// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUParameterBuffer} from '@luma-contributors/graph/gpu-dataframe';
import {
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  addHotSpotAnalysisRecipe
} from '@luma-contributors/graph/gpu-spatial-analysis';
import type {CatalogOperationBinding, CatalogRunOutput} from './catalog-binding-types';
import {COORDINATE_SYSTEM_PARAMETER} from './catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  makeCatalogTable,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeatureCollection
} from './catalog-geojson';
import {CatalogGpuUnsupportedError, withCatalogGpuSession} from './catalog-gpu-session';
import {readBufferInt32} from '../runtime/gpu-readback';
import {
  PLATFORM3_GPU_OR_CPU_CLEANUP,
  getNormalTwoSidedPValue,
  makeCellPolygonText,
  makeGridSpec,
  makeInputSpec,
  makeNumberParameter,
  makeOutputSpec,
  makeStringParameter,
  type GridSpec,
  type Values
} from './catalog-platform3-b-common';
import {type LocalFrame} from './catalog-local-frame';
import {
  MAXIMUM_RECIPE_POINTS,
  makeRecipeEnumParameter,
  makeRecipeFrame,
  readRecipePoints
} from './catalog-platform5-recipe-common';

const MAXIMUM_HOT_SPOT_CELLS = 40000;

type LatticeInputs = {
  frame: LocalFrame;
  grid: GridSpec;
  /** Per cell, row-major, point count or value sum. */
  values: Float32Array;
  pointCount: number;
  droppedCount: number;
  criterion: 'queen' | 'rook';
  radius: number;
  minimumNeighbors: number;
  multipleTesting: 'none' | 'benjamini-hochberg';
};

type HotSpotResult = {
  zScores: number[];
  pValues: number[];
  bins: number[];
  neighborCounts: number[];
};

function readLatticeInputs(
  inputs: Readonly<Record<string, CatalogFeatureCollection>>,
  values: Values
) {
  const valueProperty = (values['valueProperty'] as string).trim();
  const frame = makeRecipeFrame(values['coordinateSystem'] as CatalogCoordinateSystem, [
    inputs['points']
  ]);
  const points = readRecipePoints(inputs['points'], valueProperty, frame);
  let extentX = 0;
  let extentY = 0;
  for (let index = 0; index < points.count; index++) {
    extentX = Math.max(extentX, points.positions[2 * index]);
    extentY = Math.max(extentY, points.positions[2 * index + 1]);
  }
  const grid = makeGridSpec(extentX, extentY, values['cellSize'] as number);
  // The lattice stays small enough for the neighbor list.
  if (grid.columns * grid.rows > MAXIMUM_HOT_SPOT_CELLS)
    throw new RangeError('Lattice exceeds 40,000 cells; raise the cell size');
  // Gi* needs enough cells for a variance.
  if (grid.columns * grid.rows < 4) throw new RangeError('Too few lattice cells');
  const cellValues = new Float64Array(grid.columns * grid.rows);
  let droppedCount = points.droppedCount;
  const inverse = Math.fround(1 / Math.fround(grid.cellSize));
  const getCell = (value: number, limit: number) =>
    Math.min(limit - 1, Math.max(0, Math.floor(Math.fround(value * inverse))));
  for (let index = 0; index < points.count; index++) {
    if (!Number.isFinite(points.values[index])) {
      droppedCount++;
      continue;
    }
    cellValues[
      getCell(points.positions[2 * index + 1], grid.rows) * grid.columns +
        getCell(points.positions[2 * index], grid.columns)
    ] += points.values[index];
  }
  const lattice = Float32Array.from(cellValues);
  // A constant lattice has no variation to test.
  if (lattice.every(value => value === lattice[0])) throw new RangeError('Constant values');
  return {
    frame,
    grid,
    values: lattice,
    pointCount: points.count,
    droppedCount,
    criterion: values['criterion'] as 'queen' | 'rook',
    radius: values['radius'] as number,
    minimumNeighbors: values['minimumNeighbors'] as number,
    multipleTesting: values['multipleTesting'] as 'none' | 'benjamini-hochberg'
  } satisfies LatticeInputs;
}

function assignBins(
  zScores: readonly number[],
  pValues: readonly number[],
  multipleTesting: string
): number[] {
  const defined = pValues.flatMap((p, index) => (Number.isFinite(p) ? [index] : []));
  const adjusted = pValues.slice();
  if (multipleTesting === 'benjamini-hochberg') {
    const order = defined.sort((a, b) => pValues[a] - pValues[b]);
    let running = 1;
    for (let rank = order.length - 1; rank >= 0; rank--) {
      running = Math.min(running, (pValues[order[rank]] * order.length) / (rank + 1));
      adjusted[order[rank]] = running;
    }
  }
  return zScores.map((z, index) => {
    if (!Number.isFinite(z)) return 0;
    const p = adjusted[index];
    const level =
      multipleTesting === 'none'
        ? Math.abs(z) >= 2.576
          ? 3
          : Math.abs(z) >= 1.96
            ? 2
            : Math.abs(z) >= 1.645
              ? 1
              : 0
        : p <= 0.01
          ? 3
          : p <= 0.05
            ? 2
            : p <= 0.1
              ? 1
              : 0;
    return z < 0 ? -level : level;
  });
}

function computeHotSpotCpu(lattice: LatticeInputs): HotSpotResult {
  const {columns, rows} = lattice.grid;
  const n = columns * rows;
  let mean = 0;
  for (const value of lattice.values) mean += value;
  mean /= n;
  let variance = 0;
  for (const value of lattice.values) variance += (value - mean) ** 2;
  const deviation = Math.sqrt(variance / n);
  const zScores: number[] = [];
  const neighborCounts: number[] = [];
  for (let row = 0; row < rows; row++)
    for (let column = 0; column < columns; column++) {
      let sum = lattice.values[row * columns + column] - mean;
      let neighbors = 0;
      for (let dy = -lattice.radius; dy <= lattice.radius; dy++)
        for (let dx = -lattice.radius; dx <= lattice.radius; dx++) {
          if (!dx && !dy) continue;
          if (lattice.criterion === 'rook' && Math.abs(dx) + Math.abs(dy) > lattice.radius)
            continue;
          const x = column + dx;
          const y = row + dy;
          if (x < 0 || x >= columns || y < 0 || y >= rows) continue;
          sum += lattice.values[y * columns + x] - mean;
          neighbors++;
        }
      neighborCounts.push(neighbors);
      const weightSum = neighbors + 1;
      zScores.push(
        neighbors < Math.max(1, lattice.minimumNeighbors)
          ? Number.NaN
          : sum / (deviation * Math.sqrt((n * weightSum - weightSum * weightSum) / (n - 1)))
      );
    }
  const pValues = zScores.map(z => (Number.isFinite(z) ? getNormalTwoSidedPValue(z) : Number.NaN));
  return {
    zScores,
    pValues,
    bins: assignBins(zScores, pValues, lattice.multipleTesting),
    neighborCounts
  };
}

async function computeHotSpotGpu(
  device: Parameters<typeof withCatalogGpuSession>[0],
  lattice: LatticeInputs
): Promise<HotSpotResult> {
  const {columns, rows} = lattice.grid;
  const cellCount = columns * rows;
  const parameters = new GPUParameterBuffer(device, {
    id: 'catalog-recipe-hot-spot-parameters',
    format: 'float32',
    length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  });
  try {
    return await withCatalogGpuSession(device, 'catalog-recipe-hot-spot', async session => {
      const zScores = session.output('z-scores', 'float32', cellCount);
      const pValues = session.output('p-values', 'float32', cellCount);
      const bins = session.output('bins', 'sint32', cellCount);
      const neighborCounts = session.output('neighbor-counts', 'uint32', cellCount);
      const weightsOverflow = session.output('weights-overflow', 'uint32', 1);
      addHotSpotAnalysisRecipe(session.graph, {
        id: 'catalog-recipe-hot-spot',
        source: {
          kind: 'lattice',
          values: session.input('values', lattice.values, 'float32'),
          width: columns,
          height: rows,
          criterion: lattice.criterion,
          radius: lattice.radius,
          neighborCapacity: cellCount * (2 * lattice.radius + 1) ** 2
        },
        parameters,
        minimumNeighbors: lattice.minimumNeighbors,
        multipleTesting: lattice.multipleTesting,
        outputs: {
          zScores: zScores.view,
          pValues: pValues.view,
          bins: bins.view,
          neighborCounts: neighborCounts.view,
          status: {weightsOverflow: weightsOverflow.view}
        }
      });
      session.run();
      // The neighbor list is sized for the full window, so an overflow means the setup is wrong.
      if (await session.readScalar(weightsOverflow))
        throw new CatalogGpuUnsupportedError('The GPU neighbor list overflowed');
      return {
        zScores: Array.from(await session.readFloat32(zScores)),
        pValues: Array.from(await session.readFloat32(pValues)),
        bins: Array.from(await readBufferInt32(bins.buffer, cellCount)),
        neighborCounts: Array.from(await session.readUint32(neighborCounts))
      };
    });
  } finally {
    parameters.destroy();
  }
}

function buildHotSpotOutput(
  lattice: LatticeInputs,
  result: HotSpotResult,
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  if (lattice.droppedCount)
    addCatalogDiagnostic(
      diagnostics,
      'feature-dropped',
      'warning',
      'Features without a point position or numeric value are skipped.',
      lattice.droppedCount
    );
  const table = makeCatalogTable(
    'recipe-hot-spots',
    'Lattice hot spots (Gi*)',
    {
      cell_index: 'int32',
      column: 'int32',
      row: 'int32',
      value: 'float64',
      neighbor_count: 'int32',
      z_score: 'float64',
      p_value: 'float64',
      bin: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  const {columns, rows} = lattice.grid;
  for (let row = 0; row < rows; row++)
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      appendCatalogRow(table, {
        cell_index: index,
        column,
        row,
        value: lattice.values[index],
        neighbor_count: result.neighborCounts[index],
        z_score: Number.isFinite(result.zScores[index]) ? result.zScores[index] : null,
        p_value: Number.isFinite(result.pValues[index]) ? result.pValues[index] : null,
        bin: result.bins[index],
        geometry: makeCellPolygonText(lattice.frame, lattice.grid, column, row)
      });
    }
  addCatalogDiagnostic(
    diagnostics,
    'binary-weights-with-self',
    'info',
    'Gi* uses binary lattice weights including the cell itself; bins +-1, +-2, +-3 mean 90, 95 and 99 percent hot or cold.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {
      columns,
      rows,
      cellSize: lattice.grid.cellSize,
      pointCount: lattice.pointCount,
      testedCellCount: result.zScores.filter(z => Number.isFinite(z)).length
    },
    ...(dispatchedContributors ? {dispatchedContributors} : {})
  };
}

const HOT_SPOT_BINDING: CatalogOperationBinding = {
  id: 'recipe-hot-spot-lattice',
  title: 'Lattice hot spot analysis',
  domain: 'analysis',
  summary:
    'Counts (or sums a property of) points on a square lattice and runs Getis-Ord Gi* over the lattice with the hot spot recipe, publishing z-scores, p-values and 90, 95 and 99 percent hot and cold spot bins.',
  capabilityIds: ['spatial-autocorrelation-and-inference'],
  contributorNames: ['GPULatticeWeights', 'GPUHotSpotAnalysis'],
  gpuContributorNames: ['GPULatticeWeights', 'GPUHotSpotAnalysis'],
  engine: 'gpu-or-cpu',
  inputs: [
    makeInputSpec('points', 'Points', ['point'], MAXIMUM_RECIPE_POINTS, MAXIMUM_RECIPE_POINTS)
  ],
  parameters: [
    makeStringParameter('valueProperty', 'Value property to sum (empty = count points)'),
    makeNumberParameter('cellSize', 'Lattice cell size', 10, 1e-6, 1e7, 'crs'),
    makeRecipeEnumParameter('criterion', 'Neighborhood', ['queen', 'rook'], 'queen'),
    makeNumberParameter('radius', 'Neighborhood radius (cells)', 1, 1, 3, undefined, 'integer'),
    makeNumberParameter('minimumNeighbors', 'Minimum neighbors', 1, 1, 48, undefined, 'integer'),
    makeRecipeEnumParameter(
      'multipleTesting',
      'Multiple testing',
      ['none', 'benjamini-hochberg'],
      'none'
    ),
    COORDINATE_SYSTEM_PARAMETER
  ],
  outputs: [
    makeOutputSpec('recipe-hot-spots', 'Lattice hot spots (Gi*)', 'polygon', ['cell_index'], 40000)
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    'Points are binned on a lattice in the local meter frame (at most 40,000 cells); every cell is analysed, empty cells included, with the count or value sum as the analysed value.',
    'Binary neighbors within the radius (queen: Chebyshev, rook: Manhattan) plus the cell itself; z-scores use population moments over all cells and the Ord-Getis variance for the cell weight sum.',
    'Cells with fewer neighbors than the minimum get null z-score and p-value and bin 0. A constant lattice throws.',
    'Bins are 1, 2, 3 for 90, 95 and 99 percent (negative for cold); with Benjamini-Hochberg they use the adjusted p-values of all defined cells.'
  ],
  limitation:
    'Lattice source only: the recipe study-area, DGGS cell, distance band, permutation and color stages are not exposed; the existing hotspot-gi-star binding covers feature-based neighbors.',
  run: (inputs, values) => {
    const lattice = readLatticeInputs(inputs, values);
    return buildHotSpotOutput(lattice, computeHotSpotCpu(lattice));
  },
  runGpu: async (context, inputs, values) => {
    const lattice = readLatticeInputs(inputs, values);
    return buildHotSpotOutput(lattice, await computeHotSpotGpu(context.device, lattice), [
      'GPULatticeWeights',
      'GPUHotSpotAnalysis'
    ]);
  }
};

export const PLATFORM5_RECIPE_HOT_SPOT_BINDINGS: readonly CatalogOperationBinding[] = [
  HOT_SPOT_BINDING
];
