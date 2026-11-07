// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPURasterConnectedComponents, GPURasterNeighborhood} from '@luma.gl/graph/gpu-raster';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import type {CatalogOperationBinding, CatalogRunOutput} from './catalog-binding-types';
import {COORDINATE_SYSTEM_PARAMETER} from './catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  makeCatalogGeometryText,
  makeCatalogTable,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeatureCollection
} from './catalog-geojson';
import {CatalogGpuUnsupportedError, withCatalogGpuSession} from './catalog-gpu-session';
import {
  PLATFORM3_GPU_OR_CPU_CLEANUP,
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

// The catalog interface carries feature collections, not raster tiles, so the raster-only
// contributors are bound through a synthetic grid: points are binned onto a square lattice in the
// local frame and the contributor runs on that lattice. The published tables say so.

const MAXIMUM_RASTER_CELLS = 40000;

type PointGrid = {
  frame: LocalFrame;
  grid: GridSpec;
  /** Per cell, row-major, the sum of the point values (point count without a value property). */
  sums: Float32Array;
  /** Per cell, row-major, the number of points. */
  counts: Uint32Array;
  pointCount: number;
  droppedCount: number;
};

function readPointGrid(
  inputs: Readonly<Record<string, CatalogFeatureCollection>>,
  values: Values
): PointGrid {
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
  // The synthetic grid stays small enough for the raster kernels.
  if (grid.columns * grid.rows > MAXIMUM_RASTER_CELLS)
    throw new RangeError('Grid exceeds 40,000 cells; raise the cell size');
  const sums = new Float64Array(grid.columns * grid.rows);
  const counts = new Uint32Array(sums.length);
  const inverse = Math.fround(1 / Math.fround(grid.cellSize));
  const getCell = (value: number, limit: number) =>
    Math.min(limit - 1, Math.max(0, Math.floor(Math.fround(value * inverse))));
  let droppedCount = points.droppedCount;
  for (let index = 0; index < points.count; index++) {
    if (!Number.isFinite(points.values[index])) {
      droppedCount++;
      continue;
    }
    const cell =
      getCell(points.positions[2 * index + 1], grid.rows) * grid.columns +
      getCell(points.positions[2 * index], grid.columns);
    sums[cell] += points.values[index];
    counts[cell]++;
  }
  return {
    frame,
    grid,
    sums: Float32Array.from(sums),
    counts,
    pointCount: points.count,
    droppedCount
  };
}

function makeRasterMetadata(grid: GridSpec) {
  return makeGPURasterMetadata({
    width: grid.columns,
    height: grid.rows,
    cellSize: [grid.cellSize, grid.cellSize],
    origin: [0, 0],
    rowDirection: 'north'
  });
}

function addDroppedPoints(diagnostics: CatalogDiagnostic[], count: number): void {
  if (count)
    addCatalogDiagnostic(
      diagnostics,
      'feature-dropped',
      'warning',
      'Features without a point position or numeric value are skipped.',
      count
    );
}

function addSyntheticGridDiagnostic(diagnostics: CatalogDiagnostic[]): void {
  addCatalogDiagnostic(
    diagnostics,
    'synthetic-grid',
    'info',
    'The raster contributor runs on a square lattice built from the points (no raster tile input).'
  );
}

const GRID_PARAMETERS = [
  makeStringParameter('valueProperty', 'Value property to sum (empty = count points)'),
  makeNumberParameter('cellSize', 'Grid cell size', 10, 1e-6, 1e7, 'crs'),
  COORDINATE_SYSTEM_PARAMETER
];

// ---------------------------------------------------------------------------------------------
// Density smoothing (GPURasterNeighborhood)
// ---------------------------------------------------------------------------------------------

function makeSmoothingKernel(kernel: string, radius: number): number[] {
  const side = 2 * radius + 1;
  const coefficients: number[] = [];
  for (let dy = -radius; dy <= radius; dy++)
    for (let dx = -radius; dx <= radius; dx++)
      coefficients.push(
        kernel === 'gaussian' ? Math.exp(-(dx * dx + dy * dy) / (2 * (radius / 2) ** 2)) : 1
      );
  const total = coefficients.reduce((sum, value) => sum + value, 0);
  // Normalized so the smoothed grid keeps the mass inside the grid.
  return coefficients.map(value => Math.fround(value / total)).slice(0, side * side);
}

function smoothGridCpu(grid: PointGrid, kernel: readonly number[], radius: number): Float64Array {
  const {columns, rows} = grid.grid;
  const side = 2 * radius + 1;
  const smoothed = new Float64Array(columns * rows);
  for (let row = 0; row < rows; row++)
    for (let column = 0; column < columns; column++) {
      let sum = 0;
      for (let dy = -radius; dy <= radius; dy++)
        for (let dx = -radius; dx <= radius; dx++) {
          const x = column + dx;
          const y = row + dy;
          // Samples outside the grid count as zero.
          if (x < 0 || x >= columns || y < 0 || y >= rows) continue;
          sum += kernel[(dy + radius) * side + dx + radius] * grid.sums[y * columns + x];
        }
      smoothed[row * columns + column] = sum;
    }
  return smoothed;
}

async function smoothGridGpu(
  device: Parameters<typeof withCatalogGpuSession>[0],
  grid: PointGrid,
  kernel: readonly number[],
  radius: number
): Promise<Float64Array> {
  const {columns, rows} = grid.grid;
  return withCatalogGpuSession(device, 'catalog-raster-smoothing', async session => {
    const cellCount = columns * rows;
    const output = session.output('smoothed', 'float32', cellCount);
    const validity = session.output('validity', 'uint32', cellCount);
    const band = makeGPURasterBand(
      session.input('sums', grid.sums, 'float32'),
      makeRasterMetadata(grid.grid),
      {id: 'catalog-raster-smoothing-input'}
    );
    new GPURasterNeighborhood({
      id: 'catalog-raster-smoothing',
      width: columns,
      height: rows,
      input: band,
      output: output.view,
      outputValidity: validity.view,
      radius,
      kernel,
      borderMode: 'constant',
      borderValue: 0
    }).addToGraph(session.graph);
    session.run();
    const flags = await session.readUint32(validity);
    // Every input sample is valid, so every output must be too.
    if (flags.some(flag => flag === 0))
      throw new CatalogGpuUnsupportedError('The GPU stencil marked cells invalid');
    return Float64Array.from(await session.readFloat32(output));
  });
}

function buildSmoothingOutput(
  grid: PointGrid,
  smoothed: Float64Array,
  kernel: string,
  radius: number,
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  addDroppedPoints(diagnostics, grid.droppedCount);
  addSyntheticGridDiagnostic(diagnostics);
  const table = makeCatalogTable(
    'raster-smoothed-density',
    'Smoothed grid density',
    {
      cell_index: 'int32',
      column: 'int32',
      row: 'int32',
      point_count: 'int32',
      raw_value: 'float64',
      smoothed_value: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const {columns, rows} = grid.grid;
  for (let row = 0; row < rows; row++)
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      appendCatalogRow(table, {
        cell_index: index,
        column,
        row,
        point_count: grid.counts[index],
        raw_value: grid.sums[index],
        smoothed_value: smoothed[index],
        geometry: makeCellPolygonText(grid.frame, grid.grid, column, row)
      });
    }
  return {
    tables: [table],
    diagnostics,
    summary: {
      columns,
      rows,
      cellSize: grid.grid.cellSize,
      pointCount: grid.pointCount,
      kernel,
      radius
    },
    ...(dispatchedContributors ? {dispatchedContributors} : {})
  };
}

function readSmoothingArguments(
  inputs: Readonly<Record<string, CatalogFeatureCollection>>,
  values: Values
) {
  const radius = values['radius'] as number;
  const kernelName = values['kernel'] as string;
  return {
    grid: readPointGrid(inputs, values),
    radius,
    kernelName,
    kernel: makeSmoothingKernel(kernelName, radius)
  };
}

const SMOOTHING_BINDING: CatalogOperationBinding = {
  id: 'raster-density-smoothing',
  title: 'Grid density smoothing',
  domain: 'raster',
  summary:
    'Bins points onto a square grid and smooths the per-cell count (or value sum) with a normalized box or Gaussian stencil using the raster neighborhood contributor.',
  capabilityIds: ['density-and-interpolation'],
  contributorNames: ['GPURasterNeighborhood'],
  gpuContributorNames: ['GPURasterNeighborhood'],
  engine: 'gpu-or-cpu',
  inputs: [
    makeInputSpec('points', 'Points', ['point'], MAXIMUM_RECIPE_POINTS, MAXIMUM_RECIPE_POINTS)
  ],
  parameters: [
    ...GRID_PARAMETERS,
    makeRecipeEnumParameter('kernel', 'Kernel', ['box', 'gaussian'], 'gaussian'),
    makeNumberParameter('radius', 'Stencil radius (cells)', 2, 1, 8, undefined, 'integer')
  ],
  outputs: [
    makeOutputSpec(
      'raster-smoothed-density',
      'Smoothed grid density',
      'polygon',
      ['cell_index'],
      40000
    )
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    'Points are binned on a lattice in the local meter frame (at most 40,000 cells); cells hold the point count or the sum of the value property, and empty cells are zero, so every cell is valid.',
    'The stencil is (2 radius + 1) squared cells with coefficients summing to one (Gaussian sigma = radius / 2); cells outside the grid count as zero, so mass near the border is lost rather than renormalized.',
    'Features without a point or numeric value are skipped; all lattice cells are published, the smoothed value in the unit of the cell value.'
  ],
  limitation:
    'Synthetic-grid route: the contributor runs on a lattice built from the points because the catalog interface carries no raster tiles. Other raster operations (terrain, rank filters, zonal raster statistics on uploaded tiles) stay in the raster panels.',
  run: (inputs, values) => {
    const {grid, radius, kernelName, kernel} = readSmoothingArguments(inputs, values);
    return buildSmoothingOutput(grid, smoothGridCpu(grid, kernel, radius), kernelName, radius);
  },
  runGpu: async (context, inputs, values) => {
    const {grid, radius, kernelName, kernel} = readSmoothingArguments(inputs, values);
    if (!grid.pointCount) throw new CatalogGpuUnsupportedError('Nothing to smooth');
    return buildSmoothingOutput(
      grid,
      await smoothGridGpu(context.device, grid, kernel, radius),
      kernelName,
      radius,
      ['GPURasterNeighborhood']
    );
  }
};

// ---------------------------------------------------------------------------------------------
// Connected cell clusters (GPURasterConnectedComponents)
// ---------------------------------------------------------------------------------------------

function labelComponentsCpu(grid: PointGrid, minimumPoints: number, connectivity: number) {
  const {columns, rows} = grid.grid;
  const labels = new Uint32Array(columns * rows);
  for (let start = 0; start < labels.length; start++) {
    if (labels[start] || grid.counts[start] < minimumPoints) continue;
    // Row-major order visits the minimum cell of a component first.
    const stack = [start];
    labels[start] = start + 1;
    while (stack.length) {
      const cell = stack.pop()!;
      const column = cell % columns;
      const row = Math.floor(cell / columns);
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          if ((!dx && !dy) || (connectivity === 4 && dx && dy)) continue;
          const x = column + dx;
          const y = row + dy;
          if (x < 0 || x >= columns || y < 0 || y >= rows) continue;
          const neighbor = y * columns + x;
          if (labels[neighbor] || grid.counts[neighbor] < minimumPoints) continue;
          labels[neighbor] = start + 1;
          stack.push(neighbor);
        }
    }
  }
  return labels;
}

async function labelComponentsGpu(
  device: Parameters<typeof withCatalogGpuSession>[0],
  grid: PointGrid,
  minimumPoints: number,
  connectivity: 4 | 8
): Promise<Uint32Array> {
  const {columns, rows} = grid.grid;
  return withCatalogGpuSession(device, 'catalog-raster-components', async session => {
    const cellCount = columns * rows;
    const labels = session.output('labels', 'uint32', cellCount);
    const validity = session.output('validity', 'uint32', cellCount);
    const converged = session.output('converged', 'uint32', 1);
    const foreground = Uint32Array.from(grid.counts, count => (count >= minimumPoints ? 1 : 0));
    const band = makeGPURasterBand(
      session.input('foreground', foreground, 'uint32'),
      makeRasterMetadata(grid.grid),
      {id: 'catalog-raster-components-input'}
    );
    new GPURasterConnectedComponents({
      id: 'catalog-raster-components',
      width: columns,
      height: rows,
      input: band,
      output: labels.view,
      outputValidity: validity.view,
      converged: converged.view,
      connectivity
    }).addToGraph(session.graph);
    session.run();
    // An exhausted round budget clears every label.
    if ((await session.readScalar(converged)) !== 1)
      throw new CatalogGpuUnsupportedError('The GPU labeling did not converge');
    return await session.readUint32(labels);
  });
}

function buildComponentsOutput(
  grid: PointGrid,
  labels: Uint32Array,
  minimumPoints: number,
  connectivity: number,
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  addDroppedPoints(diagnostics, grid.droppedCount);
  addSyntheticGridDiagnostic(diagnostics);
  const {columns} = grid.grid;
  const components = new Map<
    number,
    {cells: number; points: number; sum: number; x: number; y: number}
  >();
  labels.forEach((label, cell) => {
    if (!label) return;
    const entry = components.get(label) ?? {cells: 0, points: 0, sum: 0, x: 0, y: 0};
    entry.cells++;
    entry.points += grid.counts[cell];
    entry.sum += grid.sums[cell];
    entry.x += grid.grid.minX + ((cell % columns) + 0.5) * grid.grid.cellSize;
    entry.y += grid.grid.minY + (Math.floor(cell / columns) + 0.5) * grid.grid.cellSize;
    components.set(label, entry);
  });
  const table = makeCatalogTable(
    'raster-components',
    'Connected cell clusters',
    {
      component_id: 'int32',
      cell_count: 'int32',
      point_count: 'int32',
      value_sum: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  for (const [label, entry] of [...components].sort((a, b) => a[0] - b[0]))
    appendCatalogRow(table, {
      component_id: label,
      cell_count: entry.cells,
      point_count: entry.points,
      value_sum: entry.sum,
      geometry: makeCatalogGeometryText(
        'Point',
        grid.frame.inverse(entry.x / entry.cells, entry.y / entry.cells)
      )
    });
  return {
    tables: [table],
    diagnostics,
    summary: {
      columns,
      rows: grid.grid.rows,
      cellSize: grid.grid.cellSize,
      minimumPoints,
      connectivity,
      componentCount: components.size
    },
    ...(dispatchedContributors ? {dispatchedContributors} : {})
  };
}

function readComponentsArguments(
  inputs: Readonly<Record<string, CatalogFeatureCollection>>,
  values: Values
) {
  return {
    grid: readPointGrid(inputs, values),
    minimumPoints: values['minimumPoints'] as number,
    connectivity: (values['connectivity'] === '8' ? 8 : 4) as 4 | 8
  };
}

const COMPONENTS_BINDING: CatalogOperationBinding = {
  id: 'raster-cell-clusters',
  title: 'Connected cell clusters',
  domain: 'raster',
  summary:
    'Bins points onto a square grid, marks cells holding at least a minimum number of points and labels the connected groups of marked cells with the raster connected-components contributor.',
  capabilityIds: ['spatial-neighborhoods'],
  contributorNames: ['GPURasterConnectedComponents'],
  gpuContributorNames: ['GPURasterConnectedComponents'],
  engine: 'gpu-or-cpu',
  inputs: [
    makeInputSpec('points', 'Points', ['point'], MAXIMUM_RECIPE_POINTS, MAXIMUM_RECIPE_POINTS)
  ],
  parameters: [
    ...GRID_PARAMETERS,
    makeNumberParameter(
      'minimumPoints',
      'Minimum points per cell',
      3,
      1,
      1e6,
      undefined,
      'integer'
    ),
    makeRecipeEnumParameter('connectivity', 'Connectivity', ['4', '8'], '4')
  ],
  outputs: [
    makeOutputSpec('raster-components', 'Connected cell clusters', 'point', ['component_id'], 40000)
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    'Points are binned on a lattice in the local meter frame (at most 40,000 cells); a cell is foreground when it holds at least the minimum number of points.',
    'Four-connectivity joins edge neighbors, eight also diagonal ones. A component id is the row-major index of its first cell plus one, the same on both engines.',
    'Each component publishes its cell count, point count, value sum and the mean cell center as a point.'
  ],
  limitation:
    'Synthetic-grid route: components are cell groups of a lattice built from the points, not raster tile regions; no cluster outline polygons are published.',
  run: (inputs, values) => {
    const {grid, minimumPoints, connectivity} = readComponentsArguments(inputs, values);
    return buildComponentsOutput(
      grid,
      labelComponentsCpu(grid, minimumPoints, connectivity),
      minimumPoints,
      connectivity
    );
  },
  runGpu: async (context, inputs, values) => {
    const {grid, minimumPoints, connectivity} = readComponentsArguments(inputs, values);
    if (!grid.pointCount) throw new CatalogGpuUnsupportedError('Nothing to label');
    return buildComponentsOutput(
      grid,
      await labelComponentsGpu(context.device, grid, minimumPoints, connectivity),
      minimumPoints,
      connectivity,
      ['GPURasterConnectedComponents']
    );
  }
};

export const PLATFORM5_RASTER_BINDINGS: readonly CatalogOperationBinding[] = [
  SMOOTHING_BINDING,
  COMPONENTS_BINDING
];
