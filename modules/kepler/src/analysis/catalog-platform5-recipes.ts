// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  addPointsInPolygonsChoroplethRecipe,
  addStraightLineCatchmentsRecipe,
  addZonalDistributionRecipe
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  getGPUDistanceFieldParameterValues,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
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
import {
  CatalogGpuUnsupportedError,
  withCatalogGpuSession,
  type CatalogGpuSession
} from './catalog-gpu-session';
import {PLATFORM5_RECIPE_CHANGE_OF_SUPPORT_BINDINGS} from './catalog-platform5-recipes-change-of-support';
import {PLATFORM5_RECIPE_HOT_SPOT_BINDINGS} from './catalog-platform5-recipes-hot-spot';
import {
  PLATFORM3_GPU_OR_CPU_CLEANUP,
  getFeatureGeometryText,
  makeGridSpec,
  makeInputSpec,
  makeNumberParameter,
  makeOutputSpec,
  makeStringParameter,
  type Values
} from './catalog-platform3-b-common';
import {
  MAXIMUM_RECIPE_POINTS,
  assignRecipePoints,
  getRecipeCandidateCapacity,
  makeRecipeEnumParameter,
  makeRecipeFrame,
  readRecipePoints,
  readRecipePolygons,
  type RecipePoints,
  type RecipePolygons
} from './catalog-platform5-recipe-common';

type Inputs = Readonly<Record<string, CatalogFeatureCollection>>;
type Device = Parameters<typeof withCatalogGpuSession>[0];

const RECIPE_CLEANUP = PLATFORM3_GPU_OR_CPU_CLEANUP;

function addDroppedPoints(diagnostics: CatalogDiagnostic[], count: number): void {
  if (count)
    addCatalogDiagnostic(
      diagnostics,
      'feature-dropped',
      'warning',
      'Features without a point position are skipped.',
      count
    );
}

function nullIfNaN(value: number): number | null {
  return Number.isFinite(value) ? value : null;
}

/** Quantile with the linear (type 7) convention over sorted values. */
function getLinearQuantile(sorted: readonly number[], probability: number): number {
  if (!sorted.length) return Number.NaN;
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.min(lower + 1, sorted.length - 1);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function readPolygonsAndPoints(
  inputs: Inputs,
  values: Values,
  valueProperty: string
): {points: RecipePoints; polygons: RecipePolygons} {
  const frame = makeRecipeFrame(values['coordinateSystem'] as CatalogCoordinateSystem, [
    inputs['points'],
    inputs['polygons']
  ]);
  const polygons = readRecipePolygons(inputs['polygons'], frame);
  // The join needs at least one polygon feature.
  if (!polygons.featureCount) throw new RangeError('No polygon features');
  return {points: readRecipePoints(inputs['points'], valueProperty, frame), polygons};
}

/** Uploads a polygon set as the four-level layout the join contributors read. */
function uploadPolygons(
  session: CatalogGpuSession,
  polygons: RecipePolygons,
  candidateCapacity: number
) {
  return {
    kind: 'polygons' as const,
    positions: session.input('polygon-positions', polygons.positions, 'float32x2'),
    featureOffsets: session.input('feature-offsets', polygons.featureOffsets, 'uint32'),
    polygonOffsets: session.input('polygon-offsets', polygons.polygonOffsets, 'uint32'),
    ringOffsets: session.input('ring-offsets', polygons.ringOffsets, 'uint32'),
    candidateCapacity
  };
}

// ---------------------------------------------------------------------------------------------
// Points in polygons choropleth
// ---------------------------------------------------------------------------------------------

const CHOROPLETH_STATISTICS = ['count', 'sum', 'mean', 'minimum', 'maximum'] as const;

type ChoroplethTotals = {counts: number[]; statistic: number[]; unmatchedCount: number};

function computeChoroplethCpu(
  points: RecipePoints,
  polygons: RecipePolygons,
  statistic: string
): ChoroplethTotals {
  const assignment = assignRecipePoints(points, polygons);
  const featureCount = polygons.featureCount;
  const counts = new Array<number>(featureCount).fill(0);
  const sums = new Array<number>(featureCount).fill(0);
  const valid = new Array<number>(featureCount).fill(0);
  const minimums = new Array<number>(featureCount).fill(Number.NaN);
  const maximums = new Array<number>(featureCount).fill(Number.NaN);
  let unmatchedCount = 0;
  for (let point = 0; point < points.count; point++) {
    const feature = assignment[point];
    if (feature < 0) {
      unmatchedCount++;
      continue;
    }
    counts[feature]++;
    const value = points.values[point];
    if (!Number.isFinite(value)) continue;
    sums[feature] += value;
    valid[feature]++;
    minimums[feature] = Number.isNaN(minimums[feature])
      ? value
      : Math.min(minimums[feature], value);
    maximums[feature] = Number.isNaN(maximums[feature])
      ? value
      : Math.max(maximums[feature], value);
  }
  const statisticValues = counts.map((count, feature) => {
    switch (statistic) {
      case 'sum':
        return sums[feature];
      case 'mean':
        return valid[feature] ? sums[feature] / valid[feature] : Number.NaN;
      case 'minimum':
        return minimums[feature];
      case 'maximum':
        return maximums[feature];
      default:
        return count;
    }
  });
  return {counts, statistic: statisticValues, unmatchedCount};
}

async function computeChoroplethGpu(
  device: Device,
  points: RecipePoints,
  polygons: RecipePolygons,
  statistic: string
): Promise<ChoroplethTotals> {
  return withCatalogGpuSession(device, 'catalog-recipe-choropleth', async session => {
    const featureCount = polygons.featureCount;
    const counts = session.output('counts', 'uint32', featureCount);
    const featureValues = session.output('feature-values', 'float32', featureCount);
    const overflow = session.output('overflow', 'uint32', 1);
    const uncertainCount = session.output('uncertain', 'uint32', 1);
    addPointsInPolygonsChoroplethRecipe(session.graph, {
      id: 'catalog-recipe-choropleth',
      statistic: statistic as (typeof CHOROPLETH_STATISTICS)[number],
      points: {kind: 'points', positions: session.input('points', points.positions, 'float32x2')},
      ...(statistic === 'count' ? {} : {values: session.input('values', points.values, 'float32')}),
      polygons: uploadPolygons(
        session,
        polygons,
        getRecipeCandidateCapacity(points.count, featureCount)
      ),
      outputs: {
        counts: counts.view,
        ...(statistic === 'count' ? {} : {featureValues: featureValues.view}),
        status: {overflow: overflow.view, uncertainCount: uncertainCount.view}
      }
    });
    session.run();
    // A join capacity overflow or an unproven containment makes the counts incomplete.
    if ((await session.readScalar(overflow)) || (await session.readScalar(uncertainCount)))
      throw new CatalogGpuUnsupportedError('The GPU join overflowed or left uncertain pairs');
    const countValues = Array.from(await session.readUint32(counts));
    return {
      counts: countValues,
      statistic:
        statistic === 'count' ? countValues : Array.from(await session.readFloat32(featureValues)),
      unmatchedCount: points.count - countValues.reduce((sum, count) => sum + count, 0)
    };
  });
}

function buildChoroplethOutput(
  inputs: Inputs,
  points: RecipePoints,
  polygons: RecipePolygons,
  totals: ChoroplethTotals,
  statistic: string,
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  addDroppedPoints(diagnostics, points.droppedCount);
  const table = makeCatalogTable(
    'recipe-choropleth',
    'Points per polygon choropleth',
    {
      feature_index: 'int32',
      point_count: 'int32',
      statistic: 'string',
      value: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  polygons.sourceIndices.forEach((sourceIndex, row) =>
    appendCatalogRow(table, {
      feature_index: sourceIndex,
      point_count: totals.counts[row],
      statistic,
      value: nullIfNaN(totals.statistic[row]),
      geometry: getFeatureGeometryText(inputs['polygons'].features[sourceIndex])
    })
  );
  return {
    tables: [table],
    diagnostics,
    summary: {
      polygonCount: polygons.featureCount,
      pointCount: points.count,
      unmatchedCount: totals.unmatchedCount
    },
    ...(dispatchedContributors ? {dispatchedContributors} : {})
  };
}

function readChoroplethArguments(inputs: Inputs, values: Values) {
  const statistic = values['statistic'] as string;
  const valueProperty = (values['valueProperty'] as string).trim();
  // Every statistic except count needs a value property.
  if (statistic !== 'count' && !valueProperty)
    throw new RangeError('This statistic needs a value property');
  return {
    statistic,
    ...readPolygonsAndPoints(inputs, values, statistic === 'count' ? '' : valueProperty)
  };
}

const CHOROPLETH_BINDING: CatalogOperationBinding = {
  id: 'recipe-points-in-polygons-choropleth',
  title: 'Points in polygons choropleth',
  domain: 'aggregation',
  summary:
    'Joins points to polygons and maps one statistic (count, sum, mean, minimum or maximum) per polygon, the workflow of the points-in-polygons choropleth recipe.',
  capabilityIds: ['spatial-joins', 'group-and-regional-analysis'],
  contributorNames: ['GPUZonalStatistics'],
  gpuContributorNames: ['GPUZonalStatistics'],
  engine: 'gpu-or-cpu',
  inputs: [
    makeInputSpec('points', 'Points', ['point'], MAXIMUM_RECIPE_POINTS, MAXIMUM_RECIPE_POINTS),
    makeInputSpec('polygons', 'Polygons', ['polygon'], 5000, 200000)
  ],
  parameters: [
    makeRecipeEnumParameter('statistic', 'Statistic', CHOROPLETH_STATISTICS, 'count'),
    makeStringParameter('valueProperty', 'Value property (not needed for count)'),
    COORDINATE_SYSTEM_PARAMETER
  ],
  outputs: [
    makeOutputSpec(
      'recipe-choropleth',
      'Points per polygon choropleth',
      'polygon',
      ['feature_index'],
      5000
    )
  ],
  cleanup: RECIPE_CLEANUP,
  declaredBehavior: [
    'Each point joins the smallest-row polygon feature that contains it (even-odd per polygon part, holes respected); points outside every polygon are counted in the summary.',
    'Points exactly on a polygon edge may differ between the engines (the GPU applies the half-open ownership rule).',
    'Points with a missing or non-numeric value still count toward point_count but do not contribute to sum, mean, minimum or maximum; mean, minimum and maximum of a polygon without valid values are null.',
    'Coordinates are mapped into a local meter frame and rounded to float32 for both engines; sums are float32 on the GPU.'
  ],
  limitation:
    'The recipe is wired with the zonal-statistics back-end; the class break and color scale stages of the recipe are not published (use the choropleth styling of the host).',
  run: (inputs, values) => {
    const {statistic, points, polygons} = readChoroplethArguments(inputs, values);
    return buildChoroplethOutput(
      inputs,
      points,
      polygons,
      computeChoroplethCpu(points, polygons, statistic),
      statistic
    );
  },
  runGpu: async (context, inputs, values) => {
    const {statistic, points, polygons} = readChoroplethArguments(inputs, values);
    if (!points.count) throw new CatalogGpuUnsupportedError('Nothing to join');
    return buildChoroplethOutput(
      inputs,
      points,
      polygons,
      await computeChoroplethGpu(context.device, points, polygons, statistic),
      statistic,
      ['GPUZonalStatistics']
    );
  }
};

// ---------------------------------------------------------------------------------------------
// Zonal distribution
// ---------------------------------------------------------------------------------------------

type DistributionTotals = {
  rowCounts: number[];
  validCounts: number[];
  medians: number[];
  modes: number[];
  uniqueCounts: number[];
  /** Zone-major, one row per zone and probability. */
  quantiles: number[];
};

function readProbabilities(values: Values): number[] {
  const probabilities = Array.from(values['probabilities'] as readonly number[]);
  // Probabilities must lie in [0, 1].
  if (probabilities.some(probability => !(probability >= 0 && probability <= 1)))
    throw new RangeError('Probabilities must be between 0 and 1');
  if (probabilities.length > 9) throw new RangeError('At most 9 probabilities');
  return probabilities.map(probability => Math.fround(probability));
}

function computeDistributionCpu(
  points: RecipePoints,
  polygons: RecipePolygons,
  probabilities: readonly number[]
): DistributionTotals {
  const assignment = assignRecipePoints(points, polygons);
  const zoneValues: number[][] = Array.from({length: polygons.featureCount}, () => []);
  const rowCounts = new Array<number>(polygons.featureCount).fill(0);
  for (let point = 0; point < points.count; point++) {
    const zone = assignment[point];
    if (zone < 0) continue;
    rowCounts[zone]++;
    if (Number.isFinite(points.values[point])) zoneValues[zone].push(points.values[point]);
  }
  const totals: DistributionTotals = {
    rowCounts,
    validCounts: [],
    medians: [],
    modes: [],
    uniqueCounts: [],
    quantiles: []
  };
  for (const list of zoneValues) {
    const sorted = list.sort((a, b) => a - b);
    totals.validCounts.push(sorted.length);
    totals.medians.push(getLinearQuantile(sorted, 0.5));
    let mode = Number.NaN;
    let modeCount = 0;
    let uniqueCount = 0;
    for (let start = 0; start < sorted.length; ) {
      let end = start;
      while (end < sorted.length && sorted[end] === sorted[start]) end++;
      uniqueCount++;
      // Ties resolve to the smallest value because values ascend.
      if (end - start > modeCount) {
        modeCount = end - start;
        mode = sorted[start];
      }
      start = end;
    }
    totals.modes.push(mode);
    totals.uniqueCounts.push(uniqueCount);
    for (const probability of probabilities)
      totals.quantiles.push(getLinearQuantile(sorted, probability));
  }
  return totals;
}

async function computeDistributionGpu(
  device: Device,
  points: RecipePoints,
  polygons: RecipePolygons,
  probabilities: readonly number[]
): Promise<DistributionTotals> {
  return withCatalogGpuSession(device, 'catalog-recipe-zonal-distribution', async session => {
    const zoneCount = polygons.featureCount;
    const zoneIds = session.output('zone-ids', 'uint32', zoneCount);
    const rowCounts = session.output('row-counts', 'uint32', zoneCount);
    const validCounts = session.output('valid-counts', 'uint32', zoneCount);
    const medians = session.output('medians', 'float32', zoneCount);
    const modes = session.output('modes', 'float32', zoneCount);
    const uniqueCounts = session.output('unique-counts', 'uint32', zoneCount);
    const quantiles = session.output('quantiles', 'float32', zoneCount * probabilities.length);
    addZonalDistributionRecipe(session.graph, {
      id: 'catalog-recipe-zonal-distribution',
      source: {
        kind: 'points',
        join: {
          points: {
            kind: 'points',
            positions: session.input('points', points.positions, 'float32x2')
          },
          polygons: uploadPolygons(session, polygons, 1),
          candidateCapacity: getRecipeCandidateCapacity(points.count, zoneCount)
        }
      },
      values: session.input('values', points.values, 'float32'),
      ...(probabilities.length
        ? {
            probabilities: session.input(
              'probabilities',
              Float32Array.from(probabilities),
              'float32'
            )
          }
        : {}),
      outputs: {
        zoneIds: zoneIds.view,
        rowCounts: rowCounts.view,
        validCounts: validCounts.view,
        medians: medians.view,
        modes: modes.view,
        uniqueCounts: uniqueCounts.view,
        ...(probabilities.length ? {quantiles: quantiles.view} : {})
      }
    });
    session.run();
    return {
      rowCounts: Array.from(await session.readUint32(rowCounts)),
      validCounts: Array.from(await session.readUint32(validCounts)),
      medians: Array.from(await session.readFloat32(medians)),
      modes: Array.from(await session.readFloat32(modes)),
      uniqueCounts: Array.from(await session.readUint32(uniqueCounts)),
      quantiles: probabilities.length ? Array.from(await session.readFloat32(quantiles)) : []
    };
  });
}

function buildDistributionOutput(
  inputs: Inputs,
  points: RecipePoints,
  polygons: RecipePolygons,
  probabilities: readonly number[],
  totals: DistributionTotals,
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  addDroppedPoints(diagnostics, points.droppedCount);
  const quantileColumns = Object.fromEntries(
    probabilities.map((_, index) => [`quantile_${index}`, 'float64' as const])
  );
  const table = makeCatalogTable(
    'recipe-zonal-distribution',
    'Value distribution per polygon',
    {
      feature_index: 'int32',
      row_count: 'int32',
      valid_count: 'int32',
      median: 'float64',
      mode: 'float64',
      unique_count: 'int32',
      ...quantileColumns,
      geometry: 'string'
    },
    'geometry'
  );
  polygons.sourceIndices.forEach((sourceIndex, row) =>
    appendCatalogRow(table, {
      feature_index: sourceIndex,
      row_count: totals.rowCounts[row],
      valid_count: totals.validCounts[row],
      median: nullIfNaN(totals.medians[row]),
      mode: nullIfNaN(totals.modes[row]),
      unique_count: totals.uniqueCounts[row],
      ...Object.fromEntries(
        probabilities.map((_, index) => [
          `quantile_${index}`,
          nullIfNaN(totals.quantiles[row * probabilities.length + index])
        ])
      ),
      geometry: getFeatureGeometryText(inputs['polygons'].features[sourceIndex])
    })
  );
  return {
    tables: [table],
    diagnostics,
    summary: {
      polygonCount: polygons.featureCount,
      pointCount: points.count,
      probabilities: probabilities.join(',')
    },
    ...(dispatchedContributors ? {dispatchedContributors} : {})
  };
}

function readDistributionArguments(inputs: Inputs, values: Values) {
  const valueProperty = (values['valueProperty'] as string).trim();
  // The distribution needs a numeric property.
  if (!valueProperty) throw new RangeError('Name a value property');
  return {
    probabilities: readProbabilities(values),
    ...readPolygonsAndPoints(inputs, values, valueProperty)
  };
}

const ZONAL_DISTRIBUTION_BINDING: CatalogOperationBinding = {
  id: 'recipe-zonal-distribution',
  title: 'Zonal value distribution',
  domain: 'aggregation',
  summary:
    'Assigns points to polygons and publishes the median, mode, unique value count and chosen quantiles of a numeric property per polygon.',
  capabilityIds: ['spatial-joins', 'group-and-regional-analysis'],
  contributorNames: ['GPUPointInPolygonJoin', 'GPUGroupStatistics'],
  gpuContributorNames: ['GPUPointInPolygonJoin', 'GPUGroupStatistics'],
  engine: 'gpu-or-cpu',
  inputs: [
    makeInputSpec('points', 'Points', ['point'], MAXIMUM_RECIPE_POINTS, MAXIMUM_RECIPE_POINTS),
    makeInputSpec('polygons', 'Polygons', ['polygon'], 5000, 200000)
  ],
  parameters: [
    makeStringParameter('valueProperty', 'Value property'),
    {
      name: 'probabilities',
      label: 'Quantile probabilities (0 to 1, at most 9)',
      kind: 'number-list',
      minimum: 0,
      maximum: 1,
      defaultValue: [0.25, 0.75]
    },
    COORDINATE_SYSTEM_PARAMETER
  ],
  outputs: [
    makeOutputSpec(
      'recipe-zonal-distribution',
      'Value distribution per polygon',
      'polygon',
      ['feature_index'],
      5000
    )
  ],
  cleanup: RECIPE_CLEANUP,
  declaredBehavior: [
    'Each point joins the smallest-row polygon feature that contains it; points outside every polygon are ignored.',
    'row_count counts all assigned points; the statistics use only points with a finite numeric value (valid_count). Empty polygons keep zero counts and null order statistics.',
    'Median and quantiles use linear interpolation of the sorted values; the mode is the most frequent value with ties resolved to the smallest value; values are compared after float32 rounding.',
    'Points exactly on a polygon edge may differ between the engines.'
  ],
  limitation:
    'Unweighted distributions of point values; the raster and pre-assigned zone sources of the recipe are not exposed.',
  run: (inputs, values) => {
    const {probabilities, points, polygons} = readDistributionArguments(inputs, values);
    return buildDistributionOutput(
      inputs,
      points,
      polygons,
      probabilities,
      computeDistributionCpu(points, polygons, probabilities)
    );
  },
  runGpu: async (context, inputs, values) => {
    const {probabilities, points, polygons} = readDistributionArguments(inputs, values);
    if (!points.count) throw new CatalogGpuUnsupportedError('Nothing to join');
    return buildDistributionOutput(
      inputs,
      points,
      polygons,
      probabilities,
      await computeDistributionGpu(context.device, points, polygons, probabilities),
      ['GPUPointInPolygonJoin', 'GPUGroupStatistics']
    );
  }
};

// ---------------------------------------------------------------------------------------------
// Straight line catchments
// ---------------------------------------------------------------------------------------------

type CatchmentTotals = {
  allocatedCellCount: number;
  cellCounts: number[];
  valueCounts: number[];
  sums: number[];
  means: number[];
  minimums: number[];
  maximums: number[];
};

type CatchmentGrid = {
  columns: number;
  rows: number;
  cellSize: number;
  facilities: RecipePoints;
  /** Cell of every facility. */
  seedColumns: Int32Array;
  seedRows: Int32Array;
  /** Mean observation value per cell, NaN without observations. */
  cellValues: Float32Array;
};

function readCatchmentGrid(inputs: Inputs, values: Values): CatchmentGrid {
  const frame = makeRecipeFrame(values['coordinateSystem'] as CatalogCoordinateSystem, [
    inputs['facilities'],
    inputs['observations']
  ]);
  const facilities = readRecipePoints(inputs['facilities'], '', frame);
  const observations = readRecipePoints(
    inputs['observations'],
    (values['valueProperty'] as string).trim(),
    frame
  );
  // At least one facility is needed to allocate cells.
  if (!facilities.count) throw new RangeError('No facility points');
  const cellSize = values['cellSize'] as number;
  let extentX = 0;
  let extentY = 0;
  for (const set of [facilities, observations])
    for (let index = 0; index < set.count; index++) {
      extentX = Math.max(extentX, set.positions[2 * index]);
      extentY = Math.max(extentY, set.positions[2 * index + 1]);
    }
  const grid = makeGridSpec(extentX, extentY, cellSize);
  const inverse = Math.fround(1 / Math.fround(cellSize));
  const getCell = (value: number, limit: number) =>
    Math.min(limit - 1, Math.max(0, Math.floor(Math.fround(value * inverse))));
  const seedColumns = new Int32Array(facilities.count);
  const seedRows = new Int32Array(facilities.count);
  for (let index = 0; index < facilities.count; index++) {
    seedColumns[index] = getCell(facilities.positions[2 * index], grid.columns);
    seedRows[index] = getCell(facilities.positions[2 * index + 1], grid.rows);
  }
  const sums = new Float64Array(grid.columns * grid.rows);
  const counts = new Uint32Array(grid.columns * grid.rows);
  for (let index = 0; index < observations.count; index++) {
    const value = observations.values[index];
    if (!Number.isFinite(value)) continue;
    const cell =
      getCell(observations.positions[2 * index + 1], grid.rows) * grid.columns +
      getCell(observations.positions[2 * index], grid.columns);
    sums[cell] += value;
    counts[cell]++;
  }
  const cellValues = new Float32Array(sums.length);
  for (let cell = 0; cell < cellValues.length; cell++)
    cellValues[cell] = counts[cell] ? sums[cell] / counts[cell] : Number.NaN;
  return {
    columns: grid.columns,
    rows: grid.rows,
    cellSize,
    facilities,
    seedColumns,
    seedRows,
    cellValues
  };
}

function computeCatchmentsCpu(grid: CatchmentGrid): CatchmentTotals {
  const count = grid.facilities.count;
  const totals: CatchmentTotals = {
    allocatedCellCount: 0,
    cellCounts: new Array<number>(count).fill(0),
    valueCounts: new Array<number>(count).fill(0),
    sums: new Array<number>(count).fill(0),
    means: new Array<number>(count).fill(Number.NaN),
    minimums: new Array<number>(count).fill(Number.NaN),
    maximums: new Array<number>(count).fill(Number.NaN)
  };
  for (let row = 0; row < grid.rows; row++)
    for (let column = 0; column < grid.columns; column++) {
      let best = -1;
      let bestDistance = Infinity;
      for (let facility = 0; facility < count; facility++) {
        const distance =
          (column - grid.seedColumns[facility]) ** 2 + (row - grid.seedRows[facility]) ** 2;
        if (distance < bestDistance) {
          bestDistance = distance;
          best = facility;
        }
      }
      totals.allocatedCellCount++;
      totals.cellCounts[best]++;
      const value = grid.cellValues[row * grid.columns + column];
      if (!Number.isFinite(value)) continue;
      totals.valueCounts[best]++;
      totals.sums[best] += value;
      totals.minimums[best] = Number.isNaN(totals.minimums[best])
        ? value
        : Math.min(totals.minimums[best], value);
      totals.maximums[best] = Number.isNaN(totals.maximums[best])
        ? value
        : Math.max(totals.maximums[best], value);
    }
  totals.means = totals.sums.map((sum, facility) =>
    totals.valueCounts[facility] ? sum / totals.valueCounts[facility] : Number.NaN
  );
  return totals;
}

async function computeCatchmentsGpu(device: Device, grid: CatchmentGrid): Promise<CatchmentTotals> {
  const raster = makeGPURasterMetadata({
    width: grid.columns,
    height: grid.rows,
    cellSize: [grid.cellSize, grid.cellSize],
    origin: [0, 0],
    rowDirection: 'north'
  });
  return withCatalogGpuSession(device, 'catalog-recipe-catchments', async session => {
    const count = grid.facilities.count;
    const cellCount = grid.columns * grid.rows;
    const allocation = session.output('allocation', 'uint32', cellCount);
    const distances = session.output('distances', 'float32', cellCount);
    const cellCounts = session.output('cell-counts', 'uint32', count);
    const valueCounts = session.output('value-counts', 'uint32', count);
    const sums = session.output('sums', 'float32', count);
    const means = session.output('means', 'float32', count);
    const minimums = session.output('minimums', 'float32', count);
    const maximums = session.output('maximums', 'float32', count);
    const overflow = session.output('overflow', 'uint32', 1);
    addStraightLineCatchmentsRecipe(session.graph, {
      id: 'catalog-recipe-catchments',
      grid: raster,
      parameters: session.input(
        'parameters',
        getGPUDistanceFieldParameterValues({}, raster),
        'float32'
      ),
      facilities: {
        kind: 'points',
        positions: session.input('facilities', grid.facilities.positions, 'float32x2')
      },
      values: session.input('cell-values', grid.cellValues, 'float32'),
      outputs: {
        allocation: allocation.view,
        distances: distances.view,
        statistics: {
          cellCounts: cellCounts.view,
          valueCounts: valueCounts.view,
          sums: sums.view,
          means: means.view,
          minimums: minimums.view,
          maximums: maximums.view,
          status: {overflow: overflow.view}
        }
      }
    });
    session.run();
    // A zone id past the capacity would drop cells from the statistics.
    if (await session.readScalar(overflow))
      throw new CatalogGpuUnsupportedError('The GPU zonal statistics overflowed');
    const cellCountValues = Array.from(await session.readUint32(cellCounts));
    return {
      allocatedCellCount: cellCountValues.reduce((sum, value) => sum + value, 0),
      cellCounts: cellCountValues,
      valueCounts: Array.from(await session.readUint32(valueCounts)),
      sums: Array.from(await session.readFloat32(sums)),
      means: Array.from(await session.readFloat32(means)),
      minimums: Array.from(await session.readFloat32(minimums)),
      maximums: Array.from(await session.readFloat32(maximums))
    };
  });
}

function buildCatchmentsOutput(
  inputs: Inputs,
  grid: CatchmentGrid,
  totals: CatchmentTotals,
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  addDroppedPoints(diagnostics, grid.facilities.droppedCount);
  const table = makeCatalogTable(
    'recipe-catchments',
    'Straight line catchments',
    {
      facility_index: 'int32',
      cell_count: 'int32',
      value_count: 'int32',
      sum: 'float64',
      mean: 'float64',
      minimum: 'float64',
      maximum: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  for (let facility = 0; facility < grid.facilities.count; facility++)
    appendCatalogRow(table, {
      facility_index: facility,
      cell_count: totals.cellCounts[facility],
      value_count: totals.valueCounts[facility],
      sum: totals.sums[facility],
      mean: nullIfNaN(totals.means[facility]),
      minimum: nullIfNaN(totals.minimums[facility]),
      maximum: nullIfNaN(totals.maximums[facility]),
      geometry: getFeatureGeometryText(
        inputs['facilities'].features[grid.facilities.sourceIndex[facility]]
      )
    });
  return {
    tables: [table],
    diagnostics,
    summary: {
      facilityCount: grid.facilities.count,
      columns: grid.columns,
      rows: grid.rows,
      cellSize: grid.cellSize,
      allocatedCellCount: totals.allocatedCellCount
    },
    ...(dispatchedContributors ? {dispatchedContributors} : {})
  };
}

const CATCHMENTS_BINDING: CatalogOperationBinding = {
  id: 'recipe-straight-line-catchments',
  title: 'Straight line catchments',
  domain: 'raster',
  summary:
    'Allocates every cell of a grid to its nearest facility (raster Voronoi) and summarises an observation value per catchment.',
  capabilityIds: ['zonal-and-region-statistics', 'spatial-joins'],
  contributorNames: ['GPUDistanceField', 'GPURasterZonalStatistics'],
  gpuContributorNames: ['GPUDistanceField', 'GPURasterZonalStatistics'],
  engine: 'gpu-or-cpu',
  inputs: [
    makeInputSpec('facilities', 'Facilities', ['point'], 1000, 1000),
    makeInputSpec(
      'observations',
      'Observations',
      ['point'],
      MAXIMUM_RECIPE_POINTS,
      MAXIMUM_RECIPE_POINTS
    )
  ],
  parameters: [
    makeStringParameter('valueProperty', 'Observation value property'),
    makeNumberParameter('cellSize', 'Cell size', 10, 1e-6, 1e7, 'crs'),
    COORDINATE_SYSTEM_PARAMETER
  ],
  outputs: [
    makeOutputSpec(
      'recipe-catchments',
      'Straight line catchments',
      'point',
      ['facility_index'],
      1000
    )
  ],
  cleanup: RECIPE_CLEANUP,
  declaredBehavior: [
    'The grid covers the local frame from the data minimum to the maximum of facilities and observations (capped at 250,000 cells); the value raster is the mean observation value per cell (NaN where a cell has none), built on the CPU.',
    'Facilities snap to their cell and every cell goes to the nearest facility cell by center distance; ties and facilities sharing a cell resolve to the smaller facility index, so the other facility gets an empty catchment.',
    'Distances are planar in the local meter frame (azimuthal equidistant around the data center for EPSG:4326), so large extents are approximate.',
    'Features without a point are skipped.'
  ],
  limitation:
    'Allocation and statistics only: the per-cell allocation and distance rasters are not published, and the value raster is the observation mean per cell rather than an uploaded raster tile.',
  run: (inputs, values) => {
    const grid = readCatchmentGrid(inputs, values);
    return buildCatchmentsOutput(inputs, grid, computeCatchmentsCpu(grid));
  },
  runGpu: async (context, inputs, values) => {
    const grid = readCatchmentGrid(inputs, values);
    return buildCatchmentsOutput(inputs, grid, await computeCatchmentsGpu(context.device, grid), [
      'GPUDistanceField',
      'GPURasterZonalStatistics'
    ]);
  }
};

export const PLATFORM5_RECIPE_BINDINGS: readonly CatalogOperationBinding[] = [
  CHOROPLETH_BINDING,
  ZONAL_DISTRIBUTION_BINDING,
  CATCHMENTS_BINDING,
  ...PLATFORM5_RECIPE_CHANGE_OF_SUPPORT_BINDINGS,
  ...PLATFORM5_RECIPE_HOT_SPOT_BINDINGS
];
