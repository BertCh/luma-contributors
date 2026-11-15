// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import type {
  CatalogOperationBinding,
  CatalogParameterSpec,
  CatalogRunOutput
} from './catalog-binding-types';
import type {CatalogCoordinateSystem} from './catalog-geojson';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getFeatureIdentity,
  makeCatalogTable
} from './catalog-geojson';
import {
  COORDINATE_SYSTEM_PARAMETER,
  PERMUTATION_PARAMETER_SPEC,
  PLATFORM3_CPU_CLEANUP,
  PLATFORM3_GPU_OR_CPU_CLEANUP,
  SEED_PARAMETER_SPEC,
  makeInputSpec,
  makeNumberParameter,
  makeOutputSpec,
  makeStringParameter,
  type Inputs,
  type Values
} from './catalog-platform3-b-common';
import {
  computeDbscanLabels,
  computeDbscanLabelsGpu,
  makeClusterOutput,
  readClusterSamples,
  runHierarchicalDensityClustering,
  runLocalOutlierFactor,
  runLocationAllocation,
  runOptics,
  runSimilarLocations,
  runSkaterRegions
} from './catalog-platform3-b-clusters';
import {
  addGridDiagnostics,
  computeIdwField,
  computeIdwFieldGpu,
  computeKernelDensityField,
  computeKernelDensityFieldGpu,
  computeKrigingField,
  computeLineDensity,
  computeLineLengthPerPolygon,
  computePointDensityField,
  computePointDensityFieldGpu,
  makeGridRunOutput,
  makeGridTable,
  preparePointGrid,
  prepareKrigingGrid,
  resolveKrigingSettings,
  type GridField,
  type IdwSettings,
  type KernelDensitySettings,
  type PointDensityStatistic
} from './catalog-platform3-b-grids';
import {
  runEmpiricalBayesRates,
  runGlobalSpatialStatistics,
  runHotSpotGiStar,
  runKnoxTest,
  runLocalGeary,
  runLocalMoran,
  runMultipleTestingAdjustment,
  runSpatialWeights,
  type AdjustmentMethod,
  type Neighborhood,
  type WeightsSettings
} from './catalog-platform3-b-statistics';
import {
  runArealInterpolation,
  runCellAggregation,
  runCellCompare,
  runCellRollup,
  runZonalPointStatistics
} from './catalog-platform3-b-zones';

const CRS = COORDINATE_SYSTEM_PARAMETER;
const COORDINATE = (values: Values) => values['coordinateSystem'] as CatalogCoordinateSystem;

const POINTS_INPUT = makeInputSpec('points', 'Points', ['point'], 20000, 20000);

const GRID_BEHAVIOR = [
  'Cell size is meters for EPSG:4326 (local azimuthal equidistant frame around the data center) and coordinate units for planar.',
  'Features without a point position or numeric value are skipped and counted.',
  'A grid over 250,000 cells throws before any work; the CPU reference also throws when sample-times-cell work exceeds its bound.'
];

const WEIGHT_BEHAVIOR = [
  'Features without a numeric value or position are skipped and counted; neighbors come only from the remaining features.',
  'knn and distance-band use anchors (the point, or a polygon centroid) in local meters (EPSG:4326) or coordinate units (planar); queen and rook need polygons and match exact shared vertices (queen) or shared edges (rook).',
  'Features without neighbors (islands) get a zero lag and a warning; fewer than four features or constant values throw.'
];

export function makeWeightsParameters(
  neighborhoodDefault: Neighborhood = 'knn'
): CatalogParameterSpec[] {
  return [
    {
      name: 'neighborhood',
      label: 'Neighborhood',
      kind: 'enum',
      options: ['knn', 'distance-band', 'queen', 'rook'],
      defaultValue: neighborhoodDefault
    },
    makeNumberParameter('neighborCount', 'Nearest neighbors (knn)', 6, 1, 64, undefined, 'integer'),
    makeNumberParameter('distanceBand', 'Distance band', 1000, 1e-9, 1e9, 'crs'),
    CRS
  ];
}

export function readWeightsSettings(values: Values, rowStandardize: boolean): WeightsSettings {
  return {
    neighborhood: values['neighborhood'] as Neighborhood,
    neighborCount: values['neighborCount'] as number,
    distanceBand: values['distanceBand'] as number,
    rowStandardize,
    coordinateSystem: COORDINATE(values)
  };
}

export function requireValueProperty(values: Values, name = 'valueProperty'): string {
  const property = values[name] as string;
  // A value property is required by this operation.
  if (!property) throw new Error('Property required');
  return property;
}

export function guardContiguityInput(inputs: Inputs, settings: WeightsSettings): void {
  if (settings.neighborhood !== 'queen' && settings.neighborhood !== 'rook') return;
  // queen and rook neighborhoods are defined for polygons only.
  for (const feature of inputs['features']?.features ?? [])
    if (feature.geometry && !/Polygon/.test(feature.geometry.type))
      throw new Error('Polygons required');
}

const IDW_PARAMETERS: CatalogParameterSpec[] = [
  makeStringParameter('valueProperty', 'Value property'),
  makeNumberParameter('cellSize', 'Cell size', 100, 1e-6, 1e7, 'crs'),
  makeNumberParameter('searchRadius', 'Search radius', 1000, 1e-6, 1e9, 'crs'),
  makeNumberParameter('power', 'Distance power', 2, 0, 8),
  makeNumberParameter(
    'neighborCount',
    'Nearest-sample limit (0 = all)',
    0,
    0,
    64,
    undefined,
    'integer'
  ),
  makeNumberParameter(
    'minimumNeighborCount',
    'Minimum samples per cell',
    1,
    1,
    64,
    undefined,
    'integer'
  ),
  {name: 'includeEmptyCells', label: 'Publish nodata cells', kind: 'boolean', defaultValue: false},
  CRS
];

function readIdwSettings(values: Values): IdwSettings {
  return {
    searchRadius: values['searchRadius'] as number,
    power: values['power'] as number,
    neighborCount: values['neighborCount'] as number,
    minimumNeighborCount: values['minimumNeighborCount'] as number
  };
}

export function finishGrid(
  context: ReturnType<typeof preparePointGrid>,
  field: GridField,
  tableId: string,
  label: string,
  valueColumn: string,
  isEmpty: (index: number) => boolean,
  includeEmptyCells: boolean,
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  const table = makeGridTable(context.frame, context.grid, field, {
    tableId,
    label,
    valueColumn,
    includeEmptyCells,
    isEmpty
  });
  addGridDiagnostics(
    context.diagnostics,
    context.grid.columns * context.grid.rows - table.rowCount,
    'Empty or nodata cells are not published unless includeEmptyCells is set.'
  );
  return makeGridRunOutput(
    table,
    context.diagnostics,
    context.grid,
    context.sample.count,
    dispatchedContributors
  );
}

const IDW_BINDING: CatalogOperationBinding = {
  id: 'idw-grid',
  title: 'Inverse distance weighting grid',
  domain: 'interpolation',
  summary:
    'Interpolates a numeric point property onto a regular grid with inverse distance weighting (power, radius, nearest-k).',
  capabilityIds: ['density-and-interpolation'],
  contributorNames: ['GPUInverseDistanceWeighting'],
  gpuContributorNames: ['GPUInverseDistanceWeighting'],
  engine: 'gpu-or-cpu',
  inputs: [POINTS_INPUT],
  parameters: IDW_PARAMETERS,
  outputs: [
    makeOutputSpec(
      'idw-cells',
      'Interpolated cells',
      'polygon',
      ['cell_id', 'column', 'row'],
      250000
    )
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    ...GRID_BEHAVIOR,
    'Cells are sampled at their centers; a sample exactly at the center returns its value; cells with fewer than minimumNeighborCount samples inside the radius are nodata and not published.',
    'The GPU engine needs a finite radius and throws CatalogGpuUnsupportedError otherwise; it runs in float32 local coordinates, so values match the CPU engine to float32 rounding.'
  ],
  limitation:
    'Planar distances in a local frame; no anisotropy, barriers or variance. The host IDW panel remains the interactive path.',
  run: (inputs, values) => {
    const context = preparePointGrid(
      inputs['points'],
      requireValueProperty(values),
      COORDINATE(values),
      values['cellSize'] as number
    );
    const field = computeIdwField(context.sample, context.grid, readIdwSettings(values));
    return finishGrid(
      context,
      field,
      'idw-cells',
      'Interpolated cells',
      'value',
      index => !Number.isFinite(field.values[index]),
      values['includeEmptyCells'] as boolean
    );
  },
  runGpu: async (gpuContext, inputs, values) => {
    const context = preparePointGrid(
      inputs['points'],
      requireValueProperty(values),
      COORDINATE(values),
      values['cellSize'] as number
    );
    const field = await computeIdwFieldGpu(
      gpuContext,
      context.sample,
      context.grid,
      readIdwSettings(values)
    );
    return finishGrid(
      context,
      field,
      'idw-cells',
      'Interpolated cells',
      'value',
      index => !Number.isFinite(field.values[index]),
      values['includeEmptyCells'] as boolean,
      ['GPUInverseDistanceWeighting']
    );
  }
};

const KDE_PARAMETERS: CatalogParameterSpec[] = [
  makeStringParameter('weightProperty', 'Weight property (optional)'),
  makeNumberParameter('cellSize', 'Cell size', 100, 1e-6, 1e7, 'crs'),
  makeNumberParameter('bandwidth', 'Bandwidth', 500, 1e-6, 1e9, 'crs'),
  {
    name: 'kernel',
    label: 'Kernel',
    kind: 'enum',
    options: ['gaussian', 'quartic', 'epanechnikov'],
    defaultValue: 'gaussian'
  },
  makeNumberParameter('padding', 'Grid padding around the points', 0, 0, 1e9, 'crs'),
  {
    name: 'includeEmptyCells',
    label: 'Publish zero-density cells',
    kind: 'boolean',
    defaultValue: false
  },
  CRS
];

function readKernelSettings(values: Values): KernelDensitySettings {
  return {
    bandwidth: values['bandwidth'] as number,
    kernel: values['kernel'] as KernelDensitySettings['kernel']
  };
}

const KDE_BINDING: CatalogOperationBinding = {
  id: 'kernel-density-grid',
  title: 'Kernel density grid',
  domain: 'density',
  summary:
    'Kernel density estimate of points (optionally weighted) on a regular grid, as intensity per square meter.',
  capabilityIds: ['density-and-interpolation'],
  contributorNames: ['GPUKernelDensity'],
  gpuContributorNames: ['GPUKernelDensity'],
  engine: 'gpu-or-cpu',
  inputs: [POINTS_INPUT],
  parameters: KDE_PARAMETERS,
  outputs: [
    makeOutputSpec(
      'density-cells',
      'Density cells',
      'polygon',
      ['cell_id', 'column', 'row'],
      250000
    )
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    ...GRID_BEHAVIOR,
    'Density is the weight sum per square meter (coordinate units squared for planar); the gaussian kernel is truncated at five bandwidths; negative weights are excluded.',
    'No edge correction: density falls off near the grid boundary unless padding covers the bandwidth.'
  ],
  limitation:
    'Fixed bandwidth, exact gather and no edge correction or study-window polygon; zero-density cells are not published by default.',
  run: (inputs, values) => {
    const context = preparePointGrid(
      inputs['points'],
      values['weightProperty'] as string,
      COORDINATE(values),
      values['cellSize'] as number,
      values['padding'] as number
    );
    const field = computeKernelDensityField(
      context.sample,
      context.grid,
      readKernelSettings(values)
    );
    return finishGrid(
      context,
      field,
      'density-cells',
      'Density cells',
      'density',
      index => !(field.values[index] > 0),
      values['includeEmptyCells'] as boolean
    );
  },
  runGpu: async (gpuContext, inputs, values) => {
    const context = preparePointGrid(
      inputs['points'],
      values['weightProperty'] as string,
      COORDINATE(values),
      values['cellSize'] as number,
      values['padding'] as number
    );
    const field = await computeKernelDensityFieldGpu(
      gpuContext,
      context.sample,
      context.grid,
      readKernelSettings(values),
      Boolean(values['weightProperty'])
    );
    return finishGrid(
      context,
      field,
      'density-cells',
      'Density cells',
      'density',
      index => !(field.values[index] > 0),
      values['includeEmptyCells'] as boolean,
      ['GPUKernelDensity']
    );
  }
};

const POINT_DENSITY_PARAMETERS: CatalogParameterSpec[] = [
  makeStringParameter('weightProperty', 'Weight property (sum and mean)'),
  makeNumberParameter('cellSize', 'Cell size', 100, 1e-6, 1e7, 'crs'),
  {
    name: 'statistic',
    label: 'Statistic',
    kind: 'enum',
    options: ['count', 'sum', 'mean'],
    defaultValue: 'count'
  },
  {name: 'includeEmptyCells', label: 'Publish empty cells', kind: 'boolean', defaultValue: false},
  CRS
];

function preparePointDensity(inputs: Inputs, values: Values) {
  const statistic = values['statistic'] as PointDensityStatistic;
  const weightProperty = values['weightProperty'] as string;
  // sum and mean statistics need a weight property.
  if (statistic !== 'count' && !weightProperty) throw new Error('Property required');
  return {
    statistic,
    weighted: statistic !== 'count',
    context: preparePointGrid(
      inputs['points'],
      statistic === 'count' ? '' : weightProperty,
      COORDINATE(values),
      values['cellSize'] as number
    )
  };
}

const POINT_DENSITY_BINDING: CatalogOperationBinding = {
  id: 'point-density-grid',
  title: 'Point density grid',
  domain: 'density',
  summary: 'Bins points into a square grid with count, weighted sum or weighted mean per cell.',
  capabilityIds: ['density-and-interpolation'],
  contributorNames: ['GPUPointDensity'],
  gpuContributorNames: ['GPUPointDensity'],
  engine: 'gpu-or-cpu',
  inputs: [POINTS_INPUT],
  parameters: POINT_DENSITY_PARAMETERS,
  outputs: [
    makeOutputSpec(
      'density-cells',
      'Density cells',
      'polygon',
      ['cell_id', 'column', 'row'],
      250000
    )
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    ...GRID_BEHAVIOR,
    'Values are per-cell statistics, not intensity per area; they change with the cell size.',
    'Points on a cell boundary bin by float32 arithmetic on the GPU and by float64 on the CPU, so boundary points may differ between engines.'
  ],
  limitation: 'Square cells only; no smoothing, hexagons or histogram outputs.',
  run: (inputs, values) => {
    const {context, statistic} = preparePointDensity(inputs, values);
    const field = computePointDensityField(context.sample, context.grid, statistic);
    return finishGrid(
      context,
      field,
      'density-cells',
      'Density cells',
      'value',
      index => !field.counts![index],
      values['includeEmptyCells'] as boolean
    );
  },
  runGpu: async (gpuContext, inputs, values) => {
    const {context, statistic, weighted} = preparePointDensity(inputs, values);
    const field = await computePointDensityFieldGpu(
      gpuContext,
      context.sample,
      context.grid,
      statistic,
      weighted
    );
    return finishGrid(
      context,
      field,
      'density-cells',
      'Density cells',
      'value',
      index => !field.counts![index],
      values['includeEmptyCells'] as boolean,
      ['GPUPointDensity']
    );
  }
};

const LINE_DENSITY_BINDING: CatalogOperationBinding = {
  id: 'line-density-grid',
  title: 'Line density grid',
  domain: 'density',
  summary: 'Length of line per unit area in each grid cell, splitting segments at cell borders.',
  capabilityIds: ['density-and-interpolation'],
  contributorNames: ['GPULineDensity'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [makeInputSpec('lines', 'Lines', ['line'], 20000, 200000)],
  parameters: [
    makeNumberParameter('cellSize', 'Cell size', 100, 1e-6, 1e7, 'crs'),
    {name: 'includeEmptyCells', label: 'Publish empty cells', kind: 'boolean', defaultValue: false},
    CRS
  ],
  outputs: [
    makeOutputSpec(
      'line-density-cells',
      'Line density cells',
      'polygon',
      ['cell_id', 'column', 'row'],
      250000
    )
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    'Density is line length divided by cell area (1/m for EPSG:4326 local meters, 1/unit for planar).',
    'Non-line features and degenerate lines are skipped and counted; segments are split exactly at cell borders.',
    'A grid over 250,000 cells throws before any work.'
  ],
  limitation: 'Unsmoothed per-cell length density; no search-radius kernel.',
  run: (inputs, values) => {
    const result = computeLineDensity(
      inputs['lines'],
      COORDINATE(values),
      values['cellSize'] as number
    );
    const table = makeGridTable(result.frame, result.grid, result.field, {
      tableId: 'line-density-cells',
      label: 'Line density cells',
      valueColumn: 'density',
      includeEmptyCells: values['includeEmptyCells'] as boolean,
      isEmpty: index => !(result.lengths[index] > 0)
    });
    return makeGridRunOutput(table, result.diagnostics, result.grid, result.lengths.length);
  }
};

const LINE_LENGTH_BINDING: CatalogOperationBinding = {
  id: 'line-length-per-polygon',
  title: 'Line length per polygon',
  domain: 'aggregation',
  summary: 'Total length of lines inside each polygon, with line counts and length density.',
  capabilityIds: ['density-and-interpolation'],
  contributorNames: ['GPULineLengthPerPolygon'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [
    makeInputSpec('polygons', 'Polygons', ['polygon'], 2000, 50000),
    makeInputSpec('lines', 'Lines', ['line'], 5000, 100000)
  ],
  parameters: [CRS],
  outputs: [
    makeOutputSpec(
      'polygon-lengths',
      'Polygon line lengths',
      'polygon',
      ['polygon_index', 'polygon_id'],
      2000
    )
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    'Lines are cut exactly at polygon edges; the inside parts are measured in local meters (EPSG:4326) or coordinate units (planar), holes excluded.',
    'Overlapping polygons each count the shared length; line_count is the number of source lines with a part inside.',
    'Segment-times-edge work above 60 million throws.'
  ],
  limitation: 'Straight-line segments in a local planar frame; no geodesic lengths.',
  run: (inputs, values) => {
    const result = computeLineLengthPerPolygon(
      inputs['polygons'],
      inputs['lines'],
      COORDINATE(values)
    );
    const table = makeCatalogTable(
      'polygon-lengths',
      'Polygon line lengths',
      {
        polygon_index: 'int32',
        polygon_id: 'string',
        line_length: 'float64',
        line_count: 'int32',
        area: 'float64',
        length_density: 'float64',
        geometry: 'string'
      },
      'geometry'
    );
    result.regions.forEach((region, index) => {
      const feature = inputs['polygons'].features[region.featureIndex];
      appendCatalogRow(table, {
        polygon_index: index,
        polygon_id: getFeatureIdentity(feature),
        line_length: result.lengths[index],
        line_count: result.lineCounts[index],
        area: region.area,
        length_density: region.area > 0 ? result.lengths[index] / region.area : null,
        geometry: feature.geometry ? JSON.stringify(feature.geometry) : null
      });
    });
    return {
      tables: [table],
      diagnostics: result.diagnostics,
      summary: {
        polygonCount: table.rowCount,
        totalLength: result.lengths.reduce((sum, length) => sum + length, 0)
      }
    };
  }
};

const KRIGING_BINDING: CatalogOperationBinding = {
  id: 'kriging-grid',
  title: 'Ordinary kriging grid',
  domain: 'interpolation',
  summary:
    'Ordinary kriging of a numeric point property with a stated variogram, producing value and kriging variance per grid cell.',
  capabilityIds: ['density-and-interpolation', 'coverage-local-prediction'],
  contributorNames: ['GPUKriging'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [makeInputSpec('points', 'Points', ['point'], 2000, 2000)],
  parameters: [
    makeStringParameter('valueProperty', 'Value property'),
    makeNumberParameter('cellSize', 'Cell size', 100, 1e-6, 1e7, 'crs'),
    {
      name: 'model',
      label: 'Variogram model',
      kind: 'enum',
      options: ['spherical', 'exponential', 'gaussian'],
      defaultValue: 'spherical'
    },
    makeNumberParameter(
      'nugget',
      'Nugget (default: nearest-neighbor estimate)',
      undefined,
      0,
      1e12
    ),
    makeNumberParameter(
      'sill',
      'Partial sill (default: sample variance minus nugget)',
      undefined,
      0,
      1e12
    ),
    makeNumberParameter(
      'range',
      'Range (default: extent diagonal / 3)',
      undefined,
      1e-9,
      1e9,
      'crs'
    ),
    makeNumberParameter(
      'neighborCount',
      'Nearest samples per cell',
      12,
      1,
      32,
      undefined,
      'integer'
    ),
    makeNumberParameter('searchRadius', 'Search radius (0 = unlimited)', 0, 0, 1e9, 'crs'),
    {
      name: 'includeEmptyCells',
      label: 'Publish nodata cells',
      kind: 'boolean',
      defaultValue: false
    },
    CRS
  ],
  outputs: [
    makeOutputSpec('kriging-cells', 'Kriged cells', 'polygon', ['cell_id', 'column', 'row'], 40000)
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    ...GRID_BEHAVIOR,
    'Omitted nugget defaults to the mean nearest-neighbor semivariance, capped at sample variance; omitted sill defaults to sample variance minus nugget, and omitted range uses one third of the extent diagonal. Explicit nonnegative nugget and sill (including zero), and positive range override these estimates.',
    'Cells whose kriging system is singular (for example coincident samples) are nodata and counted in a diagnostic; at most 2,000 samples, 40,000 cells and 60,000,000 cell-point work on the CPU. Increase cellSize or reduce points when the work limit is exceeded.'
  ],
  limitation:
    'Local neighborhood with an isotropic moment-estimated or explicit variogram; no variogram curve fitting, trend, anisotropy or cross-validation. Kriging variance depends on this estimated model.',
  run: (inputs, values) => {
    const context = prepareKrigingGrid(
      inputs['points'],
      requireValueProperty(values),
      COORDINATE(values),
      values['cellSize'] as number
    );
    const settings = resolveKrigingSettings(context.sample, values);
    const field = computeKrigingField(context.sample, context.grid, settings);
    if (field.singularCount)
      addCatalogDiagnostic(
        context.diagnostics,
        'singular-kriging-system',
        'warning',
        'Cells with a singular kriging system are nodata.',
        field.singularCount
      );
    return finishGrid(
      context,
      field,
      'kriging-cells',
      'Kriged cells',
      'value',
      index => !Number.isFinite(field.values[index]),
      values['includeEmptyCells'] as boolean
    );
  }
};

const LATTICE_NOTE =
  'Cells are a square lattice floor(coordinate / cellSize) in the input coordinates (degrees for EPSG:4326 data), not an H3, S2 or A5 index; cell ids are "x:y" lattice indices.';

const CELL_STATISTIC_PARAMETERS: CatalogParameterSpec[] = [
  makeStringParameter('valueProperty', 'Value property (empty = count only)'),
  makeNumberParameter(
    'cellSize',
    'Cell size (coordinate units)',
    0.01,
    1e-9,
    1e7,
    'coordinate-units'
  )
];

const CELL_AGGREGATION_BINDING: CatalogOperationBinding = {
  id: 'cell-aggregation',
  title: 'Lattice cell aggregation',
  domain: 'aggregation',
  summary:
    'Counts and summarizes a numeric point property per square lattice cell (count, sum, mean, min, max).',
  capabilityIds: ['dggs-aggregation'],
  contributorNames: ['GPUCellAggregation'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [POINTS_INPUT],
  parameters: CELL_STATISTIC_PARAMETERS,
  outputs: [
    makeOutputSpec('cells', 'Aggregated cells', 'polygon', ['cell_id', 'cell_x', 'cell_y'], 100000)
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    LATTICE_NOTE,
    'Features without a point position are skipped and counted; only occupied cells are published. Every point counts, including points without a numeric valueProperty; sum, mean, min and max use only numeric values, with null mean/min/max when none are valid. The summary and diagnostic report missingValueCount.',
    'Output over 100,000 cells throws.'
  ],
  limitation:
    'Square lattice stand-in for a discrete global grid; cells are not equal-area on EPSG:4326 data.',
  run: (inputs, values) =>
    runCellAggregation(inputs['points'], {
      cellSize: values['cellSize'] as number,
      valueProperty: values['valueProperty'] as string
    })
};

const CELL_ROLLUP_BINDING: CatalogOperationBinding = {
  id: 'cell-rollup',
  title: 'Lattice cell rollup pyramid',
  domain: 'aggregation',
  summary:
    'Aggregates points at a base cell size and at coarser levels, with parent cell links between levels.',
  capabilityIds: ['dggs-aggregation'],
  contributorNames: ['GPUCellRollup', 'GPUCellPyramid'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [POINTS_INPUT],
  parameters: [
    ...CELL_STATISTIC_PARAMETERS,
    makeNumberParameter('levels', 'Coarser levels', 3, 1, 6, undefined, 'integer'),
    makeNumberParameter('factor', 'Cell size factor per level', 2, 2, 10, undefined, 'integer')
  ],
  outputs: [
    makeOutputSpec(
      'rollup',
      'Rolled-up cells',
      'polygon',
      ['cell_id', 'parent_cell_id', 'level'],
      200000
    )
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    LATTICE_NOTE,
    'Level n uses cell size cellSize * factor^n; the parent of a cell is floor(index / factor) at the next level, so children nest exactly.',
    'Output over 200,000 rows throws.'
  ],
  limitation:
    'Integer nesting factors only; each level is recomputed from the points, not from the finer level.',
  run: (inputs, values) =>
    runCellRollup(inputs['points'], {
      cellSize: values['cellSize'] as number,
      valueProperty: values['valueProperty'] as string,
      levels: values['levels'] as number,
      factor: values['factor'] as number
    })
};

const CELL_COMPARE_BINDING: CatalogOperationBinding = {
  id: 'cell-compare',
  title: 'Lattice cell comparison',
  domain: 'aggregation',
  summary: 'Compares per-cell count, sum or mean between two point sets (before and after).',
  capabilityIds: ['dggs-outlines-and-change'],
  contributorNames: ['GPUCellTableCompare'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [
    makeInputSpec('before', 'Before points', ['point'], 20000, 20000),
    makeInputSpec('after', 'After points', ['point'], 20000, 20000)
  ],
  parameters: [
    makeStringParameter('valueProperty', 'Value property (empty = count only)'),
    makeNumberParameter(
      'cellSize',
      'Cell size (coordinate units)',
      0.01,
      1e-9,
      1e7,
      'coordinate-units'
    ),
    {
      name: 'statistic',
      label: 'Statistic',
      kind: 'enum',
      options: ['count', 'sum', 'mean'],
      defaultValue: 'count'
    }
  ],
  outputs: [
    makeOutputSpec(
      'comparison',
      'Cell comparison',
      'polygon',
      ['cell_id', 'cell_x', 'cell_y'],
      100000
    )
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    LATTICE_NOTE,
    'valueProperty is required for sum or mean; choose count to compare without a value column.',
    'Cells present in only one set are published with status appeared or disappeared; a missing mean is null, a missing count or sum is zero.',
    'relative_change is null when the before value is zero.'
  ],
  limitation: 'Compares cell statistics only; no significance test of the change.',
  run: (inputs, values) =>
    runCellCompare(inputs['before'], inputs['after'], {
      cellSize: values['cellSize'] as number,
      valueProperty: values['valueProperty'] as string,
      statistic: values['statistic'] as 'count' | 'sum' | 'mean'
    })
};

const ZONAL_BINDING: CatalogOperationBinding = {
  id: 'zonal-point-statistics',
  title: 'Zonal point statistics',
  domain: 'aggregation',
  summary:
    'Count, sum, mean, min, max and standard deviation of a point property inside each polygon.',
  capabilityIds: ['zonal-and-region-statistics'],
  contributorNames: ['GPURegionStatistics'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [
    makeInputSpec('polygons', 'Zones', ['polygon'], 5000, 100000),
    makeInputSpec('points', 'Points', ['point'], 50000, 50000)
  ],
  parameters: [makeStringParameter('valueProperty', 'Value property (empty = count only)')],
  outputs: [
    makeOutputSpec('zones', 'Zone statistics', 'polygon', ['polygon_index', 'polygon_id'], 5000)
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    'Every polygon gets a row, with count zero and null mean/min/max when no point falls inside.',
    'Points in overlapping zones count for each zone; points outside every zone are counted in a diagnostic; points on a boundary follow the even-odd ray rule.',
    'Features without a point position or numeric value are skipped and counted.'
  ],
  limitation:
    'Point-in-polygon statistics; raster zonal statistics are not bound here. Coordinates are tested as given (no projection).',
  run: (inputs, values) =>
    runZonalPointStatistics(inputs['polygons'], inputs['points'], {
      valueProperty: values['valueProperty'] as string
    })
};

const AREAL_BINDING: CatalogOperationBinding = {
  id: 'areal-interpolation',
  title: 'Areal interpolation',
  domain: 'interpolation',
  summary:
    'Transfers a polygon attribute from source zones to target zones by area-proportional overlap (extensive or intensive).',
  capabilityIds: ['change-of-support'],
  contributorNames: ['GPUArealInterpolation'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [
    makeInputSpec('sources', 'Source zones', ['polygon'], 1000, 100000),
    makeInputSpec('targets', 'Target zones', ['polygon'], 1000, 100000)
  ],
  parameters: [
    makeStringParameter('valueProperty', 'Source value property'),
    {
      name: 'variableType',
      label: 'Variable type (required: choose extensive or intensive)',
      kind: 'enum',
      options: ['', 'extensive', 'intensive'],
      defaultValue: ''
    },
    makeNumberParameter('samplesPerSide', 'Samples per side', 24, 4, 64, undefined, 'integer'),
    CRS
  ],
  outputs: [
    makeOutputSpec(
      'targets',
      'Interpolated targets',
      'polygon',
      ['target_index', 'target_id'],
      1000
    )
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    'variableType has no selected default: explicitly choose extensive for counts or intensive for rates before running.',
    'Extensive values (counts) are split by overlap fraction of each source; intensive values (rates) are the overlap-weighted mean.',
    'Overlaps are estimated from a regular sample grid inside each source polygon, so small slivers are approximate and sources smaller than one sample are skipped with a warning.',
    'Source area falling outside every target is lost for extensive values; coverage reports the covered fraction of each target.'
  ],
  limitation:
    'Sampled area weighting; no dasymetric ancillary data and no pycnophylactic smoothing.',
  run: (inputs, values) =>
    runArealInterpolation(inputs['sources'], inputs['targets'], {
      valueProperty: requireValueProperty(values),
      variableType: values['variableType'] as 'extensive' | 'intensive',
      samplesPerSide: values['samplesPerSide'] as number,
      coordinateSystem: COORDINATE(values)
    })
};

const WEIGHTS_BINDING: CatalogOperationBinding = {
  id: 'spatial-weights',
  title: 'Spatial weights',
  domain: 'analysis',
  summary:
    'Neighbor links (k nearest, distance band, queen or rook contiguity) as a link table with weights and distances.',
  capabilityIds: ['spatial-neighborhoods'],
  contributorNames: ['GPUNeighborSearch', 'GPUContiguityWeights'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [makeInputSpec('features', 'Features', ['point', 'polygon'], 3000, 200000)],
  parameters: [
    ...makeWeightsParameters(),
    {name: 'rowStandardize', label: 'Row-standardize weights', kind: 'boolean', defaultValue: true}
  ],
  outputs: [
    makeOutputSpec(
      'weights',
      'Weight links',
      'none',
      ['source_index', 'source_id', 'neighbor_index', 'neighbor_id'],
      200000
    )
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    ...WEIGHT_BEHAVIOR.slice(1),
    'Every link is directed from source to neighbor; knn is not symmetric. More than 3,000 features throw because distance search is O(n squared).'
  ],
  limitation:
    'Distance-based pairs are brute force on the CPU; use the host streamed weights for large datasets.',
  run: (inputs, values) => {
    const settings = readWeightsSettings(values, values['rowStandardize'] as boolean);
    guardContiguityInput(inputs, settings);
    return runSpatialWeights(inputs['features'], settings);
  }
};

const VALUE_AND_WEIGHTS: CatalogParameterSpec[] = [
  makeStringParameter('valueProperty', 'Value property'),
  ...makeWeightsParameters()
];

const GLOBAL_STATISTICS_BINDING: CatalogOperationBinding = {
  id: 'global-spatial-statistics',
  title: 'Global spatial autocorrelation',
  domain: 'analysis',
  summary: "Moran's I and Geary's C of a numeric property with analytic z-scores and p-values.",
  capabilityIds: ['spatial-autocorrelation-and-inference'],
  contributorNames: ['GPUGlobalSpatialStatistics'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [makeInputSpec('features', 'Features', ['point', 'polygon'], 3000, 200000)],
  parameters: VALUE_AND_WEIGHTS,
  outputs: [makeOutputSpec('global', 'Global statistics', 'none', ['statistic'], 2)],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    ...WEIGHT_BEHAVIOR,
    "Weights are row-standardized. Moran's I and Geary's C both use the randomization variance; p-values are two-sided normal approximations with no permutation test."
  ],
  limitation:
    'Two statistics with analytic inference; Getis-Ord G and permutation inference are not bound.',
  run: (inputs, values) => {
    const settings = readWeightsSettings(values, true);
    guardContiguityInput(inputs, settings);
    return runGlobalSpatialStatistics(inputs['features'], {
      valueProperty: requireValueProperty(values),
      ...settings
    });
  }
};

const LOCAL_MORAN_BINDING: CatalogOperationBinding = {
  id: 'local-moran-statistics',
  title: 'Local Moran clusters',
  domain: 'analysis',
  summary:
    'Local Moran I per feature with HH/LL/HL/LH quadrant and conditional-permutation pseudo p-values.',
  capabilityIds: ['spatial-autocorrelation-and-inference'],
  contributorNames: ['GPULocalMoran', 'GPULocalPermutationTest'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [makeInputSpec('features', 'Features', ['point', 'polygon'], 3000, 200000)],
  parameters: [
    ...VALUE_AND_WEIGHTS,
    {...PERMUTATION_PARAMETER_SPEC, defaultValue: 999},
    SEED_PARAMETER_SPEC,
    {
      name: 'multipleTesting',
      label: 'Multiple testing',
      kind: 'enum',
      options: ['none', 'benjamini-hochberg', 'bonferroni'],
      defaultValue: 'benjamini-hochberg'
    },
    makeNumberParameter('significance', 'Significance level', 0.05, 1e-6, 1)
  ],
  outputs: [
    makeOutputSpec('local-moran', 'Local Moran', 'none', ['source_index', 'source_id'], 3000)
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    ...WEIGHT_BEHAVIOR,
    'Binary neighbor links are row-standardized (each neighbor has weight 1 / neighbor count); values are standardized by the population standard deviation.',
    'The default is 999 seeded conditional permutations and Benjamini-Hochberg correction across tested rows. p_value reports raw pseudo p-values; adjusted_p_value and clusters use multipleTesting (none, Benjamini-Hochberg or Bonferroni); permutations=0 reports cluster "not-tested".',
    'The geometry column mirrors each source feature.'
  ],
  limitation:
    'Univariate local Moran on the CPU; the host Local Moran panel is the interactive path.',
  run: (inputs, values) => {
    const settings = readWeightsSettings(values, true);
    guardContiguityInput(inputs, settings);
    return runLocalMoran(inputs['features'], {
      valueProperty: requireValueProperty(values),
      permutations: values['permutations'] as number,
      seed: values['seed'] as number,
      significance: values['significance'] as number,
      multipleTesting: values['multipleTesting'] as 'none' | 'benjamini-hochberg' | 'bonferroni',
      ...settings
    });
  }
};

const GI_STAR_BINDING: CatalogOperationBinding = {
  id: 'hotspot-gi-star',
  title: 'Hot spot analysis (Gi*)',
  domain: 'analysis',
  summary:
    'Getis-Ord Gi* z-score per feature with hot and cold spot classes at 90, 95 and 99 percent.',
  capabilityIds: ['spatial-autocorrelation-and-inference'],
  contributorNames: ['GPUHotSpotAnalysis'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [makeInputSpec('features', 'Features', ['point', 'polygon'], 3000, 200000)],
  parameters: VALUE_AND_WEIGHTS,
  outputs: [
    makeOutputSpec('hotspots', 'Getis-Ord Gi*', 'none', ['source_index', 'source_id'], 3000)
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    ...WEIGHT_BEHAVIOR,
    'Features without neighbors are not tested (null z/p and classification not-tested), and counted in island-features. Binary weights including the feature itself; z-scores are normal approximations and classes use fixed two-sided thresholds 1.645, 1.96 and 2.576 with no multiple-testing correction.',
    'The geometry column mirrors each source feature.'
  ],
  limitation:
    'Analytic Gi* only; no permutation inference or false-discovery-rate correction (see multiple-testing-adjustment).',
  run: (inputs, values) => {
    const settings = readWeightsSettings(values, false);
    guardContiguityInput(inputs, settings);
    return runHotSpotGiStar(inputs['features'], {
      valueProperty: requireValueProperty(values),
      ...settings
    });
  }
};

const LOCAL_GEARY_BINDING: CatalogOperationBinding = {
  id: 'local-geary-statistics',
  title: 'Local Geary',
  domain: 'analysis',
  summary:
    'Local Geary C per feature (similarity to neighbors) with conditional-permutation pseudo p-values.',
  capabilityIds: ['local-statistics'],
  contributorNames: ['GPULocalGeary'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [makeInputSpec('features', 'Features', ['point', 'polygon'], 3000, 200000)],
  parameters: [...VALUE_AND_WEIGHTS, PERMUTATION_PARAMETER_SPEC, SEED_PARAMETER_SPEC],
  outputs: [
    makeOutputSpec('local-geary', 'Local Geary', 'none', ['source_index', 'source_id'], 3000)
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    ...WEIGHT_BEHAVIOR,
    'Row-standardized weights on standardized values; p-values are one-sided conditional-permutation pseudo p-values for small C and are null when permutations is 0.',
    'The geometry column mirrors each source feature.'
  ],
  limitation: 'Univariate local Geary; multivariate and cluster categories are not bound.',
  run: (inputs, values) => {
    const settings = readWeightsSettings(values, true);
    guardContiguityInput(inputs, settings);
    return runLocalGeary(inputs['features'], {
      valueProperty: requireValueProperty(values),
      permutations: values['permutations'] as number,
      seed: values['seed'] as number,
      ...settings
    });
  }
};

const EB_BINDING: CatalogOperationBinding = {
  id: 'empirical-bayes-rates',
  title: 'Empirical Bayes rate smoothing',
  domain: 'analysis',
  summary:
    'Shrinks small-population rates toward the overall rate (Marshall global empirical Bayes).',
  capabilityIds: ['spatial-autocorrelation-and-inference'],
  contributorNames: ['GPUEmpiricalBayesRates'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [makeInputSpec('features', 'Areas', ['point', 'polygon'], 20000, 200000)],
  parameters: [
    makeStringParameter('numeratorProperty', 'Events property'),
    makeStringParameter('denominatorProperty', 'Population property')
  ],
  outputs: [
    makeOutputSpec('rates', 'Empirical Bayes rates', 'none', ['source_index', 'source_id'], 20000)
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    'Areas with missing, negative-event or non-positive-population values are skipped and counted; events above population are rejected with row and property names; fewer than two usable areas throw.',
    'The prior variance is clamped at zero, which shrinks every rate fully to the overall rate.',
    'The geometry column mirrors each source feature.'
  ],
  limitation:
    'Global prior only; spatial empirical Bayes and Assuncao-Reis standardization are not bound.',
  run: (inputs, values) =>
    runEmpiricalBayesRates(inputs['features'], {
      numeratorProperty: requireValueProperty(values, 'numeratorProperty'),
      denominatorProperty: requireValueProperty(values, 'denominatorProperty')
    })
};

const MULTIPLE_TESTING_BINDING: CatalogOperationBinding = {
  id: 'multiple-testing-adjustment',
  title: 'Multiple-testing adjustment',
  domain: 'analysis',
  summary:
    'Adjusts a column of p-values (Bonferroni, Holm, Sidak or Benjamini-Hochberg) and flags rejections.',
  capabilityIds: ['spatial-autocorrelation-and-inference'],
  contributorNames: ['GPUMultipleTestingAdjustment'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [
    makeInputSpec('features', 'Features with p-values', ['point', 'line', 'polygon'], 20000, 200000)
  ],
  parameters: [
    makeStringParameter('pValueProperty', 'p-value property'),
    {
      name: 'method',
      label: 'Method',
      kind: 'enum',
      options: ['bonferroni', 'holm', 'benjamini-hochberg', 'sidak'],
      defaultValue: 'benjamini-hochberg'
    },
    makeNumberParameter('alpha', 'Significance level', 0.05, 1e-9, 1)
  ],
  outputs: [
    makeOutputSpec('adjusted', 'Adjusted p-values', 'none', ['source_index', 'source_id'], 20000)
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    'Features whose p-value is missing or outside [0, 1] are skipped and counted; no valid p-value throws.',
    'The number of tests is the number of valid p-values; Benjamini-Hochberg assumes independence or positive dependence.',
    'The geometry column mirrors each source feature.'
  ],
  limitation:
    'Operates on a p-value property carried by the input features, for example from a local statistic table.',
  run: (inputs, values) =>
    runMultipleTestingAdjustment(inputs['features'], {
      pValueProperty: requireValueProperty(values, 'pValueProperty'),
      method: values['method'] as AdjustmentMethod,
      alpha: values['alpha'] as number
    })
};

const KNOX_BINDING: CatalogOperationBinding = {
  id: 'knox-test',
  title: 'Knox space-time test',
  domain: 'analysis',
  summary:
    'Tests whether events close in space are also close in time, with a seeded permutation p-value.',
  capabilityIds: ['spatial-autocorrelation-and-inference'],
  contributorNames: ['GPUKnoxTest'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [makeInputSpec('points', 'Events', ['point'], 2000, 2000)],
  parameters: [
    makeStringParameter('timeProperty', 'Time property (numeric)'),
    makeNumberParameter(
      'spatialThreshold',
      'Spatial threshold (default: pairwise lower quartile)',
      undefined,
      0,
      1e9,
      'crs'
    ),
    makeNumberParameter(
      'temporalThreshold',
      'Temporal threshold (default: pairwise lower quartile)',
      undefined,
      0,
      1e15
    ),
    {...PERMUTATION_PARAMETER_SPEC, minimum: 1, defaultValue: 199, maximum: 999},
    SEED_PARAMETER_SPEC,
    CRS
  ],
  outputs: [makeOutputSpec('knox', 'Knox test', 'none', ['event_count'], 1)],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    'Events without a point or numeric time are skipped and counted; fewer than three events throw.',
    'Omitted spatialThreshold and temporalThreshold default to the lower quartile of pairwise distances and absolute time differences; explicit zero tests coincident positions or simultaneous events. Time thresholds use the numeric time column units (including epoch milliseconds); resolved thresholds are in the summary. Pairs closer than or equal to both thresholds are joint close pairs; the p-value is the upper tail of the joint count under seeded time permutations.',
    'Distances are meters for EPSG:4326 (local frame) and coordinate units for planar; at most 2,000 events.'
  ],
  limitation: 'Single pair of thresholds; no multi-threshold scan and no edge correction.',
  run: (inputs, values) =>
    runKnoxTest(inputs['points'], {
      timeProperty: requireValueProperty(values, 'timeProperty'),
      spatialThreshold: values['spatialThreshold'] as number,
      temporalThreshold: values['temporalThreshold'] as number,
      permutations: values['permutations'] as number,
      seed: values['seed'] as number,
      coordinateSystem: COORDINATE(values)
    })
};

const DBSCAN_PARAMETERS: CatalogParameterSpec[] = [
  makeNumberParameter('epsilon', 'Neighborhood radius', 200, 1e-9, 1e9, 'crs'),
  makeNumberParameter(
    'minimumPoints',
    'Minimum points (including the point)',
    4,
    1,
    1000,
    undefined,
    'integer'
  ),
  CRS
];

function makeDbscanOutput(
  inputs: Inputs,
  values: Values,
  result: {labels: ArrayLike<number>; coreFlags: ArrayLike<number>},
  context: ReturnType<typeof readClusterSamples>,
  dispatched?: readonly string[]
): CatalogRunOutput {
  return makeClusterOutput(
    inputs['points'],
    context.frame,
    context.sample,
    result.labels,
    result.coreFlags,
    context.diagnostics,
    {
      dispatchedContributors: dispatched,
      summary: {
        epsilon: values['epsilon'] as number,
        minimumPoints: values['minimumPoints'] as number
      }
    }
  );
}

const DBSCAN_BINDING: CatalogOperationBinding = {
  id: 'density-clustering',
  title: 'Density clustering (DBSCAN)',
  domain: 'clustering',
  summary: 'DBSCAN clusters and noise with core flags and cluster centroids.',
  capabilityIds: ['group-and-regional-analysis'],
  contributorNames: ['GPUSpatialClustering'],
  gpuContributorNames: ['GPUSpatialClustering'],
  engine: 'gpu-or-cpu',
  inputs: [makeInputSpec('points', 'Points', ['point'], 20000, 20000)],
  parameters: DBSCAN_PARAMETERS,
  outputs: [
    makeOutputSpec(
      'points',
      'Clustered points',
      'point',
      ['point_index', 'source_index', 'source_id'],
      20000
    ),
    makeOutputSpec('clusters', 'Clusters', 'point', ['cluster'], 20000)
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    'Neighborhoods include the point itself and use squared distance <= epsilon squared in local meters (EPSG:4326) or coordinate units (planar).',
    'Labels are canonical: clusters are numbered by their smallest core point; border points join the adjacent cluster with the smallest root; noise is cluster -1.',
    'Features without a point are skipped and counted; the CPU reference and GPU engine accept up to 20,000 points.',
    'Points exactly at distance epsilon can differ between engines because the GPU compares in float32.'
  ],
  limitation:
    'Planar DBSCAN in a local frame; no HDBSCAN stability selection (see hierarchical-density-clustering).',
  run: (inputs, values) => {
    const context = readClusterSamples(inputs['points'], COORDINATE(values));
    const result = computeDbscanLabels(
      context.sample,
      values['epsilon'] as number,
      values['minimumPoints'] as number
    );
    return makeDbscanOutput(inputs, values, result, context);
  },
  runGpu: async (gpuContext, inputs, values) => {
    const context = readClusterSamples(inputs['points'], COORDINATE(values));
    const result = await computeDbscanLabelsGpu(
      gpuContext,
      context.sample,
      values['epsilon'] as number,
      values['minimumPoints'] as number
    );
    return makeDbscanOutput(inputs, values, result, context, ['GPUSpatialClustering']);
  }
};

const OPTICS_BINDING: CatalogOperationBinding = {
  id: 'optics-ordering',
  title: 'OPTICS reachability ordering',
  domain: 'clustering',
  summary:
    'OPTICS ordering with reachability and core distances and a DBSCAN-style cluster extraction.',
  capabilityIds: ['coverage-point-processes'],
  contributorNames: ['GPUOPTICS'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [makeInputSpec('points', 'Points', ['point'], 3000, 3000)],
  parameters: [
    makeNumberParameter('maximumDistance', 'Maximum neighborhood distance', 1000, 1e-9, 1e9, 'crs'),
    makeNumberParameter(
      'minimumSamples',
      'Minimum samples (including the point)',
      5,
      1,
      1000,
      undefined,
      'integer'
    ),
    makeNumberParameter('extractionDistance', 'Extraction distance', 300, 1e-9, 1e9, 'crs'),
    CRS
  ],
  outputs: [
    makeOutputSpec(
      'points',
      'Ordered points',
      'point',
      ['point_index', 'source_index', 'source_id'],
      3000
    ),
    makeOutputSpec('clusters', 'Extracted clusters', 'point', ['cluster'], 3000)
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    'Distances are meters (EPSG:4326, local frame) or coordinate units; unreachable points have null reachability.',
    'optics_order is the position in the OPTICS ordering; extraction at extractionDistance <= maximumDistance reproduces DBSCAN clusters up to border-point assignment.',
    'Up to 3,000 points; features without a point are skipped and counted.'
  ],
  limitation: 'Brute-force O(n squared) CPU reference; xi cluster extraction is not bound.',
  run: (inputs, values) =>
    runOptics(inputs['points'], {
      maximumDistance: values['maximumDistance'] as number,
      minimumSamples: values['minimumSamples'] as number,
      extractionDistance: values['extractionDistance'] as number,
      coordinateSystem: COORDINATE(values)
    })
};

const LOF_BINDING: CatalogOperationBinding = {
  id: 'local-outlier-factor',
  title: 'Local outlier factor',
  domain: 'clustering',
  summary: 'Density-based outlier score per point (LOF) with an outlier flag at a threshold.',
  capabilityIds: ['group-and-regional-analysis'],
  contributorNames: ['GPULocalOutlierFactor'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [makeInputSpec('points', 'Points', ['point'], 3000, 3000)],
  parameters: [
    makeNumberParameter('neighborCount', 'Neighbors', 10, 1, 100, undefined, 'integer'),
    makeNumberParameter('threshold', 'Outlier threshold', 1.5, 1, 100),
    CRS
  ],
  outputs: [
    makeOutputSpec(
      'outliers',
      'Local outlier factor',
      'point',
      ['point_index', 'source_index', 'source_id'],
      3000
    )
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    'Fewer points than neighbors + 1 throws; coincident neighbors give infinite density and a score of one with a warning.',
    'Distances are meters (EPSG:4326, local frame) or coordinate units; up to 3,000 points.'
  ],
  limitation: 'Brute-force CPU reference; the threshold is a convention, not a significance test.',
  run: (inputs, values) =>
    runLocalOutlierFactor(inputs['points'], {
      neighborCount: values['neighborCount'] as number,
      threshold: values['threshold'] as number,
      coordinateSystem: COORDINATE(values)
    })
};

const HDBSCAN_BINDING: CatalogOperationBinding = {
  id: 'hierarchical-density-clustering',
  title: 'Mutual-reachability density clustering',
  domain: 'clustering',
  summary:
    'Clusters points by cutting the mutual-reachability spanning tree at a distance, dropping small clusters as noise.',
  capabilityIds: ['group-and-regional-analysis'],
  contributorNames: ['GPUHierarchicalDensityClustering'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [makeInputSpec('points', 'Points', ['point'], 3000, 3000)],
  parameters: [
    makeNumberParameter(
      'minimumSamples',
      'Core-distance neighbors',
      5,
      1,
      100,
      undefined,
      'integer'
    ),
    makeNumberParameter(
      'minimumClusterSize',
      'Minimum cluster size',
      5,
      1,
      1000,
      undefined,
      'integer'
    ),
    makeNumberParameter('cutDistance', 'Cut distance', 500, 1e-9, 1e9, 'crs'),
    CRS
  ],
  outputs: [
    makeOutputSpec(
      'points',
      'Clustered points',
      'point',
      ['point_index', 'source_index', 'source_id'],
      3000
    ),
    makeOutputSpec('clusters', 'Clusters', 'point', ['cluster'], 3000)
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    'Core distance is the distance to the minimumSamples-th other point; edge weights are max(core a, core b, distance).',
    'Clusters smaller than minimumClusterSize are noise (-1); up to 3,000 points; distances are meters (EPSG:4326, local frame) or coordinate units.'
  ],
  limitation:
    'Single flat cut (HDBSCAN*-epsilon style); no condensed tree and no stability-based selection of varying-density clusters.',
  run: (inputs, values) =>
    runHierarchicalDensityClustering(inputs['points'], {
      minimumSamples: values['minimumSamples'] as number,
      minimumClusterSize: values['minimumClusterSize'] as number,
      cutDistance: values['cutDistance'] as number,
      coordinateSystem: COORDINATE(values)
    })
};

const SIMILAR_BINDING: CatalogOperationBinding = {
  id: 'similar-locations',
  title: 'Similar locations',
  domain: 'analysis',
  summary:
    'Ranks features by attribute similarity to a reference feature (standardized Euclidean distance).',
  capabilityIds: ['group-and-regional-analysis'],
  contributorNames: ['GPUSimilarLocations'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [makeInputSpec('features', 'Features', ['point', 'line', 'polygon'], 20000, 200000)],
  parameters: [
    makeStringParameter('properties', 'Attribute properties (comma separated)'),
    makeNumberParameter(
      'referenceIndex',
      'Reference feature index (required)',
      -1,
      -1,
      1e9,
      undefined,
      'integer'
    ),
    makeNumberParameter(
      'resultCount',
      'Similar locations to flag',
      10,
      1,
      1000,
      undefined,
      'integer'
    )
  ],
  outputs: [
    makeOutputSpec('similarity', 'Similarity', 'none', ['source_index', 'source_id'], 20000)
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    'An explicit referenceIndex is required (default -1 means unselected). Features missing any attribute are skipped and counted; a reference that is missing or incomplete throws.',
    'Attributes are standardized by population mean and standard deviation (constant attributes contribute zero).',
    'The geometry column mirrors each source feature; rank 0 marks the reference.'
  ],
  limitation: 'Attribute similarity only; no spatial constraint and no weighting of attributes.',
  run: (inputs, values) => {
    validateRequirement(
      'similar-locations',
      Number.isInteger(values['referenceIndex']) &&
        (values['referenceIndex'] as number) >= 0 &&
        (values['referenceIndex'] as number) < inputs['features'].features.length,
      'referenceIndex must explicitly select a source row between 0 and feature count - 1'
    );
    return runSimilarLocations(inputs['features'], {
      properties: values['properties'] as string,
      referenceIndex: values['referenceIndex'] as number,
      resultCount: values['resultCount'] as number
    });
  }
};

const LOCATION_ALLOCATION_BINDING: CatalogOperationBinding = {
  id: 'location-allocation',
  title: 'Location allocation (p-median)',
  domain: 'analysis',
  summary:
    'Chooses p facilities among candidate points to minimize weighted straight-line distance to demand points.',
  capabilityIds: ['region-optimization'],
  contributorNames: ['GPULocationAllocation'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [
    makeInputSpec('demand', 'Demand points', ['point'], 5000, 5000),
    makeInputSpec('candidates', 'Candidate facilities', ['point'], 500, 500)
  ],
  parameters: [
    makeStringParameter('weightProperty', 'Demand weight property (empty = 1)'),
    makeNumberParameter('facilityCount', 'Facilities (p)', 3, 1, 30, undefined, 'integer'),
    CRS
  ],
  outputs: [
    makeOutputSpec(
      'facilities',
      'Selected facilities',
      'point',
      ['candidate_index', 'source_index', 'source_id'],
      30
    ),
    makeOutputSpec(
      'assignments',
      'Demand assignments',
      'none',
      ['demand_index', 'source_index', 'source_id'],
      5000
    )
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    'Greedy add followed by single-swap substitution (at most 10 passes); the result can be a local optimum and ties go to the lowest candidate index.',
    'Distances are straight-line meters (EPSG:4326, local frame) or coordinate units; negative or missing weights are skipped and counted.',
    'Fewer candidates than facilities, or work above 50 million distance evaluations, throws.'
  ],
  limitation:
    'No capacity or maximum-distance constraints and no network distances (see the network cost matrix for those).',
  run: (inputs, values) =>
    runLocationAllocation(inputs['demand'], inputs['candidates'], {
      facilityCount: values['facilityCount'] as number,
      weightProperty: values['weightProperty'] as string,
      coordinateSystem: COORDINATE(values)
    })
};

const SKATER_BINDING: CatalogOperationBinding = {
  id: 'skater-regions',
  title: 'SKATER regionalization',
  domain: 'clustering',
  summary:
    'Spatially contiguous regions by cutting a minimum spanning tree of attribute dissimilarity.',
  capabilityIds: ['group-and-regional-analysis'],
  contributorNames: ['GPUSkaterRegions'],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [makeInputSpec('features', 'Features', ['point', 'polygon'], 1500, 200000)],
  parameters: [
    makeStringParameter('properties', 'Attribute properties (comma separated)'),
    makeNumberParameter('regionCount', 'Regions', 4, 1, 64, undefined, 'integer'),
    makeNumberParameter(
      'minimumRegionSize',
      'Minimum region size',
      1,
      1,
      1000,
      undefined,
      'integer'
    ),
    ...makeWeightsParameters('queen')
  ],
  outputs: [makeOutputSpec('regions', 'Regions', 'none', ['source_index', 'source_id'], 1500)],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    ...WEIGHT_BEHAVIOR.slice(0, 2),
    'Attributes are standardized; the tree is built on squared Euclidean dissimilarity over neighbor links; each cut removes the edge that most reduces total within-region variance subject to minimumRegionSize.',
    'A disconnected neighborhood starts with one region per component; if no valid cut remains, fewer regions than requested are returned with a warning.'
  ],
  limitation:
    'Greedy SKATER on the CPU for up to 1,500 features; no floor on a spatially extensive attribute and no refinement.',
  run: (inputs, values) => {
    const settings = readWeightsSettings(values, false);
    guardContiguityInput(inputs, settings);
    return runSkaterRegions(inputs['features'], {
      properties: values['properties'] as string,
      regionCount: values['regionCount'] as number,
      minimumRegionSize: values['minimumRegionSize'] as number,
      neighborhood: settings.neighborhood,
      neighborCount: settings.neighborCount,
      distanceBand: settings.distanceBand,
      coordinateSystem: settings.coordinateSystem
    });
  }
};

/** PLATFORM3 bindings (family B): complete bindings, with gpuContributorNames and runGpu when GPU-backed. */
export const PLATFORM3_B_BINDINGS: readonly CatalogOperationBinding[] = [
  IDW_BINDING,
  KDE_BINDING,
  POINT_DENSITY_BINDING,
  LINE_DENSITY_BINDING,
  LINE_LENGTH_BINDING,
  KRIGING_BINDING,
  CELL_AGGREGATION_BINDING,
  CELL_ROLLUP_BINDING,
  CELL_COMPARE_BINDING,
  ZONAL_BINDING,
  AREAL_BINDING,
  WEIGHTS_BINDING,
  GLOBAL_STATISTICS_BINDING,
  LOCAL_MORAN_BINDING,
  GI_STAR_BINDING,
  LOCAL_GEARY_BINDING,
  EB_BINDING,
  MULTIPLE_TESTING_BINDING,
  KNOX_BINDING,
  DBSCAN_BINDING,
  OPTICS_BINDING,
  LOF_BINDING,
  HDBSCAN_BINDING,
  SIMILAR_BINDING,
  LOCATION_ALLOCATION_BINDING,
  SKATER_BINDING
];
