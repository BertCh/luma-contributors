// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createWGSLKernelNode,
  validateCapacity,
  validateFiniteNumber,
  validateRequirement,
  type GPURecipeStageStatusPort
} from '@luma-contributors/analysis/analysis-core';
import {
  addExposureByZoneRecipe,
  addRasterZonalDistributionRecipe
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPU_RASTER_ZONAL_STATISTICS_NO_ZONE} from '@luma-contributors/analysis/raster-analysis';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPURasterBufferBand} from '@luma.gl/graph/gpu-raster';
import type {
  CatalogOperationBinding,
  CatalogParameterValue,
  CatalogRunOutput
} from '../../analysis/catalog-binding-types';
import {validateCatalogTableInput, type CatalogDataInput} from '../../analysis/catalog-data-input';
import {
  appendCatalogRow,
  makeCatalogTable,
  getFeatureIdentity,
  countCollectionVertices,
  type CatalogFeatureCollection,
  type CatalogOutputTable
} from '../../analysis/catalog-geojson';
import {
  validateCatalogRasterInput,
  type CatalogRasterInput
} from '../../analysis/catalog-raster-input';
import {withCatalogGpuSession, type CatalogGpuSession} from '../../analysis/catalog-gpu-session';

const BINDING_ID = 'curated-zonal-exposure';
/** Bounded CPU materialization and GPU sort scratch for the curated, single-grid adapter. */
export const MAXIMUM_ZONAL_EXPOSURE_CELLS = 65536;
/** Dense summaries and probability outputs are allocated for every zone, including empty ones. */
export const MAXIMUM_ZONAL_EXPOSURE_ZONES = 4096;
/** Limit zone-major quantile storage to eight values per zone. */
const MAXIMUM_PROBABILITIES = 8;
/** Bounds the materialized source geometry retained by zone publication. */
const MAXIMUM_ZONE_VERTICES = 262144;
/** Float32 is the public raster recipe's sample and parameter format. */
const MAXIMUM_FLOAT32 = 3.4028234663852886e38;

type Values = Readonly<Record<string, CatalogParameterValue>>;

export type PreparedZonalExposure = {
  surface: CatalogRasterInput;
  zoneFeatures: CatalogFeatureCollection;
  zones: Uint32Array;
  assignedCounts: Uint32Array;
  hazard: Float32Array;
  hazardValidity: Uint32Array;
  zoneCount: number;
  probabilities: Float32Array;
  weightMeaning: 'none' | 'count' | 'density';
  population?: Float32Array;
  cellAreas?: Float32Array;
  threshold?: number;
  areaUnit: string;
};

function readColumn(table: CatalogOutputTable, values: Values, parameter: string) {
  const name = values[parameter];
  validateRequirement(
    BINDING_ID,
    typeof name === 'string' && name.length > 0 && Object.hasOwn(table.columns, name),
    `${parameter} must name an existing numeric cells column`
  );
  validateRequirement(
    BINDING_ID,
    table.columnTypes[name] === 'int32' || table.columnTypes[name] === 'float64',
    `${parameter} must name an int32 or float64 cells column`
  );
  return table.columns[name];
}

/** Missing numeric cells stay missing; negative populations/areas are rejected by the recipe. */
function readSample(value: unknown, parameter: string): number {
  if (value === null) return NaN;
  validateRequirement(
    BINDING_ID,
    typeof value === 'number',
    `${parameter} must contain numbers or null`
  );
  validateRequirement(
    BINDING_ID,
    !Number.isFinite(value) || Math.abs(value) <= MAXIMUM_FLOAT32,
    `${parameter} finite values must fit float32`
  );
  return value;
}

/**
 * Prepares already decoded CPU inputs once, before allocation. Cell indices form a complete
 * permutation of the grid; each cell has exactly one dense zone ID or null (unassigned).
 * Zone indices refer to the supplied polygons in feature order. No implicit polygon rasterization, fractional coverage, area calculation or unit conversion.
 */
export function prepareZonalExposure(
  inputs: Readonly<Record<string, CatalogDataInput>>,
  values: Values
): PreparedZonalExposure {
  const surface = inputs['surface'];
  const cells = inputs['cells'];
  const zoneFeatures = inputs['zones'];
  validateRequirement(
    BINDING_ID,
    zoneFeatures?.type === 'FeatureCollection',
    'zones must be polygon features'
  );
  validateCapacity(
    BINDING_ID,
    'zones.features.length',
    zoneFeatures.features.length,
    1,
    MAXIMUM_ZONAL_EXPOSURE_ZONES
  );
  validateRequirement(
    BINDING_ID,
    zoneFeatures.features.every(
      feature => feature.geometry?.type === 'Polygon' || feature.geometry?.type === 'MultiPolygon'
    ),
    'zones.geometry must be Polygon or MultiPolygon for every zone'
  );
  validateCapacity(
    BINDING_ID,
    'zones vertices',
    countCollectionVertices(zoneFeatures),
    1,
    MAXIMUM_ZONE_VERTICES
  );
  validateRequirement(BINDING_ID, surface?.type === 'raster', 'surface must be a raster band/grid');
  validateRequirement(BINDING_ID, cells?.type === 'table', 'cells must be a geometry-free table');
  validateCatalogRasterInput(BINDING_ID, surface, MAXIMUM_ZONAL_EXPOSURE_CELLS);
  validateCatalogTableInput(BINDING_ID, cells, MAXIMUM_ZONAL_EXPOSURE_CELLS);
  const cellCount = surface.grid.width * surface.grid.height;
  validateRequirement(
    BINDING_ID,
    cells.table.rowCount === cellCount,
    'cells.rowCount must equal grid cell count'
  );
  const zoneCount = zoneFeatures.features.length;
  validateCapacity(BINDING_ID, 'zoneCount', zoneCount, 1, MAXIMUM_ZONAL_EXPOSURE_ZONES);
  const weightMeaning = values['weightMeaning'];
  validateRequirement(
    BINDING_ID,
    weightMeaning === 'none' || weightMeaning === 'count' || weightMeaning === 'density',
    'weightMeaning must be none, count or density'
  );
  const probabilities = values['probabilities'];
  validateRequirement(
    BINDING_ID,
    Array.isArray(probabilities),
    'probabilities must be a numeric list'
  );
  validateCapacity(
    BINDING_ID,
    'probabilities.length',
    probabilities.length,
    1,
    MAXIMUM_PROBABILITIES
  );
  validateRequirement(
    BINDING_ID,
    probabilities.every(
      probability => Number.isFinite(probability) && probability >= 0 && probability <= 1
    ),
    'probabilities must be finite in [0, 1]'
  );
  const cellIndices = readColumn(cells.table, values, 'cellIndexProperty');
  const owners = readColumn(cells.table, values, 'zoneProperty');
  const populationColumn =
    weightMeaning !== 'none' ? readColumn(cells.table, values, 'populationProperty') : undefined;
  const areaColumn =
    weightMeaning === 'density' ? readColumn(cells.table, values, 'cellAreaProperty') : undefined;
  const areaUnit = weightMeaning === 'density' ? String(values['cellAreaUnit'] ?? '') : '';
  if (weightMeaning === 'density') {
    validateRequirement(
      BINDING_ID,
      areaUnit.trim().length > 0 && areaUnit === values['densityAreaUnit'],
      'cellAreaUnit must equal the nonempty densityAreaUnit (the density denominator); no conversion is performed'
    );
  }
  let threshold: number | undefined;
  if (weightMeaning !== 'none') {
    threshold = values['threshold'] as number;
    validateFiniteNumber(BINDING_ID, 'threshold', threshold, -MAXIMUM_FLOAT32, MAXIMUM_FLOAT32);
  }
  const zones = new Uint32Array(cellCount).fill(GPU_RASTER_ZONAL_STATISTICS_NO_ZONE);
  const assignedCounts = new Uint32Array(zoneCount);
  const seen = new Uint8Array(cellCount);
  const population = populationColumn ? new Float32Array(cellCount) : undefined;
  const cellAreas = areaColumn ? new Float32Array(cellCount) : undefined;
  for (let row = 0; row < cellCount; row++) {
    const cellIndex = cellIndices[row];
    validateRequirement(
      BINDING_ID,
      typeof cellIndex === 'number' &&
        Number.isInteger(cellIndex) &&
        cellIndex >= 0 &&
        cellIndex < cellCount &&
        !seen[cellIndex],
      'cellIndexProperty must be a permutation of integer grid indices in [0, grid cell count)'
    );
    seen[cellIndex] = 1;
    const owner = owners[row];
    validateRequirement(
      BINDING_ID,
      owner === null ||
        (typeof owner === 'number' && Number.isInteger(owner) && owner >= 0 && owner < zoneCount),
      'zoneProperty must contain null or whole-cell integer zone IDs in [0, zoneCount)'
    );
    if (owner !== null) {
      zones[cellIndex] = owner;
      assignedCounts[owner]++;
    }
    if (population && populationColumn)
      population[cellIndex] = readSample(populationColumn[row], 'populationProperty');
    if (cellAreas && areaColumn)
      cellAreas[cellIndex] = readSample(areaColumn[row], 'cellAreaProperty');
  }
  const hazard = new Float32Array(cellCount);
  const hazardValidity = Uint32Array.from(surface.band.validity, value => Number(value !== 0));
  for (let cellIndex = 0; cellIndex < cellCount; cellIndex++) {
    const value = surface.band.values[cellIndex];
    // Resolve native nodata before rounding, so a nearby valid float64 value cannot alias it.
    if (surface.band.noDataValue !== null && value === surface.band.noDataValue)
      hazardValidity[cellIndex] = 0;
    hazard[cellIndex] = hazardValidity[cellIndex] ? readSample(value, 'surface.band.values') : NaN;
  }
  return {
    surface,
    zoneFeatures,
    zones,
    assignedCounts,
    hazard,
    hazardValidity,
    zoneCount,
    probabilities: Float32Array.from(probabilities),
    weightMeaning,
    population,
    cellAreas,
    threshold,
    areaUnit
  };
}

export type ZonalExposureReadback = {
  rowCounts: Uint32Array;
  validCounts: Uint32Array;
  medians: Float32Array;
  modes: Float32Array;
  uniqueCounts: Uint32Array;
  quantiles: Float32Array;
  exposure?: {
    totalPopulation: Float32Array;
    observedPopulation: Float32Array;
    exposedPopulation: Float32Array;
    coverage: Float32Array;
    exposedShare: Float32Array;
  };
  stages: {stage: string; fields: Record<string, number>}[];
};

/** Publication adapter: preserve dense identities, missing values and every recipe status word. */
export function publishZonalExposure(
  prepared: PreparedZonalExposure,
  result: ZonalExposureReadback
): CatalogRunOutput {
  const incomplete = result.stages.some(
    ({fields}) => (fields['overflow'] ?? 0) !== 0 || (fields['candidateOverflow'] ?? 0) !== 0
  );
  const invalidThreshold = result.stages.some(({fields}) => (fields['invalidCount'] ?? 0) !== 0);
  const summary = makeCatalogTable(
    'zone-summary',
    'Zone distributions and exposure',
    {
      zone_id: 'int32',
      source_id: 'string',
      geometry: 'string',
      assigned_cell_count: 'int32',
      valid_cell_count: 'int32',
      median: 'float64',
      mode: 'float64',
      unique_count: 'int32',
      total_population: 'float64',
      observed_population: 'float64',
      exposed_population: 'float64',
      coverage: 'float64',
      exposed_share: 'float64',
      status: 'string'
    },
    'geometry'
  );
  const quantiles = makeCatalogTable('zone-quantiles', 'Unweighted zone quantiles', {
    zone_id: 'int32',
    probability_index: 'int32',
    probability: 'float64',
    value: 'float64'
  });
  const statuses = makeCatalogTable('recipe-status', 'Recipe stage status', {
    stage: 'string',
    field: 'string',
    value: 'int32'
  });
  const finite = (value: number | undefined) =>
    !incomplete && value !== undefined && Number.isFinite(value) ? value : null;
  for (let zoneIndex = 0; zoneIndex < prepared.zoneCount; zoneIndex++) {
    const exposure = result.exposure;
    appendCatalogRow(summary, {
      zone_id: zoneIndex,
      source_id: getFeatureIdentity(prepared.zoneFeatures.features[zoneIndex]),
      geometry: JSON.stringify(prepared.zoneFeatures.features[zoneIndex].geometry),
      assigned_cell_count: prepared.assignedCounts[zoneIndex],
      valid_cell_count: finite(result.validCounts[zoneIndex]),
      median: finite(result.medians[zoneIndex]),
      mode: finite(result.modes[zoneIndex]),
      unique_count: finite(result.uniqueCounts[zoneIndex]),
      total_population: finite(exposure?.totalPopulation[zoneIndex]),
      observed_population: finite(exposure?.observedPopulation[zoneIndex]),
      exposed_population: invalidThreshold ? null : finite(exposure?.exposedPopulation[zoneIndex]),
      coverage: finite(exposure?.coverage[zoneIndex]),
      exposed_share: invalidThreshold ? null : finite(exposure?.exposedShare[zoneIndex]),
      status: incomplete
        ? 'incomplete'
        : invalidThreshold
          ? 'invalid-threshold'
          : result.validCounts[zoneIndex] === 0
            ? 'no-observed-hazard'
            : 'ok'
    });
    prepared.probabilities.forEach((probability, probabilityIndex) =>
      appendCatalogRow(quantiles, {
        zone_id: zoneIndex,
        probability_index: probabilityIndex,
        probability,
        value: finite(
          result.quantiles[zoneIndex * prepared.probabilities.length + probabilityIndex]
        )
      })
    );
  }
  for (const {stage, fields} of result.stages) {
    for (const [field, value] of Object.entries(fields))
      appendCatalogRow(statuses, {stage, field, value});
  }
  return {
    tables: [summary, quantiles, statuses],
    diagnostics: [
      {
        code: 'whole-cell-ownership',
        severity: 'info',
        count: 1,
        message:
          'Cells use the supplied zone indices; zone polygons are preserved, and boundary fractions are not inferred. Quantiles are unweighted. Exposed share uses observed population; coverage is observed over total population.'
      },
      ...(incomplete
        ? [
            {
              code: 'recipe-overflow',
              severity: 'error' as const,
              count: 1,
              message:
                'A recipe stage overflowed. Statistics are withheld; inspect recipe-status before increasing capacities.'
            }
          ]
        : []),
      ...(invalidThreshold
        ? [
            {
              code: 'invalid-threshold',
              severity: 'error' as const,
              count: 1,
              message: 'Exposure classification is unknown because the recipe threshold is invalid.'
            }
          ]
        : [])
    ],
    summary: {
      cellCount: prepared.zones.length,
      zoneCount: prepared.zoneCount,
      unassignedCellCount: prepared.zones.reduce(
        (count, zone) => count + Number(zone === GPU_RASTER_ZONAL_STATISTICS_NO_ZONE),
        0
      ),
      exposureEnabled: Number(prepared.weightMeaning !== 'none'),
      weightMeaning: prepared.weightMeaning,
      densityAreaUnit: prepared.areaUnit,
      overflow: Number(incomplete),
      invalidThreshold: Number(invalidThreshold)
    }
  };
}

/** Copy recipe-owned graph views to session-owned readback buffers through public graph APIs. */
function stageReadback<Format extends 'float32' | 'uint32'>(
  session: CatalogGpuSession,
  name: string,
  source: GraphDataView<Format>
) {
  const output = session.output(name, source.format, source.length);
  const type = source.format === 'float32' ? 'f32' : 'u32';
  session.graph.add(
    createWGSLKernelNode(session.graph, {
      id: `${BINDING_ID}-${name}`,
      operation: BINDING_ID,
      invocationCount: source.length,
      bindings: [
        {name: 'source', view: source, type, access: 'read'},
        {name: 'destination', view: output.view, type, access: 'read_write'}
      ],
      body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
    })
  );
  return output;
}

async function runZonalExposure(
  session: CatalogGpuSession,
  prepared: PreparedZonalExposure
): Promise<CatalogRunOutput> {
  const grid = prepared.surface.grid;
  const hazard: GPURasterBufferBand<'float32'> = {
    id: 'hazard',
    format: 'float32',
    storage: {kind: 'buffer', values: session.input('hazard', prepared.hazard, 'float32')},
    validity: session.input('hazard-validity', prepared.hazardValidity, 'uint32')
  };
  const zones = session.input('zones', prepared.zones, 'uint32');
  const distribution = addRasterZonalDistributionRecipe(session.graph, {
    id: `${BINDING_ID}-distribution`,
    grid,
    input: hazard,
    zones,
    zoneCount: prepared.zoneCount,
    probabilities: session.input('probabilities', prepared.probabilities, 'float32')
  });
  const exposure = prepared.population
    ? addExposureByZoneRecipe(session.graph, {
        id: `${BINDING_ID}-exposure`,
        grid,
        hazard,
        zones,
        zoneCount: prepared.zoneCount,
        population: {
          id: 'population',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: session.input('population', prepared.population, 'float32')
          }
        },
        weightMeaning: prepared.weightMeaning === 'density' ? 'density' : 'count',
        ...(prepared.cellAreas
          ? {cellAreas: session.input('cell-areas', prepared.cellAreas, 'float32')}
          : {}),
        threshold: session.input('threshold', new Float32Array([prepared.threshold!]), 'float32')
      })
    : undefined;
  const distributionOutputs = {
    rowCounts: stageReadback(session, 'row-counts', distribution.outputs.rowCounts),
    validCounts: stageReadback(session, 'valid-counts', distribution.outputs.validCounts),
    uniqueCounts: stageReadback(session, 'unique-counts', distribution.outputs.uniqueCounts),
    medians: stageReadback(session, 'medians', distribution.outputs.medians),
    modes: stageReadback(session, 'modes', distribution.outputs.modes),
    quantiles: stageReadback(session, 'quantiles', distribution.outputs.quantiles!)
  };
  const exposureOutputs = exposure
    ? {
        totalPopulation: stageReadback(
          session,
          'total-population',
          exposure.outputs.totalPopulation
        ),
        observedPopulation: stageReadback(
          session,
          'observed-population',
          exposure.outputs.observedPopulation
        ),
        exposedPopulation: stageReadback(
          session,
          'exposed-population',
          exposure.outputs.exposedPopulation
        ),
        coverage: stageReadback(session, 'coverage', exposure.outputs.coverage),
        exposedShare: stageReadback(session, 'exposed-share', exposure.outputs.exposedShare)
      }
    : undefined;
  const stages: GPURecipeStageStatusPort[] = [
    ...distribution.status.stages,
    ...(exposure?.status.stages ?? [])
  ];
  const statusOutputs = stages.map(({stage, status}) => ({
    stage,
    fields: Object.entries(status).map(([field, view]) => ({
      field,
      output: stageReadback(session, `${stage}-${field}`, view)
    }))
  }));
  session.run();
  const [
    rowCounts,
    validCounts,
    uniqueCounts,
    medians,
    modes,
    quantiles,
    exposureValues,
    stageValues
  ] = await Promise.all([
    session.readUint32(distributionOutputs.rowCounts),
    session.readUint32(distributionOutputs.validCounts),
    session.readUint32(distributionOutputs.uniqueCounts),
    session.readFloat32(distributionOutputs.medians),
    session.readFloat32(distributionOutputs.modes),
    session.readFloat32(distributionOutputs.quantiles),
    exposureOutputs
      ? Promise.all(Object.values(exposureOutputs).map(output => session.readFloat32(output)))
      : undefined,
    Promise.all(
      statusOutputs.map(async ({stage, fields}) => ({
        stage,
        fields: Object.fromEntries(
          await Promise.all(
            fields.map(async ({field, output}) => [field, await session.readScalar(output)])
          )
        )
      }))
    )
  ]);
  const result = publishZonalExposure(prepared, {
    rowCounts,
    validCounts,
    uniqueCounts,
    medians,
    modes,
    quantiles,
    stages: stageValues,
    ...(exposureValues
      ? {
          exposure: {
            totalPopulation: exposureValues[0],
            observedPopulation: exposureValues[1],
            exposedPopulation: exposureValues[2],
            coverage: exposureValues[3],
            exposedShare: exposureValues[4]
          }
        }
      : {})
  });
  return {
    ...result,
    dispatchedContributors: [
      ...new Set([
        'GPURasterNeighborhood',
        ...distribution.contributors.map(contributor => contributor.constructor.name),
        ...(exposure?.contributors.map(contributor => contributor.constructor.name) ?? [])
      ])
    ]
  };
}

export const CURATED_ZONAL_EXPOSURE_BINDING: CatalogOperationBinding = {
  id: BINDING_ID,
  title: 'Zonal distribution and exposure',
  domain: 'raster',
  summary:
    'Summarise a raster by explicit whole-cell zone ownership, optionally measuring population exposure above a stated hazard threshold.',
  capabilityIds: ['raster-zonal', 'dataframe-aggregation'],
  contributorNames: ['GPUGroupStatistics', 'GPURasterZonalStatistics'],
  gpuContributorNames: ['GPURasterNeighborhood', 'GPUGroupStatistics', 'GPURasterZonalStatistics'],
  engine: 'gpu',
  inputs: [
    {
      name: 'zones',
      label: 'Zone polygons (cell zone IDs refer to this feature order)',
      accepts: ['polygon'],
      maximumFeatures: MAXIMUM_ZONAL_EXPOSURE_ZONES,
      maximumVertices: MAXIMUM_ZONE_VERTICES
    },
    {
      name: 'surface',
      label: 'Hazard raster band/grid',
      dataKind: 'raster',
      accepts: [],
      maximumFeatures: 0,
      maximumVertices: 0,
      maximumCells: MAXIMUM_ZONAL_EXPOSURE_CELLS
    },
    {
      name: 'cells',
      label: 'Whole-cell zone assignments and optional exposure weights',
      dataKind: 'table',
      accepts: [],
      maximumFeatures: MAXIMUM_ZONAL_EXPOSURE_CELLS,
      maximumVertices: 0
    }
  ],
  parameters: [
    ...Object.entries({
      cellIndexProperty: 'Raster cell index column',
      zoneProperty: 'Whole-cell zone ID column',
      populationProperty: 'Population count or density column',
      cellAreaProperty: 'Cell area column'
    }).map(([name, label]) => ({
      name,
      label,
      kind: 'string' as const,
      defaultValue: '',
      maximumLength: 128
    })),
    {
      name: 'probabilities',
      label: 'Unweighted quantile probabilities',
      kind: 'number-list',
      minimum: 0,
      maximum: 1,
      maximumLength: MAXIMUM_PROBABILITIES,
      defaultValue: [0, 0.5, 1]
    },
    {
      name: 'weightMeaning',
      label: 'Exposure weights',
      kind: 'enum',
      options: ['none', 'count', 'density'],
      defaultValue: 'none'
    },
    {
      name: 'threshold',
      label: 'Exposure threshold (hazard band units)',
      kind: 'number',
      minimum: -MAXIMUM_FLOAT32,
      maximum: MAXIMUM_FLOAT32
    },
    {
      name: 'densityAreaUnit',
      label: 'Area unit in the density denominator',
      kind: 'string',
      defaultValue: ''
    },
    {name: 'cellAreaUnit', label: 'Unit of supplied cell areas', kind: 'string', defaultValue: ''}
  ],
  outputs: [
    {
      tableId: 'zone-summary',
      label: 'Zone distributions and exposure',
      geometry: 'polygon',
      identityColumns: ['zone_id', 'source_id'],
      maximumRows: MAXIMUM_ZONAL_EXPOSURE_ZONES
    },
    {
      tableId: 'zone-quantiles',
      label: 'Zone quantiles',
      geometry: 'none',
      identityColumns: ['zone_id', 'probability_index'],
      maximumRows: MAXIMUM_ZONAL_EXPOSURE_ZONES * MAXIMUM_PROBABILITIES
    },
    {
      tableId: 'recipe-status',
      label: 'Recipe stage status',
      geometry: 'none',
      identityColumns: ['stage', 'field'],
      maximumRows: 64
    }
  ],
  cleanup:
    'The session releases uploaded buffers, graph scratch and readback buffers in finally. Returned tables own plain arrays; the caller owns the device and input arrays.',
  declaredBehavior: [
    'The cells table is a complete permutation of zero-based raster cell indices. Each zone ID is an integer in [0, zoneCount), or null for unassigned; duplicate indices and fractional ownership throw before GPU allocation.',
    'Hazard nodata and validity are applied before float32 statistics. Distributions are unweighted linear quantiles; empty zones remain present with zero counts and null statistics.',
    'weightMeaning=none runs only the distribution recipe. Count weights are people per cell; density weights require positive per-cell areas and matching explicit denominator units.',
    'Invalid or negative populations and invalid areas are omitted by the exposure recipe. Missing hazard retains population in total_population; exposed_share divides by observed_population and coverage divides observed by total.',
    'All recipe status words are published. Overflow withholds statistics. The summary retains each supplied zone polygon and source ID in feature order, including empty zones; the other outputs are reports keyed by zone ID or recipe stage.'
  ],
  limitation:
    'GPU only; numeric samples and thresholds are rounded to float32. Requires a complete bounded decoded raster and an explicit cell-ownership table; no polygon rasterization, fractional-cell allocation, density-unit conversion or new zone map geometry is inferred.',
  runDataGpu: async ({device}, inputs, values) => {
    const prepared = prepareZonalExposure(inputs, values);
    return withCatalogGpuSession(device, BINDING_ID, session =>
      runZonalExposure(session, prepared)
    );
  }
};
