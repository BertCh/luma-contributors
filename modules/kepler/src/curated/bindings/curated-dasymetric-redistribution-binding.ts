// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createWGSLKernelNode,
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '@luma-contributors/analysis/analysis-core';
import {addDasymetricRedistributionRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getGPURasterFractionalCoverageParameterValues} from '@luma-contributors/analysis/raster-analysis';
import type {
  CatalogOperationBinding,
  CatalogParameterValue,
  CatalogRunOutput
} from '../../analysis/catalog-binding-types';
import type {CatalogDataInput} from '../../analysis/catalog-data-input';
import {
  appendCatalogRow,
  countCollectionVertices,
  getFeatureIdentity,
  makeCatalogTable,
  type CatalogFeatureCollection
} from '../../analysis/catalog-geojson';
import {packCatalogPolygons} from '../../analysis/catalog-gpu-pack';
import {withCatalogGpuSession} from '../../analysis/catalog-gpu-session';
import {
  validateCatalogRasterInput,
  type CatalogRasterInput
} from '../../analysis/catalog-raster-input';
import type {KeplerCuratedAnalysis} from '../curated-analysis-types';

/** Shared question/slot metadata for the registered GPU recipe and its reusable AX template. */
export const CURATED_DASYMETRIC_REDISTRIBUTION_ANALYSIS: KeplerCuratedAnalysis = {
  id: 'dasymetric-redistribution',
  question: 'prepare-data',
  title: 'Where within each zone is the total likely to be?',
  name: 'Dasymetric redistribution',
  wave: 2,
  sampleId: 'dasymetric-fractional-support',
  method:
    'Fractional source polygon coverage is weighted by a loaded ancillary density band. Extensive source totals are transferred to explicitly owned whole target cells. The zone default retains uncovered mass; a source ledger reports allocated and unallocated totals. Incomplete evidence suppresses derived values.',
  slots: [
    {
      name: 'sources',
      label: 'Source zones with extensive totals',
      kind: 'polygons',
      rowSemantics: ['objects'],
      fields: [
        {
          role: 'total',
          label: 'Extensive total',
          accepts: ['number', 'integer'],
          measurement: ['count', 'amount'],
          required: true
        }
      ]
    },
    {name: 'targets', label: 'Whole-cell target zones', kind: 'polygons', fields: []},
    {name: 'ancillary', label: 'Decoded planar ancillary density band', kind: 'surface', fields: []}
  ],
  execution: {
    status: 'available',
    bindingId: 'curated-dasymetric-redistribution',
    inputs: {sources: 'sources', targets: 'targets', ancillary: 'ancillary'},
    fieldParameters: {'sources.total': 'totalProperty'},
    parameters: {denominator: 'zone'}
  },
  exposedParameters: [
    'denominator',
    'targetCellOwners',
    'coordinatesMatchRaster',
    'targetOwnershipConfirmed'
  ]
};

const BINDING_ID = 'curated-dasymetric-redistribution';
/** Bound the recipe's dense source-cell and target-tuple work products. */
export const MAXIMUM_DASYMETRIC_PRODUCT = 4_194_304;
/** Bound decoded uploads, serializable ownership and publication readback separately. */
export const MAXIMUM_DASYMETRIC_CELLS = 65_536;
export const MAXIMUM_DASYMETRIC_ZONES = 256;
export const MAXIMUM_DASYMETRIC_VERTICES = 32_768;
type Parameters = Readonly<Record<string, CatalogParameterValue>>;

function readZones(input: CatalogDataInput | undefined, name: string): CatalogFeatureCollection {
  validateRequirement(
    BINDING_ID,
    input?.type === 'FeatureCollection',
    `${name} must contain polygon features`
  );
  const collection = input as CatalogFeatureCollection;
  validateCapacity(
    BINDING_ID,
    `${name} feature count`,
    collection.features.length,
    1,
    MAXIMUM_DASYMETRIC_ZONES
  );
  validateMaximumCount(
    BINDING_ID,
    `${name} vertex count`,
    countCollectionVertices(collection),
    MAXIMUM_DASYMETRIC_VERTICES
  );
  validateRequirement(
    BINDING_ID,
    collection.features.every(
      feature => feature.geometry?.type === 'Polygon' || feature.geometry?.type === 'MultiPolygon'
    ),
    `${name} must contain only Polygon or MultiPolygon geometry; no rows may be dropped`
  );
  return collection;
}

/** Prepare once at the input boundary, retaining every feature's original row identity. */
export function prepareCuratedDasymetricInputs(
  inputs: Readonly<Record<string, CatalogDataInput>>,
  parameters: Parameters
) {
  const sources = readZones(inputs['sources'], 'sources');
  const targets = readZones(inputs['targets'], 'targets');
  validateRequirement(
    BINDING_ID,
    inputs['ancillary']?.type === 'raster',
    'ancillary must be a decoded raster band'
  );
  const ancillary = inputs['ancillary'] as CatalogRasterInput;
  validateCatalogRasterInput(BINDING_ID, ancillary, MAXIMUM_DASYMETRIC_CELLS);
  validateRequirement(
    BINDING_ID,
    ancillary.crs === 'planar' && ancillary.grid.pixelInterpretation === 'area',
    'ancillary crs must be planar with area pixels; density is per square affine-coordinate unit'
  );
  validateRequirement(
    BINDING_ID,
    parameters['coordinatesMatchRaster'] === true,
    'coordinatesMatchRaster must confirm that source and target coordinates use the raster planar CRS and units'
  );
  validateRequirement(
    BINDING_ID,
    parameters['targetOwnershipConfirmed'] === true,
    'targetOwnershipConfirmed must certify unique whole-cell target ownership in target row order'
  );
  const totalProperty = parameters['totalProperty'];
  validateRequirement(
    BINDING_ID,
    typeof totalProperty === 'string' &&
      totalProperty.trim().length > 0 &&
      totalProperty.length <= 64,
    'totalProperty must name an extensive count or amount column (1 to 64 characters)'
  );
  const denominator = parameters['denominator'] ?? 'zone';
  validateRequirement(
    BINDING_ID,
    denominator === 'zone' || denominator === 'overlap',
    'denominator must be zone or overlap'
  );
  const cellCount = ancillary.grid.width * ancillary.grid.height;
  const ownership = parameters['targetCellOwners'];
  validateRequirement(
    BINDING_ID,
    Array.isArray(ownership) && ownership.length === cellCount,
    'targetCellOwners must contain one target row index per raster cell in row-major order'
  );
  const owners = ownership as readonly number[];
  validateRequirement(
    BINDING_ID,
    owners.every(
      owner => Number.isInteger(owner) && owner >= -1 && owner < targets.features.length
    ),
    'targetCellOwners values must be -1 (unassigned) or a target row index below target feature count'
  );
  const tupleCapacity = sources.features.length * cellCount;
  validateMaximumCount(
    BINDING_ID,
    'sources times cells',
    tupleCapacity,
    MAXIMUM_DASYMETRIC_PRODUCT
  );
  validateMaximumCount(
    BINDING_ID,
    'targets times tupleCapacity',
    targets.features.length * tupleCapacity,
    MAXIMUM_DASYMETRIC_PRODUCT
  );
  const pairCapacity =
    parameters['pairCapacity'] ?? sources.features.length * targets.features.length;
  validateCapacity(
    BINDING_ID,
    'pairCapacity',
    pairCapacity as number,
    1,
    MAXIMUM_DASYMETRIC_PRODUCT
  );
  // Translate in float64 before packing; float32 world coordinates can move a half-cell edge.
  const [columnX, rowX, originX, columnY, rowY, originY] = ancillary.grid.affine;
  const grid = {...ancillary.grid, affine: [columnX, rowX, 0, columnY, rowY, 0] as const};
  const polygons = packCatalogPolygons(sources.features, {
    closing: 'drop',
    project: position => [position[0] - originX, position[1] - originY]
  });
  validateRequirement(
    BINDING_ID,
    polygons.sourceIndices.length === sources.features.length,
    'sources polygon packing must retain every feature'
  );
  // The recipe validates source totals on the GPU; invalid rows retain their identities.
  const sourceTotals = Float32Array.from(sources.features, feature => {
    const value = feature.properties?.[totalProperty as string];
    return typeof value === 'number' ? value : NaN;
  });
  let densityConversionOverflow = false;
  const density = Float32Array.from(ancillary.band.values, (value, index) => {
    if (!ancillary.band.validity[index] || value === ancillary.band.noDataValue) return NaN;
    const rounded = Math.fround(value);
    if (Number.isFinite(value) && !Number.isFinite(rounded)) densityConversionOverflow = true;
    return rounded;
  });
  validateRequirement(
    BINDING_ID,
    !densityConversionOverflow,
    'ancillary.band.values finite valid density samples must remain finite in float32'
  );
  return {
    sources,
    targets,
    ancillary,
    grid,
    polygons,
    sourceTotals,
    density,
    targetOwners: Uint32Array.from(owners, owner => (owner < 0 ? 0xffffffff : owner)),
    tupleCapacity,
    pairCapacity: pairCapacity as number,
    denominator: denominator as 'zone' | 'overlap'
  };
}

/** GPU-only adapter over the public recipe; the session releases every owned resource in finally. */
export const CURATED_DASYMETRIC_REDISTRIBUTION_BINDING: CatalogOperationBinding = {
  id: BINDING_ID,
  title: 'Dasymetric redistribution',
  domain: 'raster',
  summary:
    'Redistribute extensive source totals using fractional polygon coverage and a loaded ancillary density band, with explicit whole-cell target ownership and a source mass ledger.',
  capabilityIds: ['change-of-support'],
  contributorNames: ['GPURasterFractionalCoverage', 'GPUArealInterpolation', 'GPUSpatialLag'],
  gpuContributorNames: ['GPURasterFractionalCoverage', 'GPUArealInterpolation', 'GPUSpatialLag'],
  engine: 'gpu',
  inputs: [
    {
      name: 'sources',
      label: 'Source zones with extensive totals',
      accepts: ['polygon'],
      maximumFeatures: MAXIMUM_DASYMETRIC_ZONES,
      maximumVertices: MAXIMUM_DASYMETRIC_VERTICES
    },
    {
      name: 'targets',
      label: 'Target zones in ownership-index order',
      accepts: ['polygon'],
      maximumFeatures: MAXIMUM_DASYMETRIC_ZONES,
      maximumVertices: MAXIMUM_DASYMETRIC_VERTICES
    },
    {
      name: 'ancillary',
      label: 'Loaded planar ancillary density band',
      dataKind: 'raster',
      accepts: [],
      maximumCells: MAXIMUM_DASYMETRIC_CELLS,
      maximumFeatures: 0,
      maximumVertices: 0
    }
  ],
  parameters: [
    {
      name: 'totalProperty',
      label: 'Extensive source total column',
      kind: 'string',
      maximumLength: 64,
      defaultValue: ''
    },
    {
      name: 'targetCellOwners',
      label: 'Target row index for each cell (-1 unassigned)',
      kind: 'number-list',
      maximumLength: MAXIMUM_DASYMETRIC_CELLS,
      defaultValue: [-1]
    },
    {
      name: 'coordinatesMatchRaster',
      label: 'Source and target coordinates match the planar raster CRS and units',
      kind: 'boolean',
      defaultValue: false
    },
    {
      name: 'targetOwnershipConfirmed',
      label: 'Each assigned whole cell belongs to the named target row',
      kind: 'boolean',
      defaultValue: false
    },
    {
      name: 'denominator',
      label: 'Uncovered source mass',
      kind: 'enum',
      options: ['zone', 'overlap'],
      defaultValue: 'zone'
    },
    {
      name: 'pairCapacity',
      label: 'Source-target pair capacity',
      kind: 'integer',
      minimum: 1,
      maximum: MAXIMUM_DASYMETRIC_PRODUCT
    }
  ],
  outputs: [
    {
      tableId: 'dasymetric-targets',
      label: 'Redistributed target totals',
      geometry: 'polygon',
      identityColumns: ['target_index', 'target_id'],
      maximumRows: MAXIMUM_DASYMETRIC_ZONES
    },
    {
      tableId: 'dasymetric-ledger',
      label: 'Source mass ledger',
      geometry: 'none',
      identityColumns: ['source_index', 'source_id'],
      maximumRows: MAXIMUM_DASYMETRIC_ZONES
    },
    {
      tableId: 'dasymetric-status',
      label: 'Recipe evidence',
      geometry: 'none',
      identityColumns: ['stage', 'metric'],
      maximumRows: 32
    }
  ],
  cleanup:
    'The GPU session destroys uploaded buffers, output buffers and compiled graphs in finally; returned tables are plain caller-owned arrays.',
  declaredBehavior: [
    'A complete bounded planar area-pixel grid is required. Density is per square affine-coordinate unit, with matching source and target CRS confirmed before allocation.',
    'Explicit targetCellOwners uses target row order, one owner per whole cell; -1 is unassigned. Target geometry is retained for publication and is not rasterized by cell-centre sampling.',
    'Fractional source coverage includes holes and multipart geometry. Source totals are independent extensive quantities, never rates. Source coordinates and the raster origin are translated together before float32 packing.',
    'As in the underlying triangulator, collapsed source shells complete with no coverage and retain their entire totals as zero-support sources; a triangulation failure instead invalidates completion.',
    'Zero, negative, invalid and nodata ancillary samples carry no support. Finite valid densities that overflow float32 are rejected before GPU allocation. Zero-support sources retain their entire totals with validity zero.',
    'zone retains uncovered residual mass; overlap redistributes supported mass among assigned targets. Allocated and unallocated totals retain original source row order.',
    'Incomplete coverage, invalid source totals or tuples, pair overflow and arithmetic overflow suppress derived numerical columns to null; completion and stage evidence remain visible.',
    'Dense source-cell and target-tuple work products cannot exceed 4194304; cells are capped at 65536, zones at 256 and vertices at 32768 per polygon input.'
  ],
  limitation:
    'GPU device required. Explicit ownership must be supplied and certified; polygon/ownership agreement is not inferred. No reprojection, geographic density areas, tile normalization or CPU fallback. Float32 mass tolerance is absolute 1e-4 plus relative 1e-5.',
  runDataGpu: async ({device}, inputs, parameters) => {
    const prepared = prepareCuratedDasymetricInputs(inputs, parameters);
    return withCatalogGpuSession(device, BINDING_ID, async session => {
      const sourceCount = prepared.sources.features.length;
      const targetCount = prepared.targets.features.length;
      const outputs = {
        targetTotals: session.output('target-totals', 'float32', targetCount),
        targetSupportWeights: session.output('target-support', 'float32', targetCount),
        sourceEligibleWeights: session.output('source-support', 'float32', sourceCount),
        sourceTransferredFractions: session.output('source-fractions', 'float32', sourceCount),
        sourceAllocatedTotals: session.output('source-allocated', 'float32', sourceCount),
        sourceUnallocatedTotals: session.output('source-unallocated', 'float32', sourceCount),
        sourceValidity: session.output('source-validity', 'uint32', sourceCount)
      };
      const recipe = addDasymetricRedistributionRecipe(session.graph, {
        id: BINDING_ID,
        grid: prepared.grid,
        source: {
          kind: 'polygons',
          tupleCapacity: prepared.tupleCapacity,
          zones: {
            positions: session.input('positions', prepared.polygons.positions, 'float32x2'),
            featureOffsets: session.input(
              'feature-offsets',
              prepared.polygons.featureOffsets,
              'uint32'
            ),
            polygonOffsets: session.input(
              'polygon-offsets',
              prepared.polygons.polygonOffsets,
              'uint32'
            ),
            ringOffsets: session.input('ring-offsets', prepared.polygons.ringOffsets, 'uint32')
          },
          parameters: session.input(
            'coverage-parameters',
            getGPURasterFractionalCoverageParameterValues(prepared.grid),
            'float32'
          )
        },
        sourceTotals: session.input('source-totals', prepared.sourceTotals, 'float32'),
        targetZones: session.input('target-owners', prepared.targetOwners, 'uint32'),
        targetZoneCount: targetCount,
        ancillary: {
          id: prepared.ancillary.id,
          format: 'float32',
          storage: {kind: 'buffer', values: session.input('density', prepared.density, 'float32')}
        },
        denominator: prepared.denominator,
        pairCapacity: prepared.pairCapacity,
        outputs: Object.fromEntries(
          Object.entries(outputs).map(([name, output]) => [name, output.view])
        )
      });
      const evidence = [
        ...Object.entries(recipe.status).flatMap(([metric, view]) =>
          metric === 'stages'
            ? []
            : [{stage: 'recipe', metric, view: view as typeof recipe.status.complete}]
        ),
        ...recipe.status.stages.flatMap(({stage, status}) =>
          Object.entries(status).map(([metric, view]) => ({stage, metric, view}))
        )
      ].map(({stage, metric, view}, index) => {
        const output = session.output(`status-${index}`, 'uint32', 1);
        session.graph.add(
          createWGSLKernelNode(session.graph, {
            id: `read-status-${index}`,
            operation: BINDING_ID,
            invocationCount: 1,
            bindings: [
              {name: 'source', view, type: 'u32', access: 'read'},
              {name: 'output', view: output.view, type: 'u32', access: 'read_write'}
            ],
            body: 'output[outputOffset+index]=source[sourceOffset+index];'
          })
        );
        return {stage, metric, output};
      });
      session.run();
      const statusValues = await Promise.all(
        evidence.map(async entry => ({
          stage: entry.stage,
          metric: entry.metric,
          value: await session.readScalar(entry.output)
        }))
      );
      const complete =
        statusValues.find(entry => entry.stage === 'recipe' && entry.metric === 'complete')!
          .value === 1;
      const targetTotals = await session.readFloat32(outputs.targetTotals);
      const targetSupport = await session.readFloat32(outputs.targetSupportWeights);
      const eligible = await session.readFloat32(outputs.sourceEligibleWeights);
      const fractions = await session.readFloat32(outputs.sourceTransferredFractions);
      const allocated = await session.readFloat32(outputs.sourceAllocatedTotals);
      const unallocated = await session.readFloat32(outputs.sourceUnallocatedTotals);
      const validity = await session.readUint32(outputs.sourceValidity);
      const publishNumber = (value: number) => (complete && Number.isFinite(value) ? value : null);
      const targetTable = makeCatalogTable(
        'dasymetric-targets',
        'Redistributed target totals',
        {
          target_index: 'int32',
          target_id: 'string',
          status: 'string',
          total: 'float64',
          support_weight: 'float64',
          geometry: 'string'
        },
        'geometry'
      );
      targetTable.coordinateSystem = 'planar';
      prepared.targets.features.forEach((feature, index) =>
        appendCatalogRow(targetTable, {
          target_index: index,
          target_id: getFeatureIdentity(feature),
          status: complete ? 'complete' : 'incomplete',
          total: publishNumber(targetTotals[index]),
          support_weight: publishNumber(targetSupport[index]),
          geometry: JSON.stringify(feature.geometry)
        })
      );
      const ledger = makeCatalogTable('dasymetric-ledger', 'Source mass ledger', {
        source_index: 'int32',
        source_id: 'string',
        status: 'string',
        source_total: 'float64',
        eligible_weight: 'float64',
        transferred_fraction: 'float64',
        allocated_total: 'float64',
        unallocated_total: 'float64',
        validity: 'int32'
      });
      prepared.sources.features.forEach((feature, index) =>
        appendCatalogRow(ledger, {
          source_index: index,
          source_id: getFeatureIdentity(feature),
          status: !complete ? 'incomplete' : validity[index] ? 'complete' : 'zero-support',
          source_total: Number.isFinite(prepared.sourceTotals[index])
            ? prepared.sourceTotals[index]
            : null,
          eligible_weight: publishNumber(eligible[index]),
          transferred_fraction: publishNumber(fractions[index]),
          allocated_total: publishNumber(allocated[index]),
          unallocated_total: publishNumber(unallocated[index]),
          validity: validity[index]
        })
      );
      const statusTable = makeCatalogTable('dasymetric-status', 'Recipe evidence', {
        stage: 'string',
        metric: 'string',
        value: 'int32'
      });
      statusValues.forEach(entry => appendCatalogRow(statusTable, entry));
      const sumValues = (values: Float32Array) =>
        Array.from(values).reduce((sum, value) => sum + value, 0);
      const summary: CatalogRunOutput['summary'] = {
        complete: Number(complete),
        sourceCount,
        targetCount,
        cellCount: prepared.density.length,
        denominator: prepared.denominator,
        totalUnits: 'source extensive units',
        supportUnits: 'density times square affine-coordinate units',
        zeroEligibleCount: statusValues.find(
          entry => entry.stage === 'recipe' && entry.metric === 'zeroEligibleCount'
        )!.value
      };
      if (complete) {
        summary['sourceTotal'] = sumValues(prepared.sourceTotals);
        summary['targetTotal'] = sumValues(targetTotals);
        summary['unallocatedTotal'] = sumValues(unallocated);
        summary['massBalanceError'] = Math.abs(
          Number(summary['targetTotal']) +
            Number(summary['unallocatedTotal']) -
            Number(summary['sourceTotal'])
        );
      }
      return {
        tables: [targetTable, ledger, statusTable],
        diagnostics: complete
          ? [
              {
                code: 'mass-ledger',
                severity: 'info',
                count: sourceCount,
                message:
                  'Inspect allocated and unallocated totals together; zero eligible support retains all source mass.'
              }
            ]
          : [
              {
                code: 'incomplete-dasymetric',
                severity: 'error',
                count: 1,
                message:
                  'Recipe evidence is incomplete: all derived numerical outputs are null. Inspect dasymetric-status before publishing.'
              }
            ],
        summary,
        dispatchedContributors: [
          ...new Set(recipe.contributors.map(contributor => contributor.constructor.name))
        ].filter(name =>
          ['GPURasterFractionalCoverage', 'GPUArealInterpolation', 'GPUSpatialLag'].includes(name)
        )
      };
    });
  }
};
