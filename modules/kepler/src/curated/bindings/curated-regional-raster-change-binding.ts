// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createWGSLKernelNode,
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '@luma-contributors/analysis/analysis-core';
import {addRegionalRasterChangeRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  GPUPolygonRasterization,
  getGPUPolygonRasterizationParameterValues,
  getGPUChangeDetectionParameterValues
} from '@luma-contributors/analysis/raster-analysis';
import type {
  CatalogOperationBinding,
  CatalogParameterValue
} from '../../analysis/catalog-binding-types';
import type {CatalogDataInput} from '../../analysis/catalog-data-input';
import {
  areCatalogRasterGridsAligned,
  validateCatalogRasterInput,
  type CatalogRasterInput
} from '../../analysis/catalog-raster-input';
import {
  appendCatalogRow,
  getFeatureIdentity,
  getPolygonParts,
  makeCatalogTable
} from '../../analysis/catalog-geojson';
import {withCatalogGpuSession} from '../../analysis/catalog-gpu-session';
import {makeLocalFrame} from '../../analysis/catalog-local-frame';
import {readRecipePolygons} from '../../analysis/catalog-platform5-recipe-common';

const BINDING_ID = 'curated-regional-raster-change';
/** Bound host uploads/readback and the rasterizer's sorted scanline crossing storage. */
export const REGIONAL_RASTER_CHANGE_LIMITS = {
  cells: 65536,
  zones: 1024,
  vertices: 32768,
  crossings: 1048576,
  /** Leave twofold headroom below Float32 overflow for positive area reductions. */
  aggregateArea: 2 ** 127
} as const;

/** Boundary preparation only; ownership and all change/summary calculations run on the GPU. */
export function prepareRegionalRasterChange(
  inputs: Readonly<Record<string, CatalogDataInput>>,
  parameters: Readonly<Record<string, CatalogParameterValue>>
) {
  const before = inputs['before'];
  const after = inputs['after'];
  const zones = inputs['zones'];
  validateRequirement(
    BINDING_ID,
    before?.type === 'raster',
    'before must be a decoded raster band/grid'
  );
  validateRequirement(
    BINDING_ID,
    after?.type === 'raster',
    'after must be a decoded raster band/grid'
  );
  validateRequirement(
    BINDING_ID,
    zones?.type === 'FeatureCollection',
    'zones must be polygon features'
  );
  // The named checks above give callers a useful error before accessing payloads.
  if (before.type !== 'raster' || after.type !== 'raster' || zones.type !== 'FeatureCollection')
    throw new Error(`${BINDING_ID}: incompatible input types`);
  for (const raster of [before, after])
    validateCatalogRasterInput(BINDING_ID, raster, REGIONAL_RASTER_CHANGE_LIMITS.cells);
  validateRequirement(
    BINDING_ID,
    areCatalogRasterGridsAligned(before, after),
    'before and after grids must align exactly (CRS, affine, dimensions and pixel interpretation); resample upstream'
  );
  validateRequirement(
    BINDING_ID,
    before.crs === 'planar' && before.cellSizeMode === 'uniform',
    'crs must be planar with uniform cell areas; geographic and Web Mercator ground areas require upstream preparation'
  );
  const [stepX, shearX, originX, shearY, stepY, originY] = before.grid.affine;
  validateRequirement(
    BINDING_ID,
    shearX === 0 && shearY === 0 && stepX > 0,
    'grid.affine must be axis aligned with positive column spacing'
  );
  validateRequirement(
    BINDING_ID,
    parameters['confirmComparable'] === true,
    'confirmComparable must be true: bands must be calibrated to the same units and zones must use the raster CRS and units'
  );
  const threshold = parameters['threshold'] ?? 1;
  validateRequirement(
    BINDING_ID,
    typeof threshold === 'number',
    'threshold must be a number in band units'
  );
  validateFiniteNumber(BINDING_ID, 'threshold', threshold as number, 0, 1e30);
  validateCapacity(
    BINDING_ID,
    'zones.features.length',
    zones.features.length,
    1,
    REGIONAL_RASTER_CHANGE_LIMITS.zones
  );
  // Translate to the grid origin before Float32 packing, preserving small local offsets at large origins.
  const frame = makeLocalFrame('planar', [[originX, originY]], {shift: 'minimum'});
  let vertexCount = 0;
  for (const feature of zones.features) {
    const parts = getPolygonParts(feature.geometry);
    validateRequirement(
      BINDING_ID,
      parts.length > 0 &&
        parts.every(
          part =>
            part.length > 0 &&
            part.every(
              ring => ring.length >= 4 && ring.every(position => position.every(Number.isFinite))
            )
        ),
      'zones geometry must contain finite Polygon or MultiPolygon rings with at least four vertices'
    );
    for (const part of parts) for (const ring of part) vertexCount += ring.length;
  }
  validateCapacity(
    BINDING_ID,
    'zones vertexCount',
    vertexCount,
    1,
    REGIONAL_RASTER_CHANGE_LIMITS.vertices
  );
  const polygons = readRecipePolygons(zones, frame);
  validateRequirement(
    BINDING_ID,
    polygons.featureCount === zones.features.length && polygons.positions.every(Number.isFinite),
    'zones must retain every feature and finite Float32 positions'
  );
  const grid = {...before.grid, affine: [stepX, 0, 0, 0, stepY, 0] as const};
  const placement = getGPUPolygonRasterizationParameterValues(grid);
  validateRequirement(
    BINDING_ID,
    placement.every(Number.isFinite) && placement[2] > 0 && placement[3] !== 0,
    'grid spacing and extent must remain finite and nonzero in Float32'
  );
  const cellArea = Math.fround(Math.abs(stepX * stepY));
  validateRequirement(
    BINDING_ID,
    Number.isFinite(cellArea) && cellArea > 0,
    'grid cell area must be positive and finite in Float32 squared coordinate units'
  );
  const cellCount = grid.width * grid.height;
  // A single zone can own every cell. Bound the uploaded Float32 areas' worst-case sum,
  // including missing cells in totalArea, before any GPU allocation or reduction.
  validateFiniteNumber(
    BINDING_ID,
    'maximum aggregate area (cellArea * cellCount)',
    cellArea * cellCount,
    0,
    REGIONAL_RASTER_CHANGE_LIMITS.aggregateArea
  );
  const crossingCapacity = Math.min(
    REGIONAL_RASTER_CHANGE_LIMITS.crossings,
    (polygons.positions.length / 2) * grid.height
  );
  const slices = new Float32Array(cellCount * 2);
  const observationMask = new Uint32Array(cellCount * 2);
  const rasters: readonly CatalogRasterInput[] = [before, after];
  for (let cell = 0; cell < cellCount; cell++) {
    for (let slice = 0; slice < 2; slice++) {
      const band = rasters[slice].band;
      const value = Math.fround(band.values[cell]);
      const valid =
        band.validity[cell] !== 0 &&
        Number.isFinite(value) &&
        band.values[cell] !== band.noDataValue;
      slices[cell * 2 + slice] = valid ? value : NaN;
      observationMask[cell * 2 + slice] = Number(valid);
    }
  }
  return {
    before,
    after,
    zones,
    polygons,
    grid,
    placement,
    cellCount,
    cellArea,
    crossingCapacity,
    slices,
    observationMask,
    threshold: Math.fround(threshold as number)
  };
}

/** Gate positive certificates and declared loss fields; informational counts do not indicate loss. */
export function validateRegionalRasterChangeStatus(status: {
  rasterOverflow: number;
  requiredCrossings: number;
  crossingCapacity: number;
  stages: readonly {name: string; value: number}[];
}): void {
  validateRequirement(
    BINDING_ID,
    status.rasterOverflow === 0,
    `crossingCapacity limit ${status.crossingCapacity} exceeded (required ${status.requiredCrossings}); no result published`
  );
  validateRequirement(
    BINDING_ID,
    status.stages.every(stage => {
      const field = stage.name.slice(stage.name.lastIndexOf('.') + 1);
      if (field === 'complete' || field === 'converged') return stage.value === 1;
      return (
        ![
          'overflow',
          'candidateOverflow',
          'invalidInput',
          'invalidCount',
          'iterationLimit',
          'unresolvedCount',
          'incomplete',
          'uncertainCount'
        ].includes(field) || stage.value === 0
      );
    }),
    `recipe status must certify completeness without loss before publication: ${JSON.stringify(status.stages)}`
  );
}

export const CURATED_REGIONAL_RASTER_CHANGE_BINDING: CatalogOperationBinding = {
  id: BINDING_ID,
  title: 'Regional raster change',
  domain: 'raster',
  summary:
    'Compare aligned calibrated bands on common observations and report zone means, changed area and coverage.',
  capabilityIds: ['zonal-and-region-statistics'],
  contributorNames: ['GPUPolygonRasterization', 'GPUChangeDetection', 'GPURasterZonalStatistics'],
  gpuContributorNames: [
    'GPUPolygonRasterization',
    'GPUChangeDetection',
    'GPURasterZonalStatistics'
  ],
  engine: 'gpu',
  inputs: [
    ...['before', 'after'].map(name => ({
      name,
      label: name === 'before' ? 'Earlier band' : 'Later band',
      dataKind: 'raster' as const,
      accepts: [],
      maximumFeatures: 0,
      maximumVertices: 0,
      maximumCells: REGIONAL_RASTER_CHANGE_LIMITS.cells
    })),
    {
      name: 'zones',
      label: 'Zones in raster coordinates',
      accepts: ['polygon'],
      maximumFeatures: REGIONAL_RASTER_CHANGE_LIMITS.zones,
      maximumVertices: REGIONAL_RASTER_CHANGE_LIMITS.vertices
    }
  ],
  parameters: [
    {
      name: 'threshold',
      label: 'Absolute change threshold (band units, inclusive)',
      kind: 'number',
      minimum: 0,
      maximum: 1e30,
      defaultValue: 1,
      unit: 'band'
    },
    {
      name: 'confirmComparable',
      label: 'Bands have the same calibrated units; zones use the raster CRS and units',
      kind: 'boolean',
      defaultValue: false
    }
  ],
  outputs: [
    {
      tableId: 'regions',
      label: 'Regional change and coverage',
      geometry: 'polygon',
      identityColumns: ['zone_index', 'source_id'],
      maximumRows: REGIONAL_RASTER_CHANGE_LIMITS.zones
    },
    {
      tableId: 'differences',
      label: 'Common-observation cell differences',
      geometry: 'none',
      identityColumns: ['cell_index'],
      maximumRows: REGIONAL_RASTER_CHANGE_LIMITS.cells
    }
  ],
  cleanup:
    'The catalog GPU session destroys graphs and storage in finally. Returned tables own plain arrays; polygon geometry publishes through native Sources.',
  declaredBehavior: [
    'Two decoded single bands, already calibrated in the same units, must have exactly aligned grids. Validity, noDataValue and non-finite samples all mark missing observations. Float32 overflow is missing. Differences are after minus before; missing observations never become zero change.',
    'Only axis-aligned planar uniform-area grids with positive column spacing are supported, in caller-confirmed common linear units. Both row directions and area/point pixel interpretation are retained. Area means squared coordinate units, not necessarily square metres. Geographic, Web Mercator, rotated and sheared grids must be prepared upstream.',
    'Maximum aggregate area (Float32 cell area times grid cell count) must not exceed 2^127 squared coordinate units, reserving headroom below Float32 overflow. This bounds every regional total, valid and changed area, even if one zone owns the entire grid. Rescale all inputs consistently to larger linear coordinate units upstream when this bound is exceeded.',
    'The GPU polygon rasterizer assigns each whole cell by its centre, respecting holes and multipart features; overlaps belong to the smallest source row, outside cells have no zone. Coverage is common-observation area divided by assigned grid area, not full polygon area outside the grid. Boundary support is approximate.',
    'Zone means are unweighted arithmetic cell means. Changed area uses abs(after - before) >= threshold. No observations gives a null mean and changed share, zero valid area and zero coverage when a zone has assigned cells; zones with no assigned cells have null coverage. These are descriptive differences, not significance tests.',
    'Rasterization crossing overflow or any recipe stage overflow/invalid threshold rejects publication. Limits: 65536 cells per band, 1024 zones, 32768 vertices, 1048576 scanline crossings. Source row identity and optional source feature ID are preserved.'
  ],
  limitation:
    'GPU only; bounded materialization. Planar grids only, no decoding, calibration, reprojection, resampling or inferential claims. Cell-centre ownership approximates boundaries and depends on zone order for overlaps.',
  async runDataGpu(context, inputs, parameters) {
    const prepared = prepareRegionalRasterChange(inputs, parameters);
    return withCatalogGpuSession(context.device, BINDING_ID, async session => {
      const {grid, polygons, cellCount} = prepared;
      const zones = session.output('zones', 'uint32', cellCount);
      const overflow = session.output('raster-overflow', 'uint32', 1);
      const crossingCount = session.output('crossing-count', 'uint32', 1);
      session.graph.add(
        new GPUPolygonRasterization({
          id: `${BINDING_ID}-ownership`,
          grid,
          parameters: session.input('placement', prepared.placement, 'float32'),
          positions: session.input('positions', polygons.positions, 'float32x2'),
          featureOffsets: session.input('features', polygons.featureOffsets, 'uint32'),
          polygonOffsets: session.input('polygons', polygons.polygonOffsets, 'uint32'),
          ringOffsets: session.input('rings', polygons.ringOffsets, 'uint32'),
          crossingCapacity: prepared.crossingCapacity,
          output: {
            zones: zones.view,
            status: {overflow: overflow.view, crossingCount: crossingCount.view}
          }
        })
      );
      const recipe = addRegionalRasterChangeRecipe(session.graph, {
        id: BINDING_ID,
        grid,
        change: {
          slices: session.input('slices', prepared.slices, 'float32'),
          observationMask: session.input('observations', prepared.observationMask, 'uint32'),
          sliceCount: 2,
          parameters: session.input(
            'change-parameters',
            getGPUChangeDetectionParameterValues({beforeSlice: 0, afterSlice: 1}),
            'float32'
          )
        },
        zones: zones.view,
        zoneCount: polygons.featureCount,
        cellAreas: session.input(
          'areas',
          new Float32Array(cellCount).fill(prepared.cellArea),
          'float32'
        ),
        threshold: session.input('threshold', new Float32Array([prepared.threshold]), 'float32')
      });
      const reads = Object.entries(recipe.outputs).map(([name, source]) => {
        const output = session.output(`read-${name}`, source.format, source.length);
        session.graph.add(
          createWGSLKernelNode(session.graph, {
            id: `publish-${name}`,
            operation: BINDING_ID,
            invocationCount: source.length,
            bindings: [
              {
                name: 'source',
                view: source,
                type: source.format === 'uint32' ? 'u32' : 'f32',
                access: 'read'
              },
              {
                name: 'destination',
                view: output.view,
                type: source.format === 'uint32' ? 'u32' : 'f32',
                access: 'read_write'
              }
            ],
            body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
          })
        );
        return {name, output};
      });
      const statuses = recipe.status.stages.flatMap(stage =>
        Object.entries(stage.status).map(([name, source]) => {
          const output = session.output(`status-${stage.stage}-${name}`, 'uint32', 1);
          session.graph.add(
            createWGSLKernelNode(session.graph, {
              id: `publish-status-${stage.stage}-${name}`,
              operation: BINDING_ID,
              invocationCount: 1,
              bindings: [
                {name: 'source', view: source, type: 'u32', access: 'read'},
                {name: 'destination', view: output.view, type: 'u32', access: 'read_write'}
              ],
              body: 'destination[destinationOffset] = source[sourceOffset];'
            })
          );
          return {name: `${stage.stage}.${name}`, output};
        })
      );
      session.run();
      const statusValues = await Promise.all(
        statuses.map(async status => ({
          name: status.name,
          value: await session.readScalar(status.output)
        }))
      );
      const rasterOverflow = await session.readScalar(overflow);
      const requiredCrossings = await session.readScalar(crossingCount);
      validateRegionalRasterChangeStatus({
        rasterOverflow,
        requiredCrossings,
        crossingCapacity: prepared.crossingCapacity,
        stages: statusValues
      });
      const results = Object.fromEntries(
        await Promise.all(
          reads.map(async ({name, output}) => [
            name,
            output.view.format === 'uint32'
              ? await session.readUint32(output)
              : await session.readFloat32(output)
          ])
        )
      );
      const regions = makeCatalogTable(
        'regions',
        'Regional change and coverage',
        {
          zone_index: 'int32',
          source_id: 'string',
          valid_count: 'int32',
          mean_change: 'float64',
          total_area: 'float64',
          valid_area: 'float64',
          changed_area: 'float64',
          coverage: 'float64',
          changed_share: 'float64',
          geometry: 'string'
        },
        'geometry'
      );
      regions.coordinateSystem = 'planar';
      const differences = makeCatalogTable('differences', 'Common-observation cell differences', {
        cell_index: 'int32',
        zone_index: 'int32',
        common_observation: 'boolean',
        difference: 'float64'
      });
      const normalizeFinite = (value: number) => (Number.isFinite(value) ? value : null);
      for (let zone = 0; zone < polygons.featureCount; zone++) {
        const feature = prepared.zones.features[polygons.sourceIndices[zone]];
        const validArea = results['validArea'][zone];
        appendCatalogRow(regions, {
          zone_index: polygons.sourceIndices[zone],
          source_id: getFeatureIdentity(feature),
          valid_count: results['validCounts'][zone],
          mean_change: results['validCounts'][zone]
            ? normalizeFinite(results['meanChanges'][zone])
            : null,
          total_area: normalizeFinite(results['totalArea'][zone]),
          valid_area: normalizeFinite(validArea),
          changed_area: normalizeFinite(results['changedArea'][zone]),
          coverage: normalizeFinite(results['coverage'][zone]),
          changed_share:
            validArea > 0 ? normalizeFinite(results['changedArea'][zone] / validArea) : null,
          geometry: JSON.stringify(feature.geometry)
        });
      }
      const assignments = await session.readUint32(zones);
      let commonCount = 0;
      let outsideCount = 0;
      for (let cell = 0; cell < cellCount; cell++) {
        const difference = normalizeFinite(results['difference'][cell]);
        if (difference !== null) commonCount++;
        if (assignments[cell] === 0xffffffff) outsideCount++;
        appendCatalogRow(differences, {
          cell_index: cell,
          zone_index:
            assignments[cell] === 0xffffffff ? null : polygons.sourceIndices[assignments[cell]],
          common_observation: difference !== null,
          difference
        });
      }
      return {
        tables: [regions, differences],
        diagnostics: [
          {
            code: 'cell-centre-support',
            severity: 'warning',
            count: cellCount,
            message:
              'Whole-cell ownership by centre approximates boundaries; overlaps belong to the first zone. Coverage refers only to assigned grid cells.'
          },
          ...(commonCount < cellCount
            ? [
                {
                  code: 'missing-common-observations',
                  severity: 'warning' as const,
                  count: cellCount - commonCount,
                  message: 'Cells missing in either band were excluded from change summaries.'
                }
              ]
            : [])
        ],
        dispatchedContributors: CURATED_REGIONAL_RASTER_CHANGE_BINDING.gpuContributorNames,
        summary: {
          cellCount,
          zoneCount: polygons.featureCount,
          commonObservationCount: commonCount,
          outsideCellCount: outsideCount,
          threshold: prepared.threshold,
          areaUnit: 'squared coordinate units',
          differenceUnit: 'calibrated band units',
          beforeId: prepared.before.id,
          afterId: prepared.after.id,
          requiredCrossings,
          crossingCapacity: prepared.crossingCapacity,
          rasterOverflow,
          recipeStatus: JSON.stringify(statusValues),
          gridAffine: JSON.stringify(prepared.before.grid.affine)
        }
      };
    });
  }
};
