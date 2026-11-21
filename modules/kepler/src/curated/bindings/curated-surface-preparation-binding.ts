// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '@luma-contributors/analysis/analysis-core';
import type {GPURasterBufferBand} from '@luma.gl/graph/gpu-raster';
import {addSurfacePreparationRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {
  CatalogOperationBinding,
  CatalogParameterValue,
  CatalogRunOutput
} from '../../analysis/catalog-binding-types';
import type {CatalogDataInput} from '../../analysis/catalog-data-input';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  makeCatalogTable
} from '../../analysis/catalog-geojson';
import {withCatalogGpuSession, type CatalogGpuSession} from '../../analysis/catalog-gpu-session';
import {
  validateCatalogRasterInput,
  type CatalogRasterInput
} from '../../analysis/catalog-raster-input';

const BINDING_ID = 'curated-surface-preparation';
/** Bound readback, polygon publication and recipe scratch for interactive catalog runs. */
export const MAXIMUM_SURFACE_PREPARATION_CELLS = 65536;
/** Bound the brute-force disk search conservatively by its square candidate window. */
const MAXIMUM_SEARCH_SAMPLES = 16777216;
/** Smaller than the contributor cap to keep the catalog shader's search bounded. */
const MAXIMUM_FILL_DISTANCE = 16;
/** Bound the rank filter's per-invocation insertion-sort footprint to 81 samples. */
const MAXIMUM_FILTER_RADIUS = 4;

type Parameters = Readonly<Record<string, CatalogParameterValue>>;
export type SurfacePreparationSettings = {
  maximumDistance: number;
  method: 'nearest' | 'inverse-distance';
  filter: boolean;
  radius: number;
  quantile: number;
  borderMode: 'clamp' | 'nodata';
  noDataPolicy: 'ignore' | 'propagate';
};

/** Validate the bounded native grid and shader settings once before allocating GPU resources. */
export function prepareSurfacePreparation(
  inputs: Readonly<Record<string, CatalogDataInput>>,
  parameters: Parameters
): {surface: CatalogRasterInput; settings: SurfacePreparationSettings} {
  const surface = inputs['surface'];
  validateRequirement(
    BINDING_ID,
    surface?.type === 'raster',
    'surface must be a decoded raster band with a complete bounded grid'
  );
  const raster = surface as CatalogRasterInput;
  validateCatalogRasterInput(BINDING_ID, raster, MAXIMUM_SURFACE_PREPARATION_CELLS);
  validateMaximumCount(BINDING_ID, 'band.index (int32 publication)', raster.band.index, 2147483647);
  const settings = Object.fromEntries(
    CURATED_SURFACE_PREPARATION_BINDING.parameters.map(parameter => [
      parameter.name,
      parameters[parameter.name] ?? parameter.defaultValue
    ])
  ) as SurfacePreparationSettings;
  validateCapacity(
    BINDING_ID,
    'maximumDistance',
    settings.maximumDistance,
    1,
    MAXIMUM_FILL_DISTANCE
  );
  validateRequirement(
    BINDING_ID,
    ['nearest', 'inverse-distance'].includes(settings.method),
    'method must be nearest or inverse-distance'
  );
  validateRequirement(BINDING_ID, typeof settings.filter === 'boolean', 'filter must be boolean');
  validateCapacity(BINDING_ID, 'radius', settings.radius, 0, MAXIMUM_FILTER_RADIUS);
  validateRequirement(
    BINDING_ID,
    Number.isFinite(settings.quantile) && settings.quantile >= 0 && settings.quantile <= 1,
    'quantile must be finite in [0,1]'
  );
  validateRequirement(
    BINDING_ID,
    ['clamp', 'nodata'].includes(settings.borderMode),
    'borderMode must be clamp or nodata'
  );
  validateRequirement(
    BINDING_ID,
    ['ignore', 'propagate'].includes(settings.noDataPolicy),
    'noDataPolicy must be ignore or propagate'
  );
  validateMaximumCount(
    BINDING_ID,
    'search samples (grid cells times search window)',
    raster.grid.width * raster.grid.height * (2 * settings.maximumDistance + 1) ** 2,
    MAXIMUM_SEARCH_SAMPLES
  );
  return {surface: raster, settings};
}

/**
 * Encode source-domain nodata in validity before any lossy conversion. The recipe must not
 * compare a rounded sentinel against rounded donors. This is upload preparation, not a CPU
 * validation pass; non-finite sample handling remains with the recipe's canonicalizer.
 */
function uploadSurfacePreparationBand(
  session: CatalogGpuSession,
  surface: CatalogRasterInput
): GPURasterBufferBand {
  const {values, validity, noDataValue} = surface.band;
  const sourceValidity = Uint32Array.from(validity, (mask, index) =>
    Number(mask !== 0 && (noDataValue === null || values[index] !== noDataValue))
  );
  const metadata = {
    id: surface.id,
    validity: session.input('source-validity', sourceValidity, 'uint32')
  };
  if (values instanceof Int32Array || values instanceof Int16Array || values instanceof Int8Array) {
    return {
      ...metadata,
      format: 'sint32',
      storage: {
        kind: 'buffer',
        values: session.input(
          'source-values',
          values instanceof Int32Array ? values : Int32Array.from(values),
          'sint32'
        )
      }
    };
  }
  if (
    values instanceof Uint32Array ||
    values instanceof Uint16Array ||
    values instanceof Uint8Array
  ) {
    return {
      ...metadata,
      format: 'uint32',
      storage: {
        kind: 'buffer',
        values: session.input(
          'source-values',
          values instanceof Uint32Array ? values : Uint32Array.from(values),
          'uint32'
        )
      }
    };
  }
  return {
    ...metadata,
    format: 'float32',
    storage: {
      kind: 'buffer',
      values: session.input(
        'source-values',
        values instanceof Float32Array ? values : Float32Array.from(values),
        'float32'
      )
    }
  };
}

export type SurfacePreparationReadback = {
  values: Float32Array;
  validity: Uint32Array;
  originalValidity: Uint32Array;
  filled: Uint32Array;
  filtered: Uint32Array;
  unresolvedCount: number;
};

/** Publish every source cell, including gaps, with stable row-major identity and independent masks. */
export function publishSurfacePreparation(
  surface: CatalogRasterInput,
  settings: SurfacePreparationSettings,
  result: SurfacePreparationReadback
): CatalogRunOutput {
  const geographic = surface.crs === 'EPSG:4326';
  const table = makeCatalogTable(
    'prepared-surface',
    'Prepared surface and provenance',
    {
      source_id: 'string',
      band_index: 'int32',
      cell_index: 'int32',
      row: 'int32',
      column: 'int32',
      value: 'float64',
      valid: 'boolean',
      original_valid: 'boolean',
      filled: 'boolean',
      filtered: 'boolean',
      status: 'string',
      ...(geographic ? {geometry: 'string' as const} : {})
    },
    geographic ? 'geometry' : undefined
  );
  table.raster = {
    ...surface,
    id: table.id,
    band: {
      ...surface.band,
      name: table.label,
      values: result.values,
      validity: result.validity,
      // Source sentinels have already been resolved into output validity.
      noDataValue: null
    }
  };
  if (geographic) table.coordinateSystem = 'EPSG:4326';
  const {width, height, affine, pixelInterpretation} = surface.grid;
  let filledCount = 0;
  let filteredCount = 0;
  let originalCount = 0;
  let filterInvalidatedCount = 0;
  for (let cellIndex = 0; cellIndex < width * height; cellIndex++) {
    const row = Math.floor(cellIndex / width);
    const column = cellIndex % width;
    const valid = result.validity[cellIndex] !== 0;
    const original = result.originalValidity[cellIndex] !== 0;
    const filled = result.filled[cellIndex] !== 0;
    const filtered = result.filtered[cellIndex] !== 0;
    filledCount += Number(filled);
    filteredCount += Number(filtered);
    originalCount += Number(original);
    filterInvalidatedCount += Number(!valid && (original || filled));
    const offset = pixelInterpretation === 'point' ? -0.5 : 0;
    const ring = [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
      [0, 0]
    ].map(([horizontal, vertical]) => {
      const pixelColumn = column + offset + horizontal;
      const pixelRow = row + offset + vertical;
      return [
        affine[0] * pixelColumn + affine[1] * pixelRow + affine[2],
        affine[3] * pixelColumn + affine[4] * pixelRow + affine[5]
      ];
    });
    appendCatalogRow(table, {
      source_id: surface.id,
      band_index: surface.band.index,
      cell_index: cellIndex,
      row,
      column,
      value: valid ? result.values[cellIndex] : null,
      valid,
      original_valid: original,
      filled,
      filtered,
      status: !valid
        ? original || filled
          ? 'filter-invalidated'
          : 'unresolved'
        : filtered
          ? 'filtered'
          : filled
            ? 'filled'
            : 'observed',
      ...(geographic ? {geometry: JSON.stringify({type: 'Polygon', coordinates: [ring]})} : {})
    });
  }
  const diagnostics: CatalogRunOutput['diagnostics'] = [];
  if (result.unresolvedCount)
    addCatalogDiagnostic(
      diagnostics,
      'unresolved-cells',
      'warning',
      'Cells remain invalid after bounded fill and optional filtering; value is null. Increase the search distance or retain them as gaps.',
      result.unresolvedCount
    );
  if (filterInvalidatedCount)
    addCatalogDiagnostic(
      diagnostics,
      'filter-invalidated-cells',
      'warning',
      'The filter invalidated observed or filled centers because nodata was propagated.',
      filterInvalidatedCount
    );
  if (!geographic)
    addCatalogDiagnostic(
      diagnostics,
      'projected-table-only',
      'info',
      'Projected cell coordinates are retained by the source grid; map publication requires a host CRS transform. This result is a table.'
    );
  return {
    tables: [table],
    diagnostics,
    summary: {
      inputCount: width * height,
      originalCount,
      filledCount,
      filteredCount,
      unresolvedCount: result.unresolvedCount,
      filterInvalidatedCount,
      sourceId: surface.id,
      bandIndex: surface.band.index,
      crs: surface.crs,
      gridWidth: width,
      gridHeight: height,
      gridAffine: JSON.stringify(affine),
      pixelInterpretation,
      maximumDistance: settings.maximumDistance,
      distanceUnit: 'pixels',
      method: settings.method,
      filter: String(settings.filter),
      filterRadius: settings.radius,
      quantile: settings.quantile,
      borderMode: settings.borderMode,
      noDataPolicy: settings.noDataPolicy
    }
  };
}

export const CURATED_SURFACE_PREPARATION_BINDING: CatalogOperationBinding = {
  id: BINDING_ID,
  title: 'Surface preparation',
  domain: 'raster',
  summary:
    'Fill bounded gaps from original donors, optionally rank-filter, and retain independent observed, filled and filtered provenance.',
  capabilityIds: ['raster-focal'],
  contributorNames: ['GPURasterFillNoData', 'GPURasterRankFilter'],
  gpuContributorNames: ['GPURasterFillNoData', 'GPURasterRankFilter'],
  engine: 'gpu',
  inputs: [
    {
      name: 'surface',
      label: 'Raster with gaps',
      dataKind: 'raster',
      accepts: [],
      maximumFeatures: 0,
      maximumVertices: 0,
      maximumCells: MAXIMUM_SURFACE_PREPARATION_CELLS
    }
  ],
  parameters: [
    {
      name: 'maximumDistance',
      label: 'Maximum fill distance',
      kind: 'integer',
      unit: 'pixels',
      minimum: 1,
      maximum: MAXIMUM_FILL_DISTANCE,
      defaultValue: 2
    },
    {
      name: 'method',
      label: 'Fill method',
      kind: 'enum',
      options: ['nearest', 'inverse-distance'],
      defaultValue: 'inverse-distance'
    },
    {name: 'filter', label: 'Apply rank filter', kind: 'boolean', defaultValue: false},
    {
      name: 'radius',
      label: 'Filter radius',
      kind: 'integer',
      unit: 'pixels',
      minimum: 0,
      maximum: MAXIMUM_FILTER_RADIUS,
      defaultValue: 1
    },
    {
      name: 'quantile',
      label: 'Filter quantile',
      kind: 'number',
      minimum: 0,
      maximum: 1,
      defaultValue: 0.5
    },
    {
      name: 'borderMode',
      label: 'Filter border',
      kind: 'enum',
      options: ['clamp', 'nodata'],
      defaultValue: 'nodata'
    },
    {
      name: 'noDataPolicy',
      label: 'Filter nodata',
      kind: 'enum',
      options: ['ignore', 'propagate'],
      defaultValue: 'ignore'
    }
  ],
  outputs: [
    {
      tableId: 'prepared-surface',
      dataKind: 'raster',
      label: 'Prepared surface',
      geometry: 'polygon',
      identityColumns: ['source_id', 'band_index', 'cell_index'],
      maximumRows: MAXIMUM_SURFACE_PREPARATION_CELLS
    }
  ],
  cleanup:
    'Readback retains typed raster arrays beside display tables; the GPU session releases buffers and compiled graphs in finally. The caller owns the input arrays and device.',
  declaredBehavior: [
    'Input is a complete decoded native raster band; no generated grid or point conversion is used.',
    'Nodata is classified in the source representation and encoded in validity before upload; no rounded sentinel is passed to the GPU. Integer bands use signed or unsigned integer storage. Only original valid finite float32 donors fill gaps. Filled cells never become donors; distance is Euclidean pixels, not ground distance.',
    'Nearest ties use the recipe scan order; inverse-distance uses all original donors in the bounded disk with weight 1/distance.',
    'Filtering uses the recipe rank floor(quantile * sample count), capped at sample count minus one; invalid centers stay invalid.',
    'Every cell is published in source order, invalid values as null. Original, filled and filtered masks are independent; unresolvedCount is after filtering.',
    'EPSG:4326 cells publish native polygon footprints; projected grids publish a table pending a host CRS transform.',
    'At most 65536 cells and 16777216 candidate search samples; no automatic resampling or CPU fallback.'
  ],
  limitation:
    'Requires WebGPU. Integer inputs retain exact GPU storage; recipe calculations and outputs use float32. Float64 input values are converted after source nodata classification. Distances and square filter windows are in pixels, including on non-square geographic grids; no physical-distance interpolation or uncertainty is claimed. Projected grids have table publication only.',
  async runDataGpu(context, inputs, parameters) {
    const {surface, settings} = prepareSurfacePreparation(inputs, parameters);
    return withCatalogGpuSession(context.device, BINDING_ID, async session => {
      const count = surface.grid.width * surface.grid.height;
      const values = session.output('values', 'float32', count);
      const validity = session.output('validity', 'uint32', count);
      const originalValidity = session.output('original', 'uint32', count);
      const filled = session.output('filled', 'uint32', count);
      const filtered = session.output('filtered', 'uint32', count);
      const unresolvedCount = session.output('unresolved', 'uint32', 1);
      addSurfacePreparationRecipe(session.graph, {
        id: BINDING_ID,
        grid: surface.grid,
        input: uploadSurfacePreparationBand(session, surface),
        maximumDistance: settings.maximumDistance,
        method: settings.method,
        ...(settings.filter
          ? {
              filter: {
                radius: settings.radius,
                quantile: settings.quantile,
                borderMode: settings.borderMode,
                noDataPolicy: settings.noDataPolicy
              }
            }
          : {}),
        output: {
          values: values.view,
          validity: validity.view,
          originalValidity: originalValidity.view,
          filled: filled.view,
          filtered: filtered.view,
          unresolvedCount: unresolvedCount.view
        }
      });
      session.run();
      const [valueData, validityData, originalData, filledData, filteredData, unresolved] =
        await Promise.all([
          session.readFloat32(values),
          session.readUint32(validity),
          session.readUint32(originalValidity),
          session.readUint32(filled),
          session.readUint32(filtered),
          session.readScalar(unresolvedCount)
        ]);
      return {
        ...publishSurfacePreparation(surface, settings, {
          values: valueData,
          validity: validityData,
          originalValidity: originalData,
          filled: filledData,
          filtered: filteredData,
          unresolvedCount: unresolved
        }),
        dispatchedContributors: settings.filter
          ? ['GPURasterFillNoData', 'GPURasterRankFilter']
          : ['GPURasterFillNoData']
      };
    });
  }
};
