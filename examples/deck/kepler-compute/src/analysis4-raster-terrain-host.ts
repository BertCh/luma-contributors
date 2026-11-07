// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type * as arrow from 'apache-arrow';
import type {RasterTerrainOperation} from './analysis2-raster-terrain';
import {
  createTerrainRasterSourceAsync,
  type RasterTerrainEngine
} from './analysis3-raster-terrain-gpu';
import {
  TERRAIN_CELL_TABLE_LIMIT,
  createNumericRasterSourceFromGrid,
  createTerrainCellTable,
  createTerrainMosaic,
  encodeTerrainMosaicGeoTIFF,
  type DerivedGridInput,
  type TerrainMosaic
} from './analysis4-terrain-grid';
import type {NumericRasterSource} from './raster-tile-ingest';

/**
 * Terrain panel runtime: slope, aspect, hillshade and curvature over the numeric TILES raster
 * source or a derived IDW/kriging grid, GPU first with the CPU reference as fallback. The result
 * is published as a native cell dataset (one polygon per valid cell) and kept for GeoTIFF export.
 */
export type TerrainActionInput =
  | Readonly<{kind: 'tiles'; rasterLayerId: string}>
  | Readonly<{kind: 'derived-grid'}>;

export type TerrainActionRequest = Readonly<{
  input: TerrainActionInput;
  operation: RasterTerrainOperation;
  bandIndex?: number;
  zFactor?: number;
  azimuthDegrees?: number;
  altitudeDegrees?: number;
  /** Default 'gpu' (CPU reference when no device or on any GPU failure). */
  engine?: 'gpu' | 'cpu';
  /** Publish the native cell dataset. Default true; export is available either way. */
  publish?: boolean;
  /**
   * 'table' (default) reads the terrain back and publishes polygon cells. 'resident' keeps the
   * cells on the GPU (point centers, nodata rows null) and never reads them back; it needs
   * `publishResident`, a device and a TILES input, and otherwise falls back to 'table' with a note.
   */
  publication?: 'table' | 'resident';
  label?: string;
}>;

export type TerrainActionResult = Readonly<{
  published: boolean;
  label: string;
  rowCount: number;
  engine?: RasterTerrainEngine;
  engineNote?: string;
  fallbackReason?: string;
  validCellCount: number;
  width: number;
  height: number;
  /** A GeoTIFF export of this result is available. */
  exportAvailable: boolean;
  reason?: string;
  /** How the cells were published: GPU-resident without readback, or as a readback table. */
  publication?: 'resident' | 'table';
  /** Bytes read back to publish (0 for table-free resident publication except small summaries). */
  publicationReadbackBytes?: number;
}>;

export type DerivedGridReader = Readonly<{
  isAvailable(): boolean;
  read(): Promise<DerivedGridInput & {sourceId: string; operation: string}>;
}>;

export type RasterTerrainHostOptions = {
  getDevice(): Device | undefined;
  getTilesSource(rasterLayerId: string): NumericRasterSource | undefined;
  getDerivedGrid?(): DerivedGridReader | undefined;
  publishTable(options: {table: arrow.Table; label: string; geometryColumn: string}): Promise<void>;
  /** Publishes the terrain of a TILES source as a GPU-resident dataset; no cell readback. */
  publishResident?(options: {
    source: NumericRasterSource;
    operation: RasterTerrainOperation;
    bandIndex?: number;
    zFactor?: number;
    azimuthDegrees?: number;
    altitudeDegrees?: number;
    label: string;
  }): Promise<{rowCount: number; contributors: readonly string[]; readbackBytes: number}>;
};

type StoredResidentRequest = {request: TerrainActionRequest; label: string};

type StoredTerrainResult = {
  mosaic: TerrainMosaic;
  operation: RasterTerrainOperation;
  filenameBase: string;
  provenance: Record<string, unknown>;
};

const EMPTY_RESULT = {validCellCount: 0, width: 0, height: 0, exportAvailable: false};

export function createRasterTerrainHost(options: RasterTerrainHostOptions) {
  let stored: StoredTerrainResult | undefined;
  let storedResident: StoredResidentRequest | undefined;

  async function runTerrainAction(request: TerrainActionRequest): Promise<TerrainActionResult> {
    const label = request.label ?? `Terrain ${request.operation}`;
    stored = undefined;
    storedResident = undefined;
    let residentFallbackReason: string | undefined;
    let source: NumericRasterSource | undefined;
    let sourceDescription: Record<string, unknown>;
    if (request.input.kind === 'tiles') {
      source = options.getTilesSource(request.input.rasterLayerId);
      sourceDescription = {kind: 'tiles', rasterLayerId: request.input.rasterLayerId};
      if (!source || source.tiles.length === 0)
        return {
          ...EMPTY_RESULT,
          published: false,
          label,
          rowCount: 0,
          reason: 'No loaded raster tiles'
        };
    } else {
      const reader = options.getDerivedGrid?.();
      if (!reader?.isAvailable())
        return {
          ...EMPTY_RESULT,
          published: false,
          label,
          rowCount: 0,
          reason: 'Run IDW or kriging first; no derived grid is available'
        };
      const grid = await reader.read();
      source = createNumericRasterSourceFromGrid({
        ...grid,
        id: `${grid.sourceId}-${grid.operation}`
      });
      sourceDescription = {
        kind: 'derived-grid',
        operation: grid.operation,
        sourceId: grid.sourceId
      };
    }
    if (
      request.publication === 'resident' &&
      request.engine !== 'cpu' &&
      request.publish !== false &&
      request.input.kind === 'tiles' &&
      options.publishResident &&
      options.getDevice()
    ) {
      try {
        const resident = await options.publishResident({
          source,
          operation: request.operation,
          bandIndex: request.bandIndex,
          zFactor: request.zFactor,
          azimuthDegrees: request.azimuthDegrees,
          altitudeDegrees: request.altitudeDegrees,
          label
        });
        // GeoTIFF export has no stored mosaic; it recomputes with a readback on demand.
        storedResident = {request, label};
        return {
          published: true,
          label,
          rowCount: resident.rowCount,
          engine: 'gpu',
          engineNote: `Terrain ${request.operation}: GPU-resident (${resident.contributors.join(', ')}); cells were not read back`,
          validCellCount: 0,
          width: 0,
          height: 0,
          exportAvailable: true,
          publication: 'resident',
          publicationReadbackBytes: resident.readbackBytes
        };
      } catch (error) {
        // Resident publication failed; the table path below answers with a visible reason.
        residentFallbackReason =
          error instanceof Error ? error.message : 'Resident publication failed';
      }
    }
    const terrain = {
      operation: request.operation,
      bandIndex: request.bandIndex,
      zFactor: request.zFactor,
      azimuthDegrees: request.azimuthDegrees,
      altitudeDegrees: request.altitudeDegrees
    };
    const run = await createTerrainRasterSourceAsync(source, terrain, {
      device: options.getDevice(),
      engine: request.engine === 'cpu' ? 'cpu' : undefined
    });
    let mosaic: TerrainMosaic;
    try {
      mosaic = createTerrainMosaic(run.source);
    } catch (error) {
      return {
        ...EMPTY_RESULT,
        published: false,
        label,
        rowCount: 0,
        engine: run.engine,
        engineNote: run.engineNote,
        fallbackReason: run.fallbackReason,
        reason: error instanceof Error ? error.message : 'Terrain mosaic failed'
      };
    }
    stored = {
      mosaic,
      operation: request.operation,
      filenameBase: `terrain-${request.operation}-${source.id.replace(/[^a-zA-Z0-9_-]/g, '_')}`,
      provenance: {
        format: 'luma.gl-kepler-terrain',
        operation: request.operation,
        zFactor: request.zFactor ?? 1,
        azimuthDegrees: request.azimuthDegrees ?? 315,
        altitudeDegrees: request.altitudeDegrees ?? 45,
        engine: run.engine,
        engineNote: run.engineNote,
        source: sourceDescription,
        coordinateReferenceSystem: mosaic.crs
      }
    };
    const base = {
      label,
      engine: run.engine,
      engineNote: run.engineNote,
      fallbackReason: run.fallbackReason,
      validCellCount: mosaic.validCellCount,
      width: mosaic.width,
      height: mosaic.height,
      exportAvailable: true,
      publication: 'table' as const,
      ...(residentFallbackReason
        ? {fallbackReason: run.fallbackReason ?? `Resident publication: ${residentFallbackReason}`}
        : {})
    };
    if (mosaic.validCellCount === 0)
      return {...base, published: false, rowCount: 0, reason: 'Terrain produced no valid cells'};
    if (request.publish === false) return {...base, published: false, rowCount: 0};
    const table = createTerrainCellTable(mosaic, request.operation);
    if (!table)
      return {
        ...base,
        published: false,
        rowCount: 0,
        reason: `More than ${TERRAIN_CELL_TABLE_LIMIT} valid cells; export the GeoTIFF instead`
      };
    await options.publishTable({table, label, geometryColumn: 'geometry'});
    return {...base, published: true, rowCount: table.numRows};
  }

  /** GeoTIFF of the most recent terrain result. */
  async function exportTerrainGeoTIFF(): Promise<{
    data: Uint8Array<ArrayBuffer>;
    filename: string;
  }> {
    // A resident publication keeps no CPU mosaic; the export recomputes with the readback it needs.
    if (!stored && storedResident) {
      await runTerrainAction({
        ...storedResident.request,
        publication: 'table',
        publish: false,
        label: storedResident.label
      });
    }
    // Usage error: export needs a prior run.
    if (!stored) throw new Error('Run a terrain operation before exporting');
    const data = await encodeTerrainMosaicGeoTIFF(
      stored.mosaic,
      stored.operation,
      stored.provenance
    );
    return {data, filename: `${stored.filenameBase}.tif`};
  }

  return {
    runTerrainAction,
    exportTerrainGeoTIFF,
    hasResult: () => Boolean(stored || storedResident),
    isDerivedGridAvailable: () => Boolean(options.getDerivedGrid?.()?.isAvailable()),
    getResultMosaic: () => stored?.mosaic
  };
}

export type RasterTerrainHost = ReturnType<typeof createRasterTerrainHost>;
