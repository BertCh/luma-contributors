// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {KeplerDatasetRegistry} from '@luma-contributors/kepler';
import {createRasterBandGpuUploader} from './raster-band-gpu';
import {RasterBandResidency, type RasterBandResidencyStatistics} from './raster-band-model';
import {identifyRasterValue, type RasterIdentifyResult} from './raster-identify';
import {
  createNumericSourceFromKeplerTiles,
  getKeplerTileSignature,
  type KeplerLoadedRasterTile
} from './raster-live-source';
import {
  computeRasterZonalStatistics,
  type RasterZonalOptions,
  type RasterZonalPolygon,
  type RasterZonalStatistics
} from './raster-zonal';
import type {RasterWgslBandTextures} from './raster-wgsl-layer';
import type {NumericRasterSource} from './raster-tile-ingest';

export const DEFAULT_RASTER_LIVE_BUDGET_BYTES = 64 * 1024 * 1024;

/** The parts of a Kepler `RasterTileLayer` instance this controller reads. */
export type KeplerRasterLayerLike = {
  id: string;
  type: string;
  config?: {label?: string; isVisible?: boolean};
  _loadedTiles?: readonly (KeplerLoadedRasterTile | null | undefined)[];
};

type LiveEntry = {
  signature: string;
  source: NumericRasterSource;
  residency?: RasterBandResidency;
  residencyDevice?: Device;
  rebuildCount: number;
};

export type RasterLiveSnapshot = Readonly<{
  layerId: string;
  tileCount: number;
  bandCount: number;
  zooms: readonly number[];
  rebuildCount: number;
  residency: RasterBandResidencyStatistics | null;
}>;

/**
 * Feeds the tiles Kepler's native raster tile layer has fetched and decoded into
 * `NumericRasterSource` objects, and holds band 1 of each loaded tile in budgeted GPU textures.
 * Kepler still draws the layer; this controller owns the numeric view used by identify and zonal
 * statistics. The CPU source is the authority, so budget eviction never changes results.
 */
export class RasterLiveController {
  private readonly entries = new Map<string, LiveEntry>();

  constructor(
    private readonly options: {
      getLayers(): readonly KeplerRasterLayerLike[];
      getDevice(): Device | undefined;
      getMemoryRegistry?(): KeplerDatasetRegistry | undefined;
      budgetByteLength?: number;
    }
  ) {}

  getRasterLayerIds(): string[] {
    return this.options
      .getLayers()
      .filter(layer => layer.type === 'rasterTile')
      .map(layer => layer.id);
  }

  /** Rebuilds the numeric source when the loaded tile set changed; returns it, or undefined. */
  refreshSource(layerId: string): NumericRasterSource | undefined {
    const layer = this.options.getLayers().find(candidate => candidate.id === layerId);
    if (!layer || layer.type !== 'rasterTile') return undefined;
    const loadedTiles = layer._loadedTiles ?? [];
    const signature = getKeplerTileSignature(loadedTiles);
    let entry = this.entries.get(layerId);
    if (entry && entry.signature === signature) return entry.source;
    const source = createNumericSourceFromKeplerTiles(layerId, loadedTiles);
    entry = {
      signature,
      source,
      residency: entry?.residency,
      residencyDevice: entry?.residencyDevice,
      rebuildCount: (entry?.rebuildCount ?? 0) + 1
    };
    this.entries.set(layerId, entry);
    this.uploadBands(entry);
    return source;
  }

  private uploadBands(entry: LiveEntry): void {
    const device = this.options.getDevice();
    if (!device) return;
    if (entry.residency && entry.residencyDevice !== device) {
      // A new device invalidates every texture; the old residency is dropped without reuse.
      entry.residency.destroy();
      entry.residency = undefined;
    }
    entry.residency ??= new RasterBandResidency({
      budgetByteLength: this.options.budgetByteLength ?? DEFAULT_RASTER_LIVE_BUDGET_BYTES,
      upload: createRasterBandGpuUploader(device),
      memoryRegistry: this.options.getMemoryRegistry?.()
    });
    entry.residencyDevice = device;
    // Highest zoom first, so the budget keeps the tiles identify prefers.
    const tiles = [...entry.source.tiles].sort((first, second) => second.zoom - first.zoom);
    for (const tile of tiles) {
      try {
        if (tile.bands.length > 0) entry.residency.ensureBandResident(tile, tile.bands[0].index);
      } catch {
        // A band larger than the budget or a registry refusal leaves that tile CPU-only.
        break;
      }
    }
  }

  /**
   * Makes the requested bands (zero-based `imageBands` positions) of one loaded tile resident.
   * Called while layers update, never while drawing, so budget eviction cannot destroy a texture
   * that a recorded draw still uses.
   */
  ensureTileBandsResident(
    layerId: string,
    tileIndex: {x: number; y: number; z: number},
    positions: readonly number[]
  ): void {
    const device = this.options.getDevice();
    if (!device) return;
    let entry = this.entries.get(layerId);
    const tileId = `${layerId}/${tileIndex.z}/${tileIndex.x}/${tileIndex.y}`;
    let tile = entry?.source.tiles.find(candidate => candidate.id === tileId);
    if (!tile) {
      this.refreshSource(layerId);
      entry = this.entries.get(layerId);
      tile = entry?.source.tiles.find(candidate => candidate.id === tileId);
    }
    if (!entry || !tile) return;
    entry.residency ??= new RasterBandResidency({
      budgetByteLength: this.options.budgetByteLength ?? DEFAULT_RASTER_LIVE_BUDGET_BYTES,
      upload: createRasterBandGpuUploader(device),
      memoryRegistry: this.options.getMemoryRegistry?.()
    });
    entry.residencyDevice = device;
    for (const position of new Set(positions))
      try {
        if (tile.bands.some(band => band.index === position + 1))
          entry.residency.ensureBandResident(tile, position + 1);
      } catch {
        // Over budget or refused by the registry: the layer then uses private uploads.
      }
  }

  /** Resident texture pairs for zero-based band positions; undefined entries are not resident. */
  peekTileBandTextures(
    layerId: string,
    tileIndex: {x: number; y: number; z: number},
    positions: readonly number[]
  ): (RasterWgslBandTextures | undefined)[] | undefined {
    const residency = this.entries.get(layerId)?.residency;
    if (!residency) return undefined;
    const tileId = `${layerId}/${tileIndex.z}/${tileIndex.x}/${tileIndex.y}`;
    return positions.map(position => {
      const resource = residency.peekBandResource(tileId, position + 1);
      return resource?.valueTexture && resource.validityTexture
        ? {valueTexture: resource.valueTexture, validityTexture: resource.validityTexture}
        : undefined;
    });
  }

  /** Resident band resource for GPU readback and zonal statistics. */
  getResidentBandResource(layerId: string, tileId: string, bandIndex: number) {
    return this.entries.get(layerId)?.residency?.peekBandResource(tileId, bandIndex);
  }

  getSource(layerId: string): NumericRasterSource | undefined {
    return this.refreshSource(layerId);
  }

  identify(
    layerId: string,
    longitude: number,
    latitude: number,
    options?: Parameters<typeof identifyRasterValue>[3]
  ): RasterIdentifyResult | undefined {
    const source = this.refreshSource(layerId);
    return source ? identifyRasterValue(source, longitude, latitude, options) : undefined;
  }

  computeZonal(
    layerId: string,
    polygons: readonly RasterZonalPolygon[],
    options?: RasterZonalOptions
  ): RasterZonalStatistics | undefined {
    const source = this.refreshSource(layerId);
    // No loaded tile intersects: Math.max() over nothing would be -Infinity, so answer with an empty result.
    if (!source || source.tiles.length === 0) return undefined;
    return computeRasterZonalStatistics(source, polygons, options);
  }

  getSnapshot(layerId: string): RasterLiveSnapshot | undefined {
    const source = this.refreshSource(layerId);
    const entry = this.entries.get(layerId);
    if (!source || !entry) return undefined;
    return {
      layerId,
      tileCount: source.tiles.length,
      bandCount: source.bandCount,
      zooms: [...new Set(source.tiles.map(tile => tile.zoom))].sort((a, b) => a - b),
      rebuildCount: entry.rebuildCount,
      residency: entry.residency?.getStatistics() ?? null
    };
  }

  /** Releases textures and registry reservations for one layer (or all). */
  release(layerId?: string): void {
    for (const [id, entry] of [...this.entries])
      if (!layerId || id === layerId) {
        entry.residency?.destroy();
        this.entries.delete(id);
      }
  }
}
