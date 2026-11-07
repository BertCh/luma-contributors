// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createBitmapResourceScope,
  readBitmapNumericInput,
  resolveBitmapGeoreference,
  writeBitmapGeoreference,
  type BitmapGeoreference,
  type BitmapNumericInput,
  type KeplerBitmapVisConfig
} from './bitmap-georeference';
import {identifyRasterValue, type RasterIdentifyResult} from './raster-identify';

type KeplerBitmapLayerLike = {
  id: string;
  type: string;
  config: {visConfig: KeplerBitmapVisConfig & Record<string, unknown>};
};

export type BitmapLiveGeoreference = Readonly<{
  layerId: string;
  georeference?: BitmapGeoreference;
  /** Why the live visConfig is not a valid georeference (partial or out-of-range bounds). */
  error?: string;
  hasNumericSource: boolean;
}>;

/**
 * Binds `bitmap-georeference` to Kepler's live bitmap layers. The georeference is always read
 * from the layer's current `visConfig` (the native bounds editor writes those fields), never
 * cached. A GeoTIFF supplies a numeric source and its bounds; a plain image supplies neither.
 */
export class BitmapLiveBinding {
  private readonly numericInputs = new Map<string, BitmapNumericInput>();
  private readonly scopes = new Map<string, ReturnType<typeof createBitmapResourceScope>>();

  constructor(private readonly options: {getLayers(): readonly KeplerBitmapLayerLike[]}) {}

  private getBitmapLayers(): KeplerBitmapLayerLike[] {
    return this.options.getLayers().filter(layer => layer.type === 'bitmap');
  }

  getGeoreferences(): BitmapLiveGeoreference[] {
    return this.getBitmapLayers().map(layer => {
      const hasNumericSource = this.numericInputs.has(layer.id);
      try {
        return {
          layerId: layer.id,
          georeference: resolveBitmapGeoreference(layer.config.visConfig),
          hasNumericSource
        };
      } catch (error) {
        return {layerId: layer.id, error: (error as Error).message, hasNumericSource};
      }
    });
  }

  /**
   * Decodes a GeoTIFF for a bitmap layer and returns the visConfig patch that georeferences the
   * layer to the file extent. The caller dispatches it through Kepler's `layerVisConfigChange`.
   */
  attachGeoTiff(layerId: string, bytes: Uint8Array): Record<string, unknown> {
    const layer = this.getBitmapLayers().find(candidate => candidate.id === layerId);
    // Binding a GeoTIFF to a layer that is not a bitmap layer is a caller error.
    if (!layer) throw new Error(`No bitmap layer ${layerId}`);
    const input = readBitmapNumericInput(bytes, layerId, layer.config.visConfig.opacity ?? 1);
    this.releaseLayer(layerId);
    this.numericInputs.set(layerId, input);
    this.scopes.set(layerId, createBitmapResourceScope());
    const {boundsWest, boundsSouth, boundsEast, boundsNorth} = writeBitmapGeoreference(
      {} as KeplerBitmapVisConfig,
      input.georeference
    );
    return {boundsWest, boundsSouth, boundsEast, boundsNorth};
  }

  /** Pixel values only exist for GeoTIFF-backed bitmaps; plain images answer undefined. */
  identify(layerId: string, longitude: number, latitude: number): RasterIdentifyResult | undefined {
    const input = this.numericInputs.get(layerId);
    return input ? identifyRasterValue(input.source, longitude, latitude) : undefined;
  }

  /** Releases numeric sources and texture scopes of bitmap layers Kepler no longer holds. */
  pruneRemovedLayers(): number {
    const liveIds = new Set(this.getBitmapLayers().map(layer => layer.id));
    let released = 0;
    for (const layerId of [...this.numericInputs.keys(), ...this.scopes.keys()]) {
      if (liveIds.has(layerId)) continue;
      this.releaseLayer(layerId);
      released++;
    }
    return released;
  }

  releaseLayer(layerId: string): void {
    this.scopes.get(layerId)?.destroy();
    this.scopes.delete(layerId);
    this.numericInputs.delete(layerId);
  }

  destroy(): void {
    for (const layerId of [...this.numericInputs.keys()]) this.releaseLayer(layerId);
  }
}
