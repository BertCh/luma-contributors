// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp} from '@math.gl/core';

/**
 * Bounds Deck's `TileLayer` tile cache by the vector-tile working set budget.
 *
 * Deck keeps every cached tile's decoded content alive independent of the host's working set, so the
 * default cache (five times the selected tile count) can far exceed the memory budget. Deck's
 * `maxCacheByteSize` reads `tile.content.byteLength`, which a non-binary (GeoJSON array) MVT tile does
 * not have; it would log an error per tile and count zero bytes. The cache is therefore bounded by tile
 * count, derived from the observed mean decoded tile size.
 */

export const VECTOR_TILE_DEFAULT_TILE_BYTES = 256 * 1024;
export const VECTOR_TILE_MINIMUM_CACHE_TILES = 4;
export const VECTOR_TILE_MAXIMUM_CACHE_TILES = 512;

export type VectorTileCacheBudgetInput = Readonly<{
  budgetBytes: number;
  usedBytes: number;
  residentTileCount: number;
  visibleTileCount: number;
  /** A cache limit already set on the Deck layer; the smaller limit wins. */
  existingMaxCacheSize?: number | null;
}>;

/** Largest tile count whose mean-size footprint fits the budget, never below what is on screen. */
export function computeVectorTileCacheSize(input: VectorTileCacheBudgetInput): number {
  const meanTileBytes =
    input.residentTileCount > 0 && input.usedBytes > 0
      ? input.usedBytes / input.residentTileCount
      : VECTOR_TILE_DEFAULT_TILE_BYTES;
  const affordable = Math.floor(input.budgetBytes / Math.max(meanTileBytes, 1));
  const floor = Math.max(VECTOR_TILE_MINIMUM_CACHE_TILES, input.visibleTileCount);
  const bounded = clamp(affordable, floor, VECTOR_TILE_MAXIMUM_CACHE_TILES);
  return typeof input.existingMaxCacheSize === 'number' && input.existingMaxCacheSize > 0
    ? Math.min(bounded, Math.max(input.existingMaxCacheSize, floor))
    : bounded;
}
