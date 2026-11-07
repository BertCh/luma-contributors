// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Decoded vector-tile working-set admission.
 *
 * Kepler's tile layer keeps decoded GeoJSON tiles in deck.gl's TileLayer cache with no byte
 * budget. This host policy sits beside it: every decoded tile that the host wants to treat as
 * resident (for filtering, styling or analysis) is admitted against a byte budget, least recently
 * used tiles are evicted first, tiles currently visible are never evicted, and each admitted tile
 * holds a memory reservation (`KeplerDatasetRegistry.reserveMemory`, kind `cpu`) so decoded tiles
 * share the same accounting as other host resources. A tile that cannot be admitted is reported as
 * rejected; callers (see vector-tile-analysis-scope.ts) must then treat the population as
 * incomplete.
 */

import {
  VectorTileFeatureRegistry,
  type VectorTileFeatureIdOptions,
  type VectorTileFeatureLike
} from './vector-tile-feature-ids';

/** Structural subset of KeplerMemoryReservation. */
export type VectorTileReservation = Readonly<{
  resize: (byteLength: number) => void;
  release: () => void;
}>;

/** Structural subset of KeplerDatasetRegistry.reserveMemory; throws when over capacity. */
export type VectorTileReserveMemory = (
  byteLength: number,
  memoryKind: 'cpu',
  category?: 'source' | 'output' | 'scratch' | 'readback'
) => VectorTileReservation;

export type VectorTileWorkingSetOptions = Readonly<{
  budgetBytes: number;
  maxTileCount?: number;
  reserveMemory?: VectorTileReserveMemory;
  featureIdOptions?: VectorTileFeatureIdOptions;
  onEvictTile?: (tileKey: string) => void;
}>;

export type VectorTileAdmission = Readonly<{
  admitted: boolean;
  evictedTileKeys: readonly string[];
  rejectedReason?: 'larger-than-budget' | 'visible-tiles-fill-budget' | 'memory-reservation-failed';
}>;

export type VectorTileWorkingSetStatus = Readonly<{
  budgetBytes: number;
  usedBytes: number;
  tileCount: number;
  evictionCount: number;
  rejectedTileKeys: readonly string[];
}>;

type ResidentTile = {byteLength: number; reservation?: VectorTileReservation};

/** Rough decoded-size estimate: 16 bytes per vertex plus two bytes per serialized property char. */
export function estimateDecodedTileByteLength(
  features: readonly (VectorTileFeatureLike & {coordinateCount?: number})[]
): number {
  let byteLength = 0;
  for (const feature of features) {
    byteLength += 64 + (feature.coordinateCount ?? 0) * 16;
    byteLength += feature.properties ? JSON.stringify(feature.properties).length * 2 : 0;
  }
  return byteLength;
}

export class VectorTileWorkingSet {
  /** Insertion order is recency order: first key is least recently used. */
  private readonly tiles = new Map<string, ResidentTile>();
  private readonly rejectedTileKeys = new Set<string>();
  private visibleTileKeys = new Set<string>();
  private usedBytes = 0;
  private evictionCount = 0;
  readonly featureRegistry?: VectorTileFeatureRegistry;

  constructor(private readonly options: VectorTileWorkingSetOptions) {
    if (options.featureIdOptions)
      this.featureRegistry = new VectorTileFeatureRegistry(options.featureIdOptions);
  }

  /** Tiles in the current viewport; they are protected from eviction. */
  setVisibleTileKeys(tileKeys: Iterable<string>): void {
    this.visibleTileKeys = new Set(tileKeys);
  }

  has(tileKey: string): boolean {
    return this.tiles.has(tileKey);
  }

  /** Marks a tile most recently used. */
  touchTile(tileKey: string): void {
    const tile = this.tiles.get(tileKey);
    if (!tile) return;
    this.tiles.delete(tileKey);
    this.tiles.set(tileKey, tile);
  }

  /** Admits (or replaces) a decoded tile, evicting least recently used non-visible tiles. */
  admitTile(
    tileKey: string,
    byteLength: number,
    features?: readonly VectorTileFeatureLike[]
  ): VectorTileAdmission {
    const evictedTileKeys: string[] = [];
    this.removeTile(tileKey, false);
    if (byteLength > this.options.budgetBytes)
      return this.reject(tileKey, 'larger-than-budget', evictedTileKeys);

    const makeRoom = (): boolean => {
      for (const candidateKey of this.tiles.keys()) {
        if (this.visibleTileKeys.has(candidateKey)) continue;
        this.removeTile(candidateKey, true);
        evictedTileKeys.push(candidateKey);
        return true;
      }
      return false;
    };
    const fits = () =>
      this.usedBytes + byteLength <= this.options.budgetBytes &&
      this.tiles.size < (this.options.maxTileCount ?? Infinity);
    while (!fits())
      if (!makeRoom()) return this.reject(tileKey, 'visible-tiles-fill-budget', evictedTileKeys);

    let reservation: VectorTileReservation | undefined;
    while (this.options.reserveMemory) {
      try {
        reservation = this.options.reserveMemory(byteLength, 'cpu', 'source');
        break;
      } catch {
        // Registry is over capacity (possibly due to other datasets): free our own tiles first.
        if (!makeRoom()) return this.reject(tileKey, 'memory-reservation-failed', evictedTileKeys);
      }
    }
    this.tiles.set(tileKey, {byteLength, reservation});
    this.usedBytes += byteLength;
    this.rejectedTileKeys.delete(tileKey);
    if (features) this.featureRegistry?.registerTile(tileKey, features);
    return {admitted: true, evictedTileKeys};
  }

  /** Explicit removal (deck.gl dropped the tile, or dataset teardown). */
  removeTile(tileKey: string, isEviction = false): void {
    const tile = this.tiles.get(tileKey);
    if (!tile) return;
    this.tiles.delete(tileKey);
    this.usedBytes -= tile.byteLength;
    tile.reservation?.release();
    this.featureRegistry?.evictTile(tileKey);
    if (isEviction) {
      this.evictionCount++;
      this.options.onEvictTile?.(tileKey);
    }
  }

  getResidentTileKeys(): readonly string[] {
    return [...this.tiles.keys()];
  }

  getStatus(): VectorTileWorkingSetStatus {
    return {
      budgetBytes: this.options.budgetBytes,
      usedBytes: this.usedBytes,
      tileCount: this.tiles.size,
      evictionCount: this.evictionCount,
      rejectedTileKeys: [...this.rejectedTileKeys]
    };
  }

  destroy(): void {
    for (const tileKey of [...this.tiles.keys()]) this.removeTile(tileKey, false);
    this.rejectedTileKeys.clear();
  }

  private reject(
    tileKey: string,
    rejectedReason: NonNullable<VectorTileAdmission['rejectedReason']>,
    evictedTileKeys: readonly string[]
  ): VectorTileAdmission {
    this.rejectedTileKeys.add(tileKey);
    return {admitted: false, evictedTileKeys, rejectedReason};
  }
}
