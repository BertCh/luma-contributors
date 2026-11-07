// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Cross-tile stable feature identity for decoded vector tiles.
 *
 * A vector tile source clips every feature to each tile it touches, so one source feature becomes
 * several fragments with different geometry. Identity therefore must not depend on tile
 * coordinates or on clipped geometry. The key is built, in order of preference, from:
 *
 * 1. `source-id`:       the MVT feature id (source-wide unique when the encoder sets it),
 * 2. `unique-property`: a property named by the layer (Kepler `uniqueIdField`) or inferred from
 *                       Kepler's `UUID_CANDIDATES` list (same candidates as vector-tile-layer.js),
 * 3. `property-hash`:   a deterministic hash of the sorted properties. Fragments of one feature
 *                       carry identical properties, so they merge; two distinct features with
 *                       identical properties also merge (reported through `collisionRisk`).
 *
 * The id is a pure function of its inputs, so pan, evict and reload always reproduce it. Analysis
 * that needs exact per-feature counts must require stability `source-id` or `unique-property`.
 */

/** Same list Kepler uses to infer a unique id property (kepler.gl layers vector-tile-layer.js). */
export const VECTOR_TILE_UUID_CANDIDATES: readonly string[] = [
  'ufid',
  'UFID',
  'id',
  'ID',
  'fid',
  'FID',
  'objectid',
  'OBJECTID',
  'gid',
  'GID',
  'feature_id',
  'FEATURE_ID',
  '_id'
];

export type VectorTileFeatureIdStability = 'source-id' | 'unique-property' | 'property-hash';

export type VectorTileFeatureLike = Readonly<{
  id?: string | number | null;
  properties?: Readonly<Record<string, unknown>> | null;
  /** Only the geometry type is read; it keeps a point and a polygon with equal properties apart. */
  geometry?: Readonly<{type?: string}> | null;
}>;

export type VectorTileFeatureIdentity = Readonly<{
  /** Stable across tiles, pan, evict and reload. */
  featureId: string;
  stability: VectorTileFeatureIdStability;
  /** True when two distinct source features may share the id (property-hash only). */
  collisionRisk: boolean;
}>;

export type VectorTileFeatureIdOptions = Readonly<{
  /** MVT source layer; part of the key because feature ids are only unique per source layer. */
  sourceLayer: string;
  /** Kepler `uniqueIdField`; inferred from VECTOR_TILE_UUID_CANDIDATES when omitted. */
  uniqueIdProperty?: string | null;
}>;

/** 53-bit string hash (cyrb53), rendered as 14 hex digits. Deterministic and dependency free. */
export function hashVectorTileString(text: string): string {
  let firstHash = 0xdeadbeef;
  let secondHash = 0x41c6ce57;
  for (let index = 0; index < text.length; index++) {
    const code = text.charCodeAt(index);
    firstHash = Math.imul(firstHash ^ code, 2654435761);
    secondHash = Math.imul(secondHash ^ code, 1597334677);
  }
  firstHash =
    Math.imul(firstHash ^ (firstHash >>> 16), 2246822507) ^
    Math.imul(secondHash ^ (secondHash >>> 13), 3266489909);
  secondHash =
    Math.imul(secondHash ^ (secondHash >>> 16), 2246822507) ^
    Math.imul(firstHash ^ (firstHash >>> 13), 3266489909);
  return (4294967296 * (2097151 & secondHash) + (firstHash >>> 0)).toString(16).padStart(14, '0');
}

function serializeProperties(properties: Readonly<Record<string, unknown>>): string {
  return Object.keys(properties)
    .sort()
    .map(key => `${key}=${JSON.stringify(properties[key])}`)
    .join('|');
}

function isUsableId(value: unknown): value is string | number {
  return (typeof value === 'string' && value !== '') || typeof value === 'number';
}

/** Resolves the identity of one decoded feature; never reads geometry or tile coordinates. */
export function resolveVectorTileFeatureIdentity(
  feature: VectorTileFeatureLike,
  options: VectorTileFeatureIdOptions
): VectorTileFeatureIdentity {
  const properties = feature.properties ?? {};
  if (isUsableId(feature.id)) {
    return {
      featureId: `${options.sourceLayer}:i:${feature.id}`,
      stability: 'source-id',
      collisionRisk: false
    };
  }
  const uniqueProperty =
    options.uniqueIdProperty ??
    VECTOR_TILE_UUID_CANDIDATES.find(candidate => isUsableId(properties[candidate]));
  if (uniqueProperty && isUsableId(properties[uniqueProperty])) {
    return {
      featureId: `${options.sourceLayer}:p:${uniqueProperty}:${properties[uniqueProperty]}`,
      stability: 'unique-property',
      collisionRisk: false
    };
  }
  return {
    featureId: `${options.sourceLayer}:h:${hashVectorTileString(
      feature.geometry?.type
        ? `${serializeProperties(properties)}|@${feature.geometry.type}`
        : serializeProperties(properties)
    )}`,
    stability: 'property-hash',
    collisionRisk: true
  };
}

export type VectorTileFeatureRegistrySummary = Readonly<{
  uniqueFeatureCount: number;
  fragmentCount: number;
  /** Features present in more than one resident tile (split across tile borders). */
  splitFeatureCount: number;
  tileCount: number;
  stabilityCounts: Readonly<Record<VectorTileFeatureIdStability, number>>;
  /** True when any resident feature relies on property-hash identity. */
  hasCollisionRisk: boolean;
  /**
   * Fragments whose property-hash id was already taken by another feature of the same tile. A tile
   * holds at most one fragment per source feature, so these are certain collisions between distinct
   * features. They received an occurrence suffix (`#n`) to stay separate; the suffix is only stable
   * while the tile content and order are unchanged, so exact per-feature analysis must still refuse them.
   */
  ambiguousFragmentCount: number;
}>;

/**
 * Tracks which resident tiles hold fragments of which feature. Evicting a tile removes only that
 * tile's fragments, so a feature stays counted once while any fragment remains and reappears under
 * the same id when the tile reloads.
 */
export class VectorTileFeatureRegistry {
  private readonly tileFeatures = new Map<string, readonly string[]>();
  private readonly featureTiles = new Map<string, Set<string>>();
  private readonly featureStability = new Map<string, VectorTileFeatureIdStability>();
  private readonly tileAmbiguousCounts = new Map<string, number>();

  constructor(private readonly options: VectorTileFeatureIdOptions) {}

  /** Registers (or replaces) a tile's fragments; returns one stable id per input feature. */
  registerTile(tileKey: string, features: readonly VectorTileFeatureLike[]): readonly string[] {
    this.evictTile(tileKey);
    const featureIds: string[] = [];
    const occurrenceCounts = new Map<string, number>();
    let ambiguousCount = 0;
    for (const feature of features) {
      const identity = resolveVectorTileFeatureIdentity(feature, this.options);
      let featureId = identity.featureId;
      if (identity.stability === 'property-hash') {
        const occurrence = occurrenceCounts.get(featureId) ?? 0;
        occurrenceCounts.set(featureId, occurrence + 1);
        if (occurrence > 0) {
          featureId = `${featureId}#${occurrence}`;
          ambiguousCount++;
        }
      }
      featureIds.push(featureId);
      this.featureStability.set(featureId, identity.stability);
      let tiles = this.featureTiles.get(featureId);
      if (!tiles) {
        tiles = new Set();
        this.featureTiles.set(featureId, tiles);
      }
      tiles.add(tileKey);
    }
    if (ambiguousCount) this.tileAmbiguousCounts.set(tileKey, ambiguousCount);
    this.tileFeatures.set(tileKey, featureIds);
    return featureIds;
  }

  evictTile(tileKey: string): void {
    const featureIds = this.tileFeatures.get(tileKey);
    if (!featureIds) return;
    this.tileFeatures.delete(tileKey);
    this.tileAmbiguousCounts.delete(tileKey);
    for (const featureId of new Set(featureIds)) {
      const tiles = this.featureTiles.get(featureId);
      tiles?.delete(tileKey);
      if (tiles && tiles.size === 0) {
        this.featureTiles.delete(featureId);
        this.featureStability.delete(featureId);
      }
    }
  }

  getFeatureTileKeys(featureId: string): readonly string[] {
    return [...(this.featureTiles.get(featureId) ?? [])];
  }

  getTileFeatureIds(tileKey: string): readonly string[] {
    return this.tileFeatures.get(tileKey) ?? [];
  }

  summarize(): VectorTileFeatureRegistrySummary {
    const stabilityCounts = {'source-id': 0, 'unique-property': 0, 'property-hash': 0};
    let fragmentCount = 0;
    let splitFeatureCount = 0;
    let ambiguousFragmentCount = 0;
    for (const count of this.tileAmbiguousCounts.values()) ambiguousFragmentCount += count;
    for (const featureIds of this.tileFeatures.values()) fragmentCount += featureIds.length;
    for (const [featureId, tiles] of this.featureTiles) {
      stabilityCounts[this.featureStability.get(featureId) ?? 'property-hash']++;
      if (tiles.size > 1) splitFeatureCount++;
    }
    return {
      uniqueFeatureCount: this.featureTiles.size,
      fragmentCount,
      splitFeatureCount,
      tileCount: this.tileFeatures.size,
      stabilityCounts,
      hasCollisionRisk: stabilityCounts['property-hash'] > 0,
      ambiguousFragmentCount
    };
  }
}
