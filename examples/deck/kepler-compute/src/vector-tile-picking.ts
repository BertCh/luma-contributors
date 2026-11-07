// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Maps a pick on a resident vector-tile draw layer back to the decoded tile feature.
 *
 * Resident draw layers pick in primitive index spaces (point, path, polygon, polygon-outline). The
 * geometry binding of each primitive resolves a primitive index to the tile's source row, which
 * indexes the decoded feature list the tile was uploaded from.
 */

import type {DecodedTileFeature} from './vector-tile-decoded-batch';
import {resolveVectorTileFeatureIdentity} from './vector-tile-feature-ids';

export type VectorTilePrimitiveKind = 'points' | 'paths' | 'polygons' | 'polygon-outlines';

export type VectorTilePickInfo = Readonly<{
  layerId: string;
  tileKey: string;
  primitive: VectorTilePrimitiveKind;
  /** Row of the feature within the tile's decoded feature list. */
  sourceRow: number;
  /** Tile-independent id (matches the working set's feature registry). */
  featureId: string;
  properties: Readonly<Record<string, unknown>>;
  feature: DecodedTileFeature;
  coordinate: readonly number[] | null;
}>;

export type ResolveVectorTilePickInput = Readonly<{
  layerId: string;
  tileKey: string;
  primitive: VectorTilePrimitiveKind;
  primitiveIndex: number;
  features: readonly DecodedTileFeature[];
  /** Per-row ids from the feature registry; resolved from the feature when absent. */
  featureIds?: readonly string[];
  resolveSourceRow?: (primitiveIndex: number) => number;
  coordinate?: readonly number[] | null;
  uniqueIdProperty?: string | null;
}>;

/** Returns null for misses, out-of-range primitives and rows without a decoded feature. */
export function resolveVectorTilePick(
  input: ResolveVectorTilePickInput
): VectorTilePickInfo | null {
  if (!Number.isSafeInteger(input.primitiveIndex) || input.primitiveIndex < 0) return null;
  const sourceRow = input.resolveSourceRow
    ? input.resolveSourceRow(input.primitiveIndex)
    : input.primitiveIndex;
  const feature = Number.isSafeInteger(sourceRow) ? input.features[sourceRow] : undefined;
  if (!feature) return null;
  return {
    layerId: input.layerId,
    tileKey: input.tileKey,
    primitive: input.primitive,
    sourceRow,
    featureId:
      input.featureIds?.[sourceRow] ??
      resolveVectorTileFeatureIdentity(feature, {
        sourceLayer: input.layerId,
        uniqueIdProperty: input.uniqueIdProperty
      }).featureId,
    properties: feature.properties ?? {},
    feature,
    coordinate: input.coordinate ?? null
  };
}
