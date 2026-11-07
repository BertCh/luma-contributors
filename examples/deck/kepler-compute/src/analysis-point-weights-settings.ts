// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getKeplerPointWeightsCapacity,
  MAXIMUM_KEPLER_DISTANCE_BAND_ROWS,
  parseKeplerPointWeightsSettings,
  type KeplerPointWeightsSettings
} from '@luma-contributors/kepler';
import {
  getPointCoordinateSettings,
  makePointCoordinateRecipe,
  type PointCoordinateSettings
} from './analysis-point-coordinate-settings';

export type PointWeightTopology =
  | {kind?: 'knn'; k: number}
  | {
      kind: 'distance-band';
      radiusMeters: number;
      neighborCapacity: number;
      /** Permutation scratch slots per row, 64 to 512 (default 64). */
      permutationDegree?: number;
    };

/** Explicit coordinate roles; selection filters a fixed source-wide topology. */
export type PointTopologySettings = PointWeightTopology & PointCoordinateSettings;

export type PointWeightsSettings = PointTopologySettings & {
  scale: number;
  islandPolicy: 'undefined' | 'own-rate';
};

/** Produce the strict serialized topology, excluding operation-specific host settings. */
export function makePointWeightsRecipe(
  settings: PointTopologySettings
): KeplerPointWeightsSettings {
  return parseKeplerPointWeightsSettings({
    ...makePointCoordinateRecipe(settings),
    ...(settings.kind === 'distance-band'
      ? {
          kind: 'distance-band',
          radiusMeters: settings.radiusMeters,
          neighborCapacity: settings.neighborCapacity,
          ...(settings.permutationDegree === undefined
            ? {}
            : {permutationDegree: settings.permutationDegree})
        }
      : {kind: 'knn', k: settings.k})
  });
}

/** Restore topology controls without changing the legacy kNN host shape. */
export function getPointTopologySettings(
  settings: KeplerPointWeightsSettings
): PointTopologySettings {
  return {
    ...getPointCoordinateSettings(settings),
    ...(settings.kind === 'distance-band'
      ? {
          kind: 'distance-band',
          radiusMeters: settings.radiusMeters,
          neighborCapacity: settings.neighborCapacity,
          ...(settings.permutationDegree === undefined
            ? {}
            : {permutationDegree: settings.permutationDegree})
        }
      : {k: settings.k})
  };
}

export function normalizePointTopologySettings(
  settings: PointTopologySettings,
  rowCount: number
): PointTopologySettings {
  if (!Number.isSafeInteger(rowCount) || rowCount < 1 || rowCount > 2 ** 24)
    throw new Error('Point weights require 1 to 16777216 source rows');
  if (
    settings.kind !== 'distance-band' &&
    (!Number.isInteger(settings.k) || settings.k < 1 || settings.k > 32)
  )
    throw new Error('Point weights require 1 to 32 nearest neighbors');
  if (settings.kind === 'distance-band' && rowCount > MAXIMUM_KEPLER_DISTANCE_BAND_ROWS)
    throw new Error(
      `Distance-band weights support at most ${MAXIMUM_KEPLER_DISTANCE_BAND_ROWS} source rows`
    );
  return getPointTopologySettings(makePointWeightsRecipe(settings));
}

export function normalizePointWeightsSettings(
  settings: PointWeightsSettings,
  rowCount: number
): PointWeightsSettings {
  const topology = normalizePointTopologySettings(settings, rowCount);
  if (!Number.isFinite(Math.fround(settings.scale)) || settings.scale < 0)
    throw new Error('Spatial rate scale must be finite and nonnegative');
  if (settings.islandPolicy !== 'undefined' && settings.islandPolicy !== 'own-rate')
    throw new Error('Choose undefined or own-rate island behavior');
  return {...topology, scale: settings.scale, islandPolicy: settings.islandPolicy};
}

/** Coordinate packing, neighbor search and retained CSR-build scratch. */
export function getPointWeightsScratchBytes(
  settings: PointTopologySettings,
  rowCount: number
): number {
  const normalized = normalizePointTopologySettings(settings, rowCount);
  const capacity = getKeplerPointWeightsCapacity(makePointWeightsRecipe(normalized), rowCount);
  return capacity * 24 + rowCount * 128 + 4 * 1024 * 1024;
}
