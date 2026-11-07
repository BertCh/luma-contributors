// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  MAXIMUM_KEPLER_DORLING_SOURCE_ROWS,
  parseKeplerDorlingSettings,
  type KeplerDorlingSettings
} from '@luma-contributors/kepler';

/** Finite EPSG:3857 world strip, in projected meters. */
export const CARTOGRAM_MERCATOR_WORLD_LIMIT_METERS = Math.PI * 6378137;

/** Display admission uses the computed circle extents, never the declared source extent. */
export function getCartogramBoundsUnavailableReason(
  bounds: readonly [number, number, number, number] | undefined
): string | undefined {
  if (!bounds) return 'The cartogram has no eligible circles to display.';
  if (bounds.some(value => !Number.isFinite(value)))
    return 'Cartogram circle bounds are nonfinite; results remain available in the table.';
  if (bounds.some(value => Math.abs(value) > CARTOGRAM_MERCATOR_WORLD_LIMIT_METERS))
    return 'Cartogram circles extend beyond the finite Web Mercator world; results remain available in the table.';
  return undefined;
}

/** Four per-row extent channels plus conservative scratch for four scalar reductions. */
export function getCartogramBoundsWorkingBytes(rowCount: number): number {
  return rowCount * 16 + Math.max(1, Math.ceil(rowCount / 256)) * 64 + 256;
}

/**
 * Pair evaluations (rows squared per pass) one run may request. Measured on an Apple WebGPU device
 * (16,384 rows x 64 passes in about 1.4 s, 1.2e10 pair evaluations per second), so this ceiling is
 * about 30 s of GPU time: above it a single submission risks the device watchdog.
 */
export const MAXIMUM_CARTOGRAM_PAIR_EVALUATIONS = 4e11;

export function getCartogramPairEvaluations(rowCount: number, passes: number): number {
  return rowCount * rowCount * passes;
}

export function normalizeCartogramSettings(
  settings: KeplerDorlingSettings,
  rowCount: number
): KeplerDorlingSettings {
  const normalized = parseKeplerDorlingSettings(settings);
  if (
    !Number.isSafeInteger(rowCount) ||
    rowCount < 1 ||
    rowCount > MAXIMUM_KEPLER_DORLING_SOURCE_ROWS
  )
    throw new Error(`Cartograms support 1 to ${MAXIMUM_KEPLER_DORLING_SOURCE_ROWS} source rows`);
  if (
    getCartogramPairEvaluations(rowCount, normalized.maximumIterations) >
    MAXIMUM_CARTOGRAM_PAIR_EVALUATIONS
  )
    // Too much all-pairs work for one GPU submission; lower the pass count or the row count.
    throw new Error(
      `Cartogram work exceeds ${MAXIMUM_CARTOGRAM_PAIR_EVALUATIONS} pair evaluations; reduce passes for ${rowCount} rows`
    );
  return normalized;
}

/** A conservative spacing estimate at the declared source extent, not an accuracy guarantee. */
export function getCartogramCoordinateResolution(settings: KeplerDorlingSettings): number {
  const maximumMagnitude = Math.max(...settings.bounds.map(Math.abs));
  return maximumMagnitude > 0
    ? 2 ** Math.max(-149, Math.floor(Math.log2(maximumMagnitude)) - 23)
    : 0;
}
