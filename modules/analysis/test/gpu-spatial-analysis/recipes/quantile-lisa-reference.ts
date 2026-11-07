// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Independent sorted type-5 oracle, as Gda::percentile in pinned libgeoda. */
export function computeQuantileLISAReference(
  values: Float32Array,
  quantileCount: number,
  selectedQuantile: number,
  mask?: Uint32Array
) {
  const sorted = Array.from(values)
    .filter((value, row) => Number.isFinite(value) && (!mask || mask[row] !== 0))
    .sort((left, right) => left - right);
  const percentile = (probability: number): number => {
    if (!sorted.length) return NaN;
    const rank = Math.max(0, Math.min(sorted.length - 1, sorted.length * probability - 0.5));
    const lower = Math.floor(rank);
    const upper = Math.ceil(rank);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (rank - lower);
  };
  const lower =
    selectedQuantile === 1 ? -Infinity : percentile((selectedQuantile - 1) / quantileCount);
  const upper =
    selectedQuantile === quantileCount ? Infinity : percentile(selectedQuantile / quantileCount);
  const validMask = Uint32Array.from(values, (value, row) =>
    Number.isFinite(value) && (!mask || mask[row] !== 0) ? 1 : 0
  );
  const membership = Uint32Array.from(values, (value, row) =>
    validMask[row] && value >= lower && value < upper ? 1 : 0
  );
  return {boundaries: [lower, upper], membership, validMask, validCount: sorted.length};
}

/**
 * GeoDaCenter/libgeoda 71238cb58979032048081434ce6e36a53e254280,
 * gda_sa.cpp gda_quantilelisa + GenUtils.cpp QuantileBreaks/Gda::percentile.
 * These small inputs have hand-verifiable exact break values and expose R-7/tie mistakes.
 */
export const QUANTILE_LISA_REFERENCE_CASES = [
  {
    name: 'negative values and type-five quarter break',
    values: [-8, -6, -4, -2, 0, 2, 4, 6],
    quantileCount: 4,
    selectedQuantile: 1,
    boundaries: [-Infinity, -5],
    membership: [1, 1, 0, 0, 0, 0, 0, 0]
  },
  {
    name: 'interior ties go to higher class',
    values: [0, 0, 1, 1, 1, 1, 2, 2],
    quantileCount: 4,
    selectedQuantile: 2,
    boundaries: [0.5, 1],
    membership: [0, 0, 0, 0, 0, 0, 0, 0]
  },
  {
    name: 'tied class includes lower boundary',
    values: [0, 0, 1, 1, 1, 1, 2, 2],
    quantileCount: 4,
    selectedQuantile: 3,
    boundaries: [1, 1.5],
    membership: [0, 0, 1, 1, 1, 1, 0, 0]
  },
  {
    name: 'constant first class is empty',
    values: [5, 5, 5, 5, 5, 5, 5, 5],
    quantileCount: 4,
    selectedQuantile: 1,
    boundaries: [-Infinity, 5],
    membership: [0, 0, 0, 0, 0, 0, 0, 0]
  },
  {
    name: 'constant final class contains every row',
    values: [5, 5, 5, 5, 5, 5, 5, 5],
    quantileCount: 4,
    selectedQuantile: 4,
    boundaries: [5, Infinity],
    membership: [1, 1, 1, 1, 1, 1, 1, 1]
  }
] as const;
