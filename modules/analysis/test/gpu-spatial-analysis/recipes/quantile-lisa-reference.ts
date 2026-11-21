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
  const percentile = (quantile: number): number => {
    if (!sorted.length) return NaN;
    // Form the integer numerator first so exact ordinal boundaries stay exact.
    const rank = Math.max(
      0,
      Math.min(
        sorted.length - 1,
        (2 * sorted.length * quantile - quantileCount) / (2 * quantileCount)
      )
    );
    const lower = Math.floor(rank);
    const upper = Math.ceil(rank);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (rank - lower);
  };
  const lower = selectedQuantile === 1 ? -Infinity : percentile(selectedQuantile - 1);
  const upper = selectedQuantile === quantileCount ? Infinity : percentile(selectedQuantile);
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

/** Intentional libgeoda parity fixtures must keep the legacy directed tail explicitly. */
export const QUANTILE_LISA_LEGACY_INFERENCE = {alternative: 'geoda-legacy' as const};
/** Four equal values on a cycle: all six ordered conditional neighbor samples tie at 2. */
export const QUANTILE_LISA_EQUAL_CYCLE = {
  values: [4, 4, 4, 4],
  conditionalStatistics: [2, 2, 2, 2, 2, 2],
  cases: [
    {draws: 9, greater: [1, 1, 1, 1], twoSided: [1, 1, 1, 1], legacy: [0.1, 0.1, 0.1, 0.1]},
    {draws: 99, greater: [1, 1, 1, 1], twoSided: [1, 1, 1, 1], legacy: [0.01, 0.01, 0.01, 0.01]}
  ]
} as const;
