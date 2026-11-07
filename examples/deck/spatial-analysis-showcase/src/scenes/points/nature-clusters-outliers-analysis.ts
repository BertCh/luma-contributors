// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Bounded GPU sample used by the local-sparsity story. */
export const NATURE_OUTLIER_SAMPLE_LIMIT = 1200;

/** Selects stable, evenly spaced source rows when a GPU example needs a bounded frame. */
export function selectEvenlySpacedRows(rows: Uint32Array, maximumRows: number): Uint32Array {
  if (rows.length <= maximumRows) return rows.slice();
  const selected = new Uint32Array(maximumRows);
  for (let index = 0; index < maximumRows; index++) {
    selected[index] = rows[Math.floor((index * (rows.length - 1)) / (maximumRows - 1))];
  }
  return selected;
}

/** Counts LOF scores into a fixed domain; scores at or above the upper edge share the last bin. */
export function makeLocalOutlierFactorHistogram(
  scores: Float32Array,
  binCount = 24,
  maximumScore = 4
): number[] {
  const bins = new Array<number>(binCount).fill(0);
  for (const score of scores) {
    if (!Number.isFinite(score)) continue;
    const bin = Math.min(binCount - 1, Math.max(0, Math.floor((score / maximumScore) * binCount)));
    bins[bin]++;
  }
  return bins;
}
