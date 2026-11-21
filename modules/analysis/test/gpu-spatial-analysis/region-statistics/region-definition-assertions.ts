// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {GPU_REGION_STATISTICS_SUMMARY_LAYOUT} from '../../../src/gpu-spatial-analysis/region-statistics/region-statistics-types';

/** Compare the independent packed fixture without requiring correctly rounded GPU division. */
export function expectRegionDefinitionSummary(
  actual: readonly number[],
  expected: {readonly words: readonly number[]; readonly mean: number}
): void {
  expect(actual.length).toBe(expected.words.length);
  const meanWord = GPU_REGION_STATISTICS_SUMMARY_LAYOUT.mean;
  for (let word = 0; word < expected.words.length; word++) {
    if (word !== meanWord || expected.mean === 0) {
      // Counts, flags, bins and these small integer sums/extrema are exact.
      // Empty selections must still overwrite the mean with exactly zero.
      expect(actual[word], `summary word ${word}`).toBe(expected.words[word]);
      continue;
    }
    const mean = new Float32Array(Uint32Array.of(actual[word]).buffer)[0];
    expect(Number.isFinite(mean), 'summary mean must be finite').toBe(true);
    // The mathematical oracle remains 21/7 = 3. Metal produced 0x40400001
    // (3 + 2^-22) through float32 division. Allow two adjacent float32 steps
    // for this division only, without relaxing the other packed fields.
    expect(
      Math.abs(actual[word] - expected.words[word]),
      'summary mean float32 steps'
    ).toBeLessThanOrEqual(2);
  }
}
