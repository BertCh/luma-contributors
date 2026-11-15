// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PhiloxStream} from '../../../src/analysis-core/kernels/permutation-random';
import {PHILOX_PURPOSES} from '../../../src/analysis-core/kernels/philox-random';

// Six complete vectors [x, 2*x+5]; their common standardized squared distance is
// (x_i-x_j)^2 / 117.25. Each of the first five rows has one neighbor; row five is an island.
// Rows six/seven are incomplete, and row six is also masked. All nonzero distances at one
// focal row are distinct, so float32 rounding cannot turn a different candidate into a tie.
export const MULTIVARIATE_GEARY_RUNTIME_VALUES = [0, 1, 3, 7, 15, 31];
export const MULTIVARIATE_GEARY_RUNTIME_PERMUTATIONS = 19;
export const MULTIVARIATE_GEARY_RUNTIME_SEED: readonly [number, number] = [12, 0x80000001];

/** Independent degree-one oracle: integer squared distances and one bounded Philox draw per replicate. */
export function computeMultivariateGearyRuntimeOracle(
  seed: readonly [number, number],
  significance: number,
  mode: 'none' | 'bonferroni' | 'benjamini-hochberg'
) {
  const values = MULTIVARIATE_GEARY_RUNTIME_VALUES;
  const permutations = MULTIVARIATE_GEARY_RUNTIME_PERMUTATIONS;
  const statistic = new Array<number>(8).fill(NaN);
  const pValues = new Array<number>(8).fill(NaN);
  const classes = new Array<number>(8).fill(0xffffffff);
  const directions = new Array<number>(8).fill(0xffffffff);
  const simulations = new Array<number>(8 * permutations).fill(NaN);
  for (let row = 0; row < 5; row++) {
    const neighbor = (row + 1) % 5;
    const observedSquared = (values[row] - values[neighbor]) ** 2;
    const candidates = values.filter((_, candidate) => candidate !== row);
    let lower = 0;
    let upper = 0;
    let totalSquared = 0;
    for (let replicate = 1; replicate <= permutations; replicate++) {
      const stream = new PhiloxStream(
        seed,
        row,
        replicate,
        PHILOX_PURPOSES.localPermutationTest.id
      );
      const candidate = candidates[stream.nextBelow(candidates.length)];
      const squared = (values[row] - candidate) ** 2;
      lower += Number(squared <= observedSquared);
      upper += Number(squared >= observedSquared);
      totalSquared += squared;
      simulations[row * permutations + replicate - 1] = squared / 117.25;
    }
    statistic[row] = observedSquared / 117.25;
    const probability =
      mode === 'bonferroni'
        ? (Math.min(upper, permutations - upper) + 1) / (permutations + 1)
        : Math.min(1, (2 * (Math.min(lower, upper) + 1)) / (permutations + 1));
    pValues[row] = Math.min(1, probability * (mode === 'bonferroni' ? values.length : 1));
    directions[row] = observedSquared < totalSquared / permutations ? 1 : 2;
  }
  if (mode === 'benjamini-hochberg') {
    const order = [0, 1, 2, 3, 4].sort((first, second) => pValues[first] - pValues[second]);
    let previous = 1;
    for (let rank = order.length; rank > 0; rank--) {
      const row = order[rank - 1];
      previous = Math.min(previous, (pValues[row] * order.length) / rank);
      pValues[row] = previous;
    }
  }
  for (let row = 0; row < 5; row++)
    classes[row] = pValues[row] <= Math.fround(significance) ? directions[row] : 0;
  return {statistic, pValues, classes, simulations};
}
