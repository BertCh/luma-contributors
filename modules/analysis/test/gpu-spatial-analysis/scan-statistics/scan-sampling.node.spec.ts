// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {test} from 'vitest';
import {
  getSwapOrNotPermutationIndex,
  getSwapOrNotRounds
} from '../../../src/analysis-core/kernels/permutation-shuffle';
import {PHILOX_PURPOSES} from '../../../src/analysis-core/kernels/philox-random';
import {
  computeScanOracle,
  getOracleBernoulliLogLikelihoodRatio,
  getOracleLogLikelihoodRatio
} from './scan-statistic-oracle';
import {
  SCAN_SAMPLING_BOUNDS,
  createConditionalSamplingScene,
  expectConditionalLaw
} from './scan-sampling-fixtures';

for (const model of [
  'poisson',
  'bernoulli',
  'bernoulli-complement',
  'space-time-permutation'
] as const) {
  test(`conditional serial test reference follows independently enumerated ${model} maximum distribution`, () => {
    expectConditionalLaw(
      model,
      computeScanOracle(createConditionalSamplingScene(model)).maxima.slice(1)
    );
  });
}

for (const model of ['bernoulli', 'bernoulli-complement', 'space-time-permutation'] as const) {
  test(`parallel shuffle sampling follows independently enumerated ${model} distribution`, () => {
    const bernoulli = model !== 'space-time-permutation';
    const population = bernoulli ? [3, 2, 1, 2, 1, 2, 1, 2] : new Array(8).fill(2);
    const size = bernoulli ? 14 : 16;
    const drawCount = model === 'bernoulli-complement' ? 9 : bernoulli ? 5 : 8;
    const maxima = Array.from(
      {length: SCAN_SAMPLING_BOUNDS.conditionalSampleCount},
      (_, replicate) => {
        const rounds = getSwapOrNotRounds(
          [17, 0],
          replicate,
          size,
          PHILOX_PURPOSES.spatialScanStatistic.id
        );
        const selected = new Set(
          Array.from({length: drawCount}, (_, row) =>
            getSwapOrNotPermutationIndex(row, size, rounds)
          )
        );
        let first = 0;
        let maximum = 0;
        for (const count of population) {
          const observed = Array.from({length: count}, (_, row) => first + row).filter(row =>
            selected.has(row)
          ).length;
          maximum = Math.max(
            maximum,
            bernoulli
              ? getOracleBernoulliLogLikelihoodRatio(observed, count, drawCount, 14)
              : getOracleLogLikelihoodRatio(Math.max(observed, 2 - observed), 1, 16, 16)
          );
          first += count;
        }
        return maximum;
      }
    );
    expectConditionalLaw(model, maxima);
  });
}
