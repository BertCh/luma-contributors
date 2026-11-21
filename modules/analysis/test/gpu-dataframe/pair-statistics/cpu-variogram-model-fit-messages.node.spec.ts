// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  CPUVariogramModelFitOperation,
  computeVariogramModelFitOnCPU
} from '../../../src/dataframe-analysis/pair-statistics/cpu-variogram-model-fit-operation';

// The first-computation walkthrough: twenty bins h = (i + 1) * 0.2 with one pair count removed.
const DISTANCES = Array.from({length: 20}, (_, index) => (index + 1) * 0.2);
const SEMIVARIANCES = DISTANCES.map(lag => 0.2 + 1.5 * (1 - Math.exp((-3 * lag) / 2.2)));
const PAIR_COUNTS = DISTANCES.map((_, index) => 10 + index);

it('names the bin array that failed and both lengths', () => {
  expect(() =>
    computeVariogramModelFitOnCPU({
      distances: DISTANCES,
      semivariances: SEMIVARIANCES,
      pairCounts: PAIR_COUNTS.slice(0, 19)
    })
  ).toThrow(
    'variogram-model-fit invalid model or bin topology: pairCounts has 19 bins but distances has 20; every bin array must hold one entry per distance bin'
  );
  expect(() =>
    computeVariogramModelFitOnCPU({
      distances: DISTANCES,
      semivariances: [...SEMIVARIANCES, 1],
      pairCounts: PAIR_COUNTS
    })
  ).toThrow(
    /^variogram-model-fit invalid model or bin topology: semivariances has 21 bins but distances has 20;/
  );
});

it('names the failing bin array under the CPU operation ID', () => {
  const operation = new CPUVariogramModelFitOperation({
    id: 'review-variogram-fit',
    distances: 'distances',
    semivariances: 'semivariances',
    pairCounts: 'pairCounts',
    output: {parameters: 'parameters', diagnostics: 'diagnostics', status: 'status'}
  });
  const makeRows = (values: Float32Array | Uint32Array) => ({
    format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  expect(() =>
    operation.execute({
      inputs: {
        distances: makeRows(Float32Array.from(DISTANCES)),
        semivariances: makeRows(Float32Array.from(SEMIVARIANCES.slice(0, 18))),
        pairCounts: makeRows(Uint32Array.from(PAIR_COUNTS))
      }
    })
  ).toThrow(
    /^review-variogram-fit invalid model or bin topology: semivariances has 18 bins but distances has 20;/
  );
});
