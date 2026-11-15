// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPU_GLOBAL_PERMUTATION_SUMMARY as SUMMARY} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {createGlobalPermutationHarness} from './permutation-harness';
import {
  computeGlobalPermutationOracle,
  createKnnWeights,
  createSeededRandom
} from './permutation-oracle';

it('materialized stopping preserves every output and the CPU oracle stopping index for every statistic and tail', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createSeededRandom(314);
  // More than one materialization/pair block; mask creates unused compacted positions.
  const positions = Float64Array.from({length: 2051 * 2}, () => random() * 100);
  const values = Float32Array.from({length: 2051}, () => 1 + random() * 7);
  const secondValues = Float32Array.from(values, value => 2 + value * value);
  const mask = Uint32Array.from(values, (_, index) => Number(index % 11 !== 0));
  const weights = createKnnWeights(positions, 4, false);
  for (const statistic of ['moran', 'geary', 'getisOrdG', 'bivariateMoran', 'leesL'] as const) {
    for (const alternative of ['two-sided', 'greater', 'lesser', 'directed'] as const) {
      const options = {
        scene: {weights, values, secondValues, mask},
        statistic,
        alternative,
        parameters: {seed: 29, permutationCount: 31},
        maximumPermutations: 39,
        stopAfterExceedances: 5
      };
      const oracle = computeGlobalPermutationOracle({
        weights,
        values,
        secondValues,
        mask,
        statistic,
        alternative,
        seed: 29,
        permutationCount: 31
      });
      let greater = 0;
      let lesser = 0;
      let used = 31;
      for (let index = 1; index <= 31; index++) {
        greater += Number(oracle.pairSums[index] >= oracle.pairSums[0]);
        lesser += Number(oracle.pairSums[index] <= oracle.pairSums[0]);
        const exceedances =
          alternative === 'greater'
            ? greater
            : alternative === 'lesser'
              ? lesser
              : Math.min(greater, lesser);
        if (exceedances >= 5) {
          used = index;
          break;
        }
      }
      const inline = createGlobalPermutationHarness(device, options);
      const expected = await inline.run();
      expect(expected.results[SUMMARY.permutationCount]).toBe(used);
      for (const permutationBatchSize of [1, 4, 16, 40]) {
        const materialized = createGlobalPermutationHarness(device, {
          ...options,
          permutationBatchSize
        });
        const actual = await materialized.run();
        expect(actual.resultBits, `${statistic}/${alternative}/${permutationBatchSize}`).toEqual(
          expected.resultBits
        );
        expect(actual.referenceDistribution).toEqual(expected.referenceDistribution);
        expect(actual.histogram).toEqual(expected.histogram);
        expect(actual.invalidCount).toBe(expected.invalidCount);
        materialized.destroy();
      }
      inline.destroy();
    }
  }
});

it('materialized stopping handles ties at and inside batch boundaries, full runs, and re-encoding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Independent NumPy closed form: W = ones((5,5)) - eye(5), x = arange(1,6).
  // z @ W @ z / (sum(W) * (z @ z)) * 5 == -0.25 for every relabeling.
  // The shader evaluates (5 / 200) * -10: allow 2.5 ULP of division error at 0.025
  // (ULP = 2^-29), amplified by 10, plus half an ULP of final rounding at 0.25.
  const statisticTolerance = 10 * 2.5 * 2 ** -29 + 0.5 * 2 ** -25;
  const values = Float32Array.from([1, 2, 3, 4, 5]);
  const neighbors = Uint32Array.from({length: 20}, (_, index) => {
    const row = Math.floor(index / 4);
    const neighbor = index % 4;
    return neighbor >= row ? neighbor + 1 : neighbor;
  });
  const weights = {
    offsets: Uint32Array.from([0, 4, 8, 12, 16, 20]),
    neighbors,
    weights: new Float32Array(20).fill(1)
  };
  for (const target of [4, 5, 19, 24]) {
    for (const permutationBatchSize of [1, 4, 7, 25]) {
      const harness = createGlobalPermutationHarness(device, {
        scene: {weights, values},
        statistic: 'moran',
        alternative: 'greater',
        parameters: {seed: 17, permutationCount: 19},
        maximumPermutations: 24,
        stopAfterExceedances: target,
        permutationBatchSize
      });
      for (const permutationCount of [19, 3, 19]) {
        const actual = await harness.run({seed: 17, permutationCount});
        const used = Math.min(target, permutationCount);
        const observed = actual.results[SUMMARY.observed];
        expect(Math.abs(observed + 0.25)).toBeLessThanOrEqual(statisticTolerance);
        expect(actual.results[SUMMARY.permutationCount]).toBe(used);
        expect(actual.results[SUMMARY.exceedances]).toBe(used);
        // The p-value divides equal integers; apply the same WGSL division allowance at 1.
        expect(Math.abs(actual.results[SUMMARY.pseudoPValue] - 1)).toBeLessThanOrEqual(
          2.5 * 2 ** -23
        );
        // Ties must still be exact between identically scaled observed and simulated sums.
        expect(actual.referenceDistribution.slice(0, used)).toEqual(new Array(used).fill(observed));
        expect(actual.referenceDistribution.slice(used).every(Number.isNaN)).toBe(true);
        expect(actual.histogram.reduce((total, count) => total + count, 0)).toBe(used);
      }
      const invalid = await harness.run(new Uint32Array([0, 0, 0, 0]));
      expect(invalid.results[SUMMARY.permutationCount]).toBeNaN();
      expect(invalid.referenceDistribution.every(Number.isNaN)).toBe(true);
      harness.destroy();
    }
  }
});
