// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUGlobalPermutationTest} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {createGlobalPermutationHarness} from './permutation-harness';
import {createKnnWeights, createSeededRandom} from './permutation-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

function createScene(count: number, seed: number) {
  const random = createSeededRandom(seed);
  const positions = Float64Array.from({length: count * 2}, () => random() * 100);
  const values = Float32Array.from(
    {length: count},
    (_, index) =>
      2 + Math.sin(positions[index * 2] / 15) + Math.cos(positions[index * 2 + 1] / 20) + random()
  );
  return {positions, values};
}

it('GPUGlobalPermutationTest permutationBatchSize equals the inline shuffle bit for bit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 2500 rows span two 2048-row blocks, so the materialize and pair kernels both use several blocks.
  const {positions, values} = createScene(2500, 8);
  const secondValues = Float32Array.from(
    values,
    (value, index) => value + positions[index * 2] / 50
  );
  const weights = createKnnWeights(positions, 6, true);
  for (const statistic of ['moran', 'geary', 'getisOrdG', 'bivariateMoran', 'leesL'] as const) {
    const options = {
      scene: {weights, values, secondValues, rowStandardized: true},
      statistic,
      parameters: {seed: 11, permutationCount: 98},
      maximumPermutations: 127
    };
    const plain = createGlobalPermutationHarness(device, options);
    const expected = await plain.run();
    // A batch size that does not divide 128 permutations (the last batch overruns the maximum).
    for (const permutationBatchSize of [1, 25, 128]) {
      const batched = createGlobalPermutationHarness(device, {...options, permutationBatchSize});
      const actual = await batched.run();
      expect(actual.resultBits, `${statistic} batch ${permutationBatchSize} summary`).toEqual(
        expected.resultBits
      );
      expect(actual.referenceDistribution, `${statistic} batch ${permutationBatchSize}`).toEqual(
        expected.referenceDistribution
      );
      expect(actual.histogram).toEqual(expected.histogram);
      batched.destroy();
    }
    plain.destroy();
  }
});

it('GPUGlobalPermutationTest permutationBatchSize validates its limits', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {positions, values} = createScene(100, 2);
  const weights = createKnnWeights(positions, 4, true);
  const create = (extra: {permutationBatchSize?: number; stopAfterExceedances?: number}) => () =>
    createGlobalPermutationHarness(device, {
      scene: {weights, values, rowStandardized: true},
      statistic: 'moran',
      parameters: {seed: 1, permutationCount: 9},
      maximumPermutations: 15,
      ...extra
    }).destroy();
  expect(create({permutationBatchSize: 0})).toThrow();
  expect(create({permutationBatchSize: 17})).toThrow();
  expect(create({permutationBatchSize: 4, stopAfterExceedances: 3})).not.toThrow();
  expect(GPUGlobalPermutationTest).toBeDefined();
});
