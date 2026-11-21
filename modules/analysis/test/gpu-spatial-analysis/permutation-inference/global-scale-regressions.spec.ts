// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPU_GLOBAL_PERMUTATION_SUMMARY as SUMMARY} from '../../../src/gpu-spatial-analysis/permutation-inference/gpu-global-permutation-test';
import {createGlobalPermutationHarness, createLocalPermutationHarness} from './permutation-harness';
import {createWeights} from './permutation-oracle';

const path = createWeights([
  [[1, 1]],
  [
    [0, 1],
    [2, 1]
  ],
  [
    [1, 1],
    [3, 1]
  ],
  [[2, 1]]
]);

it('UST-01 global permutation General G preserves sparse ordered pair mass', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const magnitude of [10000000, 2 ** 24]) {
    const harness = createGlobalPermutationHarness(device, {
      scene: {weights: path, values: Float32Array.of(magnitude, 1, 0, 0)},
      statistic: 'getisOrdG',
      maximumPermutations: 9,
      parameters: {seed: 7, permutationCount: 9}
    });
    try {
      const result = await harness.run();
      const observed = result.results[SUMMARY.observed];
      expect(Number.isFinite(observed)).toBe(true);
      expect(observed).toBeCloseTo(1, 5);
    } finally {
      harness.destroy();
    }
  }
});

// Independent NumPy matrix formulas and Python integer Philox/swap-or-not:
// /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/permutation-inference/global-scale-reference.oracle.py
const references = {
  bivariateMoran: {
    observed: 0.29130434782608694,
    exceedances: 3,
    simulated: [
      -0.08260869565217391, -0.12608695652173912, 0.04782608695652174, 0.6391304347826087,
      0.7434782608695653, -0.7260869565217392, -0.3173913043478261, 0.8043478260869565,
      -0.6130434782608696
    ]
  },
  leesL: {
    observed: 0.21739130434782608,
    exceedances: 9,
    simulated: [
      0.4782608695652174, 0.4608695652173913, 0.4608695652173913, 1.0869565217391304,
      0.7739130434782608, 1, 0.6695652173913044, 1, 0.7739130434782608
    ]
  }
};
it('UST-05 bivariate global statistics preserve every seed-7 draw under a change of units', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const statistic of ['bivariateMoran', 'leesL'] as const)
    for (const scale of [1, 2 ** 30]) {
      const values = Float32Array.from([1, 2, 4, 8], value => value * scale);
      const harness = createGlobalPermutationHarness(device, {
        scene: {weights: path, values, secondValues: values, rowStandardized: true},
        statistic,
        alternative: 'greater',
        maximumPermutations: 9,
        parameters: {seed: 7, permutationCount: 9}
      });
      try {
        const result = await harness.run();
        const expected = references[statistic];
        expect(result.results).toHaveLength(SUMMARY.length);
        expect(result.referenceDistribution).toHaveLength(expected.simulated.length);
        expect(Number.isFinite(result.results[SUMMARY.observed])).toBe(true);
        expect(result.results[SUMMARY.observed]).toBeCloseTo(expected.observed, 5);
        for (const [index, value] of expected.simulated.entries()) {
          expect(Number.isFinite(result.referenceDistribution[index])).toBe(true);
          expect(result.referenceDistribution[index]).toBeCloseTo(value, 5);
        }
        expect(result.results[SUMMARY.exceedances]).toBe(expected.exceedances);
      } finally {
        harness.destroy();
      }
    }
});

it('DST-01 local bivariate Moran reads secondValues in both orientations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // (n-1) (x-mean(x)) .* W(y-mean(y)) / sqrt(SSx SSy), n=4, row-normalized path.
  for (const [second, expected] of [
    [
      [1, 2, 4, 8],
      [0.5021739130434782, 0.22826086956521738, 0.03260869565217391, 0.1108695652173913]
    ],
    [
      [8, 4, 2, 1],
      [-0.07173913043478261, -0.22826086956521738, -0.03260869565217391, -0.7760869565217391]
    ]
  ]) {
    const harness = createLocalPermutationHarness(device, {
      scene: {
        weights: path,
        values: Float32Array.of(1, 2, 4, 8),
        secondValues: Float32Array.from(second),
        rowStandardized: true
      },
      statistic: 'localBivariateMoran',
      maximumPermutations: 9,
      parameters: {seed: 7, permutationCount: 9}
    });
    try {
      const result = await harness.run();
      expect(result.observed).toHaveLength(expected.length);
      for (const [row, value] of expected.entries()) {
        expect(Number.isFinite(result.observed[row])).toBe(true);
        expect(result.observed[row]).toBeCloseTo(value, 5);
      }
    } finally {
      harness.destroy();
    }
  }
});

it('B1 exact scaling preserves closely separated representable values on both permutation routes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Centered ranks are proportional to [-3,-1,1,3]; row-normalized path gives I=.4, L=.2.
  const values = Float32Array.from([0, 1, 2, 3], index => 45 + index * 2 ** -18);
  for (const statistic of ['bivariateMoran', 'leesL'] as const)
    for (const permutationBatchSize of [undefined, 3]) {
      const harness = createGlobalPermutationHarness(device, {
        scene: {weights: path, values, secondValues: values, rowStandardized: true},
        statistic,
        maximumPermutations: 9,
        permutationBatchSize,
        parameters: {seed: 7, permutationCount: 9}
      });
      try {
        const result = await harness.run();
        expect(result.results).toHaveLength(SUMMARY.length);
        const observed = result.results[SUMMARY.observed];
        expect(Number.isFinite(observed)).toBe(true);
        expect(observed).toBeCloseTo(statistic === 'bivariateMoran' ? 0.4 : 0.2, 6);
      } finally {
        harness.destroy();
      }
    }
});

it('S1 General G rejects negative original values even when scaling erases their magnitude', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createGlobalPermutationHarness(device, {
    scene: {weights: path, values: Float32Array.of(2 ** 80, 2 ** 80, -(2 ** -80), 0)},
    statistic: 'getisOrdG',
    maximumPermutations: 9,
    parameters: {seed: 7, permutationCount: 9}
  });
  try {
    const result = await harness.run();
    expect(result.invalidCount).toBe(1);
    expect(result.results[SUMMARY.observed]).toBeNaN();
    expect(result.referenceDistribution).toHaveLength(9);
    expect(result.referenceDistribution.every(Number.isNaN)).toBe(true);
    const clean = await harness.run(undefined, {values: Float32Array.of(2 ** 80, 2 ** 80, 0, 0)});
    expect(clean.invalidCount).toBe(0);
    expect(Number.isFinite(clean.results[SUMMARY.observed])).toBe(true);
  } finally {
    harness.destroy();
  }
});
