// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPU_GLOBAL_PERMUTATION_SUMMARY,
  GPU_LOCAL_PERMUTATION_NOT_TESTED
} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {createGlobalPermutationHarness, createLocalPermutationHarness} from './permutation-harness';
import {createKnnWeights, createSeededRandom} from './permutation-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

const R = GPU_GLOBAL_PERMUTATION_SUMMARY;
const ROWS = 60;
const PARAMETERS = {seed: 7, permutationCount: 99, significanceLevel: 0.05};

function createScene(negativeRows: number[]) {
  const random = createSeededRandom(3);
  const positions = Float64Array.from({length: ROWS * 2}, () => random() * 100);
  const weights = createKnnWeights(positions, 5, false);
  const values = Float32Array.from({length: ROWS}, () => 1 + random() * 4);
  for (const row of negativeRows) {
    values[row] = -values[row];
  }
  return {weights, values};
}

it('GPULocalPermutationTest rejects negative data for local G and G* instead of ranking the wrong tail', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const statistic of ['localG', 'localGStar'] as const) {
    // Valid non-negative data is tested as before.
    const valid = createLocalPermutationHarness(device, {
      scene: createScene([]),
      statistic,
      parameters: PARAMETERS,
      alternative: 'greater',
      maximumPermutations: 99
    });
    const validResult = await valid.run();
    valid.destroy();
    expect(validResult.invalidCount, statistic).toBe(0);
    expect(validResult.overflow, statistic).toBe(0);
    expect(
      validResult.exceedances.filter(count => count !== GPU_LOCAL_PERMUTATION_NOT_TESTED).length,
      statistic
    ).toBe(ROWS);

    // Negative data with the 'greater' alternative: invalid, never a (lower-tail) p-value.
    const harness = createLocalPermutationHarness(device, {
      scene: createScene([2, 11, 40]),
      statistic,
      parameters: PARAMETERS,
      alternative: 'greater',
      multipleTesting: 'benjamini-hochberg',
      maximumPermutations: 99
    });
    const result = await harness.run();
    harness.destroy();
    expect(result.invalidCount, statistic).toBe(3);
    expect(result.overflow, statistic).toBe(1);
    for (let row = 0; row < ROWS; row++) {
      expect(result.exceedances[row], `${statistic} row ${row}`).toBe(
        GPU_LOCAL_PERMUTATION_NOT_TESTED
      );
      expect(result.pseudoPValues[row]).toBeNaN();
      expect(result.adjustedPValues[row]).toBeNaN();
      expect(result.observed[row]).toBeNaN();
      expect(result.significant[row]).toBe(0);
    }
  }
});

it('GPULocalPermutationTest reports negative G data with sequential stopping and leaves other statistics alone', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const stopped = createLocalPermutationHarness(device, {
    scene: createScene([5]),
    statistic: 'localG',
    parameters: PARAMETERS,
    alternative: 'greater',
    stopAfterExceedances: 5,
    maximumPermutations: 99
  });
  const stoppedResult = await stopped.run();
  stopped.destroy();
  expect(stoppedResult.invalidCount).toBe(1);
  expect(stoppedResult.exceedances.every(count => count === GPU_LOCAL_PERMUTATION_NOT_TESTED)).toBe(
    true
  );
  expect(stoppedResult.pseudoPValues.every(Number.isNaN)).toBe(true);

  // Moran is defined for negative data: tested, no invalid rows.
  const moran = createLocalPermutationHarness(device, {
    scene: createScene([5]),
    statistic: 'localMoran',
    parameters: PARAMETERS,
    maximumPermutations: 99
  });
  const moranResult = await moran.run();
  moran.destroy();
  expect(moranResult.invalidCount).toBe(0);
  expect(moranResult.overflow).toBe(0);
  expect(moranResult.exceedances.some(count => count !== GPU_LOCAL_PERMUTATION_NOT_TESTED)).toBe(
    true
  );
});

it('GPUGlobalPermutationTest rejects negative data for Getis-Ord G with NaN results and a count', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const valid = createGlobalPermutationHarness(device, {
    scene: createScene([]),
    statistic: 'getisOrdG',
    parameters: PARAMETERS,
    alternative: 'greater',
    maximumPermutations: 99
  });
  const validResult = await valid.run();
  valid.destroy();
  expect(validResult.invalidCount).toBe(0);
  expect(validResult.results[R.observed]).toBeGreaterThan(0);
  expect(validResult.results[R.pseudoPValue]).toBeGreaterThan(0);

  const harness = createGlobalPermutationHarness(device, {
    scene: createScene([2, 11, 40]),
    statistic: 'getisOrdG',
    parameters: PARAMETERS,
    alternative: 'greater',
    maximumPermutations: 99
  });
  const result = await harness.run();
  harness.destroy();
  expect(result.invalidCount).toBe(3);
  for (const field of [
    'observed',
    'pseudoPValue',
    'exceedances',
    'simulatedMean',
    'simulatedStandardDeviation',
    'zSimulated',
    'pZSimulated',
    'minimum',
    'maximum'
  ] as const) {
    expect(result.results[R[field]], field).toBeNaN();
  }
  expect(result.referenceDistribution.slice(0, 99).every(Number.isNaN)).toBe(true);
  expect(result.histogram.reduce((total, count) => total + count, 0)).toBe(0);
  expect(result.results[R.analysisCount]).toBe(ROWS);

  // Moran keeps working on the same negative data.
  const moran = createGlobalPermutationHarness(device, {
    scene: createScene([2, 11, 40]),
    statistic: 'moran',
    parameters: PARAMETERS,
    maximumPermutations: 99
  });
  const moranResult = await moran.run();
  moran.destroy();
  expect(moranResult.invalidCount).toBe(0);
  expect(moranResult.results[R.pseudoPValue]).toBeGreaterThan(0);
});
