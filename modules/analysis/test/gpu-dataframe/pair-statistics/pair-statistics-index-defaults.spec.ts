// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCrossVariogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-cross-variogram';
import {
  GPUMarkedPointStatistics,
  type GPUMarkedPointStatisticsProps
} from '../../../src/dataframe-analysis/pair-statistics/gpu-marked-point-statistics';
import {AnalysisRig} from '../../gpu-spatial-analysis/catchment-accessibility/rig';
import {requireTestDevice} from '../../utils/require-test-device';

const VALID_ROWS = 20000;
const WINDOW_AREA = 80000;

/** Disjoint two-point groups at distance one; consecutive groups are at least three apart. */
function createSparsePairs() {
  const rows = VALID_ROWS + 4;
  const positions = new Float32Array(rows * 2);
  const values = new Float32Array(rows);
  const mask = new Uint32Array(rows).fill(1);
  for (let row = 0; row < VALID_ROWS; row++) {
    positions[2 * row] = 4 * Math.floor(row / 2) + (row % 2);
    positions[2 * row + 1] = 1;
    values[row] = 2 * (row % 2);
  }
  // The explicit marked-statistics window excludes both finite outliers. Cross variogram derives
  // bounds from complete cases, so it includes them. Both exclude the NaN and masked duplicate.
  positions.set([NaN, 1, -1, 1, 40001, 1, 0, 1], 2 * VALID_ROWS);
  values.fill(100, VALID_ROWS);
  mask[rows - 1] = 0;
  return {positions, values, mask};
}

async function runMarked(
  device: Device,
  positions: Float32Array,
  parameters: Float32Array,
  options: {
    radiusCount: number;
    mask?: Uint32Array;
    gridSize?: readonly [number, number];
    searchStructure?: GPUMarkedPointStatisticsProps['searchStructure'];
  }
) {
  const rig = new AnalysisRig(device);
  const statistics = rig.output('float32', options.radiusCount * 4);
  const pairCounts = rig.output('uint32', options.radiusCount);
  const status = rig.output('uint32', options.radiusCount);
  try {
    rig.run(
      new GPUMarkedPointStatistics({
        positions: rig.input(positions, 'float32x2', positions.length / 2),
        mask: options.mask ? rig.input(options.mask, 'uint32') : undefined,
        parameters: rig.input(parameters, 'float32'),
        radiusCount: options.radiusCount,
        gridSize: options.gridSize,
        searchStructure: options.searchStructure,
        edgeCorrection: 'none',
        output: {statistics, pairCounts, status}
      })
    );
    return {
      statistics: await rig.readFloat(statistics),
      pairCounts: await rig.readUint(pairCounts),
      status: await rig.readUint(status)
    };
  } finally {
    rig.destroy();
  }
}

function expectStatistics(actual: number[], expected: number[]) {
  actual.forEach((value, index) => {
    if (Number.isNaN(expected[index])) {
      expect(value, `statistic ${index}`).toBeNaN();
    } else {
      expect(Math.abs(value - expected[index]), `statistic ${index}`).toBeLessThanOrEqual(
        1e-6 + Math.abs(expected[index]) * 1e-4
      );
    }
  });
}

it('marked statistics default to an index above 16,384 rows and match disjoint-pair counts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {positions, mask} = createSparsePairs();
  const result = await runMarked(device, positions, Float32Array.from([0, 0, 40000, 2, 1, 0]), {
    radiusCount: 2,
    mask
  });
  // First radius .5 has no pairs. Radius 1 includes one directed neighbor per valid row.
  const estimate = WINDOW_AREA / (VALID_ROWS - 1);
  expect(result.pairCounts).toEqual([0, VALID_ROWS]);
  expect(result.status).toEqual([0, 0]);
  expectStatistics(result.statistics, [
    0,
    0,
    NaN,
    NaN,
    estimate,
    Math.sqrt(estimate / Math.PI),
    estimate / (Math.PI * 0.75),
    1
  ]);
}, 120_000);

it('cross variogram default accepts more than 16,384 rows and matches complete-case pair semivariance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {positions, values, mask} = createSparsePairs();
  const rig = new AnalysisRig(device);
  const semivariances = rig.output('float32', 2);
  const pairCounts = rig.output('uint32', 2);
  const status = rig.output('uint32', 2);
  try {
    rig.run(
      new GPUCrossVariogram({
        positions: rig.input(positions, 'float32x2', positions.length / 2),
        values: rig.input(values, 'float32'),
        mask: rig.input(mask, 'uint32'),
        parameters: rig.input(Float32Array.from([1]), 'float32'),
        lagCount: 2,
        output: {semivariances, pairCounts, status}
      })
    );
    // The 10,000 disjoint pairs each contribute 2. The finite outlier at -1 also pairs with
    // the point at 0, contributing (100 - 0)^2 / 2 = 5,000. The outlier at 40,001 is isolated.
    const expectedPairCount = VALID_ROWS / 2 + 1;
    const expectedSemivariance = (2 * (VALID_ROWS / 2) + 5000) / expectedPairCount;
    expect(await rig.readUint(pairCounts)).toEqual([0, expectedPairCount]);
    expect(await rig.readUint(status)).toEqual([1, 0]);
    const actual = await rig.readFloat(semivariances);
    expect(actual[0]).toBeNaN();
    expect(actual[1]).toBeCloseTo(expectedSemivariance, 6);
  } finally {
    rig.destroy();
  }
}, 120_000);

it('marked grid statistics retain finite pairs when a finite radius square overflows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = Float32Array.from([1, 1, 2, 1, 3, 1, 1, 3]);
  for (const options of [
    {},
    {gridSize: [4, 4] as const},
    {searchStructure: 'brute-force' as const}
  ]) {
    const result = await runMarked(device, positions, Float32Array.from([0, 0, 4, 4, 1e20, 0]), {
      ...options,
      radiusCount: 1
    });
    // All n (n - 1) ordered pairs fit: K = A, L = sqrt(A/pi), unit marks correlate as one.
    // The annular area overflows to infinity, so its finite quotient is zero as in the reference.
    expect(result.pairCounts).toEqual([12]);
    expect(result.status).toEqual([0]);
    expectStatistics(result.statistics, [16, Math.sqrt(16 / Math.PI), 0, 1]);
  }
});

it('marked default and dense reference both report an empty selection without reading padding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const searchStructure of ['grid', 'brute-force'] as const) {
    const result = await runMarked(
      device,
      Float32Array.from([0, 0, 0, 0, 1, 1, NaN, NaN]),
      Float32Array.from([0, 0, 2, 2, 2, 0]),
      {radiusCount: 2, mask: new Uint32Array(4), searchStructure}
    );
    expect(result.pairCounts).toEqual([0, 0]);
    expect(result.status).toEqual([1, 1]);
    expect(result.statistics.every(Number.isNaN)).toBe(true);
  }
});
