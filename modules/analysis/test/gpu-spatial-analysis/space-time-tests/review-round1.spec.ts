// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUMantelTestOperation} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test-operation';
import {
  getGPUSpaceTimeParameterValues,
  decodeGPUSpaceTimeInferenceMetadata,
  getGPUSpaceTimeInferenceMetadata
} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-parameters';
import {runPoisonedOperation} from '../point-process/output-oracle-harness';
import {DRAW_LOCAL_EXPECTED} from './review-round1-fixture';

it('B2 preserves draw-local Pearson coefficients and all 19 usable draws', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const stopping of [false, true]) {
    const result = await runPoisonedOperation(
      device,
      new GPUMantelTestOperation({
        rowCount: 4,
        pairs: {
          offsets: 'offsets',
          neighbors: 'neighbors',
          weights: 'weights',
          distances: 'distances'
        },
        times: 'times',
        parameters: 'parameters',
        maximumPermutations: 19,
        ...(stopping ? {stopAfterExceedances: 19} : {}),
        output: {
          statistics: 'statistics',
          summary: 'summary',
          permutationCount: 'attempted',
          usablePermutationCount: 'usable'
        }
      }),
      {
        offsets: {format: 'uint32', values: Uint32Array.of(0, 3, 3, 3, 3)},
        neighbors: {format: 'uint32', values: Uint32Array.of(1, 2, 3)},
        weights: {format: 'float32', values: Float32Array.of(1, 1, 1)},
        distances: {format: 'float32', values: Float32Array.of(1, 2, 3)},
        times: {format: 'float32', values: Float32Array.of(65536, 65537, 0, 65538)},
        parameters: {
          format: 'uint32',
          values: getGPUSpaceTimeParameterValues({seed: 3, permutationCount: 19})
        }
      },
      ['statistics', 'summary', 'attempted', 'usable']
    );
    expect(result['statistics']).toHaveLength(20);
    for (const [slot, expected] of DRAW_LOCAL_EXPECTED.entries()) {
      expect(Number.isFinite(result['statistics'][slot])).toBe(true);
      expect(Math.abs(result['statistics'][slot] - expected)).toBeLessThan(2e-6);
    }
    expect(Array.from(result['attempted'])).toEqual([19]);
    expect(Array.from(result['usable'])).toEqual([19]);
  }
});

it('EARLIER Pearson denominator remains finite at nanosecond time scales', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const duration = Math.fround(36e12);
  const result = await runPoisonedOperation(
    device,
    new GPUMantelTestOperation({
      rowCount: 3,
      pairs: {
        offsets: 'offsets',
        neighbors: 'neighbors',
        weights: 'weights',
        distances: 'distances'
      },
      times: 'times',
      parameters: 'parameters',
      maximumPermutations: 19,
      output: {statistics: 'statistics', summary: 'summary'}
    }),
    {
      offsets: {format: 'uint32', values: Uint32Array.of(0, 2, 3, 3)},
      neighbors: {format: 'uint32', values: Uint32Array.of(1, 2, 2)},
      weights: {format: 'float32', values: Float32Array.of(1, 1, 1)},
      distances: {format: 'float32', values: Float32Array.of(1e6, 2e6, 1e6)},
      times: {format: 'float32', values: Float32Array.of(0, duration, 2 * duration)},
      parameters: {
        format: 'uint32',
        values: getGPUSpaceTimeParameterValues({seed: 0, permutationCount: 19})
      }
    },
    ['statistics', 'summary']
  );
  // Centered vectors are positive scalar multiples, hence exactly r=1 in real arithmetic.
  expect(Number.isFinite(result['statistics'][0])).toBe(true);
  expect(result['statistics'][0]).toBeCloseTo(1, 6);
});

it('B3 B4 publishes an attempted prefix and a readback-only inference denominator', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const stopping of [false, true]) {
    const result = await runPoisonedOperation(
      device,
      new GPUMantelTestOperation({
        rowCount: 3,
        pairs: {
          offsets: 'offsets',
          neighbors: 'neighbors',
          weights: 'weights',
          distances: 'distances'
        },
        times: 'times',
        parameters: 'parameters',
        maximumPermutations: 19,
        ...(stopping ? {stopAfterExceedances: 2} : {}),
        output: {
          statistics: 'statistics',
          summary: 'summary',
          permutationCount: 'attempted',
          usablePermutationCount: 'usable'
        }
      }),
      {
        offsets: {format: 'uint32', values: Uint32Array.of(0, 0, 0, 2)},
        neighbors: {format: 'uint32', values: Uint32Array.of(0, 1)},
        weights: {format: 'float32', values: Float32Array.of(1, 1)},
        distances: {format: 'float32', values: Float32Array.of(2, 4)},
        times: {format: 'float32', values: Float32Array.of(3, 6, 0)},
        parameters: {
          format: 'uint32',
          values: getGPUSpaceTimeParameterValues({seed: 27, permutationCount: 19})
        }
      },
      ['statistics', 'summary', 'attempted', 'usable']
    );
    expect(Array.from(result['attempted'])).toEqual([stopping ? 6 : 19]);
    expect(Array.from(result['usable'])).toEqual([stopping ? 4 : 11]);
    expect(Array.from(result['statistics'].slice(1, 7))).toEqual([-1, NaN, 1, -1, NaN, 1]);
    const metadata = getGPUSpaceTimeInferenceMetadata(
      'mantel',
      {seed: 27, permutationCount: 19},
      19,
      stopping ? 2 : undefined,
      'pseudoPTwoSided'
    );
    expect(metadata.permutationCount).toBeUndefined();
    const decoded = decodeGPUSpaceTimeInferenceMetadata(
      metadata,
      result['attempted'][0],
      result['usable'][0]
    );
    expect(decoded.inference.permutationCount).toBe(stopping ? 4 : 11);
  }
});
