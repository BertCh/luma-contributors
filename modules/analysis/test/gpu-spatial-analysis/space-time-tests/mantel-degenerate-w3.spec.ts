// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUMantelTestOperation} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test-operation';
import {getGPUSpaceTimeParameterValues} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-parameters';
import {runPoisonedOperation} from '../point-process/output-oracle-harness';

it('NUM-03 publishes undefined ordinary and band observations and excludes undefined null draws', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const band of [false, true]) {
    const operation = new GPUMantelTestOperation({
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
      output: {
        statistics: 'statistics',
        summary: 'summary',
        permutationCount: 'used',
        usablePermutationCount: 'usable'
      }
    });
    const result = await runPoisonedOperation(
      device,
      operation,
      {
        offsets: {format: 'uint32', values: Uint32Array.of(0, 2, band ? 2 : 3, band ? 2 : 3)},
        neighbors: {format: 'uint32', values: Uint32Array.of(1, 2, 2)},
        weights: {format: 'float32', values: Float32Array.of(1, 1, 1)},
        distances: {format: 'float32', values: Float32Array.of(1, 1, 1)},
        times: {format: 'float32', values: Float32Array.of(0, 1, 3)},
        parameters: {
          format: 'uint32',
          values: getGPUSpaceTimeParameterValues({seed: 0, permutationCount: 19})
        }
      },
      ['statistics', 'summary', 'used', 'usable']
    );
    expect(Array.from(result['statistics'])).toEqual(Array(20).fill(NaN));
    expect(Array.from(result['summary'])).toEqual([NaN, band ? 2 : 3, 0, 0, ...Array(8).fill(NaN)]);
    expect(Array.from(result['used'])).toEqual([0]);
    expect(Array.from(result['usable'])).toEqual([0]);
  }
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
      output: {
        statistics: 'statistics',
        summary: 'summary',
        permutationCount: 'used',
        usablePermutationCount: 'usable'
      }
    }),
    {
      offsets: {format: 'uint32', values: Uint32Array.of(0, 2, 2, 2)},
      neighbors: {format: 'uint32', values: Uint32Array.of(1, 2)},
      weights: {format: 'float32', values: Float32Array.of(1, 1)},
      distances: {format: 'float32', values: Float32Array.of(1, 2)},
      times: {format: 'float32', values: Float32Array.of(0, 1, 2)},
      parameters: {
        format: 'uint32',
        values: getGPUSpaceTimeParameterValues({seed: 0, permutationCount: 19})
      }
    },
    ['statistics', 'summary', 'used', 'usable']
  );
  // Independent Python integer shuffle + hand-worked two-pair Pearson signs, see Node companion.
  expect(Array.from(result['statistics'])).toEqual([
    1,
    -1,
    NaN,
    -1,
    1,
    1,
    -1,
    NaN,
    1,
    NaN,
    -1,
    -1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    -1
  ]);
  expect(Array.from(result['used'])).toEqual([19]);
  expect(Array.from(result['usable'])).toEqual([16]);
  expect(Number.isFinite(result['summary'][8])).toBe(true);
  expect(result['summary'][8]).toBe(Math.fround(7 / 17));
});

it('TWN-11 suppresses invalidated terminal offsets under producer loss', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const terminal of [1, 2]) {
    const result = await runPoisonedOperation(
      device,
      new GPUMantelTestOperation({
        rowCount: 2,
        pairs: {
          offsets: 'offsets',
          neighbors: 'neighbors',
          weights: 'weights',
          distances: 'distances',
          status: {overflow: 'loss'}
        },
        times: 'times',
        parameters: 'parameters',
        maximumPermutations: 3,
        output: {
          statistics: 'statistics',
          summary: 'summary',
          permutationCount: 'used',
          usablePermutationCount: 'usable'
        }
      }),
      {
        offsets: {format: 'uint32', values: Uint32Array.of(0, 1, terminal)},
        neighbors: {format: 'uint32', values: Uint32Array.of(1)},
        weights: {format: 'float32', values: Float32Array.of(1)},
        distances: {format: 'float32', values: Float32Array.of(1)},
        times: {format: 'float32', values: Float32Array.of(0, 1)},
        parameters: {
          format: 'uint32',
          values: getGPUSpaceTimeParameterValues({seed: 0, permutationCount: 3})
        },
        loss: {format: 'uint32', values: Uint32Array.of(1)}
      },
      ['statistics', 'summary', 'used', 'usable']
    );
    expect(Array.from(result['statistics'])).toEqual(Array(4).fill(NaN));
    expect(Array.from(result['summary'])).toEqual(Array(12).fill(NaN));
    expect(Array.from(result['used'])).toEqual([0]);
  }
});

it('NUM-03 handles zero usable nulls and sequential stopping with degenerate draws', async () => {
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
        ...(stopping ? {stopAfterExceedances: 1} : {}),
        output: {
          statistics: 'statistics',
          summary: 'summary',
          permutationCount: 'used',
          usablePermutationCount: 'usable'
        }
      }),
      {
        offsets: {format: 'uint32', values: Uint32Array.of(0, 2, 2, 2)},
        neighbors: {format: 'uint32', values: Uint32Array.of(1, 2)},
        weights: {format: 'float32', values: Float32Array.of(1, 1)},
        distances: {format: 'float32', values: Float32Array.of(1, 2)},
        times: {format: 'float32', values: Float32Array.of(0, 1, 2)},
        parameters: {
          format: 'uint32',
          values: getGPUSpaceTimeParameterValues({
            seed: stopping ? 0 : 5,
            permutationCount: stopping ? 19 : 1
          })
        }
      },
      ['statistics', 'summary', 'used', 'usable']
    );
    // Independent seed-5 order [1,2,0] gives undefined r. Seed 0 starts [-1,NaN,-1,1].
    expect(Array.from(result['statistics'])).toEqual(
      stopping ? [1, -1, NaN, -1, 1, ...Array(15).fill(0)] : [1, NaN, ...Array(18).fill(0)]
    );
    expect(Array.from(result['used'])).toEqual([stopping ? 4 : 1]);
    expect(Array.from(result['usable'])).toEqual([stopping ? 3 : 0]);
    if (stopping) {
      expect(Number.isFinite(result['summary'][8])).toBe(true);
      expect(result['summary'][8]).toBe(Math.fround(1 / 3));
    } else {
      expect(result['summary'][0]).toBe(1);
      expect(Array.from(result['summary'].slice(4)).every(Number.isNaN)).toBe(true);
    }
  }
});
