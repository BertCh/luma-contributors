// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {CPUProfilePeaksOperation} from '../../../src/gpu-terrain/terrain-features/cpu-profile-peaks-operation';
import {GPUProfilePeaksOperation} from '../../../src/gpu-terrain/terrain-features/gpu-profile-peaks-operation';
import {getGPUProfilePeaksParameterValues} from '../../../src/gpu-terrain/terrain-features/gpu-profile-peaks';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {expectProfileSingletonStatus, expectSingleProfilePeak} from './terrain-terminal-assertions';

for (const terminal of ['converged', 'iterationLimitReached'] as const) {
  it(`N1 status-only GPU ${terminal} rejects a resolved values count that differs from sampleCount`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const output =
      terminal === 'converged'
        ? {status: {converged: 'result'}}
        : {status: {iterationLimitReached: 'result'}};
    await expect(
      runOperationGraph(device, {
        inputs: {
          values: {format: 'float32', values: Float32Array.of(0, 1, 0, 1, 0)},
          offsets: {format: 'uint32', values: Uint32Array.of(0, 5)},
          parameters: {format: 'float32', values: Float32Array.of(1)}
        },
        operations: [
          new GPUProfilePeaksOperation({
            id: 'profile-count',
            values: 'values',
            offsets: 'offsets',
            parameters: 'parameters',
            sampleCount: 2,
            window: 1,
            minimumSide: 1,
            output
          })
        ],
        outputs: ['result']
      })
    ).rejects.toThrow(/profile-count.*values.*sampleCount.*2.*5/);
  });
}

for (const Operation of [CPUProfilePeaksOperation, GPUProfilePeaksOperation]) {
  for (const terminal of ['converged', 'iterationLimitReached'] as const) {
    for (const maximumSuppressionIterations of [0, 16] as const) {
      it(`TSC-02 ${Operation.name} writes ${terminal} as its only terminal at ${maximumSuppressionIterations} rounds`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const output =
          terminal === 'converged'
            ? {status: {converged: 'result'}}
            : {status: {iterationLimitReached: 'result'}};
        const rows = await runOperationGraph(device, {
          inputs: {
            values: {format: 'float32', values: Float32Array.of(0, 1, 0)},
            offsets: {format: 'uint32', values: Uint32Array.of(0, 3)},
            parameters: {
              format: 'float32',
              values: getGPUProfilePeaksParameterValues({minimumProminence: 1})
            }
          },
          operations: [
            new Operation({
              values: 'values',
              offsets: 'offsets',
              parameters: 'parameters',
              sampleCount: 3,
              window: 1,
              minimumSide: 1,
              maximumSuppressionIterations,
              output
            })
          ],
          outputs: ['result']
        });
        expectProfileSingletonStatus(rows['result'], terminal, maximumSuppressionIterations);
      });
    }
  }

  for (const testCase of ['threshold', 'rounding-tie'] as const) {
    it(`TWN-09 ${Operation.name} uses reported prominence precision: ${testCase}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      // 1 - 2^-26 rounds to float32 1. In the second fixture, rows 1 and 4
      // therefore tie and the earlier row wins at the inclusive radius 3 boundary.
      const values =
        testCase === 'threshold'
          ? Float32Array.of(2 ** -26, 1, 2 ** -26)
          : Float32Array.of(2 ** -26, 1, 2 ** -26, 0, 1, 0);
      const rows = await runOperationGraph(device, {
        inputs: {
          values: {format: 'float32', values},
          offsets: {format: 'uint32', values: Uint32Array.of(0, values.length)},
          parameters: {
            format: 'float32',
            values: getGPUProfilePeaksParameterValues({
              minimumProminence: testCase === 'threshold' ? 1 : 0
            })
          }
        },
        operations: [
          new Operation({
            values: 'values',
            offsets: 'offsets',
            parameters: 'parameters',
            sampleCount: values.length,
            window: 1,
            minimumSide: 1,
            suppressionRadius: 3,
            output: {
              prominence: 'prominence',
              peakMask: 'mask',
              peaks: {ids: 'ids', capacity: 1, status: {count: 'count', overflow: 'overflow'}}
            }
          })
        ],
        outputs: ['prominence', 'mask', 'ids', 'count', 'overflow']
      });
      expectSingleProfilePeak(
        {
          mask: rows['mask'],
          count: rows['count'],
          ids: rows['ids'],
          prominence: rows['prominence']
        },
        testCase === 'threshold' ? [0, 1, 0] : [0, 1, 0, 0, 0, 0]
      );
      expect(Array.from(rows['overflow'])).toEqual([0]);
    });
  }
}
