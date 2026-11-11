// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUProfilePeaksOperation} from '../../../src/gpu-terrain/terrain-features/cpu-profile-peaks-operation';
import {
  GPUProfilePeaksOperation,
  type ProfilePeaksOperationProps
} from '../../../src/gpu-terrain/terrain-features/gpu-profile-peaks-operation';
import {getGPUProfilePeaksParameterValues} from '../../../src/gpu-terrain/terrain-features/gpu-profile-peaks';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const testCase of [
  'ordinary',
  'overflow',
  'gaps',
  'seam',
  'suppression',
  'separate-profiles'
] as const) {
  it(`profile peak operations agree on every defined output: ${testCase}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const values = new Float32Array(
      testCase === 'seam'
        ? [1, 0, 0, 0, 1e-7]
        : testCase === 'suppression'
          ? [0, 0, 5, 0, 0, 5, 0, 0, 5, 0, 0]
          : testCase === 'separate-profiles'
            ? [0, 0, 5, 0, 0, 5, 0, 0]
            : [0, 1, 5, 2, 0, 0, 1, 4, 1, 0]
    );
    const props: ProfilePeaksOperationProps = {
      values: 'values',
      validity: 'validity',
      offsets: 'offsets',
      parameters: 'parameters',
      sampleCount: values.length,
      window: 2,
      minimumSide: 1,
      suppressionRadius: testCase === 'suppression' || testCase === 'separate-profiles' ? 3 : 0,
      wrap: testCase === 'seam',
      output: {
        prominence: 'prominence',
        refinedIndex: 'refinedIndex',
        refinedValue: 'refinedValue',
        peakMask: 'peakMask',
        peaks: {
          ids: 'ids',
          capacity: testCase === 'overflow' || testCase === 'seam' ? 1 : 2,
          status: {count: 'count', requiredCount: 'requiredCount', overflow: 'overflow'}
        },
        status: {converged: 'converged', iterationLimitReached: 'iterationLimitReached'}
      }
    };
    const outputs = [
      'prominence',
      'refinedIndex',
      'refinedValue',
      'peakMask',
      'ids',
      'count',
      'requiredCount',
      'overflow',
      'converged',
      'iterationLimitReached'
    ];
    const inputs = {
      values: {format: 'float32' as const, values},
      validity: {
        format: 'uint32' as const,
        values: new Uint32Array(values.length).fill(testCase === 'gaps' ? 0 : 1)
      },
      offsets: {
        format: 'uint32' as const,
        values: new Uint32Array(testCase === 'separate-profiles' ? [0, 4, 8] : [0, values.length])
      },
      parameters: {
        format: 'float32' as const,
        values: getGPUProfilePeaksParameterValues({minimumProminence: testCase === 'seam' ? 0 : 1})
      }
    };
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUProfilePeaksOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUProfilePeaksOperation(props)],
      outputs
    });
    for (const rows of [gpu, cpu]) {
      if (testCase === 'seam') {
        expect(rows['refinedIndex'][0]).toBe(0);
        expect(rows['refinedIndex'][0]).toBeGreaterThanOrEqual(0);
        expect(rows['refinedIndex'][0]).toBeLessThan(5);
        expect(rows['count'][0]).toBe(1);
        expect(Array.from(rows['ids'])).toEqual([0]);
      } else if (testCase === 'suppression' || testCase === 'separate-profiles') {
        expect(rows['count'][0]).toBe(2);
        expect(Array.from(rows['ids'])).toEqual(testCase === 'suppression' ? [2, 8] : [2, 5]);
      }
    }
    for (const name of outputs) {
      // Compact tail rows are unspecified. Integers are exact; finite floats allow 2e-5 absolute.
      const length = name === 'ids' ? cpu['count'][0] : cpu[name].length;
      for (let index = 0; index < length; index++) {
        const expected = cpu[name][index];
        if (Number.isNaN(expected)) expect(gpu[name][index]).toBeNaN();
        else if (cpu[name] instanceof Float32Array)
          expect(Math.abs(gpu[name][index] - expected)).toBeLessThanOrEqual(2e-5);
        else expect(gpu[name][index]).toBe(expected);
      }
    }
  });
}
