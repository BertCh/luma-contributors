// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUTrajectoryRecursionsOperation} from '../../../src/gpu-spatial-analysis/trajectory-recursion/cpu-trajectory-recursions-operation';
import {
  GPUTrajectoryRecursionsOperation,
  type TrajectoryRecursionsOperationProps
} from '../../../src/gpu-spatial-analysis/trajectory-recursion/gpu-trajectory-recursions-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scenario of ['ordinary', 'singleton', 'words'] as const) {
  it(`recursion operations compare every output: ${scenario}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: TrajectoryRecursionsOperationProps = {
      positions: 'positions',
      timestamps: 'timestamps',
      trackOffsets: 'trackOffsets',
      radii: 'radii',
      rowCount: 5,
      radiusCount: 3,
      timestampFormat: scenario === 'words' ? 'uint32x2' : 'float32',
      output: {
        revisits: 'revisits',
        residenceTimes: 'residenceTimes',
        firstPassageTimes: 'firstPassageTimes',
        firstPassageCensored: 'firstPassageCensored',
        status: {candidateOverflow: 'candidateOverflow'}
      }
    };
    const inputs = {
      positions: {
        format: 'float32x2' as const,
        values: new Float32Array([0, 0, 2, 0, 0, 0, 8, 0, 8, 0])
      },
      timestamps: {
        format: props.timestampFormat!,
        values:
          scenario === 'words'
            ? new Uint32Array([0xfffffffe, 100, 0, 101, 2, 101, 98, 101, 101, 101])
            : new Float32Array([0, 2, 4, 100, 103])
      },
      trackOffsets: {
        format: 'uint32' as const,
        values: new Uint32Array(scenario === 'singleton' ? [0, 1, 2, 3, 4, 5] : [0, 3, 3, 5])
      },
      radii: {format: 'float32' as const, values: new Float32Array([1, 2, 0])}
    };
    const operation = new GPUTrajectoryRecursionsOperation(props);
    const outputs = Object.values(operation.outputs).map(output => output.value);
    const gpu = await runOperationGraph(device, {inputs, operations: [operation], outputs});
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUTrajectoryRecursionsOperation(props)],
      outputs
    });
    for (const output of outputs) {
      expect(gpu[output].length).toBe(cpu[output].length);
      for (let index = 0; index < cpu[output].length; index++) {
        // Float ports: absolute 1e-5; integer ports and status: exact.
        if (cpu[output] instanceof Float32Array)
          expect(Math.abs(gpu[output][index] - cpu[output][index])).toBeLessThanOrEqual(1e-5);
        else expect(gpu[output][index]).toBe(cpu[output][index]);
      }
    }
  });
}

it('recursion review regression compares every output: clamped-offset', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: TrajectoryRecursionsOperationProps = {
    positions: 'positions',
    timestamps: 'timestamps',
    trackOffsets: 'trackOffsets',
    radii: 'radii',
    rowCount: 2,
    radiusCount: 1,
    output: {
      revisits: 'revisits',
      residenceTimes: 'residenceTimes',
      firstPassageTimes: 'firstPassageTimes',
      firstPassageCensored: 'firstPassageCensored',
      status: {candidateOverflow: 'candidateOverflow'}
    }
  };
  const inputs = {
    positions: {
      format: 'float32x2' as const,
      values: new Float32Array([0, 0, 0, 0])
    },
    timestamps: {
      format: 'float32' as const,
      values: new Float32Array([0, 1])
    },
    trackOffsets: {format: 'uint32' as const, values: new Uint32Array([0, 3])},
    radii: {format: 'float32' as const, values: new Float32Array([1])}
  };
  const expected: Record<string, readonly number[]> = {
    revisits: [0, 0],
    residenceTimes: [1, 1],
    firstPassageTimes: [1, 0],
    firstPassageCensored: [1, 1],
    candidateOverflow: [0]
  };
  const outputs = Object.keys(expected);
  const gpu = await runOperationGraph(device, {
    inputs,
    operations: [new GPUTrajectoryRecursionsOperation(props)],
    outputs
  });
  const cpu = await runOperationGraph(device, {
    inputs,
    operations: [new CPUTrajectoryRecursionsOperation(props)],
    outputs
  });
  // Every expected time here is an exact integer, so use zero tolerance for every output.
  for (const output of outputs) {
    expect(Array.from(cpu[output])).toEqual(expected[output]);
    expect(Array.from(gpu[output])).toEqual(Array.from(cpu[output]));
  }
});
