// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUTrajectoryThinningOperation} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-thinning-operation';
import {
  GPUTrajectoryThinningOperation,
  type TrajectoryThinningOperationProps
} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning-operation';
import {getGPUTrajectoryThinningParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('both thinning forms honor a one-row forward search budget', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUTrajectoryThinningOperation, CPUTrajectoryThinningOperation]) {
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: new Float32Array(8)},
        timestamps: {format: 'float32', values: Float32Array.of(0, 1, 2, 3)},
        offsets: {format: 'uint32', values: Uint32Array.of(0, 4)},
        parameters: {format: 'float32', values: Float32Array.of(10, 0, 0, 0)}
      },
      operations: [
        new Operation({
          positions: 'positions',
          timestamps: 'timestamps',
          trackOffsets: 'offsets',
          parameters: 'parameters',
          rowCount: 4,
          trackCount: 1,
          capacity: 4,
          mode: 'minimum-time',
          maximumSearchRows: 1,
          output: {
            rowIds: 'rows',
            trackOffsets: 'outputOffsets',
            status: {count: 'count', overflow: 'overflow', candidateOverflow: 'candidate'}
          }
        })
      ],
      outputs: ['rows', 'count', 'candidate', 'overflow']
    });
    expect(Array.from(result['rows']).slice(0, 3)).toEqual([0, 2, 3]);
    expect(Array.from(result['count'])).toEqual([3]);
    expect(Array.from(result['candidate'])).toEqual([1]);
    expect(Array.from(result['overflow'])).toEqual([0]);
  }
});

it('thinning compares spherical dateline and quarter-circle tracks exactly', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: TrajectoryThinningOperationProps = {
    mode: 'minimum-distance',
    positions: 'positions',
    timestamps: 'timestamps',
    trackOffsets: 'trackOffsets',
    parameters: 'parameters',
    rowCount: 7,
    trackCount: 2,
    capacity: 6,
    spatialContext: {
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      sphereRadius: 1
    },
    output: {
      rowIds: 'rowIds',
      trackOffsets: 'outputTrackOffsets',
      validity: 'validity',
      positions: 'copiedPositions',
      timestamps: 'copiedTimes',
      status: {
        count: 'count',
        overflow: 'overflow',
        requiredCount: 'requiredCount',
        candidateOverflow: 'candidateOverflow'
      }
    }
  };
  const inputs = {
    positions: {
      format: 'float32x2' as const,
      values: new Float32Array([0, 0, 30, 0, 90, 0, 180, 0, -180, 0, 180, 0, -179, 0])
    },
    timestamps: {format: 'float32' as const, values: new Float32Array([0, 1, 2, 3, 0, 1, 2])},
    trackOffsets: {format: 'uint32' as const, values: new Uint32Array([0, 4, 7])},
    parameters: {
      format: 'float32' as const,
      values: getGPUTrajectoryThinningParameterValues({tolerance: 1})
    }
  };
  const gpuOperation = new GPUTrajectoryThinningOperation(props);
  const outputs = Object.values(gpuOperation.outputs).map(output => output.value);
  const gpu = await runOperationGraph(device, {inputs, operations: [gpuOperation], outputs});
  const cpu = await runOperationGraph(device, {
    inputs,
    operations: [new CPUTrajectoryThinningOperation(props)],
    outputs
  });
  // No derived float output. All fields are integer decisions or copies: zero tolerance.
  for (const output of outputs)
    for (let row = 0; row < cpu[output].length; row++)
      expect(gpu[output][row] === cpu[output][row]).toBe(true);
  expect(Array.from(cpu['rowIds'])).toEqual([0, 2, 3, 4, 6, 0xffffffff]);
});

for (const mode of ['minimum-distance', 'minimum-time'] as const) {
  for (const overflow of [false, true]) {
    it(`thinning ${mode} compares all ports exactly (${overflow ? 'overflow' : 'ordinary'})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const props: TrajectoryThinningOperationProps = {
        mode,
        positions: 'positions',
        timestamps: 'timestamps',
        trackOffsets: 'trackOffsets',
        parameters: 'parameters',
        rowCount: 10,
        trackCount: 3,
        capacity: overflow ? 2 : 8,
        output: {
          rowIds: 'rowIds',
          trackOffsets: 'outputTrackOffsets',
          validity: 'validity',
          positions: 'copiedPositions',
          timestamps: 'copiedTimes',
          status: {
            count: 'count',
            requiredCount: 'requiredCount',
            overflow: 'overflow',
            candidateOverflow: 'candidateOverflow'
          }
        }
      };
      const inputs = {
        positions: {
          format: 'float32x2' as const,
          values: new Float32Array([0, 0, 1, 0, 0, 0, 3, 4, 3, 4, 6, 8, 10, 0, 11, 0, 12, 0, 13, 0])
        },
        timestamps: {
          format: 'float32' as const,
          values: new Float32Array([0, 1, 2, 3, 7, 8, 0, 2, 5, 6])
        },
        trackOffsets: {format: 'uint32' as const, values: new Uint32Array([0, 6, 6, 10])},
        parameters: {
          format: 'float32' as const,
          values: getGPUTrajectoryThinningParameterValues({
            tolerance: mode === 'minimum-time' ? 3 : 5
          })
        }
      };
      const gpuOperation = new GPUTrajectoryThinningOperation(props);
      const outputs = Object.values(gpuOperation.outputs).map(output => output.value);
      const gpu = await runOperationGraph(device, {inputs, operations: [gpuOperation], outputs});
      const cpu = await runOperationGraph(device, {
        inputs,
        operations: [new CPUTrajectoryThinningOperation(props)],
        outputs
      });
      // Integer decisions and copies only. Tolerance zero; fresh GPU capacity rows start at zero.
      for (const output of outputs) {
        if (output === 'copiedPositions' || output === 'copiedTimes') {
          for (let component = 0; component < cpu[output].length; component++)
            expect(gpu[output][component] === cpu[output][component]).toBe(true);
        } else expect(Array.from(gpu[output])).toEqual(Array.from(cpu[output]));
      }
      expect(cpu['requiredCount'][0]).toBe(mode === 'minimum-time' ? 7 : 5);
      expect(cpu['overflow'][0]).toBe(Number(overflow));
    });
  }
}
