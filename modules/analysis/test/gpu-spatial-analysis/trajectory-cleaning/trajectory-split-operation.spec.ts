// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUTrajectorySplitOperation} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-split-operation';
import {
  GPUTrajectorySplitOperation,
  type TrajectorySplitOperationProps
} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-split-operation';
import {getGPUTrajectorySplitParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-split';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const scenarios = [
  {mode: 'gap', overflow: false, stayOverflow: 0, stayCandidate: 0},
  {mode: 'gap', overflow: true, stayOverflow: 0, stayCandidate: 0},
  {mode: 'stop', overflow: false, stayOverflow: 0, stayCandidate: 0},
  {mode: 'stop', overflow: false, stayOverflow: 0, stayCandidate: 1},
  {mode: 'stop', overflow: true, stayOverflow: 1, stayCandidate: 0},
  {mode: 'value-change', overflow: false, stayOverflow: 0, stayCandidate: 0},
  {mode: 'value-change', overflow: true, stayOverflow: 0, stayCandidate: 0}
] as const;

it.each(
  scenarios
)('split $mode matches every graph output (capacity loss $overflow, upstream flags $stayOverflow/$stayCandidate)', async ({
  mode,
  overflow,
  stayOverflow,
  stayCandidate
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: TrajectorySplitOperationProps = {
    mode,
    positions: 'positions',
    timestamps: 'timestamps',
    trackOffsets: 'trackOffsets',
    rowCount: 10,
    ...(mode === 'gap'
      ? {parameters: 'parameters'}
      : mode === 'stop'
        ? {
            staypointIds: 'staypointIds',
            staypointStatus: new (class {
              get overflow() {
                return 'stayOverflow';
              }
              get candidateOverflow() {
                return 'stayCandidate';
              }
            })()
          }
        : {values: 'values'}),
    segmentCapacity: overflow ? 2 : 5,
    rowCapacity: overflow ? 3 : 14,
    output: {
      segmentOffsets: 'segmentOffsets',
      segmentTracks: 'segmentTracks',
      rowIds: 'rowIds',
      positions: 'copiedPositions',
      timestamps: 'copiedTimes',
      status: {
        count: 'count',
        requiredCount: 'requiredCount',
        overflow: 'overflow',
        candidateOverflow: 'candidate'
      }
    }
  };
  const inputs = {
    positions: {
      format: 'float32x2' as const,
      values: Float32Array.from({length: 20}, (_, component) => component)
    },
    timestamps: {
      format: 'float32' as const,
      values: new Float32Array([0, 1, 3, 4, 5, 0, 1, 2, 10, 11])
    },
    trackOffsets: {format: 'uint32' as const, values: new Uint32Array([0, 5, 10])},
    ...(mode === 'gap'
      ? {
          parameters: {
            format: 'float32' as const,
            values: getGPUTrajectorySplitParameterValues({maximumGap: 2})
          }
        }
      : mode === 'stop'
        ? {
            stayOverflow: {format: 'uint32' as const, values: Uint32Array.of(stayOverflow)},
            stayCandidate: {
              format: 'uint32' as const,
              values: Uint32Array.of(stayCandidate)
            },
            staypointIds: {
              format: 'uint32' as const,
              values: new Uint32Array([
                0xffffffff, 0xffffffff, 0, 0xffffffff, 0xffffffff, 0xffffffff, 1, 0xffffffff,
                0xffffffff, 0xffffffff
              ])
            }
          }
        : {
            values: {
              format: 'uint32' as const,
              values: new Uint32Array([1, 1, 2, 2, 3, 9, 8, 8, 8, 8])
            }
          })
  };
  const outputs = [
    'segmentOffsets',
    'segmentTracks',
    'rowIds',
    'copiedPositions',
    'copiedTimes',
    'count',
    'requiredCount',
    'overflow',
    'candidate'
  ];
  const gpu = await runOperationGraph(device, {
    inputs,
    operations: [new GPUTrajectorySplitOperation(props)],
    outputs
  });
  const cpu = await runOperationGraph(device, {
    inputs,
    operations: [new CPUTrajectorySplitOperation(props)],
    outputs
  });
  // All arithmetic here is integer partitioning or float32 copying: tolerance is zero.
  // Fresh graph buffers supply zero in GPU capacity rows that the gather never writes.
  for (const output of outputs) {
    if (output === 'copiedPositions' || output === 'copiedTimes') {
      for (let component = 0; component < cpu[output].length; component++)
        expect(gpu[output][component] === cpu[output][component]).toBe(true);
    } else expect(Array.from(gpu[output])).toEqual(Array.from(cpu[output]));
  }
  expect(cpu['requiredCount'][0]).toBe(mode === 'value-change' ? 4 : 3);
  expect(cpu['overflow'][0]).toBe(Number(overflow));
  expect(Array.from(gpu['candidate'])).toEqual([Number(Boolean(stayOverflow || stayCandidate))]);
});
