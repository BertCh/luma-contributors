// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUTrajectoryThinningOperation} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning-operation';
import {CPUTrajectoryThinningOperation} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-thinning-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
it('keeps original-row selection beyond compact thinning capacity in both named forms', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUTrajectoryThinningOperation, CPUTrajectoryThinningOperation])
    for (const capacity of [1, 2]) {
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: Float32Array.of(0, 0, 0.25, 0, 1, 0)},
          timestamps: {format: 'float32', values: Float32Array.of(0, 1, 2)},
          offsets: {format: 'uint32', values: Uint32Array.of(0, 3)},
          parameters: {format: 'float32', values: Float32Array.of(1, 0, 0, 0)}
        },
        operations: [
          new Operation({
            positions: 'positions',
            timestamps: 'timestamps',
            trackOffsets: 'offsets',
            parameters: 'parameters',
            mode: 'minimum-distance',
            rowCount: 3,
            trackCount: 1,
            capacity,
            output: {
              sourceRows: 'rows',
              sourceSelectionMask: 'selection',
              trackOffsets: 'newOffsets',
              status: {count: 'count', overflow: 'overflow', candidateOverflow: 'candidateOverflow'}
            }
          })
        ],
        outputs: ['rows', 'selection', 'count', 'overflow']
      });
      expect(Array.from(result['rows'])).toEqual(capacity === 1 ? [0] : [0, 2]);
      expect(Array.from(result['selection'])).toEqual([1, 0, 1]);
      expect(Array.from(result['count'])).toEqual([capacity]);
      expect(Array.from(result['overflow'])).toEqual([Number(capacity === 1)]);
    }
});
