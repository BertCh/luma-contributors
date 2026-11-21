// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUTrajectoryMetrics,
  getGPUTrajectoryMetricsParameterValues
} from '../../../src/gpu-spatial-analysis/trajectory-analysis/index';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GraphHarness, importTrajectory} from '../trajectory-staypoints/movement-spec-utils';

it('metrics draw count is the capacity-clamped count of two hand-counted stops', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const capacity of [1, 2])
    for (const withDraw of [true, false]) {
      const harness = new GraphHarness(device, `metrics-draw-${capacity}-${withDraw}`);
      // Still for [0,2], move at t=3, still for [3,5]: exactly two stops of duration 2.
      const trajectory = importTrajectory(harness, {
        positions: [
          [0, 0],
          [0, 0],
          [0, 0],
          [10, 0],
          [10, 0],
          [10, 0]
        ],
        times: Float32Array.of(0, 1, 2, 3, 4, 5),
        trackOffsets: [0, 6]
      });
      const ids = harness.output('ids', 'uint32', capacity);
      const count = harness.output('count', 'uint32', 1);
      const overflow = harness.output('overflow', 'uint32', 1);
      const draw = withDraw ? harness.output('draw', 'uint32', 1) : undefined;
      for (const output of [ids, count, overflow, draw])
        output?.buffer.write(new Uint32Array(output.buffer.byteLength / 4).fill(93));
      harness.graph.add(
        new GPUTrajectoryMetrics({
          ...trajectory,
          parameters: harness.input(
            'parameters',
            getGPUTrajectoryMetricsParameterValues({stopSpeedThreshold: 1, stopMinimumDuration: 2}),
            'float32',
            4
          ),
          output: {
            stops: {
              ids: ids.view,
              status: {count: count.view, overflow: overflow.view},
              ...(draw ? {drawInstanceCount: draw.view} : {})
            }
          }
        })
      );
      const result = await harness.run(async () => ({
        count: await readUint32(count.buffer, 1),
        overflow: await readUint32(overflow.buffer, 1),
        ids: await readUint32(ids.buffer, capacity),
        draw: draw ? await readUint32(draw.buffer, 1) : undefined
      }));
      expect(result.count).toEqual([capacity]);
      expect(result.overflow).toEqual([Number(capacity < 2)]);
      expect(result.ids).toEqual(capacity === 1 ? [0] : [0, 0]);
      if (draw) expect(result.draw).toEqual([capacity]);
    }
});
