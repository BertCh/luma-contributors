// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUMobilityIndicatorsOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-mobility-indicators-operation';
import {CPUMobilityIndicatorsOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/cpu-mobility-indicators-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

for (const cancelling of [false, true]) {
  it(`B2 omitted spherical default preserves ${cancelling ? 'grouped cancellation NaN' : 'zero stationary gyration'}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rowCount = cancelling ? 1800 : 1000;
    const positions = Float32Array.from({length: rowCount * 2}, (_, component) =>
      component % 2 === 1
        ? cancelling
          ? 0
          : 40.75
        : cancelling
          ? [17, 137, -103][Math.floor(component / 2 / 600)]
          : -74
    );
    // Independent closed forms: coincident masses have radius zero. The three
    // equally weighted directions at 120-degree intervals sum to zero.
    for (const Operation of [GPUMobilityIndicatorsOperation, CPUMobilityIndicatorsOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {
          timestamps: {
            format: 'float32',
            values: Float32Array.from({length: rowCount}, (_, row) => row)
          },
          points: {format: 'float32x2', values: positions},
          tracks: {format: 'uint32', values: new Uint32Array(rowCount)},
          starts: {format: 'uint32', values: Uint32Array.from({length: rowCount}, (_, row) => row)},
          durations: {format: 'float32', values: new Float32Array(rowCount).fill(1)},
          locations: {format: 'uint32', values: new Uint32Array(rowCount)},
          active: {format: 'uint32', values: Uint32Array.of(rowCount)}
        },
        operations: [
          new Operation({
            timestamps: 'timestamps',
            rowCount,
            trackCount: 1,
            staypointCapacity: rowCount,
            spatialContext: {
              coordinateSpace: 'longitude-latitude',
              metric: 'great-circle',
              sphereRadius: 6371000
            },
            staypoints: {
              tracks: 'tracks',
              centroids: 'points',
              startRows: 'starts',
              durations: 'durations',
              locationIds: 'locations',
              status: {count: 'active'}
            },
            output: {radiusOfGyration: 'radius'}
          })
        ],
        outputs: ['radius']
      });
      expect(result['radius'].length).toBe(1);
      if (cancelling) expect(Number.isNaN(result['radius'][0])).toBe(true);
      else expect(result['radius'][0]).toBe(0);
    }
  });
}
