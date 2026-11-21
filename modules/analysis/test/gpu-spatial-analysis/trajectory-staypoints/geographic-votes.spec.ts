// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {CPUTrajectoryStaypointsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation';
import {GPUTrajectoryStaypointsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints-operation';
import {CPUTrajectoryLocationsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation';
import {GPUTrajectoryLocationsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

for (const scenario of [
  {positions: [180, 0, 180, 0, 177, 0], geographic: true, expected: [178.5, 0]},
  {positions: [180, 0, -180, 0, 177, 0], geographic: true, expected: [178.5, 0]},
  {positions: [180, 0, 540, 0, 177, 0], geographic: true, expected: [178.5, 0]},
  {
    positions: [180 - 2 ** -16, 0, -180 - 2 ** -16, 0, 177, 0],
    geographic: true,
    expected: [178.5, 0]
  },
  {positions: [20, 90, 70, 90, 0, 87], geographic: true, expected: [0, 88.5]},
  {positions: [20, 90, 70, 90, 0, 87], geographic: false, expected: [30, 89]}
]) {
  it(`staypoint and location votes use physical identity: ${scenario.positions} geographic=${scenario.geographic}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const spatialContext = scenario.geographic
      ? ({
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle'
        } as const)
      : undefined;
    // Independent centers: seam mean 180-3/2; pole mean (0,(90+87)/2);
    // planar mean ((20+70)/3,(90+90+87)/3).
    for (const Operation of [GPUTrajectoryStaypointsOperation, CPUTrajectoryStaypointsOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: Float32Array.from(scenario.positions)},
          timestamps: {format: 'float32', values: Float32Array.of(0, 1, 2)},
          offsets: {format: 'uint32', values: Uint32Array.of(0, 3)},
          parameters: {format: 'float32', values: Float32Array.of(400000, 1, 0, 1)}
        },
        operations: [
          new Operation({
            positions: 'positions',
            timestamps: 'timestamps',
            trackOffsets: 'offsets',
            parameters: 'parameters',
            rowCount: 3,
            capacity: 1,
            spatialContext,
            output: {
              ids: 'tracks',
              centroids: 'centroids',
              startRows: 'starts',
              endRows: 'ends',
              staypointIds: 'ids',
              status: {count: 'count', overflow: 'overflow', candidateOverflow: 'candidateOverflow'}
            }
          })
        ],
        outputs: ['centroids', 'starts', 'ends', 'ids', 'count', 'overflow']
      });
      expect(Array.from(result['count'])).toEqual([1]);
      expect(Array.from(result['overflow'])).toEqual([0]);
      expect(Array.from(result['centroids'])).toEqual(scenario.expected);
      expect(Array.from(result['starts'])).toEqual([0]);
      expect(Array.from(result['ends'])).toEqual([3]);
      expect(Array.from(result['ids'])).toEqual([0, 0, 0]);
    }
    for (const Operation of [GPUTrajectoryLocationsOperation, CPUTrajectoryLocationsOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: Float32Array.from(scenario.positions)},
          tracks: {format: 'uint32', values: new Uint32Array(3)},
          active: {format: 'uint32', values: Uint32Array.of(3)},
          parameters: {format: 'float32', values: Float32Array.of(400000, 0, 0, 0)}
        },
        operations: [
          new Operation({
            centroids: 'positions',
            tracks: 'tracks',
            count: 'active',
            parameters: 'parameters',
            capacity: 3,
            maximumIterations: 2,
            spatialContext,
            output: {
              locationIds: 'ids',
              centers: 'centers',
              status: {count: 'count', iterationLimitReached: 'limit'}
            }
          })
        ],
        outputs: ['centers', 'ids', 'count']
      });
      expect(Array.from(result['count'])).toEqual([1]);
      expect(Array.from(result['centers']).slice(0, 2)).toEqual(scenario.expected);
      expect(Array.from(result['ids'])).toEqual([0, 0, 0]);
    }
  });
}
