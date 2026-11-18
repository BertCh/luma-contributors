// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUTrajectoryStaypointsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation';
import {
  GPUTrajectoryStaypointsOperation,
  type TrajectoryStaypointsOperationProps
} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints-operation';
import {getGPUTrajectoryStaypointsParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('both staypoint forms split an explicitly bounded stationary search', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUTrajectoryStaypointsOperation, CPUTrajectoryStaypointsOperation]) {
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: new Float32Array(8)},
        timestamps: {format: 'float32', values: Float32Array.of(0, 1, 2, 3)},
        offsets: {format: 'uint32', values: Uint32Array.of(0, 4)},
        parameters: {format: 'float32', values: Float32Array.of(1, 1, 0, 1)}
      },
      operations: [
        new Operation({
          positions: 'positions',
          timestamps: 'timestamps',
          trackOffsets: 'offsets',
          parameters: 'parameters',
          rowCount: 4,
          capacity: 2,
          maximumSearchRows: 1,
          output: {
            ids: 'ids',
            startRows: 'starts',
            finishRows: 'finishes',
            status: {count: 'count', overflow: 'overflow', candidateOverflow: 'candidate'}
          }
        })
      ],
      outputs: ['starts', 'finishes', 'count', 'candidate', 'overflow']
    });
    expect(Array.from(result['starts'])).toEqual([0, 2]);
    expect(Array.from(result['finishes'])).toEqual([2, 3]);
    expect(Array.from(result['count'])).toEqual([2]);
    expect(Array.from(result['candidate'])).toEqual([1]);
    expect(Array.from(result['overflow'])).toEqual([0]);
  }
});

for (const scenario of [
  'ordinary',
  'overflow',
  'no-stays',
  'words',
  'dateline',
  'half-turn'
] as const) {
  it(`staypoint operations compare every output: ${scenario}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: TrajectoryStaypointsOperationProps = {
      positions: 'positions',
      timestamps: 'timestamps',
      trackOffsets: 'trackOffsets',
      parameters: 'parameters',
      timestampFormat: scenario === 'words' ? 'uint32x2' : 'float32',
      spatialContext:
        scenario === 'dateline' || scenario === 'half-turn'
          ? {
              coordinateSpace: 'longitude-latitude',
              metric: 'great-circle',
              units: 'meters',
              sphereRadius: 1
            }
          : undefined,
      rowCount: 5,
      capacity: scenario === 'overflow' ? 1 : 3,
      output: {
        ids: 'ids',
        startRows: 'startRows',
        endRows: 'endRows',
        finishRows: 'finishRows',
        centroids: 'centroids',
        durations: 'durations',
        staypointIds: 'staypointIds',
        drawInstanceCount: 'drawInstanceCount',
        status: {
          count: 'count',
          overflow: 'overflow',
          requiredCount: 'requiredCount',
          candidateOverflow: 'candidateOverflow'
        }
      }
    };
    const timestamps =
      scenario === 'words'
        ? new Uint32Array([0xfffffffe, 123, 0xffffffff, 123, 0, 124, 1, 124, 3, 124])
        : new Float32Array([0, 1, 2, 3, 5]);
    const inputs = {
      positions: {
        format: 'float32x2' as const,
        values: new Float32Array(
          scenario === 'dateline'
            ? [179, 0, -179, 0, 179, 0, -179, 0, 179, 0]
            : scenario === 'half-turn'
              ? [0, 0, 180, 0, 0, 0, 180, 0, 0, 0]
              : [0, 0, 0, 0, 2, 0, 6, 0, 6, 0]
        )
      },
      timestamps: {format: props.timestampFormat!, values: timestamps},
      trackOffsets: {format: 'uint32' as const, values: new Uint32Array([0, 0, 5])},
      parameters: {
        format: 'float32' as const,
        values: getGPUTrajectoryStaypointsParameterValues({
          distanceThreshold: 5,
          timeThreshold: scenario === 'no-stays' ? 100 : 2,
          includeLast: true
        })
      }
    };
    const operation = new GPUTrajectoryStaypointsOperation(props);
    const outputs = Object.values(operation.outputs).map(output => output.value);
    const gpu = await runOperationGraph(device, {inputs, operations: [operation], outputs});
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUTrajectoryStaypointsOperation(props)],
      outputs
    });
    for (const output of outputs) {
      expect(gpu[output].length).toBe(cpu[output].length);
      for (let index = 0; index < cpu[output].length; index++) {
        // Float ports: absolute 1e-5. Integer ports, including all status words: exact.
        if (cpu[output] instanceof Float32Array) {
          expect(Math.abs(gpu[output][index] - cpu[output][index])).toBeLessThanOrEqual(1e-5);
        } else expect(gpu[output][index]).toBe(cpu[output][index]);
      }
    }
  });
}

for (const scenario of [
  'equivalent-antimeridian',
  'north-pole',
  'south-pole',
  'clamped-offset',
  'two-tracks'
] as const) {
  it(`staypoint review regression compares every output: ${scenario}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const isTwoTracks = scenario === 'two-tracks';
    const isSpherical = scenario !== 'clamped-offset' && !isTwoTracks;
    const positions = new Float32Array(
      scenario === 'equivalent-antimeridian'
        ? [-180, 0, 180, 0]
        : scenario === 'north-pole'
          ? [0, 90, 180, 90]
          : scenario === 'south-pole'
            ? [0, -90, 180, -90]
            : isTwoTracks
              ? [0, 0, 0, 0, 10, 0, 10, 0]
              : [0, 0, 0, 0]
    );
    const props: TrajectoryStaypointsOperationProps = {
      positions: 'positions',
      timestamps: 'timestamps',
      trackOffsets: 'trackOffsets',
      parameters: 'parameters',
      rowCount: isTwoTracks ? 4 : 2,
      capacity: isTwoTracks ? 2 : 1,
      spatialContext: isSpherical
        ? {
            coordinateSpace: 'longitude-latitude',
            metric: 'great-circle',
            units: 'meters',
            sphereRadius: 1
          }
        : undefined,
      output: {
        ids: 'ids',
        startRows: 'startRows',
        endRows: 'endRows',
        finishRows: 'finishRows',
        centroids: 'centroids',
        durations: 'durations',
        staypointIds: 'staypointIds',
        drawInstanceCount: 'drawInstanceCount',
        status: {
          count: 'count',
          requiredCount: 'requiredCount',
          overflow: 'overflow',
          candidateOverflow: 'candidateOverflow'
        }
      }
    };
    const inputs = {
      positions: {format: 'float32x2' as const, values: positions},
      timestamps: {
        format: 'float32' as const,
        values: new Float32Array(isTwoTracks ? [0, 1, 0, 1] : [0, 1])
      },
      trackOffsets: {
        format: 'uint32' as const,
        values: new Uint32Array(
          isTwoTracks ? [0, 2, 4] : scenario === 'clamped-offset' ? [0, 3] : [0, 2]
        )
      },
      parameters: {
        format: 'float32' as const,
        values: new Float32Array([isSpherical ? 1e-17 : 1, 1, 0, 1])
      }
    };
    const expected: Record<string, readonly number[]> = isTwoTracks
      ? {
          ids: [0, 1],
          startRows: [0, 2],
          endRows: [2, 4],
          finishRows: [1, 3],
          centroids: [0, 0, 10, 0],
          durations: [1, 1],
          staypointIds: [0, 0, 1, 1],
          drawInstanceCount: [2],
          count: [2],
          requiredCount: [2],
          overflow: [0],
          candidateOverflow: [0]
        }
      : {
          ids: [0],
          startRows: [0],
          endRows: [2],
          finishRows: [1],
          centroids:
            scenario === 'equivalent-antimeridian'
              ? [-180, 0]
              : scenario === 'north-pole'
                ? [90, 90]
                : scenario === 'south-pole'
                  ? [90, -90]
                  : [0, 0],
          durations: [1],
          staypointIds: [0, 0],
          drawInstanceCount: [1],
          count: [1],
          requiredCount: [1],
          overflow: [0],
          candidateOverflow: [0]
        };
    const outputs = Object.keys(expected);
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUTrajectoryStaypointsOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUTrajectoryStaypointsOperation(props)],
      outputs
    });
    // All hand-derived outputs here are exactly representable; tolerance is zero for every port.
    for (const output of outputs) {
      expect(Array.from(cpu[output])).toEqual(expected[output]);
      expect(Array.from(gpu[output])).toEqual(Array.from(cpu[output]));
    }
  });
}
