// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';

it('honors an explicit one-row search budget and marks split stays incomplete', () => {
  const result = computeTrajectoryStaypointsOnCPU(
    {
      positions: new Float32Array(8),
      timestamps: Float32Array.of(0, 1, 2, 3),
      trackOffsets: Uint32Array.of(0, 4),
      parameters: Float32Array.of(1, 1, 0, 1)
    },
    {capacity: 2, maximumSearchRows: 1}
  );
  expect(result.count[0]).toBe(2);
  expect(Array.from(result.startRows)).toEqual([0, 2]);
  expect(Array.from(result.finishRows)).toEqual([2, 3]);
  expect(result.candidateOverflow[0]).toBe(1);
});
import {
  computeTrajectoryStaypointsOnCPU,
  CPUTrajectoryStaypointsOperation
} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation';
import {
  GPUTrajectoryStaypointsOperation,
  type TrajectoryStaypointsOperationProps
} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints-operation';
import {getGPUTrajectoryStaypointsParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints';

const PARAMETERS = getGPUTrajectoryStaypointsParameterValues({
  distanceThreshold: 5,
  timeThreshold: 2,
  includeLast: true
});

for (const [name, positions, centroid] of [
  ['equivalent antimeridian', [-180, 0, 180, 0], [-180, 0]],
  ['north pole', [0, 90, 180, 90], [90, 90]],
  ['south pole', [0, -90, 180, -90], [90, -90]],
  ['clamped terminal offset', [0, 0, 0, 0], [0, 0]]
] as const) {
  it(`keeps both fixes in one stay: ${name}`, () => {
    const trackOffsets = new Uint32Array([0, name === 'clamped terminal offset' ? 3 : 2]);
    const result = computeTrajectoryStaypointsOnCPU(
      {
        positions: new Float32Array(positions),
        timestamps: new Float32Array([0, 1]),
        trackOffsets,
        parameters: new Float32Array([name === 'clamped terminal offset' ? 1 : 1e-17, 1, 0, 1])
      },
      {
        capacity: 1,
        spatialContext:
          name === 'clamped terminal offset'
            ? undefined
            : {
                coordinateSpace: 'longitude-latitude',
                metric: 'great-circle',
                units: 'meters',
                sphereRadius: 1
              }
      }
    );
    // Equivalent positions have zero distance; the only finish is the last fix at t=1.
    expect(
      Object.fromEntries(Object.entries(result).map(([port, values]) => [port, Array.from(values)]))
    ).toEqual({
      ids: [0],
      startRows: [0],
      endRows: [2],
      finishRows: [1],
      centroids: [...centroid],
      durations: [1],
      staypointIds: [0, 0],
      count: [1],
      requiredCount: [1],
      overflow: [0],
      candidateOverflow: [0],
      drawInstanceCount: [1]
    });
    expect(Array.from(trackOffsets)).toEqual([0, name === 'clamped terminal offset' ? 3 : 2]);
  });
}

it('keeps the IDs and every output of two non-empty tracks distinct', () => {
  const result = computeTrajectoryStaypointsOnCPU(
    {
      positions: new Float32Array([0, 0, 0, 0, 10, 0, 10, 0]),
      timestamps: new Float32Array([0, 1, 0, 1]),
      trackOffsets: new Uint32Array([0, 2, 4]),
      parameters: new Float32Array([1, 1, 0, 1])
    },
    {capacity: 2}
  );
  expect(
    Object.fromEntries(Object.entries(result).map(([port, values]) => [port, Array.from(values)]))
  ).toEqual({
    ids: [0, 1],
    startRows: [0, 2],
    endRows: [2, 4],
    finishRows: [1, 3],
    centroids: [0, 0, 10, 0],
    durations: [1, 1],
    staypointIds: [0, 0, 1, 1],
    count: [2],
    requiredCount: [2],
    overflow: [0],
    candidateOverflow: [0],
    drawInstanceCount: [2]
  });
});

it('matches a hand-computed sliding chain, distinct centroids and capacity truncation', () => {
  // Anchor 0 closes at row 3 (x=6); row 4 closes its final window. Repeated x=0
  // has one centroid vote, so the first centroid is (0+2)/2=1, not 2/3.
  const result = computeTrajectoryStaypointsOnCPU(
    {
      positions: new Float32Array([0, 0, 0, 0, 2, 0, 6, 0, 6, 0]),
      timestamps: new Float32Array([0, 1, 2, 3, 5]),
      trackOffsets: new Uint32Array([0, 0, 5]),
      parameters: PARAMETERS
    },
    {capacity: 3}
  );
  expect(Array.from(result.ids)).toEqual([1, 1, 0xffffffff]);
  expect(Array.from(result.startRows)).toEqual([0, 3, 0xffffffff]);
  expect(Array.from(result.endRows)).toEqual([3, 5, 0xffffffff]);
  expect(Array.from(result.finishRows)).toEqual([3, 4, 0xffffffff]);
  expect(Array.from(result.centroids)).toEqual([1, 0, 6, 0, 0, 0]);
  expect(Array.from(result.durations)).toEqual([3, 2, 0]);
  expect(Array.from(result.staypointIds)).toEqual([0, 0, 0, 1, 1]);
  expect(Array.from(result.count)).toEqual([2]);
  expect(Array.from(result.drawInstanceCount)).toEqual([2]);
  expect(Array.from(result.requiredCount)).toEqual([2]);
  expect(Array.from(result.overflow)).toEqual([0]);
  expect(Array.from(result.candidateOverflow)).toEqual([0]);
  const truncated = computeTrajectoryStaypointsOnCPU(
    {
      positions: new Float32Array([0, 0, 0, 0, 10, 0, 10, 0]),
      timestamps: new Float32Array([0, 2, 3, 5]),
      trackOffsets: new Uint32Array([0, 4]),
      parameters: PARAMETERS
    },
    {capacity: 1}
  );
  expect(Array.from(truncated.count)).toEqual([1]);
  expect(Array.from(truncated.requiredCount)).toEqual([2]);
  expect(Array.from(truncated.overflow)).toEqual([1]);
  expect(Array.from(truncated.staypointIds)).toEqual([0, 0, 0xffffffff, 0xffffffff]);
});

it('uses strict gaps and exact Int64 differences across a low-word carry', () => {
  const origin = (1n << 60n) - 1n;
  const timestamps = new Uint32Array(8);
  [0n, 2n, 5n, 7n].forEach((delta, row) => {
    const timestamp = origin + delta;
    timestamps[row * 2] = Number(timestamp & 0xffffffffn);
    timestamps[row * 2 + 1] = Number(timestamp >> 32n);
  });
  const result = computeTrajectoryStaypointsOnCPU(
    {
      positions: new Float32Array(8),
      timestamps,
      trackOffsets: new Uint32Array([0, 4]),
      parameters: getGPUTrajectoryStaypointsParameterValues({
        distanceThreshold: 1,
        timeThreshold: 2,
        maximumGap: 2,
        includeLast: true
      })
    },
    {capacity: 2}
  );
  // Gap 2->5 discards the previous window; the equal-to-limit step 5->7 is retained.
  expect(Array.from(result.startRows)).toEqual([2, 0xffffffff]);
  expect(Array.from(result.durations)).toEqual([2, 0]);
});

it('does not split a stationary window at the GPU default search cap', () => {
  const rowCount = 5000;
  const result = computeTrajectoryStaypointsOnCPU(
    {
      positions: new Float32Array(rowCount * 2),
      timestamps: Float32Array.from({length: rowCount}, (_, row) => row),
      trackOffsets: new Uint32Array([0, rowCount]),
      parameters: PARAMETERS
    },
    {capacity: 2}
  );
  expect(result.requiredCount[0]).toBe(1);
  expect(result.durations[0]).toBe(rowCount - 1);
  expect(result.candidateOverflow[0]).toBe(0);
});

it('unwraps spherical centroids across the dateline and uses nearest-even half-turn ties', () => {
  for (const [longitudes, expected] of [
    [[179, -179, 179], -180],
    [[0, 180, 0], 90],
    [[0, -180, 0], -90]
  ] as const) {
    const result = computeTrajectoryStaypointsOnCPU(
      {
        positions: new Float32Array(longitudes.flatMap(longitude => [longitude, 0])),
        timestamps: new Float32Array([0, 1, 2]),
        trackOffsets: new Uint32Array([0, 3]),
        parameters: getGPUTrajectoryStaypointsParameterValues({
          distanceThreshold: 4,
          timeThreshold: 2,
          includeLast: true
        })
      },
      {
        capacity: 1,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          units: 'meters',
          sphereRadius: 1
        }
      }
    );
    expect(result.count[0]).toBe(1);
    expect(result.centroids[0]).toBe(expected);
    expect(result.centroids[1]).toBe(0);
  }
});

it('checks inconsistent arrays and construction-time output lengths in both classes', () => {
  const input = {
    positions: new Float32Array(4),
    timestamps: new Float32Array(2),
    trackOffsets: new Uint32Array([0, 2]),
    parameters: PARAMETERS
  };
  expect(() =>
    computeTrajectoryStaypointsOnCPU({...input, timestamps: new Float32Array(1)}, {capacity: 1})
  ).toThrow(/timestamps/);
  expect(() =>
    computeTrajectoryStaypointsOnCPU({...input, positions: new Float32Array(3)}, {capacity: 1})
  ).toThrow(/positions/);
  expect(() =>
    computeTrajectoryStaypointsOnCPU(
      {...input, trackOffsets: new Uint32Array([2, 1])},
      {capacity: 1}
    )
  ).toThrow(/trackOffsets/);
  expect(() =>
    computeTrajectoryStaypointsOnCPU({...input, parameters: new Float32Array(1)}, {capacity: 1})
  ).toThrow(/parameters/);
  const props: TrajectoryStaypointsOperationProps = {
    id: 'stay-test',
    positions: 'positions',
    timestamps: 'timestamps',
    trackOffsets: 'trackOffsets',
    parameters: 'parameters',
    rowCount: 2,
    capacity: 1,
    output: {ids: 'ids', status: {count: 'count', overflow: 'overflow'}}
  };
  for (const Operation of [GPUTrajectoryStaypointsOperation, CPUTrajectoryStaypointsOperation]) {
    expect(() => new Operation({...props, capacity: 0})).toThrow(/^stay-test .*length/);
    expect(() => new Operation({...props, output: {...props.output, ids: 'positions'}})).toThrow(
      /^stay-test /
    );
  }
  expect(() => new CPUTrajectoryStaypointsOperation({...props, rowCount: -1})).toThrow(
    /^stay-test rowCount/
  );
  // GPU wrappers validate contributor-specific properties when lowering, as in the pilot.
  expect(
    () => new CPUTrajectoryStaypointsOperation({...props, maximumSearchRows: 1e9})
  ).not.toThrow();
});
