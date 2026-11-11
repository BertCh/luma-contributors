// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeTrajectoryLocationsOnCPU,
  CPUTrajectoryLocationsOperation
} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation';
import {
  GPUTrajectoryLocationsOperation,
  type TrajectoryLocationsOperationProps
} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations-operation';
import {getGPUTrajectoryLocationsParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations';

const INPUT = {
  centroids: new Float32Array([0, 0, 2, 0, 4, 0, 0, 0, 10, 0, 0, 0, 2, 0, 5, 0, 5, 0, 99, 99]),
  tracks: new Uint32Array([0, 0, 0, 0, 0, 1, 1, 1, 1, 99]),
  count: Uint32Array.of(9),
  parameters: getGPUTrajectoryLocationsParameterValues({epsilon: 2})
};

it('matches transitive closure of the radius adjacency matrix and hand-computed distinct centers', () => {
  // Independent oracle: Floyd-Warshall on squared Euclidean distances, not union-find.
  const count = INPUT.count[0];
  const reachable = Array.from({length: count}, (_, first) =>
    Array.from(
      {length: count},
      (_, second) =>
        INPUT.tracks[first] === INPUT.tracks[second] &&
        (INPUT.centroids[first * 2] - INPUT.centroids[second * 2]) ** 2 +
          (INPUT.centroids[first * 2 + 1] - INPUT.centroids[second * 2 + 1]) ** 2 <=
          4
    )
  );
  for (let middle = 0; middle < count; middle++)
    for (let first = 0; first < count; first++)
      for (let second = 0; second < count; second++)
        reachable[first][second] ||= reachable[first][middle] && reachable[middle][second];
  const result = computeTrajectoryLocationsOnCPU(INPUT);
  for (let first = 0; first < count; first++)
    for (let second = 0; second < count; second++)
      expect(result.locationIds[first] === result.locationIds[second]).toBe(
        reachable[first][second]
      );
  expect(Array.from(result.locationIds)).toEqual([0, 0, 0, 0, 1, 2, 2, 3, 3, 0xffffffff]);
  // (0+2+4)/3=2, not (0+2+4+0)/4; duplicate fixes have one centroid vote.
  expect(Array.from(result.centers)).toEqual([2, 0, 10, 0, 1, 0, 5, 0, ...new Array(12).fill(0)]);
  expect(Array.from(result.locationCount)).toEqual([4]);
  expect(Array.from(result.converged)).toEqual([1]);
  expect(Array.from(result.iterationLimitReached)).toEqual([0]);
});

it('handles no active stays and clamps counts above the table capacity', () => {
  const empty = computeTrajectoryLocationsOnCPU({...INPUT, count: Uint32Array.of(0)});
  expect(empty.locationIds.every(value => value === 0xffffffff)).toBe(true);
  expect(empty.centers.every(value => value === 0)).toBe(true);
  expect(empty.locationCount[0]).toBe(0);
  const clamped = computeTrajectoryLocationsOnCPU({...INPUT, count: Uint32Array.of(999)});
  expect(clamped.locationIds[9]).toBe(4);
  expect(clamped.locationCount[0]).toBe(5);
});

it('keeps equivalent dateline/pole fixes connected and rewraps after output rounding', () => {
  const spatialContext = {
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'meters',
    sphereRadius: 1
  } as const;
  for (const [centroids, expectedLongitude, epsilon] of [
    [[-180, 0, 180, 0], -180, 1e-10],
    [[0, 90, 180, 90], 90, 1e-10],
    [[179, 0, -179, 0], -180, 1],
    [[179.99998474121094, 0, 180, 0], -180, 1]
  ] as const) {
    const result = computeTrajectoryLocationsOnCPU(
      {
        centroids: new Float32Array(centroids),
        tracks: new Uint32Array(2),
        count: Uint32Array.of(2),
        parameters: getGPUTrajectoryLocationsParameterValues({epsilon})
      },
      {spatialContext}
    );
    expect(Array.from(result.locationIds)).toEqual([0, 0]);
    expect(result.centers[0]).toBe(expectedLongitude);
    expect(result.centers[0]).toBeLessThan(180);
  }
});

it('rejects malformed arrays and validates construction in both operation classes', () => {
  for (const [argument, replacement] of [
    ['centroids', new Float32Array(3)],
    ['tracks', new Uint32Array(1)],
    ['count', new Uint32Array(0)],
    ['parameters', new Float32Array(3)]
  ] as const)
    expect(() => computeTrajectoryLocationsOnCPU({...INPUT, [argument]: replacement})).toThrow(
      new RegExp(argument)
    );
  expect(() =>
    computeTrajectoryLocationsOnCPU({
      ...INPUT,
      tracks: new Uint32Array([1, 0, ...new Array(8).fill(0)])
    })
  ).toThrow(/tracks/);
  const props: TrajectoryLocationsOperationProps = {
    id: 'locations-test',
    centroids: 'centroids',
    tracks: 'tracks',
    count: 'count',
    parameters: 'parameters',
    capacity: 10,
    output: {locationIds: 'locationIds', status: {count: 'locationCount'}}
  };
  for (const Operation of [GPUTrajectoryLocationsOperation, CPUTrajectoryLocationsOperation]) {
    expect(() => new Operation({...props, capacity: 0})).toThrow(/^locations-test /);
    expect(
      () => new Operation({...props, output: {...props.output, locationIds: 'centroids'}})
    ).toThrow(/^locations-test /);
  }
  expect(
    () => new CPUTrajectoryLocationsOperation({...props, maximumIterations: 1e9})
  ).not.toThrow();
});
