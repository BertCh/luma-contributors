// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeTrajectoryLocationsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation';
import {computeTrajectoryStaypointsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation';

// Path x=0,1,2,3, epsilon 1. One synchronous hook/halve gives labels
// [0,0,0,1], leaving edge (2,3) unresolved; two rounds give all zero.
// Exact literals also appear in the headless contributor/operation tests.
for (const scenario of [
  {rounds: 1, status: [0, 0, 1], labels: new Array(4).fill(0xffffffff)},
  {rounds: 2, status: [1, 1, 0], labels: [0, 0, 0, 0]}
]) {
  it(`CPU locations pins unresolved-edge status after ${scenario.rounds} rounds`, () => {
    const result = computeTrajectoryLocationsOnCPU(
      {
        centroids: new Float32Array([0, 0, 1, 0, 2, 0, 3, 0]),
        tracks: new Uint32Array(4),
        count: new Uint32Array([4]),
        parameters: new Float32Array([1, 0, 0, 0])
      },
      {maximumIterations: scenario.rounds}
    );
    expect([...result.locationCount, ...result.converged, ...result.iterationLimitReached]).toEqual(
      scenario.status
    );
    expect(Array.from(result.locationIds)).toEqual(scenario.labels);
  });
}

it('CPU locations recover after an unsettled submission on the same input arrays', () => {
  const input = {
    centroids: Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0),
    tracks: new Uint32Array(4),
    count: Uint32Array.of(4),
    parameters: Float32Array.of(1, 0, 0, 0)
  };
  for (const epsilon of [1, 2, 1, 2]) {
    input.parameters[0] = epsilon;
    const result = computeTrajectoryLocationsOnCPU(input, {maximumIterations: 1});
    expect([...result.locationCount, ...result.converged, ...result.iterationLimitReached]).toEqual(
      epsilon === 1 ? [0, 0, 1] : [1, 1, 0]
    );
    expect(Array.from(result.locationIds)).toEqual(
      new Array(4).fill(epsilon === 1 ? 0xffffffff : 0)
    );
    if (epsilon === 2) expect(Array.from(result.centers).slice(0, 2)).toEqual([1.5, 0]);
  }
});

for (const scenario of [
  {capacity: 1, status: [1, 2, 1, 0, 1]},
  {capacity: 2, status: [2, 2, 0, 0, 2]}
]) {
  it(`CPU staypoints independently requires two stays at capacity ${scenario.capacity}`, () => {
    // Two stationary unit-duration two-fix tracks each meet minimum duration 1.
    const result = computeTrajectoryStaypointsOnCPU(
      {
        positions: new Float32Array([0, 0, 0, 0, 10, 0, 10, 0]),
        timestamps: new Float32Array([0, 1, 0, 1]),
        trackOffsets: new Uint32Array([0, 2, 4]),
        parameters: new Float32Array([1, 1, 0, 1])
      },
      {capacity: scenario.capacity, maximumSearchRows: 4}
    );
    expect([
      ...result.count,
      ...result.requiredCount,
      ...result.overflow,
      ...result.candidateOverflow,
      ...result.drawInstanceCount
    ]).toEqual(scenario.status);
  });
}
