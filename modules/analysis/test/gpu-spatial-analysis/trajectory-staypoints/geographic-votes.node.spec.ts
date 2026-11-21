// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeTrajectoryStaypointsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation';
import {computeTrajectoryLocationsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation';

// Each seam spelling describes only two distinct equatorial points: their
// anchor-relative mean is 180 + (0 - 3)/2 = 178.5 degrees.
for (const longitudes of [
  [180, 180, 177],
  [180, -180, 177],
  [180, 540, 177]
]) {
  it(`geographic votes identify seam spellings ${longitudes}`, () => {
    const positions = Float32Array.from(longitudes.flatMap(longitude => [longitude, 0]));
    const original = positions.slice();
    const spatialContext = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
    const stays = computeTrajectoryStaypointsOnCPU(
      {
        positions,
        timestamps: Float32Array.of(0, 1, 2),
        trackOffsets: Uint32Array.of(0, 3),
        parameters: Float32Array.of(500000, 1, 0, 1)
      },
      {capacity: 1, spatialContext}
    );
    const locations = computeTrajectoryLocationsOnCPU(
      {
        centroids: positions,
        tracks: new Uint32Array(3),
        count: Uint32Array.of(3),
        parameters: Float32Array.of(500000, 0, 0, 0)
      },
      {spatialContext, maximumIterations: 2}
    );
    expect(Array.from(stays.count)).toEqual([1]);
    expect(Array.from(stays.centroids)).toEqual([178.5, 0]);
    expect(Array.from(stays.startRows)).toEqual([0]);
    expect(Array.from(stays.endRows)).toEqual([3]);
    expect(Array.from(stays.staypointIds)).toEqual([0, 0, 0]);
    expect(Array.from(locations.locationIds)).toEqual([0, 0, 0]);
    expect(Array.from(locations.centers).slice(0, 2)).toEqual([178.5, 0]);
    expect(positions).toEqual(original);
  });
}

for (const geographic of [true, false]) {
  it(`distinct votes ${geographic ? 'ignore longitude at an exact pole' : 'preserve planar equality'}`, () => {
    // Geographic: two physical points (north pole and (0,87)), mean (0,88.5).
    // Planar: three distinct coordinates, mean ((20+70+0)/3,(90+90+87)/3).
    const positions = Float32Array.of(20, 90, 70, 90, 0, 87);
    const spatialContext = geographic
      ? ({coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const)
      : undefined;
    const stays = computeTrajectoryStaypointsOnCPU(
      {
        positions,
        timestamps: Float32Array.of(0, 1, 2),
        trackOffsets: Uint32Array.of(0, 3),
        parameters: Float32Array.of(500000, 1, 0, 1)
      },
      {capacity: 1, spatialContext}
    );
    const locations = computeTrajectoryLocationsOnCPU(
      {
        centroids: positions,
        tracks: new Uint32Array(3),
        count: Uint32Array.of(3),
        parameters: Float32Array.of(500000, 0, 0, 0)
      },
      {spatialContext}
    );
    const expected = geographic ? [0, 88.5] : [30, 89];
    expect(Array.from(stays.centroids)).toEqual(expected);
    expect(Array.from(locations.centers).slice(0, 2)).toEqual(expected);
  });
}
