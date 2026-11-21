// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeLocationChangePointsOnCPU} from '../../../src/gpu-spatial-analysis/change-points/cpu-location-change-points-operation';
import {UNIT_SCALING_PARTITIONS} from './unit-scaling-fixture';

type ChangePointOptions = Parameters<typeof computeLocationChangePointsOnCPU>[2];

for (const fixture of UNIT_SCALING_PARTITIONS) {
  it(`PTS-125 ${fixture.cost}, units x${fixture.scale}, penalty ${fixture.penalty}`, () => {
    const result = computeLocationChangePointsOnCPU(
      Float32Array.from(fixture.values),
      Float32Array.of(fixture.penalty, fixture.minimum),
      {
        layout: 'cell-major',
        cellCount: 1,
        sliceCount: 8,
        cost: fixture.cost,
        maximumChangePoints: 7
      } satisfies ChangePointOptions
    );
    const points = Array.from(result.changePoints.slice(0, result.changePointCount[0]));
    if (fixture.cost === 'mean' && (fixture.scale === 1 || fixture.penalty === 200)) {
      expect([[], [4]]).toContainEqual(points); // exactly tied optimal partitions
    } else expect(points).toEqual(fixture.points);
    expect(Number.isFinite(result.optimalCost[0])).toBe(true);
    expect(Math.abs(result.optimalCost[0] - fixture.expectedCost)).toBeLessThan(2e-5);
  });
}

it('PTS-125 Poisson rejects the negative-count series in both units', () => {
  for (const scale of [1, 10]) {
    const result = computeLocationChangePointsOnCPU(
      Float32Array.from([0, 0.1, -0.1, 0, 1, 1.1, 0.9, 1], value => value * scale),
      Float32Array.of(2, 2),
      {
        layout: 'cell-major',
        cellCount: 1,
        sliceCount: 8,
        cost: 'poisson',
        maximumChangePoints: 7
      } satisfies ChangePointOptions
    );
    expect(Number.isNaN(result.optimalCost[0])).toBe(true);
    expect(result.changePointCount[0]).toBe(0);
  }
});

it('Poisson count rescaling requires a linear penalty rescaling', () => {
  // For one split at 4, the deviance gain is
  // 2*a*(4*log(2/3) + 8*log(4/3)) = a*1.35919229436.
  // It is below penalty 2 at a=1, above 2 at a=10, and below the scaled penalty 20.
  for (const [scale, penalty, expectedPoints] of [
    [1, 2, []],
    [10, 2, [4]],
    [10, 20, []]
  ] as const) {
    const options = {
      cellCount: 1,
      sliceCount: 8,
      layout: 'cell-major',
      cost: 'poisson',
      maximumChangePoints: 7
    } satisfies ChangePointOptions;
    const result = computeLocationChangePointsOnCPU(
      Float32Array.from([1, 1, 1, 1, 2, 2, 2, 2], value => value * scale),
      Float32Array.of(penalty, 2),
      options
    );
    expect(Array.from(result.changePoints.slice(0, result.changePointCount[0]))).toEqual(
      expectedPoints
    );
    expect(Number.isFinite(result.optimalCost[0])).toBe(true);
  }
});
