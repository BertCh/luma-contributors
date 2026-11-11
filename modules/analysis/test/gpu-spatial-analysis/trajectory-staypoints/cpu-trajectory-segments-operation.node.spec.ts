// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeTrajectorySegmentsOnCPU,
  CPUTrajectorySegmentsOperation
} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-segments-operation';
import {GPUTrajectorySegmentsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-segments-operation';
import {getSegmentsFixture, getSegmentsProps} from './trajectory-segments-operation-fixture';

it('matches the hand-built timeline, including consecutive activities, nonactivities and gaps', () => {
  const result = computeTrajectorySegmentsOnCPU(getSegmentsFixture(), {
    triplegCapacity: 6,
    tripCapacity: 5
  });
  const missing = 0xffffffff;
  expect(
    Object.fromEntries(Object.entries(result).map(([port, values]) => [port, Array.from(values)]))
  ).toEqual({
    triplegIds: [0, 0, 0, 1, 1, missing],
    triplegStartRows: [2, 6, 12, 16, 20, missing],
    triplegEndRows: [4, 8, 14, 18, 22, missing],
    rowTriplegIds: [
      missing,
      missing,
      0,
      0,
      missing,
      missing,
      1,
      1,
      missing,
      missing,
      missing,
      missing,
      2,
      2,
      missing,
      missing,
      3,
      3,
      missing,
      missing,
      4,
      4
    ],
    triplegCount: [5],
    triplegRequiredCount: [5],
    triplegOverflow: [0],
    tripIds: [0, 0, 1, 1, missing],
    tripStartRows: [2, 12, 16, 20, missing],
    tripFinishRows: [7, 15, 17, 21, missing],
    originStaypoints: [0, 3, missing, missing, missing],
    destinationStaypoints: [2, missing, missing, missing, missing],
    triplegTripIds: [0, 0, 1, 2, 3, missing],
    tripCount: [4],
    tripRequiredCount: [4],
    tripOverflow: [0]
  });
});

it('keeps tripleg overflow separate and assembles trips from the stored prefix', () => {
  const result = computeTrajectorySegmentsOnCPU(getSegmentsFixture(), {
    triplegCapacity: 3,
    tripCapacity: 1
  });
  expect(Array.from(result.triplegCount)).toEqual([3]);
  expect(Array.from(result.triplegRequiredCount)).toEqual([5]);
  expect(Array.from(result.triplegOverflow)).toEqual([1]);
  expect(Array.from(result.tripRequiredCount)).toEqual([2]);
  expect(Array.from(result.tripCount)).toEqual([1]);
  expect(Array.from(result.tripOverflow)).toEqual([1]);
  expect(Array.from(result.triplegTripIds)).toEqual([0, 0, 0xffffffff]);
  expect(Array.from(result.originStaypoints)).toEqual([0]);
  expect(Array.from(result.destinationStaypoints)).toEqual([2]);
  expect(Array.from(result.rowTriplegIds.subarray(16))).toEqual(new Array(6).fill(0xffffffff));
});

it('drops stationary and singleton runs, supports triplegs alone, and clamps active staypoint count', () => {
  const input = getSegmentsFixture();
  const stationary = computeTrajectorySegmentsOnCPU(
    {...input, positions: new Float32Array(44)},
    {triplegCapacity: 2, tripCapacity: 2}
  );
  expect(stationary.triplegRequiredCount[0]).toBe(0);
  expect(stationary.tripRequiredCount[0]).toBe(0);
  expect(stationary.tripIds.every(value => value === 0xffffffff)).toBe(true);
  const triplegs = computeTrajectorySegmentsOnCPU(
    {...input, staypoints: undefined},
    {triplegCapacity: 6}
  );
  expect(Array.from(triplegs.triplegStartRows)).toEqual([2, 6, 12, 16, 20, 0xffffffff]);
  const clamped = computeTrajectorySegmentsOnCPU(
    {...input, staypoints: {...input.staypoints, count: new Uint32Array([99])}},
    {triplegCapacity: 6, tripCapacity: 5}
  );
  expect(clamped.tripRequiredCount[0]).toBe(4);
});

it('rejects malformed arrays and validates both operation constructions', () => {
  const input = getSegmentsFixture();
  for (const [argument, replacement] of [
    ['positions', new Float32Array(3)],
    ['timestamps', new Float32Array(1)],
    ['trackOffsets', new Uint32Array([4, 2])],
    ['staypointIds', new Uint32Array(1)],
    ['parameters', new Float32Array(3)]
  ] as const)
    expect(() =>
      computeTrajectorySegmentsOnCPU({...input, [argument]: replacement}, {triplegCapacity: 2})
    ).toThrow(new RegExp(argument));
  expect(() =>
    computeTrajectorySegmentsOnCPU(
      {...input, staypoints: {...input.staypoints, finishRows: new Uint32Array(1)}},
      {triplegCapacity: 2, tripCapacity: 2}
    )
  ).toThrow(/staypoints.finishRows/);
  expect(() =>
    computeTrajectorySegmentsOnCPU(
      {...input, staypoints: undefined},
      {triplegCapacity: 2, tripCapacity: 2}
    )
  ).toThrow(/staypoints/);
  const props = getSegmentsProps();
  for (const Operation of [GPUTrajectorySegmentsOperation, CPUTrajectorySegmentsOperation]) {
    expect(() => new Operation({...props, triplegCapacity: 0})).toThrow(/^segments-test /);
    expect(() => new Operation({...props, tripCapacity: 0})).toThrow(/^segments-test /);
    expect(
      () =>
        new Operation({
          ...props,
          output: {...props.output, triplegs: {...props.output.triplegs, ids: 'positions'}}
        })
    ).toThrow(/^segments-test /);
  }
  expect(() => new CPUTrajectorySegmentsOperation({...props, staypoints: undefined})).toThrow(
    /^segments-test .*staypoints/
  );
  expect(() => new CPUTrajectorySegmentsOperation({...props, rowCount: -1})).toThrow(
    /^segments-test rowCount/
  );
});
