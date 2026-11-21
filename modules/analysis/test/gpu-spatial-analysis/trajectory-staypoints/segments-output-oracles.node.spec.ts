// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeTrajectorySegmentsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-segments-operation';
import {getSegmentsFixture} from './trajectory-segments-operation-fixture';
import {SEGMENT_EXPECTED} from './segments-output-oracles-fixture';
import {expectSegmentOracleOutputs, SEGMENT_OPTIONALS} from './segments-output-oracles-fixture';

it('R7 segment assertions fail if a requested optional record disappears', () => {
  for (const optional of SEGMENT_OPTIONALS.filter(port => port !== 'none' && port !== 'trips-only'))
    expect(() => expectSegmentOracleOutputs({}, [optional])).toThrow();
});

it('R7 triplegs retain literal identities and counts without requesting trips', () => {
  const result = computeTrajectorySegmentsOnCPU(
    {...getSegmentsFixture(), staypoints: undefined},
    {triplegCapacity: 6}
  );
  expect(Array.from(result.triplegIds)).toEqual([0, 0, 0, 1, 1, 0xffffffff]);
  expect(Array.from(result.triplegCount)).toEqual([5]);
  expect(Array.from(result.triplegOverflow)).toEqual([0]);
});

it('segments match every hand-derived track, row, stay endpoint and count', () => {
  const result = computeTrajectorySegmentsOnCPU(getSegmentsFixture(), {
    triplegCapacity: 6,
    tripCapacity: 5
  });
  for (const [port, expected] of Object.entries(SEGMENT_EXPECTED))
    expect(Array.from(result[port as keyof typeof result]), port).toEqual(expected);
});

it('segments report literal capacity overflow in both record tables', () => {
  const result = computeTrajectorySegmentsOnCPU(getSegmentsFixture(), {
    triplegCapacity: 3,
    tripCapacity: 1
  });
  expect(Array.from(result.triplegCount)).toEqual([3]);
  expect(Array.from(result.triplegOverflow)).toEqual([1]);
  expect(Array.from(result.tripCount)).toEqual([1]);
  expect(Array.from(result.tripOverflow)).toEqual([1]);
});
