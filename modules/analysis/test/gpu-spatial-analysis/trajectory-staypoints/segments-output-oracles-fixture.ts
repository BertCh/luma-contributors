// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getSegmentsFixture, getSegmentsProps} from './trajectory-segments-operation-fixture';
import {expect} from 'vitest';

export function expectSegmentOracleOutputs(
  result: Record<string, ArrayLike<number>>,
  expectedKeys: readonly string[]
): void {
  expect(Object.keys(result).sort()).toEqual([...expectedKeys].sort());
  for (const port of expectedKeys)
    expect(Array.from(result[port]), port).toEqual(SEGMENT_EXPECTED[port]);
}

const MISSING = 0xffffffff;
// Hand timeline: track 0 has moving runs [2,4), [6,8), [12,14); track 1 has
// [16,18), stationary [18,20) (discarded), and [20,22), separated by gaps.
// Activity stays are 0,2,3; stays 1,4 are nonactivities (duration equals threshold).
// Thus runs 0,1 share trip 0 (origin 0, destination 2); run 2 is trip 1 (origin 3),
// and the two gap-separated track-1 runs are trips 2,3 with no stay endpoints.
export const SEGMENT_EXPECTED: Record<string, number[]> = {
  triplegIds: [0, 0, 0, 1, 1, MISSING],
  triplegCount: [5],
  triplegOverflow: [0],
  triplegStartRows: [2, 6, 12, 16, 20, MISSING],
  triplegEndRows: [4, 8, 14, 18, 22, MISSING],
  rowTriplegIds: [
    MISSING,
    MISSING,
    0,
    0,
    MISSING,
    MISSING,
    1,
    1,
    MISSING,
    MISSING,
    MISSING,
    MISSING,
    2,
    2,
    MISSING,
    MISSING,
    3,
    3,
    MISSING,
    MISSING,
    4,
    4
  ],
  tripIds: [0, 0, 1, 1, MISSING],
  tripCount: [4],
  tripOverflow: [0],
  tripCandidateOverflow: [0],
  tripStartRows: [2, 12, 16, 20, MISSING],
  tripFinishRows: [7, 15, 17, 21, MISSING],
  originStaypoints: [0, 3, MISSING, MISSING, MISSING],
  destinationStaypoints: [2, MISSING, MISSING, MISSING, MISSING],
  triplegTripIds: [0, 0, 1, 2, 3, MISSING]
};

export const SEGMENT_OPTIONALS = [
  'triplegStartRows',
  'triplegEndRows',
  'rowTriplegIds',
  'tripStartRows',
  'tripFinishRows',
  'originStaypoints',
  'destinationStaypoints',
  'triplegTripIds',
  'trips-only',
  'none'
] as const;

export function getOracleSegmentsProps(optional: (typeof SEGMENT_OPTIONALS)[number]) {
  const props = getSegmentsProps();
  const withTrips = getOracleSegmentOutputNames(optional).includes('tripIds');
  props.output = {
    triplegs: {
      ids: 'triplegIds',
      status: {count: 'triplegCount', overflow: 'triplegOverflow'},
      ...(optional === 'triplegStartRows' ? {startRows: optional} : {}),
      ...(optional === 'triplegEndRows' ? {endRows: optional} : {}),
      ...(optional === 'rowTriplegIds' ? {rowTriplegIds: optional} : {})
    },
    ...(withTrips
      ? {
          trips: {
            ids: 'tripIds',
            status: {
              count: 'tripCount',
              overflow: 'tripOverflow',
              candidateOverflow: 'tripCandidateOverflow'
            },
            ...(optional === 'tripStartRows' ? {startRows: optional} : {}),
            ...(optional === 'tripFinishRows' ? {finishRows: optional} : {}),
            ...(optional === 'originStaypoints' ? {originStaypoints: optional} : {}),
            ...(optional === 'destinationStaypoints' ? {destinationStaypoints: optional} : {}),
            ...(optional === 'triplegTripIds' ? {triplegTripIds: optional} : {})
          }
        }
      : {})
  };
  return props;
}

/** Readback keys are a test-case contract, independent of the operation descriptor. */
export function getOracleSegmentOutputNames(
  optional: (typeof SEGMENT_OPTIONALS)[number]
): string[] {
  const triplegFields = ['triplegStartRows', 'triplegEndRows', 'rowTriplegIds'];
  const tripFields = [
    'tripStartRows',
    'tripFinishRows',
    'originStaypoints',
    'destinationStaypoints',
    'triplegTripIds'
  ];
  const withTrips = optional === 'trips-only' || tripFields.includes(optional);
  return [
    'triplegIds',
    'triplegCount',
    'triplegOverflow',
    ...(triplegFields.includes(optional) ? [optional] : []),
    ...(withTrips
      ? [
          'tripIds',
          'tripCount',
          'tripOverflow',
          'tripCandidateOverflow',
          ...(tripFields.includes(optional) ? [optional] : [])
        ]
      : [])
  ];
}

export function getOracleSegmentsInputs() {
  const fixture = getSegmentsFixture();
  return {
    positions: {format: 'float32x2' as const, values: fixture.positions},
    timestamps: {format: 'float32' as const, values: fixture.timestamps},
    trackOffsets: {format: 'uint32' as const, values: fixture.trackOffsets},
    staypointIds: {format: 'uint32' as const, values: fixture.staypointIds},
    parameters: {format: 'float32' as const, values: fixture.parameters},
    stayTracks: {format: 'uint32' as const, values: fixture.staypoints.tracks},
    stayStartRows: {format: 'uint32' as const, values: fixture.staypoints.startRows},
    stayFinishRows: {format: 'uint32' as const, values: fixture.staypoints.finishRows},
    stayDurations: {format: 'float32' as const, values: fixture.staypoints.durations},
    stayCount: {format: 'uint32' as const, values: fixture.staypoints.count}
  };
}
