// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {TrajectorySegmentsOperationProps} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-segments-operation';
import {getGPUTrajectorySegmentsParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-segments';

/** Hand-built event timeline: two populated tracks, consecutive activities and two gaps. */
export function getSegmentsFixture() {
  const positions = Float32Array.from({length: 44}, (_, component) =>
    component % 2 === 0 ? component / 2 : 0
  );
  positions[38] = positions[36]; // Rows 18 and 19 are stationary and must be dropped.
  return {
    positions,
    timestamps: new Float32Array([
      ...Array.from({length: 16}, (_, row) => row),
      0,
      1,
      10,
      11,
      20,
      21
    ]),
    trackOffsets: new Uint32Array([0, 16, 22, 22]),
    staypointIds: new Uint32Array([
      0,
      0,
      0xffffffff,
      0xffffffff,
      1,
      1,
      0xffffffff,
      0xffffffff,
      2,
      2,
      3,
      3,
      0xffffffff,
      0xffffffff,
      4,
      4,
      ...new Array(6).fill(0xffffffff)
    ]),
    parameters: getGPUTrajectorySegmentsParameterValues({
      maximumGap: 3,
      activityMinimumDuration: 2
    }),
    staypoints: {
      tracks: new Uint32Array(5),
      startRows: new Uint32Array([0, 4, 8, 10, 14]),
      finishRows: new Uint32Array([1, 5, 9, 11, 15]),
      durations: new Float32Array([10, 2, 10, 10, 2]),
      count: new Uint32Array([5])
    }
  };
}

/** Requests every output port; unused capacity is deliberate. */
export function getSegmentsProps(
  triplegCapacity = 6,
  tripCapacity = 5
): TrajectorySegmentsOperationProps {
  return {
    id: 'segments-test',
    positions: 'positions',
    timestamps: 'timestamps',
    trackOffsets: 'trackOffsets',
    staypointIds: 'staypointIds',
    parameters: 'parameters',
    rowCount: 22,
    triplegCapacity,
    tripCapacity,
    staypoints: {
      tracks: 'stayTracks',
      startRows: 'stayStartRows',
      finishRows: 'stayFinishRows',
      durations: 'stayDurations',
      status: {count: 'stayCount'}
    },
    output: {
      triplegs: {
        ids: 'triplegIds',
        startRows: 'triplegStartRows',
        endRows: 'triplegEndRows',
        rowTriplegIds: 'rowTriplegIds',
        status: {
          count: 'triplegCount',
          requiredCount: 'triplegRequiredCount',
          overflow: 'triplegOverflow'
        }
      },
      trips: {
        ids: 'tripIds',
        startRows: 'tripStartRows',
        finishRows: 'tripFinishRows',
        originStaypoints: 'originStaypoints',
        destinationStaypoints: 'destinationStaypoints',
        triplegTripIds: 'triplegTripIds',
        status: {count: 'tripCount', requiredCount: 'tripRequiredCount', overflow: 'tripOverflow'}
      }
    }
  };
}
