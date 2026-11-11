// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {MobilityIndicatorsOperationProps} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-mobility-indicators-operation';

/** Two populated tracks, unequal dwell durations, a home/work tie and untracked gaps. */
export function getMobilityFixture() {
  return {
    timestamps: new Float32Array([0, 2, 5, 8, 12, 13, 20, 0, 1, 4, 6, 10, 13]),
    staypoints: {
      tracks: new Uint32Array([0, 0, 0, 0, 1, 1, 1, 99]),
      centroids: new Float32Array([0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 3, 4, 0, 0, 99, 99]),
      startRows: new Uint32Array([0, 2, 4, 6, 7, 9, 11, 99]),
      durations: new Float32Array([2, 3, 1, 0, 40, 10, 5, 99]),
      locationIds: new Uint32Array([5, 2, 5, 2, 9, 9, 4, 99]),
      count: Uint32Array.of(7)
    },
    triplegs: {
      tracks: new Uint32Array([0, 0, 1, 99]),
      startRows: new Uint32Array([1, 5, 8, 99]),
      endRows: new Uint32Array([3, 7, 10, 99]),
      count: Uint32Array.of(3),
      staypointFinishRows: new Uint32Array([1, 3, 5, 6, 8, 10, 12, 99])
    }
  };
}

export function getMobilityProps(): MobilityIndicatorsOperationProps {
  return {
    id: 'mobility-test',
    trackCount: 3,
    rowCount: 13,
    timestamps: 'timestamps',
    staypointCapacity: 8,
    triplegCapacity: 4,
    staypoints: {
      tracks: 'stayTracks',
      centroids: 'centroids',
      startRows: 'stayStartRows',
      durations: 'durations',
      locationIds: 'locationIds',
      status: {count: 'stayCount'}
    },
    triplegs: {
      tracks: 'triplegTracks',
      startRows: 'triplegStartRows',
      endRows: 'triplegEndRows',
      status: {count: 'triplegCount'},
      staypointFinishRows: 'stayFinishRows'
    },
    output: {
      radiusOfGyration: 'radiusOfGyration',
      jumpLengths: 'jumpLengths',
      waitingTimes: 'waitingTimes',
      randomEntropy: 'randomEntropy',
      uncorrelatedEntropy: 'uncorrelatedEntropy',
      realEntropy: 'realEntropy',
      homeLocations: 'homeLocations',
      workLocations: 'workLocations',
      trackingQuality: 'trackingQuality',
      status: {candidateOverflow: 'candidateOverflow'}
    }
  };
}
