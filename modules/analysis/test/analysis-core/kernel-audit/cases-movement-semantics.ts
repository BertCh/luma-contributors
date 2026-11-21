// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUMobilityIndicators,
  GPUModalSplit,
  GPUTrajectoryHampel,
  GPUTrajectoryLocations,
  GPUTrajectoryOutlierFlags,
  GPUTrajectorySegments,
  GPUTrajectorySplit,
  GPUTrajectoryStaypoints,
  GPUTrajectoryThinning,
  GPUTrajectoryTrackRelativeTimes
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

const SPATIAL_CONTEXT = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;

function trajectoryFields({view}: KernelAuditContext, rows: number, tracks: number) {
  return {
    positions: view('float32x2', rows),
    timestamps: view('uint32x2', rows),
    trackOffsets: view('uint32', tracks + 1)
  };
}

function addStaypoints(context: KernelAuditContext, rows: number): void {
  const {graph, view} = context;
  const capacity = Math.max(1, rows >> 3);
  graph.add(
    context.construct(GPUTrajectoryStaypoints, {
      ...trajectoryFields(context, rows, 1024),
      spatialContext: SPATIAL_CONTEXT,
      parameters: view('float32', 4),
      maximumSearchRows: 4096,
      output: {
        ids: view('uint32', capacity),
        status: {
          count: view('uint32', 1),
          overflow: view('uint32', 1),
          requiredCount: view('uint32', 1),
          candidateOverflow: view('uint32', 1)
        },
        drawInstanceCount: view('uint32', 1),
        startRows: view('uint32', capacity),
        endRows: view('uint32', capacity),
        finishRows: view('uint32', capacity),
        centroids: view('float32x2', capacity),
        durations: view('float32', capacity),
        staypointIds: view('uint32', rows)
      }
    })
  );
}

function addLocations(context: KernelAuditContext, capacity: number): void {
  const {graph, view} = context;
  graph.add(
    context.construct(GPUTrajectoryLocations, {
      spatialContext: SPATIAL_CONTEXT,
      centroids: view('float32x2', capacity),
      tracks: view('uint32', capacity),
      count: view('uint32', 1),
      parameters: view('float32', 4),
      maximumIterations: 32,
      output: {
        locationIds: view('uint32', capacity),
        status: {
          count: view('uint32', 1),
          converged: view('uint32', 1),
          iterationLimitReached: view('uint32', 1)
        },
        centers: view('float32x2', capacity)
      }
    })
  );
}

function addSegments(context: KernelAuditContext, rows: number): void {
  const {graph, view} = context;
  const tracks = 1024;
  const capacity = Math.max(1, rows >> 3);
  graph.add(
    context.construct(GPUTrajectorySegments, {
      ...trajectoryFields(context, rows, tracks),
      staypointIds: view('uint32', rows),
      staypoints: {
        tracks: view('uint32', capacity),
        startRows: view('uint32', capacity),
        finishRows: view('uint32', capacity),
        durations: view('float32', capacity),
        status: {count: view('uint32', 1)}
      },
      parameters: view('float32', 4),
      output: {
        triplegs: {
          ids: view('uint32', capacity),
          status: {
            count: view('uint32', 1),
            overflow: view('uint32', 1),
            requiredCount: view('uint32', 1)
          },
          startRows: view('uint32', capacity),
          endRows: view('uint32', capacity),
          rowTriplegIds: view('uint32', rows)
        },
        trips: {
          ids: view('uint32', capacity),
          status: {
            count: view('uint32', 1),
            overflow: view('uint32', 1),
            candidateOverflow: view('uint32', 1)
          },
          startRows: view('uint32', capacity),
          finishRows: view('uint32', capacity),
          originStaypoints: view('uint32', capacity),
          destinationStaypoints: view('uint32', capacity),
          triplegTripIds: view('uint32', capacity)
        }
      }
    })
  );
}

function addIndicators({graph, view, construct}: KernelAuditContext, capacity: number): void {
  const trackCount = 1024;
  graph.add(
    construct(GPUMobilityIndicators, {
      spatialContext: SPATIAL_CONTEXT,
      trackCount,
      timestamps: view('uint32x2', capacity),
      staypoints: {
        tracks: view('uint32', capacity),
        centroids: view('float32x2', capacity),
        startRows: view('uint32', capacity),
        durations: view('float32', capacity),
        locationIds: view('uint32', capacity),
        status: {count: view('uint32', 1)}
      },
      triplegs: {
        tracks: view('uint32', capacity),
        startRows: view('uint32', capacity),
        endRows: view('uint32', capacity),
        status: {count: view('uint32', 1)},
        staypointFinishRows: view('uint32', capacity)
      },
      output: {
        radiusOfGyration: view('float32', trackCount),
        jumpLengths: view('float32', capacity),
        waitingTimes: view('float32', capacity),
        randomEntropy: view('float32', trackCount),
        uncorrelatedEntropy: view('float32', trackCount),
        realEntropy: view('float32', trackCount),
        homeLocations: view('uint32', trackCount),
        workLocations: view('uint32', trackCount),
        trackingQuality: view('float32', trackCount),
        status: {
          candidateOverflow: view('uint32', 1),
          excludedLocationCount: view('uint32', 1)
        }
      }
    })
  );
}

function addModalSplit(
  context: KernelAuditContext,
  trackCount: number,
  metric: 'count' | 'distance' | 'duration'
): void {
  const {graph, view} = context;
  const rows = 1 << 20;
  const capacity = 1 << 16;
  const modeCount = 4;
  graph.add(
    context.construct(GPUModalSplit, {
      ...trajectoryFields(context, rows, trackCount),
      spatialContext: SPATIAL_CONTEXT,
      triplegs: {
        tracks: view('uint32', capacity),
        startRows: view('uint32', capacity),
        endRows: view('uint32', capacity),
        status: {count: view('uint32', 1)}
      },
      modeSpeedUpperBoundsFloat32: [15 / 3.6, 100 / 3.6, 1000, Infinity],
      modeCount,
      metric,
      output: {
        modalSplit: view('float32', trackCount * modeCount),
        predictedModes: view('uint32', capacity)
      }
    })
  );
}

function addOutlierFlags(context: KernelAuditContext, rows: number): void {
  const {graph, view} = context;
  graph.add(
    context.construct(GPUTrajectoryOutlierFlags, {
      ...trajectoryFields(context, rows, 1024),
      spatialContext: SPATIAL_CONTEXT,
      parameters: view('float32', 4),
      output: {flags: view('uint32', rows), trackFlaggedCounts: view('uint32', 1024)}
    })
  );
}

function addHampel({graph, view, construct}: KernelAuditContext, rows: number): void {
  graph.add(
    construct(GPUTrajectoryHampel, {
      values: view('float32', rows),
      trackOffsets: view('uint32', 1025),
      windowSize: 129,
      parameters: view('float32', 4),
      output: {
        filtered: view('float32', rows),
        flags: view('uint32', rows),
        outlierCount: view('uint32', 1)
      }
    })
  );
}

function addSplit(context: KernelAuditContext, rows: number): void {
  const {graph, view} = context;
  const capacity = Math.max(1, rows >> 3);
  graph.add(
    context.construct(GPUTrajectorySplit, {
      ...trajectoryFields(context, rows, 1024),
      mode: 'value-change',
      values: view('uint32', rows),
      output: {
        segmentOffsets: view('uint32', capacity + 1),
        segmentTracks: view('uint32', capacity),
        rowIds: view('uint32', capacity),
        status: {
          count: view('uint32', 1),
          overflow: view('uint32', 1),
          requiredCount: view('uint32', 1)
        },
        positions: view('float32x2', capacity),
        timestamps: view('uint32x2', capacity)
      }
    })
  );
}

function addThinning(context: KernelAuditContext, rows: number): void {
  const {graph, view} = context;
  const capacity = Math.max(1, rows >> 3);
  graph.add(
    context.construct(GPUTrajectoryThinning, {
      ...trajectoryFields(context, rows, 1024),
      mode: 'minimum-distance',
      spatialContext: SPATIAL_CONTEXT,
      parameters: view('float32', 4),
      maximumSearchRows: 4096,
      output: {
        sourceRows: view('uint32', capacity),
        trackOffsets: view('uint32', 1025),
        status: {
          count: view('uint32', 1),
          overflow: view('uint32', 1),
          candidateOverflow: view('uint32', 1)
        },
        sourceSelectionMask: view('uint32', rows),
        positions: view('float32x2', capacity),
        timestamps: view('uint32x2', capacity)
      }
    })
  );
}

function addTrackRelativeTimes({graph, view, construct}: KernelAuditContext, rows: number): void {
  graph.add(
    construct(GPUTrajectoryTrackRelativeTimes, {
      timestamps: view('uint32x2', rows),
      trackOffsets: view('uint32', 1025),
      output: view('float32', rows)
    })
  );
}

/** Cases of the staypoint, mobility and cleaning contributors (MOB-01, MOB-02, MOB-03). */
export const MOVEMENT_SEMANTICS_AUDIT_CASES: readonly KernelAuditCase[] = [
  {
    exportName: 'GPUTrajectoryStaypoints',
    name: 'great-circle words, all outputs, binding-limited rows',
    build: context => addStaypoints(context, context.maximumRows(8))
  },
  {
    exportName: 'GPUTrajectoryStaypoints',
    name: 'great-circle words, one row above the binding limit',
    build: context => addStaypoints(context, context.maximumRows(8) + 1)
  },
  {
    exportName: 'GPUTrajectoryLocations',
    name: 'great-circle, all outputs, binding-limited staypoints',
    build: context => addLocations(context, context.maximumRows(8))
  },
  {
    exportName: 'GPUTrajectoryLocations',
    name: 'one staypoint above the binding limit',
    build: context => addLocations(context, context.maximumRows(8) + 1)
  },
  {
    exportName: 'GPUTrajectorySegments',
    name: 'triplegs and trips, words, binding-limited rows',
    build: context => addSegments(context, context.maximumRows(8))
  },
  {
    exportName: 'GPUTrajectorySegments',
    name: 'triplegs and trips, one row above the binding limit',
    build: context => addSegments(context, context.maximumRows(8) + 1)
  },
  {
    exportName: 'GPUMobilityIndicators',
    name: 'all indicators, binding-limited staypoints',
    build: context => addIndicators(context, context.maximumRows(8))
  },
  {
    exportName: 'GPUMobilityIndicators',
    name: 'all indicators, one staypoint above the binding limit',
    build: context => addIndicators(context, context.maximumRows(8) + 1)
  },
  {
    exportName: 'GPUModalSplit',
    name: 'distance metric, predicted modes',
    build: context => addModalSplit(context, 4096, 'distance')
  },
  {
    exportName: 'GPUModalSplit',
    name: 'duration metric, binding-limited table',
    build: context => addModalSplit(context, context.maximumRows(16), 'duration')
  },
  {
    exportName: 'GPUModalSplit',
    name: 'count metric, one track above the binding limit',
    build: context => addModalSplit(context, context.maximumRows(16) + 1, 'count')
  },
  {
    exportName: 'GPUTrajectoryOutlierFlags',
    name: 'binding-limited rows',
    build: context => addOutlierFlags(context, context.maximumRows(8))
  },
  {
    exportName: 'GPUTrajectoryOutlierFlags',
    name: 'one row above the binding limit',
    build: context => addOutlierFlags(context, context.maximumRows(8) + 1)
  },
  {
    exportName: 'GPUTrajectoryHampel',
    name: 'largest window, binding-limited rows',
    build: context => addHampel(context, context.maximumRows(4))
  },
  {
    exportName: 'GPUTrajectoryHampel',
    name: 'one row above the binding limit',
    build: context => addHampel(context, context.maximumRows(4) + 1)
  },
  {
    exportName: 'GPUTrajectorySplit',
    name: 'value change, all outputs, binding-limited rows',
    build: context => addSplit(context, context.maximumRows(8))
  },
  {
    exportName: 'GPUTrajectorySplit',
    name: 'one row above the binding limit',
    build: context => addSplit(context, context.maximumRows(8) + 1)
  },
  {
    exportName: 'GPUTrajectoryThinning',
    name: 'minimum distance, all outputs, binding-limited rows',
    build: context => addThinning(context, context.maximumRows(8))
  },
  {
    exportName: 'GPUTrajectoryThinning',
    name: 'one row above the binding limit',
    build: context => addThinning(context, context.maximumRows(8) + 1)
  },
  {
    exportName: 'GPUTrajectoryTrackRelativeTimes',
    name: 'binding-limited rows',
    build: context => addTrackRelativeTimes(context, context.maximumRows(8))
  },
  {
    exportName: 'GPUTrajectoryTrackRelativeTimes',
    name: 'one row above the binding limit',
    build: context => addTrackRelativeTimes(context, context.maximumRows(8) + 1)
  }
];
