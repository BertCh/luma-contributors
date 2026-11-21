// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import type {GPUTrajectoryMetricsOutput} from '../../../src/gpu-spatial-analysis/trajectory-analysis/gpu-trajectory-metrics';
import type {GPUTrajectoryPlayheadOutput} from '../../../src/gpu-spatial-analysis/trajectory-interpolation/gpu-trajectory-playhead';
import type {GPUMobilityIndicatorsOutput} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-mobility-indicators';

import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {GPUTrajectoryThinning} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning';
import {GPUTrajectoryThinningOperation} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning-operation';
import {CPUTrajectoryThinningOperation} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-thinning-operation';
import {GPUTrajectoryStaypoints} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints';
import {GPUTrajectoryStaypointsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints-operation';
import {CPUTrajectoryStaypointsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation';
import {GPUTrajectoryLocations} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations';
import {GPUTrajectoryLocationsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations-operation';
import {CPUTrajectoryLocationsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation';

function checkOutputTypes(
  float: GraphDataView<'float32'>,
  integer: GraphDataView<'uint32'>,
  position: GraphDataView<'float32x2'>
) {
  const compact = {ids: integer, status: {count: integer, overflow: integer}};
  const metrics: GPUTrajectoryMetricsOutput[] = [
    {trackLengths: float},
    {trackDurations: float},
    {averageSpeeds: float},
    {maximumSpeeds: float},
    {stepSpeeds: float},
    {stepHeadings: float},
    {stepAccelerations: float},
    {trackStopCounts: integer},
    {stops: compact}
  ];
  const playhead: GPUTrajectoryPlayheadOutput[] = [
    {currentPositions: position},
    {currentElevations: float},
    {headings: float},
    {speeds: float},
    {trackStatus: integer},
    {segmentRows: integer},
    {segmentFractions: float},
    {activeTracks: compact}
  ];
  const mobility: GPUMobilityIndicatorsOutput[] = [
    {radiusOfGyration: float},
    {jumpLengths: float},
    {waitingTimes: float},
    {randomEntropy: float},
    {uncorrelatedEntropy: float},
    {realEntropy: float},
    {homeLocations: integer},
    {workLocations: integer},
    {trackingQuality: float}
  ];
  // @ts-expect-error An empty output requests no metric.
  const emptyMetrics: GPUTrajectoryMetricsOutput = {};
  // @ts-expect-error An empty nested selection requests no stop table.
  const emptyStops: GPUTrajectoryMetricsOutput = {stops: {}};
  // @ts-expect-error An empty output requests no playhead result.
  const emptyPlayhead: GPUTrajectoryPlayheadOutput = {};
  // @ts-expect-error A nested compact selection requires its actual destinations.
  const emptyActive: GPUTrajectoryPlayheadOutput = {activeTracks: {}};
  // @ts-expect-error An empty output requests no indicator.
  const emptyMobility: GPUMobilityIndicatorsOutput = {};
  // @ts-expect-error Status alone is supplementary, not an indicator.
  const statusOnly: GPUMobilityIndicatorsOutput = {status: {}};
  return [
    metrics,
    playhead,
    mobility,
    emptyMetrics,
    emptyStops,
    emptyPlayhead,
    emptyActive,
    emptyMobility,
    statusOnly
  ];
}
it('retains terminal output examples for the lead test typecheck', () => {
  expect(typeof checkOutputTypes).toBe('function');
});

it('requires a budget-loss destination when constructing each bounded trajectory form', () => {
  const context = createKernelAuditContext('required-trajectory-status');
  try {
    const {view} = context;
    const positions = view('float32x2', 1);
    const timestamps = view('float32', 1);
    const trackOffsets = view('uint32', 2);
    const parameters = view('float32', 4);
    const trajectory = {positions, timestamps, trackOffsets, parameters};
    const tableStatus = {count: view('uint32', 1), overflow: view('uint32', 1)};
    expect(
      () =>
        new GPUTrajectoryThinning({
          ...trajectory,
          mode: 'minimum-distance',
          output: {
            sourceRows: view('uint32', 1),
            trackOffsets: view('uint32', 2),
            // @ts-expect-error The search-loss destination is mandatory.
            status: tableStatus
          }
        })
    ).toThrow(
      /trajectory-thinning output\.status\.candidateOverflow is required; pass a one-row uint32 view/
    );
    expect(
      () =>
        new GPUTrajectoryStaypoints({
          ...trajectory,
          // @ts-expect-error The search-loss destination is mandatory.
          output: {ids: view('uint32', 1), status: tableStatus}
        })
    ).toThrow(
      /trajectory-staypoints output\.status\.candidateOverflow is required; pass a one-row uint32 view/
    );
    expect(
      () =>
        new GPUTrajectoryLocations({
          centroids: positions,
          tracks: view('uint32', 1),
          count: view('uint32', 1),
          parameters,
          output: {
            locationIds: view('uint32', 1),
            // @ts-expect-error The iteration-loss destination is mandatory.
            status: {count: view('uint32', 1)}
          }
        })
    ).toThrow(
      /trajectory-locations output\.status\.iterationLimitReached is required; pass a one-row uint32 view/
    );

    const trajectoryValues = {
      positions: 'p',
      timestamps: 't',
      trackOffsets: 'o',
      parameters: 'settings',
      rowCount: 1,
      trackCount: 1,
      capacity: 1
    };
    const status = {count: 'count', overflow: 'overflow'};
    for (const Operation of [GPUTrajectoryThinningOperation, CPUTrajectoryThinningOperation]) {
      expect(
        () =>
          new Operation({
            ...trajectoryValues,
            mode: 'minimum-distance',
            // @ts-expect-error Graph operations require a logical name for the loss value.
            output: {sourceRows: 'rows', trackOffsets: 'newOffsets', status}
          })
      ).toThrow(/trajectory-thinning output\.status\.candidateOverflow is required; pass the name/);
    }
    for (const Operation of [GPUTrajectoryStaypointsOperation, CPUTrajectoryStaypointsOperation]) {
      expect(
        () =>
          new Operation({
            ...trajectoryValues,
            // @ts-expect-error Graph operations require a logical name for the loss value.
            output: {ids: 'ids', status}
          })
      ).toThrow(
        /trajectory-staypoints output\.status\.candidateOverflow is required; pass the name/
      );
    }
    for (const Operation of [GPUTrajectoryLocationsOperation, CPUTrajectoryLocationsOperation]) {
      expect(
        () =>
          new Operation({
            centroids: 'p',
            tracks: 'tracks',
            count: 'stayCount',
            parameters: 'settings',
            capacity: 1,
            output: {
              locationIds: 'labels',
              // @ts-expect-error Graph operations require a logical name for the loss value.
              status: {count: 'count'}
            }
          })
      ).toThrow(
        /trajectory-locations output\.status\.iterationLimitReached is required; pass the name/
      );
    }
  } finally {
    context.device.destroy();
  }
});
