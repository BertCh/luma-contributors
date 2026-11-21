// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {addTrackQualityRecipe} from '@luma-contributors/analysis/gpu-network';
import {
  GPUTrajectoryKalman,
  GPUTrajectoryRestriction,
  prepareGPUTrajectoryObservations
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {addKernelPass} from '../../engine/mode-kernels';
import {createGraphBufferImporter} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {loadVesselTracks} from './b12-tracks';
import type {SceneContext, SceneInstance} from '../scene';

export type HarborTrackQualityOptions = {
  maximumSpeedKnots: number;
  maximumGapSeconds: number;
  maximumModelGapSeconds: number;
  processNoise: number;
  measurementVariance: number;
  covarianceModel: 'isotropic' | 'full';
  axisRatio: number;
  axisCorrelation: number;
  showUncertainty: boolean;
  windowEndHour: number;
  display: 'observed' | 'filtered' | 'smoothed' | 'gaps' | 'zone';
};

const TRACK_MMSI = 366952890;
const METERS_PER_KNOT_SECOND = 0.514444;

export async function createHarborTrackQuality(
  ctx: SceneContext<HarborTrackQualityOptions>
): Promise<SceneInstance<HarborTrackQualityOptions>> {
  const dataset = ctx.datasets.get('ais-vessels');
  const vesselSet = loadVesselTracks(dataset);
  const trackIndex = Array.from(vesselSet.mmsi).indexOf(TRACK_MMSI);
  const first = vesselSet.offsets[trackIndex];
  const last = vesselSet.offsets[trackIndex + 1];
  const observationCount = last - first;
  const polygonIntervalsPerSegment = 2;
  const polygonIntervalCapacity = observationCount * polygonIntervalsPerSegment;
  const positions = vesselSet.positions.slice(first * 2, last * 2);
  const sourceSeconds = dataset.column<Uint32Array>('timestamp').slice(first, last);
  const originTime = sourceSeconds[0];
  const offsets = Uint32Array.of(0, observationCount);
  const sourceLngLat = vesselSet.lngLat.slice(first * 2, last * 2);
  const dayEpochSeconds = BigInt(Date.UTC(2024, 5, 12) / 1000);
  const absoluteSeconds = Array.from(sourceSeconds, value => dayEpochSeconds + BigInt(value));
  const preparedObservations = prepareGPUTrajectoryObservations({
    longitudeLatitude: sourceLngLat,
    timestamps: absoluteSeconds,
    epoch: absoluteSeconds[0],
    unitsPerSecond: 1n,
    origin: vesselSet.origin
  });
  // Use the explicit epoch preparation for GPU time inputs. The map and estimators retain
  // their existing caller-owned AEQD coordinates, rather than mixing in the returned ENU frame.
  const times = preparedObservations.times;
  const naiveAbsoluteTimes = Float32Array.from(absoluteSeconds, Number);
  const maximumNaiveTimeError = Math.max(
    ...absoluteSeconds.map((value, index) => Math.abs(naiveAbsoluteTimes[index] - Number(value)))
  );
  const observedSegments = new Float32Array((observationCount - 1) * 4);
  const polygonBoundarySegments = new Float32Array(8 * 4);
  for (let index = 0; index < observationCount - 1; index++) {
    observedSegments.set(positions.subarray(index * 2, index * 2 + 2), index * 4);
    observedSegments.set(positions.subarray((index + 1) * 2, (index + 1) * 2 + 2), index * 4 + 2);
  }
  let zoneMinX = Infinity;
  let zoneMinY = Infinity;
  let zoneMaxX = -Infinity;
  let zoneMaxY = -Infinity;
  for (let row = 0; row < observationCount; row++) {
    zoneMinX = Math.min(zoneMinX, positions[row * 2]);
    zoneMinY = Math.min(zoneMinY, positions[row * 2 + 1]);
    zoneMaxX = Math.max(zoneMaxX, positions[row * 2]);
    zoneMaxY = Math.max(zoneMaxY, positions[row * 2 + 1]);
  }
  const zoneWidth = zoneMaxX - zoneMinX;
  const zoneHeight = zoneMaxY - zoneMinY;
  const zoneVertices = Float32Array.of(
    zoneMinX - 50,
    zoneMinY - 50,
    zoneMaxX + 50,
    zoneMinY - 50,
    zoneMaxX + 50,
    zoneMaxY + 50,
    zoneMinX - 50,
    zoneMaxY + 50,
    zoneMinX + zoneWidth * 0.38,
    zoneMinY + zoneHeight * 0.38,
    zoneMinX + zoneWidth * 0.62,
    zoneMinY + zoneHeight * 0.38,
    zoneMinX + zoneWidth * 0.62,
    zoneMinY + zoneHeight * 0.62,
    zoneMinX + zoneWidth * 0.38,
    zoneMinY + zoneHeight * 0.62
  );
  const zoneRings = [zoneVertices.subarray(0, 8), zoneVertices.subarray(8)];
  let boundarySegment = 0;
  for (const ring of zoneRings) {
    for (let vertex = 0; vertex < ring.length / 2; vertex++) {
      const next = (vertex + 1) % (ring.length / 2);
      polygonBoundarySegments.set(
        [ring[vertex * 2], ring[vertex * 2 + 1], ring[next * 2], ring[next * 2 + 1]],
        boundarySegment++ * 4
      );
    }
  }
  const resources = new SpatialAnalysisResources(ctx.device, 'harbor-track-quality');
  const graph = new GPUCommandGraph<void>(ctx.device, {id: 'harbor-track-quality'});
  const view = createGraphBufferImporter(graph);
  const positionsBuffer = resources.createBuffer('ais-observations', positions);
  const timeBuffer = resources.createBuffer('relative-times', times);
  const offsetsBuffer = resources.createBuffer('track-offsets', offsets);
  const thresholdsBuffer = resources.createParameterBuffer('raw-quality-thresholds', 'float32', 2);
  const modelParametersBuffer = resources.createParameterBuffer('kalman-model', 'float32', 4);
  const fullModelParametersBuffer = resources.createParameterBuffer(
    'full-kalman-model',
    'float32',
    10
  );
  const restrictionParametersBuffer = resources.createParameterBuffer(
    'time-gap-restriction',
    'float32',
    7
  );
  const restrictionParametersView = restrictionParametersBuffer.importToGraph(graph);
  const polygonRingVertexCounts = [4, 4] as const;
  const polygonParametersBuffer = resources.createParameterBuffer(
    'polygon-hole-restriction',
    'float32',
    7 + 2 * polygonRingVertexCounts.reduce((sum, count) => sum + count, 0)
  );
  const polygonParametersView = polygonParametersBuffer.importToGraph(graph);
  const observedBuffer = resources.createBuffer(
    'observed-segments',
    observedSegments,
    Buffer.VERTEX
  );
  const windowedObservedBuffer = resources.createBuffer(
    'windowed-observed-segments',
    (observationCount - 1) * 16,
    Buffer.VERTEX
  );
  const windowedObservationPointsBuffer = resources.createBuffer(
    'windowed-observation-points',
    observationCount * 8,
    Buffer.VERTEX
  );
  const filteredSegmentsBuffer = resources.createBuffer(
    'filtered-segments',
    (observationCount - 1) * 16,
    Buffer.VERTEX
  );
  const smoothedSegmentsBuffer = resources.createBuffer(
    'smoothed-segments',
    (observationCount - 1) * 16,
    Buffer.VERTEX
  );
  const restrictionSegmentsBuffer = resources.createBuffer(
    'supported-segments',
    observationCount * 16,
    Buffer.VERTEX
  );
  const filteredStateBuffer = resources.createBuffer('filtered-state', observationCount * 16);
  const filteredCovarianceBuffer = resources.createBuffer(
    'filtered-covariance',
    observationCount * 16
  );
  const smoothedStateBuffer = resources.createBuffer('smoothed-state', observationCount * 16);
  const smoothedCovarianceBuffer = resources.createBuffer(
    'smoothed-covariance',
    observationCount * 16
  );
  const kalmanStatusBuffer = resources.createBuffer('kalman-status', observationCount * 4);
  const fullFilteredStateBuffer = resources.createBuffer(
    'full-filtered-state',
    observationCount * 16
  );
  const fullSmoothedStateBuffer = resources.createBuffer(
    'full-smoothed-state',
    observationCount * 16
  );
  const fullFilteredCovarianceBuffer = resources.createBuffer(
    'full-filtered-covariance',
    observationCount * 64
  );
  const fullSmoothedCovarianceBuffer = resources.createBuffer(
    'full-smoothed-covariance',
    observationCount * 64
  );
  const fullKalmanStatusBuffer = resources.createBuffer('full-kalman-status', observationCount * 4);
  const fullFilteredSegmentsBuffer = resources.createBuffer(
    'full-filtered-segments',
    (observationCount - 1) * 16,
    Buffer.VERTEX
  );
  const fullSmoothedSegmentsBuffer = resources.createBuffer(
    'full-smoothed-segments',
    (observationCount - 1) * 16,
    Buffer.VERTEX
  );
  const uncertaintyEllipseSegmentCount = 24;
  const isoFilteredEllipsesBuffer = resources.createBuffer(
    'isotropic-filtered-ellipses',
    observationCount * uncertaintyEllipseSegmentCount * 16,
    Buffer.VERTEX
  );
  const isoSmoothedEllipsesBuffer = resources.createBuffer(
    'isotropic-smoothed-ellipses',
    observationCount * uncertaintyEllipseSegmentCount * 16,
    Buffer.VERTEX
  );
  const fullFilteredEllipsesBuffer = resources.createBuffer(
    'full-filtered-ellipses',
    observationCount * uncertaintyEllipseSegmentCount * 16,
    Buffer.VERTEX
  );
  const fullSmoothedEllipsesBuffer = resources.createBuffer(
    'full-smoothed-ellipses',
    observationCount * uncertaintyEllipseSegmentCount * 16,
    Buffer.VERTEX
  );
  const restrictionIntervalsBuffer = resources.createBuffer(
    'restriction-intervals',
    observationCount * 8
  );
  const restrictionStatusBuffer = resources.createBuffer(
    'restriction-status',
    observationCount * 4
  );
  const relationshipsBuffer = resources.createBuffer('track-relationships', 16);
  const polygonIntervalsBuffer = resources.createBuffer(
    'polygon-hole-intervals',
    polygonIntervalCapacity * 8
  );
  const polygonPositionsBuffer = resources.createBuffer(
    'polygon-hole-clipped-positions',
    polygonIntervalCapacity * 16
  );
  const polygonStatusBuffer = resources.createBuffer(
    'polygon-hole-status',
    polygonIntervalCapacity * 4
  );
  const polygonRelationshipsBuffer = resources.createBuffer('polygon-hole-relationships', 16);
  const compactIntervalsBuffer = resources.createBuffer(
    'polygon-hole-merged-intervals',
    polygonIntervalCapacity * 8
  );
  const compactPositionsBuffer = resources.createBuffer(
    'polygon-hole-merged-positions',
    polygonIntervalCapacity * 16,
    Buffer.VERTEX
  );
  const compactSourcesBuffer = resources.createBuffer(
    'polygon-hole-merged-sources',
    polygonIntervalCapacity * 8
  );
  const compactStatusBuffer = resources.createBuffer(
    'polygon-hole-merged-status',
    polygonIntervalCapacity * 4
  );
  const compactSummaryBuffer = resources.createBuffer('polygon-hole-merge-summary', 12);
  const polygonBoundaryBuffer = resources.createBuffer(
    'polygon-hole-boundaries',
    polygonBoundarySegments,
    Buffer.VERTEX
  );
  polygonBoundaryBuffer.write(polygonBoundarySegments);
  const flagsBuffer = resources.createBuffer('quality-flags', observationCount * 4);
  const speedsBuffer = resources.createBuffer('step-speeds', observationCount * 4);
  const lengthsBuffer = resources.createBuffer('track-length', 4);
  const durationsBuffer = resources.createBuffer('track-duration', 4);
  const maximumSpeedsBuffer = resources.createBuffer('maximum-speed', 4);
  const summaryBuffer = resources.createBuffer('summary', 8);
  const trajectory = {
    positions: view(positionsBuffer, 'float32x2', observationCount),
    trackOffsets: view(offsetsBuffer, 'uint32', 2),
    times: view(timeBuffer, 'float32', observationCount),
    spatialContext: vesselSet.spatialContext
  };
  const metricTrajectory = {
    positions: trajectory.positions,
    trackOffsets: trajectory.trackOffsets,
    timestamps: trajectory.times,
    spatialContext: trajectory.spatialContext
  };
  const recipe = addTrackQualityRecipe(graph, {
    id: 'harbor-raw-quality',
    trajectory: metricTrajectory,
    thresholds: thresholdsBuffer.importToGraph(graph)
  });
  graph.add(
    new GPUTrajectoryKalman({
      id: 'harbor-kalman-filter-smoother',
      positions: trajectory.positions,
      offsets: trajectory.trackOffsets,
      times: trajectory.times,
      parameters: modelParametersBuffer.importToGraph(graph),
      output: {
        filteredState: view(filteredStateBuffer, 'float32x4', observationCount),
        filteredCovariance: view(filteredCovarianceBuffer, 'float32x4', observationCount),
        smoothedState: view(smoothedStateBuffer, 'float32x4', observationCount),
        smoothedCovariance: view(smoothedCovarianceBuffer, 'float32x4', observationCount),
        status: view(kalmanStatusBuffer, 'uint32', observationCount)
      }
    })
  );
  graph.add(
    new GPUTrajectoryRestriction({
      id: 'hypothetical-ais-study-polygon-with-hole',
      positions: trajectory.positions,
      times: trajectory.times,
      offsets: trajectory.trackOffsets,
      parameters: polygonParametersView,
      polygonRingVertexCounts,
      geometryBoundary: 'closed',
      timeBoundary: 'closed',
      intervalsPerSegment: polygonIntervalsPerSegment,
      mergeIntervals: true,
      output: {
        intervals: view(polygonIntervalsBuffer, 'float32x2', polygonIntervalCapacity),
        positions: view(polygonPositionsBuffer, 'float32x4', polygonIntervalCapacity),
        status: view(polygonStatusBuffer, 'uint32', polygonIntervalCapacity),
        relationships: view(polygonRelationshipsBuffer, 'uint32x4', 1),
        compactIntervals: view(compactIntervalsBuffer, 'float32x2', polygonIntervalCapacity),
        compactPositions: view(compactPositionsBuffer, 'float32x4', polygonIntervalCapacity),
        compactSources: view(compactSourcesBuffer, 'uint32x2', polygonIntervalCapacity),
        compactStatus: view(compactStatusBuffer, 'uint32', polygonIntervalCapacity),
        compactSummary: view(compactSummaryBuffer, 'uint32', 3)
      }
    })
  );
  graph.add(
    new GPUTrajectoryKalman({
      id: 'harbor-full-covariance-kalman-filter-smoother',
      positions: trajectory.positions,
      offsets: trajectory.trackOffsets,
      times: trajectory.times,
      covarianceMode: 'full',
      parameters: fullModelParametersBuffer.importToGraph(graph),
      output: {
        filteredState: view(fullFilteredStateBuffer, 'float32x4', observationCount),
        filteredCovariance: view(fullFilteredCovarianceBuffer, 'float32x4', observationCount * 4),
        smoothedState: view(fullSmoothedStateBuffer, 'float32x4', observationCount),
        smoothedCovariance: view(fullSmoothedCovarianceBuffer, 'float32x4', observationCount * 4),
        status: view(fullKalmanStatusBuffer, 'uint32', observationCount)
      }
    })
  );
  graph.add(
    new GPUTrajectoryRestriction({
      id: 'supported-ais-observation-intervals',
      positions: trajectory.positions,
      times: trajectory.times,
      offsets: trajectory.trackOffsets,
      parameters: restrictionParametersView,
      timeBoundary: 'closed',
      output: {
        intervals: view(restrictionIntervalsBuffer, 'float32x2', observationCount),
        positions: view(restrictionSegmentsBuffer, 'float32x4', observationCount),
        status: view(restrictionStatusBuffer, 'uint32', observationCount),
        relationships: view(relationshipsBuffer, 'uint32x4', 1)
      }
    })
  );
  const filteredState = view(filteredStateBuffer, 'float32x4', observationCount);
  const smoothedState = view(smoothedStateBuffer, 'float32x4', observationCount);
  const fullFilteredState = view(fullFilteredStateBuffer, 'float32x4', observationCount);
  const fullSmoothedState = view(fullSmoothedStateBuffer, 'float32x4', observationCount);
  const filteredSegments = view(filteredSegmentsBuffer, 'float32x4', observationCount - 1);
  const smoothedSegments = view(smoothedSegmentsBuffer, 'float32x4', observationCount - 1);
  const fullFilteredSegments = view(fullFilteredSegmentsBuffer, 'float32x4', observationCount - 1);
  const fullSmoothedSegments = view(fullSmoothedSegmentsBuffer, 'float32x4', observationCount - 1);
  for (const [id, state, output, status] of [
    ['pack-filtered-track', filteredState, filteredSegments, kalmanStatusBuffer],
    ['pack-smoothed-track', smoothedState, smoothedSegments, kalmanStatusBuffer],
    ['pack-full-filtered-track', fullFilteredState, fullFilteredSegments, fullKalmanStatusBuffer],
    ['pack-full-smoothed-track', fullSmoothedState, fullSmoothedSegments, fullKalmanStatusBuffer]
  ] as const) {
    addKernelPass(graph, {
      id,
      invocationCount: observationCount - 1,
      bindings: [
        {name: 'state', view: state, type: 'f32', access: 'read'},
        {
          name: 'status',
          view: view(status, 'uint32', observationCount),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'times',
          view: view(timeBuffer, 'float32', observationCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'restriction',
          view: restrictionParametersView,
          type: 'f32',
          access: 'read'
        },
        {name: 'segments', view: output, type: 'f32', access: 'read_write'}
      ],
      body: `let previous = index; let current = index + 1u; let gap = times[timesOffset + current] - times[timesOffset + previous];
let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
if ((status[statusOffset + previous] & 4u) != 0u || (status[statusOffset + current] & 5u) != 0u || gap > restriction[restrictionOffset + 6u] || times[timesOffset + current] > restriction[restrictionOffset + 1u]) { for (var component = 0u; component < 4u; component++) { segments[segmentsOffset + index * 4u + component] = invalidValue; } return; }
segments[segmentsOffset + index * 4u] = state[stateOffset + previous * 4u]; segments[segmentsOffset + index * 4u + 1u] = state[stateOffset + previous * 4u + 1u]; segments[segmentsOffset + index * 4u + 2u] = state[stateOffset + current * 4u]; segments[segmentsOffset + index * 4u + 3u] = state[stateOffset + current * 4u + 1u];`
    });
  }
  const isoFilteredCovariance = view(filteredCovarianceBuffer, 'float32x4', observationCount);
  const isoSmoothedCovariance = view(smoothedCovarianceBuffer, 'float32x4', observationCount);
  const fullFilteredCovariance = view(
    fullFilteredCovarianceBuffer,
    'float32x4',
    observationCount * 4
  );
  const fullSmoothedCovariance = view(
    fullSmoothedCovarianceBuffer,
    'float32x4',
    observationCount * 4
  );
  const kalmanStatus = view(kalmanStatusBuffer, 'uint32', observationCount);
  const fullKalmanStatus = view(fullKalmanStatusBuffer, 'uint32', observationCount);
  function addEllipsePass(
    id: string,
    state: typeof filteredState,
    covariance: typeof isoFilteredCovariance,
    status: typeof kalmanStatus,
    output: Buffer,
    full: boolean
  ): void {
    addKernelPass(graph, {
      id,
      invocationCount: observationCount * uncertaintyEllipseSegmentCount,
      bindings: [
        {name: 'state', view: state, type: 'f32', access: 'read'},
        {name: 'covariance', view: covariance, type: 'f32', access: 'read'},
        {name: 'status', view: status, type: 'u32', access: 'read'},
        {name: 'times', view: trajectory.times, type: 'f32', access: 'read'},
        {name: 'restriction', view: restrictionParametersView, type: 'f32', access: 'read'},
        {
          name: 'segments',
          view: view(output, 'float32x4', observationCount * uncertaintyEllipseSegmentCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      declarations: `const ELLIPSE_SEGMENT_COUNT: u32 = ${uncertaintyEllipseSegmentCount}u; const PI: f32 = 3.141592653589793; fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }`,
      body: `let observation = index / ELLIPSE_SEGMENT_COUNT; let segment = index % ELLIPSE_SEGMENT_COUNT; let stateIndex = observation * 4u; let covarianceIndex = observation * ${full ? 16 : 4}u;
let positionX = state[stateOffset + stateIndex]; let positionY = state[stateOffset + stateIndex + 1u]; let positionVarianceX = covariance[covarianceOffset + covarianceIndex]; let positionCovarianceXY = ${full ? '(covariance[covarianceOffset + covarianceIndex + 1u] + covariance[covarianceOffset + covarianceIndex + 4u]) * 0.5' : '0.0'}; let positionVarianceY = ${full ? 'covariance[covarianceOffset + covarianceIndex + 5u]' : 'positionVarianceX'}; let flags = status[statusOffset + observation];
let traceHalf = (positionVarianceX + positionVarianceY) * 0.5; let halfDifference = (positionVarianceX - positionVarianceY) * 0.5; let radius = sqrt(halfDifference * halfDifference + positionCovarianceXY * positionCovarianceXY); let major = sqrt(max(0.0, traceHalf + radius)); let minor = sqrt(max(0.0, traceHalf - radius)); let rotation = 0.5 * atan2(2.0 * positionCovarianceXY, positionVarianceX - positionVarianceY); let firstAngle = 2.0 * PI * f32(segment) / f32(ELLIPSE_SEGMENT_COUNT); let secondAngle = 2.0 * PI * f32(segment + 1u) / f32(ELLIPSE_SEGMENT_COUNT); let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
if (observation % 4u != 0u || (flags & 4u) != 0u || times[timesOffset + observation] > restriction[restrictionOffset + 1u] || !finite(positionX) || !finite(positionY) || !finite(positionVarianceX) || !finite(positionCovarianceXY) || !finite(positionVarianceY)) { for (var component = 0u; component < 4u; component++) { segments[segmentsOffset + index * 4u + component] = invalidValue; } return; }
let firstX = positionX + major * cos(firstAngle) * cos(rotation) - minor * sin(firstAngle) * sin(rotation); let firstY = positionY + major * cos(firstAngle) * sin(rotation) + minor * sin(firstAngle) * cos(rotation); let secondX = positionX + major * cos(secondAngle) * cos(rotation) - minor * sin(secondAngle) * sin(rotation); let secondY = positionY + major * cos(secondAngle) * sin(rotation) + minor * sin(secondAngle) * cos(rotation); segments[segmentsOffset + index * 4u] = firstX; segments[segmentsOffset + index * 4u + 1u] = firstY; segments[segmentsOffset + index * 4u + 2u] = secondX; segments[segmentsOffset + index * 4u + 3u] = secondY;`
    });
  }
  addEllipsePass(
    'pack-isotropic-filtered-ellipses',
    filteredState,
    isoFilteredCovariance,
    kalmanStatus,
    isoFilteredEllipsesBuffer,
    false
  );
  addEllipsePass(
    'pack-isotropic-smoothed-ellipses',
    smoothedState,
    isoSmoothedCovariance,
    kalmanStatus,
    isoSmoothedEllipsesBuffer,
    false
  );
  addEllipsePass(
    'pack-full-filtered-ellipses',
    fullFilteredState,
    fullFilteredCovariance,
    fullKalmanStatus,
    fullFilteredEllipsesBuffer,
    true
  );
  addEllipsePass(
    'pack-full-smoothed-ellipses',
    fullSmoothedState,
    fullSmoothedCovariance,
    fullKalmanStatus,
    fullSmoothedEllipsesBuffer,
    true
  );

  addKernelPass(graph, {
    id: 'clip-observed-track-window',
    invocationCount: observationCount - 1,
    bindings: [
      {name: 'positions', view: trajectory.positions, type: 'f32', access: 'read'},
      {name: 'times', view: trajectory.times, type: 'f32', access: 'read'},
      {
        name: 'restriction',
        view: restrictionParametersView,
        type: 'f32',
        access: 'read'
      },
      {
        name: 'segments',
        view: view(windowedObservedBuffer, 'float32x4', observationCount - 1),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: `let previous = index; let current = index + 1u;
let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
if (times[timesOffset + current] > restriction[restrictionOffset + 1u]) { for (var component = 0u; component < 4u; component++) { segments[segmentsOffset + index * 4u + component] = invalidValue; } return; }
segments[segmentsOffset + index * 4u] = positions[positionsOffset + previous * 2u]; segments[segmentsOffset + index * 4u + 1u] = positions[positionsOffset + previous * 2u + 1u]; segments[segmentsOffset + index * 4u + 2u] = positions[positionsOffset + current * 2u]; segments[segmentsOffset + index * 4u + 3u] = positions[positionsOffset + current * 2u + 1u];`
  });
  addKernelPass(graph, {
    id: 'clip-observation-points-to-window',
    invocationCount: observationCount,
    bindings: [
      {name: 'positions', view: trajectory.positions, type: 'f32', access: 'read'},
      {name: 'times', view: trajectory.times, type: 'f32', access: 'read'},
      {
        name: 'restriction',
        view: restrictionParametersView,
        type: 'f32',
        access: 'read'
      },
      {
        name: 'visiblePositions',
        view: view(windowedObservationPointsBuffer, 'float32x2', observationCount),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: `let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
if (times[timesOffset + index] > restriction[restrictionOffset + 1u]) { visiblePositions[visiblePositionsOffset + index * 2u] = invalidValue; visiblePositions[visiblePositionsOffset + index * 2u + 1u] = invalidValue; return; }
visiblePositions[visiblePositionsOffset + index * 2u] = positions[positionsOffset + index * 2u]; visiblePositions[visiblePositionsOffset + index * 2u + 1u] = positions[positionsOffset + index * 2u + 1u];`
  });
  const metricOutputs = recipe.outputs.metrics;
  const publishBindings = [
    {name: 'flags', view: recipe.outputs.flags, type: 'u32' as const, access: 'read' as const}
  ];
  addKernelPass(graph, {
    id: 'publish-quality-summary',
    invocationCount: 1,
    bindings: [
      ...publishBindings,
      {
        name: 'summary',
        view: view(summaryBuffer, 'uint32', 2),
        type: 'u32' as const,
        access: 'read_write' as const
      }
    ],
    body:
      'var flagged = 0u; var gaps = 0u; for (var row = 0u; row < ' +
      observationCount +
      'u; row++) { let flag = flags[flagsOffset + row]; flagged += select(0u, 1u, flag != 0u); gaps += select(0u, 1u, (flag & 8u) != 0u); } summary[summaryOffset] = flagged; summary[summaryOffset + 1u] = gaps;'
  });
  let generation = 1;
  let pendingGeneration = 0;
  let dirty = true;
  let destroyed = false;
  let summary: {
    flags: Uint32Array;
    speeds: Float32Array;
    filtered: Float32Array;
    smoothed: Float32Array;
    filteredCovariance: Float32Array;
    smoothedCovariance: Float32Array;
    kalmanStatus: Uint32Array;
    fullFiltered: Float32Array;
    fullSmoothed: Float32Array;
    fullFilteredCovariance: Float32Array;
    fullSmoothedCovariance: Float32Array;
    fullKalmanStatus: Uint32Array;
    flagged: number;
    gaps: number;
    length: number;
    duration: number;
    maximumSpeed: number;
  } | null = null;
  const reader = new SummaryReader(
    resources,
    'quality-summary',
    [
      {buffer: flagsBuffer, size: observationCount * 4},
      {buffer: speedsBuffer, size: observationCount * 4},
      {buffer: filteredStateBuffer, size: observationCount * 16},
      {buffer: smoothedStateBuffer, size: observationCount * 16},
      {buffer: filteredCovarianceBuffer, size: observationCount * 16},
      {buffer: smoothedCovarianceBuffer, size: observationCount * 16},
      {buffer: kalmanStatusBuffer, size: observationCount * 4},
      {buffer: fullFilteredStateBuffer, size: observationCount * 16},
      {buffer: fullSmoothedStateBuffer, size: observationCount * 16},
      {buffer: fullFilteredCovarianceBuffer, size: observationCount * 64},
      {buffer: fullSmoothedCovarianceBuffer, size: observationCount * 64},
      {buffer: fullKalmanStatusBuffer, size: observationCount * 4},
      {buffer: summaryBuffer, size: 8},
      {buffer: lengthsBuffer, size: 4},
      {buffer: durationsBuffer, size: 4},
      {buffer: maximumSpeedsBuffer, size: 4},
      {buffer: polygonStatusBuffer, size: polygonIntervalCapacity * 4},
      {buffer: compactSummaryBuffer, size: 12}
    ],
    bytes => {
      if (destroyed || pendingGeneration !== generation) return;
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      const summaryOffset = observationCount * 240;
      const metricOffset = summaryOffset + 8;
      const polygonStatusOffset = (metricOffset + 12) / 4;
      const compactSummaryOffset = polygonStatusOffset + polygonIntervalCapacity;
      summary = {
        flags: words.slice(0, observationCount),
        speeds: floats.slice(observationCount, observationCount * 2),
        filtered: floats.slice(observationCount * 2, observationCount * 2 + observationCount * 4),
        smoothed: floats.slice(observationCount * 6, observationCount * 6 + observationCount * 4),
        filteredCovariance: floats.slice(
          observationCount * 10,
          observationCount * 10 + observationCount * 4
        ),
        smoothedCovariance: floats.slice(
          observationCount * 14,
          observationCount * 14 + observationCount * 4
        ),
        kalmanStatus: words.slice(observationCount * 18, observationCount * 19),
        fullFiltered: floats.slice(observationCount * 19, observationCount * 23),
        fullSmoothed: floats.slice(observationCount * 23, observationCount * 27),
        fullFilteredCovariance: floats.slice(observationCount * 27, observationCount * 43),
        fullSmoothedCovariance: floats.slice(observationCount * 43, observationCount * 59),
        fullKalmanStatus: words.slice(observationCount * 59, observationCount * 60),
        flagged: words[summaryOffset / 4],
        gaps: words[summaryOffset / 4 + 1],
        length: floats[metricOffset / 4],
        duration: floats[metricOffset / 4 + 1],
        maximumSpeed: floats[metricOffset / 4 + 2]
      };
      ctx.setReadout(
        'quality',
        `${summary.flagged} flagged observations · ${summary.gaps} long-gap endpoints`
      );
      const polygonStatuses = words.slice(polygonStatusOffset, compactSummaryOffset);
      let polygonUnknown = 0;
      for (let row = 0; row < observationCount; row++) {
        if ((polygonStatuses[row * polygonIntervalsPerSegment] & 8) !== 0) polygonUnknown++;
      }
      const polygonInvalid = polygonStatuses.filter(status => (status & 4) !== 0).length;
      ctx.setReadout(
        'zoneRestriction',
        `${words[compactSummaryOffset + 1]} merged intervals · ${polygonUnknown} source rows unknown across gaps · ${polygonInvalid} invalid`
      );
      ctx.setReadout(
        'zoneCompaction',
        `${words[compactSummaryOffset]} required · ${words[compactSummaryOffset + 1]} written · ${words[compactSummaryOffset + 2]} overflow`
      );
      ctx.setReadout('trackLength', `${(summary.length / 1000).toFixed(1)} km`);
      ctx.setReadout('trackDuration', `${(summary.duration / 3600).toFixed(1)} h observed`);
      ctx.setReadout(
        'maximumSpeed',
        `${(summary.maximumSpeed / METERS_PER_KNOT_SECOND).toFixed(1)} kn`
      );
      ctx.setReadout(
        'kalmanResets',
        `${(ctx.options.covarianceModel === 'full' ? summary.fullKalmanStatus : summary.kalmanStatus).filter(status => status & 1).length} filter resets`
      );
      const useFullCovariance = ctx.options.covarianceModel === 'full';
      const filteredStateValues = useFullCovariance ? summary.fullFiltered : summary.filtered;
      const smoothedStateValues = useFullCovariance ? summary.fullSmoothed : summary.smoothed;
      const kalmanStatusValues = useFullCovariance
        ? summary.fullKalmanStatus
        : summary.kalmanStatus;
      const invalidStateCount = kalmanStatusValues.filter(status => status & 4).length;
      const smoothingFallbackCount = kalmanStatusValues.filter(status => status & 8).length;
      ctx.setReadout(
        'kalmanValidity',
        `${observationCount - invalidStateCount} valid · ${invalidStateCount} invalid states · ${smoothingFallbackCount} RTS fallback steps`
      );
      const bins = [0, 0, 0, 0];
      let filteredSquaredError = 0;
      let smoothedSquaredError = 0;
      let validEstimates = 0;
      for (const flag of summary.flags) {
        if (flag & 8) bins[0]++;
        else if (flag & 4) bins[1]++;
        else bins[2]++;
      }
      for (let row = 0; row < observationCount; row++) {
        if (kalmanStatusValues[row] & 4) continue;
        const observedX = positions[row * 2],
          observedY = positions[row * 2 + 1];
        filteredSquaredError +=
          (filteredStateValues[row * 4] - observedX) ** 2 +
          (filteredStateValues[row * 4 + 1] - observedY) ** 2;
        smoothedSquaredError +=
          (smoothedStateValues[row * 4] - observedX) ** 2 +
          (smoothedStateValues[row * 4 + 1] - observedY) ** 2;
        validEstimates++;
      }
      ctx.setReadout(
        'modelResidual',
        validEstimates
          ? `filter ${(Math.sqrt(filteredSquaredError / validEstimates)).toFixed(0)} m · smoother ${(Math.sqrt(smoothedSquaredError / validEstimates)).toFixed(0)} m RMS`
          : 'no valid model states'
      );
      let maximumFilteredPositionDifference = 0;
      let maximumSmoothedPositionDifference = 0;
      let comparableEstimates = 0;
      for (let row = 0; row < observationCount; row++) {
        if (summary.kalmanStatus[row] & 4 || summary.fullKalmanStatus[row] & 4) continue;
        const filteredDifference = Math.hypot(
          summary.filtered[row * 4] - summary.fullFiltered[row * 4],
          summary.filtered[row * 4 + 1] - summary.fullFiltered[row * 4 + 1]
        );
        const smoothedDifference = Math.hypot(
          summary.smoothed[row * 4] - summary.fullSmoothed[row * 4],
          summary.smoothed[row * 4 + 1] - summary.fullSmoothed[row * 4 + 1]
        );
        maximumFilteredPositionDifference = Math.max(
          maximumFilteredPositionDifference,
          filteredDifference
        );
        maximumSmoothedPositionDifference = Math.max(
          maximumSmoothedPositionDifference,
          smoothedDifference
        );
        comparableEstimates++;
      }
      ctx.setReadout(
        'covarianceParity',
        comparableEstimates
          ? `max position difference: filter ${maximumFilteredPositionDifference.toFixed(3)} m · smoother ${maximumSmoothedPositionDifference.toFixed(3)} m${ctx.options.axisRatio === 1 && ctx.options.axisCorrelation === 0 ? ' · ratio 1 / ρ 0 parity check' : ''}`
          : 'no comparable valid states'
      );
      const ellipseHours: number[] = [];
      const majorStandardDeviations: number[] = [];
      const minorStandardDeviations: number[] = [];
      for (let row = 0; row < observationCount; row++) {
        if (kalmanStatusValues[row] & 4) continue;
        let varianceX: number;
        let covarianceXY = 0;
        let varianceY: number;
        if (useFullCovariance) {
          const covariance =
            ctx.options.display === 'filtered'
              ? summary.fullFilteredCovariance
              : summary.fullSmoothedCovariance;
          const base = row * 16;
          varianceX = covariance[base];
          covarianceXY = (covariance[base + 1] + covariance[base + 4]) / 2;
          varianceY = covariance[base + 5];
        } else {
          const covariance =
            ctx.options.display === 'filtered'
              ? summary.filteredCovariance
              : summary.smoothedCovariance;
          varianceX = covariance[row * 4];
          varianceY = varianceX;
        }
        const halfTrace = (varianceX + varianceY) / 2;
        const halfDifference = (varianceX - varianceY) / 2;
        const spread = Math.hypot(halfDifference, covarianceXY);
        majorStandardDeviations.push(Math.sqrt(Math.max(0, halfTrace + spread)));
        minorStandardDeviations.push(Math.sqrt(Math.max(0, halfTrace - spread)));
        ellipseHours.push(times[row] / 3600);
      }
      const sortedMajor = majorStandardDeviations.slice().sort((first, second) => first - second);
      const sortedMinor = minorStandardDeviations.slice().sort((first, second) => first - second);
      ctx.setReadout(
        'uncertaintySummary',
        sortedMajor.length
          ? `${useFullCovariance ? 'Full covariance' : 'Isotropic covariance'} · median 1σ axes ${sortedMajor[Math.floor(sortedMajor.length / 2)].toFixed(1)} / ${sortedMinor[Math.floor(sortedMinor.length / 2)].toFixed(1)} m${useFullCovariance ? ` · ρ ${ctx.options.axisCorrelation.toFixed(1)}, axis ratio ${ctx.options.axisRatio.toFixed(1)}` : ''}`
          : 'no valid position covariance'
      );
      ctx.setChart('uncertaintyChart', {
        kind: 'line',
        xLabel: 'hours since first fix',
        yLabel: 'position standard deviation (m, 1σ)',
        series: [
          {label: 'major axis', x: ellipseHours, y: majorStandardDeviations, color: 0, width: 2},
          {label: 'minor axis', x: ellipseHours, y: minorStandardDeviations, color: 1, width: 1.7}
        ],
        description: `Model-implied ${ctx.options.display === 'filtered' ? 'filtered' : 'smoothed'} position covariance; this is conditional uncertainty from the chosen ${useFullCovariance ? 'correlated full-covariance' : 'isotropic'} model, not calibrated AIS accuracy.`
      });
      ctx.setChart('flagsChart', {
        kind: 'bars',
        height: 110,
        values: bins.slice(0, 3),
        labels: ['Gap > limit', 'Speed > limit', 'No threshold flag'],
        highlight: [0, 1],
        yLabel: 'fixes',
        description:
          'GPU raw-observation flags; a threshold exceedance is a review cue, not evidence of a bad AIS fix.'
      });
      ctx.requestLayers();
    }
  );
  function markDirty(): void {
    generation++;
    dirty = true;
    summary = null;
    ctx.setReadout('quality', 'Waiting for GPU estimate');
    ctx.setReadout('trackLength', '—');
    ctx.setReadout('trackDuration', '—');
    ctx.setReadout('maximumSpeed', '—');
    ctx.setReadout('kalmanResets', '—');
    ctx.setReadout('kalmanValidity', '—');
    ctx.setReadout('modelResidual', '—');
    ctx.setReadout('covarianceParity', '—');
    ctx.setReadout('uncertaintySummary', 'Waiting for GPU estimate');
    ctx.setReadout('zoneRestriction', 'Waiting for GPU geometry clip');
    ctx.setReadout('zoneCompaction', 'Waiting for GPU compaction');
    ctx.setChart('flagsChart', null);
    ctx.setChart('uncertaintyChart', null);
    ctx.requestLayers();
  }
  function updateParameters(): void {
    thresholdsBuffer.write(
      Float32Array.of(
        ctx.options.maximumSpeedKnots * METERS_PER_KNOT_SECOND,
        ctx.options.maximumGapSeconds
      )
    );
    modelParametersBuffer.write(
      Float32Array.of(
        ctx.options.processNoise,
        ctx.options.measurementVariance,
        2500,
        ctx.options.maximumModelGapSeconds
      )
    );
    const axisRatio = ctx.options.axisRatio;
    const correlation = ctx.options.axisCorrelation;
    const axisScale = Math.sqrt(axisRatio);
    const processX = ctx.options.processNoise * axisScale;
    const processY = ctx.options.processNoise / axisScale;
    const measurementX = ctx.options.measurementVariance * axisScale;
    const measurementY = ctx.options.measurementVariance / axisScale;
    fullModelParametersBuffer.write(
      Float32Array.of(
        processX,
        correlation * Math.sqrt(processX * processY),
        processY,
        measurementX,
        correlation * Math.sqrt(measurementX * measurementY),
        measurementY,
        2500,
        0,
        2500,
        ctx.options.maximumModelGapSeconds
      )
    );
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let row = 0; row < observationCount; row++) {
      minX = Math.min(minX, positions[row * 2]);
      minY = Math.min(minY, positions[row * 2 + 1]);
      maxX = Math.max(maxX, positions[row * 2]);
      maxY = Math.max(maxY, positions[row * 2 + 1]);
    }
    const windowEnd = Math.min(
      times[times.length - 1],
      ctx.options.windowEndHour * 3600 - originTime
    );
    restrictionParametersBuffer.write(
      Float32Array.of(
        0,
        Math.max(0, windowEnd),
        minX - 10,
        minY - 10,
        maxX + 10,
        maxY + 10,
        ctx.options.maximumModelGapSeconds
      )
    );
    polygonParametersBuffer.write(
      Float32Array.of(
        0,
        Math.max(0, windowEnd),
        zoneMinX - 50,
        zoneMinY - 50,
        zoneMaxX + 50,
        zoneMaxY + 50,
        ctx.options.maximumModelGapSeconds,
        ...zoneVertices
      )
    );
    markDirty();
  }
  // Recipe intermediates are graph-owned; publish their compact track metrics into app-owned buffers.
  addKernelPass(graph, {
    id: 'publish-track-metrics',
    invocationCount: 1,
    bindings: [
      {name: 'length', view: metricOutputs.trackLengths, type: 'f32', access: 'read'},
      {name: 'duration', view: metricOutputs.trackDurations, type: 'f32', access: 'read'},
      {name: 'maximumSpeed', view: metricOutputs.maximumSpeeds, type: 'f32', access: 'read'},
      {
        name: 'outLength',
        view: view(lengthsBuffer, 'float32', 1),
        type: 'f32',
        access: 'read_write'
      },
      {
        name: 'outDuration',
        view: view(durationsBuffer, 'float32', 1),
        type: 'f32',
        access: 'read_write'
      },
      {
        name: 'outMaximumSpeed',
        view: view(maximumSpeedsBuffer, 'float32', 1),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: 'outLength[outLengthOffset] = length[lengthOffset]; outDuration[outDurationOffset] = duration[durationOffset]; outMaximumSpeed[outMaximumSpeedOffset] = maximumSpeed[maximumSpeedOffset];'
  });
  // Copy raw recipe outputs to stable app-owned buffers for the summary readback.
  addKernelPass(graph, {
    id: 'publish-raw-quality',
    invocationCount: observationCount,
    bindings: [
      {name: 'flags', view: recipe.outputs.flags, type: 'u32', access: 'read'},
      {name: 'speed', view: metricOutputs.stepSpeeds, type: 'f32', access: 'read'},
      {
        name: 'outFlags',
        view: view(flagsBuffer, 'uint32', observationCount),
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'outSpeed',
        view: view(speedsBuffer, 'float32', observationCount),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: 'outFlags[outFlagsOffset + index] = flags[flagsOffset + index]; outSpeed[outSpeedOffset + index] = speed[speedOffset + index];'
  });
  const compiled: CompiledGPUCommandGraph<void> = resources.track(await graph.compileAsync());
  updateParameters();
  ctx.setReadout('trackId', `MMSI ${TRACK_MMSI} · ${observationCount} fixes`);
  ctx.setReadout(
    'clockPrecision',
    `${new Set(times).size}/${observationCount} distinct relative times · naïve epoch float32 ${new Set(naiveAbsoluteTimes).size}/${observationCount}, max error ${maximumNaiveTimeError.toFixed(0)} s`
  );
  ctx.setReadout(
    'trackWindow',
    `${new Date(originTime * 1000 + Date.UTC(2024, 5, 12)).toISOString().slice(11, 16)}–${new Date(vesselSet.timestamps[last - 1] * 1000 + Date.UTC(2024, 5, 12)).toISOString().slice(11, 16)} UTC`
  );
  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder) {
      if (dirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        reader.markStale();
      }
      if (!reader.isPending && generation !== pendingGeneration) pendingGeneration = generation;
      reader.flush(commandEncoder);
    },
    setOption(id) {
      updateParameters();
    },
    getLayers() {
      const dark = ctx.ground() === 'dark';
      const layerOptions = {
        analysisFrame: vesselSet.analysisFrame,
        instanceCount: observationCount - 1,
        widthPixels: 3.2
      };
      const observedColor = dark ? ([156, 177, 191, 140] as const) : ([99, 120, 132, 160] as const);
      if (!summary)
        return [
          new SpatialAnalysisSegmentLayer({
            id: 'harbor-ais-observed-pending',
            analysisFrame: vesselSet.analysisFrame,
            segments: observedBuffer,
            instanceCount: observationCount - 1,
            widthPixels: 1.8,
            color: observedColor
          })
        ];
      const estimate =
        ctx.options.covarianceModel === 'full'
          ? ctx.options.display === 'filtered'
            ? fullFilteredSegmentsBuffer
            : fullSmoothedSegmentsBuffer
          : ctx.options.display === 'filtered'
            ? filteredSegmentsBuffer
            : smoothedSegmentsBuffer;
      const uncertaintyEllipses =
        ctx.options.covarianceModel === 'full'
          ? ctx.options.display === 'filtered'
            ? fullFilteredEllipsesBuffer
            : fullSmoothedEllipsesBuffer
          : ctx.options.display === 'filtered'
            ? isoFilteredEllipsesBuffer
            : isoSmoothedEllipsesBuffer;
      const observedLayer = new SpatialAnalysisSegmentLayer({
        id: 'harbor-ais-observed-context',
        segments: windowedObservedBuffer,
        ...layerOptions,
        widthPixels: 1.2,
        color: observedColor
      });
      return [
        ...(ctx.options.display === 'observed'
          ? [
              new SpatialAnalysisSegmentLayer({
                id: 'harbor-ais-observed',
                segments: windowedObservedBuffer,
                ...layerOptions,
                widthPixels: 1.8,
                color: observedColor
              })
            ]
          : []),
        ...(ctx.options.display === 'gaps'
          ? [
              new SpatialAnalysisSegmentLayer({
                id: 'harbor-ais-gap-supported',
                segments: restrictionSegmentsBuffer,
                ...layerOptions,
                instanceCount: observationCount,
                widthPixels: 4,
                color: [45, 155, 183, 245]
              })
            ]
          : []),
        ...(ctx.options.display === 'zone'
          ? [
              new SpatialAnalysisSegmentLayer({
                id: 'harbor-ais-zone-union',
                segments: compactPositionsBuffer,
                ...layerOptions,
                instanceCount: polygonIntervalCapacity,
                widthPixels: 4.5,
                color: [236, 178, 74, 245]
              }),
              new SpatialAnalysisSegmentLayer({
                id: 'harbor-ais-study-zone-rings',
                segments: polygonBoundaryBuffer,
                ...layerOptions,
                instanceCount: 8,
                widthPixels: 2.2,
                color: [99, 216, 202, 230]
              })
            ]
          : []),
        ...(ctx.options.display === 'filtered' || ctx.options.display === 'smoothed'
          ? [
              observedLayer,
              new SpatialAnalysisSegmentLayer({
                id: 'harbor-ais-estimate',
                segments: estimate,
                ...layerOptions,
                widthPixels: 3.8,
                color:
                  ctx.options.display === 'smoothed' ? [232, 112, 50, 245] : [134, 70, 201, 240]
              }),
              ...(ctx.options.showUncertainty
                ? [
                    new SpatialAnalysisSegmentLayer({
                      id: 'harbor-kalman-one-sigma-ellipses',
                      analysisFrame: vesselSet.analysisFrame,
                      segments: uncertaintyEllipses,
                      instanceCount: observationCount * uncertaintyEllipseSegmentCount,
                      widthPixels: 1.25,
                      color: [67, 223, 218, 170]
                    })
                  ]
                : [])
            ]
          : []),
        new SpatialAnalysisPointLayer({
          id: 'harbor-ais-observation-fixes',
          analysisFrame: vesselSet.analysisFrame,
          positions: windowedObservationPointsBuffer,
          instanceCount: observationCount,
          radiusPixels: 2.2,
          color: [235, 241, 244, 175]
        }),
        new SpatialAnalysisPointLayer({
          id: 'harbor-ais-flagged-fixes',
          analysisFrame: vesselSet.analysisFrame,
          positions: windowedObservationPointsBuffer,
          instanceCount: observationCount,
          values: flagsBuffer,
          valueFormat: 'uint32',
          colormap: 'mask',
          color: [239, 90, 69, 255],
          noDataColor: [0, 0, 0, 0],
          radiusPixels: 4.3,
          outlineColor: [26, 30, 39, 255],
          outlineWidthPixels: 0.8
        })
      ] as Layer[];
    },
    getTooltip(event) {
      if (!summary) return null;
      const viewport = ctx.getViewport();
      if (!viewport) return null;
      let nearest = -1;
      let nearestDistance = 14 * 14;
      for (let row = 0; row < observationCount; row++) {
        if (vesselSet.timestamps[first + row] - originTime > ctx.options.windowEndHour * 3600)
          continue;
        const projected = viewport.project([sourceLngLat[row * 2], sourceLngLat[row * 2 + 1]]);
        const distance =
          (projected[0] - event.pixel[0]) ** 2 + (projected[1] - event.pixel[1]) ** 2;
        if (distance < nearestDistance) {
          nearest = row;
          nearestDistance = distance;
        }
      }
      if (nearest < 0) return null;
      const flag = summary.flags[nearest];
      const useFullCovariance = ctx.options.covarianceModel === 'full';
      const useFiltered = ctx.options.display === 'filtered';
      const selectedStatus = useFullCovariance
        ? summary.fullKalmanStatus[nearest]
        : summary.kalmanStatus[nearest];
      const covariance = useFullCovariance
        ? useFiltered
          ? summary.fullFilteredCovariance
          : summary.fullSmoothedCovariance
        : useFiltered
          ? summary.filteredCovariance
          : summary.smoothedCovariance;
      const covarianceOffset = nearest * (useFullCovariance ? 16 : 4);
      const varianceX = covariance[covarianceOffset];
      const covarianceXY = useFullCovariance
        ? (covariance[covarianceOffset + 1] + covariance[covarianceOffset + 4]) / 2
        : 0;
      const varianceY = useFullCovariance ? covariance[covarianceOffset + 5] : varianceX;
      const halfTrace = (varianceX + varianceY) / 2;
      const spread = Math.hypot((varianceX - varianceY) / 2, covarianceXY);
      const majorDeviation = Math.sqrt(Math.max(0, halfTrace + spread));
      const minorDeviation = Math.sqrt(Math.max(0, halfTrace - spread));
      const labels = [
        flag & 1 ? 'invalid position' : '',
        flag & 2 ? 'invalid time' : '',
        flag & 4 ? 'speed above threshold' : '',
        flag & 8 ? 'sampling gap above threshold' : ''
      ].filter(Boolean);
      return {
        title: `AIS observation ${nearest + 1}`,
        subtitle:
          new Date(vesselSet.timestamps[first + nearest] * 1000 + Date.UTC(2024, 5, 12))
            .toISOString()
            .slice(11, 19) + ' UTC',
        rows: [
          {
            label: 'Derived step speed',
            value: `${(summary.speeds[nearest] / METERS_PER_KNOT_SECOND).toFixed(1)} kn`
          },
          {label: 'Raw flags', value: labels.join(', ') || 'none at chosen thresholds'},
          {
            label: `${useFullCovariance ? 'Full' : 'Isotropic'} ${useFiltered ? 'filtered' : 'smoothed'} 1σ axes`,
            value: `${majorDeviation.toFixed(0)} m / ${minorDeviation.toFixed(0)} m`
          },
          {label: 'Position covariance xy', value: `${covarianceXY.toFixed(1)} m²`},
          {
            label: 'Kalman state',
            value:
              selectedStatus & 4
                ? 'invalid'
                : selectedStatus & 2
                  ? 'prediction only'
                  : 'measurement updated'
          }
        ],
        note: 'Threshold flags identify values to review. The Kalman estimate is a motion model, not verified position truth.'
      };
    },
    onThemeChange() {
      ctx.requestLayers();
    },
    destroy() {
      destroyed = true;
      reader.stop();
      resources.destroy();
    }
  };
}
