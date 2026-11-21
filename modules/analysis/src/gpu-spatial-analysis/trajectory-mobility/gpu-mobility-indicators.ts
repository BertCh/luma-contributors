// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RequireTrajectoryOutput} from '../trajectory-analysis/trajectory-output-choice';
import {resolveGyrationCenter, type GyrationCenter} from './gyration-center';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {COMPENSATED_SUM_WGSL} from '../../analysis-core/kernels/compensated-sum-wgsl';
import {
  validateTrajectoryBindings,
  validateTrajectoryWritableDispatches,
  validateTrajectoryScratch,
  validateTrajectoryOutputRanges
} from '../trajectory-analysis/trajectory-binding-validation';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {
  GPUCandidateStatusPort,
  GPUStatusScalar,
  GPUTrajectoryTimestampFormat
} from '../../analysis-core/contracts/index';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getTrajectoryTimeBindings,
  getTrajectoryTimeMode,
  getTrajectoryTimeSource
} from '../trajectory-analysis/trajectory-metrics-kernels';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';
import {
  createTrackTableOffsetsNode,
  MOVEMENT_SENTINEL_WGSL
} from '../trajectory-staypoints/movement-common';

const OPERATION = 'GPUMobilityIndicators';
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);

/** Default cap on the staypoint sequence length of the real-entropy estimator. */
export const GPU_MOBILITY_DEFAULT_ENTROPY_SEQUENCE_ROWS = 1024;

/** Largest accepted cap on the staypoint sequence length of the real-entropy estimator. */
export const GPU_MOBILITY_MAXIMUM_ENTROPY_SEQUENCE_ROWS = 16384;

/** Staypoint columns consumed by {@link GPUMobilityIndicators}. */
export type GPUMobilityStaypointColumns = {
  /** Track of each staypoint, sorted (`GPUTrajectoryStaypoints` output `ids`). */
  tracks: GraphDataView<'uint32'>;
  /** Staypoint centroids, positions of the indicators. */
  centroids: GraphDataView<'float32x2'>;
  /** First row of each staypoint, for `started_at`. */
  startRows: GraphDataView<'uint32'>;
  /** Staypoint durations in the timestamps' unit. */
  durations: GraphDataView<'float32'>;
  /** Location of each staypoint (`GPUTrajectoryLocations` `locationIds`). */
  locationIds?: GraphDataView<'uint32'>;
  /** One row in `count`: active staypoint count, as `GPUTrajectoryStaypoints` writes it. */
  status: {
    count: GPUStatusScalar;
    overflow?: GPUStatusScalar;
    candidateOverflow?: GPUStatusScalar;
    /** Optional locations iteration-loss evidence; suppresses only location-dependent outputs. */
    locationIterationLimitReached?: GPUStatusScalar;
  };
};

/**
 * Caller-owned outputs. Supplied staypoint loss suppresses every numeric row to NaN and every
 * location row to 0xffffffff, including unused capacity rows. At least one indicator is required.
 */
export type GPUMobilityIndicatorsOutput = RequireTrajectoryOutput<{
  /** Per-track radius of gyration of the staypoint centroids (`trackCount` rows). */
  radiusOfGyration?: GraphDataView<'float32'>;
  /** Per-staypoint jump length from the previous staypoint of the track (capacity rows; first row of a track 0). */
  jumpLengths?: GraphDataView<'float32'>;
  /** Per-staypoint waiting time `start_i - start_{i-1}` (capacity rows; first of a track 0). */
  waitingTimes?: GraphDataView<'float32'>;
  /** Per-track `log2(distinct locations)`. */
  randomEntropy?: GraphDataView<'float32'>;
  /** Per-track temporal-uncorrelated entropy of the location shares, bits. */
  uncorrelatedEntropy?: GraphDataView<'float32'>;
  /**
   * Per-track real entropy (Lempel-Ziv estimator of the location sequence), bits. NaN for a track
   * with more than `maximumEntropySequenceRows` usable location keys.
   */
  realEntropy?: GraphDataView<'float32'>;
  /** Per-track location with the largest summed duration; `0xffffffff` for a track without staypoints. */
  homeLocations?: GraphDataView<'uint32'>;
  /** Per-track location with the second largest summed duration; `0xffffffff` when absent. */
  workLocations?: GraphDataView<'uint32'>;
  /** Per-track tracking quality; NaN for all tracks when supplied tripleg loss is nonzero. */
  trackingQuality?: GraphDataView<'float32'>;
  /** 1 for supplied staypoint/location loss or a sequence exceeding maximumEntropySequenceRows. */
  status?: GPUCandidateStatusPort & {
    /** Number of active staypoint rows with missing or unsettled keys, across all tracks. */
    excludedLocationCount?: GPUStatusScalar;
  };
}>;

/** Properties for {@link GPUMobilityIndicators}. */
export type GPUMobilityIndicatorsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'mobility-indicators'`. */
  id?: string;
  /** Planar, or longitude/latitude great-circle (haversine). Distances in the context's length unit. */
  spatialContext?: GPUSpatialContext;
  /** Number of tracks, the row count of per-track outputs. */
  trackCount: number;
  /** Trajectory timestamps, aligned with the rows `startRows` and `finishRows` index. */
  timestamps: GraphDataView<GPUTrajectoryTimestampFormat>;
  staypoints: GPUMobilityStaypointColumns;
  /** Tripleg table, for `trackingQuality`: `tracks`, `startRows`, `endRows`(exclusive), `status.count`. */
  triplegs?: {
    tracks: GraphDataView<'uint32'>;
    startRows: GraphDataView<'uint32'>;
    endRows: GraphDataView<'uint32'>;
    status: {
      count: GPUStatusScalar;
      overflow?: GPUStatusScalar;
      candidateOverflow?: GPUStatusScalar;
    };
    /** Staypoint `finishRows`. */
    staypointFinishRows: GraphDataView<'uint32'>;
  };
  /** Cap on usable location keys per track for `realEntropy` (default 1024, at most 16384). */
  maximumEntropySequenceRows?: number;
  /** Defaults to spherical mean for great-circle coordinates, coordinate mean for planar. Explicit coordinate-mean preserves arithmetic centering; a mean unit-vector norm <= 1e-6 yields NaN. */
  gyrationCenter?: GyrationCenter;
  output: GPUMobilityIndicatorsOutput;
};

/**
 * Per-user mobility indicators on staypoints, following the definitions of scikit-mobility 1.3.1
 * (`individual` measures) and trackintel (`freq_method`, `temporal_tracking_quality`):
 *
 * - `radius_of_gyration`: root mean squared distance from the spherical mean in great-circle
 *   contexts or coordinate mean in planar contexts. Explicit `gyrationCenter: 'coordinate-mean'`
 *   retains scikit-mobility arithmetic centering; an indeterminate spherical center yields NaN.
 * - `jump_lengths`: distance between consecutive staypoints.
 * - `waiting_times`: difference between consecutive staypoint start times.
 * - `random_entropy`: `log2(distinct locations)`. `uncorrelated_entropy`: Shannon entropy in bits of
 *   location shares. `real_entropy`: the Lempel-Ziv estimator `n * log2(n) / sum(lambda_i)` of
 *   Song et al. as implemented in scikit-mobility, on the location sequence.
 * - home and work: the locations with the longest and second longest summed duration (trackintel
 *   FREQ, no pre-filter). Ties go to the lower location index.
 * - tracking quality: `sum(positive durations) / (last finish - first start)` over staypoints and
 *   triplegs (trackintel `temporal_tracking_quality`, granularity `all`).
 *
 * Each indicator kernel runs one thread per track. Location statistics are `O(m^2)` per track and
 * real entropy `O(m^3)` worst case per track with `m` staypoints; the sequence cap bounds the last.
 * Modal split is {@link GPUModalSplit}.
 * Supplied `staypoints.status.locationIterationLimitReached` suppresses the three location
 * entropies to NaN and home/work IDs to 0xffffffff, and sets `status.candidateOverflow` when
 * present. Gyration, jumps, waits and tracking quality remain independent of location labels.
 * All active rows are excluded from location indicators when the flag is set. Otherwise,
 * rows with the sentinel key or an omitted location column are excluded. The optional one-row
 * uint32 `status.excludedLocationCount` counts these rows. Tracks with stays but no usable
 * keys have NaN entropies and sentinel home/work IDs.
 */
export class GPUMobilityIndicators implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUMobilityIndicatorsProps;
  readonly spatialContext: GPUSpatialContext;
  readonly maximumEntropySequenceRows: number;

  constructor(props: GPUMobilityIndicatorsProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'trackCount',
      'timestamps',
      'staypoints',
      'triplegs',
      'maximumEntropySequenceRows',
      'gyrationCenter',
      'output'
    ]);
    const stays = snapshotConstructionFields(props.staypoints, [
      'tracks',
      'centroids',
      'startRows',
      'durations',
      'locationIds',
      'status'
    ]);
    const legs =
      props.triplegs &&
      snapshotConstructionFields(props.triplegs, [
        'tracks',
        'startRows',
        'endRows',
        'status',
        'staypointFinishRows'
      ]);
    const outputs = snapshotConstructionFields(props.output, [
      'radiusOfGyration',
      'jumpLengths',
      'waitingTimes',
      'randomEntropy',
      'uncorrelatedEntropy',
      'realEntropy',
      'homeLocations',
      'workLocations',
      'trackingQuality',
      'status'
    ]);
    props = {
      ...props,
      staypoints: {
        ...stays,
        status: snapshotConstructionFields(stays.status, [
          'count',
          'overflow',
          'candidateOverflow',
          'locationIterationLimitReached'
        ])
      },
      triplegs: legs && {
        ...legs,
        status: snapshotConstructionFields(legs.status, ['count', 'overflow', 'candidateOverflow'])
      },
      output: {
        ...outputs,
        status:
          outputs.status &&
          snapshotConstructionFields(outputs.status, ['candidateOverflow', 'excludedLocationCount'])
      }
    };
    props = {
      ...props,
      gyrationCenter: resolveGyrationCenter(
        props.id ?? 'mobility-indicators',
        props.gyrationCenter,
        props.spatialContext
      )
    };
    this.id = props.id ?? 'mobility-indicators';
    this.props = props;
    const id = this.id;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    const {staypoints, triplegs, output, timestamps} = props;
    for (const [name, flag] of Object.entries(staypoints.status)) {
      if (!flag) continue;
      validateSinglePackedView(flag, ['uint32'], `${id} staypoints.status.${name}`);
      validateRequirement(
        id,
        flag.length >= 1,
        `staypoints.status.${name} must contain at least one row`
      );
    }
    for (const [name, flag] of Object.entries(triplegs?.status ?? {})) {
      if (!flag) continue;
      validateSinglePackedView(flag, ['uint32'], `${id} triplegs.status.${name}`);
      validateRequirement(
        id,
        flag.length >= 1,
        `triplegs.status.${name} must contain at least one row`
      );
    }
    validateRequirement(
      id,
      props.gyrationCenter !== 'spherical-mean' ||
        getGPUGreatCircleRadius(this.spatialContext) !== undefined,
      'gyrationCenter spherical-mean requires a great-circle spatialContext; use coordinate-mean for planar positions'
    );
    validateCapacity(id, 'trackCount', props.trackCount);
    validateSinglePackedView(timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
    validateSinglePackedView(staypoints.tracks, ['uint32'], `${id} staypoints.tracks`);
    validateSinglePackedView(staypoints.centroids, ['float32x2'], `${id} staypoints.centroids`);
    validateSinglePackedView(staypoints.startRows, ['uint32'], `${id} staypoints.startRows`);
    validateSinglePackedView(staypoints.durations, ['float32'], `${id} staypoints.durations`);
    if (staypoints.locationIds)
      validateSinglePackedView(staypoints.locationIds, ['uint32'], `${id} staypoints.locationIds`);
    const capacity = staypoints.tracks.length;
    for (const view of [
      staypoints.centroids,
      staypoints.startRows,
      staypoints.durations,
      staypoints.locationIds
    ]) {
      if (view) validateViewLength(id, 'staypoint column', view, capacity);
    }
    for (const [name, view, format] of [
      ['radiusOfGyration', output.radiusOfGyration, 'float32'],
      ['randomEntropy', output.randomEntropy, 'float32'],
      ['uncorrelatedEntropy', output.uncorrelatedEntropy, 'float32'],
      ['realEntropy', output.realEntropy, 'float32'],
      ['homeLocations', output.homeLocations, 'uint32'],
      ['workLocations', output.workLocations, 'uint32'],
      ['trackingQuality', output.trackingQuality, 'float32']
    ] as const) {
      if (view) {
        validateSinglePackedView(view, [format], `${id} output.${name}`);
        validateViewLength(id, `output.${name}`, view, props.trackCount);
      }
    }
    for (const [name, view] of [
      ['jumpLengths', output.jumpLengths],
      ['waitingTimes', output.waitingTimes]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['float32'], `${id} output.${name}`);
        validateViewLength(id, `output.${name}`, view, capacity);
      }
    }
    validateRequirement(
      id,
      Boolean(
        output.radiusOfGyration ||
          output.jumpLengths ||
          output.waitingTimes ||
          output.randomEntropy ||
          output.uncorrelatedEntropy ||
          output.realEntropy ||
          output.homeLocations ||
          output.workLocations ||
          output.trackingQuality
      ),
      'requires at least one output'
    );
    validateRequirement(
      id,
      Boolean(!output.trackingQuality || triplegs),
      'trackingQuality requires triplegs'
    );
    for (const [name, view] of Object.entries(output.status ?? {})) {
      if (!view) continue;
      validateSinglePackedView(view, ['uint32'], `${id} output.status.${name}`);
      validateViewLength(id, `output.status.${name}`, view, 1);
    }
    this.maximumEntropySequenceRows =
      props.maximumEntropySequenceRows ?? GPU_MOBILITY_DEFAULT_ENTROPY_SEQUENCE_ROWS;
    validateMaximumCount(
      id,
      'maximumEntropySequenceRows',
      this.maximumEntropySequenceRows,
      GPU_MOBILITY_MAXIMUM_ENTROPY_SEQUENCE_ROWS
    );
    validateTrajectoryOutputRanges(this.id, this.props.output);
    validateTrajectoryWritableDispatches(this.id, [
      {
        randomEntropy: output.randomEntropy,
        uncorrelatedEntropy: output.uncorrelatedEntropy,
        homeLocations: output.homeLocations,
        workLocations: output.workLocations
      },
      {realEntropy: output.realEntropy, candidateOverflow: output.status?.candidateOverflow},
      {jumpLengths: output.jumpLengths, waitingTimes: output.waitingTimes}
    ]);
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
  }

  private getInputViews(): (GraphDataView | undefined)[] {
    const {staypoints, triplegs, timestamps} = this.props;
    return [
      timestamps,
      staypoints.tracks,
      staypoints.centroids,
      staypoints.startRows,
      staypoints.durations,
      staypoints.locationIds,
      staypoints.status.count,
      staypoints.status.overflow,
      staypoints.status.candidateOverflow,
      staypoints.status.locationIterationLimitReached,
      triplegs?.tracks,
      triplegs?.startRows,
      triplegs?.endRows,
      triplegs?.status.count,
      triplegs?.status.overflow,
      triplegs?.status.candidateOverflow,
      triplegs?.staypointFinishRows
    ];
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {output} = this.props;
    return [
      output.radiusOfGyration,
      output.jumpLengths,
      output.waitingTimes,
      output.randomEntropy,
      output.uncorrelatedEntropy,
      output.realEntropy,
      output.homeLocations,
      output.workLocations,
      output.trackingQuality,
      output.status?.candidateOverflow,
      output.status?.excludedLocationCount
    ];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {staypoints, triplegs, output, timestamps, trackCount} = props;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    const capacity = staypoints.tracks.length;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'centroids',
      capacity * 8,
      'use fewer staypoints per graph'
    );
    validateTrajectoryBindings(graph, this.id, this.props);
    validateTrajectoryScratch(graph, id, {'track offsets': (trackCount + 1) * 4});
    const radius = getGPUGreatCircleRadius(this.spatialContext);
    const timeMode = getTrajectoryTimeMode({timestamps});
    const trackOffsets = createTransientView(
      graph,
      `${id}-track-offsets`,
      'uint32',
      trackCount + 1
    );
    const nodes: GPUCommandNode<Parameters>[] = [
      createTrackTableOffsetsNode<Parameters>(graph, {
        id: `${id}-offsets`,
        operation: OPERATION,
        tracks: staypoints.tracks,
        count: staypoints.status.count,
        offsets: trackOffsets,
        capacity,
        trackCount
      })
    ];
    const distanceDeclarations = `${radius === undefined ? '' : GEODESIC_WGSL}
fn stayPoint(row: u32) -> vec2<f32> {
  return vec2<f32>(centroids[centroidsOffset + 2u * row], centroids[centroidsOffset + 2u * row + 1u]);
}
fn pointDistance(a: vec2<f32>, b: vec2<f32>) -> f32 {
  return ${radius === undefined ? 'length(a - b)' : `geodesicHaversineDistance(a, b, ${getWGSLFloatLiteral(radius)})`};
}`;
    if (output.radiusOfGyration) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-radius-of-gyration`,
          operation: OPERATION,
          invocationCount: trackCount,
          bindings: [
            {name: 'centroids', view: staypoints.centroids, type: 'f32', access: 'read'},
            {name: 'trackOffsets', view: trackOffsets, type: 'u32', access: 'read'},
            {
              name: 'radiusOfGyration',
              view: output.radiusOfGyration,
              type: 'f32',
              access: 'read_write'
            }
          ],
          declarations: `${distanceDeclarations}\n${GET_NAN_WGSL}\n${props.gyrationCenter === 'spherical-mean' ? COMPENSATED_SUM_WGSL : ''}`,
          body: `let begin = trackOffsets[trackOffsetsOffset + index];
  let end = trackOffsets[trackOffsetsOffset + index + 1u];
  var result = 0.0;
  if (end > begin) {
    let first = stayPoint(begin);
    var center = first;
    var undefinedCenter = false;
    ${
      props.gyrationCenter === 'spherical-mean'
        ? `var unitX = vec2<f32>(0.0);
    var unitY = vec2<f32>(0.0);
    var unitZ = vec2<f32>(0.0);
    var stationary = true;
    for (var row = begin; row < end; row++) {
      let point = stayPoint(row);
      stationary = stationary && all(point == first);
      let angles = point * 0.017453292519943295;
      unitX = addCompensatedValue(unitX, cos(angles.y) * cos(angles.x));
      unitY = addCompensatedValue(unitY, cos(angles.y) * sin(angles.x));
      unitZ = addCompensatedValue(unitZ, sin(angles.y));
    }
    let unitSum = vec3<f32>(getCompensated(unitX), getCompensated(unitY), getCompensated(unitZ));
    undefinedCenter = length(unitSum) <= f32(end - begin) * 1e-6;
    // Identical positions retain their exact stored center, avoiding a trigonometric round trip.
    if (!stationary && !undefinedCenter) {
      center = vec2<f32>(atan2(unitSum.y, unitSum.x), atan2(unitSum.z, length(unitSum.xy))) * 57.29577951308232;
    }`
        : `var sum = vec2<f32>(0.0, 0.0);
    for (var row = begin; row < end; row++) { sum += stayPoint(row) - first; }
    center = first + sum / f32(end - begin);`
    }
    var squares = 0.0;
    for (var row = begin; row < end; row++) {
      let separation = pointDistance(stayPoint(row), center);
      squares += separation * separation;
    }
    result = sqrt(squares / f32(end - begin));
    if (undefinedCenter) { result = getNaN(); }
  }
  radiusOfGyration[radiusOfGyrationOffset + index] = result;`
        })
      );
    }
    if (output.jumpLengths || output.waitingTimes) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-jumps`,
          operation: OPERATION,
          variant: timeMode,
          invocationCount: capacity,
          bindings: [
            {name: 'centroids', view: staypoints.centroids, type: 'f32', access: 'read'},
            {name: 'tracks', view: staypoints.tracks, type: 'u32', access: 'read'},
            {name: 'startRows', view: staypoints.startRows, type: 'u32', access: 'read'},
            {name: 'stayCount', view: staypoints.status.count, type: 'u32', access: 'read'},
            ...getTrajectoryTimeBindings({timestamps}),
            ...(output.jumpLengths
              ? [
                  {
                    name: 'jumpLengths',
                    view: output.jumpLengths,
                    type: 'f32',
                    access: 'read_write'
                  } as const
                ]
              : []),
            ...(output.waitingTimes
              ? [
                  {
                    name: 'waitingTimes',
                    view: output.waitingTimes,
                    type: 'f32',
                    access: 'read_write'
                  } as const
                ]
              : [])
          ],
          declarations: `${distanceDeclarations}\n${getTrajectoryTimeSource(timeMode)}`,
          body: `let isActive = index < min(stayCount[stayCountOffset], ${capacity}u);
  let hasPrevious = isActive && index > 0u && tracks[tracksOffset + index] == tracks[tracksOffset + index - 1u];
  ${
    output.jumpLengths
      ? `jumpLengths[jumpLengthsOffset + index] = select(0.0, pointDistance(stayPoint(index - 1u), stayPoint(index)), hasPrevious);`
      : ''
  }
  ${
    output.waitingTimes
      ? `waitingTimes[waitingTimesOffset + index] = select(0.0, timeDifferenceRows(startRows[startRowsOffset + index], startRows[startRowsOffset + max(index, 1u) - 1u]), hasPrevious);`
      : ''
  }`
        })
      );
    }
    const locationLoss = staypoints.status.locationIterationLimitReached;
    const locationLossBindings = locationLoss
      ? [{name: 'locationLoss', view: locationLoss, type: 'u32', access: 'read'} as const]
      : [];
    const locationIds = staypoints.locationIds;
    const locationIdBindings = locationIds
      ? [{name: 'locationIds', view: locationIds, type: 'u32', access: 'read'} as const]
      : [];
    const excludedLocationCount = output.status?.excludedLocationCount;
    if (excludedLocationCount) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-excluded-location-count-reset`,
          operation: OPERATION,
          view: excludedLocationCount,
          type: 'u32',
          value: '0u',
          componentCount: 1
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-excluded-location-count`,
          operation: OPERATION,
          invocationCount: capacity,
          bindings: [
            {name: 'stayCount', view: staypoints.status.count, type: 'u32', access: 'read'},
            ...locationIdBindings,
            ...(locationIds ? locationLossBindings : []),
            {
              name: 'excluded',
              view: excludedLocationCount,
              type: 'atomic<u32>',
              access: 'read_write'
            }
          ],
          body: `if (index >= stayCount[stayCountOffset]) { return; }
  ${
    locationIds
      ? `if (${locationLoss ? 'locationLoss[locationLossOffset] == 0u && ' : ''}locationIds[locationIdsOffset + index] != ${MOVEMENT_SENTINEL_WGSL}) { return; }`
      : ''
  }
  atomicAdd(&excluded[excludedOffset], 1u);`
        })
      );
    }
    const summarizeLocations =
      output.randomEntropy ||
      output.uncorrelatedEntropy ||
      output.homeLocations ||
      output.workLocations;
    if (summarizeLocations || output.realEntropy) {
      // When the column is omitted the count is zero and the sequence buffers are never read.
      const settledCapacity = locationIds ? Math.max(1, capacity) : 1;
      validateTrajectoryScratch(graph, id, {
        'settled location keys': settledCapacity * 4,
        'settled location durations': settledCapacity * 4,
        'settled location counts': trackCount * 4
      });
      const settledKeys = createTransientView(
        graph,
        `${id}-settled-location-keys`,
        'uint32',
        settledCapacity
      );
      const settledDurations = createTransientView(
        graph,
        `${id}-settled-location-durations`,
        'float32',
        settledCapacity
      );
      const settledCounts = createTransientView(
        graph,
        `${id}-settled-location-counts`,
        'uint32',
        trackCount
      );
      if (!locationIds) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-empty-location-keys`,
            operation: OPERATION,
            view: settledKeys,
            type: 'u32',
            value: '0u',
            componentCount: 1
          }),
          createFillNode<Parameters>(graph, {
            id: `${id}-empty-location-durations`,
            operation: OPERATION,
            view: settledDurations,
            type: 'f32',
            value: '0.0',
            componentCount: 1
          })
        );
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-filter-locations`,
          operation: OPERATION,
          invocationCount: trackCount,
          bindings: [
            ...locationIdBindings,
            ...(locationIds
              ? [
                  ...locationLossBindings,
                  {name: 'trackOffsets', view: trackOffsets, type: 'u32', access: 'read'} as const,
                  {
                    name: 'durations',
                    view: staypoints.durations,
                    type: 'f32',
                    access: 'read'
                  } as const
                ]
              : []),
            ...(locationIds
              ? [
                  {
                    name: 'settledKeys',
                    view: settledKeys,
                    type: 'u32',
                    access: 'read_write'
                  } as const,
                  {
                    name: 'settledDurations',
                    view: settledDurations,
                    type: 'f32',
                    access: 'read_write'
                  } as const
                ]
              : []),
            {name: 'settledCounts', view: settledCounts, type: 'u32', access: 'read_write'}
          ],
          body: locationIds
            ? `let begin = trackOffsets[trackOffsetsOffset + index];
  let end = trackOffsets[trackOffsetsOffset + index + 1u];
  var kept = 0u;
  ${locationLoss ? 'if (locationLoss[locationLossOffset] == 0u) {' : ''}
  for (var row = begin; row < end; row++) {
    let locationId = locationIds[locationIdsOffset + row];
    if (locationId == ${MOVEMENT_SENTINEL_WGSL}) { continue; }
    settledKeys[settledKeysOffset + begin + kept] = locationId;
    settledDurations[settledDurationsOffset + begin + kept] = durations[durationsOffset + row];
    kept++;
  }
  ${locationLoss ? '}' : ''}
  settledCounts[settledCountsOffset + index] = kept;`
            : `settledCounts[settledCountsOffset + index] = 0u;`
        })
      );
      if (summarizeLocations) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-locations`,
            operation: OPERATION,
            invocationCount: trackCount,
            bindings: [
              {name: 'locationIds', view: settledKeys, type: 'u32', access: 'read'},
              {name: 'durations', view: settledDurations, type: 'f32', access: 'read'},
              {name: 'trackOffsets', view: trackOffsets, type: 'u32', access: 'read'},
              {name: 'settledCounts', view: settledCounts, type: 'u32', access: 'read'},
              ...(output.randomEntropy
                ? [
                    {
                      name: 'randomEntropy',
                      view: output.randomEntropy,
                      type: 'f32',
                      access: 'read_write'
                    } as const
                  ]
                : []),
              ...(output.uncorrelatedEntropy
                ? [
                    {
                      name: 'uncorrelatedEntropy',
                      view: output.uncorrelatedEntropy,
                      type: 'f32',
                      access: 'read_write'
                    } as const
                  ]
                : []),
              ...(output.homeLocations
                ? [
                    {
                      name: 'homeLocations',
                      view: output.homeLocations,
                      type: 'u32',
                      access: 'read_write'
                    } as const
                  ]
                : []),
              ...(output.workLocations
                ? [
                    {
                      name: 'workLocations',
                      view: output.workLocations,
                      type: 'u32',
                      access: 'read_write'
                    } as const
                  ]
                : [])
            ],
            declarations: GET_NAN_WGSL,
            body: `let begin = trackOffsets[trackOffsetsOffset + index];
  let kept = settledCounts[settledCountsOffset + index];
  let end = begin + kept;
  let total = f32(kept);
  ${output.randomEntropy || output.uncorrelatedEntropy ? 'let allExcluded = kept == 0u && trackOffsets[trackOffsetsOffset + index + 1u] > begin;' : ''}
  var distinctCount = 0u;
  var entropy = 0.0;
  var bestDuration = -1.0;
  var bestLocation = ${MOVEMENT_SENTINEL_WGSL};
  var secondDuration = -1.0;
  var secondLocation = ${MOVEMENT_SENTINEL_WGSL};
  for (var row = begin; row < end; row++) {
    let locationId = locationIds[locationIdsOffset + row];
    var isFirst = true;
    for (var earlier = begin; earlier < row; earlier++) {
      if (locationIds[locationIdsOffset + earlier] == locationId) { isFirst = false; break; }
    }
    if (!isFirst) { continue; }
    distinctCount++;
    var count = 0u;
    var summed = 0.0;
    for (var other = row; other < end; other++) {
      if (locationIds[locationIdsOffset + other] == locationId) {
        count++;
        summed += durations[durationsOffset + other];
      }
    }
    let share = f32(count) / total;
    entropy -= share * log2(share);
    if (summed > bestDuration || (summed == bestDuration && locationId < bestLocation)) {
      secondDuration = bestDuration;
      secondLocation = bestLocation;
      bestDuration = summed;
      bestLocation = locationId;
    } else if (summed > secondDuration || (summed == secondDuration && locationId < secondLocation)) {
      secondDuration = summed;
      secondLocation = locationId;
    }
  }
  ${output.randomEntropy ? `randomEntropy[randomEntropyOffset + index] = select(log2(f32(max(1u, distinctCount))), getNaN(), allExcluded);` : ''}
  ${output.uncorrelatedEntropy ? `uncorrelatedEntropy[uncorrelatedEntropyOffset + index] = select(entropy, getNaN(), allExcluded);` : ''}
  ${output.homeLocations ? `homeLocations[homeLocationsOffset + index] = bestLocation;` : ''}
  ${output.workLocations ? `workLocations[workLocationsOffset + index] = secondLocation;` : ''}`
          })
        );
      }
      if (output.realEntropy) {
        if (output.status?.candidateOverflow) {
          nodes.push(
            createFillNode<Parameters>(graph, {
              id: `${id}-entropy-overflow-reset`,
              operation: OPERATION,
              view: output.status.candidateOverflow,
              type: 'u32',
              value: '0u',
              componentCount: 1
            })
          );
        }
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-real-entropy`,
            operation: OPERATION,
            invocationCount: trackCount,
            bindings: [
              {name: 'locationIds', view: settledKeys, type: 'u32', access: 'read'},
              {name: 'trackOffsets', view: trackOffsets, type: 'u32', access: 'read'},
              {name: 'settledCounts', view: settledCounts, type: 'u32', access: 'read'},
              {name: 'realEntropy', view: output.realEntropy, type: 'f32', access: 'read_write'},
              ...(output.status?.candidateOverflow
                ? [
                    {
                      name: 'candidateOverflow',
                      view: output.status.candidateOverflow,
                      type: 'atomic<u32>',
                      access: 'read_write'
                    } as const
                  ]
                : [])
            ],
            declarations: GET_NAN_WGSL,
            body: `let begin = trackOffsets[trackOffsetsOffset + index];
  let end = trackOffsets[trackOffsetsOffset + index + 1u];
  let sequenceLength = settledCounts[settledCountsOffset + index];
  var result = 0.0;
  if (sequenceLength == 0u && end > begin) {
    result = getNaN();
  } else if (sequenceLength > ${this.maximumEntropySequenceRows}u) {
    // Not computed: NaN, never the valid entropy 0, whether or not the status view is supplied.
    result = getNaN();
    ${output.status?.candidateOverflow ? 'atomicStore(&candidateOverflow[candidateOverflowOffset], 1u);' : ''}
  } else if (sequenceLength >= 2u) {
    // Scikit-mobility true_entropy: first and last elements contribute 1 and 2.
    var sumLambda = 3.0;
    for (var position = 1u; position + 1u < sequenceLength; position++) {
      var longestMatch = 0u;
      for (var candidate = 0u; candidate < position; candidate++) {
        var matchRows = 0u;
        while (candidate + matchRows < position && position + matchRows < sequenceLength &&
          locationIds[locationIdsOffset + begin + candidate + matchRows] == locationIds[locationIdsOffset + begin + position + matchRows]) {
          matchRows++;
        }
        longestMatch = max(longestMatch, matchRows);
      }
      // scikit-mobility: the first unseen length, plus one when the match reaches the sequence end.
      var symbolLength = longestMatch + 1u;
      if (longestMatch + 1u >= sequenceLength - position) { symbolLength = sequenceLength - position + 1u; }
      sumLambda += f32(symbolLength);
    }
    let n = f32(sequenceLength);
    result = n * log2(n) / sumLambda;
  }
  realEntropy[realEntropyOffset + index] = result;`
          })
        );
      }
    }
    if (output.trackingQuality && triplegs) {
      const lossBindings = [triplegs.status.overflow, triplegs.status.candidateOverflow].flatMap(
        (view, index) =>
          view
            ? [{name: `triplegLoss${index}`, view, type: 'u32' as const, access: 'read' as const}]
            : []
      );
      const hasLoss =
        lossBindings.map(binding => `${binding.name}[${binding.name}Offset] != 0u`).join(' || ') ||
        'false';
      const triplegCapacity = triplegs.tracks.length;
      const triplegOffsets = createTransientView(
        graph,
        `${id}-tripleg-offsets`,
        'uint32',
        trackCount + 1
      );
      nodes.push(
        createTrackTableOffsetsNode<Parameters>(graph, {
          id: `${id}-tripleg-offset-table`,
          operation: OPERATION,
          tracks: triplegs.tracks,
          count: triplegs.status.count,
          offsets: triplegOffsets,
          capacity: triplegCapacity,
          trackCount
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-tracking-quality`,
          operation: OPERATION,
          variant: timeMode,
          invocationCount: trackCount,
          bindings: [
            ...getTrajectoryTimeBindings({timestamps}),
            {name: 'stayStarts', view: staypoints.startRows, type: 'u32', access: 'read'},
            {name: 'stayFinishes', view: triplegs.staypointFinishRows, type: 'u32', access: 'read'},
            {name: 'stayOffsets', view: trackOffsets, type: 'u32', access: 'read'},
            {name: 'triplegStarts', view: triplegs.startRows, type: 'u32', access: 'read'},
            {name: 'triplegEnds', view: triplegs.endRows, type: 'u32', access: 'read'},
            {name: 'triplegOffsets', view: triplegOffsets, type: 'u32', access: 'read'},
            {
              name: 'trackingQuality',
              view: output.trackingQuality,
              type: 'f32',
              access: 'read_write'
            }
          ],
          declarations: getTrajectoryTimeSource(timeMode),
          body: `var tracked = 0.0;
  var firstStart = ${MOVEMENT_SENTINEL_WGSL};
  var lastFinish = ${MOVEMENT_SENTINEL_WGSL};
  for (var row = stayOffsets[stayOffsetsOffset + index]; row < stayOffsets[stayOffsetsOffset + index + 1u]; row++) {
    let start = stayStarts[stayStartsOffset + row];
    let finish = stayFinishes[stayFinishesOffset + row];
    if (timeDifferenceRows(finish, start) <= 0.0) { continue; }
    tracked += timeDifferenceRows(finish, start);
    if (firstStart == ${MOVEMENT_SENTINEL_WGSL} || timeDifferenceRows(start, firstStart) < 0.0) { firstStart = start; }
    if (lastFinish == ${MOVEMENT_SENTINEL_WGSL} || timeDifferenceRows(finish, lastFinish) > 0.0) { lastFinish = finish; }
  }
  for (var row = triplegOffsets[triplegOffsetsOffset + index]; row < triplegOffsets[triplegOffsetsOffset + index + 1u]; row++) {
    let start = triplegStarts[triplegStartsOffset + row];
    let finish = triplegEnds[triplegEndsOffset + row] - 1u;
    if (timeDifferenceRows(finish, start) <= 0.0) { continue; }
    tracked += timeDifferenceRows(finish, start);
    if (firstStart == ${MOVEMENT_SENTINEL_WGSL} || timeDifferenceRows(start, firstStart) < 0.0) { firstStart = start; }
    if (lastFinish == ${MOVEMENT_SENTINEL_WGSL} || timeDifferenceRows(finish, lastFinish) > 0.0) { lastFinish = finish; }
  }
  var quality = 0.0;
  if (firstStart != ${MOVEMENT_SENTINEL_WGSL} && lastFinish != ${MOVEMENT_SENTINEL_WGSL}) {
    let extent = timeDifferenceRows(lastFinish, firstStart);
    if (extent > 0.0) { quality = tracked / extent; }
  }
  trackingQuality[trackingQualityOffset + index] = quality;`
        })
      );
      if (lossBindings.length)
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-suppress-incomplete-quality`,
            operation: OPERATION,
            invocationCount: trackCount,
            bindings: [
              ...lossBindings,
              {
                name: 'trackingQuality',
                view: output.trackingQuality,
                type: 'f32',
                access: 'read_write'
              }
            ],
            declarations: GET_NAN_WGSL,
            body: `if (${hasLoss}) { trackingQuality[trackingQualityOffset + index] = getNaN(); }`
          })
        );
    }
    const stayLossBindings = [
      staypoints.status.overflow,
      staypoints.status.candidateOverflow
    ].flatMap((view, index) =>
      view ? [{name: `stayLoss${index}`, view, type: 'u32' as const, access: 'read' as const}] : []
    );
    const hasStayLoss =
      stayLossBindings
        .map(binding => `${binding.name}[${binding.name}Offset] != 0u`)
        .join(' || ') || 'false';
    const allLossBindings = [...stayLossBindings, ...locationLossBindings];
    const hasAnyLoss =
      allLossBindings.map(binding => `${binding.name}[${binding.name}Offset] != 0u`).join(' || ') ||
      'false';
    for (const port of [
      'radiusOfGyration',
      'jumpLengths',
      'waitingTimes',
      'randomEntropy',
      'uncorrelatedEntropy',
      'realEntropy',
      'homeLocations',
      'workLocations',
      'trackingQuality'
    ] as const) {
      const view = output[port];
      if (!view) continue;
      const location = port === 'homeLocations' || port === 'workLocations';
      const dependsOnLocation =
        location ||
        port === 'randomEntropy' ||
        port === 'uncorrelatedEntropy' ||
        port === 'realEntropy';
      const lossBindings = dependsOnLocation ? allLossBindings : stayLossBindings;
      if (!lossBindings.length) continue;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-suppress-incomplete-stays-${port}`,
          operation: OPERATION,
          invocationCount: view.length,
          bindings: [
            ...lossBindings,
            {name: 'values', view, type: location ? 'u32' : 'f32', access: 'read_write'}
          ],
          declarations: location ? '' : GET_NAN_WGSL,
          body: `if (${dependsOnLocation ? hasAnyLoss : hasStayLoss}) { values[valuesOffset + index] = ${location ? '0xffffffffu' : 'getNaN()'}; }`
        })
      );
    }
    if (output.status?.candidateOverflow) {
      if (!output.realEntropy)
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-candidate-overflow-reset`,
            operation: OPERATION,
            view: output.status.candidateOverflow,
            type: 'u32',
            value: '0u',
            componentCount: 1
          })
        );
      if (allLossBindings.length)
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-staypoint-loss-status`,
            operation: OPERATION,
            invocationCount: 1,
            bindings: [
              ...allLossBindings,
              {
                name: 'candidateOverflow',
                view: output.status.candidateOverflow,
                type: 'u32',
                access: 'read_write'
              }
            ],
            body: `if (${hasAnyLoss}) { candidateOverflow[candidateOverflowOffset] = 1u; }`
          })
        );
    }
    return nodes;
  }
}
