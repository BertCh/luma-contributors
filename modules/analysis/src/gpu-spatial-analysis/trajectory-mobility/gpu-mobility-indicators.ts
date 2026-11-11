// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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
  locationIds: GraphDataView<'uint32'>;
  /** One row in `count`: active staypoint count, as `GPUTrajectoryStaypoints` writes it. */
  status: {count: GPUStatusScalar};
};

/** Caller-owned outputs of {@link GPUMobilityIndicators}. At least one is required. */
export type GPUMobilityIndicatorsOutput = {
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
  /** Per-track real entropy (Lempel-Ziv estimator of the location sequence), bits. */
  realEntropy?: GraphDataView<'float32'>;
  /** Per-track location with the largest summed duration; `0xffffffff` for a track without staypoints. */
  homeLocations?: GraphDataView<'uint32'>;
  /** Per-track location with the second largest summed duration; `0xffffffff` when absent. */
  workLocations?: GraphDataView<'uint32'>;
  /** Per-track temporal tracking quality: tracked duration / extent (requires `triplegs`). */
  trackingQuality?: GraphDataView<'float32'>;
  /** `candidateOverflow` is 1 when a track's sequence exceeded `maximumEntropySequenceRows`. */
  status?: GPUCandidateStatusPort;
};

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
    status: {count: GPUStatusScalar};
    /** Staypoint `finishRows`. */
    staypointFinishRows: GraphDataView<'uint32'>;
  };
  /** Cap on a track's staypoint count for `realEntropy` (default 1024, at most 16384). */
  maximumEntropySequenceRows?: number;
  output: GPUMobilityIndicatorsOutput;
};

/**
 * Per-user mobility indicators on staypoints, following the definitions of scikit-mobility 1.3.1
 * (`individual` measures) and trackintel (`freq_method`, `temporal_tracking_quality`):
 *
 * - `radius_of_gyration`: root mean squared distance of the staypoints from their arithmetic mean
 *   position, haversine on `lat, lon` (no longitude wrap, as in scikit-mobility).
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
 */
export class GPUMobilityIndicators implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUMobilityIndicatorsProps;
  readonly spatialContext: GPUSpatialContext;
  readonly maximumEntropySequenceRows: number;

  constructor(props: GPUMobilityIndicatorsProps) {
    this.id = props.id ?? 'mobility-indicators';
    this.props = props;
    const id = this.id;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    const {staypoints, triplegs, output, timestamps} = props;
    validateCapacity(id, 'trackCount', props.trackCount);
    validateSinglePackedView(timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
    validateSinglePackedView(staypoints.tracks, ['uint32'], `${id} staypoints.tracks`);
    validateSinglePackedView(staypoints.centroids, ['float32x2'], `${id} staypoints.centroids`);
    validateSinglePackedView(staypoints.startRows, ['uint32'], `${id} staypoints.startRows`);
    validateSinglePackedView(staypoints.durations, ['float32'], `${id} staypoints.durations`);
    validateSinglePackedView(staypoints.locationIds, ['uint32'], `${id} staypoints.locationIds`);
    const capacity = staypoints.tracks.length;
    for (const view of [
      staypoints.centroids,
      staypoints.startRows,
      staypoints.durations,
      staypoints.locationIds
    ]) {
      validateViewLength(id, 'staypoint column', view, capacity);
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
    this.maximumEntropySequenceRows =
      props.maximumEntropySequenceRows ?? GPU_MOBILITY_DEFAULT_ENTROPY_SEQUENCE_ROWS;
    validateMaximumCount(
      id,
      'maximumEntropySequenceRows',
      this.maximumEntropySequenceRows,
      GPU_MOBILITY_MAXIMUM_ENTROPY_SEQUENCE_ROWS
    );
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
      triplegs?.tracks,
      triplegs?.startRows,
      triplegs?.endRows,
      triplegs?.status.count,
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
      output.status?.candidateOverflow
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
          declarations: distanceDeclarations,
          body: `let begin = trackOffsets[trackOffsetsOffset + index];
  let end = trackOffsets[trackOffsetsOffset + index + 1u];
  var result = 0.0;
  if (end > begin) {
    let first = stayPoint(begin);
    var sum = vec2<f32>(0.0, 0.0);
    for (var row = begin; row < end; row++) { sum += stayPoint(row) - first; }
    let center = first + sum / f32(end - begin);
    var squares = 0.0;
    for (var row = begin; row < end; row++) {
      let separation = pointDistance(stayPoint(row), center);
      squares += separation * separation;
    }
    result = sqrt(squares / f32(end - begin));
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
    if (
      output.randomEntropy ||
      output.uncorrelatedEntropy ||
      output.homeLocations ||
      output.workLocations
    ) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-locations`,
          operation: OPERATION,
          invocationCount: trackCount,
          bindings: [
            {name: 'locationIds', view: staypoints.locationIds, type: 'u32', access: 'read'},
            {name: 'durations', view: staypoints.durations, type: 'f32', access: 'read'},
            {name: 'trackOffsets', view: trackOffsets, type: 'u32', access: 'read'},
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
          body: `let begin = trackOffsets[trackOffsetsOffset + index];
  let end = trackOffsets[trackOffsetsOffset + index + 1u];
  let total = f32(end - begin);
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
  ${output.randomEntropy ? `randomEntropy[randomEntropyOffset + index] = select(0.0, log2(f32(distinctCount)), distinctCount > 0u);` : ''}
  ${output.uncorrelatedEntropy ? `uncorrelatedEntropy[uncorrelatedEntropyOffset + index] = entropy;` : ''}
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
            {name: 'locationIds', view: staypoints.locationIds, type: 'u32', access: 'read'},
            {name: 'trackOffsets', view: trackOffsets, type: 'u32', access: 'read'},
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
          body: `let begin = trackOffsets[trackOffsetsOffset + index];
  let end = trackOffsets[trackOffsetsOffset + index + 1u];
  let sequenceLength = end - begin;
  var result = 0.0;
  if (sequenceLength > ${this.maximumEntropySequenceRows}u) {
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
    if (output.trackingQuality && triplegs) {
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
    }
    return nodes;
  }
}
