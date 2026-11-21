// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  validateTrajectoryBindings,
  validateTrajectoryScratch,
  validateTrajectoryOutputRanges,
  validateTrajectoryWritableDispatches
} from '../trajectory-analysis/trajectory-binding-validation';
import {
  createTransientView,
  GPUCompaction,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {
  GPUStatusScalar,
  GPUTrajectoryGeometryFields
} from '../../analysis-core/contracts/index';
import type {GPUCompactOutput} from '../../analysis-core/kernels/gpu-contributor-types';
import {
  validateCapacity,
  validateCompactOutput,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createFillNode,
  createPublishNode,
  createWGSLKernelNode
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {resolveGPUTrajectoryTemporalContext} from '../trajectory-analysis/trajectory-common';
import {
  getTrajectoryTimeBindings,
  getTrajectoryTimeMode,
  getTrajectoryTimeSource
} from '../trajectory-analysis/trajectory-metrics-kernels';
import {
  createRowIdsNode,
  createTrackTableOffsetsNode,
  MOVEMENT_SENTINEL_WGSL
} from './movement-common';

const OPERATION = 'GPUTrajectorySegments';

/** Number of float32 elements in a segments parameter buffer. */
export const GPU_TRAJECTORY_SEGMENTS_PARAMETER_LENGTH = 4;

/** Per-frame settings of {@link GPUTrajectorySegments}. */
export type GPUTrajectorySegmentsSettings = {
  /**
   * Shared gap rule (`GPUTrajectoryGeometryPort`): a step strictly longer than this limit breaks a
   * tripleg and a trip (trackintel `gap_threshold`); 0 or absent means no limit.
   */
  maximumGap?: number;
  /**
   * A staypoint is an activity when its duration is strictly longer than this, in the timestamps'
   * unit (trackintel `create_activity_flag`). Finite and non-negative.
   */
  activityMinimumDuration: number;
};

/**
 * Packs segment settings into the float32 layout read by {@link GPUTrajectorySegments}.
 *
 * Layout: `[maximumGap, activityMinimumDuration, 0, 0]`.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If a value is negative or not finite.
 */
export function getGPUTrajectorySegmentsParameterValues(
  settings: GPUTrajectorySegmentsSettings,
  target: Float32Array = new Float32Array(GPU_TRAJECTORY_SEGMENTS_PARAMETER_LENGTH)
): Float32Array {
  const maximumGap = settings.maximumGap ?? 0;
  const {activityMinimumDuration} = settings;
  if (!Number.isFinite(Math.fround(maximumGap)) || maximumGap < 0) {
    throw new Error('Segment maximumGap must be finite and non-negative');
  }
  if (!Number.isFinite(Math.fround(activityMinimumDuration)) || activityMinimumDuration < 0) {
    throw new Error('Segment activityMinimumDuration must be finite and non-negative');
  }
  target.set([maximumGap, activityMinimumDuration, 0, 0]);
  return target;
}

/** Caller-owned tripleg table of {@link GPUTrajectorySegments}. `ids` hold the track of each tripleg. */
export type GPUTrajectoryTriplegOutput = GPUCompactOutput & {
  /** Optional first row of each tripleg. */
  startRows?: GraphDataView<'uint32'>;
  /** Optional row after the last row of each tripleg (exclusive). */
  endRows?: GraphDataView<'uint32'>;
  /** Optional per-row tripleg index, `positions.length` rows; other rows are `0xffffffff`. */
  rowTriplegIds?: GraphDataView<'uint32'>;
};

/**
 * Caller-owned trip table of {@link GPUTrajectorySegments}. `ids` hold the track of each trip.
 * Trips aggregate triplegs and non-activity staypoints between two activities, split at gaps
 * (trackintel `generate_trips`).
 */
export type GPUTrajectoryTripOutput = GPUCompactOutput & {
  /** Optional row of `started_at`: the first tripleg or non-activity staypoint of the trip. */
  startRows?: GraphDataView<'uint32'>;
  /** Optional row of `finished_at`. */
  finishRows?: GraphDataView<'uint32'>;
  /** Optional origin activity staypoint index; `0xffffffff` when the origin is unknown. */
  originStaypoints?: GraphDataView<'uint32'>;
  /** Optional destination activity staypoint index; `0xffffffff` when unknown. */
  destinationStaypoints?: GraphDataView<'uint32'>;
  /** Optional trip index of each tripleg, tripleg capacity rows; `0xffffffff` past the tripleg count. */
  triplegTripIds?: GraphDataView<'uint32'>;
};

/** Caller-owned outputs of {@link GPUTrajectorySegments}. */
export type GPUTrajectorySegmentsOutput = {
  triplegs: GPUTrajectoryTriplegOutput;
  /** Trips require status.candidateOverflow to expose loss from their bounded input table. */
  trips?: GPUTrajectoryTripOutput;
};

/**
 * Properties for {@link GPUTrajectorySegments}: the trajectory layout and the staypoint columns of
 * {@link GPUTrajectoryStaypoints}.
 */
export type GPUTrajectorySegmentsProps = GPUTrajectoryGeometryFields & {
  /** Prefix for node and transient IDs. Defaults to `'trajectory-segments'`. */
  id?: string;
  /** Per-row staypoint index (`GPUTrajectoryStaypoints` `staypointIds`). */
  staypointIds: GraphDataView<'uint32'>;
  /** Staypoint table columns, required for trips. */
  staypoints?: {
    tracks: GraphDataView<'uint32'>;
    startRows: GraphDataView<'uint32'>;
    finishRows: GraphDataView<'uint32'>;
    durations: GraphDataView<'float32'>;
    status: {
      count: GPUStatusScalar;
      overflow?: GPUStatusScalar;
      candidateOverflow?: GPUStatusScalar;
    };
  };
  /** Per-frame float32 view of {@link GPU_TRAJECTORY_SEGMENTS_PARAMETER_LENGTH} elements. */
  parameters: GraphDataView<'float32'>;
  output: GPUTrajectorySegmentsOutput;
};

/**
 * Triplegs and trips from positionfixes and staypoints, with trackintel's `between_staypoints`
 * tripleg rule and `generate_trips` state machine. Per-track trip counts use a public
 * hierarchical exclusive scan; a separate constant-work node publishes the terminal offset
 * and exact total. Scan nodes contribute their workload estimates to graph planning.
 *
 * A tripleg is a maximal run of rows outside every staypoint, broken wherever a step is strictly
 * longer than `maximumGap`, with at least two rows; a run that does not move (a tripleg whose
 * every coordinate is the same, an invalid LineString in trackintel) is dropped. A trip is formed
 * per track from the events (staypoints and triplegs) ordered by start: a new trip starts at the
 * first event of a track, after a gap between events, and at the last activity of an activity run;
 * activities are removed; a trip without a tripleg is dropped. Origin and destination are the
 * activity staypoints directly around the trip, unknown across a gap or at a track end.
 *
 * Trip events walk one track on one thread (`O(events)`), the same shape as staypoint
 * consolidation, in two passes: count, then write. The `overlap_staypoints` tripleg method is out
 * of scope.
 */
export class GPUTrajectorySegments implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUTrajectorySegmentsProps;

  constructor(props: GPUTrajectorySegmentsProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'positions',
      'timestamps',
      'temporalContext',
      'trackOffsets',
      'staypointIds',
      'staypoints',
      'parameters',
      'output'
    ]);
    const outputSnapshot = snapshotConstructionFields(props.output, ['triplegs', 'trips']);
    const snapshotTable = <Table extends GPUTrajectoryTriplegOutput>(table: Table): Table => {
      const captured = snapshotConstructionFields(table, [
        'ids',
        'status',
        'startRows',
        'endRows',
        'rowTriplegIds',
        'finishRows',
        'originStaypoints',
        'destinationStaypoints',
        'triplegTripIds'
      ] as (keyof Table)[]);
      return {
        ...captured,
        status: snapshotConstructionFields(captured.status, [
          'count',
          'overflow',
          'requiredCount',
          'candidateOverflow'
        ])
      };
    };
    const stays =
      props.staypoints &&
      snapshotConstructionFields(props.staypoints, [
        'tracks',
        'startRows',
        'finishRows',
        'durations',
        'status'
      ]);
    props = {
      ...props,
      output: {
        triplegs: snapshotTable(outputSnapshot.triplegs),
        trips: outputSnapshot.trips && snapshotTable(outputSnapshot.trips)
      },
      staypoints: stays && {
        ...stays,
        status: snapshotConstructionFields(stays.status, ['count', 'overflow', 'candidateOverflow'])
      }
    };
    this.id = props.id ?? 'trajectory-segments';
    this.props = props;
    const id = this.id;
    const {positions, timestamps, trackOffsets, staypointIds, parameters, output} = props;
    validateRequirement(
      id,
      !output.trips || output.trips.status.candidateOverflow !== undefined,
      'output.trips.status.candidateOverflow is required because triplegs can be truncated'
    );
    validateRequirement(
      id,
      !(
        props.staypoints?.status.overflow !== undefined ||
        props.staypoints?.status.candidateOverflow !== undefined
      ) || output.triplegs.status.candidateOverflow !== undefined,
      'output.triplegs.status.candidateOverflow is required with supplied staypoint loss'
    );
    validateSinglePackedView(positions, ['float32x2'], `${id} positions`);
    validateSinglePackedView(timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
    validateSinglePackedView(trackOffsets, ['uint32'], `${id} trackOffsets`);
    validateSinglePackedView(staypointIds, ['uint32'], `${id} staypointIds`);
    validateSinglePackedView(parameters, ['float32'], `${id} parameters`);
    validateViewLength(id, 'timestamps', timestamps, positions.length);
    validateViewLength(id, 'staypointIds', staypointIds, positions.length);
    resolveGPUTrajectoryTemporalContext(id, timestamps, props.temporalContext);
    validateRequirement(
      id,
      trackOffsets.length >= 1,
      'trackOffsets must contain at least one terminal row'
    );
    validateRequirement(
      id,
      parameters.length >= GPU_TRAJECTORY_SEGMENTS_PARAMETER_LENGTH,
      `parameters must hold ${GPU_TRAJECTORY_SEGMENTS_PARAMETER_LENGTH} float32 values`
    );
    validateCompactOutput(id, output.triplegs);
    validateCapacity(id, 'tripleg capacity', output.triplegs.ids.length);
    for (const [name, view] of [
      ['startRows', output.triplegs.startRows],
      ['endRows', output.triplegs.endRows]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} output.triplegs.${name}`);
        validateViewLength(id, `output.triplegs.${name}`, view, output.triplegs.ids.length);
      }
    }
    if (output.triplegs.rowTriplegIds) {
      validateSinglePackedView(
        output.triplegs.rowTriplegIds,
        ['uint32'],
        `${id} output.triplegs.rowTriplegIds`
      );
      validateViewLength(
        id,
        'output.triplegs.rowTriplegIds',
        output.triplegs.rowTriplegIds,
        positions.length
      );
    }
    const {trips} = output;
    if (trips) {
      validateRequirement(id, Boolean(props.staypoints), 'staypoints are required for trips');
      validateCompactOutput(id, trips);
      validateCapacity(id, 'trip capacity', trips.ids.length);
      for (const [name, view] of [
        ['startRows', trips.startRows],
        ['finishRows', trips.finishRows],
        ['originStaypoints', trips.originStaypoints],
        ['destinationStaypoints', trips.destinationStaypoints]
      ] as const) {
        if (view) {
          validateSinglePackedView(view, ['uint32'], `${id} output.trips.${name}`);
          validateViewLength(id, `output.trips.${name}`, view, trips.ids.length);
        }
      }
      if (trips.triplegTripIds) {
        validateSinglePackedView(
          trips.triplegTripIds,
          ['uint32'],
          `${id} output.trips.triplegTripIds`
        );
        validateViewLength(
          id,
          'output.trips.triplegTripIds',
          trips.triplegTripIds,
          output.triplegs.ids.length
        );
      }
    }
    validateTrajectoryOutputRanges(this.id, this.props.output);
    for (const [name, view] of Object.entries(props.staypoints?.status ?? {})) {
      if (!view) continue;
      validateSinglePackedView(view, ['uint32'], `${id} staypoints.status.${name}`);
      validateRequirement(
        id,
        view.length >= 1,
        `staypoints.status.${name} must have at least 1 row`
      );
    }
    validateTrajectoryWritableDispatches(id, [
      {
        ids: output.triplegs.ids,
        startRows: output.triplegs.startRows,
        endRows: output.triplegs.endRows
      },
      {...output.triplegs.status},
      ...(output.trips && props.staypoints
        ? [
            {
              startRows: output.trips.startRows,
              finishRows: output.trips.finishRows,
              originStaypoints: output.trips.originStaypoints,
              destinationStaypoints: output.trips.destinationStaypoints
            },
            {...output.trips.status}
          ]
        : [])
    ]);
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.triplegs.rowTriplegIds],
      [output.triplegs.startRows, output.triplegs.endRows]
    );
    if (output.trips && props.staypoints) {
      validateGraphOutputsDisjointFromInputs(
        id,
        [output.trips.triplegTripIds],
        [output.triplegs.startRows, output.trips.startRows, output.trips.finishRows]
      );
    }
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
  }

  private getInputViews(): (GraphDataView | undefined)[] {
    const {props} = this;
    return [
      props.positions,
      props.timestamps,
      props.trackOffsets,
      props.staypointIds,
      props.parameters,
      props.staypoints?.tracks,
      props.staypoints?.startRows,
      props.staypoints?.finishRows,
      props.staypoints?.durations,
      props.staypoints?.status.count,
      props.staypoints?.status.overflow,
      props.staypoints?.status.candidateOverflow
    ];
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {triplegs, trips} = this.props.output;
    return [
      triplegs.ids,
      triplegs.status.count,
      triplegs.status.overflow,
      triplegs.status.candidateOverflow,
      triplegs.status.requiredCount,
      triplegs.status.candidateOverflow,
      triplegs.startRows,
      triplegs.endRows,
      triplegs.rowTriplegIds,
      trips?.ids,
      trips?.status.count,
      trips?.status.overflow,
      trips?.status.candidateOverflow,
      trips?.status.requiredCount,
      trips?.status.candidateOverflow,
      trips?.startRows,
      trips?.finishRows,
      trips?.originStaypoints,
      trips?.destinationStaypoints,
      trips?.triplegTripIds
    ];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, trackOffsets, staypointIds, parameters, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    const rowCount = positions.length;
    const trackCount = trackOffsets.length - 1;
    const triplegCapacity = output.triplegs.ids.length;
    const stayLossSources = [
      props.staypoints?.status.overflow,
      props.staypoints?.status.candidateOverflow
    ].filter((view): view is GraphDataView<'uint32'> => Boolean(view));
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'positions',
      rowCount * 8,
      'split the tracks over several graphs'
    );
    validateTrajectoryBindings(graph, this.id, this.props);
    validateTrajectoryScratch(graph, id, {
      'track offsets': (trackCount + 1) * 4,
      ...(output.trips && props.staypoints
        ? {
            'trip records': output.trips.ids.length * 16,
            'merged trip events': (props.staypoints.startRows.length + triplegCapacity) * 4
          }
        : {})
    });
    const timeMode = getTrajectoryTimeMode(props);
    const timeSource = getTrajectoryTimeSource(timeMode);

    const startFlags = createTransientView(graph, `${id}-tripleg-flags`, 'uint32', rowCount);
    const rowEnds = createTransientView(graph, `${id}-tripleg-ends`, 'uint32', rowCount);
    const rowTracks = createTransientView(graph, `${id}-row-tracks`, 'uint32', rowCount);
    const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', rowCount);
    const compact = createTransientView(graph, `${id}-tripleg-rows`, 'uint32', rowCount);
    const triplegTotal = createTransientView(graph, `${id}-tripleg-total`, 'uint32', 1);
    const triplegStarts =
      output.triplegs.startRows ??
      createTransientView(graph, `${id}-tripleg-starts`, 'uint32', triplegCapacity);
    const triplegEnds =
      output.triplegs.endRows ??
      createTransientView(graph, `${id}-tripleg-ends-out`, 'uint32', triplegCapacity);

    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-reset-tripleg-flags`,
        operation: OPERATION,
        invocationCount: rowCount,
        bindings: [
          {name: 'startFlags', view: startFlags, type: 'u32', access: 'read_write'},
          {name: 'rowEnds', view: rowEnds, type: 'u32', access: 'read_write'},
          {name: 'rowTracks', view: rowTracks, type: 'u32', access: 'read_write'}
        ],
        body: `startFlags[startFlagsOffset + index] = 0u;
  rowEnds[rowEndsOffset + index] = ${MOVEMENT_SENTINEL_WGSL};
  rowTracks[rowTracksOffset + index] = ${MOVEMENT_SENTINEL_WGSL};`
      }),
      // One serial thread per track; O(track rows). Tripleg rows are disjoint.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-tripleg-scan`,
        operation: OPERATION,
        variant: timeMode,
        invocationCount: trackCount,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          ...getTrajectoryTimeBindings(props),
          {name: 'trackOffsets', view: trackOffsets, type: 'u32', access: 'read'},
          {name: 'staypointIds', view: staypointIds, type: 'u32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          {name: 'startFlags', view: startFlags, type: 'u32', access: 'read_write'},
          {name: 'rowEnds', view: rowEnds, type: 'u32', access: 'read_write'},
          {name: 'rowTracks', view: rowTracks, type: 'u32', access: 'read_write'}
        ],
        declarations: `${timeSource}
fn isGapStep(row: u32) -> bool {
  let maximumGap = parameters[parametersOffset];
  return maximumGap > 0.0 && timeDifferenceRows(row, row - 1u) > maximumGap;
}
fn isOutsideStaypoint(row: u32) -> bool {
  return staypointIds[staypointIdsOffset + row] == ${MOVEMENT_SENTINEL_WGSL};
}`,
        body: `let begin = min(trackOffsets[trackOffsetsOffset + index], ${rowCount}u);
  let end = min(trackOffsets[trackOffsetsOffset + index + 1u], ${rowCount}u);
  var row = begin;
  while (row < end) {
    if (!isOutsideStaypoint(row)) { row++; continue; }
    var runEnd = row + 1u;
    while (runEnd < end && isOutsideStaypoint(runEnd) && !isGapStep(runEnd)) { runEnd++; }
    var isValid = runEnd - row >= 2u;
    if (isValid) {
      var moves = false;
      for (var scan = row + 1u; scan < runEnd; scan++) {
        if (positions[positionsOffset + 2u * scan] != positions[positionsOffset + 2u * row] || positions[positionsOffset + 2u * scan + 1u] != positions[positionsOffset + 2u * row + 1u]) { moves = true; break; }
      }
      isValid = moves;
    }
    if (isValid) {
      startFlags[startFlagsOffset + row] = 1u;
      rowEnds[rowEndsOffset + row] = runEnd;
      for (var member = row; member < runEnd; member++) { rowTracks[rowTracksOffset + member] = index; }
    }
    row = runEnd;
  }`
      }),
      createRowIdsNode<Parameters>(graph, {id: `${id}-row-ids`, operation: OPERATION, rowIds}),
      ...new GPUCompaction({
        id: `${id}-compact`,
        input: rowIds,
        flags: startFlags,
        output: compact,
        count: triplegTotal
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-tripleg-table`,
        operation: OPERATION,
        invocationCount: triplegCapacity,
        bindings: [
          {name: 'compact', view: compact, type: 'u32', access: 'read'},
          {name: 'total', view: triplegTotal, type: 'u32', access: 'read'},
          {name: 'rowEnds', view: rowEnds, type: 'u32', access: 'read'},
          {name: 'rowTracks', view: rowTracks, type: 'u32', access: 'read'},
          {name: 'ids', view: output.triplegs.ids, type: 'u32', access: 'read_write'},
          {name: 'starts', view: triplegStarts, type: 'u32', access: 'read_write'},
          {name: 'ends', view: triplegEnds, type: 'u32', access: 'read_write'}
        ],
        body: `if (index < total[totalOffset]) {
    let row = compact[compactOffset + index];
    ids[idsOffset + index] = rowTracks[rowTracksOffset + row];
    starts[startsOffset + index] = row;
    ends[endsOffset + index] = rowEnds[rowEndsOffset + row];
  } else {
    ids[idsOffset + index] = ${MOVEMENT_SENTINEL_WGSL};
    starts[startsOffset + index] = ${MOVEMENT_SENTINEL_WGSL};
    ends[endsOffset + index] = ${MOVEMENT_SENTINEL_WGSL};
  }`
      }),
      createPublishNode<Parameters>(graph, {
        id: `${id}-tripleg-publish`,
        operation: OPERATION,
        requiredCount: triplegTotal,
        output: {
          ...output.triplegs,
          status: output.triplegs.status
        },
        candidateOverflowSources: stayLossSources
      })
    ];
    if (output.triplegs.rowTriplegIds) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-row-tripleg-reset`,
          operation: OPERATION,
          view: output.triplegs.rowTriplegIds,
          type: 'u32',
          value: MOVEMENT_SENTINEL_WGSL
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-row-triplegs`,
          operation: OPERATION,
          invocationCount: triplegCapacity,
          bindings: [
            {name: 'total', view: triplegTotal, type: 'u32', access: 'read'},
            {name: 'starts', view: triplegStarts, type: 'u32', access: 'read'},
            {name: 'ends', view: triplegEnds, type: 'u32', access: 'read'},
            {
              name: 'rowTriplegIds',
              view: output.triplegs.rowTriplegIds,
              type: 'u32',
              access: 'read_write'
            }
          ],
          body: `if (index < min(total[totalOffset], ${triplegCapacity}u)) {
    for (var row = starts[startsOffset + index]; row < ends[endsOffset + index]; row++) {
      rowTriplegIds[rowTriplegIdsOffset + row] = index;
    }
  }`
        })
      );
    }
    if (output.trips && props.staypoints) {
      nodes.push(
        ...this.createTripNodes(graph, {
          trackCount,
          rowCount,
          triplegTotal,
          triplegStarts,
          triplegEnds
        })
      );
    }
    return nodes;
  }

  private createTripNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    context: {
      trackCount: number;
      rowCount: number;
      triplegTotal: GraphDataView<'uint32'>;
      triplegStarts: GraphDataView<'uint32'>;
      triplegEnds: GraphDataView<'uint32'>;
    }
  ): GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const trips = props.output.trips!;
    const staypoints = props.staypoints!;
    const {trackCount, triplegTotal, triplegStarts, triplegEnds} = context;
    const triplegTracks = props.output.triplegs.ids;
    const stayCapacity = staypoints.startRows.length;
    const triplegCapacity = triplegStarts.length;
    const eventCapacity = stayCapacity + triplegCapacity;
    const timeMode = getTrajectoryTimeMode(props);
    const timeSource = getTrajectoryTimeSource(timeMode);
    const trackStayOffsets = createTransientView(
      graph,
      `${id}-track-stay-offsets`,
      'uint32',
      trackCount + 1
    );
    const trackTriplegOffsets = createTransientView(
      graph,
      `${id}-track-tripleg-offsets`,
      'uint32',
      trackCount + 1
    );
    const trackEventCounts = createTransientView(
      graph,
      `${id}-track-event-counts`,
      'uint32',
      trackCount + 1
    );
    const eventIdentifiers = createTransientView(graph, `${id}-events`, 'uint32', eventCapacity);
    const eventStartRows = createTransientView(
      graph,
      `${id}-event-starts`,
      'uint32',
      eventCapacity
    );
    const eventFinishRows = createTransientView(
      graph,
      `${id}-event-finishes`,
      'uint32',
      eventCapacity
    );
    const trackTripOffsets = createTransientView(
      graph,
      `${id}-track-trip-offsets`,
      'uint32',
      trackCount + 1
    );
    const tripCounts = createTransientView(graph, `${id}-track-trip-counts`, 'uint32', trackCount);
    const tripTotal = createTransientView(graph, `${id}-trip-total`, 'uint32', 1);
    const tripRecords = createTransientView(
      graph,
      `${id}-trip-records`,
      'uint32',
      trips.ids.length * 4
    );

    const events = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-merge-events`,
      operation: OPERATION,
      invocationCount: trackCount,
      bindings: [
        {name: 'stayStarts', view: staypoints.startRows, type: 'u32', access: 'read'},
        {name: 'stayDurations', view: staypoints.durations, type: 'f32', access: 'read'},
        {name: 'triplegStarts', view: triplegStarts, type: 'u32', access: 'read'},
        {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
        {name: 'trackStayOffsets', view: trackStayOffsets, type: 'u32', access: 'read'},
        {name: 'trackTriplegOffsets', view: trackTriplegOffsets, type: 'u32', access: 'read'},
        {name: 'eventIdentifiers', view: eventIdentifiers, type: 'u32', access: 'read_write'},
        {name: 'trackEventCounts', view: trackEventCounts, type: 'u32', access: 'read_write'}
      ],
      // eventId = (id << 2) | (kind << 1) | activity; kind 0 staypoint, 1 tripleg.
      body: `let activityThreshold = parameters[parametersOffset + 1u];
  let stayEnd = trackStayOffsets[trackStayOffsetsOffset + index + 1u];
  let triplegEnd = trackTriplegOffsets[trackTriplegOffsetsOffset + index + 1u];
  var stayCursor = trackStayOffsets[trackStayOffsetsOffset + index];
  var triplegCursor = trackTriplegOffsets[trackTriplegOffsetsOffset + index];
  var event = stayCursor + triplegCursor;
  trackEventCounts[trackEventCountsOffset + index] = event;
  while (stayCursor < stayEnd || triplegCursor < triplegEnd) {
    let takeStay = triplegCursor >= triplegEnd || (stayCursor < stayEnd && stayStarts[stayStartsOffset + stayCursor] <= triplegStarts[triplegStartsOffset + triplegCursor]);
    if (takeStay) {
      let isActivity = stayDurations[stayDurationsOffset + stayCursor] > activityThreshold;
      eventIdentifiers[eventIdentifiersOffset + event] = (stayCursor << 2u) | select(0u, 1u, isActivity);
      stayCursor++;
    } else {
      eventIdentifiers[eventIdentifiersOffset + event] = (triplegCursor << 2u) | 2u;
      triplegCursor++;
    }
    event++;
  }
  if (index == ${trackCount - 1}u) { trackEventCounts[trackEventCountsOffset + ${trackCount}u] = event; }`
    });
    const eventColumns = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-event-rows`,
      operation: OPERATION,
      invocationCount: eventCapacity,
      bindings: [
        {name: 'eventIdentifiers', view: eventIdentifiers, type: 'u32', access: 'read'},
        {name: 'stayStarts', view: staypoints.startRows, type: 'u32', access: 'read'},
        {name: 'stayFinishes', view: staypoints.finishRows, type: 'u32', access: 'read'},
        {name: 'triplegStarts', view: triplegStarts, type: 'u32', access: 'read'},
        {name: 'triplegEnds', view: triplegEnds, type: 'u32', access: 'read'},
        {name: 'eventCount', view: trackEventCounts, type: 'u32', access: 'read'},
        {name: 'eventStartRows', view: eventStartRows, type: 'u32', access: 'read_write'},
        {name: 'eventFinishRows', view: eventFinishRows, type: 'u32', access: 'read_write'}
      ],
      body: `if (index < eventCount[eventCountOffset + ${trackCount}u]) {
    let packed = eventIdentifiers[eventIdentifiersOffset + index];
    let identifier = packed >> 2u;
    if ((packed & 2u) != 0u) {
      eventStartRows[eventStartRowsOffset + index] = triplegStarts[triplegStartsOffset + identifier];
      eventFinishRows[eventFinishRowsOffset + index] = triplegEnds[triplegEndsOffset + identifier] - 1u;
    } else {
      eventStartRows[eventStartRowsOffset + index] = stayStarts[stayStartsOffset + identifier];
      eventFinishRows[eventFinishRowsOffset + index] = stayFinishes[stayFinishesOffset + identifier];
    }
  }`
    });
    // The shared WGSL of the trip walk: emit=false counts trips, emit=true writes records.
    const walkSource = `${timeSource}
struct TripState { count: u32, activeTripleg: bool, nonActivityCount: u32, firstStart: u32, lastFinish: u32, origin: u32, hasDestination: bool, destination: u32, openDestination: bool }
fn isEventActivity(event: u32) -> bool { return (eventIdentifiers[eventIdentifiersOffset + event] & 1u) != 0u; }
fn isEventTripleg(event: u32) -> bool { return (eventIdentifiers[eventIdentifiersOffset + event] & 2u) != 0u; }
fn eventId(event: u32) -> u32 { return eventIdentifiers[eventIdentifiersOffset + event] >> 2u; }
fn isGapBetween(a: u32, b: u32) -> bool {
  let maximumGap = parameters[parametersOffset];
  return maximumGap > 0.0 && timeDifferenceRows(eventStartRows[eventStartRowsOffset + b], eventFinishRows[eventFinishRowsOffset + a]) > maximumGap;
}`;
    const walkBody = (
      emit: boolean
    ) => `let firstEvent = trackEventCounts[trackEventCountsOffset + index];
  let lastEvent = trackEventCounts[trackEventCountsOffset + index + 1u];
  var state: TripState;
  state.count = 0u; state.activeTripleg = false; state.nonActivityCount = 0u;
  state.origin = ${MOVEMENT_SENTINEL_WGSL}; state.destination = ${MOVEMENT_SENTINEL_WGSL}; state.hasDestination = false; state.openDestination = false;
  state.firstStart = ${MOVEMENT_SENTINEL_WGSL}; state.lastFinish = ${MOVEMENT_SENTINEL_WGSL};
  ${emit ? `let baseRow = trackTripOffsets[trackTripOffsetsOffset + index];` : ''}
  for (var event = firstEvent; event < lastEvent; event++) {
    let isActivity = isEventActivity(event);
    let previousIsActivity = event > firstEvent && isEventActivity(event - 1u);
    let nextIsActivity = event + 1u < lastEvent && isEventActivity(event + 1u);
    let isGapBefore = event > firstEvent && isGapBetween(event - 1u, event);
    let isLastActivity = isActivity && !nextIsActivity;
    let isFirstOfTrack = event == firstEvent;
    let startsTrip = isFirstOfTrack || isGapBefore || isLastActivity;
    // A destination pending from the previous non-activity event resolves on this event.
    if (state.openDestination) {
      state.openDestination = false;
      if (isActivity && !isGapBefore) { state.destination = eventId(event); }
      else { state.destination = ${MOVEMENT_SENTINEL_WGSL}; }
    }
    if (startsTrip && event > firstEvent) {
      if (state.activeTripleg) {
        ${
          emit
            ? `let row = baseRow + state.count;
        if (row < ${trips.ids.length}u) {
          tripRecords[tripRecordsOffset + 4u * row] = state.firstStart;
          tripRecords[tripRecordsOffset + 4u * row + 1u] = state.lastFinish;
          tripRecords[tripRecordsOffset + 4u * row + 2u] = state.origin;
          tripRecords[tripRecordsOffset + 4u * row + 3u] = state.destination;
        }`
            : ''
        }
        state.count++;
      }
      state.activeTripleg = false;
      state.nonActivityCount = 0u;
      state.destination = ${MOVEMENT_SENTINEL_WGSL};
      state.origin = ${MOVEMENT_SENTINEL_WGSL};
    }
    if (!isActivity) {
      if (state.nonActivityCount == 0u) {
        state.firstStart = eventStartRows[eventStartRowsOffset + event];
        state.origin = select(${MOVEMENT_SENTINEL_WGSL}, eventId(event - 1u), event > firstEvent && previousIsActivity && !isGapBefore);
      }
      state.nonActivityCount++;
      state.lastFinish = eventFinishRows[eventFinishRowsOffset + event];
      if (isEventTripleg(event)) { state.activeTripleg = true; }
      state.openDestination = true;
    }
  }
  if (state.activeTripleg) {
    if (state.openDestination) { state.destination = ${MOVEMENT_SENTINEL_WGSL}; }
    ${
      emit
        ? `let row = baseRow + state.count;
    if (row < ${trips.ids.length}u) {
      tripRecords[tripRecordsOffset + 4u * row] = state.firstStart;
      tripRecords[tripRecordsOffset + 4u * row + 1u] = state.lastFinish;
      tripRecords[tripRecordsOffset + 4u * row + 2u] = state.origin;
      tripRecords[tripRecordsOffset + 4u * row + 3u] = state.destination;
    }`
        : ''
    }
    state.count++;
  }
  ${emit ? '' : 'tripCounts[tripCountsOffset + index] = state.count;'}`;
    const walkBindings = [
      ...getTrajectoryTimeBindings(props),
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'} as const,
      {name: 'trackEventCounts', view: trackEventCounts, type: 'u32', access: 'read'} as const,
      {name: 'eventIdentifiers', view: eventIdentifiers, type: 'u32', access: 'read'} as const,
      {name: 'eventStartRows', view: eventStartRows, type: 'u32', access: 'read'} as const,
      {name: 'eventFinishRows', view: eventFinishRows, type: 'u32', access: 'read'} as const
    ];
    const nodes: GPUCommandNode<Parameters>[] = [
      createTrackTableOffsetsNode<Parameters>(graph, {
        id: `${id}-offsets-stays`,
        operation: OPERATION,
        tracks: staypoints.tracks,
        count: staypoints.status.count,
        offsets: trackStayOffsets,
        capacity: stayCapacity,
        trackCount
      }),
      createTrackTableOffsetsNode<Parameters>(graph, {
        id: `${id}-offsets-triplegs`,
        operation: OPERATION,
        tracks: triplegTracks,
        count: triplegTotal,
        offsets: trackTriplegOffsets,
        capacity: triplegCapacity,
        trackCount
      }),
      events,
      eventColumns,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-trip-count`,
        operation: OPERATION,
        variant: timeMode,
        invocationCount: trackCount,
        bindings: [
          ...walkBindings,
          {name: 'tripCounts', view: tripCounts, type: 'u32', access: 'read_write'}
        ],
        declarations: walkSource,
        body: walkBody(false)
      }),
      ...new GPUScan({
        id: `${id}-trip-offset-scan`,
        input: tripCounts,
        output: trackTripOffsets,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-trip-offsets`,
        operation: OPERATION,
        invocationCount: 1,
        bindings: [
          ...(trackCount > 0
            ? [{name: 'tripCounts', view: tripCounts, type: 'u32', access: 'read'} as const]
            : []),
          {name: 'trackTripOffsets', view: trackTripOffsets, type: 'u32', access: 'read_write'},
          {name: 'tripTotal', view: tripTotal, type: 'u32', access: 'read_write'}
        ],
        body: `let required = ${trackCount > 0 ? `trackTripOffsets[trackTripOffsetsOffset + ${trackCount - 1}u] + tripCounts[tripCountsOffset + ${trackCount - 1}u]` : '0u'};
  trackTripOffsets[trackTripOffsetsOffset + ${trackCount}u] = required;
  tripTotal[tripTotalOffset] = required;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-trip-write`,
        operation: OPERATION,
        variant: timeMode,
        invocationCount: trackCount,
        bindings: [
          ...walkBindings,
          {name: 'trackTripOffsets', view: trackTripOffsets, type: 'u32', access: 'read'},
          {name: 'tripRecords', view: tripRecords, type: 'u32', access: 'read_write'}
        ],
        declarations: walkSource,
        body: walkBody(true)
      }),
      ...this.createTripTableNodes(graph, {
        tripTotal,
        tripRecords,
        trackTripOffsets,
        triplegStarts,
        triplegTotal
      })
    ];
    return nodes;
  }

  private createTripTableNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    context: {
      tripTotal: GraphDataView<'uint32'>;
      tripRecords: GraphDataView<'uint32'>;
      trackTripOffsets: GraphDataView<'uint32'>;
      triplegStarts: GraphDataView<'uint32'>;
      triplegTotal: GraphDataView<'uint32'>;
    }
  ): GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const trips = props.output.trips!;
    const {tripTotal, tripRecords, trackTripOffsets, triplegStarts, triplegTotal} = context;
    const triplegCapacity = props.output.triplegs.ids.length;
    const triplegLoss = createTransientView(graph, `${id}-trip-input-loss`, 'uint32', 1);
    const stayLossSources = [
      props.staypoints?.status.overflow,
      props.staypoints?.status.candidateOverflow
    ].filter((view): view is GraphDataView<'uint32'> => Boolean(view));
    const capacity = trips.ids.length;
    const trackCount = props.trackOffsets.length - 1;
    const startRows =
      trips.startRows ?? createTransientView(graph, `${id}-trip-starts`, 'uint32', capacity);
    const finishRows =
      trips.finishRows ?? createTransientView(graph, `${id}-trip-finishes`, 'uint32', capacity);
    const origins =
      trips.originStaypoints ??
      createTransientView(graph, `${id}-trip-origins`, 'uint32', capacity);
    const destinations =
      trips.destinationStaypoints ??
      createTransientView(graph, `${id}-trip-destinations`, 'uint32', capacity);
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-trip-table`,
        operation: OPERATION,
        invocationCount: capacity,
        bindings: [
          {name: 'tripRecords', view: tripRecords, type: 'u32', access: 'read'},
          {name: 'tripTotal', view: tripTotal, type: 'u32', access: 'read'},
          {name: 'starts', view: startRows, type: 'u32', access: 'read_write'},
          {name: 'finishes', view: finishRows, type: 'u32', access: 'read_write'},
          {name: 'origins', view: origins, type: 'u32', access: 'read_write'},
          {name: 'destinations', view: destinations, type: 'u32', access: 'read_write'}
        ],
        body: `if (index < tripTotal[tripTotalOffset]) {
    starts[startsOffset + index] = tripRecords[tripRecordsOffset + 4u * index];
    finishes[finishesOffset + index] = tripRecords[tripRecordsOffset + 4u * index + 1u];
    origins[originsOffset + index] = tripRecords[tripRecordsOffset + 4u * index + 2u];
    destinations[destinationsOffset + index] = tripRecords[tripRecordsOffset + 4u * index + 3u];
  } else {
    starts[startsOffset + index] = ${MOVEMENT_SENTINEL_WGSL};
    finishes[finishesOffset + index] = ${MOVEMENT_SENTINEL_WGSL};
    origins[originsOffset + index] = ${MOVEMENT_SENTINEL_WGSL};
    destinations[destinationsOffset + index] = ${MOVEMENT_SENTINEL_WGSL};
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-trip-ids`,
        operation: OPERATION,
        invocationCount: capacity,
        bindings: [
          {name: 'tripTotal', view: tripTotal, type: 'u32', access: 'read'},
          {name: 'trackTripOffsets', view: trackTripOffsets, type: 'u32', access: 'read'},
          {name: 'ids', view: trips.ids, type: 'u32', access: 'read_write'}
        ],
        body: `if (index < tripTotal[tripTotalOffset]) {
    var low = 0u;
    var high = ${trackCount}u;
    while (low + 1u < high) {
      let middle = (low + high) / 2u;
      if (trackTripOffsets[trackTripOffsetsOffset + middle] <= index) { low = middle; } else { high = middle; }
    }
    ids[idsOffset + index] = low;
  } else {
    ids[idsOffset + index] = ${MOVEMENT_SENTINEL_WGSL};
  }`
      }),
      createPublishNode<Parameters>(graph, {
        id: `${id}-trip-publish`,
        operation: OPERATION,
        requiredCount: tripTotal,
        output: {
          ...trips,
          status: trips.status
        },
        candidateOverflowSources: [triplegLoss, ...stayLossSources]
      })
    ];
    // Only the stored tripleg prefix participates in trip assembly. Its lost rows are
    // intermediate loss for trips, independently of the final trip capacity.
    nodes.unshift(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-trip-input-loss`,
        operation: OPERATION,
        invocationCount: 1,
        bindings: [
          {name: 'triplegTotal', view: triplegTotal, type: 'u32', access: 'read'},
          {name: 'loss', view: triplegLoss, type: 'u32', access: 'read_write'}
        ],
        body: `loss[lossOffset] = select(0u, 1u, triplegTotal[triplegTotalOffset] > ${triplegCapacity}u);`
      })
    );
    if (trips.triplegTripIds) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-tripleg-trips`,
          operation: OPERATION,
          invocationCount: props.output.triplegs.ids.length,
          bindings: [
            {name: 'triplegStarts', view: triplegStarts, type: 'u32', access: 'read'},
            {name: 'triplegTotal', view: triplegTotal, type: 'u32', access: 'read'},
            {name: 'tripTotal', view: tripTotal, type: 'u32', access: 'read'},
            {name: 'tripStarts', view: startRows, type: 'u32', access: 'read'},
            {name: 'tripFinishes', view: finishRows, type: 'u32', access: 'read'},
            {name: 'triplegTripIds', view: trips.triplegTripIds, type: 'u32', access: 'read_write'}
          ],
          body: `var trip = ${MOVEMENT_SENTINEL_WGSL};
  if (index < min(triplegTotal[triplegTotalOffset], ${props.output.triplegs.ids.length}u)) {
    let row = triplegStarts[triplegStartsOffset + index];
    let total = min(tripTotal[tripTotalOffset], ${capacity}u);
    var low = 0u;
    var high = total;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (tripStarts[tripStartsOffset + middle] <= row) { low = middle + 1u; } else { high = middle; }
    }
    if (low > 0u && tripStarts[tripStartsOffset + low - 1u] <= row && row <= tripFinishes[tripFinishesOffset + low - 1u]) { trip = low - 1u; }
  }
  triplegTripIds[triplegTripIdsOffset + index] = trip;`
        })
      );
    }
    return nodes;
  }
}
