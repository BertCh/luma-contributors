// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateTrajectoryBindings,
  validateTrajectoryWritableDispatches,
  validateTrajectoryOutputRanges
} from '../trajectory-analysis/trajectory-binding-validation';
import {
  createTransientView,
  GPUCompaction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius,
  resolveGPUSpatialContext,
  warnGPUShortLongitudeLatitudeDistance,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {
  GPUCandidateStatusPort,
  GPUTrajectoryGeometryFields
} from '../../analysis-core/contracts/index';
import type {GPUCompactOutput} from '../../analysis-core/kernels/gpu-contributor-types';
import {
  validateCapacity,
  validateCompactOutput,
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
  createChainReachNodes,
  createRowBoundsNodes,
  createRowIdsNode,
  getMovementDistanceWGSL,
  MOVEMENT_SENTINEL_WGSL
} from './movement-common';

const OPERATION = 'GPUTrajectoryStaypoints';

/** Spatial contexts of this distance-measuring operation. */
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);

/** Number of float32 elements in a staypoint parameter buffer. */
export const GPU_TRAJECTORY_STAYPOINTS_PARAMETER_LENGTH = 4;

/** Default compile-time forward search cap per anchor, in rows. */
export const GPU_TRAJECTORY_STAYPOINTS_DEFAULT_MAXIMUM_SEARCH_ROWS = 4096;

/** Largest accepted forward search cap per anchor, in rows. */
export const GPU_TRAJECTORY_STAYPOINTS_MAXIMUM_SEARCH_ROWS = 1 << 24;

/** Per-frame settings of {@link GPUTrajectoryStaypoints}. */
export type GPUTrajectoryStaypointsSettings = {
  /** Spatial context of the positions, for the short-distance warning only. */
  spatialContext?: GPUSpatialContext;
  /**
   * Distance at or beyond which a fix leaves the current staypoint (trackintel `dist_threshold`),
   * in position units, or the context length unit for great-circle. Finite and positive.
   */
  distanceThreshold: number;
  /**
   * Smallest staypoint duration (`finished - started`, trackintel `time_threshold`) in the
   * timestamps' unit. Finite and non-negative.
   */
  timeThreshold: number;
  /**
   * Shared gap rule (`GPUTrajectoryGeometryPort`): a step is a gap only when strictly longer than
   * this limit, in the timestamps' unit; 0 or absent means no limit. A gap restarts the search at
   * the fix after it without emitting a staypoint (trackintel `gap_threshold`).
   */
  maximumGap?: number;
  /** Emit a staypoint for the window still open at the end of a track (trackintel `include_last`). */
  includeLast?: boolean;
};

/**
 * Packs staypoint settings into the 4-element float32 layout read by {@link GPUTrajectoryStaypoints}.
 *
 * Layout: `[distanceThreshold, timeThreshold, maximumGap, includeLast]`, `includeLast` as 0 or 1.
 * Write the result into a `GPUParameterBuffer` between encodings to change thresholds without
 * recompiling.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements. A new array is returned when omitted.
 * @throws If a threshold is out of range or `target` is too short.
 */
export function getGPUTrajectoryStaypointsParameterValues(
  settings: GPUTrajectoryStaypointsSettings,
  target: Float32Array = new Float32Array(GPU_TRAJECTORY_STAYPOINTS_PARAMETER_LENGTH)
): Float32Array {
  const {distanceThreshold, timeThreshold, maximumGap = 0, includeLast = false} = settings;
  if (target.length < GPU_TRAJECTORY_STAYPOINTS_PARAMETER_LENGTH) {
    throw new Error(
      `Staypoint target must hold ${GPU_TRAJECTORY_STAYPOINTS_PARAMETER_LENGTH} elements`
    );
  }
  if (!Number.isFinite(Math.fround(distanceThreshold)) || Math.fround(distanceThreshold) <= 0) {
    throw new Error('Staypoint distanceThreshold must be finite and positive');
  }
  if (!Number.isFinite(Math.fround(timeThreshold)) || timeThreshold < 0) {
    throw new Error('Staypoint timeThreshold must be finite and non-negative');
  }
  if (!Number.isFinite(Math.fround(maximumGap)) || maximumGap < 0) {
    throw new Error('Staypoint maximumGap must be finite and non-negative');
  }
  warnGPUShortLongitudeLatitudeDistance(
    'GPUTrajectoryStaypoints',
    settings.spatialContext,
    distanceThreshold
  );
  target.set([distanceThreshold, timeThreshold, maximumGap, includeLast ? 1 : 0]);
  return target;
}

/**
 * Caller-owned staypoint table of {@link GPUTrajectoryStaypoints}: a {@link GPUCompactOutput}
 * whose `ids` hold the track of each staypoint, ordered by track and then time. Rows at and after
 * `count` are rewritten: `0xffffffff` for IDs and rows, `0` for floats.
 *
 * `status.candidateOverflow` is 1 when a chain anchor's forward search hit `maximumSearchRows`
 * before deciding: the window is then cut at that row, so a longer stay reports as consecutive
 * staypoints. Raise `maximumSearchRows`.
 */
export type GPUTrajectoryStaypointsOutput = GPUCompactOutput & {
  status: GPUCompactOutput['status'] & GPUCandidateStatusPort;
  /** Optional one-row scalar receiving the clamped count again, for an indirect draw record. */
  drawInstanceCount?: GraphDataView<'uint32'>;
  /** Optional first row of each staypoint (`started_at`). */
  startRows?: GraphDataView<'uint32'>;
  /** Optional row after the last row of the staypoint (exclusive); the rows `[start, end)` are its fixes. */
  endRows?: GraphDataView<'uint32'>;
  /** Optional row whose time is `finished_at`: `endRows` before a break, the last row at a track end. */
  finishRows?: GraphDataView<'uint32'>;
  /** Optional centroid of the staypoint's distinct coordinates (trackintel `union_all`). */
  centroids?: GraphDataView<'float32x2'>;
  /** Optional `finished_at - started_at` in the timestamps' unit. */
  durations?: GraphDataView<'float32'>;
  /** Optional per-row staypoint index (`staypoint_id`), `positions.length` rows; other rows are `0xffffffff`. */
  staypointIds?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUTrajectoryStaypoints}: the trajectory layout plus the options below.
 * Per-frame (no recompile): buffer contents and `parameters`. Compile-time: view lengths, the
 * capacity, `maximumSearchRows`.
 */
export type GPUTrajectoryStaypointsProps = GPUTrajectoryGeometryFields & {
  /** Prefix for node and transient IDs. Defaults to `'trajectory-staypoints'`. */
  id?: string;
  /** Planar (default) or longitude/latitude great-circle (trackintel `haversine`, set `sphereRadius` 6371000 to match). */
  spatialContext?: GPUSpatialContext;
  /** Per-frame float32 view of {@link GPU_TRAJECTORY_STAYPOINTS_PARAMETER_LENGTH} elements. */
  parameters: GraphDataView<'float32'>;
  /** Forward-search cap per anchor in rows (default 4096, at most 16,777,216). */
  maximumSearchRows?: number;
  /** Caller-owned staypoint table. `ids.length` is the capacity. */
  output: GPUTrajectoryStaypointsOutput;
};

/**
 * Staypoint detection with trackintel's `sliding` semantics (Li et al. 2008), computed in a
 * bounded number of GPU passes.
 *
 * trackintel keeps a window anchor and walks forward: a step longer than `maximumGap` restarts
 * the anchor at the fix after the gap and emits nothing; a fix at `distanceThreshold` or farther
 * from the anchor closes the window, emits a staypoint when `timeThreshold` has elapsed since the
 * anchor (`finished_at` is the breaking fix, its geometry and rows exclude it) and restarts the
 * anchor at the breaking fix either way. With `includeLast` the window open at the end of a track
 * becomes a staypoint when long enough, its geometry and rows then including the last fix.
 *
 * Because that decision depends only on the fixes after an anchor, every row searches forward
 * independently (up to `maximumSearchRows` fixes, with early exit at the first break), giving a
 * successor row. Pointer doubling then marks the rows reachable from each track's first row in
 * `ceil(log2(rowCount))` fixed passes, which reproduces the sequential chain exactly. Those two
 * steps are the whole serial part of the algorithm; the rest is compaction and gathers.
 *
 * Time comparisons use exact word differences for Int64 words (exact below 2^24 ticks).
 *
 * Non-goals: `exclude_duplicate_pfs` row removal (dedupe upstream with `GPURunLengthEncode` or a
 * sort); elevation medians; staypoints from positionfixes without a `staypoint_id` join.
 */
export class GPUTrajectoryStaypoints implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTrajectoryStaypointsProps;
  /** Resolved spatial context. */
  readonly spatialContext: GPUSpatialContext;
  /** Forward search cap in rows. */
  readonly maximumSearchRows: number;

  constructor(props: GPUTrajectoryStaypointsProps) {
    this.id = props.id ?? 'trajectory-staypoints';
    this.props = props;
    const id = this.id;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    const {output, positions, timestamps, trackOffsets, parameters} = props;
    validateSinglePackedView(positions, ['float32x2'], `${id} positions`);
    validateSinglePackedView(timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
    validateSinglePackedView(trackOffsets, ['uint32'], `${id} trackOffsets`);
    validateSinglePackedView(parameters, ['float32'], `${id} parameters`);
    validateViewLength(id, 'timestamps', timestamps, positions.length);
    validateRequirement(
      id,
      trackOffsets.length >= 2,
      'trackOffsets must contain at least two rows'
    );
    resolveGPUTrajectoryTemporalContext(id, timestamps, props.temporalContext);
    validateRequirement(
      id,
      parameters.length >= GPU_TRAJECTORY_STAYPOINTS_PARAMETER_LENGTH,
      `parameters must hold ${GPU_TRAJECTORY_STAYPOINTS_PARAMETER_LENGTH} float32 values`
    );
    this.maximumSearchRows =
      props.maximumSearchRows ?? GPU_TRAJECTORY_STAYPOINTS_DEFAULT_MAXIMUM_SEARCH_ROWS;
    validateMaximumCount(
      id,
      'maximumSearchRows',
      this.maximumSearchRows,
      GPU_TRAJECTORY_STAYPOINTS_MAXIMUM_SEARCH_ROWS
    );
    validateCompactOutput(id, output);
    validateCapacity(id, 'staypoint capacity', output.ids.length);
    const capacity = output.ids.length;
    for (const [name, view, format] of [
      ['startRows', output.startRows, 'uint32'],
      ['endRows', output.endRows, 'uint32'],
      ['finishRows', output.finishRows, 'uint32'],
      ['centroids', output.centroids, 'float32x2'],
      ['durations', output.durations, 'float32']
    ] as const) {
      if (view) {
        validateSinglePackedView(view, [format], `${id} output.${name}`);
        validateViewLength(id, `output.${name}`, view, capacity);
      }
    }
    if (output.staypointIds) {
      validateSinglePackedView(output.staypointIds, ['uint32'], `${id} output.staypointIds`);
      validateViewLength(id, 'output.staypointIds', output.staypointIds, positions.length);
    }
    validateTrajectoryOutputRanges(this.id, this.props.output);
    validateTrajectoryWritableDispatches(this.id, [
      {ids: output.ids, startRows: output.startRows},
      {endRows: output.endRows, finishRows: output.finishRows},
      {
        count: output.status.count,
        overflow: output.status.overflow,
        requiredCount: output.status.requiredCount,
        drawInstanceCount: output.drawInstanceCount
      }
    ]);
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.durations],
      [output.startRows, output.finishRows]
    );
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.centroids, output.staypointIds],
      [output.startRows, output.endRows]
    );
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
  }

  private getInputViews(): (GraphDataView | undefined)[] {
    const {positions, timestamps, trackOffsets, parameters} = this.props;
    return [positions, timestamps, trackOffsets, parameters];
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {output} = this.props;
    return [
      output.ids,
      output.status.count,
      output.status.overflow,
      output.status.requiredCount,
      output.status.candidateOverflow,
      output.drawInstanceCount,
      output.startRows,
      output.endRows,
      output.finishRows,
      output.centroids,
      output.durations,
      output.staypointIds
    ];
  }

  /** Returns the staypoint nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output, positions, trackOffsets, parameters} = props;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    const rowCount = positions.length;
    const capacity = output.ids.length;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'positions',
      rowCount * 8,
      'split the tracks over several graphs'
    );
    validateTrajectoryBindings(graph, this.id, this.props);
    const timeMode = getTrajectoryTimeMode(props);
    const radius = getGPUGreatCircleRadius(this.spatialContext);
    const timeBindings = getTrajectoryTimeBindings(props);

    const rowStarts = createTransientView(graph, `${id}-row-starts`, 'uint32', rowCount);
    const rowEnds = createTransientView(graph, `${id}-row-ends`, 'uint32', rowCount);
    const rowTracks = createTransientView(graph, `${id}-row-tracks`, 'uint32', rowCount);
    const next = createTransientView(graph, `${id}-next`, 'uint32', rowCount);
    const flags = createTransientView(graph, `${id}-flags`, 'uint32', rowCount);
    const reach = createTransientView(graph, `${id}-reach`, 'uint32', rowCount);
    const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', rowCount);
    const stayFlags = createTransientView(graph, `${id}-stay-flags`, 'uint32', rowCount);
    const stayRows = createTransientView(graph, `${id}-stay-rows`, 'uint32', rowCount);
    const stayTotal = createTransientView(graph, `${id}-stay-total`, 'uint32', 1);

    const nodes: GPUCommandNode<Parameters>[] = [
      ...createRowBoundsNodes<Parameters>(graph, {
        id: `${id}-row-bounds`,
        operation: OPERATION,
        trackOffsets,
        rowCount,
        rowStarts,
        rowEnds,
        rowTracks
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-search`,
        operation: OPERATION,
        variant: timeMode,
        invocationCount: rowCount,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          ...timeBindings,
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          {name: 'rowEnds', view: rowEnds, type: 'u32', access: 'read'},
          {name: 'next', view: next, type: 'u32', access: 'read_write'},
          {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
        ],
        declarations: `${getTrajectoryTimeSource(timeMode)}\n${getMovementDistanceWGSL(radius)}`,
        body: `let distanceThreshold = parameters[parametersOffset];
  let timeThreshold = parameters[parametersOffset + 1u];
  let maximumGap = parameters[parametersOffset + 2u];
  let includeLast = parameters[parametersOffset + 3u] != 0.0;
  let trackEnd = rowEnds[rowEndsOffset + index];
  var successor = index;
  var state = 0u;
  if (trackEnd != ${MOVEMENT_SENTINEL_WGSL}) {
    let searchEnd = min(trackEnd, index + 1u + ${this.maximumSearchRows}u);
    var decided = false;
    for (var row = index + 1u; row < searchEnd; row++) {
      if (maximumGap > 0.0 && timeDifferenceRows(row, row - 1u) > maximumGap) {
        successor = row;
        decided = true;
        break;
      }
      if (rowDistance(index, row) >= distanceThreshold) {
        successor = row;
        if (timeDifferenceRows(row, index) >= timeThreshold) { state = 1u; }
        decided = true;
        break;
      }
    }
    if (!decided) {
      if (searchEnd == trackEnd) {
        if (includeLast && timeDifferenceRows(trackEnd - 1u, index) >= timeThreshold) { state = 5u; }
      } else {
        successor = searchEnd;
        state = 2u;
        if (timeDifferenceRows(searchEnd, index) >= timeThreshold) { state = 3u; }
      }
    }
  }
  next[nextOffset + index] = successor;
  flags[flagsOffset + index] = state;`
      }),
      ...createChainReachNodes<Parameters>(graph, {
        id: `${id}-chain`,
        operation: OPERATION,
        trackOffsets,
        next,
        reach,
        rowCount
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-select`,
        operation: OPERATION,
        invocationCount: rowCount,
        bindings: [
          {name: 'reach', view: reach, type: 'u32', access: 'read'},
          {name: 'flags', view: flags, type: 'u32', access: 'read'},
          {name: 'stayFlags', view: stayFlags, type: 'u32', access: 'read_write'}
        ],
        body: `stayFlags[stayFlagsOffset + index] = select(0u, 1u, reach[reachOffset + index] != 0u && (flags[flagsOffset + index] & 1u) != 0u);`
      })
    ];
    if (output.status.candidateOverflow) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-candidate-reset`,
          operation: OPERATION,
          view: output.status.candidateOverflow,
          type: 'u32',
          value: '0u',
          componentCount: 1
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-candidate-overflow`,
          operation: OPERATION,
          invocationCount: rowCount,
          bindings: [
            {name: 'reach', view: reach, type: 'u32', access: 'read'},
            {name: 'flags', view: flags, type: 'u32', access: 'read'},
            {
              name: 'candidateOverflow',
              view: output.status.candidateOverflow,
              type: 'atomic<u32>',
              access: 'read_write'
            }
          ],
          body: `if (reach[reachOffset + index] != 0u && (flags[flagsOffset + index] & 2u) != 0u) { atomicStore(&candidateOverflow[candidateOverflowOffset], 1u); }`
        })
      );
    }
    nodes.push(
      createRowIdsNode<Parameters>(graph, {id: `${id}-row-ids`, operation: OPERATION, rowIds}),
      ...new GPUCompaction({
        id: `${id}-compact`,
        input: rowIds,
        flags: stayFlags,
        output: stayRows,
        count: stayTotal
      }).getCommandNodes(graph),
      ...this.createGatherNodes(graph, {
        rowCount,
        capacity,
        stayRows,
        stayTotal,
        next,
        flags,
        rowEnds,
        rowTracks
      }),
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        requiredCount: stayTotal,
        output,
        extraCounts: output.drawInstanceCount ? [output.drawInstanceCount] : []
      })
    );
    return nodes;
  }

  private createGatherNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    context: {
      rowCount: number;
      capacity: number;
      stayRows: GraphDataView<'uint32'>;
      stayTotal: GraphDataView<'uint32'>;
      next: GraphDataView<'uint32'>;
      flags: GraphDataView<'uint32'>;
      rowEnds: GraphDataView<'uint32'>;
      rowTracks: GraphDataView<'uint32'>;
    }
  ): GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output, positions} = props;
    const {rowCount, capacity, stayRows, stayTotal, next, flags, rowEnds, rowTracks} = context;
    const startRows =
      output.startRows ?? createTransientView(graph, `${id}-start-rows`, 'uint32', capacity);
    const endRows =
      output.endRows ?? createTransientView(graph, `${id}-end-rows`, 'uint32', capacity);
    const finishRows =
      output.finishRows ?? createTransientView(graph, `${id}-finish-rows`, 'uint32', capacity);
    const radius = getGPUGreatCircleRadius(this.spatialContext);
    const timeMode = getTrajectoryTimeMode(props);
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-gather-ends`,
        operation: OPERATION,
        invocationCount: capacity,
        bindings: [
          {name: 'stayRows', view: stayRows, type: 'u32', access: 'read'},
          {name: 'stayTotal', view: stayTotal, type: 'u32', access: 'read'},
          {name: 'next', view: next, type: 'u32', access: 'read'},
          {name: 'flags', view: flags, type: 'u32', access: 'read'},
          {name: 'rowEnds', view: rowEnds, type: 'u32', access: 'read'},
          {name: 'endRows', view: endRows, type: 'u32', access: 'read_write'},
          {name: 'finishRows', view: finishRows, type: 'u32', access: 'read_write'}
        ],
        body: `if (index < stayTotal[stayTotalOffset]) {
    let row = stayRows[stayRowsOffset + index];
    let isTrackEnd = (flags[flagsOffset + row] & 4u) != 0u;
    let trackEnd = rowEnds[rowEndsOffset + row];
    endRows[endRowsOffset + index] = select(next[nextOffset + row], trackEnd, isTrackEnd);
    finishRows[finishRowsOffset + index] = select(next[nextOffset + row], trackEnd - 1u, isTrackEnd);
  } else {
    endRows[endRowsOffset + index] = ${MOVEMENT_SENTINEL_WGSL};
    finishRows[finishRowsOffset + index] = ${MOVEMENT_SENTINEL_WGSL};
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-gather-rows`,
        operation: OPERATION,
        invocationCount: capacity,
        bindings: [
          {name: 'stayRows', view: stayRows, type: 'u32', access: 'read'},
          {name: 'stayTotal', view: stayTotal, type: 'u32', access: 'read'},
          {name: 'rowTracks', view: rowTracks, type: 'u32', access: 'read'},
          {name: 'ids', view: output.ids, type: 'u32', access: 'read_write'},
          {name: 'startRows', view: startRows, type: 'u32', access: 'read_write'}
        ],
        body: `if (index < stayTotal[stayTotalOffset]) {
    let row = stayRows[stayRowsOffset + index];
    ids[idsOffset + index] = rowTracks[rowTracksOffset + row];
    startRows[startRowsOffset + index] = row;
  } else {
    ids[idsOffset + index] = ${MOVEMENT_SENTINEL_WGSL};
    startRows[startRowsOffset + index] = ${MOVEMENT_SENTINEL_WGSL};
  }`
      })
    ];
    if (output.durations) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-durations`,
          operation: OPERATION,
          variant: timeMode,
          invocationCount: capacity,
          bindings: [
            ...getTrajectoryTimeBindings(props),
            {name: 'stayTotal', view: stayTotal, type: 'u32', access: 'read'},
            {name: 'startRows', view: startRows, type: 'u32', access: 'read'},
            {name: 'finishRows', view: finishRows, type: 'u32', access: 'read'},
            {name: 'durations', view: output.durations, type: 'f32', access: 'read_write'}
          ],
          declarations: getTrajectoryTimeSource(timeMode),
          body: `durations[durationsOffset + index] = select(0.0, timeDifferenceRows(finishRows[finishRowsOffset + index], startRows[startRowsOffset + index]), index < stayTotal[stayTotalOffset]);`
        })
      );
    }
    if (output.centroids) {
      // One thread per staypoint. Distinct coordinates need a backward scan, so the cost is
      // quadratic in the staypoint's row count, the same order as the forward search.
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-centroids`,
          operation: OPERATION,
          invocationCount: capacity,
          bindings: [
            {name: 'positions', view: positions, type: 'f32', access: 'read'},
            {name: 'stayTotal', view: stayTotal, type: 'u32', access: 'read'},
            {name: 'startRows', view: startRows, type: 'u32', access: 'read'},
            {name: 'endRows', view: endRows, type: 'u32', access: 'read'},
            {name: 'centroids', view: output.centroids, type: 'f32', access: 'read_write'}
          ],
          declarations: `fn longitudeOffset(delta: f32) -> f32 { return ${radius === undefined ? 'delta' : 'delta - 360.0 * round(delta / 360.0)'}; }
fn rowPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`,
          body: `var centroid = vec2<f32>(0.0, 0.0);
  if (index < stayTotal[stayTotalOffset]) {
    let start = startRows[startRowsOffset + index];
    let end = endRows[endRowsOffset + index];
    let anchor = rowPosition(start);
    var offset = vec2<f32>(0.0, 0.0);
    var distinctCount = 0u;
    for (var row = start; row < end; row++) {
      let position = rowPosition(row);
      var isDuplicate = false;
      for (var earlier = start; earlier < row; earlier++) {
        let other = rowPosition(earlier);
        if (other.x == position.x && other.y == position.y) { isDuplicate = true; break; }
      }
      if (!isDuplicate) {
        offset += vec2<f32>(longitudeOffset(position.x - anchor.x), position.y - anchor.y);
        distinctCount++;
      }
    }
    centroid = anchor + offset / f32(max(distinctCount, 1u));
    centroid.x = ${radius === undefined ? 'centroid.x' : '((centroid.x + 180.0) - 360.0 * floor((centroid.x + 180.0) / 360.0)) - 180.0'};
  }
  centroids[centroidsOffset + index * 2u] = centroid.x;
  centroids[centroidsOffset + index * 2u + 1u] = centroid.y;`
        })
      );
    }
    if (output.staypointIds) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-row-ids-reset`,
          operation: OPERATION,
          view: output.staypointIds,
          type: 'u32',
          value: MOVEMENT_SENTINEL_WGSL
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-row-staypoints`,
          operation: OPERATION,
          invocationCount: capacity,
          bindings: [
            {name: 'stayTotal', view: stayTotal, type: 'u32', access: 'read'},
            {name: 'startRows', view: startRows, type: 'u32', access: 'read'},
            {name: 'endRows', view: endRows, type: 'u32', access: 'read'},
            {name: 'staypointIds', view: output.staypointIds, type: 'u32', access: 'read_write'}
          ],
          body: `if (index < stayTotal[stayTotalOffset]) {
    for (var row = startRows[startRowsOffset + index]; row < min(endRows[endRowsOffset + index], ${rowCount}u); row++) {
      staypointIds[staypointIdsOffset + row] = index;
    }
  }`
        })
      );
    }
    return nodes;
  }
}
