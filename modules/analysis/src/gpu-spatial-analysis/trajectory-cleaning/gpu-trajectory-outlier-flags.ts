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
import type {GPUTrajectoryGeometryFields} from '../../analysis-core/contracts/index';
import {
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {resolveGPUTrajectoryTemporalContext} from '../trajectory-analysis/trajectory-common';
import {
  getTrajectoryTimeBindings,
  getTrajectoryTimeMode,
  getTrajectoryTimeSource
} from '../trajectory-analysis/trajectory-metrics-kernels';
import {
  createRowBoundsNodes,
  getMovementDistanceWGSL
} from '../trajectory-staypoints/movement-common';

const OPERATION = 'GPUTrajectoryOutlierFlags';

/** Point-centred outlier flags of {@link GPUTrajectoryOutlierFlags}. */
export const GPU_TRAJECTORY_OUTLIER_FLAGS = {
  /** The fix is displaced: both adjacent step speeds exceed the limit. */
  speed: 1,
  /**
   * The fix is displaced: the step speed rises by more than the limit into the fix and falls by
   * more than the limit after the step out of it.
   */
  acceleration: 2
} as const;

/** Number of float32 elements in an outlier-flag parameter buffer. */
export const GPU_TRAJECTORY_OUTLIER_FLAGS_PARAMETER_LENGTH = 4;

/** Per-frame settings of {@link GPUTrajectoryOutlierFlags}. */
export type GPUTrajectoryOutlierFlagsSettings = {
  /** Speed limit, length unit per timestamp unit. Finite and non-negative; 0 disables the test. */
  maximumSpeed: number;
  /** Acceleration limit, length per timestamp unit squared. 0 disables the test. */
  maximumAcceleration?: number;
};

/**
 * Packs outlier-flag settings into the float32 layout read by {@link GPUTrajectoryOutlierFlags}.
 *
 * Layout: `[maximumSpeed, maximumAcceleration, 0, 0]`.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If a limit is negative or not finite.
 */
export function getGPUTrajectoryOutlierFlagsParameterValues(
  settings: GPUTrajectoryOutlierFlagsSettings,
  target: Float32Array = new Float32Array(GPU_TRAJECTORY_OUTLIER_FLAGS_PARAMETER_LENGTH)
): Float32Array {
  const maximumAcceleration = settings.maximumAcceleration ?? 0;
  for (const value of [settings.maximumSpeed, maximumAcceleration]) {
    if (!Number.isFinite(value) || value < 0) {
      throw new Error('Outlier limits must be finite and non-negative');
    }
  }
  target.set([settings.maximumSpeed, maximumAcceleration, 0, 0]);
  return target;
}

/** Properties for {@link GPUTrajectoryOutlierFlags}. */
export type GPUTrajectoryOutlierFlagsProps = GPUTrajectoryGeometryFields & {
  /** Prefix for node and transient IDs. Defaults to `'trajectory-outlier-flags'`. */
  id?: string;
  /** Planar or longitude/latitude great-circle. Speeds use the context's length unit per timestamp unit. */
  spatialContext?: GPUSpatialContext;
  /** Per-frame float32 view of {@link GPU_TRAJECTORY_OUTLIER_FLAGS_PARAMETER_LENGTH} elements. */
  parameters: GraphDataView<'float32'>;
  output: {
    /** Per-row flag bits of {@link GPU_TRAJECTORY_OUTLIER_FLAGS}, `positions.length` rows. */
    flags: GraphDataView<'uint32'>;
    /** Optional per-track flagged row count. */
    trackFlaggedCounts?: GraphDataView<'uint32'>;
  };
};

/**
 * Point-centred speed and acceleration outliers (review finding FX-08): unlike the `'step'` test of
 * `addTrackQualityRecipe`, which flags the row ending a fast step, a fix is an outlier only when
 * both adjacent step speeds exceed the limit, so the valid fix after a spike is left alone. An
 * endpoint has one step and is flagged when that step fails unless the neighbour's step away from
 * it also fails.
 *
 * The acceleration bit marks the same displaced fix from the change of step speed. With `v_j` the
 * speed of the step ending at row `j` and `a_j = (v_j - v_{j-1}) / dt_j` its signed acceleration, a
 * displaced fix `i` has two fast steps of about equal speed, so `a_{i+1}` is near zero and the
 * excess shows on either side of the pair: row `i` is flagged when `a_i > limit` (the speed rises
 * into the fix) and `a_{i+2} < -limit` (it falls after the step out of the fix). A sustained jump,
 * where every later row is shifted, has one fast step, so its rise and fall are adjacent and no row
 * is flagged. At a track end the side that does not exist is waived, as for the speed bit: the first
 * row needs only the fall `a_{i+2}`, the last row only the rise `a_i`. The second row has no
 * `a_i` and is flagged on the fall unless `a_{i+1} > limit` (its step in was the slow one); the
 * second-to-last row has no `a_{i+2}` and is flagged on the rise unless `a_{i+1} < -limit`. A
 * track of two rows has no acceleration and is never flagged.
 *
 * This is a single pass: two consecutive displaced fixes are not caught by either bit, because the
 * step between them is slow. `movingpandas.OutlierCleaner` compares each fix to the last kept fix
 * instead, which catches such bursts but is inherently sequential. Rows are not removed here; drop
 * flagged rows with `GPUCompaction` or the row mask of `GPUTrajectorySplit`.
 */
export class GPUTrajectoryOutlierFlags implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUTrajectoryOutlierFlagsProps;
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUTrajectoryOutlierFlagsProps) {
    this.id = props.id ?? 'trajectory-outlier-flags';
    this.props = props;
    const id = this.id;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT
    );
    const {positions, timestamps, trackOffsets, parameters, output} = props;
    validateSinglePackedView(positions, ['float32x2'], `${id} positions`);
    validateSinglePackedView(timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
    validateSinglePackedView(trackOffsets, ['uint32'], `${id} trackOffsets`);
    validateSinglePackedView(parameters, ['float32'], `${id} parameters`);
    validateViewLength(id, 'timestamps', timestamps, positions.length);
    resolveGPUTrajectoryTemporalContext(id, timestamps, props.temporalContext);
    validateRequirement(
      id,
      trackOffsets.length >= 2,
      'trackOffsets must contain at least two rows'
    );
    validateRequirement(
      id,
      parameters.length >= GPU_TRAJECTORY_OUTLIER_FLAGS_PARAMETER_LENGTH,
      'parameters must hold 4 float32 values'
    );
    validateSinglePackedView(output.flags, ['uint32'], `${id} output.flags`);
    validateViewLength(id, 'output.flags', output.flags, positions.length);
    if (output.trackFlaggedCounts) {
      validateSinglePackedView(
        output.trackFlaggedCounts,
        ['uint32'],
        `${id} output.trackFlaggedCounts`
      );
      validateViewLength(
        id,
        'output.trackFlaggedCounts',
        output.trackFlaggedCounts,
        trackOffsets.length - 1
      );
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.flags, output.trackFlaggedCounts],
      [positions, timestamps, trackOffsets, parameters]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, timestamps, trackOffsets, parameters, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      timestamps,
      trackOffsets,
      parameters,
      output.flags,
      output.trackFlaggedCounts
    ]);
    const rowCount = positions.length;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'positions',
      rowCount * 8,
      'split the tracks over several graphs'
    );
    const timeMode = getTrajectoryTimeMode(props);
    const rowStarts = createTransientView(graph, `${id}-row-starts`, 'uint32', rowCount);
    const rowEnds = createTransientView(graph, `${id}-row-ends`, 'uint32', rowCount);
    const rowTracks = createTransientView(graph, `${id}-row-tracks`, 'uint32', rowCount);
    const radius = getGPUGreatCircleRadius(this.spatialContext);
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
        id: `${id}-flags`,
        operation: OPERATION,
        variant: timeMode,
        invocationCount: rowCount,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          ...getTrajectoryTimeBindings(props),
          {name: 'rowStarts', view: rowStarts, type: 'u32', access: 'read'},
          {name: 'rowEnds', view: rowEnds, type: 'u32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          {name: 'flags', view: output.flags, type: 'u32', access: 'read_write'}
        ],
        declarations: `${getTrajectoryTimeSource(timeMode)}\n${getMovementDistanceWGSL(radius)}
fn stepSpeed(row: u32) -> f32 {
  let elapsed = timeDifferenceRows(row, row - 1u);
  return select(0.0, rowDistance(row, row - 1u) / elapsed, elapsed > 0.0);
}
fn isExcessiveStep(row: u32, limit: f32) -> bool {
  let elapsed = timeDifferenceRows(row, row - 1u);
  return limit > 0.0 && elapsed > 0.0 && stepSpeed(row) > limit;
}
// Signed acceleration of the step ending at row: positive when the step is faster than the one
// before it. 0 where no previous step exists in the track or no time has passed.
fn stepAcceleration(row: u32, begin: u32) -> f32 {
  if (row < begin + 2u) { return 0.0; }
  let elapsed = timeDifferenceRows(row, row - 1u);
  if (elapsed <= 0.0) { return 0.0; }
  return (stepSpeed(row) - stepSpeed(row - 1u)) / elapsed;
}
// A displaced fix has two fast steps of about equal speed, so the excessive accelerations are the
// rise into the row and the fall after the step out of it, two rows later. The acceleration between
// the two fast steps is near zero and is not tested. A side that the track end removes is waived.
fn isDisplacedByAcceleration(row: u32, begin: u32, last: u32, limit: f32) -> bool {
  if (limit <= 0.0) { return false; }
  let hasRise = row >= begin + 2u;
  let hasFall = row + 2u <= last;
  if (!hasRise && !hasFall) { return false; }
  if (hasRise) {
    if (!(stepAcceleration(row, begin) > limit)) { return false; }
  } else if (row > begin) {
    // Second row: its step in has no earlier step to rise from. A rise on its step out means the
    // step in was the slow one, which is a jump after this row, not a displaced fix.
    if (stepAcceleration(row + 1u, begin) > limit) { return false; }
  }
  if (hasFall) {
    if (!(stepAcceleration(row + 2u, begin) < -limit)) { return false; }
  } else if (row < last) {
    // Second-to-last row: a fall on its step out means the fix stayed where the step in took it.
    if (stepAcceleration(row + 1u, begin) < -limit) { return false; }
  }
  return true;
}`,
        body: `let begin = rowStarts[rowStartsOffset + index];
  let end = rowEnds[rowEndsOffset + index];
  var state = 0u;
  if (begin != 0xffffffffu && end - begin >= 2u) {
    let speedLimit = parameters[parametersOffset];
    let accelerationLimit = parameters[parametersOffset + 1u];
    let hasIncoming = index > begin;
    let hasOutgoing = index + 1u < end;
    let incoming = hasIncoming && isExcessiveStep(index, speedLimit);
    let outgoing = hasOutgoing && isExcessiveStep(index + 1u, speedLimit);
    var isOutlier = false;
    if (hasIncoming && hasOutgoing) { isOutlier = incoming && outgoing; }
    else if (hasOutgoing) { isOutlier = outgoing && !(index + 2u < end && isExcessiveStep(index + 2u, speedLimit)); }
    else if (hasIncoming) { isOutlier = incoming && !(index >= begin + 2u && isExcessiveStep(index - 1u, speedLimit)); }
    if (isOutlier) { state |= 1u; }
    if (isDisplacedByAcceleration(index, begin, end - 1u, accelerationLimit)) { state |= 2u; }
  }
  flags[flagsOffset + index] = state;`
      })
    ];
    if (output.trackFlaggedCounts) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-counts`,
          operation: OPERATION,
          invocationCount: trackOffsets.length - 1,
          bindings: [
            {name: 'trackOffsets', view: trackOffsets, type: 'u32', access: 'read'},
            {name: 'flags', view: output.flags, type: 'u32', access: 'read'},
            {name: 'counts', view: output.trackFlaggedCounts, type: 'u32', access: 'read_write'}
          ],
          body: `var total = 0u;
  for (var row = min(trackOffsets[trackOffsetsOffset + index], ${rowCount}u); row < min(trackOffsets[trackOffsetsOffset + index + 1u], ${rowCount}u); row++) {
    total += select(0u, 1u, flags[flagsOffset + row] != 0u);
  }
  counts[countsOffset + index] = total;`
        })
      );
    }
    return nodes;
  }
}
