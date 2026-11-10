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
  validatePackedView,
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
  /** The fix is displaced: both adjacent accelerations exceed the limit in magnitude. */
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
 * it also fails. Acceleration uses the same rule on `(v_i - v_{i-1}) / dt_i` and
 * `(v_{i+1} - v_i) / dt_{i+1}`.
 *
 * This is a single pass: two consecutive displaced fixes are not caught, because the step between
 * them is slow. `movingpandas.OutlierCleaner` compares each fix to the last kept fix instead, which
 * catches such bursts but is inherently sequential. Rows are not removed here; drop flagged rows
 * with `GPUCompaction` or the row mask of `GPUTrajectorySplit`.
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
    validatePackedView(positions, ['float32x2'], `${id} positions`);
    validatePackedView(timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
    validatePackedView(trackOffsets, ['uint32'], `${id} trackOffsets`);
    validatePackedView(parameters, ['float32'], `${id} parameters`);
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
    validatePackedView(output.flags, ['uint32'], `${id} output.flags`);
    validateViewLength(id, 'output.flags', output.flags, positions.length);
    if (output.trackFlaggedCounts) {
      validatePackedView(output.trackFlaggedCounts, ['uint32'], `${id} output.trackFlaggedCounts`);
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
// Acceleration of the step ending at row, where a previous step exists.
fn isExcessiveAcceleration(row: u32, begin: u32, limit: f32) -> bool {
  if (limit <= 0.0 || row < begin + 2u) { return false; }
  let elapsed = timeDifferenceRows(row, row - 1u);
  if (elapsed <= 0.0) { return false; }
  return abs(stepSpeed(row) - stepSpeed(row - 1u)) / elapsed > limit;
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
    let accelerationIn = hasIncoming && isExcessiveAcceleration(index, begin, accelerationLimit);
    let accelerationOut = hasOutgoing && isExcessiveAcceleration(index + 1u, begin, accelerationLimit);
    if (accelerationIn && accelerationOut) { state |= 2u; }
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
