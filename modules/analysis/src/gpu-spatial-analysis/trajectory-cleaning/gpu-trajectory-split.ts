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
import type {
  GPUCardinalityPort,
  GPUOverflowStatusPort,
  GPUTrajectoryGeometryFields
} from '../../analysis-core/contracts/index';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {resolveGPUTrajectoryTemporalContext} from '../trajectory-analysis/trajectory-common';
import {
  getTrajectoryTimeBindings,
  getTrajectoryTimeMode,
  getTrajectoryTimeSource
} from '../trajectory-analysis/trajectory-metrics-kernels';
import {MOVEMENT_SENTINEL_WGSL} from '../trajectory-staypoints/movement-common';

const OPERATION = 'GPUTrajectorySplit';

/** How {@link GPUTrajectorySplit} breaks a track into segments. */
export type GPUTrajectorySplitMode = 'gap' | 'stop' | 'value-change';

/** Number of float32 elements in a split parameter buffer. */
export const GPU_TRAJECTORY_SPLIT_PARAMETER_LENGTH = 4;

/** Per-frame settings of {@link GPUTrajectorySplit}. */
export type GPUTrajectorySplitSettings = {
  /** Mode `'gap'`: the shared gap rule; a step strictly longer than this splits. 0 means no limit. */
  maximumGap: number;
};

/**
 * Packs split settings into the float32 layout read by {@link GPUTrajectorySplit}.
 *
 * Layout: `[maximumGap, 0, 0, 0]`.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If `maximumGap` is negative or not finite.
 */
export function getGPUTrajectorySplitParameterValues(
  settings: GPUTrajectorySplitSettings,
  target: Float32Array = new Float32Array(GPU_TRAJECTORY_SPLIT_PARAMETER_LENGTH)
): Float32Array {
  if (!Number.isFinite(settings.maximumGap) || settings.maximumGap < 0) {
    throw new Error('Split maximumGap must be finite and non-negative');
  }
  target.set([settings.maximumGap, 0, 0, 0]);
  return target;
}

/** Caller-owned outputs of {@link GPUTrajectorySplit}. */
export type GPUTrajectorySplitOutput = {
  /**
   * New track offsets: segment `s` owns `rowIds[segmentOffsets[s]..segmentOffsets[s + 1])`.
   * `segmentCapacity + 1` rows; rows past the count hold the sentinel.
   */
  segmentOffsets: GraphDataView<'uint32'>;
  /** Source track of each segment, `segmentCapacity` rows. */
  segmentTracks: GraphDataView<'uint32'>;
  /** Source row of each output row, `rowCapacity` rows. */
  rowIds: GraphDataView<'uint32'>;
  /** Segment count; `overflow` is 1 when segments or rows exceeded a capacity. */
  status: GPUCardinalityPort & GPUOverflowStatusPort;
  /** Optional copied positions of the output rows, `rowCapacity` rows. */
  positions?: GraphDataView<'float32x2'>;
  /** Optional copied timestamps of the output rows, `rowCapacity` rows, the input format. */
  timestamps?: GPUTrajectoryGeometryFields['timestamps'];
};

/** Properties for {@link GPUTrajectorySplit}. */
export type GPUTrajectorySplitProps = GPUTrajectoryGeometryFields & {
  /** Prefix for node and transient IDs. Defaults to `'trajectory-split'`. */
  id?: string;
  /** Split mode (compile-time). */
  mode: GPUTrajectorySplitMode;
  /** Mode `'gap'`: per-frame float32 view of {@link GPU_TRAJECTORY_SPLIT_PARAMETER_LENGTH} elements. */
  parameters?: GraphDataView<'float32'>;
  /** Mode `'stop'`: per-row staypoint index from `GPUTrajectoryStaypoints`. */
  staypointIds?: GraphDataView<'uint32'>;
  /** Mode `'value-change'`: one uint32 value per row. */
  values?: GraphDataView<'uint32'>;
  output: GPUTrajectorySplitOutput;
};

/**
 * Splits tracks into segments and rewrites the offsets. MovingPandas 0.23 semantics:
 *
 * - `'gap'` (`ObservationGapSplitter`): a new segment starts after every step strictly longer than
 *   `maximumGap`.
 * - `'stop'` (`StopSplitter`, on staypoints): segments are the maximal runs of rows outside every
 *   staypoint.
 * - `'value-change'` (`ValueChangeSplitter`): a new segment starts where the value changes, and
 *   each segment but the last repeats the first row of the next segment.
 *
 * A segment needs at least two rows; shorter ones are dropped (`len(df) > 1` in MovingPandas). The
 * rows of the new port are `rowIds`, optionally with copied `positions` and `timestamps`, so the
 * output feeds another trajectory contributor without readback.
 *
 * Two passes per track, one thread each: count, then write after prefix offsets.
 */
export class GPUTrajectorySplit implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUTrajectorySplitProps;
  readonly segmentCapacity: number;
  readonly rowCapacity: number;

  constructor(props: GPUTrajectorySplitProps) {
    this.id = props.id ?? 'trajectory-split';
    this.props = props;
    const id = this.id;
    const {positions, timestamps, trackOffsets, output, mode} = props;
    validatePackedView(positions, ['float32x2'], `${id} positions`);
    validatePackedView(timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
    validatePackedView(trackOffsets, ['uint32'], `${id} trackOffsets`);
    validateViewLength(id, 'timestamps', timestamps, positions.length);
    resolveGPUTrajectoryTemporalContext(id, timestamps, props.temporalContext);
    validateRequirement(
      id,
      trackOffsets.length >= 2,
      'trackOffsets must contain at least two rows'
    );
    validateRequirement(
      id,
      ['gap', 'stop', 'value-change'].includes(mode),
      "mode must be 'gap', 'stop' or 'value-change'"
    );
    if (mode === 'gap') {
      validateRequirement(id, Boolean(props.parameters), "mode 'gap' requires parameters");
      validatePackedView(props.parameters!, ['float32'], `${id} parameters`);
      validateRequirement(
        id,
        props.parameters!.length >= GPU_TRAJECTORY_SPLIT_PARAMETER_LENGTH,
        'parameters must hold 4 float32 values'
      );
    } else if (mode === 'stop') {
      validateRequirement(id, Boolean(props.staypointIds), "mode 'stop' requires staypointIds");
      validatePackedView(props.staypointIds!, ['uint32'], `${id} staypointIds`);
      validateViewLength(id, 'staypointIds', props.staypointIds!, positions.length);
    } else {
      validateRequirement(id, Boolean(props.values), "mode 'value-change' requires values");
      validatePackedView(props.values!, ['uint32'], `${id} values`);
      validateViewLength(id, 'values', props.values!, positions.length);
    }
    validatePackedView(output.segmentOffsets, ['uint32'], `${id} output.segmentOffsets`);
    validatePackedView(output.segmentTracks, ['uint32'], `${id} output.segmentTracks`);
    validatePackedView(output.rowIds, ['uint32'], `${id} output.rowIds`);
    validateCapacity(id, 'output.segmentTracks', output.segmentTracks.length);
    validateCapacity(id, 'output.rowIds', output.rowIds.length);
    this.segmentCapacity = output.segmentTracks.length;
    this.rowCapacity = output.rowIds.length;
    validateViewLength(
      id,
      'output.segmentOffsets',
      output.segmentOffsets,
      this.segmentCapacity + 1
    );
    if (output.positions) {
      validatePackedView(output.positions, ['float32x2'], `${id} output.positions`);
      validateViewLength(id, 'output.positions', output.positions, this.rowCapacity);
    }
    if (output.timestamps) {
      validatePackedView(output.timestamps, [timestamps.format], `${id} output.timestamps`);
      validateViewLength(id, 'output.timestamps', output.timestamps, this.rowCapacity);
    }
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
  }

  private getInputViews(): (GraphDataView | undefined)[] {
    const {props} = this;
    return [
      props.positions,
      props.timestamps,
      props.trackOffsets,
      props.parameters,
      props.staypointIds,
      props.values
    ];
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {output} = this.props;
    return [
      output.segmentOffsets,
      output.segmentTracks,
      output.rowIds,
      output.status.count,
      output.status.overflow,
      output.status.requiredCount,
      output.positions,
      output.timestamps
    ];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, segmentCapacity, rowCapacity} = this;
    const {positions, trackOffsets, output, mode} = props;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    const rowCount = positions.length;
    const trackCount = trackOffsets.length - 1;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'positions',
      rowCount * 8,
      'split the tracks over several graphs'
    );
    const timeMode = getTrajectoryTimeMode(props);
    const segmentCounts = createTransientView(graph, `${id}-segment-counts`, 'uint32', trackCount);
    const rowCounts = createTransientView(graph, `${id}-row-counts`, 'uint32', trackCount);
    const segmentOffsets = createTransientView(
      graph,
      `${id}-track-segment-offsets`,
      'uint32',
      trackCount + 1
    );
    const rowOffsets = createTransientView(
      graph,
      `${id}-track-row-offsets`,
      'uint32',
      trackCount + 1
    );
    const totalSegments = createTransientView(graph, `${id}-total-segments`, 'uint32', 1);
    const totalRows = createTransientView(graph, `${id}-total-rows`, 'uint32', 1);
    // `stepTimeDifference` is declared only in gap mode, so non-gap walks need no time buffer.
    const modeSource = `${mode === 'gap' ? getTrajectoryTimeSource(timeMode) : ''}
fn breakBefore(row: u32) -> bool {
  ${
    mode === 'gap'
      ? 'return parameters[parametersOffset] > 0.0 && timeDifferenceRows(row, row - 1u) > parameters[parametersOffset];'
      : mode === 'stop'
        ? `return staypointIds[staypointIdsOffset + row] != ${MOVEMENT_SENTINEL_WGSL};`
        : 'return values[valuesOffset + row] != values[valuesOffset + row - 1u];'
  }
}
fn isExcludedRow(row: u32) -> bool {
  ${mode === 'stop' ? `return staypointIds[staypointIdsOffset + row] != ${MOVEMENT_SENTINEL_WGSL};` : 'return false;'}
}`;
    const modeBindings =
      mode === 'gap'
        ? [
            ...getTrajectoryTimeBindings(props),
            {name: 'parameters', view: props.parameters!, type: 'f32', access: 'read'} as const
          ]
        : mode === 'stop'
          ? [
              {
                name: 'staypointIds',
                view: props.staypointIds!,
                type: 'u32',
                access: 'read'
              } as const
            ]
          : [{name: 'values', view: props.values!, type: 'u32', access: 'read'} as const];
    const overlap = mode === 'value-change';
    const walkBody = (
      emit: boolean
    ) => `let begin = min(trackOffsets[trackOffsetsOffset + index], ${rowCount}u);
  let end = min(trackOffsets[trackOffsetsOffset + index + 1u], ${rowCount}u);
  var segmentTotal = 0u;
  var rowTotal = 0u;
  ${
    emit
      ? `let segmentBase = trackSegmentOffsets[trackSegmentOffsetsOffset + index];
  let rowBase = trackRowOffsets[trackRowOffsetsOffset + index];`
      : ''
  }
  var runStart = begin;
  while (runStart < end) {
    if (isExcludedRow(runStart)) { runStart++; continue; }
    var runEnd = runStart + 1u;
    while (runEnd < end && !breakBefore(runEnd)) { runEnd++; }
    let hasOverlap = ${overlap ? 'runEnd < end' : 'false'};
    let emittedRows = runEnd - runStart + select(0u, 1u, hasOverlap);
    if (emittedRows >= 2u) {
      ${
        emit
          ? `let segmentIndex = segmentBase + segmentTotal;
      let firstOut = rowBase + rowTotal;
      if (segmentIndex < ${segmentCapacity}u) {
        segmentOffsetsOut[segmentOffsetsOutOffset + segmentIndex] = firstOut;
        segmentTracks[segmentTracksOffset + segmentIndex] = index;
      }
      for (var offset = 0u; offset < emittedRows; offset++) {
        let target0 = firstOut + offset;
        if (target0 < ${rowCapacity}u) {
          rowIds[rowIdsOffset + target0] = runStart + offset;
        }
      }`
          : ''
      }
      segmentTotal++;
      rowTotal += emittedRows;
    }
    runStart = runEnd;
  }
  ${emit ? '' : 'segmentCounts[segmentCountsOffset + index] = segmentTotal;\n  rowCounts[rowCountsOffset + index] = rowTotal;'}`;
    const walkBindings = [
      ...modeBindings,
      {name: 'trackOffsets', view: trackOffsets, type: 'u32', access: 'read'} as const
    ];
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-count`,
        operation: OPERATION,
        variant: `${mode}-${timeMode}`,
        invocationCount: trackCount,
        bindings: [
          ...walkBindings,
          {name: 'segmentCounts', view: segmentCounts, type: 'u32', access: 'read_write'},
          {name: 'rowCounts', view: rowCounts, type: 'u32', access: 'read_write'}
        ],
        declarations: modeSource,
        body: walkBody(false)
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-prefix`,
        operation: OPERATION,
        invocationCount: 1,
        bindings: [
          {name: 'segmentCounts', view: segmentCounts, type: 'u32', access: 'read'},
          {name: 'rowCounts', view: rowCounts, type: 'u32', access: 'read'},
          {name: 'trackSegmentOffsets', view: segmentOffsets, type: 'u32', access: 'read_write'},
          {name: 'trackRowOffsets', view: rowOffsets, type: 'u32', access: 'read_write'},
          {name: 'totalSegments', view: totalSegments, type: 'u32', access: 'read_write'},
          {name: 'totalRows', view: totalRows, type: 'u32', access: 'read_write'}
        ],
        body: `var segments = 0u;
  var rows = 0u;
  for (var track = 0u; track < ${trackCount}u; track++) {
    trackSegmentOffsets[trackSegmentOffsetsOffset + track] = segments;
    trackRowOffsets[trackRowOffsetsOffset + track] = rows;
    segments += segmentCounts[segmentCountsOffset + track];
    rows += rowCounts[rowCountsOffset + track];
  }
  trackSegmentOffsets[trackSegmentOffsetsOffset + ${trackCount}u] = segments;
  trackRowOffsets[trackRowOffsetsOffset + ${trackCount}u] = rows;
  totalSegments[totalSegmentsOffset] = segments;
  totalRows[totalRowsOffset] = rows;`
      }),
      createFillNode<Parameters>(graph, {
        id: `${id}-offsets-sentinel`,
        operation: OPERATION,
        view: output.segmentOffsets,
        type: 'u32',
        value: MOVEMENT_SENTINEL_WGSL
      }),
      createFillNode<Parameters>(graph, {
        id: `${id}-tracks-sentinel`,
        operation: OPERATION,
        view: output.segmentTracks,
        type: 'u32',
        value: MOVEMENT_SENTINEL_WGSL
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write`,
        operation: OPERATION,
        variant: `${mode}-${timeMode}`,
        invocationCount: trackCount,
        bindings: [
          ...walkBindings,
          {name: 'trackSegmentOffsets', view: segmentOffsets, type: 'u32', access: 'read'},
          {name: 'trackRowOffsets', view: rowOffsets, type: 'u32', access: 'read'},
          {
            name: 'segmentOffsetsOut',
            view: output.segmentOffsets,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'segmentTracks', view: output.segmentTracks, type: 'u32', access: 'read_write'},
          {name: 'rowIds', view: output.rowIds, type: 'u32', access: 'read_write'}
        ],
        declarations: modeSource,
        body: walkBody(true)
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-status`,
        operation: OPERATION,
        invocationCount: 1,
        bindings: [
          {name: 'totalSegments', view: totalSegments, type: 'u32', access: 'read'},
          {name: 'totalRows', view: totalRows, type: 'u32', access: 'read'},
          {
            name: 'segmentOffsetsOut',
            view: output.segmentOffsets,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'count', view: output.status.count, type: 'u32', access: 'read_write'},
          {name: 'overflow', view: output.status.overflow, type: 'u32', access: 'read_write'},
          ...(output.status.requiredCount
            ? [
                {
                  name: 'requiredCount',
                  view: output.status.requiredCount,
                  type: 'u32',
                  access: 'read_write'
                } as const
              ]
            : [])
        ],
        body: `let segments = totalSegments[totalSegmentsOffset];
  let rows = totalRows[totalRowsOffset];
  let clampedSegments = min(segments, ${segmentCapacity}u);
  count[countOffset] = clampedSegments;
  overflow[overflowOffset] = select(0u, 1u, segments > ${segmentCapacity}u || rows > ${rowCapacity}u);
  segmentOffsetsOut[segmentOffsetsOutOffset + clampedSegments] = min(rows, ${rowCapacity}u);
  ${output.status.requiredCount ? 'requiredCount[requiredCountOffset] = segments;' : ''}`
      })
    ];
    if (output.positions || output.timestamps) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-gather`,
          operation: OPERATION,
          invocationCount: rowCapacity,
          bindings: [
            {name: 'rowIds', view: output.rowIds, type: 'u32', access: 'read'},
            {name: 'totalRows', view: totalRows, type: 'u32', access: 'read'},
            ...(output.positions
              ? [
                  {name: 'positions', view: positions, type: 'f32', access: 'read'} as const,
                  {
                    name: 'outPositions',
                    view: output.positions,
                    type: 'f32',
                    access: 'read_write'
                  } as const
                ]
              : []),
            ...(output.timestamps
              ? [
                  ...getTrajectoryTimeBindings(props),
                  {
                    name: 'outTimes',
                    view: output.timestamps,
                    type: output.timestamps.format === 'uint32x2' ? 'u32' : 'f32',
                    access: 'read_write'
                  } as const
                ]
              : [])
          ],
          body: `if (index < min(totalRows[totalRowsOffset], ${rowCapacity}u)) {
    let row = rowIds[rowIdsOffset + index];
    ${
      output.positions
        ? `outPositions[outPositionsOffset + 2u * index] = positions[positionsOffset + 2u * row];
    outPositions[outPositionsOffset + 2u * index + 1u] = positions[positionsOffset + 2u * row + 1u];`
        : ''
    }
    ${
      output.timestamps
        ? output.timestamps.format === 'uint32x2'
          ? `outTimes[outTimesOffset + 2u * index] = timestamps[timestampsOffset + 2u * row];
    outTimes[outTimesOffset + 2u * index + 1u] = timestamps[timestampsOffset + 2u * row + 1u];`
          : 'outTimes[outTimesOffset + index] = timestamps[timestampsOffset + row];'
        : ''
    }
  }`
        })
      );
    }
    return nodes;
  }
}
