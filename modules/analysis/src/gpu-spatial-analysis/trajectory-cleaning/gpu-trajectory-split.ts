// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  createTransientView,
  getViewBindingRange,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {
  GPUCandidateStatusPort,
  GPUCardinalityPort,
  GPUOverflowStatusPort,
  GPUTrajectoryGeometryFields
} from '../../analysis-core/contracts/index';
import {
  validateCapacity,
  validateFiniteNumber,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateTrajectoryBindings,
  validateTrajectoryWritableDispatches
} from '../trajectory-analysis/trajectory-binding-validation';
import {
  createSaturatedCountScanNodes,
  SCAN_TOTAL_WGSL
} from '../../analysis-core/kernels/scan-total';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {resolveGPUTrajectoryTemporalContext} from '../trajectory-analysis/trajectory-common';
import {
  getTrajectoryTimeBindings,
  getTrajectoryTimeMode,
  getTrajectoryTimeSource
} from '../trajectory-analysis/trajectory-metrics-kernels';
import {MOVEMENT_SENTINEL_WGSL} from '../trajectory-staypoints/movement-common';

const OPERATION = 'GPUTrajectorySplit';

// A uint32 count scan needs at most four radix-256 levels for any uint32 track count.
// Two uint64 scans use two uint32 scans each, plus carry, high-clear and saturation passes.
const SPLIT_SCAN_COMMAND_NODE_ALLOWANCE = 4 * 8 + 2 * 3;
// Row indices and track indices are unsigned words in the shaders.
const SPLIT_MAXIMUM_ROW_COUNT = 0xffffffff;
// segmentOffsets needs one terminal row and its fill dispatch count must fit uint32.
const SPLIT_MAXIMUM_SEGMENT_CAPACITY = 0xfffffffe;

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
  validateFiniteNumber(OPERATION, 'maximumGap', settings.maximumGap, 0);
  validateRequirement(
    OPERATION,
    target.length >= GPU_TRAJECTORY_SPLIT_PARAMETER_LENGTH,
    `parameter target must contain ${GPU_TRAJECTORY_SPLIT_PARAMETER_LENGTH} float32 values`
  );
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
  /** Source row of each retained output row, `rowCapacity` rows. Read [0, segmentOffsets[status.count]); the unused capacity tail is unspecified. */
  rowIds: GraphDataView<'uint32'>;
  /** Segment count; `overflow` is 1 when segments or rows exceeded a capacity. */
  status: GPUCardinalityPort & GPUOverflowStatusPort & GPUCandidateStatusPort;
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
  /** Mode 'stop': source staypoint loss to propagate to output.status.candidateOverflow. */
  staypointStatus?: GPUCandidateStatusPort & Partial<GPUOverflowStatusPort>;
  /** Mode `'value-change'`: one uint32 value per row. */
  values?: GraphDataView<'uint32'>;
  output: GPUTrajectorySplitOutput;
};

/**
 * Splits tracks into segments and rewrites the offsets. `'gap'` and `'value-change'` follow
 * MovingPandas 0.23 (pinned by a committed fixture); `'stop'` is the trackintel tripleg rule:
 *
 * - `'gap'` (`ObservationGapSplitter`): a new segment starts after every step strictly longer than
 *   `maximumGap`.
 * - `'stop'`: segments are the maximal runs of rows outside every staypoint, the rule trackintel
 *   uses for triplegs between staypoints. The rows of a staypoint belong to no segment, so a stay
 *   on rows 3 to 8 ends the segment before it at row 2 and starts the next at row 9. This is not
 *   the `StopSplitter` of MovingPandas, whose segments are described as running from the last fix
 *   of one stop to the first fix of the next, both included (not run here).
 * - `'value-change'` (`ValueChangeSplitter`): a new segment starts where the value changes, and
 *   each segment but the last repeats the first row of the next segment.
 *
 * A segment needs at least two rows; shorter ones are dropped (`len(df) > 1` in MovingPandas). The
 * rows of the new port are `rowIds`, optionally with copied `positions` and `timestamps`, so the
 * output feeds another trajectory contributor without readback. In stop mode, wire the source
 * staypoint loss flags through `staypointStatus` into `output.status.candidateOverflow`.
 *
 * Two walks per track: count, then write after parallel saturated prefix scans.
 * Track walks remain serial; scans never wrap or overwrite an earlier output prefix.
 */
export class GPUTrajectorySplit implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUTrajectorySplitProps;
  readonly segmentCapacity: number;
  readonly rowCapacity: number;
  /** Upper bound including both published uint64 scans and optional geometry gathering. */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUTrajectorySplitProps) {
    const {staypointStatus, output: destinations} = props;
    props = {
      ...props,
      staypointStatus:
        staypointStatus &&
        snapshotConstructionFields(staypointStatus, ['overflow', 'candidateOverflow']),
      output: {
        ...destinations,
        status: snapshotConstructionFields(destinations.status, [
          'count',
          'overflow',
          'requiredCount',
          'candidateOverflow'
        ])
      }
    };
    this.id = props.id ?? 'trajectory-split';
    this.props = props;
    const id = this.id;
    const {positions, timestamps, trackOffsets, output, mode} = props;
    validateSinglePackedView(positions, ['float32x2'], `${id} positions`);
    validateSinglePackedView(timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
    validateSinglePackedView(trackOffsets, ['uint32'], `${id} trackOffsets`);
    validateCapacity(id, 'positions rows', positions.length, 0, SPLIT_MAXIMUM_ROW_COUNT);
    validateCapacity(id, 'trackOffsets rows', trackOffsets.length, 1, SPLIT_MAXIMUM_ROW_COUNT);
    validateViewLength(id, 'timestamps', timestamps, positions.length);
    resolveGPUTrajectoryTemporalContext(id, timestamps, props.temporalContext);
    validateRequirement(
      id,
      ['gap', 'stop', 'value-change'].includes(mode),
      "mode must be 'gap', 'stop' or 'value-change'"
    );
    if (mode === 'gap') {
      validateRequirement(id, Boolean(props.parameters), "mode 'gap' requires parameters");
      validateSinglePackedView(props.parameters!, ['float32'], `${id} parameters`);
      validateRequirement(
        id,
        props.parameters!.length >= GPU_TRAJECTORY_SPLIT_PARAMETER_LENGTH,
        'parameters must hold 4 float32 values'
      );
    } else if (mode === 'stop') {
      validateRequirement(id, Boolean(props.staypointIds), "mode 'stop' requires staypointIds");
      validateSinglePackedView(props.staypointIds!, ['uint32'], `${id} staypointIds`);
      validateViewLength(id, 'staypointIds', props.staypointIds!, positions.length);
    } else {
      validateRequirement(id, Boolean(props.values), "mode 'value-change' requires values");
      validateSinglePackedView(props.values!, ['uint32'], `${id} values`);
      validateViewLength(id, 'values', props.values!, positions.length);
    }
    const {overflow, candidateOverflow} = props.staypointStatus ?? {};
    validateRequirement(
      id,
      !(overflow || candidateOverflow) || mode === 'stop',
      "staypointStatus is supported in mode 'stop'"
    );
    validateRequirement(
      id,
      !(overflow || candidateOverflow) || Boolean(output.status.candidateOverflow),
      'output.status.candidateOverflow is required with staypointStatus; pass a one-row uint32 view'
    );
    for (const [name, view] of Object.entries({overflow, candidateOverflow})) {
      if (!view) continue;
      validateSinglePackedView(view, ['uint32'], `${id} staypointStatus.${name}`);
      validateViewLength(id, `staypointStatus.${name}`, view, 1);
    }
    validateSinglePackedView(output.segmentOffsets, ['uint32'], `${id} output.segmentOffsets`);
    validateSinglePackedView(output.segmentTracks, ['uint32'], `${id} output.segmentTracks`);
    validateSinglePackedView(output.rowIds, ['uint32'], `${id} output.rowIds`);
    validateCapacity(
      id,
      'output.segmentTracks',
      output.segmentTracks.length,
      1,
      SPLIT_MAXIMUM_SEGMENT_CAPACITY
    );
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
      validateSinglePackedView(output.positions, ['float32x2'], `${id} output.positions`);
      validateViewLength(id, 'output.positions', output.positions, this.rowCapacity);
    }
    if (output.timestamps) {
      validateSinglePackedView(output.timestamps, [timestamps.format], `${id} output.timestamps`);
      validateViewLength(id, 'output.timestamps', output.timestamps, this.rowCapacity);
    }
    for (const [name, view] of Object.entries(output.status)) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} output.status.${name}`);
        validateViewLength(id, `output.status.${name}`, view, 1);
      }
    }
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.positions, output.timestamps],
      [output.rowIds]
    );
    validateTrajectoryWritableDispatches(id, [
      {
        segmentOffsets: output.segmentOffsets,
        segmentTracks: output.segmentTracks,
        rowIds: output.rowIds
      },
      {
        segmentOffsets: output.segmentOffsets,
        count: output.status.count,
        overflow: output.status.overflow,
        requiredCount: output.status.requiredCount
      },
      {candidateOverflow: output.status.candidateOverflow},
      {positions: output.positions, timestamps: output.timestamps}
    ]);
    this.estimatedCommandNodeCount =
      SPLIT_SCAN_COMMAND_NODE_ALLOWANCE + 6 + Number(Boolean(output.status.candidateOverflow));
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  private getInputViews(): (GraphDataView | undefined)[] {
    const {props} = this;
    return [
      props.positions,
      props.timestamps,
      props.trackOffsets,
      props.parameters,
      props.staypointIds,
      props.staypointStatus?.overflow,
      props.staypointStatus?.candidateOverflow,
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
      output.status.candidateOverflow,
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
    for (const [name, view] of Object.entries({
      positions,
      timestamps: props.timestamps,
      trackOffsets,
      parameters: props.parameters,
      staypointIds: props.staypointIds,
      values: props.values,
      'output.segmentOffsets': output.segmentOffsets,
      'output.segmentTracks': output.segmentTracks,
      'output.rowIds': output.rowIds,
      'output.positions': output.positions,
      'output.timestamps': output.timestamps,
      ...Object.fromEntries(
        Object.entries(output.status).map(([name, view]) => [`output.status.${name}`, view])
      )
    })) {
      if (view) {
        validateGPUStorageBindingByteLength(
          graph.device,
          id,
          name,
          getViewBindingRange(view).size,
          name === 'positions'
            ? 'split the tracks over several graphs'
            : 'reduce the view binding range'
        );
      }
    }
    validateTrajectoryBindings(graph, id, props);
    // Count buffers, prefixes and all uint64-scan intermediates hold at most one word per
    // track. Hierarchical block summaries are smaller; the two total words are separate.
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'track scan scratch',
      trackCount * 4,
      'split the tracks over several graphs'
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'total scratch',
      4,
      'use a device with storage bindings'
    );
    const timeMode = getTrajectoryTimeMode(props);
    const segmentCounts = createTransientView(graph, `${id}-segment-counts`, 'uint32', trackCount);
    const rowCounts = createTransientView(graph, `${id}-row-counts`, 'uint32', trackCount);
    const segmentOffsets = createTransientView(
      graph,
      `${id}-track-segment-offsets`,
      'uint32',
      trackCount
    );
    const rowOffsets = createTransientView(graph, `${id}-track-row-offsets`, 'uint32', trackCount);
    const totalSegments = createTransientView(graph, `${id}-total-segments`, 'uint32', 1);
    const totalRows = createTransientView(graph, `${id}-total-rows`, 'uint32', 1);
    // `stepTimeDifference` is declared only in gap mode, so non-gap walks need no time buffer.
    const modeSource = `${SCAN_TOTAL_WGSL}
${mode === 'gap' ? getTrajectoryTimeSource(timeMode) : ''}
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
          ? `let segmentIndex = getSaturatedScanSum(segmentBase, segmentTotal);
      let firstOut = getSaturatedScanSum(rowBase, rowTotal);
      if (segmentIndex < ${segmentCapacity}u) {
        segmentOffsetsOut[segmentOffsetsOutOffset + segmentIndex] = firstOut;
        segmentTracks[segmentTracksOffset + segmentIndex] = index;
      }
      for (var offset = 0u; offset < emittedRows; offset++) {
        let target0 = getSaturatedScanSum(firstOut, offset);
        if (target0 < ${rowCapacity}u) {
          rowIds[rowIdsOffset + target0] = runStart + offset;
        }
      }`
          : ''
      }
      segmentTotal++;
      rowTotal = getSaturatedScanSum(rowTotal, emittedRows);
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
      ...createSaturatedCountScanNodes(graph, {
        id: `${id}-segment-prefix`,
        operation: OPERATION,
        counts: segmentCounts,
        offsets: segmentOffsets,
        rowCount: trackCount,
        total: totalSegments
      }),
      ...createSaturatedCountScanNodes(graph, {
        id: `${id}-row-prefix`,
        operation: OPERATION,
        counts: rowCounts,
        offsets: rowOffsets,
        rowCount: trackCount,
        total: totalRows
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
    if (output.status.candidateOverflow) {
      const lossBindings = [
        props.staypointStatus?.overflow,
        props.staypointStatus?.candidateOverflow
      ].flatMap((view, index) =>
        view ? [{name: `loss${index}`, view, type: 'u32' as const, access: 'read' as const}] : []
      );
      const hasLoss =
        lossBindings.map(({name}) => `${name}[${name}Offset] != 0u`).join(' || ') || 'false';
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-staypoint-loss-status`,
          operation: OPERATION,
          invocationCount: 1,
          bindings: [
            ...lossBindings,
            {
              name: 'candidateOverflow',
              view: output.status.candidateOverflow,
              type: 'u32',
              access: 'read_write'
            }
          ],
          body: `candidateOverflow[candidateOverflowOffset] = select(0u, 1u, ${hasLoss});`
        })
      );
    }
    return nodes;
  }
}
