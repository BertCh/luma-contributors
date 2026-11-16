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
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {
  GPUCandidateStatusPort,
  GPUCardinalityPort,
  GPUOverflowStatusPort,
  GPUTrajectoryGeometryFields
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
} from '../trajectory-staypoints/movement-common';

const OPERATION = 'GPUTrajectoryThinning';

/** How {@link GPUTrajectoryThinning} decides the next kept row. */
export type GPUTrajectoryThinningMode = 'minimum-distance' | 'minimum-time';

/** Number of float32 elements in a thinning parameter buffer. */
export const GPU_TRAJECTORY_THINNING_PARAMETER_LENGTH = 4;

/** Default compile-time forward search cap per row. */
export const GPU_TRAJECTORY_THINNING_DEFAULT_MAXIMUM_SEARCH_ROWS = 4096;

/** Largest accepted forward search cap per row. */
export const GPU_TRAJECTORY_THINNING_MAXIMUM_SEARCH_ROWS = 1 << 24;

/** Per-frame settings of {@link GPUTrajectoryThinning}. */
export type GPUTrajectoryThinningSettings = {
  /** Minimum distance (context length unit) or time (timestamps' unit) between consecutive kept rows. */
  tolerance: number;
};

/**
 * Packs thinning settings into the float32 layout read by {@link GPUTrajectoryThinning}.
 *
 * Layout: `[tolerance, 0, 0, 0]`.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If `tolerance` is negative or not finite.
 */
export function getGPUTrajectoryThinningParameterValues(
  settings: GPUTrajectoryThinningSettings,
  target: Float32Array = new Float32Array(GPU_TRAJECTORY_THINNING_PARAMETER_LENGTH)
): Float32Array {
  if (!Number.isFinite(Math.fround(settings.tolerance)) || settings.tolerance < 0) {
    throw new Error('Thinning tolerance must be finite and non-negative');
  }
  target.set([settings.tolerance, 0, 0, 0]);
  return target;
}

/** Caller-owned outputs of {@link GPUTrajectoryThinning}. */
export type GPUTrajectoryThinningOutput = {
  /** Kept row indices in row order, rows past the count hold the sentinel. Capacity rows. */
  rowIds: GraphDataView<'uint32'>;
  /** New per-track offsets into `rowIds`, `trackCount + 1` rows. */
  trackOffsets: GraphDataView<'uint32'>;
  /** Kept row count, overflow, and `candidateOverflow` when a search hit `maximumSearchRows`. */
  status: GPUCardinalityPort & GPUOverflowStatusPort & GPUCandidateStatusPort;
  /** Optional `1`/`0` validity of the rows: `1` for a kept row, `positions.length` rows. */
  validity?: GraphDataView<'uint32'>;
  /** Optional copied positions of the kept rows, capacity rows. */
  positions?: GraphDataView<'float32x2'>;
  /** Optional copied timestamps of the kept rows, capacity rows, the input format. */
  timestamps?: GPUTrajectoryGeometryFields['timestamps'];
};

/** Properties for {@link GPUTrajectoryThinning}. */
export type GPUTrajectoryThinningProps = GPUTrajectoryGeometryFields & {
  /** Prefix for node and transient IDs. Defaults to `'trajectory-thinning'`. */
  id?: string;
  mode: GPUTrajectoryThinningMode;
  /** Planar or longitude/latitude great-circle, for `'minimum-distance'`. */
  spatialContext?: GPUSpatialContext;
  /** Per-frame float32 view of {@link GPU_TRAJECTORY_THINNING_PARAMETER_LENGTH} elements. */
  parameters: GraphDataView<'float32'>;
  /** Forward-search cap per row (default 4096, at most 16,777,216). */
  maximumSearchRows?: number;
  output: GPUTrajectoryThinningOutput;
};

/**
 * Minimum-distance and minimum-time generalization with MovingPandas' sequential semantics in a
 * bounded number of passes: the first and last rows of a track are always kept, and each kept row
 * is followed by the first later row at or beyond the tolerance. Like staypoints, the sequential
 * rule becomes a bounded forward search per row plus pointer doubling to the chain from each
 * track start; a search that hits `maximumSearchRows` reports `status.candidateOverflow`.
 */
export class GPUTrajectoryThinning implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUTrajectoryThinningProps;
  readonly spatialContext: GPUSpatialContext;
  readonly maximumSearchRows: number;
  readonly capacity: number;

  constructor(input: GPUTrajectoryThinningProps) {
    this.id = input.id ?? 'trajectory-thinning';
    const id = this.id;
    const props = input;
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT
    );
    const {positions, timestamps, trackOffsets, parameters, output, mode} = props;
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
      parameters.length >= GPU_TRAJECTORY_THINNING_PARAMETER_LENGTH,
      'parameters must hold 4 float32 values'
    );
    validateRequirement(
      id,
      mode === 'minimum-distance' || mode === 'minimum-time',
      "mode must be 'minimum-distance' or 'minimum-time'"
    );
    this.maximumSearchRows =
      props.maximumSearchRows ?? GPU_TRAJECTORY_THINNING_DEFAULT_MAXIMUM_SEARCH_ROWS;
    validateMaximumCount(
      id,
      'maximumSearchRows',
      this.maximumSearchRows,
      GPU_TRAJECTORY_THINNING_MAXIMUM_SEARCH_ROWS
    );
    validateSinglePackedView(output.rowIds, ['uint32'], `${id} output.rowIds`);
    validateSinglePackedView(output.trackOffsets, ['uint32'], `${id} output.trackOffsets`);
    validateCapacity(id, 'output.rowIds', output.rowIds.length);
    this.capacity = output.rowIds.length;
    validateViewLength(id, 'output.trackOffsets', output.trackOffsets, trackOffsets.length);
    if (output.validity) {
      validateSinglePackedView(output.validity, ['uint32'], `${id} output.validity`);
      validateViewLength(id, 'output.validity', output.validity, positions.length);
    }
    if (output.positions) {
      validateSinglePackedView(output.positions, ['float32x2'], `${id} output.positions`);
      validateViewLength(id, 'output.positions', output.positions, this.capacity);
    }
    if (output.timestamps) {
      validateSinglePackedView(output.timestamps, [timestamps.format], `${id} output.timestamps`);
      validateViewLength(id, 'output.timestamps', output.timestamps, this.capacity);
    }
    validateTrajectoryOutputRanges(this.id, this.props.output);
    validateTrajectoryWritableDispatches(this.id, [
      {positions: output.positions, timestamps: output.timestamps},
      {
        count: output.status.count,
        overflow: output.status.overflow,
        requiredCount: output.status.requiredCount
      }
    ]);
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.positions, output.timestamps],
      [output.rowIds]
    );
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), [
      positions,
      timestamps,
      trackOffsets,
      parameters
    ]);
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {output} = this.props;
    return [
      output.rowIds,
      output.trackOffsets,
      output.status.count,
      output.status.overflow,
      output.status.requiredCount,
      output.status.candidateOverflow,
      output.validity,
      output.positions,
      output.timestamps
    ];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, capacity} = this;
    const {positions, timestamps, trackOffsets, parameters, output, mode} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      timestamps,
      trackOffsets,
      parameters,
      ...this.getOutputViews()
    ]);
    const rowCount = positions.length;
    const trackCount = trackOffsets.length - 1;
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
    const rowStarts = createTransientView(graph, `${id}-row-starts`, 'uint32', rowCount);
    const rowEnds = createTransientView(graph, `${id}-row-ends`, 'uint32', rowCount);
    const rowTracks = createTransientView(graph, `${id}-row-tracks`, 'uint32', rowCount);
    const next = createTransientView(graph, `${id}-next`, 'uint32', rowCount);
    const reach = createTransientView(graph, `${id}-reach`, 'uint32', rowCount);
    const keep = output.validity ?? createTransientView(graph, `${id}-keep`, 'uint32', rowCount);
    const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', rowCount);
    const keptRows = createTransientView(graph, `${id}-kept-rows`, 'uint32', rowCount);
    const keptTotal = createTransientView(graph, `${id}-kept-total`, 'uint32', 1);
    const truncated = createTransientView(graph, `${id}-truncated`, 'uint32', rowCount);
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
        variant: `${mode}-${timeMode}`,
        invocationCount: rowCount,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          ...getTrajectoryTimeBindings(props),
          {name: 'rowEnds', view: rowEnds, type: 'u32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          {name: 'next', view: next, type: 'u32', access: 'read_write'},
          {name: 'truncated', view: truncated, type: 'u32', access: 'read_write'}
        ],
        declarations: `${getTrajectoryTimeSource(timeMode)}\n${getMovementDistanceWGSL(radius)}`,
        body: `let trackEnd = rowEnds[rowEndsOffset + index];
  var successor = index;
  var wasTruncated = 0u;
  if (trackEnd != ${MOVEMENT_SENTINEL_WGSL}) {
    let searchEnd = min(trackEnd, index + 1u + ${this.maximumSearchRows}u);
    let tolerance = parameters[parametersOffset];
    var found = false;
    for (var row = index + 1u; row < searchEnd; row++) {
      if (${mode === 'minimum-distance' ? 'rowDistance(row, index)' : 'timeDifferenceRows(row, index)'} >= tolerance) {
        successor = row;
        found = true;
        break;
      }
    }
    if (!found && searchEnd < trackEnd) {
      successor = searchEnd;
      wasTruncated = 1u;
    }
  }
  next[nextOffset + index] = successor;
  truncated[truncatedOffset + index] = wasTruncated;`
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
        id: `${id}-keep`,
        operation: OPERATION,
        invocationCount: rowCount,
        bindings: [
          {name: 'reach', view: reach, type: 'u32', access: 'read'},
          {name: 'rowEnds', view: rowEnds, type: 'u32', access: 'read'},
          {name: 'keep', view: keep, type: 'u32', access: 'read_write'}
        ],
        body: `let trackEnd = rowEnds[rowEndsOffset + index];
  keep[keepOffset + index] = select(0u, 1u, trackEnd != ${MOVEMENT_SENTINEL_WGSL} && (reach[reachOffset + index] != 0u || index + 1u == trackEnd));`
      }),
      createRowIdsNode<Parameters>(graph, {id: `${id}-row-ids`, operation: OPERATION, rowIds}),
      ...new GPUCompaction({
        id: `${id}-compact`,
        input: rowIds,
        flags: keep,
        output: keptRows,
        count: keptTotal
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-rows-out`,
        operation: OPERATION,
        invocationCount: capacity,
        bindings: [
          {name: 'keptRows', view: keptRows, type: 'u32', access: 'read'},
          {name: 'keptTotal', view: keptTotal, type: 'u32', access: 'read'},
          {name: 'rowIds', view: output.rowIds, type: 'u32', access: 'read_write'}
        ],
        body: `rowIds[rowIdsOffset + index] = select(${MOVEMENT_SENTINEL_WGSL}, keptRows[keptRowsOffset + index], index < keptTotal[keptTotalOffset]);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-offsets`,
        operation: OPERATION,
        invocationCount: trackCount + 1,
        bindings: [
          {name: 'keptRows', view: keptRows, type: 'u32', access: 'read'},
          {name: 'keptTotal', view: keptTotal, type: 'u32', access: 'read'},
          {name: 'trackOffsets', view: trackOffsets, type: 'u32', access: 'read'},
          {name: 'outOffsets', view: output.trackOffsets, type: 'u32', access: 'read_write'}
        ],
        // Stable compaction preserves source-row order. Each track boundary independently
        // searches the compact list, including duplicate boundaries for empty tracks.
        body: `let boundary = min(trackOffsets[trackOffsetsOffset + index], ${rowCount}u);
  var low = 0u;
  var high = min(keptTotal[keptTotalOffset], ${rowCount}u);
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (keptRows[keptRowsOffset + middle] < boundary) { low = middle + 1u; }
    else { high = middle; }
  }
  outOffsets[outOffsetsOffset + index] = min(low, ${capacity}u);`
      }),
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        requiredCount: keptTotal,
        output: {ids: output.rowIds, status: output.status}
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
            {name: 'truncated', view: truncated, type: 'u32', access: 'read'},
            {
              name: 'candidateOverflow',
              view: output.status.candidateOverflow,
              type: 'atomic<u32>',
              access: 'read_write'
            }
          ],
          body: `if (reach[reachOffset + index] != 0u && truncated[truncatedOffset + index] != 0u) { atomicStore(&candidateOverflow[candidateOverflowOffset], 1u); }`
        })
      );
    }
    if (output.positions || output.timestamps) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-gather`,
          operation: OPERATION,
          invocationCount: capacity,
          bindings: [
            {name: 'rowIds', view: output.rowIds, type: 'u32', access: 'read'},
            {name: 'keptTotal', view: keptTotal, type: 'u32', access: 'read'},
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
          body: `if (index < min(keptTotal[keptTotalOffset], ${capacity}u)) {
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
