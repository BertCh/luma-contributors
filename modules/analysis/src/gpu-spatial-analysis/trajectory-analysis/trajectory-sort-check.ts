// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {
  GPUStatusScalar,
  GPUTrajectoryGeometryFields
} from '../../analysis-core/contracts/index';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getTrajectoryTimeBindings,
  getTrajectoryTimeMode,
  getTrajectoryTimeSource
} from './trajectory-metrics-kernels';
import {resolveGPUTrajectoryTemporalContext} from './trajectory-common';

const OPERATION = 'addTrajectorySortCheck';

/** Properties of {@link addTrajectorySortCheck}: the trajectory port plus the status output. */
export type AddTrajectorySortCheckProps = GPUTrajectoryGeometryFields & {
  /** Prefix for generated node IDs. Defaults to `'trajectory-sort-check'`. */
  id?: string;
  /** Double-single low parts, with float32 `timestamps`. */
  timestampsLow?: GraphDataView<'float32'>;
  output: {
    status: {
      /**
       * One row: rows out of order. A row counts when its time is earlier than the previous row's
       * in the same track (non-finite float times also count, since they cannot be ordered). A
       * track whose end offset is before its start counts the rows of that reversed range.
       */
      invalidCount: GPUStatusScalar;
    };
  };
};

/**
 * Adds an optional GPU check of the sort-by-track-then-time precondition. Nothing runs unless this
 * is called; no trajectory contributor enables it itself. One invocation per track compares each
 * row with its predecessor, exactly for Int64 words.
 */
export function addTrajectorySortCheck<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: AddTrajectorySortCheckProps
): void {
  const id = props.id ?? 'trajectory-sort-check';
  const {invalidCount} = props.output.status;
  validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
  validateSinglePackedView(props.timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
  validateSinglePackedView(props.trackOffsets, ['uint32'], `${id} trackOffsets`);
  validateSinglePackedView(invalidCount, ['uint32'], `${id} status.invalidCount`);
  if (props.timestampsLow) {
    validateSinglePackedView(props.timestampsLow, ['float32'], `${id} timestampsLow`);
  }
  if (
    props.timestamps.length !== props.positions.length ||
    (props.timestampsLow && props.timestampsLow.length !== props.positions.length)
  ) {
    throw new Error(`${id} timestamps length must equal positions length`);
  }
  if (props.trackOffsets.length < 2) {
    throw new Error(`${id} trackOffsets must contain at least two rows`);
  }
  resolveGPUTrajectoryTemporalContext(
    id,
    props.timestamps,
    props.temporalContext,
    props.timestampsLow
  );
  const inputs = [props.positions, props.timestamps, props.timestampsLow, props.trackOffsets];
  validateGraphViewsBelongToGraph(id, graph, [...inputs, invalidCount]);
  validateGraphOutputsDisjointFromInputs(id, [invalidCount], inputs);
  const rowCount = props.positions.length;
  graph.add(
    createFillNode<Parameters>(graph, {
      id: `${id}-clear`,
      operation: OPERATION,
      view: invalidCount,
      type: 'u32',
      value: '0u',
      componentCount: 1
    })
  );
  graph.add(
    createWGSLKernelNode<Parameters>(graph, {
      id,
      operation: OPERATION,
      variant: getTrajectoryTimeMode(props),
      invocationCount: props.trackOffsets.length - 1,
      bindings: [
        ...getTrajectoryTimeBindings(props),
        {name: 'trackOffsets', view: props.trackOffsets, type: 'u32', access: 'read'},
        {name: 'invalidCount', view: invalidCount, type: 'atomic<u32>', access: 'read_write'}
      ],
      declarations: getTrajectoryTimeSource(getTrajectoryTimeMode(props)),
      body: `let begin = min(trackOffsets[trackOffsetsOffset + index], ${rowCount}u);
  let end = min(trackOffsets[trackOffsetsOffset + index + 1u], ${rowCount}u);
  var disordered = 0u;
  if (end < begin) { disordered = begin - end; }
  for (var row = begin + 1u; row < end; row++) {
    if (!isTimeNonDecreasing(row, row - 1u)) { disordered++; }
  }
  if (disordered > 0u) { atomicAdd(&invalidCount[invalidCountOffset], disordered); }`
    })
  );
}
