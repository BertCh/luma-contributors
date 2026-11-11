// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {resolveGPUTrajectoryTemporalContext} from '../trajectory-analysis/trajectory-common';
import type {GPUTemporalContext} from '../../analysis-core/contracts/index';
import {
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {TIME_WORDS_WGSL} from '../../dataframe-analysis/time-window-filter/time-words';

const OPERATION = 'GPUTrajectoryTrackRelativeTimes';

/** Properties for {@link GPUTrajectoryTrackRelativeTimes}. */
export type GPUTrajectoryTrackRelativeTimesProps = {
  /** Prefix for generated node IDs. Defaults to `'trajectory-track-relative-times'`. */
  id?: string;
  /** Int64 word timestamps, `(low, high)` per row. */
  timestamps: GraphDataView<'uint32x2'>;
  /** Track-to-row offsets with one terminal entry. */
  trackOffsets: GraphDataView<'uint32'>;
  /** Optional declaration of `timestamps`; must be `int64-words`. */
  temporalContext?: GPUTemporalContext;
  /** Caller-owned `float32` output: ticks after the first row of each track. */
  output: GraphDataView<'float32'>;
};

/**
 * Int64 word timestamps to `float32` ticks after each track's own first row. Differences are taken
 * exactly on the words, so a track is exact while it spans fewer than 2^24 ticks and has a relative
 * error below 2^-23 beyond. Its output is the float32 time column that time-ratio simplification
 * (`GPULineSimplification` `metric: 'time-ratio'`) and other float32 trajectory paths read.
 */
export class GPUTrajectoryTrackRelativeTimes implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUTrajectoryTrackRelativeTimesProps;

  constructor(props: GPUTrajectoryTrackRelativeTimesProps) {
    this.id = props.id ?? 'trajectory-track-relative-times';
    this.props = props;
    const id = this.id;
    validateSinglePackedView(props.timestamps, ['uint32x2'], `${id} timestamps`);
    validateSinglePackedView(props.trackOffsets, ['uint32'], `${id} trackOffsets`);
    validateSinglePackedView(props.output, ['float32'], `${id} output`);
    validateViewLength(id, 'output', props.output, props.timestamps.length);
    validateRequirement(
      id,
      props.trackOffsets.length >= 2,
      'trackOffsets must contain at least two rows'
    );
    resolveGPUTrajectoryTemporalContext(id, props.timestamps, props.temporalContext);
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output],
      [props.timestamps, props.trackOffsets]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {timestamps, trackOffsets, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [timestamps, trackOffsets, output]);
    const rowCount = timestamps.length;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'timestamps',
      rowCount * 8,
      'split the tracks over several graphs'
    );
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-convert`,
        operation: OPERATION,
        invocationCount: trackOffsets.length - 1,
        bindings: [
          {name: 'timestamps', view: timestamps, type: 'u32', access: 'read'},
          {name: 'trackOffsets', view: trackOffsets, type: 'u32', access: 'read'},
          {name: 'relative', view: output, type: 'f32', access: 'read_write'}
        ],
        declarations: `${TIME_WORDS_WGSL}
fn timeWords(row: u32) -> vec2<u32> {
  return vec2<u32>(timestamps[timestampsOffset + 2u * row], timestamps[timestampsOffset + 2u * row + 1u]);
}`,
        body: `let begin = min(trackOffsets[trackOffsetsOffset + index], ${rowCount}u);
  let end = min(trackOffsets[trackOffsetsOffset + index + 1u], ${rowCount}u);
  for (var row = begin; row < end; row++) {
    relative[relativeOffset + row] = timeWordsDifferenceToF32(timeWords(row), timeWords(begin));
  }`
      })
    ];
  }
}
