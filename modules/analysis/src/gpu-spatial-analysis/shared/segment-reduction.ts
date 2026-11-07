// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Reduction applied to each CSR segment. */
export type SegmentReductionOperation = 'sum' | 'min' | 'max';

/** Properties for {@link createSegmentReductionNode}. */
export type SegmentReductionProps = {
  id: string;
  operation: string;
  input: GraphDataView<'float32'>;
  /** CSR-style element offsets; length is `output.length + 1`. */
  segmentOffsets: GraphDataView<'uint32'>;
  /** One reduced value per segment; empty segments write `0`. */
  output: GraphDataView<'float32'>;
  reduction: SegmentReductionOperation;
};

/**
 * Reduces float32 rows independently per CSR segment, one invocation per segment walking its rows
 * in order (so sums are deterministic). Slower than a workgroup-per-segment tree for very large
 * segments, but dispatch is bounded by the kernel helper for any segment count.
 */
export function createSegmentReductionNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SegmentReductionProps
): GPUCommandNode<Parameters> {
  const identity =
    props.reduction === 'sum'
      ? '0.0'
      : props.reduction === 'min'
        ? '3.402823466e+38'
        : '-3.402823466e+38';
  const combine = props.reduction === 'sum' ? 'value + next' : `${props.reduction}(value, next)`;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: `segment-${props.reduction}`,
    bindings: [
      {name: 'inputValues', view: props.input, type: 'f32', access: 'read'},
      {name: 'segmentOffsets', view: props.segmentOffsets, type: 'u32', access: 'read'},
      {name: 'outputValues', view: props.output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.output.length,
    body: `let begin = segmentOffsets[segmentOffsetsOffset + index];
  let end = segmentOffsets[segmentOffsetsOffset + index + 1u];
  var value: f32 = ${identity};
  for (var row = begin; row < end; row++) {
    let next = inputValues[inputValuesOffset + row];
    value = ${combine};
  }
  outputValues[outputValuesOffset + index] = select(value, 0.0, begin == end);`
  });
}
