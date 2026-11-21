// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GPUSegmentIntersection, type GPUSegmentIntersectionProps} from './gpu-segment-intersection';

/**
 * Internal pair stages for consumers whose overflow denotes an incomplete intermediate pair list.
 * Preserve the child's distinct final/candidate flags, then fold both into the consumer's loss word.
 */
export function createSegmentPairNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUSegmentIntersectionProps
): readonly GPUCommandNode<Parameters>[] {
  const id = props.id ?? 'segment-pairs';
  const pairs = props.output.pairs;
  const overflow =
    pairs.status.overflow ?? createTransientView(graph, `${id}-pair-overflow`, 'uint32', 1);
  const candidateOverflow =
    pairs.status.candidateOverflow ??
    createTransientView(graph, `${id}-candidate-overflow`, 'uint32', 1);
  return [
    ...new GPUSegmentIntersection({
      ...props,
      output: {
        ...props.output,
        pairs: {...pairs, status: {...pairs.status, overflow, candidateOverflow}}
      }
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-completeness`,
      operation: 'GPUSegmentPairCompleteness',
      invocationCount: 1,
      bindings: [
        {name: 'candidateOverflow', view: candidateOverflow, type: 'u32', access: 'read'},
        {name: 'overflow', view: overflow, type: 'u32', access: 'read_write'}
      ],
      body: 'overflow[overflowOffset] = select(0u, 1u, overflow[overflowOffset] != 0u || candidateOverflow[candidateOverflowOffset] != 0u);'
    })
  ];
}
