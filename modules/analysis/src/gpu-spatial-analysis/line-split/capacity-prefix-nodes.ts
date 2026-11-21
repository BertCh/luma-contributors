// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Parallel selection of a monotone prefix: no serial search and no unsigned midpoint sum. */
export function createCapacityPrefixNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    length: number;
    bindings: WGSLKernelBinding[];
    declarations?: string;
    /** Body must return except at the unique longest fitting prefix end (monotone end offsets). */
    body: string;
  }
) {
  const count = createTransientView(graph, `${props.id}-fitting-count`, 'uint32', 1);
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${props.id}-clear`,
      operation: props.operation,
      invocationCount: 1,
      bindings: [{name: 'fitting', view: count, type: 'u32', access: 'read_write'}],
      body: 'fitting[fittingOffset]=0u;'
    }),
    createWGSLKernelNode(graph, {
      id: props.id,
      operation: props.operation,
      invocationCount: props.length,
      bindings: [
        ...props.bindings,
        {name: 'fitting', view: count, type: 'u32', access: 'read_write'}
      ],
      declarations: props.declarations,
      body: `${props.body}\nfitting[fittingOffset]=index+1u;`
    })
  ];
  return {count, nodes};
}
