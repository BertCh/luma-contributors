// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createStableLexicographicIndexSortNodes} from '../../analysis-core/kernels/stable-lexicographic-index-sort';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Sorts emitted pair slots by full-width (row, neighbor), retaining attribute alignment. @internal */
export function getFuzzyContiguityPairSortNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    rows: number;
    pairRows: GraphDataView<'uint32'>;
    pairNeighbors: GraphDataView<'uint32'>;
  }
) {
  const indices = createTransientView(
    graph,
    `${props.id}-indices`,
    'uint32',
    props.pairRows.length
  );
  const keyBits = Math.max(1, Math.ceil(Math.log2(props.rows)));
  const sorted = createStableLexicographicIndexSortNodes(graph, {
    id: props.id,
    operation: 'GPUFuzzyContiguity',
    indices,
    indicesAreIdentity: true,
    algorithm: 'radix',
    keys: [
      {view: props.pairRows, keyBits},
      {view: props.pairNeighbors, keyBits}
    ]
  });
  return {
    sortedIndices: sorted.sortedIndices,
    nodes: [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${props.id}-initialize-indices`,
        operation: 'GPUFuzzyContiguity',
        variant: 'pair-sort-indices',
        invocationCount: indices.length,
        bindings: [{name: 'indices', view: indices, type: 'u32', access: 'read_write'}],
        body: 'indices[indicesOffset + index] = index;'
      }),
      ...sorted.nodes
    ]
  };
}
