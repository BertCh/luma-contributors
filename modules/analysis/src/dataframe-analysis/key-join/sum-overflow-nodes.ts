// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createGroupStatisticsKernelNode} from '../group-statistics/group-statistics-kernel';
import {getFixedPointWGSL} from '../../analysis-core/kernels/cell-key-wgsl';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateGPUStorageBindingByteLength} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Optional caller-scale sum-loss status, separate from capacity loss and float reductions. @internal */
export function getSumOverflowNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    capacity: number;
    sumScale: number;
    /** Reuse group runtime chunk metadata rather than specializing each chunk offset/size. */
    useGroupDispatchUniforms?: boolean;
    chunks: readonly {
      values: GraphDataView<'float32' | 'uint32'>;
      groups: GraphDataView<'uint32'>;
      rowOffset?: number;
      permutation?: GraphDataView<'uint32'>;
    }[];
    /** If supplied, only nonzero groups contribute published results and numeric loss. */
    retainedGroups?: GraphDataView<'uint32'>;
    output: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  // Split the exact 96-bit caller-scale total to keep each binding no larger than the existing
  // two-word sums. The extra high word retains carries across every chunk, until publication.
  validateGPUStorageBindingByteLength(
    graph.device,
    props.id,
    'sum overflow words',
    props.capacity * 8,
    'reduce sum capacity'
  );
  validateGPUStorageBindingByteLength(
    graph.device,
    props.id,
    'sum overflow guards',
    (props.capacity + 1) * 4,
    'reduce sum capacity'
  );
  const words = createTransientView(graph, `${props.id}-words`, 'uint32x2', props.capacity);
  const guards = createTransientView(graph, `${props.id}-guards`, 'uint32', props.capacity + 1);
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${props.id}-clear`,
      operation: 'DataframeSumOverflow',
      variant: 'clear',
      invocationCount: props.capacity,
      bindings: [
        {name: 'words', view: words, type: 'u32', access: 'read_write'},
        {name: 'guards', view: guards, type: 'u32', access: 'read_write'}
      ],
      body: `words[wordsOffset + 2u * index] = 0u;
  words[wordsOffset + 2u * index + 1u] = 0u;
  guards[guardsOffset + index] = 0u;
  if (index == 0u) { guards[guardsOffset + ${props.capacity}u] = 0u; }`
    })
  ];
  const createAccumulationNode = props.useGroupDispatchUniforms
    ? createGroupStatisticsKernelNode
    : createWGSLKernelNode;
  for (const [chunkIndex, chunk] of props.chunks.entries()) {
    nodes.push(
      createAccumulationNode<Parameters>(graph, {
        rowOffset: chunk.rowOffset ?? 0,
        id: `${props.id}-accumulate-${chunkIndex}`,
        operation: 'DataframeSumOverflow',
        variant: 'accumulate',
        invocationCount: chunk.values.length,
        bindings: [
          {name: 'values', view: chunk.values, type: 'u32', access: 'read'},
          {name: 'groups', view: chunk.groups, type: 'u32', access: 'read'},
          ...(props.retainedGroups
            ? [
                {
                  name: 'retainedGroups',
                  view: props.retainedGroups,
                  type: 'u32',
                  access: 'read'
                } as const
              ]
            : []),
          ...(chunk.permutation
            ? [{name: 'permutation', view: chunk.permutation, type: 'u32', access: 'read'} as const]
            : []),
          {name: 'words', view: words, type: 'atomic<u32>', access: 'read_write'},
          {name: 'guards', view: guards, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations: getFixedPointWGSL(props.sumScale),
        body: `let group = groups[groupsOffset + index + ${props.useGroupDispatchUniforms ? 'GROUP_ROW_OFFSET' : `${chunk.rowOffset ?? 0}u`}];
  if (group == 0xffffffffu) { return; }
  ${props.retainedGroups ? 'if (retainedGroups[retainedGroupsOffset + group] == 0u) { return; }' : ''}
  let row = ${chunk.permutation ? 'permutation[permutationOffset + index]' : 'index'};
  let bits = values[valuesOffset + row];
  if ((bits & 0x7f800000u) == 0x7f800000u) { return; }
  let value = bitcast<f32>(bits);
  if (abs(value * SUM_SCALE) > MAXIMUM_SCALED_VALUE) {
    atomicOr(&guards[guardsOffset + ${props.capacity}u], 1u);
  }
  let scaled = cellScaleValue(value);
  let previousLow = atomicAdd(&words[wordsOffset + 2u * group], scaled.y);
  let high = scaled.x + select(0u, 1u, previousLow + scaled.y < previousLow);
  var guardCarry = select(0u, 0xffffffffu, (scaled.x >> 31u) != 0u) + select(0u, 1u, high < scaled.x);
  if (high != 0u) {
    let previousHigh = atomicAdd(&words[wordsOffset + 2u * group + 1u], high);
    guardCarry += select(0u, 1u, previousHigh + high < previousHigh);
  }
  if (guardCarry != 0u) { atomicAdd(&guards[guardsOffset + group], guardCarry); }`
      })
    );
  }
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${props.id}-check`,
      operation: 'DataframeSumOverflow',
      variant: 'check',
      invocationCount: props.capacity,
      bindings: [
        {name: 'words', view: words, type: 'u32', access: 'read'},
        {name: 'guards', view: guards, type: 'atomic<u32>', access: 'read_write'}
      ],
      body: `let signWord = select(0u, 0xffffffffu, (words[wordsOffset + 2u * index + 1u] >> 31u) != 0u);
  if (atomicLoad(&guards[guardsOffset + index]) != signWord) {
    atomicOr(&guards[guardsOffset + ${props.capacity}u], 1u);
  }`
    }),
    createWGSLKernelNode(graph, {
      id: `${props.id}-publish`,
      operation: 'DataframeSumOverflow',
      variant: 'publish',
      invocationCount: 1,
      bindings: [
        {name: 'guards', view: guards, type: 'u32', access: 'read'},
        {name: 'sumOverflow', view: props.output, type: 'u32', access: 'read_write'}
      ],
      body: `sumOverflow[sumOverflowOffset] = guards[guardsOffset + ${props.capacity}u];`
    })
  );
  return nodes;
}
