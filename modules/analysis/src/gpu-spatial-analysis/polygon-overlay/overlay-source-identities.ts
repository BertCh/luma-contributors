// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Publish identities after all row-based ownership reductions. Separate dispatches allow the
 * caller's operand and identity columns to occupy disjoint slices of one physical buffer. */
export function createOverlaySourceIdentityNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    rows: GraphDataView<'uint32'>;
    operands: GraphDataView<'uint32'>;
    count: GraphDataView<'uint32'>;
    leftIds?: GraphDataView<'uint32'>;
    rightIds?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  if (!props.leftIds && !props.rightIds) return [];
  const mapped = createTransientView(
    graph,
    `${props.id}-mapped-source-identities`,
    'uint32',
    props.rows.length
  );
  return [
    createWGSLKernelNode(graph, {
      id: `${props.id}-source-feature-ids`,
      operation: props.operation,
      variant: 'source-feature-ids',
      invocationCount: props.rows.length,
      bindings: [
        {name: 'sources', view: props.rows, type: 'u32', access: 'read'},
        {name: 'operands', view: props.operands, type: 'u32', access: 'read'},
        {name: 'count', view: props.count, type: 'u32', access: 'read'},
        ...(props.leftIds
          ? [{name: 'leftIds', view: props.leftIds, type: 'u32' as const, access: 'read' as const}]
          : []),
        ...(props.rightIds
          ? [
              {
                name: 'rightIds',
                view: props.rightIds,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        {name: 'mapped', view: mapped, type: 'u32', access: 'read_write'}
      ],
      body: `let row=sources[sourcesOffset+index];
mapped[mappedOffset+index]=row;
if(index>=count[countOffset]){return;}
if(operands[operandsOffset+index]==0u){
${props.leftIds ? 'mapped[mappedOffset+index]=leftIds[leftIdsOffset+row];' : ''}
}else{
${props.rightIds ? 'mapped[mappedOffset+index]=rightIds[rightIdsOffset+row];' : ''}
}`
    }),
    createWGSLKernelNode(graph, {
      id: `${props.id}-publish-source-feature-ids`,
      operation: props.operation,
      variant: 'publish-source-feature-ids',
      invocationCount: props.rows.length,
      bindings: [
        {name: 'mapped', view: mapped, type: 'u32', access: 'read'},
        {name: 'identities', view: props.rows, type: 'u32', access: 'read_write'}
      ],
      body: 'identities[identitiesOffset+index]=mapped[mappedOffset+index];'
    })
  ];
}
