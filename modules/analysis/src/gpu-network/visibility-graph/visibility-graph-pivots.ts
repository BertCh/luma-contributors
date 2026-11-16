// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {VISIBILITY_GRAPH_CSR_MAXIMUM_POINT_COUNT} from '../../gpu-spatial-analysis/urban-morphology/visibility-graph-csr';

/** Twenty input bits cover the one-million-point CSR cap; no intermediate exceeds three million. */
const VISIBILITY_GRAPH_PIVOT_POINT_BIT_COUNT = Math.ceil(
  Math.log2(VISIBILITY_GRAPH_CSR_MAXIMUM_POINT_COUNT + 1)
);

/** Exact systematic pivot selection for validated CSR counts, using bounded uint32 arithmetic. @internal */
export const VISIBILITY_GRAPH_PIVOT_WGSL = `
fn selectVisibilityGraphPivot(pivotIndex:u32,pointCount:u32,pivotCount:u32)->u32{
 if(pointCount==pivotCount){return pivotIndex;}
 var selected=0u;var remainder=0u;
 // Horner multiplication of the point-count bits with quotient/remainder carried separately.
 // remainder < pivotCount and pivotIndex < pivotCount, so next < 3*pivotCount <= 3,000,000.
 // Each next quotient digit is 0, 1 or 2: no wide arithmetic or reciprocal approximation.
 for(var remaining=${VISIBILITY_GRAPH_PIVOT_POINT_BIT_COUNT}u;remaining>0u;remaining--){
  let incoming=(pointCount>>(remaining-1u))&1u;
  let next=2u*remainder+select(0u,pivotIndex,incoming!=0u);
  let digit=select(select(0u,1u,next>=pivotCount),2u,next>=2u*pivotCount);
  selected=2u*selected+digit;remainder=next-digit*pivotCount;
 }
 return selected;
}
`;

/** Exact floor(pivotIndex * pointCount / pivotCount), without wide multiplication or division. @internal */
export function createVisibilityGraphPivotsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; operation: string; pointCount: number; pivots: GraphDataView<'uint32'>}
) {
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: props.operation,
    invocationCount: props.pivots.length,
    declarations: VISIBILITY_GRAPH_PIVOT_WGSL,
    bindings: [{name: 'pivots', view: props.pivots, type: 'u32', access: 'read_write'}],
    body: `pivots[pivotsOffset+index]=selectVisibilityGraphPivot(index,${props.pointCount}u,${props.pivots.length}u);`
  });
}
