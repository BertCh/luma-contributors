// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma-contributors/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';

type SparseResistanceComponentsProps = {
  id: string;
  edges: GraphDataView<'uint32x2'>;
  conductances: GraphDataView<'float32'>;
  components: GraphDataView<'uint32'>;
  status: GraphDataView<'uint32x4'>;
  vertexCount: number;
  source: number;
  sink: number;
  maximumIterations: number;
  /** Internal budgets can be reduced to exercise the exact fallback. */
  maximumUnionAttempts?: number;
  maximumRootSteps?: number;
};

/** Bounded edge unions with checked completion and an exact serial fallback. @internal */
export function addSparseResistanceComponents<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SparseResistanceComponentsProps
) {
  const {id, vertexCount, source, sink, maximumIterations, components, status} = props;
  const edgeCount = props.edges.length;
  const maximumUnionAttempts = props.maximumUnionAttempts ?? 64;
  const maximumRootSteps = props.maximumRootSteps ?? 64;
  const completionNodeId = `${id}-components-complete`;
  // Internal budgets remain bounded even when deliberately reduced for fallback validation.
  assert(
    Number.isInteger(maximumUnionAttempts) &&
      maximumUnionAttempts >= 0 &&
      maximumUnionAttempts <= 64
  );
  assert(Number.isInteger(maximumRootSteps) && maximumRootSteps >= 0 && maximumRootSteps <= 64);
  // Union exhaustion, root exhaustion, failed certificate, and executed fallback.
  const control = getRecipeView(graph, `${id}-component-control`, 'uint32x4', 1);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-components-initialize`,
      operation: id,
      invocationCount: Math.max(vertexCount, 4),
      bindings: [
        {name: 'parents', view: components, type: 'u32', access: 'read_write'},
        {name: 'status', view: status, type: 'u32', access: 'read_write'},
        {name: 'control', view: control, type: 'u32', access: 'read_write'}
      ],
      body: `if (index < ${vertexCount}u) { parents[parentsOffset + index] = index; }
if (index < 4u) { control[controlOffset + index] = 0u; }
if (index == 0u) {
  status[statusOffset] = 0u;
  status[statusOffset + 1u] = ${maximumIterations}u;
  status[statusOffset + 2u] = 0u;
  status[statusOffset + 3u] = 0u;
}`
    })
  );
  const rootDeclaration = `
fn findRoot(vertex: u32) -> u32 {
  var current = vertex;
  for (var step = 0u; step < ${maximumRootSteps}u; step++) {
    let parent = atomicLoad(&parents[parentsOffset + current]);
    if (parent == current) { return current; }
    let grandparent = atomicLoad(&parents[parentsOffset + parent]);
    atomicMin(&parents[parentsOffset + current], grandparent);
    current = grandparent;
  }
  atomicStore(&control[controlOffset + 1u], 1u);
  return 0xffffffffu;
}`;
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-components-union`,
      operation: id,
      invocationCount: edgeCount,
      declarations: IS_FINITE_VALUE_WGSL + rootDeclaration,
      bindings: [
        {name: 'edges', view: props.edges, type: 'u32', access: 'read'},
        {name: 'weights', view: props.conductances, type: 'f32', access: 'read'},
        {name: 'parents', view: components, type: 'atomic<u32>', access: 'read_write'},
        {name: 'status', view: status, type: 'atomic<u32>', access: 'read_write'},
        {name: 'control', view: control, type: 'atomic<u32>', access: 'read_write'}
      ],
      body: `let first = edges[edgesOffset + 2u * index];
let second = edges[edgesOffset + 2u * index + 1u];
let conductance = weights[weightsOffset + index];
if (first >= ${vertexCount}u || second >= ${vertexCount}u || !isFiniteValue(conductance) || conductance < 0.0) {
  atomicAdd(&status[statusOffset + 3u], 1u);
  return;
}
if (conductance == 0.0) { return; }
for (var attempt = 0u; attempt < ${maximumUnionAttempts}u; attempt++) {
  let firstRoot = findRoot(first);
  let secondRoot = findRoot(second);
  if (firstRoot == 0xffffffffu || secondRoot == 0xffffffffu) { return; }
  if (firstRoot == secondRoot) { return; }
  let higher = max(firstRoot, secondRoot);
  let lower = min(firstRoot, secondRoot);
  // Only a current root can be hooked; spurious failure retries within the fixed budget.
  let exchange = atomicCompareExchangeWeak(&parents[parentsOffset + higher], higher, lower);
  if (exchange.exchanged) { return; }
}
atomicStore(&control[controlOffset], 1u);`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-components-flatten`,
      operation: id,
      invocationCount: vertexCount,
      declarations: rootDeclaration,
      bindings: [
        {name: 'parents', view: components, type: 'atomic<u32>', access: 'read_write'},
        {name: 'control', view: control, type: 'atomic<u32>', access: 'read_write'}
      ],
      body: `let root = findRoot(index);
if (root != 0xffffffffu) { atomicMin(&parents[parentsOffset + index], root); }`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-components-verify`,
      operation: id,
      invocationCount: Math.max(vertexCount, edgeCount),
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'edges', view: props.edges, type: 'u32', access: 'read'},
        {name: 'weights', view: props.conductances, type: 'f32', access: 'read'},
        {name: 'parents', view: components, type: 'u32', access: 'read'},
        {name: 'control', view: control, type: 'atomic<u32>', access: 'read_write'}
      ],
      body: `if (index < ${vertexCount}u) {
  let parent = parents[parentsOffset + index];
  if (parent > index) { atomicStore(&control[controlOffset + 2u], 1u); }
  else if (parents[parentsOffset + parent] != parent) { atomicStore(&control[controlOffset + 2u], 1u); }
}
if (index < ${edgeCount}u) {
  let first = edges[edgesOffset + 2u * index];
  let second = edges[edgesOffset + 2u * index + 1u];
  let conductance = weights[weightsOffset + index];
  if (first < ${vertexCount}u && second < ${vertexCount}u && isFiniteValue(conductance) && conductance > 0.0) {
    if (parents[parentsOffset + first] != parents[parentsOffset + second]) {
      atomicStore(&control[controlOffset + 2u], 1u);
    }
  }
}`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: completionNodeId,
      operation: id,
      invocationCount: 1,
      workgroupSize: 2,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'edges', view: props.edges, type: 'u32', access: 'read'},
        {name: 'weights', view: props.conductances, type: 'f32', access: 'read'},
        {name: 'parents', view: components, type: 'u32', access: 'read_write'},
        {name: 'status', view: status, type: 'u32', access: 'read_write'},
        {name: 'control', view: control, type: 'u32', access: 'read_write'}
      ],
      body: `if ((control[controlOffset] | control[controlOffset + 1u] | control[controlOffset + 2u]) != 0u) {
control[controlOffset + 3u] = 1u;
${getSerialComponentsBody(props)}
} else {
status[statusOffset + 2u] = u32(status[statusOffset + 3u] == 0u && parents[parentsOffset + ${source}u] == parents[parentsOffset + ${sink}u]);
}`
    })
  );
  return {control, completionNodeId};
}

/** Preserve the single-dispatch path when bounded-union setup would dominate. @internal */
export function addSerialSparseResistanceComponents<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SparseResistanceComponentsProps
) {
  const completionNodeId = `${props.id}-components`;
  graph.add(
    createRecipeKernelNode(graph, {
      id: completionNodeId,
      operation: props.id,
      invocationCount: 1,
      workgroupSize: 2,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'edges', view: props.edges, type: 'u32', access: 'read'},
        {name: 'weights', view: props.conductances, type: 'f32', access: 'read'},
        {name: 'parents', view: props.components, type: 'u32', access: 'read_write'},
        {name: 'status', view: props.status, type: 'u32', access: 'read_write'}
      ],
      body: getSerialComponentsBody(props)
    })
  );
  return {completionNodeId};
}

function getSerialComponentsBody(props: SparseResistanceComponentsProps): string {
  const {vertexCount, source, sink, maximumIterations} = props;
  const edgeCount = props.edges.length;
  return `for(var vertex=0u;vertex<${vertexCount}u;vertex++){parents[parentsOffset+vertex]=vertex;}
var invalid=0u;
for(var edge=0u;edge<${edgeCount}u;edge++){
 var first=edges[edgesOffset+2u*edge];var second=edges[edgesOffset+2u*edge+1u];let conductance=weights[weightsOffset+edge];
 if(first>=${vertexCount}u||second>=${vertexCount}u||!isFiniteValue(conductance)||conductance<0.0){invalid++;continue;}
 if(conductance==0.0){continue;}
 loop{let parent=parents[parentsOffset+first];if(parent==first){break;}parents[parentsOffset+first]=parents[parentsOffset+parent];first=parent;}
 loop{let parent=parents[parentsOffset+second];if(parent==second){break;}parents[parentsOffset+second]=parents[parentsOffset+parent];second=parent;}
 parents[parentsOffset+max(first,second)]=min(first,second);
}
for(var vertex=0u;vertex<${vertexCount}u;vertex++){var root=vertex;loop{let parent=parents[parentsOffset+root];if(parent==root){break;}root=parent;}parents[parentsOffset+vertex]=root;}
status[statusOffset]=0u;status[statusOffset+1u]=${maximumIterations}u;status[statusOffset+2u]=u32(invalid==0u&&parents[parentsOffset+${source}u]==parents[parentsOffset+${sink}u]);status[statusOffset+3u]=invalid;`;
}
