// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {validateGPUStorageBindingByteLength} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Reconcile globally unique tile-local IDs across class-compatible seams on the complete resident lattice. @internal */
export function reconcileLandscapePatchLabels<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    patches: GraphDataView<'uint32'>;
    patchClasses: GraphDataView<'uint32'>;
    width: number;
    height: number;
    connectivity: 4 | 8;
    backgroundClass?: number;
    classCount?: number;
    /** Caller-owned result views: the merged label of every cell, and the root of every patch. */
    output?: {patches?: GraphDataView<'uint32'>; parents?: GraphDataView<'uint32'>};
  }
) {
  const count = props.width * props.height;
  const patchCount = props.patchClasses.length;
  validateGPUStorageBindingByteLength(
    graph.device,
    props.id,
    'patch parents',
    patchCount * 4,
    'reduce the patch count'
  );
  validateGPUStorageBindingByteLength(
    graph.device,
    props.id,
    'reconciled patches',
    count * 4,
    'reduce the raster size'
  );
  const parents = getRecipeView(
    graph,
    `${props.id}-patch-parents`,
    'uint32',
    patchCount,
    props.output?.parents
  );
  const output = getRecipeView(
    graph,
    `${props.id}-reconciled-patches`,
    'uint32',
    count,
    props.output?.patches
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${props.id}-patch-parents-init`,
      operation: props.id,
      invocationCount: patchCount,
      bindings: [{name: 'parents', view: parents, type: 'u32', access: 'read_write'}],
      body: 'parents[parentsOffset+index]=index;'
    })
  );
  // Roots only move toward lower patch IDs. Atomic root linking cannot create a cycle; a failed
  // compare/exchange retries against the new roots, so no seam edge is dropped under contention.
  const declarations = `
fn getRoot(owner:u32)->u32 {
  var root=owner;
  loop {
    let parent=atomicLoad(&parents[parentsOffset+root]);
    if(parent==root){return root;}
    let grandparent=atomicLoad(&parents[parentsOffset+parent]);
    atomicMin(&parents[parentsOffset+root],grandparent);
    root=parent;
  }
}`;
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${props.id}-reconcile`,
      operation: props.id,
      invocationCount: count,
      declarations,
      bindings: [
        {name: 'patches', view: props.patches, type: 'u32', access: 'read'},
        {name: 'classes', view: props.patchClasses, type: 'u32', access: 'read'},
        {name: 'parents', view: parents, type: 'atomic<u32>', access: 'read_write'}
      ],
      body: `let owner=patches[patchesOffset+index];
if(owner>=${patchCount}u){return;}
let category=classes[classesOffset+owner];
if(category==${props.backgroundClass ?? 0xffffffff}u||category>=${props.classCount ?? 0xffffffff}u){return;}
let column=i32(index%${props.width}u);let row=i32(index/${props.width}u);
let directions=array<vec2<i32>,4>(vec2<i32>(1,0),vec2<i32>(0,1),vec2<i32>(1,1),vec2<i32>(-1,1));
for(var direction=0u;direction<${props.connectivity === 8 ? 4 : 2}u;direction++){
  let adjacent=vec2<i32>(column,row)+directions[direction];
  if(any(adjacent < vec2<i32>(0))||adjacent.x>=${props.width}||adjacent.y>=${props.height}){continue;}
  let neighbor=patches[patchesOffset+u32(adjacent.y)*${props.width}u+u32(adjacent.x)];
  if(neighbor>=${patchCount}u||classes[classesOffset+neighbor]!=category){continue;}
  loop {
    let first=getRoot(owner);let second=getRoot(neighbor);
    if(first==second){break;}
    let higher=max(first,second);let lower=min(first,second);
    let linked=atomicCompareExchangeWeak(&parents[parentsOffset+higher],higher,lower);
    if(linked.exchanged){break;}
  }
}`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${props.id}-patch-roots`,
      operation: props.id,
      invocationCount: patchCount,
      declarations,
      bindings: [{name: 'parents', view: parents, type: 'atomic<u32>', access: 'read_write'}],
      body: 'let root=getRoot(index);atomicStore(&parents[parentsOffset+index],root);'
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${props.id}-relabel`,
      operation: props.id,
      invocationCount: count,
      bindings: [
        {name: 'patches', view: props.patches, type: 'u32', access: 'read'},
        {name: 'parents', view: parents, type: 'u32', access: 'read'},
        {name: 'output', view: output, type: 'u32', access: 'read_write'}
      ],
      body: `let owner=patches[patchesOffset+index];var label=0xffffffffu;if(owner<${patchCount}u){label=parents[parentsOffset+owner];}output[outputOffset+index]=label;`
    })
  );
  return {patches: output, parents};
}
