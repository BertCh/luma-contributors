// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';

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
      id: `${props.id}-reconcile`,
      operation: props.id,
      invocationCount: 1,
      workgroupSize: 2,
      // One thread unions every cell with its neighbors: width times height visits. It is the
      // slow step above about a million cells; a parallel component labeling would replace it.
      bindings: [
        {name: 'patches', view: props.patches, type: 'u32', access: 'read'},
        {name: 'classes', view: props.patchClasses, type: 'u32', access: 'read'},
        {name: 'parents', view: parents, type: 'u32', access: 'read_write'}
      ],
      body: `for(var owner=0u;owner<${patchCount}u;owner++){parents[parentsOffset+owner]=owner;}
for(var cell=0u;cell<${count}u;cell++){let owner=patches[patchesOffset+cell];if(owner>=${patchCount}u){continue;}let category=classes[classesOffset+owner];if(category==${props.backgroundClass ?? 0xffffffff}u||category>=${props.classCount ?? 0xffffffff}u){continue;}let column=i32(cell%${props.width}u);let row=i32(cell/${props.width}u);let directions=array<vec2<i32>,4>(vec2<i32>(1,0),vec2<i32>(0,1),vec2<i32>(1,1),vec2<i32>(-1,1));
for(var direction=0u;direction<${props.connectivity === 8 ? 4 : 2}u;direction++){let adjacent=vec2<i32>(column,row)+directions[direction];if(any(adjacent < vec2<i32>(0))||adjacent.x>=${props.width}||adjacent.y>=${props.height}){continue;}let neighbor=patches[patchesOffset+u32(adjacent.y)*${props.width}u+u32(adjacent.x)];if(neighbor>=${patchCount}u||classes[classesOffset+neighbor]!=category){continue;}var first=owner;var second=neighbor;loop{let parent=parents[parentsOffset+first];if(first==parent){break;}parents[parentsOffset+first]=parents[parentsOffset+parent];first=parent;}loop{let parent=parents[parentsOffset+second];if(second==parent){break;}parents[parentsOffset+second]=parents[parentsOffset+parent];second=parent;}parents[parentsOffset+max(first,second)]=min(first,second);}}
for(var owner=0u;owner<${patchCount}u;owner++){var root=owner;loop{let parent=parents[parentsOffset+root];if(parent==root){break;}root=parent;}parents[parentsOffset+owner]=root;}`
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
