// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  URBAN_RAY_STACK_CAPACITY,
  getUrbanRayBVHBindings,
  getUrbanRayDeclarations,
  type UrbanRayBVH
} from './urban-ray-utils';

export type UrbanFreeSpaceRegion = {
  /** Closed directed segment rings; nonzero supports oriented shells and holes. */
  fillRule: 'even-odd' | 'nonzero';
  /** Choose the unfilled obstacle exterior or the filled enclosure interior. */
  freeSpace: 'inside' | 'outside';
};
/** Internal shared classification: 0 free, 1 blocked interior/exterior, 2 boundary, 3 nonfinite. */
export function classifyUrbanViewpoints<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    segments: GraphDataView<'float32x4'>;
    tolerance: number;
    region?: UrbanFreeSpaceRegion;
    /** Caller-owned classification view with one row per viewpoint. */
    output?: GraphDataView<'uint32'>;
    /**
     * Segment BVH over `segments` (see `addUrbanObstacleBVH`). When given, each viewpoint visits
     * only the segments whose box meets the horizontal ray to its right (the only segments that
     * change the winding number or lie within the tolerance), and non-finite segments are found
     * by one reduction instead of by every viewpoint. The classification is identical.
     */
    obstacleBVH?: UrbanRayBVH;
  }
) {
  const classifications = getRecipeView(
    graph,
    `${props.id}-viewpoint-classifications`,
    'uint32',
    props.positions.length,
    props.output
  );
  const regionTest = `let inside=${props.region?.fillRule === 'nonzero' ? 'winding!=0' : '(abs(winding)%2)==1'};
let allowed=${props.region ? (props.region.freeSpace === 'inside' ? 'inside' : '!inside') : 'true'};
classifications[classificationsOffset+index]=select(select(select(1u,0u,allowed),2u,boundary),3u,!valid);`;
  const windingStep = `let cross=crossVectors(segment.zw-segment.xy,point-segment.xy);
  if(segment.y<=point.y&&segment.w>point.y&&cross>0.0){winding++;}
  if(segment.y>point.y&&segment.w<=point.y&&cross<0.0){winding--;}`;
  const bvh = props.obstacleBVH;
  if (bvh) {
    // A segment that is not finite makes every viewpoint non-finite (3). The tree skips such
    // segments, so one reduction over the segments counts them.
    const segmentCount = props.segments.length;
    const nonFiniteFlags = createTransientView(
      graph,
      `${props.id}-non-finite-segment-flags`,
      'uint32',
      segmentCount
    );
    const nonFiniteCount = createTransientView(
      graph,
      `${props.id}-non-finite-segment-count`,
      'uint32',
      1
    );
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${props.id}-non-finite-segment-flags`,
        operation: props.id,
        invocationCount: segmentCount,
        declarations: IS_FINITE_VALUE_WGSL,
        bindings: [
          {name: 'segments', view: props.segments, type: 'f32', access: 'read'},
          {name: 'flags', view: nonFiniteFlags, type: 'u32', access: 'read_write'}
        ],
        body: `let base=segmentsOffset+4u*index;
flags[flagsOffset+index]=u32(!(isFiniteValue(segments[base])&&isFiniteValue(segments[base+1u])&&isFiniteValue(segments[base+2u])&&isFiniteValue(segments[base+3u])));`
      })
    );
    graph.add(
      new GPUReduction({
        id: `${props.id}-non-finite-segment-count`,
        input: nonFiniteFlags,
        output: nonFiniteCount,
        operation: 'sum'
      }).getCommandNodes(graph)
    );
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${props.id}-classify-viewpoints`,
        operation: props.id,
        invocationCount: props.positions.length,
        declarations:
          IS_FINITE_VALUE_WGSL + getUrbanRayDeclarations(segmentCount, props.tolerance, bvh),
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'segments', view: props.segments, type: 'f32', access: 'read'},
          ...getUrbanRayBVHBindings(bvh),
          {name: 'nonFiniteCount', view: nonFiniteCount, type: 'u32', access: 'read'},
          {name: 'classifications', view: classifications, type: 'u32', access: 'read_write'}
        ],
        // One thread per viewpoint visits the segments whose box meets the ray to its right.
        body: `let point=vec2<f32>(positions[positionsOffset+2u*index],positions[positionsOffset+2u*index+1u]);
let valid=isFiniteValue(point.x)&&isFiniteValue(point.y)&&nonFiniteCount[nonFiniteCountOffset]==0u;
var boundary=false;
var winding=0;
var stack:array<u32,${URBAN_RAY_STACK_CAPACITY}>;var depth=1u;stack[0u]=0u;
loop{
  if(depth==0u){break;}
  depth--;let node=stack[depth];
  if(getBoxEntry(node,point,vec2<f32>(1.0,0.0),1.0e30)>=3.0e38){continue;}
  if(node<${bvh.internalNodeCount}u){stack[depth]=node*2u+2u;stack[depth+1u]=node*2u+1u;depth+=2u;continue;}
  let row=leafIds[leafIdsOffset+node-${bvh.internalNodeCount}u];
  if(row>=${segmentCount}u){continue;}
  let segment=readSegment(row);
  boundary=boundary||distanceToSegment(point,segment)<=${props.tolerance};
  ${windingStep}
}
${regionTest}`
      })
    );
    return classifications;
  }
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${props.id}-classify-viewpoints`,
      operation: props.id,
      invocationCount: props.positions.length,
      declarations:
        IS_FINITE_VALUE_WGSL + getUrbanRayDeclarations(props.segments.length, props.tolerance),
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'segments', view: props.segments, type: 'f32', access: 'read'},
        {name: 'classifications', view: classifications, type: 'u32', access: 'read_write'}
      ],
      // One thread per viewpoint walks every obstacle segment (winding number): cost is viewpoints times segments.
      body: `let point=vec2<f32>(positions[positionsOffset+2u*index],positions[positionsOffset+2u*index+1u]);
var valid=isFiniteValue(point.x)&&isFiniteValue(point.y);
var boundary=false;
var winding=0;
for(var segmentIndex=0u;segmentIndex<${props.segments.length}u;segmentIndex++){
  let segment=readSegment(segmentIndex);
  valid=valid&&isFiniteValue(segment.x)&&isFiniteValue(segment.y)&&isFiniteValue(segment.z)&&isFiniteValue(segment.w);
  boundary=boundary||distanceToSegment(point,segment)<=${props.tolerance};
  ${windingStep}
}
${regionTest}`
    })
  );
  return classifications;
}
