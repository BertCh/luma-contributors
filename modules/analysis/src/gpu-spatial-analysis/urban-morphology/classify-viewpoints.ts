// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getUrbanRayDeclarations} from './urban-ray-utils';

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
  }
) {
  const classifications = getRecipeView(
    graph,
    `${props.id}-viewpoint-classifications`,
    'uint32',
    props.positions.length,
    props.output
  );
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
  let cross=crossVectors(segment.zw-segment.xy,point-segment.xy);
  if(segment.y<=point.y&&segment.w>point.y&&cross>0.0){winding++;}
  if(segment.y>point.y&&segment.w<=point.y&&cross<0.0){winding--;}
}
let inside=${props.region?.fillRule === 'nonzero' ? 'winding!=0' : '(abs(winding)%2)==1'};
let allowed=${props.region ? (props.region.freeSpace === 'inside' ? 'inside' : '!inside') : 'true'};
classifications[classificationsOffset+index]=select(select(select(1u,0u,allowed),2u,boundary),3u,!valid);`
    })
  );
  return classifications;
}
