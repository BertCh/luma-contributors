// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {classifyUrbanViewpoints, type UrbanFreeSpaceRegion} from './classify-viewpoints';
import {getUrbanRayDeclarations} from './urban-ray-utils';

export type GPUIsovistsProps = {
  id?: string;
  /** Free-space viewpoints. Points inside opaque obstacle interiors are not meaningful viewpoints. */
  positions: GraphDataView<'float32x2'>;
  segments: GraphDataView<'float32x4'>;
  rayCount: number;
  /** Add rays just before/at/after each obstacle endpoint, then sort by angle on GPU. */
  sampling?: 'uniform' | 'vertices';
  angularEpsilon?: number;
  maximumDistance: number;
  tolerance?: number;
  /** Optional explicit filled-ring free-space classification. Default checks boundaries only. */
  region?: UrbanFreeSpaceRegion;
};

/** Angularly sampled obstacle isovist polygons, counterclockwise from +X, with an explicit circular range boundary. */
export function addIsovistsRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUIsovistsProps
) {
  const id = props.id ?? 'isovists';
  const maximumDistance = props.maximumDistance;
  const vertexSampling = props.sampling === 'vertices';
  const angularEpsilon = props.angularEpsilon ?? 1e-5;
  const rayCount = props.rayCount + (vertexSampling ? 6 * props.segments.length : 0);
  assert(!vertexSampling || rayCount <= 4096);
  assert(Number.isFinite(angularEpsilon) && angularEpsilon > 0 && angularEpsilon < 0.01);
  const tolerance = props.tolerance ?? 1e-6;
  assert(
    props.positions.length > 0 && props.segments.length > 0 && props.segments.length <= 16777216
  );
  assert(Number.isSafeInteger(rayCount) && rayCount >= 3 && rayCount <= 65536);
  assert(
    Number.isFinite(maximumDistance) &&
      maximumDistance > 0 &&
      Number.isFinite(tolerance) &&
      tolerance > 0
  );
  const viewpointClassifications = classifyUrbanViewpoints(graph, {
    id,
    positions: props.positions,
    segments: props.segments,
    tolerance,
    region: props.region
  });
  const angles = getRecipeView(graph, `${id}-angles`, 'float32', props.positions.length * rayCount);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-angular-events`,
      operation: id,
      invocationCount: props.positions.length,
      workgroupSize: 2,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'segments', view: props.segments, type: 'f32', access: 'read'},
        {name: 'angles', view: angles, type: 'f32', access: 'read_write'}
      ],
      body: vertexSampling
        ? `
let origin=vec2<f32>(positions[positionsOffset+2u*index],positions[positionsOffset+2u*index+1u]);
for(var ray=0u;ray<${props.rayCount}u;ray++){angles[anglesOffset+index*${rayCount}u+ray]=6.283185307179586*f32(ray)/${props.rayCount}.0;}
${
  vertexSampling
    ? `for(var endpoint=0u;endpoint<${props.segments.length * 2}u;endpoint++){
 let point=vec2<f32>(segments[segmentsOffset+2u*endpoint],segments[segmentsOffset+2u*endpoint+1u])-origin;
 let center=atan2(point.y,point.x);
 for(var side=0u;side<3u;side++){var angle=center+(f32(side)-1.0)*${angularEpsilon};angle=angle-floor(angle/6.283185307179586)*6.283185307179586;angles[anglesOffset+index*${rayCount}u+${props.rayCount}u+3u*endpoint+side]=angle;}
}
for(var position=1u;position<${rayCount}u;position++){let value=angles[anglesOffset+index*${rayCount}u+position];var insertion=position;loop{if(insertion==0u||angles[anglesOffset+index*${rayCount}u+insertion-1u]<=value){break;}angles[anglesOffset+index*${rayCount}u+insertion]=angles[anglesOffset+index*${rayCount}u+insertion-1u];insertion--;}angles[anglesOffset+index*${rayCount}u+insertion]=value;}`
    : ''
}
`
        : `for(var ray=0u;ray<${rayCount}u;ray++){angles[anglesOffset+index*${rayCount}u+ray]=6.283185307179586*f32(ray)/${rayCount}.0;}`
    })
  );
  const endpoints = getRecipeView(
    graph,
    `${id}-endpoints`,
    'float32x2',
    props.positions.length * rayCount
  );
  const distances = getRecipeView(
    graph,
    `${id}-distances`,
    'float32',
    props.positions.length * rayCount
  );
  const segments = getRecipeView(
    graph,
    `${id}-segments`,
    'uint32',
    props.positions.length * rayCount
  );
  const validity = getRecipeView(
    graph,
    `${id}-validity`,
    'uint32',
    props.positions.length * rayCount
  );
  const measures = getRecipeView(graph, `${id}-measures`, 'float32x4', props.positions.length);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-cast`,
      operation: id,
      invocationCount: props.positions.length * rayCount,
      declarations:
        IS_FINITE_VALUE_WGSL + getUrbanRayDeclarations(props.segments.length, tolerance),
      bindings: [
        {name: 'classifications', view: viewpointClassifications, type: 'u32', access: 'read'},
        {name: 'angles', view: angles, type: 'f32', access: 'read'},
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'segments', view: props.segments, type: 'f32', access: 'read'},
        {name: 'endpoints', view: endpoints, type: 'f32', access: 'read_write'},
        {name: 'distances', view: distances, type: 'f32', access: 'read_write'},
        {name: 'hits', view: segments, type: 'u32', access: 'read_write'},
        {name: 'validity', view: validity, type: 'u32', access: 'read_write'}
      ],
      body: `let viewpoint=index/${rayCount}u;let angle=angles[anglesOffset+index];let direction=vec2<f32>(cos(angle),sin(angle));
let origin=vec2<f32>(positions[positionsOffset+2u*viewpoint],positions[positionsOffset+2u*viewpoint+1u]);
var valid=classifications[classificationsOffset+viewpoint]==0u&&isFiniteValue(origin.x)&&isFiniteValue(origin.y);
for(var segment=0u;segment<${props.segments.length}u;segment++){let values=readSegment(segment);valid=valid&&isFiniteValue(values.x)&&isFiniteValue(values.y)&&isFiniteValue(values.z)&&isFiniteValue(values.w);}
let hit=castRay(origin,direction,${maximumDistance});valid=valid&&hit.x>${tolerance};
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));let endpoint=origin+direction*hit.x;
endpoints[endpointsOffset+2u*index]=select(missing,endpoint.x,valid);endpoints[endpointsOffset+2u*index+1u]=select(missing,endpoint.y,valid);distances[distancesOffset+index]=select(missing,hit.x,valid);hits[hitsOffset+index]=select(0xffffffffu,u32(max(hit.y,0.0)),hit.y>=0.0&&valid);validity[validityOffset+index]=u32(valid);`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-measures`,
      operation: id,
      invocationCount: props.positions.length,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'endpoints', view: endpoints, type: 'f32', access: 'read'},
        {name: 'validity', view: validity, type: 'u32', access: 'read'},
        {name: 'output', view: measures, type: 'f32', access: 'read_write'}
      ],
      body: `let origin=vec2<f32>(positions[positionsOffset+2u*index],positions[positionsOffset+2u*index+1u]);var area=0.0;var perimeter=0.0;var maximum=0.0;var valid=true;
for(var ray=0u;ray<${rayCount}u;ray++){let first=index*${rayCount}u+ray;let second=index*${rayCount}u+(ray+1u)%${rayCount}u;valid=valid&&validity[validityOffset+first]!=0u;
let start=vec2<f32>(endpoints[endpointsOffset+2u*first],endpoints[endpointsOffset+2u*first+1u])-origin;let end=vec2<f32>(endpoints[endpointsOffset+2u*second],endpoints[endpointsOffset+2u*second+1u])-origin;
area+=0.5*(start.x*end.y-start.y*end.x);perimeter+=distance(start,end);maximum=max(maximum,length(start));}
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));output[outputOffset+4u*index]=select(missing,area,valid);output[outputOffset+4u*index+1u]=select(missing,perimeter,valid);output[outputOffset+4u*index+2u]=select(missing,maximum,valid);output[outputOffset+4u*index+3u]=f32(valid);`
    })
  );
  return {
    outputs: {endpoints, distances, segments, validity, measures, angles, viewpointClassifications},
    provenance: {
      positions: props.positions,
      obstacles: props.segments,
      rayCount,
      maximumDistance,
      angularSpacing: (2 * Math.PI) / props.rayCount,
      sampling: props.sampling ?? 'uniform',
      angularEpsilon,
      tolerance
    }
  };
}
