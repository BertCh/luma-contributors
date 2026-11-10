// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, type GPUBVH, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getSegmentBVHNodes} from '../segment-intersection/segment-bvh';

/** Obstacle segment ids travel through float32 hit indices, which are exact only up to 2^24. */
export const URBAN_MAXIMUM_OBSTACLE_SEGMENT_COUNT = 16777216;

/**
 * Largest obstacle segment count for which the default `'auto'` obstacle index tests every
 * segment for every ray. Above it, `'auto'` builds a segment BVH, because brute force costs
 * rays times segments and a few thousand segments already dominate the frame.
 */
export const URBAN_BRUTE_FORCE_MAXIMUM_SEGMENT_COUNT = 4096;

/** Smallest BVH leaf capacity, so that the tree always has one internal node and a valid root. */
const URBAN_MINIMUM_BVH_LEAF_CAPACITY = 2;

/** Depth bound of the ray traversal stack: a 2^24 leaf tree has 25 levels, two pushes per level. */
const URBAN_RAY_STACK_CAPACITY = 64;

/** How rays find obstacle segments: every segment ('brute-force'), a segment BVH, or by count. */
export type UrbanObstacleIndex = 'auto' | 'brute-force' | 'bvh';

/** Segment BVH views a ray kernel binds when the obstacle index is a BVH. */
export type UrbanRayBVH = Pick<
  GPUBVH,
  'nodeMinima' | 'nodeMaxima' | 'leafIds' | 'internalNodeCount'
>;

/** Resolves `'auto'` to the index used for `segmentCount` obstacle segments. */
export function resolveUrbanObstacleIndex(
  index: UrbanObstacleIndex | undefined,
  segmentCount: number
): 'brute-force' | 'bvh' {
  if (index === 'brute-force' || index === 'bvh') return index;
  return segmentCount > URBAN_BRUTE_FORCE_MAXIMUM_SEGMENT_COUNT ? 'bvh' : 'brute-force';
}

/**
 * Adds the nodes that build a Morton-sorted segment BVH over the obstacle segments. Bounds are
 * padded by twice the ray tolerance (the brute-force collinear branch accepts a segment up to
 * two tolerances off the ray: one for the parallel test, one for the offset) plus float32 rounding of the coordinates, so the tree never prunes
 * a segment the brute-force test would report. Segments with non-finite coordinates get inverted
 * bounds and are absent from every traversal.
 */
export function addUrbanObstacleBVH<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    segments: GraphDataView<'float32x4'>;
    tolerance: number;
  }
): UrbanRayBVH {
  const segmentCount = props.segments.length;
  const minima = createTransientView(graph, `${props.id}-bvh-minima`, 'float32x2', segmentCount);
  const maxima = createTransientView(graph, `${props.id}-bvh-maxima`, 'float32x2', segmentCount);
  graph.add(
    createWGSLKernelNode(graph, {
      id: `${props.id}-bvh-bounds`,
      operation: props.operation,
      invocationCount: segmentCount,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'segments', view: props.segments, type: 'f32', access: 'read'},
        {name: 'minima', view: minima, type: 'f32', access: 'read_write'},
        {name: 'maxima', view: maxima, type: 'f32', access: 'read_write'}
      ],
      body: `let base=segmentsOffset+4u*index;let start=vec2<f32>(segments[base],segments[base+1u]);let end=vec2<f32>(segments[base+2u],segments[base+3u]);
let finiteSegment=isFiniteValue(start.x)&&isFiniteValue(start.y)&&isFiniteValue(end.x)&&isFiniteValue(end.y);
let largest=max(max(abs(start.x),abs(start.y)),max(abs(end.x),abs(end.y)));
let padding=${getWGSLFloatLiteral(2 * props.tolerance)}+1.0e-6*largest;
let lower=min(start,end)-vec2<f32>(padding);let upper=max(start,end)+vec2<f32>(padding);
minima[minimaOffset+2u*index]=select(1.0,lower.x,finiteSegment);minima[minimaOffset+2u*index+1u]=select(1.0,lower.y,finiteSegment);
maxima[maximaOffset+2u*index]=select(0.0,upper.x,finiteSegment);maxima[maximaOffset+2u*index+1u]=select(0.0,upper.y,finiteSegment);`
    })
  );
  let leafCapacity = URBAN_MINIMUM_BVH_LEAF_CAPACITY;
  while (leafCapacity < segmentCount) leafCapacity *= 2;
  const {bvh, nodes} = getSegmentBVHNodes(graph, {
    id: props.id,
    operation: props.operation,
    minima,
    maxima,
    leafCapacity,
    spatialSort: true
  });
  graph.add(nodes);
  return bvh;
}

/** Storage bindings of the segment BVH, in the names the BVH ray declarations read. */
export function getUrbanRayBVHBindings(bvh: UrbanRayBVH): WGSLKernelBinding[] {
  return [
    {name: 'nodeMinima', view: bvh.nodeMinima, type: 'f32', access: 'read'},
    {name: 'nodeMaxima', view: bvh.nodeMaxima, type: 'f32', access: 'read'},
    {name: 'leafIds', view: bvh.leafIds, type: 'u32', access: 'read'}
  ];
}

/**
 * Shared bounded planar ray predicates. Collinear overlap blocks at its nearest forward point.
 * `castRay` returns `(distance, segment row)`; the nearest segment wins and the lowest row breaks
 * ties. With `bvh`, `castRay` traverses the segment BVH and returns exactly the same result as the
 * loop over every segment; the kernel must also bind {@link getUrbanRayBVHBindings}.
 */
export function getUrbanRayDeclarations(
  segmentCount: number,
  tolerance: number,
  bvh?: UrbanRayBVH
): string {
  const toleranceLiteral = getWGSLFloatLiteral(tolerance);
  const shared = `
fn readSegment(row:u32)->vec4<f32>{let base=segmentsOffset+row*4u;return vec4<f32>(segments[base],segments[base+1u],segments[base+2u],segments[base+3u]);}
fn crossVectors(first:vec2<f32>,second:vec2<f32>)->f32{return first.x*second.y-first.y*second.x;}
fn distanceToSegment(position:vec2<f32>,segment:vec4<f32>)->f32{
 let direction=segment.zw-segment.xy;let squared=dot(direction,direction);var fraction=0.0;
 if(squared>0.0){fraction=clamp(dot(position-segment.xy,direction)/squared,0.0,1.0);}
 return distance(position,segment.xy+fraction*direction);
}
// Returns (candidate distance along the ray, 1.0 when the segment blocks the ray, else 0.0).
fn getSegmentCandidate(origin:vec2<f32>,direction:vec2<f32>,maximum:f32,segmentIndex:u32)->vec2<f32>{
 let segment=readSegment(segmentIndex);let edge=segment.zw-segment.xy;let displacement=segment.xy-origin;let denominator=crossVectors(direction,edge);var candidate=maximum;var intersects=false;
 if(distanceToSegment(origin,segment)<=${toleranceLiteral}){candidate=0.0;intersects=true;}
 else if(abs(denominator)>${toleranceLiteral}){
  let rayDistance=crossVectors(displacement,edge)/denominator;let fraction=crossVectors(displacement,direction)/denominator;
  if(rayDistance>=0.0&&fraction>=0.0&&fraction<=1.0){candidate=rayDistance;intersects=true;}
 }else if(abs(crossVectors(displacement,direction))<=${toleranceLiteral}){
  let first=dot(displacement,direction);let second=dot(segment.zw-origin,direction);
  if(max(first,second)>=0.0){candidate=max(0.0,min(first,second));intersects=true;}
 }
 return vec2<f32>(candidate,f32(intersects));
}`;
  if (!bvh) {
    return `${shared}
fn castRay(origin:vec2<f32>,direction:vec2<f32>,maximum:f32)->vec2<f32>{
 var nearest=maximum;var hit=-1.0;
 for(var segmentIndex=0u;segmentIndex<${segmentCount}u;segmentIndex++){
  let blocked=getSegmentCandidate(origin,direction,maximum,segmentIndex);
  if(blocked.y>0.0&&blocked.x<=nearest&&(blocked.x<nearest||hit<0.0)){nearest=blocked.x;hit=f32(segmentIndex);}
 }
 return vec2<f32>(nearest,hit);
}`;
  }
  return `${shared}
// Entry distance of the ray into a node box, or a huge value when the ray misses it or the box is empty.
fn getBoxEntry(node:u32,origin:vec2<f32>,direction:vec2<f32>,limit:f32)->f32{
 let component=node*2u;
 let lower=vec2<f32>(nodeMinima[nodeMinimaOffset+component],nodeMinima[nodeMinimaOffset+component+1u]);
 let upper=vec2<f32>(nodeMaxima[nodeMaximaOffset+component],nodeMaxima[nodeMaximaOffset+component+1u]);
 if(!(lower.x<=upper.x&&lower.y<=upper.y)){return 3.0e38;}
 var entry=0.0;var exit=limit;
 for(var axis=0u;axis<2u;axis++){
  if(abs(direction[axis])<1.0e-20){if(origin[axis]<lower[axis]||origin[axis]>upper[axis]){return 3.0e38;}}
  else{let first=(lower[axis]-origin[axis])/direction[axis];let second=(upper[axis]-origin[axis])/direction[axis];entry=max(entry,min(first,second));exit=min(exit,max(first,second));}
 }
 return select(3.0e38,entry,entry<=exit);
}
fn castRay(origin:vec2<f32>,direction:vec2<f32>,maximum:f32)->vec2<f32>{
 var nearest=maximum;var hit=-1.0;var stack:array<u32,${URBAN_RAY_STACK_CAPACITY}>;var depth=1u;stack[0u]=0u;
 loop{
  if(depth==0u){break;}
  depth--;let node=stack[depth];
  // Rounding slack on the pruning distance keeps tied and near-tied segments in the search.
  let limit=nearest+nearest*1.0e-5+${getWGSLFloatLiteral(4 * tolerance)};
  if(getBoxEntry(node,origin,direction,limit)>=3.0e38){continue;}
  if(node<${bvh.internalNodeCount}u){stack[depth]=node*2u+2u;stack[depth+1u]=node*2u+1u;depth+=2u;continue;}
  let row=leafIds[leafIdsOffset+node-${bvh.internalNodeCount}u];
  if(row>=${segmentCount}u){continue;}
  let blocked=getSegmentCandidate(origin,direction,maximum,row);
  if(blocked.y>0.0&&blocked.x<=nearest&&(blocked.x<nearest||hit<0.0||f32(row)<hit)){nearest=blocked.x;hit=f32(row);}
 }
 return vec2<f32>(nearest,hit);
}`;
}
