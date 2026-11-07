// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getUrbanRayDeclarations} from './urban-ray-utils';

export type GPUStreetProfilesProps = {
  id?: string;
  /** Caller samples street centerlines with explicit spacing; normals point to the left side. */
  positions: GraphDataView<'float32x2'>;
  normals: GraphDataView<'float32x2'>;
  segments: GraphDataView<'float32x4'>;
  segmentBuildingIds: GraphDataView<'uint32'>;
  buildingHeights: GraphDataView<'float32'>;
  maximumDistance: number;
  sampleSpacing: number;
  tolerance?: number;
  /** Optional straight street start/end segment for exact visible facade arc length in its rectangular corridor. */
  streetSegment?: GraphDataView<'float32x4'>;
};

/** Side-aware first-building profiles; missing sides remain range-censored and never become observed walls. */
export function addStreetProfilesRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUStreetProfilesProps
) {
  const id = props.id ?? 'street-profiles';
  const count = props.positions.length;
  const segmentCount = props.segments.length;
  const tolerance = props.tolerance ?? 1e-6;
  assert(
    count > 0 && props.normals.length === count && segmentCount > 0 && segmentCount <= 16777216
  );
  assert(props.segmentBuildingIds.length === segmentCount && props.buildingHeights.length > 0);
  assert(Number.isFinite(props.maximumDistance) && props.maximumDistance > 0);
  assert(
    Number.isFinite(props.sampleSpacing) &&
      props.sampleSpacing > 0 &&
      Number.isFinite(tolerance) &&
      tolerance > 0
  );
  const profiles = getRecipeView(graph, `${id}-profiles`, 'float32x4', count);
  const buildings = getRecipeView(graph, `${id}-buildings`, 'uint32x2', count);
  const validity = getRecipeView(graph, `${id}-validity`, 'uint32', count);
  const canyon = getRecipeView(graph, `${id}-canyon`, 'float32x4', count);
  const frontage = getRecipeView(
    graph,
    `${id}-frontage`,
    'float32x2',
    props.buildingHeights.length
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-rays`,
      operation: id,
      invocationCount: count,
      declarations: IS_FINITE_VALUE_WGSL + getUrbanRayDeclarations(segmentCount, tolerance),
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'normals', view: props.normals, type: 'f32', access: 'read'},
        {name: 'segments', view: props.segments, type: 'f32', access: 'read'},
        {name: 'owners', view: props.segmentBuildingIds, type: 'u32', access: 'read'},
        {name: 'heights', view: props.buildingHeights, type: 'f32', access: 'read'},
        {name: 'profiles', view: profiles, type: 'f32', access: 'read_write'},
        {name: 'buildings', view: buildings, type: 'u32', access: 'read_write'},
        {name: 'validity', view: validity, type: 'u32', access: 'read_write'}
      ],
      body: `let origin=vec2<f32>(positions[positionsOffset+2u*index],positions[positionsOffset+2u*index+1u]);let normal=vec2<f32>(normals[normalsOffset+2u*index],normals[normalsOffset+2u*index+1u]);
var valid=isFiniteValue(origin.x)&&isFiniteValue(origin.y)&&isFiniteValue(normal.x)&&isFiniteValue(normal.y)&&isFiniteValue(length(normal))&&length(normal)>0.0;
for(var segment=0u;segment<${segmentCount}u;segment++){let values=readSegment(segment);valid=valid&&isFiniteValue(values.x)&&isFiniteValue(values.y)&&isFiniteValue(values.z)&&isFiniteValue(values.w)&&owners[ownersOffset+segment]<${props.buildingHeights.length}u;}
var flags=0u;let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
for(var side=0u;side<2u;side++){
 var distance=missing;var height=missing;var owner=0xffffffffu;
 if(valid){let hit=castRay(origin,normalize(normal)*select(1.0,-1.0,side==1u),${props.maximumDistance});distance=hit.x;
 if(hit.y>=0.0){owner=owners[ownersOffset+u32(hit.y)];height=heights[heightsOffset+owner];if(!isFiniteValue(height)||height<0.0){height=missing;}flags|=1u<<side;}
 if(hit.x<=${tolerance}){flags|=4u;}
 }
 profiles[profilesOffset+4u*index+side]=distance;profiles[profilesOffset+4u*index+side+2u]=height;buildings[buildingsOffset+2u*index+side]=owner;
}
if(!valid){flags|=8u;}validity[validityOffset+index]=flags;`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-canyon`,
      operation: id,
      invocationCount: count,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'profiles', view: profiles, type: 'f32', access: 'read'},
        {name: 'validity', view: validity, type: 'u32', access: 'read'},
        {name: 'output', view: canyon, type: 'f32', access: 'read_write'}
      ],
      body: `let base=profilesOffset+4u*index;let flags=validity[validityOffset+index];let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
let complete=flags==3u;let width=profiles[base]+profiles[base+1u];let meanHeight=0.5*(profiles[base+2u]+profiles[base+3u]);
output[outputOffset+4u*index]=select(missing,width,complete);output[outputOffset+4u*index+1u]=select(missing,meanHeight,complete);
output[outputOffset+4u*index+2u]=select(missing,meanHeight/width,complete&&width>0.0&&isFiniteValue(meanHeight));output[outputOffset+4u*index+3u]=select(missing,width,(flags&12u)==0u);`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-frontage`,
      operation: id,
      invocationCount: props.buildingHeights.length,
      bindings: [
        {name: 'buildings', view: buildings, type: 'u32', access: 'read'},
        {name: 'validity', view: validity, type: 'u32', access: 'read'},
        {name: 'output', view: frontage, type: 'f32', access: 'read_write'}
      ],
      body: `for(var side=0u;side<2u;side++){var length=0.0;for(var sample=0u;sample<${count}u;sample++){let flags=validity[validityOffset+sample];if((flags&12u)==0u&&(flags&(1u<<side))!=0u&&buildings[buildingsOffset+2u*sample+side]==index){length+=${props.sampleSpacing};}}output[outputOffset+2u*index+side]=length;}`
    })
  );
  let facades:
    | {lengths: GraphDataView<'float32x2'>; frontage: GraphDataView<'float32x2'>}
    | undefined;
  if (props.streetSegment) {
    assert(props.streetSegment.length === 1 && segmentCount <= 1024);
    const facadeEvents = getRecipeView(
      graph,
      `${id}-facade-events`,
      'float32',
      segmentCount * (3 * segmentCount + 8)
    );
    const lengths = getRecipeView(graph, `${id}-facade-lengths`, 'float32x2', segmentCount);
    const facadeFrontage = getRecipeView(
      graph,
      `${id}-facade-frontage`,
      'float32x2',
      props.buildingHeights.length
    );
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-facade-intervals`,
        operation: id,
        invocationCount: segmentCount,
        workgroupSize: 2,
        declarations: IS_FINITE_VALUE_WGSL + getUrbanRayDeclarations(segmentCount, tolerance),
        bindings: [
          {name: 'segments', view: props.segments, type: 'f32', access: 'read'},
          {name: 'street', view: props.streetSegment, type: 'f32', access: 'read'},
          {name: 'output', view: lengths, type: 'f32', access: 'read_write'},
          {name: 'facadeEvents', view: facadeEvents, type: 'f32', access: 'read_write'}
        ],
        body: `let streetStart=vec2<f32>(street[streetOffset],street[streetOffset+1u]);let streetEnd=vec2<f32>(street[streetOffset+2u],street[streetOffset+3u]);let streetLength=distance(streetStart,streetEnd);let axis=(streetEnd-streetStart)/streetLength;let normal=vec2<f32>(-axis.y,axis.x);let segment=readSegment(index);let start=segment.xy;let end=segment.zw;let displacement=end-start;let projectedStart=dot(start-streetStart,axis);let projectedDelta=dot(displacement,axis);let sideStart=dot(start-streetStart,normal);let sideDelta=dot(displacement,normal);
facadeEvents[facadeEventsOffset+index*${3 * segmentCount + 8}u+0u]=0.0;facadeEvents[facadeEventsOffset+index*${3 * segmentCount + 8}u+1u]=1.0;var size=2u;
if(abs(projectedDelta)>${tolerance}){facadeEvents[facadeEventsOffset+index*${3 * segmentCount + 8}u+size]=clamp(-projectedStart/projectedDelta,0.0,1.0);size++;facadeEvents[facadeEventsOffset+index*${3 * segmentCount + 8}u+size]=clamp((streetLength-projectedStart)/projectedDelta,0.0,1.0);size++;}
if(abs(sideDelta)>${tolerance}){for(var boundary=0u;boundary<3u;boundary++){facadeEvents[facadeEventsOffset+index*${3 * segmentCount + 8}u+size]=clamp((f32(i32(boundary)-1)*${props.maximumDistance}-sideStart)/sideDelta,0.0,1.0);size++;}}
for(var other=0u;other<${segmentCount}u;other++){let obstacle=readSegment(other);if(abs(projectedDelta)>${tolerance}){facadeEvents[facadeEventsOffset+index*${3 * segmentCount + 8}u+size]=clamp((dot(obstacle.xy-streetStart,axis)-projectedStart)/projectedDelta,0.0,1.0);size++;facadeEvents[facadeEventsOffset+index*${3 * segmentCount + 8}u+size]=clamp((dot(obstacle.zw-streetStart,axis)-projectedStart)/projectedDelta,0.0,1.0);size++;}
let otherDelta=obstacle.zw-obstacle.xy;let denominator=displacement.x*otherDelta.y-displacement.y*otherDelta.x;if(abs(denominator)>${tolerance}){let difference=obstacle.xy-start;let crossing=(difference.x*otherDelta.y-difference.y*otherDelta.x)/denominator;facadeEvents[facadeEventsOffset+index*${3 * segmentCount + 8}u+size]=clamp(crossing,0.0,1.0);size++;}}
for(var event=1u;event<size;event++){let value=facadeEvents[facadeEventsOffset+index*${3 * segmentCount + 8}u+event];var insertion=event;loop{if(insertion==0u||facadeEvents[facadeEventsOffset+index*${3 * segmentCount + 8}u+insertion-1u]<=value){break;}facadeEvents[facadeEventsOffset+index*${3 * segmentCount + 8}u+insertion]=facadeEvents[facadeEventsOffset+index*${3 * segmentCount + 8}u+insertion-1u];insertion--;}facadeEvents[facadeEventsOffset+index*${3 * segmentCount + 8}u+insertion]=value;}
var total=vec2<f32>(0.0);for(var event=1u;event<size;event++){let middle=0.5*(facadeEvents[facadeEventsOffset+index*${3 * segmentCount + 8}u+event]+facadeEvents[facadeEventsOffset+index*${3 * segmentCount + 8}u+event-1u]);let point=start+middle*displacement;let longitudinal=projectedStart+middle*projectedDelta;let lateral=sideStart+middle*sideDelta;if(longitudinal<0.0||longitudinal>streetLength||abs(lateral)<=${tolerance}||abs(lateral)>${props.maximumDistance}){continue;}let origin=streetStart+longitudinal*axis;let hit=castRay(origin,normal*sign(lateral),${props.maximumDistance});if(abs(hit.x-abs(lateral))<=${tolerance}&&u32(max(hit.y,0.0))==index){let side=u32(lateral<0.0);total[side]+=(facadeEvents[facadeEventsOffset+index*${3 * segmentCount + 8}u+event]-facadeEvents[facadeEventsOffset+index*${3 * segmentCount + 8}u+event-1u])*length(displacement);}}
var valid=isFiniteValue(streetLength)&&streetLength>${tolerance};for(var obstacleIndex=0u;obstacleIndex<${segmentCount}u;obstacleIndex++){let obstacle=readSegment(obstacleIndex);valid=valid&&isFiniteValue(obstacle.x)&&isFiniteValue(obstacle.y)&&isFiniteValue(obstacle.z)&&isFiniteValue(obstacle.w);}let missing=bitcast<f32>(0x7fc00000u | (index & 0u));output[outputOffset+2u*index]=select(missing,total.x,valid);output[outputOffset+2u*index+1u]=select(missing,total.y,valid);`
      })
    );
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-facade-summary`,
        operation: id,
        invocationCount: props.buildingHeights.length,
        bindings: [
          {name: 'owners', view: props.segmentBuildingIds, type: 'u32', access: 'read'},
          {name: 'lengths', view: lengths, type: 'f32', access: 'read'},
          {name: 'output', view: facadeFrontage, type: 'f32', access: 'read_write'}
        ],
        body: `var total=vec2<f32>(0.0);for(var segment=0u;segment<${segmentCount}u;segment++){if(owners[ownersOffset+segment]==index){total+=vec2<f32>(lengths[lengthsOffset+2u*segment],lengths[lengthsOffset+2u*segment+1u]);}}output[outputOffset+2u*index]=total.x;output[outputOffset+2u*index+1u]=total.y;`
      })
    );
    facades = {lengths, frontage: facadeFrontage};
  }
  return {
    outputs: {profiles, buildings, validity, canyon, frontage},
    facades,
    provenance: {
      positions: props.positions,
      normals: props.normals,
      segments: props.segments,
      segmentBuildingIds: props.segmentBuildingIds,
      maximumDistance: props.maximumDistance,
      sampleSpacing: props.sampleSpacing,
      tolerance
    }
  };
}
