// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getSpatialAnalysisPackedKernelNodes} from '../geographic-distribution/spatial-analysis-packed-kernel';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  getBuildingTessellationIndexDeclarations,
  getBuildingTessellationGridWalk,
  type BuildingTessellationIndex
} from './building-tessellation-index';

/** Builds one building-parallel clipping dispatch. All sizes and bindings are validated by the recipe. @internal */
export function getBuildingTessellationCellKernel<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    buildingCount: number;
    maximumSamplesPerBuilding: number;
    candidateCapacity: number;
    cellVertexCapacity: number;
    maximumGridVisits: number;
    parameters: GraphDataView<'float32'>;
    enclosure: GraphDataView<'float32x2'>;
    pointIndex: BuildingTessellationIndex;
    simpleEnclosure: boolean;
    enclosurePieceCount: number;
    sampleData: GraphDataView<'uint32'>;
    buildingIndex: GraphDataView<'uint32'>;
    positions: GraphDataView<'float32x2'>;
    cells: GraphDataView<'uint32x4'>;
    buildingStatus: GraphDataView<'uint32x4'>;
  }
) {
  const {buildingCount, maximumSamplesPerBuilding, candidateCapacity, cellVertexCapacity} = props;
  const enclosureCount = props.simpleEnclosure ? 3 : props.enclosure.length;
  const piecesPerBuilding = maximumSamplesPerBuilding * props.enclosurePieceCount;
  // A convex polygon gains at most one vertex per clipping half-plane.
  const clippingCapacity = enclosureCount + candidateCapacity;
  const sampleCapacity = props.sampleData.length / 3;
  return getSpatialAnalysisPackedKernelNodes(graph, {
    id: props.id,
    operation: 'GPUBuildingTessellation',
    variant: 'local-sample-cell-union',
    invocationCount: buildingCount,
    workgroupSize: 4,
    bindings: [
      {name: 'samples', view: props.sampleData, type: 'u32', access: 'read'},
      {name: 'buildingIndex', view: props.buildingIndex, type: 'u32', access: 'read'},
      {name: 'pointIndex', view: props.pointIndex.data, type: 'u32', access: 'read'},
      {name: 'enclosure', view: props.enclosure, type: 'f32', access: 'read'},
      {name: 'positions', view: props.positions, type: 'f32', access: 'read_write'},
      {name: 'cells', view: props.cells, type: 'u32', access: 'read_write'},
      {name: 'buildingStatus', view: props.buildingStatus, type: 'u32', access: 'read_write'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'}
    ],
    declarations:
      IS_FINITE_VALUE_WGSL +
      getBuildingTessellationIndexDeclarations(props.pointIndex) +
      `
var<private> candidateRows:array<u32,${candidateCapacity}>;
fn readSample(row:u32)->vec2<f32>{return vec2<f32>(bitcast<f32>(samples[samplesOffset+2u*row]),bitcast<f32>(samples[samplesOffset+2u*row+1u]));}
fn readOwner(row:u32)->u32{return samples[samplesOffset+${2 * sampleCapacity}u+row];}
fn readOwnSample(slot:u32)->u32{return buildingIndex[buildingIndexOffset+${buildingCount + 5}u+slot];}
fn readEnclosure(row:u32)->vec2<f32>{return vec2<f32>(enclosure[enclosureOffset+2u*row],enclosure[enclosureOffset+2u*row+1u]);}
fn crossVectors(first:vec2<f32>,second:vec2<f32>)->f32{return first.x*second.y-first.y*second.x;}
// No constant-trip-count fill of candidateRows: all stores use a data-dependent append index.
fn gatherGridCandidates(center:vec2<f32>,searchRadius:f32)->vec2<u32>{
 var count=0u;var visited=0u;
 ${getBuildingTessellationGridWalk(
   props.pointIndex,
   `
  visited++;
  let displacement=readSample(candidate)-center;
  if(any(abs(displacement)>vec2<f32>(searchRadius))){continue;}
  let candidateDistance=length(displacement);
  if(!isFiniteValue(candidateDistance)){return vec2<u32>(count,2u);}
  if(candidateDistance<=searchRadius){
   if(count>=${candidateCapacity}u){return vec2<u32>(count,1u);}
   candidateRows[count]=candidate;count++;
  }`,
   `if(visited>=${props.maximumGridVisits}u){return vec2<u32>(count,1u);}`
 )}
 return vec2<u32>(count,0u);
}
fn gatherCandidates(center:vec2<f32>,searchRadius:f32)->vec2<u32>{
 let gridResult=gatherGridCandidates(center,searchRadius);
 if(gridResult.y!=1u){return gridResult;}
 // Retry a dense/coarse grid query independently; neither interval depends on global extent.
 var count=0u;var visited=0u;
 let interval=getCandidateInterval(center,searchRadius);
 for(var slot=interval.x;slot<interval.y;slot++){
  if(visited>=${props.maximumGridVisits}u){return vec2<u32>(count,1u);}
  visited++;
  let candidate=pointIndex[pointIndexOffset+interval.z+slot];
  let displacement=readSample(candidate)-center;
  if(any(abs(displacement)>vec2<f32>(searchRadius))){continue;}
  let candidateDistance=length(displacement);
  if(!isFiniteValue(candidateDistance)){return vec2<u32>(count,2u);}
  if(candidateDistance<=searchRadius){
   if(count>=${candidateCapacity}u){return vec2<u32>(count,1u);}
   candidateRows[count]=candidate;count++;
  }
 }
 return vec2<u32>(count,0u);
}
`,
    body: `let first=buildingIndex[buildingIndexOffset+4u+index];
let end=buildingIndex[buildingIndexOffset+5u+index];
let ownCount=end-first;
let rowBase=buildingStatusOffset+4u*index;
buildingStatus[rowBase]=0u;buildingStatus[rowBase+1u]=0u;buildingStatus[rowBase+2u]=0u;buildingStatus[rowBase+3u]=0u;
for(var part=0u;part<${piecesPerBuilding}u;part++){
 let record=cellsOffset+4u*(index*${piecesPerBuilding}u+part);
 cells[record]=0u;cells[record+1u]=select(0u,0xffffffffu,part<ownCount*${props.enclosurePieceCount}u);cells[record+2u]=index;cells[record+3u]=0xffffffffu;
 if(part<ownCount*${props.enclosurePieceCount}u){cells[record+3u]=readOwnSample(first+part/${props.enclosurePieceCount}u);}
}
if(buildingIndex[buildingIndexOffset+3u]!=0u){buildingStatus[rowBase+3u]=1u;return;}
if(buildingIndex[buildingIndexOffset+2u]!=0u||ownCount>${maximumSamplesPerBuilding}u){buildingStatus[rowBase+2u]=1u;return;}
if(ownCount==0u){buildingStatus[rowBase+3u]=1u;return;}
let radius=parameters[parametersOffset+1u];let tolerance=parameters[parametersOffset+2u];
if(!isFiniteValue(radius)||radius<=0.0||!isFiniteValue(tolerance)||tolerance<=0.0){buildingStatus[rowBase+3u]=1u;return;}
var lower=vec2<f32>(3.402823e38);var upper=vec2<f32>(-3.402823e38);
for(var part=0u;part<ownCount;part++){let point=readSample(readOwnSample(first+part));lower=min(lower,point);upper=max(upper,point);}
let center=0.5*lower+0.5*upper;
// Slack covers float32 cell assignment, the radius predicate and the twice-radius certificate.
let slack=max(4.0*tolerance,1.0e-5*(max(abs(center.x),abs(center.y))+radius));
let searchRadius=radius+slack;
if(!isFiniteValue(searchRadius)||!isFiniteValue(center.x)||!isFiniteValue(center.y)){buildingStatus[rowBase+3u]=1u;return;}
for(var part=0u;part<ownCount;part++){
 let ownDistance=distance(readSample(readOwnSample(first+part)),center)+slack;
 if(!isFiniteValue(ownDistance)){buildingStatus[rowBase+3u]=1u;return;}
 if(ownDistance>=radius){buildingStatus[rowBase+2u]=1u;return;}
}
${
  props.simpleEnclosure
    ? `if(enclosure[enclosureOffset]!=1.0){buildingStatus[rowBase+3u]=1u;return;}`
    : `// Convexity is certified against every edge, not only consecutive turns (a star ring can
// turn consistently). This is bounded by the named enclosure capacity, never by building count.
var enclosureValid=true;var enclosureArea=0.0;let enclosureAnchor=readEnclosure(0u)-center;
for(var vertex=0u;vertex<${enclosureCount}u;vertex++){
 let point=readEnclosure(vertex)-center;let next=readEnclosure((vertex+1u)%${enclosureCount}u)-center;
 enclosureValid=enclosureValid&&isFiniteValue(point.x)&&isFiniteValue(point.y);
 enclosureArea+=crossVectors(point-enclosureAnchor,next-enclosureAnchor);
 for(var otherVertex=0u;otherVertex<${enclosureCount}u;otherVertex++){
  let other=readEnclosure(otherVertex)-center;let side=crossVectors(next-point,other-point);
  enclosureValid=enclosureValid&&isFiniteValue(side)&&side>=0.0;
  if(otherVertex!=vertex&&all(other==point)){enclosureValid=false;}
 }
}
if(!enclosureValid||!isFiniteValue(enclosureArea)||enclosureArea<=0.0){buildingStatus[rowBase+3u]=1u;return;}
`
}
let gathered=gatherCandidates(center,searchRadius);
if(gathered.y!=0u){buildingStatus[rowBase+select(2u,3u,gathered.y==2u)]=1u;return;}
let exhaustive=gathered.x==buildingIndex[buildingIndexOffset];
var polygon:array<vec2<f32>,${clippingCapacity}>;
var scratch:array<vec2<f32>,${clippingCapacity}>;
var candidateOverflow=false;var overflow=false;var invalid=false;
// A building is one feature represented by the union of its own sample cells. Clipping against
// its other own sites partitions this union without changing its area or exterior boundary.
for(var part=0u;part<ownCount;part++){
 let sampleRow=readOwnSample(first+part);let site=readSample(sampleRow)-center;
 for(var enclosurePart=0u;enclosurePart<${props.enclosurePieceCount}u;enclosurePart++){
 let piece=part*${props.enclosurePieceCount}u+enclosurePart;
 var size=${enclosureCount}u;
 for(var vertex=0u;vertex<${enclosureCount}u;vertex++){
  polygon[vertex]=readEnclosure(${props.simpleEnclosure ? '1u+3u*enclosurePart+vertex' : 'vertex'})-center;
 }
 ${
   props.simpleEnclosure
     ? `let triangleArea=crossVectors(polygon[1u]-polygon[0u],polygon[2u]-polygon[0u]);
 let firstVertex=readEnclosure(1u+3u*enclosurePart);
 let secondVertex=readEnclosure(2u+3u*enclosurePart);
 let thirdVertex=readEnclosure(3u+3u*enclosurePart);
 let padding=all(firstVertex==secondVertex)&&all(secondVertex==thirdVertex);
 // Padding is exactly repeated source vertices. A real triangle lost to translation/underflow
 // must invalidate the building instead of silently dropping area.
 if(padding){size=0u;}
 else if(!isFiniteValue(triangleArea)||triangleArea<=0.0){invalid=true;}`
     : ''
 }
 for(var candidateIndex=0u;candidateIndex<gathered.x;candidateIndex++){
  if(size==0u||invalid){break;}
  let other=candidateRows[candidateIndex];if(other==sampleRow){continue;}
  let displacement=(readSample(other)-center)-site;let squared=dot(displacement,displacement);
  if(!isFiniteValue(squared)){invalid=true;break;}
  if(squared==0.0){
   if(readOwner(other)!=index){invalid=true;}
   else if(other<sampleRow){size=0u;}
   continue;
  }
  let threshold=0.5*squared;var outputSize=0u;
  for(var vertex=0u;vertex<size;vertex++){
   let start=polygon[vertex];let finish=polygon[(vertex+1u)%size];
   let startProducts=(start-site)*displacement;let endProducts=(finish-site)*displacement;
   let startSide=startProducts.x+startProducts.y-threshold;let endSide=endProducts.x+endProducts.y-threshold;
   if(!isFiniteValue(startProducts.x)||!isFiniteValue(startProducts.y)||!isFiniteValue(endProducts.x)||!isFiniteValue(endProducts.y)){invalid=true;break;}
   let inside=startSide<=0.0;let nextInside=endSide<=0.0;
   if(!isFiniteValue(startSide)||!isFiniteValue(endSide)){invalid=true;break;}
   if(inside){if(outputSize<${clippingCapacity}u){scratch[outputSize]=start;}outputSize++;}
   if(inside!=nextInside){
    // Normalize before subtracting: finite opposite sides can have an infinite difference.
    let sideScale=max(abs(startSide),abs(endSide));
    let scaledStart=startSide/sideScale;let scaledEnd=endSide/sideScale;
    let denominator=scaledStart-scaledEnd;let fraction=scaledStart/denominator;
    if(!isFiniteValue(sideScale)||sideScale==0.0||!isFiniteValue(denominator)||denominator==0.0||!isFiniteValue(fraction)||fraction<0.0||fraction>1.0){invalid=true;break;}
    let intersection=(1.0-fraction)*start+fraction*finish;
    invalid=invalid||!isFiniteValue(intersection.x)||!isFiniteValue(intersection.y);
    if(outputSize<${clippingCapacity}u){scratch[outputSize]=intersection;}outputSize++;
   }
  }
  if(outputSize>${clippingCapacity}u){invalid=true;break;}
  size=outputSize;for(var vertex=0u;vertex<size;vertex++){polygon[vertex]=scratch[vertex];}
 }
 if(size<3u){size=0u;}
 if(size>0u&&!invalid&&!exhaustive){
  var greatest=0.0;
  for(var vertex=0u;vertex<size;vertex++){greatest=max(greatest,distance(polygon[vertex],site));}
  // If an excluded site q could cut a point x, |q-site| <= 2|x-site|. Every excluded
  // site is beyond radius from center; this triangle bound therefore certifies completeness.
  let certificate=length(site)+2.0*greatest+slack;
  if(!isFiniteValue(greatest)||!isFiniteValue(certificate)){invalid=true;}
  else if(certificate>=radius){candidateOverflow=true;}
 }
 let record=cellsOffset+4u*(index*${piecesPerBuilding}u+piece);
 cells[record+1u]=select(size,0xffffffffu,invalid);
 if(!invalid&&size>${cellVertexCapacity}u){overflow=true;}
 if(!invalid&&size<=${cellVertexCapacity}u){
  cells[record]=size;
  for(var vertex=0u;vertex<size;vertex++){
   let destination=positionsOffset+2u*((index*${piecesPerBuilding}u+piece)*${cellVertexCapacity}u+vertex);
   let point=polygon[vertex]+center;invalid=invalid||!isFiniteValue(point.x)||!isFiniteValue(point.y);positions[destination]=point.x;positions[destination+1u]=point.y;
  }
 }
}
}
let valid=!candidateOverflow&&!overflow&&!invalid;
buildingStatus[rowBase]=u32(valid);buildingStatus[rowBase+1u]=u32(overflow&&!candidateOverflow&&!invalid);buildingStatus[rowBase+2u]=u32(candidateOverflow);buildingStatus[rowBase+3u]=u32(invalid);
if(!valid){for(var part=0u;part<${piecesPerBuilding}u;part++){
 let record=cellsOffset+4u*(index*${piecesPerBuilding}u+part);cells[record]=0u;
 if(part<ownCount*${props.enclosurePieceCount}u&&(candidateOverflow||invalid)){cells[record+1u]=0xffffffffu;}
}}`
  });
}
