// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Shared bounded planar ray predicates. Collinear overlap blocks at its nearest forward point. */
export function getUrbanRayDeclarations(segmentCount: number, tolerance: number): string {
  return `
fn readSegment(row:u32)->vec4<f32>{let base=segmentsOffset+row*4u;return vec4<f32>(segments[base],segments[base+1u],segments[base+2u],segments[base+3u]);}
fn crossVectors(first:vec2<f32>,second:vec2<f32>)->f32{return first.x*second.y-first.y*second.x;}
fn distanceToSegment(position:vec2<f32>,segment:vec4<f32>)->f32{
 let direction=segment.zw-segment.xy;let squared=dot(direction,direction);var fraction=0.0;
 if(squared>0.0){fraction=clamp(dot(position-segment.xy,direction)/squared,0.0,1.0);}
 return distance(position,segment.xy+fraction*direction);
}
fn castRay(origin:vec2<f32>,direction:vec2<f32>,maximum:f32)->vec2<f32>{
 var nearest=maximum;var hit=-1.0;
 for(var segmentIndex=0u;segmentIndex<${segmentCount}u;segmentIndex++){
  let segment=readSegment(segmentIndex);let edge=segment.zw-segment.xy;let displacement=segment.xy-origin;let denominator=crossVectors(direction,edge);var candidate=maximum;var intersects=false;
  if(distanceToSegment(origin,segment)<=${tolerance}){candidate=0.0;intersects=true;}
  else if(abs(denominator)>${tolerance}){
   let rayDistance=crossVectors(displacement,edge)/denominator;let fraction=crossVectors(displacement,direction)/denominator;
   if(rayDistance>=0.0&&fraction>=0.0&&fraction<=1.0){candidate=rayDistance;intersects=true;}
  }else if(abs(crossVectors(displacement,direction))<=${tolerance}){
   let first=dot(displacement,direction);let second=dot(segment.zw-origin,direction);
   if(max(first,second)>=0.0){candidate=max(0.0,min(first,second));intersects=true;}
  }
  if(intersects&&candidate<=nearest&&(candidate<nearest||hit<0.0)){nearest=candidate;hit=f32(segmentIndex);}
 }
 return vec2<f32>(nearest,hit);
}`;
}
