// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {dggs} from '@luma.gl/shaders';
import {S2_ANALYSIS_WGSL} from './s2-analysis-wgsl';

/** Fixed input-work tile, independent of ring size and the traversal visit budget. */
export const MAXIMUM_S2_COVERING_INPUT_EDGES_PER_INVOCATION = 4096;
/** Binary searches address uint32 ranges, so at most 32 bisections are needed. */
export const MAXIMUM_S2_COVERING_RANGE_SEARCH_STEPS = 32;

/** Stable planes and integer winding contributions shared by preparation and cell predicates. @internal */
export function getS2CoveringParallelWGSL(tolerance: number, includePositions = true): string {
  let source = `${dggs.source}\n${S2_ANALYSIS_WGSL}
const COVER_TOLERANCE:f32=${tolerance};
const COVER_INPUT_TILE:u32=${MAXIMUM_S2_COVERING_INPUT_EDGES_PER_INVOCATION}u;
fn coverPosition(row:u32)->vec3f {let offset=positionsOffset+row*3u;return vec3f(positions[offset],positions[offset+1u],positions[offset+2u]);}
fn coverEdgeCross(first:vec3f,second:vec3f)->vec3f {
  let difference=second-select(-first,first,dot(first,second)>=0.0);
  return cross(first,difference);
}
fn coverOnArc(first:vec3f,second:vec3f,point:vec3f)->bool {
  let firstDirection=normalize(first);
  let secondDirection=normalize(second);
  let pointDirection=normalize(point);
  if(min(distance(firstDirection,pointDirection),distance(secondDirection,pointDirection))<=2.0*sin(COVER_TOLERANCE*0.5)){return true;}
  let normal=normalize(coverEdgeCross(first,second));
  let firstSide=dot(coverEdgeCross(firstDirection,pointDirection),normal);
  let secondSide=dot(coverEdgeCross(pointDirection,secondDirection),normal);
  return dot(firstDirection+secondDirection,pointDirection)>0.0&&firstSide>=0.0&&secondSide>=0.0&&abs(dot(normal,pointDirection))<=sin(COVER_TOLERANCE);
}
fn coverArcsIntersect(first:vec3f,second:vec3f,third:vec3f,fourth:vec3f)->bool {
  let crossing=cross(coverEdgeCross(first,second),coverEdgeCross(third,fourth));
  if(length(crossing)>COVER_TOLERANCE*COVER_TOLERANCE) {
    let point=normalize(crossing);
    return (coverOnArc(first,second,point)&&coverOnArc(third,fourth,point))||(coverOnArc(first,second,-point)&&coverOnArc(third,fourth,-point));
  }
  return coverOnArc(first,second,third)||coverOnArc(first,second,fourth);
}
struct CoverContribution { winding:vec4i, boundary:u32, touched:u32 }
// Signed ray crossings compute the same integer winding degree as the angle sum. Integer
// partials combine exactly across tiles; no floating atomic sum or polygon-sized reduction.
fn coverRingContribution(cell:vec2u,ringBegin:u32,ringEnd:u32,begin:u32,end:u32)->CoverContribution {
  var vertices:array<vec3f,4>;var normals:array<vec3f,4>;
  for(var corner=0u;corner<4u;corner++){vertices[corner]=s2AnalysisVertex(cell,corner);}
  for(var corner=0u;corner<4u;corner++){normals[corner]=normalize(coverEdgeCross(vertices[corner],vertices[(corner+1u)%4u]));}
  var winding=vec4i(0);var boundary=0u;var touched=0u;
  for(var vertex=begin;vertex<end&&vertex-begin<COVER_INPUT_TILE;vertex++) {
    let first=coverPosition(vertex);let second=coverPosition(select(vertex+1u,ringBegin,vertex+1u==ringEnd));
    var vertexInside=true;
    for(var corner=0u;corner<4u;corner++){vertexInside=vertexInside&&dot(normals[corner],first)>=-COVER_TOLERANCE;}
    if(vertexInside){touched=1u;}
    for(var corner=0u;corner<4u;corner++) {
      let query=vertices[corner];let direction=normalize(query);
      if(coverOnArc(first,second,query)){boundary|=1u<<corner;}
      let firstDifference=first-select(-direction,direction,dot(direction,first)>=0.0);
      let secondDifference=second-select(-direction,direction,dot(direction,second)>=0.0);
      let firstTangent=cross(direction,firstDifference);let secondTangent=cross(direction,secondDifference);
      let magnitude=abs(direction);var axis=vec3f(0.0,0.0,1.0);
      if(magnitude.x<=magnitude.y&&magnitude.x<=magnitude.z){axis=vec3f(1.0,0.0,0.0);}
      else if(magnitude.y<=magnitude.z){axis=vec3f(0.0,1.0,0.0);}
      let firstHeight=dot(firstTangent,axis);let secondHeight=dot(secondTangent,axis);
      let side=dot(direction,cross(firstTangent,secondTangent));
      if(firstHeight<=0.0&&secondHeight>0.0&&side>0.0){winding[corner]+=1;}
      if(firstHeight>0.0&&secondHeight<=0.0&&side<0.0){winding[corner]-=1;}
      if(coverArcsIntersect(query,vertices[(corner+1u)%4u],first,second)){touched=1u;}
    }
  }
  return CoverContribution(winding,boundary,touched);
}
fn coverPositiveWindingMask(cell:vec2u,pole:vec3f,winding:vec4i)->u32 {
  var mask=0u;
  for(var corner=0u;corner<4u;corner++) {
    // A valid ring and its interior lie in its validated hemisphere. This also excludes
    // the antipodal image, including the undefined tangent at an antipodal ring vertex.
    if(winding[corner]>0&&dot(pole,s2AnalysisVertex(cell,corner))>0.0){mask|=1u<<corner;}
  }
  return mask;
}`;
  if (!includePositions) {
    const positionBegin = source.indexOf('fn coverPosition(');
    const positionEnd = source.indexOf('fn coverEdgeCross(', positionBegin);
    source = source.slice(0, positionBegin) + source.slice(positionEnd);
    const contributionBegin = source.indexOf('fn coverRingContribution(');
    const contributionEnd = source.indexOf('fn coverPositiveWindingMask(', contributionBegin);
    source = source.slice(0, contributionBegin) + source.slice(contributionEnd);
  }
  return source;
}
