// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {assertUnitVector} from '../global-coordinates/assert-unit-vector';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {dggs} from '@luma.gl/shaders';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validatePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {S2_ANALYSIS_WGSL} from './s2-analysis-wgsl';

export type GPUS2PolygonCoveringProps = {
  id?: string;
  /** One prepared simple spherical polygon: CCW shell then CCW holes, unit xyz minor arcs. */
  positions: GraphDataView<'float32x3'>;
  ringOffsets: GraphDataView<'uint32'>;
  complement?: boolean;
  minimumLevel?: number;
  maximumLevel: number;
  mode?: 'exterior' | 'interior';
  capacity: number;
  /** Maximum visited hierarchy cells. Exhaustion is explicit; default 65536. */
  maximumVisitedCells?: number;
  tolerance?: number;
  /** Optional common hemisphere pole. Omitted: each ring uses its normalized vertex mean. */
  hemisphereCenter?: readonly [number, number, number];
  output: {
    cells: GraphDataView<'uint32x2'>;
    /** (written, required so far), not a complete requirement when status has bit 4. */
    counts: GraphDataView<'uint32x2'>;
    /** One word: 1 invalid input, 2 output capacity, 4 traversal budget exhausted. */
    status: GraphDataView<'uint32'>;
    visitedCount?: GraphDataView<'uint32'>;
  };
};

/** Adaptive mixed-level S2 interior/exterior cover, entirely within a bounded GPU traversal. */
export class GPUS2PolygonCovering implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUS2PolygonCoveringProps) {
    this.id = props.id ?? 's2-polygon-covering';
    if (props.hemisphereCenter) assertUnitVector(props.hemisphereCenter);
    validatePackedView(props.positions, ['float32x3'], this.id);
    validatePackedView(props.ringOffsets, ['uint32'], this.id);
    validatePackedView(props.output.cells, ['uint32x2'], this.id);
    validatePackedView(props.output.counts, ['uint32x2'], this.id);
    validatePackedView(props.output.status, ['uint32'], this.id);
    if (props.output.visitedCount) {
      validatePackedView(props.output.visitedCount, ['uint32'], this.id);
      assert(props.output.visitedCount.length === 1);
    }
    assert(
      Number.isInteger(props.maximumLevel) && props.maximumLevel >= 0 && props.maximumLevel <= 15
    );
    assert(
      Number.isInteger(props.minimumLevel ?? 0) &&
        (props.minimumLevel ?? 0) >= 0 &&
        (props.minimumLevel ?? 0) <= props.maximumLevel
    );
    assert(
      Number.isInteger(props.capacity) &&
        props.capacity >= 1 &&
        props.output.cells.length === props.capacity &&
        props.output.counts.length === 1 &&
        props.output.status.length === 1
    );
    assert(
      Number.isInteger(props.maximumVisitedCells ?? 65536) &&
        (props.maximumVisitedCells ?? 65536) >= 6 &&
        (props.maximumVisitedCells ?? 65536) <= 1048576
    );
    assert(
      Number.isFinite(props.tolerance ?? 1e-6) &&
        (props.tolerance ?? 1e-6) >= 1e-7 &&
        (props.tolerance ?? 1e-6) <= 1e-3
    );
    assert(props.ringOffsets.length >= 2);
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      props.ringOffsets
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.ringOffsets,
      ...Object.values(props.output)
    ]);
    return [
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPUS2PolygonCovering',
        variant: props.mode ?? 'exterior',
        invocationCount: 1,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'offsets', view: props.ringOffsets, type: 'u32', access: 'read'},
          {name: 'cells', view: props.output.cells, type: 'u32', access: 'read_write'},
          {name: 'counts', view: props.output.counts, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
          ...(props.output.visitedCount
            ? [
                {
                  name: 'visited',
                  view: props.output.visitedCount,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        declarations: `${dggs.source}\n${S2_ANALYSIS_WGSL}
const COVER_TOLERANCE:f32=${props.tolerance ?? 1e-6};
fn coverPosition(row:u32)->vec3f {let offset=positionsOffset+row*3u;return vec3f(positions[offset],positions[offset+1u],positions[offset+2u]);}
fn coverOnArc(first:vec3f,second:vec3f,point:vec3f)->bool {let normal=normalize(cross(first,second));return abs(dot(normal,point))<=COVER_TOLERANCE&&dot(cross(first,point),normal)>=-COVER_TOLERANCE&&dot(cross(point,second),normal)>=-COVER_TOLERANCE;}
fn coverContains(point:vec3f)->vec2u {
  var inside=false;var boundary=false;
  for(var ring=0u;ring<${props.ringOffsets.length - 1}u;ring++) {
    let begin=offsets[offsetsOffset+ring];let end=offsets[offsetsOffset+ring+1u];var winding=0.0;
    for(var edge=begin;edge<end;edge++) {let first=coverPosition(edge);let second=coverPosition(select(edge+1u,begin,edge+1u==end));boundary=boundary||coverOnArc(first,second,point);let numerator=dot(point,cross(first,second));let denominator=dot(first,second)-dot(point,first)*dot(point,second);if(abs(numerator)+abs(denominator)>1e-12){winding+=atan2(numerator,denominator);}}
    if(ring==0u){inside=winding>3.14159265;}else{inside=inside&&winding<=3.14159265;}
  }
  return vec2u(select(0u,1u,${props.complement ? '!inside' : 'inside'}),select(0u,1u,boundary));
}
fn coverCellContains(normals:ptr<function,array<vec3f,4>>,point:vec3f)->bool {
  for(var edge=0u;edge<4u;edge++){if(dot((*normals)[edge],point)<-COVER_TOLERANCE){return false;}}
  return true;
}
fn coverCellOnArc(first:vec3f,second:vec3f,normal:vec3f,point:vec3f)->bool {
  return abs(dot(normal,point))<=COVER_TOLERANCE&&dot(cross(first,point),normal)>=-COVER_TOLERANCE&&dot(cross(point,second),normal)>=-COVER_TOLERANCE;
}
fn coverClassify(cell:vec2u)->vec2u {
  var intersects=false;var contained=true;var touched=false;
  var vertices:array<vec3f,4>;var normals:array<vec3f,4>;var crosses:array<vec3f,4>;
  for(var edge=0u;edge<4u;edge++){vertices[edge]=s2AnalysisVertex(cell,edge);}
  for(var edge=0u;edge<4u;edge++){crosses[edge]=cross(vertices[edge],vertices[(edge+1u)%4u]);normals[edge]=normalize(crosses[edge]);}
  // Polygon vertex containment does not depend on the current cell edge.
  for(var ring=0u;ring<${props.ringOffsets.length - 1}u;ring++){
    let begin=offsets[offsetsOffset+ring];let end=offsets[offsetsOffset+ring+1u];
    for(var vertex=begin;vertex<end;vertex++){if(coverCellContains(&normals,coverPosition(vertex))){touched=true;intersects=true;}}
  }
  for(var edge=0u;edge<4u;edge++) {
    let first=vertices[edge];let second=vertices[(edge+1u)%4u];let normal=normals[edge];
    let classification=coverContains(first);intersects=intersects||classification.x!=0u||classification.y!=0u;contained=contained&&classification.x!=0u&&classification.y==0u;
    for(var ring=0u;ring<${props.ringOffsets.length - 1}u;ring++){
      let begin=offsets[offsetsOffset+ring];let end=offsets[offsetsOffset+ring+1u];
      for(var vertex=begin;vertex<end;vertex++){
        let third=coverPosition(vertex);let fourth=coverPosition(select(vertex+1u,begin,vertex+1u==end));
        let crossing=cross(crosses[edge],cross(third,fourth));
        if(length(crossing)>COVER_TOLERANCE*COVER_TOLERANCE){
          let point=normalize(crossing);
          if((coverCellOnArc(first,second,normal,point)&&coverOnArc(third,fourth,point))||(coverCellOnArc(first,second,normal,-point)&&coverOnArc(third,fourth,-point))){touched=true;intersects=true;}
        }else if(coverCellOnArc(first,second,normal,third)||coverCellOnArc(first,second,normal,fourth)){touched=true;intersects=true;}
      }
    }
  }
  return vec2u(select(0u,1u,intersects),select(0u,1u,contained&&!touched));
}`,
        body: `var failure=0u;var required=0u;var visitedCount=0u;
for(var ring=0u;ring<${props.ringOffsets.length - 1}u;ring++){let begin=offsets[offsetsOffset+ring];let end=offsets[offsetsOffset+ring+1u];if(begin>end||end>${props.positions.length}u||end-begin<3u){failure=1u;break;}
  var center=vec3f(0.0);for(var vertex=begin;vertex<end;vertex++){let point=coverPosition(vertex);let squared=dot(point,point);if((bitcast<u32>(squared)&0x7f800000u)==0x7f800000u||abs(squared-1.0)>1e-4){failure=1u;}center+=point;}
  ${props.hemisphereCenter ? `center = vec3f(${props.hemisphereCenter.join(',')});` : ''}
  if(length(center)<=COVER_TOLERANCE){failure=1u;break;}center=normalize(center);
  for(var vertex=begin;vertex<end&&failure==0u;vertex++){
    let next=select(vertex+1u,begin,vertex+1u==end);let first=coverPosition(vertex);let second=coverPosition(next);let normal=cross(first,second);
    if(dot(center,first)<=COVER_TOLERANCE||length(normal)<=COVER_TOLERANCE){failure=1u;break;}
    // A crossing or nonadjacent touch cannot define a simple ring or a guaranteed cover.
    for(var other=vertex+1u;other<end;other++){
      let otherNext=select(other+1u,begin,other+1u==end);if(other==next||otherNext==vertex){continue;}
      let third=coverPosition(other);let fourth=coverPosition(otherNext);let crossing=cross(normal,cross(third,fourth));
      if(length(crossing)>COVER_TOLERANCE*COVER_TOLERANCE){
        let intersection=normalize(crossing);
        if((coverOnArc(first,second,intersection)&&coverOnArc(third,fourth,intersection))||(coverOnArc(first,second,-intersection)&&coverOnArc(third,fourth,-intersection))){failure=1u;}
      }else if(coverOnArc(first,second,third)||coverOnArc(first,second,fourth)){failure=1u;}
    }
  }
}
var stack:array<vec2u,52>;var size=6u;
for(var face=0u;face<6u;face++){stack[5u-face]=vec2u((2u*face+1u)<<28u,0u);}
loop {if(size==0u||failure!=0u){break;}if(visitedCount>=${props.maximumVisitedCells ?? 65536}u){failure|=4u;break;}size-=1u;let cell=stack[size];visitedCount+=1u;let classification=coverClassify(cell);if(classification.x==0u){continue;}let level=dggs_s2_get_level(cell);
  let emit=level>=${props.minimumLevel ?? 0}u&&(classification.y!=0u||(${props.mode !== 'interior'}&&level==${props.maximumLevel}u));
  if(emit){if(required<${props.capacity}u){cells[cellsOffset+required*2u]=cell.y;cells[cellsOffset+required*2u+1u]=cell.x;}required+=1u;}
  else if(level<${props.maximumLevel}u){let marker=dggs_u64_shift_left(vec2u(0u,1u),2u*(30u-level));let childMarker=dggs_u64_shift_right(marker,2u);let first=dggs_u64_add(dggs_u64_subtract(cell,marker),childMarker);let step=dggs_u64_shift_left(childMarker,1u);var child=first;for(var offset=0u;offset<4u;offset++){stack[size+3u-offset]=child;child=dggs_u64_add(child,step);}size+=4u;}
}
for(var slot=min(required,${props.capacity}u);slot<${props.capacity}u;slot++){cells[cellsOffset+slot*2u]=0u;cells[cellsOffset+slot*2u+1u]=0u;}
if(required>${props.capacity}u){failure|=2u;}counts[countsOffset]=min(required,${props.capacity}u);counts[countsOffset+1u]=required;status[statusOffset]=failure;
${props.output.visitedCount ? 'visited[visitedOffset]=visitedCount;' : ''}`
      })
    ];
  }
}
