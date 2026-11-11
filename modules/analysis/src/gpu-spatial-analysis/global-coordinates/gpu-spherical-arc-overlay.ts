// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assertUnitVector} from './assert-unit-vector';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateSinglePackedView,
  validateMaximumCount,
  validateRequirement,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {GUARDED_ATAN2_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  GPU_SPHERICAL_ARC_OVERLAY_MAXIMUM_INPUT_VERTICES,
  GPU_SPHERICAL_ARC_OVERLAY_MAXIMUM_OUTPUT_VERTICES,
  GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
  GPU_SPHERICAL_MAXIMUM_ARRANGEMENT_TOLERANCE,
  GPU_SPHERICAL_MINIMUM_TOLERANCE,
  GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT,
  validateAngularTolerance
} from './spherical-limits';

/** Status bit: invalid topology, input or reference point. */
export const GPU_SPHERICAL_ARC_OVERLAY_STATUS_INVALID = 1;
/** Status bit: the result needs more vertices or rings than the output holds; see `counts[2]`. */
export const GPU_SPHERICAL_ARC_OVERLAY_STATUS_CAPACITY = 2;
/** Status bit: an unresolved numerical separation or junction. */
export const GPU_SPHERICAL_ARC_OVERLAY_STATUS_UNRESOLVED = 4;

export type GPUSphericalArcOperand = {
  positions: GraphDataView<'float32x3'>;
  /** Simple disjoint boundary rings, no duplicate closing vertices. Nesting uses parity. */
  ringOffsets: GraphDataView<'uint32'>;
  /** Known membership of referencePoint, which must be separated from every boundary. */
  referenceInside: boolean;
};
export type GPUSphericalArcOverlayProps = {
  id?: string;
  /**
   * Coordinate space of the unit vectors: longitude-latitude (their embedding on the sphere) with
   * great-circle edges. Planar contexts are rejected. Defaults to longitude-latitude, no metric.
   */
  spatialContext?: GPUSpatialContext;
  left: GPUSphericalArcOperand;
  right: GPUSphericalArcOperand;
  referencePoint: readonly [number, number, number];
  operation: 'intersection' | 'union' | 'difference' | 'symmetric-difference';
  tolerance?: number;
  output: {
    /** Open assembled unit-xyz rings, with result interior on the left. Capacity 3 to {@link GPU_SPHERICAL_ARC_OVERLAY_MAXIMUM_OUTPUT_VERTICES} (1024). */
    positions: GraphDataView<'float32x3'>;
    ringOffsets: GraphDataView<'uint32'>;
    /**
     * Four words: written vertices (`counts[0]`), written rings (`counts[1]`), required segments
     * (`counts[2]`), reference membership (`counts[3]`, 1 when the reference point is inside).
     */
    counts: GraphDataView<'uint32'>;
    /**
     * One word, a bit set: {@link GPU_SPHERICAL_ARC_OVERLAY_STATUS_INVALID} (1),
     * {@link GPU_SPHERICAL_ARC_OVERLAY_STATUS_CAPACITY} (2),
     * {@link GPU_SPHERICAL_ARC_OVERLAY_STATUS_UNRESOLVED} (4); 0 is complete.
     */
    status: GraphDataView<'uint32'>;
  };
};

/** Intrinsic spherical arc arrangement and ring assembly, including non-hemisphere regions. */
export class GPUSphericalArcOverlay implements GPUCommandNodeProducer {
  readonly id: string;
  /** Validated coordinate space of the unit vectors. */
  readonly spatialContext: GPUSpatialContext;
  readonly referenceInside: boolean;
  constructor(readonly props: GPUSphericalArcOverlayProps) {
    this.id = props.id ?? 'spherical-arc-overlay';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext ?? GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
      GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT
    );
    assertUnitVector(props.referencePoint, this.id, 'referencePoint');
    validateAngularTolerance(
      this.id,
      props.tolerance ?? 1e-6,
      GPU_SPHERICAL_MINIMUM_TOLERANCE,
      GPU_SPHERICAL_MAXIMUM_ARRANGEMENT_TOLERANCE
    );
    for (const operand of [props.left, props.right]) {
      validateSinglePackedView(operand.positions, ['float32x3'], this.id);
      validateSinglePackedView(operand.ringOffsets, ['uint32'], this.id);
      validateRequirement(
        this.id,
        operand.ringOffsets.length >= 1,
        'left and right ringOffsets must hold at least one entry'
      );
    }
    validateMaximumCount(
      this.id,
      'left plus right positions',
      props.left.positions.length + props.right.positions.length,
      GPU_SPHERICAL_ARC_OVERLAY_MAXIMUM_INPUT_VERTICES
    );
    validateSinglePackedView(props.output.positions, ['float32x3'], this.id);
    for (const view of [props.output.ringOffsets, props.output.counts, props.output.status])
      validateSinglePackedView(view, ['uint32'], this.id);
    validateRequirement(
      this.id,
      props.output.positions.length >= 3 &&
        props.output.positions.length <= GPU_SPHERICAL_ARC_OVERLAY_MAXIMUM_OUTPUT_VERTICES,
      `output.positions must hold 3 to ${GPU_SPHERICAL_ARC_OVERLAY_MAXIMUM_OUTPUT_VERTICES} entries`
    );
    validateRequirement(
      this.id,
      props.output.ringOffsets.length >= 2,
      'output.ringOffsets must hold at least 2 entries'
    );
    validateRequirement(
      this.id,
      props.output.counts.length === 4 && props.output.status.length === 1,
      'output.counts must hold 4 entries and output.status 1 entry'
    );
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.left.positions,
      props.right.positions,
      props.left.ringOffsets,
      props.right.ringOffsets
    ]);
    const left = props.left.referenceInside;
    const right = props.right.referenceInside;
    this.referenceInside =
      props.operation === 'intersection'
        ? left && right
        : props.operation === 'union'
          ? left || right
          : props.operation === 'difference'
            ? left && !right
            : left !== right;
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.left.positions,
      props.right.positions,
      props.left.ringOffsets,
      props.right.ringOffsets,
      ...Object.values(props.output)
    ]);
    const capacity = props.output.positions.length;
    const packedOffsets = createTransientView(
      graph,
      `${id}-offsets`,
      'uint32',
      props.left.ringOffsets.length + props.right.ringOffsets.length
    );
    const scratch = createTransientView(graph, `${id}-segments`, 'float32', capacity * 7);
    const nodes: GPUCommandNode<Parameters>[] = [];
    for (const [operand, offset] of [
      [props.left, 0],
      [props.right, props.left.ringOffsets.length]
    ] as const) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-pack-${offset}`,
          operation: 'GPUSphericalArcOverlay',
          variant: 'pack-offsets',
          invocationCount: operand.ringOffsets.length,
          bindings: [
            {name: 'source', view: operand.ringOffsets, type: 'u32', access: 'read'},
            {name: 'packed', view: packedOffsets, type: 'u32', access: 'read_write'}
          ],
          body: `packed[packedOffset+${offset}u+index]=source[sourceOffset+index];`
        })
      );
    }
    return [
      ...nodes,
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPUSphericalArcOverlay',
        variant: props.operation,
        invocationCount: 1,
        bindings: [
          {name: 'leftPositions', view: props.left.positions, type: 'f32', access: 'read'},
          {name: 'rightPositions', view: props.right.positions, type: 'f32', access: 'read'},
          {name: 'offsets', view: packedOffsets, type: 'u32', access: 'read'},
          {name: 'scratch', view: scratch, type: 'f32', access: 'read_write'},
          {name: 'output', view: props.output.positions, type: 'f32', access: 'read_write'},
          {name: 'rings', view: props.output.ringOffsets, type: 'u32', access: 'read_write'},
          {name: 'counts', view: props.output.counts, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `const ARC_TOLERANCE:f32=${props.tolerance ?? 1e-6};
${GUARDED_ATAN2_WGSL}
const ARC_REFERENCE=vec3f(${props.referencePoint.join(',')});
fn arcStart(row:u32)->vec3f{let offset=scratchOffset+row*7u;return vec3f(scratch[offset],scratch[offset+1u],scratch[offset+2u]);}
fn arcEnd(row:u32)->vec3f{let offset=scratchOffset+row*7u+3u;return vec3f(scratch[offset],scratch[offset+1u],scratch[offset+2u]);}
fn arcStore(row:u32,start:vec3f,end:vec3f){let offset=scratchOffset+row*7u;scratch[offset]=start.x;scratch[offset+1u]=start.y;scratch[offset+2u]=start.z;scratch[offset+3u]=end.x;scratch[offset+4u]=end.y;scratch[offset+5u]=end.z;}
fn arcUsed(row:u32)->bool{return scratch[scratchOffset+row*7u+6u]!=0.0;}

fn arcLength(first:vec3f,second:vec3f)->f32{return guardedAtan2(length(cross(first,second)),dot(first,second));}
fn arcVertexCount(operand:u32)->u32{return select(${props.left.positions.length}u,${props.right.positions.length}u,operand==1u);}
fn arcRingCount(operand:u32)->u32{return select(${props.left.ringOffsets.length - 1}u,${props.right.ringOffsets.length - 1}u,operand==1u);}
fn arcOffset(operand:u32,row:u32)->u32{if(operand==0u){return offsets[offsetsOffset+row];}return offsets[offsetsOffset+${props.left.ringOffsets.length}u+row];}
fn arcPosition(operand:u32,row:u32)->vec3f{if(operand==0u){let offset=leftPositionsOffset+row*3u;return vec3f(leftPositions[offset],leftPositions[offset+1u],leftPositions[offset+2u]);}let offset=rightPositionsOffset+row*3u;return vec3f(rightPositions[offset],rightPositions[offset+1u],rightPositions[offset+2u]);}
fn arcNext(operand:u32,row:u32)->u32{for(var ring=0u;ring<arcRingCount(operand);ring++){let begin=arcOffset(operand,ring);let end=arcOffset(operand,ring+1u);if(row>=begin&&row<end){return select(row+1u,begin,row+1u==end);}}return row;}
fn arcOn(first:vec3f,second:vec3f,point:vec3f,margin:f32)->bool{let normal=normalize(cross(first,second));return abs(dot(normal,point))<=max(margin,ARC_TOLERANCE)&&dot(cross(first,point),normal)>=-margin&&dot(cross(point,second),normal)>=-margin;}
fn arcBoolean(left:bool,right:bool)->bool{return ${props.operation === 'intersection' ? 'left&&right' : props.operation === 'union' ? 'left||right' : props.operation === 'difference' ? 'left&&!right' : 'left!=right'};}
// Parity is anchored at a known point, so a ring may enclose more than a hemisphere.
// Alternate two-arc paths avoid rays through vertices without choosing longitude seams.
fn arcContains(operand:u32,point:vec3f)->vec2u{
  let referenceInside=select(${Number(props.left.referenceInside)}u,${Number(props.right.referenceInside)}u,operand==1u);
  for(var edge=0u;edge<arcVertexCount(operand);edge++){if(arcOn(arcPosition(operand,edge),arcPosition(operand,arcNext(operand,edge)),point,ARC_TOLERANCE)){return vec2u(0u,2u);}}
  let choices=array<vec3f,7>(ARC_REFERENCE,normalize(vec3f(1.0,0.371,0.127)),normalize(vec3f(-0.253,1.0,0.419)),normalize(vec3f(0.173,-0.391,1.0)),normalize(vec3f(-1.0,0.137,-0.293)),normalize(vec3f(0.317,-1.0,0.193)),normalize(vec3f(-0.227,0.433,-1.0)));
  for(var attempt=0u;attempt<7u;attempt++){
    var parity=referenceInside;var ambiguous=false;
    for(var part=0u;part<2u;part++){
      let first=select(ARC_REFERENCE,choices[attempt],part==1u);let second=select(choices[attempt],point,part==1u);let crossing=cross(first,second);
      if(length(crossing)<ARC_TOLERANCE){if(dot(first,second)<0.0){ambiguous=true;}continue;}
      let normal=normalize(crossing);
      for(var edge=0u;edge<arcVertexCount(operand);edge++){
        let third=arcPosition(operand,edge);let fourth=arcPosition(operand,arcNext(operand,edge));let firstSide=dot(normal,third);let secondSide=dot(normal,fourth);
        if((abs(firstSide)<=ARC_TOLERANCE&&arcOn(first,second,third,ARC_TOLERANCE))||(abs(secondSide)<=ARC_TOLERANCE&&arcOn(first,second,fourth,ARC_TOLERANCE))){ambiguous=true;}
        if(firstSide*secondSide<0.0){let intersection=normalize(cross(crossing,cross(third,fourth)));if((arcOn(first,second,intersection,-ARC_TOLERANCE)&&arcOn(third,fourth,intersection,-ARC_TOLERANCE))||(arcOn(first,second,-intersection,-ARC_TOLERANCE)&&arcOn(third,fourth,-intersection,-ARC_TOLERANCE))){parity^=1u;}}
      }
    }
    if(!ambiguous){return vec2u(parity,0u);}
  }
  return vec2u(0u,4u);
}
fn arcDistance(first:vec3f,second:vec3f,point:vec3f)->f32{let normal=normalize(cross(first,second));let projected=cross(normal,cross(point,normal));var distance=min(arcLength(first,point),arcLength(second,point));if(length(projected)>1e-10){let nearest=normalize(projected);if(arcOn(first,second,nearest,0.0)){distance=min(distance,arcLength(nearest,point));}}return distance;}`,
        body: `var failure=0u;var required=0u;for(var row=0u;row<${capacity}u;row++){scratch[scratchOffset+row*7u+6u]=0.0;}
for(var operand=0u;operand<2u;operand++){
  if(arcOffset(operand,0u)!=0u||arcOffset(operand,arcRingCount(operand))!=arcVertexCount(operand)){failure=1u;}
  for(var ring=0u;ring<arcRingCount(operand);ring++){let begin=arcOffset(operand,ring);let end=arcOffset(operand,ring+1u);if(begin>end||end>arcVertexCount(operand)||end-begin<3u){failure=1u;}}
  if(failure!=0u){break;}
  for(var edge=0u;edge<arcVertexCount(operand);edge++){
    let first=arcPosition(operand,edge);let second=arcPosition(operand,arcNext(operand,edge));let squared=dot(first,first);
    if((bitcast<u32>(squared)&0x7f800000u)==0x7f800000u||abs(squared-1.0)>1e-4||length(cross(first,second))<=ARC_TOLERANCE||arcOn(first,second,ARC_REFERENCE,ARC_TOLERANCE)){failure=1u;}
    for(var other=edge+1u;other<arcVertexCount(operand);other++){
      if(other==arcNext(operand,edge)||arcNext(operand,other)==edge){continue;}
      let third=arcPosition(operand,other);let fourth=arcPosition(operand,arcNext(operand,other));let crossing=cross(cross(first,second),cross(third,fourth));
      if(length(crossing)>ARC_TOLERANCE*ARC_TOLERANCE){let point=normalize(crossing);if((arcOn(first,second,point,ARC_TOLERANCE)&&arcOn(third,fourth,point,ARC_TOLERANCE))||(arcOn(first,second,-point,ARC_TOLERANCE)&&arcOn(third,fourth,-point,ARC_TOLERANCE))){failure=1u;}}
      else if(arcOn(first,second,third,ARC_TOLERANCE)||arcOn(first,second,fourth,ARC_TOLERANCE)){failure=1u;}
    }
  }
}
for(var operand=0u;operand<2u&&failure==0u;operand++){
  for(var edge=0u;edge<arcVertexCount(operand)&&failure==0u;edge++){
    let first=arcPosition(operand,edge);let second=arcPosition(operand,arcNext(operand,edge));var points:array<vec3f,130>;var angles:array<f32,130>;var pointCount=2u;points[0]=first;points[1]=second;angles[0]=0.0;angles[1]=arcLength(first,second);
    for(var other=0u;other<arcVertexCount(1u-operand);other++){
      let third=arcPosition(1u-operand,other);let fourth=arcPosition(1u-operand,arcNext(1u-operand,other));let crossing=cross(cross(first,second),cross(third,fourth));var candidates:array<vec3f,2>;
      if(length(crossing)>ARC_TOLERANCE*ARC_TOLERANCE){candidates[0]=normalize(crossing);candidates[1]=-candidates[0];}else{candidates[0]=third;candidates[1]=fourth;}
      for(var candidate=0u;candidate<2u;candidate++){let point=candidates[candidate];if(arcOn(first,second,point,ARC_TOLERANCE)&&arcOn(third,fourth,point,ARC_TOLERANCE)){
        let angle=arcLength(first,point);var duplicate=false;for(var previous=0u;previous<pointCount;previous++){duplicate=duplicate||abs(angles[previous]-angle)<=2.0*ARC_TOLERANCE;}
        if(!duplicate){if(pointCount>=130u){failure=2u;break;}var position=pointCount;loop{if(position==0u||angles[position-1u]<=angle){break;}angles[position]=angles[position-1u];points[position]=points[position-1u];position-=1u;}angles[position]=angle;points[position]=point;pointCount+=1u;}
      }}
    }
    for(var piece=0u;piece+1u<pointCount&&failure==0u;piece++){
      let start=points[piece];let end=points[piece+1u];if(arcLength(start,end)<=4.0*ARC_TOLERANCE){failure=4u;break;}
      let middle=normalize(start+end);let normal=normalize(cross(start,end));var separation=16.0*ARC_TOLERANCE;
      for(var source=0u;source<2u;source++){for(var other=0u;other<arcVertexCount(source);other++){let third=arcPosition(source,other);let fourth=arcPosition(source,arcNext(source,other));if(!arcOn(third,fourth,middle,ARC_TOLERANCE)){separation=min(separation,arcDistance(third,fourth,middle)*0.25);}}}
      if(separation<=2.0*ARC_TOLERANCE){failure=4u;break;}
      let leftPoint=normalize(middle+normal*separation);let rightPoint=normalize(middle-normal*separation);
      let firstLeft=arcContains(0u,leftPoint);let secondLeft=arcContains(1u,leftPoint);let firstRight=arcContains(0u,rightPoint);let secondRight=arcContains(1u,rightPoint);
      if((firstLeft.y|secondLeft.y|firstRight.y|secondRight.y)!=0u){failure=4u;break;}
      let leftInside=arcBoolean(firstLeft.x!=0u,secondLeft.x!=0u);let rightInside=arcBoolean(firstRight.x!=0u,secondRight.x!=0u);
      if(leftInside==rightInside){continue;}
      let startPoint=select(end,start,leftInside);let endPoint=select(start,end,leftInside);var duplicate=false;
      for(var previous=0u;previous<min(required,${capacity}u);previous++){duplicate=duplicate||(distance(arcStart(previous),startPoint)<=4.0*ARC_TOLERANCE&&distance(arcEnd(previous),endPoint)<=4.0*ARC_TOLERANCE);}
      if(!duplicate){if(required<${capacity}u){arcStore(required,startPoint,endPoint);}required+=1u;}
    }
  }
}
if(required>${capacity}u){failure|=2u;}
var vertexCount=0u;var ringCount=0u;
if(failure==0u){for(var seed=0u;seed<required&&failure==0u;seed++){
  if(arcUsed(seed)){continue;}if(ringCount>=${props.output.ringOffsets.length - 1}u){failure=2u;break;}
  rings[ringsOffset+ringCount]=vertexCount;var current=seed;var closed=false;
  for(var step=0u;step<required;step++){
    if(arcUsed(current)){failure=4u;break;}scratch[scratchOffset+current*7u+6u]=1.0;let point=arcStart(current);output[outputOffset+vertexCount*3u]=point.x;output[outputOffset+vertexCount*3u+1u]=point.y;output[outputOffset+vertexCount*3u+2u]=point.z;vertexCount+=1u;
    let junction=arcEnd(current);let incoming=normalize(junction*dot(arcStart(current),junction)-arcStart(current));
    var next=0xffffffffu;var bestTurn=-4.0;
    for(var candidate=0u;candidate<required;candidate++){if((!arcUsed(candidate)||candidate==seed)&&distance(arcStart(candidate),junction)<=4.0*ARC_TOLERANCE){
      let outgoing=normalize(arcEnd(candidate)-junction*dot(junction,arcEnd(candidate)));let turn=guardedAtan2(dot(junction,cross(incoming,outgoing)),dot(incoming,outgoing));
      if(abs(turn-bestTurn)<=ARC_TOLERANCE){failure=4u;}if(turn>bestTurn){bestTurn=turn;next=candidate;}
    }}
    if(next==seed){closed=true;break;}
    if(next==0xffffffffu||failure!=0u){failure=4u;break;}current=next;
  }
  if(!closed){failure|=4u;}ringCount+=1u;rings[ringsOffset+ringCount]=vertexCount;
}}
if(failure!=0u){vertexCount=0u;ringCount=0u;}
for(var row=vertexCount;row<${capacity}u;row++){output[outputOffset+row*3u]=0.0;output[outputOffset+row*3u+1u]=0.0;output[outputOffset+row*3u+2u]=0.0;}
for(var ring=ringCount+1u;ring<${props.output.ringOffsets.length}u;ring++){rings[ringsOffset+ring]=vertexCount;}rings[ringsOffset]=0u;
counts[countsOffset]=vertexCount;counts[countsOffset+1u]=ringCount;counts[countsOffset+2u]=required;counts[countsOffset+3u]=${Number(this.referenceInside)}u;status[statusOffset]=failure;`
      })
    ];
  }
}
