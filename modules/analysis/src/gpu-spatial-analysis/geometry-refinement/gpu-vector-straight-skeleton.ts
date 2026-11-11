// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUPolygonGeometryFields} from '../../analysis-core/contracts/index';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getReadPointWGSL} from '../shared/spatial-wgsl-point-helpers';

export type GPUVectorStraightSkeletonProps = GPUPolygonGeometryFields & {
  id?: string;
  /** Original vertices per polygon, excluding repeated closing vertices; default 128. */
  maximumVertices?: number;
  /** Edge collapse / reflex split events per polygon; default 128. */
  maximumEvents?: number;
  offsetDistance: number;
  eventTolerance?: number;
  segmentCapacity: number;
  output: {
    segments: GraphDataView<'float32x4'>;
    sourceEdges: GraphDataView<'uint32x2'>;
    /** Fixed polygon blocks of maximumVertices+2*maximumEvents slots. Inactive slots are identified by links. */
    offsetPositions: GraphDataView<'float32x2'>;
    /** Previous/next offset vertex IDs, local to the polygon block; UINT_MAX for unused slots. */
    offsetLinks: GraphDataView<'uint32x2'>;
    /** [writtenArcs, requiredArcs, activeOffsetVertices, flags]: input=1, capacity=2, work=4, numerical=8. */
    status: GraphDataView<'uint32x4'>;
  };
};

/**
 * Event-driven planar wavefront with explicit linked half-edge topology, reflex splits and hole merges.
 * Independent polygons run in parallel. Internal storage reserves 19 words per wavefront slot and
 * one count per polygon, using source-position ranges when smaller than maximumVertices blocks.
 * Unused event slots are initialized only when born. Stable nondegenerate components are measured
 * once per event; uncertain near-zero components retain the original ordered floating-point checks.
 */
export class GPUVectorStraightSkeleton implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUVectorStraightSkeletonProps;
  readonly maximumVertices: number;
  readonly maximumEvents: number;
  constructor(props: GPUVectorStraightSkeletonProps) {
    this.id = props.id ?? 'vector-straight-skeleton';
    this.props = props;
    this.maximumVertices = props.maximumVertices ?? 128;
    this.maximumEvents = props.maximumEvents ?? 128;
    validateCapacity(this.id, 'maximumVertices', this.maximumVertices, 3, 1024);
    validateCapacity(this.id, 'maximumEvents', this.maximumEvents, 0, 4096);
    validateCapacity(this.id, 'segmentCapacity', props.segmentCapacity, 1);
    if (
      !(
        Number.isFinite(props.offsetDistance) &&
        props.offsetDistance >= 0 &&
        Number.isFinite(props.eventTolerance ?? 1e-5) &&
        (props.eventTolerance ?? 1e-5) > 0
      )
    )
      throw new Error('Invalid skeleton event parameters');
    validateSinglePackedView(props.positions, ['float32x2'], this.id);
    for (const view of [props.ringOffsets, props.polygonOffsets, props.featureOffsets])
      validateSinglePackedView(view, ['uint32'], this.id);
    const polygons = props.polygonOffsets.length - 1;
    const slots = this.maximumVertices + 2 * this.maximumEvents;
    for (const [name, format, length] of [
      ['segments', 'float32x4', polygons * props.segmentCapacity],
      ['sourceEdges', 'uint32x2', polygons * props.segmentCapacity],
      ['offsetPositions', 'float32x2', polygons * slots],
      ['offsetLinks', 'uint32x2', polygons * slots],
      ['status', 'uint32x4', polygons]
    ] as const) {
      validateSinglePackedView(props.output[name], [format], this.id);
      if (props.output[name].length !== length) throw new Error('Skeleton output size mismatch');
    }
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      props.ringOffsets,
      props.polygonOffsets,
      props.featureOffsets
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const slots = this.maximumVertices + 2 * this.maximumEvents;
    const polygons = props.polygonOffsets.length - 1;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.ringOffsets,
      props.polygonOffsets,
      props.featureOffsets,
      ...Object.values(props.output)
    ]);
    // Motion/topology (11), original edge plane/source (4), snapshot links (2), visitation (1), ordered active successor (1).
    // Source ranges include closing vertices, so they conservatively bound initial live vertices.
    const packedWorkspace = props.positions.length < polygons * this.maximumVertices;
    const workspaceSlots = packedWorkspace
      ? props.positions.length + polygons * 2 * this.maximumEvents
      : polygons * slots;
    const getWorkspaceBase = (polygon: string) =>
      packedWorkspace
        ? `${polygon} + (rings[ringsOffset + polygons[polygonsOffset + ${polygon}]] + ${polygon} * ${2 * this.maximumEvents}u) * 19u`
        : `${polygon} * ${1 + slots * 19}u`;
    const workspace = createTransientView(
      graph,
      `${id}-wavefront`,
      'uint32',
      polygons + workspaceSlots * 19
    );
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-events`,
        operation: 'GPUVectorStraightSkeleton',
        invocationCount: polygons,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'rings', view: props.ringOffsets, type: 'u32', access: 'read'},
          {name: 'polygons', view: props.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'workspace', view: workspace, type: 'u32', access: 'read_write'},
          {name: 'segments', view: props.output.segments, type: 'f32', access: 'read_write'},
          {name: 'sources', view: props.output.sourceEdges, type: 'u32', access: 'read_write'},
          {name: 'offsets', view: props.output.offsetPositions, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `const SLOTS: u32 = ${slots}u;
const TOLERANCE: f32 = ${getWGSLFloatLiteral(props.eventTolerance ?? 1e-5)};
fn readWord(base:u32, vertex:u32, field:u32)->u32 { return workspace[workspaceOffset+base+1u+vertex*19u+field]; }
fn put(base:u32, vertex:u32, field:u32, value:u32) { workspace[workspaceOffset+base+1u+vertex*19u+field]=value; }
fn scalar(base:u32, vertex:u32, field:u32)->f32 { return bitcast<f32>(readWord(base,vertex,field)); }
fn store(base:u32, vertex:u32, field:u32, value:f32) { put(base,vertex,field,bitcast<u32>(value)); }
${getReadPointWGSL('point', 'positions')}
fn cross(left:vec2f,right:vec2f)->f32 { return left.x*right.y-left.y*right.x; }
fn normal(base:u32,edge:u32)->vec2f { return vec2f(scalar(base,edge,11u),scalar(base,edge,12u)); }
fn at(base:u32,vertex:u32,time:f32)->vec2f { return vec2f(scalar(base,vertex,0u),scalar(base,vertex,1u))+time*vec2f(scalar(base,vertex,2u),scalar(base,vertex,3u)); }
fn initialize(base:u32,vertex:u32,left:u32,right:u32,previous:u32,next:u32,origin:vec2f,time:f32) {
  put(base,vertex,4u,left); put(base,vertex,5u,right); put(base,vertex,6u,previous); put(base,vertex,7u,next); put(base,vertex,8u,1u);
  put(base,vertex,15u,0xffffffffu); put(base,vertex,16u,0xffffffffu); put(base,vertex,17u,0u);
  store(base,vertex,9u,origin.x); store(base,vertex,10u,origin.y);
  let first=normal(base,left); let second=normal(base,right); let determinant=cross(first,second);
  var start=origin; var velocity=vec2f(0.0);
  if (abs(determinant)>1e-7) {
    let firstConstant=scalar(base,left,13u); let secondConstant=scalar(base,right,13u);
    start=vec2f(firstConstant*second.y-first.y*secondConstant,first.x*secondConstant-firstConstant*second.x)/determinant;
    velocity=vec2f(second.y-first.y,first.x-second.x)/determinant;
  }
  else if(dot(first,second)>0.999999) { velocity=first; start=origin-time*velocity; }
  store(base,vertex,0u,start.x); store(base,vertex,1u,start.y); store(base,vertex,2u,velocity.x); store(base,vertex,3u,velocity.y);
}
fn emit(polygon:u32,base:u32,vertex:u32,end:vec2f,required:u32)->u32 {
  let start=vec2f(scalar(base,vertex,9u),scalar(base,vertex,10u));
  if (distance(start,end)<=TOLERANCE) { return required; }
  if (required<${props.segmentCapacity}u) {
    let slot=polygon*${props.segmentCapacity}u+required;
    segments[segmentsOffset+slot*4u]=start.x; segments[segmentsOffset+slot*4u+1u]=start.y; segments[segmentsOffset+slot*4u+2u]=end.x; segments[segmentsOffset+slot*4u+3u]=end.y;
    sources[sourcesOffset+slot*2u]=readWord(base,readWord(base,vertex,4u),14u); sources[sourcesOffset+slot*2u+1u]=readWord(base,readWord(base,vertex,5u),14u);
  }
  return required+1u;
}
fn snapshot(polygon:u32,base:u32,count:u32,time:f32)->u32 {
  var activeCount=0u;
  for(var vertex=0u;vertex<count;vertex++) { if(readWord(base,vertex,8u)!=0u) {
    let location=at(base,vertex,time); let slot=polygon*SLOTS+vertex;
    offsets[offsetsOffset+slot*2u]=location.x; offsets[offsetsOffset+slot*2u+1u]=location.y;
    put(base,vertex,15u,readWord(base,vertex,6u)); put(base,vertex,16u,readWord(base,vertex,7u)); activeCount++;
  } }
  return activeCount;
}`,
        body: `let base=${getWorkspaceBase('index')};
var count=0u; var flags=0u; var required=0u;
let firstRing=polygons[polygonsOffset+index]; let lastRing=polygons[polygonsOffset+index+1u];
for(var ring=firstRing;ring<lastRing;ring++) {
  let first=rings[ringsOffset+ring]; var last=rings[ringsOffset+ring+1u];
  if(last>first && all(point(first)==point(last-1u))) { last--; }
  if(last-first<3u) { flags|=1u; continue; }
  var area=0.0; for(var vertex=first;vertex<last;vertex++) { let next=select(vertex+1u,first,vertex+1u==last); area+=cross(point(vertex),point(next)); }
  let reverse=select((area < 0.0), (area > 0.0), (ring > firstRing));
  let ringStart=count;
  for(var ordinal=0u;ordinal<last-first;ordinal++) {
    let vertex=first+select(ordinal,last-first-1u-ordinal,reverse);
    let before=first+(vertex-first+last-first-1u)%(last-first); let after=first+(vertex-first+1u)%(last-first);
    if(all(point(vertex)==point(before)) || all(point(vertex)==point(after))) { flags|=1u; continue; }
    if(count>=${this.maximumVertices}u) { flags|=1u; break; }
    let location=point(vertex); if(any((bitcast<vec2u>(location)&vec2u(0x7f800000u))==vec2u(0x7f800000u))) { flags|=1u; }
    store(base,count,9u,location.x); store(base,count,10u,location.y); put(base,count,14u,vertex); count++;
  }
  if(count-ringStart<3u) { flags|=1u; continue; }
  for(var vertex=ringStart;vertex<count;vertex++) {
    let next=select(vertex+1u,ringStart,vertex+1u==count); let previous=select(vertex-1u,count-1u,vertex==ringStart);
    let start=vec2f(scalar(base,vertex,9u),scalar(base,vertex,10u)); let finish=vec2f(scalar(base,next,9u),scalar(base,next,10u)); let edge=finish-start;
    let inward=vec2f(-edge.y,edge.x)/length(edge); store(base,vertex,11u,inward.x); store(base,vertex,12u,inward.y); store(base,vertex,13u,dot(inward,start)); put(base,vertex,6u,previous); put(base,vertex,7u,next);
  }
}
for(var vertex=0u;vertex<count;vertex++) { initialize(base,vertex,readWord(base,vertex,6u),vertex,readWord(base,vertex,6u),readWord(base,vertex,7u),vec2f(scalar(base,vertex,9u),scalar(base,vertex,10u)),0.0); }
var time=0.0; var offsetCount=0u; var captured=false; var eventCount=0u;
loop {
  if(flags!=0u) { break; }
  // Preserve ascending slot order while skipping every historical inactive vertex in searches.
  var firstActive=count;
  for(var remaining=count;remaining>0u;remaining--) { let vertex=remaining-1u; if(readWord(base,vertex,8u)!=0u) { put(base,vertex,18u,firstActive); firstActive=vertex; } }
  // A terminal two-vertex component is a ridge, including simultaneous opposing edge collapses.
  for(var vertex=firstActive;vertex<count;vertex=readWord(base,vertex,18u)) { if(readWord(base,vertex,8u)!=0u) {
    let next=readWord(base,vertex,7u);
    if(next==vertex) { put(base,vertex,8u,0u); }
    else if(readWord(base,next,7u)==vertex) {
      required=emit(index,base,vertex,vec2f(scalar(base,next,9u),scalar(base,next,10u)),required); put(base,vertex,8u,0u); put(base,next,8u,0u);
    }
  } }
  // Simultaneous parallel collisions leave zero-area wavefront components. Their
  // elementary undirected edges are terminal ridges; collinear vertices split them.
  let visitGeneration=eventCount+1u;
  for(var root=firstActive;root<count;root=readWord(base,root,18u)) { if(readWord(base,root,8u)==0u || readWord(base,root,17u)==visitGeneration) { continue; }
    var vertex=root; var twiceArea=0.0; var areaMagnitude=0.0; var perimeter=0.0; var traversed=0u;
    loop { let next=readWord(base,vertex,7u); let start=at(base,vertex,time); let finish=at(base,next,time); let areaTerm=cross(start,finish); twiceArea+=areaTerm; areaMagnitude+=abs(areaTerm); perimeter+=distance(start,finish); put(base,vertex,17u,visitGeneration); vertex=next; traversed++; if(vertex==root || traversed>count) { break; } }
    if(traversed>count) { flags|=8u; break; }
    if(abs(twiceArea)>TOLERANCE*(1.0+perimeter)) {
      // Every rotated sum uses the same terms. A conservative rounding envelope preserves
      // the original per-root checks when their order could change the tolerance decision.
      let roundingEnvelope=8.0*f32(traversed)*1.1920928955078125e-7*(areaMagnitude+TOLERANCE*(1.0+perimeter));
      if (!(abs(twiceArea)>TOLERANCE*(1.0+perimeter)+roundingEnvelope)) {
        vertex=root; loop { put(base,vertex,17u,0u); vertex=readWord(base,vertex,7u); if(vertex==root) { break; } }
      }
      continue;
    }
    vertex=root;
    loop { required=emit(index,base,vertex,at(base,vertex,time),required); vertex=readWord(base,vertex,7u); if(vertex==root) { break; } }
    vertex=root;
    loop {
      let next=readWord(base,vertex,7u); let start=at(base,vertex,time); let finish=at(base,next,time); let direction=finish-start; let lengthSquared=dot(direction,direction); var keep=lengthSquared>TOLERANCE*TOLERANCE;
      var other=root;
      loop {
        let otherNext=readWord(base,other,7u); let otherStart=at(base,other,time); let otherFinish=at(base,otherNext,time);
        let fraction=dot(otherStart-start,direction)/max(lengthSquared,1e-30);
        if(fraction>TOLERANCE && fraction<1.0-TOLERANCE && abs(cross(otherStart-start,direction))<=TOLERANCE*sqrt(lengthSquared)) { keep=false; }
        if(other<vertex && ((distance(start,otherStart)<=TOLERANCE && distance(finish,otherFinish)<=TOLERANCE) || (distance(start,otherFinish)<=TOLERANCE && distance(finish,otherStart)<=TOLERANCE))) { keep=false; }
        other=otherNext; if(other==root) { break; }
      }
      if(keep) { store(base,vertex,9u,start.x); store(base,vertex,10u,start.y); required=emit(index,base,vertex,finish,required); }
      vertex=next; if(vertex==root) { break; }
    }
    vertex=root; loop { put(base,vertex,8u,0u); vertex=readWord(base,vertex,7u); if(vertex==root) { break; } }
  }
  var activeCount=0u; for(var vertex=firstActive;vertex<count;vertex=readWord(base,vertex,18u)) { activeCount+=readWord(base,vertex,8u); }
  if(activeCount==0u) { break; }
  var eventTime=3.0e38; var eventVertex=0xffffffffu; var eventEdge=0xffffffffu; var split=false;
  for(var vertex=firstActive;vertex<count;vertex=readWord(base,vertex,18u)) { if(readWord(base,vertex,8u)==0u) { continue; }
    let next=readWord(base,vertex,7u); let delta=at(base,next,0.0)-at(base,vertex,0.0);
    let velocity=vec2f(scalar(base,next,2u)-scalar(base,vertex,2u),scalar(base,next,3u)-scalar(base,vertex,3u)); let denominator=dot(velocity,velocity);
    if(denominator>1e-12) {
      let candidate=-dot(delta,velocity)/denominator;
      if(candidate>=time-TOLERANCE && candidate<eventTime && length(delta+candidate*velocity)<=TOLERANCE*(1.0+length(at(base,vertex,candidate)))) { eventTime=max(time,candidate); eventVertex=vertex; eventEdge=next; split=false; }
    }
  }
  for(var vertex=firstActive;vertex<count;vertex=readWord(base,vertex,18u)) { if(readWord(base,vertex,8u)==0u || cross(normal(base,readWord(base,vertex,4u)),normal(base,readWord(base,vertex,5u)))>=-1e-7) { continue; }
    let origin=at(base,vertex,0.0); let velocity=vec2f(scalar(base,vertex,2u),scalar(base,vertex,3u));
    for(var edge=firstActive;edge<count;edge=readWord(base,edge,18u)) { if(readWord(base,edge,8u)==0u) { continue; } let next=readWord(base,edge,7u);
      if(edge==vertex || next==vertex) { continue; }
      let edgeId=readWord(base,edge,5u); let inward=normal(base,edgeId); let rate=dot(inward,velocity)-1.0;
      if(rate>=-1e-7) { continue; }
      let candidate=(scalar(base,edgeId,13u)-dot(inward,origin))/rate;
      if(candidate<time-TOLERANCE || candidate>=eventTime-TOLERANCE) { continue; }
      let location=origin+candidate*velocity; let start=at(base,edge,candidate); let finish=at(base,next,candidate); let direction=finish-start;
      let fraction=dot(location-start,direction)/dot(direction,direction);
      if(fraction>=-TOLERANCE && fraction<=1.0+TOLERANCE) { eventTime=max(time,candidate); eventVertex=vertex; eventEdge=edge; split=true; }
    }
  }
  if(!captured && ${getWGSLFloatLiteral(props.offsetDistance)}<=eventTime) { offsetCount=snapshot(index,base,count,${getWGSLFloatLiteral(props.offsetDistance)}); captured=true; }
  if(eventVertex==0xffffffffu) { flags|=8u; break; }
  if(eventCount>=${this.maximumEvents}u || count+select(1u,2u,split)>SLOTS) { flags|=4u; break; }
  let location=at(base,eventVertex,eventTime); let previous=readWord(base,eventVertex,6u); let next=readWord(base,eventVertex,7u);
  required=emit(index,base,eventVertex,location,required); put(base,eventVertex,8u,0u);
  if(split) {
    let edgeNext=readWord(base,eventEdge,7u); let edgeId=readWord(base,eventEdge,5u);
    initialize(base,count,readWord(base,eventVertex,4u),edgeId,previous,edgeNext,location,eventTime);
    initialize(base,count+1u,edgeId,readWord(base,eventVertex,5u),eventEdge,next,location,eventTime);
    put(base,previous,7u,count); put(base,edgeNext,6u,count); put(base,eventEdge,7u,count+1u); put(base,next,6u,count+1u); count+=2u;
  } else {
    let after=readWord(base,eventEdge,7u); required=emit(index,base,eventEdge,location,required); put(base,eventEdge,8u,0u);
    initialize(base,count,readWord(base,eventVertex,4u),readWord(base,eventEdge,5u),previous,after,location,eventTime);
    put(base,previous,7u,count); put(base,after,6u,count); count++;
  }
  time=eventTime; eventCount++;
}
workspace[workspaceOffset+base]=count;
if(required>${props.segmentCapacity}u) { flags|=2u; }
status[statusOffset+index*4u]=min(required,${props.segmentCapacity}u); status[statusOffset+index*4u+1u]=required; status[statusOffset+index*4u+2u]=offsetCount; status[statusOffset+index*4u+3u]=flags;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-offset-links`,
        operation: 'GPUVectorStraightSkeleton',
        invocationCount: polygons * slots,
        bindings: [
          {name: 'workspace', view: workspace, type: 'u32', access: 'read'},
          {name: 'rings', view: props.ringOffsets, type: 'u32', access: 'read'},
          {name: 'polygons', view: props.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'links', view: props.output.offsetLinks, type: 'u32', access: 'read_write'}
        ],
        body: `let polygon=index/${slots}u; let vertex=index%${slots}u;
let base=${getWorkspaceBase('polygon')};
var previous=0xffffffffu; var next=0xffffffffu;
if(vertex<workspace[workspaceOffset+base]) { previous=workspace[workspaceOffset+base+1u+vertex*19u+15u]; next=workspace[workspaceOffset+base+1u+vertex*19u+16u]; }
links[linksOffset+index*2u]=previous; links[linksOffset+index*2u+1u]=next;`
      })
    ];
  }
}
