// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getRingQuantizerWGSL} from '../../analysis-core/ring-assembly/ring-vertex-quantizer';

/** Direct contacts are incidences, never transitive vertex identities. */
export function createPolygonizeContactPeelingNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    endpoints: GraphDataView<'float32x4'>;
    groups: GraphDataView<'uint32'>;
    count: GraphDataView<'uint32'>;
    frame: GraphDataView<'uint32'>;
    order: GraphDataView<'uint32'>;
    owners: GraphDataView<'uint32'>;
    state: GraphDataView<'uint32'>;
    removed: GraphDataView<'uint32'>;
    queue: GraphDataView<'uint32'>;
    vertexTolerance: number;
  }
): GPUCommandNode<Parameters>[] {
  const {
    id,
    endpoints,
    groups,
    count,
    frame,
    order,
    owners,
    state,
    removed,
    queue,
    vertexTolerance
  } = props;
  const endpointCapacity = endpoints.length;
  const segmentCapacity = endpointCapacity / 2;
  const declarations = `${getRingQuantizerWGSL(getWGSLFloatLiteral(vertexTolerance), true, 0)}
const CONTACT_TOLERANCE:f32=${getWGSLFloatLiteral(vertexTolerance)};
fn contactPoint(row:u32)->vec2f{return vec2f(endpoints[endpointsOffset+4u*row],endpoints[endpointsOffset+4u*row+1u]);}
fn beforeContactCell(row:u32,group:u32,cellX:u32,cellY:u32)->bool{
 let origin=ringOrigin();let point=contactPoint(row);let x=ringQuantize(point.x,origin.x);let y=ringQuantize(point.y,origin.y);let otherGroup=groups[groupsOffset+row];
 return otherGroup<group || (otherGroup==group && (x<cellX || (x==cellX && y<cellY)));
}
fn firstContactCell(group:u32,cellX:u32,cellY:u32)->u32{
 var low=0u;var high=atomicLoad(&counters[countersOffset+1u]);
 while(low<high){let middle=low+(high-low)/2u;if(beforeContactCell(order[orderOffset+middle],group,cellX,cellY)){low=middle+1u;}else{high=middle;}}
 return low;
}
fn inContactCell(row:u32,group:u32,cellX:u32,cellY:u32)->bool{
 let origin=ringOrigin();let point=contactPoint(row);
 return groups[groupsOffset+row]==group && ringQuantize(point.x,origin.x)==cellX && ringQuantize(point.y,origin.y)==cellY;
}`;
  const common = [
    {name: 'endpoints', view: endpoints, type: 'f32', access: 'read'},
    {name: 'groups', view: groups, type: 'u32', access: 'read'},
    {name: 'order', view: order, type: 'u32', access: 'read'},
    {name: 'owners', view: owners, type: 'u32', access: 'read'},
    {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'},
    {name: 'removed', view: removed, type: 'atomic<u32>', access: 'read_write'},
    {name: 'counters', view: queue, type: 'atomic<u32>', access: 'read_write'}
  ] as const;
  const enqueue = `fn enqueueContactEdge(edge:u32){
 let previous=atomicOr(&removed[removedOffset+edge],2u);
 if(previous==0u){let slot=atomicAdd(&counters[countersOffset+3u],1u);atomicStore(&counters[countersOffset+4u+slot],edge);}
}`;
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-contact-queue-clear`,
      operation: 'GPUPolygonize',
      invocationCount: 1,
      bindings: [
        {name: 'frame', view: frame, type: 'u32', access: 'read'},
        {name: 'count', view: count, type: 'u32', access: 'read'},
        {name: 'queue', view: queue, type: 'u32', access: 'read_write'}
      ],
      body: 'queue[queueOffset]=frame[frameOffset];queue[queueOffset+1u]=count[countOffset];queue[queueOffset+2u]=0u;queue[queueOffset+3u]=0u;'
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-contact-counts`,
      operation: 'GPUPolygonize',
      invocationCount: endpointCapacity,
      bindings: common,
      declarations,
      body: `var neighbors=0u;
 let total=atomicLoad(&counters[countersOffset+1u]);
 if(index<total && owners[ownersOffset+index/2u]==index/2u && atomicLoad(&removed[removedOffset+index/2u])==0u){
  let position=contactPoint(index);let origin=ringOrigin();let group=groups[groupsOffset+index];let ownX=ringQuantize(position.x,origin.x);let ownY=ringQuantize(position.y,origin.y);
  for(var deltaY=-1;deltaY<=1;deltaY++){for(var deltaX=-1;deltaX<=1;deltaX++){
   let cellX=u32(i32(ownX)+deltaX);let cellY=u32(i32(ownY)+deltaY);var slot=firstContactCell(group,cellX,cellY);
   while(slot<total){let candidate=order[orderOffset+slot];if(!inContactCell(candidate,group,cellX,cellY)){break;}
    let edge=candidate/2u;
    if(edge!=index/2u && owners[ownersOffset+edge]==edge && atomicLoad(&removed[removedOffset+edge])==0u && all(abs(contactPoint(candidate)-position)<=vec2f(CONTACT_TOLERANCE))){neighbors++;}
    slot++;
   }
  }}
 }
 atomicStore(&state[stateOffset+index],neighbors);`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-contact-seed`,
      operation: 'GPUPolygonize',
      invocationCount: segmentCapacity,
      bindings: [
        {name: 'owners', view: owners, type: 'u32', access: 'read'},
        {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'},
        {name: 'removed', view: removed, type: 'atomic<u32>', access: 'read_write'},
        {name: 'counters', view: queue, type: 'atomic<u32>', access: 'read_write'}
      ],
      declarations: enqueue,
      body: `if(owners[ownersOffset+index]!=index || atomicLoad(&removed[removedOffset+index])!=0u){return;}
 if(atomicLoad(&state[stateOffset+2u*index])==0u || atomicLoad(&state[stateOffset+2u*index+1u])==0u){enqueueContactEdge(index);}`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-peel-direct`,
      operation: 'GPUPolygonize',
      invocationCount: 64,
      workgroupSize: 64,
      guardIndex: false,
      bindings: common,
      declarations: `${declarations}\n${enqueue}
const CONTACT_LANE_COUNT:u32=64u;
var<workgroup> directPeelEdge:u32;`,
      body: `loop {
 if(localInvocationIndex==0u){
  let head=atomicLoad(&counters[countersOffset+2u]);let tail=atomicLoad(&counters[countersOffset+3u]);
  directPeelEdge=0xffffffffu;
  if(head<tail){let edge=atomicLoad(&counters[countersOffset+4u+head]);atomicStore(&counters[countersOffset+2u],head+1u);atomicStore(&removed[removedOffset+edge],1u);directPeelEdge=edge;}
 }
 storageBarrier();
 let edge=workgroupUniformLoad(&directPeelEdge);if(edge==0xffffffffu){break;}
 let total=atomicLoad(&counters[countersOffset+1u]);let origin=ringOrigin();
 for(var side=0u;side<2u;side++){
  let endpoint=2u*edge+side;let position=contactPoint(endpoint);let group=groups[groupsOffset+endpoint];let ownX=ringQuantize(position.x,origin.x);let ownY=ringQuantize(position.y,origin.y);
  for(var deltaY=-1;deltaY<=1;deltaY++){for(var deltaX=-1;deltaX<=1;deltaX++){
   let cellX=u32(i32(ownX)+deltaX);let cellY=u32(i32(ownY)+deltaY);var slot=firstContactCell(group,cellX,cellY)+localInvocationIndex;
   while(slot<total){let candidate=order[orderOffset+slot];if(!inContactCell(candidate,group,cellX,cellY)){break;}
    let otherEdge=candidate/2u;
    if(otherEdge!=edge && owners[ownersOffset+otherEdge]==otherEdge && atomicLoad(&removed[removedOffset+otherEdge])!=1u && all(abs(contactPoint(candidate)-position)<=vec2f(CONTACT_TOLERANCE))){
     let previous=atomicSub(&state[stateOffset+candidate],1u);if(previous==1u){enqueueContactEdge(otherEdge);}
    }
    slot+=CONTACT_LANE_COUNT;
   }
  }}
 }
 storageBarrier();workgroupBarrier();
}`
    })
  ];
}
