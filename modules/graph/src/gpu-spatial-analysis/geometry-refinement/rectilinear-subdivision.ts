// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma-contributors/graph/gpu-core';
import {GPUScan} from '@luma-contributors/graph/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {createZeroPiecesNode, GPUSubdividePolygons} from './gpu-subdivide-polygons';

/** Axis-aligned clipping of hole-aware convex seeds. Every accepted leaf is a simple ring. */
export function getRectilinearSubdivisionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  producer: GPUSubdividePolygons
): readonly GPUCommandNode<Parameters>[] {
  const {props, id, maximumVertices} = producer;
  const seedCapacity = Math.max(1, props.positions.length + 2 * (props.ringOffsets.length - 1));
  const seeds = {
    positions: createTransientView(graph, `${id}-seed-positions`, 'float32x2', seedCapacity * 128),
    pieces: createTransientView(graph, `${id}-seed-pieces`, 'uint32x4', seedCapacity),
    status: createTransientView(graph, `${id}-seed-status`, 'uint32x4', 1)
  };
  // Each queue thread owns one 128 x 260 float split queue and handles seeds strided by the thread
  // count, so queue memory stays bounded (about 33 MiB) however many seeds there are.
  const queueThreadCount = Math.min(seedCapacity, RECTILINEAR_QUEUE_THREADS);
  const queue = createTransientView(
    graph,
    `${id}-split-queue`,
    'float32',
    128 * 260 * queueThreadCount
  );
  const leafCounts = createTransientView(graph, `${id}-leaf-counts`, 'uint32', seedCapacity);
  const leafOffsets = createTransientView(graph, `${id}-leaf-offsets`, 'uint32', seedCapacity);
  const exhaustedFlag = createTransientView(graph, `${id}-exhausted`, 'uint32', 1);
  // The splitting of a seed is independent of every other seed, so it runs twice: a `count` pass
  // fixes the leaf count of each seed, a scan numbers the leaves in seed order, and a `write` pass
  // repeats the splitting and stores each leaf at its number (the serial algorithm's order).
  const getSplitPass = (pass: 'count' | 'write') =>
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-rectilinear-${pass}`,
      operation: 'GPUSubdividePolygons',
      variant: `rectilinear-queue-${pass}`,
      invocationCount: queueThreadCount,
      bindings: [
        {name: 'inputPositions', view: seeds.positions, type: 'f32', access: 'read'},
        {name: 'inputPieces', view: seeds.pieces, type: 'u32', access: 'read'},
        {name: 'inputStatus', view: seeds.status, type: 'u32', access: 'read'},
        {name: 'queue', view: queue, type: 'f32', access: 'read_write'},
        ...(pass === 'count'
          ? [
              {
                name: 'leafCounts',
                view: leafCounts,
                type: 'u32' as const,
                access: 'read_write' as const
              },
              {
                name: 'exhaustedFlag',
                view: exhaustedFlag,
                type: 'atomic<u32>' as const,
                access: 'read_write' as const
              }
            ]
          : [
              {
                name: 'leafOffsets',
                view: leafOffsets,
                type: 'u32' as const,
                access: 'read' as const
              },
              {
                name: 'positions',
                view: props.output.positions,
                type: 'f32' as const,
                access: 'read_write' as const
              },
              {
                name: 'pieces',
                view: props.output.pieces,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ])
      ],
      declarations: `const LIMIT:u32=${maximumVertices}u;
const CAPACITY:u32=${props.output.pieces.length}u;
const THREADS:u32=${queueThreadCount}u;
const QUEUE_BASE:u32=${128 * 260}u;
fn queuePoint(slot:u32,vertex:u32,thread:u32)->vec2f {let base=queueOffset+thread*QUEUE_BASE+slot*260u+vertex*2u;return vec2f(queue[base],queue[base+1u]);}
fn storePoint(slot:u32,vertex:u32,thread:u32,point:vec2f) {let base=queueOffset+thread*QUEUE_BASE+slot*260u+vertex*2u;queue[base]=point.x;queue[base+1u]=point.y;}`,
      body: getRectilinearSplitBody(pass, props.maximumSplits ?? 4096)
    });
  return [
    ...new GPUSubdividePolygons({
      ...props,
      id: `${id}-seeds`,
      method: 'convex',
      maximumVertices: 128,
      output: seeds
    }).getCommandNodes(graph),
    createZeroPiecesNode(graph, id, props.output.pieces),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-clear-counts`,
      operation: 'GPUSubdividePolygons',
      variant: 'rectilinear-clear',
      invocationCount: seedCapacity,
      bindings: [
        {name: 'leafCounts', view: leafCounts, type: 'u32', access: 'read_write'},
        {name: 'exhaustedFlag', view: exhaustedFlag, type: 'u32', access: 'read_write'}
      ],
      body: 'leafCounts[leafCountsOffset + index] = 0u; if (index == 0u) { exhaustedFlag[exhaustedFlagOffset] = 0u; }'
    }),
    getSplitPass('count'),
    ...new GPUScan({
      id: `${id}-leaf-scan`,
      input: leafCounts,
      output: leafOffsets,
      mode: 'exclusive'
    }).getCommandNodes(graph),
    getSplitPass('write'),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-rectilinear-status`,
      operation: 'GPUSubdividePolygons',
      variant: 'rectilinear-status',
      invocationCount: 1,
      bindings: [
        {name: 'leafCounts', view: leafCounts, type: 'u32', access: 'read'},
        {name: 'leafOffsets', view: leafOffsets, type: 'u32', access: 'read'},
        {name: 'exhaustedFlag', view: exhaustedFlag, type: 'u32', access: 'read'},
        {name: 'inputStatus', view: seeds.status, type: 'u32', access: 'read'},
        {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
      ],
      declarations: `const CAPACITY:u32=${props.output.pieces.length}u;`,
      body: `let required=leafOffsets[leafOffsetsOffset+${seedCapacity - 1}u]+leafCounts[leafCountsOffset+${seedCapacity - 1}u];
status[statusOffset]=min(required,CAPACITY);status[statusOffset+1u]=required;
status[statusOffset+2u]=inputStatus[inputStatusOffset+2u]|select(0u,1u,required>CAPACITY);
status[statusOffset+3u]=inputStatus[inputStatusOffset+3u]|select(0u,0x80000000u,exhaustedFlag[exhaustedFlagOffset]!=0u);`
    })
  ];
}

/** Queue threads of the rectilinear splitter: one 133 KB queue each. */
const RECTILINEAR_QUEUE_THREADS = 256;

/**
 * Splitting of every seed handled by thread `index` (seeds `index`, `index + THREADS`, ...). The
 * `count` pass records the leaf count of each seed; the `write` pass stores leaf `n` of seed `s`
 * at piece `leafOffsets[s] + n`.
 */
function getRectilinearSplitBody(pass: 'count' | 'write', maximumSplits: number): string {
  return `for(var seed=index;seed<inputStatus[inputStatusOffset];seed+=THREADS) {
 var leaves=0u;
 let initialCount=inputPieces[inputPiecesOffset+seed*4u];
 for(var vertex=0u;vertex<initialCount;vertex++) {let base=inputPositionsOffset+(seed*128u+vertex)*2u;storePoint(0u,vertex,index,vec2f(inputPositions[base],inputPositions[base+1u]));}
 queue[queueOffset+index*QUEUE_BASE+256u]=f32(initialCount);var pending=1u;var work=0u;var exhausted=false;
 loop {
  if(pending==0u) {break;}
  if(work>=${maximumSplits}u) {exhausted=true;break;} work++;pending--;
  let count=u32(queue[queueOffset+index*QUEUE_BASE+pending*260u+256u]);
  var polygon:array<vec2f,128>;var lower=vec2f(3e38);var upper=vec2f(-3e38);
  for(var vertex=0u;vertex<count;vertex++) {let point=queuePoint(pending,vertex,index);polygon[vertex]=point;lower=min(lower,point);upper=max(upper,point);}
  if(count<=LIMIT) {
   ${
     pass === 'write'
       ? `let required=leafOffsets[leafOffsetsOffset+seed]+leaves;
   if(required<CAPACITY) {
    pieces[piecesOffset+required*4u]=count;
    for(var column=1u;column<4u;column++) {pieces[piecesOffset+required*4u+column]=inputPieces[inputPiecesOffset+seed*4u+column];}
    for(var vertex=0u;vertex<count;vertex++) {let base=positionsOffset+(required*LIMIT+vertex)*2u;positions[base]=polygon[vertex].x;positions[base+1u]=polygon[vertex].y;}
   }`
       : ''
   } leaves++;continue;
  }
  let axis=select(1u,0u,upper.x-lower.x>=upper.y-lower.y);let middle=lower[axis]+(upper[axis]-lower[axis])*0.5;
  if(!(middle>lower[axis] && middle<upper[axis]) || pending+2u>128u) {exhausted=true;break;}
  for(var half=0u;half<2u;half++) {
   var clipped:array<vec2f,130>;var clippedCount=0u;
   for(var vertex=0u;vertex<count;vertex++) {
    let first=polygon[vertex];let last=polygon[(vertex+1u)%count];let firstInside=select(first[axis]<=middle,first[axis]>=middle,half==1u);let lastInside=select(last[axis]<=middle,last[axis]>=middle,half==1u);
    if(firstInside) {clipped[clippedCount]=first;clippedCount++;}
    if(firstInside!=lastInside) {let fraction=(middle-first[axis])/(last[axis]-first[axis]);if(fraction>0.0 && fraction<1.0) {var crossing=first+fraction*(last-first);crossing[axis]=middle;clipped[clippedCount]=crossing;clippedCount++;}}
   }
   // Exact collinear vertices carry no corner information and need not consume the vertex limit.
   var compact:array<vec2f,130>;var compactCount=0u;
   for(var vertex=0u;vertex<clippedCount;vertex++) {
    let previous=clipped[(vertex+clippedCount-1u)%clippedCount];let current=clipped[vertex];let next=clipped[(vertex+1u)%clippedCount];let incoming=current-previous;let outgoing=next-current;
    if(all(current==previous) || incoming.x*outgoing.y-incoming.y*outgoing.x==0.0) {continue;}
    compact[compactCount]=current;compactCount++;
   }
   if(compactCount<3u) {continue;}
   if(compactCount>128u) {exhausted=true;break;}
   for(var vertex=0u;vertex<compactCount;vertex++) {storePoint(pending,vertex,index,compact[vertex]);}
   queue[queueOffset+index*QUEUE_BASE+pending*260u+256u]=f32(compactCount);pending++;
  }
 }
 ${pass === 'count' ? 'leafCounts[leafCountsOffset+seed]=leaves;if(exhausted) {atomicStore(&exhaustedFlag[exhaustedFlagOffset],1u);}' : ''}
}`;
}
