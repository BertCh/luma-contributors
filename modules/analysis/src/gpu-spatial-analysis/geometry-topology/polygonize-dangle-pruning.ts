// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createStableLexicographicIndexSortNodes} from '../../analysis-core/kernels/stable-lexicographic-index-sort';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelProps
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createPolygonizeContactPeelingNodes} from './polygonize-contact-peeling';
import {getRingQuantizerWGSL} from '../../analysis-core/ring-assembly/ring-vertex-quantizer';

/**
 * Peel the undirected degree-one fringe before left-face traversal. Source/directed
 * rows remain intact; peeled rows become zero-length only in the assembly input.
 * Duplicate source edges share one degree contribution and one peeling decision.
 */
export function createPolygonizeDanglePruningNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    endpoints: GraphDataView<'float32x4'>;
    groups: GraphDataView<'uint32'>;
    count: GraphDataView<'uint32'>;
    vertexTolerance: number;
    /** Internal diagnostic for bounded spatial lookup tests, not a public contributor output. */
    candidateCounts?: GraphDataView<'uint32'>;
  }
): {nodes: GPUCommandNode<Parameters>[]; endpoints: GraphDataView<'float32x4'>} {
  const {id, endpoints, groups, count, vertexTolerance, candidateCounts} = props;
  const endpointCapacity = endpoints.length;
  const segmentCapacity = endpointCapacity / 2;
  const transient = <Format extends 'uint32' | 'float32x4'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `${id}-${name}`, format, length);
  const keysX = transient('endpoint-keys-x', 'uint32', endpointCapacity);
  const keysY = transient('endpoint-keys-y', 'uint32', endpointCapacity);
  const frame = transient('frame', 'uint32', 1);
  const endpointOrder = transient('endpoint-order', 'uint32', endpointCapacity);
  const vertexRows = transient('vertex-rows', 'uint32', endpointCapacity);
  const firstRows = transient('first-rows', 'uint32', segmentCapacity);
  const lastRows = transient('last-rows', 'uint32', segmentCapacity);
  const edgeOrder = transient('edge-order', 'uint32', segmentCapacity);
  const owners = transient('owners', 'uint32', segmentCapacity);
  // Degree followed by the modular u32 sum of incident (owner segment row + 1).
  // Removing a row subtracts its term; at degree one the residual is exactly
  // the surviving row + 1, even if a high-degree vertex's original sum wrapped.
  const state = transient('vertex-state', 'uint32', 2 * endpointCapacity);
  const queue = transient('leaf-queue', 'uint32', Math.max(endpointCapacity, segmentCapacity + 4));
  const removed = transient('removed', 'uint32', segmentCapacity);
  const faceEndpoints = transient('face-endpoints', 'float32x4', endpointCapacity);
  const kernel = (
    name: string,
    configuration: Omit<WGSLKernelProps<Parameters>, 'id' | 'operation'>
  ) =>
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-${name}`,
      operation: 'GPUPolygonize',
      ...configuration
    });
  const sortedEndpoints = createStableLexicographicIndexSortNodes(graph, {
    id: `${id}-endpoint-sort`,
    operation: 'GPUPolygonize',
    algorithm: 'radix',
    indices: endpointOrder,
    indicesAreIdentity: true,
    keys: [
      {view: groups, keyBits: 32},
      {view: keysX, keyBits: 32},
      {view: keysY, keyBits: 32}
    ]
  });
  const sortedEdges = createStableLexicographicIndexSortNodes(graph, {
    id: `${id}-edge-sort`,
    operation: 'GPUPolygonize',
    algorithm: 'radix',
    indices: edgeOrder,
    indicesAreIdentity: true,
    keys: [
      {view: firstRows, keyBits: 32},
      {view: lastRows, keyBits: 32}
    ]
  });
  const nodes: GPUCommandNode<Parameters>[] = [
    // Polygonize supplies its tolerance, so the shared quantizer has no derived extent.
    kernel('frame-clear', {
      invocationCount: 1,
      bindings: [{name: 'counters', view: frame, type: 'u32', access: 'read_write'}],
      body: 'counters[countersOffset]=0u;'
    }),
    kernel('endpoint-keys', {
      invocationCount: endpointCapacity,
      bindings: [
        {name: 'endpoints', view: endpoints, type: 'f32', access: 'read'},
        {name: 'count', view: count, type: 'u32', access: 'read'},
        {name: 'counters', view: frame, type: 'u32', access: 'read'},
        {name: 'keysX', view: keysX, type: 'u32', access: 'read_write'},
        {name: 'keysY', view: keysY, type: 'u32', access: 'read_write'},
        {name: 'order', view: endpointOrder, type: 'u32', access: 'read_write'},
        {name: 'parents', view: vertexRows, type: 'u32', access: 'read_write'}
      ],
      declarations: getRingQuantizerWGSL(getWGSLFloatLiteral(vertexTolerance), false, 0),
      body: `order[orderOffset+index]=index;
  var cellX=0xffffffffu;var cellY=0xffffffffu;var parent=0xffffffffu;
  if(index<count[countOffset]) {
    cellX=ringQuantize(endpoints[endpointsOffset+index*4u],0.0);
    cellY=ringQuantize(endpoints[endpointsOffset+index*4u+1u],0.0);
    parent=index;
  }
  keysX[keysXOffset+index]=cellX;keysY[keysYOffset+index]=cellY;
  parents[parentsOffset+index]=parent;`
    }),
    ...sortedEndpoints.nodes,
    kernel('vertices', {
      invocationCount: endpointCapacity,
      bindings: [
        {name: 'endpoints', view: endpoints, type: 'f32', access: 'read'},
        {name: 'groups', view: groups, type: 'u32', access: 'read'},
        {name: 'count', view: count, type: 'u32', access: 'read'},
        {name: 'keysX', view: keysX, type: 'u32', access: 'read'},
        {name: 'keysY', view: keysY, type: 'u32', access: 'read'},
        {name: 'order', view: sortedEndpoints.sortedIndices, type: 'u32', access: 'read'},
        {name: 'parents', view: vertexRows, type: 'atomic<u32>', access: 'read_write'},
        ...(candidateCounts
          ? [
              {
                name: 'candidateCounts',
                view: candidateCounts,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      declarations: `const TOLERANCE:f32=${getWGSLFloatLiteral(vertexTolerance)};
fn point(row:u32)->vec2f {return vec2f(endpoints[endpointsOffset+row*4u],endpoints[endpointsOffset+row*4u+1u]);}
fn findRoot(vertex:u32)->u32 {
  var root=vertex;var parent=atomicLoad(&parents[parentsOffset+root]);
  while(parent!=root) {root=parent;parent=atomicLoad(&parents[parentsOffset+root]);}
  return root;
}
fn joinVertices(first:u32,second:u32) {
  var left=first;var right=second;
  loop {
    left=findRoot(left);right=findRoot(right);
    if(left==right) {return;}
    let upper=max(left,right);let lower=min(left,right);
    let previous=atomicMin(&parents[parentsOffset+upper],lower);
    if(previous==upper) {return;}
    left=previous;right=lower;
  }
}
fn beforeCell(row:u32,group:u32,cellX:u32,cellY:u32)->bool {
  let candidateGroup=groups[groupsOffset+row];
  let candidateX=keysX[keysXOffset+row];let candidateY=keysY[keysYOffset+row];
  return candidateGroup<group || (candidateGroup==group && (candidateX<cellX || (candidateX==cellX && candidateY<cellY)));
}`,
      body: `let total=count[countOffset];
  var visits=0u;
  if(index<total) {
    let position=point(index);let group=groups[groupsOffset+index];
    let ownX=keysX[keysXOffset+index];let ownY=keysY[keysYOffset+index];
    // Matching vertices lie in the same or adjacent cells on each axis.
    for(var deltaY=-1;deltaY<=1;deltaY++) {
      for(var deltaX=-1;deltaX<=1;deltaX++) {
        let cellX=u32(i32(ownX)+deltaX);let cellY=u32(i32(ownY)+deltaY);
        var low=0u;var high=total;
        while(low<high) {
          let middle=low+(high-low)/2u;
          if(beforeCell(order[orderOffset+middle],group,cellX,cellY)) {low=middle+1u;} else {high=middle;}
        }
        var slot=low;
        while(slot<total) {
          let candidate=order[orderOffset+slot];
          if(groups[groupsOffset+candidate]!=group || keysX[keysXOffset+candidate]!=cellX || keysY[keysYOffset+candidate]!=cellY) {break;}
          visits++;
          if(candidate>index && all(point(candidate)==position)) {joinVertices(index,candidate);}
          slot++;
        }
      }
    }
  }
  ${candidateCounts ? 'candidateCounts[candidateCountsOffset+index]=visits;' : ''}`
    }),
    ...Array.from({length: Math.ceil(Math.log2(Math.max(endpointCapacity, 2)))}, (_, pass) =>
      kernel(`compress-${pass}`, {
        invocationCount: endpointCapacity,
        bindings: [
          {name: 'parents', view: vertexRows, type: 'atomic<u32>', access: 'read_write'},
          {name: 'count', view: count, type: 'u32', access: 'read'}
        ],
        body: `if(index>=count[countOffset]) {return;}
  let parent=atomicLoad(&parents[parentsOffset+index]);
  let ancestor=atomicLoad(&parents[parentsOffset+parent]);
  atomicMin(&parents[parentsOffset+index],ancestor);`
      })
    ),
    kernel('edge-keys', {
      invocationCount: segmentCapacity,
      bindings: [
        {name: 'endpoints', view: endpoints, type: 'f32', access: 'read'},
        {name: 'vertices', view: vertexRows, type: 'u32', access: 'read'},
        {name: 'first', view: firstRows, type: 'u32', access: 'read_write'},
        {name: 'last', view: lastRows, type: 'u32', access: 'read_write'},
        {name: 'order', view: edgeOrder, type: 'u32', access: 'read_write'}
      ],
      body: `let firstVertex=vertices[verticesOffset+2u*index];var lastVertex=vertices[verticesOffset+2u*index+1u];
  let start=vec2f(endpoints[endpointsOffset+8u*index],endpoints[endpointsOffset+8u*index+1u]);
  let end=vec2f(endpoints[endpointsOffset+8u*index+4u],endpoints[endpointsOffset+8u*index+5u]);
  if(all(abs(start-end)<=vec2f(${getWGSLFloatLiteral(vertexTolerance)}))) {lastVertex=firstVertex;}
  first[firstOffset+index]=min(firstVertex,lastVertex);last[lastOffset+index]=max(firstVertex,lastVertex);order[orderOffset+index]=index;`
    }),
    ...sortedEdges.nodes,
    kernel('clear', {
      invocationCount: 2 * endpointCapacity,
      bindings: [
        {name: 'state', view: state, type: 'u32', access: 'read_write'},
        {name: 'removed', view: removed, type: 'u32', access: 'read_write'}
      ],
      body: `state[stateOffset+index]=0u;
  if(index<${segmentCapacity}u) {removed[removedOffset+index]=0u;}`
    }),
    kernel('degrees', {
      invocationCount: segmentCapacity,
      bindings: [
        {name: 'first', view: firstRows, type: 'u32', access: 'read'},
        {name: 'last', view: lastRows, type: 'u32', access: 'read'},
        {name: 'order', view: sortedEdges.sortedIndices, type: 'u32', access: 'read'},
        {name: 'owners', view: owners, type: 'u32', access: 'read_write'},
        {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'},
        {name: 'removed', view: removed, type: 'u32', access: 'read_write'}
      ],
      body: `let firstVertex=first[firstOffset+index];let lastVertex=last[lastOffset+index];
  var owner=0xffffffffu;
  if(firstVertex!=0xffffffffu) {
    var low=0u;var high=${segmentCapacity}u;
    while(low<high) {
      let middle=low+(high-low)/2u;let candidate=order[orderOffset+middle];
      let before=first[firstOffset+candidate]<firstVertex || (first[firstOffset+candidate]==firstVertex && last[lastOffset+candidate]<lastVertex);
      if(before) {low=middle+1u;} else {high=middle;}
    }
    owner=order[orderOffset+low];
    // An edge whose ends are identified by tolerance contributes no degree.
    // Retain its source row as a dangle and allow the remaining fringe to peel.
    if(firstVertex==lastVertex) {
      if(owner==index) {removed[removedOffset+index]=1u;}
    } else if(owner==index) {
      atomicAdd(&state[stateOffset+firstVertex],1u);atomicAdd(&state[stateOffset+lastVertex],1u);
      atomicAdd(&state[stateOffset+${endpointCapacity}u+firstVertex],index+1u);
      atomicAdd(&state[stateOffset+${endpointCapacity}u+lastVertex],index+1u);
    }
  }
  owners[ownersOffset+index]=owner;`
    }),
    // contract-allow: single-invocation — linear leaf queue; each vertex enqueues once and each source owner is removed at most once. No CPU readback or per-round graph expansion.
    kernel('peel', {
      invocationCount: 1,
      bindings: [
        {name: 'vertices', view: vertexRows, type: 'u32', access: 'read'},
        {name: 'state', view: state, type: 'u32', access: 'read_write'},
        {name: 'queue', view: queue, type: 'u32', access: 'read_write'},
        {name: 'removed', view: removed, type: 'u32', access: 'read_write'},
        {name: 'count', view: count, type: 'u32', access: 'read'}
      ],
      body: `let total=count[countOffset];
  var head=0u;var tail=0u;
  for(var vertex=0u;vertex<total;vertex++) {
    if(state[stateOffset+vertex]==1u) {queue[queueOffset+tail]=vertex;tail++;}
  }
  while(head<tail) {
    let vertex=queue[queueOffset+head];head++;
    if(state[stateOffset+vertex]!=1u) {continue;}
    let edge=state[stateOffset+${endpointCapacity}u+vertex] - 1u;
    if(removed[removedOffset+edge]!=0u) {continue;}
    removed[removedOffset+edge]=1u;
    for(var side=0u;side<2u;side++) {
      let endpoint=vertices[verticesOffset+edge*2u+side];
      state[stateOffset+endpoint]=state[stateOffset+endpoint] - 1u;
      state[stateOffset+${endpointCapacity}u+endpoint]=state[stateOffset+${endpointCapacity}u+endpoint] - (edge+1u);
      if(state[stateOffset+endpoint]==1u) {queue[queueOffset+tail]=endpoint;tail++;}
    }
  }`
    }),
    kernel('face-endpoints', {
      invocationCount: endpointCapacity,
      bindings: [
        {name: 'endpoints', view: endpoints, type: 'f32', access: 'read'},
        {name: 'owners', view: owners, type: 'u32', access: 'read'},
        {name: 'removed', view: removed, type: 'u32', access: 'read'},
        {name: 'output', view: faceEndpoints, type: 'f32', access: 'read_write'}
      ],
      body: `let owner=owners[ownersOffset+index/2u];
  var peeled=false;
  if(owner!=0xffffffffu) {peeled=removed[removedOffset+owner]!=0u;}
  let base=endpointsOffset+index*4u;let destination=outputOffset+index*4u;
  output[destination]=endpoints[base];output[destination+1u]=endpoints[base+1u];
  output[destination+2u]=select(endpoints[base+2u],endpoints[base],peeled);
  output[destination+3u]=select(endpoints[base+3u],endpoints[base+1u],peeled);`
    })
  ];
  if (vertexTolerance > 0) {
    const direct = createPolygonizeContactPeelingNodes(graph, {
      id,
      endpoints,
      groups,
      count,
      frame,
      order: sortedEndpoints.sortedIndices,
      owners,
      state,
      removed,
      queue,
      vertexTolerance
    });
    const peel = nodes.findIndex(node => node.id === `${id}-peel`);
    nodes.splice(peel, 1, ...direct);
  }
  return {nodes, endpoints: faceEndpoints};
}
