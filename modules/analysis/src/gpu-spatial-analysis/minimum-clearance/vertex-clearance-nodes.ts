// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getSegmentBVHNodes} from '../segment-intersection/segment-bvh';
/** @internal Distinct vertex candidates include collapsed parts absent from the segment BVH. */
export function createVertexClearanceNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    features: GraphDataView<'uint32'>;
    nearest: GraphDataView<'float32x4'>;
    leafCapacity: number;
    spatialSort: boolean;
    overflow?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, positions, features, nearest, leafCapacity, spatialSort} = props;
  const count = positions.length;
  const minima = createTransientView(graph, `${id}-vertex-minima`, 'float32x2', count);
  const maxima = createTransientView(graph, `${id}-vertex-maxima`, 'float32x2', count);
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-vertex-bounds`,
      operation: 'GPUMinimumClearance',
      invocationCount: count,
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        {name: 'minima', view: minima, type: 'f32', access: 'read_write'},
        {name: 'maxima', view: maxima, type: 'f32', access: 'read_write'}
      ],
      body: `let point=vec2f(positions[positionsOffset+index*2u],positions[positionsOffset+index*2u+1u]);let usable=all((bitcast<vec2u>(point)&vec2u(0x7f800000u))!=vec2u(0x7f800000u));let minimum=select(vec2f(3.402823466e+38),point,usable);let maximum=select(vec2f(-3.402823466e+38),point,usable);minima[minimaOffset+index*2u]=minimum.x;minima[minimaOffset+index*2u+1u]=minimum.y;maxima[maximaOffset+index*2u]=maximum.x;maxima[maximaOffset+index*2u+1u]=maximum.y;`
    })
  ];
  const {bvh, nodes: bvhNodes} = getSegmentBVHNodes(graph, {
    id: `${id}-vertices`,
    operation: 'GPUMinimumClearance',
    minima,
    maxima,
    leafCapacity,
    spatialSort
  });
  nodes.push(...bvhNodes);
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-vertex-probe`,
      operation: 'GPUMinimumClearance',
      invocationCount: count,
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        {name: 'features', view: features, type: 'u32', access: 'read'},
        {name: 'minima', view: bvh.nodeMinima, type: 'f32', access: 'read'},
        {name: 'maxima', view: bvh.nodeMaxima, type: 'f32', access: 'read'},
        {name: 'leaves', view: bvh.leafIds, type: 'u32', access: 'read'},
        {name: 'nearest', view: nearest, type: 'f32', access: 'read_write'}
      ],
      declarations: `fn vertex(row:u32)->vec2f{return vec2f(positions[positionsOffset+row*2u],positions[positionsOffset+row*2u+1u]);}
// Expand the rounded incumbent before squaring: sqrt(2) rounds downward in f32.
fn squaredUpperBound(distance:f32)->f32 {
  let bits=bitcast<u32>(distance);
  if(bits>=0x7f7fffffu) {return distance*distance;}
  let upper=bitcast<f32>(bits+1u);return upper*upper;
}
fn boxDistance(node:u32,point:vec2f)->f32{let minimum=vec2f(minima[minimaOffset+node*2u],minima[minimaOffset+node*2u+1u]);let maximum=vec2f(maxima[maximaOffset+node*2u],maxima[maximaOffset+node*2u+1u]);let gap=max(max(minimum-point,point-maximum),vec2f(0.0));return dot(gap,gap);}`,
      body: `let feature=features[featuresOffset+index];if(feature==0xffffffffu){return;}let point=vertex(index);if(any((bitcast<vec2u>(point)&vec2u(0x7f800000u))==vec2u(0x7f800000u))){return;}
let base=nearestOffset+index*4u;var best=nearest[base];var bestPoint=vec2f(nearest[base+1u],nearest[base+2u]);let incumbentPoint=bestPoint;let incumbentDistance=best;var incumbentIsVertex=false;var bestPartner=0xffffffffu;var bestSquared=squaredUpperBound(best);var stack:array<u32,64>;var depth=1u;stack[0]=0u;
loop{if(depth==0u){break;}depth-=1u;let node=stack[depth];if(boxDistance(node,point)>bestSquared){continue;}if(node<${bvh.internalNodeCount}u){stack[depth]=node*2u+1u;stack[depth+1u]=node*2u+2u;depth+=2u;continue;}let row=leaves[leavesOffset+node-${bvh.internalNodeCount}u];if(row>=${count}u||features[featuresOffset+row]!=feature){continue;}let candidate=vertex(row);let delta=point-candidate;let distanceSquared=dot(delta,delta);if(distanceSquared>0.0){let distance=sqrt(distanceSquared);if(distance==incumbentDistance && all(candidate==incumbentPoint)){incumbentIsVertex=true;}if(distance<best || (distance==best && row<bestPartner)){best=distance;bestSquared=max(distanceSquared,squaredUpperBound(distance));bestPoint=candidate;bestPartner=row;}}}
// Keep an interior segment witness when no stored endpoint represents its tied point.
if(best==incumbentDistance && !incumbentIsVertex){bestPoint=incumbentPoint;}
nearest[base]=best;nearest[base+1u]=bestPoint.x;nearest[base+2u]=bestPoint.y;`
    })
  );
  if (props.overflow)
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-vertex-overflow`,
        operation: 'GPUMinimumClearance',
        invocationCount: 1,
        bindings: [
          {name: 'input', view: bvh.overflow, type: 'u32', access: 'read'},
          {name: 'output', view: props.overflow, type: 'u32', access: 'read_write'}
        ],
        body: 'output[outputOffset]=select(0u,1u,output[outputOffset]!=0u||input[inputOffset]!=0u);'
      })
    );
  return nodes;
}
