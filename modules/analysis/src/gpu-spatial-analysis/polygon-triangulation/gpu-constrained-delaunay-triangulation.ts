// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGraphOutputViews} from '../../analysis-core/kernels/gpu-contributor-utils';
import {assert} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPUPolygonTriangulation,
  type GPUPolygonTriangulationProps,
  type GPUPolygonTriangulationOutput
} from './gpu-polygon-triangulation';
import {ADAPTIVE_INCIRCLE_WGSL} from '../delaunay-tessellation/gpu-delaunay-tessellation';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateGraphOutputsDisjointFromInputs} from '../../analysis-core/kernels/gpu-contributor-utils';

export type GPUConstrainedDelaunayTriangulationProps = Omit<
  GPUPolygonTriangulationProps,
  'output'
> & {
  maximumFlips?: number;
  output: GPUPolygonTriangulationOutput & {
    /** Per-polygon convergence. Triangulation validity alone does not certify Delaunay completion. */
    status: {converged: GraphDataView<'uint32'>};
  };
};

/**
 * Triangulates polygon shells/holes, then performs bounded Lawson flips of shared interior edges.
 * Boundary edges remain constrained; only strictly convex quadrilaterals with positive incircle
 * tests flip. Cocircular diagonals stay unchanged. Exact predicates follow the tessellator's
 * documented f32 exponent-span fallback. Iteration exhaustion preserves a valid mesh but reports
 * converged=0, so consumers never confuse an unfinished mesh with a certified CDT.
 */
export class GPUConstrainedDelaunayTriangulation implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUConstrainedDelaunayTriangulationProps;
  constructor(props: GPUConstrainedDelaunayTriangulationProps) {
    this.id = props.id ?? 'constrained-delaunay';
    this.props = props;
    assert(props.output.status.converged.length >= props.polygonOffsets.length - 1);
    assert(Number.isInteger(props.maximumFlips ?? 4096) && (props.maximumFlips ?? 4096) > 0);
    new GPUPolygonTriangulation(props);
    const outputs = getGraphOutputViews(props.output).filter(view => view !== undefined);
    assert(new Set(outputs.map(view => view.buffer)).size === outputs.length);
    validateGraphOutputsDisjointFromInputs(this.id, outputs, [
      props.positions,
      props.featureOffsets,
      props.polygonOffsets,
      props.ringOffsets
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const polygonCount = props.polygonOffsets.length - 1;
    const triangleOffsets =
      output.triangleOffsets ??
      createTransientView(graph, `${id}-triangle-offsets`, 'uint32', polygonCount + 1);
    return [
      ...new GPUPolygonTriangulation({
        ...props,
        id: `${id}-initial`,
        output: {...output, triangleOffsets}
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-lawson`,
        operation: 'GPUConstrainedDelaunayTriangulation',
        invocationCount: polygonCount,
        workgroupSize: 32,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'indices', view: output.indices, type: 'u32', access: 'read_write'},
          {name: 'offsets', view: triangleOffsets, type: 'u32', access: 'read'},
          {name: 'valid', view: output.valid, type: 'u32', access: 'read'},
          {name: 'converged', view: output.status.converged, type: 'u32', access: 'read_write'}
        ],
        declarations: `${EXACT_ORIENTATION_WGSL}\n${ADAPTIVE_INCIRCLE_WGSL}
fn getTriangle(triangle:u32)->vec3<u32>{let offset=indicesOffset+triangle*3u;return vec3<u32>(indices[offset],indices[offset+1u],indices[offset+2u]);}
fn setTriangle(triangle:u32,first:u32,second:u32,third:u32){let offset=indicesOffset+triangle*3u;indices[offset]=first;indices[offset+1u]=second;indices[offset+2u]=third;}
fn point(vertex:u32)->vec2<f32>{return vec2<f32>(positions[positionsOffset+vertex*2u],positions[positionsOffset+vertex*2u+1u]);}`,
        body: `converged[convergedOffset+index]=0u;if(valid[validOffset+index]==0u){return;}
let begin=offsets[offsetsOffset+index];let end=offsets[offsetsOffset+index+1u];var complete=false;
for(var iteration=0u;iteration<=${props.maximumFlips ?? 4096}u;iteration++){
 var changed=false;
 for(var firstTriangle=begin;firstTriangle<end;firstTriangle++){
  let first=getTriangle(firstTriangle);if(first.x==first.y||first.y==first.z||first.z==first.x){continue;}
  for(var secondTriangle=firstTriangle+1u;secondTriangle<end;secondTriangle++){
   let second=getTriangle(secondTriangle);if(second.x==second.y||second.y==second.z||second.z==second.x){continue;}
   for(var side=0u;side<3u;side++){
    let firstVertex=first[side];let secondVertex=first[(side+1u)%3u];let firstOpposite=first[(side+2u)%3u];
    for(var otherSide=0u;otherSide<3u;otherSide++){
     if(second[otherSide]!=secondVertex||second[(otherSide+1u)%3u]!=firstVertex){continue;}
     let secondOpposite=second[(otherSide+2u)%3u];
     if(orientSign(point(firstOpposite),point(secondOpposite),point(secondVertex))!=1||orientSign(point(secondOpposite),point(firstOpposite),point(firstVertex))!=1){continue;}
     if(incircleSign(point(firstVertex),point(secondVertex),point(firstOpposite),point(secondOpposite))<=0){continue;}
     changed=true;
     if(iteration<${props.maximumFlips ?? 4096}u){setTriangle(firstTriangle,firstOpposite,secondOpposite,secondVertex);setTriangle(secondTriangle,secondOpposite,firstOpposite,firstVertex);}
     break;
    }if(changed){break;}
   }if(changed){break;}
  }if(changed){break;}
 }
 if(!changed){complete=true;break;}
}
converged[convergedOffset+index]=u32(complete);`
      })
    ];
  }
}
