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
import {
  GPUPolygonTriangulation,
  type GPUPolygonTriangulationProps,
  type GPUPolygonTriangulationOutput,
  validatePolygonTriangulationScratch
} from './gpu-polygon-triangulation';
import {ADAPTIVE_INCIRCLE_WGSL} from '../delaunay-tessellation/gpu-delaunay-tessellation';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getGraphOutputViews,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateRequirement,
  validateGPUStorageBindingByteLength,
  validateSinglePackedView,
  validateCommandNodeBudget,
  GPU_MAXIMUM_COMMAND_NODE_COUNT
} from '../../analysis-core/kernels/gpu-contributor-utils';

import {
  getGPUPolygonTriangulationMonotoneMaximumVertexCount,
  validatePolygonTriangulationMonotoneCapacity
} from '../../analysis-core/polygon-triangulation/polygon-triangulation-monotone';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  CONSTRAINED_DELAUNAY_FIXED_NODE_ALLOWANCE,
  CONSTRAINED_DELAUNAY_NODES_PER_ROUND,
  CONSTRAINED_DELAUNAY_PARALLEL_MINIMUM_TRIANGLE_COUNT,
  getConstrainedDelaunayParallelNodes,
  validateConstrainedDelaunayParallelScratch
} from './constrained-delaunay-parallel';

/** A final verification iteration must increment without wrapping the uint32 loop counter. */
const CONSTRAINED_DELAUNAY_MAXIMUM_REFERENCE_FLIPS = 0xfffffffe;
/** Parallel flip records use eight uint32 words per polygon. */
const CONSTRAINED_DELAUNAY_MAXIMUM_PARALLEL_POLYGON_COUNT = Math.floor(0xffffffff / 8);
/** Indexed half-edges and candidate records use three uint32 words per triangle. */
const CONSTRAINED_DELAUNAY_MAXIMUM_PARALLEL_TRIANGLE_COUNT = Math.floor(0xffffffff / 3);

export type GPUConstrainedDelaunayTriangulationProps = Omit<
  GPUPolygonTriangulationProps,
  'output'
> & {
  maximumFlips?: number;
  /** Auto parallelizes polygons averaging at least 256 vertices, or when the caller declares
   * a largest polygon of that size, when the node budget fits.
   * Serial retains the original first-illegal-pair reference. Parallel preserves that same order. */
  flipMethod?: 'auto' | 'serial' | 'parallel';
  /** Optional compile-time bound used to select the parallel route for a skewed polygon set. */
  maximumVerticesPerPolygon?: number;
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
  readonly useParallelFlips: boolean;
  readonly estimatedCommandNodeCount: number;
  constructor(props: GPUConstrainedDelaunayTriangulationProps) {
    this.id = props.id ?? 'constrained-delaunay';
    this.props = props;
    const polygonCount = props.polygonOffsets.length - 1;
    validateRequirement(
      this.id,
      props.output.status.converged.length >= polygonCount,
      `output.status.converged must have at least ${polygonCount} rows (one per polygon)`
    );
    validateCapacity(
      this.id,
      'maximumFlips',
      props.maximumFlips ?? 4096,
      1,
      CONSTRAINED_DELAUNAY_MAXIMUM_REFERENCE_FLIPS
    );
    validateSinglePackedView(
      props.output.status.converged,
      ['uint32'],
      `${this.id} output.status.converged`
    );
    if (props.maximumVerticesPerPolygon !== undefined)
      validateCapacity(this.id, 'maximumVerticesPerPolygon', props.maximumVerticesPerPolygon);
    const method = props.flipMethod ?? 'auto';
    validateRequirement(
      this.id,
      ['auto', 'serial', 'parallel'].includes(method),
      "flipMethod must be 'auto', 'serial' or 'parallel'"
    );
    const parallelNodes =
      CONSTRAINED_DELAUNAY_FIXED_NODE_ALLOWANCE +
      CONSTRAINED_DELAUNAY_NODES_PER_ROUND * ((props.maximumFlips ?? 4096) + 1);
    this.useParallelFlips =
      method === 'parallel' ||
      (method === 'auto' &&
        (props.positions.length >=
          CONSTRAINED_DELAUNAY_PARALLEL_MINIMUM_TRIANGLE_COUNT * Math.max(polygonCount, 1) ||
          (props.maximumVerticesPerPolygon ?? 0) >=
            CONSTRAINED_DELAUNAY_PARALLEL_MINIMUM_TRIANGLE_COUNT) &&
        parallelNodes <= GPU_MAXIMUM_COMMAND_NODE_COUNT);
    this.estimatedCommandNodeCount = this.useParallelFlips
      ? parallelNodes
      : CONSTRAINED_DELAUNAY_FIXED_NODE_ALLOWANCE;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    if (this.useParallelFlips) {
      validateCapacity(
        this.id,
        'polygon count',
        polygonCount,
        0,
        CONSTRAINED_DELAUNAY_MAXIMUM_PARALLEL_POLYGON_COUNT
      );
      validateCapacity(
        this.id,
        'triangle capacity',
        Math.floor(props.output.indices.length / 3),
        0,
        CONSTRAINED_DELAUNAY_MAXIMUM_PARALLEL_TRIANGLE_COUNT
      );
    }
    const {status: ignoredStatus, ...triangulationOutput} = props.output;
    void ignoredStatus;
    new GPUPolygonTriangulation({...props, output: triangulationOutput});
    const outputs = getGraphOutputViews(props.output).filter(view => view !== undefined);
    validateRequirement(
      this.id,
      new Set(outputs.map(view => view.buffer)).size === outputs.length,
      'output views must use distinct buffers'
    );
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
    validateGraphChunkBindingRanges(graph, id, {
      positions: props.positions,
      featureOffsets: props.featureOffsets,
      polygonOffsets: props.polygonOffsets,
      ringOffsets: props.ringOffsets,
      ...Object.fromEntries(
        getGraphOutputViews(output).map((view, index) => [`output ${index}`, view])
      )
    });
    validatePolygonTriangulationScratch(
      graph.device,
      id,
      props.positions.length,
      props.ringOffsets.length - 1,
      polygonCount,
      props.useZOrderHash ?? true
    );
    for (const [name, words] of [
      ['triangle offsets', polygonCount + 1],
      ['polygon layout', 4 * polygonCount + 2],
      ['ring layout', 2 * (props.ringOffsets.length - 1) + 1]
    ] as const)
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        Math.max(words, 1) * 4,
        'reduce the geometry count'
      );
    if (this.useParallelFlips)
      validateConstrainedDelaunayParallelScratch(
        graph,
        id,
        Math.floor(output.indices.length / 3),
        polygonCount
      );
    if (
      (props.largePolygonMethod ?? 'monotone') === 'monotone' &&
      props.positions.length <=
        getGPUPolygonTriangulationMonotoneMaximumVertexCount(graph.device) &&
      props.monotoneMaximumVertices !== undefined
    ) {
      validatePolygonTriangulationMonotoneCapacity(graph.device, id, props.monotoneMaximumVertices);
    }
    const triangleOffsets =
      output.triangleOffsets ??
      createTransientView(graph, `${id}-triangle-offsets`, 'uint32', polygonCount + 1);
    const initialNodes = new GPUPolygonTriangulation({
      ...props,
      id: `${id}-initial`,
      output: {
        indices: output.indices,
        valid: output.valid,
        triangleCount: output.triangleCount,
        triangleOffsets
      }
    }).getCommandNodes(graph);
    if (this.useParallelFlips)
      return [
        ...initialNodes,
        ...getConstrainedDelaunayParallelNodes(graph, {...props, id}, triangleOffsets)
      ];
    return [
      ...initialNodes,
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
