// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateSinglePackedView,
  validateRequirement,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  GPU_SPHERICAL_CONVEX_MAXIMUM_VERTICES,
  GPU_SPHERICAL_CONVEX_MINIMUM_VERTICES,
  GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
  GPU_SPHERICAL_MAXIMUM_TOLERANCE,
  GPU_SPHERICAL_MINIMUM_TOLERANCE,
  GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT,
  validateAngularTolerance
} from './spherical-limits';

/** Status of one containment row: the value of `output.status`. */
export const GPU_SPHERICAL_CONVEX_CONTAINS_STATUS_VALID = 0;
/** Status: invalid or unsupported polygon, polygon index or query (nonconvex, non-unit, too few or too many vertices). */
export const GPU_SPHERICAL_CONVEX_CONTAINS_STATUS_INVALID = 1;
/** Status: the query lies within `tolerance` of the polygon boundary; `contains` follows `boundary`. */
export const GPU_SPHERICAL_CONVEX_CONTAINS_STATUS_BOUNDARY = 2;
export type GPUSphericalConvexContainsProps = {
  id?: string;
  /**
   * Coordinate space of the unit vectors: longitude-latitude (their embedding on the sphere) with
   * great-circle edges. Planar contexts are rejected. Defaults to longitude-latitude, no metric.
   */
  spatialContext?: GPUSpatialContext;
  /** Unit xyz vertices, CCW minor arcs; each polygon is convex and lies in an open hemisphere. */
  positions: GraphDataView<'float32x3'>;
  offsets: GraphDataView<'uint32'>;
  queries: GraphDataView<'float32x3'>;
  polygonIndices: GraphDataView<'uint32'>;
  boundary?: 'closed' | 'open';
  /** Query the complementary region; boundary policy still applies explicitly. */
  complement?: boolean;
  tolerance?: number;
  output: {
    /** 0 false, 1 true; invalid rows must be interpreted using status. */
    contains: GraphDataView<'uint32'>;
    /**
     * Per query: {@link GPU_SPHERICAL_CONVEX_CONTAINS_STATUS_VALID} (0),
     * {@link GPU_SPHERICAL_CONVEX_CONTAINS_STATUS_INVALID} (1, invalid or unsupported polygon or
     * query) or {@link GPU_SPHERICAL_CONVEX_CONTAINS_STATUS_BOUNDARY} (2, boundary tolerance band).
     */
    status: GraphDataView<'uint32'>;
  };
};
/** Convex spherical containment with explicit boundary and complement semantics. */
export class GPUSphericalConvexContains implements GPUCommandNodeProducer {
  readonly id: string;
  /** Validated coordinate space of the unit vectors. */
  readonly spatialContext: GPUSpatialContext;
  constructor(readonly props: GPUSphericalConvexContainsProps) {
    this.id = props.id ?? 'spherical-convex-contains';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext ?? GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
      GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT
    );
    for (const view of [props.positions, props.queries])
      validateSinglePackedView(view, ['float32x3'], this.id);
    for (const view of [props.offsets, props.polygonIndices, ...Object.values(props.output)])
      validateSinglePackedView(view, ['uint32'], this.id);
    // One polygon selector and output per query; polygons contain at most 32 vertices.
    validateRequirement(this.id, props.offsets.length >= 1, 'offsets must hold at least one entry');
    validateRequirement(
      this.id,
      props.polygonIndices.length === props.queries.length,
      'polygonIndices must hold one entry per query'
    );
    validateRequirement(
      this.id,
      props.output.contains.length === props.queries.length &&
        props.output.status.length === props.queries.length,
      'output.contains and output.status must hold one entry per query'
    );
    validateAngularTolerance(
      this.id,
      props.tolerance ?? 1e-6,
      GPU_SPHERICAL_MINIMUM_TOLERANCE,
      GPU_SPHERICAL_MAXIMUM_TOLERANCE
    );
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      props.offsets,
      props.queries,
      props.polygonIndices
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.offsets,
      props.queries,
      props.polygonIndices,
      ...Object.values(props.output)
    ]);
    return [
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPUSphericalConvexContains',
        variant: `${props.boundary ?? 'closed'}-${props.complement ? 'complement' : 'interior'}`,
        invocationCount: props.queries.length,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
          {name: 'queries', view: props.queries, type: 'f32', access: 'read'},
          {name: 'polygonIndices', view: props.polygonIndices, type: 'u32', access: 'read'},
          {name: 'contains', view: props.output.contains, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `const CONTAINS_TOLERANCE: f32 = ${props.tolerance ?? 1e-6};
fn containsPosition(row: u32) -> vec3f { let offset = positionsOffset + row * 3u; return vec3f(positions[offset], positions[offset + 1u], positions[offset + 2u]); }
fn containsUnit(point: vec3f) -> bool { let squared = dot(point, point); return (bitcast<u32>(squared) & 0x7f800000u) != 0x7f800000u && abs(squared - 1.0) <= 1e-4; }`,
        body: `let queryOffset = queriesOffset + index * 3u; let point = vec3f(queries[queryOffset], queries[queryOffset + 1u], queries[queryOffset + 2u]);
let polygon = polygonIndices[polygonIndicesOffset + index]; var failure = 0u; var inside = true; var boundary = false;
if (!containsUnit(point) || polygon >= ${props.offsets.length - 1}u) { failure = 1u; }
if (failure == 0u) {
  let begin = offsets[offsetsOffset + polygon]; let end = offsets[offsetsOffset + polygon + 1u];
  if (begin > end || end > ${props.positions.length}u || end - begin < ${GPU_SPHERICAL_CONVEX_MINIMUM_VERTICES}u || end - begin > ${GPU_SPHERICAL_CONVEX_MAXIMUM_VERTICES}u) { failure = 1u; }
  if (failure == 0u) {
    var center = vec3f(0.0);
    for (var vertex = begin; vertex < end; vertex++) { let position = containsPosition(vertex); if (!containsUnit(position)) { failure = 1u; } center += position; }
    if (length(center) < CONTAINS_TOLERANCE) { failure = 1u; } else { center = normalize(center); }
    for (var edge = begin; edge < end && failure == 0u; edge++) {
      let first = containsPosition(edge); let second = containsPosition(select(edge + 1u, begin, edge + 1u == end));
      let crossing = cross(first, second);
      if (length(crossing) <= CONTAINS_TOLERANCE || dot(center, first) <= CONTAINS_TOLERANCE) { failure = 1u; break; }
      let normal = normalize(crossing);
      if (dot(normal, center) <= CONTAINS_TOLERANCE) { failure = 1u; break; }
      for (var vertex = begin; vertex < end; vertex++) { if (dot(normal, containsPosition(vertex)) < -CONTAINS_TOLERANCE) { failure = 1u; } }
      let side = dot(normal, point); inside = inside && side >= -CONTAINS_TOLERANCE; boundary = boundary || abs(side) <= CONTAINS_TOLERANCE;
    }
  }
}
boundary = boundary && inside;
var accepted = ${props.complement ? '!inside' : 'inside'};
if (boundary) { accepted = ${props.boundary !== 'open'}; if (failure == 0u) { failure = 2u; } }
contains[containsOffset + index] = select(0u, 1u, accepted && failure != 1u); status[statusOffset + index] = failure;`
      })
    ];
  }
}
