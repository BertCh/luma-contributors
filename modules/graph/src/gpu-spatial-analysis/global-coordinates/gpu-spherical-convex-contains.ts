// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {assert} from '@luma.gl/core';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validatePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../utils/gpu-contributor-utils';
export type GPUSphericalConvexContainsProps = {
  id?: string;
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
    /** 0 valid, 1 invalid/unsupported polygon or query, 2 boundary tolerance band. */
    status: GraphDataView<'uint32'>;
  };
};
/** Convex spherical containment with explicit boundary and complement semantics. */
export class GPUSphericalConvexContains implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUSphericalConvexContainsProps) {
    this.id = props.id ?? 'spherical-convex-contains';
    for (const view of [props.positions, props.queries])
      validatePackedView(view, ['float32x3'], this.id);
    for (const view of [props.offsets, props.polygonIndices, ...Object.values(props.output)])
      validatePackedView(view, ['uint32'], this.id);
    // One polygon selector and output per query; polygons contain at most 32 vertices.
    assert(
      props.offsets.length >= 1 &&
        props.polygonIndices.length === props.queries.length &&
        props.output.contains.length === props.queries.length &&
        props.output.status.length === props.queries.length
    );
    assert(
      Number.isFinite(props.tolerance ?? 1e-6) &&
        (props.tolerance ?? 1e-6) >= 1e-7 &&
        (props.tolerance ?? 1e-6) <= 1e-2
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
  if (begin > end || end > ${props.positions.length}u || end - begin < 3u || end - begin > 32u) { failure = 1u; }
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
