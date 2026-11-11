// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assertUnitVector} from './assert-unit-vector';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getReadPoint3WGSL} from '../shared/spatial-wgsl-point-helpers';
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
  GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
  GPU_SPHERICAL_MAXIMUM_TOLERANCE,
  GPU_SPHERICAL_MINIMUM_TOLERANCE,
  GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT,
  validateAngularTolerance
} from './spherical-limits';

/** Status of one query: inside or outside the polygon (or its complement). */
export const GPU_SPHERICAL_POLYGON_CONTAINS_STATUS_VALID = 0;
/** Status: invalid query, polygon index or ring (including an edge shorter than `tolerance`). */
export const GPU_SPHERICAL_POLYGON_CONTAINS_STATUS_INVALID = 1;
/** Status: the query lies on the boundary within `tolerance`; `contains` follows `boundary`. */
export const GPU_SPHERICAL_POLYGON_CONTAINS_STATUS_BOUNDARY = 2;

export type GPUSphericalPolygonContainsProps = {
  id?: string;
  /**
   * Coordinate space of the unit vectors: longitude-latitude (their embedding on the sphere) with
   * great-circle edges. Planar contexts are rejected. Defaults to longitude-latitude, no metric.
   */
  spatialContext?: GPUSpatialContext;
  /** Unit xyz, simple CCW minor-arc rings, with no repeated closing vertex. */
  positions: GraphDataView<'float32x3'>;
  ringOffsets: GraphDataView<'uint32'>;
  /** Ring offsets per polygon: first ring is the shell; subsequent rings are holes. */
  polygonOffsets: GraphDataView<'uint32'>;
  queries: GraphDataView<'float32x3'>;
  polygonIndices: GraphDataView<'uint32'>;
  boundary?: 'closed' | 'open';
  complement?: boolean;
  tolerance?: number;
  /** Optional common hemisphere pole. Omitted: each ring uses its normalized vertex mean. */
  hemisphereCenter?: readonly [number, number, number];
  output: {
    /** 0 outside, 1 inside (after `complement`); read it together with `status`. */
    contains: GraphDataView<'uint32'>;
    /**
     * Per query: {@link GPU_SPHERICAL_POLYGON_CONTAINS_STATUS_VALID} (0),
     * {@link GPU_SPHERICAL_POLYGON_CONTAINS_STATUS_INVALID} (1) or
     * {@link GPU_SPHERICAL_POLYGON_CONTAINS_STATUS_BOUNDARY} (2).
     */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Nonconvex spherical winding containment with holes and explicit complementary-region semantics.
 *
 * Ring validity (unit vertices, hemisphere condition, no crossing or touching nonadjacent arcs) is
 * checked once per ring, not once per query; a query then costs one pass over its rings' edges.
 *
 * Status per query: `0` inside or outside, `1` invalid query, polygon index or ring (this includes
 * a ring with an edge whose cross product is within `tolerance`, i.e. an edge shorter than about
 * `tolerance` radians or nearly antipodal endpoints), `2` on the boundary within `tolerance`. The
 * f32 winding sum is only trustworthy for queries farther than `tolerance` from every vertex, so
 * keep `tolerance` (at least 1e-7, default 1e-6 or about 6 m) below the shortest edge of interest.
 */
export class GPUSphericalPolygonContains implements GPUCommandNodeProducer {
  readonly id: string;
  /** Validated coordinate space of the unit vectors. */
  readonly spatialContext: GPUSpatialContext;
  constructor(readonly props: GPUSphericalPolygonContainsProps) {
    this.id = props.id ?? 'spherical-polygon-contains';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext ?? GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
      GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT
    );
    if (props.hemisphereCenter)
      assertUnitVector(props.hemisphereCenter, this.id, 'hemisphereCenter');
    for (const view of [props.positions, props.queries])
      validateSinglePackedView(view, ['float32x3'], this.id);
    for (const view of [
      props.ringOffsets,
      props.polygonOffsets,
      props.polygonIndices,
      ...Object.values(props.output)
    ])
      validateSinglePackedView(view, ['uint32'], this.id);
    validateRequirement(
      this.id,
      props.ringOffsets.length >= 1 && props.polygonOffsets.length >= 1,
      'ringOffsets and polygonOffsets must hold at least one entry'
    );
    validateRequirement(
      this.id,
      props.queries.length === props.polygonIndices.length &&
        props.output.contains.length === props.queries.length &&
        props.output.status.length === props.queries.length,
      'polygonIndices, output.contains and output.status must hold one entry per query'
    );
    validateAngularTolerance(
      this.id,
      props.tolerance ?? 1e-6,
      GPU_SPHERICAL_MINIMUM_TOLERANCE,
      GPU_SPHERICAL_MAXIMUM_TOLERANCE
    );
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      props.ringOffsets,
      props.polygonOffsets,
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
      props.ringOffsets,
      props.polygonOffsets,
      props.queries,
      props.polygonIndices,
      ...Object.values(props.output)
    ]);
    const tolerance = getWGSLFloatLiteral(props.tolerance ?? 1e-6);
    const ringCount = props.ringOffsets.length - 1;
    // Query-independent ring validation is computed once per ring instead of once per query.
    const ringCenters = createTransientView(
      graph,
      `${id}-ring-centers`,
      'float32x3',
      Math.max(ringCount, 1)
    );
    const ringFailures = createTransientView(
      graph,
      `${id}-ring-failures`,
      'uint32',
      Math.max(ringCount, 1)
    );
    const declarations = `const POLYGON_TOLERANCE: f32 = ${tolerance};
${getReadPoint3WGSL('polygonPosition', 'positions')}
fn polygonUnit(point: vec3f) -> bool { let squared = dot(point, point); return (bitcast<u32>(squared) & 0x7f800000u) != 0x7f800000u && abs(squared - 1.0) <= 1e-4; }
fn polygonOnArc(first: vec3f, second: vec3f, point: vec3f, margin: f32) -> bool {
  let normal = normalize(cross(first, second));
  return abs(dot(normal, point)) <= margin && dot(cross(first, point), normal) >= -margin && dot(cross(point, second), normal) >= -margin;
}`;
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-ring-center`,
        operation: 'GPUSphericalPolygonContains',
        variant: 'ring-center',
        invocationCount: Math.max(ringCount, 1),
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
          {name: 'ringCenters', view: ringCenters, type: 'f32', access: 'read_write'},
          {name: 'ringFailures', view: ringFailures, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations,
        body: `if (index >= ${ringCount}u) { return; }
let begin = ringOffsets[ringOffsetsOffset + index]; let end = ringOffsets[ringOffsetsOffset + index + 1u];
var failure = 0u; var center = vec3f(0.0);
if (begin > end || end > ${props.positions.length}u || end - begin < 3u) { failure = 1u; }
else {
  for (var vertex = begin; vertex < end; vertex++) { let value = polygonPosition(vertex); if (!polygonUnit(value)) { failure = 1u; } center += value; }
  // The supplied ring represents its smaller region; complement is explicit at polygon level.
  ${props.hemisphereCenter ? `center = vec3f(${props.hemisphereCenter.join(',')});` : ''}
  if (length(center) <= POLYGON_TOLERANCE) { failure = 1u; } else { center = normalize(center); }
}
ringCenters[ringCentersOffset + index * 3u] = center.x; ringCenters[ringCentersOffset + index * 3u + 1u] = center.y; ringCenters[ringCentersOffset + index * 3u + 2u] = center.z;
atomicStore(&ringFailures[ringFailuresOffset + index], failure);`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-ring-edges`,
        operation: 'GPUSphericalPolygonContains',
        variant: 'ring-edges',
        invocationCount: Math.max(props.positions.length, 1),
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
          {name: 'ringCenters', view: ringCenters, type: 'f32', access: 'read'},
          {name: 'ringFailures', view: ringFailures, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations,
        body: `if (index >= ${props.positions.length}u || ${ringCount}u == 0u) { return; }
// Locate the ring that owns this vertex by binary search over the ring offsets.
var low = 0u; var high = ${ringCount}u;
for (var step = 0u; step < 32u && low + 1u < high; step++) {
  let middle = (low + high) / 2u;
  if (ringOffsets[ringOffsetsOffset + middle] <= index) { low = middle; } else { high = middle; }
}
let ring = low;
let begin = ringOffsets[ringOffsetsOffset + ring]; let end = ringOffsets[ringOffsetsOffset + ring + 1u];
if (index < begin || index >= end || atomicLoad(&ringFailures[ringFailuresOffset + ring]) != 0u) { return; }
let center = vec3f(ringCenters[ringCentersOffset + ring * 3u], ringCenters[ringCentersOffset + ring * 3u + 1u], ringCenters[ringCentersOffset + ring * 3u + 2u]);
let edge = index;
let next = select(edge + 1u, begin, edge + 1u == end);
let first = polygonPosition(edge); let second = polygonPosition(next);
let normal = cross(first, second);
var failure = 0u;
if (length(normal) <= POLYGON_TOLERANCE || dot(center, first) <= POLYGON_TOLERANCE) { failure = 1u; }
// Reject crossing/touching nonadjacent arcs instead of assigning an invalid ring a fill.
for (var other = begin; other < end && failure == 0u; other++) {
  let otherNext = select(other + 1u, begin, other + 1u == end);
  if (other == edge || other == next || otherNext == edge) { continue; }
  let third = polygonPosition(other); let fourth = polygonPosition(otherNext);
  let crossing = cross(normal, cross(third, fourth));
  if (length(crossing) > POLYGON_TOLERANCE * POLYGON_TOLERANCE) {
    let intersection = normalize(crossing);
    if ((polygonOnArc(first, second, intersection, POLYGON_TOLERANCE) && polygonOnArc(third, fourth, intersection, POLYGON_TOLERANCE)) || (polygonOnArc(first, second, -intersection, POLYGON_TOLERANCE) && polygonOnArc(third, fourth, -intersection, POLYGON_TOLERANCE))) { failure = 1u; }
  } else if (polygonOnArc(first, second, third, POLYGON_TOLERANCE) || polygonOnArc(first, second, fourth, POLYGON_TOLERANCE)) { failure = 1u; }
}
if (failure != 0u) { atomicMax(&ringFailures[ringFailuresOffset + ring], 1u); }`
      }),
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPUSphericalPolygonContains',
        variant: `${props.boundary ?? 'closed'}-${Boolean(props.complement)}`,
        invocationCount: props.queries.length,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
          {name: 'polygonOffsets', view: props.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'queries', view: props.queries, type: 'f32', access: 'read'},
          {name: 'polygonIndices', view: props.polygonIndices, type: 'u32', access: 'read'},
          {name: 'contains', view: props.output.contains, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
          {name: 'ringFailures', view: ringFailures, type: 'u32', access: 'read'}
        ],
        declarations,
        body: `let queryOffset = queriesOffset + index * 3u;
let point = vec3f(queries[queryOffset], queries[queryOffset + 1u], queries[queryOffset + 2u]);
let polygon = polygonIndices[polygonIndicesOffset + index];
var failure = 0u; var inside = false; var boundary = false;
if (!polygonUnit(point) || polygon >= ${props.polygonOffsets.length - 1}u) { failure = 1u; }
if (failure == 0u) {
  let firstRing = polygonOffsets[polygonOffsetsOffset + polygon]; let lastRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
  if (firstRing > lastRing || lastRing >= ${props.ringOffsets.length}u) { failure = 1u; }
  for (var ring = firstRing; ring < lastRing && failure == 0u; ring++) {
    // Ring validity (range, unit vectors, hemisphere, arc crossings) is precomputed per ring.
    if (ringFailures[ringFailuresOffset + ring] != 0u) { failure = 1u; break; }
    let begin = ringOffsets[ringOffsetsOffset + ring]; let end = ringOffsets[ringOffsetsOffset + ring + 1u];
    var winding = 0.0; var ringBoundary = false;
    for (var edge = begin; edge < end; edge++) {
      let next = select(edge + 1u, begin, edge + 1u == end);
      let first = polygonPosition(edge); let second = polygonPosition(next);
      let normal = cross(first, second);
      ringBoundary = ringBoundary || polygonOnArc(first, second, point, POLYGON_TOLERANCE);
      let numerator = dot(point, normal); let denominator = dot(first, second) - dot(point, first) * dot(point, second);
      if (abs(numerator) + abs(denominator) > 1e-12) { winding += atan2(numerator, denominator); }
    }
    let ringInside = winding > 3.141592653589793;
    if (ring == firstRing) { inside = ringInside; } else { inside = inside && !ringInside; }
    boundary = boundary || ringBoundary;
  }
}
var accepted = ${props.complement ? '!inside' : 'inside'};
if (boundary && failure == 0u) { failure = 2u; accepted = ${props.boundary !== 'open'}; }
contains[containsOffset + index] = select(0u, 1u, accepted && failure != 1u);
status[statusOffset + index] = failure;`
      })
    ];
  }
}
