// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateSinglePackedView,
  validateCapacity,
  validateRequirement,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {GUARDED_ATAN2_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  GPU_SPHERICAL_CONVEX_INTERSECTION_MAXIMUM_CAPACITY,
  GPU_SPHERICAL_CONVEX_MAXIMUM_VERTICES,
  GPU_SPHERICAL_CONVEX_MINIMUM_VERTICES,
  GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
  GPU_SPHERICAL_MAXIMUM_TOLERANCE,
  GPU_SPHERICAL_MINIMUM_TOLERANCE,
  GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT,
  validateAngularTolerance
} from './spherical-limits';

/** Status bit: invalid, nonconvex or unsupported input in the row. */
export const GPU_SPHERICAL_CONVEX_INTERSECTION_STATUS_INVALID = 1;
/** Status bit: the intersection has more vertices than `capacity`; `counts` holds the required count. */
export const GPU_SPHERICAL_CONVEX_INTERSECTION_STATUS_CAPACITY = 2;
/** Status bit: the row is numerically ambiguous (near-parallel distinct edges or candidate overflow). */
export const GPU_SPHERICAL_CONVEX_INTERSECTION_STATUS_AMBIGUOUS = 4;

export type GPUSphericalConvexIntersectionProps = {
  id?: string;
  /**
   * Coordinate space of the unit vectors: longitude-latitude (their embedding on the sphere) with
   * great-circle edges. Planar contexts are rejected. Defaults to longitude-latitude, no metric.
   */
  spatialContext?: GPUSpatialContext;
  /** Unit direction vectors, counterclockwise minor great-circle arcs, no repeated closing vertex. */
  firstPositions: GraphDataView<'float32x3'>;
  secondPositions: GraphDataView<'float32x3'>;
  /** One polygon per paired row, each with 3–32 vertices. */
  firstOffsets: GraphDataView<'uint32'>;
  secondOffsets: GraphDataView<'uint32'>;
  capacity: number;
  /** Angular predicate/deduplication tolerance in radians, default 1e-6. */
  tolerance?: number;
  output: {
    /** Fixed row stride `capacity`, CCW output; unused slots are zero. */
    positions: GraphDataView<'float32x3'>;
    /** (written, required) counts. Incomplete output must not be interpreted as a polygon. */
    counts: GraphDataView<'uint32x2'>;
    /**
     * Per row, a bit set: {@link GPU_SPHERICAL_CONVEX_INTERSECTION_STATUS_INVALID} (1),
     * {@link GPU_SPHERICAL_CONVEX_INTERSECTION_STATUS_CAPACITY} (2),
     * {@link GPU_SPHERICAL_CONVEX_INTERSECTION_STATUS_AMBIGUOUS} (4); 0 is complete.
     */
    status: GraphDataView<'uint32'>;
    /** Optional (intersection steradians, source fraction, target fraction, union steradians). Invalid/overflow rows are NaN. */
    measures?: GraphDataView<'float32x4'>;
  };
};

/** Intersection of paired convex spherical polygons contained strictly in an open hemisphere.
 * Uses oriented 3D great-circle planes, so poles and antimeridian need no longitude seam.
 * Empty intersections have zero counts; boundary contacts may contain one or two vertices.
 */
export class GPUSphericalConvexIntersection implements GPUCommandNodeProducer {
  readonly id: string;
  /** Validated coordinate space of the unit vectors. */
  readonly spatialContext: GPUSpatialContext;
  constructor(readonly props: GPUSphericalConvexIntersectionProps) {
    this.id = props.id ?? 'spherical-convex-intersection';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext ?? GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
      GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT
    );
    for (const positions of [props.firstPositions, props.secondPositions, props.output.positions])
      validateSinglePackedView(positions, ['float32x3'], this.id);
    for (const offsets of [props.firstOffsets, props.secondOffsets, props.output.status])
      validateSinglePackedView(offsets, ['uint32'], this.id);
    validateSinglePackedView(props.output.counts, ['uint32x2'], this.id);
    if (props.output.measures) {
      validateSinglePackedView(props.output.measures, ['float32x4'], this.id);
      validateRequirement(
        this.id,
        props.output.measures.length === props.firstOffsets.length - 1,
        'output.measures must hold one entry per row'
      );
    }
    const rows = props.firstOffsets.length - 1;
    // Private candidate storage is bounded by the sum of the two convex polygon vertex counts.
    validateRequirement(
      this.id,
      rows >= 0 && props.secondOffsets.length === rows + 1,
      'firstOffsets and secondOffsets must describe the same number of rows'
    );
    validateCapacity(
      this.id,
      'capacity',
      props.capacity,
      1,
      GPU_SPHERICAL_CONVEX_INTERSECTION_MAXIMUM_CAPACITY
    );
    validateRequirement(
      this.id,
      props.output.positions.length === rows * props.capacity &&
        props.output.counts.length === rows &&
        props.output.status.length === rows,
      'output.positions must hold rows times capacity entries, and output.counts and output.status one entry per row'
    );
    validateAngularTolerance(
      this.id,
      props.tolerance ?? 1e-6,
      GPU_SPHERICAL_MINIMUM_TOLERANCE,
      GPU_SPHERICAL_MAXIMUM_TOLERANCE
    );
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.firstPositions,
      props.secondPositions,
      props.firstOffsets,
      props.secondOffsets
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.firstPositions,
      props.secondPositions,
      props.firstOffsets,
      props.secondOffsets,
      ...Object.values(props.output)
    ]);
    return [
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPUSphericalConvexIntersection',
        variant: 'closed-hemisphere-convex',
        invocationCount: props.firstOffsets.length - 1,
        bindings: [
          {name: 'firstPositions', view: props.firstPositions, type: 'f32', access: 'read'},
          {name: 'secondPositions', view: props.secondPositions, type: 'f32', access: 'read'},
          {name: 'firstOffsets', view: props.firstOffsets, type: 'u32', access: 'read'},
          {name: 'secondOffsets', view: props.secondOffsets, type: 'u32', access: 'read'},
          {name: 'output', view: props.output.positions, type: 'f32', access: 'read_write'},
          {name: 'counts', view: props.output.counts, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
          ...(props.output.measures
            ? [
                {
                  name: 'measures',
                  view: props.output.measures,
                  type: 'f32',
                  access: 'read_write'
                } satisfies WGSLKernelBinding
              ]
            : [])
        ],
        declarations: `const SPHERICAL_TOLERANCE: f32 = ${props.tolerance ?? 1e-6};
${GUARDED_ATAN2_WGSL}
fn spherePosition(second: bool, row: u32) -> vec3f {
  if (second) { let offset = secondPositionsOffset + row * 3u; return vec3f(secondPositions[offset], secondPositions[offset + 1u], secondPositions[offset + 2u]); }
  let offset = firstPositionsOffset + row * 3u; return vec3f(firstPositions[offset], firstPositions[offset + 1u], firstPositions[offset + 2u]);
}
fn sphereContains(second: bool, begin: u32, end: u32, point: vec3f) -> bool {
  for (var edge = begin; edge < end; edge++) {
    let next = select(edge + 1u, begin, edge + 1u == end);
    let normal = normalize(cross(spherePosition(second, edge), spherePosition(second, next)));
    if (dot(normal, point) < -SPHERICAL_TOLERANCE) { return false; }
  }
  return true;
}
fn sphereValid(second: bool, begin: u32, end: u32) -> bool {
  let capacity = select(${props.firstPositions.length}u, ${props.secondPositions.length}u, second);
  if (begin > end || end > capacity || end - begin < ${GPU_SPHERICAL_CONVEX_MINIMUM_VERTICES}u || end - begin > ${GPU_SPHERICAL_CONVEX_MAXIMUM_VERTICES}u) { return false; }
  var center = vec3f(0.0);
  for (var vertex = begin; vertex < end; vertex++) {
    let point = spherePosition(second, vertex);
    let squared = dot(point, point);
    if ((bitcast<u32>(squared) & 0x7f800000u) == 0x7f800000u || abs(squared - 1.0) > 1e-4) { return false; }
    let next = spherePosition(second, select(vertex + 1u, begin, vertex + 1u == end));
    if (length(cross(point, next)) <= SPHERICAL_TOLERANCE || !sphereContains(second, begin, end, point)) { return false; }
    center += point;
  }
  if (length(center) <= SPHERICAL_TOLERANCE) { return false; }
  center = normalize(center);
  for (var vertex = begin; vertex < end; vertex++) { if (dot(center, spherePosition(second, vertex)) <= SPHERICAL_TOLERANCE) { return false; } }
  for (var edge = begin; edge < end; edge++) {
    let next = select(edge + 1u, begin, edge + 1u == end);
    if (dot(center, normalize(cross(spherePosition(second, edge), spherePosition(second, next)))) <= SPHERICAL_TOLERANCE) { return false; }
  }
  return true;
}
fn sphereTriangleArea(first: vec3f, second: vec3f, third: vec3f) -> f32 {
  return 2.0 * atan2(abs(dot(first, cross(second - first, third - first))), 1.0 + dot(first, second) + dot(second, third) + dot(third, first));
}
fn spherePolygonArea(second: bool, begin: u32, end: u32) -> f32 {
  var area = 0.0;
  for (var vertex = begin + 1u; vertex + 1u < end; vertex++) { area += sphereTriangleArea(spherePosition(second, begin), spherePosition(second, vertex), spherePosition(second, vertex + 1u)); }
  return area;
}
fn sphereOnArc(first: vec3f, second: vec3f, point: vec3f) -> bool {
  let normal = normalize(cross(first, second));
  return dot(cross(first, point), normal) >= -SPHERICAL_TOLERANCE && dot(cross(point, second), normal) >= -SPHERICAL_TOLERANCE;
}
fn sphereAppend(candidates: ptr<function, array<vec3f, ${GPU_SPHERICAL_CONVEX_INTERSECTION_MAXIMUM_CAPACITY}>>, count: ptr<function, u32>, point: vec3f) {
  for (var previous = 0u; previous < min(*count, ${GPU_SPHERICAL_CONVEX_INTERSECTION_MAXIMUM_CAPACITY}u); previous++) { if (length((*candidates)[previous] - point) <= SPHERICAL_TOLERANCE * 4.0) { return; } }
  if (*count < ${GPU_SPHERICAL_CONVEX_INTERSECTION_MAXIMUM_CAPACITY}u) { (*candidates)[*count] = point; }
  *count += 1u;
}`,
        body: `let firstBegin = firstOffsets[firstOffsetsOffset + index]; let firstEnd = firstOffsets[firstOffsetsOffset + index + 1u];
let secondBegin = secondOffsets[secondOffsetsOffset + index]; let secondEnd = secondOffsets[secondOffsetsOffset + index + 1u];
var failure = 0u; var count = 0u; var candidates: array<vec3f, ${GPU_SPHERICAL_CONVEX_INTERSECTION_MAXIMUM_CAPACITY}>;
if (!sphereValid(false, firstBegin, firstEnd) || !sphereValid(true, secondBegin, secondEnd)) { failure = 1u; }
if (failure == 0u) {
  for (var vertex = firstBegin; vertex < firstEnd; vertex++) { let point = spherePosition(false, vertex); if (sphereContains(true, secondBegin, secondEnd, point)) { sphereAppend(&candidates, &count, point); } }
  for (var vertex = secondBegin; vertex < secondEnd; vertex++) { let point = spherePosition(true, vertex); if (sphereContains(false, firstBegin, firstEnd, point)) { sphereAppend(&candidates, &count, point); } }
  for (var firstEdge = firstBegin; firstEdge < firstEnd; firstEdge++) {
    let firstStart = spherePosition(false, firstEdge); let firstStop = spherePosition(false, select(firstEdge + 1u, firstBegin, firstEdge + 1u == firstEnd));
    let firstNormal = normalize(cross(firstStart, firstStop));
    for (var secondEdge = secondBegin; secondEdge < secondEnd; secondEdge++) {
      let secondStart = spherePosition(true, secondEdge); let secondStop = spherePosition(true, select(secondEdge + 1u, secondBegin, secondEdge + 1u == secondEnd));
      let secondNormal = normalize(cross(secondStart, secondStop));
      let crossing = cross(firstNormal, secondNormal); let magnitude = length(crossing);
      // Coincident planes are handled by contained endpoints. Near-parallel distinct planes are uncertain.
      if (magnitude <= SPHERICAL_TOLERANCE) {
        if (abs(dot(firstNormal, secondStart)) > SPHERICAL_TOLERANCE) { failure |= 4u; }
        continue;
      }
      for (var side = 0u; side < 2u; side++) {
        let point = crossing / magnitude * select(1.0, -1.0, side == 1u);
        if (sphereOnArc(firstStart, firstStop, point) && sphereOnArc(secondStart, secondStop, point) && sphereContains(false, firstBegin, firstEnd, point) && sphereContains(true, secondBegin, secondEnd, point)) { sphereAppend(&candidates, &count, point); }
      }
    }
  }
}
if (count > ${GPU_SPHERICAL_CONVEX_INTERSECTION_MAXIMUM_CAPACITY}u) { failure |= 4u; }
if (count > ${props.capacity}u) { failure |= 2u; }
var center = vec3f(0.0);
for (var vertex = 0u; vertex < min(count, ${GPU_SPHERICAL_CONVEX_INTERSECTION_MAXIMUM_CAPACITY}u); vertex++) { center += candidates[vertex]; }
if (count > 2u && length(center) > SPHERICAL_TOLERANCE) {
  center = normalize(center);
  let tangent = normalize(candidates[0] - center * dot(center, candidates[0]));
  let bitangent = cross(center, tangent);
  for (var vertex = 1u; vertex < min(count, ${GPU_SPHERICAL_CONVEX_INTERSECTION_MAXIMUM_CAPACITY}u); vertex++) {
    let point = candidates[vertex]; let angle = guardedAtan2(dot(point, bitangent), dot(point, tangent)); var insertion = vertex;
    while (insertion > 0u) {
      let previous = candidates[insertion - 1u];
      if (guardedAtan2(dot(previous, bitangent), dot(previous, tangent)) <= angle) { break; }
      candidates[insertion] = previous; insertion -= 1u;
    }
    candidates[insertion] = point;
  }
}
let written = select(min(count, ${props.capacity}u), 0u, (failure & 5u) != 0u);
for (var slot = 0u; slot < ${props.capacity}u; slot++) {
  var point = vec3f(0.0); if (slot < written) { point = candidates[slot]; }
  let offset = outputOffset + (index * ${props.capacity}u + slot) * 3u;
  output[offset] = point.x; output[offset + 1u] = point.y; output[offset + 2u] = point.z;
}
counts[countsOffset + index * 2u] = written; counts[countsOffset + index * 2u + 1u] = count;
status[statusOffset + index] = failure;
${
  props.output.measures
    ? `var intersectionArea = 0.0; var sourceFraction = 0.0; var targetFraction = 0.0; var unionArea = 0.0;
if (failure == 0u) {
  for (var vertex = 1u; vertex + 1u < count; vertex++) { intersectionArea += sphereTriangleArea(candidates[0], candidates[vertex], candidates[vertex + 1u]); }
  let firstArea = spherePolygonArea(false, firstBegin, firstEnd); let secondArea = spherePolygonArea(true, secondBegin, secondEnd);
  sourceFraction = intersectionArea / firstArea; targetFraction = intersectionArea / secondArea; unionArea = firstArea + secondArea - intersectionArea;
} else { intersectionArea = bitcast<f32>(0x7fc00000u | (index & 0u)); sourceFraction = intersectionArea; targetFraction = intersectionArea; unionArea = intersectionArea; }
measures[measuresOffset + index * 4u] = intersectionArea; measures[measuresOffset + index * 4u + 1u] = sourceFraction; measures[measuresOffset + index * 4u + 2u] = targetFraction; measures[measuresOffset + index * 4u + 3u] = unionArea;`
    : ''
}`
      })
    ];
  }
}
