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
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getReadPointWGSL} from '../shared/spatial-wgsl-point-helpers';

export type GPUConvexStraightSkeletonProps = {
  id?: string;
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  /** One strictly convex simple polygon per run, implicit or explicit closure. */
  ringOffsets: GraphDataView<'uint32'>;
  maximumVertices?: number;
  maximumIterations?: number;
  /** Inward parallel offset requested in planar units. */
  offsetDistance: number;
  /** Relative simultaneous-event tolerance, default 1e-5. */
  eventTolerance?: number;
  /** Number of skeleton arcs reserved per polygon; default twice maximumVertices. */
  segmentCapacity?: number;
  output: {
    /** Per-polygon blocks of segmentCapacity (startX,startY,endX,endY) arcs. */
    segments: GraphDataView<'float32x4'>;
    /** Original left/right wavefront edge IDs for each arc; terminal ridge uses UINT_MAX. */
    sourceEdges: GraphDataView<'uint32x2'>;
    /** Per-polygon blocks of maximumVertices for the requested inward offset. */
    offsetPositions: GraphDataView<'float32x2'>;
    /** [writtenArcs, requiredArcs, offsetVertexCount, flags]: unsupported=1, capacity=2, budget=4, numerical=8. */
    status: GraphDataView<'uint32x4'>;
  };
};

/**
 * Vector wavefront prototype for independent strictly convex polygons. Parallel
 * edges move inward at unit speed; adjacent intersections trace skeleton arcs.
 * Simultaneous edge collapses are handled together, including rectangle ridges.
 * Nonconvex polygons, holes and collinear corners are explicitly unsupported;
 * they require split events and general half-edge topology, not this prototype.
 */
export class GPUConvexStraightSkeleton implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUConvexStraightSkeletonProps;
  readonly spatialContext: GPUSpatialContext;
  readonly maximumVertices: number;
  readonly maximumIterations: number;
  readonly segmentCapacity: number;
  constructor(props: GPUConvexStraightSkeletonProps) {
    this.id = props.id ?? 'convex-straight-skeleton';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    this.maximumVertices = props.maximumVertices ?? 32;
    this.maximumIterations = props.maximumIterations ?? this.maximumVertices;
    this.segmentCapacity = props.segmentCapacity ?? this.maximumVertices * 2;
    validateCapacity(this.id, 'maximumVertices', this.maximumVertices, 3, 64);
    validateCapacity(this.id, 'maximumIterations', this.maximumIterations, 0, 128);
    validateCapacity(this.id, 'segmentCapacity', this.segmentCapacity, 1);
    if (
      !(
        Number.isFinite(props.offsetDistance) &&
        props.offsetDistance >= 0 &&
        Number.isFinite(props.eventTolerance ?? 1e-5) &&
        (props.eventTolerance ?? 1e-5) > 0
      )
    ) {
      throw new Error('Invalid wavefront parameters');
    }
    validateSinglePackedView(props.positions, ['float32x2'], this.id);
    validateSinglePackedView(props.ringOffsets, ['uint32'], this.id);
    validateSinglePackedView(props.output.segments, ['float32x4'], this.id);
    validateSinglePackedView(props.output.sourceEdges, ['uint32x2'], this.id);
    validateSinglePackedView(props.output.offsetPositions, ['float32x2'], this.id);
    validateSinglePackedView(props.output.status, ['uint32x4'], this.id);
    const rings = props.ringOffsets.length - 1;
    if (
      props.output.segments.length !== rings * this.segmentCapacity ||
      props.output.sourceEdges.length !== props.output.segments.length ||
      props.output.offsetPositions.length !== rings * this.maximumVertices ||
      props.output.status.length !== rings
    ) {
      throw new Error('Skeleton output capacity mismatch');
    }
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      props.ringOffsets
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, maximumVertices, segmentCapacity, maximumIterations} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.ringOffsets,
      ...Object.values(props.output)
    ]);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-events`,
        operation: 'GPUConvexStraightSkeleton',
        variant: 'convex-wavefront',
        invocationCount: props.ringOffsets.length - 1,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'offsets', view: props.ringOffsets, type: 'u32', access: 'read'},
          {name: 'segments', view: props.output.segments, type: 'f32', access: 'read_write'},
          {name: 'sourceEdges', view: props.output.sourceEdges, type: 'u32', access: 'read_write'},
          {
            name: 'offsetPositions',
            view: props.output.offsetPositions,
            type: 'f32',
            access: 'read_write'
          },
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `const MAXIMUM_VERTICES: u32 = ${maximumVertices}u;
const SEGMENT_CAPACITY: u32 = ${segmentCapacity}u;
const REQUESTED_DISTANCE: f32 = ${getWGSLFloatLiteral(props.offsetDistance)};
const EVENT_TOLERANCE: f32 = ${getWGSLFloatLiteral(props.eventTolerance ?? 1e-5)};
${getReadPointWGSL('point', 'positions')}
fn cross(left: vec2f, right: vec2f) -> f32 { return left.x * right.y - left.y * right.x; }
fn intersection(left: vec3f, right: vec3f, time: f32) -> vec2f {
  let determinant = cross(left.xy, right.xy);
  return vec2f((left.z + time) * right.y - left.y * (right.z + time), left.x * (right.z + time) - (left.z + time) * right.x) / determinant;
}
fn writeArc(polygon: u32, slot: u32, start: vec2f, end: vec2f, left: u32, right: u32) {
  if (slot >= SEGMENT_CAPACITY) { return; }
  let destination = polygon * SEGMENT_CAPACITY + slot;
  segments[segmentsOffset + destination * 4u] = start.x; segments[segmentsOffset + destination * 4u + 1u] = start.y;
  segments[segmentsOffset + destination * 4u + 2u] = end.x; segments[segmentsOffset + destination * 4u + 3u] = end.y;
  sourceEdges[sourceEdgesOffset + destination * 2u] = left; sourceEdges[sourceEdgesOffset + destination * 2u + 1u] = right;
}`,
        body: `let first = min(offsets[offsetsOffset + index], ${props.positions.length}u);
let last = min(offsets[offsetsOffset + index + 1u], ${props.positions.length}u);
var count = last - first;
if (count > 1u && all(point(first) == point(last - 1u))) { count--; }
var flags = 0u; var required = 0u; var offsetCount = 0u; var offsetWritten = false;
if (count < 3u || count > MAXIMUM_VERTICES) { flags = 1u; }
var edges: array<vec3f, MAXIMUM_VERTICES>;
var edgeIds: array<u32, MAXIMUM_VERTICES>;
var births: array<vec2f, MAXIMUM_VERTICES>;
var orientation = 0.0;
if (flags == 0u) {
  orientation = sign(cross(point(first + 1u) - point(first), point(first + 2u) - point(first + 1u)));
  for (var vertex = 0u; vertex < count; vertex++) {
    let nextVertexIndex = (vertex + 1u) % count;
    let current = point(first + vertex); let next = point(first + nextVertexIndex);
    if (any((bitcast<vec2u>(current) & vec2u(0x7f800000u)) == vec2u(0x7f800000u))) { flags |= 8u; }
    let tangent = next - current;
    for (var otherVertexIndex = 0u; otherVertexIndex < count; otherVertexIndex++) {
      // The edge endpoints are on its own supporting line by definition. With
      // slanted float32 edges, recomputing that cross product can round slightly
      // negative and falsely reject an otherwise strictly convex polygon.
      if (otherVertexIndex == vertex || otherVertexIndex == nextVertexIndex) { continue; }
      if (cross(tangent, point(first + otherVertexIndex) - current) * orientation < 0.0) { flags |= 1u; }
    }
    let following = point(first + (vertex + 2u) % count) - next;
    if (cross(tangent, following) * orientation <= 0.0) { flags |= 1u; }
    let normal = orientation * vec2f(-tangent.y, tangent.x) / length(tangent);
    edges[vertex] = vec3f(normal, dot(normal, current)); edgeIds[vertex] = first + vertex; births[vertex] = current;
  }
}
var time = 0.0; var iterations = 0u;
loop {
  if (flags != 0u || count < 3u) { break; }
  var eventTimes: array<f32, MAXIMUM_VERTICES>;
  var eventPositions: array<vec2f, MAXIMUM_VERTICES>;
  var nextTime = 3.0e38;
  for (var vertex = 0u; vertex < count; vertex++) {
    let previous = (vertex + count - 1u) % count; let next = (vertex + 1u) % count;
    let currentPoint = intersection(edges[previous], edges[vertex], time);
    let nextPoint = intersection(edges[vertex], edges[next], time);
    let currentVelocity = intersection(vec3f(edges[previous].xy, 0.0), vec3f(edges[vertex].xy, 0.0), 1.0);
    let nextVelocity = intersection(vec3f(edges[vertex].xy, 0.0), vec3f(edges[next].xy, 0.0), 1.0);
    let tangent = orientation * vec2f(edges[vertex].y, -edges[vertex].x);
    let shrinkRate = dot(nextVelocity - currentVelocity, tangent);
    eventTimes[vertex] = 3.0e38;
    if (shrinkRate < 0.0) { eventTimes[vertex] = max(time, time - dot(nextPoint - currentPoint, tangent) / shrinkRate); }
    nextTime = min(nextTime, eventTimes[vertex]);
  }
  if (nextTime == 3.0e38) { flags |= 8u; break; }
  if (!offsetWritten && REQUESTED_DISTANCE < nextTime) {
    offsetCount = count; offsetWritten = true;
    for (var vertex = 0u; vertex < count; vertex++) {
      let current = intersection(edges[(vertex + count - 1u) % count], edges[vertex], REQUESTED_DISTANCE);
      let destination = (index * MAXIMUM_VERTICES + vertex) * 2u;
      offsetPositions[offsetPositionsOffset + destination] = current.x; offsetPositions[offsetPositionsOffset + destination + 1u] = current.y;
    }
  }
  if (iterations >= ${maximumIterations}u) { flags |= 4u; break; }
  let epsilon = EVENT_TOLERANCE * max(1.0, abs(nextTime));
  var removed: array<u32, MAXIMUM_VERTICES>; var survivors = 0u;
  for (var vertex = 0u; vertex < count; vertex++) {
    removed[vertex] = select(0u, 1u, eventTimes[vertex] <= nextTime + epsilon);
    if (removed[vertex] == 0u) { survivors++; }
    eventPositions[vertex] = intersection(edges[(vertex + count - 1u) % count], edges[vertex], nextTime);
  }
  for (var vertex = 0u; vertex < count; vertex++) {
    let previous = (vertex + count - 1u) % count;
    if (removed[previous] != 0u || removed[vertex] != 0u || survivors < 3u) {
      if (distance(births[vertex], eventPositions[vertex]) > epsilon) {
        writeArc(index, required, births[vertex], eventPositions[vertex], edgeIds[previous], edgeIds[vertex]); required++;
      }
    }
  }
  if (survivors < 3u) {
    // Simultaneous opposing collapses leave a terminal ridge (e.g. a rectangle).
    var left = eventPositions[0]; var right = left; var longest = 0.0;
    for (var vertex = 0u; vertex < count; vertex++) {
      for (var other = vertex + 1u; other < count; other++) {
        let separation = distance(eventPositions[vertex], eventPositions[other]);
        if (separation > longest) { longest = separation; left = eventPositions[vertex]; right = eventPositions[other]; }
      }
    }
    if (longest > epsilon) { writeArc(index, required, left, right, 0xffffffffu, 0xffffffffu); required++; }
    count = 0u; time = nextTime; break;
  }
  var retained: array<u32, MAXIMUM_VERTICES>; var kept = 0u;
  for (var vertex = 0u; vertex < count; vertex++) { if (removed[vertex] == 0u) { retained[kept] = vertex; kept++; } }
  // Event positions are no longer needed after arc emission. Reuse them for the next
  // births before compacting edges in place; ascending retained IDs never read an overwritten edge.
  for (var vertex = 0u; vertex < survivors; vertex++) {
    let current = retained[vertex]; let previous = retained[(vertex + survivors - 1u) % survivors];
    if ((previous + 1u) % count == current) { eventPositions[vertex] = births[current]; }
    else { eventPositions[vertex] = intersection(edges[previous], edges[current], nextTime); }
  }
  for (var vertex = 0u; vertex < survivors; vertex++) {
    edges[vertex] = edges[retained[vertex]]; edgeIds[vertex] = edgeIds[retained[vertex]]; births[vertex] = eventPositions[vertex];
  }
  count = survivors; time = nextTime; iterations++;
}
if (required > SEGMENT_CAPACITY) { flags |= 2u; }
status[statusOffset + index * 4u] = min(required, SEGMENT_CAPACITY); status[statusOffset + index * 4u + 1u] = required;
status[statusOffset + index * 4u + 2u] = offsetCount; status[statusOffset + index * 4u + 3u] = flags;`
      })
    ];
  }
}
