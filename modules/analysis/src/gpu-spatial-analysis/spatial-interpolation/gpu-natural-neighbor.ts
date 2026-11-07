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
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  validateCapacity,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getPackedPredictionAccessors,
  validatePredictionViews
} from '../../analysis-core/kernels/gpu-prediction-utils';

/** Bounded exact Sibson interpolation in 2D planar coordinates. */
export type GPUNaturalNeighborProps = {
  id?: string;
  positions: GraphDataView<'float32x2'>;
  values: GraphDataView<'float32'>;
  queries: GraphDataView<'float32x2'>;
  /** Maximum polygon vertices in [3, 64]; a deliberately smaller budget exposes exhaustion. */
  maximumVertexCount?: number;
  /** Optional GPU-resident Delaunay index: six u32 per triangle [a,b,c,neighborAB,neighborBC,neighborCA]; boundary=0xffffffff. */
  triangles?: GraphDataView<'uint32'>;
  /** One containing triangle per query; 0xffffffff explicitly marks outside-hull queries. */
  queryTriangles?: GraphDataView<'uint32'>;
  /** Bounded natural-neighbor cavity vertices, 3..64; default 32. */
  maximumNeighborCount?: number;
  output: {
    /** [prediction, query Voronoi area in position units squared, weight sum, contributing count]. */
    values: GraphDataView<'float32x4'>;
    /** 0 success; 1 outside hull/invalid; 2 duplicate/collinear/numerically degenerate; 3 capacity. */
    status: GraphDataView<'uint32'>;
    /** Without index: scalar weights in sample order. Indexed: float32x2 [sampleIndex,weight] in cavity order, unused [-1,0]. */
    weights?: GraphDataView<'float32' | 'float32x2'>;
  };
};

/**
 * Computes true stolen-Voronoi-area coordinates, independently of triangulation implementation.
 * The direct path uses at most 64 samples. The indexed path traverses a supplied GPU-resident
 * Delaunay cavity from a containing-triangle seed, with up to 64 natural neighbors and 256
 * inspected triangles. Original Voronoi cells clip the query cell, giving stolen-area weights.
 * The index is caller-owned valid planar Delaunay topology (unique sites, reciprocal adjacency);
 * seed location/build cost is separate from interpolation. Each query uses bounded local work,
 * independent of the full site count. Float32 output indices limit indexed sites to2^24.
 * Outside-hull queries are invalid; hull edges use the limiting linear
 * weights of adjacent collinear edge samples. Sample hits reproduce samples (duplicate hits
 * average); duplicates away from the hit and collinear sets report degeneracy. Nonfinite input
 * rows make a query invalid. Coordinates are query-centered and scaled before geometry.
 */
export class GPUNaturalNeighbor implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNaturalNeighborProps;
  readonly maximumVertexCount: number;
  readonly maximumNeighborCount: number;
  constructor(props: GPUNaturalNeighborProps) {
    this.props = props;
    this.id = props.id ?? 'natural-neighbor';
    this.maximumVertexCount = props.maximumVertexCount ?? 64;
    this.maximumNeighborCount = props.maximumNeighborCount ?? 32;
    validateCapacity(this.id, 'maximumNeighborCount', this.maximumNeighborCount, 3);
    validateCapacity(this.id, 'maximumVertexCount', this.maximumVertexCount, 3);
    if (
      this.maximumVertexCount > 64 ||
      props.positions.length > 0x1000000 ||
      (props.output.weights &&
        props.output.weights.format !== (props.triangles ? 'float32x2' : 'float32')) ||
      (!props.triangles && props.positions.length > this.maximumNeighborCount) ||
      this.maximumNeighborCount > 64 ||
      Boolean(props.triangles) !== Boolean(props.queryTriangles) ||
      (props.triangles &&
        (props.triangles.length % 6 !== 0 ||
          props.queryTriangles!.length !== props.queries.length)) ||
      props.positions.length !== props.values.length ||
      props.output.values.length < props.queries.length ||
      props.output.status.length < props.queries.length ||
      (props.output.weights &&
        props.output.weights.length <
          props.queries.length *
            (props.triangles ? this.maximumNeighborCount : props.positions.length))
    ) {
      throw new Error(`${this.id} invalid capacities`);
    }
    validatePredictionViews(
      this.id,
      [
        props.positions,
        props.values,
        props.queries,
        ...(props.triangles ? [props.triangles, props.queryTriangles!] : [])
      ],
      [
        props.output.values,
        props.output.status,
        ...(props.output.weights ? [props.output.weights] : [])
      ]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maximumVertexCount} = this;
    const sampleCapacity = props.triangles ? this.maximumNeighborCount : props.positions.length;
    const sampleCount = sampleCapacity;
    const bindings: WGSLKernelBinding[] = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'values', view: props.values, type: 'f32', access: 'read'},
      {name: 'queries', view: props.queries, type: 'f32', access: 'read'},
      {name: 'results', view: props.output.values, type: 'f32', access: 'read_write'},
      {name: 'statuses', view: props.output.status, type: 'u32', access: 'read_write'}
    ];
    if (props.triangles) {
      bindings.push(
        {name: 'triangles', view: props.triangles, type: 'u32', access: 'read'},
        {name: 'queryTriangles', view: props.queryTriangles!, type: 'u32', access: 'read'}
      );
    }
    if (props.output.weights) {
      bindings.push({
        name: 'weights',
        view: props.output.weights,
        type: 'f32',
        access: 'read_write'
      });
    }
    validateGraphViewsBelongToGraph(
      id,
      graph,
      bindings.map(binding => binding.view)
    );
    return [
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPUNaturalNeighbor',
        variant: 'sibson-area',
        bindings,
        invocationCount: props.queries.length,
        workgroupSize: 2,
        declarations: `${getPackedPredictionAccessors({positions: 2, queries: 2}, ['results'])}\n${GET_NAN_WGSL}\n${IS_FINITE_VALUE_WGSL}
const SAMPLE_CAPACITY: u32 = ${sampleCount}u;
const VERTEX_CAPACITY: u32 = ${maximumVertexCount}u;
${props.output.weights ? `fn writeWeight(queryIndex: u32, slot: u32, sampleIndex: u32, weight: f32) { ${props.triangles ? 'weights[weightsOffset + queryIndex * SAMPLE_CAPACITY * 2u + slot * 2u] = f32(sampleIndex); weights[weightsOffset + queryIndex * SAMPLE_CAPACITY * 2u + slot * 2u + 1u] = weight;' : 'weights[weightsOffset + queryIndex * SAMPLE_CAPACITY + slot] = weight;'} }` : ''}
fn crossProduct(first: vec2<f32>, second: vec2<f32>) -> f32 { return first.x * second.y - first.y * second.x; }
var<private> polygon: array<vec2<f32>, ${maximumVertexCount}>;
var<private> clipped: array<vec2<f32>, ${maximumVertexCount}>;
var<private> polygonCount: u32;
fn clipPolygon(normal: vec2<f32>, limit: f32) -> bool {
  var count = 0u;
  for (var vertex = 0u; vertex < polygonCount; vertex++) {
    let start = polygon[vertex]; let finish = polygon[(vertex + 1u) % polygonCount];
    let startDistance = dot(normal, start) - limit; let finishDistance = dot(normal, finish) - limit;
    if (startDistance <= 0.0) { if (count >= VERTEX_CAPACITY) { return false; } clipped[count] = start; count++; }
    if ((startDistance < 0.0 && finishDistance > 0.0) || (startDistance > 0.0 && finishDistance < 0.0)) {
      if (count >= VERTEX_CAPACITY) { return false; }
      clipped[count] = start + (finish - start) * (startDistance / (startDistance - finishDistance)); count++;
    }
  }
  polygonCount = count; for (var vertex = 0u; vertex < count; vertex++) { polygon[vertex] = clipped[vertex]; }
  return true;
}
fn polygonArea() -> f32 {
  var area = 0.0; for (var vertex = 0u; vertex < polygonCount; vertex++) { area += crossProduct(polygon[vertex], polygon[(vertex + 1u) % polygonCount]); }
  return abs(area) * 0.5;
}`,
        body: `
write_results(index, vec4<f32>(getNaN(), 0.0, 0.0, 0.0)); statuses[statusesOffset + index] = 1u;
${props.output.weights ? `for (var sample = 0u; sample < SAMPLE_CAPACITY; sample++) { weights[weightsOffset + index * SAMPLE_CAPACITY * ${props.triangles ? 2 : 1}u + sample * ${props.triangles ? 2 : 1}u] = ${props.triangles ? '-1.0' : '0.0'}; ${props.triangles ? 'weights[weightsOffset + index * SAMPLE_CAPACITY * 2u + sample * 2u + 1u] = 0.0;' : ''} }` : ''}
let query = read_queries(index); if (!isFiniteValue(query.x) || !isFiniteValue(query.y)) { return; }
var sampleIndices: array<u32, ${Math.max(sampleCapacity, 1)}>;
var sampleCount = ${props.triangles ? 0 : sampleCount}u;
${
  props.triangles
    ? `
let seedTriangle = queryTriangles[queryTrianglesOffset + index];
if (seedTriangle >= ${props.triangles.length / 6}u) { return; }
var triangleQueue: array<u32, ${this.maximumNeighborCount * 4}>;
triangleQueue[0] = seedTriangle; var queueCount = 1u; var queueCursor = 0u;
loop {
  if (queueCursor >= queueCount) { break; }
  let triangle = triangleQueue[queueCursor]; queueCursor++;
  let triangleStart = trianglesOffset + triangle * 6u;
  let firstIndex = triangles[triangleStart]; let secondIndex = triangles[triangleStart + 1u]; let thirdIndex = triangles[triangleStart + 2u];
  if (firstIndex >= ${props.positions.length}u || secondIndex >= ${props.positions.length}u || thirdIndex >= ${props.positions.length}u) { statuses[statusesOffset + index] = 2u; return; }
  var first = read_positions(firstIndex) - query; var second = read_positions(secondIndex) - query; var third = read_positions(thirdIndex) - query;
  let triangleScale = max(max(length(first), length(second)), length(third));
  if (!(triangleScale > 0.0) || !isFiniteValue(triangleScale)) { statuses[statusesOffset + index] = 2u; return; }
  first /= triangleScale; second /= triangleScale; third /= triangleScale;
  let orientation = crossProduct(second - first, third - first);
  if (abs(orientation) < 1e-9) { statuses[statusesOffset + index] = 2u; return; }
  if (triangle == seedTriangle) {
    let sideFirst = crossProduct(second - first, -first) * sign(orientation);
    let sideSecond = crossProduct(third - second, -second) * sign(orientation);
    let sideThird = crossProduct(first - third, -third) * sign(orientation);
    if (min(min(sideFirst, sideSecond), sideThird) < -1e-6) { return; }
  }
  let inCircle = (dot(first, first) * crossProduct(second, third) - dot(second, second) * crossProduct(first, third) + dot(third, third) * crossProduct(first, second)) * sign(orientation);
  if (inCircle < -1e-7) { continue; }
  for (var corner = 0u; corner < 3u; corner++) {
    let sampleIndex = triangles[triangleStart + corner]; var duplicate = false;
    for (var previous = 0u; previous < sampleCount; previous++) { duplicate = duplicate || sampleIndices[previous] == sampleIndex; }
    if (!duplicate) {
      if (sampleCount >= SAMPLE_CAPACITY) { statuses[statusesOffset + index] = 3u; return; }
      sampleIndices[sampleCount] = sampleIndex; sampleCount++;
    }
    let adjacent = triangles[triangleStart + 3u + corner];
    if (adjacent == 0xffffffffu) { continue; }
    if (adjacent >= ${props.triangles.length / 6}u) { statuses[statusesOffset + index] = 2u; return; }
    var visited = false; for (var previous = 0u; previous < queueCount; previous++) { visited = visited || triangleQueue[previous] == adjacent; }
    if (!visited) {
      if (queueCount >= ${this.maximumNeighborCount * 4}u) { statuses[statusesOffset + index] = 3u; return; }
      triangleQueue[queueCount] = adjacent; queueCount++;
    }
  }
}
`
    : 'for (var sample = 0u; sample < sampleCount; sample++) { sampleIndices[sample] = sample; }'
}
var points: array<vec2<f32>, ${Math.max(sampleCount, 1)}>;
var scale = 0.0; var hitCount = 0u; var hitValue = 0.0;
for (var sample = 0u; sample < sampleCount; sample++) {
  let point = read_positions(sampleIndices[sample]); let value = values[valuesOffset + sampleIndices[sample]];
  if (!isFiniteValue(point.x) || !isFiniteValue(point.y) || !isFiniteValue(value)) { return; }
  points[sample] = point - query; scale = max(scale, length(points[sample]));
  if (all(point == query)) { hitCount++; }
}
if (hitCount > 0u) {
  for (var sample = 0u; sample < sampleCount; sample++) { if (all(points[sample] == vec2<f32>(0.0))) { hitValue += values[valuesOffset + sampleIndices[sample]] / f32(hitCount); } }
  if (!isFiniteValue(hitValue)) { statuses[statusesOffset + index] = 2u; return; }
  write_results(index, vec4<f32>(hitValue, 0.0, 1.0, f32(hitCount))); statuses[statusesOffset + index] = 0u;
  ${props.output.weights ? 'for (var sample = 0u; sample < sampleCount; sample++) { if (all(points[sample] == vec2<f32>(0.0))) { writeWeight(index, sample, sampleIndices[sample], 1.0 / f32(hitCount)); } }' : ''}
  return;
}
if (sampleCount < 3u || !(scale > 0.0)) { statuses[statusesOffset + index] = 2u; return; }
for (var sample = 0u; sample < sampleCount; sample++) { points[sample] /= scale; }
var noncollinear = false; var edgeFirst = 0u; var edgeSecond = 0u; var edgeLength = 3.402823e38;
for (var first = 0u; first < sampleCount; first++) {
  for (var second = first + 1u; second < sampleCount; second++) {
    let direction = points[second] - points[first]; let magnitude = length(direction);
    if (magnitude <= 1e-7) { statuses[statusesOffset + index] = 2u; return; }
    var positive = false; var negative = false;
    for (var other = 0u; other < sampleCount; other++) {
      let side = crossProduct(direction, points[other] - points[first]); positive = positive || side > 1e-7; negative = negative || side < -1e-7;
    }
    noncollinear = noncollinear || positive || negative;
    if (positive && negative) { continue; }
    let side = crossProduct(direction, -points[first]);
    if ((positive && side < -1e-7) || (negative && side > 1e-7)) { return; }
    if (abs(side) <= 1e-7 && dot(-points[first], -points[second]) <= 0.0 && magnitude < edgeLength) { edgeFirst = first; edgeSecond = second; edgeLength = magnitude; }
  }
}
if (!noncollinear) { statuses[statusesOffset + index] = 2u; return; }
if (edgeLength < 3.402823e38) {
  let fraction = length(points[edgeFirst]) / edgeLength;
  write_results(index, vec4<f32>(mix(values[valuesOffset + sampleIndices[edgeFirst]], values[valuesOffset + sampleIndices[edgeSecond]], fraction), 0.0, 1.0, 2.0)); statuses[statusesOffset + index] = 0u;
  ${props.output.weights ? 'writeWeight(index, edgeFirst, sampleIndices[edgeFirst], 1.0 - fraction); writeWeight(index, edgeSecond, sampleIndices[edgeSecond], fraction);' : ''}
  return;
}
var queryPolygon: array<vec2<f32>, ${maximumVertexCount}>; var vertexCount = 0u;
for (var first = 0u; first < sampleCount; first++) {
  for (var second = first + 1u; second < sampleCount; second++) {
    let firstPoint = points[first]; let secondPoint = points[second]; let determinant = crossProduct(firstPoint, secondPoint);
    if (abs(determinant) <= 1e-9) { continue; }
    let firstLimit = dot(firstPoint, firstPoint) * 0.5; let secondLimit = dot(secondPoint, secondPoint) * 0.5;
    let vertex = vec2<f32>(firstLimit * secondPoint.y - secondLimit * firstPoint.y, firstPoint.x * secondLimit - secondPoint.x * firstLimit) / determinant;
    var feasible = true; for (var other = 0u; other < sampleCount; other++) { feasible = feasible && dot(points[other], vertex) <= dot(points[other], points[other]) * 0.5 + 1e-6; }
    if (!feasible) { continue; }
    var duplicate = false; for (var previous = 0u; previous < vertexCount; previous++) { duplicate = duplicate || distance(vertex, queryPolygon[previous]) <= 1e-6; }
    if (duplicate) { continue; }
    if (vertexCount >= VERTEX_CAPACITY) { statuses[statusesOffset + index] = 3u; return; }
    queryPolygon[vertexCount] = vertex; vertexCount++;
  }
}
if (vertexCount < 3u) { statuses[statusesOffset + index] = 2u; return; }
for (var vertex = 1u; vertex < vertexCount; vertex++) {
  let point = queryPolygon[vertex]; let angle = atan2(point.y, point.x); var insertion = vertex;
  loop { if (insertion == 0u) { break; } let previous = queryPolygon[insertion - 1u]; if (atan2(previous.y, previous.x) <= angle) { break; } queryPolygon[insertion] = previous; insertion--; }
  queryPolygon[insertion] = point;
}
polygonCount = vertexCount; for (var vertex = 0u; vertex < vertexCount; vertex++) { polygon[vertex] = queryPolygon[vertex]; }
let queryArea = polygonArea(); if (!(queryArea > 1e-12) || !isFiniteValue(queryArea)) { statuses[statusesOffset + index] = 2u; return; }
var areas: array<f32, ${Math.max(sampleCount, 1)}>; var areaSum = 0.0; var prediction = 0.0; var contributing = 0u;
for (var sample = 0u; sample < sampleCount; sample++) {
  polygonCount = vertexCount; for (var vertex = 0u; vertex < vertexCount; vertex++) { polygon[vertex] = queryPolygon[vertex]; }
  for (var other = 0u; other < sampleCount; other++) {
    if (other == sample) { continue; }
    if (!clipPolygon(points[other] - points[sample], (dot(points[other], points[other]) - dot(points[sample], points[sample])) * 0.5)) { statuses[statusesOffset + index] = 3u; return; }
    if (polygonCount < 3u) { break; }
  }
  let area = polygonArea(); areas[sample] = area; areaSum += area; prediction += area * values[valuesOffset + sampleIndices[sample]]; if (area > 0.0) { contributing++; }
}
if (!(areaSum > 0.0) || abs(areaSum - queryArea) > queryArea * 1e-3 || !isFiniteValue(prediction)) { statuses[statusesOffset + index] = 2u; return; }
write_results(index, vec4<f32>(prediction / areaSum, queryArea * scale * scale, 1.0, f32(contributing))); statuses[statusesOffset + index] = 0u;
${props.output.weights ? 'for (var sample = 0u; sample < sampleCount; sample++) { writeWeight(index, sample, sampleIndices[sample], areas[sample] / areaSum); }' : ''}`
      })
    ];
  }
}
