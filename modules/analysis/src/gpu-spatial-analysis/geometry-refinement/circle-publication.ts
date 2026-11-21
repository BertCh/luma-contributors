// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getReadPointWGSL} from '../shared/spatial-wgsl-point-helpers';
import type {GPUMaximumInscribedCircleProps} from './gpu-maximum-inscribed-circle';

/**
 * Closest point of segment `start`-`end` to `query`, projected from the nearer endpoint so that
 * the rounding of a long edge's fraction cannot displace the witness by up to an ulp of the
 * edge length. Axis-aligned edges return the exact foot. @internal
 */
export const CLOSEST_SEGMENT_POINT_WGSL = /* wgsl */ `
fn closestSegmentPoint(query: vec2f, start: vec2f, end: vec2f) -> vec2f {
  let edge = end - start;
  let denominator = dot(edge, edge);
  if (!(denominator > 0.0)) { return start; }
  let fromStart = dot(query - start, edge);
  if (fromStart <= 0.0) { return start; }
  let fromEnd = dot(end - query, edge);
  if (fromEnd <= 0.0) { return end; }
  if (edge.y == 0.0) { return vec2f(query.x, start.y); }
  if (edge.x == 0.0) { return vec2f(start.x, query.y); }
  if (fromStart <= fromEnd) { return start + (fromStart / denominator) * edge; }
  return end - (fromEnd / denominator) * edge;
}
`;

/**
 * Exact sign of a sum of float32 products `values[2i] * values[2i + 1]`, from integer
 * significands: no product, difference or sum is rounded. Returns -1, 0 or 1, or 2 when the
 * exponents span more than the 384-bit accumulator (the caller then treats the test as
 * uncertified). Integer-only arithmetic, so no float reassociation can change the result. @internal
 */
export const EXACT_PRODUCT_SUM_SIGN_WGSL = /* wgsl */ `
const EXACT_SUM_WORDS: u32 = 12u;
fn exactLowestExponent(value: f32) -> i32 {
  let biased = (bitcast<u32>(value) >> 23u) & 255u;
  return select(i32(biased) - 150, -149, biased == 0u);
}
fn exactSignificand(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return (bits & 8388607u) | select(8388608u, 0u, ((bits >> 23u) & 255u) == 0u);
}
fn exactMultiply(first: u32, second: u32) -> vec2u {
  let low = (first & 65535u) * (second & 65535u);
  let middleFirst = (first >> 16u) * (second & 65535u) + (low >> 16u);
  let middleSecond = (first & 65535u) * (second >> 16u) + (middleFirst & 65535u);
  return vec2u((middleSecond << 16u) | (low & 65535u),
    (first >> 16u) * (second >> 16u) + (middleFirst >> 16u) + (middleSecond >> 16u));
}
fn exactAdd(words: ptr<function, array<u32, 12>>, index: u32, value: u32) {
  var carry = value;
  for (var slot = index; slot < EXACT_SUM_WORDS && carry != 0u; slot++) {
    let previous = (*words)[slot];
    let sum = previous + carry;
    (*words)[slot] = sum;
    carry = select(0u, 1u, sum < previous);
  }
}
fn exactProductSumSign(values: ptr<function, array<f32, 16>>, count: u32) -> i32 {
  var base = 2147483647;
  for (var term = 0u; term < count; term++) {
    let first = (*values)[term * 2u]; let second = (*values)[term * 2u + 1u];
    if (first != 0.0 && second != 0.0) { base = min(base, exactLowestExponent(first) + exactLowestExponent(second)); }
  }
  if (base == 2147483647) { return 0; }
  var positive: array<u32, 12>;
  var negative: array<u32, 12>;
  for (var term = 0u; term < count; term++) {
    let first = (*values)[term * 2u]; let second = (*values)[term * 2u + 1u];
    if (first == 0.0 || second == 0.0) { continue; }
    let shift = exactLowestExponent(first) + exactLowestExponent(second) - base;
    if (shift > i32(EXACT_SUM_WORDS * 32u) - 49) { return 2; }
    let product = exactMultiply(exactSignificand(first), exactSignificand(second));
    let word = u32(shift) / 32u;
    let remainder = u32(shift) % 32u;
    var parts = vec3u(product, 0u);
    if (remainder != 0u) {
      parts = vec3u(product.x << remainder,
        (product.y << remainder) | (product.x >> (32u - remainder)),
        product.y >> (32u - remainder));
    }
    let subtract = (first < 0.0) != (second < 0.0);
    for (var part = 0u; part < 3u; part++) {
      if (word + part >= EXACT_SUM_WORDS) { continue; }
      if (subtract) { exactAdd(&negative, word + part, parts[part]); }
      else { exactAdd(&positive, word + part, parts[part]); }
    }
  }
  for (var word = i32(EXACT_SUM_WORDS) - 1; word >= 0; word--) {
    if (positive[u32(word)] > negative[u32(word)]) { return 1; }
    if (positive[u32(word)] < negative[u32(word)]) { return -1; }
  }
  return 0;
}
// Sign of dot(c - a, b - a).
fn exactDotSign(c: vec2f, a: vec2f, b: vec2f) -> i32 {
  var values = array<f32, 16>(c.x, b.x, -c.x, a.x, -a.x, b.x, a.x, a.x,
    c.y, b.y, -c.y, a.y, -a.y, b.y, a.y, a.y);
  return exactProductSumSign(&values, 8u);
}
// The stored witness w is the exact closest point of segment a-b to center c.
fn isExactClosestPoint(w: vec2f, c: vec2f, a: vec2f, b: vec2f) -> bool {
  if (all(w == a)) { return exactDotSign(c, a, b) <= 0; }
  if (all(w == b)) { return exactDotSign(c, b, a) <= 0; }
  if (exactDotSign(c, a, b) != 1 || exactDotSign(c, b, a) != 1) { return false; }
  // On the line: cross(b - a, w - a) == 0.
  var line = array<f32, 16>(b.x, w.y, -b.x, a.y, -a.x, w.y, -b.y, w.x, b.y, a.x, a.y, w.x,
    0.0, 0.0, 0.0, 0.0);
  if (exactProductSumSign(&line, 6u) != 0) { return false; }
  // Perpendicular foot: dot(w - c, b - a) == 0.
  var foot = array<f32, 16>(w.x, b.x, -w.x, a.x, -c.x, b.x, c.x, a.x,
    w.y, b.y, -w.y, a.y, -c.y, b.y, c.y, a.y);
  return exactProductSumSign(&foot, 8u) == 0;
}
`;

/** Re-evaluate geometry in the row domain of the published float32 center. @internal */
export function getCirclePublicationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUMaximumInscribedCircleProps
): GPUCommandNode<Parameters>[] {
  const count = props.featureOffsets.length - 1;
  const publication = createTransientView(graph, `${id}-publication`, 'float32x4', count);
  const nodes = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-published-circle`,
      operation: 'GPUMaximumInscribedCircle',
      variant: 'published-circle',
      invocationCount: count,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
        {name: 'polygonOffsets', view: props.polygonOffsets, type: 'u32', access: 'read'},
        {name: 'featureOffsets', view: props.featureOffsets, type: 'u32', access: 'read'},
        {name: 'circles', view: props.output.circles, type: 'f32', access: 'read_write'},
        {name: 'witnesses', view: props.output.witnesses, type: 'f32', access: 'read_write'},
        {name: 'publication', view: publication, type: 'f32', access: 'read_write'}
      ],
      declarations: `${getReadPointWGSL('point', 'positions')}
${CLOSEST_SEGMENT_POINT_WGSL}
${EXACT_PRODUCT_SUM_SIGN_WGSL}
const VERTEX_COUNT: u32 = ${props.positions.length}u;
const RING_COUNT: u32 = ${props.ringOffsets.length - 1}u;
const POLYGON_COUNT: u32 = ${props.polygonOffsets.length - 1}u;
struct Nearest { absolute: vec2f, relative: vec2f, start: vec2f, end: vec2f, distance: f32, exact: f32 }
fn nearestBoundary(center: vec2f, firstRing: u32, lastRing: u32) -> Nearest {
  var inside = false;
  var result = Nearest(center, vec2f(0.0), center, center, 3.4028234663852886e38, 0.0);
  var distanceSquared = 3.4028234663852886e38;
  for (var ring = firstRing; ring < lastRing; ring++) {
    let first = min(ringOffsets[ringOffsetsOffset + ring], VERTEX_COUNT);
    let last = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
    if (last <= first) { continue; }
    var previousPoint = point(last - 1u);
    var previous = previousPoint - center;
    for (var vertex = first; vertex < last; vertex++) {
      let currentPoint = point(vertex);
      let current = currentPoint - center;
      // Center-relative coordinates keep sub-ULP clearance; the projection starts from the
      // nearer endpoint (stable on long edges). Vertices and axis-aligned feet are exact.
      let relative = closestSegmentPoint(vec2f(0.0), previous, current);
      let squared = dot(relative, relative);
      if (squared < distanceSquared) {
        distanceSquared = squared;
        let edge = currentPoint - previousPoint;
        var absolute = center + relative;
        var exactWitness = false;
        if (all(relative == previous)) { absolute = previousPoint; exactWitness = true; }
        else if (all(relative == current)) { absolute = currentPoint; exactWitness = true; }
        else if (edge.y == 0.0) { absolute = vec2f(center.x, previousPoint.y); exactWitness = true; }
        else if (edge.x == 0.0) { absolute = vec2f(previousPoint.x, center.y); exactWitness = true; }
        var published = relative;
        if (exactWitness) { published = absolute - center; }
        result = Nearest(absolute, published, previousPoint, currentPoint, 0.0, select(0.0, 1.0, exactWitness));
      }
      if ((current.y > 0.0) != (previous.y > 0.0)) {
        if (0.0 < previous.x - previous.y * (current.x - previous.x) / (current.y - previous.y)) { inside = !inside; }
      }
      previous = current;
      previousPoint = currentPoint;
    }
  }
  result.distance = select(-sqrt(distanceSquared), sqrt(distanceSquared), inside);
  return result;
}`,
      body: `let address = circlesOffset + index * 4u;
var center = vec2f(circles[address], circles[address + 1u]);
let searchRadius = circles[address + 2u];
var relative = vec2f(searchRadius);
var clearance = searchRadius;
var limited = 0.0;
if (!any((bitcast<vec2u>(center) & vec2u(0x7f800000u)) == vec2u(0x7f800000u))) {
  let firstPolygon = min(featureOffsets[featureOffsetsOffset + index], POLYGON_COUNT);
  let lastPolygon = min(featureOffsets[featureOffsetsOffset + index + 1u], POLYGON_COUNT);
  let firstRing = min(polygonOffsets[polygonOffsetsOffset + firstPolygon], RING_COUNT);
  let lastRing = min(polygonOffsets[polygonOffsetsOffset + lastPolygon], RING_COUNT);
  var nearest = nearestBoundary(center, firstRing, lastRing);
  if (nearest.distance < 0.0) {
    // A stored source vertex is a representable boundary fallback, including with empty rings.
    for (var ring = firstRing; ring < lastRing; ring++) {
      let first = min(ringOffsets[ringOffsetsOffset + ring], VERTEX_COUNT);
      let last = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
      if (last > first) { center = point(first); break; }
    }
    nearest = nearestBoundary(center, firstRing, lastRing);
    limited = 1.0;
  }
  clearance = max(0.0, nearest.distance);
  relative = nearest.relative;
  if (clearance < searchRadius) { limited = 1.0; }
  // An oblique foot is published only when the stored float32 witness is certified, with exact
  // integer arithmetic, as the closest point of its supporting edge to the stored center.
  if (nearest.exact == 0.0 && !isExactClosestPoint(nearest.absolute, center, nearest.start, nearest.end)) {
    limited = 1.0;
  }
  circles[address] = center.x; circles[address + 1u] = center.y;
  circles[address + 2u] = clearance;
  witnesses[witnessesOffset + index * 2u] = nearest.absolute.x;
  witnesses[witnessesOffset + index * 2u + 1u] = nearest.absolute.y;
}
publication[publicationOffset + index * 4u] = relative.x;
publication[publicationOffset + index * 4u + 1u] = relative.y;
publication[publicationOffset + index * 4u + 2u] = clearance;
publication[publicationOffset + index * 4u + 3u] = limited;`
    })
  ];
  for (const [name, view] of Object.entries(props.output)) {
    if (!view || !['publishedCenterClearance', 'precisionLimited', 'witnessOffsets'].includes(name))
      continue;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-published-${name}`,
        operation: 'GPUMaximumInscribedCircle',
        variant: `published-${name}`,
        invocationCount: count,
        bindings: [
          {name: 'publication', view: publication, type: 'f32', access: 'read'},
          {
            name: 'output',
            view,
            type: name === 'precisionLimited' ? 'u32' : 'f32',
            access: 'read_write'
          }
        ],
        body:
          name === 'precisionLimited'
            ? 'output[outputOffset + index] = select(0u, 1u, publication[publicationOffset + index * 4u + 3u] != 0.0);'
            : name === 'witnessOffsets'
              ? `output[outputOffset + index * 2u] = publication[publicationOffset + index * 4u]; output[outputOffset + index * 2u + 1u] = publication[publicationOffset + index * 4u + 1u];`
              : 'output[outputOffset + index] = publication[publicationOffset + index * 4u + 2u];'
      })
    );
  }
  return nodes;
}
