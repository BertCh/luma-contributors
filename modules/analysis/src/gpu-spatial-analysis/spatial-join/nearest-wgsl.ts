// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getStorageTopKAccessors} from '../../analysis-core/kernels/storage-backed-top-k';

import {NEAREST_PRIVATE_NEIGHBOR_CAPACITY} from './nearest-types';

import type {NearestSide} from './nearest-types';
import {forEachEdge, getMinimumRingVertices} from './spatial-predicate-wgsl';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {SPATIAL_GEODESIC_WGSL} from './spatial-geodesic-wgsl';
import {SCALED_PLANAR_DISTANCE_WGSL} from '../../analysis-core/kernels/scaled-planar-distance-wgsl';

/**
 * Squared-distance ranks retain a normalized f32 significand and a biased exponent. The two
 * finite normal float bit ranges provide 508 exponent buckets, enough to square every normal
 * finite f32 linear distance without underflow or overflow. Bits skip NaN/infinity/subnormal
 * encodings; comparisons use unsigned words, never the encoded float's numeric value. The top
 * words are reserved for unbounded/missing, invalid radius and unrepresentable linear distance.
 * The last two finite squared bins coalesce to leave those sentinels. @internal
 */
export const NEAREST_DISTANCE_RANK_WGSL = /* wgsl */ `
const NEAREST_NO_DISTANCE: f32 = -3.402823466e+38;
const NEAREST_INVALID_DISTANCE: f32 = bitcast<f32>(0xff7ffffeu);
const NEAREST_OVERFLOW_DISTANCE: u32 = 0xff7ffffdu;
const NEAREST_MAXIMUM_FINITE_RANK: u32 = 0xff7ffffcu;
fn nearestDistanceLess(first: f32, second: f32) -> bool { return bitcast<u32>(first) < bitcast<u32>(second); }
fn nearestDistanceAtMost(first: f32, second: f32) -> bool { return bitcast<u32>(first) <= bitcast<u32>(second); }
fn minimumNearestDistance(first: f32, second: f32) -> f32 { return select(second, first, nearestDistanceLess(first, second)); }
fn encodeNearestSquaredRank(significand: f32, exponent: i32) -> f32 {
  if (significand == 0.0) { return 0.0; }
  let parts = frexp(significand);
  let power = exponent + parts.exp;
  if (power > 256 || (power == 256 && parts.fract > 0.9999998807907104)) {
    return bitcast<f32>(NEAREST_OVERFLOW_DISTANCE);
  }
  if (power < -251) { return bitcast<f32>(0x00800000u); }
  var word = (u32(power + 252) << 23u) | (bitcast<u32>(parts.fract) & 0x7fffffu);
  if (word >= 0x7f800000u) { word += 0x01000000u; }
  return bitcast<f32>(min(word, NEAREST_MAXIMUM_FINITE_RANK));
}
fn encodeNearestDistance(distance: f32) -> f32 {
  let parts = frexp(abs(distance));
  return encodeNearestSquaredRank(parts.fract * parts.fract, parts.exp * 2);
}
fn decodeNearestDistance(rank: f32) -> f32 {
  let bits = bitcast<u32>(rank);
  if (bits == 0u) { return 0.0; }
  if (bits == NEAREST_OVERFLOW_DISTANCE) { return bitcast<f32>(0x7f800000u | (bits & 0u)); }
  if (bits == NEAREST_MAXIMUM_FINITE_RANK) { return 3.402823466e+38; }
  var word = bits;
  if (word >= 0x80800000u) { word -= 0x01000000u; }
  var exponent = i32(word >> 23u) - 252;
  var significand = bitcast<f32>(0x3f000000u | (word & 0x7fffffu));
  if ((exponent & 1) != 0) { significand *= 2.0; exponent -= 1; }
  let firstExponent = min(exponent / 2, 126);
  return ldexp(ldexp(sqrt(significand), firstExponent), exponent / 2 - firstExponent);
}
fn widenNearestDistanceBound(rank: f32) -> f32 {
  let bits = bitcast<u32>(rank);
  if (bits == 0u || bits >= NEAREST_OVERFLOW_DISTANCE) { return rank; }
  var word = min(bits, NEAREST_OVERFLOW_DISTANCE - 16u) + 16u;
  if (word >= 0x7f800000u && word < 0x80800000u) { word += 0x01000000u; }
  return bitcast<f32>(word);
}
`;

/** Slots of the per-query header in the traversal result buffer: count, flags, bound bits. @internal */
export const NEAREST_HEADER_WORDS = 3;

/** u32 words per BVH node in the packed node table: min.xy, max.xy (f32 bits), feature row. @internal */
export const NEAREST_NODE_WORDS = 5;

/**
 * Constants and finite checks shared by every nearest kernel, including the bounds kernels that do
 * not evaluate geometry.
 *
 * @internal
 */
export const NEAREST_BASE_WGSL = /* wgsl */ `
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
const NO_FEATURE: u32 = 0xffffffffu;
const NO_SEGMENT: u32 = 0xffffffffu;
${IS_FINITE_VALUE_WGSL}
${NEAREST_DISTANCE_RANK_WGSL}
`;

const NEAREST_ORIENTATION_WGSL = /* wgsl */ `${NEAREST_BASE_WGSL}
${EXACT_ORIENTATION_WGSL}
// Exact sign (-1, 0, 1) of cross(b - a, c - a).
fn orientSide(a: vec2f, b: vec2f, c: vec2f) -> i32 {
  if (a.x == b.x && a.y == b.y) { return 0; }
  let sign = orientSign(a, b, c);
  if (sign != 2) { return sign; }
  let determinant = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  return select(select(0, -1, determinant < 0.0), 1, determinant > 0.0);
}
fn inSegmentBox(a: vec2f, b: vec2f, q: vec2f) -> bool {
  return q.x >= min(a.x, b.x) && q.x <= max(a.x, b.x) &&
    q.y >= min(a.y, b.y) && q.y <= max(a.y, b.y);
}
// Each orientation is evaluated once; a zero sign plus the range test is the on-segment test.
fn segmentsIntersect(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> bool {
  let o1 = orientSide(a, b, c);
  let o2 = orientSide(a, b, d);
  let o3 = orientSide(c, d, a);
  let o4 = orientSide(c, d, b);
  if (((o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0)) &&
      ((o3 > 0 && o4 < 0) || (o3 < 0 && o4 > 0))) {
    return true;
  }
  return (o1 == 0 && inSegmentBox(a, b, c)) || (o2 == 0 && inSegmentBox(a, b, d)) ||
    (o3 == 0 && inSegmentBox(c, d, a)) || (o4 == 0 && inSegmentBox(c, d, b));
}
fn crossProduct(a: vec2f, b: vec2f) -> f32 { return a.x * b.y - a.y * b.x; }
`;

/**
 * Planar helpers shared by the nearest kernels: segment/segment distance, the foot point on the
 * second segment, and the topological tests (on-segment, segment intersection, containment).
 *
 * Ranks preserve squared significands across the full normal linear-distance range.
 * Foot points and published linear distances are f32. Every orientation sign is exact: `orientSide` wraps the
 * shared `orientSign` (an f32 filter with an exact 256-bit integer fallback, see
 * `exact-orientation-wgsl.ts`), so "distance 0" agrees with the point-in-polygon join and the
 * predicate join for the same vertices, including points exactly on an edge. Only an orientation
 * with no exact sign (non-finite coordinate, product exponents spanning more than 200 bits) falls
 * back to the sign of the f32 determinant.
 *
 * @internal
 */
export const NEAREST_COMMON_WGSL = /* wgsl */ `${NEAREST_ORIENTATION_WGSL}
${SCALED_PLANAR_DISTANCE_WGSL}
fn rankPlanarSquaredDistance(value: PlanarDistanceValue) -> f32 {
  return encodeNearestSquaredRank(value.significand, value.exponent);
}
fn nearestCrossProduct(first: PlanarDistanceVector, second: PlanarDistanceVector) -> PlanarDistanceValue {
  let left = multiplyPlanarDistanceValues(first.x, second.y);
  let right = multiplyPlanarDistanceValues(first.y, second.x);
  let leftExponent = first.x.exponent + second.y.exponent;
  let rightExponent = first.y.exponent + second.x.exponent;
  if (left.significand != 0.0 && right.significand != 0.0 &&
      (left.significand < 0.0) == (right.significand < 0.0) && abs(leftExponent - rightExponent) <= 2) {
    // Exact 48-bit products and integer subtraction survive shader reassociation. Floating
    // error-free transforms (including fma residuals) can be optimized away on Metal.
    let exponent = min(leftExponent, rightExponent);
    var leftWords = multiply24((bitcast<u32>(first.x.significand) & 0x7fffffu) | 0x800000u,
      (bitcast<u32>(second.y.significand) & 0x7fffffu) | 0x800000u);
    var rightWords = multiply24((bitcast<u32>(first.y.significand) & 0x7fffffu) | 0x800000u,
      (bitcast<u32>(second.x.significand) & 0x7fffffu) | 0x800000u);
    let leftShift = u32(leftExponent - exponent);
    let rightShift = u32(rightExponent - exponent);
    if (leftShift != 0u) { leftWords = vec2u(leftWords.x << leftShift, (leftWords.y << leftShift) | (leftWords.x >> (32u - leftShift))); }
    if (rightShift != 0u) { rightWords = vec2u(rightWords.x << rightShift, (rightWords.y << rightShift) | (rightWords.x >> (32u - rightShift))); }
    let reverse = leftWords.y < rightWords.y || (leftWords.y == rightWords.y && leftWords.x < rightWords.x);
    let larger = select(leftWords, rightWords, reverse);
    let smaller = select(rightWords, leftWords, reverse);
    let low = larger.x - smaller.x;
    let high = larger.y - smaller.y - select(0u, 1u, larger.x < smaller.x);
    let magnitude = f32(high) * 4294967296.0 + f32(low);
    return getPlanarDistanceValue(select(magnitude, -magnitude, reverse != (left.significand < 0.0)), exponent - 48);
  }
  return addPlanarDistanceValues(left, negatePlanarDistanceValue(right));
}
fn interpolateNearestSegment(start: vec2f, direction: PlanarDistanceVector, fraction: PlanarDistanceValue) -> vec2f {
  return vec2f(convertPlanarDistanceValue(addPlanarDistanceValues(getPlanarDistanceValue(start.x, 0), multiplyPlanarDistanceValues(direction.x, fraction))),
    convertPlanarDistanceValue(addPlanarDistanceValues(getPlanarDistanceValue(start.y, 0), multiplyPlanarDistanceValues(direction.y, fraction))));
}
fn closestPointOnSegment(p: vec2f, a: vec2f, b: vec2f) -> vec2f {
  let direction = getPlanarDistanceVector(b, a);
  let lengthSq = dotPlanarDistanceVectors(direction, direction);
  let projection = dotPlanarDistanceVectors(getPlanarDistanceVector(p, a), direction);
  if (lengthSq.significand == 0.0 || projection.significand <= 0.0) { return a; }
  if (dotPlanarDistanceVectors(getPlanarDistanceVector(p, b), direction).significand >= 0.0) { return b; }
  return interpolateNearestSegment(a, direction, getPlanarDistanceValue(projection.significand / lengthSq.significand, projection.exponent - lengthSq.exponent));
}
fn pointSegmentDistanceSq(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let direction = getPlanarDistanceVector(b, a);
  let fromStart = getPlanarDistanceVector(p, a);
  let lengthSq = dotPlanarDistanceVectors(direction, direction);
  if (lengthSq.significand == 0.0 || dotPlanarDistanceVectors(fromStart, direction).significand <= 0.0) {
    return rankPlanarSquaredDistance(dotPlanarDistanceVectors(fromStart, fromStart));
  }
  let fromEnd = getPlanarDistanceVector(p, b);
  if (dotPlanarDistanceVectors(fromEnd, direction).significand >= 0.0) { return rankPlanarSquaredDistance(dotPlanarDistanceVectors(fromEnd, fromEnd)); }
  let cross = nearestCrossProduct(fromStart, direction);
  return rankPlanarSquaredDistance(getPlanarDistanceValue(cross.significand * cross.significand / lengthSq.significand, 2 * cross.exponent - lengthSq.exponent));
}
fn segmentPairDistanceSq(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> f32 {
  if (a.x == b.x && a.y == b.y) { return pointSegmentDistanceSq(a, c, d); }
  if (c.x == d.x && c.y == d.y) { return pointSegmentDistanceSq(c, a, b); }
  if (segmentsIntersect(a, b, c, d)) { return 0.0; }
  return minimumNearestDistance(
    minimumNearestDistance(pointSegmentDistanceSq(a, c, d), pointSegmentDistanceSq(b, c, d)),
    minimumNearestDistance(pointSegmentDistanceSq(c, a, b), pointSegmentDistanceSq(d, a, b))
  );
}
// The point of segment c-d nearest to segment a-b (the crossing point when they cross).
fn segmentFoot(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> vec2f {
  if (c.x == d.x && c.y == d.y) { return c; }
  if (a.x == b.x && a.y == b.y) { return closestPointOnSegment(a, c, d); }
  let ab = getPlanarDistanceVector(b, a);
  let cd = getPlanarDistanceVector(d, c);
  let denominator = nearestCrossProduct(cd, ab);
  if (denominator.significand != 0.0 && segmentsIntersect(a, b, c, d)) {
    let numerator = nearestCrossProduct(getPlanarDistanceVector(a, c), ab);
    return interpolateNearestSegment(c, cd, getPlanarDistanceValue(numerator.significand / denominator.significand, numerator.exponent - denominator.exponent));
  }
  var best = closestPointOnSegment(a, c, d);
  var bestSq = pointSegmentDistanceSq(a, c, d);
  let second = closestPointOnSegment(b, c, d);
  let secondSq = pointSegmentDistanceSq(b, c, d);
  if (nearestDistanceLess(secondSq, bestSq)) { best = second; bestSq = secondSq; }
  let thirdSq = pointSegmentDistanceSq(c, a, b);
  if (nearestDistanceLess(thirdSq, bestSq)) { best = c; bestSq = thirdSq; }
  if (nearestDistanceLess(pointSegmentDistanceSq(d, a, b), bestSq)) { best = d; }
  return best;
}
`;

/**
 * Great-circle counterpart of the distance half of {@link NEAREST_COMMON_WGSL}, with the same
 * function names. Positions are `vec2f(longitude, latitude)` in degrees and every "distance
 * squared" is a central angle in radians (the traversal compares and bounds angles; multiply by the
 * sphere radius for meters).
 *
 * Edges are straight in longitude/latitude with the longitude difference wrapped at +-180 (the same
 * edge model as the containment and crossing tests and as the network snapping contributors).
 * Point-to-edge distances use `geodesicSegmentFraction` and `geodesicSegmentDistance` of
 * `GEODESIC_WGSL`. The crossing and containment tests stay exact and planar in degrees; before the
 * crossing test the second edge is moved by a multiple of 360 degrees to overlap the first, so an
 * edge across the antimeridian still crosses its neighbors.
 *
 * @internal
 */
export const NEAREST_GEODESIC_WGSL = /* wgsl */ `${NEAREST_ORIENTATION_WGSL}
${SPATIAL_GEODESIC_WGSL}
fn unwrapLongitude(x: f32, reference: f32) -> f32 {
  return x + 360.0 * round((reference - x) / 360.0);
}
fn wrapLongitude(x: f32) -> f32 {
  return x - 360.0 * round(x / 360.0);
}
struct AlignedEdges { b: vec2f, c: vec2f, d: vec2f, overlaps: bool }
// Edges a-b and c-d with b, c and d shifted by multiples of 360 degrees so that b is within 180 of
// a, c within 180 of a and d within 180 of c, and c-d on the copy that overlaps a-b in longitude.
// Coordinates that need no shift are returned unchanged.
fn alignEdges(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> AlignedEdges {
  let alignedB = vec2f(unwrapLongitude(b.x, a.x), b.y);
  var alignedC = vec2f(unwrapLongitude(c.x, a.x), c.y);
  var alignedD = vec2f(unwrapLongitude(d.x, alignedC.x), d.y);
  let low = min(a.x, alignedB.x);
  let high = max(a.x, alignedB.x);
  var overlaps = max(alignedC.x, alignedD.x) >= low && min(alignedC.x, alignedD.x) <= high;
  if (!overlaps) {
    let shift = select(360.0, -360.0, alignedC.x > a.x);
    alignedC.x = alignedC.x + shift;
    alignedD.x = alignedD.x + shift;
    overlaps = max(alignedC.x, alignedD.x) >= low && min(alignedC.x, alignedD.x) <= high;
  }
  return AlignedEdges(alignedB, alignedC, alignedD, overlaps);
}
fn edgesIntersect(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> bool {
  let edges = alignEdges(a, b, c, d);
  return edges.overlaps && segmentsIntersect(a, edges.b, edges.c, edges.d);
}
fn closestPointOnSegment(p: vec2f, a: vec2f, b: vec2f) -> vec2f {
  let fraction = geodesicSegmentFraction(p, a, b);
  if (fraction <= 0.0) { return a; }
  if (fraction >= 1.0) { return b; }
  let foot = a + vec2f(geodesicWrapLongitudeDelta(b.x - a.x), b.y - a.y) * fraction;
  return vec2f(wrapLongitude(foot.x), foot.y);
}
// Central angle from p to the segment a-b.
fn pointSegmentDistanceSq(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  return geodesicSegmentDistance(p, a, b, geodesicSegmentFraction(p, a, b), 1.0);
}
fn segmentPairDistanceSq(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> f32 {
  if (a.x == b.x && a.y == b.y) { return pointSegmentDistanceSq(a, c, d); }
  if (c.x == d.x && c.y == d.y) { return pointSegmentDistanceSq(c, a, b); }
  if (edgesIntersect(a, b, c, d)) { return 0.0; }
  return min(
    min(pointSegmentDistanceSq(a, c, d), pointSegmentDistanceSq(b, c, d)),
    min(pointSegmentDistanceSq(c, a, b), pointSegmentDistanceSq(d, a, b))
  );
}
// The point of segment c-d nearest to segment a-b (the crossing point when they cross).
fn segmentFoot(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> vec2f {
  if (c.x == d.x && c.y == d.y) { return c; }
  if (a.x == b.x && a.y == b.y) { return closestPointOnSegment(a, c, d); }
  let edges = alignEdges(a, b, c, d);
  if (edges.overlaps) {
    let ab = edges.b - a;
    let cd = edges.d - edges.c;
    let denominator = crossProduct(cd, ab);
    if (denominator != 0.0 && segmentsIntersect(a, edges.b, edges.c, edges.d)) {
      let crossing = edges.c + cd * clamp(crossProduct(a - edges.c, ab) / denominator, 0.0, 1.0);
      return vec2f(wrapLongitude(crossing.x), crossing.y);
    }
  }
  var best = closestPointOnSegment(a, c, d);
  var bestAngle = geodesicCentralAngle(a, best);
  let second = closestPointOnSegment(b, c, d);
  let secondAngle = geodesicCentralAngle(b, second);
  if (secondAngle < bestAngle) { best = second; bestAngle = secondAngle; }
  let thirdAngle = pointSegmentDistanceSq(c, a, b);
  if (thirdAngle < bestAngle) { best = c; bestAngle = thirdAngle; }
  if (pointSegmentDistanceSq(d, a, b) < bestAngle) { best = d; }
  return best;
}
`;

/**
 * Declares the accessors of one geometry side: ring and vertex ranges, vertices, the first usable
 * vertex, and (for polygons) an even/odd containment test.
 *
 * Bindings read by the accessors: points `${p}Positions`; segments `${p}Starts`, `${p}Ends`; lines
 * `${p}Positions`, `${p}LineOffsets`; polygons `${p}Positions`, `${p}FeatureRings`, `${p}RingOffsets`.
 * With `packedRingOffsetsBase`, polygon ring offsets share the `${p}FeatureRings` binding.
 *
 * @internal
 */
export function getNearestSideWGSL(side: NearestSide): string {
  const p = side.prefix;
  const P = p.toUpperCase();
  const {kind} = side;
  let ringStart = 'f';
  let ringEnd = 'f + 1u';
  let vertexStart = 'r';
  let vertexEnd = 'r + 1u';
  let vertex = `vec2f(${p}Positions[${p}PositionsOffset + i * 2u], ${p}Positions[${p}PositionsOffset + i * 2u + 1u])`;
  if (kind === 'segments') {
    vertexStart = '2u * r';
    vertexEnd = '2u * r + 2u';
    vertex = `select(
    vec2f(${p}Ends[${p}EndsOffset + (i >> 1u) * 2u], ${p}Ends[${p}EndsOffset + (i >> 1u) * 2u + 1u]),
    vec2f(${p}Starts[${p}StartsOffset + (i >> 1u) * 2u], ${p}Starts[${p}StartsOffset + (i >> 1u) * 2u + 1u]),
    (i & 1u) == 0u)`;
  } else if (kind === 'lines') {
    const offsets =
      side.packedRingOffsetsBase === undefined
        ? `${p}LineOffsets[${p}LineOffsetsOffset`
        : `${p}FeatureOffsets[${p}FeatureOffsetsOffset + ${side.packedRingOffsetsBase}u`;
    vertexStart = `${offsets} + r]`;
    vertexEnd = `min(${offsets} + r + 1u], ${P}_VERTEX_COUNT)`;
  } else if (kind === 'polygons') {
    ringStart = `${p}FeatureRings[${p}FeatureRingsOffset + f * 2u]`;
    ringEnd = `${p}FeatureRings[${p}FeatureRingsOffset + f * 2u + 1u]`;
    const ringOffsets =
      side.packedRingOffsetsBase === undefined
        ? `${p}RingOffsets[${p}RingOffsetsOffset`
        : `${p}FeatureRings[${p}FeatureRingsOffset + ${side.packedRingOffsetsBase}u`;
    vertexStart = `${ringOffsets} + r]`;
    vertexEnd = `min(${ringOffsets} + r + 1u], ${P}_VERTEX_COUNT)`;
  }
  const contains =
    kind === 'polygons'
      ? `
fn ${p}Contains(q: vec2f, f: u32) -> bool {
  var inside = false;
  ${forEachEdge(
    side,
    'f',
    'a',
    'b',
    `if ((a.y > q.y) != (b.y > q.y)) {
      let side = orientSide(a, b, q);
      if ((b.y > a.y && side > 0) || (b.y < a.y && side < 0)) { inside = !inside; }
    }`
  )}
  return inside;
}`
      : '';
  if (side.partCount !== undefined) {
    ringStart = `${p}FeatureOffsets[${p}FeatureOffsetsOffset + f]`;
    ringEnd = `${p}FeatureOffsets[${p}FeatureOffsetsOffset + f + 1u]`;
  }
  return `
${kind === 'points' ? '' : `const ${P}_VERTEX_COUNT: u32 = ${side.vertexCount}u;`}
fn ${p}RingStart(f: u32) -> u32 { ${side.partCount === undefined ? '' : `if (${ringStart} > ${ringEnd} || ${ringEnd} > ${side.partCount}u) { return 0u; }`} return ${ringStart}; }
fn ${p}RingEnd(f: u32) -> u32 { ${side.partCount === undefined ? '' : `if (${ringStart} > ${ringEnd} || ${ringEnd} > ${side.partCount}u) { return 0u; }`} return ${ringEnd}; }
fn ${p}RingVertexStart(r: u32) -> u32 { return ${vertexStart}; }
fn ${p}RingVertexEnd(r: u32) -> u32 { return ${vertexEnd}; }
fn ${p}Vertex(i: u32) -> vec2f { return ${vertex}; }
// (x, y, valid): first finite vertex of the first usable ring; valid is 0 for an empty feature.
fn ${p}First(f: u32) -> vec3f {
  for (var r = ${p}RingStart(f); r < ${p}RingEnd(f); r++) {
    let vs = ${p}RingVertexStart(r);
    let ve = ${p}RingVertexEnd(r);
    if (ve > vs && ve - vs >= ${getMinimumRingVertices(kind)}u) {
      let v = ${p}Vertex(vs);
      return vec3f(v, select(0.0, 1.0, isFiniteValue(v.x) && isFiniteValue(v.y)));
    }
  }
  return vec3f(0.0);
}${contains}
`;
}

/**
 * Declares `pairResult(queryRow, featureRow)`: the minimum planar distance between two geometries,
 * the nearest point on the feature, the nearest point on the query, and the absolute start-vertex index of the feature edge that
 * attains it.
 *
 * Distance is the minimum over all usable edge pairs (the first minimum in vertex order wins, so
 * the foot point is deterministic). Polygon containment either way gives distance 0 with the
 * contained vertex as foot point and no segment. A pair with no usable edges returns
 * `NEAREST_NO_DISTANCE` (an internal rank sentinel, never a published distance).
 *
 * @internal
 */
export function getNearestPairWGSL(query: NearestSide, feature: NearestSide): string {
  const featureSegment =
    feature.kind === 'points' ? 'NO_SEGMENT' : feature.kind === 'segments' ? '0u' : 'rvs + rk';
  const inner = forEachEdge(
    feature,
    'rowF',
    'ra',
    'rb',
    `let distanceSq = segmentPairDistanceSq(qa, qb, ra, rb);
      if (nearestDistanceLess(distanceSq, result.distanceSq)) {
        result.distanceSq = distanceSq;
        result.foot = segmentFoot(qa, qb, ra, rb);
        result.queryFoot = segmentFoot(ra, rb, qa, qb);
        result.segment = ${featureSegment};
        if (distanceSq == 0.0) { return result; }
      }`
  );
  const outer = forEachEdge(query, 'rowQ', 'qa', 'qb', inner);
  const featureInQuery =
    query.kind === 'polygons'
      ? `
  if (bitcast<u32>(result.distanceSq) != 0u) {
    ${forEachEdge(feature, 'rowF', 'containedStart', 'containedEnd', `if (qContains(containedStart, rowQ)) { return PairResult(0.0, containedStart, NO_SEGMENT, containedStart); }`)}
  }`
      : '';
  const queryInFeature =
    feature.kind === 'polygons'
      ? `
  if (bitcast<u32>(result.distanceSq) != 0u) {
    ${forEachEdge(query, 'rowQ', 'containedStart', 'containedEnd', `if (rContains(containedStart, rowF)) { return PairResult(0.0, containedStart, NO_SEGMENT, containedStart); }`)}
  }`
      : '';
  return `
struct PairResult { distanceSq: f32, foot: vec2f, segment: u32, queryFoot: vec2f }
fn pairResult(rowQ: u32, rowF: u32) -> PairResult {
  var result = PairResult(NEAREST_NO_DISTANCE, vec2f(0.0), NO_SEGMENT, vec2f(0.0));
  ${outer}${queryInFeature}${featureInQuery}
  return result;
}
`;
}

/** Options of {@link getNearestTraversalWGSL}. @internal */
export type NearestTraversalOptions = {
  /** Neighbors wanted per query. */
  k: number;
  /** Result slots per query, at least `k`; larger than `k` only in all-ties mode. */
  capacity: number;
  /** Report dropped k-th-distance ties even when capacity equals k. */
  ties?: 'all' | 'lowest-id';
  /** Number of features. */
  featureCount: number;
  /** Number of internal BVH nodes. */
  internalNodeCount: number;
  /** Maximum traversal stack depth. */
  stackSize: number;
  /** Tail slot holding the per-dispatch first row and count. */
  dispatchWord: number;
  /**
   * Set when the `topk` scratch holds only one chunk of this dispatch's queries. The chunk's first
   * query (relative to the dispatch) is then read from `topk[CURSOR_WORD]`, a GPU word, and the
   * dispatch's first row and query count are GPU words at `dispatchWord`, so no chunk
   * index reaches the WGSL text or pipeline constants. `cursorWord` is the
   * `topk` element that holds the cursor.
   */
  scratchChunk?: {cursorWord: number};
  /**
   * Set when the kernel binds `queryOrder` (a permutation of the dispatch's query rows, see
   * `getHilbertQueryOrderNodes`). Invocation `index` (or scratch position `cursor + index`) then
   * handles query `queryOrder[position]`. Without `scratchChunk` the `topk` row of a query stays
   * its own row, so the clear and column kernels do not change; with it, the scratch row is the
   * position, and the column kernels map it back through the same order.
   */
  queryOrder?: boolean;
  /** Query geometry kind, for the usable-ring minimum. */
  queryKind: NearestSide['kind'];
  /**
   * Candidate filter read from the tail of `nodeData`: `[id, key]` words per query row from
   * `queryBase`, then per feature row from `featureBase` (word offsets inside `nodeData`).
   */
  filter?: {exclusive: boolean; onAttribute: boolean; queryBase: number; featureBase: number};
  /**
   * Measure distances as great-circle central angles between longitude/latitude positions instead
   * of squared planar distances; the node bounds are then lower-bounded with
   * `boxLowerAngle`. The kernel must include {@link NEAREST_GEODESIC_WGSL}.
   */
  greatCircle?: boolean;
};

/**
 * Lower bound, as a central angle in radians, of the great-circle distance between two
 * longitude/latitude boxes, for the great-circle traversal.
 *
 * The haversine of any two points is `hav(dlat) + cos(lat1) cos(lat2) hav(dlon)`. With the latitude
 * gap of the boxes, the circular longitude gap (through the antimeridian when shorter) and the
 * cosine of the latitude farthest from the equator over both boxes (a lower bound of both
 * cosines), the sum never overestimates. A box 180 degrees or wider in longitude is treated as
 * covering every longitude, which also covers features whose box straddles the antimeridian.
 *
 * @internal
 */
export const NEAREST_BOX_LOWER_ANGLE_WGSL = /* wgsl */ `
fn boxLowerAngle(
  nodeMinimum: vec2f, nodeMaximum: vec2f, queryMinimum: vec2f, queryMaximum: vec2f
) -> f32 {
  let latitudeGap = max(max(nodeMinimum.y - queryMaximum.y, queryMinimum.y - nodeMaximum.y), 0.0);
  var longitudeGap = 0.0;
  if (nodeMaximum.x - nodeMinimum.x < 180.0 && queryMaximum.x - queryMinimum.x < 180.0) {
    let direct = max(nodeMinimum.x - queryMaximum.x, queryMinimum.x - nodeMaximum.x);
    let around = 360.0 - (max(nodeMaximum.x, queryMaximum.x) - min(nodeMinimum.x, queryMinimum.x));
    longitudeGap = max(min(direct, around), 0.0);
  }
  let farLatitude = max(
    max(abs(nodeMinimum.y), abs(nodeMaximum.y)),
    max(abs(queryMinimum.y), abs(queryMaximum.y))
  );
  let cosine = geodesicCosLatitude(farLatitude);
  let sinHalfLatitude = sin(0.5 * latitudeGap * GEODESIC_DEGREES_TO_RADIANS);
  let sinHalfLongitude = sin(0.5 * longitudeGap * GEODESIC_DEGREES_TO_RADIANS);
  let term = sinHalfLatitude * sinHalfLatitude + cosine * cosine * sinHalfLongitude * sinHalfLongitude;
  return 2.0 * asin(sqrt(clamp(term, 0.0, 1.0)));
}`;

/**
 * Declarations and body of the per-query branch-and-bound BVH traversal.
 *
 * Each invocation walks the packed node table `nodeData` with an explicit stack, visiting the
 * nearer child first and pruning every node whose box distance exceeds the current k-th distance
 * (slightly widened so f32 rounding in the box bound never drops a tied candidate). It keeps the
 * best `capacity` rows in a list ordered by `(distance squared, feature row)` and writes the
 * result into `topk`: `[count, flags, bound rank bits, (row, distance rank bits) * capacity]`.
 * Slot `[2]` must hold the initial encoded bound (the clear pass writes it).
 *
 * @internal
 */
export function getNearestTraversalWGSL(options: NearestTraversalOptions): {
  declarations: string;
  body: string;
} {
  const {k, capacity} = options;
  const storageBacked = capacity > NEAREST_PRIVATE_NEIGHBOR_CAPACITY;
  const storageList = getStorageTopKAccessors('topk', 'header + 3u');
  const readRow = (slot: string) =>
    storageBacked ? storageList.readRow(slot) : `listRow[${slot}]`;
  const readDistance = (slot: string) =>
    storageBacked ? storageList.readDistance(slot) : `listDistance[${slot}]`;
  const writeRow = (slot: string, value: string) =>
    storageBacked ? storageList.writeRow(slot, value) : `listRow[${slot}] = ${value};`;
  const writeDistance = (slot: string, value: string) =>
    storageBacked ? storageList.writeDistance(slot, value) : `listDistance[${slot}] = ${value};`;
  const stride = NEAREST_HEADER_WORDS + 2 * capacity;
  const declarations = `
const DISPATCH_WORD: u32 = ${options.dispatchWord}u;
const K: u32 = ${k}u;
const CAPACITY: u32 = ${capacity}u;
const STRIDE: u32 = ${stride}u;
const FEATURE_COUNT: u32 = ${options.featureCount}u;
const INTERNAL_NODE_COUNT: u32 = ${options.internalNodeCount}u;
const STACK_SIZE: u32 = ${options.stackSize}u;
${options.scratchChunk ? `const CURSOR_WORD: u32 = ${options.scratchChunk.cursorWord}u;` : ''}
const BOUND_SLACK: f32 = 1.000001;
${options.greatCircle ? NEAREST_BOX_LOWER_ANGLE_WGSL : ''}
// ${options.greatCircle ? 'Central-angle lower bound' : 'Squared-distance rank'} from the query box to a node box, or NEAREST_NO_DISTANCE for an empty node.
fn boxLower(node: u32, queryMinimum: vec2f, queryMaximum: vec2f) -> f32 {
  let base = nodeDataOffset + node * ${NEAREST_NODE_WORDS}u;
  let nodeMinimum = vec2f(bitcast<f32>(nodeData[base]), bitcast<f32>(nodeData[base + 1u]));
  let nodeMaximum = vec2f(bitcast<f32>(nodeData[base + 2u]), bitcast<f32>(nodeData[base + 3u]));
  if (nodeMinimum.x > nodeMaximum.x || nodeMinimum.y > nodeMaximum.y) { return NEAREST_NO_DISTANCE; }
  ${
    options.greatCircle
      ? 'return boxLowerAngle(nodeMinimum, nodeMaximum, queryMinimum, queryMaximum);'
      : `var gap = PlanarDistanceVector(PlanarDistanceValue(0.0, 0), PlanarDistanceValue(0.0, 0));
  if (nodeMinimum.x > queryMaximum.x) { gap.x = getPlanarDistanceDifference(nodeMinimum.x, queryMaximum.x); }
  else if (queryMinimum.x > nodeMaximum.x) { gap.x = getPlanarDistanceDifference(queryMinimum.x, nodeMaximum.x); }
  if (nodeMinimum.y > queryMaximum.y) { gap.y = getPlanarDistanceDifference(nodeMinimum.y, queryMaximum.y); }
  else if (queryMinimum.y > nodeMaximum.y) { gap.y = getPlanarDistanceDifference(queryMinimum.y, nodeMaximum.y); }
  return rankPlanarSquaredDistance(dotPlanarDistanceVectors(gap, gap));`
  }
}`;
  // `localQuery` is the query's index inside this dispatch's query side, `row` its global row, and
  // `header` its slot in the (possibly chunked) `topk` scratch.
  const body = `let CHUNK_FIRST_ROW = topk[topkOffset + DISPATCH_WORD];
  let CHUNK_QUERY_COUNT = topk[topkOffset + DISPATCH_WORD + 1u];
  ${
    options.scratchChunk
      ? options.queryOrder
        ? `let position = topk[topkOffset + CURSOR_WORD] + index;
  if (position >= CHUNK_QUERY_COUNT) { return; }
  let localQuery = queryOrder[queryOrderOffset + position] - CHUNK_FIRST_ROW;
  let row = CHUNK_FIRST_ROW + localQuery;
  let header = topkOffset + index * STRIDE;`
        : `let localQuery = topk[topkOffset + CURSOR_WORD] + index;
  if (localQuery >= CHUNK_QUERY_COUNT) { return; }
  let row = CHUNK_FIRST_ROW + localQuery;
  let header = topkOffset + index * STRIDE;`
      : options.queryOrder
        ? `if (index >= CHUNK_QUERY_COUNT) { return; }
  let localQuery = queryOrder[queryOrderOffset + index] - CHUNK_FIRST_ROW;
  let row = CHUNK_FIRST_ROW + localQuery;
  let header = topkOffset + row * STRIDE;`
        : `if (index >= CHUNK_QUERY_COUNT) { return; }
  let localQuery = index;
  let row = CHUNK_FIRST_ROW + index;
  let header = topkOffset + row * STRIDE;`
  }
  var bound = bitcast<f32>(topk[header + 2u]);
  topk[header] = 0u;
  topk[header + 1u] = 0u;
  if (bitcast<u32>(bound) == bitcast<u32>(NEAREST_INVALID_DISTANCE) ||
      bitcast<u32>(bound) > bitcast<u32>(NEAREST_NO_DISTANCE)) { return; }
  var queryMinimum = vec2f(FLOAT32_MAXIMUM);
  var queryMaximum = vec2f(-FLOAT32_MAXIMUM);
  for (var qRing = qRingStart(localQuery); qRing < qRingEnd(localQuery); qRing++) {
    let vertexStart = qRingVertexStart(qRing);
    let vertexEnd = qRingVertexEnd(qRing);
    if (vertexEnd <= vertexStart || vertexEnd - vertexStart < ${getMinimumRingVertices(options.queryKind)}u) { continue; }
    for (var vertex = vertexStart; vertex < vertexEnd; vertex++) {
      let position = qVertex(vertex);
      if (isFiniteValue(position.x) && isFiniteValue(position.y)) {
        queryMinimum = min(queryMinimum, position);
        queryMaximum = max(queryMaximum, position);
      }
    }
  }
  if (queryMinimum.x > queryMaximum.x) { return; }

  ${storageBacked ? '' : `var listDistance: array<f32, ${capacity}>;\n  var listRow: array<u32, ${capacity}>;`}
  var count = 0u;
  var droppedMinimum = NEAREST_NO_DISTANCE;
  var stackNode: array<u32, ${options.stackSize}>;
  var stackLower: array<f32, ${options.stackSize}>;
  var top = 0u;
  let rootLower = boxLower(0u, queryMinimum, queryMaximum);
  if (nearestDistanceLess(rootLower, NEAREST_NO_DISTANCE)) {
    stackNode[0] = 0u;
    stackLower[0] = rootLower;
    top = 1u;
  }
  loop {
    if (top == 0u) { break; }
    top = top - 1u;
    let node = stackNode[top];
    let limit = widenNearestDistanceBound(bound);
    if (!nearestDistanceAtMost(stackLower[top], limit)) { continue; }
    if (node < INTERNAL_NODE_COUNT) {
      let first = node * 2u + 1u;
      let second = first + 1u;
      let firstLower = boxLower(first, queryMinimum, queryMaximum);
      let secondLower = boxLower(second, queryMinimum, queryMaximum);
      let firstUsable = nearestDistanceLess(firstLower, NEAREST_NO_DISTANCE) && nearestDistanceAtMost(firstLower, limit);
      let secondUsable = nearestDistanceLess(secondLower, NEAREST_NO_DISTANCE) && nearestDistanceAtMost(secondLower, limit);
      // Push the farther child first so the nearer one is popped first.
      if (firstUsable && secondUsable && top + 2u <= STACK_SIZE) {
        if (nearestDistanceAtMost(firstLower, secondLower)) {
          stackNode[top] = second; stackLower[top] = secondLower;
          stackNode[top + 1u] = first; stackLower[top + 1u] = firstLower;
        } else {
          stackNode[top] = first; stackLower[top] = firstLower;
          stackNode[top + 1u] = second; stackLower[top + 1u] = secondLower;
        }
        top = top + 2u;
      } else if (firstUsable && top + 1u <= STACK_SIZE) {
        stackNode[top] = first; stackLower[top] = firstLower;
        top = top + 1u;
      } else if (secondUsable && top + 1u <= STACK_SIZE) {
        stackNode[top] = second; stackLower[top] = secondLower;
        top = top + 1u;
      }
      continue;
    }
    let featureRow = nodeData[nodeDataOffset + node * ${NEAREST_NODE_WORDS}u + 4u];
    if (featureRow >= FEATURE_COUNT) { continue; }
    ${
      options.filter
        ? `let queryFilter = nodeDataOffset + ${options.filter.queryBase}u + row * 2u;
    let featureFilter = nodeDataOffset + ${options.filter.featureBase}u + featureRow * 2u;
    ${options.filter.exclusive ? 'if (nodeData[queryFilter] == nodeData[featureFilter]) { continue; }' : ''}
    ${options.filter.onAttribute ? 'if (nodeData[queryFilter + 1u] != nodeData[featureFilter + 1u]) { continue; }' : ''}`
        : ''
    }
    let candidate = pairResult(localQuery, featureRow).distanceSq;
    if (!nearestDistanceLess(candidate, NEAREST_NO_DISTANCE) || !nearestDistanceAtMost(candidate, bound)) { continue; }
    // Find the slot of (candidate, featureRow) in the ordered list.
    var position = count;
    loop {
      if (position == 0u) { break; }
      let previousDistance = ${readDistance('position - 1u')};
      if (nearestDistanceLess(previousDistance, candidate) || (previousDistance == candidate && ${readRow('position - 1u')} < featureRow)) { break; }
      position = position - 1u;
    }
    if (position >= CAPACITY) {
      droppedMinimum = minimumNearestDistance(droppedMinimum, candidate);
      continue;
    }
    var last = count;
    if (count == CAPACITY) {
      droppedMinimum = minimumNearestDistance(droppedMinimum, ${readDistance('CAPACITY - 1u')});
      last = CAPACITY - 1u;
    }
    var slot = last;
    loop {
      if (slot <= position) { break; }
      ${writeDistance('slot', readDistance('slot - 1u'))}
      ${writeRow('slot', readRow('slot - 1u'))}
      slot = slot - 1u;
    }
    ${writeDistance('position', 'candidate')}
    ${writeRow('position', 'featureRow')}
    count = min(count + 1u, CAPACITY);
    // Beyond k, keep only rows tied with the k-th distance.
    loop {
      if (count > K && nearestDistanceLess(${readDistance('K - 1u')}, ${readDistance('count - 1u')})) { count = count - 1u; } else { break; }
    }
    if (count >= K) { bound = minimumNearestDistance(bound, ${readDistance('K - 1u')}); }
  }
  var overflowFlag = 0u;
  if (${options.ties === 'all' ? 'true' : 'CAPACITY > K'} && count >= K && nearestDistanceAtMost(droppedMinimum, ${readDistance('K - 1u')})) { overflowFlag = 1u; }
  ${
    storageBacked
      ? ''
      : `for (var slot = 0u; slot < count; slot++) {
    topk[header + 3u + slot * 2u] = listRow[slot];
    topk[header + 4u + slot * 2u] = bitcast<u32>(listDistance[slot]);
  }`
  }
  topk[header] = count;
  topk[header + 1u] = overflowFlag;`;
  return {declarations, body};
}
