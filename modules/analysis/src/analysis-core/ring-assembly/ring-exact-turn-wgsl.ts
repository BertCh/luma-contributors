// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {EXACT_GEOMETRY_WGSL} from '../kernels/exact-geometry-wgsl';
import {EXACT_ORIENTATION_WGSL} from '../kernels/exact-orientation-wgsl';

/**
 * Exact turn classification for the planar successor choice (`match` kernel).
 *
 * `ringExactTurnClass(origin, vertex, end, side)` places the turn from `vertex - origin` to
 * `end - vertex` in the order of its angle toward the interior side, in (-pi, pi]: 0 for a turn
 * away from the interior side, 1 for straight on (also a zero-length incoming or outgoing
 * segment), 2 for a turn toward the interior side or a u-turn. The caller supplies `side` through
 * its single `exactOrientSigns` call chain and disables exact choice when it cannot be certified.
 * It agrees with the sign classes of `ringTurnKey`, so the exact order
 * (class, then `orientSign` between two candidates of one open half-plane) is the f32 key order
 * without its rounding. The collinear case compares coordinates only.
 *
 * The two interior sides are mirror images: the orientation signs are multiplied by `SIDE_ORDER`
 * (1 for `'left'`, -1 for `'right'`) before they are classed or compared, so the kernel for
 * `'right'` decides as the kernel for `'left'` does on the mirror image of the input. In
 * particular an exact u-turn is the last class for both: the face rule takes the first edge met
 * when turning from the reversed incoming direction toward the interior side, and an edge that
 * exactly reverses the incoming one is met first whichever way that turn goes. (Until the
 * re-review of 2026-11-11 the turn was classed left or right and the order reversed for
 * `'right'`, which put the u-turn first for `'left'` and last for `'right'`.)
 *
 * `ringSegmentTurnClass(origin, vertex, start, end)` gives the same classes for the turn from the
 * segment `origin -> vertex` to the segment `start -> end`, which need not start at `vertex`. It
 * is used when a candidate starts within the tolerance of the end vertex but not on it, so that
 * there is no common vertex to measure the turn at. The order is that of the directions
 * `vertex - origin` and `end - start` as real vectors, not of their f32 differences:
 * `ringCrossSign(a, b, c, d)` is the exact sign of `cross(b - a, d - c)` for four f32 points
 * (`-1`, `0`, `1`, or `2` when it cannot be certified). The eight products of raw coordinates of
 *
 *     bx dy - bx cy - ax dy + ax cy - by dx + by cx + ay dx - ay cx
 *
 * are summed exactly with the integer machinery of `orientSign` (`multiply24`,
 * `accumulateShifted`: 48-bit products aligned to the smallest exponent in two 256-bit
 * accumulators), behind a float filter on the rounded differences. With `c == b` it is
 * `orientSign(a, b, d)`. The float filter and the exact sum hold no table: the terms are
 * recomputed instead of stored in a local array (contributor contract, section 11).
 *
 * None of these signs changes when one axis is scaled by a positive factor, so the same order
 * holds for longitude/latitude input, whose turns are measured with longitude differences scaled
 * by the cosine of the latitude.
 */
export const RING_EXACT_TURN_WGSL = /* wgsl */ `${EXACT_ORIENTATION_WGSL}
${EXACT_GEOMETRY_WGSL}
fn ringExactTurnClass(origin: vec2f, vertex: vec2f, end: vec2f, side: i32) -> i32 {
  // Positive toward the interior side.
  let toward = side * SIDE_ORDER;
  if (toward < 0) { return 0; }
  if (toward > 0) { return 2; }
  // Collinear: behind the vertex (dot product negative) is the u-turn.
  var behind = false;
  if (vertex.x != origin.x) {
    behind = (vertex.x > origin.x && end.x < vertex.x) || (vertex.x < origin.x && end.x > vertex.x);
  } else if (vertex.y != origin.y) {
    behind = (vertex.y > origin.y && end.y < vertex.y) || (vertex.y < origin.y && end.y > vertex.y);
  }
  return select(1, 2, behind);
}
// Bit patterns of the two factors of term 'term' of
// cross(b - a, d - c) = bx dy - bx cy - ax dy + ax cy - by dx + by cx + ay dx - ay cx.
// The sign of the term is folded into the first factor.
fn ringCrossFactors(term: u32, a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> vec2u {
  let negate = 0x80000000u;
  switch (term) {
    case 0u: { return vec2u(bitcast<u32>(b.x), bitcast<u32>(d.y)); }
    case 1u: { return vec2u(bitcast<u32>(b.x) ^ negate, bitcast<u32>(c.y)); }
    case 2u: { return vec2u(bitcast<u32>(a.x) ^ negate, bitcast<u32>(d.y)); }
    case 3u: { return vec2u(bitcast<u32>(a.x), bitcast<u32>(c.y)); }
    case 4u: { return vec2u(bitcast<u32>(b.y) ^ negate, bitcast<u32>(d.x)); }
    case 5u: { return vec2u(bitcast<u32>(b.y), bitcast<u32>(c.x)); }
    case 6u: { return vec2u(bitcast<u32>(a.y), bitcast<u32>(d.x)); }
    default: { return vec2u(bitcast<u32>(a.y) ^ negate, bitcast<u32>(c.x)); }
  }
}

// One term as an exact 48-bit product times 2^exponent; 'present' is false for a zero term.
struct RingCrossTerm {
  low: u32,
  high: u32,
  exponent: i32,
  negative: bool,
  present: bool,
}

fn ringCrossTerm(term: u32, a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> RingCrossTerm {
  let factors = ringCrossFactors(term, a, b, c, d);
  var result = RingCrossTerm(0u, 0u, 0, false, false);
  let firstMagnitude = factors.x & 0x7fffffffu;
  let secondMagnitude = factors.y & 0x7fffffffu;
  if (firstMagnitude == 0u || secondMagnitude == 0u) { return result; }
  let firstExponentBits = firstMagnitude >> 23u;
  let secondExponentBits = secondMagnitude >> 23u;
  let firstFraction = firstMagnitude & 0x7fffffu;
  let secondFraction = secondMagnitude & 0x7fffffu;
  let firstSignificand = select(0x800000u | firstFraction, firstFraction, firstExponentBits == 0u);
  let secondSignificand = select(0x800000u | secondFraction, secondFraction, secondExponentBits == 0u);
  let firstExponent = select(i32(firstExponentBits) - 150, -149, firstExponentBits == 0u);
  let secondExponent = select(i32(secondExponentBits) - 150, -149, secondExponentBits == 0u);
  let product = multiply24(firstSignificand, secondSignificand);
  result.low = product.x;
  result.high = product.y;
  result.exponent = firstExponent + secondExponent;
  result.negative = ((factors.x ^ factors.y) >> 31u) == 1u;
  result.present = true;
  return result;
}

fn ringCrossExact(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> i32 {
  var minimumExponent = 100000;
  var maximumExponent = -100000;
  for (var term = 0u; term < 8u; term++) {
    let value = ringCrossTerm(term, a, b, c, d);
    if (!value.present) { continue; }
    minimumExponent = min(minimumExponent, value.exponent);
    maximumExponent = max(maximumExponent, value.exponent);
  }
  if (minimumExponent > maximumExponent) { return 0; }
  if (maximumExponent - minimumExponent > ORIENT_MAXIMUM_SPAN) { return 2; }
  var positive: array<u32, 8>;
  var negative: array<u32, 8>;
  for (var term = 0u; term < 8u; term++) {
    let value = ringCrossTerm(term, a, b, c, d);
    if (!value.present) { continue; }
    let shift = u32(value.exponent - minimumExponent);
    if (value.negative) {
      accumulateShifted(&negative, value.low, value.high, shift);
    } else {
      accumulateShifted(&positive, value.low, value.high, shift);
    }
  }
  for (var slot = 8u; slot > 0u; slot--) {
    let plus = positive[slot - 1u];
    let minus = negative[slot - 1u];
    if (plus != minus) { return select(-1, 1, plus > minus); }
  }
  return 0;
}

// Exact sign of cross(b - a, d - c) for four f32 points: -1, 0, 1, or 2 when it cannot be
// certified (non-finite coordinates, or terms that span more than 200 binary orders).
fn ringCrossSign(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> i32 {
  if (!(orientIsFinite(a.x) && orientIsFinite(a.y) && orientIsFinite(b.x) && orientIsFinite(b.y) &&
        orientIsFinite(c.x) && orientIsFinite(c.y) && orientIsFinite(d.x) && orientIsFinite(d.y))) {
    return 2;
  }
  // Each difference is rounded once and each product once, so the float determinant is within
  // 2.5e-7 of the magnitude sum of the exact one; the bound keeps a factor of two. Products near
  // the smallest f32 values lose more than that and go to the exact sum.
  let first = b - a;
  let second = d - c;
  let detLeft = first.x * second.y;
  let detRight = first.y * second.x;
  let det = detLeft - detRight;
  let magnitude = abs(detLeft) + abs(detRight);
  let bound = 5.0e-7 * magnitude;
  if (magnitude > 1.0e-30) {
    if (det > bound) { return 1; }
    if (-det > bound) { return -1; }
  }
  return ringCrossExact(a, b, c, d);
}

// The same classes for the turn from segment origin -> vertex to segment start -> end (neither
// of zero length), by the exact sign of the cross product of their directions.
fn ringSegmentTurnClass(origin: vec2f, vertex: vec2f, start: vec2f, end: vec2f) -> i32 {
  let side = ringCrossSign(origin, vertex, start, end);
  if (side == 2) { return -1; }
  // Positive toward the interior side.
  let toward = side * SIDE_ORDER;
  if (toward < 0) { return 0; }
  if (toward > 0) { return 2; }
  // Parallel: opposite signs of a non-zero component are the u-turn. A component of one
  // direction is zero exactly when it is zero in the other.
  var behind = false;
  if (vertex.x != origin.x) {
    behind = (vertex.x > origin.x) != (end.x > start.x);
  } else {
    behind = (vertex.y > origin.y) != (end.y > start.y);
  }
  return select(1, 2, behind);
}
`;
