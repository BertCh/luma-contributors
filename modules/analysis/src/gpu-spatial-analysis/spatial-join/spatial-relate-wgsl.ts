// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  forEachEdge,
  getMinimumRingVertices,
  getPieceInsideWGSL,
  getSideWGSL,
  getSpatialPredicateCommonWGSL,
  type SpatialPredicateSide
} from './spatial-predicate-wgsl';
import {GPU_SPATIAL_RELATE_UNCERTAIN_BIT} from './spatial-relate-types';

/** Matrix cell index in `raise(a, b, dim)`: interior, boundary, exterior. */
const INTERIOR = '0u';
const BOUNDARY = '1u';
const EXTERIOR = '2u';

/** Emits a loop over every vertex of `feature`, binding `v` and the OGC part class `vc`. */
function forEachVertex(side: SpatialPredicateSide, feature: string, body: string): string {
  const p = side.prefix;
  const partClass =
    side.kind === 'points'
      ? INTERIOR
      : side.kind === 'polygons'
        ? BOUNDARY
        : `select(0u, 1u, ${p}IsBoundary(v, ${feature}))`;
  return `for (var ${p}r = ${p}RingStart(${feature}); ${p}r < ${p}RingEnd(${feature}); ${p}r++) {
    let ${p}vs = ${p}RingVertexStart(${p}r);
    let ${p}ve = ${p}RingVertexEnd(${p}r);
    if (${p}ve <= ${p}vs || ${p}ve - ${p}vs < ${getMinimumRingVertices(side.kind)}u) { continue; }
    let ${p}n = ${p}ve - ${p}vs;
    for (var ${p}k = relateLane; ${p}k < ${p}n; ${p}k += relateStride) {
      let v = ${p}Vertex(${p}vs + ${p}k);
      let vc = ${partClass};
      ${body}
    }
  }`;
}

/**
 * WGSL boolean: the crossing of edges `a-b` and `c-d` is an endpoint of the unclosed line `side`.
 * That point belongs to the line boundary (the vertex passes record it), not to the interior,
 * even when the line passes over it again. False for other kinds.
 */
function endpointCrossing(side: SpatialPredicateSide, feature: string): string {
  if (side.kind !== 'lines') {
    return 'false';
  }
  return `${side.prefix}CrossingOnBoundary(a, b, c, d, ${feature})`;
}

/** A crossing can coincide with an endpoint of another part; use the whole mod-2 boundary. */
function getCrossingBoundaryWGSL(side: SpatialPredicateSide): string {
  if (side.kind !== 'lines') return '';
  const prefix = side.prefix;
  return `fn ${prefix}IsBoundary(query: vec2f, feature: u32) -> bool {
  var endpoints = 0u;
  for (var part = ${prefix}RingStart(feature); part < ${prefix}RingEnd(feature); part++) {
    let start = ${prefix}RingVertexStart(part);
    let end = ${prefix}RingVertexEnd(part);
    if (end <= start || end - start < 2u) { continue; }
    let first = ${prefix}Vertex(start);
    let last = ${prefix}Vertex(end - 1u);
    if (first.x == last.x && first.y == last.y) { continue; }
    if (query.x == first.x && query.y == first.y) { endpoints++; }
    if (query.x == last.x && query.y == last.y) { endpoints++; }
  }
  return (endpoints & 1u) != 0u;
}
fn ${prefix}CrossingOnBoundary(firstStart: vec2f, firstEnd: vec2f, secondStart: vec2f, secondEnd: vec2f, feature: u32) -> bool {
  for (var part = ${prefix}RingStart(feature); part < ${prefix}RingEnd(feature); part++) {
    let start = ${prefix}RingVertexStart(part);
    let end = ${prefix}RingVertexEnd(part);
    if (end <= start || end - start < 2u) { continue; }
    let first = ${prefix}Vertex(start);
    let last = ${prefix}Vertex(end - 1u);
    if (first.x == last.x && first.y == last.y) { continue; }
    if (onSegment(firstStart, firstEnd, first) && onSegment(secondStart, secondEnd, first) && ${prefix}IsBoundary(first, feature)) { return true; }
    if (onSegment(firstStart, firstEnd, last) && onSegment(secondStart, secondEnd, last) && ${prefix}IsBoundary(last, feature)) { return true; }
  }
  return false;
}`;
}

/**
 * Like {@link forEachEdge} over the edges of `feature` (binding `c` and `d`) whose y range may reach
 * `[yLow, yHigh]`, using the per-pair y-slab index of a polygon side when it is active.
 *
 * An edge outside the slabs of the query range has a y range disjoint from it, so the callers' bodies
 * (which all start with a bounding-box rejection) would skip it without any orientation call. Edges
 * are bucketed into every slab their y range overlaps; each is visited once, in the first slab of
 * the query range that holds it. The closing edge of every ring is not indexed and is visited first.
 * Without an active index (small feature, or `withSlabIndex` off) this is the plain edge loop.
 * `body` may `return` or `continue`.
 */
function forEachSlabCandidateEdge(
  side: SpatialPredicateSide,
  feature: string,
  c: string,
  d: string,
  yLow: string,
  yHigh: string,
  body: string,
  withSlabIndex: boolean
): string {
  const plain = forEachEdge(side, feature, c, d, body);
  if (!withSlabIndex || side.kind !== 'polygons') {
    return plain;
  }
  const p = side.prefix;
  return `if (${p}SlabActive) {
    for (var ${p}r = ${p}RingStart(${feature}); ${p}r < ${p}RingEnd(${feature}); ${p}r++) {
      let ${p}vs = ${p}RingVertexStart(${p}r);
      let ${p}ve = ${p}RingVertexEnd(${p}r);
      if (${p}ve <= ${p}vs || ${p}ve - ${p}vs < ${getMinimumRingVertices(side.kind)}u) { continue; }
      let ${c} = ${p}Vertex(${p}ve - 1u);
      let ${d} = ${p}Vertex(${p}vs);
      ${body}
    }
    let ${p}QueryFirst = ${p}SlabOf(${yLow});
    let ${p}QueryLast = ${p}SlabOf(${yHigh});
    for (var ${p}Cursor = ${p}QueryFirst; ${p}Cursor <= ${p}QueryLast; ${p}Cursor++) {
      for (var ${p}Entry = ${p}SlabStarts[${p}Cursor]; ${p}Entry < ${p}SlabStarts[${p}Cursor + 1u]; ${p}Entry++) {
        let ${p}EntryEdge = ${p}SlabEntries[${p}Entry];
        let ${c} = ${p}Vertex(${p}EntryEdge);
        let ${d} = ${p}Vertex(${p}EntryEdge + 1u);
        if (max(${p}QueryFirst, ${p}SlabOf(min(${c}.y, ${d}.y))) != ${p}Cursor) { continue; }
        ${body}
      }
    }
  } else {
    ${plain}
  }`;
}

/** Declares the piece helpers of `side` acting as the "other" geometry of a pass. */
function getPieceHelpersWGSL(side: SpatialPredicateSide, withSlabIndex: boolean): string {
  const p = side.prefix;
  const box = `fn ${p}RelateBox(f: u32) -> vec4f {
  var box = vec4f(FLOAT32_MAXIMUM, FLOAT32_MAXIMUM, -FLOAT32_MAXIMUM, -FLOAT32_MAXIMUM);
  ${forEachEdge(
    side,
    'f',
    'a',
    'b',
    'box = vec4f(min(box.xy, min(a, b)), max(box.zw, max(a, b)));'
  )}
  return box;
}
// Locate, short-cut for points outside the bounding box of the feature (the box is closed).
fn ${p}RelateLocate(q: vec2f, f: u32) -> u32 {
  if (q.x < ${p}Box.x || q.x > ${p}Box.z || q.y < ${p}Box.y || q.y > ${p}Box.w) { return 0u; }
  ${withSlabIndex && side.kind === 'polygons' ? `if (${p}SlabActive) { return ${p}SlabLocate(q, f); }` : ''}
  return ${p}Locate(q, f);
}
// False when the segment a-b is outside the bounding box of the feature: nothing there can touch it.
fn ${p}RelateNearEdge(a: vec2f, b: vec2f) -> bool {
  return !(max(a.x, b.x) < ${p}Box.x || min(a.x, b.x) > ${p}Box.z ||
    max(a.y, b.y) < ${p}Box.y || min(a.y, b.y) > ${p}Box.w);
}`;
  if (side.kind === 'points') {
    return `${box}
fn ${p}RelateNextBreak(a: vec2f, b: vec2f, f: u32, t: f32) -> f32 { return 1.0; }
fn ${p}RelateCollectBreaks(a: vec2f, b: vec2f, f: u32) -> u32 { return 0u; }
fn ${p}RelateBreakAfter(count: u32, t: f32, cursor: ptr<function, u32>) -> f32 { return 1.0; }
fn ${p}RelatePieceCode(a: vec2f, b: vec2f, f: u32, t0: f32, t1: f32) -> u32 { return 0u; }`;
  }
  const classify =
    side.kind === 'polygons'
      ? `return select(0u, 2u, ${p}RelatePieceInside(a, b, f, tm));`
      : 'return 0u;';
  const pieceInside =
    side.kind === 'polygons'
      ? getPieceInsideWGSL(`${p}RelatePieceInside`, `${p}RelateLocate`, (yLow, yHigh, body) =>
          forEachSlabCandidateEdge(side, 'f', 'c', 'd', yLow, yHigh, body, withSlabIndex)
        )
      : '';
  return `${box}
${pieceInside}
// Next parameter after t in (0, 1] where segment a-b crosses or starts to overlap an edge.
fn ${p}RelateNextBreak(a: vec2f, b: vec2f, f: u32, t: f32) -> f32 {
  var best = 1.0;
  if (!${p}RelateNearEdge(a, b)) { return best; }
  ${forEachEdge(
    side,
    'f',
    'c',
    'd',
    `if (dot(d - c, d - c) == 0.0 || boxesDisjoint(a, b, c, d)) { continue; }
    let breaks = edgeBreaks(a, b, c, d);
    if (breaks.x >= 1.0 && breaks.y > t && breaks.y < best) { best = breaks.y; }
    if (breaks.x >= 2.0 && breaks.z > t && breaks.z < best) { best = breaks.z; }`
  )}
  return best;
}
// Every break parameter of a-b against the edges of the feature, ascending, in one pass over those
// edges: the piece loop then reads them in order instead of rescanning the edges for each piece.
// Returns the break count, or RELATE_BREAK_OVERFLOW when more than RELATE_BREAK_CAPACITY exist (the
// caller then falls back to \`RelateNextBreak\`, which finds each break by its own scan).
var<private> ${p}Breaks: array<f32, RELATE_BREAK_CAPACITY>;
// Whether some edge of the feature may be collinear with a-b, so \`RelateCovers\` has to run. The
// collection scan answers it once per edge instead of once per piece.
var<private> ${p}CoverNeeded: bool = false;
fn ${p}RelateCollectBreaks(a: vec2f, b: vec2f, f: u32) -> u32 {
  var count = 0u;
  ${p}CoverNeeded = false;
  if (!${p}RelateNearEdge(a, b)) { return 0u; }
  ${forEachSlabCandidateEdge(
    side,
    'f',
    'c',
    'd',
    'min(a.y, b.y)',
    'max(a.y, b.y)',
    `if (boxesDisjoint(a, b, c, d)) { continue; }
    if (orient(a, b, c) == 0.0 && orient(a, b, d) == 0.0) { ${p}CoverNeeded = true; }
    if (dot(d - c, d - c) == 0.0) { continue; }
    let breaks = edgeBreaks(a, b, c, d);
    let found = u32(breaks.x);
    for (var index = 0u; index < found; index++) {
      if (count == RELATE_BREAK_CAPACITY) { ${p}CoverNeeded = true; return RELATE_BREAK_OVERFLOW; }
      var position = count;
      let value = select(breaks.y, breaks.z, index == 1u);
      loop {
        if (position == 0u || ${p}Breaks[position - 1u] <= value) { break; }
        ${p}Breaks[position] = ${p}Breaks[position - 1u];
        position = position - 1u;
      }
      ${p}Breaks[position] = value;
      count = count + 1u;
    }`,
    withSlabIndex
  )}
  return count;
}
// First collected break after t, or 1.0. \`cursor\` only moves forward as t increases.
fn ${p}RelateBreakAfter(count: u32, t: f32, cursor: ptr<function, u32>) -> f32 {
  while (*cursor < count && ${p}Breaks[*cursor] <= t) { *cursor = *cursor + 1u; }
  return select(1.0, ${p}Breaks[*cursor], *cursor < count);
}
fn ${p}RelateCovers(a: vec2f, b: vec2f, f: u32, tm: f32) -> bool {
  let ab = b - a;
  let lengthSq = dot(ab, ab);
  if (!${p}RelateNearEdge(a, b)) { return false; }
  ${forEachEdge(
    side,
    'f',
    'c',
    'd',
    `if (boxesDisjoint(a, b, c, d)) { continue; }
    if (orient(a, b, c) == 0.0 && orient(a, b, d) == 0.0) {
      let tc = dot(c - a, ab) / lengthSq;
      let td = dot(d - a, ab) / lengthSq;
      if (min(tc, td) <= tm && tm <= max(tc, td)) { return true; }
    }`
  )}
  return false;
}
// 0: the piece [t0, t1] of a-b lies outside, 1: on the boundary or line, 2: strictly inside.
fn ${p}RelatePieceCode(a: vec2f, b: vec2f, f: u32, t0: f32, t1: f32) -> u32 {
  let tm = 0.5 * (t0 + t1);
  if (${p}CoverNeeded && ${p}RelateCovers(a, b, f, tm)) { return 1u; }
  ${classify}
}`;
}

/**
 * One direction of the relate computation: every part of `x` is located against `y`.
 *
 * - Vertices of `x` give point-sized (dimension 0) contributions.
 * - Proper crossings and collinear overlaps of `x` edges with `y` edges give dimension 0 and 1
 *   contributions between the two edge classes (polygon ring edges are boundary, line edges are
 *   interior).
 * - Pieces of each `x` edge between its crossings with `y` lie outside `y`, in its interior or on it,
 *   which gives dimension 1 contributions. For two polygons, the same pieces decide the areal cells:
 *   a piece inside `y` has `x` interior and `x` exterior both in the interior of `y`, a piece outside
 *   `y` has the `x` interior side outside `y`, and a piece shared with the boundary of `y` is probed
 *   on both sides.
 *
 * Running both directions covers every cell, because every area of the arrangement of two
 * polygons is bounded by edge pieces of one of them.
 */
function getDirectionWGSL(
  x: SpatialPredicateSide,
  y: SpatialPredicateSide,
  xIsLeft: boolean,
  withSlabIndex: boolean
): string {
  const fx = xIsLeft ? 'l' : 'r';
  const fy = xIsLeft ? 'r' : 'l';
  const raiseXY = (xClass: string, yClass: string, dimension: number) =>
    xIsLeft
      ? `raise(${xClass}, ${yClass}, ${dimension}u);`
      : `raise(${yClass}, ${xClass}, ${dimension}u);`;
  const xEdgeClass = x.kind === 'polygons' ? BOUNDARY : INTERIOR;
  const yEdgeClass = y.kind === 'polygons' ? BOUNDARY : INTERIOR;
  const probeAreas = (query: string, forward: string): string =>
    x.kind === 'polygons' && y.kind === 'polygons'
      ? `{
    let query = ${query};
    for (var side = 0u; side < 2u; side++) {
      let normalSide = select(1i, -1i, side == 1u);
      let inX = ${x.prefix}ProjectiveLocate(query, a, b, ${forward}, normalSide, ${fx}) == 1u;
      let inY = ${y.prefix}ProjectiveLocate(query, a, b, ${forward}, normalSide, ${fy}) == 1u;
      if (inX && inY) { ${raiseXY(INTERIOR, INTERIOR, 2)} }
      else if (inX) { ${raiseXY(INTERIOR, EXTERIOR, 2)} }
      else if (inY) { ${raiseXY(EXTERIOR, INTERIOR, 2)} }
    }
  }`
      : '';
  const parts: string[] = [];

  parts.push(
    forEachVertex(x, fx, raiseXY('vc', `locationClass(${y.prefix}RelateLocate(v, ${fy}))`, 0))
  );

  if (x.kind !== 'points' && y.kind !== 'points') {
    parts.push(
      forEachEdge(
        x,
        fx,
        'a',
        'b',
        `if (!${y.prefix}RelateNearEdge(a, b)) { continue; }
        ${forEachSlabCandidateEdge(
          y,
          fy,
          'c',
          'd',
          'min(a.y, b.y)',
          'max(a.y, b.y)',
          `if (boxesDisjoint(a, b, c, d)) { continue; }
      let o1 = orient(a, b, c);
      let o2 = orient(a, b, d);
      let o3 = orient(c, d, a);
      let o4 = orient(c, d, b);
      if (((o1 > 0.0 && o2 < 0.0) || (o1 < 0.0 && o2 > 0.0)) &&
          ((o3 > 0.0 && o4 < 0.0) || (o3 < 0.0 && o4 > 0.0)) &&
          !(${endpointCrossing(x, fx)} || ${endpointCrossing(y, fy)})) {
        ${raiseXY(xEdgeClass, yEdgeClass, 0)}
      }
      if (o1 == 0.0 && o2 == 0.0) {
        // Supplied coordinates order a collinear overlap exactly, even on an oblique edge.
        let useX = a.x != b.x;
        let low = max(select(min(a.y, b.y), min(a.x, b.x), useX), select(min(c.y, d.y), min(c.x, d.x), useX));
        let high = min(select(max(a.y, b.y), max(a.x, b.x), useX), select(max(c.y, d.y), max(c.x, d.x), useX));
        if (high > low) { ${raiseXY(xEdgeClass, yEdgeClass, 1)} }
      }`,
          withSlabIndex
        )}`,
        true
      )
    );
  }

  if (x.kind !== 'points' && y.kind !== 'polygons') {
    // Exterior means outside the UNION of all parts. Part-pair exterior maxima are wrong.
    // A non-degenerate edge has exterior dimension 1 exactly when interval-union coverage
    // fails. SegmentWithin merges supplied endpoint coordinates without projected fractions.
    parts.push(
      forEachEdge(
        x,
        fx,
        'a',
        'b',
        `
      if (a.x == b.x && a.y == b.y) { continue; }
      ${y.kind === 'points' ? raiseXY(xEdgeClass, EXTERIOR, 1) : `if ((${y.prefix}SegmentWithin(a, b, ${fy}) & 1u) == 0u) { ${raiseXY(xEdgeClass, EXTERIOR, 1)} }`}
    `,
        true
      )
    );
  }

  if (x.kind !== 'points' && y.kind === 'polygons') {
    const bothPolygons = x.kind === 'polygons' && y.kind === 'polygons';
    const areaCases = bothPolygons
      ? `if (code == 2u) {
        ${raiseXY(INTERIOR, INTERIOR, 2)}
        ${raiseXY(EXTERIOR, INTERIOR, 2)}
      }
      if (code == 0u) { ${raiseXY(INTERIOR, EXTERIOR, 2)} }
      ${xIsLeft ? 'if (code == 1u) {' : 'if (false) {'}
        // Symbolic side probes on an exact rational midpoint. No finite offset can be
        // guaranteed representable for a polygon only one ulp wide after translation.
        let query = spatialParameterQuery(a, b, t, next);
        for (var side = 0u; side < 2u; side++) {
          let normalSide = select(1i, -1i, side == 1u);
          let inX = ${x.prefix}ProjectiveLocate(query, a, b, true, normalSide, ${fx}) == 1u;
          let inY = ${y.prefix}ProjectiveLocate(query, a, b, true, normalSide, ${fy}) == 1u;
          if (inX && inY) { ${raiseXY(INTERIOR, INTERIOR, 2)} }
          else if (inX) { ${raiseXY(INTERIOR, EXTERIOR, 2)} }
          else if (inY) { ${raiseXY(EXTERIOR, INTERIOR, 2)} }
        }
      }`
      : '';
    parts.push(
      forEachEdge(
        x,
        fx,
        'a',
        'b',
        `let ab = b - a;
    if (dot(ab, ab) == 0.0) { continue; }
    if (${y.prefix}NeedsExactPieces(a, b, ${fy})) {
      let cells = ${y.prefix}ExactSegmentCells(a, b, ${fy});
      if ((cells & 1u) != 0u) { ${raiseXY(xEdgeClass, EXTERIOR, 1)} }
      if ((cells & 2u) != 0u) { ${raiseXY(xEdgeClass, INTERIOR, 1)} }
      if ((cells & 4u) != 0u) { ${raiseXY(xEdgeClass, BOUNDARY, 1)} }
      ${probeAreas('spatialEndpointQuery(a)', 'true')}
      ${
        bothPolygons
          ? forEachEdge(
              y,
              fy,
              'c',
              'd',
              `
        if (orient(c, d, a) * orient(c, d, b) >= 0.0 || !segmentsIntersect(a, b, c, d)) { continue; }
        ${probeAreas('spatialCrossingQuery(a, b, c, d)', 'false')}
        ${probeAreas('spatialCrossingQuery(a, b, c, d)', 'true')}
      `
            )
          : ''
      }
      continue;
    }
    var t = 0.0;
    let breakCount = ${y.prefix}RelateCollectBreaks(a, b, ${fy});
    var breakCursor = 0u;
    for (var guard = 0u; guard < ${y.prefix.toUpperCase()}_VERTEX_COUNT * 2u + 4u; guard++) {
      if (t >= 1.0) { break; }
      var next = 1.0;
      if (breakCount == RELATE_BREAK_OVERFLOW) {
        next = ${y.prefix}RelateNextBreak(a, b, ${fy}, t);
      } else {
        next = ${y.prefix}RelateBreakAfter(breakCount, t, &breakCursor);
      }
      let code = ${y.prefix}RelatePieceCode(a, b, ${fy}, t, next);
      if (code == 0u) { ${raiseXY(xEdgeClass, EXTERIOR, 1)} }
      if (code == 2u) { ${raiseXY(xEdgeClass, INTERIOR, 1)} }
      ${areaCases}
      t = next;
    }`,
        true
      )
    );
  }

  if (x.kind === 'polygons' && y.kind !== 'polygons') {
    // The interior of a polygon always extends outside a point or a line.
    parts.push(raiseXY(INTERIOR, EXTERIOR, 2));
  }
  return parts.join('\n  ');
}

/** Slabs of the per-pair y index of {@link getSlabIndexWGSL}. */
const SLAB_COUNT = 32;
/** Edge entries per side in workgroup memory; a feature needing more falls back to the full scan. */
const SLAB_ENTRY_CAPACITY = 1280;
/** Features with fewer edges than this are located by the plain scan. */
const SLAB_MINIMUM_EDGES = 48;

/** Emits `body` for every non-closing edge of the polygon feature `f` owned by this lane. */
function forEachSlabEdge(side: SpatialPredicateSide, body: string): string {
  const p = side.prefix;
  return `for (var ${p}r = ${p}RingStart(f); ${p}r < ${p}RingEnd(f); ${p}r++) {
    let ${p}vs = ${p}RingVertexStart(${p}r);
    let ${p}ve = ${p}RingVertexEnd(${p}r);
    if (${p}ve <= ${p}vs || ${p}ve - ${p}vs < ${getMinimumRingVertices(side.kind)}u) { continue; }
    for (var ${p}k = relateLane; ${p}k + 1u < ${p}ve - ${p}vs; ${p}k += relateStride) {
      let a = ${p}Vertex(${p}vs + ${p}k);
      let b = ${p}Vertex(${p}vs + ${p}k + 1u);
      let edge = ${p}vs + ${p}k;
      ${body}
    }
  }`;
}

/**
 * Returns WGSL for a per-pair y-slab index of one polygon side in workgroup memory, which turns the
 * point location of the relate engine from a scan of every edge into a scan of the edges whose y
 * range reaches the query's slab.
 *
 * \`${'${p}'}BuildSlabs(f, enabled)\` counts, scans and fills the index with the lanes of the workgroup
 * (three barriers, always reached, so call it from uniform control flow). It leaves
 * \`SlabActive\` false, and location falls back to the plain scan, for small features, a degenerate
 * or non-finite y extent, or more entries than the workgroup memory holds. Edges are bucketed by
 * the slabs their y range overlaps; the closing edge of every ring is not indexed and is always
 * tested. Both scans test exactly the edges the plain loop would act on (an edge whose y range
 * misses the query neither contains it nor straddles it, and costs no orientation call), and the
 * result is a union and a parity, so the order of the edges does not matter.
 */
function getSlabIndexWGSL(side: SpatialPredicateSide): string {
  const p = side.prefix;
  return `
var<workgroup> ${p}SlabCounts: array<atomic<u32>, ${SLAB_COUNT + 1}>;
var<workgroup> ${p}SlabFill: array<atomic<u32>, ${SLAB_COUNT}>;
var<workgroup> ${p}SlabStarts: array<u32, ${SLAB_COUNT + 1}>;
var<workgroup> ${p}SlabEntries: array<u32, ${SLAB_ENTRY_CAPACITY}>;
var<workgroup> ${p}SlabFlag: u32;
var<private> ${p}SlabActive: bool = false;
var<private> ${p}SlabLow: f32 = 0.0;
var<private> ${p}SlabScale: f32 = 0.0;
fn ${p}SlabOf(y: f32) -> u32 {
  let t = (y - ${p}SlabLow) * ${p}SlabScale;
  return select(0u, u32(min(t, ${SLAB_COUNT - 1}.0)), t > 0.0);
}
fn ${p}BuildSlabs(f: u32, enabled: bool) {
  ${p}SlabActive = false;
  if (relateLane <= ${SLAB_COUNT}u) { atomicStore(&${p}SlabCounts[relateLane], 0u); }
  if (relateLane < ${SLAB_COUNT}u) { atomicStore(&${p}SlabFill[relateLane], 0u); }
  workgroupBarrier();
  var usable = false;
  if (enabled) {
    let box = ${p}RelateBox(f);
    let range = box.w - box.y;
    var edges = 0u;
    for (var r = ${p}RingStart(f); r < ${p}RingEnd(f); r++) {
      let vs = ${p}RingVertexStart(r);
      let ve = ${p}RingVertexEnd(r);
      if (ve > vs && ve - vs >= ${getMinimumRingVertices(side.kind)}u) { edges += ve - vs; }
    }
    ${p}SlabLow = box.y;
    ${p}SlabScale = ${SLAB_COUNT}.0 / range;
    usable = edges >= ${SLAB_MINIMUM_EDGES}u && box.y >= -FLOAT32_MAXIMUM && box.w <= FLOAT32_MAXIMUM &&
      range > 0.0 && ${p}SlabScale <= FLOAT32_MAXIMUM;
  }
  if (usable) {
    ${forEachSlabEdge(
      side,
      `let first = ${p}SlabOf(min(a.y, b.y));
      let last = ${p}SlabOf(max(a.y, b.y));
      for (var slab = first; slab <= last; slab++) { atomicAdd(&${p}SlabCounts[slab], 1u); }`
    )}
  }
  workgroupBarrier();
  if (relateLane == 0u) {
    var running = 0u;
    for (var slab = 0u; slab < ${SLAB_COUNT}u; slab++) {
      ${p}SlabStarts[slab] = running;
      running += atomicLoad(&${p}SlabCounts[slab]);
    }
    ${p}SlabStarts[${SLAB_COUNT}] = running;
    ${p}SlabFlag = select(0u, 1u, usable && running <= ${SLAB_ENTRY_CAPACITY}u);
  }
  workgroupBarrier();
  ${p}SlabActive = ${p}SlabFlag != 0u;
  if (${p}SlabActive) {
    ${forEachSlabEdge(
      side,
      `let first = ${p}SlabOf(min(a.y, b.y));
      let last = ${p}SlabOf(max(a.y, b.y));
      for (var slab = first; slab <= last; slab++) {
        let position = atomicAdd(&${p}SlabFill[slab], 1u);
        ${p}SlabEntries[${p}SlabStarts[slab] + position] = edge;
      }`
    )}
  }
  workgroupBarrier();
}
// Same answer as ${p}Locate for a polygon: 2 on the boundary, 1 inside, 0 outside.
fn ${p}SlabLocate(q: vec2f, f: u32) -> u32 {
  var inside = false;
  for (var r = ${p}RingStart(f); r < ${p}RingEnd(f); r++) {
    let vs = ${p}RingVertexStart(r);
    let ve = ${p}RingVertexEnd(r);
    if (ve <= vs || ve - vs < ${getMinimumRingVertices(side.kind)}u) { continue; }
    let a = ${p}Vertex(ve - 1u);
    let b = ${p}Vertex(vs);
    if (onSegment(a, b, q)) { return 2u; }
    if ((a.y > q.y) != (b.y > q.y)) {
      let o = orient(a, b, q);
      if ((b.y > a.y && o > 0.0) || (b.y < a.y && o < 0.0)) { inside = !inside; }
    }
  }
  let slab = ${p}SlabOf(q.y);
  for (var i = ${p}SlabStarts[slab]; i < ${p}SlabStarts[slab + 1u]; i++) {
    let edge = ${p}SlabEntries[i];
    let a = ${p}Vertex(edge);
    let b = ${p}Vertex(edge + 1u);
    if (onSegment(a, b, q)) { return 2u; }
    if ((a.y > q.y) != (b.y > q.y)) {
      let o = orient(a, b, q);
      if ((b.y > a.y && o > 0.0) || (b.y < a.y && o < 0.0)) { inside = !inside; }
    }
  }
  return select(0u, 1u, inside);
}`;
}

/**
 * Returns module-scope WGSL declaring `pairRelate(l, r) -> u32` for one pair of geometry kinds.
 *
 * The result is the packed DE-9IM matrix (see `packGPUSpatialRelate`), `0` when either feature is
 * empty or non-finite, with {@link GPU_SPATIAL_RELATE_UNCERTAIN_BIT} set when an orientation test
 * could not be certified. The caller must bind `${prefix}Positions` for each side, plus
 * `${prefix}RingOffsets` for lines and polygons and `${prefix}FeatureRings` for polygons.
 *
 * The matrix is derived from structure, not sampling: vertex locations, edge crossings and
 * collinear overlaps, and edge pieces between crossings located against the other geometry. Every
 * orientation sign is exact (shared `orientSign`), so the uncertain bit is set only for input with
 * no sign (non-finite coordinates, product exponents spanning more than 200 bits). A piece is
 * located from an exactly located endpoint and the exact crossings between it and the piece, not
 * from its rounded midpoint. Colliding/adjacent f32 breaks use exact projective ordering and
 * symbolic locations; shared-boundary area cells use symbolic normal probes.
 *
 * @internal
 */
export function getSpatialRelateWGSL(
  left: SpatialPredicateSide,
  right: SpatialPredicateSide,
  withSlabIndex = false
): string {
  return `${getSpatialPredicateCommonWGSL(true)}
const RELATE_BREAK_CAPACITY: u32 = 24u;
const RELATE_BREAK_OVERFLOW: u32 = 0xffffffffu;
${getSideWGSL(left)}
${getSideWGSL(right)}
${getCrossingBoundaryWGSL(left)}
${getCrossingBoundaryWGSL(right)}
${getPieceHelpersWGSL(left, withSlabIndex)}
${getPieceHelpersWGSL(right, withSlabIndex)}
var<private> relateMatrix: u32 = 0u;
// Raises cell (a, b) of the matrix to at least dimension \`dimension\`.
fn raise(a: u32, b: u32, dimension: u32) {
  let shift = (a * 3u + b) * 2u;
  let current = (relateMatrix >> shift) & 3u;
  if (dimension + 1u > current) {
    relateMatrix = (relateMatrix & ~(3u << shift)) | ((dimension + 1u) << shift);
  }
}
// Maps an OGC point location (0 exterior, 1 interior, 2 boundary) to a matrix part (I=0, B=1, E=2).
fn locationClass(location: u32) -> u32 {
  return select(select(1u, 0u, location == 1u), 2u, location == 0u);
}
${
  withSlabIndex
    ? [left, right]
        .filter(side => side.kind === 'polygons')
        .map(getSlabIndexWGSL)
        .join('\n')
    : ''
}
var<private> relateLane: u32 = 0u;
var<private> relateStride: u32 = 1u;
var<private> leftBox: vec4f;
var<private> rightBox: vec4f;
// One lane of the matrix of (l, r): this lane handles every relateStride-th edge of the outer loops.
fn pairRelateLane(l: u32, r: u32) -> u32 {
  relateMatrix = 0u;
  uncertainOrientation = false;
  if (leftFirst(l).z == 0.0 || rightFirst(r).z == 0.0) { return 0u; }
  leftBox = leftRelateBox(l);
  rightBox = rightRelateBox(r);
  raise(${EXTERIOR}, ${EXTERIOR}, 2u);
  ${getDirectionWGSL(left, right, true, withSlabIndex)}
  ${getDirectionWGSL(right, left, false, withSlabIndex)}
  return relateMatrix | select(0u, ${GPU_SPATIAL_RELATE_UNCERTAIN_BIT}u, uncertainOrientation);
}
// The whole matrix of (l, r) in one invocation.
fn pairRelate(l: u32, r: u32) -> u32 {
  relateLane = 0u;
  relateStride = 1u;
  return pairRelateLane(l, r);
}`;
}

/** Invocations per pair of the workgroup relate kernel. @internal */
export const SPATIAL_RELATE_WORKGROUP_SIZE = 256;

/**
 * Returns module-scope WGSL declaring `relateWorkgroupPair(l, r, valid, lane) -> u32`, which splits
 * one pair across the `SPATIAL_RELATE_WORKGROUP_SIZE` lanes of a workgroup and merges the lane
 * matrices (cell-wise maximum, uncertain bit ORed). Append after {@link getSpatialRelateWGSL}.
 *
 * Every lane of the workgroup must call it with the same `(l, r, valid)` from uniform control
 * flow (it contains a barrier); only lane 0 receives the merged matrix.
 *
 * @internal
 */
export function getSpatialRelateWorkgroupWGSL(
  slabSides: readonly SpatialPredicateSide[] = []
): string {
  const size = SPATIAL_RELATE_WORKGROUP_SIZE;
  const slabBuilds = slabSides
    .filter(side => side.kind === 'polygons')
    .map(side => `${side.prefix}BuildSlabs(${side.prefix === 'left' ? 'l' : 'r'}, usable);`)
    .join('\n  ');
  return `
// Lane matrices merge with atomic ORs: each cell holds a thermometer code of its value (a value v
// sets the low v bits of a 3-bit field), so the OR of the lanes is the cell-wise maximum.
var<workgroup> relateMergedCells: atomic<u32>;
var<workgroup> relateMergedUncertain: atomic<u32>;
fn relateWorkgroupPair(l: u32, r: u32, valid: bool, lane: u32) -> u32 {
  relateLane = lane;
  relateStride = ${size}u;
  ${
    slabBuilds
      ? `// Per-pair y-slab indexes of the polygon sides (barriers: reached by every lane).
  let usable = valid && leftFirst(l).z != 0.0 && rightFirst(r).z != 0.0;
  ${slabBuilds}`
      : ''
  }
  if (valid) {
    let lanes = pairRelateLane(l, r);
    var thermometer = 0u;
    for (var cell = 0u; cell < 9u; cell++) {
      thermometer = thermometer | (((1u << ((lanes >> (cell * 2u)) & 3u)) - 1u) << (cell * 3u));
    }
    if (thermometer != 0u) { atomicOr(&relateMergedCells, thermometer); }
    if ((lanes & ${GPU_SPATIAL_RELATE_UNCERTAIN_BIT}u) != 0u) { atomicOr(&relateMergedUncertain, 1u); }
  }
  workgroupBarrier();
  var merged = 0u;
  if (lane == 0u) {
    let cells = atomicLoad(&relateMergedCells);
    for (var cell = 0u; cell < 9u; cell++) {
      merged = merged | (countOneBits((cells >> (cell * 3u)) & 7u) << (cell * 2u));
    }
    merged = merged | select(0u, ${GPU_SPATIAL_RELATE_UNCERTAIN_BIT}u, atomicLoad(&relateMergedUncertain) != 0u);
  }
  return merged;
}`;
}
