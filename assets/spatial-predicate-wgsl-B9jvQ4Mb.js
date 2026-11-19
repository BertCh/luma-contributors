import{t as e}from"./scaled-planar-distance-wgsl-BTfGg91B.js";import{t}from"./exact-orientation-wgsl-B_jI-HEc.js";import{c as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as r}from"./geodesic-wgsl-I0CYDKcz.js";import{r as i}from"./spatial-join-passes-CNxNpnDa.js";var a=1.2566,o=r,s={kind:`planar`},c=2**63,l=u(!0);function u(r,a=s){let l=r?`${t}
var<private> uncertainOrientation: bool = false;
fn orient(a: vec2f, b: vec2f, c: vec2f) -> f32 {
  // Coincident points make the determinant exactly zero. Shared edges hit this constantly, and an
  // exact zero is the one answer the f32 filter cannot certify (it falls to the slow exact path).
  if ((c.x == a.x && c.y == a.y) || (c.x == b.x && c.y == b.y) || (a.x == b.x && a.y == b.y)) {
    return 0.0;
  }
  let sign = orientSign(a, b, c);
  if (sign == 2) {
    uncertainOrientation = true;
    return 0.0;
  }
  return f32(sign);
}`:`fn orient(a: vec2f, b: vec2f, c: vec2f) -> f32 {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}`,u=a.kind===`great-circle`?`// Central angle in radians from p to the segment a-b, straight in longitude/latitude.
fn pointSegmentDistanceSq(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  return geodesicSegmentDistance(p, a, b, geodesicSegmentFraction(p, a, b), 1.0);
}`:`fn pointSegmentDistanceSq(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let ab = b - a;
  let lengthSq = dot(ab, ab);
  if (lengthSq == 0.0) { let d = p - a; return dot(d, d); }
  let t = clamp(dot(p - a, ab) / lengthSq, 0.0, 1.0);
  let d = p - (a + ab * t);
  return dot(d, d);
}`;return`${i}
const PLANAR_MAXIMUM_SQUAREABLE_DISTANCE: f32 = ${n(c)};
${a.kind===`great-circle`?o:e}
${l}
// Differs from nearest-wgsl: range test first, orient() may flag an uncertain sign.
fn onSegment(a: vec2f, b: vec2f, q: vec2f) -> bool {
  // The range test comes first so a sign that cannot matter is never reported as uncertain.
  return q.x >= min(a.x, b.x) && q.x <= max(a.x, b.x) &&
    q.y >= min(a.y, b.y) && q.y <= max(a.y, b.y) &&
    orient(a, b, q) == 0.0;
}
// True when the bounding boxes of segments a-b and c-d do not touch: no intersection is possible.
fn boxesDisjoint(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> bool {
  return max(a.x, b.x) < min(c.x, d.x) || max(c.x, d.x) < min(a.x, b.x) ||
    max(a.y, b.y) < min(c.y, d.y) || max(c.y, d.y) < min(a.y, b.y);
}
fn segmentsIntersect(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> bool {
  if (boxesDisjoint(a, b, c, d)) { return false; }
  let o1 = orient(a, b, c);
  let o2 = orient(a, b, d);
  let o3 = orient(c, d, a);
  let o4 = orient(c, d, b);
  if (((o1 > 0.0 && o2 < 0.0) || (o1 < 0.0 && o2 > 0.0)) &&
      ((o3 > 0.0 && o4 < 0.0) || (o3 < 0.0 && o4 > 0.0))) {
    return true;
  }
  return onSegment(a, b, c) || onSegment(a, b, d) || onSegment(c, d, a) || onSegment(c, d, b);
}
${u}
fn pointSegmentDistance(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  return ${a.kind===`planar`?`getPlanarPointSegmentDistance(p, a, b)`:`pointSegmentDistanceSq(p, a, b)`};
}
fn segmentDistance(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> f32 {
  if (segmentsIntersect(a, b, c, d)) { return 0.0; }
  return min(
    min(pointSegmentDistance(a, c, d), pointSegmentDistance(b, c, d)),
    min(pointSegmentDistance(c, a, b), pointSegmentDistance(d, a, b))
  );
}
fn segmentDistanceSq(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> f32 {
  if (segmentsIntersect(a, b, c, d)) { return 0.0; }
  return min(
    min(pointSegmentDistanceSq(a, c, d), pointSegmentDistanceSq(b, c, d)),
    min(pointSegmentDistanceSq(c, a, b), pointSegmentDistanceSq(d, a, b))
  );
}
fn crossProduct(a: vec2f, b: vec2f) -> f32 { return a.x * b.y - a.y * b.x; }
// Parameters t in the open interval (0, 1) along a-b where segment c-d crosses or overlaps it.
// Returns (count, t1, t2). Requires a != b.
fn edgeBreaks(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> vec3f {
  if (boxesDisjoint(a, b, c, d)) { return vec3f(0.0); }
  let ab = b - a;
  // Every branch is decided by orientation signs only. Rounded arithmetic (the cross product of the
  // two directions, divisions) only places a break that the signs already proved to exist: an f32
  // \`denominator == 0\` test lost near-parallel crossings and missed exactly collinear overlaps.
  let o1 = orient(a, b, c);
  let o2 = orient(a, b, d);
  if (o1 == 0.0 && o2 == 0.0) {
    // Collinear: c and d break a-b where they lie strictly between a and b. On a common line the
    // coordinate along a non-degenerate axis orders the points exactly.
    let useX = a.x != b.x;
    let low = select(min(a.y, b.y), min(a.x, b.x), useX);
    let high = select(max(a.y, b.y), max(a.x, b.x), useX);
    let cValue = select(c.y, c.x, useX);
    let dValue = select(d.y, d.x, useX);
    let lengthSq = dot(ab, ab);
    var count = 0.0;
    var first = 0.0;
    var second = 0.0;
    let tc = dot(c - a, ab) / lengthSq;
    let td = dot(d - a, ab) / lengthSq;
    // A break that rounds onto an endpoint is dropped: its piece is below the parameter resolution.
    if (cValue > low && cValue < high && tc > 0.0 && tc < 1.0) { first = tc; count = 1.0; }
    if (dValue > low && dValue < high && td > 0.0 && td < 1.0) {
      if (count == 0.0) { first = td; } else { second = td; }
      count = count + 1.0;
    }
    return vec3f(count, first, second);
  }
  if ((o1 > 0.0 && o2 > 0.0) || (o1 < 0.0 && o2 < 0.0)) { return vec3f(0.0); }
  // c-d must cross the open segment a-b: a and b strictly on opposite sides of c-d.
  let o3 = orient(c, d, a);
  let o4 = orient(c, d, b);
  if (!((o3 > 0.0 && o4 < 0.0) || (o3 < 0.0 && o4 > 0.0))) { return vec3f(0.0); }
  let t = crossingParameter(a, b, c, d);
  if (t > 0.0 && t < 1.0) { return vec3f(1.0, t, 0.0); }
  return vec3f(0.0);
}
// Rounded parameter along a-b of its crossing with c-d, for a and b strictly on opposite sides of
// c-d: t = s3 / (s3 - s4) with the signed areas of a and b against c-d. Algebraically equal to
// cross(c - a, cd) / cross(ab, cd), but it cannot divide by a denominator that rounded to zero,
// and when the rounded areas keep opposite signs t stays inside [0, 1]. edgeBreaks and
// shiftedLineCrossing share it, so a crossing gets the same f32 value as its break.
fn crossingParameter(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> f32 {
  let cd = d - c;
  let s3 = crossProduct(cd, a - c);
  let s4 = crossProduct(cd, b - c);
  let span = s3 - s4;
  return select(0.5, clamp(s3 / span, 0.0, 1.0), span != 0.0 && (s3 * s4) < 0.0);
}
const NO_SHIFTED_CROSSING: f32 = -2.0;
// Where the line through a-b, shifted infinitesimally toward its positive side, crosses the edge c-d:
// NO_SHIFTED_CROSSING when it does not, -1 at or before a, 2 at or beyond b, otherwise the
// crossingParameter of a crossing strictly inside a-b. A vertex on the line counts as on its
// negative side, so the shifted line crosses c-d when exactly one endpoint is strictly positive;
// crossings of all edges then flip the even-odd location exactly, through vertices and collinear
// runs too. Every decision is an exact orientation sign; only the inside parameter is rounded.
fn shiftedLineCrossing(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> f32 {
  let o1 = orient(a, b, c);
  let o2 = orient(a, b, d);
  if ((o1 > 0.0) == (o2 > 0.0)) { return NO_SHIFTED_CROSSING; }
  // orient(c, d, a + s (b - a)) is linear in s, increasing exactly when c is the positive endpoint.
  let direction = select(-1.0, 1.0, o1 > 0.0);
  if (orient(c, d, b) * direction <= 0.0) { return 2.0; }
  if (orient(c, d, a) * direction >= 0.0) { return -1.0; }
  return crossingParameter(a, b, c, d);
}
`}function d(e,t,n){return`fn ${e}(a: vec2f, b: vec2f, f: u32, tm: f32) -> bool {
  // Reference b counts crossings in (tm, 1], reference a crossings in [0, tm).
  var reference = ${t}(b, f);
  var low = tm;
  var high = 1.5;
  var bounded = true;
  var yLow = min(a.y, b.y);
  var yHigh = max(a.y, b.y);
  if (reference == 2u) {
    reference = ${t}(a, f);
    low = -0.5;
    high = tm;
    if (reference == 2u) {
      reference = 0u;
      low = tm;
      high = 3.0;
      bounded = false;
      yLow = -FLOAT32_MAXIMUM;
      yHigh = FLOAT32_MAXIMUM;
    }
  }
  var inside = reference == 1u;
  ${n(`yLow`,`yHigh`,`if (bounded && boxesDisjoint(a, b, c, d)) { continue; }
    let crossing = shiftedLineCrossing(a, b, c, d);
    if (crossing > low && crossing < high) { inside = !inside; }`)}
  return inside;
}`}function f(e){return e===`points`?1:e===`polygons`?3:2}function p(e,t,n,r,i,a=!1,o=!0){let s=e.prefix,c=e.kind===`points`?`1u`:e.kind===`polygons`?`${s}n`:`${s}n - 1u`,l=e.kind===`points`?`${s}k`:e.kind===`polygons`?`(${s}k + 1u) % ${s}n`:`${s}k + 1u`,u=o?`
      if (!(isFiniteValue(${n}.x) && isFiniteValue(${n}.y) && isFiniteValue(${r}.x) && isFiniteValue(${r}.y))) { continue; }`:``;return`for (var ${s}r = ${s}RingStart(${t}); ${s}r < ${s}RingEnd(${t}); ${s}r++) {
    let ${s}vs = ${s}RingVertexStart(${s}r);
    let ${s}ve = ${s}RingVertexEnd(${s}r);
    if (${s}ve <= ${s}vs || ${s}ve - ${s}vs < ${f(e.kind)}u) { continue; }
    let ${s}n = ${s}ve - ${s}vs;
    for (var ${s}k = ${a?`relateLane`:`0u`}; ${s}k < ${c}; ${s}k += ${a?`relateStride`:`1u`}) {
      let ${n} = ${s}Vertex(${s}vs + ${s}k);
      let ${r} = ${s}Vertex(${s}vs + ${l});${u}
      ${i}
    }
  }`}function m(e){let t=e.prefix,n=t.toUpperCase(),{kind:r}=e,i=e.partCount===void 0?r===`polygons`?`${t}FeatureRings[${t}FeatureRingsOffset + f * 2u]`:`f`:`${t}FeatureOffsets[${t}FeatureOffsetsOffset + f]`,a=e.partCount===void 0?r===`polygons`?`${t}FeatureRings[${t}FeatureRingsOffset + f * 2u + 1u]`:`f + 1u`:`${t}FeatureOffsets[${t}FeatureOffsetsOffset + f + 1u]`,o=r===`points`?`r`:`${t}RingOffsets[${t}RingOffsetsOffset + r]`,s=r===`points`?`r + 1u`:`min(${t}RingOffsets[${t}RingOffsetsOffset + r + 1u], ${n}_VERTEX_COUNT)`,c;c=r===`points`?`for (var part = ${t}RingStart(f); part < ${t}RingEnd(f); part++) {
    let v = ${t}Vertex(part);
    if (v.x == q.x && v.y == q.y) { return 1u; }
  }
  return 0u;`:r===`lines`?`var found = false;
  ${p(e,`f`,`a`,`b`,`if (onSegment(a, b, q)) { found = true; }`)}
  if (!found) { return 0u; }
  var endpointCount = 0u;
  for (var part = ${t}RingStart(f); part < ${t}RingEnd(f); part++) {
    let start = ${t}RingVertexStart(part);
    let end = ${t}RingVertexEnd(part);
    if (end <= start || end - start < 2u) { continue; }
    let first = ${t}Vertex(start);
    let last = ${t}Vertex(end - 1u);
    if (first.x == last.x && first.y == last.y) { continue; }
    if (q.x == first.x && q.y == first.y) { endpointCount++; }
    if (q.x == last.x && q.y == last.y) { endpointCount++; }
  }
  return select(1u, 2u, (endpointCount & 1u) != 0u);`:`var inside = false;
  ${p(e,`f`,`a`,`b`,`if (onSegment(a, b, q)) { return 2u; }
      if ((a.y > q.y) != (b.y > q.y)) {
        let o = orient(a, b, q);
        if ((b.y > a.y && o > 0.0) || (b.y < a.y && o < 0.0)) { inside = !inside; }
      }`)}
  return select(0u, 1u, inside);`;let l=``;return r===`lines`?l=`
// Bit 0: segment a-b lies entirely on the linestring. Bit 1: it overlaps the linestring interior.
fn ${t}SegmentWithin(a: vec2f, b: vec2f, f: u32) -> u32 {
  if (a.x == b.x && a.y == b.y) {
    let location = ${t}Locate(a, f);
    return select(0u, select(1u, 3u, location == 1u), location != 0u);
  }
  let useX = abs(b.x - a.x) >= abs(b.y - a.y);
  let first = select(a.y, a.x, useX);
  let last = select(b.y, b.x, useX);
  let limit = max(first, last);
  var reach = min(first, last);
  var passes = 0u;
  loop {
    if (reach >= limit || passes > ${n}_VERTEX_COUNT) { break; }
    passes = passes + 1u;
    var progressed = false;
    ${p(e,`f`,`c`,`d`,`if (orient(a, b, c) == 0.0 && orient(a, b, d) == 0.0) {
        let tc = select(c.y, c.x, useX);
        let td = select(d.y, d.x, useX);
        let low = min(tc, td);
        let high = max(tc, td);
        if (low <= reach && high > reach) { reach = min(high, limit); progressed = true; }
      }`)}
    if (!progressed) { break; }
  }
  return select(0u, 3u, reach >= limit);
}`:r===`polygons`&&(l=`
fn ${t}CollinearCovers(a: vec2f, b: vec2f, f: u32, tm: f32) -> bool {
  let ab = b - a;
  let lengthSq = dot(ab, ab);
  ${p(e,`f`,`c`,`d`,`if (orient(a, b, c) == 0.0 && orient(a, b, d) == 0.0) {
      let tc = dot(c - a, ab) / lengthSq;
      let td = dot(d - a, ab) / lengthSq;
      if (min(tc, td) <= tm && tm <= max(tc, td)) { return true; }
    }`)}
  return false;
}
fn ${t}NextBreak(a: vec2f, b: vec2f, f: u32, t: f32) -> f32 {
  var best = 1.0;
  ${p(e,`f`,`c`,`d`,`let breaks = edgeBreaks(a, b, c, d);
    if (breaks.x >= 1.0 && breaks.y > t && breaks.y < best) { best = breaks.y; }
    if (breaks.x >= 2.0 && breaks.z > t && breaks.z < best) { best = breaks.z; }`)}
  return best;
}
${d(`${t}PieceInside`,`${t}Locate`,(t,n,r)=>p(e,`f`,`c`,`d`,r))}
// 0: the piece starting at t0 lies outside, 1: on the boundary, 2: strictly inside.
fn ${t}PieceCode(a: vec2f, b: vec2f, f: u32, t0: f32) -> u32 {
  let tm = 0.5 * (t0 + ${t}NextBreak(a, b, f, t0));
  if (${t}CollinearCovers(a, b, f, tm)) { return 1u; }
  return select(0u, 2u, ${t}PieceInside(a, b, f, tm));
}
// Bit 0: segment a-b lies in the closed polygon. Bit 1: part of it lies strictly inside.
fn ${t}SegmentWithin(a: vec2f, b: vec2f, f: u32) -> u32 {
  let ab = b - a;
  if (dot(ab, ab) == 0.0) {
    let location = ${t}Locate(a, f);
    return select(0u, select(1u, 3u, location == 1u), location != 0u);
  }
  var interior = false;
  var code = ${t}PieceCode(a, b, f, 0.0);
  if (code == 0u) { return 0u; }
  interior = code == 2u;
  ${p(e,`f`,`c`,`d`,`let breaks = edgeBreaks(a, b, c, d);
    if (breaks.x >= 1.0) {
      code = ${t}PieceCode(a, b, f, breaks.y);
      if (code == 0u) { return 0u; }
      interior = interior || code == 2u;
    }
    if (breaks.x >= 2.0) {
      code = ${t}PieceCode(a, b, f, breaks.z);
      if (code == 0u) { return 0u; }
      interior = interior || code == 2u;
    }`)}
  return select(1u, 3u, interior);
}`),`
const ${n}_VERTEX_COUNT: u32 = ${e.vertexCount}u;
fn ${t}RingStart(f: u32) -> u32 { ${e.partCount===void 0?``:`if (${i} > ${a} || ${a} > ${e.partCount}u) { return 0u; }`} return ${i}; }
fn ${t}RingEnd(f: u32) -> u32 { ${e.partCount===void 0?``:`if (${i} > ${a} || ${a} > ${e.partCount}u) { return 0u; }`} return ${a}; }
fn ${t}RingVertexStart(r: u32) -> u32 { return ${o}; }
fn ${t}RingVertexEnd(r: u32) -> u32 { return ${s}; }
fn ${t}Vertex(i: u32) -> vec2f {
  return vec2f(${t}Positions[${t}PositionsOffset + i * 2u], ${t}Positions[${t}PositionsOffset + i * 2u + 1u]);
}
// (x, y, valid): first vertex of the first usable ring; valid is 0 for an empty feature.
fn ${t}First(f: u32) -> vec3f {
  for (var r = ${t}RingStart(f); r < ${t}RingEnd(f); r++) {
    let vs = ${t}RingVertexStart(r);
    let ve = ${t}RingVertexEnd(r);
    if (ve > vs && ve - vs >= ${f(r)}u) {
      let v = ${t}Vertex(vs);
      return vec3f(v, select(0.0, 1.0, isFiniteValue(v.x) && isFiniteValue(v.y)));
    }
  }
  return vec3f(0.0);
}
// 0: exterior, 1: interior, 2: boundary (OGC point location).
fn ${t}Locate(q: vec2f, f: u32) -> u32 {
  ${c}
}
fn ${t}EdgesIntersect(qa: vec2f, qb: vec2f, f: u32) -> bool {
  ${p(e,`f`,`c`,`d`,`if (segmentsIntersect(qa, qb, c, d)) { return true; }`)}
  return false;
}
fn ${t}MinimumDistance(qa: vec2f, qb: vec2f, f: u32) -> f32 {
  var best = FLOAT32_MAXIMUM;
  ${p(e,`f`,`c`,`d`,`best = min(best, segmentDistance(qa, qb, c, d));`)}
  return best;
}
fn ${t}MinimumDistanceSq(qa: vec2f, qb: vec2f, f: u32) -> f32 {
  var best = FLOAT32_MAXIMUM;
  ${p(e,`f`,`c`,`d`,`best = min(best, segmentDistanceSq(qa, qb, c, d));`)}
  return best;
}${l}
`}function h(e,t){return`
fn pairIntersects(l: u32, r: u32) -> bool {
  if (leftFirst(l).z == 0.0 || rightFirst(r).z == 0.0) { return false; }
  ${p(e,`l`,`la`,`lb`,`if (rightEdgesIntersect(la, lb, r)) { return true; }
    ${t.kind===`polygons`?`if (rightLocate(la, r) != 0u) { return true; }`:``}`)}
  // Every component can be contained without a boundary crossing; the first feature vertex alone
  // cannot detect containment of a later polygon in a multipolygon.
  ${e.kind===`polygons`?p(t,`r`,`ra`,`rb`,`if (leftLocate(ra, l) != 0u) { return true; }`):``}
  return false;
}`}function g(e){return`
fn pairDistance(l: u32, r: u32) -> f32 {
  if (leftFirst(l).z == 0.0 || rightFirst(r).z == 0.0) { return FLOAT32_MAXIMUM; }
  if (pairIntersects(l, r)) { return 0.0; }
  var best = FLOAT32_MAXIMUM;
  ${p(e,`l`,`la`,`lb`,`best = min(best, rightMinimumDistance(la, lb, r));`)}
  return best;
}
fn pairDistanceSq(l: u32, r: u32) -> f32 {
  if (leftFirst(l).z == 0.0 || rightFirst(r).z == 0.0) { return FLOAT32_MAXIMUM; }
  if (pairIntersects(l, r)) { return 0.0; }
  var best = FLOAT32_MAXIMUM;
  ${p(e,`l`,`la`,`lb`,`best = min(best, rightMinimumDistanceSq(la, lb, r));`)}
  return best;
}`}function _(e){return`
fn ${e.prefix}ProbeInside(origin: vec2f, ahead: vec2f, leftSide: bool, f: u32) -> bool {
  // probe.y - origin.y = s (ahead.y - origin.y) +/- t (ahead.x - origin.x).
  let probeAbove = ahead.y > origin.y || (ahead.y == origin.y && (ahead.x > origin.x) == leftSide);
  // On the line through origin and ahead a non-degenerate axis orders points exactly.
  let useX = ahead.x != origin.x;
  let aheadPositive = select(ahead.y > origin.y, ahead.x > origin.x, useX);
  var inside = false;
  ${p(e,`f`,`a`,`b`,`let aAbove = a.y > origin.y || (a.y == origin.y && !probeAbove);
      let bAbove = b.y > origin.y || (b.y == origin.y && !probeAbove);
      if (aAbove != bAbove) {
        // orient(a, b, probe) = orient(a, b, origin) + s cross(b - a, ahead - origin) +/- t dot(b - a, ahead - origin).
        // With origin on the line a-b the second coefficient is orient(a, b, ahead). With ahead on
        // it too the edge is parallel to the direction, and the third is positive when it points the same way.
        var o = orient(a, b, origin);
        if (o == 0.0) { o = orient(a, b, ahead); }
        if (o == 0.0) {
          let edgePositive = select(b.y > a.y, b.x > a.x, useX);
          o = select(-1.0, 1.0, (edgePositive == aheadPositive) == leftSide);
        }
        if ((b.y > a.y) == (o > 0.0)) { inside = !inside; }
      }`)}
  return inside;
}`}function v(e,t,n=!1){let r=e.prefix,i=t.prefix,a,o=``;return t.kind===`points`?a=`var interior = false;
  ${p(t,`inner`,`a`,`b`,`let location = ${r}Locate(a, outer);
    if (location == 0u) { return false; }
    interior = interior || location == 1u;`)}
  return ${n?`true`:`interior`};`:e.kind===`points`||e.kind===`lines`&&t.kind===`polygons`?a=`return false;`:e.kind===`lines`?a=`${p(t,`inner`,`ia`,`ib`,`if ((${r}SegmentWithin(ia, ib, outer) & 1u) == 0u) { return false; }`)}
  return true;`:t.kind===`lines`?a=`var interior = false;
  ${p(t,`inner`,`ia`,`ib`,`let within = ${r}SegmentWithin(ia, ib, outer);
    if ((within & 1u) == 0u) { return false; }
    interior = interior || (within & 2u) != 0u;`)}
  return ${n?`true`:`interior`};`:(o=`${_(e)}
${_(t)}`,a=`${p(t,`inner`,`ia`,`ib`,`if ((${r}SegmentWithin(ia, ib, outer) & 1u) == 0u) { return false; }`)}
  ${p(e,`outer`,`oa`,`ob`,`let location = ${i}Locate(oa, inner);
      if (location == 1u) { return false; }
      if (location == 2u && (oa.x != ob.x || oa.y != ob.y)) {
        if (${i}ProbeInside(oa, ob, true, inner) && !${r}ProbeInside(oa, ob, true, outer)) { return false; }
        if (${i}ProbeInside(oa, ob, false, inner) && !${r}ProbeInside(oa, ob, false, outer)) { return false; }
      }`)}
  return true;`),`${o}
fn pairContains(outer: u32, inner: u32) -> bool {
  if (${r}First(outer).z == 0.0 || ${i}First(inner).z == 0.0) { return false; }
  ${a}
}`}function y(e,t=s){if(t.kind===`great-circle`){let r=n(Math.fround(t.radius)),i=n(Math.fround(a));if(typeof e==`number`){let r=Math.fround(Math.fround(e)/Math.fround(t.radius));return`fn distanceLimitSq() -> f32 { return ${n(r<=Math.fround(1.2566)?r:-1)}; }`}return`fn distanceLimitSq() -> f32 {
  let distance = ${e.expression};
  let angle = distance / ${r};
  return select(-1.0, angle, distance >= 0.0 && angle <= ${i});
}`}return`fn distanceLimit() -> f32 {
  let distance = ${typeof e==`number`?n(Math.fround(e)):e.expression};
  return select(-1.0, distance, distance >= 0.0 && distance <= FLOAT32_MAXIMUM);
}
fn distanceLimitSq() -> f32 {
  let distance = distanceLimit();
  if (distance > PLANAR_MAXIMUM_SQUAREABLE_DISTANCE) { return FLOAT32_MAXIMUM; }
  return select(-1.0, distance * distance, distance >= 0.0);
}`}function b(e,t,n,r,i=s){let a;switch(n){case`intersects`:a=`${h(e,t)}
fn pairMatches(l: u32, r: u32) -> bool { return pairIntersects(l, r); }`;break;case`contains`:a=`${v(e,t)}
fn pairMatches(l: u32, r: u32) -> bool { return pairContains(l, r); }`;break;case`within`:a=`${v(t,e)}
fn pairMatches(l: u32, r: u32) -> bool { return pairContains(r, l); }`;break;case`coveredBy`:a=`${v(t,e,!0)}
fn pairMatches(l: u32, r: u32) -> bool { return pairContains(r, l); }`;break;case`dwithin`:a=`${h(e,t)}
${g(e)}
${y(r,i)}
fn pairMatches(l: u32, r: u32) -> bool {
  ${i.kind===`planar`?`if (distanceLimit() > PLANAR_MAXIMUM_SQUAREABLE_DISTANCE) {
    return pairDistance(l, r) <= distanceLimit();
  }`:``}
  return pairDistanceSq(l, r) <= distanceLimitSq();
}`;break;default:throw Error(`predicate ${n} is evaluated by the relate engine`)}return`${i.kind===`planar`?l:u(!0,i)}
${m(e)}
${m(t)}
${a}`}export{u as a,o as c,m as i,f as n,b as o,d as r,a as s,p as t};