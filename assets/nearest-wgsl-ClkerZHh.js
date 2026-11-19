import{n as e}from"./storage-backed-top-k-Dm9YLNx9.js";import{c as t,n,t as r}from"./spatial-predicate-wgsl-B9jvQ4Mb.js";import{u as i}from"./wgsl-common-DfyXiYB0.js";import{t as a}from"./exact-orientation-wgsl-B_jI-HEc.js";import{t as o}from"./scaled-planar-distance-wgsl-BTfGg91B.js";var s=2147483646,c=`
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
`,l=`${`
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
const NO_FEATURE: u32 = 0xffffffffu;
const NO_SEGMENT: u32 = 0xffffffffu;
${i}
${c}
`}
${a}
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
`,u=`${l}
${o}
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
`,d=`${l}
${t}
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
`;function f(e){let t=e.prefix,i=t.toUpperCase(),{kind:a}=e,o=`f`,s=`f + 1u`,c=`r`,l=`r + 1u`,u=`vec2f(${t}Positions[${t}PositionsOffset + i * 2u], ${t}Positions[${t}PositionsOffset + i * 2u + 1u])`;if(a===`segments`)c=`2u * r`,l=`2u * r + 2u`,u=`select(
    vec2f(${t}Ends[${t}EndsOffset + (i >> 1u) * 2u], ${t}Ends[${t}EndsOffset + (i >> 1u) * 2u + 1u]),
    vec2f(${t}Starts[${t}StartsOffset + (i >> 1u) * 2u], ${t}Starts[${t}StartsOffset + (i >> 1u) * 2u + 1u]),
    (i & 1u) == 0u)`;else if(a===`lines`){let n=e.packedRingOffsetsBase===void 0?`${t}LineOffsets[${t}LineOffsetsOffset`:`${t}FeatureOffsets[${t}FeatureOffsetsOffset + ${e.packedRingOffsetsBase}u`;c=`${n} + r]`,l=`min(${n} + r + 1u], ${i}_VERTEX_COUNT)`}else if(a===`polygons`){o=`${t}FeatureRings[${t}FeatureRingsOffset + f * 2u]`,s=`${t}FeatureRings[${t}FeatureRingsOffset + f * 2u + 1u]`;let n=e.packedRingOffsetsBase===void 0?`${t}RingOffsets[${t}RingOffsetsOffset`:`${t}FeatureRings[${t}FeatureRingsOffset + ${e.packedRingOffsetsBase}u`;c=`${n} + r]`,l=`min(${n} + r + 1u], ${i}_VERTEX_COUNT)`}let d=a===`polygons`?`
fn ${t}Contains(q: vec2f, f: u32) -> bool {
  var inside = false;
  ${r(e,`f`,`a`,`b`,`if ((a.y > q.y) != (b.y > q.y)) {
      let side = orientSide(a, b, q);
      if ((b.y > a.y && side > 0) || (b.y < a.y && side < 0)) { inside = !inside; }
    }`)}
  return inside;
}`:``;return e.partCount!==void 0&&(o=`${t}FeatureOffsets[${t}FeatureOffsetsOffset + f]`,s=`${t}FeatureOffsets[${t}FeatureOffsetsOffset + f + 1u]`),`
${a===`points`?``:`const ${i}_VERTEX_COUNT: u32 = ${e.vertexCount}u;`}
fn ${t}RingStart(f: u32) -> u32 { ${e.partCount===void 0?``:`if (${o} > ${s} || ${s} > ${e.partCount}u) { return 0u; }`} return ${o}; }
fn ${t}RingEnd(f: u32) -> u32 { ${e.partCount===void 0?``:`if (${o} > ${s} || ${s} > ${e.partCount}u) { return 0u; }`} return ${s}; }
fn ${t}RingVertexStart(r: u32) -> u32 { return ${c}; }
fn ${t}RingVertexEnd(r: u32) -> u32 { return ${l}; }
fn ${t}Vertex(i: u32) -> vec2f { return ${u}; }
// (x, y, valid): first finite vertex of the first usable ring; valid is 0 for an empty feature.
fn ${t}First(f: u32) -> vec3f {
  for (var r = ${t}RingStart(f); r < ${t}RingEnd(f); r++) {
    let vs = ${t}RingVertexStart(r);
    let ve = ${t}RingVertexEnd(r);
    if (ve > vs && ve - vs >= ${n(a)}u) {
      let v = ${t}Vertex(vs);
      return vec3f(v, select(0.0, 1.0, isFiniteValue(v.x) && isFiniteValue(v.y)));
    }
  }
  return vec3f(0.0);
}${d}
`}function p(e,t){let n=t.kind===`points`?`NO_SEGMENT`:t.kind===`segments`?`0u`:`rvs + rk`,i=r(t,`rowF`,`ra`,`rb`,`let distanceSq = segmentPairDistanceSq(qa, qb, ra, rb);
      if (nearestDistanceLess(distanceSq, result.distanceSq)) {
        result.distanceSq = distanceSq;
        result.foot = segmentFoot(qa, qb, ra, rb);
        result.queryFoot = segmentFoot(ra, rb, qa, qb);
        result.segment = ${n};
        if (distanceSq == 0.0) { return result; }
      }`),a=r(e,`rowQ`,`qa`,`qb`,i),o=e.kind===`polygons`?`
  if (bitcast<u32>(result.distanceSq) != 0u) {
    ${r(t,`rowF`,`containedStart`,`containedEnd`,`if (qContains(containedStart, rowQ)) { return PairResult(0.0, containedStart, NO_SEGMENT, containedStart); }`)}
  }`:``;return`
struct PairResult { distanceSq: f32, foot: vec2f, segment: u32, queryFoot: vec2f }
fn pairResult(rowQ: u32, rowF: u32) -> PairResult {
  var result = PairResult(NEAREST_NO_DISTANCE, vec2f(0.0), NO_SEGMENT, vec2f(0.0));
  ${a}${t.kind===`polygons`?`
  if (bitcast<u32>(result.distanceSq) != 0u) {
    ${r(e,`rowQ`,`containedStart`,`containedEnd`,`if (rContains(containedStart, rowF)) { return PairResult(0.0, containedStart, NO_SEGMENT, containedStart); }`)}
  }`:``}${o}
  return result;
}
`}var m=`
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
}`;function h(t){let{k:r,capacity:i}=t,a=i>64,o=e(`topk`,`header + 3u`),s=e=>a?o.readRow(e):`listRow[${e}]`,c=e=>a?o.readDistance(e):`listDistance[${e}]`,l=(e,t)=>a?o.writeRow(e,t):`listRow[${e}] = ${t};`,u=(e,t)=>a?o.writeDistance(e,t):`listDistance[${e}] = ${t};`,d=3+2*i;return{declarations:`
const DISPATCH_WORD: u32 = ${t.dispatchWord}u;
const K: u32 = ${r}u;
const CAPACITY: u32 = ${i}u;
const STRIDE: u32 = ${d}u;
const FEATURE_COUNT: u32 = ${t.featureCount}u;
const INTERNAL_NODE_COUNT: u32 = ${t.internalNodeCount}u;
const STACK_SIZE: u32 = ${t.stackSize}u;
${t.scratchChunk?`const CURSOR_WORD: u32 = ${t.scratchChunk.cursorWord}u;`:``}
const BOUND_SLACK: f32 = 1.000001;
${t.greatCircle?m:``}
// ${t.greatCircle?`Central-angle lower bound`:`Squared-distance rank`} from the query box to a node box, or NEAREST_NO_DISTANCE for an empty node.
fn boxLower(node: u32, queryMinimum: vec2f, queryMaximum: vec2f) -> f32 {
  let base = nodeDataOffset + node * 5u;
  let nodeMinimum = vec2f(bitcast<f32>(nodeData[base]), bitcast<f32>(nodeData[base + 1u]));
  let nodeMaximum = vec2f(bitcast<f32>(nodeData[base + 2u]), bitcast<f32>(nodeData[base + 3u]));
  if (nodeMinimum.x > nodeMaximum.x || nodeMinimum.y > nodeMaximum.y) { return NEAREST_NO_DISTANCE; }
  ${t.greatCircle?`return boxLowerAngle(nodeMinimum, nodeMaximum, queryMinimum, queryMaximum);`:`var gap = PlanarDistanceVector(PlanarDistanceValue(0.0, 0), PlanarDistanceValue(0.0, 0));
  if (nodeMinimum.x > queryMaximum.x) { gap.x = getPlanarDistanceDifference(nodeMinimum.x, queryMaximum.x); }
  else if (queryMinimum.x > nodeMaximum.x) { gap.x = getPlanarDistanceDifference(queryMinimum.x, nodeMaximum.x); }
  if (nodeMinimum.y > queryMaximum.y) { gap.y = getPlanarDistanceDifference(nodeMinimum.y, queryMaximum.y); }
  else if (queryMinimum.y > nodeMaximum.y) { gap.y = getPlanarDistanceDifference(queryMinimum.y, nodeMaximum.y); }
  return rankPlanarSquaredDistance(dotPlanarDistanceVectors(gap, gap));`}
}`,body:`let CHUNK_FIRST_ROW = topk[topkOffset + DISPATCH_WORD];
  let CHUNK_QUERY_COUNT = topk[topkOffset + DISPATCH_WORD + 1u];
  ${t.scratchChunk?t.queryOrder?`let position = topk[topkOffset + CURSOR_WORD] + index;
  if (position >= CHUNK_QUERY_COUNT) { return; }
  let localQuery = queryOrder[queryOrderOffset + position] - CHUNK_FIRST_ROW;
  let row = CHUNK_FIRST_ROW + localQuery;
  let header = topkOffset + index * STRIDE;`:`let localQuery = topk[topkOffset + CURSOR_WORD] + index;
  if (localQuery >= CHUNK_QUERY_COUNT) { return; }
  let row = CHUNK_FIRST_ROW + localQuery;
  let header = topkOffset + index * STRIDE;`:t.queryOrder?`if (index >= CHUNK_QUERY_COUNT) { return; }
  let localQuery = queryOrder[queryOrderOffset + index] - CHUNK_FIRST_ROW;
  let row = CHUNK_FIRST_ROW + localQuery;
  let header = topkOffset + row * STRIDE;`:`if (index >= CHUNK_QUERY_COUNT) { return; }
  let localQuery = index;
  let row = CHUNK_FIRST_ROW + index;
  let header = topkOffset + row * STRIDE;`}
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
    if (vertexEnd <= vertexStart || vertexEnd - vertexStart < ${n(t.queryKind)}u) { continue; }
    for (var vertex = vertexStart; vertex < vertexEnd; vertex++) {
      let position = qVertex(vertex);
      if (isFiniteValue(position.x) && isFiniteValue(position.y)) {
        queryMinimum = min(queryMinimum, position);
        queryMaximum = max(queryMaximum, position);
      }
    }
  }
  if (queryMinimum.x > queryMaximum.x) { return; }

  ${a?``:`var listDistance: array<f32, ${i}>;\n  var listRow: array<u32, ${i}>;`}
  var count = 0u;
  var droppedMinimum = NEAREST_NO_DISTANCE;
  var stackNode: array<u32, ${t.stackSize}>;
  var stackLower: array<f32, ${t.stackSize}>;
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
    let featureRow = nodeData[nodeDataOffset + node * 5u + 4u];
    if (featureRow >= FEATURE_COUNT) { continue; }
    ${t.filter?`let queryFilter = nodeDataOffset + ${t.filter.queryBase}u + row * 2u;
    let featureFilter = nodeDataOffset + ${t.filter.featureBase}u + featureRow * 2u;
    ${t.filter.exclusive?`if (nodeData[queryFilter] == nodeData[featureFilter]) { continue; }`:``}
    ${t.filter.onAttribute?`if (nodeData[queryFilter + 1u] != nodeData[featureFilter + 1u]) { continue; }`:``}`:``}
    let candidate = pairResult(localQuery, featureRow).distanceSq;
    if (!nearestDistanceLess(candidate, NEAREST_NO_DISTANCE) || !nearestDistanceAtMost(candidate, bound)) { continue; }
    // Find the slot of (candidate, featureRow) in the ordered list.
    var position = count;
    loop {
      if (position == 0u) { break; }
      let previousDistance = ${c(`position - 1u`)};
      if (nearestDistanceLess(previousDistance, candidate) || (previousDistance == candidate && ${s(`position - 1u`)} < featureRow)) { break; }
      position = position - 1u;
    }
    if (position >= CAPACITY) {
      droppedMinimum = minimumNearestDistance(droppedMinimum, candidate);
      continue;
    }
    var last = count;
    if (count == CAPACITY) {
      droppedMinimum = minimumNearestDistance(droppedMinimum, ${c(`CAPACITY - 1u`)});
      last = CAPACITY - 1u;
    }
    var slot = last;
    loop {
      if (slot <= position) { break; }
      ${u(`slot`,c(`slot - 1u`))}
      ${l(`slot`,s(`slot - 1u`))}
      slot = slot - 1u;
    }
    ${u(`position`,`candidate`)}
    ${l(`position`,`featureRow`)}
    count = min(count + 1u, CAPACITY);
    // Beyond k, keep only rows tied with the k-th distance.
    loop {
      if (count > K && nearestDistanceLess(${c(`K - 1u`)}, ${c(`count - 1u`)})) { count = count - 1u; } else { break; }
    }
    if (count >= K) { bound = minimumNearestDistance(bound, ${c(`K - 1u`)}); }
  }
  var overflowFlag = 0u;
  if (${t.ties===`all`?`true`:`CAPACITY > K`} && count >= K && nearestDistanceAtMost(droppedMinimum, ${c(`K - 1u`)})) { overflowFlag = 1u; }
  ${a?``:`for (var slot = 0u; slot < count; slot++) {
    topk[header + 3u + slot * 2u] = listRow[slot];
    topk[header + 4u + slot * 2u] = bitcast<u32>(listDistance[slot]);
  }`}
  topk[header] = count;
  topk[header + 1u] = overflowFlag;`}}export{p as a,s as c,d as i,u as n,f as o,c as r,h as s,m as t};