import{o as e}from"./wgsl-kernel-nodes-DagJklH2.js";import{t}from"./exact-orientation-wgsl-B_jI-HEc.js";import{u as n}from"./gpu-contributor-utils-BIwLQON5.js";var r={proper:1,touch:2,collinearTouch:3,overlap:4,uncertain:5},i=4294967295;function a(e){return n(e,32)}var o=`
const SEGMENT_NONE: u32 = ${i}u;
const KIND_NONE: u32 = 0u;
const KIND_PROPER: u32 = ${r.proper}u;
const KIND_TOUCH: u32 = ${r.touch}u;
const KIND_COLLINEAR_TOUCH: u32 = ${r.collinearTouch}u;
const KIND_OVERLAP: u32 = ${r.overlap}u;
const KIND_UNCERTAIN: u32 = ${r.uncertain}u;
${t}

struct SegmentHit {
  kind: u32,
  point: vec2f,
  endPoint: vec2f,
  // KIND_TOUCH only: which orientation vanished (0: c on a-b, 1: d on a-b, 2: a on c-d, 3: b on c-d).
  selector: u32,
}

fn axisValue(point: vec2f, useX: bool) -> f32 { return select(point.y, point.x, useX); }

// Picks the point among the four endpoints whose axis coordinate equals value.
fn pointAtAxis(a: vec2f, b: vec2f, c: vec2f, d: vec2f, useX: bool, value: f32) -> vec2f {
  if (axisValue(a, useX) == value) { return a; }
  if (axisValue(b, useX) == value) { return b; }
  if (axisValue(c, useX) == value) { return c; }
  return d;
}

// Classifies segments a-b and c-d, both non-degenerate. withGeometry also fills the points.
fn classifySegments(a: vec2f, b: vec2f, c: vec2f, d: vec2f, withGeometry: bool) -> SegmentHit {
  var hit = SegmentHit(KIND_NONE, a, a, 0u);
  if (max(a.x, b.x) < min(c.x, d.x) || max(c.x, d.x) < min(a.x, b.x) ||
      max(a.y, b.y) < min(c.y, d.y) || max(c.y, d.y) < min(a.y, b.y)) {
    return hit;
  }
  let o1 = orientSign(a, b, c);
  let o2 = orientSign(a, b, d);
  if (o1 == 2 || o2 == 2) { hit.kind = KIND_UNCERTAIN; return hit; }
  if (o1 != 0 && o1 == o2) { return hit; }
  let o3 = orientSign(c, d, a);
  let o4 = orientSign(c, d, b);
  if (o3 == 2 || o4 == 2) { hit.kind = KIND_UNCERTAIN; return hit; }
  if (o3 != 0 && o3 == o4) { return hit; }
  if (o1 == 0 && o2 == 0) {
    // All four orientations vanish: collinear. Compare along the axis the line is not parallel to.
    let useX = a.x != b.x;
    let low = max(min(axisValue(a, useX), axisValue(b, useX)), min(axisValue(c, useX), axisValue(d, useX)));
    let high = min(max(axisValue(a, useX), axisValue(b, useX)), max(axisValue(c, useX), axisValue(d, useX)));
    if (low > high) { return hit; }
    hit.kind = select(KIND_OVERLAP, KIND_COLLINEAR_TOUCH, low == high);
    hit.point = pointAtAxis(a, b, c, d, useX, low);
    hit.endPoint = pointAtAxis(a, b, c, d, useX, high);
    return hit;
  }
  if (o1 != 0 && o2 != 0 && o3 != 0 && o4 != 0) {
    hit.kind = KIND_PROPER;
    if (withGeometry) {
      let r = b - a;
      let s = d - c;
      let denominator = r.x * s.y - r.y * s.x;
      let t = clamp(((c.x - a.x) * s.y - (c.y - a.y) * s.x) / denominator, 0.0, 1.0);
      let crossing = a + r * t;
      hit.point = clamp(crossing, max(min(a, b), min(c, d)), min(max(a, b), max(c, d)));
      hit.endPoint = hit.point;
    }
    return hit;
  }
  hit.kind = KIND_TOUCH;
  if (o1 == 0) { hit.selector = 0u; hit.point = c; } else if (o2 == 0) { hit.selector = 1u; hit.point = d; } else if (o3 == 0) { hit.selector = 2u; hit.point = a; } else { hit.selector = 3u; hit.point = b; }
  hit.endPoint = hit.point;
  return hit;
}

// Rebuilds the points of a classified pair from its kind and touch selector without evaluating any
// orientation predicate, so a pair is classified once and its geometry is derived afterwards.
fn getSegmentHitGeometry(kind: u32, selector: u32, a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> SegmentHit {
  var hit = SegmentHit(kind, a, a, selector);
  if (kind == KIND_PROPER) {
    let r = b - a;
    let s = d - c;
    let denominator = r.x * s.y - r.y * s.x;
    let t = clamp(((c.x - a.x) * s.y - (c.y - a.y) * s.x) / denominator, 0.0, 1.0);
    let crossing = a + r * t;
    hit.point = clamp(crossing, max(min(a, b), min(c, d)), min(max(a, b), max(c, d)));
    hit.endPoint = hit.point;
  } else if (kind == KIND_TOUCH) {
    let touchPoints = array<vec2f, 4>(c, d, a, b);
    hit.point = touchPoints[selector];
    hit.endPoint = hit.point;
  } else if (kind == KIND_OVERLAP || kind == KIND_COLLINEAR_TOUCH) {
    let useX = a.x != b.x;
    let low = max(min(axisValue(a, useX), axisValue(b, useX)), min(axisValue(c, useX), axisValue(d, useX)));
    let high = min(max(axisValue(a, useX), axisValue(b, useX)), max(axisValue(c, useX), axisValue(d, useX)));
    hit.point = pointAtAxis(a, b, c, d, useX, low);
    hit.endPoint = pointAtAxis(a, b, c, d, useX, high);
  }
  return hit;
}

`;function s(e){let t=`${e}Table`,n=e=>`${t}Offset + ${e} * 8u`;return`
fn ${e}Ring(row: u32) -> u32 { return ${t}[${n(`row`)} + 4u]; }
fn ${e}Valid(row: u32) -> bool { return ${e}Ring(row) != SEGMENT_NONE; }
fn ${e}Successor(row: u32) -> u32 { return ${t}[${n(`row`)} + 5u]; }
fn ${e}Feature(row: u32) -> u32 { return ${t}[${n(`row`)} + 6u]; }
fn ${e}Start(row: u32) -> vec2f {
  return vec2f(bitcast<f32>(${t}[${n(`row`)}]), bitcast<f32>(${t}[${n(`row`)} + 1u]));
}
fn ${e}End(row: u32) -> vec2f {
  return vec2f(bitcast<f32>(${t}[${n(`row`)} + 2u]), bitcast<f32>(${t}[${n(`row`)} + 3u]));
}`}var c=1024;function l(e){return e.positions.length}function u(t,n){let{geometry:r}=n,i=r.kind===`polygons`,a=[{name:`positions`,view:r.positions,type:`f32`,access:`read`}],s;i?(a.push({name:`ringOffsets`,view:r.ringOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:r.polygonOffsets,type:`u32`,access:`read`},{name:`featureOffsets`,view:r.featureOffsets,type:`u32`,access:`read`}),s=`
const RING_COUNT: u32 = ${r.ringOffsets.length-1}u;
const POLYGON_COUNT: u32 = ${r.polygonOffsets.length-1}u;
const FEATURE_COUNT: u32 = ${r.featureOffsets.length-1}u;
const MINIMUM_RING_VERTICES: u32 = 3u;
fn lastAtMost(offsetsKind: u32, count: u32, value: u32) -> u32 {
  if (count == 0u) { return SEGMENT_NONE; }
  var low = 0u;
  var high = count;
  if (readOffset(offsetsKind, 0u) > value) { return SEGMENT_NONE; }
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (readOffset(offsetsKind, middle) <= value) { low = middle; } else { high = middle; }
  }
  return low;
}
fn readOffset(offsetsKind: u32, row: u32) -> u32 {
  if (offsetsKind == 0u) { return ringOffsets[ringOffsetsOffset + row]; }
  if (offsetsKind == 1u) { return polygonOffsets[polygonOffsetsOffset + row]; }
  return featureOffsets[featureOffsetsOffset + row];
}
fn ringStartOf(ring: u32) -> u32 { return ringOffsets[ringOffsetsOffset + ring]; }
fn ringEndOf(ring: u32) -> u32 { return min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT); }
fn featureOfRing(ring: u32) -> u32 {
  if (ring >= polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT]) { return SEGMENT_NONE; }
  let polygon = lastAtMost(1u, POLYGON_COUNT, ring);
  if (polygon == SEGMENT_NONE) { return SEGMENT_NONE; }
  if (polygon >= featureOffsets[featureOffsetsOffset + FEATURE_COUNT]) { return SEGMENT_NONE; }
  return lastAtMost(2u, FEATURE_COUNT, polygon);
}
fn ringOfVertex(vertex: u32) -> u32 {
  let ring = lastAtMost(0u, RING_COUNT, vertex);
  if (ring == SEGMENT_NONE || vertex >= ringEndOf(ring)) { return SEGMENT_NONE; }
  return ring;
}`):(a.push({name:`lineOffsets`,view:r.lineOffsets,type:`u32`,access:`read`}),s=`
const RING_COUNT: u32 = ${r.lineOffsets.length-1}u;
const MINIMUM_RING_VERTICES: u32 = 2u;
fn readOffset(offsetsKind: u32, row: u32) -> u32 { return lineOffsets[lineOffsetsOffset + row]; }
fn lastAtMost(offsetsKind: u32, count: u32, value: u32) -> u32 {
  if (count == 0u) { return SEGMENT_NONE; }
  var low = 0u;
  var high = count;
  if (readOffset(offsetsKind, 0u) > value) { return SEGMENT_NONE; }
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (readOffset(offsetsKind, middle) <= value) { low = middle; } else { high = middle; }
  }
  return low;
}
fn ringStartOf(ring: u32) -> u32 { return lineOffsets[lineOffsetsOffset + ring]; }
fn ringEndOf(ring: u32) -> u32 { return min(lineOffsets[lineOffsetsOffset + ring + 1u], VERTEX_COUNT); }
fn featureOfRing(ring: u32) -> u32 { return ring; }
fn ringOfVertex(vertex: u32) -> u32 {
  let ring = lastAtMost(0u, RING_COUNT, vertex);
  if (ring == SEGMENT_NONE || vertex >= ringEndOf(ring)) { return SEGMENT_NONE; }
  return ring;
}`),a.push({name:`table`,view:n.table,type:`u32`,access:`read_write`}),n.minima&&n.maxima&&a.push({name:`segmentMinima`,view:n.minima,type:`f32`,access:`read_write`},{name:`segmentMaxima`,view:n.maxima,type:`f32`,access:`read_write`});let u=!!(n.minima&&n.maxima);return e(t,{id:n.id,operation:n.operation,variant:`segment-table-${r.kind}`,bindings:a,invocationCount:l(r),declarations:`${o}
const VERTEX_COUNT: u32 = ${r.positions.length}u;
const CYCLIC: bool = ${i};
const MAXIMUM_SUCCESSOR_SCAN: u32 = ${c}u;
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
${s}
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
fn finiteValue(value: f32) -> bool { return isFiniteBits(value); }
fn finitePoint(point: vec2f) -> bool { return finiteValue(point.x) && finiteValue(point.y); }
// The vertex that ends the segment starting at vertex, or SEGMENT_NONE when there is none.
fn nextVertex(vertex: u32, ringStart: u32, ringEnd: u32, closedLine: bool) -> u32 {
  if (vertex + 1u < ringEnd) { return vertex + 1u; }
  if (CYCLIC || closedLine) { return ringStart; }
  return SEGMENT_NONE;
}
fn segmentIsUsable(vertex: u32, ringStart: u32, ringEnd: u32, closedLine: bool) -> bool {
  let next = nextVertex(vertex, ringStart, ringEnd, closedLine);
  if (next == SEGMENT_NONE) { return false; }
  // A closed linestring's last vertex repeats its first and owns no segment.
  if (!CYCLIC && vertex + 1u == ringEnd) { return false; }
  let start = vertexAt(vertex);
  let end = vertexAt(next);
  return finitePoint(start) && finitePoint(end) && (start.x != end.x || start.y != end.y);
}`,body:`var row0 = 0u; var row1 = 0u; var row2 = 0u; var row3 = 0u;
  var rowRing = SEGMENT_NONE;
  var rowSuccessor = SEGMENT_NONE;
  var rowFeature = SEGMENT_NONE;
  var minimum = vec2f(FLOAT32_MAXIMUM);
  var maximum = vec2f(-FLOAT32_MAXIMUM);
  let ring = ringOfVertex(index);
  if (ring != SEGMENT_NONE) {
    let ringStart = ringStartOf(ring);
    let ringEnd = ringEndOf(ring);
    let feature = featureOfRing(ring);
    var closedLine = false;
    if (!CYCLIC && ringEnd - ringStart >= 3u) {
      let first = vertexAt(ringStart);
      let last = vertexAt(ringEnd - 1u);
      closedLine = first.x == last.x && first.y == last.y;
    }
    if (ringEnd - ringStart >= MINIMUM_RING_VERTICES && feature != SEGMENT_NONE &&
        segmentIsUsable(index, ringStart, ringEnd, closedLine)) {
      let start = vertexAt(index);
      let end = vertexAt(nextVertex(index, ringStart, ringEnd, closedLine));
      row0 = bitcast<u32>(start.x); row1 = bitcast<u32>(start.y);
      row2 = bitcast<u32>(end.x); row3 = bitcast<u32>(end.y);
      rowRing = ring;
      rowFeature = feature;
      minimum = min(start, end);
      maximum = max(start, end);
      var candidate = index;
      for (var step = 0u; step < min(ringEnd - ringStart, MAXIMUM_SUCCESSOR_SCAN); step++) {
        let following = nextVertex(candidate, ringStart, ringEnd, closedLine);
        if (following == SEGMENT_NONE) { break; }
        candidate = following;
        if (segmentIsUsable(candidate, ringStart, ringEnd, closedLine)) {
          rowSuccessor = candidate;
          break;
        }
      }
    }
  }
  let base = tableOffset + index * 8u;
  table[base] = row0; table[base + 1u] = row1; table[base + 2u] = row2; table[base + 3u] = row3;
  table[base + 4u] = rowRing; table[base + 5u] = rowSuccessor; table[base + 6u] = rowFeature;
  table[base + 7u] = 0u;
  ${u?`segmentMinima[segmentMinimaOffset + index * 2u] = minimum.x;
  segmentMinima[segmentMinimaOffset + index * 2u + 1u] = minimum.y;
  segmentMaxima[segmentMaximaOffset + index * 2u] = maximum.x;
  segmentMaxima[segmentMaximaOffset + index * 2u + 1u] = maximum.y;`:``}`})}export{s as a,a as i,l as n,r as o,o as r,i as s,u as t};