import{t as e}from"./common-CshkX-d_.js";import{r as t}from"./kernel-C3tfQOn4.js";import{i as n,o as r,t as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as a}from"./gpu-scan-B4NNTfF9.js";import{t as o}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{E as s,S as c,f as l,u,v as d,y as f}from"./gpu-contributor-utils-BIwLQON5.js";import{t as ee}from"./exact-orientation-wgsl-B_jI-HEc.js";import{i as p,n as m,r as te,t as ne}from"./polygon-edge-slabs-CRnrFvP2.js";import{u as h}from"./wgsl-common-DfyXiYB0.js";import{t as g}from"./exact-geometry-wgsl-y7TUnqOh.js";import{v as _}from"./spatial-context-DwZEd4EZ.js";import{n as v,t as y}from"./construction-snapshot-DdkX9Pc0.js";import{a as b,i as x}from"./graph-chunk-dispatch-zBt8k2-o.js";var S={routed:0,clean:1,pieces:2,jumpPairs:3,maximumClean:4,jumpActive:5},C={nonFinite:1,shell:2,orientation:4,ray:8,diagonal:16,piece:32,stack:64},w={start:0,end:1,split:2,merge:3,regularLeft:4,regularRight:5};function T(e){return`
const NONE: u32 = 0xffffffffu;
const OVERFLOW_BIT: u32 = 0x80000000u;
const VERTEX_COUNT: u32 = ${e.vertexCount}u;
const POLYGON_COUNT: u32 = ${e.polygonCount}u;
const PREPARED_LAYOUT: bool = ${!!e.preparedLayout};
const RING_COUNT: u32 = ${e.ringCount}u;
const CAPACITY: u32 = ${e.capacity}u;
const INDEX_LENGTH: u32 = ${e.indexLength}u;
const MINIMUM_VERTICES: u32 = ${e.minimumVertices}u;
const COUNT_ROUTED: u32 = ${S.routed}u;
const COUNT_CLEAN: u32 = ${S.clean}u;
const COUNT_PIECES: u32 = ${S.pieces}u;
const COUNT_JUMP_PAIRS: u32 = ${S.jumpPairs}u;
const COUNT_MAXIMUM_CLEAN: u32 = ${S.maximumClean}u;
const COUNT_JUMP_ACTIVE: u32 = ${S.jumpActive}u;
const COUNT_WORDS: u32 = 8u;
const STATS_STRIDE: u32 = 8u;
const STATS_FAIL: u32 = 0u;
const STATS_CLEAN: u32 = 1u;
const STATS_HOLES: u32 = 2u;
const STATS_TRIANGLES: u32 = 3u;
const STATS_WRITTEN: u32 = 4u;
const FAIL_NON_FINITE: u32 = ${C.nonFinite}u;
const FAIL_SHELL: u32 = ${C.shell}u;
const FAIL_ORIENTATION: u32 = ${C.orientation}u;
const FAIL_RAY: u32 = ${C.ray}u;
const FAIL_DIAGONAL: u32 = ${C.diagonal}u;
const FAIL_PIECE: u32 = ${C.piece}u;
const FAIL_STACK: u32 = ${C.stack}u;
const TYPE_START: u32 = ${w.start}u;
const TYPE_END: u32 = ${w.end}u;
const TYPE_SPLIT: u32 = ${w.split}u;
const TYPE_MERGE: u32 = ${w.merge}u;
const TYPE_REGULAR_LEFT: u32 = ${w.regularLeft}u;
const TYPE_REGULAR_RIGHT: u32 = ${w.regularRight}u;
const A_POLYGON: u32 = 0u;
const A_RING: u32 = 1u;
const A_ROW: u32 = 2u;
const A_TYPE: u32 = 3u;
const B_RANK: u32 = 0u;
const B_NEXT: u32 = 1u;
const B_PREVIOUS: u32 = 2u;
const B_LEFT: u32 = 3u;
const PIECE_STRIDE: u32 = 8u;
`}var E=`
fn polygonOf(vertex: u32) -> u32 { return vertexA[vertexAOffset + 4u * vertex + A_POLYGON]; }
fn ringOf(vertex: u32) -> u32 { return vertexA[vertexAOffset + 4u * vertex + A_RING]; }
fn rowOf(vertex: u32) -> u32 { return vertexA[vertexAOffset + 4u * vertex + A_ROW]; }
fn typeOf(vertex: u32) -> u32 { return vertexA[vertexAOffset + 4u * vertex + A_TYPE] & 255u; }
fn isFlipped(vertex: u32) -> bool { return ((vertexA[vertexAOffset + 4u * vertex + A_TYPE] >> 8u) & 1u) == 1u; }
`,D=`
fn rankOf(vertex: u32) -> u32 { return vertexB[vertexBOffset + 4u * vertex + B_RANK]; }
fn nextOf(vertex: u32) -> u32 { return vertexB[vertexBOffset + 4u * vertex + B_NEXT]; }
fn previousOf(vertex: u32) -> u32 { return vertexB[vertexBOffset + 4u * vertex + B_PREVIOUS]; }
`,O=`
fn positionOf(vertex: u32) -> vec2f {
  return vec2f(vertexPosition[vertexPositionOffset + 2u * vertex], vertexPosition[vertexPositionOffset + 2u * vertex + 1u]);
}
`,k=`
var<private> orientUncertain: bool;
fn orientOf(a: vec2f, b: vec2f, c: vec2f) -> i32 {
  let result = orientSign(a, b, c);
  if (result == 2) {
    orientUncertain = true;
    return 0;
  }
  return result;
}
`,A=`
fn failPolygon(failedPolygon: u32, failureBit: u32) {
  atomicOr(&polygonStats[polygonStatsOffset + STATS_STRIDE * failedPolygon + STATS_FAIL], failureBit);
}
`,j=`
fn tailOf(halfEdge: u32) -> u32 {
  if (halfEdge < CAPACITY) { return halfEdge; }
  return (halfEdge - CAPACITY) / 4u;
}
`,M=`
fn orderedKey(x: f32) -> u32 {
  var bits = bitcast<u32>(x);
  if (x == 0.0) { bits = 0u; }
  return bits ^ select(0x80000000u, 0xffffffffu, (bits & 0x80000000u) != 0u);
}
`,N=`
fn baseSlotOf(polygon: u32) -> i32 {
  if (PREPARED_LAYOUT) {
    return 3 * i32(polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT + 1u + polygon]);
  }
  let ringPrefix = i32(polygonOffsets[polygonOffsetsOffset + polygon]);
  let vertexPrefix = i32(ringOffsets[ringOffsetsOffset + u32(ringPrefix)]);
  return 3 * (vertexPrefix + 2 * ringPrefix - 4 * i32(polygon));
}
`,P=`
fn polygonFirstRow(polygon: u32) -> u32 {
  return min(ringOffsets[ringOffsetsOffset + polygonOffsets[polygonOffsetsOffset + polygon]], VERTEX_COUNT);
}
fn polygonEndRow(polygon: u32) -> u32 {
  return min(ringOffsets[ringOffsetsOffset + polygonOffsets[polygonOffsetsOffset + polygon + 1u]], VERTEX_COUNT);
}
`;function re(e,t=!1){return`
const MONOTONE_MINIMUM_VERTICES: u32 = ${e}u;
fn sizeRouteOf(polygon: u32) -> u32 {
  ${t?`let count = polygonOffsets[polygonOffsetsOffset + 2u * (POLYGON_COUNT + 1u) + polygon];
  return select(0u, 1u, count >= MONOTONE_MINIMUM_VERTICES);`:`
  let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
  let endRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
  if (endRing <= firstRing) { return 0u; }
  let first = min(ringOffsets[ringOffsetsOffset + firstRing], VERTEX_COUNT);
  let end = min(ringOffsets[ringOffsetsOffset + endRing], VERTEX_COUNT);
  return select(0u, 1u, end >= first + MONOTONE_MINIMUM_VERTICES);`}
}
`}function ie(e){return{declarations:`${T(e)}${P}`,body:`var count = 0u;
  if (index < POLYGON_COUNT) {
    if (routing[routingOffset + index] != 0u) {
      count = polygonEndRow(index) - min(polygonFirstRow(index), polygonEndRow(index));
    }
    for (var word = 0u; word < STATS_STRIDE; word++) {
      polygonStats[polygonStatsOffset + STATS_STRIDE * index + word] = 0u;
    }
  }
  routedCounts[routedCountsOffset + index] = count;
  if (index == 0u) {
    for (var word = 0u; word < COUNT_WORDS; word++) { counts[countsOffset + word] = 0u; }
  }`}}function ae(e){return{declarations:T(e),body:`let count = routedCounts[routedCountsOffset + index];
  if (count > 0u && routedOffsets[routedOffsetsOffset + index] + count > CAPACITY) {
    routing[routingOffset + index] = 3u;
    routedCounts[routedCountsOffset + index] = 0u;
  }
  if (index == 0u) {
    counts[countsOffset + COUNT_ROUTED] = routedOffsets[routedOffsetsOffset + POLYGON_COUNT];
  }`}}function oe(e){return{declarations:`${T(e)}${A}
fn vertexAt(row: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
${h}`,body:`let j = index;
  keepFlags[keepFlagsOffset + j] = 0u;
  // The largest polygon whose routed range starts at or before j.
  var low = 0u;
  var high = POLYGON_COUNT;
  while (low + 1u < high) {
    let middle = (low + high) / 2u;
    if (routedOffsets[routedOffsetsOffset + middle] <= j) { low = middle; } else { high = middle; }
  }
  let polygon = low;
  let polygonBegin = routedOffsets[routedOffsetsOffset + polygon];
  let polygonEnd = routedOffsets[routedOffsetsOffset + polygon + 1u];
  let info = jInfoOffset + 4u * j;
  jInfo[info + 1u] = NONE;
  if (j >= polygonEnd || routing[routingOffset + polygon] == 0u) { return; }
  let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
  let endRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
  let row = ringOffsets[ringOffsetsOffset + firstRing] + (j - polygonBegin);
  if (row >= VERTEX_COUNT) { return; }
  // The ring containing the row: the largest ring of the polygon that starts at or before it.
  var ringLow = firstRing;
  var ringHigh = endRing;
  while (ringLow + 1u < ringHigh) {
    let middle = (ringLow + ringHigh) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] <= row) { ringLow = middle; } else { ringHigh = middle; }
  }
  let ring = ringLow;
  jInfo[info] = row;
  jInfo[info + 2u] = ring;
  jInfo[info + 3u] = firstRing;
  if (routing[routingOffset + polygon] == 3u) {
    jInfo[info + 1u] = polygon | OVERFLOW_BIT;
    return;
  }
  jInfo[info + 1u] = polygon;
  ${e.preparedLayout?`if (ringOffsets[ringOffsetsOffset + RING_COUNT + 1u + ring] == 0u) { return; }`:``}
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
  let point = vertexAt(row);
  if (!isFiniteValue(point.x) || !isFiniteValue(point.y)) {
    failPolygon(polygon, FAIL_NON_FINITE);
    return;
  }
  let previousRow = select(row - 1u, ringEnd - 1u, row == ringBegin);
  let previous = vertexAt(previousRow);
  if (point.x != previous.x || point.y != previous.y) { keepFlags[keepFlagsOffset + j] = 1u; }`}}function se(e){return{declarations:`${T(e)}${A}`,body:`let info = jInfoOffset + 4u * index;
  let polygon = jInfo[info + 1u];
  if (polygon >= OVERFLOW_BIT) { return; }
  let ring = jInfo[info + 2u];
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  if (jInfo[info] != ringBegin) { return; }
  let ringEnd = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
  let begin = keepOffsets[keepOffsetsOffset + index];
  let count = keepOffsets[keepOffsetsOffset + index + (ringEnd - ringBegin)] - begin;
  let shellRing = jInfo[info + 3u];
  let shellBegin = ringOffsets[ringOffsetsOffset + shellRing];
  let shellEnd = min(ringOffsets[ringOffsetsOffset + shellRing + 1u], VERTEX_COUNT);
  let polygonBegin = routedOffsets[routedOffsetsOffset + polygon];
  let shellCount = keepOffsets[keepOffsetsOffset + polygonBegin + (shellEnd - shellBegin)] -
    keepOffsets[keepOffsetsOffset + polygonBegin];
  let alive = count >= 3u && shellCount >= 3u;
  ringInfo[ringInfoOffset + 4u * ring] = begin;
  ringInfo[ringInfoOffset + 4u * ring + 1u] = count;
  ringInfo[ringInfoOffset + 4u * ring + 2u] = select(0u, 1u, alive);
  ringInfo[ringInfoOffset + 4u * ring + 3u] = 0u;
  atomicStore(&ringTop[ringTopOffset + ring], NONE);
  let statsBase = polygonStatsOffset + STATS_STRIDE * polygon;
  if (alive) {
    let before = atomicAdd(&polygonStats[statsBase + STATS_CLEAN], count);
    atomicMax(&counts[countsOffset + COUNT_MAXIMUM_CLEAN], before + count);
    if (ring != shellRing) { atomicAdd(&polygonStats[statsBase + STATS_HOLES], 1u); }
  }
  if (ring == shellRing && shellCount < 3u) { failPolygon(polygon, FAIL_SHELL); }`}}function ce(e){return{declarations:T(e),body:`let routed = min(counts[countsOffset + COUNT_ROUTED], VERTEX_COUNT);
  counts[countsOffset + COUNT_CLEAN] = min(keepOffsets[keepOffsetsOffset + routed], CAPACITY);
  // A cycle has at most 3 n half-edges (n ring edges and up to 2 n diagonal half-edges); a
  // pointer window of 2^r half-edges closes it once 2^r reaches that bound.
  let bound = 3u * counts[countsOffset + COUNT_MAXIMUM_CLEAN] + 3u;
  var rounds = 0u;
  var reach = 1u;
  while (reach < bound) {
    reach = reach * 2u;
    rounds += 1u;
  }
  counts[countsOffset + COUNT_JUMP_PAIRS] = (rounds + 1u) / 2u;`}}function le(e){return{declarations:T(e),body:`let info = jInfoOffset + 4u * index;
  let polygon = jInfo[info + 1u];
  if (polygon >= OVERFLOW_BIT || keepFlags[keepFlagsOffset + index] == 0u) { return; }
  let row = jInfo[info];
  let ring = jInfo[info + 2u];
  let vertex = keepOffsets[keepOffsetsOffset + index];
  if (vertex >= CAPACITY) { return; }
  let alive = (ringInfo[ringInfoOffset + 4u * ring + 2u] & 1u) != 0u;
  let record = vertexAOffset + 4u * vertex;
  vertexA[record + A_POLYGON] = select(NONE, polygon, alive);
  vertexA[record + A_RING] = ring;
  vertexA[record + A_ROW] = row;
  vertexA[record + A_TYPE] = 0u;
  vertexPosition[vertexPositionOffset + 2u * vertex] = positions[positionsOffset + 2u * row];
  vertexPosition[vertexPositionOffset + 2u * vertex + 1u] = positions[positionsOffset + 2u * row + 1u];
  rowToClean[rowToCleanOffset + row] = vertex;`}}function ue(e){return{declarations:`${T(e)}${M}`,body:`let vertex = index;
  var high = NONE;
  var low = NONE;
  if (vertex < counts[countsOffset + COUNT_CLEAN]) {
    if (vertexA[vertexAOffset + 4u * vertex + A_POLYGON] != NONE) {
      let x = vertexPosition[vertexPositionOffset + 2u * vertex];
      let y = vertexPosition[vertexPositionOffset + 2u * vertex + 1u];
      high = min(~orderedKey(y), 0xfffffffeu);
      low = min(orderedKey(x), 0xfffffffeu);
    }
  } else {
    vertexA[vertexAOffset + 4u * vertex + A_POLYGON] = NONE;
  }
  sortKeyHigh[sortKeyHighOffset + vertex] = high;
  sortKeyLow[sortKeyLowOffset + vertex] = low;
  identity[identityOffset + vertex] = vertex;`}}function de(e){return{declarations:`${T(e)}${E}`,body:`let vertex = sortedIndices[sortedIndicesOffset + index];
  if (vertex >= CAPACITY) { return; }
  if (polygonOf(vertex) == NONE) { return; }
  vertexB[vertexBOffset + 4u * vertex + B_RANK] = index;
  atomicMin(&ringTop[ringTopOffset + ringOf(vertex)], index);`}}function fe(e){return{declarations:`${T(e)}${E}${D}${O}${A}${k}`,body:`let vertex = index;
  let polygon = polygonOf(vertex);
  if (polygon == NONE) { return; }
  let ring = ringOf(vertex);
  if (rankOf(vertex) != atomicLoad(&ringTop[ringTopOffset + ring])) { return; }
  let info = ringInfoOffset + 4u * ring;
  let begin = ringInfo[info];
  let count = ringInfo[info + 1u];
  let local = vertex - begin;
  let nextVertex = begin + (local + 1u) % count;
  let previousVertex = begin + (local + count - 1u) % count;
  let orientation = orientOf(positionOf(previousVertex), positionOf(vertex), positionOf(nextVertex));
  var flip = false;
  if (orientation == 0 || orientUncertain) {
    failPolygon(polygon, FAIL_ORIENTATION);
  } else if (ring == polygonOffsets[polygonOffsetsOffset + polygon]) {
    flip = orientation < 0;
  } else {
    flip = orientation > 0;
  }
  if (flip) { ringInfo[info + 2u] = ringInfo[info + 2u] | 2u; }`}}function pe(e){return{declarations:`${T(e)}${E}${D}${O}${A}${k}`,body:`let vertex = index;
  let polygon = polygonOf(vertex);
  if (polygon == NONE) { return; }
  let info = ringInfoOffset + 4u * ringOf(vertex);
  let begin = ringInfo[info];
  let count = ringInfo[info + 1u];
  let flip = (ringInfo[info + 2u] & 2u) != 0u;
  let local = vertex - begin;
  let rawNext = begin + (local + 1u) % count;
  let rawPrevious = begin + (local + count - 1u) % count;
  let nextVertex = select(rawNext, rawPrevious, flip);
  let previousVertex = select(rawPrevious, rawNext, flip);
  let record = vertexBOffset + 4u * vertex;
  vertexB[record + B_NEXT] = nextVertex;
  vertexB[record + B_PREVIOUS] = previousVertex;
  vertexB[record + B_LEFT] = NONE;
  let rank = rankOf(vertex);
  let nextBelow = rankOf(nextVertex) > rank;
  let previousBelow = rankOf(previousVertex) > rank;
  var kind = TYPE_REGULAR_LEFT;
  if (previousBelow == nextBelow) {
    let orientation = orientOf(positionOf(previousVertex), positionOf(vertex), positionOf(nextVertex));
    if (orientation == 0 || orientUncertain) {
      failPolygon(polygon, FAIL_ORIENTATION);
    } else if (previousBelow) {
      kind = select(TYPE_SPLIT, TYPE_START, orientation > 0);
    } else {
      kind = select(TYPE_MERGE, TYPE_END, orientation > 0);
    }
  } else {
    kind = select(TYPE_REGULAR_LEFT, TYPE_REGULAR_RIGHT, previousBelow);
  }
  vertexA[vertexAOffset + 4u * vertex + A_TYPE] = kind | select(0u, 256u, flip);
  nextHalfEdge[nextHalfEdgeOffset + vertex] = vertex;
  for (var slot = 0u; slot < 4u; slot++) {
    vertexDiagonal[vertexDiagonalOffset + 4u * vertex + slot] = NONE;
    nextHalfEdge[nextHalfEdgeOffset + CAPACITY + 4u * vertex + slot] = CAPACITY + 4u * vertex + slot;
  }`}}var me=`
// Positive when edge \`second\` lies to the right of edge \`first\` at the common sweep position.
fn isRightOf(second: u32, first: u32) -> i32 {
  let upper1 = first;
  let lower1 = nextOf(first);
  let upper2 = second;
  let lower2 = nextOf(second);
  var a = upper1;
  var b = lower1;
  var c = upper2;
  var d = lower2;
  var multiplier = 1;
  if (upper2 == upper1) {
    c = lower2;
  } else if (rankOf(upper2) < rankOf(upper1)) {
    a = upper2;
    b = lower2;
    c = upper1;
    d = lower1;
    multiplier = -1;
  }
  var side = multiplier * orientOf(positionOf(a), positionOf(b), positionOf(c));
  if (side == 0 && upper2 != upper1) {
    side = multiplier * orientOf(positionOf(a), positionOf(b), positionOf(d));
  }
  if (side != 0) { return side; }
  // Touching edges (invalid input): a deterministic order by name.
  return select(-1, 1, second > first);
}
`;function he(e){return`let vertex = index;
  let polygon = polygonOf(vertex);
  if (polygon == NONE) { return; }
  let kind = typeOf(vertex);
  if (kind != TYPE_SPLIT && kind != TYPE_MERGE && kind != TYPE_REGULAR_RIGHT) { return; }
  let point = positionOf(vertex);
  let queryRank = rankOf(vertex);
  var best = NONE;
  ${e(`let cleanEnd = rowToClean[rowToCleanOffset + edgeEnd];
        if (cleanEnd >= CAPACITY) { continue; }
        if (polygonOf(cleanEnd) != polygon || rowOf(cleanEnd) != edgeEnd) { continue; }
        let flipped = isFlipped(cleanEnd);
        let rawPrevious = select(previousOf(cleanEnd), nextOf(cleanEnd), flipped);
        let upperVertex = select(rawPrevious, cleanEnd, flipped);
        let lowerVertex = select(cleanEnd, rawPrevious, flipped);
        let upperRank = rankOf(upperVertex);
        if (!(upperRank < queryRank && queryRank < rankOf(lowerVertex))) { continue; }
        if (orientOf(positionOf(upperVertex), positionOf(lowerVertex), point) <= 0) { continue; }
        var closer = best == NONE;
        if (!closer) { closer = isRightOf(upperVertex, best) > 0; }
        if (closer) { best = upperVertex; }`)}
  if (orientUncertain) { best = NONE; }
  vertexB[vertexBOffset + 4u * vertex + B_LEFT] = best;`}function ge(e){return{declarations:`${T(e)}${E}${D}${A}`,body:`let rank = index;
  var left = CAPACITY;
  var lower = CAPACITY;
  var upper = CAPACITY;
  if (rank < counts[countsOffset + COUNT_CLEAN]) {
    let vertex = sortedIndices[sortedIndicesOffset + rank];
    var usable = vertex < CAPACITY;
    if (usable) { usable = polygonOf(vertex) != NONE; }
    if (usable) {
      let kind = typeOf(vertex);
      if (kind == TYPE_SPLIT || kind == TYPE_MERGE || kind == TYPE_REGULAR_RIGHT) {
        let edge = vertexB[vertexBOffset + 4u * vertex + B_LEFT];
        if (edge == NONE) { failPolygon(polygonOf(vertex), FAIL_RAY); } else { left = edge; }
      }
      // The edge into the vertex is a down edge when its start is above; the edge out of the
      // vertex is a down edge when its end is below.
      if (rankOf(previousOf(vertex)) < rank) { lower = previousOf(vertex); }
      if (rankOf(nextOf(vertex)) > rank) { upper = vertex; }
    }
  }
  memberGroup[memberGroupOffset + 3u * rank] = left;
  memberGroup[memberGroupOffset + 3u * rank + 1u] = lower;
  memberGroup[memberGroupOffset + 3u * rank + 2u] = upper;
  memberIdentity[memberIdentityOffset + 3u * rank] = 3u * rank;
  memberIdentity[memberIdentityOffset + 3u * rank + 1u] = 3u * rank + 1u;
  memberIdentity[memberIdentityOffset + 3u * rank + 2u] = 3u * rank + 2u;`}}function _e(e){return{declarations:`${T(e)}${E}${A}`,body:`let member = sortedMembers[sortedMembersOffset + index];
  let groupId = memberGroup[memberGroupOffset + member];
  if (groupId == CAPACITY || index == 0u) { return; }
  let previousMember = sortedMembers[sortedMembersOffset + index - 1u];
  if (memberGroup[memberGroupOffset + previousMember] != groupId) { return; }
  let later = sortedIndices[sortedIndicesOffset + member / 3u];
  let earlier = sortedIndices[sortedIndicesOffset + previousMember / 3u];
  if (later >= CAPACITY || earlier >= CAPACITY) { return; }
  // A diagonal (later, earlier) is added when the later vertex is a split vertex or the earlier
  // one (the helper) is a merge vertex.
  if (!(typeOf(later) == TYPE_SPLIT || typeOf(earlier) == TYPE_MERGE)) { return; }
  let laterRole = member % 3u;
  let earlierRole = previousMember % 3u;
  // The insertion (role 2) is always first in its group and the deletion (role 1) always last.
  if (earlierRole == 1u || laterRole == 2u) {
    failPolygon(polygonOf(later), FAIL_DIAGONAL);
    return;
  }
  let laterSlot = select(1u, 0u, laterRole == 0u);
  let earlierSlot = select(3u, 2u, earlierRole == 0u);
  vertexDiagonal[vertexDiagonalOffset + 4u * later + laterSlot] = (earlier << 2u) | earlierSlot;
  vertexDiagonal[vertexDiagonalOffset + 4u * earlier + earlierSlot] = (later << 2u) | laterSlot;`}}function ve(e){return{declarations:`${T(e)}${E}${D}${O}${A}${k}
var<private> sortCenter: vec2f;
var<private> sortBase: vec2f;
var<private> sortInvalid: bool;
fn sameDirection(center: vec2f, first: vec2f, second: vec2f) -> bool {
  let dx1 = select(select(-1i, 0i, first.x == center.x), 1i, first.x > center.x);
  let dy1 = select(select(-1i, 0i, first.y == center.y), 1i, first.y > center.y);
  let dx2 = select(select(-1i, 0i, second.x == center.x), 1i, second.x > center.x);
  let dy2 = select(select(-1i, 0i, second.y == center.y), 1i, second.y > center.y);
  return dx1 == dx2 && dy1 == dy2;
}
fn angleCategory(candidate: vec2f) -> i32 {
  let orientation = orientOf(sortCenter, sortBase, candidate);
  if (orientation > 0) { return 0; }
  if (orientation < 0) { return 2; }
  return select(1i, -1i, sameDirection(sortCenter, sortBase, candidate));
}
fn angleBefore(first: vec2f, second: vec2f) -> bool {
  let firstCategory = angleCategory(first);
  let secondCategory = angleCategory(second);
  if (firstCategory != secondCategory) { return firstCategory < secondCategory; }
  if (firstCategory == 1 || firstCategory == -1) {
    sortInvalid = true;
    return false;
  }
  return orientOf(sortCenter, first, second) > 0;
}`,body:`let vertex = index;
  let polygon = polygonOf(vertex);
  if (polygon == NONE) { return; }
  let center = positionOf(vertex);
  let nextVertex = nextOf(vertex);
  let previousVertex = previousOf(vertex);
  sortCenter = center;
  sortBase = positionOf(nextVertex);
  let backPoint = positionOf(previousVertex);
  var others = array<u32, 4>(0u, 0u, 0u, 0u);
  var slots = array<u32, 4>(0u, 0u, 0u, 0u);
  var count = 0u;
  var invalid = false;
  for (var slot = 0u; slot < 4u; slot++) {
    let value = vertexDiagonal[vertexDiagonalOffset + 4u * vertex + slot];
    if (value == NONE) { continue; }
    let other = value >> 2u;
    let reverseSlot = value & 3u;
    // The reverse slot must point back, and a diagonal joins two vertices of one polygon that
    // are not neighbors on the ring.
    if (other >= CAPACITY || other == vertex || other == nextVertex || other == previousVertex) {
      invalid = true;
      continue;
    }
    if (polygonOf(other) != polygon ||
        vertexDiagonal[vertexDiagonalOffset + 4u * other + reverseSlot] != ((vertex << 2u) | slot)) {
      invalid = true;
      continue;
    }
    others[count] = other;
    slots[count] = slot;
    count += 1u;
  }
  // Insertion sort of at most four diagonals by counter-clockwise angle from the ring edge.
  for (var item = 1u; item < count; item++) {
    let heldOther = others[item];
    let heldSlot = slots[item];
    let heldPoint = positionOf(heldOther);
    var hole = item;
    loop {
      if (hole == 0u) { break; }
      if (!angleBefore(heldPoint, positionOf(others[hole - 1u]))) { break; }
      others[hole] = others[hole - 1u];
      slots[hole] = slots[hole - 1u];
      hole -= 1u;
    }
    others[hole] = heldOther;
    slots[hole] = heldSlot;
  }
  // Every diagonal lies strictly inside the wedge from the ring edge to the back edge, and the
  // directions are pairwise distinct.
  for (var item = 0u; item < count; item++) {
    let point = positionOf(others[item]);
    if (point.x == center.x && point.y == center.y) { invalid = true; }
    if (angleCategory(point) == -1) { invalid = true; }
    if (!angleBefore(point, backPoint)) { invalid = true; }
    if (item > 0u) {
      if (!angleBefore(positionOf(others[item - 1u]), point)) { invalid = true; }
    }
  }
  if (invalid || sortInvalid || orientUncertain) {
    failPolygon(polygon, FAIL_DIAGONAL);
    return;
  }
  // The list is [ring edge, d_1 .. d_k]; the successor of a half-edge ending at this vertex is the
  // entry before its reverse in the list (the back edge being the end of the list).
  var lastHalfEdge = vertex;
  if (count > 0u) { lastHalfEdge = CAPACITY + 4u * vertex + slots[count - 1u]; }
  nextHalfEdge[nextHalfEdgeOffset + previousVertex] = lastHalfEdge;
  for (var item = 0u; item < count; item++) {
    let value = vertexDiagonal[vertexDiagonalOffset + 4u * vertex + slots[item]];
    let incoming = CAPACITY + 4u * (value >> 2u) + (value & 3u);
    var before = vertex;
    if (item > 0u) { before = CAPACITY + 4u * vertex + slots[item - 1u]; }
    nextHalfEdge[nextHalfEdgeOffset + incoming] = before;
  }`}}function ye(e){return{declarations:`${T(e)}${E}`,body:`let vertex = index;
  let alive = polygonOf(vertex) != NONE;
  for (var slot = 0u; slot < 5u; slot++) {
    let halfEdge = select(CAPACITY + 4u * vertex + slot - 1u, vertex, slot == 0u);
    var valid = false;
    if (alive) {
      if (slot == 0u) {
        valid = true;
      } else {
        valid = vertexDiagonal[vertexDiagonalOffset + 4u * vertex + slot - 1u] != NONE;
      }
    }
    jumpA[jumpAOffset + 2u * halfEdge] = select(halfEdge, nextHalfEdge[nextHalfEdgeOffset + halfEdge], valid);
    jumpA[jumpAOffset + 2u * halfEdge + 1u] = select(NONE, halfEdge, valid);
  }`}}function be(e){return{declarations:T(e),body:`let vertex = index;
  for (var slot = 0u; slot < 5u; slot++) {
    let halfEdge = select(CAPACITY + 4u * vertex + slot - 1u, vertex, slot == 0u);
    let pointer = min(jumpSource[jumpSourceOffset + 2u * halfEdge], 5u * CAPACITY - 1u);
    let label = jumpSource[jumpSourceOffset + 2u * halfEdge + 1u];
    jumpTarget[jumpTargetOffset + 2u * halfEdge] = jumpSource[jumpSourceOffset + 2u * pointer];
    jumpTarget[jumpTargetOffset + 2u * halfEdge + 1u] = min(label, jumpSource[jumpSourceOffset + 2u * pointer + 1u]);
  }`}}function xe(e){return{declarations:T(e),body:`let remaining = counts[countsOffset + COUNT_JUMP_PAIRS];
  if (remaining > 0u) {
    counts[countsOffset + COUNT_JUMP_PAIRS] = remaining - 1u;
    counts[countsOffset + COUNT_JUMP_ACTIVE] = counts[countsOffset + COUNT_CLEAN];
  } else {
    counts[countsOffset + COUNT_JUMP_ACTIVE] = 0u;
  }`}}function Se(e){return{declarations:T(e),body:`let vertex = index;
  var count = 0u;
  for (var slot = 0u; slot < 5u; slot++) {
    let halfEdge = select(CAPACITY + 4u * vertex + slot - 1u, vertex, slot == 0u);
    if (jumpA[jumpAOffset + 2u * halfEdge + 1u] == halfEdge) { count += 1u; }
  }
  repCounts[repCountsOffset + vertex] = count;`}}function Ce(e){return{declarations:T(e),body:`let vertex = index;
  var piece = repOffsets[repOffsetsOffset + vertex];
  for (var slot = 0u; slot < 5u; slot++) {
    let halfEdge = select(CAPACITY + 4u * vertex + slot - 1u, vertex, slot == 0u);
    if (jumpA[jumpAOffset + 2u * halfEdge + 1u] == halfEdge) {
      if (piece < CAPACITY) { pieceTable[pieceTableOffset + PIECE_STRIDE * piece] = halfEdge; }
      piece += 1u;
    }
  }`}}function we(e){return{declarations:T(e),body:`let clean = counts[countsOffset + COUNT_CLEAN];
  counts[countsOffset + COUNT_PIECES] = min(repOffsets[repOffsetsOffset + clean], CAPACITY);`}}function Te(e){return{declarations:`${T(e)}${E}${A}${j}`,body:`let piece = index;
  if (piece >= counts[countsOffset + COUNT_PIECES]) { return; }
  let table = pieceTableOffset + PIECE_STRIDE * piece;
  let head = pieceTable[table];
  let polygon = polygonOf(tailOf(head));
  var vertexCountOfPiece = 0u;
  if (polygon != NONE) {
    // A cycle holds at most 3 n half-edges; a longer walk is not a cycle.
    let limit = 3u * atomicLoad(&polygonStats[polygonStatsOffset + STATS_STRIDE * polygon + STATS_CLEAN]) + 3u;
    var id = head;
    var steps = 0u;
    loop {
      id = min(nextHalfEdge[nextHalfEdgeOffset + id], 5u * CAPACITY - 1u);
      steps += 1u;
      if (id == head || steps > limit) { break; }
    }
    if (id == head && steps >= 3u) {
      vertexCountOfPiece = steps;
      atomicAdd(&polygonStats[polygonStatsOffset + STATS_STRIDE * polygon + STATS_TRIANGLES], steps - 2u);
    } else {
      failPolygon(polygon, FAIL_PIECE);
    }
  }
  pieceTable[table + 1u] = vertexCountOfPiece;
  pieceTriangles[pieceTrianglesOffset + piece] = select(0u, vertexCountOfPiece - 2u, vertexCountOfPiece >= 3u);`}}function Ee(e){return{declarations:`${T(e)}${N}`,body:`let route = routing[routingOffset + index];
  let plan = polygonPlanOffset + 4u * index;
  polygonPlan[plan] = 0u;
  polygonPlan[plan + 1u] = 0u;
  polygonPlan[plan + 2u] = 0u;
  polygonPlan[plan + 3u] = 0u;
  if (route == 0u || route == 3u) { return; }
  let sliceStart = baseSlotOf(index);
  let sliceEnd = min(baseSlotOf(index + 1u), i32(INDEX_LENGTH));
  if (sliceStart < 0 || sliceEnd <= sliceStart) { return; }
  let compact = keepOffsets[keepOffsetsOffset + routedOffsets[routedOffsetsOffset + index]];
  let firstPiece = repOffsets[repOffsetsOffset + min(compact, CAPACITY)];
  polygonPlan[plan] = u32(sliceStart);
  polygonPlan[plan + 1u] = u32(sliceEnd - sliceStart) / 3u;
  polygonPlan[plan + 2u] = pieceTriangleOffsets[pieceTriangleOffsetsOffset + min(firstPiece, CAPACITY)];
  polygonPlan[plan + 3u] = 1u;`}}function De(e){return{declarations:`${T(e)}${E}${A}${j}`,body:`let piece = index;
  let table = pieceTableOffset + PIECE_STRIDE * piece;
  if (piece >= counts[countsOffset + COUNT_PIECES]) {
    pieceTable[table + 1u] = 0u;
    return;
  }
  let polygon = polygonOf(tailOf(pieceTable[table]));
  if (polygon == NONE) {
    pieceTable[table + 1u] = 0u;
    return;
  }
  let plan = polygonPlanOffset + 4u * polygon;
  let triangleOffset = pieceTriangleOffsets[pieceTriangleOffsetsOffset + piece];
  let firstOffset = polygonPlan[plan + 2u];
  if (polygonPlan[plan + 3u] == 0u || triangleOffset < firstOffset) {
    pieceTable[table + 1u] = 0u;
    failPolygon(polygon, FAIL_PIECE);
    return;
  }
  pieceTable[table + 2u] = polygonPlan[plan] + 3u * (triangleOffset - firstOffset);
  pieceTable[table + 3u] = polygonPlan[plan] + 3u * polygonPlan[plan + 1u];
  pieceTable[table + 4u] = triangleOffset + 2u * piece;
  pieceTable[table + 5u] = polygon;`}}function Oe(e){return{declarations:`${T(e)}${E}${D}${O}${A}${j}${k}
const SCRATCH_REGION: u32 = ${3*e.capacity}u;
var<private> indexBase: u32;
var<private> indexLimit: u32;
var<private> triangleCursor: u32;
var<private> pieceFailed: bool;
var<private> sequenceBase: u32;
var<private> sequenceCount: u32;
var<private> sequenceTop: u32;
var<private> stackBase: u32;
var<private> stackSize: u32;

fn sequenceAt(position: u32) -> u32 {
  return pieceScratch[pieceScratchOffset + sequenceBase + (sequenceTop + position) % sequenceCount];
}
fn pushStack(pushedVertex: u32, pushedChain: u32) {
  if (stackSize >= sequenceCount) {
    pieceFailed = true;
    return;
  }
  pieceScratch[pieceScratchOffset + stackBase + stackSize] = pushedVertex * 2u + pushedChain;
  stackSize += 1u;
}
fn stackVertexAt(position: u32) -> u32 { return pieceScratch[pieceScratchOffset + stackBase + position] >> 1u; }
fn stackChainAt(position: u32) -> u32 { return pieceScratch[pieceScratchOffset + stackBase + position] & 1u; }
fn emitTriangle(cornerA: u32, cornerB: u32, cornerC: u32) {
  let triangleSlot = indexBase + 3u * triangleCursor;
  if (triangleSlot + 3u > indexLimit) {
    pieceFailed = true;
    return;
  }
  indices[indicesOffset + triangleSlot] = rowOf(cornerA);
  indices[indicesOffset + triangleSlot + 1u] = rowOf(cornerB);
  indices[indicesOffset + triangleSlot + 2u] = rowOf(cornerC);
  triangleCursor += 1u;
}
// Fan from \`vertex\` over the whole stack: counter-clockwise on a left-chain stack as
// (upper, lower, vertex) and on a right-chain stack as (lower, upper, vertex). In a monotone
// piece of a simple polygon no fan triangle is clockwise. One that is means the piece is not
// such a polygon (rings that touch can pass every other check, frc-geometry-02) and fails it:
// the triangles of a piece sum to its boundary, so they tile it exactly when none is clockwise.
fn fanStack(fanVertex: u32, fanChain: u32) {
  for (var fanItem = 0u; fanItem + 1u < stackSize; fanItem++) {
    let fanUpper = stackVertexAt(fanItem);
    let fanLower = stackVertexAt(fanItem + 1u);
    let fanFirst = select(fanLower, fanUpper, fanChain == 0u);
    let fanSecond = select(fanUpper, fanLower, fanChain == 0u);
    if (orientOf(positionOf(fanFirst), positionOf(fanSecond), positionOf(fanVertex)) < 0) {
      pieceFailed = true;
      return;
    }
    emitTriangle(fanFirst, fanSecond, fanVertex);
  }
}

fn triangulatePiece(piece: u32) {
  let table = pieceTableOffset + PIECE_STRIDE * piece;
  let pieceVertexCount = pieceTable[table + 1u];
  if (pieceVertexCount < 3u) { return; }
  let head = pieceTable[table];
  let polygon = pieceTable[table + 5u];
  indexBase = pieceTable[table + 2u];
  indexLimit = pieceTable[table + 3u];
  sequenceBase = pieceTable[table + 4u];
  sequenceCount = pieceVertexCount;
  stackBase = SCRATCH_REGION + sequenceBase;
  stackSize = 0u;
  triangleCursor = 0u;
  pieceFailed = false;
  if (sequenceBase + pieceVertexCount > SCRATCH_REGION) {
    failPolygon(polygon, FAIL_STACK);
    return;
  }
  // Copy the cycle and find its top and bottom vertices.
  var id = head;
  var topIndex = 0u;
  var bottomIndex = 0u;
  var topRank = 0u;
  var bottomRank = 0u;
  for (var position = 0u; position < pieceVertexCount; position++) {
    let vertex = tailOf(id);
    pieceScratch[pieceScratchOffset + sequenceBase + position] = vertex;
    let rank = rankOf(vertex);
    if (position == 0u || rank < topRank) {
      topRank = rank;
      topIndex = position;
    }
    if (position == 0u || rank > bottomRank) {
      bottomRank = rank;
      bottomIndex = position;
    }
    id = min(nextHalfEdge[nextHalfEdgeOffset + id], 5u * CAPACITY - 1u);
  }
  if (id != head) {
    failPolygon(polygon, FAIL_PIECE);
    return;
  }
  sequenceTop = topIndex;
  let bottom = (bottomIndex + pieceVertexCount - topIndex) % pieceVertexCount;
  // Merge the chains by rank. Positions 1..bottom are the left chain (the bottom included) and
  // pieceVertexCount - 1 down to bottom + 1 the right chain.
  var leftPosition = 1u;
  var rightPosition = pieceVertexCount - 1u;
  var leftRank = topRank;
  var rightRank = topRank;
  var processed = 1u;
  pushStack(sequenceAt(0u), 0u);
  loop {
    let leftAvailable = leftPosition <= bottom;
    let rightAvailable = rightPosition > bottom;
    if (!leftAvailable && !rightAvailable) { break; }
    var takeLeft = leftAvailable;
    var vertex = 0u;
    if (leftAvailable) { vertex = sequenceAt(leftPosition); }
    if (rightAvailable && (!leftAvailable || rankOf(sequenceAt(rightPosition)) < rankOf(vertex))) {
      takeLeft = false;
      vertex = sequenceAt(rightPosition);
    }
    let rank = rankOf(vertex);
    if (takeLeft) {
      if (rank <= leftRank) { pieceFailed = true; break; }
      leftRank = rank;
      leftPosition += 1u;
    } else {
      if (rank <= rightRank) { pieceFailed = true; break; }
      rightRank = rank;
      rightPosition -= 1u;
    }
    processed += 1u;
    let chain = select(1u, 0u, takeLeft);
    if (processed == 2u) {
      pushStack(vertex, chain);
      continue;
    }
    let topChain = stackChainAt(stackSize - 1u);
    if (leftPosition > bottom && rightPosition <= bottom) {
      // The bottom vertex: fan over everything left on the stack.
      fanStack(vertex, topChain);
      break;
    }
    if (chain != topChain) {
      // Opposite chains: fan over the whole stack, then keep its top and the new vertex.
      let previousTop = stackVertexAt(stackSize - 1u);
      fanStack(vertex, topChain);
      stackSize = 0u;
      pushStack(previousTop, topChain);
      pushStack(vertex, chain);
    } else {
      // Same chain: pop while the diagonal to the vertex stays inside the piece.
      var last = stackVertexAt(stackSize - 1u);
      stackSize -= 1u;
      loop {
        if (stackSize == 0u || pieceFailed) { break; }
        let upper = stackVertexAt(stackSize - 1u);
        var first = upper;
        var third = vertex;
        if (chain == 1u) {
          first = vertex;
          third = upper;
        }
        if (orientOf(positionOf(first), positionOf(last), positionOf(third)) < 0) { break; }
        if (chain == 0u) { emitTriangle(upper, last, vertex); } else { emitTriangle(vertex, last, upper); }
        last = upper;
        stackSize -= 1u;
      }
      pushStack(last, chain);
      pushStack(vertex, chain);
    }
    if (pieceFailed) { break; }
  }
  if (pieceFailed || orientUncertain || triangleCursor != pieceVertexCount - 2u) {
    failPolygon(polygon, FAIL_STACK);
  }
}
`,body:`triangulatePiece(index);`}}function ke(e,t){return{declarations:T(e),body:`let route = routing[routingOffset + index];
  if (route == 0u) { return; }
  let statsBase = polygonStatsOffset + STATS_STRIDE * index;
  let clean = polygonStats[statsBase + STATS_CLEAN];
  let holes = polygonStats[statsBase + STATS_HOLES];
  let triangles = polygonStats[statsBase + STATS_TRIANGLES];
  var ok = route != 3u && polygonStats[statsBase + STATS_FAIL] == 0u && clean >= 3u;
  // The decomposition must account for every triangle of the polygon, no more and no less.
  if (ok) { ok = triangles == clean + 2u * holes - 2u; }
  // A size-routed or overflowed polygon that did not complete goes to the Earcut fallback pass.
  if (!ok && (route == 1u || route == 3u)) { routing[routingOffset + index] = 4u; }
  polygonStats[statsBase + STATS_WRITTEN] = select(0u, triangles, ok);
  valid[validOffset + index] = select(0u, 1u, ok);
  ${t?`triangleCount[triangleCountOffset + index] = select(0u, triangles, ok);`:``}`}}function Ae(e){return{declarations:`${T(e)}${N}${P}
fn padUnit(padIndex: u32, padSliceStart: u32, padCapacity: u32, padWritten: u32, padVertex: u32) {
  if (padIndex >= padCapacity || padIndex < padWritten) { return; }
  let padSlot = indicesOffset + padSliceStart + 3u * padIndex;
  indices[padSlot] = padVertex;
  indices[padSlot + 1u] = padVertex;
  indices[padSlot + 2u] = padVertex;
}`,body:`let info = jInfoOffset + 4u * index;
  let rawPolygon = jInfo[info + 1u];
  if (rawPolygon == NONE) { return; }
  let polygon = rawPolygon & ~OVERFLOW_BIT;
  let sliceStart = baseSlotOf(polygon);
  let sliceEnd = min(baseSlotOf(polygon + 1u), i32(INDEX_LENGTH));
  if (sliceStart < 0 || sliceEnd <= sliceStart) { return; }
  let capacity = u32(sliceEnd - sliceStart) / 3u;
  let written = polygonStats[polygonStatsOffset + STATS_STRIDE * polygon + STATS_WRITTEN];
  let padRow = polygonFirstRow(polygon);
  let local = index - routedOffsets[routedOffsetsOffset + polygon];
  padUnit(local, u32(sliceStart), capacity, written, padRow);
  let ring = jInfo[info + 2u];
  let firstRing = jInfo[info + 3u];
  if (jInfo[info] == ringOffsets[ringOffsetsOffset + ring] && ring >= firstRing + 2u) {
    let polygonVertexCount = polygonEndRow(polygon) - padRow;
    let extra = polygonVertexCount + 2u * (ring - firstRing - 2u);
    padUnit(extra, u32(sliceStart), capacity, written, padRow);
    padUnit(extra + 1u, u32(sliceStart), capacity, written, padRow);
  }`}}function F(e){return u(e,40)}function I(e,t,n){c(t,`monotoneMaximumVertices`,n,F(e))}var je=256,Me=64;function Ne(e){return{preparedLayout:!!e.layout,vertexCount:e.positions.length,polygonCount:e.polygonOffsets.length-1,ringCount:e.ringOffsets.length-1,capacity:e.capacity,indexLength:e.output.indices.length,minimumVertices:e.minimumVertices}}function Pe(e){return Math.max(1,32-Math.clz32(e))}function Fe(e){let t=Math.ceil(Math.log2(3*e+3));return Math.ceil(t/2)}function L(e,i){let{id:s,operation:c,positions:l,routing:u,output:d}=i,f=i.layout?.polygonOffsets??i.polygonOffsets,h=i.layout?.ringOffsets??i.ringOffsets,g=Ne(i),{vertexCount:_,polygonCount:v,ringCount:y,capacity:b}=g,x=[],C=i.recorder,w=(n,r)=>t(e,`${s}-${n}`,`uint32`,Math.max(r,1)),A=e=>Math.max(e,1),j=w(`counts`,8),M=w(`routed-counts`,v+1),N=w(`routed-offsets`,v+1),P=w(`polygon-stats`,8*v),re=w(`polygon-plan`,4*v),F=w(`keep-flags`,_+1),I=w(`keep-offsets`,_+1),L=w(`vertex-info`,4*_),Le=w(`row-to-clean`,_),R=w(`ring-info`,4*y),z=w(`ring-top`,y),B=w(`vertex-a`,4*b),V=w(`vertex-b`,4*b),H=t(e,`${s}-vertex-position`,`float32x2`,A(b)),Re=w(`sort-key-high`,b),ze=w(`sort-key-low`,b),Be=w(`identity`,b),Ve=w(`member-group`,3*b),He=w(`member-identity`,3*b),Ue=w(`vertex-diagonal`,4*b),U=w(`next-half-edge`,5*b),W=w(`jump-a`,10*b),We=w(`jump-b`,10*b),Ge=w(`rep-counts`,b+1),Ke=w(`rep-offsets`,b+1),qe=w(`piece-table`,8*b),Je=w(`piece-triangles`,b+1),Ye=w(`piece-triangle-offsets`,b+1),Xe=w(`piece-scratch`,6*b),Ze=(t,n)=>{x.push(r(e,t)),C?.({kind:`kernel`,id:t.id,props:t,gate:n})},G=(e,t,n,r,i,a={})=>Ze({id:`${s}-${e}`,operation:c,variant:t,bindings:n,invocationCount:Math.max(r,1),declarations:i.declarations,body:i.body,...a}),Qe=(t,n,r)=>{x.push(...new a({id:`${s}-${t}`,input:n,output:r,mode:`exclusive`}).getCommandNodes(e)),C?.({kind:`scan`,id:`${s}-${t}`,input:n,output:r})},K=(t,r,i,a,o=1)=>{let l=n(e,{id:`${s}-${t}-dispatch`,operation:c,count:j,countRow:r,maximumItemCount:i,workgroupSize:a,invocationsPerItem:o});return x.push(l.updateNode),{condition:l.condition,workgroupSize:a,gate:{countRow:r,invocationsPerItem:o,workgroupSize:a}}},q=(e,t,n,r,i,a)=>Ze({id:`${s}-${t}`,operation:c,variant:n,bindings:r,invocationCount:Math.max(i,1),workgroupSize:e.workgroupSize,condition:{...e.condition,id:`${s}-${t}-gate`},declarations:a.declarations,body:a.body},e.gate),J=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read`}),Y=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),X=(e,t)=>Y(e,t,`atomic<u32>`);G(`route-counts`,`monotone-route-counts`,[J(`polygonOffsets`,f),J(`ringOffsets`,h),J(`routing`,u),Y(`routedCounts`,M),Y(`polygonStats`,P),Y(`counts`,j)],v+1,ie(g)),Qe(`route-scan`,M,N),G(`route-overflow`,`monotone-route-overflow`,[Y(`routing`,u),Y(`routedCounts`,M),J(`routedOffsets`,N),Y(`counts`,j)],v,ae(g));let $e=K(`routed`,S.routed,_,je),Z=te(e,{id:`${s}-slabs`,operation:c,owner:`polygon`,positions:l,polygonOffsets:i.polygonOffsets,ringOffsets:i.ringOffsets,gate:{counts:M,minimum:1}});x.push(...Z.nodes),C?.({kind:`slabs`,id:`${s}-slabs`,params:Z.slabs.params,offsets:Z.slabs.slabOffsets,entries:Z.slabs.slabEntries}),q($e,`index`,`monotone-index`,[J(`polygonOffsets`,f),J(`ringOffsets`,h),J(`positions`,l,`f32`),J(`routing`,u),J(`routedOffsets`,N),Y(`keepFlags`,F),Y(`jInfo`,L),X(`polygonStats`,P)],_,oe(g)),Qe(`keep-scan`,F,I),q($e,`ring`,`monotone-ring`,[J(`jInfo`,L),J(`ringOffsets`,h),J(`routedOffsets`,N),J(`keepOffsets`,I),Y(`ringInfo`,R),X(`ringTop`,z),X(`polygonStats`,P),X(`counts`,j)],_,se(g)),G(`counts`,`monotone-counts`,[Y(`counts`,j),J(`keepOffsets`,I)],1,ce(g));let Q=K(`clean`,S.clean,b,je),et=K(`members`,S.clean,b,je,3);q($e,`compact`,`monotone-compact`,[J(`jInfo`,L),J(`keepFlags`,F),J(`keepOffsets`,I),J(`ringInfo`,R),J(`positions`,l,`f32`),Y(`vertexA`,B),Y(`vertexPosition`,H,`f32`),Y(`rowToClean`,Le)],_,le(g)),G(`keys`,`monotone-keys`,[J(`counts`,j),Y(`vertexA`,B),J(`vertexPosition`,H,`f32`),Y(`sortKeyHigh`,Re),Y(`sortKeyLow`,ze),Y(`identity`,Be)],b,ue(g));let tt=o(e,{id:`${s}-sweep-sort`,operation:c,indices:Be,keys:[{view:Re},{view:ze}]});x.push(...tt.nodes);let $=tt.sortedIndices;C?.({kind:`sort`,id:`${s}-sweep-sort`,keys:[Re,ze],indices:Be,sortedIndices:$}),q(Q,`rank`,`monotone-rank`,[J(`sortedIndices`,$),J(`vertexA`,B),Y(`vertexB`,V),X(`ringTop`,z)],b,de(g)),q(Q,`flip`,`monotone-flip`,[J(`vertexA`,B),J(`vertexB`,V),Y(`ringInfo`,R),X(`ringTop`,z),J(`vertexPosition`,H,`f32`),J(`polygonOffsets`,f),X(`polygonStats`,P)],b,Ie(fe(g))),q(Q,`link`,`monotone-link`,[Y(`vertexA`,B),Y(`vertexB`,V),J(`ringInfo`,R),J(`vertexPosition`,H,`f32`),Y(`vertexDiagonal`,Ue),Y(`nextHalfEdge`,U),X(`polygonStats`,P)],b,Ie(pe(g)));let nt=ne(Z.slabs),rt=`${ee}${T(g)}${E}${D}${O}${k}${p(Z.slabs)}${me}`;q(Q,`left-edge`,`monotone-left-edge`,[...nt,J(`rowToClean`,Le),J(`vertexA`,B),Y(`vertexB`,V),J(`vertexPosition`,H,`f32`)],b,{declarations:rt,body:he(e=>m({owner:`polygon`,lowY:`point.y`,highY:`point.y`,body:e}))}),G(`members`,`monotone-members`,[J(`counts`,j),J(`sortedIndices`,$),J(`vertexA`,B),J(`vertexB`,V),Y(`memberGroup`,Ve),Y(`memberIdentity`,He),X(`polygonStats`,P)],b,ge(g));let it=o(e,{id:`${s}-member-sort`,operation:c,indices:He,keys:[{view:Ve,keyBits:Pe(b)}]});x.push(...it.nodes),C?.({kind:`sort`,id:`${s}-member-sort`,keys:[Ve],indices:He,sortedIndices:it.sortedIndices}),q(et,`diagonals`,`monotone-diagonals`,[J(`sortedMembers`,it.sortedIndices),J(`memberGroup`,Ve),J(`sortedIndices`,$),J(`vertexA`,B),Y(`vertexDiagonal`,Ue),X(`polygonStats`,P)],3*b,_e(g)),q(Q,`next`,`monotone-next`,[J(`vertexA`,B),J(`vertexB`,V),J(`vertexPosition`,H,`f32`),J(`vertexDiagonal`,Ue),Y(`nextHalfEdge`,U),X(`polygonStats`,P)],b,Ie(ve(g))),q(Q,`jump-init`,`monotone-jump-init`,[J(`vertexA`,B),J(`vertexDiagonal`,Ue),J(`nextHalfEdge`,U),Y(`jumpA`,W)],b,ye(g));let at=i.jumpPairCount??Fe(b),ot=be(g),st=xe(g);for(let e=0;e<at;e++){G(`jump-${e}-state`,`monotone-jump-state`,[Y(`counts`,j)],1,st);let t=K(`jump-${e}`,S.jumpActive,b,je);q(t,`jump-${e}-forward`,`monotone-jump`,[J(`jumpSource`,W),Y(`jumpTarget`,We)],b,ot),q(t,`jump-${e}-backward`,`monotone-jump`,[J(`jumpSource`,We),Y(`jumpTarget`,W)],b,ot)}q(Q,`rep-count`,`monotone-rep-count`,[J(`jumpA`,W),Y(`repCounts`,Ge)],b,Se(g)),Qe(`rep-scan`,Ge,Ke),q(Q,`pieces`,`monotone-pieces`,[J(`jumpA`,W),J(`repOffsets`,Ke),Y(`pieceTable`,qe)],b,Ce(g)),G(`piece-count`,`monotone-piece-count`,[Y(`counts`,j),J(`repOffsets`,Ke)],1,we(g));let ct=K(`pieces`,S.pieces,b,Me);return q(ct,`measure`,`monotone-measure`,[J(`counts`,j),Y(`pieceTable`,qe),J(`nextHalfEdge`,U),J(`vertexA`,B),X(`polygonStats`,P),Y(`pieceTriangles`,Je)],b,Te(g)),Qe(`piece-scan`,Je,Ye),G(`polygon-plan`,`monotone-polygon-plan`,[J(`polygonOffsets`,f),J(`ringOffsets`,h),J(`routing`,u),J(`routedOffsets`,N),J(`keepOffsets`,I),J(`repOffsets`,Ke),J(`pieceTriangleOffsets`,Ye),Y(`polygonPlan`,re)],v,Ee(g)),q(ct,`piece-plan`,`monotone-piece-plan`,[J(`counts`,j),Y(`pieceTable`,qe),J(`pieceTriangleOffsets`,Ye),J(`polygonPlan`,re),J(`vertexA`,B),X(`polygonStats`,P)],b,De(g)),q(ct,`triangulate`,`monotone-triangulate`,[J(`pieceTable`,qe),J(`nextHalfEdge`,U),J(`vertexA`,B),J(`vertexB`,V),J(`vertexPosition`,H,`f32`),Y(`pieceScratch`,Xe),Y(`indices`,d.indices),X(`polygonStats`,P)],b,Ie(Oe(g))),G(`final`,`monotone-final`,[Y(`routing`,u),Y(`polygonStats`,P),Y(`valid`,d.valid),...d.triangleCount?[Y(`triangleCount`,d.triangleCount)]:[]],v,ke(g,!!d.triangleCount)),q($e,`pad`,`monotone-pad`,[J(`jInfo`,L),J(`polygonOffsets`,f),J(`ringOffsets`,h),J(`routedOffsets`,N),J(`polygonStats`,P),Y(`indices`,d.indices)],_,Ae(g)),x}function Ie(e){return{declarations:`${ee}${e.declarations}`,body:e.body}}function Le(e,n){let{id:i,positions:o,polygonOffsets:s,ringOffsets:c,triangleOffsets:l}=n,u=s.length-1,d=c.length-1,f=`GPUPolygonTriangulation`,p={polygonOffsets:t(e,`${i}-polygon-layout`,`uint32`,4*u+2),ringOffsets:t(e,`${i}-ring-layout`,`uint32`,2*d+1)},m=t(e,`${i}-triangle-capacities`,`uint32`,u+1);return{nodes:[r(e,{id:`${i}-classify-rings`,operation:f,variant:`classify-rings`,bindings:[{name:`positions`,view:o,type:`f32`,access:`read`},{name:`ringOffsets`,view:c,type:`u32`,access:`read`},{name:`ringLayout`,view:p.ringOffsets,type:`u32`,access:`read_write`}],invocationCount:d+1,declarations:`${ee}
fn readVertex(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * vertex], positions[positionsOffset + 2u * vertex + 1u]);
}`,body:`let first = ringOffsets[ringOffsetsOffset + index];
  ringLayout[ringLayoutOffset + index] = first;
  if (index == ${d}u) { return; }
  let end = min(ringOffsets[ringOffsetsOffset + index + 1u], ${o.length}u);
  var nonDegenerate = 0u;
  if (end >= first + 3u) {
    let origin = readVertex(first);
    for (var vertex = first + 1u; vertex + 1u < end; vertex++) {
      // Match study-window preparation: uncertified (2) is not exact zero.
      if (orientSign(origin, readVertex(vertex), readVertex(vertex + 1u)) != 0) {
        nonDegenerate = 1u;
        break;
      }
    }
  }
  ringLayout[ringLayoutOffset + ${d+1}u + index] = nonDegenerate;`}),r(e,{id:`${i}-polygon-capacities`,operation:f,variant:`polygon-capacities`,bindings:[{name:`polygonOffsets`,view:s,type:`u32`,access:`read`},{name:`ringLayout`,view:p.ringOffsets,type:`u32`,access:`read`},{name:`polygonLayout`,view:p.polygonOffsets,type:`u32`,access:`read_write`},{name:`counts`,view:m,type:`u32`,access:`read_write`}],invocationCount:u+1,body:`let first = polygonOffsets[polygonOffsetsOffset + index];
  polygonLayout[polygonLayoutOffset + index] = first;
  counts[countsOffset + index] = 0u;
  if (index == ${u}u) { return; }
  let end = polygonOffsets[polygonOffsetsOffset + index + 1u];
  var vertices = 0u;
  var rings = 0u;
  if (first < end && ringLayout[ringLayoutOffset + ${d+1}u + first] != 0u) {
    for (var ring = first; ring < end; ring++) {
      if (ringLayout[ringLayoutOffset + ${d+1}u + ring] == 0u) { continue; }
      vertices += ringLayout[ringLayoutOffset + ring + 1u] - ringLayout[ringLayoutOffset + ring];
      rings += 1u;
    }
  }
  if (rings > 0u) { counts[countsOffset + index] = vertices + 2u * rings - 4u; }
  polygonLayout[polygonLayoutOffset + ${2*(u+1)}u + index] = vertices;
  polygonLayout[polygonLayoutOffset + ${3*u+2}u + index] = rings;`}),...new a({id:`${i}-triangle-offsets`,input:m,output:l,mode:`exclusive`}).getCommandNodes(e),r(e,{id:`${i}-publish-layout`,operation:f,variant:`publish-layout`,bindings:[{name:`triangleOffsets`,view:l,type:`u32`,access:`read`},{name:`polygonLayout`,view:p.polygonOffsets,type:`u32`,access:`read_write`}],invocationCount:u+1,body:`polygonLayout[polygonLayoutOffset + ${u+1}u + index] = triangleOffsets[triangleOffsetsOffset + index];`})],layout:p}}var R=`GPUPolygonTriangulation`;function z(e,t,n){return n>0?Math.max(0,3*(e+2*t-4)):0}function B(e,t=!0){return u(e,t?28:16)}function V(e,t,n){let r=t-n;return 2*(e+2*r)+8*n+r}function H(e,t,n,r,i,a=!0){c(t,`scratch nodes`,V(n,r,i),B(e,a)),l(t,`scratch byteLength`,Math.max((a?28:16)*V(n,r,i),16),0,e.limits.maxBufferSize)}var Re=80,ze=class{id;props;spatialContext;polygonCount;ringCount;indexCount;constructor(e){let t=y(e,[`id`,`spatialContext`]);this.id=t.id??`polygon-triangulation`;let n=t.spatialContext&&v(t.spatialContext);this.spatialContext=v(_(this.id,n));let r=y(e,[`positions`,`featureOffsets`,`polygonOffsets`,`ringOffsets`,`output`,`maximumWork`,`useZOrderHash`,`largePolygonMethod`,`monotoneMinimumVertices`,`monotoneMaximumVertices`]);e=Object.freeze({...t,...r,spatialContext:n,output:y(r.output,[`indices`,`valid`,`triangleCount`,`triangleOffsets`])}),this.props=e;let{id:i}=this,{output:a}=e;s(e.positions,[`float32x2`],`${i} positions`);for(let t of[`featureOffsets`,`polygonOffsets`,`ringOffsets`])if(s(e[t],[`uint32`],`${i} ${t}`),e[t].length<1)throw Error(`${i} ${t} requires a terminal entry`);this.polygonCount=e.polygonOffsets.length-1,this.ringCount=e.ringOffsets.length-1,this.indexCount=z(e.positions.length,this.ringCount,this.polygonCount),s(a.indices,[`uint32`],`${i} output.indices`);let o=Math.max(0,3*(e.positions.length+2*this.ringCount-4*this.polygonCount));if(a.indices.length<o)throw Error(`${i} output.indices needs at least ${o} rows`);if(a.triangleOffsets&&(s(a.triangleOffsets,[`uint32`],`${i} output.triangleOffsets`),a.triangleOffsets.length!==this.polygonCount+1))throw Error(`${i} triangleOffsets length mismatch`);if(s(a.valid,[`uint32`],`${i} output.valid`),a.valid.length!==this.polygonCount)throw Error(`${i} output.valid length must equal the polygon count`);if(a.triangleCount&&(s(a.triangleCount,[`uint32`],`${i} output.triangleCount`),a.triangleCount.length!==this.polygonCount))throw Error(`${i} output.triangleCount length must equal the polygon count`);if(e.maximumWork!==void 0&&l(i,`maximumWork`,e.maximumWork),e.largePolygonMethod!==void 0&&![`monotone`,`none`].includes(e.largePolygonMethod))throw Error(`${i} largePolygonMethod must be 'monotone' or 'none'`);e.monotoneMinimumVertices!==void 0&&l(i,`monotoneMinimumVertices`,e.monotoneMinimumVertices),e.monotoneMaximumVertices!==void 0&&l(i,`monotoneMaximumVertices`,e.monotoneMaximumVertices);let c={indices:a.indices,triangleOffsets:a.triangleOffsets,valid:this.polygonCount>0?a.valid:void 0,triangleCount:this.polygonCount>0?a.triangleCount:void 0};x(i,c,{separateDispatches:!0}),x(i,{indices:a.indices,valid:c.valid,triangleCount:c.triangleCount}),f(i,Object.values(c),[e.positions,e.featureOffsets,e.polygonOffsets,e.ringOffsets])}getViews(){let{positions:e,featureOffsets:t,polygonOffsets:n,ringOffsets:r,output:i}=this.props,{indices:a,valid:o,triangleCount:s,triangleOffsets:c}=i;return[e,t,n,r,a,this.polygonCount>0?o:void 0,this.polygonCount>0?s:void 0,c]}getCommandNodes(n){let{id:a,props:o,polygonCount:s,ringCount:c}=this,{output:l}=o;b(n,a,Object.fromEntries(this.getViews().map((e,t)=>[`view ${t}`,e]))),d(n.device,a,`triangle offsets scratch`,(s+1)*4,`reduce the polygon count`),H(n.device,a,o.positions.length,c,s,o.useZOrderHash??!0);let u=o.positions.length,f=(o.largePolygonMethod??`monotone`)===`monotone`&&u<=F(n.device);f&&o.monotoneMaximumVertices!==void 0&&I(n.device,a,o.monotoneMaximumVertices);let p=V(u,c,s),m=o.useZOrderHash??!0,te=t(n,`${a}-scratch`,`uint32`,Math.max((m?7:4)*p,4)),ne=l.triangleOffsets??t(n,`${a}-polygon-triangle-offsets`,`uint32`,s+1),{nodes:h,layout:_}=Le(n,{id:a,positions:o.positions,polygonOffsets:o.polygonOffsets,ringOffsets:o.ringOffsets,triangleOffsets:ne});if(h.push(i(n,{id:`${a}-clear-indices`,operation:R,view:l.indices,type:`u32`,value:`0u`})),s===0)return h;let v=[{name:`positions`,view:o.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:_.ringOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:_.polygonOffsets,type:`u32`,access:`read`},{name:`indices`,view:l.indices,type:`u32`,access:`read_write`},{name:`valid`,view:l.valid,type:`u32`,access:`read_write`},{name:`scratch`,view:te,type:`u32`,access:`read_write`}];l.triangleCount&&v.push({name:`triangleCount`,view:l.triangleCount,type:`u32`,access:`read_write`});let y=f?t(n,`${a}-routing`,`uint32`,Math.max(s,1)):void 0;y&&v.push({name:`routing`,view:y,type:`u32`,access:`read_write`});let x=[];y&&(x=L(n,{id:`${a}-monotone`,operation:R,positions:o.positions,polygonOffsets:o.polygonOffsets,ringOffsets:o.ringOffsets,routing:y,layout:_,output:l,capacity:e(u,1,o.monotoneMaximumVertices??u),minimumVertices:o.monotoneMinimumVertices??4096}));let S=o.monotoneMinimumVertices??4096,C=(e,t)=>r(n,{id:e,operation:R,variant:`earcut`,bindings:v,invocationCount:s,workgroupSize:64,declarations:`${ee}
// Only sosPointSide and exactCompareSign are called; exactOrientSigns (the helper's orientSign
// call site) is never called here and is dropped as dead code.
${g}
${Be({vertexCount:u,polygonCount:s,ringCount:c,indexLength:l.indices.length,nodeCount:p,useZOrderHash:m,maximumWork:o.maximumWork??4e6,writeTriangleCount:!!l.triangleCount,routingPass:t,minimumVertices:S})}`,body:`triangulatePolygon(index);`});return h.push(C(`${a}-earcut`,y?`first`:void 0)),y&&x.push(C(`${a}-earcut-fallback`,`fallback`)),[...h,...x]}};function Be(e){return`
const VERTEX_COUNT: u32 = ${e.vertexCount}u;
const POLYGON_COUNT: u32 = ${e.polygonCount}u;
const INDEX_LENGTH: u32 = ${e.indexLength}u;
const SCRATCH_NODE_COUNT: u32 = ${e.nodeCount}u;
const Z_BASE: u32 = ${4*e.nodeCount}u;
const USE_Z_HASH: bool = ${e.useZOrderHash};
const Z_HASH_MINIMUM_VERTICES: u32 = ${Re}u;
const WORK_LIMIT: u32 = ${e.maximumWork}u;
const NONE: u32 = 0xffffffffu;
${e.routingPass===`first`?re(e.minimumVertices,!0):``}
var<private> nodeBase: u32;
var<private> nodeCursor: u32;
var<private> nodeLimit: u32;
var<private> workDone: u32;
var<private> failed: bool;
var<private> incomplete: bool;
var<private> filteredOut: bool;
var<private> triangleBase: u32;
var<private> triangleSlots: u32;
var<private> trianglesWritten: u32;
var<private> hashed: bool;
var<private> hashMinimum: vec2f;
var<private> hashScale: f32;

// Node n lives in scratch words 4n..4n+3: next, previous, vertex row, spare (hole queue, ring stack).
fn nextOf(node: u32) -> u32 { return scratch[scratchOffset + 4u * node]; }
fn prevOf(node: u32) -> u32 { return scratch[scratchOffset + 4u * node + 1u]; }
fn vertexOf(node: u32) -> u32 { return scratch[scratchOffset + 4u * node + 2u]; }
fn setNext(node: u32, value: u32) { scratch[scratchOffset + 4u * node] = value; }
// z-order list of node n: z, previous in z order, next in z order (NONE-terminated).
fn zOf(node: u32) -> u32 { return scratch[scratchOffset + Z_BASE + 3u * node]; }
fn prevZOf(node: u32) -> u32 { return scratch[scratchOffset + Z_BASE + 3u * node + 1u]; }
fn nextZOf(node: u32) -> u32 { return scratch[scratchOffset + Z_BASE + 3u * node + 2u]; }
fn setZ(node: u32, value: u32) { scratch[scratchOffset + Z_BASE + 3u * node] = value; }
fn setPrevZ(node: u32, value: u32) { scratch[scratchOffset + Z_BASE + 3u * node + 1u] = value; }
fn setNextZ(node: u32, value: u32) { scratch[scratchOffset + Z_BASE + 3u * node + 2u] = value; }
fn setPrev(node: u32, value: u32) { scratch[scratchOffset + 4u * node + 1u] = value; }
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
fn pointOf(node: u32) -> vec2f { return vertexAt(vertexOf(node)); }
fn samePoint(first: vec2f, second: vec2f) -> bool { return first.x == second.x && first.y == second.y; }
fn spend(amount: u32) {
  workDone += amount;
  if (workDone > WORK_LIMIT) { failed = true; }
}

// Sign of Earcut's area(p, q, r), which is minus the orientation: negative at a convex vertex
// of the (counter-clockwise) working order. A predicate that cannot be certified (non-finite
// coordinates) fails the polygon: no decision may rest on it.
fn areaSign(p: vec2f, q: vec2f, r: vec2f) -> i32 {
  let result = orientSign(p, q, r);
  if (result == 2) {
    failed = true;
    return 0;
  }
  return -result;
}
fn orientOf(p: vec2f, q: vec2f, r: vec2f) -> i32 {
  let result = orientSign(p, q, r);
  if (result == 2) {
    failed = true;
    return 0;
  }
  return result;
}

fn createNode(vertex: u32) -> u32 {
  if (nodeCursor >= nodeLimit) {
    failed = true;
    return nodeBase;
  }
  let node = nodeCursor;
  nodeCursor += 1u;
  scratch[scratchOffset + 4u * node + 2u] = vertex;
  if (hashed) {
    setPrevZ(node, NONE);
    setNextZ(node, NONE);
  }
  return node;
}
fn insertNode(vertex: u32, last: u32) -> u32 {
  let node = createNode(vertex);
  if (last == NONE) {
    setPrev(node, node);
    setNext(node, node);
  } else {
    let following = nextOf(last);
    setNext(node, following);
    setPrev(node, last);
    setPrev(following, node);
    setNext(last, node);
  }
  return node;
}
fn removeNode(node: u32) {
  if (hashed) {
    let previousZ = prevZOf(node);
    let followingZ = nextZOf(node);
    if (previousZ != NONE) { setNextZ(previousZ, followingZ); }
    if (followingZ != NONE) { setPrevZ(followingZ, previousZ); }
  }
  setPrev(nextOf(node), prevOf(node));
  setNext(prevOf(node), nextOf(node));
}
fn equalNodes(first: u32, second: u32) -> bool { return samePoint(pointOf(first), pointOf(second)); }

// Winding of a ring at its lexicographically smallest vertex: 1 counter-clockwise, -1 clockwise,
// 0 undetermined, 2 uncertain.
fn ringWinding(start: u32, count: u32) -> i32 {
  var extreme = 0u;
  for (var k = 1u; k < count; k++) {
    let candidate = vertexAt(start + k);
    let best = vertexAt(start + extreme);
    if (candidate.x < best.x || (candidate.x == best.x && candidate.y < best.y)) { extreme = k; }
  }
  let center = vertexAt(start + extreme);
  var previous = extreme;
  var next = extreme;
  var foundPrevious = false;
  var foundNext = false;
  for (var step = 1u; step < count; step++) {
    if (!foundPrevious) {
      previous = (extreme + count - step) % count;
      foundPrevious = !samePoint(vertexAt(start + previous), center);
    }
    if (!foundNext) {
      next = (extreme + step) % count;
      foundNext = !samePoint(vertexAt(start + next), center);
    }
  }
  if (!foundPrevious || !foundNext) { return 0; }
  return orientSign(vertexAt(start + previous), center, vertexAt(start + next));
}

// Circular list of a ring in counter-clockwise (shell) or clockwise (hole) working order.
fn linkedList(start: u32, count: u32, wantCounterClockwise: bool) -> u32 {
  if (count == 0u) { return NONE; }
  let winding = ringWinding(start, count);
  var forward = true;
  if (winding == 1 || winding == -1) { forward = (winding == 1) == wantCounterClockwise; }
  var last = NONE;
  for (var k = 0u; k < count; k++) {
    let row = select(start + count - 1u - k, start + k, forward);
    last = insertNode(row, last);
  }
  if (last != NONE && equalNodes(last, nextOf(last))) {
    removeNode(last);
    last = nextOf(last);
  }
  return last;
}

// Removes collinear and coincident nodes; with end == start the whole ring until stable. The
// ring keeps its boundary as a sum of directed segments: two collinear edges add up to the edge
// that replaces them, also where one folds back over the other.
fn filterPoints(start: u32, endIn: u32) -> u32 {
  var end = endIn;
  let full = end == start;
  var p = start;
  loop {
    if (failed) { break; }
    spend(1u);
    var again = false;
    let following = nextOf(p);
    if (p != following &&
        (equalNodes(p, following) || areaSign(pointOf(prevOf(p)), pointOf(p), pointOf(following)) == 0)) {
      if (full || p == end) { end = prevOf(p); }
      filteredOut = true;
      removeNode(p);
      p = prevOf(p);
      again = true;
    } else if (full || p != end) {
      p = following;
      again = !full;
    }
    if (!(again || p != end)) { break; }
  }
  return end;
}

fn pointInTriangle(a: vec2f, b: vec2f, c: vec2f, p: vec2f) -> bool {
  return orientOf(p, c, a) >= 0 && orientOf(p, a, b) >= 0 && orientOf(p, b, c) >= 0;
}

fn isEar(ear: u32) -> bool {
  let a = prevOf(ear);
  let c = nextOf(ear);
  let pa = pointOf(a);
  let pb = pointOf(ear);
  let pc = pointOf(c);
  let low = min(pa, min(pb, pc));
  let high = max(pa, max(pb, pc));
  var p = nextOf(c);
  while (p != a) {
    spend(1u);
    if (failed) { return false; }
    let pp = pointOf(p);
    if (pp.x >= low.x && pp.x <= high.x && pp.y >= low.y && pp.y <= high.y && !samePoint(pa, pp) &&
        pointInTriangle(pa, pb, pc, pp) &&
        areaSign(pointOf(prevOf(p)), pp, pointOf(nextOf(p))) >= 0) {
      return false;
    }
    p = nextOf(p);
  }
  return true;
}

// Morton code of a point on a 32768 x 32768 grid over the shell's bounding box. Quantisation is
// monotone in each coordinate (and clamped), so every point inside a box has a code between the
// codes of the box's low and high corners.
fn spreadBits(value: u32) -> u32 {
  var x = value;
  x = (x | (x << 8u)) & 0x00ff00ffu;
  x = (x | (x << 4u)) & 0x0f0f0f0fu;
  x = (x | (x << 2u)) & 0x33333333u;
  x = (x | (x << 1u)) & 0x55555555u;
  return x;
}
fn zOrderOf(point: vec2f) -> u32 {
  let scaled = clamp((point - hashMinimum) * hashScale, vec2f(0.0), vec2f(32767.0));
  return spreadBits(u32(scaled.x)) | (spreadBits(u32(scaled.y)) << 1u);
}

// Earcut's indexCurves: every node of the ring gets a code and joins a list sorted by it (stable
// bottom-up merge sort of the linked list).
fn indexCurves(start: u32) {
  var p = start;
  loop {
    spend(1u);
    setZ(p, zOrderOf(pointOf(p)));
    setPrevZ(p, prevOf(p));
    setNextZ(p, nextOf(p));
    p = nextOf(p);
    if (p == start) { break; }
  }
  setNextZ(prevZOf(start), NONE);
  setPrevZ(start, NONE);
  var list = start;
  var inSize = 1u;
  loop {
    var left = list;
    list = NONE;
    var tail = NONE;
    var merges = 0u;
    while (left != NONE) {
      merges += 1u;
      var right = left;
      var leftSize = 0u;
      for (var i = 0u; i < inSize; i++) {
        leftSize += 1u;
        right = nextZOf(right);
        if (right == NONE) { break; }
      }
      var rightSize = inSize;
      while (leftSize > 0u || (rightSize > 0u && right != NONE)) {
        spend(1u);
        var element = NONE;
        if (leftSize != 0u && (rightSize == 0u || right == NONE || zOf(left) <= zOf(right))) {
          element = left;
          left = nextZOf(left);
          leftSize -= 1u;
        } else {
          element = right;
          right = nextZOf(right);
          rightSize -= 1u;
        }
        if (tail != NONE) { setNextZ(tail, element); } else { list = element; }
        setPrevZ(element, tail);
        tail = element;
      }
      left = right;
    }
    setNextZ(tail, NONE);
    if (failed || merges <= 1u) { break; }
    inSize *= 2u;
  }
}

fn blocksEar(p: u32, a: u32, c: u32, pa: vec2f, pb: vec2f, pc: vec2f, low: vec2f, high: vec2f) -> bool {
  spend(1u);
  let pp = pointOf(p);
  return p != a && p != c && pp.x >= low.x && pp.x <= high.x && pp.y >= low.y && pp.y <= high.y &&
    !samePoint(pa, pp) && pointInTriangle(pa, pb, pc, pp) &&
    areaSign(pointOf(prevOf(p)), pp, pointOf(nextOf(p))) >= 0;
}

// Same decision as isEar, but only visits the nodes whose z-order code lies in the code range of
// the triangle's bounding box (Earcut's isEarHashed).
fn isEarHashed(ear: u32) -> bool {
  let a = prevOf(ear);
  let c = nextOf(ear);
  let pa = pointOf(a);
  let pb = pointOf(ear);
  let pc = pointOf(c);
  let low = min(pa, min(pb, pc));
  let high = max(pa, max(pb, pc));
  let minZ = zOrderOf(low);
  let maxZ = zOrderOf(high);
  var before = prevZOf(ear);
  var after = nextZOf(ear);
  while (before != NONE && zOf(before) >= minZ && after != NONE && zOf(after) <= maxZ) {
    if (failed) { return false; }
    if (blocksEar(before, a, c, pa, pb, pc, low, high)) { return false; }
    before = prevZOf(before);
    if (blocksEar(after, a, c, pa, pb, pc, low, high)) { return false; }
    after = nextZOf(after);
  }
  while (before != NONE && zOf(before) >= minZ) {
    if (failed) { return false; }
    if (blocksEar(before, a, c, pa, pb, pc, low, high)) { return false; }
    before = prevZOf(before);
  }
  while (after != NONE && zOf(after) <= maxZ) {
    if (failed) { return false; }
    if (blocksEar(after, a, c, pa, pb, pc, low, high)) { return false; }
    after = nextZOf(after);
  }
  return true;
}

fn isEarTest(ear: u32) -> bool {
  if (hashed) { return isEarHashed(ear); }
  return isEar(ear);
}

// Bounding-box test only: the caller has already established collinearity with areaSign().
fn onSegment(p: vec2f, q: vec2f, r: vec2f) -> bool {
  return q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) && q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y);
}
fn segmentsIntersect(p1: vec2f, q1: vec2f, p2: vec2f, q2: vec2f, includeBoundary: bool) -> bool {
  let o1 = areaSign(p1, q1, p2);
  let o2 = areaSign(p1, q1, q2);
  let o3 = areaSign(p2, q2, p1);
  let o4 = areaSign(p2, q2, q1);
  if (((o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0)) && ((o3 > 0 && o4 < 0) || (o3 < 0 && o4 > 0))) { return true; }
  if (!includeBoundary) { return false; }
  if (o1 == 0 && onSegment(p1, p2, q1)) { return true; }
  if (o2 == 0 && onSegment(p1, q2, q1)) { return true; }
  if (o3 == 0 && onSegment(p2, p1, q2)) { return true; }
  if (o4 == 0 && onSegment(p2, q1, q2)) { return true; }
  return false;
}

fn locallyInside(a: u32, b: u32) -> bool {
  let pa = pointOf(a);
  let pb = pointOf(b);
  let before = pointOf(prevOf(a));
  let after = pointOf(nextOf(a));
  if (areaSign(before, pa, after) < 0) {
    return areaSign(pa, pb, after) >= 0 && areaSign(pa, before, pb) >= 0;
  }
  return areaSign(pa, pb, before) < 0 || areaSign(pa, after, pb) < 0;
}

// Whether the diagonal midpoint is inside the working polygon: a topology decision, so the parity
// of a ray toward +x uses the exact rule of utils/exact-geometry-wgsl.ts. The midpoint is only a
// rounded f32 placement; its side of each straddling edge is the exact orientSign (through
// orientOf, no new orientSign call site) with Simulation-of-Simplicity ties from sosPointSide.
fn middleInside(a: u32, b: u32) -> bool {
  let middle = (pointOf(a) + pointOf(b)) * 0.5;
  var p = a;
  var inside = false;
  loop {
    spend(1u);
    if (failed) { return false; }
    let following = nextOf(p);
    let edgeStart = pointOf(p);
    let edgeEnd = pointOf(following);
    if ((edgeStart.y >= middle.y) != (edgeEnd.y >= middle.y)) {
      // orientOf is orientSign (positive when the midpoint is left of edgeStart -> edgeEnd) with
      // uncertain counted as zero, which sosPointSide then breaks by perturbation.
      let side = sosPointSide(orientOf(edgeStart, edgeEnd, middle), edgeStart, edgeEnd);
      let isUpward = edgeEnd.y > edgeStart.y;
      if ((isUpward && side > 0) || (!isUpward && side < 0)) {
        inside = !inside;
      }
    }
    p = following;
    if (p == a) { break; }
  }
  return inside;
}

fn intersectsPolygon(a: u32, b: u32) -> bool {
  let pa = pointOf(a);
  let pb = pointOf(b);
  let low = min(pa, pb);
  let high = max(pa, pb);
  let indexA = vertexOf(a);
  let indexB = vertexOf(b);
  var p = a;
  loop {
    spend(1u);
    if (failed) { return true; }
    let following = nextOf(p);
    let edgeStart = pointOf(p);
    let edgeEnd = pointOf(following);
    let outside = (edgeStart.x > high.x && edgeEnd.x > high.x) || (edgeStart.x < low.x && edgeEnd.x < low.x) ||
      (edgeStart.y > high.y && edgeEnd.y > high.y) || (edgeStart.y < low.y && edgeEnd.y < low.y);
    if (!outside) {
      let indexFrom = vertexOf(p);
      let indexTo = vertexOf(following);
      if (indexFrom != indexA && indexTo != indexA && indexFrom != indexB && indexTo != indexB &&
          segmentsIntersect(edgeStart, edgeEnd, pa, pb, true)) {
        return true;
      }
    }
    p = following;
    if (p == a) { break; }
  }
  return false;
}

fn isValidDiagonal(a: u32, b: u32) -> bool {
  let pa = pointOf(a);
  let pb = pointOf(b);
  let zeroLength = samePoint(pa, pb) &&
    areaSign(pointOf(prevOf(a)), pa, pointOf(nextOf(a))) > 0 &&
    areaSign(pointOf(prevOf(b)), pb, pointOf(nextOf(b))) > 0;
  if (vertexOf(nextOf(a)) == vertexOf(b)) { return false; }
  // Two nodes at one point (rings that touch at a vertex, or a bridge at a shared vertex) can
  // only be split in Earcut's special case, where both sectors are reflex and the split pairs
  // the edges at the point the other way. Between two convex sectors the split would turn one
  // loop inside out. Earcut leaves that case to its float midpoint test at the shared point;
  // the exact midpoint parity below perturbs the point to one side and can say inside.
  if (samePoint(pa, pb) && !zeroLength) { return false; }
  if (!(zeroLength || (locallyInside(a, b) && locallyInside(b, a) &&
      (areaSign(pointOf(prevOf(a)), pa, pointOf(prevOf(b))) != 0 ||
       areaSign(pa, pointOf(prevOf(b)), pb) != 0)))) {
    return false;
  }
  if (intersectsPolygon(a, b)) { return false; }
  return zeroLength || middleInside(a, b);
}

// Splits (a, b) in two rings (or merges a hole ring into the shell); returns b's duplicate.
fn splitPolygon(a: u32, b: u32) -> u32 {
  let a2 = createNode(vertexOf(a));
  let b2 = createNode(vertexOf(b));
  let an = nextOf(a);
  let bp = prevOf(b);
  setNext(a, b);
  setPrev(b, a);
  setNext(a2, an);
  setPrev(an, a2);
  setNext(b2, a2);
  setPrev(a2, b2);
  setNext(bp, b2);
  setPrev(b2, bp);
  return b2;
}

fn emitTriangle(first: u32, second: u32, third: u32) {
  if (trianglesWritten >= triangleSlots) {
    failed = true;
    return;
  }
  let slot = triangleBase + 3u * trianglesWritten;
  indices[indicesOffset + slot] = first;
  indices[indicesOffset + slot + 1u] = second;
  indices[indicesOffset + slot + 2u] = third;
  trianglesWritten += 1u;
}

// Whether Earcut's cureLocalIntersections would act: two edges of the ring, one node apart, cross
// properly. Earcut writes the triangle (a, p, b) across the crossing and drops the two nodes
// between. That triangle and the piece dropped do not add up to the ring they replace, so a
// triangulation that contains one is never a tiling: the caller fails the polygon instead. On a
// valid polygon the working ring only crosses itself after an ear that was no ear, and every
// wrong triangulation found for valid polygons with touching rings (frc-geometry-01) had passed
// through this step.
fn hasLocalIntersection(start: u32) -> bool {
  var p = start;
  loop {
    spend(1u);
    if (failed) { break; }
    let a = prevOf(p);
    let following = nextOf(p);
    let b = nextOf(following);
    if (segmentsIntersect(pointOf(a), pointOf(p), pointOf(following), pointOf(b), false) &&
        locallyInside(a, b) && locallyInside(b, a)) {
      return true;
    }
    p = nextOf(p);
    if (p == start) { break; }
  }
  return false;
}

// First valid diagonal of the ring as (a, b), or (NONE, NONE).
fn findSplit(start: u32) -> vec2<u32> {
  var a = start;
  loop {
    var b = nextOf(nextOf(a));
    while (b != prevOf(a)) {
      spend(1u);
      if (failed) { return vec2<u32>(NONE, NONE); }
      if (vertexOf(a) != vertexOf(b) && isValidDiagonal(a, b)) { return vec2<u32>(a, b); }
      b = nextOf(b);
    }
    a = nextOf(a);
    if (a == start) { break; }
  }
  return vec2<u32>(NONE, NONE);
}

fn pushRing(depth: u32, head: u32) {
  if (nodeBase + depth >= nodeLimit) {
    failed = true;
    return;
  }
  scratch[scratchOffset + 4u * (nodeBase + depth) + 3u] = head;
}
fn popRing(depth: u32) -> u32 {
  return scratch[scratchOffset + 4u * (nodeBase + depth) + 3u];
}

// Earcut's earcutLinked, with the recursion of splitEarcut replaced by a stack of ring heads.
fn clipEars(first: u32) {
  var ear = first;
  var depth = 0u;
  loop {
    if (hashed) { indexCurves(ear); }
    var stop = ear;
    var cured = false;
    loop {
      if (failed) { break; }
      spend(1u);
      let before = prevOf(ear);
      let after = nextOf(ear);
      if (before == after) { break; }
      var earFound = false;
      if (areaSign(pointOf(before), pointOf(ear), pointOf(after)) < 0) {
        earFound = isEarTest(ear);
      }
      if (earFound) {
        emitTriangle(vertexOf(before), vertexOf(ear), vertexOf(after));
        removeNode(ear);
        ear = after;
        stop = after;
        continue;
      }
      ear = after;
      if (ear == stop) {
        filteredOut = false;
        ear = filterPoints(ear, ear);
        if (filteredOut) {
          stop = ear;
          continue;
        }
        if (!cured) {
          // Where Earcut would cure a local intersection the polygon fails (hasLocalIntersection).
          // Otherwise the ring is scanned once more, as Earcut does after its cure pass.
          if (hasLocalIntersection(ear)) { failed = true; }
          stop = ear;
          cured = true;
          continue;
        }
        let diagonal = findSplit(ear);
        if (diagonal.x == NONE) {
          incomplete = true;
          break;
        }
        let other = splitPolygon(diagonal.x, diagonal.y);
        let keptHead = filterPoints(diagonal.x, nextOf(diagonal.x));
        let otherHead = filterPoints(other, nextOf(other));
        pushRing(depth, otherHead);
        depth += 1u;
        ear = keptHead;
        stop = keptHead;
        cured = false;
      }
    }
    if (failed || incomplete || depth == 0u) { break; }
    depth -= 1u;
    ear = popRing(depth);
  }
}

// The first node after 'node' that is not at the point of 'node': a ring can repeat a vertex.
fn nextDistinctOf(node: u32) -> u32 {
  var following = nextOf(node);
  while (following != node && equalNodes(following, node)) {
    spend(1u);
    if (failed) { break; }
    following = nextOf(following);
  }
  return following;
}

fn isLeftOf(first: u32, second: u32) -> bool {
  let a = pointOf(first);
  let b = pointOf(second);
  if (a.x != b.x) { return a.x < b.x; }
  if (a.y != b.y) { return a.y < b.y; }
  // Two holes that meet at their leftmost vertex: Earcut's order by the slope of the edge that
  // leaves it. Both edges point into x >= a.x (straight up at most), so the smaller slope is the
  // clockwise one, and the exact orientation decides it where two f32 quotients can tie. The
  // edge that leaves the vertex is the one to the next different point: with the leftmost vertex
  // repeated, the next node is at the same point and the orientation would be zero both ways.
  return orientOf(a, pointOf(nextDistinctOf(first)), pointOf(nextDistinctOf(second))) > 0;
}

fn sectorContainsSector(m: u32, p: u32) -> bool {
  let pm = pointOf(m);
  return areaSign(pointOf(prevOf(m)), pm, pointOf(prevOf(p))) < 0 &&
    areaSign(pointOf(nextOf(p)), pm, pointOf(nextOf(m))) < 0;
}

// Whether edge start -> end meets the row y = rowY strictly right of where bestStart -> bestEnd
// meets it. Both edges run downward across the row. Two edges of a simple polygon do not cross,
// so one of them lies on one closed side of the other's line, and the exact orientation of its
// ends gives the order of the two crossings; a positive orientation against a downward edge is
// the +x side. An end that lies on the other line at the row itself is the same crossing.
// Edges that do cross (the input is not a simple polygon) take crossingIsRight, the f32 order.
fn crossesRowRightOf(
  start: vec2f, end: vec2f, bestStart: vec2f, bestEnd: vec2f, rowY: f32, crossingIsRight: bool
) -> bool {
  let sideStart = orientOf(bestStart, bestEnd, start);
  let sideEnd = orientOf(bestStart, bestEnd, end);
  if (sideStart >= 0 && sideEnd >= 0) {
    return (sideStart > 0 && sideEnd > 0) || (sideStart > 0 && end.y != rowY) ||
      (sideEnd > 0 && start.y != rowY);
  }
  if (sideStart <= 0 && sideEnd <= 0) { return false; }
  let bestSideStart = orientOf(start, end, bestStart);
  let bestSideEnd = orientOf(start, end, bestEnd);
  if (bestSideStart <= 0 && bestSideEnd <= 0) {
    return (bestSideStart < 0 && bestSideEnd < 0) || (bestSideStart < 0 && bestEnd.y != rowY) ||
      (bestSideEnd < 0 && bestStart.y != rowY);
  }
  if (bestSideStart >= 0 && bestSideEnd >= 0) { return false; }
  return crossingIsRight;
}

// The end of the edge from 'node' that lies on the row y = rowY, or NONE when the row meets the
// edge between its ends. For an edge that is not horizontal.
fn hitEndOnRow(node: u32, rowY: f32) -> u32 {
  if (pointOf(node).y == rowY) { return node; }
  if (pointOf(nextOf(node)).y == rowY) { return nextOf(node); }
  return NONE;
}

// David Eberly's bridge between a hole's leftmost vertex and the outer ring, or NONE. Every
// decision is an exact orientation or a coordinate comparison, so the bridge does not depend on
// f32 rounding of the ray crossing or of the tangents:
// - a hole vertex that lies inside an edge of the outer ring (any direction, horizontal too)
//   becomes a vertex of that edge: a node with the hole vertex's row is inserted between the
//   edge's ends and returned, so the bridge has zero length, like the bridge to a coincident
//   vertex. The edge is the same set of points with or without the node. Earcut bridges such a
//   vertex to an end of the edge, along the edge, which is right only when the collinear filter
//   that follows removes the right node (frc-geometry-01),
// - an edge is hit by the ray toward -x when the hole vertex is on its +x side,
// - of two hit edges the nearer is found by crossesRowRightOf; of two that meet the row at one
//   point, an edge that ends there at a node whose sector does not hold the hole gives way to
//   one that passes through the point or whose node there holds the hole. Two nodes can share
//   the point (an earlier bridge ends at that vertex), and a node can lie inside an edge (the
//   collinear filter removed the other node of an earlier bridge); the walk order chose between
//   them before, and a hole joined to the wrong sector cannot be clipped (frc-geometry-04),
// - a vertex is inside the triangle (hole vertex, crossing, first candidate) by its side of the
//   ray's row, of the hit edge's line (the crossing lies on it) and of the line to the candidate,
// - of two candidates the one with the smaller angle to the ray is on the ray's side of the line
//   from the hole vertex to the other; collinear candidates keep the nearer one, as Earcut does.
// The f32 crossing x is kept only as the order of two hit edges that cross each other.
// As in Earcut, the chosen bridge is not checked afterwards: eliminateHoles splits on it
// directly, and isValidDiagonal guards only the later splits of findSplit.
fn findHoleBridge(hole: u32, outerNode: u32) -> u32 {
  let hp = pointOf(hole);
  var p = outerNode;
  var qx = -3.0e38;
  var m = NONE;
  var hitNode = NONE;
  var hitStart = hp;
  var hitEnd = hp;
  if (equalNodes(hole, p)) { return p; }
  loop {
    spend(1u);
    if (failed) { return NONE; }
    let following = nextOf(p);
    let edgeStart = pointOf(p);
    let edgeEnd = pointOf(following);
    if (equalNodes(hole, following)) { return following; }
    // Inside this edge (its ends were compared above): the box test, then the exact line test.
    if (onSegment(edgeStart, hp, edgeEnd) && orientOf(edgeStart, edgeEnd, hp) == 0) {
      return insertNode(vertexOf(hole), p);
    }
    if (hp.y <= edgeStart.y && hp.y >= edgeEnd.y && edgeEnd.y != edgeStart.y) {
      let side = orientOf(edgeStart, edgeEnd, hp);
      if (side > 0) {
        let x = edgeStart.x + (hp.y - edgeStart.y) * (edgeEnd.x - edgeStart.x) / (edgeEnd.y - edgeStart.y);
        var isNearer = m == NONE;
        if (!isNearer) {
          isNearer = crossesRowRightOf(edgeStart, edgeEnd, hitStart, hitEnd, hp.y, x > qx);
        }
        if (!isNearer) {
          // The same crossing as the hit edge so far, which reaches the row at one of its ends,
          // and the sector of that node does not hold the hole (a sliver between a bridge and an
          // edge, say): this edge is the one the hole faces when the row meets it between its
          // ends, or at a node whose sector holds the hole.
          let hitRowEnd = hitEndOnRow(hitNode, hp.y);
          if (hitRowEnd != NONE) {
            if (!locallyInside(hitRowEnd, hole) && orientOf(edgeStart, edgeEnd, pointOf(hitRowEnd)) == 0) {
              let rowEnd = hitEndOnRow(p, hp.y);
              isNearer = rowEnd == NONE;
              if (!isNearer) { isNearer = locallyInside(rowEnd, hole); }
            }
          }
        }
        if (isNearer) {
          qx = x;
          hitStart = edgeStart;
          hitEnd = edgeEnd;
          hitNode = p;
          m = select(following, p, edgeStart.x < edgeEnd.x);
        }
      }
    }
    p = following;
    if (p == outerNode) { break; }
  }
  if (m == NONE) { return NONE; }
  let pm = pointOf(m);
  let stop = m;
  let isAbove = hp.y < pm.y;
  // Whether m is a candidate that passed the visibility test (Earcut's tanMin < Infinity).
  var hasCandidate = false;
  p = m;
  loop {
    spend(1u);
    if (failed) { return NONE; }
    let pp = pointOf(p);
    if (hp.x >= pp.x && pp.x >= pm.x && hp.x != pp.x) {
      let onTriangleSideOfRay = select(pp.y <= hp.y, pp.y >= hp.y, isAbove);
      let sideOfCandidateLine = select(orientOf(pm, hp, pp), orientOf(hp, pm, pp), isAbove);
      if (onTriangleSideOfRay && sideOfCandidateLine >= 0 && orientOf(hitStart, hitEnd, pp) >= 0) {
        if (locallyInside(p, hole)) {
          var isBetter = !hasCandidate;
          if (hasCandidate) {
            let best = pointOf(m);
            // Positive when p makes the smaller angle with the ray than the current best.
            let turn = select(orientOf(hp, pp, best), orientOf(hp, best, pp), isAbove);
            isBetter = turn > 0 ||
              (turn == 0 && (pp.x > best.x || (pp.x == best.x && sectorContainsSector(m, p))));
          }
          if (isBetter) {
            m = p;
            hasCandidate = true;
          }
        }
      }
    }
    p = nextOf(p);
    if (p == stop) { break; }
  }
  return m;
}

fn getLeftmost(start: u32) -> u32 {
  var leftmost = start;
  var p = start;
  loop {
    spend(1u);
    if (failed) { return leftmost; }
    let a = pointOf(p);
    let b = pointOf(leftmost);
    if (a.x < b.x || (a.x == b.x && a.y < b.y)) { leftmost = p; }
    p = nextOf(p);
    if (p == start) { break; }
  }
  return leftmost;
}

// Links every hole ring (rings ringFirst + 1 ..< ringLast) into the outer ring.
fn eliminateHoles(ringFirst: u32, ringLast: u32, outerIn: u32) -> u32 {
  var outerNode = outerIn;
  var queued = 0u;
  for (var ring = ringFirst + 1u; ring < ringLast; ring++) {
    let start = ringOffsets[ringOffsetsOffset + ring];
    let end = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
    if (ringOffsets[ringOffsetsOffset + ${e.ringCount+1}u + ring] == 0u) { continue; }
    let list = linkedList(start, end - start, false);
    if (list == NONE || failed) { continue; }
    scratch[scratchOffset + 4u * (nodeBase + queued) + 3u] = getLeftmost(list);
    queued += 1u;
  }
  for (var round = 0u; round < queued; round++) {
    // Selection sort by Earcut's compareXYSlope: leftmost, then lowest, then smallest slope.
    var best = NONE;
    var bestSlot = 0u;
    for (var slot = 0u; slot < queued; slot++) {
      spend(1u);
      let candidate = scratch[scratchOffset + 4u * (nodeBase + slot) + 3u];
      if (candidate == NONE) { continue; }
      if (best == NONE || isLeftOf(candidate, best)) {
        best = candidate;
        bestSlot = slot;
      }
    }
    if (best == NONE || failed) { break; }
    scratch[scratchOffset + 4u * (nodeBase + bestSlot) + 3u] = NONE;
    let bridge = findHoleBridge(best, outerNode);
    // Earcut drops a hole it cannot bridge and triangulates the rest, which covers the hole.
    if (bridge == NONE) {
      failed = true;
      break;
    }
    let bridgeReverse = splitPolygon(bridge, best);
    filterPoints(bridgeReverse, nextOf(bridgeReverse));
    outerNode = filterPoints(bridge, nextOf(bridge));
  }
  return filterPoints(outerNode, outerNode);
}

// Scratch nodes before polygon 'polygon': getPolygonTriangulationNodeCount of the layout prefix.
fn baseNodeOf(polygon: u32) -> i32 {
  let ringPrefix = i32(polygonOffsets[polygonOffsetsOffset + polygon]);
  let vertexPrefix = i32(ringOffsets[ringOffsetsOffset + u32(ringPrefix)]);
  let holePrefix = ringPrefix - i32(polygon);
  return 2 * (vertexPrefix + 2 * holePrefix) + 8 * i32(polygon) + holePrefix;
}
fn baseSlotOf(polygon: u32) -> i32 {
  return 3 * i32(polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT + 1u + polygon]);
}

fn triangulatePolygon(polygon: u32) {
  ${e.routingPass===`first`?`routing[routingOffset + polygon] = sizeRouteOf(polygon);
  if (routing[routingOffset + polygon] == 1u) { return; }`:``}
  ${e.routingPass===`fallback`?`if (routing[routingOffset + polygon] != 4u) { return; }`:``}
  failed = false;
  incomplete = false;
  filteredOut = false;
  workDone = 0u;
  trianglesWritten = 0u;
  let slotStart = baseSlotOf(polygon);
  let slotEnd = min(baseSlotOf(polygon + 1u), i32(INDEX_LENGTH));
  let slotCount = max(slotEnd - slotStart, 0);
  triangleBase = u32(max(slotStart, 0));
  triangleSlots = u32(slotCount) / 3u;
  let nodeStart = baseNodeOf(polygon);
  let nodeEnd = min(baseNodeOf(polygon + 1u), i32(SCRATCH_NODE_COUNT));
  nodeBase = u32(max(nodeStart, 0));
  nodeCursor = nodeBase;
  nodeLimit = u32(max(nodeEnd, nodeStart));
  let ringFirst = polygonOffsets[polygonOffsetsOffset + polygon];
  let ringLast = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
  let shellStart = ringOffsets[ringOffsetsOffset + ringFirst];
  var shellEnd = shellStart;
  if (ringLast > ringFirst) {
    shellEnd = min(ringOffsets[ringOffsetsOffset + ringFirst + 1u], VERTEX_COUNT);
  }
  let padVertex = shellStart;
  let liveVertices = polygonOffsets[polygonOffsetsOffset + 2u * (POLYGON_COUNT + 1u) + polygon];
  let liveRings = polygonOffsets[polygonOffsetsOffset + 3u * POLYGON_COUNT + 2u + polygon];
  hashed = USE_Z_HASH && liveVertices > Z_HASH_MINIMUM_VERTICES;
  if (liveRings == 0u) {
    valid[validOffset + polygon] = 1u;
    ${e.writeTriangleCount?`triangleCount[triangleCountOffset + polygon] = 0u;`:``}
    return;
  }
  if (baseSlotOf(polygon + 1u) > i32(INDEX_LENGTH) || triangleSlots == 0u ||
      nodeLimit < nodeBase + ringLast - ringFirst) {
    failed = true;
  } else {
    var outerNode = linkedList(shellStart, shellEnd - shellStart, true);
    if (outerNode == NONE || nextOf(outerNode) == prevOf(outerNode)) {
      failed = true;
    } else {
      if (hashed) {
        var low = vertexAt(shellStart);
        var high = low;
        for (var row = shellStart + 1u; row < shellEnd; row++) {
          low = min(low, vertexAt(row));
          high = max(high, vertexAt(row));
        }
        let extent = max(high.x - low.x, high.y - low.y);
        hashMinimum = low;
        hashScale = select(0.0, 32767.0 / extent, extent > 0.0);
      }
      if (liveRings > 1u) { outerNode = eliminateHoles(ringFirst, ringLast, outerNode); }
      clipEars(outerNode);
    }
  }
  let ok = !failed && !incomplete;
  ${e.routingPass===`first`?`if (!ok && ringLast > ringFirst && shellEnd >= shellStart + 3u) { routing[routingOffset + polygon] = 2u; }`:``}
  let written = select(0u, trianglesWritten, ok);
  // Failed polygons overwrite everything with degenerate triangles; partial output is unusable.
  for (var slot = written; slot < triangleSlots; slot++) {
    let base = indicesOffset + triangleBase + 3u * slot;
    indices[base] = padVertex;
    indices[base + 1u] = padVertex;
    indices[base + 2u] = padVertex;
  }
  valid[validOffset + polygon] = select(0u, 1u, ok);
  ${e.writeTriangleCount?`triangleCount[triangleCountOffset + polygon] = written;`:``}
}
`}export{F as a,H as i,z as n,I as o,V as r,ze as t};