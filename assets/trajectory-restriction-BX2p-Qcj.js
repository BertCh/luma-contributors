import{a as e,i as t,n,r}from"./trajectory-binding-validation-C89oJIAZ.js";import{i,r as a}from"./kernel-C3tfQOn4.js";import{n as o,o as s,r as c}from"./trajectory-common-DFp8lX7h.js";import{s as l,v as u}from"./spatial-context-DwZEd4EZ.js";import{o as d}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as f}from"./exact-geometry-wgsl-y7TUnqOh.js";import{t as p}from"./exact-orientation-wgsl-B_jI-HEc.js";import{t as m}from"./time-words-BFl8KCxq.js";import{E as h,T as g,x as _,y as v}from"./gpu-contributor-utils-BIwLQON5.js";import{t as y}from"./define-run-B-brkCe4.js";import{o as b}from"./pointer-overlay-Qh8qflcA.js";import{t as x}from"./fleet-Cm7PfbI5.js";import{er as S}from"./entry-preview-D70TdSbK.js";function C(e,t){return t?{bindings:[{name:`timeTable`,view:t.view,type:`f32`,access:`read`}],source:`fn rowTime(row: u32) -> f32 { return timeTable[timeTableOffset + 2u * row]; }
fn rowElapsed(second: u32, first: u32) -> f32 { _ = second; return timeTable[timeTableOffset + 2u * first + 1u]; }`,suffix:`-words`}:{bindings:[{name:`times`,view:e.times,type:`f32`,access:`read`}],source:`fn rowTime(row: u32) -> f32 { return times[timesOffset + row]; }
fn rowElapsed(second: u32, first: u32) -> f32 { return times[timesOffset + second] - times[timesOffset + first]; }`,suffix:``}}function w(e,t,n){let r=n.positions.length,i=a(e,`${t}-word-times`,`float32`,Math.max(2*r,1)),o=[d(e,{id:`${t}-word-times`,operation:`GPUTrajectoryRestriction`,variant:`word-times`,invocationCount:n.offsets.length-1,workgroupSize:32,bindings:[{name:`times`,view:n.times,type:`u32`,access:`read`},{name:`timeOrigin`,view:n.timeOrigin,type:`u32`,access:`read`},{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`timeTable`,view:i,type:`f32`,access:`read_write`}],declarations:`const OBSERVATION_COUNT: u32 = ${r}u;
${m}
fn getTimeWords(row: u32) -> vec2<u32> { return vec2<u32>(times[timesOffset + 2u * row], times[timesOffset + 2u * row + 1u]); }`,body:`let start = min(offsets[offsetsOffset + index], OBSERVATION_COUNT); let end = min(offsets[offsetsOffset + index + 1u], OBSERVATION_COUNT);
let origin = vec2<u32>(timeOrigin[timeOriginOffset], timeOrigin[timeOriginOffset + 1u]);
for (var row = start; row < end; row++) {
  timeTable[timeTableOffset + 2u * row] = timeWordsDifferenceToF32(getTimeWords(row), origin);
  timeTable[timeTableOffset + 2u * row + 1u] = select(0.0, timeWordsDifferenceToF32(getTimeWords(row + 1u), getTimeWords(row)), row + 1u < end);
}`})];return{timeTable:{view:i},nodes:o}}function T(e,t,n,r){let i=n.polygonRingVertexCounts??[],a=i.reduce((e,t)=>e+t,0),s=n.intervalsPerSegment??1,c=2+Math.max(a*2,4),l=n.geometryBoundary!==`open`,u=n.timeBoundary!==`open`&&n.timeBoundary!==`right-closed`,m=n.timeBoundary!==`open`&&n.timeBoundary!==`left-closed`,h=0,g=i.flatMap(e=>{let t=h;return h+=e,Array.from({length:e},(n,r)=>t+(r+1)%e)}),_=C(n,r),v=`for (var positionIndex = 1u; positionIndex < boundaryCount; positionIndex++) {
    let value = boundaries[positionIndex]; var cursor = positionIndex;
    while (cursor > 0u && boundaries[cursor - 1u] > value) { boundaries[cursor] = boundaries[cursor - 1u]; cursor--; } boundaries[cursor] = value;
  }
  for (var cursor = 1u; cursor < boundaryCount; cursor++) { if (boundaries[cursor] != boundaries[uniqueCount - 1u]) { boundaries[uniqueCount] = boundaries[cursor]; uniqueCount++; } }
  queryCount = select(2u * uniqueCount - 1u, 1u, isGap || duration <= 0.0);`,y=`{
      let slot = query / 2u; var fraction = boundaries[slot];
      var onBoundary = ${a?`fraction > lower && fraction < upper`:`false`};
      if ((query & 1u) == 1u) {
        let finish = boundaries[slot + 1u]; let betweenCrossings = onBoundary && finish < upper;
        let beginPoint = first + direction * fraction; let finishPoint = first + direction * finish;
        fraction = (fraction + finish) * 0.5; let middlePoint = first + direction * fraction;
        onBoundary = betweenCrossings && all((middlePoint == beginPoint) | (middlePoint == finishPoint));
        ${a?`for (var range = 0u; range < collinearCount; range++) { if (fraction >= collinearRanges[range].x && fraction <= collinearRanges[range].y) { onBoundary = true; } }`:``}
      }
      queryPoint = first + direction * fraction;
      if (fraction <= 0.0) { queryPoint = first; }
      if (fraction >= 1.0) { queryPoint = second; }
      queryOnBoundary = onBoundary;
    }`,b=[d(e,{id:`${t}-intervals`,operation:`GPUTrajectoryRestriction`,variant:`${a?`polygon-intervals`:`rectangle-intervals`}${_.suffix}`,invocationCount:n.offsets.length-1,workgroupSize:32,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},..._.bindings,{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`intervals`,view:n.output.intervals,type:`f32`,access:`read_write`},{name:`clipped`,view:n.output.positions,type:`f32`,access:`read_write`},{name:`status`,view:n.output.status,type:`u32`,access:`read_write`},{name:`relationships`,view:n.output.relationships,type:`u32`,access:`read_write`}],declarations:`const OBSERVATION_COUNT = ${n.positions.length}u;
const SLOT_COUNT = ${s}u;
${_.source}
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn cross(first: vec2<f32>, second: vec2<f32>) -> f32 { return first.x * second.y - first.y * second.x; }
fn position(row: u32) -> vec2<f32> { return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]); }
fn vertex(row: u32) -> vec2<f32> { return vec2<f32>(parameters[parametersOffset + 7u + 2u * row], parameters[parametersOffset + 8u + 2u * row]); }
${a?`const NEXT = array<u32, ${a}>(${g.map(e=>`${e}u`).join(`,`)});\n${p}\n${f}`:``}
const CLOSED_GEOMETRY: bool = ${l};
${a?``:`fn inside(point: vec2<f32>) -> bool {
let minimum = vec2<f32>(parameters[parametersOffset + 2u], parameters[parametersOffset + 3u]); let maximum = vec2<f32>(parameters[parametersOffset + 4u], parameters[parametersOffset + 5u]);
return all(point ${l?`>=`:`>`} minimum) && all(point ${l?`<=`:`<`} maximum);
}`}
fn timeIncluded(time: f32) -> bool { return time ${u?`>=`:`>`} parameters[parametersOffset] && time ${m?`<=`:`<`} parameters[parametersOffset + 1u]; }
fn writeInterval(row: u32, slot: u32, firstTime: f32, duration: f32, first: vec2<f32>, direction: vec2<f32>, lower: f32, upper: f32, includeStart: bool, includeEnd: bool) {
  if (slot >= SLOT_COUNT) { status[statusOffset + row * SLOT_COUNT] |= 256u; return; }
  let outputRow = row * SLOT_COUNT + slot;
  intervals[intervalsOffset + 2u * outputRow] = firstTime + duration * lower; intervals[intervalsOffset + 2u * outputRow + 1u] = firstTime + duration * upper;
  let begin = first + direction * lower; let end = first + direction * upper;
  clipped[clippedOffset + 4u * outputRow] = begin.x; clipped[clippedOffset + 4u * outputRow + 1u] = begin.y; clipped[clippedOffset + 4u * outputRow + 2u] = end.x; clipped[clippedOffset + 4u * outputRow + 3u] = end.y;
  status[statusOffset + outputRow] |= select(1u, 2u, duration == 0.0 || lower == upper) | select(64u, 0u, includeStart) | select(128u, 0u, includeEnd);
}`,body:`let start = min(offsets[offsetsOffset + index], OBSERVATION_COUNT); let end = min(offsets[offsetsOffset + index + 1u], OBSERVATION_COUNT);
let minimumTime = parameters[parametersOffset]; let maximumTime = parameters[parametersOffset + 1u]; let maximumGap = ${o(`parameters[parametersOffset + 6u]`)};
var validParameters = finite(minimumTime) && finite(maximumTime) && minimumTime <= maximumTime && maximumGap > 0.0;
for (var parameter = 2u; parameter < ${7+a*2}u; parameter++) { if (parameter != 6u) { validParameters = validParameters && finite(parameters[parametersOffset + parameter]); } }
${a?``:`validParameters = validParameters && parameters[parametersOffset + 2u] <= parameters[parametersOffset + 4u] && parameters[parametersOffset + 3u] <= parameters[parametersOffset + 5u];`}
var nanBits = 0x7fc00000u; let missing = bitcast<f32>(nanBits);
var anyInside = false; var allInside = true; var supportCount = 0u; var unknownCount = 0u;
for (var row = start; row < end; row++) {
  for (var slot = 0u; slot < SLOT_COUNT; slot++) {
    let outputRow = row * SLOT_COUNT + slot; status[statusOffset + outputRow] = 0u;
    for (var component = 0u; component < 2u; component++) { intervals[intervalsOffset + 2u * outputRow + component] = missing; }
    for (var component = 0u; component < 4u; component++) { clipped[clippedOffset + 4u * outputRow + component] = missing; }
  }
  let next = min(row + 1u, end - 1u); let first = position(row); let second = position(next); let direction = second - first;
  let firstTime = rowTime(row); let secondTime = rowTime(next); let duration = rowElapsed(next, row);
  if (!validParameters || !finite(firstTime) || !finite(secondTime) || duration < 0.0 || !finite(first.x) || !finite(first.y) || !finite(second.x) || !finite(second.y) || (duration == 0.0 && any(direction != vec2<f32>(0.0)))) { status[statusOffset + row * SLOT_COUNT] = 4u; unknownCount++; continue; }
  if (secondTime < minimumTime || firstTime > maximumTime) { continue; }
  let isGap = duration > maximumGap;
  var lower = 0.0; var upper = select(1.0, 0.0, duration == 0.0);
  var boundaries: array<f32, ${c}>; var uniqueCount = 1u; var boundaryCount = 1u;
  ${a?`var collinearRanges: array<vec2<f32>, ${a}>; var collinearCount = 0u;`:``}
  if (isGap) {
    status[statusOffset + row * SLOT_COUNT] = 8u;
    if (max(minimumTime, firstTime) < min(maximumTime, secondTime)) { unknownCount++; }
    if (!timeIncluded(firstTime)) { continue; }
    supportCount++; status[statusOffset + row * SLOT_COUNT] |= 32u;
    // Only the fix itself is classified.
    upper = 0.0; boundaries[0] = 0.0;
  } else {
    if (duration > 0.0) { lower = max(0.0, (minimumTime - firstTime) / duration); upper = min(1.0, (maximumTime - firstTime) / duration); }
    if (lower > upper || ((lower == upper || duration == 0.0) && !timeIncluded(firstTime + duration * lower))) { continue; }
    supportCount++; status[statusOffset + row * SLOT_COUNT] = 32u;
    boundaries[0] = lower; boundaries[1] = upper; boundaryCount = 2u;
    ${a?``:`for (var axis = 0u; axis < 2u; axis++) { if (direction[axis] != 0.0) { for (var side = 0u; side < 2u; side++) { let fraction = (parameters[parametersOffset + 2u + axis + 2u * side] - first[axis]) / direction[axis]; if (fraction > lower && fraction < upper) { boundaries[boundaryCount] = fraction; boundaryCount++; } } } }`}
  }
  // Classify every boundary point (even query) and the midpoint of every piece between two of
  // them (odd query) in one place. A fix is tested as it was read. A polygon crossing is on the
  // boundary because it was constructed there, not because its rounded coordinates say so; so is
  // a piece that runs along an edge, and a piece between two crossings that has no float32 point
  // of its own. Everything else is the exact test of the computed point.
  var containment: array<u32, ${2*c}>; var queryCount = 1u;
  var queryPoint = first; var queryOnBoundary = false;
  ${a?`// One loop holds the only call of exactOrientSigns (the orientation code is large and is
  // inlined at every call). Jobs 0 .. V - 1 decide, edge by edge, whether the segment meets the
  // edge, from the exact sides of the two fixes against the edge and of the edge's two ends
  // against the segment; float32 only places a crossing that exists. The following jobs are the
  // exact even-odd test of each query point against each edge, with the rule of
  // exactProbeCrossing (the shared predicate GPUZoneEvents uses): on the boundary only when the
  // exact orientation is zero inside the edge's box.
  var job = select(0u, ${a}u, isGap || all(direction == vec2<f32>(0.0)));
  var jobCount = ${a+1}u; var winding = 0; var boundaryHit = false;
  loop {
    if (job >= jobCount) { break; }
    if (job == ${a}u) {
      ${v}
      jobCount = ${a}u + queryCount * ${a}u;
    }
    let edge = job % ${a}u; let isProbe = job >= ${a}u;
    let edgeStart = vertex(edge); let edgeEnd = vertex(NEXT[edge]);
    var signCount = 4u; var straddles = false; var inBox = false;
    var starts = array<vec2<f32>, 4>(edgeStart, edgeStart, first, first);
    var ends = array<vec2<f32>, 4>(edgeEnd, edgeEnd, second, second);
    var points = array<vec2<f32>, 4>(first, second, edgeStart, edgeEnd);
    if (isProbe) {
      let query = job / ${a}u - 1u;
      if (edge == 0u) {
        ${y}
        winding = 0; boundaryHit = false;
      }
      signCount = 0u;
      if (!queryOnBoundary) {
        straddles = (edgeStart.y >= queryPoint.y) != (edgeEnd.y >= queryPoint.y);
        inBox = all(queryPoint >= min(edgeStart, edgeEnd)) && all(queryPoint <= max(edgeStart, edgeEnd));
        if (straddles || inBox) {
          signCount = 1u;
          starts = array<vec2<f32>, 4>(edgeStart, edgeStart, edgeStart, edgeStart);
          ends = array<vec2<f32>, 4>(edgeEnd, edgeEnd, edgeEnd, edgeEnd);
          points = array<vec2<f32>, 4>(queryPoint, queryPoint, queryPoint, queryPoint);
        }
      }
    } else if (all(edgeStart == edgeEnd)) {
      // A repeated vertex is not an edge; its neighbours carry the crossing.
      signCount = 0u;
    }
    let exact = exactOrientSigns(signCount, starts, ends, points);
    if (!isProbe) {
      if (signCount == 4u) {
        // x, y: the two fixes against the edge. z, w: the edge's two ends against the segment.
        let sides = exact.signs; let lengthSquared = dot(direction, direction);
        if (sides.x == 0 && sides.y == 0) {
          // The segment runs along this edge's line: between the edge's two ends it is on the boundary.
          if (lengthSquared > 0.0) {
            let startFraction = dot(edgeStart - first, direction) / lengthSquared; let endFraction = dot(edgeEnd - first, direction) / lengthSquared;
            if (startFraction > lower && startFraction < upper) { boundaries[boundaryCount] = startFraction; boundaryCount++; }
            if (endFraction > lower && endFraction < upper) { boundaries[boundaryCount] = endFraction; boundaryCount++; }
            collinearRanges[collinearCount] = vec2<f32>(min(startFraction, endFraction), max(startFraction, endFraction)); collinearCount++;
          }
        } else if (sides.x * sides.y <= 0 && sides.z * sides.w <= 0) {
          // The segment and the edge share exactly one point. A fix on the edge is the query at
          // 0 or 1; a vertex on the segment is placed by its own projection, so the two edges
          // that share it give the same parameter; otherwise the float32 quotient places it.
          var fraction = 0.5;
          if (sides.x == 0) { fraction = 0.0; }
          else if (sides.y == 0) { fraction = 1.0; }
          else if (sides.z == 0 && lengthSquared > 0.0) { fraction = dot(edgeStart - first, direction) / lengthSquared; }
          else if (sides.w == 0 && lengthSquared > 0.0) { fraction = dot(edgeEnd - first, direction) / lengthSquared; }
          else {
            let edgeDirection = edgeEnd - edgeStart; let denominator = cross(direction, edgeDirection);
            if (denominator != 0.0) { fraction = cross(edgeStart - first, edgeDirection) / denominator; }
          }
          fraction = clamp(fraction, 0.0, 1.0);
          if (fraction > lower && fraction < upper) { boundaries[boundaryCount] = fraction; boundaryCount++; }
        }
      }
    } else {
      if (signCount == 1u) {
        if (inBox && exact.signs.x == 0) { boundaryHit = true; }
        let side = sosPointSide(exact.signs.x, edgeStart, edgeEnd);
        if (straddles) {
          let upward = edgeEnd.y > edgeStart.y;
          if (upward && side > 0) { winding += 1; }
          if (!upward && side < 0) { winding -= 1; }
        }
      }
      if (edge == ${a-1}u) {
        var contained = CLOSED_GEOMETRY;
        if (!queryOnBoundary && !boundaryHit) { contained = (winding & 1) != 0; }
        containment[job / ${a}u - 1u] = select(0u, 1u, contained);
      }
    }
    job++;
  }`:`${v}
  for (var query = 0u; query < queryCount; query++) {
    ${y}
    containment[query] = select(0u, 1u, inside(queryPoint));
  }`}
  if (isGap) {
    let contained = containment[0] != 0u; anyInside = anyInside || contained; allInside = allInside && contained;
    if (contained) { writeInterval(row, 0u, firstTime, 0.0, first, vec2<f32>(0.0), 0.0, 0.0, true, true); status[statusOffset + row * SLOT_COUNT] |= 16u; }
    continue;
  }
  var written = 0u; var fullyInside = true; var previousInterior = false;
  for (var cursor = 0u; cursor < uniqueCount; cursor++) {
    let begin = boundaries[cursor]; let beginTime = firstTime + duration * begin;
    let pointSupported = timeIncluded(beginTime); let pointInside = containment[2u * cursor] != 0u;
    let hasNext = cursor + 1u < uniqueCount && duration > 0.0;
    var nextInterior = false; var finish = begin;
    if (hasNext) { finish = boundaries[cursor + 1u]; nextInterior = containment[2u * cursor + 1u] != 0u; fullyInside = fullyInside && nextInterior; }
    if (pointSupported) { fullyInside = fullyInside && pointInside; }
    if (nextInterior) {
      let endIncluded = timeIncluded(firstTime + duration * finish) && containment[2u * cursor + 2u] != 0u;
      writeInterval(row, written, firstTime, duration, first, direction, begin, finish, pointSupported && pointInside, endIncluded); written++; anyInside = true;
    } else if (pointSupported && pointInside && !previousInterior) {
      writeInterval(row, written, firstTime, duration, first, direction, begin, begin, true, true); written++; anyInside = true;
    }
    previousInterior = nextInterior;
  }
  allInside = allInside && fullyInside;
  if (fullyInside) { status[statusOffset + row * SLOT_COUNT] |= 16u; }
}
var ever = select(0u, 1u, anyInside); var always = select(0u, 1u, allInside);
if (supportCount == 0u || unknownCount > 0u) { if (!anyInside) { ever = 0xffffffffu; } if (allInside) { always = 0xffffffffu; } }
relationships[relationshipsOffset + 4u * index] = ever; relationships[relationshipsOffset + 4u * index + 1u] = always; relationships[relationshipsOffset + 4u * index + 2u] = supportCount; relationships[relationshipsOffset + 4u * index + 3u] = unknownCount;`})],x=n.output;return x.compactIntervals&&x.compactPositions&&x.compactSources&&x.compactStatus&&x.compactSummary&&b.push(d(e,{id:`${t}-compact`,operation:`GPUTrajectoryRestriction`,variant:n.mergeIntervals?`merge-intervals`:`compact-intervals`,invocationCount:1,workgroupSize:2,bindings:[{name:`intervals`,view:x.intervals,type:`f32`,access:`read`},{name:`status`,view:x.status,type:`u32`,access:`read`},{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`result`,view:x.compactIntervals,type:`f32`,access:`read_write`},{name:`sources`,view:x.compactSources,type:`u32`,access:`read_write`},{name:`resultStatus`,view:x.compactStatus,type:`u32`,access:`read_write`},{name:`summary`,view:x.compactSummary,type:`u32`,access:`read_write`}],body:`var nanBits = 0x7fc00000u; let missing = bitcast<f32>(nanBits);
for (var row = 0u; row < ${x.compactIntervals.length}u; row++) {
  result[resultOffset + 2u * row] = missing; result[resultOffset + 2u * row + 1u] = missing;
  sources[sourcesOffset + 2u * row] = 0xffffffffu; sources[sourcesOffset + 2u * row + 1u] = 0xffffffffu; resultStatus[resultStatusOffset + row] = 0u;
}
var required = 0u; var upstreamOverflow = 0u; var track = 0u; var previousTrack = 0xffffffffu; var previousBegin = 0.0; var previousEnd = 0.0; var previousFlags = 0u;
for (var row = 0u; row < ${x.intervals.length}u; row++) {
  let flags = status[statusOffset + row]; if ((flags & 256u) != 0u) { upstreamOverflow++; }
  if ((flags & 3u) == 0u) { continue; }
  let sourceRow = row / ${s}u;
  while (track + 1u < ${n.offsets.length-1}u && sourceRow >= offsets[offsetsOffset + track + 1u]) { track++; }
  let begin = intervals[intervalsOffset + 2u * row]; let finish = intervals[intervalsOffset + 2u * row + 1u];
  let merge = ${n.mergeIntervals?`required > 0u && track == previousTrack && (begin < previousEnd || (begin == previousEnd && ((previousFlags & 128u) == 0u || (flags & 64u) == 0u)))`:`false`};
  if (merge) {
    let extendsInterval = finish >= previousEnd;
    if (extendsInterval) { previousEnd = finish; previousFlags = (previousFlags & ~128u) | (flags & 128u); }
    previousFlags = (previousFlags & ~3u) | select(1u,2u,previousBegin == previousEnd);
    if (required <= ${x.compactIntervals.length}u) {
      let previous = required - 1u; result[resultOffset + 2u * previous + 1u] = previousEnd; resultStatus[resultStatusOffset + previous] = previousFlags;
      if (extendsInterval) { sources[sourcesOffset + 2u * previous + 1u] = row; }
    }
    continue;
  }
  previousTrack = track; previousBegin = begin; previousEnd = finish; previousFlags = flags;
  if (required < ${x.compactIntervals.length}u) {
    result[resultOffset + 2u * required] = begin; result[resultOffset + 2u * required + 1u] = finish;
    sources[sourcesOffset + 2u * required] = row; sources[sourcesOffset + 2u * required + 1u] = row; resultStatus[resultStatusOffset + required] = flags;
  }
  required++;
}
summary[summaryOffset] = required; summary[summaryOffset + 1u] = min(required, ${x.compactIntervals.length}u); summary[summaryOffset + 2u] = required - min(required, ${x.compactIntervals.length}u) + upstreamOverflow;`}),d(e,{id:`${t}-compact-positions`,operation:`GPUTrajectoryRestriction`,variant:`compact-positions-and-provenance`,invocationCount:x.compactIntervals.length,bindings:[{name:`positions`,view:x.positions,type:`f32`,access:`read`},{name:`sources`,view:x.compactSources,type:`u32`,access:`read_write`},{name:`result`,view:x.compactPositions,type:`f32`,access:`read_write`}],body:`let first = sources[sourcesOffset + 2u * index]; let last = sources[sourcesOffset + 2u * index + 1u];
if (first == 0xffffffffu) { var bits = 0x7fc00000u; for (var component = 0u; component < 4u; component++) { result[resultOffset + 4u * index + component] = bitcast<f32>(bits); } return; }
result[resultOffset + 4u * index] = positions[positionsOffset + 4u * first]; result[resultOffset + 4u * index + 1u] = positions[positionsOffset + 4u * first + 1u];
result[resultOffset + 4u * index + 2u] = positions[positionsOffset + 4u * last + 2u]; result[resultOffset + 4u * index + 3u] = positions[positionsOffset + 4u * last + 3u];
sources[sourcesOffset + 2u * index] = first / ${s}u; sources[sourcesOffset + 2u * index + 1u] = last / ${s}u;`})),b}function E(e,t,n,r,i){let s=n.intervalsPerPair;if(!r.capacity)return[];let c=n.timeBoundary!==`open`&&n.timeBoundary!==`right-closed`,l=n.timeBoundary!==`open`&&n.timeBoundary!==`left-closed`,u=n.geometryBoundary!==`open`,f=C(n,i),p=n.offsets.length-1,m=a(e,`${t}-pair-table`,`uint32`,4*r.capacity);return[d(e,{id:`${t}-pairwise-resolve`,operation:`GPUTrajectoryRestriction`,variant:`pairwise-resolve-pairs${r.staticPairs?`-static`:``}${r.count?`-counted`:``}`,invocationCount:r.capacity,bindings:[{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},...r.staticPairs?[]:[{name:`pairFirst`,view:r.first,type:`u32`,access:`read`},{name:`pairSecond`,view:r.second,type:`u32`,access:`read`},...r.count?[{name:`pairCount`,view:r.count,type:`u32`,access:`read`}]:[]],{name:`pairTable`,view:m,type:`u32`,access:`read_write`}],declarations:r.staticPairs?`const PAIRS = array<vec2<u32>, ${r.staticPairs.length}>(${r.staticPairs.map(e=>`vec2<u32>(${e[0]}u,${e[1]}u)`).join(`,`)});`:``,body:`${r.staticPairs?`let isLive = true; let tracks = PAIRS[index];`:`let isLive = ${r.count?`index < pairCount[pairCountOffset]`:`true`}; let tracks = vec2<u32>(pairFirst[pairFirstOffset + index], pairSecond[pairSecondOffset + index]);`}
var row = vec4<u32>(0xffffffffu);
if (isLive) {
  row = vec4<u32>(0xfffffffeu);
  if (tracks.x < ${p}u && tracks.y < ${p}u) {
    row = vec4<u32>(min(offsets[offsetsOffset + tracks.x], ${n.positions.length}u), min(offsets[offsetsOffset + tracks.x + 1u], ${n.positions.length}u), min(offsets[offsetsOffset + tracks.y], ${n.positions.length}u), min(offsets[offsetsOffset + tracks.y + 1u], ${n.positions.length}u));
  }
}
for (var component = 0u; component < 4u; component++) { pairTable[pairTableOffset + 4u * index + component] = row[component]; }`}),d(e,{id:`${t}-pairwise-clear`,operation:`GPUTrajectoryRestriction`,variant:`pairwise-clear-slots`,invocationCount:r.capacity*s,workgroupSize:64,bindings:[{name:`intervals`,view:n.output.intervals,type:`f32`,access:`read_write`},{name:`clipped`,view:n.output.positions,type:`f32`,access:`read_write`},{name:`status`,view:n.output.status,type:`u32`,access:`read_write`}],body:`var nanBits = 0x7fc00000u; let missing = bitcast<f32>(nanBits);
status[statusOffset + index] = 0u;
for (var component = 0u; component < 2u; component++) { intervals[intervalsOffset + 2u * index + component] = missing; }
for (var component = 0u; component < 4u; component++) { clipped[clippedOffset + 4u * index + component] = missing; }`}),d(e,{id:`${t}-pairwise-merge`,operation:`GPUTrajectoryRestriction`,variant:`pairwise-time-merge${f.suffix}`,invocationCount:r.capacity,workgroupSize:32,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},...f.bindings,{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`pairTable`,view:m,type:`u32`,access:`read`},{name:`intervals`,view:n.output.intervals,type:`f32`,access:`read_write`},{name:`clipped`,view:n.output.positions,type:`f32`,access:`read_write`},{name:`status`,view:n.output.status,type:`u32`,access:`read_write`},{name:`relationships`,view:n.output.relationships,type:`u32`,access:`read_write`}],declarations:`fn pairActive(pair: u32) -> bool { return pairTable[pairTableOffset + 4u * pair] != 0xffffffffu; }
fn pairInRange(pair: u32) -> bool { return pairTable[pairTableOffset + 4u * pair] != 0xfffffffeu; }
${f.source}
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn position(row: u32) -> vec2<f32> { return vec2<f32>(positions[positionsOffset + 2u * row],positions[positionsOffset + 2u * row + 1u]); }
fn timeIncluded(time: f32) -> bool { return time ${c?`>=`:`>`} parameters[parametersOffset] && time ${l?`<=`:`<`} parameters[parametersOffset + 1u]; }
// One segment of a track: from \`row\` to the next row, or a singleton at the last row and before a gap.
struct Segment {
  time: f32, finish: f32, rawFinish: f32, duration: f32, position: vec2<f32>, movement: vec2<f32>, gap: bool, invalid: bool
}
fn loadSegment(row: u32, end: u32, maximumGap: f32) -> Segment {
  let next = min(row + 1u, end - 1u);
  let time = rowTime(row); let rawFinish = rowTime(next); let rawDuration = rowElapsed(next, row);
  let gap = rawDuration > maximumGap;
  let finish = select(rawFinish, time, gap); let duration = select(rawDuration, 0.0, gap);
  let start = position(row); let movement = select(position(next) - start, vec2<f32>(0.0), gap);
  let invalid = !finite(time) || !finite(finish) || duration < 0.0 || !finite(start.x) || !finite(start.y) || !finite(movement.x) || !finite(movement.y) || (duration == 0.0 && any(movement != vec2<f32>(0.0)));
  return Segment(time, finish, rawFinish, duration, start, movement, gap, invalid);
}`,body:`let base = index * ${s}u;
var anyInside = false; var allInside = true; var support = 0u; var unknown = 0u; var written = 0u;
let minimumTime = parameters[parametersOffset]; let maximumTime = parameters[parametersOffset + 1u]; let distanceLimit = parameters[parametersOffset + 2u]; let maximumGap = ${o(`parameters[parametersOffset + 6u]`)};
let validParameters = finite(minimumTime) && finite(maximumTime) && minimumTime <= maximumTime && finite(distanceLimit) && distanceLimit >= 0.0 && maximumGap > 0.0;
if (!pairActive(index)) {
  relationships[relationshipsOffset + 4u * index] = 0xffffffffu; relationships[relationshipsOffset + 4u * index + 1u] = 0xffffffffu; relationships[relationshipsOffset + 4u * index + 2u] = 0u; relationships[relationshipsOffset + 4u * index + 3u] = 0u;
  return;
}
if (!pairInRange(index) || !validParameters) {
  status[statusOffset + base] |= 4u;
  relationships[relationshipsOffset + 4u * index] = 0xffffffffu; relationships[relationshipsOffset + 4u * index + 1u] = 0xffffffffu; relationships[relationshipsOffset + 4u * index + 2u] = 0u; relationships[relationshipsOffset + 4u * index + 3u] = 1u;
  return;
}
let startA = pairTable[pairTableOffset + 4u * index]; let endA = pairTable[pairTableOffset + 4u * index + 1u];
let startB = pairTable[pairTableOffset + 4u * index + 2u]; let endB = pairTable[pairTableOffset + 4u * index + 3u];
// Invalid segments never overlap anything; count each once.
for (var rowA = startA; rowA < endA; rowA++) { if (loadSegment(rowA, endA, maximumGap).invalid) { unknown++; } }
for (var rowB = startB; rowB < endB; rowB++) { if (loadSegment(rowB, endB, maximumGap).invalid) { unknown++; } }
if (unknown > 0u) { status[statusOffset + base] |= 4u; }
// Two-pointer merge: \`firstB\` is the first segment of B that does not end before the current segment of A starts.
var firstB = startB;
for (var rowA = startA; rowA < endA; rowA++) {
  let segmentA = loadSegment(rowA, endA, maximumGap);
  if (segmentA.invalid) { continue; }
  if (segmentA.time > maximumTime) { break; }
  if (segmentA.rawFinish < minimumTime) { continue; }
  for (; firstB < endB; firstB++) {
    let segmentB = loadSegment(firstB, endB, maximumGap);
    if (!segmentB.invalid && segmentB.rawFinish >= segmentA.time) { break; }
  }
  for (var rowB = firstB; rowB < endB; rowB++) {
    let segmentB = loadSegment(rowB, endB, maximumGap);
    if (segmentB.invalid) { continue; }
    if (segmentB.time > segmentA.rawFinish || segmentB.time > maximumTime) { break; }
    let timeA = segmentA.time; let timeB = segmentB.time;
    let finishA = segmentA.finish; let finishB = segmentB.finish;
    let durationA = segmentA.duration; let durationB = segmentB.duration;
    let positionA = segmentA.position; let movementA = segmentA.movement; let movementB = segmentB.movement;
    let lower = max(minimumTime, max(timeA, timeB)); let upper = min(maximumTime, min(finishA, finishB));
    if ((segmentA.gap || segmentB.gap) && lower < min(maximumTime, min(segmentA.rawFinish, segmentB.rawFinish))) { unknown++; status[statusOffset + base] |= 8u; }
    if (lower > upper || (lower == upper && !timeIncluded(lower))) { continue; }
    support++;
    var velocityA = vec2<f32>(0.0); var velocityB = vec2<f32>(0.0); if (durationA > 0.0) { velocityA = movementA / durationA; } if (durationB > 0.0) { velocityB = movementB / durationB; }
    let firstA = positionA + velocityA * (lower - timeA); let firstPointB = segmentB.position + velocityB * (lower - timeB); let relative = firstA - firstPointB; let velocity = velocityA - velocityB;
    let quadratic = dot(velocity, velocity); let projection = dot(relative, velocity); let limitSquared = distanceLimit * distanceLimit;
    if (!finite(quadratic) || !finite(projection) || !finite(limitSquared)) { unknown++; status[statusOffset + base] |= 4u; continue; }
    var begin = 0.0; var finish = upper - lower; var matches = true;
    if (quadratic == 0.0) {
      let distanceSquared = dot(relative, relative);
      if (!finite(distanceSquared)) { unknown++; status[statusOffset + base] |= 4u; continue; }
      matches = distanceSquared ${u?`<=`:`<`} limitSquared;
    } else {
      // Evaluate distance at closest approach before subtracting the radius. The conventional
      // b*b-4*a*c discriminant loses narrow encounters when absolute separation is large.
      let center = -projection / quadratic; let closest = relative + velocity * center;
      let remainingSquared = limitSquared - dot(closest, closest);
      if (!finite(center) || !finite(remainingSquared)) { unknown++; status[statusOffset + base] |= 4u; continue; }
      if (remainingSquared < 0.0) { matches = false; }
      else {
        let radius = sqrt(remainingSquared / quadratic);
        if (!finite(radius)) { unknown++; status[statusOffset + base] |= 4u; continue; }
        begin = max(begin, center - radius); finish = min(finish, center + radius); matches = begin <= finish;
      }
    }
    let beginTime = lower + begin; let finishTime = lower + finish;
    let beginInside = dot(relative + velocity * begin, relative + velocity * begin) ${u?`<=`:`<`} distanceLimit * distanceLimit;
    let finishInside = dot(relative + velocity * finish, relative + velocity * finish) ${u?`<=`:`<`} distanceLimit * distanceLimit;
    let includeStart = timeIncluded(beginTime) && beginInside; let includeEnd = timeIncluded(finishTime) && finishInside;
    if (begin == finish && !(includeStart && includeEnd)) { matches = false; }
    let fullyInside = matches && begin == 0.0 && finish == upper - lower && (!timeIncluded(lower) || includeStart) && (!timeIncluded(upper) || includeEnd);
    allInside = allInside && fullyInside;
    if (!matches) { continue; } anyInside = true;
    if (written >= ${s}u) { status[statusOffset + base] |= 256u; continue; }
    let outputRow = base + written; intervals[intervalsOffset + 2u * outputRow] = beginTime; intervals[intervalsOffset + 2u * outputRow + 1u] = finishTime;
    let beginPosition = firstA + velocityA * begin; let finishPosition = firstA + velocityA * finish;
    clipped[clippedOffset + 4u * outputRow] = beginPosition.x; clipped[clippedOffset + 4u * outputRow + 1u] = beginPosition.y; clipped[clippedOffset + 4u * outputRow + 2u] = finishPosition.x; clipped[clippedOffset + 4u * outputRow + 3u] = finishPosition.y;
    status[statusOffset + outputRow] |= 32u | select(1u, 2u, begin == finish) | select(0u,16u,fullyInside) | select(64u,0u,includeStart) | select(128u,0u,includeEnd); written++;
  }
}
var ever = select(0u,1u,anyInside); var always = select(0u,1u,allInside);
if (support == 0u || unknown > 0u) { if (!anyInside) { ever = 0xffffffffu; } if (allInside) { always = 0xffffffffu; } }
relationships[relationshipsOffset + 4u * index] = ever; relationships[relationshipsOffset + 4u * index + 1u] = always; relationships[relationshipsOffset + 4u * index + 2u] = support; relationships[relationshipsOffset + 4u * index + 3u] = unknown;`})]}var D=class{id;props;spatialContext;constructor(t){this.id=t.id??`trajectory-restriction`,this.spatialContext=u(this.id,t.spatialContext,t.pairs||t.pairwiseTracks?Object.freeze({...l,measure:`distance`}):l);let{times:n,offsets:a}=c(this.id,t);s(this.id,n,t.temporalContext),this.props={...t,times:n,offsets:a};let{props:o}=this;for(let e of[o.positions,o.output.intervals])h(e,[`float32x2`],this.id);h(o.times,[`float32`,`uint32x2`],this.id),h(o.parameters,[`float32`],this.id),g(this.id,o.times.format===`uint32x2`==!!o.timeOrigin,`word times need timeOrigin and float times must not have one`),o.timeOrigin&&(h(o.timeOrigin,[`uint32`],this.id),g(this.id,o.timeOrigin.length>=2,`timeOrigin must hold at least two rows`));for(let e of[o.offsets,o.output.status])h(e,[`uint32`],this.id);h(o.output.positions,[`float32x4`],this.id),h(o.output.relationships,[`uint32x4`],this.id);let d=o.intervalsPerSegment??1;g(this.id,Number.isInteger(d)&&d>0&&d<=256,`intervalsPerSegment must be an integer in [1, 256]`);let f=o.polygonRingVertexCounts??[];g(this.id,f.every(e=>Number.isInteger(e)&&e>=3),`polygon rings need at least 3 vertices each`);let p=f.reduce((e,t)=>e+t,0);if(g(this.id,p<=128,`polygon vertices must total at most 128`),o.output.compactIntervals){let e=o.output.compactIntervals.length;g(this.id,!!(o.output.compactPositions&&o.output.compactSources&&o.output.compactStatus&&o.output.compactSummary),`compact outputs need compactPositions, compactSources, compactStatus and compactSummary`),h(o.output.compactIntervals,[`float32x2`],this.id),h(o.output.compactPositions,[`float32x4`],this.id),h(o.output.compactSources,[`uint32x2`],this.id),h(o.output.compactStatus,[`uint32`],this.id),h(o.output.compactSummary,[`uint32`],this.id),g(this.id,o.output.compactPositions.length===e&&o.output.compactSources.length===e&&o.output.compactStatus.length===e&&o.output.compactSummary.length===3,`compact outputs need the length of compactIntervals and a 3-row compactSummary`)}else g(this.id,!o.output.compactPositions&&!o.output.compactSources&&!o.output.compactStatus&&!o.output.compactSummary&&!o.mergeIntervals,`compact outputs are absent unless compactIntervals is given, and mergeIntervals needs them`);if(g(this.id,!(o.pairwiseTracks&&o.pairs),`takes pairwiseTracks or pairs`),o.pairs){for(let e of[o.pairs.first,o.pairs.second,o.pairs.count])e&&h(e,[`uint32`],this.id);g(this.id,o.pairs.first.length===o.pairs.second.length,`pairs.first and pairs.second must have the same length`),g(this.id,!o.pairs.count||o.pairs.count.length>=1,`pairs.count must hold at least one row`)}o.pairwiseTracks||o.pairs?(g(this.id,!o.polygonRingVertexCounts&&!o.output.compactIntervals&&o.intervalsPerSegment===void 0,`pairwise mode excludes polygons, compact outputs and intervalsPerSegment`),g(this.id,Number.isInteger(o.intervalsPerPair)&&o.intervalsPerPair>0&&o.intervalsPerPair<=65536,`intervalsPerPair must be an integer in [1, 65536]`),g(this.id,(o.pairwiseTracks??[]).length<=4096&&(o.pairwiseTracks??[]).every(e=>e.every(e=>Number.isInteger(e)&&e>=0&&e<o.offsets.length-1)),`pairwiseTracks must hold at most 4096 pairs of track indices below the track count`)):g(this.id,o.intervalsPerPair===void 0,`intervalsPerPair needs pairwise mode`);let m=o.pairs?.first.length??o.pairwiseTracks?.length,_=m===void 0?o.positions.length*d:m*o.intervalsPerPair;g(this.id,o.offsets.length>=1&&o.parameters.length===7+2*p&&o.times.length===o.positions.length&&o.output.intervals.length===_&&o.output.positions.length===_&&o.output.status.length===_&&o.output.relationships.length===(m??o.offsets.length-1),`view lengths must match the observation, track, parameter and interval counts`);let y=Object.values(o.output).filter(e=>e!==void 0);for(let[e,t]of y.entries())g(this.id,!y.slice(0,e).some(e=>i(e,t)),`output views must not overlap`);r(this.id,this.props.output),e(this.id,[{intervals:o.output.intervals,positions:o.output.positions,status:o.output.status,relationships:o.output.relationships}]),v(this.id,Object.values(o.output),[o.positions,o.times,o.offsets,o.parameters,o.timeOrigin,o.pairs?.first,o.pairs?.second,o.pairs?.count])}getCommandNodes(e){n(e,this.id,this.props);let{id:r,props:i}=this;_(r,e,[i.positions,i.times,i.offsets,i.parameters,i.timeOrigin,i.pairs?.first,i.pairs?.second,i.pairs?.count,...Object.values(i.output)]),i.times.format===`uint32x2`&&t(e,r,{"relative word times":Math.max(2*i.positions.length,1)*4});let a=i.times.format===`uint32x2`?w(e,r,i):void 0,s=a?.timeTable,c=a?.nodes??[];if(i.pairwiseTracks||i.pairs)return[...c,...E(e,r,i,i.pairs?{...i.pairs,capacity:i.pairs.first.length}:{staticPairs:i.pairwiseTracks,capacity:i.pairwiseTracks.length},s)];if(i.polygonRingVertexCounts||i.timeBoundary||i.geometryBoundary||i.intervalsPerSegment||i.output.compactIntervals)return[...c,...T(e,r,i,s)];let l=C(i,s);return[...c,d(e,{id:r,operation:`GPUTrajectoryRestriction`,variant:`closed-planar-rectangle${l.suffix}`,invocationCount:i.offsets.length-1,workgroupSize:32,bindings:[{name:`positions`,view:i.positions,type:`f32`,access:`read`},...l.bindings,{name:`offsets`,view:i.offsets,type:`u32`,access:`read`},{name:`parameters`,view:i.parameters,type:`f32`,access:`read`},{name:`intervals`,view:i.output.intervals,type:`f32`,access:`read_write`},{name:`clipped`,view:i.output.positions,type:`f32`,access:`read_write`},{name:`status`,view:i.output.status,type:`u32`,access:`read_write`},{name:`relationships`,view:i.output.relationships,type:`u32`,access:`read_write`}],declarations:`const OBSERVATION_COUNT: u32 = ${i.positions.length}u;
${l.source}
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn loadPosition(row: u32) -> vec2<f32> { return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]); }`,body:`let start = min(offsets[offsetsOffset + index], OBSERVATION_COUNT); let end = min(offsets[offsetsOffset + index + 1u], OBSERVATION_COUNT);
let minimumTime = parameters[parametersOffset]; let maximumTime = parameters[parametersOffset + 1u];
let minimum = vec2<f32>(parameters[parametersOffset + 2u], parameters[parametersOffset + 3u]); let maximum = vec2<f32>(parameters[parametersOffset + 4u], parameters[parametersOffset + 5u]); let maximumGap = ${o(`parameters[parametersOffset + 6u]`)};
var parametersValid = finite(minimumTime) && finite(maximumTime) && minimumTime <= maximumTime && all(minimum <= maximum) && maximumGap > 0.0;
for (var parameter = 2u; parameter < 6u; parameter++) { parametersValid = parametersValid && finite(parameters[parametersOffset + parameter]); }
var nanBits = 0x7fc00000u; let missing = bitcast<f32>(nanBits);
var anyInside = false; var allInside = true; var supportCount = 0u; var unknownCount = 0u;
for (var row = start; row < end; row++) {
  for (var component = 0u; component < 2u; component++) { intervals[intervalsOffset + 2u * row + component] = missing; }
  for (var component = 0u; component < 4u; component++) { clipped[clippedOffset + 4u * row + component] = missing; }
  status[statusOffset + row] = 0u;
  let next = min(row + 1u, end - 1u); let first = loadPosition(row); let second = loadPosition(next);
  let firstTime = rowTime(row); let secondTime = rowTime(next); let duration = rowElapsed(next, row);
  if (!parametersValid || !finite(firstTime) || !finite(secondTime) || duration < 0.0 || !finite(first.x) || !finite(first.y) || !finite(second.x) || !finite(second.y) || (duration == 0.0 && any(first != second))) { status[statusOffset + row] = 4u; unknownCount++; continue; }
  if (secondTime < minimumTime || firstTime > maximumTime) { continue; }
  if (duration > maximumGap) {
    let unknownInterior = max(minimumTime, firstTime) < min(maximumTime, secondTime);
    status[statusOffset + row] = 8u;
    if (unknownInterior) { unknownCount++; }
    // A known endpoint remains a measured singleton even when the following gap is unsupported.
    if (firstTime >= minimumTime && firstTime <= maximumTime) {
      supportCount++;
      let inside = all(first >= minimum) && all(first <= maximum);
      allInside = allInside && inside; anyInside = anyInside || inside;
      status[statusOffset + row] |= 32u;
      if (inside) {
        status[statusOffset + row] |= 18u;
        intervals[intervalsOffset + 2u * row] = firstTime; intervals[intervalsOffset + 2u * row + 1u] = firstTime;
        clipped[clippedOffset + 4u * row] = first.x; clipped[clippedOffset + 4u * row + 1u] = first.y; clipped[clippedOffset + 4u * row + 2u] = first.x; clipped[clippedOffset + 4u * row + 3u] = first.y;
      }
    }
    continue;
  }
  supportCount++;
  var lower = 0.0; var upper = 1.0;
  if (duration > 0.0) { lower = max(0.0, (minimumTime - firstTime) / duration); upper = min(1.0, (maximumTime - firstTime) / duration); }
  let timeLower = lower; let timeUpper = upper; let direction = second - first;
  var intersects = true;
  for (var axis = 0u; axis < 2u; axis++) {
    if (direction[axis] == 0.0) { if (first[axis] < minimum[axis] || first[axis] > maximum[axis]) { intersects = false; } }
    else { let firstCrossing = (minimum[axis] - first[axis]) / direction[axis]; let secondCrossing = (maximum[axis] - first[axis]) / direction[axis]; lower = max(lower, min(firstCrossing, secondCrossing)); upper = min(upper, max(firstCrossing, secondCrossing)); }
  }
  intersects = intersects && lower <= upper;
  let fullyInside = intersects && lower == timeLower && upper == timeUpper;
  allInside = allInside && fullyInside;
  if (!intersects) { status[statusOffset + row] = 32u; continue; }
  anyInside = true;
  let clippedStart = first + direction * lower; let clippedEnd = first + direction * upper;
  intervals[intervalsOffset + 2u * row] = firstTime + duration * lower; intervals[intervalsOffset + 2u * row + 1u] = firstTime + duration * upper;
  clipped[clippedOffset + 4u * row] = clippedStart.x; clipped[clippedOffset + 4u * row + 1u] = clippedStart.y; clipped[clippedOffset + 4u * row + 2u] = clippedEnd.x; clipped[clippedOffset + 4u * row + 3u] = clippedEnd.y;
  status[statusOffset + row] = 32u | select(1u, 2u, duration == 0.0 || lower == upper) | select(0u, 16u, fullyInside);
}
var ever = select(0u, 1u, anyInside); var always = select(0u, 1u, allInside);
if (supportCount == 0u || unknownCount > 0u) { if (!anyInside) { ever = 0xffffffffu; } if (allInside) { always = 0xffffffffu; } }
relationships[relationshipsOffset + 4u * index] = ever; relationships[relationshipsOffset + 4u * index + 1u] = always; relationships[relationshipsOffset + 4u * index + 2u] = supportCount; relationships[relationshipsOffset + 4u * index + 3u] = unknownCount;`})]}},O=.45,k=.3;function A(e,t){let n=(t-.2)/2*e.worldWidth;return{minimumX:e.center[0]-n,maximumX:e.center[0]+n,minimumY:e.center[1]-k,maximumY:e.center[1]+k}}var j=y({parameters:S,caption:`Track parts inside the box and the time window.`,createInputs:({world:e})=>{let t=x(e,{trackCount:14,minimumFixes:22,maximumFixes:34});return{...t,center:[O*t.worldWidth,.5]}},build(e,t){let n=e.allocate(`parameters`,`float32`,7),r=e.allocate(`intervals`,`float32x2`,t.rowCount),i=e.allocate(`clipped`,`float32x4`,t.rowCount),a=e.allocate(`status`,`uint32`,t.rowCount),o=e.allocate(`relationships`,`uint32x4`,t.trackCount),s=e.allocate(`rectangle`,`float32x4`,4);return e.graph.add(new D({positions:e.upload(`positions`,`float32x2`,t.positions).view,times:e.upload(`times`,`float32`,t.timestamps).view,offsets:e.upload(`offsets`,`uint32`,t.trackOffsets).view,parameters:n.view,output:{intervals:r.view,positions:i.view,status:a.view,relationships:o.view}})),{outputs:{clipped:i,rectangle:s,segments:e.upload(`segments`,`float32x4`,t.segments)},update:e=>{let{minimumX:r,maximumX:i,minimumY:a,maximumY:o}=A(t,e.maximumX);n.write(Float32Array.of(e.minimumTime,e.maximumTime,r,a,i,o,1e3)),s.write(Float32Array.of(r,a,i,a,i,a,i,o,i,o,r,o,r,o,r,a))}}},interaction:{kind:`window`,hint:`move the box`,rest:({toUnit:e,inputs:t})=>e(t.center),resolve(e,{inputs:t,values:n}){let r=[b(Math.round(e.x*50)/50,0,t.worldWidth),b(Math.round(e.y*50)/50,.1,.9)];t.center=r;let i=A(t,n.maximumX),a=0;for(let e=0;e<t.trackCount;e++)for(let r=t.trackOffsets[e];r<t.trackOffsets[e+1];r++){let e=t.positions[r*2],o=t.positions[r*2+1],s=t.timestamps[r];if(e>=i.minimumX&&e<=i.maximumX&&o>=i.minimumY&&o<=i.maximumY&&s>=n.minimumTime&&s<=n.maximumTime){a++;break}}return{rerun:!0,glyphs:[{shape:`box`,minimum:[i.minimumX,i.minimumY],maximum:[i.maximumX,i.maximumY],dashed:!0}],readout:`box (${r[0].toFixed(2)}, ${r[1].toFixed(2)}) · ${a} tracks inside`}}},draw:({inputs:e,outputs:t})=>[{layer:`lines`,context:!0,segments:t.segments,count:e.segmentCount},{layer:`lines`,context:!0,segments:t.rectangle,count:4,radius:1.5},{layer:`lines`,segments:t.clipped,count:e.rowCount,radius:3}]});export{j as default};