import{n as e,o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-sort-xRm8l5yn.js";import{t as i}from"./gpu-compaction-B6IzcF0P.js";import{E as a,S as o,f as s,l as c,m as l,w as u,x as d,y as f}from"./gpu-contributor-utils-BIwLQON5.js";import{_ as p,v as m}from"./spatial-context-DwZEd4EZ.js";import{p as h}from"./wgsl-common-DfyXiYB0.js";import{t as g}from"./exact-orientation-wgsl-B_jI-HEc.js";import{t as _}from"./exact-geometry-wgsl-y7TUnqOh.js";import{i as ee,r as v,v as te}from"./spatial-join-passes-CNxNpnDa.js";import{t as ne}from"./sorted-segment-offsets-ChmOtA9F.js";import{n as re}from"./load-balanced-for-each-GB0Xy_f0.js";import{a as y,i as b,o as x}from"./trajectory-common-DFp8lX7h.js";import{a as S,i as ie,n as ae,r as C}from"./trajectory-binding-validation-C89oJIAZ.js";import{n as w,t as T}from"./trajectory-counts-wgsl-CVOZcx6G.js";var E=`GPUZoneEvents`,D=`0x80000000u`;function O(e){return e?{bindings:[{name:`timestamps`,view:e.timestamps,type:e.isWordMode?`u32`:`f32`,access:`read`},{name:`maximumGap`,view:e.maximumGap,type:`f32`,access:`read`}],declarations:`${y(e.isWordMode)}
fn isGapSegment(row: u32) -> bool {
  let limit = maximumGap[maximumGapOffset];
  return limit > 0.0 && rowTimeDifference(row, row - 1u) > limit;
}`}:{bindings:[],declarations:``}}var k=`
const NO_TRACK: u32 = 0xffffffffu;
fn findTrack(row: u32) -> u32 {
  if (row < trackOffsets[trackOffsetsOffset] || row >= trackOffsets[trackOffsetsOffset + TRACK_COUNT]) {
    return NO_TRACK;
  }
  var low = 0u;
  var high = TRACK_COUNT;
  loop {
    if (low >= high) { break; }
    let middle = (low + high) / 2u;
    if (trackOffsets[trackOffsetsOffset + middle] > row) { high = middle; } else { low = middle + 1u; }
  }
  return low - 1u;
}`;function A(e){return`const TRACK_COUNT: u32 = ${e.trackCount}u;
const ZONE_COUNT: u32 = ${e.zoneCount}u;
const EDGE_COUNT: u32 = ${e.edgeCount}u;
const CANDIDATE_CAPACITY: u32 = ${e.candidateCapacity}u;
const MAXIMUM_EVENTS_PER_TRACK: u32 = ${e.maximumEventsPerTrack}u;`}function j(e,t,n){return`fn ${e}(value: u32) -> u32 {
  if (value < ${t}[${t}Offset] || value >= ${t}[${t}Offset + ${n}u]) { return NO_INTERVAL; }
  var low = 0u;
  var high = ${n}u;
  loop {
    if (low >= high) { break; }
    let middle = (low + high) / 2u;
    if (${t}[${t}Offset + middle] > value) { high = middle; } else { low = middle + 1u; }
  }
  return low - 1u;
}`}function oe(e,n){let{zones:r}=n;return t(e,{id:n.id,operation:E,variant:`edges`,bindings:[{name:`positions`,view:r.positions,type:`f32`,access:`read`},{name:`featureOffsets`,view:r.featureOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:r.polygonOffsets,type:`u32`,access:`read`},{name:`ringOffsets`,view:r.ringOffsets,type:`u32`,access:`read`},{name:`edgeStarts`,view:n.edgeStarts,type:`f32`,access:`read_write`},{name:`edgeEnds`,view:n.edgeEnds,type:`f32`,access:`read_write`},{name:`edgeZones`,view:n.edgeZones,type:`u32`,access:`read_write`}],invocationCount:r.positions.length,declarations:`const NO_INTERVAL: u32 = 0xffffffffu;
${j(`findRing`,`ringOffsets`,r.ringOffsets.length-1)}
${j(`findPolygon`,`polygonOffsets`,r.polygonOffsets.length-1)}
${j(`findFeature`,`featureOffsets`,r.featureOffsets.length-1)}`,body:`let vertex = index;
  // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
  let notANumber = bitcast<f32>(0x7fc00000u | (index & 0u));
  var zone = NO_INTERVAL;
  var start = vec2f(notANumber, notANumber);
  var end = vec2f(notANumber, notANumber);
  let ring = findRing(vertex);
  if (ring != NO_INTERVAL) {
    let ringStart = ringOffsets[ringOffsetsOffset + ring];
    let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
    let polygon = findPolygon(ring);
    var feature = NO_INTERVAL;
    if (polygon != NO_INTERVAL) { feature = findFeature(polygon); }
    if (ringEnd - ringStart >= 3u && feature != NO_INTERVAL) {
      let next = select(vertex + 1u, ringStart, vertex + 1u >= ringEnd);
      let a = vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
      let b = vec2f(positions[positionsOffset + next * 2u], positions[positionsOffset + next * 2u + 1u]);
      let isRepeat = bitcast<u32>(a.x) == bitcast<u32>(b.x) && bitcast<u32>(a.y) == bitcast<u32>(b.y);
      if (!isRepeat) {
        zone = feature;
        start = a;
        end = b;
      }
    }
  }
  edgeStarts[edgeStartsOffset + vertex * 2u] = start.x;
  edgeStarts[edgeStartsOffset + vertex * 2u + 1u] = start.y;
  edgeEnds[edgeEndsOffset + vertex * 2u] = end.x;
  edgeEnds[edgeEndsOffset + vertex * 2u + 1u] = end.y;
  edgeZones[edgeZonesOffset + vertex] = zone;`})}function se(e,n){return t(e,{id:n.id,operation:E,variant:`clear`,bindings:[{name:`state`,view:n.state,type:`u32`,access:`read_write`},{name:`initialInside`,view:n.initialInside,type:`u32`,access:`read_write`}],invocationCount:Math.max(n.cellCount,4),body:`if (index < 4u) { state[stateOffset + index] = 0u; }
  if (index < ${n.cellCount}u) { initialInside[initialInsideOffset + index] = 0u; }`})}function ce(e,n){let{bvh:r,shape:i}=n;return t(e,{id:n.id,operation:E,variant:`probe`,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`trackOffsets`,view:n.trackOffsets,type:`u32`,access:`read`},{name:`nodeMinima`,view:r.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:r.nodeMaxima,type:`f32`,access:`read`},{name:`leafIds`,view:r.leafIds,type:`u32`,access:`read`},{name:`state`,view:n.state,type:`atomic<u32>`,access:`read_write`},{name:`candidatePairs`,view:n.candidatePairs,type:`u32`,access:`read_write`}],invocationCount:i.rowCount,declarations:`${w}
${v}
${A(i)}
const INTERNAL_NODE_COUNT: u32 = ${r.internalNodeCount}u;
${k}
${h}`,body:`let row = index;
  let track = findTrack(row);
  if (track == NO_TRACK) { return; }
  let isFirst = row == trackOffsets[trackOffsetsOffset + track];
  let current = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
  if (!isFiniteValue(current.x) || !isFiniteValue(current.y)) { return; }
  var queryMinimum = current;
  var queryMaximum = vec2f(FLOAT32_MAXIMUM, current.y);
  if (!isFirst) {
    let previous = vec2f(positions[positionsOffset + (row - 1u) * 2u], positions[positionsOffset + (row - 1u) * 2u + 1u]);
    if (!isFiniteValue(previous.x) || !isFiniteValue(previous.y)) { return; }
    queryMinimum = min(previous, current);
    queryMaximum = max(previous, current);
  }
  let rowWord = select(row, row | ${D}, isFirst);
  var node = 0u;
  loop {
    if (nodeOverlaps(node, queryMinimum, queryMaximum)) {
      if (node < INTERNAL_NODE_COUNT) {
        node = node * 2u + 1u;
        continue;
      }
      let edgeRow = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
      if (edgeRow < EDGE_COUNT) {
        let slot = reserveTrajectoryCount();
        if (slot < CANDIDATE_CAPACITY) {
          candidatePairs[candidatePairsOffset + slot * 2u] = rowWord;
          candidatePairs[candidatePairsOffset + slot * 2u + 1u] = edgeRow;
        }
      }
    }
    loop {
      if (node == 0u || (node & 1u) == 1u) { break; }
      node = (node - 1u) / 2u;
    }
    if (node == 0u) { break; }
    node = node + 1u;
  }`})}function le(e,n){return t(e,{id:n.id,operation:E,variant:`crossing`,bindings:[{name:`state`,view:n.state,type:`u32`,access:`read`},{name:`candidatePairs`,view:n.candidatePairs,type:`u32`,access:`read`},{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`edgeStarts`,view:n.edgeStarts,type:`f32`,access:`read`},{name:`edgeEnds`,view:n.edgeEnds,type:`f32`,access:`read`},{name:`edgeZones`,view:n.edgeZones,type:`u32`,access:`read`},{name:`candidateKinds`,view:n.candidateKinds,type:`u32`,access:`read_write`},{name:`candidateParameters`,view:n.candidateParameters,type:`f32`,access:`read_write`}],invocationCount:n.shape.candidateCapacity,declarations:`${v}
${A(n.shape)}
${g}
${_}
fn cross2(a: vec2f, b: vec2f) -> f32 { return a.x * b.y - a.y * b.x; }`,body:`candidateKinds[candidateKindsOffset + index] = 0u;
  candidateParameters[candidateParametersOffset + index] = 0.0;
  if (index >= min(state[stateOffset], CANDIDATE_CAPACITY)) { return; }
  let rowWord = candidatePairs[candidatePairsOffset + index * 2u];
  let edge = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  if (edgeZones[edgeZonesOffset + edge] >= ZONE_COUNT) { return; }
  let a = vec2f(edgeStarts[edgeStartsOffset + edge * 2u], edgeStarts[edgeStartsOffset + edge * 2u + 1u]);
  let b = vec2f(edgeEnds[edgeEndsOffset + edge * 2u], edgeEnds[edgeEndsOffset + edge * 2u + 1u]);
  if (!isFiniteValue(a.x) || !isFiniteValue(a.y) || !isFiniteValue(b.x) || !isFiniteValue(b.y)) { return; }
  let isRay = (rowWord & ${D}) != 0u;
  let row = rowWord & 0x7fffffffu;
  let current = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
  // Straddle rule of the perturbed point (see the boundary rule above): a.y >= y differs from b.y >= y.
  if (isRay && ((a.y >= current.y) == (b.y >= current.y))) { return; }
  var previous = current;
  if (!isRay) {
    previous = vec2f(positions[positionsOffset + (row - 1u) * 2u], positions[positionsOffset + (row - 1u) * 2u + 1u]);
  }
  // One batch of exact signs for both candidate kinds keeps a single orientSign call site:
  // the ray needs orient(a, b, current); the segment test all four.
  let exact = exactOrientSigns(
    select(4u, 1u, isRay),
    array<vec2f, 4>(a, a, previous, previous),
    array<vec2f, 4>(b, b, current, current),
    array<vec2f, 4>(current, previous, a, b)
  );
  if (isRay) {
    let side = sosPointSide(exact.signs.x, a, b);
    if (select((side < 0), (side > 0), (b.y > a.y))) { candidateKinds[candidateKindsOffset + index] = 2u; }
    return;
  }
  if (sosPointSide(exact.signs.y, a, b) == sosPointSide(exact.signs.x, a, b)) { return; }
  if (sosSegmentSide(exact.signs.z, previous, current) == sosSegmentSide(exact.signs.w, previous, current)) { return; }
  // Only the decision above is exact. The parameter places the event in time.
  let direction = current - previous;
  let edgeDirection = b - a;
  let offset = a - previous;
  let denominator = cross2(direction, edgeDirection);
  var along = 0.0;
  if (denominator != 0.0) {
    along = cross2(offset, edgeDirection) / denominator;
  } else {
    // Parallel track and edge: the perturbed crossing lies at the edge endpoint nearest the track.
    let length2 = dot(direction, direction);
    if (length2 > 0.0) { along = dot(offset, direction) / length2; }
  }
  along = select(0.0, clamp(along, 0.0, 1.0), isFiniteValue(along));
  candidateKinds[candidateKindsOffset + index] = 1u;
  candidateParameters[candidateParametersOffset + index] = along;`})}function ue(e,n){return t(e,{id:n.id,operation:E,variant:`initial-state`,bindings:[{name:`candidatePairs`,view:n.candidatePairs,type:`u32`,access:`read`},{name:`candidateKinds`,view:n.candidateKinds,type:`u32`,access:`read`},{name:`trackOffsets`,view:n.trackOffsets,type:`u32`,access:`read`},{name:`edgeZones`,view:n.edgeZones,type:`u32`,access:`read`},{name:`initialInside`,view:n.initialInside,type:`atomic<u32>`,access:`read_write`}],invocationCount:n.shape.candidateCapacity,declarations:`${A(n.shape)}
${k}`,body:`if (candidateKinds[candidateKindsOffset + index] != 2u) { return; }
  let row = candidatePairs[candidatePairsOffset + index * 2u] & 0x7fffffffu;
  let edge = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  let track = findTrack(row);
  if (track == NO_TRACK) { return; }
  atomicXor(&initialInside[initialInsideOffset + track * ZONE_COUNT + edgeZones[edgeZonesOffset + edge]], 1u);`})}function de(e,n){return t(e,{id:n.id,operation:E,variant:n.isWordMode?`event-time-words`:`event-time`,bindings:[{name:`candidatePairs`,view:n.candidatePairs,type:`u32`,access:`read`},{name:`candidateKinds`,view:n.candidateKinds,type:`u32`,access:`read`},{name:`candidateParameters`,view:n.candidateParameters,type:`f32`,access:`read`},{name:`timestamps`,view:n.timestamps,type:n.isWordMode?`u32`:`f32`,access:`read`},{name:`trackOffsets`,view:n.trackOffsets,type:`u32`,access:`read`},{name:`candidateTracks`,view:n.candidateTracks,type:`u32`,access:`read_write`},{name:`candidateTimes`,view:n.candidateTimes,type:`f32`,access:`read_write`},{name:`state`,view:n.state,type:`atomic<u32>`,access:`read_write`}],invocationCount:n.shape.candidateCapacity,declarations:`${A(n.shape)}
${k}
${y(n.isWordMode)}`,body:`candidateTracks[candidateTracksOffset + index] = TRACK_COUNT;
  candidateTimes[candidateTimesOffset + index] = 0.0;
  if (candidateKinds[candidateKindsOffset + index] != 1u) { return; }
  let row = candidatePairs[candidatePairsOffset + index * 2u];
  let track = findTrack(row);
  if (track == NO_TRACK) { return; }
  let firstRow = trackOffsets[trackOffsetsOffset + track];
  let base = rowTimeDifference(row - 1u, firstRow);
  let span = rowTimeDifference(row, row - 1u);
  let eventTime = max(base + candidateParameters[candidateParametersOffset + index] * span, 0.0);
  candidateTracks[candidateTracksOffset + index] = track;
  candidateTimes[candidateTimesOffset + index] = eventTime;
  atomicAdd(&state[stateOffset + 1u], 1u);`})}function fe(e,n){let r=[],i;return n.variant===`edge`?(r.push({name:`candidatePairs`,view:n.candidatePairs,type:`u32`,access:`read`},{name:`candidateKinds`,view:n.candidateKinds,type:`u32`,access:`read`},{name:`identity`,view:n.identity,type:`u32`,access:`read_write`}),i=`identity[identityOffset + index] = index;
  let isEvent = candidateKinds[candidateKindsOffset + index] == 1u;
  keys[keysOffset + index] = select(0u, candidatePairs[candidatePairsOffset + index * 2u + 1u], isEvent);`):n.variant===`time`?(r.push({name:`order`,view:n.order,type:`u32`,access:`read`},{name:`candidateTimes`,view:n.candidateTimes,type:`f32`,access:`read`}),i=`keys[keysOffset + index] = bitcast<u32>(candidateTimes[candidateTimesOffset + order[orderOffset + index]]);`):(r.push({name:`order`,view:n.order,type:`u32`,access:`read`},{name:`candidateTracks`,view:n.candidateTracks,type:`u32`,access:`read`}),i=`keys[keysOffset + index] = candidateTracks[candidateTracksOffset + order[orderOffset + index]];`),r.push({name:`keys`,view:n.keys,type:`u32`,access:`read_write`}),t(e,{id:n.id,operation:E,variant:`sort-key-${n.variant}`,bindings:r,invocationCount:n.shape.candidateCapacity,body:i})}function pe(e,n){return t(e,{id:n.id,operation:E,variant:`sorted-columns`,bindings:[{name:`state`,view:n.state,type:`u32`,access:`read`},{name:`order`,view:n.order,type:`u32`,access:`read`},{name:`candidatePairs`,view:n.candidatePairs,type:`u32`,access:`read`},{name:`edgeZones`,view:n.edgeZones,type:`u32`,access:`read`},{name:`candidateTimes`,view:n.candidateTimes,type:`f32`,access:`read`},{name:`sortedZones`,view:n.sortedZones,type:`u32`,access:`read_write`},{name:`sortedTimes`,view:n.sortedTimes,type:`f32`,access:`read_write`},{name:`sortedRows`,view:n.sortedRows,type:`u32`,access:`read_write`}],invocationCount:n.shape.candidateCapacity,body:`if (index >= state[stateOffset + 1u]) {
    sortedZones[sortedZonesOffset + index] = 0xffffffffu;
    sortedTimes[sortedTimesOffset + index] = 0.0;
    sortedRows[sortedRowsOffset + index] = 0xffffffffu;
    return;
  }
  let slot = order[orderOffset + index];
  let edge = candidatePairs[candidatePairsOffset + slot * 2u + 1u];
  sortedZones[sortedZonesOffset + index] = edgeZones[edgeZonesOffset + edge];
  sortedTimes[sortedTimesOffset + index] = candidateTimes[candidateTimesOffset + slot];
  sortedRows[sortedRowsOffset + index] = candidatePairs[candidatePairsOffset + slot * 2u];`})}function me(e,n){return t(e,{id:n.id,operation:E,variant:`cell-keys`,bindings:[{name:`state`,view:n.state,type:`u32`,access:`read`},{name:`sortedTracks`,view:n.sortedTracks,type:`u32`,access:`read`},{name:`sortedZones`,view:n.sortedZones,type:`u32`,access:`read`},{name:`cellKeys`,view:n.cellKeys,type:`u32`,access:`read_write`},{name:`identity`,view:n.identity,type:`u32`,access:`read_write`}],invocationCount:n.shape.candidateCapacity,declarations:A(n.shape),body:`identity[identityOffset + index] = index;
  var key = TRACK_COUNT * ZONE_COUNT;
  if (index < state[stateOffset + 1u]) {
    key = sortedTracks[sortedTracksOffset + index] * ZONE_COUNT + sortedZones[sortedZonesOffset + index];
  }
  cellKeys[cellKeysOffset + index] = key;`})}function he(e,n){let r=O(n.gap);return t(e,{id:n.id,operation:E,variant:n.gap.isWordMode?`gap-snap-words`:`gap-snap`,bindings:[...r.bindings,{name:`candidatePairs`,view:n.candidatePairs,type:`u32`,access:`read`},{name:`candidateKinds`,view:n.candidateKinds,type:`u32`,access:`read`},{name:`candidateParameters`,view:n.candidateParameters,type:`f32`,access:`read_write`}],invocationCount:n.shape.candidateCapacity,declarations:r.declarations,body:`if (candidateKinds[candidateKindsOffset + index] != 1u) { return; }
  if (isGapSegment(candidatePairs[candidatePairsOffset + index * 2u])) {
    candidateParameters[candidateParametersOffset + index] = 1.0;
  }`})}function ge(e,n){return t(e,{id:n.id,operation:E,variant:`observed-times`,bindings:[{name:`state`,view:n.state,type:`u32`,access:`read`},{name:`cellOrder`,view:n.cellOrder,type:`u32`,access:`read`},{name:`sortedRows`,view:n.sortedRows,type:`u32`,access:`read`},{name:`sortedTimes`,view:n.sortedTimes,type:`f32`,access:`read`},{name:`gapPrefix`,view:n.gapPrefix,type:`f32`,access:`read`},{name:`cellObservedTimes`,view:n.cellObservedTimes,type:`f32`,access:`read_write`}],invocationCount:n.shape.candidateCapacity,body:`if (index >= state[stateOffset + 1u]) {
    cellObservedTimes[cellObservedTimesOffset + index] = 0.0;
    return;
  }
  let slot = cellOrder[cellOrderOffset + index];
  cellObservedTimes[cellObservedTimesOffset + index] = sortedTimes[sortedTimesOffset + slot] - gapPrefix[gapPrefixOffset + sortedRows[sortedRowsOffset + slot]];`})}function _e(e,n){let r=O(n.gap);return t(e,{id:n.id,operation:E,variant:n.gap.isWordMode?`gap-prefix-words`:`gap-prefix`,bindings:[...r.bindings,{name:`trackOffsets`,view:n.trackOffsets,type:`u32`,access:`read`},{name:`gapPrefix`,view:n.gapPrefix,type:`f32`,access:`read_write`}],invocationCount:n.shape.trackCount,declarations:r.declarations,body:`let start = min(trackOffsets[trackOffsetsOffset + index], ${n.shape.rowCount}u);
  let end = min(trackOffsets[trackOffsetsOffset + index + 1u], ${n.shape.rowCount}u);
  if (start >= end) { return; }
  var total = 0.0;
  gapPrefix[gapPrefixOffset + start] = 0.0;
  for (var row = start + 1u; row < end; row++) {
    if (isGapSegment(row)) { total = total + rowTimeDifference(row, row - 1u); }
    gapPrefix[gapPrefixOffset + row] = total;
  }`})}function ve(e,n){return t(e,{id:n.id,operation:E,variant:`cell-events`,bindings:[{name:`state`,view:n.state,type:`u32`,access:`read`},{name:`cellOrder`,view:n.cellOrder,type:`u32`,access:`read`},{name:`sortedCellKeys`,view:n.sortedCellKeys,type:`u32`,access:`read`},{name:`cellOffsets`,view:n.cellOffsets,type:`u32`,access:`read`},{name:`initialInside`,view:n.initialInside,type:`u32`,access:`read`},{name:`sortedTimes`,view:n.sortedTimes,type:`f32`,access:`read`},{name:`eventTypes`,view:n.eventTypes,type:`u32`,access:`read_write`},{name:`cellTimes`,view:n.cellTimes,type:`f32`,access:`read_write`}],invocationCount:n.shape.candidateCapacity,body:`if (index >= state[stateOffset + 1u]) {
    cellTimes[cellTimesOffset + index] = 0.0;
    return;
  }
  let cell = sortedCellKeys[sortedCellKeysOffset + index];
  let rank = index - cellOffsets[cellOffsetsOffset + cell];
  let after = (initialInside[initialInsideOffset + cell] & 1u) ^ ((rank + 1u) & 1u);
  let slot = cellOrder[cellOrderOffset + index];
  eventTypes[eventTypesOffset + slot] = select(1u, 0u, after == 1u);
  cellTimes[cellTimesOffset + index] = sortedTimes[sortedTimesOffset + slot];`})}var ye=`
fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> f32 {
  let rank = item - loadBalancedOffsets[loadBalancedOffsetsOffset + row];
  let seed = initialInside[initialInsideOffset + row] & 1u;
  let after = seed ^ ((rank + 1u) & 1u);
  if (after == 1u) {
    return 0.0;
  }
  var enterTime = 0.0;
  if (rank > 0u) {
    enterTime = cellTimes[cellTimesOffset + item - 1u];
  }
  return cellTimes[cellTimesOffset + item] - enterTime;
}`;function be(e,n){return t(e,{id:n.id,operation:E,variant:`event-ranks`,bindings:[{name:`state`,view:n.state,type:`u32`,access:`read`},{name:`sortedTracks`,view:n.sortedTracks,type:`u32`,access:`read`},{name:`eventRanks`,view:n.eventRanks,type:`u32`,access:`read_write`}],invocationCount:n.shape.candidateCapacity,body:`let eventTotal = state[stateOffset + 1u];
  if (index >= eventTotal) {
    eventRanks[eventRanksOffset + index] = 0xffffffffu;
    return;
  }
  let track = sortedTracks[sortedTracksOffset + index];
  var low = 0u;
  var high = index;
  loop {
    if (low >= high) { break; }
    let middle = (low + high) / 2u;
    if (sortedTracks[sortedTracksOffset + middle] < track) { low = middle + 1u; } else { high = middle; }
  }
  eventRanks[eventRanksOffset + index] = index - low;`})}function xe(e,n){return t(e,{id:n.id,operation:E,variant:`track-counts`,bindings:[{name:`state`,view:n.state,type:`atomic<u32>`,access:`read_write`},{name:`sortedTracks`,view:n.sortedTracks,type:`u32`,access:`read`},{name:`trackEventCounts`,view:n.trackEventCounts,type:`u32`,access:`read_write`}],invocationCount:n.shape.trackCount,declarations:`${A(n.shape)}
// First event position whose track is at least (or, with strict, above) the given track.
fn findTrackBound(track: u32, eventTotal: u32, strict: bool) -> u32 {
  var low = 0u;
  var high = eventTotal;
  loop {
    if (low >= high) { break; }
    let middle = (low + high) / 2u;
    let value = sortedTracks[sortedTracksOffset + middle];
    if (value < track || (strict && value == track)) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}`,body:`let eventTotal = atomicLoad(&state[stateOffset + 1u]);
  let count = findTrackBound(index, eventTotal, true) - findTrackBound(index, eventTotal, false);
  trackEventCounts[trackEventCountsOffset + index] = count;
  if (count > MAXIMUM_EVENTS_PER_TRACK) { atomicOr(&state[stateOffset + 2u], 1u); }`})}function Se(e,n){return t(e,{id:n.id,operation:E,variant:`${n.isWordMode?`cell-finalize-words`:`cell-finalize`}${n.gapPrefix?`-gap`:``}`,bindings:[...n.gapPrefix?[{name:`gapPrefix`,view:n.gapPrefix,type:`f32`,access:`read`}]:[],{name:`timestamps`,view:n.timestamps,type:n.isWordMode?`u32`:`f32`,access:`read`},{name:`trackOffsets`,view:n.trackOffsets,type:`u32`,access:`read`},{name:`cellOffsets`,view:n.cellOffsets,type:`u32`,access:`read`},{name:`initialInside`,view:n.initialInside,type:`u32`,access:`read`},{name:`cellTimes`,view:n.cellTimes,type:`f32`,access:`read`},{name:`dwellTimes`,view:n.dwellTimes,type:`f32`,access:`read_write`},{name:`visitCounts`,view:n.visitCounts,type:`u32`,access:`read_write`}],invocationCount:n.shape.trackCount*n.shape.zoneCount,declarations:`${A(n.shape)}
${y(n.isWordMode)}`,body:`let start = cellOffsets[cellOffsetsOffset + index];
  let eventCount = cellOffsets[cellOffsetsOffset + index + 1u] - start;
  let seed = initialInside[initialInsideOffset + index] & 1u;
  let enters = select((eventCount + 1u) / 2u, eventCount / 2u, seed == 1u);
  visitCounts[visitCountsOffset + index] = seed + enters;
  if ((seed ^ (eventCount & 1u)) == 0u) { return; }
  var enterTime = 0.0;
  if (eventCount > 0u) { enterTime = cellTimes[cellTimesOffset + start + eventCount - 1u]; }
  let track = index / ZONE_COUNT;
  let firstRow = trackOffsets[trackOffsetsOffset + track];
  let lastRow = trackOffsets[trackOffsetsOffset + track + 1u] - 1u;
  let duration = rowTimeDifference(lastRow, firstRow)${n.gapPrefix?` - gapPrefix[gapPrefixOffset + lastRow]`:``};
  dwellTimes[dwellTimesOffset + index] = dwellTimes[dwellTimesOffset + index] + (duration - enterTime);`})}function Ce(e,n){return t(e,{id:n.id,operation:E,variant:n.isWordMode?`cell-spans-words`:`cell-spans`,bindings:[{name:`timestamps`,view:n.timestamps,type:n.isWordMode?`u32`:`f32`,access:`read`},{name:`trackOffsets`,view:n.trackOffsets,type:`u32`,access:`read`},{name:`cellOffsets`,view:n.cellOffsets,type:`u32`,access:`read`},{name:`initialInside`,view:n.initialInside,type:`u32`,access:`read`},{name:`cellTimes`,view:n.cellTimes,type:`f32`,access:`read`},{name:`firstEnterTimes`,view:n.firstEnterTimes,type:`f32`,access:`read_write`},{name:`lastExitTimes`,view:n.lastExitTimes,type:`f32`,access:`read_write`}],invocationCount:n.shape.trackCount*n.shape.zoneCount,declarations:`${A(n.shape)}
${y(n.isWordMode)}`,body:`let start = cellOffsets[cellOffsetsOffset + index];
  let eventCount = cellOffsets[cellOffsetsOffset + index + 1u] - start;
  let seed = initialInside[initialInsideOffset + index] & 1u;
  var firstEnter = -1.0;
  if (seed == 1u) {
    firstEnter = 0.0;
  } else if (eventCount > 0u) {
    firstEnter = cellTimes[cellTimesOffset + start];
  }
  firstEnterTimes[firstEnterTimesOffset + index] = firstEnter;
  var lastExit = 0.0;
  if ((seed ^ (eventCount & 1u)) == 1u) {
    let track = index / ZONE_COUNT;
    let firstRow = trackOffsets[trackOffsetsOffset + track];
    let lastRow = trackOffsets[trackOffsetsOffset + track + 1u] - 1u;
    lastExit = rowTimeDifference(lastRow, firstRow);
  } else if (eventCount > 0u) {
    lastExit = cellTimes[cellTimesOffset + start + eventCount - 1u];
  }
  lastExitTimes[lastExitTimesOffset + index] = lastExit;`})}function we(e,n){return t(e,{id:n.id,operation:E,variant:`keep-flags`,bindings:[{name:`state`,view:n.state,type:`u32`,access:`read`},{name:`eventRanks`,view:n.eventRanks,type:`u32`,access:`read`},{name:`keepFlags`,view:n.keepFlags,type:`u32`,access:`read_write`},{name:`rowIds`,view:n.rowIds,type:`u32`,access:`read_write`}],invocationCount:n.shape.candidateCapacity,declarations:A(n.shape),body:`rowIds[rowIdsOffset + index] = index;
  let isKept = index < state[stateOffset + 1u] && eventRanks[eventRanksOffset + index] < MAXIMUM_EVENTS_PER_TRACK;
  keepFlags[keepFlagsOffset + index] = select(0u, 1u, isKept);`})}function Te(e,n){let r=[{name:`keptTotal`,view:n.keptTotal,type:`u32`,access:`read`},{name:`keptRows`,view:n.keptRows,type:`u32`,access:`read`}],i=[];for(let[e,t]of n.columns.entries()){r.push({name:`source${e}`,view:t.source,type:`u32`,access:`read`},{name:`destination${e}`,view:t.destination,type:`u32`,access:`read_write`});let n=t.stride??1;for(let r=0;r<n;r++)i.push(`destination${e}[destination${e}Offset + index * ${n}u + ${r}u] = select(${t.sentinel}, source${e}[source${e}Offset + row * ${n}u + ${r}u], isLive);`)}return t(e,{id:n.id,operation:E,variant:`gather`,bindings:r,invocationCount:n.capacity,body:`let isLive = index < keptTotal[keptTotalOffset];
  let row = select(0u, keptRows[keptRowsOffset + index], isLive);
  ${i.join(`
  `)}`})}function Ee(e,n){return t(e,{id:n.id,operation:E,variant:`overflow`,bindings:[{name:`state`,view:n.state,type:`u32`,access:`read`},{name:`flag`,view:n.flag,type:`u32`,access:`read_write`}],invocationCount:1,declarations:A(n.shape),body:`let candidateOverflow = state[stateOffset] > CANDIDATE_CAPACITY;
  let trackOverflow = state[stateOffset + 2u] != 0u;
  flag[flagOffset] = select(0u, 1u, candidateOverflow || trackOverflow);`})}function De(e,n){let r=[{name:`state`,view:n.state,type:`u32`,access:`read`},{name:`keptTotal`,view:n.keptTotal,type:`u32`,access:`read`}],i=[],a=[[`candidateCount`,n.candidateCount,`state[stateOffset]`],[`candidateOverflow`,n.candidateOverflow,`select(0u, 1u, state[stateOffset] > CANDIDATE_CAPACITY)`],[`trackOverflow`,n.trackOverflow,`select(0u, 1u, state[stateOffset + 2u] != 0u)`],[`eventOverflow`,n.eventOverflow,`select(0u, 1u, keptTotal[keptTotalOffset] > EVENT_CAPACITY)`]];for(let[e,t,n]of a)t&&(r.push({name:e,view:t,type:`u32`,access:`read_write`}),i.push(`${e}[${e}Offset] = ${n};`));return t(e,{id:n.id,operation:E,variant:`diagnostics`,bindings:r,invocationCount:1,declarations:`${A(n.shape)}
const EVENT_CAPACITY: u32 = ${n.eventCapacity}u;`,body:i.join(`
  `)})}function Oe(e,n){return t(e,{id:n.id,operation:E,variant:`sorted-positions`,bindings:[{name:`state`,view:n.state,type:`u32`,access:`read`},{name:`order`,view:n.order,type:`u32`,access:`read`},{name:`candidatePairs`,view:n.candidatePairs,type:`u32`,access:`read`},{name:`candidateParameters`,view:n.candidateParameters,type:`f32`,access:`read`},{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`sortedPositions`,view:n.sortedPositions,type:`f32`,access:`read_write`}],invocationCount:n.shape.candidateCapacity,body:`var crossing = vec2f(0.0, 0.0);
  if (index < state[stateOffset + 1u]) {
    let slot = order[orderOffset + index];
    let row = candidatePairs[candidatePairsOffset + slot * 2u];
    let along = candidateParameters[candidateParametersOffset + slot];
    let previous = vec2f(positions[positionsOffset + (row - 1u) * 2u], positions[positionsOffset + (row - 1u) * 2u + 1u]);
    let current = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
    crossing = previous + along * (current - previous);
  }
  sortedPositions[sortedPositionsOffset + index * 2u] = crossing.x;
  sortedPositions[sortedPositionsOffset + index * 2u + 1u] = crossing.y;`})}function M(e,n){return t(e,{id:n.id,operation:E,variant:`table-flags`,bindings:[{name:`visitCounts`,view:n.visitCounts,type:`u32`,access:`read`},{name:`flags`,view:n.flags,type:`u32`,access:`read_write`},{name:`cellIds`,view:n.cellIds,type:`u32`,access:`read_write`}],invocationCount:n.cellCount,body:`cellIds[cellIdsOffset + index] = index;
  flags[flagsOffset + index] = select(0u, 1u, visitCounts[visitCountsOffset + index] > 0u);`})}function N(e){return e.kind===`value`?2:1}function P(e,n){let r=[{name:`keptTotal`,view:n.keptTotal,type:`u32`,access:`read`},{name:`keptCells`,view:n.keptCells,type:`u32`,access:`read`}],i=[];for(let[e,t]of n.columns.entries()){let n=t.destination.format===`float32`?`f32`:`u32`,a=n===`f32`?`0.0`:`0xffffffffu`;r.push({name:`destination${e}`,view:t.destination,type:n,access:`read_write`});let o;t.kind===`value`&&t.source?(r.push({name:`source${e}`,view:t.source,type:n,access:`read`}),o=`source${e}[source${e}Offset + cell]`):o=t.kind===`track`?`cell / ZONE_COUNT`:`cell % ZONE_COUNT`,i.push(`destination${e}[destination${e}Offset + index] = select(${a}, ${o}, isLive);`)}return o(n.id,`table gather storage bindings`,r.length,8),t(e,{id:n.id,operation:E,variant:`table-gather`,bindings:r,invocationCount:n.capacity,declarations:`const ZONE_COUNT: u32 = ${n.zoneCount}u;`,body:`let isLive = index < keptTotal[keptTotalOffset];
  let cell = select(0u, keptCells[keptCellsOffset + index], isLive);
  ${i.join(`
  `)}`})}var F=`GPUZoneEvents`,I=8,L=2,R=3,z=class{id;props;spatialContext;trackCount;zoneCount;edgeCount;leafCapacity;constructor(e){this.id=e.id??`zone-events`,this.props=e;let{id:t}=this;this.spatialContext=m(t,e.spatialContext),p(this.id,`zones`,e.zonesSpatialContext,this.spatialContext);let{tracks:n,zones:r,output:i}=e;if(a(n.positions,[`float32x2`],`${t} tracks.positions`),a(n.timestamps,[`float32`,`uint32x2`],`${t} tracks.timestamps`),a(n.trackOffsets,[`uint32`],`${t} tracks.trackOffsets`),a(r.positions,[`float32x2`],`${t} zones.positions`),a(r.featureOffsets,[`uint32`],`${t} zones.featureOffsets`),a(r.polygonOffsets,[`uint32`],`${t} zones.polygonOffsets`),a(r.ringOffsets,[`uint32`],`${t} zones.ringOffsets`),n.timestamps.length!==n.positions.length)throw Error(`${t} timestamps length must equal positions length`);if(n.trackOffsets.length<2)throw Error(`${t} trackOffsets must contain at least two rows`);if(e.maximumGap&&a(e.maximumGap,[`float32`],`${t} maximumGap`),x(t,n.timestamps,n.temporalContext),r.featureOffsets.length<2)throw Error(`${t} zones need at least one feature`);if(r.polygonOffsets.length<1||r.ringOffsets.length<1)throw Error(`${t} zones polygonOffsets and ringOffsets must not be empty`);if(this.trackCount=n.trackOffsets.length-1,this.zoneCount=r.featureOffsets.length-1,this.edgeCount=r.positions.length,this.edgeCount<1)throw Error(`${t} zones need at least one vertex`);if(s(t,`candidateCapacity`,e.candidateCapacity,1,T),s(t,`trackEventCapacity`,e.trackEventCapacity),n.positions.length>=2147483648)throw Error(`${t} supports fewer than 2^31 rows`);let o=this.trackCount*this.zoneCount;if(o>4294967295)throw Error(`${t} trackCount * zoneCount must fit in 32 bits`);for(let[e,n,r]of[[`dwellTimes`,i.dwellTimes,`float32`],[`visitCounts`,i.visitCounts,`uint32`]])if(n&&(a(n,[r],`${t} ${e}`),n.length!==o))throw Error(`${t} ${e} length must equal trackCount * zoneCount`);if(i.trackEventCounts&&(a(i.trackEventCounts,[`uint32`],`${t} trackEventCounts`),i.trackEventCounts.length!==this.trackCount))throw Error(`${t} trackEventCounts length must equal the track count`);let{events:d}=i;l(t,d);let h=d.ids.length;for(let[e,n,r]of[[`eventZones`,d.eventZones,`uint32`],[`eventTypes`,d.eventTypes,`uint32`],[`eventTimes`,d.eventTimes,`float32`],[`eventRows`,d.eventRows,`uint32`]])if(n&&(a(n,[r],`${t} events.${e}`),n.length!==h))throw Error(`${t} events.${e} length must equal the capacity`);if(d.eventPositions&&(a(d.eventPositions,[`float32x2`],`${t} events.eventPositions`),d.eventPositions.length!==h))throw Error(`${t} events.eventPositions length must equal the capacity`);let{visitTable:g}=i;if(g){l(`${t} visitTable`,g);let e=g.ids.length;for(let[n,r,i]of[[`zones`,g.zones,`uint32`],[`visits`,g.visits,`uint32`],[`dwellTimes`,g.dwellTimes,`float32`],[`firstEnterTimes`,g.firstEnterTimes,`float32`],[`lastExitTimes`,g.lastExitTimes,`float32`]])if(r&&(a(r,[i],`${t} visitTable.${n}`),r.length!==e))throw Error(`${t} visitTable.${n} length must equal the capacity`)}for(let[e,n]of Object.entries(i.status??{}))if(n&&(a(n,[`uint32`],`${t} status.${e}`),n.length!==1))throw Error(`${t} status.${e} must have one row`);this.leafCapacity=e.leafCapacity??c(this.edgeCount),u(t,`leafCapacity`,this.leafCapacity),C(this.id,this.props.output),S(t,[{dwellTimes:i.dwellTimes,visitCounts:i.visitCounts},{...i.status},{...d.status},{...g?.status}]);let _=[d.ids,d.eventZones,d.eventRows,d.eventTypes,d.eventTimes,d.eventPositions].filter(e=>e!==void 0);for(let e=0;e<_.length;e+=R)S(t,[Object.fromEntries(_.slice(e,e+R).map((e,t)=>[`eventColumn${t}`,e]))]);if(g)for(let e of V(B(g,i)))f(t,e.map(e=>e.destination),e.map(e=>e.source)),S(t,[Object.fromEntries(e.map((e,t)=>[`tableColumn${t}`,e.destination]))]);f(t,[d.ids,d.status.count,d.status.overflow,d.status.candidateOverflow,d.status.requiredCount,d.eventZones,d.eventTypes,d.eventTimes,d.eventRows,d.eventPositions,...this.getTableViews(),i.dwellTimes,i.visitCounts,i.trackEventCounts,...Object.values(i.status??{})],[n.positions,n.timestamps,n.trackOffsets,r.positions,r.featureOffsets,r.polygonOffsets,r.ringOffsets,e.maximumGap])}getCommandNodes(t){ae(t,this.id,this.props);let{id:a,props:o,trackCount:s,edgeCount:c,leafCapacity:l}=this,{tracks:u,zones:f,output:p}=o,{events:m}=p;d(a,t,[u.positions,u.timestamps,u.trackOffsets,o.maximumGap,f.positions,f.featureOffsets,f.polygonOffsets,f.ringOffsets,m.ids,m.status.count,m.status.overflow,m.status.candidateOverflow,m.status.requiredCount,m.eventZones,m.eventTypes,m.eventTimes,m.eventRows,m.eventPositions,...this.getTableViews(),p.dwellTimes,p.visitCounts,p.trackEventCounts,...Object.values(p.status??{})]);let h=u.positions.length,g=o.candidateCapacity,_=s*this.zoneCount,v=u.timestamps.format===`uint32x2`,y={rowCount:h,trackCount:s,edgeCount:c,zoneCount:this.zoneCount,candidateCapacity:g,maximumEventsPerTrack:o.trackEventCapacity};ie(t,a,{"edge bounds":c*8,"BVH nodes":(2*l-1)*8,"candidate pairs":g*8,"track-zone cells":_*4,"row scratch":h*4});let x=(e,r)=>n(t,`${a}-${e}`,`uint32`,Math.max(r,1)),S=(e,r)=>n(t,`${a}-${e}`,`float32`,Math.max(r,1)),C=[],w=n(t,`${a}-edge-starts`,`float32x2`,c),T=n(t,`${a}-edge-ends`,`float32x2`,c),E=x(`edge-zones`,c);C.push(oe(t,{id:`${a}-edges`,zones:f,edgeStarts:w,edgeEnds:T,edgeZones:E}));let D=n(t,`${a}-edge-minima`,`float32x2`,c),O=n(t,`${a}-edge-maxima`,`float32x2`,c);C.push(ee(t,{id:`${a}-bounds`,operation:F,featureCount:c,source:{kind:`segments`,starts:w,ends:T},minima:D,maxima:O}));let{bvh:k,nodes:A}=te(t,a,F,D,O,l,o.spatialSort??!1);C.push(...A);let j=o.maximumGap?{maximumGap:o.maximumGap,timestamps:u.timestamps,isWordMode:v}:void 0,M=j?S(`gap-prefix`,h):void 0;j&&M&&C.push(_e(t,{id:`${a}-gap-prefix`,shape:y,gap:j,trackOffsets:u.trackOffsets,gapPrefix:M}));let N=x(`state`,4),P=x(`initial-inside`,_),I=n(t,`${a}-candidate-pairs`,`uint32x2`,g),L=x(`candidate-kinds`,g),z=S(`candidate-parameters`,g),B=x(`candidate-tracks`,g),V=S(`candidate-times`,g);C.push(se(t,{id:`${a}-clear`,state:N,initialInside:P,cellCount:_}),ce(t,{id:`${a}-probe`,shape:y,positions:u.positions,trackOffsets:u.trackOffsets,bvh:k,state:N,candidatePairs:I}),le(t,{id:`${a}-crossing`,shape:y,positions:u.positions,edgeStarts:w,edgeEnds:T,edgeZones:E,state:N,candidatePairs:I,candidateKinds:L,candidateParameters:z}),...j?[he(t,{id:`${a}-gap-snap`,shape:y,gap:j,candidatePairs:I,candidateKinds:L,candidateParameters:z})]:[],ue(t,{id:`${a}-initial-state`,shape:y,trackOffsets:u.trackOffsets,edgeZones:E,candidatePairs:I,candidateKinds:L,initialInside:P}),de(t,{id:`${a}-event-time`,shape:y,isWordMode:v,timestamps:u.timestamps,trackOffsets:u.trackOffsets,state:N,candidatePairs:I,candidateKinds:L,candidateParameters:z,candidateTracks:B,candidateTimes:V}));let ke=x(`identity`,g),Ae=x(`edge-keys`,g),je=x(`edge-sorted-keys`,g),H=x(`edge-order`,g),Me=x(`time-keys`,g),Ne=x(`time-sorted-keys`,g),U=x(`time-order`,g),Pe=x(`track-keys`,g),W=x(`sorted-tracks`,g),Fe=x(`order`,g);C.push(fe(t,{id:`${a}-edge-keys`,shape:y,variant:`edge`,candidatePairs:I,candidateKinds:L,keys:Ae,identity:ke}),...new r({id:`${a}-edge-sort`,keys:Ae,values:ke,outputKeys:je,outputValues:H,keyBits:b(c-1)}).getCommandNodes(t),fe(t,{id:`${a}-time-keys`,shape:y,variant:`time`,order:H,candidateTimes:V,keys:Me}),...new r({id:`${a}-time-sort`,keys:Me,values:H,outputKeys:Ne,outputValues:U,keyBits:32}).getCommandNodes(t),fe(t,{id:`${a}-track-keys`,shape:y,variant:`track`,order:U,candidateTracks:B,keys:Pe}),...new r({id:`${a}-track-sort`,keys:Pe,values:U,outputKeys:W,outputValues:Fe,keyBits:b(s)}).getCommandNodes(t));let Ie=x(`sorted-zones`,g),G=S(`sorted-times`,g),Le=x(`sorted-rows`,g),Re=x(`event-types`,g),ze=x(`event-ranks`,g),{visitTable:K}=p,Be=!!(K?.firstEnterTimes||K?.lastExitTimes),Ve=Be?S(`first-enter-times`,_):void 0,He=Be?S(`last-exit-times`,_):void 0,Ue=p.dwellTimes??S(`dwell-times`,_),We=p.visitCounts??x(`visit-counts`,_),Ge=p.trackEventCounts??x(`track-event-counts`,s),Ke=x(`cell-keys`,g),qe=x(`cell-identity`,g),q=x(`sorted-cell-keys`,g),Je=x(`cell-order`,g),J=x(`cell-offsets`,_+1),Y=S(`cell-times`,g),X=M?S(`cell-observed-times`,g):void 0;C.push(pe(t,{id:`${a}-sorted-columns`,shape:y,state:N,order:Fe,candidatePairs:I,edgeZones:E,candidateTimes:V,sortedZones:Ie,sortedTimes:G,sortedRows:Le}),be(t,{id:`${a}-event-ranks`,shape:y,state:N,sortedTracks:W,eventRanks:ze}),xe(t,{id:`${a}-track-counts`,shape:y,state:N,sortedTracks:W,trackEventCounts:Ge}),me(t,{id:`${a}-cell-keys`,shape:y,state:N,sortedTracks:W,sortedZones:Ie,cellKeys:Ke,identity:qe}),...new r({id:`${a}-cell-sort`,keys:Ke,values:qe,outputKeys:q,outputValues:Je,keyBits:b(_)}).getCommandNodes(t),ne(t,{id:`${a}-cell-offsets`,operation:F,variant:`cell-offsets`,segmentCount:_,sortedKeys:q,segmentOffsets:J}),ve(t,{id:`${a}-cell-events`,shape:y,state:N,cellOrder:Je,sortedCellKeys:q,cellOffsets:J,initialInside:P,sortedTimes:G,eventTypes:Re,cellTimes:Y}),...M&&X?[ge(t,{id:`${a}-observed-times`,shape:y,state:N,cellOrder:Je,sortedRows:Le,sortedTimes:G,gapPrefix:M,cellObservedTimes:X})]:[],...re(t,{id:`${a}-dwell`,operation:F,variant:`dwell-sum`,offsets:J,itemCapacity:g,accumulatorType:`f32`,output:Ue,bindings:[{name:`initialInside`,view:P,type:`u32`,access:`read`},{name:`cellTimes`,view:X??Y,type:`f32`,access:`read`}],declarations:ye}),Se(t,{id:`${a}-cell-finalize`,shape:y,isWordMode:v,timestamps:u.timestamps,trackOffsets:u.trackOffsets,cellOffsets:J,initialInside:P,cellTimes:X??Y,dwellTimes:Ue,visitCounts:We,gapPrefix:M})),Ve&&He&&C.push(Ce(t,{id:`${a}-cell-spans`,shape:y,isWordMode:v,timestamps:u.timestamps,trackOffsets:u.trackOffsets,cellOffsets:J,initialInside:P,cellTimes:Y,firstEnterTimes:Ve,lastExitTimes:He}));let Z;m.eventPositions&&(Z=n(t,`${a}-sorted-positions`,`float32x2`,Math.max(g,1)),C.push(Oe(t,{id:`${a}-sorted-positions`,shape:y,state:N,order:Fe,candidatePairs:I,candidateParameters:z,positions:u.positions,sortedPositions:Z})));let Ye=x(`keep-flags`,g),Xe=x(`row-ids`,g),Ze=x(`kept-rows`,g),Q=x(`kept-total`,1),Qe=x(`overflow-flag`,1);C.push(we(t,{id:`${a}-keep-flags`,shape:y,state:N,eventRanks:ze,keepFlags:Ye,rowIds:Xe}),...new i({id:`${a}-kept`,input:Xe,flags:Ye,output:Ze,count:Q}).getCommandNodes(t),Ee(t,{id:`${a}-overflow`,shape:y,state:N,flag:Qe})),p.status&&Object.values(p.status).some(Boolean)&&C.push(De(t,{id:`${a}-diagnostics`,shape:y,eventCapacity:m.ids.length,state:N,keptTotal:Q,...p.status}));let $e=p.status?.candidateOverflow??x(`visit-candidate-overflow`,1);K?.status.candidateOverflow&&!p.status?.candidateOverflow&&C.push(De(t,{id:`${a}-visit-candidate-status`,shape:y,state:N,keptTotal:Q,eventCapacity:m.ids.length,candidateOverflow:$e}));let $=[{source:W,destination:m.ids,sentinel:`0xffffffffu`}];m.eventZones&&$.push({source:Ie,destination:m.eventZones,sentinel:`0xffffffffu`}),m.eventRows&&$.push({source:Le,destination:m.eventRows,sentinel:`0xffffffffu`}),m.eventTypes&&$.push({source:Re,destination:m.eventTypes,sentinel:`0u`}),m.eventTimes&&$.push({source:G,destination:m.eventTimes,sentinel:`0u`}),m.eventPositions&&Z&&$.push({source:Z,destination:m.eventPositions,sentinel:`0u`,stride:2});for(let e=0,n=0;e<$.length;e+=R,n++)C.push(Te(t,{id:`${a}-gather-${n}`,capacity:m.ids.length,keptTotal:Q,keptRows:Ze,columns:$.slice(e,e+R)}));return C.push(e(t,{id:`${a}-publish`,operation:F,requiredCount:Q,output:m,candidateOverflowSources:m.status.candidateOverflow?[Qe]:[]})),K&&C.push(...this.getTableNodes(t,{cellCount:_,visitCounts:We,dwellTimes:Ue,firstEnterTimes:Ve,lastExitTimes:He,candidateOverflow:$e})),C}getTableViews(){let e=this.props.output.visitTable;return e?[e.ids,e.status.count,e.status.overflow,e.status.candidateOverflow,e.status.requiredCount,e.zones,e.visits,e.dwellTimes,e.firstEnterTimes,e.lastExitTimes]:[]}getTableNodes(t,r){let{id:a}=this,o=this.props.output.visitTable,{cellCount:s}=r,c=o.ids.length,l=(e,r)=>n(t,`${a}-${e}`,`uint32`,Math.max(r,1)),u=l(`table-flags`,s),d=l(`table-cell-ids`,s),f=l(`table-kept-cells`,s),p=l(`table-kept-total`,1),m=[M(t,{id:`${a}-table-flags`,cellCount:s,visitCounts:r.visitCounts,flags:u,cellIds:d}),...new i({id:`${a}-table-cells`,input:d,flags:u,output:f,count:p}).getCommandNodes(t)],h=V(B(o,r));for(let[e,n]of h.entries())m.push(P(t,{id:`${a}-table-gather-${e}`,capacity:c,zoneCount:this.zoneCount,keptTotal:p,keptCells:f,columns:n}));return m.push(e(t,{id:`${a}-table-publish`,operation:F,requiredCount:p,output:o,candidateOverflowSources:o.status.candidateOverflow?[r.candidateOverflow]:[]})),m}};function B(e,t){let n=[{kind:`track`,destination:e.ids}];return e.zones&&n.push({kind:`zone`,destination:e.zones}),e.visits&&n.push({kind:`value`,source:t.visitCounts,destination:e.visits}),e.dwellTimes&&n.push({kind:`value`,source:t.dwellTimes,destination:e.dwellTimes}),e.firstEnterTimes&&n.push({kind:`value`,source:t.firstEnterTimes,destination:e.firstEnterTimes}),e.lastExitTimes&&n.push({kind:`value`,source:t.lastExitTimes,destination:e.lastExitTimes}),n}function V(e){let t=[],n=[],r=L;for(let i of e){let e=N(i);r+e>I&&(t.push(n),n=[],r=L),n.push(i),r+=e}return n.length&&t.push(n),t}var ke=Uint32Array.from({length:4},(e,t)=>t),Ae=Uint32Array.from({length:4},(e,t)=>t*4),je=.02;function H(e){return Math.round(e/je)*je}function Me(e){let t=[];for(let n=0;n<e.rowCount-1;n++){if(e.trackOffsets.includes(n+1))continue;let r=Math.hypot(e.positions[n*2+2]-e.positions[n*2],e.positions[n*2+3]-e.positions[n*2+1]),i=e.timestamps[n+1]-e.timestamps[n];i>0&&r/i<.001&&t.push([e.positions[n*2],e.positions[n*2+1]])}let n=t,r=[];for(r=[n[0]];r.length<3;){let e=n[0],t=-1;for(let i of n){let n=Math.min(...r.map(e=>Math.hypot(i[0]-e[0],i[1]-e[1])));n>t&&(t=n,e=i)}r.push(e)}for(let e=0;e<8;e++){let e=r.map(()=>[0,0,0]);for(let t of n){let n=0;for(let e=1;e<r.length;e++)Math.hypot(t[0]-r[e][0],t[1]-r[e][1])<Math.hypot(t[0]-r[n][0],t[1]-r[n][1])&&(n=e);e[n][0]+=t[0],e[n][1]+=t[1],e[n][2]++}r=r.map((t,n)=>e[n][2]?[e[n][0]/e[n][2],e[n][1]/e[n][2]]:t)}return r.sort((e,t)=>e[0]-t[0]).map(([e,t])=>[H(e),H(t)])}function Ne(e,t){let n=0;for(let r=1;r<e.length;r++)Math.hypot(t.x-e[r][0],t.y-e[r][1])<Math.hypot(t.x-e[n][0],t.y-e[n][1])&&(n=r);return{centers:e.map((e,r)=>r===n?[H(t.x),H(t.y)]:e),zone:n}}function U(e,t){return[.2*t*Math.min(e.width*.6,1.4),.2*t]}function Pe(e,t,n){let r=[],i=[];for(let[a,o]of n){let[n,s]=U(e,t),c=[[a-n,o-s],[a+n,o-s],[a+n,o+s],[a-n,o+s]];for(let[e,t]of c.entries()){let n=c[(e+1)%4];r.push(t[0],t[1]),i.push(t[0],t[1],n[0],n[1])}}return{positions:Float32Array.from(r),outline:Float32Array.from(i)}}function W(){let e=[],t=[];for(let n=0;n<3;n++){let r=n*4;e.push(r,r+1,r+2,r,r+2,r+3),t.push(n,n)}return{indices:Uint32Array.from(e),owners:Uint32Array.from(t)}}export{W as a,z as c,Pe as i,Ae as n,Ne as o,Me as r,U as s,ke as t};