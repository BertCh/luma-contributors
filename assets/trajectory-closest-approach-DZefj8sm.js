import{n as e,t}from"./trajectory-counts-wgsl-CVOZcx6G.js";import{a as n,n as r,r as i}from"./trajectory-binding-validation-C89oJIAZ.js";import{r as a}from"./kernel-C3tfQOn4.js";import{t as o}from"./gpu-compaction-B6IzcF0P.js";import{t as s}from"./gpu-scan-B4NNTfF9.js";import{t as c}from"./gpu-sort-xRm8l5yn.js";import{c as l,n as u,o as d,t as f}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as p}from"./scan-total-C4QddMmi.js";import{E as m,f as h,m as g,v as _,x as v,y}from"./gpu-contributor-utils-BIwLQON5.js";import{s as b,v as x}from"./spatial-context-DwZEd4EZ.js";import{a as S,i as ee,o as C}from"./trajectory-common-DFp8lX7h.js";import{t as w}from"./define-run-B-brkCe4.js";import{s as T}from"./pick-Bo3t3hjv.js";import{c as E,r as D}from"./text-DYSgVV-n.js";import{r as O,t as k}from"./fleet-Cm7PfbI5.js";import{n as A}from"./pair-links-BjRFatl4.js";import{i as j,n as M,r as N,t as P}from"./track-lookup-BfxbJhpy.js";import{or as F}from"./entry-preview-D70TdSbK.js";var I=`GPUTrajectoryClosestApproach`,L=4096;function R(e){return`const ROW_COUNT: u32 = ${e.rowCount}u;
const TRACK_COUNT: u32 = ${e.trackCount}u;
const CELL_SIZE: f32 = ${l(e.cellSize)};
const TIME_CELL_SIZE: f32 = ${l(e.timeCellSize)};
const ENTRY_CAPACITY: u32 = ${e.entryCapacity}u;
const HIT_CAPACITY: u32 = ${e.hitCapacity}u;
const PAIR_CAPACITY: u32 = ${e.pairCapacity}u;
const SEGMENT_WORDS: u32 = 8u;
const ENTRY_WORDS: u32 = 4u;
const HIT_WORDS: u32 = 6u;
const MAXIMUM_CELLS_PER_SEGMENT: u32 = ${L}u;
const NONE: u32 = 0xffffffffu;
// Cell coordinates are clamped to +-2^30 so the hash arithmetic never meets i32 overflow.
const CELL_LIMIT: f32 = 1073741824.0;
fn getCellCoordinate(value: f32, size: f32) -> i32 {
  return i32(clamp(floor(value / size), -CELL_LIMIT, CELL_LIMIT));
}
// Hash of a space-time cell. Different cells may share a hash; every user compares coordinates.
fn getCellHash(cell: vec3<i32>) -> u32 {
  return (bitcast<u32>(cell.x) * 73856093u) ^ (bitcast<u32>(cell.y) * 19349663u) ^
    (bitcast<u32>(cell.z) * 83492791u);
}`}function z(e,t){return{name:`timestamps`,view:e,type:t?`u32`:`f32`,access:`read`}}function B(e,t){let{shape:n}=t;return d(e,{id:t.id,operation:I,variant:`${n.isWordMode?`segments-words`:`segments`}${t.trackValid?`-valid`:``}${t.rowSkip?`-gap`:``}`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},z(t.timestamps,n.isWordMode),{name:`trackOffsets`,view:t.trackOffsets,type:`u32`,access:`read`},...t.trackValid&&!t.rowSkip?[{name:`trackValid`,view:t.trackValid,type:`u32`,access:`read`}]:[],...t.rowSkip?[{name:`rowSkip`,view:t.rowSkip,type:`u32`,access:`read`}]:[],{name:`distance`,view:t.distance,type:`f32`,access:`read`},{name:`segmentRecords`,view:t.segmentRecords,type:`u32`,access:`read_write`},{name:`segmentCounts`,view:t.segmentCounts,type:`u32`,access:`read_write`},{name:`state`,view:t.state,type:`atomic<u32>`,access:`read_write`}],invocationCount:n.rowCount,declarations:`${R(n)}
${S(n.isWordMode)}
fn findTrack(row: u32) -> u32 {
  if (row < trackOffsets[trackOffsetsOffset] || row >= trackOffsets[trackOffsetsOffset + TRACK_COUNT]) {
    return NONE;
  }
  var low = 0u;
  var high = TRACK_COUNT;
  loop {
    if (low >= high) { break; }
    let middle = (low + high) / 2u;
    if (trackOffsets[trackOffsetsOffset + middle] > row) { high = middle; } else { low = middle + 1u; }
  }
  return low - 1u;
}
fn isFiniteNumber(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}`,body:`segmentCounts[segmentCountsOffset + index] = 0u;
  let base = segmentRecordsOffset + index * SEGMENT_WORDS;
  segmentRecords[base + 6u] = NONE;
  let track = findTrack(index);
  if (track == NONE || index + 1u >= trackOffsets[trackOffsetsOffset + track + 1u]) { return; }
  ${t.rowSkip?`if (rowSkip[rowSkipOffset + index] != 0u) { return; }`:t.trackValid?`if (trackValid[trackValidOffset + track] == 0u) { return; }`:``}
  let radius = distance[distanceOffset];
  if (!(isFiniteNumber(radius) && radius >= 0.0)) { return; }
  let start = vec2<f32>(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
  let end = vec2<f32>(positions[positionsOffset + index * 2u + 2u], positions[positionsOffset + index * 2u + 3u]);
  if (!(isFiniteNumber(start.x) && isFiniteNumber(start.y) && isFiniteNumber(end.x) && isFiniteNumber(end.y))) {
    return;
  }
  if (!(rowTimeDifference(index + 1u, index) > 0.0)) { return; }
  let startTime = rowTimeDifference(index, 0u);
  let endTime = rowTimeDifference(index + 1u, 0u);
  // Half the distance on each box (plus a relative margin for the f32 corners); the time pad
  // covers the rounding of times measured from row 0 (two roundings of 2^-24 relative).
  let halfDistance = 0.5 * radius;
  let spatialPad = halfDistance + 1.0e-6 * max(max(abs(start.x), abs(end.x)), max(abs(start.y), abs(end.y)));
  let minimum = min(start, end) - vec2<f32>(spatialPad);
  let maximum = max(start, end) + vec2<f32>(spatialPad);
  let timePad = 1.0e-6 * max(abs(startTime), abs(endTime)) + 1.0e-6 * TIME_CELL_SIZE;
  let minimumTime = min(startTime, endTime) - timePad;
  let maximumTime = max(startTime, endTime) + timePad;
  let firstCell = vec3<i32>(getCellCoordinate(minimum.x, CELL_SIZE), getCellCoordinate(minimum.y, CELL_SIZE),
    getCellCoordinate(minimumTime, TIME_CELL_SIZE));
  let lastCell = vec3<i32>(getCellCoordinate(maximum.x, CELL_SIZE), getCellCoordinate(maximum.y, CELL_SIZE),
    getCellCoordinate(maximumTime, TIME_CELL_SIZE));
  let extent = vec3<f32>(lastCell - firstCell) + vec3<f32>(1.0);
  let cellCount = extent.x * extent.y * extent.z;
  if (!(cellCount <= f32(MAXIMUM_CELLS_PER_SEGMENT))) {
    atomicStore(&state[stateOffset + 2u], 1u);
    return;
  }
  segmentRecords[base] = bitcast<u32>(minimum.x);
  segmentRecords[base + 1u] = bitcast<u32>(minimum.y);
  segmentRecords[base + 2u] = bitcast<u32>(minimumTime);
  segmentRecords[base + 3u] = bitcast<u32>(maximum.x);
  segmentRecords[base + 4u] = bitcast<u32>(maximum.y);
  segmentRecords[base + 5u] = bitcast<u32>(maximumTime);
  segmentRecords[base + 6u] = track;
  segmentRecords[base + 7u] = trackOffsets[trackOffsetsOffset + track];
  segmentCounts[segmentCountsOffset + index] = u32(cellCount);`})}function V(e,t){let{shape:n}=t,r=[z(t.timestamps,n.isWordMode),{name:`trackOffsets`,view:t.trackOffsets,type:`u32`,access:`read`},{name:`maximumGap`,view:t.maximumGap,type:`f32`,access:`read`},{name:`rowSkip`,view:t.rowSkip,type:`u32`,access:`read_write`}];return t.trackValid&&r.push({name:`trackValid`,view:t.trackValid,type:`u32`,access:`read`}),d(e,{id:t.id,operation:I,variant:`row-skip${n.isWordMode?`-words`:``}${t.trackValid?`-valid`:``}`,bindings:r,invocationCount:n.rowCount,declarations:`${R(n)}
${S(n.isWordMode)}
fn findTrack(row: u32) -> u32 {
  if (row < trackOffsets[trackOffsetsOffset] || row >= trackOffsets[trackOffsetsOffset + TRACK_COUNT]) {
    return NONE;
  }
  var low = 0u;
  var high = TRACK_COUNT;
  loop {
    if (low >= high) { break; }
    let middle = (low + high) / 2u;
    if (trackOffsets[trackOffsetsOffset + middle] > row) { high = middle; } else { low = middle + 1u; }
  }
  return low - 1u;
}`,body:`rowSkip[rowSkipOffset + index] = 1u;
  let track = findTrack(index);
  if (track == NONE || index + 1u >= trackOffsets[trackOffsetsOffset + track + 1u]) { return; }
  ${t.trackValid?`if (trackValid[trackValidOffset + track] == 0u) { return; }`:``}
  let gapLimit = maximumGap[maximumGapOffset];
  // A segment longer than the limit is a data gap, not linear motion; equal is still a segment.
  if (gapLimit > 0.0 && rowTimeDifference(index + 1u, index) > gapLimit) { return; }
  rowSkip[rowSkipOffset + index] = 0u;`})}function H(e,t){return d(e,{id:t.id,operation:I,variant:`entry-clear`,bindings:[{name:`entryKeys`,view:t.entryKeys,type:`u32`,access:`read_write`},{name:`entryIdentity`,view:t.entryIdentity,type:`u32`,access:`read_write`}],invocationCount:t.entryKeys.length,body:`entryKeys[entryKeysOffset + index] = 0xffffffffu;
  entryIdentity[entryIdentityOffset + index] = index;`})}function U(e,t){return d(e,{id:t.id,operation:I,variant:`entries`,bindings:[{name:`segmentRecords`,view:t.segmentRecords,type:`u32`,access:`read`},{name:`segmentCounts`,view:t.segmentCounts,type:`u32`,access:`read`},{name:`segmentStarts`,view:t.segmentStarts,type:`u32`,access:`read`},{name:`entryKeys`,view:t.entryKeys,type:`u32`,access:`read_write`},{name:`entryRecords`,view:t.entryRecords,type:`u32`,access:`read_write`}],invocationCount:t.shape.rowCount,declarations:R(t.shape),body:`let count = segmentCounts[segmentCountsOffset + index];
  if (count == 0u) { return; }
  let base = segmentRecordsOffset + index * SEGMENT_WORDS;
  let firstCell = vec3<i32>(
    getCellCoordinate(bitcast<f32>(segmentRecords[base]), CELL_SIZE),
    getCellCoordinate(bitcast<f32>(segmentRecords[base + 1u]), CELL_SIZE),
    getCellCoordinate(bitcast<f32>(segmentRecords[base + 2u]), TIME_CELL_SIZE));
  let lastCell = vec3<i32>(
    getCellCoordinate(bitcast<f32>(segmentRecords[base + 3u]), CELL_SIZE),
    getCellCoordinate(bitcast<f32>(segmentRecords[base + 4u]), CELL_SIZE),
    getCellCoordinate(bitcast<f32>(segmentRecords[base + 5u]), TIME_CELL_SIZE));
  let extent = vec3<u32>(lastCell - firstCell) + vec3<u32>(1u);
  let start = segmentStarts[segmentStartsOffset + index];
  for (var cellIndex = 0u; cellIndex < count; cellIndex++) {
    let slot = start + cellIndex;
    // A wrapped (saturated) prefix or a full scratch stops here; the total reports the overflow.
    if (slot < start || slot >= ENTRY_CAPACITY) { break; }
    let offset = vec3<u32>(cellIndex % extent.x, (cellIndex / extent.x) % extent.y, cellIndex / (extent.x * extent.y));
    let cell = firstCell + vec3<i32>(offset);
    entryKeys[entryKeysOffset + slot] = getCellHash(cell);
    let entry = entryRecordsOffset + slot * ENTRY_WORDS;
    entryRecords[entry] = index;
    entryRecords[entry + 1u] = bitcast<u32>(cell.x);
    entryRecords[entry + 2u] = bitcast<u32>(cell.y);
    entryRecords[entry + 3u] = bitcast<u32>(cell.z);
  }`})}function W(t,n){let{shape:r}=n;return d(t,{id:n.id,operation:I,variant:r.isWordMode?`pairs-words`:`pairs`,bindings:[{name:`sortedEntries`,view:n.sortedEntries,type:`u32`,access:`read`},{name:`entryRecords`,view:n.entryRecords,type:`u32`,access:`read`},{name:`segmentRecords`,view:n.segmentRecords,type:`u32`,access:`read`},{name:`positions`,view:n.positions,type:`f32`,access:`read`},z(n.timestamps,r.isWordMode),{name:`distance`,view:n.distance,type:`f32`,access:`read`},{name:`state`,view:n.state,type:`atomic<u32>`,access:`read_write`},{name:`hits`,view:n.hits,type:`u32`,access:`read_write`}],invocationCount:r.entryCapacity,declarations:`${e}
${R(r)}
${S(r.isWordMode)}
fn getEntryCell(entry: u32) -> vec3<i32> {
  let base = entryRecordsOffset + entry * ENTRY_WORDS;
  return vec3<i32>(bitcast<i32>(entryRecords[base + 1u]), bitcast<i32>(entryRecords[base + 2u]),
    bitcast<i32>(entryRecords[base + 3u]));
}
fn getBoxMinimum(segment: u32) -> vec3<f32> {
  let base = segmentRecordsOffset + segment * SEGMENT_WORDS;
  return vec3<f32>(bitcast<f32>(segmentRecords[base]), bitcast<f32>(segmentRecords[base + 1u]),
    bitcast<f32>(segmentRecords[base + 2u]));
}
fn getBoxMaximum(segment: u32) -> vec3<f32> {
  let base = segmentRecordsOffset + segment * SEGMENT_WORDS;
  return vec3<f32>(bitcast<f32>(segmentRecords[base + 3u]), bitcast<f32>(segmentRecords[base + 4u]),
    bitcast<f32>(segmentRecords[base + 5u]));
}
fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
}
// Position of segment row 'row' at time 'time' measured from row 'origin'.
fn getSegmentPosition(row: u32, origin: u32, time: f32) -> vec2<f32> {
  let startTime = rowTimeDifference(row, origin);
  let duration = rowTimeDifference(row + 1u, row);
  let fraction = clamp((time - startTime) / duration, 0.0, 1.0);
  return mix(getPosition(row), getPosition(row + 1u), fraction);
}
// Tests segments 'first' and 'second' of different tracks and appends a hit when they come within
// the distance during their common time interval.
fn testSegments(first: u32, second: u32, radius: f32) {
  let firstTrack = segmentRecords[segmentRecordsOffset + first * SEGMENT_WORDS + 6u];
  let secondTrack = segmentRecords[segmentRecordsOffset + second * SEGMENT_WORDS + 6u];
  let lowIsFirst = firstTrack < secondTrack;
  let low = select(second, first, lowIsFirst);
  let high = select(first, second, lowIsFirst);
  // Times are measured from the low segment's start row.
  let lowEnd = rowTimeDifference(low + 1u, low);
  let highStart = rowTimeDifference(high, low);
  let highEnd = rowTimeDifference(high + 1u, low);
  let commonStart = max(0.0, highStart);
  let commonEnd = min(lowEnd, highEnd);
  if (commonStart > commonEnd) { return; }
  let separationStart = getSegmentPosition(high, low, commonStart) - getSegmentPosition(low, low, commonStart);
  let separationEnd = getSegmentPosition(high, low, commonEnd) - getSegmentPosition(low, low, commonEnd);
  let span = commonEnd - commonStart;
  var velocity = vec2<f32>(0.0);
  if (span > 0.0) { velocity = (separationEnd - separationStart) / span; }
  let a = dot(velocity, velocity);
  let b = dot(separationStart, velocity);
  let c = dot(separationStart, separationStart) - radius * radius;
  var enter = 0.0;
  var exit = span;
  var closest = 0.0;
  if (a > 0.0) {
    closest = clamp(-b / a, 0.0, span);
    let discriminant = b * b - a * c;
    if (discriminant < 0.0) { return; }
    let root = sqrt(discriminant);
    enter = max((-b - root) / a, 0.0);
    exit = min((-b + root) / a, span);
    if (enter > exit) { return; }
  } else if (c > 0.0) {
    return;
  }
  let nearest = separationStart + velocity * closest;
  let slot = reserveTrajectoryCount();
  if (slot >= HIT_CAPACITY) { return; }
  // Report times relative to the first sample of the lower track.
  let lowTrack = segmentRecords[segmentRecordsOffset + low * SEGMENT_WORDS + 6u];
  let lowFirstRow = segmentRecords[segmentRecordsOffset + low * SEGMENT_WORDS + 7u];
  let base = rowTimeDifference(low, lowFirstRow) + commonStart;
  let highTrack = segmentRecords[segmentRecordsOffset + high * SEGMENT_WORDS + 6u];
  let hit = hitsOffset + slot * HIT_WORDS;
  hits[hit] = lowTrack;
  hits[hit + 1u] = highTrack;
  hits[hit + 2u] = bitcast<u32>(max(base + enter, 0.0));
  hits[hit + 3u] = bitcast<u32>(max(base + exit, 0.0));
  hits[hit + 4u] = bitcast<u32>(length(nearest));
  hits[hit + 5u] = bitcast<u32>(max(base + closest, 0.0));
}`,body:`let entryTotal = min(atomicLoad(&state[stateOffset + 1u]), ENTRY_CAPACITY);
  if (index >= entryTotal) { return; }
  let radius = distance[distanceOffset];
  let entry = sortedEntries[sortedEntriesOffset + index];
  let cell = getEntryCell(entry);
  let hash = getCellHash(cell);
  let segment = entryRecords[entryRecordsOffset + entry * ENTRY_WORDS];
  let track = segmentRecords[segmentRecordsOffset + segment * SEGMENT_WORDS + 6u];
  let boxMinimum = getBoxMinimum(segment);
  let boxMaximum = getBoxMaximum(segment);
  for (var next = index + 1u; next < entryTotal; next++) {
    let other = sortedEntries[sortedEntriesOffset + next];
    let otherCell = getEntryCell(other);
    if (getCellHash(otherCell) != hash) { break; }
    if (any(otherCell != cell)) { continue; }
    let otherSegment = entryRecords[entryRecordsOffset + other * ENTRY_WORDS];
    if (segmentRecords[segmentRecordsOffset + otherSegment * SEGMENT_WORDS + 6u] == track) { continue; }
    let corner = max(boxMinimum, getBoxMinimum(otherSegment));
    let farCorner = min(boxMaximum, getBoxMaximum(otherSegment));
    if (any(corner > farCorner)) { continue; }
    let referenceCell = vec3<i32>(getCellCoordinate(corner.x, CELL_SIZE), getCellCoordinate(corner.y, CELL_SIZE),
      getCellCoordinate(corner.z, TIME_CELL_SIZE));
    if (any(referenceCell != cell)) { continue; }
    testSegments(segment, otherSegment, radius);
  }`})}function G(e,t){let n=t.variant===`enter`?2:+(t.variant===`high`),r=t.variant===`low`?`TRACK_COUNT`:`0u`,i=[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`hits`,view:t.hits,type:`u32`,access:`read`}];return t.order&&i.push({name:`order`,view:t.order,type:`u32`,access:`read`}),t.identity&&i.push({name:`identity`,view:t.identity,type:`u32`,access:`read_write`}),i.push({name:`keys`,view:t.keys,type:`u32`,access:`read_write`}),d(e,{id:t.id,operation:I,variant:`sort-key-${t.variant}`,bindings:i,invocationCount:t.shape.hitCapacity,declarations:R(t.shape),body:`let activeCount = min(state[stateOffset], HIT_CAPACITY);
  ${t.identity?`identity[identityOffset + index] = index;`:``}
  let slot = ${t.order?`order[orderOffset + index]`:`index`};
  keys[keysOffset + index] = select(${r}, hits[hitsOffset + slot * HIT_WORDS + ${n}u], slot < activeCount);`})}function K(e,t){return d(e,{id:t.id,operation:I,variant:`sorted-hits`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`order`,view:t.order,type:`u32`,access:`read`},{name:`hits`,view:t.hits,type:`u32`,access:`read`},{name:`sortedLow`,view:t.sortedLow,type:`u32`,access:`read`},{name:`sortedHigh`,view:t.sortedHigh,type:`u32`,access:`read_write`},{name:`sortedValues`,view:t.sortedValues,type:`f32`,access:`read_write`},{name:`flags`,view:t.flags,type:`u32`,access:`read_write`},{name:`rowIds`,view:t.rowIds,type:`u32`,access:`read_write`}],invocationCount:t.shape.hitCapacity,declarations:R(t.shape),body:`rowIds[rowIdsOffset + index] = index;
  let activeCount = min(state[stateOffset], HIT_CAPACITY);
  if (index >= activeCount) {
    sortedHigh[sortedHighOffset + index] = NONE;
    for (var word = 0u; word < 4u; word++) { sortedValues[sortedValuesOffset + index * 4u + word] = 0.0; }
    flags[flagsOffset + index] = 0u;
    return;
  }
  let hit = hitsOffset + order[orderOffset + index] * HIT_WORDS;
  let high = hits[hit + 1u];
  sortedHigh[sortedHighOffset + index] = high;
  for (var word = 0u; word < 4u; word++) {
    sortedValues[sortedValuesOffset + index * 4u + word] = bitcast<f32>(hits[hit + 2u + word]);
  }
  // The first hit of each (low, high) run starts a pair. The previous high is read from the hit
  // table, not from sortedHigh, which other invocations are writing.
  var isStart = index == 0u;
  if (!isStart) {
    let previous = hitsOffset + order[orderOffset + index - 1u] * HIT_WORDS;
    isStart = sortedLow[sortedLowOffset + index] != sortedLow[sortedLowOffset + index - 1u] ||
      hits[previous + 1u] != high;
  }
  flags[flagsOffset + index] = select(0u, 1u, isStart);`})}function te(e,t){let{columns:n}=t,r=R(t.shape),i=`let isLive = index < min(runCount[runCountOffset], PAIR_CAPACITY);
  let start = select(0u, runStarts[runStartsOffset + index], isLive);`,a=[d(e,{id:`${t.id}-identity`,operation:I,variant:`reduce-identity`,bindings:[{name:`runStarts`,view:t.runStarts,type:`u32`,access:`read`},{name:`runCount`,view:t.runCount,type:`u32`,access:`read`},{name:`sortedLow`,view:t.sortedLow,type:`u32`,access:`read`},{name:`sortedHigh`,view:t.sortedHigh,type:`u32`,access:`read`},{name:`ids`,view:n.ids,type:`u32`,access:`read_write`},...n.partners?[{name:`partners`,view:n.partners,type:`u32`,access:`read_write`}]:[]],invocationCount:t.shape.pairCapacity,declarations:r,body:`${i}
  ids[idsOffset + index] = select(NONE, sortedLow[sortedLowOffset + start], isLive);
  ${n.partners?`partners[partnersOffset + index] = select(NONE, sortedHigh[sortedHighOffset + start], isLive);`:``}`})],o=[[`minimumDistances`,n.minimumDistances,`minimumDistance`],[`closestTimes`,n.closestTimes,`closestTime`],[`firstTimes`,n.firstTimes,`firstTime`],[`durations`,n.durations,`duration`]].filter(([,e])=>e);return o.length>0&&a.push(d(e,{id:`${t.id}-statistics`,operation:I,variant:`reduce-statistics-${o.map(([e])=>e).join(`-`)}`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`runStarts`,view:t.runStarts,type:`u32`,access:`read`},{name:`runCount`,view:t.runCount,type:`u32`,access:`read`},{name:`sortedValues`,view:t.sortedValues,type:`f32`,access:`read`},...o.map(([e,t])=>({name:e,view:t,type:`f32`,access:`read_write`}))],invocationCount:t.shape.pairCapacity,declarations:r,body:`${i}
  var end = start;
  if (isLive) {
    end = min(state[stateOffset], HIT_CAPACITY);
    if (index + 1u < min(runCount[runCountOffset], PAIR_CAPACITY)) { end = runStarts[runStartsOffset + index + 1u]; }
  }
  // Hits of one pair are in enter-time order.
  var firstTime = 0.0;
  var minimumDistance = 0.0;
  var closestTime = 0.0;
  var duration = 0.0;
  if (isLive) {
    firstTime = sortedValues[sortedValuesOffset + start * 4u];
    minimumDistance = 3.402823466e+38;
  }
  for (var hit = start; hit < end; hit++) {
    let base = sortedValuesOffset + hit * 4u;
    let hitDistance = sortedValues[base + 2u];
    let hitClosest = sortedValues[base + 3u];
    if (hitDistance < minimumDistance || (hitDistance == minimumDistance && hitClosest < closestTime)) {
      minimumDistance = hitDistance;
      closestTime = hitClosest;
    }
    duration = duration + (sortedValues[base + 1u] - sortedValues[base]);
  }
  ${o.map(([e,,t])=>`${e}[${e}Offset + index] = ${t};`).join(`
  `)}`})),a}function ne(e,t){return d(e,{id:t.id,operation:I,variant:t.candidateOverflow?`status-candidate`:`status`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`entryOverflow`,view:t.entryOverflow,type:`u32`,access:`read_write`},{name:`hitOverflow`,view:t.hitOverflow,type:`u32`,access:`read_write`},...t.candidateOverflow?[{name:`candidateOverflow`,view:t.candidateOverflow,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:R(t.shape),body:`let entryOverflowed = state[stateOffset + 1u] > ENTRY_CAPACITY || state[stateOffset + 2u] != 0u;
  let hitOverflowed = state[stateOffset] > HIT_CAPACITY;
  entryOverflow[entryOverflowOffset] = select(0u, 1u, entryOverflowed);
  hitOverflow[hitOverflowOffset] = select(0u, 1u, hitOverflowed);${t.candidateOverflow?`
  candidateOverflow[candidateOverflowOffset] = select(0u, 1u, entryOverflowed || hitOverflowed);`:``}`})}var q=Object.freeze({...b,measure:`distance`}),J=`GPUTrajectoryClosestApproach`,Y=class{id;props;spatialContext;trackCount;constructor(e){this.id=e.id??`trajectory-closest-approach`;let r=e;this.props=r;let{id:a}=this;if(this.spatialContext=x(a,r.spatialContext,q),m(r.positions,[`float32x2`],`${a} positions`),m(r.timestamps,[`float32`,`uint32x2`],`${a} timestamps`),m(r.trackOffsets,[`uint32`],`${a} trackOffsets`),r.timestamps.length!==r.positions.length)throw Error(`${a} timestamps length must equal positions length`);if(r.trackOffsets.length<2)throw Error(`${a} trackOffsets must contain at least two rows`);if(this.trackCount=r.trackOffsets.length-1,r.mask&&(m(r.mask,[`uint32`],`${a} mask`),r.mask.length!==this.trackCount))throw Error(`${a} mask length must equal the track count`);if(m(r.distance,[`float32`],`${a} distance`),r.distance.length<1)throw Error(`${a} distance must contain one float32 row`);r.maximumGap&&m(r.maximumGap,[`float32`],`${a} maximumGap`),C(a,r.timestamps,r.temporalContext);for(let[e,t]of[[`cellSize`,r.cellSize],[`timeCellSize`,r.timeCellSize]])if(!(Number.isFinite(Math.fround(t))&&Math.fround(t)>0))throw Error(`${a} ${e} must be a positive finite number`);h(a,`entryCapacity`,r.entryCapacity,1,t),h(a,`hitCapacity`,r.hitCapacity,1,t);let{pairs:o,status:s}=r.output;g(a,o);for(let[e,t]of[[`status.entryOverflow`,s.entryOverflow],[`status.hitOverflow`,s.hitOverflow],[`status.candidateOverflow`,s.candidateOverflow]])if(t!==void 0&&(m(t,[`uint32`],`${a} ${e}`),t.length<1))throw Error(`${a} ${e} must hold one row`);let c=o.ids.length;for(let[e,t,n]of[[`partners`,o.partners,`uint32`],[`minimumDistances`,o.minimumDistances,`float32`],[`closestTimes`,o.closestTimes,`float32`],[`firstTimes`,o.firstTimes,`float32`],[`durations`,o.durations,`float32`]])if(t&&(m(t,[n],`${a} pairs.${e}`),t.length!==c))throw Error(`${a} pairs.${e} length must equal the capacity`);i(this.id,this.props.output),n(a,[{ids:o.ids,partners:o.partners},{minimumDistances:o.minimumDistances,closestTimes:o.closestTimes,firstTimes:o.firstTimes,durations:o.durations},{...r.output.status},{...o.status}]),y(a,this.getOutputs(),this.getInputs())}getInputs(){let{props:e}=this;return[e.positions,e.timestamps,e.trackOffsets,e.mask,e.distance,e.maximumGap]}getOutputs(){let{pairs:e,status:t}=this.props.output;return[e.ids,e.status.count,e.status.overflow,e.status.candidateOverflow,e.status.requiredCount,e.partners,e.minimumDistances,e.closestTimes,e.firstTimes,e.durations,t.entryOverflow,t.hitOverflow,t.candidateOverflow]}getCommandNodes(e){r(e,this.id,this.props);let{id:t,props:n,trackCount:i}=this,{pairs:l,status:d}=n.output;v(t,e,[...this.getInputs(),...this.getOutputs()]);let m=n.positions.length;for(let[r,i,a]of[[`segment records`,m*8*4,`split the tracks`],[`cell entries`,n.entryCapacity*4*4,`lower entryCapacity`],[`segment-pair hits`,n.hitCapacity*6*4,`lower hitCapacity`]])_(e.device,t,r,i,a);let h={rowCount:m,trackCount:i,cellSize:n.cellSize,timeCellSize:n.timeCellSize,entryCapacity:n.entryCapacity,hitCapacity:n.hitCapacity,pairCapacity:l.ids.length,isWordMode:n.timestamps.format===`uint32x2`},g=(n,r)=>a(e,`${t}-${n}`,`uint32`,Math.max(r,1)),y=(n,r)=>a(e,`${t}-${n}`,`float32`,Math.max(r,1)),b=g(`state`,3),x=d.candidateOverflow??g(`candidate-overflow`,1),S=g(`segment-records`,m*8),C=g(`segment-counts`,m),w=g(`segment-starts`,m),T=g(`entry-keys`,n.entryCapacity),E=g(`entry-identity`,n.entryCapacity),D=g(`entry-records`,n.entryCapacity*4),O=g(`sorted-entry-keys`,n.entryCapacity),k=g(`sorted-entries`,n.entryCapacity),A=g(`hits`,n.hitCapacity*6),j=n.maximumGap?g(`row-skip`,m):void 0,M=[...n.maximumGap&&j?[V(e,{id:`${t}-row-skip`,shape:h,timestamps:n.timestamps,trackOffsets:n.trackOffsets,trackValid:n.mask,maximumGap:n.maximumGap,rowSkip:j})]:[],f(e,{id:`${t}-clear`,operation:J,view:b,type:`u32`,value:`0u`}),B(e,{id:`${t}-segments`,shape:h,positions:n.positions,timestamps:n.timestamps,trackOffsets:n.trackOffsets,trackValid:n.mask,distance:n.distance,rowSkip:j,segmentRecords:S,segmentCounts:C,state:b}),...new s({id:`${t}-segment-scan`,input:C,output:w,mode:`exclusive`}).getCommandNodes(e),...p(e,{id:`${t}-entry`,operation:J,counts:C,offsets:w,rowCount:m,total:b,totalRow:1}),H(e,{id:`${t}-entry-clear`,entryKeys:T,entryIdentity:E}),U(e,{id:`${t}-entries`,shape:h,segmentRecords:S,segmentCounts:C,segmentStarts:w,entryKeys:T,entryRecords:D}),...new c({id:`${t}-entry-sort`,keys:T,values:E,outputKeys:O,outputValues:k}).getCommandNodes(e),W(e,{id:`${t}-pairs`,shape:h,sortedEntries:k,entryRecords:D,segmentRecords:S,positions:n.positions,timestamps:n.timestamps,distance:n.distance,state:b,hits:A})],N=n.hitCapacity,P=g(`hit-identity`,N),F=ee(i),I,L;for(let[n,r]of[[`enter`,32],[`high`,F],[`low`,F]]){let i=g(`${n}-keys`,N),a=g(`${n}-sorted-keys`,N),o=g(`${n}-order`,N);M.push(G(e,{id:`${t}-${n}-keys`,shape:h,variant:n,state:b,hits:A,order:I,identity:I?void 0:P,keys:i}),...new c({id:`${t}-${n}-sort`,keys:i,values:I??P,outputKeys:a,outputValues:o,keyBits:r}).getCommandNodes(e)),I=o,L=a}let R=I,z=L,q=g(`sorted-high`,N),Y=y(`sorted-values`,N*4),X=g(`run-flags`,N),Z=g(`row-ids`,N),Q=g(`run-starts`,N),$=g(`run-count`,1);return M.push(K(e,{id:`${t}-sorted-hits`,shape:h,state:b,order:R,hits:A,sortedLow:z,sortedHigh:q,sortedValues:Y,flags:X,rowIds:Z}),...new o({id:`${t}-runs`,input:Z,flags:X,output:Q,count:$}).getCommandNodes(e),...te(e,{id:`${t}-reduce`,shape:h,state:b,runStarts:Q,runCount:$,sortedLow:z,sortedHigh:q,sortedValues:Y,columns:{ids:l.ids,partners:l.partners,minimumDistances:l.minimumDistances,closestTimes:l.closestTimes,firstTimes:l.firstTimes,durations:l.durations}}),ne(e,{id:`${t}-status`,shape:h,state:b,entryOverflow:d.entryOverflow,hitOverflow:d.hitOverflow,candidateOverflow:x}),u(e,{id:`${t}-publish`,operation:J,requiredCount:$,output:l,candidateOverflowSources:l.status.candidateOverflow?[x]:[]})),M}},X=64,Z=[.06,.05,.94,.38],Q=w({parameters:F,caption:`Exact time spent within the distance, per pair of tracks.`,createInputs:({world:e})=>{let t=O(k(e,{trackCount:14,skipTracks:4,minimumFixes:24,maximumFixes:34}),.42);return{...t,busiestFix:P(t,.12)}},build(e,t){let n=e.allocate(`distance`,`float32`,1),r=e.allocate(`pair-ids`,`uint32`,X),i=e.allocate(`pair-count`,`uint32`,1),a=e.allocate(`pair-overflow`,`uint32`,1),o=e.allocate(`durations`,`float32`,X),s=e.allocate(`partners`,`uint32`,X),c=e.allocate(`closest-times`,`float32`,X),l=e.allocate(`entry-overflow`,`uint32`,1),u=e.allocate(`hit-overflow`,`uint32`,1),d=e.upload(`positions`,`float32x2`,t.positions),f=e.upload(`timestamps`,`float32`,t.timestamps),p=e.upload(`offsets`,`uint32`,t.trackOffsets);e.graph.add(new Y({positions:d.view,timestamps:f.view,trackOffsets:p.view,distance:n.view,cellSize:.16,timeCellSize:60,entryCapacity:65536,hitCapacity:65536,output:{pairs:{ids:r.view,status:{count:i.view,overflow:a.view},partners:s.view,closestTimes:c.view,durations:o.view},status:{entryOverflow:l.view,hitOverflow:u.view}}}));let m=A(e,{ids:r,partners:s,closestTimes:c,count:i,positions:d,timestamps:f,trackOffsets:p,pairCapacity:X});return{outputs:{links:m.links,linkEnds:m.ends,ids:r,count:i,partners:s,closestTimes:c,durations:o,nodes:e.upload(`nodes`,`float32x2`,t.positions),segments:e.upload(`segments`,`float32x4`,t.segments)},update:e=>n.write(Float32Array.of(e.distance))}},interaction:{kind:`probe`,hint:`hover a track`,rest:({toUnit:e,inputs:t})=>e(t.busiestFix),resolve(e,{inputs:t,outputs:n,values:r,peek:i,world:a}){let{track:o}=M(t,e.x,e.y),s=i(n.ids),c=i(n.partners),l=i(n.closestTimes),u=i(n.durations),d=i(n.count),f=[{shape:`polyline`,points:j(t,o)}];if(!s||!c||!l||!u||!d)return{glyphs:f};let p=-1;for(let e=0;e<Math.min(d[0],s.length);e++)(s[e]===o||c[e]===o)&&(p<0||u[e]>u[p])&&(p=e);if(p<0)return{glyphs:f,readout:`track ${o} · none within ${r.distance}`};let m=s[p]===o?c[p]:s[p];f.push({shape:`polyline`,points:j(t,m),dashed:!0});let h=N(t,o,l[p]);h&&f.push({shape:`ring`,at:h,radius:r.distance});let g=T(Z,a,(p+.5)/X);return f.push({shape:`dot`,at:[g,(Z[1]-.025)*a.height]}),{glyphs:f,readout:`tracks ${o} & ${m} · ${Math.round(u[p])} s within ${r.distance}`}}},draw:({inputs:e,outputs:t,theme:n,world:r})=>[{layer:`lines`,context:!0,segments:t.segments,count:e.segmentCount},{layer:`dots`,context:!0,positions:t.nodes,radius:3},{layer:`lines`,segments:t.links,radius:2.6,color:[...n.categories[1],1]},{layer:`dots`,positions:t.linkEnds,radius:6,color:[...n.categories[1],1]},{layer:`chart`,kind:`bars`,values:t.durations,ramp:`sequential`,rectangle:[...Z]},{layer:`text`,...E({labels:[{text:`bars: time each pair spent within the distance, one bar per pair`,at:[.06*r.width,.385],anchor:`bottom-left`,role:`title`}]},D({corner:`top-left`,entries:[{label:`raw fix`,swatch:[...n.muted]},{label:`closest approach of a pair (between fixes)`,swatch:1}]}))}]});export{Q as default};