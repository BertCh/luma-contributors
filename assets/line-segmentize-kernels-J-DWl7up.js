import{t as e}from"./scan-total-C4QddMmi.js";import{t}from"./addition-rounding-error-wgsl-DVwPAqKC.js";import{t as n}from"./lon-lat-ring-wgsl-DpZBIfuE.js";import{c as r,o as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as a}from"./geodesic-wgsl-I0CYDKcz.js";import{T as o}from"./gpu-contributor-utils-BIwLQON5.js";function s(t){return`${e}
const BALANCED_ROWS: u32 = ${t}u;
const BALANCED_LANES: u32 = 128u;
const BALANCED_NONE: u32 = 0xffffffffu;
var<workgroup> balancedWindow: array<u32, 2>;

// Last row in [low, high] whose output range starts at or before the slot.
fn balancedLastAtMost(slot: u32, lowRow: u32, highRow: u32) -> u32 {
  var low = lowRow;
  var high = highRow;
  while (low < high) {
    let middle = (low + high + 1u) >> 1u;
    if (balancedStart(middle) <= slot) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  return low;
}

fn findBalancedOwner(slot: u32, firstSlot: u32, lane: u32) -> u32 {
  if (lane == 0u) {
    let lowRow = balancedLastAtMost(firstSlot, 0u, BALANCED_ROWS - 1u);
    balancedWindow[0] = lowRow;
    balancedWindow[1] = balancedLastAtMost(firstSlot + (BALANCED_LANES - 1u), lowRow, BALANCED_ROWS - 1u);
  }
  let windowLow = workgroupUniformLoad(&balancedWindow[0]);
  let windowHigh = workgroupUniformLoad(&balancedWindow[1]);
  let total = getSaturatedScanSum(balancedStart(BALANCED_ROWS - 1u), balancedCount(BALANCED_ROWS - 1u));
  if (slot >= total) {
    return BALANCED_NONE;
  }
  return balancedLastAtMost(slot, windowLow, windowHigh);
}
`}var c=`
const NO_LINE: u32 = 0xffffffffu;

fn findLine(row: u32) -> u32 {
  var low = 0u;
  var high = LINE_COUNT + 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] <= row) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low == 0u || low > LINE_COUNT) {
    return NO_LINE;
  }
  return low - 1u;
}
`;function l(e,t){let n=e===`spherical`?`return geodesicCentralAngle(a, b) * RADIUS;`:`return length(b - a);`;return`
${e===`spherical`?a:``}
const RADIUS: f32 = ${r(t)};

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getSegmentLength(a: vec2<f32>, b: vec2<f32>) -> f32 {
  ${n}
}
`}var u=64;function d(e,r){o(r.id,!(r.rowTurns&&r.rowShifts),`line prefix takes rowTurns or rowShifts, not both`);let a=r.rowTurns??r.rowShifts,s=r.rowTurns?`i32`:`f32`,c=r.rowTurns?`0i`:`0.0`,d=r.rowTurns?`-i32(round((current.x - previous.x) / 360.0))`:`-360.0 * round((current.x - previous.x) / 360.0)`,f=r.polygonEdges?`select(${d}, ${c}, lonLatIsPoleEdge(previous, current))`:d,p=r.rowTurns?`turnCarry[0]`:`prefixCarry[2]`,m=r.lineOffsets.length-1,h=[{name:`positions`,view:r.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:r.lineOffsets,type:`u32`,access:`read`}];r.rowMeasures&&h.push({name:`rowMeasures`,view:r.rowMeasures,type:`f32`,access:`read_write`}),a&&h.push({name:`rowShifts`,view:a,type:s,access:`read_write`});let{rowMeasures:g}=r,_=a;return i(e,{id:r.id,operation:r.operation,variant:`line-prefix`,bindings:h,invocationCount:Math.ceil(m/u)*u,workgroupSize:u,guardIndex:!1,declarations:`const LINE_COUNT: u32 = ${m}u;
const ROW_COUNT: u32 = ${r.positions.length}u;
const PREFIX_LANES: u32 = ${u}u;
var<workgroup> largeBits: array<atomic<u32>, 2>;
var<workgroup> largeBitsCopy: array<u32, 2>;
var<workgroup> largeStart: array<u32, ${u}>;
var<workgroup> largeEnd: array<u32, ${u}>;
var<workgroup> prefixCarry: array<f32, 3>;
var<workgroup> tileLengths: array<f32, ${u}>;
var<workgroup> tileShifts: array<${s}, ${u}>;
${r.rowTurns?`var<workgroup> turnCarry: array<i32, 1>;`:``}
${t}
${l(r.coordinateSystem,r.radius)}
${r.polygonEdges?n:``}`,body:`let lane = localInvocationIndex;
  let lineBase = workgroupIndex * PREFIX_LANES;
  let ownLine = lineBase + lane;
  // Phase 1: this lane's own line, serially when small; larger lines are queued for phase 2.
  if (ownLine < LINE_COUNT) {
    let first = min(lineOffsets[lineOffsetsOffset + ownLine], ROW_COUNT);
    let last = min(max(lineOffsets[lineOffsetsOffset + ownLine + 1u], first), ROW_COUNT);
    if (last > first) {
      if (last - first <= PREFIX_LANES) {
        var sum = 0.0;
        var compensation = 0.0;
        var shift = ${c};
        var previous = getPosition(first);
        ${g?`rowMeasures[rowMeasuresOffset + first] = 0.0;`:``}
        ${_?`rowShifts[rowShiftsOffset + first] = ${c};`:``}
        for (var row = first + 1u; row < last; row++) {
          let current = getPosition(row);
          ${g?`let segmentLength = getSegmentLength(previous, current);
          let nextSum = sum + segmentLength;
          compensation = compensation + getAdditionRoundingError(sum, segmentLength, nextSum);
          sum = nextSum;
          rowMeasures[rowMeasuresOffset + row] = sum + compensation;`:``}
          ${_?`shift = shift + ${f};
          rowShifts[rowShiftsOffset + row] = shift;`:``}
          previous = current;
        }
      } else {
        largeStart[lane] = first;
        largeEnd[lane] = last;
        atomicOr(&largeBits[lane / 32u], 1u << (lane % 32u));
      }
    }
  }
  workgroupBarrier();
  if (lane == 0u) {
    largeBitsCopy[0] = atomicLoad(&largeBits[0]);
    largeBitsCopy[1] = atomicLoad(&largeBits[1]);
  }
  // Phase 2: every barrier below is in workgroup-uniform control flow (the queue is read through
  // workgroupUniformLoad), and the queued lines are visited in ascending order.
  for (var word = 0u; word < 2u; word++) {
    var pending = workgroupUniformLoad(&largeBitsCopy[word]);
    while (pending != 0u) {
      let slot = word * 32u + firstTrailingBit(pending);
      pending = pending & (pending - 1u);
      let lineStart = workgroupUniformLoad(&largeStart[slot]);
      let lineEnd = workgroupUniformLoad(&largeEnd[slot]);
      if (lane == 0u) {
        ${g?`rowMeasures[rowMeasuresOffset + lineStart] = 0.0;`:``}
        ${_?`rowShifts[rowShiftsOffset + lineStart] = ${c};`:``}
        prefixCarry[0] = 0.0;
        prefixCarry[1] = 0.0;
        ${p} = ${c};
      }
      for (var tileStart = lineStart + 1u; tileStart < lineEnd; tileStart += PREFIX_LANES) {
        let row = tileStart + lane;
        var segmentLength = 0.0;
        var segmentShift = ${c};
        if (row < lineEnd) {
          let previous = getPosition(row - 1u);
          let current = getPosition(row);
          ${g?`segmentLength = getSegmentLength(previous, current);`:``}
          ${_?`segmentShift = ${f};`:``}
        }
        tileLengths[lane] = segmentLength;
        tileShifts[lane] = segmentShift;
        workgroupBarrier();
        for (var offset = 1u; offset < PREFIX_LANES; offset = offset << 1u) {
          var addedLength = 0.0;
          var addedShift = ${c};
          if (lane >= offset) {
            addedLength = tileLengths[lane - offset];
            addedShift = tileShifts[lane - offset];
          }
          workgroupBarrier();
          tileLengths[lane] += addedLength;
          tileShifts[lane] += addedShift;
          workgroupBarrier();
        }
        if (row < lineEnd) {
          ${g?`rowMeasures[rowMeasuresOffset + row] = (prefixCarry[0] + tileLengths[lane]) + prefixCarry[1];`:``}
          ${_?`rowShifts[rowShiftsOffset + row] = ${p} + tileShifts[lane];`:``}
        }
        workgroupBarrier();
        if (lane == 0u) {
          // Neumaier-compensated carry of the tile total into the running measure.
          let total = tileLengths[PREFIX_LANES - 1u];
          let sum = prefixCarry[0];
          let nextSum = sum + total;
          prefixCarry[1] += getAdditionRoundingError(sum, total, nextSum);
          prefixCarry[0] = nextSum;
          ${p} += tileShifts[PREFIX_LANES - 1u];
        }
        workgroupBarrier();
      }
    }
  }`})}function f(e,t){return i(e,{id:t.id,operation:t.operation,variant:`clear-approximation`,bindings:[{name:`approximation`,view:t.approximation,type:`u32`,access:`read_write`}],invocationCount:1,body:`approximation[approximationOffset] = 0u;`})}function p(e,t){let n=t.parameterIndex??0;o(t.id,Number.isInteger(n)&&n>=0&&n<t.parameters.length,`parameterIndex must be an integer in [0, ${t.parameters.length-1}]`);let r=(t.interpolation??(t.coordinateSystem===`spherical`?`great-circle`:`planar`))===`great-circle`,a=t.lineOffsets.length-1;return i(e,{id:t.id,operation:t.operation,variant:`count`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`counts`,view:t.counts,type:`u32`,access:`read_write`},...t.approximation?[{name:`approximation`,view:t.approximation,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:t.positions.length,declarations:`const LINE_COUNT: u32 = ${a}u;
const MAXIMUM_PIECES: u32 = ${t.maximumPieceCount}u;
${c}
${l(r||t.polygonEdges?`spherical`:t.coordinateSystem,t.radius)}

fn getPieceExtent(start: vec2<f32>, end: vec2<f32>) -> f32 {
  ${t.pieceCountModel===`degree-step`?`let rawLongitude = end.x - start.x;
  let longitudeStep = rawLongitude - 360.0 * round(rawLongitude / 360.0);
  return max(abs(longitudeStep), ${r||t.polygonEdges?`geodesicCentralAngle(start, end) * GEODESIC_RADIANS_TO_DEGREES`:`abs(end.y - start.y)`});`:`return getSegmentLength(start, end);`}
}

fn getPieceCount(segmentLength: f32, maximumLength: f32) -> u32 {
  if (!(maximumLength > 0.0) || !(segmentLength > 0.0)) {
    return 1u;
  }
  let requestedPieces = ceil(segmentLength / maximumLength);
  ${t.approximation?`if (requestedPieces > f32(MAXIMUM_PIECES)) {
    atomicStore(&approximation[approximationOffset], 1u);
  }`:``}
  let pieces = min(requestedPieces, f32(MAXIMUM_PIECES));
  return clamp(u32(pieces), 1u, MAXIMUM_PIECES);
}`,body:`let line = findLine(index);
  var count = 0u;
  if (line != NO_LINE) {
    let first = lineOffsets[lineOffsetsOffset + line];
    let last = min(lineOffsets[lineOffsetsOffset + line + 1u], ${t.positions.length}u);
    ${t.closedLines?`if (last > first + 1u && index + 1u == last &&
        all(bitcast<vec2<u32>>(getPosition(index)) == bitcast<vec2<u32>>(getPosition(first)))) {
      counts[countsOffset + index] = 0u;
      return;
    }`:``}
    if (index + 1u >= last ${t.closedLines?`&& last <= first + 1u`:``}) {
      count = 1u;
    } else {
      count = getPieceCount(
        getPieceExtent(getPosition(index), getPosition(${t.closedLines?`select(index + 1u, first, index + 1u >= last)`:`index + 1u`})),
        parameters[parametersOffset + ${n}u]
      );
    }
  }
  counts[countsOffset + index] = count;`})}function m(e,t){o(t.id,!t.closedLines||t.lineOffsets!==void 0,`closedLines needs lineOffsets`),o(t.id,!t.outputTurns||t.outputTurns.length>=t.outputPositions.length,`outputTurns must have at least ${t.outputPositions.length} rows (one per output position)`);let r=t.interpolation??(t.coordinateSystem===`spherical`?`great-circle`:`planar`),a=r===`great-circle`,u=r!==`planar`,d=!!(t.wrapOutputLongitudes||t.outputTurns),f=t.rowTurns??t.rowShifts,p=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`counts`,view:t.counts,type:`u32`,access:`read`},{name:`starts`,view:t.starts,type:`u32`,access:`read`},{name:`outputPositions`,view:t.outputPositions,type:`f32`,access:`read_write`}];t.closedLines&&t.lineOffsets&&p.push({name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`}),t.outputTurns&&p.push({name:`outputTurns`,view:t.outputTurns,type:`i32`,access:`read_write`}),u&&f&&p.push({name:`rowShifts`,view:f,type:t.rowTurns?`i32`:`f32`,access:`read`}),t.outputMeasures&&t.rowMeasures&&p.push({name:`rowMeasures`,view:t.rowMeasures,type:`f32`,access:`read`},{name:`outputMeasures`,view:t.outputMeasures,type:`f32`,access:`read_write`}),t.outputSourceRows&&p.push({name:`outputSourceRows`,view:t.outputSourceRows,type:`u32`,access:`read_write`});let m=!!(t.outputMeasures&&t.rowMeasures),h=a?`geodesicInterpolate(start, end, angle, fraction)`:r===`linear-degrees`?`vec2<f32>(start.x + (end.x - start.x - 360.0 * round((end.x - start.x) / 360.0)) * fraction, mix(start.y, end.y, fraction))`:`mix(start, end, fraction)`,g=t.polygonEdges?`interpolatePolygonEdge(start, end, fraction)`:h;return i(e,{id:t.id,operation:t.operation,variant:`emit`,bindings:p,invocationCount:t.outputPositions.length,workgroupSize:128,guardIndex:!1,declarations:`const CAPACITY: u32 = ${t.outputPositions.length}u;
${l(a?`spherical`:t.coordinateSystem,t.radius)}
${t.polygonEdges?`${n}
fn interpolatePolygonEdge(start: vec2<f32>, end: vec2<f32>, fraction: f32) -> vec2<f32> {
  if (lonLatIsPoleEdge(start, end)) { return mix(start, end, fraction); }
  ${a?`let angle = geodesicCentralAngle(start, end);`:``}
  return ${h};
}`:``}
${t.closedLines?`const LINE_COUNT: u32 = ${t.lineOffsets.length-1}u;
${c}`:``}
${s(t.positions.length)}
fn balancedStart(row: u32) -> u32 { return starts[startsOffset + row]; }
fn balancedCount(row: u32) -> u32 { return counts[countsOffset + row]; }

fn writeVertex(outputRow: u32, vertex: vec2<f32>, measure: f32, sourceRow: u32, prefixTurn: i32) {
  if (outputRow >= CAPACITY) {
    return;
  }
  var longitude = vertex.x;
  var turn = prefixTurn;
  ${d?`// Preserve canonical input endpoints exactly, including signed zero and both seam signs.
  if (longitude < -180.0 || longitude > 180.0) {
    let localTurn = i32(floor((longitude + 180.0) / 360.0));
    longitude -= 360.0 * f32(localTurn);
    turn += localTurn;
  }`:``}
  outputPositions[outputPositionsOffset + 2u * outputRow] = longitude;
  ${t.outputTurns?`outputTurns[outputTurnsOffset + outputRow] = turn;`:``}
  outputPositions[outputPositionsOffset + 2u * outputRow + 1u] = vertex.y;
  ${m?`outputMeasures[outputMeasuresOffset + outputRow] = measure;`:``}
  ${t.outputSourceRows?`outputSourceRows[outputSourceRowsOffset + outputRow] = sourceRow;`:``}
}`,body:`let row = findBalancedOwner(index, workgroupIndex * BALANCED_LANES, localInvocationIndex);
  if (row == BALANCED_NONE) {
    return;
  }
  let count = counts[countsOffset + row];
  let piece = index - starts[startsOffset + row];
  var start = getPosition(row);
  let prefixTurn = ${u&&f?t.rowTurns?`rowShifts[rowShiftsOffset + row]`:`i32(rowShifts[rowShiftsOffset + row] / 360.0)`:`0i`};
  ${!d&&u&&f?t.rowTurns?`start.x += 360.0 * f32(prefixTurn);`:`start.x = start.x + rowShifts[rowShiftsOffset + row];`:``}
  let measure = ${m?`rowMeasures[rowMeasuresOffset + row]`:`0.0`};
  if (piece == 0u) {
    writeVertex(index, start, measure, row, prefixTurn);
    return;
  }
  ${t.closedLines?`let line = findLine(row);
  let first = lineOffsets[lineOffsetsOffset + line];
  let last = min(lineOffsets[lineOffsetsOffset + line + 1u], ${t.positions.length}u);
  let next = select(row + 1u, first, row + 1u >= last);`:`let next = row + 1u;`}
  let end = getPosition(next);
  let segmentLength = getSegmentLength(start, end);
  ${a?`let angle = geodesicCentralAngle(start, end);`:``}
  let fraction = f32(piece) / f32(count);
  writeVertex(index, ${g}, measure + segmentLength * fraction, row, prefixTurn);`})}function h(e,t){return i(e,{id:t.id,operation:t.operation,variant:`source-rows`,bindings:[{name:`counts`,view:t.counts,type:`u32`,access:`read`},{name:`starts`,view:t.starts,type:`u32`,access:`read`},{name:`outputSourceRows`,view:t.outputSourceRows,type:`u32`,access:`read_write`}],invocationCount:t.outputSourceRows.length,workgroupSize:128,guardIndex:!1,declarations:`${s(t.counts.length)}
fn balancedStart(row: u32) -> u32 { return starts[startsOffset + row]; }
fn balancedCount(row: u32) -> u32 { return counts[countsOffset + row]; }`,body:`let row = findBalancedOwner(index, workgroupIndex * BALANCED_LANES, localInvocationIndex);
  if (index < ${t.outputSourceRows.length}u && row != BALANCED_NONE) {
    outputSourceRows[outputSourceRowsOffset + index] = row;
  }`})}var g=`
fn getSource(row: u32) -> vec2<f32> {
  return vec2<f32>(sources[sourcesOffset + 2u * row], sources[sourcesOffset + 2u * row + 1u]);
}

fn getTarget(row: u32) -> vec2<f32> {
  return vec2<f32>(targets[targetsOffset + 2u * row], targets[targetsOffset + 2u * row + 1u]);
}
`;function _(e,t){return i(e,{id:t.id,operation:t.operation,variant:`count`,bindings:[{name:`sources`,view:t.sources,type:`f32`,access:`read`},{name:`targets`,view:t.targets,type:`f32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`counts`,view:t.counts,type:`u32`,access:`read_write`},...t.approximation?[{name:`approximation`,view:t.approximation,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:t.sources.length,declarations:`${a}
const RADIUS: f32 = ${r(t.radius)};
const MAXIMUM_SEGMENTS: u32 = ${t.maximumSegmentCount}u;
${g}`,body:`let distance = geodesicCentralAngle(getSource(index), getTarget(index)) * RADIUS;
  let maximumLength = parameters[parametersOffset];
  let minimumSegments = parameters[parametersOffset + 1u];
  var segments = 1.0;
  if (maximumLength > 0.0 && distance > 0.0) {
    ${t.approximation?`if (ceil(distance / maximumLength) > f32(MAXIMUM_SEGMENTS)) {
      atomicStore(&approximation[approximationOffset], 1u);
    }`:``}
    segments = ceil(min(distance / maximumLength, f32(MAXIMUM_SEGMENTS)));
  }
  if (minimumSegments > segments) {
    segments = min(round(minimumSegments), f32(MAXIMUM_SEGMENTS));
  }
  counts[countsOffset + index] = clamp(u32(segments), 1u, MAXIMUM_SEGMENTS) + 1u;`})}function v(e,t){let n=[{name:`sources`,view:t.sources,type:`f32`,access:`read`},{name:`targets`,view:t.targets,type:`f32`,access:`read`},{name:`counts`,view:t.counts,type:`u32`,access:`read`},{name:`starts`,view:t.starts,type:`u32`,access:`read`},{name:`outputPositions`,view:t.outputPositions,type:`f32`,access:`read_write`}];return t.outputMeasures&&n.push({name:`outputMeasures`,view:t.outputMeasures,type:`f32`,access:`read_write`}),t.outputSourceRows&&n.push({name:`outputSourceRows`,view:t.outputSourceRows,type:`u32`,access:`read_write`}),i(e,{id:t.id,operation:t.operation,variant:`emit`,bindings:n,invocationCount:t.outputPositions.length,workgroupSize:128,guardIndex:!1,declarations:`${a}
const RADIUS: f32 = ${r(t.radius)};
const CAPACITY: u32 = ${t.outputPositions.length}u;
${g}
${s(t.sources.length)}
fn balancedStart(row: u32) -> u32 { return starts[startsOffset + row]; }
fn balancedCount(row: u32) -> u32 { return counts[countsOffset + row]; }

fn writeVertex(outputRow: u32, vertex: vec2<f32>, measure: f32, sourceRow: u32) {
  if (outputRow >= CAPACITY) {
    return;
  }
  outputPositions[outputPositionsOffset + 2u * outputRow] = vertex.x;
  outputPositions[outputPositionsOffset + 2u * outputRow + 1u] = vertex.y;
  ${t.outputMeasures?`outputMeasures[outputMeasuresOffset + outputRow] = measure;`:``}
  ${t.outputSourceRows?`outputSourceRows[outputSourceRowsOffset + outputRow] = sourceRow;`:``}
}`,body:`let pair = findBalancedOwner(index, workgroupIndex * BALANCED_LANES, localInvocationIndex);
  if (pair == BALANCED_NONE) {
    return;
  }
  let segments = counts[countsOffset + pair] - 1u;
  let vertex = index - starts[startsOffset + pair];
  let source = getSource(pair);
  if (vertex == 0u) {
    writeVertex(index, source, 0.0, pair);
    return;
  }
  let destination = getTarget(pair);
  // Measure and interpolate from the original endpoints. Reconstructing an unwrapped
  // target first rounds away short antimeridian separations even with a stable angle.
  let angle = geodesicCentralAngle(source, destination);
  if (vertex == segments) {
    let unwrapped = vec2<f32>(source.x + geodesicLongitudeDelta(source.x, destination.x), destination.y);
    writeVertex(index, unwrapped, angle * RADIUS, pair);
    return;
  }
  let fraction = f32(vertex) / f32(segments);
  writeVertex(
    index,
    geodesicInterpolate(source, destination, angle, fraction),
    angle * RADIUS * fraction,
    pair
  );`})}function y(t,n){let r=[];n.lineOffsets&&r.push({name:`lineOffsets`,view:n.lineOffsets,type:`u32`,access:`read`}),r.push({name:`counts`,view:n.counts,type:`u32`,access:`read`},{name:`starts`,view:n.starts,type:`u32`,access:`read`},{name:`outputLineOffsets`,view:n.outputLineOffsets,type:`u32`,access:`read_write`},{name:`overflowOut`,view:n.overflow,type:`u32`,access:`read_write`}),n.vertexCount&&r.push({name:`vertexCountOut`,view:n.vertexCount,type:`u32`,access:`read_write`}),n.requiredVertexCount&&r.push({name:`totalOut`,view:n.requiredVertexCount,type:`u32`,access:`read_write`});let a=n.lineOffsets?`min(lineOffsets[lineOffsetsOffset + index], ROW_COUNT)`:`min(index, ROW_COUNT)`;return i(t,{id:n.id,operation:n.operation,variant:`publish`,bindings:r,invocationCount:n.lineCount+1,declarations:`${e}
const ROW_COUNT: u32 = ${n.rowCount}u;
const CAPACITY: u32 = ${n.capacity}u;`,body:`let total = getSaturatedScanSum(starts[startsOffset + ROW_COUNT - 1u], counts[countsOffset + ROW_COUNT - 1u]);
  let row = ${a};
  var start = total;
  if (row < ROW_COUNT) {
    start = starts[startsOffset + row];
  }
  outputLineOffsets[outputLineOffsetsOffset + index] = min(start, CAPACITY);
  if (index == 0u) {
    overflowOut[overflowOutOffset] = select(0u, 1u, total > CAPACITY);
    ${n.vertexCount?`vertexCountOut[vertexCountOutOffset] = min(total, CAPACITY);`:``}
    ${n.requiredVertexCount?`totalOut[totalOutOffset] = total;`:``}
  }`})}function b(e,t){return i(e,{id:t.id,operation:t.operation,variant:`source-line-indexs`,bindings:[{name:`sourceLineIndices`,view:t.sourceLineIndices,type:`u32`,access:`read_write`}],invocationCount:t.sourceLineIndices.length,body:`sourceLineIndices[sourceLineIndicesOffset + index] = index;`})}export{b as a,p as c,s as d,v as i,m as l,f as n,y as o,_ as r,d as s,c as t,h as u};