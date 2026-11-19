import{a as e,o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{c as n}from"./wgsl-common-DfyXiYB0.js";import{t as r}from"./exact-orientation-wgsl-B_jI-HEc.js";import{i}from"./spatial-wgsl-point-helpers-DcOgpqmI.js";var a=`
${i(`readPoint`,`positions`)}
`;function o(e,t,n){return`
fn ${e}(row: u32) -> u32 {
  var low = 0u;
  var high = ${n};
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (${t}[${t}Offset + middle] <= row) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return low;
}
`}function s(e,n){let i=n.mode===`orient-polygons`,s=[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:n.ringOffsets,type:`u32`,access:`read`}];i&&s.push({name:`polygonOffsets`,view:n.polygonOffsets,type:`u32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`}),s.push({name:`ringFlags`,view:n.ringFlags,type:`u32`,access:`read_write`});let c=n.ringOffsets.length-1;return t(e,{id:n.id,operation:n.operation,variant:`ring-orientation`,bindings:s,invocationCount:c,declarations:`const VERTEX_COUNT: u32 = ${n.positions.length}u;
${i?`const POLYGON_COUNT: u32 = ${n.polygonOffsets.length-1}u;
${a}
${r}
${o(`findPolygon`,`polygonOffsets`,`POLYGON_COUNT`)}`:``}`,body:i?`
  let start = ringOffsets[ringOffsetsOffset + index];
  let end = min(ringOffsets[ringOffsetsOffset + index + 1u], VERTEX_COUNT);
  var shouldReverse = false;
  if (end > start && end - start >= 3u) {
    // Winding at the lexicographically smallest vertex (first on ties), decided by exact
    // orientation so sliver rings are not misjudged by f32 shoelace cancellation.
    var extreme = start;
    var center = readPoint(start);
    for (var row = start + 1u; row < end; row++) {
      let candidate = readPoint(row);
      if (candidate.x < center.x || (candidate.x == center.x && candidate.y < center.y)) {
        extreme = row;
        center = candidate;
      }
    }
    let count = end - start;
    var previous = center;
    var next = center;
    var foundPrevious = false;
    var foundNext = false;
    for (var step = 1u; step < count; step++) {
      if (!foundPrevious) {
        previous = readPoint(start + (extreme - start + count - step) % count);
        foundPrevious = any(previous != center);
      }
      if (!foundNext) {
        next = readPoint(start + (extreme - start + step) % count);
        foundNext = any(next != center);
      }
      if (foundPrevious && foundNext) { break; }
    }
    var winding = 0;
    if (foundPrevious && foundNext) { winding = orientSign(previous, center, next); }
    // Non-finite or uncertain winding leaves the ring as stored.
    if (winding == 2) { winding = 0; }
    var isExterior = true;
    if (index >= polygonOffsets[polygonOffsetsOffset] &&
        index < polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT]) {
      isExterior = polygonOffsets[polygonOffsetsOffset + findPolygon(index)] == index;
    }
    let exteriorClockwise = parameters[parametersOffset] != 0.0;
    let wantClockwise = select(!exteriorClockwise, exteriorClockwise, isExterior);
    shouldReverse = (winding > 0 && wantClockwise) || (winding < 0 && !wantClockwise);
  }
  ringFlags[ringFlagsOffset + index] = select(0u, 1u, shouldReverse);`:`ringFlags[ringFlagsOffset + index] = 1u;`})}function c(e,n){let r=n.ringOffsets.length-1;return t(e,{id:n.id,operation:n.operation,variant:`ring-reverse`,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:n.ringOffsets,type:`u32`,access:`read`},{name:`ringFlags`,view:n.ringFlags,type:`u32`,access:`read`},{name:`outputPositions`,view:n.outputPositions,type:`f32`,access:`read_write`}],invocationCount:n.positions.length,declarations:`const VERTEX_COUNT: u32 = ${n.positions.length}u;
const RING_COUNT: u32 = ${r}u;
${o(`findRing`,`ringOffsets`,`RING_COUNT`)}`,body:`
  var source = index;
  let first = ringOffsets[ringOffsetsOffset];
  let last = min(ringOffsets[ringOffsetsOffset + RING_COUNT], VERTEX_COUNT);
  if (index >= first && index < last) {
    let ring = findRing(index);
    if (ringFlags[ringFlagsOffset + ring] != 0u) {
      let start = ringOffsets[ringOffsetsOffset + ring];
      let end = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
      source = start + end - 1u - index;
    }
  }
  outputPositions[outputPositionsOffset + 2u * index] = positions[positionsOffset + 2u * source];
  outputPositions[outputPositionsOffset + 2u * index + 1u] =
    positions[positionsOffset + 2u * source + 1u];`})}function l(e,n){let r=!!n.ringOffsets,i=[{name:`positions`,view:n.positions,type:`f32`,access:`read`}];n.ringOffsets&&i.push({name:`ringOffsets`,view:n.ringOffsets,type:`u32`,access:`read`}),n.featureOffsets&&i.push({name:`featureOffsets`,view:n.featureOffsets,type:`u32`,access:`read`}),n.polygonOffsets&&i.push({name:`polygonOffsets`,view:n.polygonOffsets,type:`u32`,access:`read`}),n.parameters&&i.push({name:`parameters`,view:n.parameters,type:`f32`,access:`read`}),n.featureTransforms&&i.push({name:`featureTransforms`,view:n.featureTransforms,type:`f32`,access:`read`}),n.bounds&&i.push({name:`bounds`,view:n.bounds,type:`f32`,access:`read`}),n.centroids&&i.push({name:`centroids`,view:n.centroids,type:`f32`,access:`read`}),i.push({name:`outputPositions`,view:n.outputPositions,type:`f32`,access:`read_write`});let s=r?`let ring = findRing(index);
  ${n.polygonOffsets?`featureIndex = min(findFeature(findPolygon(ring)), FEATURE_COUNT - 1u);`:n.featureOffsets?`featureIndex = min(findFeature(ring), FEATURE_COUNT - 1u);`:`featureIndex = min(ring, FEATURE_COUNT - 1u);`}`:``,c=n.featureTransforms?`let base = 6u * featureIndex + featureTransformsOffset;
  let result = vec2<f32>(
    featureTransforms[base] * point.x + featureTransforms[base + 1u] * point.y +
      featureTransforms[base + 4u],
    featureTransforms[base + 2u] * point.x + featureTransforms[base + 3u] * point.y +
      featureTransforms[base + 5u]
  );`:`let a = parameters[parametersOffset];
  let b = parameters[parametersOffset + 1u];
  let d = parameters[parametersOffset + 2u];
  let e = parameters[parametersOffset + 3u];
  let translation = vec2<f32>(parameters[parametersOffset + 4u], parameters[parametersOffset + 5u]);
  var origin = vec2<f32>(parameters[parametersOffset + 6u], parameters[parametersOffset + 7u]);
  let originMode = u32(parameters[parametersOffset + 8u] + 0.5);
  ${n.bounds?`if (originMode == 1u) {
    let box = vec4<f32>(
      bounds[boundsOffset + 4u * featureIndex],
      bounds[boundsOffset + 4u * featureIndex + 1u],
      bounds[boundsOffset + 4u * featureIndex + 2u],
      bounds[boundsOffset + 4u * featureIndex + 3u]
    );
    origin = 0.5 * (box.xy + box.zw);
  }`:``}
  ${n.centroids?`if (originMode == 2u) {
    origin = vec2<f32>(
      centroids[centroidsOffset + 2u * featureIndex],
      centroids[centroidsOffset + 2u * featureIndex + 1u]
    );
  }`:``}
  let relative = point - origin;
  let result = vec2<f32>(a * relative.x + b * relative.y, d * relative.x + e * relative.y) +
    origin + translation;`;return t(e,{id:n.id,operation:n.operation,variant:`affine`,bindings:i,invocationCount:n.positions.length,declarations:`const FEATURE_COUNT: u32 = ${Math.max(n.featureCount,1)}u;
${n.ringOffsets?`const RING_COUNT: u32 = ${n.ringOffsets.length-1}u;
${o(`findRing`,`ringOffsets`,`RING_COUNT`)}`:``}
${n.polygonOffsets?`const POLYGON_COUNT: u32 = ${n.polygonOffsets.length-1}u;
${o(`findPolygon`,`polygonOffsets`,`POLYGON_COUNT`)}`:``}
${n.featureOffsets?`const FEATURE_OFFSET_COUNT: u32 = ${n.featureOffsets.length-1}u;
${o(`findFeature`,`featureOffsets`,`FEATURE_OFFSET_COUNT`)}`:``}
${a}`,body:`
  let point = readPoint(index);
  var featureIndex = 0u;
  ${s}
  ${c}
  outputPositions[outputPositionsOffset + 2u * index] = result.x;
  outputPositions[outputPositionsOffset + 2u * index + 1u] = result.y;`})}var u=`
fn snapValue(value: f32, gridSize: f32) -> f32 {
  if (gridSize > 0.0) {
    return floor(value / gridSize + 0.5) * gridSize;
  }
  return value;
}

fn readSnappedPoint(row: u32, gridSize: f32) -> vec2<f32> {
  return vec2<f32>(
    snapValue(positions[positionsOffset + 2u * row], gridSize),
    snapValue(positions[positionsOffset + 2u * row + 1u], gridSize)
  );
}
`,d=`
${n}
fn isWithinTolerance(first: vec2<f32>, second: vec2<f32>, tolerance: f32) -> bool {
  // Test operands before subtraction: Metal can fold (a - b) == 0 into a == b,
  // which would incorrectly remove repeated infinities in the constant-tolerance path.
  if (!isFiniteBits(first.x) || !isFiniteBits(first.y) ||
      !isFiniteBits(second.x) || !isFiniteBits(second.y)) { return false; }
  let delta = first - second;
  if (tolerance > 0.0) {
    return dot(delta, delta) <= tolerance * tolerance;
  }
  return delta.x == 0.0 && delta.y == 0.0;
}`;function f(e,n){let r=[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`outputPositions`,view:n.outputPositions,type:`f32`,access:`read_write`},{name:`countOut`,view:n.count,type:`u32`,access:`read_write`},{name:`overflowOut`,view:n.overflow,type:`u32`,access:`read_write`}];return n.requiredCount&&r.push({name:`totalOut`,view:n.requiredCount,type:`u32`,access:`read_write`}),t(e,{id:n.id,operation:n.operation,variant:`cleanup-points`,bindings:r,invocationCount:n.positions.length,declarations:`const POINT_COUNT: u32 = ${n.positions.length}u;
const CAPACITY: u32 = ${n.outputPositions.length}u;
${u}`,body:`
  if (index == 0u) {
    countOut[countOutOffset] = min(POINT_COUNT, CAPACITY);
    overflowOut[overflowOutOffset] = select(0u, 1u, POINT_COUNT > CAPACITY);
    ${n.requiredCount?`totalOut[totalOutOffset] = POINT_COUNT;`:``}
  }
  if (index >= CAPACITY) {
    return;
  }
  let point = readSnappedPoint(index, parameters[parametersOffset]);
  outputPositions[outputPositionsOffset + 2u * index] = point.x;
  outputPositions[outputPositionsOffset + 2u * index + 1u] = point.y;`})}function p(e,n){return t(e,{id:n.id,operation:n.operation,variant:`cleanup-flags`,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:n.ringOffsets,type:`u32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`keepFlags`,view:n.keepFlags,type:`u32`,access:`read_write`},{name:`collapsedCount`,view:n.collapsedCount,type:`atomic<u32>`,access:`read_write`}],invocationCount:n.ringOffsets.length-1,condition:n.condition,extraResources:n.extraResources,declarations:`const VERTEX_COUNT: u32 = ${n.positions.length}u;
const IS_POLYGON: bool = ${n.isPolygon?`true`:`false`};
${u}
${d}`,body:`
  let gridSize = parameters[parametersOffset];
  let tolerance = max(parameters[parametersOffset + 1u], 0.0);
  let removeRepeated = parameters[parametersOffset + 2u] != 0.0;
  let start = ringOffsets[ringOffsetsOffset + index];
  let end = min(ringOffsets[ringOffsetsOffset + index + 1u], VERTEX_COUNT);
  if (end <= start) {
    return;
  }
  if (!removeRepeated || end - start < 2u) {
    for (var row = start; row < end; row++) {
      keepFlags[keepFlagsOffset + row] = 1u;
    }
    return;
  }
  var lastKept = start;
  var keptCount = 1u;
  keepFlags[keepFlagsOffset + start] = 1u;
  var lastKeptPoint = readSnappedPoint(start, gridSize);
  for (var row = start + 1u; row + 1u < end; row++) {
    let point = readSnappedPoint(row, gridSize);
    if (isWithinTolerance(point, lastKeptPoint, tolerance)) {
      keepFlags[keepFlagsOffset + row] = 0u;
    } else {
      keepFlags[keepFlagsOffset + row] = 1u;
      lastKept = row;
      lastKeptPoint = point;
      keptCount++;
    }
  }
  let finalRow = end - 1u;
  keepFlags[keepFlagsOffset + finalRow] = 1u;
  keptCount++;
  if (lastKept != start &&
      isWithinTolerance(readSnappedPoint(finalRow, gridSize), lastKeptPoint, tolerance)) {
    keepFlags[keepFlagsOffset + lastKept] = 0u;
    keptCount--;
  }
  if (IS_POLYGON && keptCount < 3u) {
    for (var row = start; row < end; row++) {
      keepFlags[keepFlagsOffset + row] = 0u;
    }
    atomicAdd(&collapsedCount[collapsedCountOffset], 1u);
  }`})}function m(e,n){return t(e,{id:n.id,operation:n.operation,variant:`cleanup-emit`,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`keepFlags`,view:n.keepFlags,type:`u32`,access:`read`},{name:`ranks`,view:n.ranks,type:`u32`,access:`read`},{name:`outputPositions`,view:n.outputPositions,type:`f32`,access:`read_write`}],invocationCount:n.positions.length,declarations:`const CAPACITY: u32 = ${n.outputPositions.length}u;
${u}`,body:`
  if (keepFlags[keepFlagsOffset + index] == 0u) {
    return;
  }
  let outputRow = ranks[ranksOffset + index];
  if (outputRow >= CAPACITY) {
    return;
  }
  let point = readSnappedPoint(index, parameters[parametersOffset]);
  outputPositions[outputPositionsOffset + 2u * outputRow] = point.x;
  outputPositions[outputPositionsOffset + 2u * outputRow + 1u] = point.y;`})}function h(e,n){let r=n.keepFlags.length,i=[{name:`ringOffsets`,view:n.ringOffsets,type:`u32`,access:`read`},{name:`keepFlags`,view:n.keepFlags,type:`u32`,access:`read`},{name:`ranks`,view:n.ranks,type:`u32`,access:`read`},{name:`outputRingOffsets`,view:n.outputRingOffsets,type:`u32`,access:`read_write`},{name:`countOut`,view:n.count,type:`u32`,access:`read_write`},{name:`overflowOut`,view:n.overflow,type:`u32`,access:`read_write`}];return n.requiredCount&&i.push({name:`totalOut`,view:n.requiredCount,type:`u32`,access:`read_write`}),t(e,{id:n.id,operation:n.operation,variant:`cleanup-publish`,bindings:i,invocationCount:n.ringOffsets.length,declarations:`const VERTEX_COUNT: u32 = ${r}u;
const CAPACITY: u32 = ${n.capacity}u;
const RING_OFFSET_COUNT: u32 = ${n.ringOffsets.length}u;`,body:`
  let total = select(
    ranks[ranksOffset + VERTEX_COUNT - 1u] + keepFlags[keepFlagsOffset + VERTEX_COUNT - 1u],
    0u,
    VERTEX_COUNT == 0u
  );
  let vertex = min(ringOffsets[ringOffsetsOffset + index], VERTEX_COUNT);
  let rank = select(ranks[ranksOffset + vertex], total, vertex >= VERTEX_COUNT);
  outputRingOffsets[outputRingOffsetsOffset + index] = min(rank, CAPACITY);
  if (index == 0u) {
    countOut[countOutOffset] = min(total, CAPACITY);
    overflowOut[overflowOutOffset] = select(0u, 1u, total > CAPACITY);
    ${n.requiredCount?`totalOut[totalOutOffset] = total;`:``}
  }`})}var g=0,_=1,v=2,y=3,b=`0xffffffffu`;function x(t,n){return e(t,{id:n.id,operation:n.operation,shapes:[[n.ringCount],[Math.ceil(n.vertexCount/32)],[n.ringCount],[n.vertexCount]]})}function S(e,t,n){let r={sequential:g,flag:_,resolve:v,apply:y}[t],{condition:i,extraResources:a}=e.getGate(r);return{condition:{...i,id:`${n}-gate`},extraResources:a}}function C(e,n){let r=e=>`dispatch[dispatchOffset + ${3*e}u] = 0u;`;return t(e,{id:n.id,operation:n.operation,variant:`cleanup-repeat-gate`,bindings:[{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},n.gate.binding],invocationCount:1,body:`
  ${n.gate.armWGSL}
  if (max(parameters[parametersOffset + 1u], 0.0) > 0.0) {
    ${r(_)}
    ${r(v)}
    ${r(y)}
  } else {
    ${r(g)}
  }`})}function w(e,n){return t(e,{id:n.id,operation:n.operation,variant:`cleanup-repeat-flags`,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:n.ringOffsets,type:`u32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`keepFlags`,view:n.keepFlags,type:`u32`,access:`read_write`},{name:`ringState`,view:n.ringState,type:`atomic<u32>`,access:`read_write`}],invocationCount:Math.ceil(n.positions.length/32),condition:n.condition,extraResources:n.extraResources,declarations:`const VERTEX_COUNT: u32 = ${n.positions.length}u;
const RING_COUNT: u32 = ${n.ringOffsets.length-1}u;
const CHUNK: u32 = 32u;
${u}
${d}
${o(`findRing`,`ringOffsets`,`RING_COUNT`)}`,body:`
  let chunkEnd = min((index + 1u) * CHUNK, min(ringOffsets[ringOffsetsOffset + RING_COUNT], VERTEX_COUNT));
  var row = max(index * CHUNK, ringOffsets[ringOffsetsOffset]);
  let removeRepeated = parameters[parametersOffset + 2u] != 0.0;
  let gridSize = parameters[parametersOffset];
  while (row < chunkEnd) {
    let ring = findRing(row);
    let start = ringOffsets[ringOffsetsOffset + ring];
    let end = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
    if (row < start || row >= end) {
      row++;
      continue;
    }
    let segmentEnd = min(end, chunkEnd);
    let plain = !removeRepeated || end - start < 2u;
    var count = 0u;
    var last = 0u;
    for (; row < segmentEnd; row++) {
      if (plain || row == start || row + 1u == end) {
        keepFlags[keepFlagsOffset + row] = 1u;
        continue;
      }
      if (isWithinTolerance(readSnappedPoint(row, gridSize), readSnappedPoint(row - 1u, gridSize), 0.0)) {
        keepFlags[keepFlagsOffset + row] = 0u;
      } else {
        keepFlags[keepFlagsOffset + row] = 1u;
        count++;
        last = row - start;
      }
    }
    if (count > 0u) {
      let stateBase = ringStateOffset + 3u * ring;
      atomicAdd(&ringState[stateBase], count);
      atomicMax(&ringState[stateBase + 1u], last);
    }
  }`})}function T(e,n){return t(e,{id:n.id,operation:n.operation,variant:`cleanup-repeat-resolve`,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:n.ringOffsets,type:`u32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`ringState`,view:n.ringState,type:`u32`,access:`read_write`},{name:`collapsedCount`,view:n.collapsedCount,type:`atomic<u32>`,access:`read_write`}],invocationCount:n.ringOffsets.length-1,condition:n.condition,extraResources:n.extraResources,declarations:`const VERTEX_COUNT: u32 = ${n.positions.length}u;
const IS_POLYGON: bool = ${n.isPolygon?`true`:`false`};
${u}
${d}`,body:`
  let start = ringOffsets[ringOffsetsOffset + index];
  let end = min(ringOffsets[ringOffsetsOffset + index + 1u], VERTEX_COUNT);
  if (end <= start || end - start < 2u || parameters[parametersOffset + 2u] == 0.0) {
    return;
  }
  let stateBase = ringStateOffset + 3u * index;
  // First and final vertex plus the kept interior vertices.
  var keptCount = 2u + ringState[stateBase];
  let lastKept = ringState[stateBase + 1u];
  var replaced = 0u;
  if (lastKept != 0u) {
    let gridSize = parameters[parametersOffset];
    if (isWithinTolerance(readSnappedPoint(end - 1u, gridSize), readSnappedPoint(end - 2u, gridSize), 0.0)) {
      replaced = lastKept;
      keptCount--;
    }
  }
  if (IS_POLYGON && keptCount < 3u) {
    ringState[stateBase + 2u] = ${b};
    atomicAdd(&collapsedCount[collapsedCountOffset], 1u);
  } else {
    ringState[stateBase + 2u] = replaced;
  }`})}function E(e,n){return t(e,{id:n.id,operation:n.operation,variant:`cleanup-repeat-apply`,bindings:[{name:`ringOffsets`,view:n.ringOffsets,type:`u32`,access:`read`},{name:`ringState`,view:n.ringState,type:`u32`,access:`read`},{name:`keepFlags`,view:n.keepFlags,type:`u32`,access:`read_write`}],invocationCount:n.positionCount,condition:n.condition,extraResources:n.extraResources,declarations:`const VERTEX_COUNT: u32 = ${n.positionCount}u;
const RING_COUNT: u32 = ${n.ringOffsets.length-1}u;
${o(`findRing`,`ringOffsets`,`RING_COUNT`)}`,body:`
  let first = ringOffsets[ringOffsetsOffset];
  if (index < first || index >= min(ringOffsets[ringOffsetsOffset + RING_COUNT], VERTEX_COUNT)) {
    return;
  }
  let ring = findRing(index);
  let start = ringOffsets[ringOffsetsOffset + ring];
  let end = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
  if (index < start || index >= end) {
    return;
  }
  let resolution = ringState[ringStateOffset + 3u * ring + 2u];
  if (resolution == ${b} || (resolution != 0u && index - start == resolution)) {
    keepFlags[keepFlagsOffset + index] = 0u;
  }`})}export{h as a,x as c,s as d,c as f,f as i,C as l,m as n,E as o,S as p,p as r,w as s,l as t,T as u};