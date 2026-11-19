import{o as e}from"./wgsl-kernel-nodes-DagJklH2.js";var t=1e6,n=180/Math.PI;function r(e){return e.format.startsWith(`uint32`)?`u32`:e.format.startsWith(`sint32`)?`i32`:`f32`}function i(t,n){t.graph.add({getCommandNodes:t=>[e(t,{id:n.id,operation:`CellsRunAdapter`,bindings:n.bindings.map(e=>({name:e.name,view:e.buffer.view,type:r(e.buffer),access:e.access})),invocationCount:n.invocationCount,body:n.body})]})}function a(e,t,r){let a=e.allocate(`${t}-lonlat`,`float32x2`,r.length);return i(e,{id:`${t}-unit-to-lonlat`,invocationCount:r.length,bindings:[{name:`vectors`,buffer:r,access:`read`},{name:`positions`,buffer:a,access:`read_write`}],body:`let base = vectorsOffset + index * 3u;
  let vector = vec3<f32>(vectors[base], vectors[base + 1u], vectors[base + 2u]);
  let outBase = positionsOffset + index * 2u;
  positions[outBase] = atan2(vector.y, vector.x) * ${n};
  positions[outBase + 1u] = asin(clamp(vector.z, -1.0, 1.0)) * ${n};`}),a}function o(e,t,n,r=n.length){let a=e.allocate(`${t}-float`,`float32`,r);return i(e,{id:`${t}-uint-to-float`,invocationCount:r,bindings:[{name:`values`,buffer:n,access:`read`},{name:`result`,buffer:a,access:`read_write`}],body:`let value = values[valuesOffset + index];
  result[resultOffset + index] = select(f32(value), 1.0e30, value == 0xffffffffu);`}),a}function s(e,n,r,a,o){let s=e.allocate(`${n}-ring-segments`,`float32x4`,r.length),c=a.format===`uint32x2`?2:1;return i(e,{id:`${n}-ring-segments`,invocationCount:r.length,bindings:[{name:`positions`,buffer:r,access:`read`},{name:`counts`,buffer:a,access:`read`},{name:`segments`,buffer:s,access:`read_write`}],body:`let group = index / ${o}u;
  let slot = index % ${o}u;
  let count = counts[countsOffset + group * ${c}u];
  let outBase = segmentsOffset + index * 4u;
  if (slot >= count || count < 2u) {
    segments[outBase] = ${t};
    segments[outBase + 1u] = ${t};
    segments[outBase + 2u] = ${t};
    segments[outBase + 3u] = ${t};
    return;
  }
  let fromBase = positionsOffset + (group * ${o}u + slot) * 2u;
  let toBase = positionsOffset + (group * ${o}u + (slot + 1u) % count) * 2u;
  segments[outBase] = positions[fromBase];
  segments[outBase + 1u] = positions[fromBase + 1u];
  segments[outBase + 2u] = positions[toBase];
  segments[outBase + 3u] = positions[toBase + 1u];`}),s}function c(e,n,r){let a=r.leftRows.length,o=e.allocate(`${n}-pair-segments`,`float32x4`,a);return i(e,{id:`${n}-pair-segments`,invocationCount:a,bindings:[{name:`leftPositions`,buffer:r.leftPositions,access:`read`},{name:`rightPositions`,buffer:r.rightPositions,access:`read`},{name:`leftRows`,buffer:r.leftRows,access:`read`},{name:`rightRows`,buffer:r.rightRows,access:`read`},{name:`count`,buffer:r.count,access:`read`},{name:`segments`,buffer:o,access:`read_write`}],body:`let outBase = segmentsOffset + index * 4u;
  if (index >= count[countOffset]) {
    segments[outBase] = ${t};
    segments[outBase + 1u] = ${t};
    segments[outBase + 2u] = ${t};
    segments[outBase + 3u] = ${t};
    return;
  }
  let leftBase = leftPositionsOffset + leftRows[leftRowsOffset + index] * 2u;
  let rightBase = rightPositionsOffset + rightRows[rightRowsOffset + index] * 2u;
  segments[outBase] = leftPositions[leftBase];
  segments[outBase + 1u] = leftPositions[leftBase + 1u];
  segments[outBase + 2u] = rightPositions[rightBase];
  segments[outBase + 3u] = rightPositions[rightBase + 1u];`}),o}function l(e,n,r){let a=r.positions.length,o=e.allocate(`${n}-offset-segments`,`float32x4`,a);return i(e,{id:`${n}-offset-segments`,invocationCount:a,bindings:[{name:`positions`,buffer:r.positions,access:`read`},{name:`featureOffsets`,buffer:r.featureOffsets,access:`read`},{name:`polygonOffsets`,buffer:r.polygonOffsets,access:`read`},{name:`ringOffsets`,buffer:r.ringOffsets,access:`read`},{name:`count`,buffer:r.count,access:`read`},{name:`segments`,buffer:o,access:`read_write`}],body:`let features = count[countOffset];
  let polygonCount = featureOffsets[featureOffsetsOffset + features];
  let ringCount = polygonOffsets[polygonOffsetsOffset + polygonCount];
  let total = ringOffsets[ringOffsetsOffset + ringCount];
  let outBase = segmentsOffset + index * 4u;
  if (index >= total) {
    segments[outBase] = ${t};
    segments[outBase + 1u] = ${t};
    segments[outBase + 2u] = ${t};
    segments[outBase + 3u] = ${t};
    return;
  }
  var start = 0u;
  var end = total;
  for (var ring = 0u; ring < ringCount; ring++) {
    let ringStart = ringOffsets[ringOffsetsOffset + ring];
    let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
    if (index >= ringStart && index < ringEnd) {
      start = ringStart;
      end = ringEnd;
    }
  }
  let next = select(index + 1u, start, index + 1u == end);
  let fromBase = positionsOffset + index * 2u;
  let toBase = positionsOffset + next * 2u;
  segments[outBase] = positions[fromBase];
  segments[outBase + 1u] = positions[fromBase + 1u];
  segments[outBase + 2u] = positions[toBase];
  segments[outBase + 3u] = positions[toBase + 1u];`}),o}function u(e,n,r){let a=r.positions.length,o=e.allocate(`${n}-counted-segments`,`float32x4`,a);return i(e,{id:`${n}-counted-segments`,invocationCount:a,bindings:[{name:`positions`,buffer:r.positions,access:`read`},{name:`ringOffsets`,buffer:r.ringOffsets,access:`read`},{name:`counts`,buffer:r.counts,access:`read`},{name:`segments`,buffer:o,access:`read_write`}],body:`let total = counts[countsOffset];
  let ringCount = counts[countsOffset + 1u];
  let outBase = segmentsOffset + index * 4u;
  if (index >= total) {
    segments[outBase] = ${t};
    segments[outBase + 1u] = ${t};
    segments[outBase + 2u] = ${t};
    segments[outBase + 3u] = ${t};
    return;
  }
  var start = 0u;
  var end = total;
  for (var ring = 0u; ring < ringCount; ring++) {
    let ringStart = ringOffsets[ringOffsetsOffset + ring];
    let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
    if (index >= ringStart && index < ringEnd) {
      start = ringStart;
      end = ringEnd;
    }
  }
  let next = select(index + 1u, start, index + 1u == end);
  let fromBase = positionsOffset + index * 2u;
  let toBase = positionsOffset + next * 2u;
  segments[outBase] = positions[fromBase];
  segments[outBase + 1u] = positions[fromBase + 1u];
  segments[outBase + 2u] = positions[toBase];
  segments[outBase + 3u] = positions[toBase + 1u];`}),o}function d(e,n,r,a){let o=e.allocate(`${n}-point-pair-segments`,`float32x4`,r.length);return i(e,{id:`${n}-point-pair-segments`,invocationCount:r.length,bindings:[{name:`starts`,buffer:r,access:`read`},{name:`ends`,buffer:a,access:`read`},{name:`segments`,buffer:o,access:`read_write`}],body:`let startBase = startsOffset + index * 2u;
  let endBase = endsOffset + index * 2u;
  let startPoint = vec2<f32>(starts[startBase], starts[startBase + 1u]);
  let endPoint = vec2<f32>(ends[endBase], ends[endBase + 1u]);
  let outBase = segmentsOffset + index * 4u;
  let isValid = startPoint.x == startPoint.x && startPoint.y == startPoint.y && endPoint.x == endPoint.x && endPoint.y == endPoint.y;
  segments[outBase] = select(${t}, startPoint.x, isValid);
  segments[outBase + 1u] = select(${t}, startPoint.y, isValid);
  segments[outBase + 2u] = select(${t}, endPoint.x, isValid);
  segments[outBase + 3u] = select(${t}, endPoint.y, isValid);`}),o}function f(e,t,n){let r=e.allocate(`${t}-csr-segments`,`float32x4`,n.neighbors.length),a=e.allocate(`${t}-csr-count`,`uint32`,1);return i(e,{id:`${t}-csr-segments`,invocationCount:n.rowCount,bindings:[{name:`positions`,buffer:n.positions,access:`read`},{name:`offsets`,buffer:n.offsets,access:`read`},{name:`neighbors`,buffer:n.neighbors,access:`read`},{name:`segments`,buffer:r,access:`read_write`}],body:`let first = offsets[offsetsOffset + index];
  let last = offsets[offsetsOffset + index + 1u];
  let rowBase = positionsOffset + index * 2u;
  for (var entry = first; entry < last; entry++) {
    let neighborBase = positionsOffset + neighbors[neighborsOffset + entry] * 2u;
    let outBase = segmentsOffset + entry * 4u;
    segments[outBase] = positions[rowBase];
    segments[outBase + 1u] = positions[rowBase + 1u];
    segments[outBase + 2u] = positions[neighborBase];
    segments[outBase + 3u] = positions[neighborBase + 1u];
  }`}),i(e,{id:`${t}-csr-count`,invocationCount:1,bindings:[{name:`offsets`,buffer:n.offsets,access:`read`},{name:`count`,buffer:a,access:`read_write`}],body:`count[countOffset] = offsets[offsetsOffset + ${n.rowCount}u];`}),{segments:r,count:a}}function p(e,t,n){let r=e.allocate(`${t}-row-sums`,`float32`,n.rowCount);return i(e,{id:`${t}-row-sums`,invocationCount:n.rowCount,bindings:[{name:`offsets`,buffer:n.offsets,access:`read`},{name:`weights`,buffer:n.weights,access:`read`},{name:`sums`,buffer:r,access:`read_write`}],body:`var total = 0.0;
  for (var entry = offsets[offsetsOffset + index]; entry < offsets[offsetsOffset + index + 1u]; entry++) {
    total += weights[weightsOffset + entry];
  }
  sums[sumsOffset + index] = total;`}),r}function m(e,t,n,r){let a=e.allocate(`${t}-field`,`float32`,n.length);return i(e,{id:`${t}-high-word-field`,invocationCount:n.length,bindings:[{name:`identifiers`,buffer:n,access:`read`},{name:`result`,buffer:a,access:`read_write`}],body:`let base = identifiersOffset + index * 2u;
  let isEmpty = identifiers[base] == 0u && identifiers[base + 1u] == 0u;
  result[resultOffset + index] = select(f32((identifiers[base + 1u] >> ${r}u) & 7u), -1.0, isEmpty);`}),a}function h(e,t,n){let r=e.allocate(`${t}-levels`,`float32`,n.length);return i(e,{id:`${t}-s2-levels`,invocationCount:n.length,bindings:[{name:`cells`,buffer:n,access:`read`},{name:`levels`,buffer:r,access:`read_write`}],body:`let base = cellsOffset + index * 2u;
  let low = cells[base];
  let high = cells[base + 1u];
  let zeros = select(countTrailingZeros(low), 32u + countTrailingZeros(high), low == 0u);
  levels[levelsOffset + index] = select(f32(30u - zeros / 2u), -1.0, low == 0u && high == 0u);`}),r}function g(e,t,n){let r=e.allocate(`${t}-paired-ids`,`uint32`,n.rowCount);return i(e,{id:`${t}-paired-ids`,invocationCount:n.rowCount,bindings:[{name:`leftRows`,buffer:n.leftRows,access:`read`},{name:`rightRows`,buffer:n.rightRows,access:`read`},{name:`count`,buffer:n.count,access:`read`},{name:`zoneIds`,buffer:r,access:`read_write`}],body:`var zone = 0xffffffffu;
  let total = min(count[countOffset], ${n.leftRows.length}u);
  for (var pair = 0u; pair < total; pair++) {
    if (leftRows[leftRowsOffset + pair] == index) {
      zone = rightRows[rightRowsOffset + pair];
      break;
    }
  }
  zoneIds[zoneIdsOffset + index] = zone;`}),r}function _(e,t,n,r,a,o){let[s,c,l,u]=r,d=e.allocate(`${t}-fill`,`float32`,a*o);return i(e,{id:`${t}-segment-fill`,invocationCount:a*o,bindings:[{name:`segments`,buffer:n,access:`read`},{name:`fill`,buffer:d,access:`read_write`}],body:`let column = index % ${a}u;
  let row = index / ${a}u;
  let x = ${s} + (f32(column) + 0.5) / ${a}.0 * ${l-s};
  let y = ${u} - (f32(row) + 0.5) / ${o}.0 * ${u-c};
  var isInside = false;
  for (var segment = 0u; segment < ${n.length}u; segment++) {
    let base = segmentsOffset + segment * 4u;
    let x0 = segments[base];
    let y0 = segments[base + 1u];
    let x1 = segments[base + 2u];
    let y1 = segments[base + 3u];
    if ((y0 > y) != (y1 > y) && x < (x1 - x0) * (y - y0) / (y1 - y0) + x0) {
      isInside = !isInside;
    }
  }
  fill[fillOffset + index] = select(1.0e30, 1.0, isInside);`}),d}export{m as a,g as c,h as d,_ as f,s as i,d as l,a as m,f as n,i as o,o as p,p as r,c as s,u as t,l as u};