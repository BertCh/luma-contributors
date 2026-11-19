import{i as e,r as t}from"./cell-dispatch-validation-PIkQyAK_.js";import{t as n}from"./common-CshkX-d_.js";import{r}from"./kernel-C3tfQOn4.js";import{t as i}from"./gpu-scan-B4NNTfF9.js";import{n as a,r as o,t as s}from"./cell-keys-Dl1tIQ_Z.js";import{o as c,s as l}from"./cell-index-wgsl-Cw7syfdA.js";import{c as u,n as d,o as f,t as p}from"./wgsl-kernel-nodes-DagJklH2.js";import{n as m,r as h}from"./gpu-cell-geometry-CDEk1ld7.js";import{t as g}from"./s2-analysis-wgsl-CSV2LEdb.js";import{n as _,r as v,t as y}from"./s2-covering-support-BNzE_85J.js";import{t as b}from"./lon-lat-ring-wgsl-DpZBIfuE.js";import{n as x,t as S}from"./exact-geometry-wgsl-y7TUnqOh.js";import{t as C}from"./exact-orientation-wgsl-B_jI-HEc.js";import{r as w}from"./gpu-point-to-cell-CYNSElfx.js";import{i as T,n as E,r as D,t as O}from"./polygon-edge-slabs-CRnrFvP2.js";import{t as ee}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{a as te}from"./sorted-segment-sums-Dvmm6OdR.js";import{i as k,n as A,r as j,v as M}from"./spatial-context-DwZEd4EZ.js";import{E as N,S as P,f as F,p as I,u as L,x as R,y as z}from"./gpu-contributor-utils-BIwLQON5.js";var B=[3.7,.56,.5,.32,.12,.046,.017,.0066,.00245,94e-5,35e-5,...[1,2,3,4,5].map(e=>35e-5/Math.sqrt(7)**e)].map(e=>Math.fround(.5*e)),V=2**20,H=4e-5,U=1e-6,W=2e-4,G=3e-5,K=6e-5;function q(e,t,n){let r=Math.max(e,1);return t?r*8+3*t.slotCapacity*8:r*n}var J=`0xffffffffu`;function Y(e,t,n){return`
fn ${e}(value: u32) -> u32 {
  var low = 0u;
  var high = ${n}u;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (${t}[${t}Offset + middle] <= value) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low == 0u) { return ${J}; }
  let range = low - 1u;
  if (value >= ${t}[${t}Offset + range + 1u]) { return ${J}; }
  return range;
}`}function ne(e){let t=`
const X_SLACK_BASE: f32 = ${u(H)};
const X_SLACK_RELATIVE: f32 = ${u(U)};
fn coverFloorToInt(value: f32) -> i32 { return i32(clamp(floor(value), -2.0e9, 2.0e9)); }
fn coverCeilToInt(value: f32) -> i32 { return i32(clamp(ceil(value), -2.0e9, 2.0e9)); }
`;if(e.family===`quadbin`){let n=e.resolution;return`${t}${s}${a}
const TILE_WIDTH: f32 = ${u(45*2**(3-n))};
const ROW_SCALE: f32 = ${u(2**(1-n))};
const PI: f32 = 3.14159265358979;
const RESOLUTION: u32 = ${n}u;
const Y_SLACK: f32 = ${u(W)};
fn coverTileLatitude(row: f32) -> f32 {
  let t = PI * (1.0 - row * ROW_SCALE);
  return (2.0 * atan(exp(t)) - PI * 0.5) * 57.29577951308232;
}
fn coverRowCount(rb: u32) -> u32 { return ranges[rb + 3u]; }
fn coverColumnCount(rb: u32) -> u32 { return ranges[rb + 2u]; }
fn coverRowLine(rb: u32, row: u32) -> f32 {
  return coverTileLatitude(f32(ranges[rb + 1u] + row) + 0.5);
}
fn coverRowBand(rb: u32, row: u32) -> vec2f {
  let y = ranges[rb + 1u] + row;
  return vec2f(coverTileLatitude(f32(y + 1u)) - Y_SLACK, coverTileLatitude(f32(y)) + Y_SLACK);
}
fn coverEdgeRows(rb: u32, lowY: f32, highY: f32) -> vec2i {
  let origin = i32(ranges[rb + 1u]);
  let first = i32(quadbinGetTileY(highY + Y_SLACK, RESOLUTION)) - origin - 1;
  let last = i32(quadbinGetTileY(lowY - Y_SLACK, RESOLUTION)) - origin + 1;
  return vec2i(max(first, 0), min(last, i32(ranges[rb + 3u]) - 1));
}
fn coverColumnRange(rb: u32, lowX: f32, highX: f32) -> vec2i {
  let origin = i32(ranges[rb]);
  let first = coverFloorToInt((lowX - X_SLACK_BASE + 180.0) / TILE_WIDTH) - 1 - origin;
  let last = coverFloorToInt((highX + X_SLACK_BASE + 180.0) / TILE_WIDTH) + 1 - origin;
  return vec2i(max(first, 0), min(last, i32(ranges[rb + 2u]) - 1));
}
`}let n=B[e.resolution];return`${t}
const LATTICE_SPACING: f32 = ${u(n)};
const Y_SLACK: f32 = ${u(G)};
const X_MARGIN_SLACK: f32 = ${u(K)};
fn coverRowCount(rb: u32) -> u32 { return ranges[rb + 1u]; }
fn coverColumnCount(rb: u32) -> u32 { return ranges[rb]; }
fn coverOriginLng(rb: u32) -> f32 { return bitcast<f32>(ranges[rb + 2u]); }
fn coverOriginLat(rb: u32) -> f32 { return bitcast<f32>(ranges[rb + 3u]); }
fn coverSpacingLng(rb: u32) -> f32 { return bitcast<f32>(ranges[rb + 4u]); }
fn coverRowLine(rb: u32, row: u32) -> f32 {
  return coverOriginLat(rb) + f32(row) * LATTICE_SPACING;
}
// An accepted cell's centre rounds to its lattice row: its latitude is within half a spacing of the
// row line, plus the f32 error of the rounding quotient (3e-7 of a spacing per row) and of the line.
fn coverRowHalfBand(rb: u32) -> f32 {
  return LATTICE_SPACING * (0.501 + 3.0e-7 * f32(ranges[rb + 1u])) + Y_SLACK;
}
fn coverRowBand(rb: u32, row: u32) -> vec2f {
  let line = coverRowLine(rb, row);
  let half = coverRowHalfBand(rb);
  return vec2f(line - half, line + half);
}
fn coverEdgeRows(rb: u32, lowY: f32, highY: f32) -> vec2i {
  let origin = coverOriginLat(rb);
  let half = coverRowHalfBand(rb);
  let first = coverFloorToInt((lowY - half - origin) / LATTICE_SPACING) - 1;
  let last = coverCeilToInt((highY + half - origin) / LATTICE_SPACING) + 1;
  return vec2i(max(first, 0), min(last, i32(ranges[rb + 1u]) - 1));
}
fn coverColumnMargin(rb: u32) -> f32 {
  return coverSpacingLng(rb) * (0.501 + 3.0e-7 * f32(ranges[rb])) + X_MARGIN_SLACK;
}
fn coverColumnRange(rb: u32, lowX: f32, highX: f32) -> vec2i {
  let origin = coverOriginLng(rb);
  let spacing = coverSpacingLng(rb);
  let margin = coverColumnMargin(rb) + X_SLACK_BASE;
  let first = coverFloorToInt((lowX - margin - origin) / spacing) - 1;
  let last = coverCeilToInt((highX + margin - origin) / spacing) + 1;
  return vec2i(max(first, 0), min(last, i32(ranges[rb]) - 1));
}
`}var re=`
fn coverSortableKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn coverFromSortableKey(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key & 0x7fffffffu, (key & 0x80000000u) != 0u));
}
`,ie=`${re}
fn coverGetVertex(vertex: u32) -> vec2f {
  let base = positionsOffset + 2u * vertex;
  return vec2f(positions[base], positions[base + 1u]);
}

/** (valid, xLow, xHigh): x extent of the part of edge a-b inside the latitude band, with slack. */
fn coverClipEdge(a: vec2f, b: vec2f, band: vec2f) -> vec3f {
  let aIsLower = a.y <= b.y;
  let lower = select(b, a, aIsLower);
  let upper = select(a, b, aIsLower);
  if ((upper.y < band.x) || (lower.y > band.y)) {
    return vec3f(0.0, 0.0, 0.0);
  }
  var x0 = lower.x;
  var x1 = upper.x;
  let dy = upper.y - lower.y;
  if (dy > 0.0) {
    let dx = upper.x - lower.x;
    x0 = lower.x + dx * clamp((band.x - lower.y) / dy, 0.0, 1.0);
    x1 = lower.x + dx * clamp((band.y - lower.y) / dy, 0.0, 1.0);
  }
  let slack = X_SLACK_BASE + X_SLACK_RELATIVE * abs(upper.x - lower.x);
  return vec3f(1.0, min(x0, x1) - slack, max(x0, x1) + slack);
}

/** x where edge a-b crosses latitude line; only called when (a.y <= line) != (b.y <= line). */
fn coverGetCrossing(a: vec2f, b: vec2f, line: f32) -> f32 {
  let aIsLower = a.y <= b.y;
  let lower = select(b, a, aIsLower);
  let upper = select(a, b, aIsLower);
  let t = clamp((line - lower.y) / (upper.y - lower.y), 0.0, 1.0);
  return lower.x + (upper.x - lower.x) * t;
}

`;function X(e,t,n){let{id:a,featureCount:o,candidateCapacity:s,positions:c}=t,{slotCapacity:l,force:d}=t.scanline,{counts:m,ranges:h}=n,g=`GPUCellCover`,_=c.length,v=3*l,y=`rangesOffset + ${o}u * 8u`,b=[],x=ne(t),S=e=>`${a}-scanline-${e}`,C=(t,n)=>r(e,S(t),`uint32`,Math.max(n,1)),w=`const FEATURE_COUNT: u32 = ${o}u;
const VERTEX_COUNT: u32 = ${_}u;
const STRIDE: u32 = 8u;
const SLOT_CAPACITY: u32 = ${l}u;
const INVALID_ROW: u32 = ${l}u;`,T=[{name:`positions`,view:c,type:`f32`,access:`read`}],E=C(`edge-feature`,_+1),D=C(`edge-next`,_+1);b.push(f(e,{id:S(`edges`),operation:g,variant:`scanline-edges`,bindings:[{name:`ringOffsets`,view:t.ringOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:t.polygonOffsets,type:`u32`,access:`read`},{name:`featureOffsets`,view:t.featureOffsets,type:`u32`,access:`read`},{name:`ranges`,view:h,type:`u32`,access:`read`},{name:`counts`,view:m,type:`u32`,access:`read`},{name:`edgeFeature`,view:E,type:`u32`,access:`read_write`},{name:`edgeNext`,view:D,type:`u32`,access:`read_write`}],invocationCount:_+1,declarations:`${w}
${Y(`coverFindRing`,`ringOffsets`,t.ringOffsets.length-1)}
${Y(`coverFindPolygon`,`polygonOffsets`,t.polygonOffsets.length-1)}
${Y(`coverFindFeatureOfPolygon`,`featureOffsets`,o)}`,body:`var feature = ${J};
  var next = ${J};
  if (index < VERTEX_COUNT) {
    let ring = coverFindRing(index);
    if (ring != ${J}) {
      let polygon = coverFindPolygon(ring);
      if (polygon != ${J}) {
        let candidate = coverFindFeatureOfPolygon(polygon);
        if (candidate != ${J}) {
          let rb = rangesOffset + candidate * STRIDE;
          // Non-finite vertices (flag bit 0) or no lattice: the feature keeps the bounding box.
          if (counts[countsOffset + candidate] > 0u && (ranges[rb + 5u] & 1u) == 0u) {
            let ringStart = ringOffsets[ringOffsetsOffset + ring];
            let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
            feature = candidate;
            next = select(index + 1u, ringStart, (index + 1u) >= ringEnd);
          }
        }
      }
    }
  }
  edgeFeature[edgeFeatureOffset + index] = feature;
  edgeNext[edgeNextOffset + index] = next;`}));let O=C(`edge-slots`,_+1),k=C(`demand`,2*o);b.push(p(e,{id:S(`clear-demand`),operation:g,view:k,type:`u32`,value:`0u`}),f(e,{id:S(`rows`),operation:g,variant:`scanline-rows`,bindings:[...T,{name:`ranges`,view:h,type:`u32`,access:`read`},{name:`edgeFeature`,view:E,type:`u32`,access:`read`},{name:`edgeNext`,view:D,type:`u32`,access:`read`},{name:`edgeSlots`,view:O,type:`u32`,access:`read_write`},{name:`demand`,view:k,type:`atomic<u32>`,access:`read_write`}],invocationCount:_+1,declarations:`${w}
${x}${ie}`,body:`var count = 0u;
  let feature = edgeFeature[edgeFeatureOffset + index];
  if (feature != ${J}) {
    let a = coverGetVertex(index);
    let b = coverGetVertex(edgeNext[edgeNextOffset + index]);
    let rows = coverEdgeRows(rangesOffset + feature * STRIDE, min(a.y, b.y), max(a.y, b.y));
    if (rows.x <= rows.y) {
      count = u32(rows.y - rows.x + 1);
      // Exact 64-bit total: a wrapped low word carries one into the high word.
      let before = atomicAdd(&demand[demandOffset + 2u * feature], count);
      if (before + count < before) {
        atomicAdd(&demand[demandOffset + 2u * feature + 1u], 1u);
      }
    }
  }
  edgeSlots[edgeSlotsOffset + index] = count;`}));let A=C(`cost`,o+1),j=C(`cost-starts`,o+1);b.push(f(e,{id:S(`choose`),operation:g,variant:`scanline-choose`,bindings:[{name:`counts`,view:m,type:`u32`,access:`read`},{name:`ranges`,view:h,type:`u32`,access:`read`},{name:`demand`,view:k,type:`u32`,access:`read`},{name:`cost`,view:A,type:`u32`,access:`read_write`}],invocationCount:o+1,declarations:`${w}
const FORCE: bool = ${d};
const COST_RATIO: f32 = ${u(4)};
${x}`,body:`var featureCost = 0u;
  if (index < FEATURE_COUNT) {
    let rb = rangesOffset + index * STRIDE;
    if (counts[countsOffset + index] > 0u && (ranges[rb + 5u] & 1u) == 0u) {
      let rows = coverRowCount(rb);
      let slots = select(demand[demandOffset + 2u * index], 0xffffffffu, (demand[demandOffset + 2u * index + 1u] != 0u));
      let cells = f32(rows) * f32(coverColumnCount(rb));
      // Work of the scanline enumeration (events and runs per row, one slot per edge row) against
      // the cells of the bounding box. The row count and the slots must also fit the stage.
      let cheaper = cells > COST_RATIO * (2.0 * f32(rows) + f32(slots));
      if ((FORCE || cheaper) && rows <= SLOT_CAPACITY && slots <= SLOT_CAPACITY) {
        featureCost = min(slots + rows, SLOT_CAPACITY + 1u);
      }
    }
  }
  cost[costOffset + index] = featureCost;`}),...new i({id:S(`cost-scan`),input:A,output:j,mode:`exclusive`}).getCommandNodes(e));let M=C(`count-carry`,o);b.push(f(e,{id:S(`select`),operation:g,variant:`scanline-select`,bindings:[{name:`counts`,view:m,type:`u32`,access:`read_write`},{name:`ranges`,view:h,type:`u32`,access:`read_write`},{name:`cost`,view:A,type:`u32`,access:`read`},{name:`costStarts`,view:j,type:`u32`,access:`read`},{name:`countCarry`,view:M,type:`u32`,access:`read_write`}],invocationCount:o,declarations:w,body:`let rb = rangesOffset + index * STRIDE;
  countCarry[countCarryOffset + index] = 0u;
  let featureCost = cost[costOffset + index];
  // The sum of capped costs stays below 2^32 because (featureCount + 1) * (capacity + 1) does.
  let selected = featureCost > 0u && (costStarts[costStartsOffset + index] + featureCost) <= SLOT_CAPACITY;
  ranges[rb + 5u] = (ranges[rb + 5u] & 1u) | select(0u, 2u, selected);
  ranges[rb + 6u] = 0u;
  ranges[rb + 7u] = 0u;
  if (selected) {
    counts[countsOffset + index] = 0u;
  }`}));let N=C(`edge-offsets`,_+1);b.push(f(e,{id:S(`slots`),operation:g,variant:`scanline-slots`,bindings:[{name:`ranges`,view:h,type:`u32`,access:`read`},{name:`edgeFeature`,view:E,type:`u32`,access:`read`},{name:`edgeSlots`,view:O,type:`u32`,access:`read_write`}],invocationCount:_+1,declarations:w,body:`let feature = edgeFeature[edgeFeatureOffset + index];
  if (feature == ${J} || (ranges[rangesOffset + feature * STRIDE + 5u] & 2u) == 0u) {
    edgeSlots[edgeSlotsOffset + index] = 0u;
  }`}),...new i({id:S(`edge-scan`),input:O,output:N,mode:`exclusive`}).getCommandNodes(e));let P=C(`event-row`,v),F=C(`event-position`,v);b.push(f(e,{id:S(`events`),operation:g,variant:`scanline-events`,bindings:[...T,{name:`ranges`,view:h,type:`u32`,access:`read`},{name:`edgeFeature`,view:E,type:`u32`,access:`read`},{name:`edgeNext`,view:D,type:`u32`,access:`read`},{name:`edgeOffsets`,view:N,type:`u32`,access:`read`},{name:`costStarts`,view:j,type:`u32`,access:`read`},{name:`eventRow`,view:P,type:`u32`,access:`read_write`},{name:`eventPosition`,view:F,type:`u32`,access:`read_write`}],invocationCount:l,declarations:`${w}
${x}${ie}`,body:`var rowOpen = INVALID_ROW;
  var rowClose = INVALID_ROW;
  var rowCross = INVALID_ROW;
  var keyOpen = 0u;
  var keyClose = 0u;
  var keyCross = 0u;
  if (index < edgeOffsets[edgeOffsetsOffset + VERTEX_COUNT]) {
    // The edge owning slot index: the last edge whose first slot is at most index.
    var low = 0u;
    var high = VERTEX_COUNT;
    while (low < high) {
      let middle = (low + high) >> 1u;
      if (edgeOffsets[edgeOffsetsOffset + middle] <= index) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    let edge = low - 1u;
    let feature = edgeFeature[edgeFeatureOffset + edge];
    if (feature != ${J}) {
      let rb = rangesOffset + feature * STRIDE;
      let a = coverGetVertex(edge);
      let b = coverGetVertex(edgeNext[edgeNextOffset + edge]);
      let rows = coverEdgeRows(rb, min(a.y, b.y), max(a.y, b.y));
      let row = rows.x + i32(index - edgeOffsets[edgeOffsetsOffset + edge]);
      if (row <= rows.y) {
        let rowKey = costStarts[costStartsOffset + feature] + u32(row);
        let clip = coverClipEdge(a, b, coverRowBand(rb, u32(row)));
        if (clip.x > 0.0) {
          rowOpen = rowKey;
          rowClose = rowKey;
          keyOpen = coverSortableKey(clip.y);
          keyClose = coverSortableKey(clip.z);
        }
        let line = coverRowLine(rb, u32(row));
        if ((a.y <= line) != (b.y <= line)) {
          rowCross = rowKey;
          keyCross = coverSortableKey(coverGetCrossing(a, b, line));
        }
      }
    }
  }
  let base = 3u * index;
  eventRow[eventRowOffset + base] = rowOpen;
  eventRow[eventRowOffset + base + 1u] = rowClose;
  eventRow[eventRowOffset + base + 2u] = rowCross;
  eventPosition[eventPositionOffset + base] = keyOpen;
  eventPosition[eventPositionOffset + base + 1u] = keyClose;
  eventPosition[eventPositionOffset + base + 2u] = keyCross;`}));let I=C(`event-ids`,v);b.push(f(e,{id:S(`iota`),operation:g,variant:`scanline-iota`,bindings:[{name:`eventIds`,view:I,type:`u32`,access:`read_write`}],invocationCount:v,body:`eventIds[eventIdsOffset + index] = index;`}));let L=ee(e,{id:S(`sort`),operation:g,indices:I,keys:[{view:P,keyBits:te(l)},{view:F}]});b.push(...L.nodes);let R=L.sortedIndices,z=L.sortedPrimaryKeys,B=C(`depth-delta`,v),V=C(`crossing-flag`,v),H=C(`segment-flags`,v),U=C(`depth-sum`,v),W=C(`crossing-sum`,v);b.push(f(e,{id:S(`flags`),operation:g,variant:`scanline-flags`,bindings:[{name:`sortedIndices`,view:R,type:`u32`,access:`read`},{name:`sortedRows`,view:z,type:`u32`,access:`read`},{name:`depthDelta`,view:B,type:`u32`,access:`read_write`},{name:`crossingFlag`,view:V,type:`u32`,access:`read_write`},{name:`segmentFlags`,view:H,type:`u32`,access:`read_write`}],invocationCount:v,declarations:w,body:`let row = sortedRows[sortedRowsOffset + index];
  let valid = row < INVALID_ROW;
  // Event e is slot e / 3: kind 0 opens a span, 1 closes it, 2 is a crossing.
  let kind = sortedIndices[sortedIndicesOffset + index] % 3u;
  var head = index == 0u;
  if (!head) {
    head = row != sortedRows[sortedRowsOffset + index - 1u];
  }
  segmentFlags[segmentFlagsOffset + index] = select(0u, 1u, head);
  depthDelta[depthDeltaOffset + index] = select(0u, select(0u, select(0xffffffffu, 1u, (kind == 0u)), (kind < 2u)), valid);
  crossingFlag[crossingFlagOffset + index] = select(0u, 1u, (valid && kind == 2u));`}),...new i({id:S(`depth-scan`),input:B,output:U,mode:`inclusive`,segmentFlags:H}).getCommandNodes(e),...new i({id:S(`crossing-scan`),input:V,output:W,mode:`inclusive`,segmentFlags:H}).getCommandNodes(e));let G=C(`covered`,v),K=C(`gap-columns`,2*v);b.push(f(e,{id:S(`cover-state`),operation:g,variant:`scanline-cover-state`,bindings:[{name:`sortedRows`,view:z,type:`u32`,access:`read`},{name:`depthSum`,view:U,type:`u32`,access:`read`},{name:`crossingSum`,view:W,type:`u32`,access:`read`},{name:`covered`,view:G,type:`u32`,access:`read_write`}],invocationCount:v,declarations:`${w}
const EVENT_COUNT: u32 = ${v}u;`,body:`let row = sortedRows[sortedRowsOffset + index];
  var keep = false;
  // Depth is never negative after all events of one position, so nonzero means a span covers.
  if (row < INVALID_ROW && (index + 1u) < EVENT_COUNT && sortedRows[sortedRowsOffset + index + 1u] == row) {
    keep = depthSum[depthSumOffset + index] != 0u || (crossingSum[crossingSumOffset + index] & 1u) != 0u;
  }
  covered[coveredOffset + index] = select(0u, 1u, keep);`}),f(e,{id:S(`gaps`),operation:g,variant:`scanline-gaps`,bindings:[{name:`sortedIndices`,view:R,type:`u32`,access:`read`},{name:`sortedRows`,view:z,type:`u32`,access:`read`},{name:`eventPosition`,view:F,type:`u32`,access:`read`},{name:`ranges`,view:h,type:`u32`,access:`read`},{name:`costStarts`,view:j,type:`u32`,access:`read`},{name:`covered`,view:G,type:`u32`,access:`read_write`},{name:`gapColumns`,view:K,type:`u32`,access:`read_write`}],invocationCount:v,declarations:`${w}
${x}${re}
${Y(`coverFindCostFeature`,`costStarts`,o)}`,body:`if (covered[coveredOffset + index] == 0u) {
    return;
  }
  let lowKey = eventPosition[eventPositionOffset + sortedIndices[sortedIndicesOffset + index]];
  let highKey = eventPosition[eventPositionOffset + sortedIndices[sortedIndicesOffset + index + 1u]];
  var keep = 0u;
  // Zero-length gaps add no cell the neighbouring gaps and spans do not already contain.
  if (lowKey < highKey) {
    let feature = coverFindCostFeature(sortedRows[sortedRowsOffset + index]);
    let columns = coverColumnRange(rangesOffset + feature * STRIDE, coverFromSortableKey(lowKey), coverFromSortableKey(highKey));
    if (columns.x <= columns.y) {
      keep = 1u;
      gapColumns[gapColumnsOffset + 2u * index] = u32(columns.x);
      gapColumns[gapColumnsOffset + 2u * index + 1u] = u32(columns.y);
    }
  }
  covered[coveredOffset + index] = keep;`}));let q=C(`gap-slots`,v);b.push(...new i({id:S(`gap-scan`),input:G,output:q,mode:`exclusive`}).getCommandNodes(e),f(e,{id:S(`compact`),operation:g,variant:`scanline-compact`,bindings:[{name:`covered`,view:G,type:`u32`,access:`read`},{name:`gapSlots`,view:q,type:`u32`,access:`read`},{name:`sortedRows`,view:z,type:`u32`,access:`read`},{name:`gapColumns`,view:K,type:`u32`,access:`read`},{name:`ranges`,view:h,type:`u32`,access:`read_write`}],invocationCount:v,declarations:w,body:`if (covered[coveredOffset + index] == 0u) {
    return;
  }
  let run = ${y} + gapSlots[gapSlotsOffset + index] * 8u;
  ranges[run] = sortedRows[sortedRowsOffset + index];
  ranges[run + 1u] = gapColumns[gapColumnsOffset + 2u * index];
  ranges[run + 2u] = gapColumns[gapColumnsOffset + 2u * index + 1u];`}));let X=C(`run-lengths`,v),Z=C(`run-offsets`,v),Q=C(`run-flags`,v),ae=`(gapSlots[gapSlotsOffset + ${v-1}u] + covered[coveredOffset + ${v-1}u])`;return b.push(f(e,{id:S(`runs`),operation:g,variant:`scanline-runs`,bindings:[{name:`ranges`,view:h,type:`u32`,access:`read_write`},{name:`costStarts`,view:j,type:`u32`,access:`read`},{name:`covered`,view:G,type:`u32`,access:`read`},{name:`gapSlots`,view:q,type:`u32`,access:`read`},{name:`runLengths`,view:X,type:`u32`,access:`read_write`},{name:`runFlags`,view:Q,type:`u32`,access:`read_write`}],invocationCount:v,declarations:`${w}
${Y(`coverFindCostFeature`,`costStarts`,o)}`,body:`var runLength = 0u;
  var isHead = false;
  if (index < ${ae}) {
    let run = ${y} + index * 8u;
    let rowKey = ranges[run];
    let columnLow = ranges[run + 1u];
    let columnHigh = ranges[run + 2u];
    let feature = coverFindCostFeature(rowKey);
    var start = columnLow;
    isHead = true;
    if (index > 0u) {
      let previous = run - 8u;
      let previousRow = ranges[previous];
      if (previousRow == rowKey) {
        start = max(start, ranges[previous + 2u] + 1u);
      }
      isHead = coverFindCostFeature(previousRow) != feature;
    }
    runLength = select(0u, columnHigh + 1u - start, (columnHigh >= start));
    ranges[run + 3u] = feature;
    ranges[run + 5u] = start;
    ranges[run + 6u] = rowKey - costStarts[costStartsOffset + feature];
  }
  runLengths[runLengthsOffset + index] = runLength;
  runFlags[runFlagsOffset + index] = select(0u, 1u, isHead);`}),...new i({id:S(`run-scan`),input:X,output:Z,mode:`exclusive`,segmentFlags:Q}).getCommandNodes(e),f(e,{id:S(`finish`),operation:g,variant:`scanline-finish`,bindings:[{name:`ranges`,view:h,type:`atomic<u32>`,access:`read_write`},{name:`runOffsets`,view:Z,type:`u32`,access:`read`},{name:`runLengths`,view:X,type:`u32`,access:`read`},{name:`covered`,view:G,type:`u32`,access:`read`},{name:`gapSlots`,view:q,type:`u32`,access:`read`},{name:`counts`,view:m,type:`atomic<u32>`,access:`read_write`},{name:`countCarry`,view:M,type:`atomic<u32>`,access:`read_write`}],invocationCount:v,declarations:w,body:`if (index >= ${ae}) {
    return;
  }
  let run = ${y} + index * 8u;
  let feature = atomicLoad(&ranges[run + 3u]);
  atomicStore(&ranges[run + 4u], runOffsets[runOffsetsOffset + index]);
  let featureBase = rangesOffset + feature * STRIDE;
  if (index == 0u || atomicLoad(&ranges[run - 8u + 3u]) != feature) {
    atomicStore(&ranges[featureBase + 6u], index);
  }
  atomicAdd(&ranges[featureBase + 7u], 1u);
  // Exact 64-bit candidate count: a wrapped low word carries one into the high word.
  let runLength = runLengths[runLengthsOffset + index];
  let before = atomicAdd(&counts[countsOffset + feature], runLength);
  if (before + runLength < before) {
    atomicAdd(&countCarry[countCarryOffset + feature], 1u);
  }`}),f(e,{id:S(`clamp`),operation:g,variant:`scanline-clamp`,bindings:[{name:`ranges`,view:h,type:`u32`,access:`read_write`},{name:`counts`,view:m,type:`u32`,access:`read_write`},{name:`countCarry`,view:M,type:`u32`,access:`read`}],invocationCount:o,declarations:`${w}
const CAPACITY_PLUS_ONE: u32 = ${s+1}u;`,body:`let flagIndex = rangesOffset + index * STRIDE + 5u;
  if ((ranges[flagIndex] & 2u) != 0u) {
    counts[countsOffset + index] = min(counts[countsOffset + index], CAPACITY_PLUS_ONE);
    // 2^32 or more candidates: far beyond any capacity, and the 32-bit run offsets of the feature
    // wrapped. Its count saturates (candidate overflow) and it falls back to the bounding box.
    if (countCarry[countCarryOffset + index] != 0u) {
      counts[countsOffset + index] = CAPACITY_PLUS_ONE;
      ranges[flagIndex] = ranges[flagIndex] & 1u;
    }
  }`})),b}function Z(e,t){let n=`rangesOffset + ${e}u * 8u`;return`var column = candidateLocal % ${t};
  var row = candidateLocal / ${t};
  if ((ranges[rangeBase + 5u] & 2u) != 0u) {
    // Runs of this feature are [first, first + runCount); offsets are local to the feature, and
    // the run holding the candidate is the last one that starts at or before it.
    let firstRun = ranges[rangeBase + 6u];
    var low = firstRun;
    var high = firstRun + ranges[rangeBase + 7u];
    while (low < high) {
      let middle = (low + high) >> 1u;
      if (ranges[${n} + middle * 8u + 4u] <= candidateLocal) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    let run = ${n} + (low - 1u) * 8u;
    column = ranges[run + 5u] + (candidateLocal - ranges[run + 4u]);
    row = ranges[run + 6u];
  }`}var Q=`GPUCellCover`,ae={center:0,full:1,intersects:2},oe=2147483648,se=2e-4;function ce(e){return e.scanline?8:5}function le(e){return[{name:`positions`,view:e.positions,type:`f32`,access:`read`},{name:`featureOffsets`,view:e.featureOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:e.polygonOffsets,type:`u32`,access:`read`},{name:`ringOffsets`,view:e.ringOffsets,type:`u32`,access:`read`}]}var ue=`${C}
${S}
fn coverGetVertex(vertex: u32) -> vec2f {
  let rangeBase = positionsOffset + 2u * vertex;
  return vec2f(positions[rangeBase], positions[rangeBase + 1u]);
}

fn coverIsFinite(p: vec2f) -> bool {
  return isFiniteBits(p.x) && isFiniteBits(p.y) && abs(p.x) < 1e30 && abs(p.y) < 1e30;
}

/** Whether the closed segment a-b meets the open rectangle (lo, hi). */
fn coverSegmentHitsRect(a: vec2f, b: vec2f, lo: vec2f, hi: vec2f) -> bool {
  if (!coverIsFinite(a) || !coverIsFinite(b)) { return false; }
  var lower = -1e30;
  var upper = 1e30;
  let d = b - a;
  if (d.x == 0.0) {
    if (!(lo.x < a.x && a.x < hi.x)) { return false; }
  } else {
    let t1 = (lo.x - a.x) / d.x;
    let t2 = (hi.x - a.x) / d.x;
    lower = max(lower, min(t1, t2));
    upper = min(upper, max(t1, t2));
  }
  if (d.y == 0.0) {
    if (!(lo.y < a.y && a.y < hi.y)) { return false; }
  } else {
    let t1 = (lo.y - a.y) / d.y;
    let t2 = (hi.y - a.y) / d.y;
    lower = max(lower, min(t1, t2));
    upper = min(upper, max(t1, t2));
  }
  return lower < upper && lower < 1.0 && upper > 0.0;
}
`,$=`${ue}
${b}
struct CoverFeatureInfo {
  bounds: vec4f,
  vertexCount: u32,
  vertexBegin: u32,
  vertexEnd: u32,
  hasNonFinite: bool,
  hasCrossingEdge: bool
}

fn coverGetFeatureInfo(feature: u32) -> CoverFeatureInfo {
  var info = CoverFeatureInfo(vec4f(1e30, 1e30, -1e30, -1e30), 0u, 0u, 0u, false, false);
  var isFirstRing = true;
  let firstPolygon = featureOffsets[featureOffsetsOffset + feature];
  let endPolygon = featureOffsets[featureOffsetsOffset + feature + 1u];
  for (var polygon = firstPolygon; polygon < endPolygon; polygon++) {
    let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
    let endRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
    for (var ring = firstRing; ring < endRing; ring++) {
      let firstVertex = ringOffsets[ringOffsetsOffset + ring];
      let endVertex = ringOffsets[ringOffsetsOffset + ring + 1u];
      if (isFirstRing) {
        info.vertexBegin = firstVertex;
        isFirstRing = false;
      }
      info.vertexEnd = endVertex;
      if (endVertex > firstVertex) {
        info.vertexCount += endVertex - firstVertex;
      }
      for (var vertex = firstVertex; vertex < endVertex; vertex++) {
        let p = coverGetVertex(vertex);
        let next = coverGetVertex(select(vertex + 1u, firstVertex, vertex + 1u == endVertex));
        info.hasCrossingEdge = info.hasCrossingEdge || lonLatIsCrossingEdge(p, next);
        if (coverIsFinite(p)) {
          info.bounds = vec4f(min(info.bounds.xy, p), max(info.bounds.zw, p));
        } else {
          info.hasNonFinite = true;
        }
      }
    }
  }
  return info;
}

${x(`coverGetRingWinding`,`coverGetVertex`)}
// Even-odd containment of the SoS-perturbed point p over every ring of the feature.
fn coverContains(feature: u32, p: vec2f) -> bool {
  let probe = ExactProbe(p, p, p, EXACT_PROBE_POINT);
  var winding = 0;
  let firstPolygon = featureOffsets[featureOffsetsOffset + feature];
  let endPolygon = featureOffsets[featureOffsetsOffset + feature + 1u];
  for (var polygon = firstPolygon; polygon < endPolygon; polygon++) {
    let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
    let endRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
    for (var ring = firstRing; ring < endRing; ring++) {
      let firstVertex = ringOffsets[ringOffsetsOffset + ring];
      let endVertex = ringOffsets[ringOffsetsOffset + ring + 1u];
      // Rings of two vertices or fewer enclose nothing (the helper returns 0 for them).
      winding += coverGetRingWinding(probe, firstVertex, endVertex).x;
    }
  }
  return (winding & 1) != 0;
}

fn coverEdgesHitRect(feature: u32, lo: vec2f, hi: vec2f) -> bool {
  let firstPolygon = featureOffsets[featureOffsetsOffset + feature];
  let endPolygon = featureOffsets[featureOffsetsOffset + feature + 1u];
  for (var polygon = firstPolygon; polygon < endPolygon; polygon++) {
    let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
    let endRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
    for (var ring = firstRing; ring < endRing; ring++) {
      let firstVertex = ringOffsets[ringOffsetsOffset + ring];
      let endVertex = ringOffsets[ringOffsetsOffset + ring + 1u];
      for (var vertex = firstVertex; vertex < endVertex; vertex++) {
        let a = coverGetVertex(vertex);
        let b = coverGetVertex(select(vertex + 1u, firstVertex, vertex + 1u == endVertex));
        if (coverSegmentHitsRect(a, b, lo, hi)) { return true; }
      }
    }
  }
  return false;
}
`;function de(e){return`${ue}${T(e)}
// Even-odd containment of the SoS-perturbed point p. Slabbed and walked features share one edge
// loop, so the exact crossing helper has a single call site.
fn coverContains(feature: u32, p: vec2f) -> bool {
  let probe = ExactProbe(p, p, p, EXACT_PROBE_POINT);
  var winding = 0;
  ${E({owner:`feature`,lowY:`p.y`,highY:`p.y`,getVertex:`coverGetVertex`,body:`winding += exactProbeCrossing(probe, a, b).x;`})}
  return (winding & 1) != 0;
}

fn coverEdgesHitRect(feature: u32, lo: vec2f, hi: vec2f) -> bool {
  ${E({owner:`feature`,lowY:`lo.y`,highY:`hi.y`,getVertex:`coverGetVertex`,body:`if (coverSegmentHitsRect(a, b, lo, hi)) { return true; }`})}
  return false;
}
`}function fe(e){return`
const FEATURE_COUNT: u32 = ${e}u;

/** Feature whose candidate range [starts[f], starts[f + 1]) contains \`candidate\`. */
fn coverFindFeature(candidate: u32) -> u32 {
  var low = 0u;
  var high = FEATURE_COUNT;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (starts[startsOffset + middle + 1u] > candidate) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  return min(low, FEATURE_COUNT - 1u);
}
`}function pe(e){let t=e.resolution;return`
const TILE_WIDTH: f32 = ${u(45*2**(3-t))};
const ROW_SCALE: f32 = ${u(2**(1-t))};
const PI: f32 = 3.14159265358979;
`}function me(e,t){let n=t.invalidCount?`if (hasCrossingEdge) { atomicAdd(&invalidCount[invalidCountOffset], 1u); }`:``,r=e===`h3`?89:85.051129;return`let hasCrossingEdge = info.hasCrossingEdge;
  ${n}
  ${t.uncertainCount?`if (!hasCrossingEdge && bounds.x <= bounds.z && (bounds.w > ${u(r)} || bounds.y < -${u(r)})) { atomicAdd(&uncertainCount[uncertainCountOffset], 1u); }`:``}`}function he(e){return[`invalidCount`,`uncertainCount`].flatMap(t=>{let n=e[t];return n?[{name:t,view:n,type:`atomic<u32>`,access:`read_write`}]:[]})}function ge(e,t,n,r,i={}){let{resolution:o,candidateCapacity:c,family:l}=t,d=me(l,i),p=c+1,m=t.scanline?`ranges[rangeBase + 5u] = select(0u, 1u, info.hasNonFinite);`:``,h=`${$}
const CAPACITY_PLUS_ONE: u32 = ${p}u;
const RANGE_STRIDE: u32 = ${ce(t)}u;
fn coverClampCount(width: u32, height: u32) -> u32 {
  if (width == 0u || height == 0u) { return 0u; }
  if (width > CAPACITY_PLUS_ONE || height > CAPACITY_PLUS_ONE) { return CAPACITY_PLUS_ONE; }
  if (width > CAPACITY_PLUS_ONE / height) { return CAPACITY_PLUS_ONE; }
  return width * height;
}
`,g;if(l===`quadbin`)h+=`${s}${a}const RESOLUTION: u32 = ${o}u;`,g=`if (index == FEATURE_COUNT) { counts[countsOffset + index] = 0u; return; }
  let info = coverGetFeatureInfo(index);
  let bounds = info.bounds;
  let rangeBase = rangesOffset + index * RANGE_STRIDE;
  var count = 0u;
  ${m}
  ${d}
  if (!hasCrossingEdge && bounds.x <= bounds.z && bounds.y <= bounds.w) {
    // Point indexing wraps +180 to column zero; bounds must retain the easternmost column.
    let lastColumn = (1u << RESOLUTION) - 1u;
    let x0 = select(quadbinGetTileX(bounds.x, RESOLUTION), lastColumn, bounds.x >= 180.0);
    let x1 = select(quadbinGetTileX(bounds.z, RESOLUTION), lastColumn, bounds.z >= 180.0);
    let y0 = quadbinGetTileY(bounds.w, RESOLUTION);
    let y1 = quadbinGetTileY(bounds.y, RESOLUTION);
    if (x1 >= x0 && y1 >= y0) {
      ranges[rangeBase] = x0;
      ranges[rangeBase + 1u] = y0;
      ranges[rangeBase + 2u] = x1 - x0 + 1u;
      ranges[rangeBase + 3u] = y1 - y0 + 1u;
      count = coverClampCount(x1 - x0 + 1u, y1 - y0 + 1u);
    }
  }
  counts[countsOffset + index] = count;`;else{let e=B[o];h+=`const LATTICE_SPACING: f32 = ${u(e)};
const MAXIMUM_LATITUDE: f32 = ${u(89)};
const DEGREES_TO_RADIANS: f32 = 0.017453292519943295;
fn coverToCount(value: f32) -> u32 {
  return u32(clamp(value, 1.0, 4.0e9));
}`,g=`if (index == FEATURE_COUNT) { counts[countsOffset + index] = 0u; return; }
  let info = coverGetFeatureInfo(index);
  let bounds = info.bounds;
  let rangeBase = rangesOffset + index * RANGE_STRIDE;
  var count = 0u;
  ${m}
  let originLat = max(bounds.y - LATTICE_SPACING, -MAXIMUM_LATITUDE);
  let endLat = min(bounds.w + LATTICE_SPACING, MAXIMUM_LATITUDE);
  ${d}
  if (!hasCrossingEdge && bounds.x <= bounds.z && bounds.y <= bounds.w && originLat <= endLat) {
    // Longitude spacing from the most equatorward latitude of the band, where a degree of longitude
    // is longest: a spacing taken from the poleward edge is too coarse for the rows nearer the
    // equator and loses cells there.
    let equatorwardLat = select(min(abs(originLat), abs(endLat)), 0.0, originLat <= 0.0 && endLat >= 0.0);
    let spacingLng = LATTICE_SPACING / cos(equatorwardLat * DEGREES_TO_RADIANS);
    let originLng = bounds.x - spacingLng;
    let width = coverToCount(ceil((bounds.z + spacingLng - originLng) / spacingLng) + 1.0);
    let height = coverToCount(ceil((endLat - originLat) / LATTICE_SPACING) + 1.0);
    ranges[rangeBase] = width;
    ranges[rangeBase + 1u] = height;
    ranges[rangeBase + 2u] = bitcast<u32>(originLng);
    ranges[rangeBase + 3u] = bitcast<u32>(originLat);
    ranges[rangeBase + 4u] = bitcast<u32>(spacingLng);
    count = coverClampCount(width, height);
  }
  counts[countsOffset + index] = count;`}return h+=`const FEATURE_COUNT: u32 = ${t.featureCount}u;`,f(e,{id:`${t.id}-count`,operation:Q,variant:`count`,bindings:[...le(t),{name:`counts`,view:n,type:`u32`,access:`read_write`},{name:`ranges`,view:r,type:`u32`,access:`read_write`},...he(i)],invocationCount:t.featureCount+1,declarations:h,body:g})}function _e(e,t,n){let{family:r,resolution:i,candidateCapacity:c,edgeSlabs:d}=t,p=ae[t.containment],m=t.scanline?Z(t.featureCount,`width`):`let column = candidateLocal % width;
  let row = candidateLocal / width;`,g=d?O(d):[],_=`${d?de(d):$}${fe(t.featureCount)}
const CANDIDATE_CAPACITY: u32 = ${c}u;
const RANGE_STRIDE: u32 = ${ce(t)}u;
const MODE: u32 = ${p}u;
const RESOLUTION: u32 = ${i}u;
const CORE_BIT: u32 = ${oe}u;
const CORE_MARGIN: f32 = ${u(se)};
`,v;if(r===`quadbin`){let e=o(`quadbin`,i);_+=`${pe(t)}${s}${a}
fn coverTileLatitude(row: f32) -> f32 {
  let t = PI * (1.0 - row * ROW_SCALE);
  return (2.0 * atan(exp(t)) - PI * 0.5) * 57.29577951308232;
}
`,v=`let width = ranges[rangeBase + 2u];
  ${m}
  let x = ranges[rangeBase] + column;
  let y = ranges[rangeBase + 1u] + row;
  let lng0 = f32(x) * TILE_WIDTH - 180.0;
  let lng1 = f32(x + 1u) * TILE_WIDTH - 180.0;
  let latNorth = coverTileLatitude(f32(y));
  let latSouth = coverTileLatitude(f32(y + 1u));
  let centre = vec2f((f32(x) + 0.5) * TILE_WIDTH - 180.0, coverTileLatitude(f32(y) + 0.5));
  // One call site for the exact containment test (see COMMON_WGSL).
  let centreInside = coverContains(feature, centre);
  var accepted = false;
  if (MODE == 0u) {
    accepted = centreInside;
  } else {
    let hit = coverEdgesHitRect(feature, vec2f(lng0, latSouth), vec2f(lng1, latNorth));
    if (MODE == 1u) {
      accepted = !hit && centreInside;
    } else {
      accepted = hit || centreInside;
    }
  }
  var key = vec2u(0u);
  if (accepted) {
    key = cellGetKey(quadbinGetCompactKey(x, y), ${e.headerHigh}u, RESOLUTION, ${e.lowBit}u);
    ${t.computeCore?`// Core: the centre is inside and no polygon edge enters the cell grown by a safety margin.
    let slack = vec2f(CORE_MARGIN + 0.001 * (lng1 - lng0));
    if (centreInside && !coverEdgesHitRect(feature, vec2f(lng0, latSouth) - slack, vec2f(lng1, latNorth) + slack)) {
      key.x = key.x | CORE_BIT;
    }`:``}
  }`}else _+=`${l.source}\n${w}\n${t.computeCore?h:``}`,v=`let width = ranges[rangeBase];
  let originLng = bitcast<f32>(ranges[rangeBase + 2u]);
  let originLat = bitcast<f32>(ranges[rangeBase + 3u]);
  let spacingLng = bitcast<f32>(ranges[rangeBase + 4u]);
  let spacingLat = ${u(B[i])};
  ${m}
  let point = vec2f(originLng + f32(column) * spacingLng, originLat + f32(row) * spacingLat);
  var key = vec2u(0u);
  let cell = cellIndexH3FromLngLat(point, RESOLUTION);
  if (cell.x != 0u || cell.y != 0u) {
    let centre = dggs_h3_get_center_lnglat(cell);
    // Shift the center by whole turns next to the lattice, then round it to the nearest lattice
    // index with one floating point expression of the center alone, so exactly one lattice point
    // claims each cell (no tie ambiguity between neighbouring points).
    let middleLng = originLng + 0.5 * f32(width) * spacingLng;
    let centreLng = centre.x + 360.0 * round((middleLng - centre.x) / 360.0);
    let nearestColumn = floor((centreLng - originLng) / spacingLng + 0.5);
    let nearestRow = floor((centre.y - originLat) / spacingLat + 0.5);
    if (nearestColumn == f32(column) && nearestRow == f32(row) && coverContains(feature, centre)) {
      key = cell;
      ${t.computeCore?`// Core: no polygon edge enters the bounding box of the cell boundary, grown by a margin.
      let boundary = cellIndexH3GetBoundary(cell);
      var lo = boundary.points[0];
      var hi = boundary.points[0];
      for (var vertex = 1u; vertex < boundary.count; vertex++) {
        lo = min(lo, boundary.points[vertex]);
        hi = max(hi, boundary.points[vertex]);
      }
      let slack = vec2f(CORE_MARGIN) + 0.001 * (hi - lo);
      if (boundary.count > 0u && hi.x - lo.x < 90.0 && !coverEdgesHitRect(feature, lo - slack, hi + slack)) {
        key.x = key.x | CORE_BIT;
      }`:``}
    }
  }
  let accepted = key.x != 0u || key.y != 0u;`;return f(e,{id:`${t.id}-test`,operation:Q,variant:d?`test-slabs`:`test`,bindings:[...d?[{name:`positions`,view:t.positions,type:`f32`,access:`read`}]:le(t),{name:`ranges`,view:n.ranges,type:`u32`,access:`read`},{name:`starts`,view:n.starts,type:`u32`,access:`read`},{name:`flags`,view:n.flags,type:`u32`,access:`read_write`},{name:`candidateCells`,view:n.candidateCells,type:`u32`,access:`read_write`},...g],invocationCount:c,declarations:_,body:`let activeCount = min(starts[startsOffset + FEATURE_COUNT], CANDIDATE_CAPACITY);
  if (index >= activeCount) {
    flags[flagsOffset + index] = 0u;
    return;
  }
  let feature = coverFindFeature(index);
  let candidateLocal = index - starts[startsOffset + feature];
  let rangeBase = rangesOffset + feature * RANGE_STRIDE;
  ${v}
  flags[flagsOffset + index] = select(0u, 1u, accepted);
  candidateCells[candidateCellsOffset + 2u * index] = key.y;
  candidateCells[candidateCellsOffset + 2u * index + 1u] = key.x;`})}function ve(e,t,n){let r=[{name:`starts`,view:n.starts,type:`u32`,access:`read`},{name:`accepted`,view:n.accepted,type:`u32`,access:`read`},{name:`candidateCells`,view:n.candidateCells,type:`u32`,access:`read`}];return n.featureIds&&r.push({name:`featureIds`,view:n.featureIds,type:`u32`,access:`read`}),r.push({name:`outputFeatureIds`,view:n.outputFeatureIds,type:`u32`,access:`read_write`},{name:`outputCells`,view:n.outputCells,type:`u32`,access:`read_write`}),n.outputCore&&r.push({name:`outputCore`,view:n.outputCore,type:`u32`,access:`read_write`}),f(e,{id:`${t.id}-write`,operation:Q,variant:`write`,bindings:r,invocationCount:t.candidateCapacity,declarations:`${fe(t.featureCount)}
const OUTPUT_CAPACITY: u32 = ${n.outputFeatureIds.length}u;
const CORE_BIT: u32 = ${oe}u;`,body:`var beforeCount = 0u;
  if (index > 0u) { beforeCount = accepted[acceptedOffset + index - 1u]; }
  let afterCount = accepted[acceptedOffset + index];
  if (afterCount == beforeCount) { return; }
  let slot = afterCount - 1u;
  if (slot >= OUTPUT_CAPACITY) { return; }
  let feature = coverFindFeature(index);
  outputFeatureIds[outputFeatureIdsOffset + slot] = ${n.featureIds?`featureIds[featureIdsOffset + feature]`:`feature`};
  outputCells[outputCellsOffset + 2u * slot] = candidateCells[candidateCellsOffset + 2u * index];
  let highWord = candidateCells[candidateCellsOffset + 2u * index + 1u];
  outputCells[outputCellsOffset + 2u * slot + 1u] = highWord & ~CORE_BIT;
  ${n.outputCore?`outputCore[outputCoreOffset + slot] = select(0u, 1u, (highWord & CORE_BIT) != 0u);`:``}`})}function ye(e,t,n){return f(e,{id:`${t.id}-finalize`,operation:Q,variant:`finalize`,bindings:[{name:`starts`,view:n.starts,type:`u32`,access:`read`},{name:`accepted`,view:n.accepted,type:`u32`,access:`read`},{name:`total`,view:n.total,type:`u32`,access:`read_write`},{name:`candidateOverflow`,view:n.candidateOverflow,type:`u32`,access:`read_write`},...n.candidateCount?[{name:`candidateCount`,view:n.candidateCount,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:`const FEATURE_COUNT: u32 = ${t.featureCount}u;
const CANDIDATE_CAPACITY: u32 = ${t.candidateCapacity}u;`,body:`total[totalOffset] = accepted[acceptedOffset + CANDIDATE_CAPACITY - 1u];
  candidateOverflow[candidateOverflowOffset] =
    select(0u, 1u, starts[startsOffset + FEATURE_COUNT] > CANDIDATE_CAPACITY);
  ${n.candidateCount?`candidateCount[candidateCountOffset] = starts[startsOffset + FEATURE_COUNT];`:``}`})}var be=1073741823,xe=715827882,Se=.001,Ce=6;function we(e){return 5+Number(!!e.output.status.invalidCount)+Number(!!e.output.status.uncertainCount)+(e.resolution+1)*(1+_(e.candidateCapacity+1))+e.resolution}function Te(e,n,r){let i=r.featureOffsets.length-1,a=r.candidateCapacity;v(e,n,r.resolution+1),t(e,n,{positions:r.positions,featureOffsets:r.featureOffsets,polygonOffsets:r.polygonOffsets,ringOffsets:r.ringOffsets,featureIds:r.featureIds,outputFeatureIds:r.output.featureIds,outputCells:r.output.cells,outputCore:r.output.core,...r.output.status},[[`S2 frontier cells`,a*8],[`S2 frontier feature rows`,a*4],[`S2 split scan`,(a+1)*4],[`S2 feature metadata`,(i*Ce+1)*4],[`S2 control`,8],[`S2 round parameters`,(r.resolution+1)*64*4]])}function Ee(e,t,n){let{output:a,candidateCapacity:o,resolution:s}=n,c=n.featureOffsets.length-1,u=s+1,d=y(e,t,u),h=d.slots.map(t=>e.createDataView(t.buffer,{format:`uint32`,length:2,byteOffset:t.byteOffset})),_=e.createDataView(d.slots[0].buffer,{format:`uint32`,length:u*64}),v=[r(e,`${t}-s2-cells-a`,`uint32x2`,o),r(e,`${t}-s2-cells-b`,`uint32x2`,o)],b=[r(e,`${t}-s2-features-a`,`uint32`,o),r(e,`${t}-s2-features-b`,`uint32`,o)],x=r(e,`${t}-s2-feature-metadata`,`uint32`,Ce*c+1),S=r(e,`${t}-s2-counts`,`uint32`,o+1),C=r(e,`${t}-s2-offsets`,`uint32`,o+1),w=r(e,`${t}-s2-control`,`uint32`,2),T=[...d.nodes];for(let n of[`invalidCount`,`uncertainCount`]){let r=a.status[n];r&&T.push(p(e,{id:`${t}-s2-clear-${n}`,operation:`GPUCellCover`,view:r,type:`u32`,value:`0u`}))}let E=[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`featureOffsets`,view:n.featureOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:n.polygonOffsets,type:`u32`,access:`read`},{name:`ringOffsets`,view:n.ringOffsets,type:`u32`,access:`read`}];T.push(f(e,{id:`${t}-s2-features`,operation:`GPUCellCover`,variant:`s2-features`,invocationCount:c+1,bindings:[...E,{name:`metadata`,view:x,type:`u32`,access:`read_write`},...n.featureIds?[{name:`sourceIds`,view:n.featureIds,type:`u32`,access:`read`}]:[],...a.status.invalidCount?[{name:`invalidCount`,view:a.status.invalidCount,type:`atomic<u32>`,access:`read_write`}]:[]],declarations:$,body:`metadata[metadataOffset + index] = featureOffsets[featureOffsetsOffset + index];
  if (index < ${c}u) {
    let info = coverGetFeatureInfo(index);
    let valid = !info.hasCrossingEdge && !info.hasNonFinite && info.bounds.x <= info.bounds.z && info.bounds.y <= info.bounds.w;
    let bounds = select(vec4f(1.0, 1.0, -1.0, -1.0), info.bounds, valid);
    let base = metadataOffset + ${c+1}u + 5u * index;
    for (var axis = 0u; axis < 4u; axis++) { metadata[base + axis] = bitcast<u32>(bounds[axis]); }
    metadata[base + 4u] = ${n.featureIds?`sourceIds[sourceIdsOffset + index]`:`index`};
    ${a.status.invalidCount?`if (info.hasCrossingEdge) { atomicAdd(&invalidCount[invalidCountOffset], 1u); }`:``}
  }`}),f(e,{id:`${t}-s2-start`,operation:`GPUCellCover`,variant:`s2-start`,invocationCount:o,bindings:[{name:`cells`,view:v[0],type:`u32`,access:`read_write`},{name:`features`,view:b[0],type:`u32`,access:`read_write`},{name:`parameters`,view:_,type:`u32`,access:`read_write`},{name:`control`,view:w,type:`u32`,access:`read_write`}],body:`if (index == 0u) { parameters[parametersOffset + 1u] = ${Math.min(c*6,o)}u;
    control[controlOffset] = ${Number(c*6>o)}u; control[controlOffset + 1u] = 0u; }
  if (index < ${Math.min(c*6,o)}u) {
    cells[cellsOffset + 2u * index] = 0u;
    cells[cellsOffset + 2u * index + 1u] = (2u * (index % 6u) + 1u) << 28u;
    features[featuresOffset + index] = index / 6u;
  }`}));let D=`${l.source}\n${m}\n${g}\n${$}
fn s2CoverCellBounds(cell: vec2u) -> vec4f {
  let surface = dggs_s2_ij_to_st(dggs_s2_get_ij(cell), dggs_s2_get_level(cell), vec2f(0.5));
  let center = normalize(dggs_s2_face_uv_to_xyz(dggs_s2_get_face(cell), dggs_s2_st_to_uv(surface)));
  var radius = 0.0;
  for (var vertex = 0u; vertex < 4u; vertex++) { radius = max(radius, s2AnalysisAngle(center, s2AnalysisVertex(cell, vertex))); }
  let angularMargin = ${Se} * 0.017453292519943295;
  radius += angularMargin;
  let longitudeLatitude = geometryS2XyzToLngLat(center);
  let latitudeMargin = radius * 57.29577951308232;
  var lowerLongitude = -180.0; var upperLongitude = 180.0;
  if (abs(longitudeLatitude.y) + latitudeMargin < 90.0) {
    let width = asin(clamp(sin(radius) / cos(radians(longitudeLatitude.y)), 0.0, 1.0)) * 57.29577951308232 + ${Se};
    if (longitudeLatitude.x - width > -180.0 && longitudeLatitude.x + width < 180.0) {
      lowerLongitude = longitudeLatitude.x - width; upperLongitude = longitudeLatitude.x + width;
    }
  }
  return vec4f(lowerLongitude, max(-90.0, longitudeLatitude.y - latitudeMargin), upperLongitude, min(90.0, longitudeLatitude.y + latitudeMargin));
}`;for(let n=0;n<u;n++){let r=v[n%2],u=b[n%2];T.push(f(e,{id:`${t}-s2-classify-${n}`,operation:`GPUCellCover`,variant:`s2-classify`,invocationCount:o+1,bindings:[E[0],{name:`featureOffsets`,view:x,type:`u32`,access:`read`},E[2],E[3],{name:`cells`,view:r,type:`u32`,access:`read`},{name:`features`,view:u,type:`u32`,access:`read_write`},{name:`counts`,view:S,type:`u32`,access:`read_write`},{name:`parameters`,view:h[n],type:`u32`,access:`read`}],declarations:D,body:`var count = 0u;
  if (index < min(parameters[parametersOffset + 1u], ${o}u)) {
    let feature = features[featuresOffset + index];
    let base = featureOffsetsOffset + ${c+1}u + 5u * feature;
    let bounds = vec4f(bitcast<f32>(featureOffsets[base]), bitcast<f32>(featureOffsets[base + 1u]), bitcast<f32>(featureOffsets[base + 2u]), bitcast<f32>(featureOffsets[base + 3u]));
    let cell = vec2u(cells[cellsOffset + 2u * index + 1u], cells[cellsOffset + 2u * index]);
    let cellBounds = s2CoverCellBounds(cell);
    if (bounds.x <= bounds.z && all(cellBounds.xy <= bounds.zw) && all(bounds.xy <= cellBounds.zw)) {
      if (parameters[parametersOffset] < ${s}u) { count = 4u; }
      else if (coverContains(feature, geometryGetCenter(cell))) {
        count = 1u;
        ${a.core?`if (!coverEdgesHitRect(feature, cellBounds.xy, cellBounds.zw)) { features[featuresOffset + index] = feature | 0x80000000u; }`:``}
      }
    }
  }
  counts[countsOffset + index] = count;`}),...new i({id:`${t}-s2-scan-${n}`,input:S,output:C,mode:`exclusive`}).getCommandNodes(e)),n<s&&T.push(f(e,{id:`${t}-s2-write-${n}`,operation:`GPUCellCover`,variant:`s2-write`,invocationCount:o,bindings:[{name:`cells`,view:r,type:`u32`,access:`read`},{name:`features`,view:u,type:`u32`,access:`read`},{name:`counts`,view:S,type:`u32`,access:`read`},{name:`offsets`,view:C,type:`u32`,access:`read`},{name:`nextCells`,view:v[(n+1)%2],type:`u32`,access:`read_write`},{name:`nextFeatures`,view:b[(n+1)%2],type:`u32`,access:`read_write`},{name:`parameters`,view:_,type:`u32`,access:`read_write`},{name:`control`,view:w,type:`atomic<u32>`,access:`read_write`}],declarations:l.source,body:`if (index == 0u) {
    let required = offsets[offsetsOffset + ${o}u];
    let roundIndex = atomicAdd(&control[controlOffset + 1u], 1u);
    parameters[parametersOffset + (roundIndex + 1u) * 64u + 1u] = min(required, ${o}u);
    if (required > ${o}u) { atomicOr(&control[controlOffset], 1u); }
  }
  if (counts[countsOffset + index] != 0u) {
    let cell = vec2u(cells[cellsOffset + 2u * index + 1u], cells[cellsOffset + 2u * index]);
    let level = dggs_s2_get_level(cell);
    let marker = dggs_u64_shift_left(vec2u(0u, 1u), 2u * (30u - level));
    let childMarker = dggs_u64_shift_right(marker, 2u);
    var child = dggs_u64_add(dggs_u64_subtract(cell, marker), childMarker);
    for (var slot = 0u; slot < 4u; slot++) {
      let destination = offsets[offsetsOffset + index] + slot;
      if (destination < ${o}u) { nextCells[nextCellsOffset + 2u * destination] = child.y; nextCells[nextCellsOffset + 2u * destination + 1u] = child.x; nextFeatures[nextFeaturesOffset + destination] = features[featuresOffset + index]; }
      child = dggs_u64_add(child, dggs_u64_shift_left(childMarker, 1u));
    }
  }`}))}let O=v[s%2],ee=b[s%2];return T.push(f(e,{id:`${t}-s2-emit`,operation:`GPUCellCover`,variant:`s2-emit`,invocationCount:o,bindings:[{name:`cells`,view:O,type:`u32`,access:`read`},{name:`features`,view:ee,type:`u32`,access:`read`},{name:`counts`,view:S,type:`u32`,access:`read`},{name:`offsets`,view:C,type:`u32`,access:`read`},{name:`metadata`,view:x,type:`u32`,access:`read`},{name:`outputCells`,view:a.cells,type:`u32`,access:`read_write`},{name:`outputIds`,view:a.featureIds,type:`u32`,access:`read_write`},...a.core?[{name:`core`,view:a.core,type:`u32`,access:`read_write`}]:[]],body:`let destination = offsets[offsetsOffset + index];
  if (counts[countsOffset + index] != 0u && destination < ${a.cells.length}u) {
    let featureWord = features[featuresOffset + index];
    let feature = featureWord & 0x7fffffffu;
    outputCells[outputCellsOffset + 2u * destination] = cells[cellsOffset + 2u * index];
    outputCells[outputCellsOffset + 2u * destination + 1u] = cells[cellsOffset + 2u * index + 1u];
    outputIds[outputIdsOffset + destination] = metadata[metadataOffset + ${c+1}u + 5u * feature + 4u];
    ${a.core?`core[coreOffset + destination] = featureWord >> 31u;`:``}
  }`}),f(e,{id:`${t}-s2-publish`,operation:`GPUCellCover`,variant:`s2-publish`,invocationCount:1,bindings:[{name:`offsets`,view:C,type:`u32`,access:`read`},{name:`control`,view:w,type:`u32`,access:`read`},{name:`parameters`,view:h[s],type:`u32`,access:`read`},{name:`count`,view:a.status.count,type:`u32`,access:`read_write`},{name:`overflow`,view:a.status.overflow,type:`u32`,access:`read_write`},{name:`candidateOverflow`,view:a.status.candidateOverflow,type:`u32`,access:`read_write`},...a.status.requiredCount?[{name:`requiredCount`,view:a.status.requiredCount,type:`u32`,access:`read_write`}]:[],...a.status.candidateCount?[{name:`candidateCount`,view:a.status.candidateCount,type:`u32`,access:`read_write`}]:[]],body:`let required = offsets[offsetsOffset + ${o}u];
  count[countOffset] = min(required, ${a.cells.length}u); overflow[overflowOffset] = select(0u, 1u, required > ${a.cells.length}u);
  candidateOverflow[candidateOverflowOffset] = control[controlOffset];
  ${a.status.requiredCount?`requiredCount[requiredCountOffset] = required;`:``}
  ${a.status.candidateCount?`candidateCount[candidateCountOffset] = parameters[parametersOffset + 1u];`:``}`})),T}function De(e){return L(e,8)}function Oe(e){return L(e,8)}var ke=`cell-cover`,Ae=4294967295,je=4e6,Me=class{id;props;spatialContext;featureCount;containment;edgeSlabs;candidateEnumeration;scanline;scanlineSlotCapacity;estimatedCommandNodeCount;constructor(t){this.id=t.id??ke,this.props=t;let{id:n}=this;if(this.spatialContext=M(n,t.spatialContext??j,{...k,edgeModels:A}),![`quadbin`,`h3`,`s2`].includes(t.family))throw Error(`${n} family must be 'quadbin', 'h3' or 's2'`);if(c(n,t.family,t.resolution),t.family===`s2`?F(n,`s2 resolution`,t.resolution,0,15):o(t.family,t.resolution),this.containment=t.containment??`center`,![`center`,`full`,`intersects`].includes(this.containment))throw Error(`${n} containment must be 'center', 'full' or 'intersects'`);if(t.family!==`quadbin`&&this.containment!==`center`)throw Error(`${n} ${t.family} supports only containment 'center'`);N(t.positions,[`float32x2`],`${n} positions`),F(n,`positions components`,t.positions.length*2,0,Ae);for(let[e,r]of[[`featureOffsets`,t.featureOffsets],[`polygonOffsets`,t.polygonOffsets],[`ringOffsets`,t.ringOffsets]])if(N(r,[`uint32`],`${n} ${e}`),F(n,`${e}.length`,r.length,1,Ae),r.length<1)throw Error(`${n} ${e} requires a terminal entry`);if(this.featureCount=t.featureOffsets.length-1,t.family===`s2`&&(F(n,`featureCount`,this.featureCount,0,xe),F(n,`candidateCapacity`,t.candidateCapacity,1,be)),this.edgeSlabs=t.family!==`s2`&&(t.edgeSlabs??t.candidateCapacity*t.positions.length>=je)&&t.ringOffsets.length>=2&&t.polygonOffsets.length>=2,t.edgeSlabEntryCapacity!==void 0&&F(n,`edgeSlabEntryCapacity`,t.edgeSlabEntryCapacity),t.featureIds&&(N(t.featureIds,[`uint32`],`${n} featureIds`),t.featureIds.length!==this.featureCount))throw Error(`${n} featureIds length must equal the feature count`);if(F(n,`candidateCapacity`,t.candidateCapacity),t.family!==`s2`&&(this.featureCount+1)*(t.candidateCapacity+1)>=4294967295)throw Error(`${n} (featureCount + 1) * (candidateCapacity + 1) must be below 2^32`);if(this.candidateEnumeration=t.candidateEnumeration??`auto`,![`bounds`,`scanline`,`auto`].includes(this.candidateEnumeration))throw Error(`${n} candidateEnumeration must be 'bounds', 'scanline' or 'auto'`);this.scanline=t.family!==`s2`&&(this.candidateEnumeration===`scanline`||this.candidateEnumeration===`auto`&&t.candidateCapacity>=65536)&&this.featureCount>=1&&t.positions.length>=1&&t.ringOffsets.length>=2&&t.polygonOffsets.length>=2,this.scanlineSlotCapacity=t.scanlineSlotCapacity??Math.min(t.candidateCapacity,V,Math.max(16*t.positions.length,16384)),t.family!==`s2`&&(t.scanlineSlotCapacity!==void 0||this.scanline)&&F(n,`scanlineSlotCapacity`,this.scanlineSlotCapacity,1,Math.min(t.candidateCapacity,V));let{output:r}=t,{status:i}=r;if(N(r.featureIds,[`uint32`],`${n} output.featureIds`),N(r.cells,[`uint32x2`],`${n} output.cells`),F(n,`output.cells components`,r.cells.length*2,1,Ae),r.cells.length!==r.featureIds.length)throw Error(`${n} output.cells and output.featureIds must have equal lengths`);if(r.featureIds.length<1)throw Error(`${n} output capacity must be at least 1`);if(r.core&&(N(r.core,[`uint32`],`${n} output.core`),r.core.length!==r.featureIds.length))throw Error(`${n} output.core must have the same length as output.featureIds`);for(let[e,t]of[[`status.count`,i.count],[`status.requiredCount`,i.requiredCount],[`status.overflow`,i.overflow],[`status.candidateOverflow`,i.candidateOverflow],[`status.candidateCount`,i.candidateCount],[`status.invalidCount`,i.invalidCount],[`status.uncertainCount`,i.uncertainCount]])if(t&&(N(t,[`uint32`],`${n} output.${e}`),t.length<1))throw Error(`${n} output.${e} must contain one uint32 row`);e(n,{featureIds:r.featureIds,cells:r.cells,core:r.core,...i},[{featureIds:r.featureIds,cells:r.cells,core:r.core},t.family===`s2`?{count:i.count,overflow:i.overflow,requiredCount:i.requiredCount,candidateOverflow:i.candidateOverflow,candidateCount:i.candidateCount}:{candidateOverflow:i.candidateOverflow,candidateCount:i.candidateCount},t.family===`s2`?{}:{invalidCount:i.invalidCount,uncertainCount:i.uncertainCount},{count:i.count,overflow:i.overflow,requiredCount:i.requiredCount}]),z(n,[r.featureIds,r.cells,r.core,i.count,i.requiredCount,i.overflow,i.candidateOverflow,i.candidateCount,i.invalidCount,i.uncertainCount],[t.positions,t.featureOffsets,t.polygonOffsets,t.ringOffsets,t.featureIds]),t.family===`s2`&&(this.estimatedCommandNodeCount=we(t),I(n,this.estimatedCommandNodeCount))}getCommandNodes(e){let{id:a,props:o,featureCount:s}=this,{output:c,candidateCapacity:l}=o;if(o.family===`s2`)return Te(e,a,o),Ee(e,a,o);let u=this.scanline?{slotCapacity:this.scanlineSlotCapacity,force:this.candidateEnumeration===`scanline`}:void 0,f=q(s,u,5),m=Oe(e.device);F(a,`candidateCapacity`,l,1,De(e.device)),this.edgeSlabs&&P(a,`positions rows + edgeSlabEntryCapacity`,o.positions.length+(o.edgeSlabEntryCapacity??1),m);let h=o.edgeSlabEntryCapacity??n(m-o.positions.length,1,Math.max(4*o.positions.length,1024)),g=[[`feature count scan scratch`,(s+1)*4],[`candidate cells scratch`,l*8],[`range scratch`,f*4],[`total scratch`,4]];u&&g.push([`scanline edge scratch`,(o.positions.length+1)*4],[`scanline demand scratch`,s*8],[`scanline event and sort scratch`,this.scanlineSlotCapacity*24]),this.edgeSlabs&&g.push([`edge slab parameters`,Math.max(s,1)*24],[`edge slab owner scan`,(s+1)*4],[`edge slab ring owners`,Math.max(o.ringOffsets.length-1,1)*4],[`edge slab vertex owners`,o.positions.length*4],[`edge slab offsets`,(Math.floor(o.positions.length/4)+2)*4],[`edge slab entries`,(o.positions.length+h)*8]),t(e,a,{positions:o.positions,featureOffsets:o.featureOffsets,polygonOffsets:o.polygonOffsets,ringOffsets:o.ringOffsets,featureIds:o.featureIds,outputFeatureIds:c.featureIds,outputCells:c.cells,outputCore:c.core,...c.status},g),R(a,e,[o.positions,o.featureOffsets,o.polygonOffsets,o.ringOffsets,o.featureIds,c.featureIds,c.cells,c.core,c.status.count,c.status.requiredCount,c.status.overflow,c.status.candidateOverflow,c.status.candidateCount,c.status.invalidCount,c.status.uncertainCount]);let _=r(e,`${a}-counts`,`uint32`,s+1),v=r(e,`${a}-starts`,`uint32`,s+1);P(a,`range table words`,f,L(e.device,4));let y=r(e,`${a}-ranges`,`uint32`,f),b=r(e,`${a}-flags`,`uint32`,l),x=r(e,`${a}-accepted`,`uint32`,l),S=r(e,`${a}-candidate-cells`,`uint32x2`,l),C=r(e,`${a}-total`,`uint32`,1),w=this.edgeSlabs?D(e,{id:a,operation:`GPUCellCover`,owner:`feature`,positions:o.positions,featureOffsets:o.featureOffsets,polygonOffsets:o.polygonOffsets,ringOffsets:o.ringOffsets,entryCapacity:h,gate:{counts:_,minimum:16}}):void 0,T={id:a,family:o.family,resolution:o.resolution,containment:this.containment,computeCore:!!c.core,featureCount:s,candidateCapacity:l,edgeSlabs:w?.slabs,scanline:u,positions:o.positions,featureOffsets:o.featureOffsets,polygonOffsets:o.polygonOffsets,ringOffsets:o.ringOffsets},E=[`invalidCount`,`uncertainCount`].flatMap(t=>{let n=c.status[t];return n?[p(e,{id:`${a}-clear-${t}`,operation:`GPUCellCover`,view:n,type:`u32`,value:`0u`})]:[]}),O={invalidCount:c.status.invalidCount,uncertainCount:c.status.uncertainCount};return[...E,ge(e,T,_,y,O),...u?X(e,{...T,scanline:u},{counts:_,ranges:y}):[],...new i({id:`${a}-candidate-scan`,input:_,output:v,mode:`exclusive`}).getCommandNodes(e),...w?.nodes??[],_e(e,T,{ranges:y,starts:v,flags:b,candidateCells:S}),...new i({id:`${a}-accept-scan`,input:b,output:x,mode:`inclusive`}).getCommandNodes(e),ve(e,T,{starts:v,accepted:x,candidateCells:S,featureIds:o.featureIds,outputFeatureIds:c.featureIds,outputCells:c.cells,outputCore:c.core}),ye(e,T,{starts:v,accepted:x,total:C,candidateOverflow:c.status.candidateOverflow,candidateCount:c.status.candidateCount}),d(e,{id:`${a}-publish`,operation:`GPUCellCover`,requiredCount:C,output:{ids:c.featureIds,status:{count:c.status.count,overflow:c.status.overflow,requiredCount:c.status.requiredCount}},overflowSources:[]})]}};export{Me as t};