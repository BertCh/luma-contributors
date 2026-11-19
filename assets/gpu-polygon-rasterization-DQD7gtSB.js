import{n as e}from"./raster-storage-preflight-CIxFc-0g.js";import{n as t,v as n}from"./spatial-context-DwZEd4EZ.js";import{r}from"./scan-total-C4QddMmi.js";import{r as i}from"./kernel-C3tfQOn4.js";import{t as a}from"./gpu-scan-B4NNTfF9.js";import{t as o}from"./gpu-sort-xRm8l5yn.js";import{o as s}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as c}from"./sorted-segment-sums-Dvmm6OdR.js";import{E as l,s as ee,x as te,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{h as d,u as ne}from"./raster-band-utils-BcTxc8_Y.js";import{i as f,n as p,t as re}from"./polygon-rasterization-parameters-B9SJE1ko.js";var m=`GPUPolygonRasterization`,h=`0xffffffffu`,g=class{id;spatialContext;props;featureCount;polygonCount;ringCount;singleSort;constructor(e){this.id=e.id??`polygon-rasterization`,this.spatialContext=n(this.id,e.spatialContext,{coordinateSpaces:[`planar`,`longitude-latitude`],metrics:[`none`,`native`],edgeModels:t}),this.props=e;let{id:r}=this,{output:i}=e,a=ne(r,e.grid),{width:o,height:s}=e.grid;if(s*(o+1)>4294967295)throw Error(`${r} height * (width + 1) must fit in uint32`);if(d(r,e.parameters,4),l(e.positions,[`float32x2`],`${r} positions`),e.positions.length<1)throw Error(`${r} positions must not be empty`);for(let[t,n]of[[`featureOffsets`,e.featureOffsets],[`polygonOffsets`,e.polygonOffsets],[`ringOffsets`,e.ringOffsets]])if(l(n,[`uint32`],`${r} ${t}`),n.length<2)throw Error(`${r} ${t} requires at least one range and a terminal entry`);if(this.featureCount=e.featureOffsets.length-1,this.polygonCount=e.polygonOffsets.length-1,this.ringCount=e.ringOffsets.length-1,this.featureCount>=4294967295)throw Error(`${r} feature count must be below the no-zone sentinel`);if(!Number.isSafeInteger(e.crossingCapacity)||e.crossingCapacity<2||e.crossingCapacity>=4294967295)throw Error(`${r} crossingCapacity must be an integer of at least 2`);for(let[e,t,n]of[[`zones`,i.zones,a],[`boundary`,i.boundary,a],[`status.overflow`,i.status.overflow,1],[`status.crossingCount`,i.status.crossingCount,1]])if(t&&(l(t,[`uint32`],`${r} output.${e}`),t.length!==n))throw Error(`${r} output.${e} must contain ${n} uint32 rows`);u(r,[i.zones,i.boundary,i.status.overflow,i.status.crossingCount],[e.parameters,e.positions,e.featureOffsets,e.polygonOffsets,e.ringOffsets]),this.singleSort=c(this.polygonCount)+c(a+s-1)<=32}getCommandNodes(t){let n=this.props;e(t,this.id,[n.parameters,n.positions,n.featureOffsets,n.polygonOffsets,n.ringOffsets,...ee(n.output)],{"crossing and sort planes":n.crossingCapacity*4,"edge planes":n.positions.length*(this.spatialContext.coordinateSpace===`longitude-latitude`?3:1)*4,"crossing total":4});let{id:l,props:u,polygonCount:d,ringCount:ne,featureCount:g,singleSort:_}=this,{parameters:v,positions:y,output:b}=u,{zones:x,boundary:S}=b,{overflow:C,crossingCount:w}=b.status,{width:T,height:E}=u.grid;te(l,t,[v,y,u.featureOffsets,u.polygonOffsets,u.ringOffsets,x,S,C,w]);let D=y.length,O=this.spatialContext.coordinateSpace===`longitude-latitude`,k=O?D*3:D,A=O?`index % ${D}u`:`index`,j=O?`let longitudeShift = f32(i32(index / ${D}u) - 1) * 360.0;`:``,M=e=>O?`loadPosition(${e}) + vec2<f32>(longitudeShift, 0.0)`:`loadPosition(${e})`,N=u.crossingCapacity,P=T*E,F=T+1,I=E*F,L=c(I-1),R=`const WIDTH: u32 = ${T}u;
const HEIGHT: u32 = ${E}u;
const ROW_STRIDE: u32 = ${F}u;
const CAPACITY: u32 = ${N}u;
const POLYGON_COUNT: u32 = ${d}u;
const SINGLE_SORT: bool = ${_};
// Only the single-sort key packs polygons above the row-column bits.
const ROW_COLUMN_BITS: u32 = ${_?L:0}u;
const ROW_COLUMN_LIMIT: u32 = ${I}u;`,z=[],B=i(t,`${l}-primary`,`uint32`,N),V=i(t,`${l}-secondary`,`uint32`,N),ie=`${_?d*2**L:I}u`,ae=_?`0u`:`${d}u`,H=[{name:`zonesOut`,view:x,type:`u32`,access:`read_write`},{name:`primary`,view:B,type:`u32`,access:`read_write`},{name:`secondary`,view:V,type:`u32`,access:`read_write`}];S&&H.push({name:`boundaryOut`,view:S,type:`u32`,access:`read_write`}),z.push(s(t,{id:`${l}-reset`,operation:m,variant:`reset`,bindings:H,invocationCount:Math.max(P,N),declarations:`const CELL_COUNT: u32 = ${P}u;
const CAPACITY: u32 = ${N}u;`,body:`if (index < CELL_COUNT) {
    zonesOut[zonesOutOffset + index] = ${re}u;
    ${S?`boundaryOut[boundaryOutOffset + index] = 0u;`:``}
  }
  if (index < CAPACITY) {
    primary[primaryOffset + index] = ${ie};
    secondary[secondaryOffset + index] = ${ae};
  }`}));let U=i(t,`${l}-edge-counts`,`uint32`,k),W=i(t,`${l}-edge-next`,`uint32`,k),G=i(t,`${l}-edge-polygons`,`uint32`,k),K=`${p}
fn getRowThreshold(y: f32, rasterExtent: RasterExtent) -> u32 {
  // Smallest row whose center line is at or above y.
  let row = ceil((y - rasterExtent.origin.y) / rasterExtent.cellSize.y - 0.5);
  return u32(clamp(row, 0.0, f32(HEIGHT)));
}
fn loadPosition(vertex: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
fn isFinitePosition(position: vec2<f32>) -> bool {
  return isFiniteValue(position.x) && isFiniteValue(position.y);
}`;z.push(s(t,{id:`${l}-edges`,operation:m,variant:O?`edges-longitude-latitude`:`edges`,bindings:[{name:`positions`,view:y,type:`f32`,access:`read`},{name:`ringOffsets`,view:u.ringOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:u.polygonOffsets,type:`u32`,access:`read`},{name:`featureOffsets`,view:u.featureOffsets,type:`u32`,access:`read`},{name:`extent`,view:v,type:`f32`,access:`read`},{name:`edgeCounts`,view:U,type:`u32`,access:`read_write`},{name:`edgeNext`,view:W,type:`u32`,access:`read_write`},{name:`edgePolygons`,view:G,type:`u32`,access:`read_write`}],invocationCount:k,declarations:`${R}
${K}
${f(`findRing`,`ringOffsets`,ne)}
${f(`findPolygon`,`polygonOffsets`,d)}
${f(`findFeature`,`featureOffsets`,g)}`,body:`var next = ${h};
  var polygon = ${h};
  var count = 0u;
  ${j}
  let ring = findRing(${A});
  if (ring != ${h}) {
    polygon = findPolygon(ring);
  }
  if (polygon != ${h} && findFeature(polygon) != ${h}) {
    let ringStart = ringOffsets[ringOffsetsOffset + ring];
    let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
    next = select(${A} + 1u, ringStart, ${A} + 1u >= ringEnd);
    let start = ${M(A)};
    let end = ${M(`next`)};
    let rasterExtent = getRasterExtent();
    if (rasterExtent.valid && isFinitePosition(start) && isFinitePosition(end)) {
      let low = getRowThreshold(min(start.y, end.y), rasterExtent);
      let high = getRowThreshold(max(start.y, end.y), rasterExtent);
      count = high - min(low, high);
      ${O?`// Select longitude strips, retaining off-grid crossings to preserve even-odd parity.
      let gridMaximumX = rasterExtent.origin.x + f32(WIDTH) * rasterExtent.cellSize.x;
      if (gridMaximumX <= longitudeShift - 180.0 || rasterExtent.origin.x >= longitudeShift + 180.0) {
        count = 0u;
        next = ${h};
      }`:``}
    } else {
      next = ${h};
    }
  } else {
    polygon = ${h};
  }
  edgeCounts[edgeCountsOffset + index] = count;
  edgeNext[edgeNextOffset + index] = next;
  edgePolygons[edgePolygonsOffset + index] = polygon;`}));let q=i(t,`${l}-edge-offsets`,`uint32`,k);z.push(...new a({id:`${l}-edge-scan`,input:U,output:q,mode:`exclusive`}).getCommandNodes(t));let J=i(t,`${l}-total`,`uint32`,1),Y=[{name:`total`,view:J,type:`u32`,access:`read`},{name:`overflowOut`,view:C,type:`u32`,access:`read_write`}];w&&Y.push({name:`crossingCountOut`,view:w,type:`u32`,access:`read_write`}),z.push(...r(t,{id:l,operation:m,counts:U,offsets:q,rowCount:k,total:J}),s(t,{id:`${l}-publish`,operation:m,variant:`publish`,bindings:Y,invocationCount:1,declarations:`const LAST_EDGE: u32 = ${D-1}u;
const CAPACITY: u32 = ${N}u;`,body:`let crossings = total[totalOffset];
  overflowOut[overflowOutOffset] = select(0u, 1u, crossings > CAPACITY);
  ${w?`crossingCountOut[crossingCountOutOffset] = crossings;`:``}`})),z.push(s(t,{id:`${l}-emit`,operation:m,variant:O?`emit-longitude-latitude`:`emit`,bindings:[{name:`positions`,view:y,type:`f32`,access:`read`},{name:`extent`,view:v,type:`f32`,access:`read`},{name:`edgeCounts`,view:U,type:`u32`,access:`read`},{name:`edgeNext`,view:W,type:`u32`,access:`read`},{name:`edgePolygons`,view:G,type:`u32`,access:`read`},{name:`edgeOffsets`,view:q,type:`u32`,access:`read`},{name:`primary`,view:B,type:`u32`,access:`read_write`},{name:`secondary`,view:V,type:`u32`,access:`read_write`}],invocationCount:k,declarations:`${R}
${K}`,body:`let count = edgeCounts[edgeCountsOffset + index];
  if (count == 0u) {
    return;
  }
  let next = edgeNext[edgeNextOffset + index];
  let polygon = edgePolygons[edgePolygonsOffset + index];
  let rasterExtent = getRasterExtent();
  ${j}
  let first = ${M(A)};
  let second = ${M(`next`)};
  // Orient bottom to top so both traversal directions compute bitwise identical crossings.
  let lower = select(second, first, first.y < second.y);
  let upper = select(first, second, first.y < second.y);
  let firstRow = getRowThreshold(lower.y, rasterExtent);
  let base = edgeOffsets[edgeOffsetsOffset + index];
  for (var step = 0u; step < count; step++) {
    let slot = base + step;
    if (slot >= CAPACITY) {
      break;
    }
    let row = firstRow + step;
    let centerY = rasterExtent.origin.y + (f32(row) + 0.5) * rasterExtent.cellSize.y;
    let t = clamp((centerY - lower.y) / (upper.y - lower.y), 0.0, 1.0);
    let crossingX = lower.x + (upper.x - lower.x) * t;
    // First column whose center is at or right of the crossing.
    let column = u32(clamp(ceil((crossingX - rasterExtent.origin.x) / rasterExtent.cellSize.x - 0.5), 0.0, f32(WIDTH)));
    let rowColumn = row * ROW_STRIDE + column;
    if (SINGLE_SORT) {
      primary[primaryOffset + slot] = (polygon << ROW_COLUMN_BITS) | rowColumn;
      secondary[secondaryOffset + slot] = rowColumn;
    } else {
      primary[primaryOffset + slot] = rowColumn;
      secondary[secondaryOffset + slot] = polygon;
    }
  }`}));let X=i(t,`${l}-sorted-keys`,`uint32`,N),Z=i(t,`${l}-sorted-values`,`uint32`,N),Q,$;if(_)z.push(...new o({id:`${l}-sort`,keys:B,values:V,outputKeys:X,outputValues:Z,keyBits:c(d)+L}).getCommandNodes(t)),Q=X,$=Z;else{let e=i(t,`${l}-polygon-keys`,`uint32`,N),n=i(t,`${l}-row-columns`,`uint32`,N);z.push(...new o({id:`${l}-sort-row-column`,keys:B,values:V,outputKeys:X,outputValues:Z,keyBits:c(I)}).getCommandNodes(t),...new o({id:`${l}-sort-polygon`,keys:Z,values:X,outputKeys:e,outputValues:n,keyBits:c(d)}).getCommandNodes(t)),Q=e,$=n}let oe=`fn getRecordPolygon(record: u32) -> u32 {
  let key = recordKeys[recordKeysOffset + record];
  return select(key, key >> ROW_COLUMN_BITS, SINGLE_SORT);
}
fn getRecordRowColumn(record: u32) -> u32 {
  return recordRowColumns[recordRowColumnsOffset + record];
}`,se=i(t,`${l}-segment-flags`,`uint32`,N),ce=i(t,`${l}-ones`,`uint32`,N),le=i(t,`${l}-ranks`,`uint32`,N);return z.push(s(t,{id:`${l}-segments`,operation:m,variant:`segments`,bindings:[{name:`recordKeys`,view:Q,type:`u32`,access:`read`},{name:`recordRowColumns`,view:$,type:`u32`,access:`read`},{name:`segmentFlags`,view:se,type:`u32`,access:`read_write`},{name:`ones`,view:ce,type:`u32`,access:`read_write`}],invocationCount:N,declarations:`${R}
${oe}`,body:`var isHead = index == 0u;
  if (!isHead) {
    isHead = getRecordPolygon(index) != getRecordPolygon(index - 1u) ||
      getRecordRowColumn(index) / ROW_STRIDE != getRecordRowColumn(index - 1u) / ROW_STRIDE;
  }
  segmentFlags[segmentFlagsOffset + index] = select(0u, 1u, isHead);
  ones[onesOffset + index] = 1u;`})),z.push(...new a({id:`${l}-rank-scan`,input:ce,output:le,mode:`exclusive`,segmentFlags:se}).getCommandNodes(t)),z.push(s(t,{id:`${l}-fill`,operation:m,variant:`fill`,bindings:[{name:`recordKeys`,view:Q,type:`u32`,access:`read`},{name:`recordRowColumns`,view:$,type:`u32`,access:`read`},{name:`ranks`,view:le,type:`u32`,access:`read`},{name:`featureOffsets`,view:u.featureOffsets,type:`u32`,access:`read`},{name:`total`,view:J,type:`u32`,access:`read`},{name:`extent`,view:v,type:`f32`,access:`read`},{name:`zonesOut`,view:x,type:`atomic<u32>`,access:`read_write`}],invocationCount:N-1,declarations:`${R}
${p}
${oe}
${f(`findFeature`,`featureOffsets`,g)}`,body:`if (total[totalOffset] > CAPACITY || (ranks[ranksOffset + index] & 1u) != 0u) {
    return;
  }
  let polygon = getRecordPolygon(index);
  if (polygon >= POLYGON_COUNT || getRecordPolygon(index + 1u) != polygon) {
    return;
  }
  let start = getRecordRowColumn(index);
  let end = getRecordRowColumn(index + 1u);
  let row = start / ROW_STRIDE;
  if (end / ROW_STRIDE != row) {
    return;
  }
  let feature = findFeature(polygon);
  let rowBase = getStoredRow(row, getRasterExtent()) * WIDTH;
  for (var column = start % ROW_STRIDE; column < end % ROW_STRIDE; column++) {
    atomicMin(&zonesOut[zonesOutOffset + rowBase + column], feature);
  }`})),S&&z.push(s(t,{id:`${l}-boundary`,operation:m,variant:O?`boundary-longitude-latitude`:`boundary`,bindings:[{name:`positions`,view:y,type:`f32`,access:`read`},{name:`extent`,view:v,type:`f32`,access:`read`},{name:`edgeNext`,view:W,type:`u32`,access:`read`},{name:`boundaryOut`,view:S,type:`atomic<u32>`,access:`read_write`}],invocationCount:k,declarations:`${R}
${p}
// Cell padding that keeps the marking conservative under float32 rounding: a fixed part plus
// about 8 ulps of the largest magnitude involved, in cell units.
const BASE_PADDING: f32 = 1e-3;
const RELATIVE_PADDING: f32 = 1e-6;
fn loadPosition(vertex: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
fn clampCell(value: f32, limit: u32) -> i32 {
  return i32(clamp(floor(value), -1.0, f32(limit)));
}`,body:`let next = edgeNext[edgeNextOffset + index];
  if (next == ${h}) {
    return;
  }
  let rasterExtent = getRasterExtent();
  ${j}
  let startPosition = ${M(A)};
  let endPosition = ${M(`next`)};
  // Edge in cell units.
  let start = (startPosition - rasterExtent.origin) / rasterExtent.cellSize;
  let end = (endPosition - rasterExtent.origin) / rasterExtent.cellSize;
  let magnitude = (max(abs(startPosition), abs(endPosition)) + abs(rasterExtent.origin)) / rasterExtent.cellSize;
  let padding = BASE_PADDING + RELATIVE_PADDING * max(magnitude.x, magnitude.y);
  let lower = select(end, start, start.y < end.y);
  let upper = select(start, end, start.y < end.y);
  let firstRow = max(clampCell(lower.y - padding, HEIGHT), 0);
  let lastRow = min(clampCell(upper.y + padding, HEIGHT), i32(HEIGHT) - 1);
  for (var row = firstRow; row <= lastRow; row++) {
    // x extent of the segment inside the padded row slab.
    var slabX = vec2<f32>(lower.x, upper.x);
    let deltaY = upper.y - lower.y;
    if (deltaY > 0.0) {
      let t0 = clamp((f32(row) - padding - lower.y) / deltaY, 0.0, 1.0);
      let t1 = clamp((f32(row + 1) + padding - lower.y) / deltaY, 0.0, 1.0);
      slabX = vec2<f32>(lower.x + (upper.x - lower.x) * t0, lower.x + (upper.x - lower.x) * t1);
    }
    let firstColumn = max(clampCell(min(slabX.x, slabX.y) - padding, WIDTH), 0);
    let lastColumn = min(clampCell(max(slabX.x, slabX.y) + padding, WIDTH), i32(WIDTH) - 1);
    for (var column = firstColumn; column <= lastColumn; column++) {
      atomicStore(&boundaryOut[boundaryOutOffset + getStoredRow(u32(row), rasterExtent) * WIDTH + u32(column)], 1u);
    }
  }`})),z}};export{g as t};