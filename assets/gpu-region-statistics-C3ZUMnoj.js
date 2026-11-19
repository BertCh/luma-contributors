import{t as e}from"./spatial-statistics-counts-wgsl-B8N7XLDs.js";import{t}from"./spatial-statistics-collective-node-BlCi_ESE.js";import{i as n}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as r}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{c as i,d as a,r as o,u as s}from"./kernel-C3tfQOn4.js";import{t as c}from"./gpu-histogram-fxACbUIT.js";import{n as l,r as u}from"./gpu-command-dispatch-metadata-Bpi5I1Ez.js";import{n as d,t as f}from"./gpu-dispatch-utils-CbRzFAgT.js";import{n as p,t as m}from"./gpu-spatial-utils-BiKW3pHz.js";import{t as h}from"./gpu-reduction-tNgOVQ_C.js";import{t as g}from"./gpu-visibility-workflow-CkjKbbbA.js";import{l as _}from"./gpu-command-graph-BdzdmV6u.js";import{c as v,n as y,o as b,t as x}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as S,c as C,f as w,m as T,x as E,y as D}from"./gpu-contributor-utils-BIwLQON5.js";import{t as O}from"./gpu-pick-region-mask-Bc8gG-q9.js";import{a as k,i as A,n as j,o as M,r as N,s as P,t as F}from"./gpu-region-mask-CAp57qz5.js";import{t as I}from"./gpu-grid-coordinate-DwQPYH7E.js";import{u as L}from"./wgsl-common-DfyXiYB0.js";import{t as R}from"./status-aOwAAXek.js";var z=256,B=class{id;positions;kind;query;outputMask;overflow;candidates;dimension;constructor(e){this.id=e.id??`gpu-point-spatial-filter`,this.positions=e.positions,this.kind=e.kind,this.query=e.query,this.outputMask=e.outputMask,this.overflow=e.overflow,this.candidates=e.candidates,this.dimension=this.positions.format===`float32x2`?2:3;for(let e of u(this.positions))a(e,[`float32x2`,`float32x3`],`${this.id} positions`);a(this.query,[`float32`],`${this.id} query`);for(let e of u(this.outputMask))s(e,`${this.id} outputMask`);if(s(this.overflow,`${this.id} overflow`),this.outputMask.length!==this.positions.length)throw Error(`${this.id} outputMask.length must equal positions.length`);if(this.overflow.length<1)throw Error(`${this.id} overflow must contain one uint32 row`);let t=this.kind===`bounds`?this.dimension*2:this.dimension+1;if(this.query.length!==t)throw Error(`${this.id} ${this.kind} query must contain ${t} floats`);if(this.candidates){for(let e of u(this.candidates.ids))s(e,`${this.id} candidate IDs`);if(s(this.candidates.count,`${this.id} candidate count`),this.candidates.count.length<1)throw Error(`${this.id} candidate count must contain one uint32 row`);if(this.candidates.overflow&&(s(this.candidates.overflow,`${this.id} candidate overflow`),this.candidates.overflow.length<1))throw Error(`${this.id} candidate overflow must contain one uint32 row`)}if(this.positions.length>4294967295||(this.candidates?.ids.length??0)>4294967295)throw Error(`${this.id} logical row counts must fit in uint32`);p(this.id,[...u(this.positions),this.query,...this.candidates?[...u(this.candidates.ids),this.candidates.count,...this.candidates.overflow?[this.candidates.overflow]:[]]:[]],[...u(this.outputMask),this.overflow])}getCommandNodes(e){let t=[];if([this.positions,this.query,this.outputMask,this.overflow,...this.candidates?[this.candidates.ids,this.candidates.count,...this.candidates.overflow?[this.candidates.overflow]:[]]:[]].flatMap(e=>u(e)).some(t=>t.buffer.graph!==e))throw Error(`${this.id} views must belong to the target graph`);t.push(...V(e,this));let n=0;for(let[r,[i,a]]of l(e,[this.positions,this.outputMask]).entries()){if(this.candidates){let o=0;for(let[s,c]of u(this.candidates.ids).entries())c.length&&t.push(...H(e,{...this,id:`${this.id}-positions-${r}-candidates-${s}`,positions:i,outputMask:a,positionStart:n,candidateStart:o,candidateCapacity:this.candidates.ids.length,candidates:{...this.candidates,ids:c}})),o+=c.length}else t.push(...H(e,{...this,id:`${this.id}-positions-${r}`,positions:i,outputMask:a,positionStart:n,candidateStart:0,candidateCapacity:0,candidates:void 0}));n+=i.length}return t}};function V(e,t){let n=[];for(let[r,a]of u(t.outputMask).entries()){if(!a.length)continue;let o=f(t.id,a.length,z,e.device.limits.maxComputeWorkgroupsPerDimension);n.push(...m(e,{id:`${t.id}-clear-mask-${r}`,source:`
@group(0) @binding(0) var<storage, read_write> outputMask: array<u32>;
@compute @workgroup_size(${z}) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${d(o,z)}
  if (index < ${a.length}u) {
    outputMask[${i(a)}u + index] = 0u;
  }
}`,resources:[{buffer:a,usage:`storage-write`}],bindings:{outputMask:a},dispatch:o}))}let r=t.candidates,a=`
@group(0) @binding(0) var<storage, read_write> outputOverflow: array<u32>;
${r?`@group(0) @binding(1) var<storage, read> candidateCount: array<u32>;`:``}
${r?.overflow?`@group(0) @binding(2) var<storage, read> sourceOverflow: array<u32>;`:``}
@compute @workgroup_size(1) fn main() {
  outputOverflow[${i(t.overflow)}u] = ${r?`select(0u, 1u, candidateCount[${i(r.count)}u] > ${r.ids.length}u${r.overflow?` || sourceOverflow[${i(r.overflow)}u] != 0u`:``})`:`0u`};
}`;return n.push(...m(e,{id:`${t.id}-initialize-overflow`,source:a,resources:[{buffer:t.overflow,usage:`storage-write`},...r?[{buffer:r.count,usage:`storage-read`}]:[],...r?.overflow?[{buffer:r.overflow,usage:`storage-read`}]:[]],bindings:{outputOverflow:t.overflow,...r?{candidateCount:r.count}:{},...r?.overflow?{sourceOverflow:r.overflow}:{}},dispatch:{x:1,y:1,z:1}})),n}function H(e,t){let n=[],r=t.candidates?.ids.length??t.positions.length,a=f(t.id,r,z,e.device.limits.maxComputeWorkgroupsPerDimension),o=t.candidates?`@group(0) @binding(3) var<storage, read> candidateIds: array<u32>;
@group(0) @binding(4) var<storage, read> candidateCount: array<u32>;`:``,s=t.candidates?`const CANDIDATE_IDS_OFFSET: u32 = ${i(t.candidates.ids)}u;
const CANDIDATE_COUNT_OFFSET: u32 = ${i(t.candidates.count)}u;
const CANDIDATE_CAPACITY: u32 = ${t.candidateCapacity}u;
const CANDIDATE_START: u32 = ${t.candidateStart}u;
const CANDIDATE_CHUNK_LENGTH: u32 = ${t.candidates.ids.length}u;`:``,c=t.candidates?`let storedCandidateCount = min(candidateCount[CANDIDATE_COUNT_OFFSET], CANDIDATE_CAPACITY);
  if (index >= CANDIDATE_CHUNK_LENGTH || CANDIDATE_START + index >= storedCandidateCount) { return; }
  let globalRow = candidateIds[CANDIDATE_IDS_OFFSET + index];
  if (globalRow < ${t.positionStart}u || globalRow - ${t.positionStart}u >= POSITION_COUNT) { return; }
  let sourceRow = globalRow - ${t.positionStart}u;`:`if (index >= POSITION_COUNT) { return; }
  let sourceRow = index;`,l=U(t),u=`
const POSITION_COUNT: u32 = ${t.positions.length}u;
const POSITIONS_OFFSET: u32 = ${i(t.positions)}u;
const QUERY_OFFSET: u32 = ${i(t.query)}u;
const MASK_OFFSET: u32 = ${i(t.outputMask)}u;
${s}
@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> queryValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputMask: array<atomic<u32>>;
${o}

fn finite(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823466e+38;
}

@compute @workgroup_size(${z}) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${d(a,z)}
  ${c}
  if (sourceRow >= POSITION_COUNT) { return; }
  ${l}
  if (selected) { atomicStore(&outputMask[MASK_OFFSET + sourceRow], 1u); }
}`,p=[{buffer:t.positions,usage:`storage-read`},{buffer:t.query,usage:`storage-read`},{buffer:t.outputMask,usage:`storage-read-write`},...t.candidates?[{buffer:t.candidates.ids,usage:`storage-read`},{buffer:t.candidates.count,usage:`storage-read`}]:[]];return n.push(...m(e,{id:t.id,source:u,resources:p,bindings:{positions:t.positions,queryValues:t.query,outputMask:t.outputMask,...t.candidates?{candidateIds:t.candidates.ids,candidateCount:t.candidates.count}:{}},dispatch:a})),n}function U(e){let t=[`X`,`Y`,...e.dimension===3?[`Z`]:[]],n=t.map((t,n)=>`let position${t} = positions[POSITIONS_OFFSET + sourceRow * ${e.dimension}u + ${n}u];`).join(`
  `),r=t.map(e=>`finite(position${e})`).join(` && `);if(e.kind===`bounds`)return`${n}
  ${t.map((t,n)=>`let queryMin${t} = queryValues[QUERY_OFFSET + ${n}u];
  let queryMax${t} = queryValues[QUERY_OFFSET + ${n+e.dimension}u];`).join(`
  `)}
  let selected = ${r} && ${t.map(e=>`finite(queryMin${e}) && finite(queryMax${e}) && queryMin${e} <= queryMax${e}`).join(` && `)} && ${t.map(e=>`position${e} >= queryMin${e} && position${e} <= queryMax${e}`).join(` && `)};`;let i=t.map((e,t)=>`let query${e} = queryValues[QUERY_OFFSET + ${t}u];`).join(`
  `),a=t.map(e=>`finite(query${e})`).join(` && `),o=W([`radius`,...t.flatMap(e=>[`abs(position${e})`,`abs(query${e})`])]),s=t.map(e=>`(position${e} / scale - query${e} / scale) * (position${e} / scale - query${e} / scale)`).join(` + `);return`${n}
  ${i}
  let radius = queryValues[QUERY_OFFSET + ${e.dimension}u];
  let scale = ${o};
  let selected = ${r} && ${a} && finite(radius) && radius >= 0.0 && (scale == 0.0 || ${s} <= (radius / scale) * (radius / scale));`}function W(e){return e.slice(1).reduce((e,t)=>`max(${e}, ${t})`,e[0])}var G=`GPURegionStatistics`;function K(e,t){let{id:n,region:r,spatialIndex:i,rowCount:a,regionOverflow:s}=t,c=t.positions,l=Math.min(i.candidateCapacity,a),u=[],d,f;r.kind===`rectangle`?(d=r.bounds,f=`bounds`):r.kind===`radius`?(d=r.circle,f=`radius`):(d=o(e,`${n}-polygon-bounds`,`float32`,4),f=`bounds`,u.push(b(e,{id:`${n}-polygon-bounds-kernel`,operation:G,variant:`polygon-bounds`,bindings:[{name:`vertices`,view:r.vertices,type:`f32`,access:`read`},{name:`vertexCount`,view:r.vertexCount,type:`u32`,access:`read`},{name:`bounds`,view:d,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`const VERTEX_CAPACITY: u32 = ${r.vertices.length}u;
${L}`,body:`let activeCount = min(vertexCount[vertexCountOffset], VERTEX_CAPACITY);
  var minimum = vec2f(3.402823466e+38);
  var maximum = vec2f(-3.402823466e+38);
  var allFinite = true;
  for (var vertexIndex = 0u; vertexIndex < activeCount; vertexIndex++) {
    let vertex = vec2f(vertices[verticesOffset + vertexIndex * 2u], vertices[verticesOffset + vertexIndex * 2u + 1u]);
    if (isFiniteValue(vertex.x) && isFiniteValue(vertex.y)) {
      minimum = min(minimum, vertex);
      maximum = max(maximum, vertex);
    } else {
      allFinite = false;
    }
  }
  if (!allFinite) {
    minimum = vec2f(-3.402823466e+38);
    maximum = vec2f(3.402823466e+38);
  }
  // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
  let quietNaN = bitcast<f32>(0x7fc00000u | (index & 0u));
  let valid = activeCount >= 3u;
  bounds[boundsOffset] = select(quietNaN, minimum.x, valid);
  bounds[boundsOffset + 1u] = select(quietNaN, minimum.y, valid);
  bounds[boundsOffset + 2u] = select(quietNaN, maximum.x, valid);
  bounds[boundsOffset + 3u] = select(quietNaN, maximum.y, valid);`})));let{index:p}=i,[m,h]=p.gridSize,[g,_,y,x]=p.bounds,S=o(e,`${n}-candidate-rows`,`uint32`,l),C=o(e,`${n}-candidate-count`,`uint32`,1),w=o(e,`${n}-candidate-overflow`,`uint32`,1),T=o(e,`${n}-range-start`,`uint32`,h),E=o(e,`${n}-range-prefix`,`uint32`,h+1),D=f===`radius`;u.push(b(e,{id:`${n}-cell-ranges`,operation:G,variant:`grid-cell-ranges`,bindings:[{name:`cellOffsets`,view:p.cellOffsets,type:`u32`,access:`read`},{name:`indexOverflow`,view:p.overflow,type:`u32`,access:`read`},{name:`query`,view:d,type:`f32`,access:`read`},{name:`rangeStart`,view:T,type:`u32`,access:`read_write`},{name:`rangePrefix`,view:E,type:`u32`,access:`read_write`},{name:`candidateCount`,view:C,type:`u32`,access:`read_write`},{name:`candidateOverflow`,view:w,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const WIDTH: u32 = ${m}u;
const HEIGHT: u32 = ${h}u;
const CAPACITY: u32 = ${l}u;
const DOMAIN_MIN: vec2f = vec2f(${v(g)}, ${v(_)});
const DOMAIN_MAX: vec2f = vec2f(${v(y)}, ${v(x)});
${L}
// Same cell mapping as GPUGridIndex.
${I}`,body:`${D?`let center = vec2f(query[queryOffset], query[queryOffset + 1u]);
  let radius = query[queryOffset + 2u];
  let valid = isFiniteValue(center.x) && isFiniteValue(center.y) && isFiniteValue(radius) && radius >= 0.0;
  let queryMin = center - vec2f(radius);
  let queryMax = center + vec2f(radius);`:`let queryMin = vec2f(query[queryOffset], query[queryOffset + 1u]);
  let queryMax = vec2f(query[queryOffset + 2u], query[queryOffset + 3u]);
  let valid = isFiniteValue(queryMin.x) && isFiniteValue(queryMin.y) &&
    isFiniteValue(queryMax.x) && isFiniteValue(queryMax.y) && all(queryMin <= queryMax);`}
  let overlaps = valid && all(queryMax >= DOMAIN_MIN) && all(queryMin <= DOMAIN_MAX);
  let clampedMin = max(queryMin, DOMAIN_MIN);
  let clampedMax = min(queryMax, DOMAIN_MAX);
  var columnLow = 0u;
  var columnHigh = 0u;
  var rowLow = 0u;
  var rowHigh = 0u;
  if (overlaps) {
    columnLow = getCoordinate(clampedMin.x, DOMAIN_MIN.x, DOMAIN_MAX.x, WIDTH);
    columnHigh = getCoordinate(clampedMax.x, DOMAIN_MIN.x, DOMAIN_MAX.x, WIDTH);
    rowLow = getCoordinate(clampedMin.y, DOMAIN_MIN.y, DOMAIN_MAX.y, HEIGHT);
    rowHigh = getCoordinate(clampedMax.y, DOMAIN_MIN.y, DOMAIN_MAX.y, HEIGHT);
    columnLow = columnLow - min(columnLow, 1u);
    rowLow = rowLow - min(rowLow, 1u);
    columnHigh = min(columnHigh + 1u, WIDTH - 1u);
    rowHigh = min(rowHigh + 1u, HEIGHT - 1u);
  }
  var total = 0u;
  for (var row = 0u; row < HEIGHT; row++) {
    var start = 0u;
    var length = 0u;
    if (overlaps && row >= rowLow && row <= rowHigh) {
      start = cellOffsets[cellOffsetsOffset + row * WIDTH + columnLow];
      let end = cellOffsets[cellOffsetsOffset + row * WIDTH + columnHigh + 1u];
      length = select(0u, end - start, end > start);
    }
    rangeStart[rangeStartOffset + row] = start;
    rangePrefix[rangePrefixOffset + row] = total;
    total += length;
  }
  rangePrefix[rangePrefixOffset + HEIGHT] = total;
  candidateCount[candidateCountOffset] = total;
  candidateOverflow[candidateOverflowOffset] = select(0u, 1u, total > CAPACITY || indexOverflow[indexOverflowOffset] != 0u);`})),u.push(b(e,{id:`${n}-cell-gather`,operation:G,variant:`grid-cell-gather`,bindings:[{name:`objectIds`,view:p.objectIds,type:`u32`,access:`read`},{name:`rangeStart`,view:T,type:`u32`,access:`read`},{name:`rangePrefix`,view:E,type:`u32`,access:`read`},{name:`candidateCount`,view:C,type:`u32`,access:`read`},{name:`candidateRows`,view:S,type:`u32`,access:`read_write`}],invocationCount:l,declarations:`const HEIGHT: u32 = ${h}u;
const CAPACITY: u32 = ${l}u;
const OBJECT_COUNT: u32 = ${p.objectIds.length}u;`,body:`let storedCount = min(candidateCount[candidateCountOffset], CAPACITY);
  var row = 0xffffffffu;
  if (index < storedCount) {
    var low = 0u;
    var high = HEIGHT + 1u;
    loop {
      if (low >= high) { break; }
      let middle = low + (high - low) / 2u;
      if (rangePrefix[rangePrefixOffset + middle] <= index) { low = middle + 1u; } else { high = middle; }
    }
    let gridRow = low - 1u;
    let objectIndex = rangeStart[rangeStartOffset + gridRow] + index - rangePrefix[rangePrefixOffset + gridRow];
    if (objectIndex < OBJECT_COUNT) { row = objectIds[objectIdsOffset + objectIndex]; }
  }
  candidateRows[candidateRowsOffset + index] = row;`}));let O=o(e,`${n}-candidate-positions`,`float32x2`,l),k=t.values?o(e,`${n}-candidate-values`,`float32`,l):void 0,A=[{name:`positions`,view:c,type:`f32`,access:`read`},...t.values?[{name:`values`,view:t.values,type:`f32`,access:`read`}]:[],{name:`candidateRows`,view:S,type:`u32`,access:`read`},{name:`candidateCount`,view:C,type:`u32`,access:`read`},{name:`candidatePositions`,view:O,type:`f32`,access:`read_write`},...k?[{name:`candidateValues`,view:k,type:`f32`,access:`read_write`}]:[]];u.push(b(e,{id:`${n}-gather`,operation:G,variant:`grid-gather`,bindings:A,invocationCount:l,declarations:`const ROW_COUNT: u32 = ${a}u;
const CAPACITY: u32 = ${l}u;`,body:`let storedCount = min(candidateCount[candidateCountOffset], CAPACITY);
  // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
  let quietNaN = bitcast<f32>(0x7fc00000u | (index & 0u));
  var position = vec2f(quietNaN);
  var value = quietNaN;
  if (index < storedCount) {
    let row = candidateRows[candidateRowsOffset + index];
    if (row < ROW_COUNT) {
      position = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
      ${t.values?`value = values[valuesOffset + row];`:``}
    }
  }
  candidatePositions[candidatePositionsOffset + index * 2u] = position.x;
  candidatePositions[candidatePositionsOffset + index * 2u + 1u] = position.y;
  ${k?`candidateValues[candidateValuesOffset + index] = value;`:``}`}));let j=o(e,`${n}-candidate-mask`,`uint32`,l);r.kind===`polygon`?u.push(...new F({id:`${n}-region`,positions:O,region:r,output:{mask:j,status:{overflow:s}}}).getCommandNodes(e)):u.push(...new B({id:`${n}-filter`,positions:O,kind:f,query:d,outputMask:j,overflow:s}).getCommandNodes(e));let M=o(e,`${n}-candidate-flag`,`uint32`,1);return{nodes:u,capacity:l,rows:S,count:C,positions:O,values:k,mask:j,candidateFlag:M,getStatusNode:t=>b(e,{id:`${n}-candidate-status`,operation:G,variant:`grid-status`,bindings:[{name:`candidateCount`,view:C,type:`u32`,access:`read`},{name:`candidateOverflow`,view:w,type:`u32`,access:`read`},{name:`indexCount`,view:i.index.count,type:`u32`,access:`read`},{name:`counts`,view:t,type:`u32`,access:`read_write`},{name:`candidateFlag`,view:M,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const ROW_COUNT: u32 = ${a}u;
const CAPACITY: u32 = ${l}u;`,body:`let truncated = candidateOverflow[candidateOverflowOffset] != 0u ||
    candidateCount[candidateCountOffset] > CAPACITY ||
    indexCount[indexCountOffset] < ROW_COUNT;
  counts[countsOffset + 2u] = select(0u, 1u, truncated);
  candidateFlag[candidateFlagOffset] = select(0u, 1u, truncated);`})}}function q(e,t){let{candidates:n}=t;return b(e,{id:t.id,operation:G,variant:`grid-scatter`,bindings:[{name:`candidateRows`,view:n.rows,type:`u32`,access:`read`},{name:`candidateCount`,view:n.count,type:`u32`,access:`read`},{name:`candidateMask`,view:n.mask,type:`u32`,access:`read`},{name:`mask`,view:t.mask,type:`u32`,access:`read_write`}],invocationCount:n.capacity,declarations:`const ROW_COUNT: u32 = ${t.rowCount}u;
const CAPACITY: u32 = ${n.capacity}u;`,body:`let storedCount = min(candidateCount[candidateCountOffset], CAPACITY);
  if (index < storedCount && candidateMask[candidateMaskOffset + index] != 0u) {
    let row = candidateRows[candidateRowsOffset + index];
    if (row < ROW_COUNT) { mask[maskOffset + row] = 1u; }
  }`})}function J(e,t){let{region:n,spatialIndex:r}=t;if(n.kind!==`rectangle`&&n.kind!==`radius`&&n.kind!==`polygon`)throw Error(`${e} spatialIndex supports rectangle, radius and polygon regions`);if(n.screenTransform)throw Error(`${e} spatialIndex rejects screenTransform`);for(let[n,r]of[[`positions`,t.positions],[`values`,t.values],[`sourceIds`,t.sourceIds]])if(r instanceof _)throw Error(`${e} spatialIndex requires packed (non-vector) ${n}`);if(r.kind!==`grid`)throw Error(`${e} spatialIndex.kind must be 'grid'`);let{index:i,candidateCapacity:a}=r;if(i.cellOffsets instanceof _||i.objectIds instanceof _)throw Error(`${e} spatialIndex.index requires packed storage`);if(i.gridSize.length!==2||i.bounds.length!==4)throw Error(`${e} spatialIndex.index must be two-dimensional`);w(e,`spatialIndex.candidateCapacity`,a)}var Y=256,X=`GPURegionStatistics`,Z=65536,ee=class{id;props;rowCount;spatialContext;constructor(e){this.id=e.id??`region-statistics`,this.props=e;let{id:t}=this;this.spatialContext=M(t,e.spatialContext,e.region.kind),P(this.id,e.regionSpatialContext,this.spatialContext,e.region);let{region:r,positions:i,values:a,sourceIds:o,histogram:c}=e,{mask:l,selectedRows:u,summary:d,drawInstanceCount:f}=e.output,p=r.kind===`mask`?r.mask.length:i?.length??a?.length??o?.length??l?.length;if(p===void 0)throw Error(`${t} cannot infer the source row count`);if(p===0)throw Error(`${t} requires at least one source row`);if(this.rowCount=p,(r.kind===`rectangle`||r.kind===`polygon`||r.kind===`radius`)&&!i)throw Error(`${t} positions are required for ${r.kind} regions`);for(let[e,n]of[[`positions`,i],[`values`,a],[`sourceIds`,o],[`output.mask`,l],[`region.mask`,r.kind===`mask`?r.mask:void 0]])if(n&&n.length!==p)throw Error(`${t} ${e} length must equal the source row count`);for(let e of a?C(a):[])S(e,[`float32`],`${t} values`);if(c){if(!a)throw Error(`${t} histogram requires values`);w(t,`histogram.binCount`,c.binCount,1,Z);let{domain:e}=c;if(Array.isArray(e)){let[n,r]=e;if(!Number.isFinite(Math.fround(n))||!Number.isFinite(Math.fround(r))||n>r)throw Error(`${t} histogram.domain must have finite float32 endpoints with min <= max`)}else if(e&&e!==`selection`&&(S(e,[`float32`],`${t} histogram.domain`),e.length!==2))throw Error(`${t} histogram.domain view must hold two float32 values`)}if(r.kind===`radius`&&(S(r.circle,[`float32`],`${t} region.circle`),r.circle.length!==3))throw Error(`${t} region.circle must contain three float32 values`);if(r.kind===`mask`&&l)throw Error(`${t} output.mask cannot be combined with a mask region`);l&&s(l,`${t} output.mask`),s(d,`${t} output.summary`);let m=c?.binCount??0;if(d.length!==8+m)throw Error(`${t} output.summary must hold 8 + binCount uint32 words`);if(u&&(T(t,u),u.ids.length<1))throw Error(`${t} output.selectedRows.ids must hold at least one row`);if(f){if(s(f,`${t} output.drawInstanceCount`),f.length<1)throw Error(`${t} output.drawInstanceCount must contain one uint32 row`);if(!u)throw Error(`${t} output.drawInstanceCount requires output.selectedRows`)}e.output.status&&R(`${t} output.status`,e.output.status),e.spatialIndex&&J(t,{...e,spatialIndex:e.spatialIndex}),n(t,{summary:d,mask:l,ids:u?.ids,selectedCount:u?.status.count,selectedOverflow:u?.status.overflow,selectedCandidateOverflow:u?.status.candidateOverflow,selectedRequiredCount:u?.status.requiredCount,drawInstanceCount:f,...e.output.status}),D(t,[d,l,u?.ids,u?.status.count,u?.status.overflow,u?.status.candidateOverflow,u?.status.requiredCount,f,e.output.status?.antimeridianUncertainCount,e.output.status?.poleUncertainCount,e.output.status?.overflow,e.output.status?.candidateOverflow],$(e))}getCommandNodes(e){r(e,this.id,this.props,{"row scratch":this.rowCount*4,"candidate positions":this.props.spatialIndex?Math.min(this.props.spatialIndex.candidateCapacity,this.rowCount)*8:0});let{id:t,props:n,rowCount:i}=this,{region:a,positions:s,values:l,sourceIds:u,histogram:d}=n,{mask:f,selectedRows:p,summary:m,drawInstanceCount:_}=n.output;E(t,e,[m,f,p?.ids,p?.status.count,p?.status.overflow,p?.status.candidateOverflow,p?.status.requiredCount,_,n.output.status?.antimeridianUncertainCount,n.output.status?.poleUncertainCount,n.output.status?.overflow,n.output.status?.candidateOverflow,...$(n)]);let v=N(e,t,X,this.spatialContext,a,n.output.status),b=o(e,`${t}-region-overflow`,`uint32`,1),{spatialIndex:S}=n,C=!S||!!(f||p),w=a.kind===`mask`?a.mask:f??(C?o(e,`${t}-mask`,`uint32`,i):void 0),T;if(S)T=K(e,{id:t,region:a,positions:s,values:l,spatialIndex:S,rowCount:i,regionOverflow:b}),v.push(...T.nodes),w&&v.push(x(e,{id:`${t}-mask-clear`,operation:X,view:w,type:`u32`,value:`0u`}),q(e,{id:`${t}-mask-scatter`,candidates:T,mask:w,rowCount:i}));else switch(a.kind){case`rectangle`:case`radius`:if(a.kind===`radius`||!a.screenTransform){v.push(...new B({id:`${t}-filter`,positions:s,kind:a.kind===`radius`?`radius`:`bounds`,query:a.kind===`radius`?a.circle:a.bounds,outputMask:w,overflow:b}).getCommandNodes(e));break}v.push(...new F({id:`${t}-region`,spatialContext:this.spatialContext,regionSpatialContext:n.regionSpatialContext,positions:s,region:a,output:{mask:w,status:{overflow:b}}}).getCommandNodes(e));break;case`polygon`:v.push(...new F({id:`${t}-region`,spatialContext:this.spatialContext,regionSpatialContext:n.regionSpatialContext,positions:s,region:a,output:{mask:w,status:{overflow:b}}}).getCommandNodes(e));break;case`pick-region`:v.push(...new O({id:`${t}-pick`,picks:a.picks,batchIndex:a.batchIndex,output:{mask:w,status:{overflow:b}}}).getCommandNodes(e));break;case`mask`:v.push(x(e,{id:`${t}-region-overflow-clear`,operation:X,view:b,type:`u32`,value:`0u`}))}let D=T?T.capacity:i,k=T?T.mask:w,A=T?T.values:l,j=o(e,`${t}-counts`,`uint32`,T?3:2),M=A?o(e,`${t}-value-mask`,`uint32`,D):void 0;v.push(x(e,{id:`${t}-counts-clear`,operation:X,view:j,type:`u32`,value:`0u`})),T&&v.push(T.getStatusNode(j)),v.push(...te(e,t,k,j,D,A,M));let P,I;A&&M&&(I=o(e,`${t}-sum-value`,`float32`,1),P=o(e,`${t}-extent-value`,`float32`,2),v.push(...new h({id:`${t}-sum`,input:A,mask:M,output:I,operation:`sum`}).getCommandNodes(e)),v.push(...new h({id:`${t}-extent`,input:A,mask:M,output:P,operation:`extent`}).getCommandNodes(e)));let L,R;if(d&&A&&M&&P){L=o(e,`${t}-bins`,`uint32`,d.binCount),R=o(e,`${t}-bins-total-value`,`uint32`,1);let n=d.domain===void 0||d.domain===`selection`?P:d.domain;v.push(...new c({id:`${t}-histogram`,input:A,mask:M,output:L,domain:n}).getCommandNodes(e)),v.push(...new h({id:`${t}-bins-total`,input:L,output:R,operation:`sum`}).getCommandNodes(e))}let z;if(p){let n=o(e,`${t}-selection-ids`,`uint32`,i);z=o(e,`${t}-selection-total`,`uint32`,1),v.push(...new g({id:`${t}-visibility`,predicates:[{kind:`selection`,mask:w}],output:n,count:z,sourceIds:u}).getCommandNodes(e)),v.push(y(e,{id:`${t}-selection-finalize`,operation:X,requiredCount:z,compactIds:n,output:{ids:p.ids,status:{count:p.status.count,overflow:p.status.overflow,requiredCount:p.status.requiredCount}},extraCounts:_?[_]:[]}))}v.push(ne(e,{id:`${t}-summary`,counts:j,hasCandidateFlag:!!T,regionOverflow:b,sumValue:I,extentValue:P,bins:L,binsTotal:R,selectionTotal:z,selectionCapacity:p?.ids.length??0,summary:m}));let{overflow:V,candidateOverflow:H}=n.output.status??{};return(V||H)&&v.push(Q(e,{id:`${t}-status`,summary:m,overflow:V,candidateOverflow:H})),p?.status.candidateOverflow&&v.push(Q(e,{id:`${t}-selection-candidate-status`,summary:m,includeRegionLoss:!0,candidateOverflow:p.status.candidateOverflow})),v}};function Q(e,t){let{overflow:n,candidateOverflow:r}=t,i=[{name:`summary`,view:t.summary,type:`u32`,access:`read`}];n&&i.push({name:`overflow`,view:n,type:`u32`,access:`read_write`}),r&&i.push({name:`candidateOverflow`,view:r,type:`u32`,access:`read_write`});let{flags:a}=k,{selectionTruncated:o,regionTruncated:s,candidatesTruncated:c}=A;return b(e,{id:t.id,operation:X,variant:`status-${n?`overflow`:`none`}-${r?`candidate`:`none`}`,bindings:i,invocationCount:1,body:`let flags = summary[summaryOffset + ${a}u];
  ${n?`overflow[overflowOffset] = select(0u, 1u, (flags & ${o|s}u) != 0u);`:``}
  ${r?`candidateOverflow[candidateOverflowOffset] = select(0u, 1u, (flags & ${c|(t.includeRegionLoss?s:0)}u) != 0u);`:``}`})}function $(e){let{region:t,histogram:n}=e,r=[];switch(t.kind){case`rectangle`:r.push(t.bounds,t.screenTransform);break;case`polygon`:r.push(t.vertices,t.vertexCount,t.screenTransform);break;case`radius`:r.push(t.circle);break;case`pick-region`:r.push(t.picks);break;case`mask`:r.push(t.mask)}let i=n?.domain&&typeof n.domain==`object`&&!Array.isArray(n.domain)?n.domain:void 0,{index:a}=e.spatialIndex??{};return[e.positions,e.values,e.sourceIds,i,...r,...a?[a.cellOffsets,a.objectIds,a.count,a.overflow]:[]]}function te(e,n,r,i,a,o,s){let c=o?C(o):[void 0],l=o instanceof _,u=[],d=0;for(let[o,f]of c.entries()){let c=f?f.length:a;if(c>0){let a=[{name:`mask`,view:r,type:`u32`,access:`read`}];f&&s&&(a.push({name:`values`,view:f,type:`f32`,access:`read`}),a.push({name:`valueMask`,view:s,type:`u32`,access:`read_write`})),a.push({name:`counts`,view:i,type:`atomic<u32>`,access:`read_write`}),u.push(t(e,{id:l?`${n}-count-chunk-${o}`:`${n}-count`,operation:X,variant:`count`,bindings:a,invocationCount:c,workgroupSize:Y,parameterValues:[d],declarations:`
var<workgroup> localSelectedCount: atomic<u32>;
var<workgroup> localValueCount: atomic<u32>;
${j}`,body:`if (localInvocationIndex == 0u) {
    atomicStore(&localSelectedCount, 0u);
    atomicStore(&localValueCount, 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let row = getChunkParameter(36u) + index;
    let selected = mask[maskOffset + row] != 0u;
    ${f?`let value = values[valuesOffset + index];
    let valued = selected && isFiniteValue(value);
    valueMask[valueMaskOffset + row] = select(0u, 1u, valued);
    if (valued) { atomicAdd(&localValueCount, 1u); }`:``}
    if (selected) { atomicAdd(&localSelectedCount, 1u); }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let selectedTotal = atomicLoad(&localSelectedCount);
    let valueTotal = atomicLoad(&localValueCount);
    if (selectedTotal > 0u) { atomicAdd(&counts[countsOffset], selectedTotal); }
    if (valueTotal > 0u) { atomicAdd(&counts[countsOffset + 1u], valueTotal); }
  }`}))}d+=c}return u}function ne(t,n){let{sumValue:r,extentValue:i,bins:a,binsTotal:o,selectionTotal:s}=n,c=[{name:`counts`,view:n.counts,type:`u32`,access:`read`},{name:`regionOverflow`,view:n.regionOverflow,type:`u32`,access:`read`}];r&&c.push({name:`sumValue`,view:r,type:`f32`,access:`read`}),i&&c.push({name:`extentValue`,view:i,type:`f32`,access:`read`}),a&&c.push({name:`bins`,view:a,type:`u32`,access:`read`}),o&&c.push({name:`binsTotal`,view:o,type:`u32`,access:`read`}),s&&c.push({name:`selectionTotal`,view:s,type:`u32`,access:`read`}),c.push({name:`summary`,view:n.summary,type:`u32`,access:`read_write`});let l=a?.length??0;return b(t,{id:n.id,operation:X,variant:`summary`,bindings:c,invocationCount:Math.max(l,1),declarations:`${e}
const BIN_COUNT: u32 = ${l}u;
const SELECTION_CAPACITY: u32 = ${n.selectionCapacity}u;`,body:`${a?`if (index < BIN_COUNT) { summary[summaryOffset + 8u + index] = bins[binsOffset + index]; }`:``}
  if (index == 0u) {
    let selectedCount = counts[countsOffset];
    let valueCount = counts[countsOffset + 1u];
    let hasValues = valueCount > 0u;
    ${r&&i?`let sum = select(0.0, sumValue[sumValueOffset], hasValues);
    let mean = select(0.0, divideBySpatialCount(sum, max(valueCount, 1u)), hasValues);
    let minimum = select(0.0, extentValue[extentValueOffset], hasValues);
    let maximum = select(0.0, extentValue[extentValueOffset + 1u], hasValues);`:`let sum = 0.0;
    let mean = 0.0;
    let minimum = 0.0;
    let maximum = 0.0;`}
    ${o?`let binnedCount = binsTotal[binsTotalOffset];
    let outsideCount = select(0u, valueCount - binnedCount, valueCount > binnedCount);`:`let outsideCount = 0u;`}
    let selectionTruncated = ${s?`selectionTotal[selectionTotalOffset] > SELECTION_CAPACITY`:`false`};
    let regionTruncated = regionOverflow[regionOverflowOffset] != 0u;
    let candidatesTruncated = ${n.hasCandidateFlag?`counts[countsOffset + 2u] != 0u`:`false`};
    summary[summaryOffset + 0u] = selectedCount;
    summary[summaryOffset + 1u] = valueCount;
    summary[summaryOffset + 2u] = bitcast<u32>(sum);
    summary[summaryOffset + 3u] = bitcast<u32>(mean);
    summary[summaryOffset + 4u] = bitcast<u32>(minimum);
    summary[summaryOffset + 5u] = bitcast<u32>(maximum);
    summary[summaryOffset + 6u] = outsideCount;
    summary[summaryOffset + 7u] = select(0u, 1u, selectionTruncated) | select(0u, 2u, regionTruncated) | select(0u, 4u, candidatesTruncated);
  }`})}export{ee as t};