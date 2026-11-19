import{t as e}from"./construction-snapshot-DdkX9Pc0.js";import{r as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-visibility-workflow-CkjKbbbA.js";import{l as r,o as i}from"./gpu-command-graph-BdzdmV6u.js";import{i as a,n as o,o as s}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as c,S as l,c as u,f as d,m as f,u as p,v as m,y as h}from"./gpu-contributor-utils-BIwLQON5.js";import{t as g}from"./time-words-BFl8KCxq.js";import{t as _}from"./temporal-reduction-word-division-DIM5475u.js";import{d as v,i as y,r as b,t as x}from"./wgsl-common-DfyXiYB0.js";import{a as S,i as C}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as w}from"./define-run-B-brkCe4.js";import{a as T}from"./pointer-overlay-Qh8qflcA.js";import{c as E}from"./pick-Bo3t3hjv.js";import{n as D}from"./convert-CaKqeDF0.js";import{t as O}from"./lightning-C0gvbE9a.js";import{uc as k}from"./entry-preview-D70TdSbK.js";function A(e,t){let n=t.timestamps.format===`uint32x2`,r=t.timestamps.length,i=n?`${g}
${_}
override BUCKET_COUNT: u32;
override ROW_COUNT: u32;

fn getRowTime(row: u32) -> vec2<u32> {
  return vec2<u32>(rowTimes[rowTimesOffset + 2u * row], rowTimes[rowTimesOffset + 2u * row + 1u]);
}

fn isQuotientAtLeastBucketCount(difference: vec2<u32>, width: vec2<u32>) -> bool {
  let division = divideTimeDifference(difference, width);
  return division.overflow || division.quotient >= BUCKET_COUNT;
}

fn isRowBelowRange(row: u32, origin: vec2<u32>) -> bool {
  return timeWordsCompare(getRowTime(row), origin) < 0i;
}

// Only called for rows at or after the lower bound, which are not before the origin.
fn isRowAboveRange(row: u32, origin: vec2<u32>, width: vec2<u32>) -> bool {
  return isQuotientAtLeastBucketCount(timeWordsSubtract(getRowTime(row), origin), width);
}`:`${v}
override BUCKET_COUNT: u32;
override ROW_COUNT: u32;

fn isRowBelowRange(row: u32, origin: f32) -> bool {
  return rowTimes[rowTimesOffset + row] - origin < 0.0;
}

fn isRowAboveRange(row: u32, origin: f32, limit: f32) -> bool {
  return !(rowTimes[rowTimesOffset + row] - origin < limit);
}`,a=n?`let origin = vec2<u32>(params[paramsOffset], params[paramsOffset + 1u]);
  let width = vec2<u32>(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let isValid = width.x != 0u || width.y != 0u;`:`let origin = params[paramsOffset];
  let width = params[paramsOffset + 1u];
  let isValid = !isNanBits(origin) && !isNanBits(width) && width > 0.0 &&
    (bitcast<u32>(width) & 0x7fffffffu) != 0x7f800000u;
  let limit = f32(BUCKET_COUNT) * width;`,o=n?`isRowAboveRange(middle, origin, width)`:`isRowAboveRange(middle, origin, limit)`;return s(e,{id:t.id,operation:`GPUTemporalReduction`,variant:n?`range-search-words`:`range-search`,bindings:[{name:`rowTimes`,view:t.timestamps,type:n?`u32`:`f32`,access:`read`},{name:`params`,view:t.parameters,type:n?`u32`:`f32`,access:`read`},{name:`rowSlots`,view:t.rowSlots,type:`u32`,access:`read_write`}],invocationCount:1,workgroupSize:32,declarations:i,constants:{BUCKET_COUNT:t.bucketCount,ROW_COUNT:r},body:`
  ${a}
  var firstRow = 0u;
  var endRow = 0u;
  if (isValid) {
    // Lower bound: the first row that is not below the range.
    var low = 0u;
    var high = ROW_COUNT;
    loop {
      if (low >= high) { break; }
      let middle = low + (high - low) / 2u;
      if (isRowBelowRange(middle, origin)) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    firstRow = low;
    // Upper bound: the first row above the range, searched from the lower bound.
    high = ROW_COUNT;
    loop {
      if (low >= high) { break; }
      let middle = low + (high - low) / 2u;
      if (${o}) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    endRow = low;
  }
  rowSlots[rowSlotsOffset] = firstRow;
  rowSlots[rowSlotsOffset + 1u] = endRow - firstRow;`})}function j(e,t,n=new Float32Array(2)){if(n.length<2)throw Error(`Temporal reduction parameter target must hold 2 elements`);return n[0]=e,n[1]=t,n}var M=`GPUTemporalReduction`,N=256,P=`0xffffffffu`,F=256,I=4294967294,L=4294967295,R=2**20,z=class{id;props;slotCount;constructor(t){t=e(t,[`id`,`cellIds`,`timestamps`,`values`,`mask`,`parameters`,`cellCount`,`bucketCount`,`output`,`sorted`]);let n=e(t.output,[`counts`,`minimums`,`maximums`,`first`,`last`,`occupiedSlots`]),a=e(n.occupiedSlots,[`ids`,`status`]);t=Object.freeze({...t,output:Object.freeze({...n,occupiedSlots:Object.freeze({...a,status:e(a.status,[`count`,`overflow`,`candidateOverflow`,`requiredCount`])})})}),this.id=t.id??`temporal-reduction`,this.props=t;let o=this.id,{cellCount:s,bucketCount:l,output:p}=t;if(d(o,`cellCount`,s),d(o,`bucketCount`,l),this.slotCount=s*l,this.slotCount>I)throw Error(`${o} cellCount * bucketCount must not exceed ${I}`);let m=t.values.length;d(o,`values length`,m,1,L);let g=t.timestamps.format===`uint32x2`;c(t.cellIds,[`uint32`],`${o} cellIds`),c(t.timestamps,[g?`uint32x2`:`float32`],`${o} timestamps`),c(t.values,[`float32`],`${o} values`),t.mask&&c(t.mask,[`uint32`],`${o} mask`);for(let[e,n]of[[`cellIds`,t.cellIds],[`timestamps`,t.timestamps],[`mask`,t.mask]])if(n&&n.length!==m)throw Error(`${o} ${e} length must equal values length`);if(g){if(c(t.parameters,[`uint32`],`${o} parameters`),t.parameters.length<4)throw Error(`${o} parameters must hold 4 uint32 values`)}else{if(c(t.parameters,[`float32`],`${o} parameters`),t.parameters.length<2)throw Error(`${o} parameters must hold 2 float32 values`);if(l>R)throw Error(`${o} float32 times support at most ${R} buckets`)}c(p.counts,[`uint32`],`${o} output.counts`);for(let e of[`minimums`,`maximums`,`first`,`last`])c(p[e],[`float32`],`${o} output.${e}`);for(let e of[`counts`,`minimums`,`maximums`,`first`,`last`])if(p[e].length<this.slotCount)throw Error(`${o} output.${e} must hold cellCount * bucketCount rows`);f(o,p.occupiedSlots);{let e={counts:p.counts,minimums:p.minimums,maximums:p.maximums,first:p.first,last:p.last,ids:p.occupiedSlots.ids,count:p.occupiedSlots.status.count,overflow:p.occupiedSlots.status.overflow,candidateOverflow:p.occupiedSlots.status.candidateOverflow,requiredCount:p.occupiedSlots.status.requiredCount},t=[[p.counts,p.first,p.last],[p.counts,p.minimums,p.maximums],[p.occupiedSlots.status.count,p.occupiedSlots.status.overflow,p.occupiedSlots.status.candidateOverflow,p.occupiedSlots.status.requiredCount,...p.occupiedSlots.ids.length<this.slotCount?[p.occupiedSlots.ids]:[]]],n=Object.values(e).flatMap(e=>e?u(e):[]).map(e=>new i(e.buffer,{format:`uint32`,length:e.length*e.rowByteLength/4,byteOffset:e.byteOffset,byteStride:4,rowByteLength:4})),a=n.reduce((e,t)=>e+t.length,0);C(o,{outputs:new r({id:`${o}-writable-ranges`,name:`outputs`,format:`uint32`,length:a,valueLength:a,stride:1,byteStride:4,rowByteLength:4,data:n})});for(let e of t)C(o,Object.fromEntries(e.map((e,t)=>[String(t),e])))}h(o,[p.counts,p.minimums,p.maximums,p.first,p.last,p.occupiedSlots.ids,p.occupiedSlots.status.count,p.occupiedSlots.status.overflow,p.occupiedSlots.status.candidateOverflow,p.occupiedSlots.status.requiredCount],[t.cellIds,t.timestamps,t.values,t.mask,t.parameters])}getCommandNodes(e){let{props:r,id:i,slotCount:c}=this,{output:u,cellCount:d,bucketCount:f}=r;S(e,i,{cellIds:r.cellIds,timestamps:r.timestamps,values:r.values,mask:r.mask,parameters:r.parameters,counts:u.counts,minimums:u.minimums,maximums:u.maximums,first:u.first,last:u.last,occupiedIds:u.occupiedSlots.ids,count:u.occupiedSlots.status.count,overflow:u.occupiedSlots.status.overflow,candidateOverflow:u.occupiedSlots.status.candidateOverflow,requiredCount:u.occupiedSlots.status.requiredCount});let h=r.values.length,v=r.timestamps.format===`uint32x2`;if(m(e.device,i,`timestamps`,h*r.timestamps.rowByteLength,`reduce values.length or process the rows in several calls`),v||m(e.device,i,`slot columns`,c*4,`reduce cellCount * bucketCount`),v){let t=p(e.device,8);if(c>t)throw Error(`${i} cellCount * bucketCount must be at most ${t} with uint32x2 times`)}let y=Math.max(1,h+(r.sorted?2:0),h*(v?2:1),c*(v?2:1));m(e.device,i,`temporal scratch`,y*4,`reduce rows or slot count`),l(i,`temporal scratch byteLength`,y*4,e.device.limits.maxBufferSize);let b=(n,r)=>t(e,`${i}-${n}`,`uint32`,r),x=r.sorted===!0,C=x?`2u + `:``,w=b(`row-slots`,h+(x?2:0)),T=v?2:1,E=b(`row-time-keys`,T*h),D=x?b(`row-value-keys`,h):void 0,O=b(`min-keys`,c),k=b(`max-keys`,c),j=b(`first-time-keys`,T*c),I=b(`last-time-keys`,T*c),L=b(`first-rows`,c),R=b(`last-rows`,c),z=b(`occupied`,c),{counts:U}=u,W=D?`rowValueKeys[rowValueKeysOffset + index]`:`getOrderedKey(values[valuesOffset + index])`,G=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),K=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),q=[],J;x&&(q.push(A(e,{id:`${i}-range-search`,timestamps:r.timestamps,parameters:r.parameters,rowSlots:w,bucketCount:f})),J=a(e,{id:`${i}-range-dispatch`,operation:M,count:w,countRow:1,maximumItemCount:h,workgroupSize:F}),q.push(J.updateNode)),q.push(s(e,{id:`${i}-init`,operation:M,variant:`init`,bindings:[K(`counts`,U),K(`minKeys`,O),K(`maxKeys`,k),K(`firstTimeKeys`,j),K(`lastTimeKeys`,I),K(`firstRows`,L),K(`lastRows`,R)],invocationCount:c,body:`counts[countsOffset + index] = 0u;
  minKeys[minKeysOffset + index] = 0xffffffffu;
  maxKeys[maxKeysOffset + index] = 0u;
  firstTimeKeys[firstTimeKeysOffset + index] = 0xffffffffu;
  lastTimeKeys[lastTimeKeysOffset + index] = 0u;${v?`
  firstTimeKeys[firstTimeKeysOffset + ${c}u + index] = 0xffffffffu;
  lastTimeKeys[lastTimeKeysOffset + ${c}u + index] = 0u;`:``}
  firstRows[firstRowsOffset + index] = 0xffffffffu;
  lastRows[lastRowsOffset + index] = 0xffffffffu;`})),q.push(s(e,{id:`${i}-classify`,operation:M,variant:v?`classify-words`:`classify`,bindings:[G(`cellIds`,r.cellIds,`u32`),G(`timestamps`,r.timestamps,v?`u32`:`f32`),G(`values`,r.values,`f32`),...r.mask?[G(`rowMask`,r.mask,`u32`)]:[],G(`params`,r.parameters,v?`u32`:`f32`),K(`rowSlots`,w),K(`rowTimeKeys`,E),...D?[K(`rowValueKeys`,D)]:[]],invocationCount:h,...x?{workgroupSize:F,condition:J.condition}:{},declarations:`const CELL_COUNT: u32 = ${d}u;
const BUCKET_COUNT: u32 = ${f}u;
${B}
${v?`${g}\n${_}`:``}`,body:`${x?`let rangeFirstRow = rowSlots[rowSlotsOffset];
  let rangeRowCount = rowSlots[rowSlotsOffset + 1u];
  rowSlots[rowSlotsOffset + 2u + index] = ${P};
  if (index >= rangeRowCount) {
    return;
  }
  let row = rangeFirstRow + index;`:`rowSlots[rowSlotsOffset + index] = ${P};
  let row = index;`}
  let cell = cellIds[cellIdsOffset + row];
  let value = values[valuesOffset + row];
  if (cell >= CELL_COUNT || isNanBits(value)) {
    return;
  }
  ${r.mask?`if (rowMask[rowMaskOffset + row] == 0u) {
    return;
  }`:``}
  var bucket: u32;
  var timeKey: u32;
  var timeKeyLow = 0u;
  ${v?H:V}
  rowSlots[rowSlotsOffset + ${C}index] = cell * BUCKET_COUNT + bucket;
  rowTimeKeys[rowTimeKeysOffset + index] = timeKey;${v?`\n  rowTimeKeys[rowTimeKeysOffset + ${h}u + index] = timeKeyLow;`:``}${D?`
  rowValueKeys[rowValueKeysOffset + index] = getOrderedKey(value);`:``}`})),q.push(s(e,{id:`${i}-extremes`,operation:M,variant:`extremes`,bindings:[G(`rowSlots`,w,`u32`),G(`rowTimeKeys`,E,`u32`),D?G(`rowValueKeys`,D,`u32`):G(`values`,r.values,`f32`),K(`counts`,U,`atomic<u32>`),K(`minKeys`,O,`atomic<u32>`),K(`maxKeys`,k,`atomic<u32>`),K(`firstTimeKeys`,j,`atomic<u32>`),K(`lastTimeKeys`,I,`atomic<u32>`)],invocationCount:h,...x?{condition:J.condition}:{},...c<=N?{guardIndex:!1,declarations:`${B}
const SLOT_COUNT: u32 = ${c}u;
var<workgroup> localCounts: array<atomic<u32>, ${c}>;
var<workgroup> localMinimums: array<atomic<u32>, ${c}>;
var<workgroup> localMaximums: array<atomic<u32>, ${c}>;
var<workgroup> localFirstTimes: array<atomic<u32>, ${c}>;
var<workgroup> localLastTimes: array<atomic<u32>, ${c}>;`,body:`for (var slot = localInvocationIndex; slot < SLOT_COUNT; slot += 256u) {
    atomicStore(&localCounts[slot], 0u);
    atomicStore(&localMinimums[slot], 0xffffffffu);
    atomicStore(&localMaximums[slot], 0u);
    atomicStore(&localFirstTimes[slot], 0xffffffffu);
    atomicStore(&localLastTimes[slot], 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let slot = rowSlots[rowSlotsOffset + ${C}index];
    if (slot != ${P}) {
      let valueKey = ${W};
      let timeKey = rowTimeKeys[rowTimeKeysOffset + index];
      atomicAdd(&localCounts[slot], 1u);
      atomicMin(&localMinimums[slot], valueKey);
      atomicMax(&localMaximums[slot], valueKey);
      atomicMin(&localFirstTimes[slot], timeKey);
      atomicMax(&localLastTimes[slot], timeKey);
    }
  }
  workgroupBarrier();
  for (var slot = localInvocationIndex; slot < SLOT_COUNT; slot += 256u) {
    let count = atomicLoad(&localCounts[slot]);
    if (count != 0u) {
      atomicAdd(&counts[countsOffset + slot], count);
      atomicMin(&minKeys[minKeysOffset + slot], atomicLoad(&localMinimums[slot]));
      atomicMax(&maxKeys[maxKeysOffset + slot], atomicLoad(&localMaximums[slot]));
      atomicMin(&firstTimeKeys[firstTimeKeysOffset + slot], atomicLoad(&localFirstTimes[slot]));
      atomicMax(&lastTimeKeys[lastTimeKeysOffset + slot], atomicLoad(&localLastTimes[slot]));
    }
  }`}:{declarations:B,...x?{workgroupSize:F}:{},body:`let slot = rowSlots[rowSlotsOffset + ${C}index];
  if (slot == ${P}) {
    return;
  }
  let valueKey = ${W};
  let timeKey = rowTimeKeys[rowTimeKeysOffset + index];
  atomicAdd(&counts[countsOffset + slot], 1u);
  atomicMin(&minKeys[minKeysOffset + slot], valueKey);
  atomicMax(&maxKeys[maxKeysOffset + slot], valueKey);
  atomicMin(&firstTimeKeys[firstTimeKeysOffset + slot], timeKey);
  atomicMax(&lastTimeKeys[lastTimeKeysOffset + slot], timeKey);`}})),v&&q.push(s(e,{id:`${i}-time-low-words`,operation:M,variant:`time-low-words`,bindings:[G(`rowSlots`,w,`u32`),G(`rowTimeKeys`,E,`u32`),K(`firstTimeKeys`,j,`atomic<u32>`),K(`lastTimeKeys`,I,`atomic<u32>`)],invocationCount:h,...x?{workgroupSize:F,condition:J.condition}:{},body:`let slot = rowSlots[rowSlotsOffset + ${C}index];
  if (slot == ${P}) {
    return;
  }
  let timeKeyHigh = rowTimeKeys[rowTimeKeysOffset + index];
  let timeKeyLow = rowTimeKeys[rowTimeKeysOffset + ${h}u + index];
  if (timeKeyHigh == atomicLoad(&firstTimeKeys[firstTimeKeysOffset + slot])) {
    atomicMin(&firstTimeKeys[firstTimeKeysOffset + ${c}u + slot], timeKeyLow);
  }
  if (timeKeyHigh == atomicLoad(&lastTimeKeys[lastTimeKeysOffset + slot])) {
    atomicMax(&lastTimeKeys[lastTimeKeysOffset + ${c}u + slot], timeKeyLow);
  }`})),q.push(s(e,{id:`${i}-tie-rows`,operation:M,variant:`tie-rows`,bindings:[G(`rowSlots`,w,`u32`),G(`rowTimeKeys`,E,`u32`),G(`firstTimeKeys`,j,`u32`),G(`lastTimeKeys`,I,`u32`),K(`firstRows`,L,`atomic<u32>`),K(`lastRows`,R,`atomic<u32>`)],invocationCount:h,...x?{workgroupSize:F,condition:J.condition}:{},body:`let slot = rowSlots[rowSlotsOffset + ${C}index];
  if (slot == ${P}) {
    return;
  }
  let timeKey = rowTimeKeys[rowTimeKeysOffset + index];
  ${x?`let row = rowSlots[rowSlotsOffset] + index;`:`let row = index;`}
  ${v?`let timeKeyLow = rowTimeKeys[rowTimeKeysOffset + ${h}u + index];
  if (timeKey == firstTimeKeys[firstTimeKeysOffset + slot] &&
      timeKeyLow == firstTimeKeys[firstTimeKeysOffset + ${c}u + slot]) {
    atomicMin(&firstRows[firstRowsOffset + slot], row);
  }
  if (timeKey == lastTimeKeys[lastTimeKeysOffset + slot] &&
      timeKeyLow == lastTimeKeys[lastTimeKeysOffset + ${c}u + slot]) {
    atomicMin(&lastRows[lastRowsOffset + slot], row);
  }`:`if (timeKey == firstTimeKeys[firstTimeKeysOffset + slot]) {
    atomicMin(&firstRows[firstRowsOffset + slot], row);
  }
  if (timeKey == lastTimeKeys[lastTimeKeysOffset + slot]) {
    atomicMin(&lastRows[lastRowsOffset + slot], row);
  }`}`})),q.push(s(e,{id:`${i}-finish-values`,operation:M,variant:`finish-values`,bindings:[G(`values`,r.values,`f32`),G(`firstRows`,L,`u32`),G(`lastRows`,R,`u32`),K(`counts`,U,`u32`),K(`firstOut`,u.first,`f32`),K(`lastOut`,u.last,`f32`),K(`occupied`,z)],invocationCount:c,declarations:B,body:`let count = counts[countsOffset + index];
  let isOccupied = count != 0u;
  occupied[occupiedOffset + index] = select(0u, 1u, isOccupied);
  let nan = getNaN();
  firstOut[firstOutOffset + index] = select(nan, values[valuesOffset + firstRows[firstRowsOffset + index]], isOccupied);
  lastOut[lastOutOffset + index] = select(nan, values[valuesOffset + lastRows[lastRowsOffset + index]], isOccupied);`})),q.push(s(e,{id:`${i}-finish-extrema`,operation:M,variant:`finish-extrema`,bindings:[G(`minKeys`,O,`u32`),G(`maxKeys`,k,`u32`),K(`counts`,U,`u32`),K(`minOut`,u.minimums,`f32`),K(`maxOut`,u.maximums,`f32`)],invocationCount:c,declarations:B,body:`let isOccupied = counts[countsOffset + index] != 0u;
  let nan = getNaN();
  minOut[minOutOffset + index] = select(nan, decodeOrderedKey(minKeys[minKeysOffset + index]), isOccupied);
  maxOut[maxOutOffset + index] = select(nan, decodeOrderedKey(maxKeys[maxKeysOffset + index]), isOccupied);`}));let Y=u.occupiedSlots.ids.length>=c,X=b(`total`,1),Z=Y?u.occupiedSlots.ids:b(`compact-ids`,c);return q.push(...new n({id:`${i}-compact`,predicates:[{kind:`selection`,mask:z}],output:Z,count:X}).getCommandNodes(e)),q.push(o(e,{id:`${i}-publish`,operation:M,requiredCount:X,compactIds:Y?void 0:Z,output:u.occupiedSlots})),q}},B=`
// WGSL rejects NaN constants, so build one from a runtime bit pattern.
${b}

${v}

${y}

${x}
`,V=`let time = timestamps[timestampsOffset + row];
  let origin = params[paramsOffset];
  let width = params[paramsOffset + 1u];
  if (isNanBits(time) || isNanBits(origin) || isNanBits(width)) {
    return;
  }
  if (!(width > 0.0) || (bitcast<u32>(width) & 0x7fffffffu) == 0x7f800000u) {
    return;
  }
  let difference = time - origin;
  var bucketFloat = floor(difference / width);
  if (isNanBits(bucketFloat) || bucketFloat < -1.0 || bucketFloat > f32(BUCKET_COUNT) + 1.0) {
    return;
  }
  // Correct the estimate with correctly rounded products: one step down, then one step up.
  if (difference < bucketFloat * width) {
    bucketFloat = bucketFloat - 1.0;
  } else if (difference >= (bucketFloat + 1.0) * width) {
    bucketFloat = bucketFloat + 1.0;
  }
  if (bucketFloat < 0.0 || bucketFloat >= f32(BUCKET_COUNT)) {
    return;
  }
  bucket = u32(bucketFloat);
  timeKey = getOrderedKey(select(time, 0.0, time == 0.0));`,H=`let origin = vec2<u32>(params[paramsOffset], params[paramsOffset + 1u]);
  let width = vec2<u32>(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  if (width.x == 0u && width.y == 0u) {
    return;
  }
  let time = vec2<u32>(timestamps[timestampsOffset + 2u * row], timestamps[timestampsOffset + 2u * row + 1u]);
  // Exact signed order first: the sign of a wrapped difference is wrong when |time - origin| >=
  // 2^63. Past the check, time - origin is a non-negative value below 2^64 (unsigned words).
  if (timeWordsCompare(time, origin) < 0i) {
    return;
  }
  let difference = timeWordsSubtract(time, origin);
  // Exact 64-bit by 64-bit division; the remainder is the 64-bit offset inside the bucket.
  let division = divideTimeDifference(difference, width);
  if (division.overflow || division.quotient >= BUCKET_COUNT) {
    return;
  }
  bucket = division.quotient;
  timeKey = division.remainder.y;
  timeKeyLow = division.remainder.x;`,U=4e3,W=12,G=18;function K({world:e}){let t=O({width:1,height:1},U),n=Uint32Array.from(t.hours,(e,n)=>Math.min(11,Math.floor(t.positions[n*2]*W))),r=1;for(let e=0;e<=12;e+=.5){let i=new Uint32Array(216);t.hours.forEach((t,a)=>{let o=Math.floor((t-e)/2);o>=0&&o<G&&(r=Math.max(r,++i[n[a]*G+o]))})}return{world:e,cells:n,hours:t.hours,energy:t.energy,tallest:r}}var q=w({parameters:k,caption:`Midwest lightning flashes by longitude band and hours since May 21, 2024, 12:00 UTC.`,createInputs:K,build(e,t){let n=e.upload(`cell-ids`,`uint32`,t.cells),r=e.upload(`timestamps`,`float32`,t.hours),i=e.upload(`energy`,`float32`,t.energy),a=e.allocate(`parameters`,`float32`,2),o=e.allocate(`counts`,`uint32`,216);e.graph.add(new z({cellIds:n.view,timestamps:r.view,values:i.view,cellCount:W,bucketCount:G,parameters:a.view,output:{counts:o.view,minimums:e.allocate(`minimums`,`float32`,216).view,maximums:e.allocate(`maximums`,`float32`,216).view,first:e.allocate(`first`,`float32`,216).view,last:e.allocate(`last`,`float32`,216).view,occupiedSlots:{ids:e.allocate(`occupied`,`uint32`,216).view,status:{count:e.allocate(`occupied-count`,`uint32`,1).view,overflow:e.allocate(`occupied-overflow`,`uint32`,1).view}}}}));let s=D(e,`slot-counts`,o),c=e.allocate(`ticks`,`float32x4`,4);return{outputs:{slotCounts:s,ticks:c},update:e=>{c.write(Float32Array.from({length:16},(n,r)=>{let i=((r>>2)*6-e.origin)/e.width,a=i>=0&&i<=G,o=a?i/G*t.world.width:0;return[o,1,o,a?.96:1][r%4]})),a.write(j(e.origin,e.width))}}},rebuildOn:[],interaction:T(k,{kind:`pad`,hint:`slide origin, width up`,x:`origin`,y:`width`,decorate:(e,t,{inputs:n,outputs:r,peek:i})=>{let a=E(G,W,[0,0,n.world.width,1],t.x,t.y),o=i(r.slotCounts)?.[a.index],s=a.column/G*n.world.width,c=(a.column+1)/G*n.world.width;return{glyphs:[{shape:`box`,minimum:[s,1-(a.row+1)/W],maximum:[c,1-a.row/W]}],readout:`origin ${e.origin} h · width ${e.width} h · ${o??`…`} flashes here`}}}),draw:({inputs:e,outputs:t})=>[{layer:`raster`,values:t.slotCounts,columns:G,rows:W,range:[0,e.tallest]},{layer:`lines`,context:!0,segments:t.ticks,count:4,color:[1,1,1,.8]}]});export{q as default};