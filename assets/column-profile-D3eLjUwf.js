import{a as e,i as t}from"./graph-chunk-dispatch-zBt8k2-o.js";import{l as n,o as r}from"./gpu-command-graph-BdzdmV6u.js";import{E as i,S as a,T as o,c as s,f as c,u as l,v as u,y as d}from"./gpu-contributor-utils-BIwLQON5.js";import{r as f}from"./kernel-C3tfQOn4.js";import{c as p,o as m,t as h}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as g}from"./column-classification-shared-BTIVlxdJ.js";import{c as ee}from"./wgsl-common-DfyXiYB0.js";import{t as _}from"./define-run-B-brkCe4.js";import{t as v}from"./lightning-C0gvbE9a.js";import{n as y}from"./convert-CaKqeDF0.js";import{o as b,p as x,t as S}from"./helpers-DPAd8TjN.js";import{Wc as C}from"./entry-preview-D70TdSbK.js";function w(e){return l(e,4)}var T=4294967295,E={count:0,nullCount:1,minimum:2,maximum:3,sum:4,mean:5,variance:6,sampleVariance:7,standardDeviation:8,distinctEstimate:9,overflowCount:10};function D(e){return e*2}function O(e,t=e.length,n){if(c(`GPUColumnProfile`,`columnCount`,t),e.length>t)throw Error(`GPUColumnProfile domains exceed columnCount`);let r=D(t),i=n??new Float32Array(r);if(i.length<r)throw Error(`GPUColumnProfile parameter target must hold ${r} elements`);for(let n=0;n<t;n++){let t=e[n],r=NaN,a=NaN;t&&(Array.isArray(t)?[r,a]=t:(r=t.minimum??NaN,a=t.maximum??NaN)),i[n*2]=r,i[n*2+1]=a}return i}var k=2048,A=7,j=4,M=4,N=2**100,P=2**-24,F=`
struct Moments {
  count: u32,
  nullCount: u32,
  finiteCount: u32,
  mean: f32,
  m2: f32,
  sum: f32,
  origin: f32,
}

fn emptyMoments() -> Moments {
  return Moments(0u, 0u, 0u, 0.0, 0.0, 0.0, 0.0);
}

fn mergeMoments(left: Moments, right: Moments) -> Moments {
  var result = emptyMoments();
  result.count = left.count + right.count;
  result.nullCount = left.nullCount + right.nullCount;
  result.sum = left.sum + right.sum;
  if (right.finiteCount == 0u) {
    result.finiteCount = left.finiteCount;
    result.origin = left.origin;
    result.mean = left.mean;
    result.m2 = left.m2;
  } else if (left.finiteCount == 0u) {
    result.finiteCount = right.finiteCount;
    result.origin = right.origin;
    result.mean = right.mean;
    result.m2 = right.m2;
  } else {
    let count = left.finiteCount + right.finiteCount;
    let delta = (right.origin - left.origin) + (right.mean - left.mean);
    result.origin = left.origin;
    result.finiteCount = count;
    result.mean = left.mean + delta * (f32(right.finiteCount) / f32(count));
    result.m2 = left.m2 + right.m2 + delta * delta * (f32(left.finiteCount) * f32(right.finiteCount) / f32(count));
  }
  return result;
}

var<workgroup> sharedMoments: array<Moments, 256>;

fn reduceSharedMoments(local: u32) {
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    workgroupBarrier();
    if (local < stride) {
      sharedMoments[local] = mergeMoments(sharedMoments[local], sharedMoments[local + stride]);
    }
  }
}
`,te=`
fn getBinIndex(value: f32, minimum: f32, maximum: f32) -> u32 {
  var difference = value - minimum;
  var range = maximum - minimum;
  if (range > ${p(N)}) {
    difference = difference * ${p(P)};
    range = range * ${p(P)};
  }
  let scaled = difference * f32(BIN_COUNT);
  var bin = clamp(floor(scaled / range), 0.0, f32(BIN_COUNT - 1u));
  if (scaled < bin * range) {
    bin = bin - 1.0;
  } else if (scaled >= (bin + 1.0) * range) {
    bin = bin + 1.0;
  }
  return u32(clamp(bin, 0.0, f32(BIN_COUNT - 1u)));
}
`,ne=`
fn hashBits(bits: u32) -> u32 {
  var h = bits;
  h = h ^ (h >> 16u);
  h = h * 0x85ebca6bu;
  h = h ^ (h >> 13u);
  h = h * 0xc2b2ae35u;
  h = h ^ (h >> 16u);
  return h;
}
`,I=1024;function L(e){let{maximumSlotCount:t,slotCount:n,targetName:r,targetBase:i,slotBody:a}=e,o=`const NO_SLOT: u32 = 0xffffffffu;
fn getSlot(index: u32, column: u32) -> u32 {
  ${a}
}`,s=`let column = columnCursor[columnCursorOffset];`,c=`${r}[${r}Offset + ${i} + `;return t>I||t<1?{workgroupSize:256,declarations:`${e.declarations}\n${o}`,body:`${s}
  let slot = getSlot(index, column);
  if (slot != NO_SLOT) {
    atomicAdd(&${c}slot], 1u);
  }`}:{declarations:`${e.declarations}
${o}
var<workgroup> localCounts: array<atomic<u32>, ${t}>;`,body:`${s}
  let slotCount = ${n};
  for (var slot = localInvocationIndex; slot < slotCount; slot += 256u) {
    atomicStore(&localCounts[slot], 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let slot = getSlot(index, column);
    if (slot != NO_SLOT) {
      atomicAdd(&localCounts[slot], 1u);
    }
  }
  workgroupBarrier();
  for (var slot = localInvocationIndex; slot < slotCount; slot += 256u) {
    let count = atomicLoad(&localCounts[slot]);
    if (count != 0u) {
      atomicAdd(&${c}slot], count);
    }
  }`,guardIndex:!1,workgroupSize:256}}function R(e){return e===16?.673:e===32?.697:e===64?.709:.7213/(1+1.079/e)}function z(e){return e.reduce((e,t)=>e+(t.kind===`category`?t.categoryCount:0),0)}var B=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),V=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`});function re(e){let t=t=>`array<u32, ${e.length}>(${t.map(e=>`${e}u`).join(`, `)})`;return`const COLUMN_COUNT: u32 = ${e.length}u;
var<private> columnCategoryCounts: array<u32, ${e.length}> = ${t(e.map(e=>e.categoryCount))};
var<private> columnCategoryBases: array<u32, ${e.length}> = ${t(e.map(e=>e.categoryBase))};`}var H=`let column = columnCursor[columnCursorOffset];`;function ie(t,n){let{id:r,columns:i,output:o}=n;e(t,r,{...Object.fromEntries(i.map((e,t)=>[`columns[${t}].values`,e.values])),mask:n.mask,parameters:n.parameters,...o});let s=i.length,c={partials:s*Math.ceil(n.rowCount/k)*A,records:s*A,keys:s*j,domains:s*M,registers:o.hyperLogLogRegisters?0:s*2**n.hyperLogLogPrecision,categoryCounts:o.histograms||o.topCategories?z(i):0,columnCursor:1};for(let[e,n]of Object.entries(c))n>0&&(a(r,`${e} scratch byteLength`,n*4,t.device.limits.maxBufferSize),u(t.device,r,`${e} scratch`,n*Uint32Array.BYTES_PER_ELEMENT,`reduce the column row count or category count`))}function U(e,t){ie(e,t);let{id:n,operation:r,columns:i,rowCount:a,mask:o,parameters:s,output:c}=t,{histogramBinCount:l,hyperLogLogPrecision:u,topCategoryCount:d}=t,_=i.length,v=2**u,y=Math.ceil(a/k),b=z(i),x=!!c.histograms,S=!!(c.topCategories&&c.topCategoryCounts),C=i.some(e=>e.kind===`category`)&&(x||S),w=o?[B(`rowMask`,o,`u32`)]:[],E=e=>e.kind===`numeric`?`f32`:`u32`,D=(t,r)=>f(e,`${n}-${t}`,`uint32`,r),O=(t,r)=>f(e,`${n}-${t}`,`float32`,r),N=D(`partials`,_*y*A),P=D(`records`,_*A),I=D(`keys`,_*j),R=O(`domains`,_*M),U=c.hyperLogLogRegisters??D(`registers`,_*v),W=C?D(`category-counts`,b):void 0,G=D(`column-cursor`,1),K=re(i),q=B(`columnCursor`,G,`u32`),J=[];J.push(m(e,{id:`${n}-init`,operation:r,variant:`init`,bindings:[V(`keys`,I),V(`columnCursor`,G)],invocationCount:_*j,body:`keys[keysOffset + index] = select(0xffffffffu, 0u, (index & 1u) == 1u);
  if (index == 0u) {
    columnCursor[columnCursorOffset] = 0u;
  }`}));let Y=(t,i,a,o)=>J.push(h(e,{id:`${n}-${t}`,operation:r,view:i,type:`u32`,value:o,componentCount:a}));Y(`clear-registers`,U,_*v,`0u`),c.histograms&&Y(`clear-histograms`,c.histograms,_*l,`0u`),W&&Y(`clear-category-counts`,W,b,`0u`),S&&(Y(`clear-top-categories`,c.topCategories,_*d,`0xffffffffu`),Y(`clear-top-category-counts`,c.topCategoryCounts,_*d,`0u`));for(let[t,o]of i.entries())J.push(ae(e,{id:`${n}-column-${t}-moments-tile`,operation:r,column:o,rowCount:a,tileCount:y,maskBinding:w,partials:N,keys:I,columnCursor:G,columnConstants:K})),J.push(m(e,{id:`${n}-column-${t}-advance`,operation:r,variant:`column-advance`,bindings:[V(`columnCursor`,G)],invocationCount:1,declarations:`const COLUMN_COUNT: u32 = ${_}u;`,body:`columnCursor[columnCursorOffset] = (columnCursor[columnCursorOffset] + 1u) % COLUMN_COUNT;`}));J.push(m(e,{id:`${n}-moments-merge`,operation:r,variant:`moments-merge`,bindings:[B(`partials`,N,`u32`),V(`records`,P)],invocationCount:_*256,workgroupSize:256,guardIndex:!1,declarations:`const TILE_COUNT: u32 = ${y}u;
const STEP_COUNT: u32 = ${Math.ceil(y/256)}u;
${F}`,body:`let local = localInvocationIndex;
  let column = workgroupIndex;
  var accumulated = emptyMoments();
  for (var step = 0u; step < STEP_COUNT; step = step + 1u) {
    let tile = step * 256u + local;
    if (tile < TILE_COUNT) {
      let base = partialsOffset + (column * TILE_COUNT + tile) * ${A}u;
      accumulated = mergeMoments(
        accumulated,
        Moments(partials[base], partials[base + 1u], partials[base + 2u], bitcast<f32>(partials[base + 3u]), bitcast<f32>(partials[base + 4u]), bitcast<f32>(partials[base + 5u]), bitcast<f32>(partials[base + 6u]))
      );
    }
  }
  sharedMoments[local] = accumulated;
  reduceSharedMoments(local);
  if (local == 0u) {
    let record = sharedMoments[0];
    let base = recordsOffset + column * ${A}u;
    records[base] = record.count;
    records[base + 1u] = record.nullCount;
    records[base + 2u] = record.finiteCount;
    records[base + 3u] = bitcast<u32>(record.mean);
    records[base + 4u] = bitcast<u32>(record.m2);
    records[base + 5u] = bitcast<u32>(record.sum);
    records[base + 6u] = bitcast<u32>(record.origin);
  }`})),x&&J.push(m(e,{id:`${n}-domain`,operation:r,variant:`domain`,bindings:[B(`keys`,I,`u32`),...s?[B(`params`,s,`f32`)]:[],V(`domains`,R,`f32`)],invocationCount:_,declarations:`const INVERSE_BIN_COUNT: f32 = ${p(1/l)};
${g}
${ee}`,body:`let column = index;
  let keyBase = keysOffset + column * ${j}u;
  var lo = getNaN();
  var hi = getNaN();
  ${s?`lo = params[paramsOffset + column * 2u];
  hi = params[paramsOffset + column * 2u + 1u];`:``}
  let finiteMinimumKey = keys[keyBase + 2u];
  let finiteMaximumKey = keys[keyBase + 3u];
  if (isNanBits(lo)) {
    lo = select(getNaN(), decodeOrderedKey(finiteMinimumKey), finiteMinimumKey != 0xffffffffu);
  }
  if (isNanBits(hi)) {
    hi = select(getNaN(), decodeOrderedKey(finiteMaximumKey), finiteMaximumKey != 0u);
  }
  var mode = 0.0;
  var width = 0.0;
  if (isFiniteBits(lo) && isFiniteBits(hi) && lo <= hi) {
    if (lo == hi) {
      mode = 2.0;
    } else {
      let range = hi - lo;
      width = range * INVERSE_BIN_COUNT;
      if (isFiniteBits(range) && isFiniteBits(width) && width > 0.0) {
        mode = 1.0;
      }
    }
  }
  let base = domainsOffset + column * ${M}u;
  domains[base] = lo;
  domains[base + 1u] = hi;
  domains[base + 2u] = width;
  domains[base + 3u] = mode;`}));let X=Math.max(0,...i.map(e=>e.kind===`category`?e.categoryCount:0));for(let[t,s]of i.entries()){let i=`${n}-column-${t}`,f=s.kind===`numeric`,p=E(s);x&&f&&J.push(m(e,{id:`${i}-histogram`,operation:r,variant:`histogram`,bindings:[B(`values`,s.values,`f32`),...w,B(`domains`,R,`f32`),V(`histogram`,c.histograms,`atomic<u32>`),q],invocationCount:a,...L({maximumSlotCount:l,slotCount:`${l}u`,targetName:`histogram`,targetBase:`column * BIN_COUNT`,declarations:`const BIN_COUNT: u32 = ${l}u;
${te}`,slotBody:`${o?`if (rowMask[rowMaskOffset + index] == 0u) {
    return NO_SLOT;
  }`:``}
  let value = values[valuesOffset + index];
  if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
    return NO_SLOT;
  }
  let base = domainsOffset + column * ${M}u;
  let mode = domains[base + 3u];
  let lo = domains[base];
  let hi = domains[base + 1u];
  if (mode == 0.0 || value < lo || value > hi) {
    return NO_SLOT;
  }
  var bin = 0u;
  if (mode == 1.0) {
    bin = getBinIndex(value, lo, hi);
  }
  return bin;`})})),J.push(m(e,{id:`${i}-hyperloglog`,operation:r,variant:`hyperloglog`,bindings:[B(`values`,s.values,p),...w,V(`registers`,U,`atomic<u32>`),q],invocationCount:a,declarations:`const PRECISION: u32 = ${u}u;
const REGISTER_COUNT: u32 = ${v}u;
${ne}`,body:`${H}
  // Equals getColumnProfileHashSeed(column).
  let seed = (column + 1u) * 0x9e3779b9u;
  ${o?`if (rowMask[rowMaskOffset + index] == 0u) {
    return;
  }`:``}
  ${f?`var bits = bitcast<u32>(values[valuesOffset + index]);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return;
  }
  if ((bits & 0x7fffffffu) == 0u) {
    bits = 0u;
  }`:`let bits = values[valuesOffset + index];
  if (bits == ${T}u) {
    return;
  }`}
  let hash = hashBits(bits ^ seed);
  let registerIndex = hash >> (32u - PRECISION);
  let rank = min(countLeadingZeros(hash << PRECISION), 32u - PRECISION) + 1u;
  atomicMax(&registers[registersOffset + column * REGISTER_COUNT + registerIndex], rank);`})),W&&!f&&J.push(m(e,{id:`${i}-category-counts`,operation:r,variant:`category-counts`,bindings:[B(`values`,s.values,`u32`),...w,V(`categoryCounts`,W,`atomic<u32>`),q],invocationCount:a,...L({maximumSlotCount:X,slotCount:`columnCategoryCounts[column]`,targetName:`categoryCounts`,targetBase:`columnCategoryBases[column]`,declarations:K,slotBody:`${o?`if (rowMask[rowMaskOffset + index] == 0u) {
    return NO_SLOT;
  }`:``}
  let code = values[valuesOffset + index];
  if (code < columnCategoryCounts[column]) {
    return code;
  }
  return NO_SLOT;`})})),S&&W&&!f&&J.push(oe(e,{id:`${i}-top-categories`,operation:r,topCategoryCount:d,columnConstants:K,columnCursor:G,categoryCounts:W,topCategories:c.topCategories,topCategoryCounts:c.topCategoryCounts})),J.push(se(e,{id:`${i}-finish`,operation:r,column:s,columnConstants:K,columnCursor:G,registerCount:v,hyperLogLogPrecision:u,histogramBinCount:l,records:P,keys:I,registers:U,categoryCounts:W,output:c}))}return J}function ae(e,t){let{column:n,rowCount:r,tileCount:i,maskBinding:a}=t,o=n.kind===`numeric`,s=a.length>0,c=o?`let bits = bitcast<u32>(value);
      if ((bits & 0x7fffffffu) > 0x7f800000u) {
        accumulated.nullCount = accumulated.nullCount + 1u;
      } else {
        accumulated.count = accumulated.count + 1u;
        let key = getOrderedKey(value);
        minimumKey = min(minimumKey, key);
        maximumKey = max(maximumKey, key);
        if ((bits & 0x7fffffffu) != 0x7f800000u) {
          finiteMinimumKey = min(finiteMinimumKey, key);
          finiteMaximumKey = max(finiteMaximumKey, key);
          if (accumulated.finiteCount == 0u) {
            accumulated.origin = value;
          }
          let shiftedValue = value - accumulated.origin;
          let count = accumulated.finiteCount + 1u;
          let delta = shiftedValue - accumulated.mean;
          accumulated.finiteCount = count;
          accumulated.mean = accumulated.mean + delta / f32(count);
          accumulated.m2 = accumulated.m2 + delta * (shiftedValue - accumulated.mean);
          accumulated.sum = accumulated.sum + value;
        }
      }`:`if (value == ${T}u) {
        accumulated.nullCount = accumulated.nullCount + 1u;
      } else {
        accumulated.count = accumulated.count + 1u;
        if (value >= categoryCount) {
          accumulated.finiteCount = accumulated.finiteCount + 1u;
        }
        minimumKey = min(minimumKey, value);
        maximumKey = max(maximumKey, value);
      }`;return m(e,{id:t.id,operation:t.operation,variant:`moments-tile`,bindings:[B(`values`,n.values,o?`f32`:`u32`),...a,V(`partials`,t.partials),V(`keys`,t.keys,`atomic<u32>`),B(`columnCursor`,t.columnCursor,`u32`)],invocationCount:i*256,workgroupSize:256,guardIndex:!1,declarations:`const ROW_COUNT: u32 = ${r}u;
const TILE_COUNT: u32 = ${i}u;
const TILE_SIZE: u32 = ${k}u;
const ROWS_PER_INVOCATION: u32 = 8u;
${t.columnConstants}
${g}
${F}
var<workgroup> sharedKeys: array<atomic<u32>, ${j}>;`,body:`let local = localInvocationIndex;
  let tile = workgroupIndex;
  ${H}
  let categoryCount = columnCategoryCounts[column];
  if (local < ${j}u) {
    atomicStore(&sharedKeys[local], select(0xffffffffu, 0u, (local & 1u) == 1u));
  }
  workgroupBarrier();
  var accumulated = emptyMoments();
  var minimumKey = 0xffffffffu;
  var maximumKey = 0u;
  var finiteMinimumKey = 0xffffffffu;
  var finiteMaximumKey = 0u;
  for (var step = 0u; step < ROWS_PER_INVOCATION; step = step + 1u) {
    // Uniform trip count: rows past the end are skipped by a predicate, never by a break.
    let row = tile * TILE_SIZE + step * 256u + local;
    if (row < ROW_COUNT${s?` && rowMask[rowMaskOffset + row] != 0u`:``}) {
      let value = values[valuesOffset + row];
      ${c}
    }
  }
  sharedMoments[local] = accumulated;
  atomicMin(&sharedKeys[0], minimumKey);
  atomicMax(&sharedKeys[1], maximumKey);
  atomicMin(&sharedKeys[2], finiteMinimumKey);
  atomicMax(&sharedKeys[3], finiteMaximumKey);
  reduceSharedMoments(local);
  workgroupBarrier();
  if (local == 0u) {
    let record = sharedMoments[0];
    let base = partialsOffset + (column * TILE_COUNT + tile) * ${A}u;
    partials[base] = record.count;
    partials[base + 1u] = record.nullCount;
    partials[base + 2u] = record.finiteCount;
    partials[base + 3u] = bitcast<u32>(record.mean);
    partials[base + 4u] = bitcast<u32>(record.m2);
    partials[base + 5u] = bitcast<u32>(record.sum);
    partials[base + 6u] = bitcast<u32>(record.origin);
    let keyBase = keysOffset + column * ${j}u;
    atomicMin(&keys[keyBase], atomicLoad(&sharedKeys[0]));
    atomicMax(&keys[keyBase + 1u], atomicLoad(&sharedKeys[1]));
    atomicMin(&keys[keyBase + 2u], atomicLoad(&sharedKeys[2]));
    atomicMax(&keys[keyBase + 3u], atomicLoad(&sharedKeys[3]));
  }`})}function oe(e,t){let{topCategoryCount:n}=t;return m(e,{id:t.id,operation:t.operation,variant:`top-categories`,bindings:[B(`categoryCounts`,t.categoryCounts,`u32`),V(`topCategories`,t.topCategories),V(`topCategoryCounts`,t.topCategoryCounts),B(`columnCursor`,t.columnCursor,`u32`)],invocationCount:256,workgroupSize:256,guardIndex:!1,declarations:`${t.columnConstants}
const TOP_COUNT: u32 = ${n}u;
var<workgroup> bestCounts: array<u32, 256>;
var<workgroup> bestCodes: array<u32, 256>;`,body:`let local = localInvocationIndex;
  ${H}
  let categoryCount = columnCategoryCounts[column];
  let base = columnCategoryBases[column];
  let stepCount = (categoryCount + 255u) / 256u;
  var previousCount = 0u;
  var previousCode = 0u;
  var hasPrevious = false;
  for (var rank = 0u; rank < TOP_COUNT; rank = rank + 1u) {
    var bestCount = 0u;
    var bestCode = 0xffffffffu;
    for (var step = 0u; step < stepCount; step = step + 1u) {
      let code = step * 256u + local;
      if (code < categoryCount) {
        let count = categoryCounts[categoryCountsOffset + base + code];
        // Strictly after the previous pick in (count descending, code ascending) order.
        let isCandidate = count > 0u && (!hasPrevious || count < previousCount || (count == previousCount && code > previousCode));
        if (isCandidate && count > bestCount) {
          bestCount = count;
          bestCode = code;
        }
      }
    }
    bestCounts[local] = bestCount;
    bestCodes[local] = bestCode;
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
      workgroupBarrier();
      if (local < stride) {
        let otherCount = bestCounts[local + stride];
        let otherCode = bestCodes[local + stride];
        if (otherCount > bestCounts[local] || (otherCount == bestCounts[local] && otherCode < bestCodes[local])) {
          bestCounts[local] = otherCount;
          bestCodes[local] = otherCode;
        }
      }
    }
    workgroupBarrier();
    let pickedCount = bestCounts[0];
    let pickedCode = bestCodes[0];
    if (local == 0u) {
      topCategories[topCategoriesOffset + column * TOP_COUNT + rank] = select(0xffffffffu, pickedCode, pickedCount > 0u);
      topCategoryCounts[topCategoryCountsOffset + column * TOP_COUNT + rank] = pickedCount;
    }
    previousCount = pickedCount;
    previousCode = pickedCode;
    hasPrevious = true;
    workgroupBarrier();
  }`})}function se(e,t){let{column:n,registerCount:r,histogramBinCount:i,output:a}=t,o=n.kind===`numeric`,s=R(r)*r*r,c=!o&&a.histograms&&t.categoryCounts,l=E,u=[B(`records`,t.records,`u32`),B(`keys`,t.keys,`u32`),V(`registers`,t.registers,`u32`),V(`statistics`,a.statistics,`f32`)];a.counts&&u.push(V(`countsOut`,a.counts)),c&&u.push(B(`categoryCounts`,t.categoryCounts,`u32`),V(`histogram`,a.histograms)),u.push(V(`columnCursor`,t.columnCursor));let d=o?`let hasValues = count > 0.0;
    let hasFinite = finiteCount > 0.0;
    let variance = select(nan, m2 / finiteCount, hasFinite);
    statistics[statisticsBase + ${l.minimum}u] = select(nan, decodeOrderedKey(minimumKey), hasValues);
    statistics[statisticsBase + ${l.maximum}u] = select(nan, decodeOrderedKey(maximumKey), hasValues);
    statistics[statisticsBase + ${l.sum}u] = select(nan, sum, hasFinite);
    statistics[statisticsBase + ${l.mean}u] = select(nan, mean, hasFinite);
    statistics[statisticsBase + ${l.variance}u] = variance;
    statistics[statisticsBase + ${l.sampleVariance}u] = select(nan, m2 / f32(exactFiniteCount - min(exactFiniteCount, 1u)), exactFiniteCount >= 2u);
    statistics[statisticsBase + ${l.standardDeviation}u] = select(nan, sqrt(variance), hasFinite);
    statistics[statisticsBase + ${l.overflowCount}u] = 0.0;`:`let hasValues = count > 0.0;
    statistics[statisticsBase + ${l.minimum}u] = select(nan, f32(minimumKey), hasValues);
    statistics[statisticsBase + ${l.maximum}u] = select(nan, f32(maximumKey), hasValues);
    statistics[statisticsBase + ${l.sum}u] = nan;
    statistics[statisticsBase + ${l.mean}u] = nan;
    statistics[statisticsBase + ${l.variance}u] = nan;
    statistics[statisticsBase + ${l.sampleVariance}u] = nan;
    statistics[statisticsBase + ${l.standardDeviation}u] = nan;
    statistics[statisticsBase + ${l.overflowCount}u] = finiteCount;`;return m(e,{id:t.id,operation:t.operation,variant:`finish`,bindings:u,invocationCount:256,workgroupSize:256,guardIndex:!1,declarations:`${t.columnConstants}
const REGISTER_COUNT: u32 = ${r}u;
const REGISTER_STEP_COUNT: u32 = ${Math.ceil(r/256)}u;
const ALPHA_TIMES_SQUARE: f32 = ${p(s)};
const BIN_COUNT: u32 = ${i}u;
${g}
var<workgroup> sharedColumn: u32;
var<workgroup> sharedInverseSums: array<f32, 256>;
var<workgroup> sharedZeroCounts: array<u32, 256>;`,body:`let local = localInvocationIndex;
  if (local == 0u) {
    sharedColumn = columnCursor[columnCursorOffset];
  }
  let column = workgroupUniformLoad(&sharedColumn);
  var inverseSum = 0.0;
  var zeroCount = 0u;
  for (var step = 0u; step < REGISTER_STEP_COUNT; step = step + 1u) {
    let registerIndex = step * 256u + local;
    if (registerIndex < REGISTER_COUNT) {
      let rank = registers[registersOffset + column * REGISTER_COUNT + registerIndex];
      // 2^-rank is exact: rank is at most ${33-t.hyperLogLogPrecision}.
      inverseSum = inverseSum + bitcast<f32>((127u - rank) << 23u);
      zeroCount = zeroCount + select(0u, 1u, rank == 0u);
    }
  }
  sharedInverseSums[local] = inverseSum;
  sharedZeroCounts[local] = zeroCount;
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    workgroupBarrier();
    if (local < stride) {
      sharedInverseSums[local] = sharedInverseSums[local] + sharedInverseSums[local + stride];
      sharedZeroCounts[local] = sharedZeroCounts[local] + sharedZeroCounts[local + stride];
    }
  }
  ${c?`for (var bin = local; bin < BIN_COUNT; bin = bin + 256u) {
    histogram[histogramOffset + column * BIN_COUNT + bin] = select(0u, categoryCounts[categoryCountsOffset + columnCategoryBases[column] + bin], bin < columnCategoryCounts[column]);
  }`:``}
  if (local == 0u) {
    let recordBase = recordsOffset + column * ${A}u;
    let exactCount = records[recordBase];
    let count = f32(exactCount);
    let exactNullCount = records[recordBase + 1u];
    let nullCount = f32(exactNullCount);
    let exactFiniteCount = records[recordBase + 2u];
    let finiteCount = f32(exactFiniteCount);
    let mean = bitcast<f32>(records[recordBase + 6u]) + bitcast<f32>(records[recordBase + 3u]);
    let m2 = bitcast<f32>(records[recordBase + 4u]);
    let sum = bitcast<f32>(records[recordBase + 5u]);
    let keyBase = keysOffset + column * ${j}u;
    let minimumKey = keys[keyBase];
    let maximumKey = keys[keyBase + 1u];
    let nan = getNaN();
    let statisticsBase = statisticsOffset + column * 11u;
    statistics[statisticsBase + ${l.count}u] = count;
    statistics[statisticsBase + ${l.nullCount}u] = nullCount;
    ${d}
    let registerCountFloat = f32(REGISTER_COUNT);
    var estimate = ALPHA_TIMES_SQUARE / sharedInverseSums[0];
    let zeros = sharedZeroCounts[0];
    if (zeros == REGISTER_COUNT) {
      estimate = 0.0;
    } else if (estimate <= 2.5 * registerCountFloat && zeros > 0u) {
      // Linear counting m * ln(m / zeros). For few occupied registers the ratio is close to 1 and
      // log() cancels, so use the series of ln(1 + u) with the exact integer u = (m - zeros) / zeros.
      let occupied = f32(REGISTER_COUNT - zeros);
      let ratio = occupied / f32(zeros);
      if (ratio < 0.0625) {
        let r2 = ratio * ratio;
        estimate = registerCountFloat * (ratio - r2 * (0.5 - ratio * (0.33333334 - ratio * (0.25 - ratio * 0.2))));
      } else {
        estimate = registerCountFloat * log(registerCountFloat / f32(zeros));
      }
    }
    statistics[statisticsBase + ${l.distinctEstimate}u] = estimate;
    ${a.counts?`countsOut[countsOutOffset + column * 2u] = exactCount;
    countsOut[countsOutOffset + column * 2u + 1u] = exactNullCount;`:``}
    columnCursor[columnCursorOffset] = (column + 1u) % COLUMN_COUNT;
  }`})}var W=`GPUColumnProfile`,G=4294967295,K=2**24,q=2**20,J=12,Y=10,X=class{id;props;rowCount;hyperLogLogPrecision;topCategoryCount;columns;constructor(e){this.id=e.id??`column-profile`,this.props=e;let a=this.id,{output:l}=e,u=e.columns.length;c(a,`column count`,u,1,16);let{histogramBinCount:f}=e;c(a,`histogramBinCount`,f,1,q),this.hyperLogLogPrecision=e.hyperLogLogPrecision??J,c(a,`hyperLogLogPrecision`,this.hyperLogLogPrecision,4,16),this.topCategoryCount=e.topCategoryCount??Y,c(a,`topCategoryCount`,this.topCategoryCount,1,64);let p=e.columns[0].values.length;if(p<1)throw Error(`${a} needs at least one row`);o(a,p<=G,`columns[0] length must be at most ${G}: row indices and counts use uint32`),this.rowCount=p;let m=0;if(this.columns=e.columns.map((e,t)=>{let n=`columns[${t}]`;if(e.values.length!==p)throw Error(`${a} ${n} length must equal columns[0] length`);if(e.kind===`category`){i(e.values,[`uint32`],`${a} ${n}.values`),c(a,`${n}.categoryCount`,e.categoryCount,1,K);let t=m;return m+=e.categoryCount,{kind:`category`,values:e.values,categoryCount:e.categoryCount,categoryBase:t}}if(e.kind!==void 0&&e.kind!==`numeric`)throw Error(`${a} ${n}.kind must be 'numeric' or 'category'`);return i(e.values,[`float32`],`${a} ${n}.values`),{kind:`numeric`,values:e.values,categoryCount:0,categoryBase:0}}),e.mask&&(i(e.mask,[`uint32`],`${a} mask`),e.mask.length!==p))throw Error(`${a} mask length must equal the column length`);if(e.parameters){i(e.parameters,[`float32`],`${a} parameters`);let t=D(u);if(e.parameters.length<t)throw Error(`${a} parameters must hold ${t} float32 values`)}let h=(e,t,n)=>{if(t.length<n)throw Error(`${a} output.${e} must hold ${n} rows`)};i(l.statistics,[`float32`],`${a} output.statistics`),h(`statistics`,l.statistics,u*11);for(let[e,t,n]of[[`counts`,l.counts,u*2],[`histograms`,l.histograms,u*f],[`hyperLogLogRegisters`,l.hyperLogLogRegisters,u*2**this.hyperLogLogPrecision],[`topCategories`,l.topCategories,u*this.topCategoryCount],[`topCategoryCounts`,l.topCategoryCounts,u*this.topCategoryCount]])t&&(i(t,[`uint32`],`${a} output.${e}`),h(e,t,n));if(!!l.topCategories!=!!l.topCategoryCounts)throw Error(`${a} output.topCategories and output.topCategoryCounts go together`);{let i=l,o=[[l.topCategories,l.topCategoryCounts],[l.statistics,l.counts,l.hyperLogLogRegisters,...e.columns.some(e=>e.kind===`category`)?[l.histograms]:[]]],c=Object.values(i).flatMap(e=>e?s(e):[]).map(e=>new r(e.buffer,{format:`uint32`,length:e.length*e.rowByteLength/4,byteOffset:e.byteOffset,byteStride:4,rowByteLength:4})),u=c.reduce((e,t)=>e+t.length,0);t(a,{outputs:new n({id:`${a}-writable-ranges`,name:`outputs`,format:`uint32`,length:u,valueLength:u,stride:1,byteStride:4,rowByteLength:4,data:c})});for(let e of o)t(a,Object.fromEntries(e.map((e,t)=>[String(t),e])))}d(a,[l.statistics,l.counts,l.histograms,l.hyperLogLogRegisters,l.topCategories,l.topCategoryCounts],[...e.columns.map(e=>e.values),e.mask,e.parameters])}getCommandNodes(e){let{props:t,id:n}=this;return a(n,`category columns categoryCount total`,this.columns.reduce((e,t)=>e+t.categoryCount,0),w(e.device)),U(e,{id:n,operation:W,columns:this.columns,rowCount:this.rowCount,mask:t.mask,parameters:t.parameters,histogramBinCount:t.histogramBinCount,hyperLogLogPrecision:this.hyperLogLogPrecision,topCategoryCount:this.topCategoryCount,output:t.output})}},Z=2e3,Q=.45,$=[.07,.06,.95,.38];function ce(e,t){let n=x(e.world.width,e.mapHeight,t,[e.center[0],e.center[1]-Q]);return{minimumX:n.minimumX,maximumX:n.maximumX,minimumY:n.minimumY+Q,maximumY:n.maximumY+Q}}var le=_({parameters:C,caption:`Flash energy histogram in units of 10 fJ for the flashes inside the box (bright bars), over all flashes (faint bars).`,createInputs:({world:e})=>{let t=.55,n=v({width:e.width,height:t},Z);for(let e=0;e<Z;e++)n.positions[e*2+1]+=Q;return{world:e,mapHeight:t,points:n,column:Float32Array.from(n.energy,e=>e/10),center:[e.width/2,.7250000000000001],kept:{count:0,sum:0,bounds:{minimumX:0,minimumY:0,maximumX:0,maximumY:0}}}},build(e,t,n){let r=n.histogramBinCount,i=e.upload(`flash-energy`,`float32`,t.column),a=e.upload(`positions`,`float32x2`,t.points.positions),o=e.allocate(`kept-positions`,`float32x2`,Z),s=e.allocate(`kept-values`,`float32`,Z),c=e.allocate(`mask`,`uint32`,Z),l=e.allocate(`domain`,`float32`,2),u=e.allocate(`histograms`,`uint32`,r),d=e.allocate(`all-rows`,`float32`,r);return e.graph.add(new X({columns:[{values:i.view}],mask:c.view,parameters:l.view,histogramBinCount:r,output:{statistics:e.allocate(`statistics`,`float32`,11).view,counts:e.allocate(`counts`,`uint32`,2).view,histograms:u.view}})),{outputs:{column:i,positions:a,keptPositions:o,keptValues:s,bars:y(e,`bars`,u),allRows:d},update:e=>{let n=ce(t,e.mask),i=S(t.points.positions,n);c.write(i);let a=0,u=0,f=new Float32Array(Z*2),p=new Float32Array(Z);for(let e=0;e<Z;e++)i[e]&&(f.set([t.points.positions[e*2],t.points.positions[e*2+1]],a*2),p[a]=t.column[e],u+=t.column[e],a++);o.write(f),s.write(p),t.kept={count:a,sum:u,bounds:n},l.write(O([[0,e.domainMaximum]])),d.write(b(t.column,r,0,e.domainMaximum))}}},rebuildOn:[`histogramBinCount`],interaction:{kind:`window`,hint:`move the box`,rest:({inputs:e})=>[.5,1-(Q+e.mapHeight/2)],resolve(e,{inputs:t,values:n}){t.center=[e.x,Math.max(e.y,Q)];let r=ce(t,n.mask),{count:i,sum:a}=t.kept;return{rerun:!0,glyphs:[{shape:`box`,minimum:[r.minimumX,r.minimumY],maximum:[r.maximumX,r.maximumY]}],readout:`mask ${i} of ${Z} rows · mean ${i?Math.round(a/i*10):0} fJ`}}},draw:({inputs:e,outputs:t,values:n})=>{let r=[0,Math.max(...b(e.column,n.histogramBinCount,0,n.domainMaximum))*1.05];return[{layer:`dots`,context:!0,positions:t.positions,values:t.column,range:[0,50],radius:3},{layer:`dots`,positions:t.keptPositions,values:t.keptValues,count:Math.max(1,e.kept.count),range:[0,50],radius:3.6},{layer:`chart`,kind:`bars`,context:!0,values:t.allRows,range:r,rectangle:$},{layer:`chart`,kind:`bars`,color:[.2,.55,.9,1],values:t.bars,range:r,rectangle:$}]}});export{le as default};