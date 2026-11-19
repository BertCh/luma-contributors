import{r as e,s as t,u as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-sort-xRm8l5yn.js";import{t as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as a}from"./statistics-kernel-node-lacKNERm.js";import{E as o,f as s,v as c,x as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{a as d,i as f}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as p}from"./spatial-statistics-counts-wgsl-B8N7XLDs.js";import{r as m}from"./wgsl-common-DfyXiYB0.js";var h=`GPUMultipleTestingAdjustment`,g=256,_=1065353217,v=30,y=2**31+1,b=[`none`,`bonferroni`,`holm`,`benjamini-hochberg`,`benjamini-yekutieli`];function x(e,t){let n=t??`none`;if(!b.includes(n))throw Error(`${e} method must be one of ${b.join(`, `)}, got ${String(t)}`);return n}function S(e,t){let n=t??0;return s(e,`minimumFamilySize`,n,0,y-1),n}var C=class{id;props;constructor(e){let t=e.id??`multiple-testing-adjustment`;this.id=t,this.props=e,x(t,e.method),S(t,e.minimumFamilySize),o(e.pValues,[`float32`],`${t} pValues`),o(e.output.adjustedPValues,[`float32`],`${t} output.adjustedPValues`);let r=e.pValues.length;if(s(t,`pValues length`,r,1,y-1),e.output.adjustedPValues.length!==r)throw Error(`${t} adjustedPValues length must equal the pValues length`);e.output.testedCount&&(n(e.output.testedCount,`${t} output.testedCount`),s(t,`output.testedCount length`,e.output.testedCount.length)),u(t,[e.output.adjustedPValues,e.output.testedCount],[e.pValues]),f(t,e.output)}getCommandNodes(n){let{id:o,props:s}=this;l(o,n,[s.pValues,s.output.adjustedPValues,s.output.testedCount]),d(n,o,Object.fromEntries([s.pValues,s.output.adjustedPValues,s.output.testedCount].map((e,t)=>[`binding ${t}`,e])));for(let e of[s.pValues,s.output.adjustedPValues,s.output.testedCount])e&&c(n.device,o,`multiple testing binding`,t(e).size,`reduce rows or request larger device bindings`);c(n.device,o,`multiple testing row scratch`,s.pValues.length*4,`reduce rows or request larger device bindings`);let u=x(o,s.method),f=S(o,s.minimumFamilySize),y=s.pValues.length,b=[],C=`${m}
${p}
// A p-value is tested when it is in [0, 1]: +0..1.0 as positive bit patterns, or -0.
fn isTestedBits(bits: u32) -> bool {
  return bits <= 0x3f800000u || bits == 0x80000000u;
}`;if(u===`none`&&!s.output.testedCount)return b.push(a(n,{id:`${o}-apply`,operation:h,variant:`none`,bindings:[{name:`pValues`,view:s.pValues,type:`f32`,access:`read`},{name:`adjusted`,view:s.output.adjustedPValues,type:`f32`,access:`read_write`}],invocationCount:y,declarations:C,body:`let p = pValues[pValuesOffset + index];
  adjusted[adjustedOffset + index] = select(getNaN(), p + 0.0, isTestedBits(bitcast<u32>(p)));`})),b;let w=u===`holm`||u===`benjamini-hochberg`||u===`benjamini-yekutieli`,T=e(n,`${o}-counters`,`uint32`,1),E=e(n,`${o}-family`,`uint32`,2),D=e(n,`${o}-family-scale`,`float32`,1),O=w?e(n,`${o}-keys`,`uint32`,y):void 0,k=w?e(n,`${o}-row-ids`,`uint32`,y):void 0,A=[{name:`pValues`,view:s.pValues,type:`f32`,access:`read`},{name:`counters`,view:T,type:`atomic<u32>`,access:`read_write`}];O&&k&&A.push({name:`keys`,view:O,type:`u32`,access:`read_write`},{name:`rowIds`,view:k,type:`u32`,access:`read_write`}),b.push(i(n,{id:`${o}-counters-clear`,operation:h,view:T,type:`u32`,value:`0u`}),a(n,{id:`${o}-count`,operation:h,variant:O?`count-keys`:`count`,bindings:A,invocationCount:y,declarations:`${C}
const UNTESTED_KEY: u32 = ${_}u;`,body:`let bits = bitcast<u32>(pValues[pValuesOffset + index]);
  let tested = isTestedBits(bits);
  if (tested) {
    atomicAdd(&counters[countersOffset], 1u);
  }
  ${O?`keys[keysOffset + index] = select(UNTESTED_KEY, bits & 0x7fffffffu, tested);
  rowIds[rowIdsOffset + index] = index;`:``}`}));let j=[{name:`counters`,view:T,type:`u32`,access:`read`},{name:`family`,view:E,type:`u32`,access:`read_write`},{name:`familyScale`,view:D,type:`f32`,access:`read_write`}];if(s.output.testedCount&&j.push({name:`testedCount`,view:s.output.testedCount,type:`u32`,access:`read_write`}),b.push(a(n,{id:`${o}-family`,operation:h,variant:`family`,bindings:j,invocationCount:1,declarations:`const MINIMUM_FAMILY_SIZE: u32 = ${f}u;
const EULER_MASCHERONI: f32 = 0.5772157;
// Harmonic number c(m) = sum_{i=1..m} 1/i. Exact fixed-order sum (smallest terms first) for small
// m; asymptotic series ln m + gamma + 1/(2m) - 1/(12 m^2) (error < 1/(120 m^4)) beyond.
fn getHarmonicNumber(m: u32) -> f32 {
  if (m <= 64u) {
    var sum = 0.0;
    for (var i = m; i >= 1u; i--) {
      sum += 1.0 / f32(i);
    }
    return sum;
  }
  let x = f32(m);
  return log(x) + EULER_MASCHERONI + 0.5 / x - 1.0 / (12.0 * x * x);
}`,body:`let tested = counters[countersOffset];
  let m = max(tested, MINIMUM_FAMILY_SIZE);
  family[familyOffset] = tested;
  family[familyOffset + 1u] = m;
  familyScale[familyScaleOffset] = ${u===`benjamini-yekutieli`?`getHarmonicNumber(m)`:`1.0`};
  ${s.output.testedCount?`testedCount[testedCountOffset] = m;`:``}`})),u===`none`||u===`bonferroni`)return b.push(a(n,{id:`${o}-apply`,operation:h,variant:u,bindings:[{name:`pValues`,view:s.pValues,type:`f32`,access:`read`},{name:`family`,view:E,type:`u32`,access:`read`},{name:`adjusted`,view:s.output.adjustedPValues,type:`f32`,access:`read_write`}],invocationCount:y,declarations:C,body:`let p = pValues[pValuesOffset + index];
  let m = family[familyOffset + 1u];
  let value = ${u===`none`?`p + 0.0`:`min(multiplyBySpatialCount(p, m), 1.0)`};
  adjusted[adjustedOffset + index] = select(getNaN(), value, isTestedBits(bitcast<u32>(p)));`})),b;let M=e(n,`${o}-sorted-keys`,`uint32`,y),N=e(n,`${o}-sorted-rows`,`uint32`,y),P=Math.ceil(y/g),F=Math.ceil(P/g),I=e(n,`${o}-scanned`,`float32`,y),L=e(n,`${o}-block-totals`,`float32`,P),R=u===`holm`,z=!R,B=`${p}
const ROWS: u32 = ${y}u;
const BLOCK_COUNT: u32 = ${P}u;
const BLOCK_CHUNK: u32 = ${F}u;
const REVERSE: bool = ${z};
const IDENTITY: f32 = ${R?`0.0`:`3.0e38`};
var<workgroup> scanValues: array<f32, ${g}>;

fn combine(left: f32, right: f32) -> f32 {
  return ${R?`max(left, right)`:`min(left, right)`};
}

// Raw step value of sorted rank r (rank k = r + 1) given tested count, family size m and c(m);
// IDENTITY beyond the tested rows.
fn getStepValue(r: u32, key: u32, tested: u32, m: u32, scale: f32) -> f32 {
  if (r >= tested) {
    return IDENTITY;
  }
  let p = bitcast<f32>(key);
  ${R?`return multiplyBySpatialCount(p, m - r);`:`return divideBySpatialCount(multiplyBySpatialCount(p, m), r + 1u) * scale;`}
}

// Hillis-Steele inclusive scan of scanValues across the workgroup. Every invocation must call it.
fn scanWorkgroup(localIndex: u32) {
  workgroupBarrier();
  for (var offset = 1u; offset < ${g}u; offset = offset * 2u) {
    var other = IDENTITY;
    if (localIndex >= offset) {
      other = scanValues[localIndex - offset];
    }
    let combined = combine(scanValues[localIndex], other);
    workgroupBarrier();
    scanValues[localIndex] = combined;
    workgroupBarrier();
  }
}`;return b.push(...new r({id:`${o}-sort`,keys:O,values:k,outputKeys:M,outputValues:N,keyBits:v}).getCommandNodes(n),a(n,{id:`${o}-block-scan`,operation:h,variant:`block-scan`,bindings:[{name:`sortedKeys`,view:M,type:`u32`,access:`read`},{name:`family`,view:E,type:`u32`,access:`read`},{name:`familyScale`,view:D,type:`f32`,access:`read`},{name:`scanned`,view:I,type:`f32`,access:`read_write`},{name:`blockTotals`,view:L,type:`f32`,access:`read_write`}],invocationCount:y,workgroupSize:g,guardIndex:!1,declarations:B,body:`// Scan position j; the sorted rank it reads is r.
  let j = index;
  var value = IDENTITY;
  if (j < ROWS) {
    let r = select(j, ROWS - 1u - j, REVERSE);
    value = getStepValue(r, sortedKeys[sortedKeysOffset + r], family[familyOffset], family[familyOffset + 1u], familyScale[familyScaleOffset]);
  }
  scanValues[localInvocationIndex] = value;
  scanWorkgroup(localInvocationIndex);
  if (j < ROWS) {
    scanned[scannedOffset + j] = scanValues[localInvocationIndex];
  }
  if (localInvocationIndex == 255u) {
    blockTotals[blockTotalsOffset + workgroupIndex] = scanValues[localInvocationIndex];
  }`})),P>1&&b.push(a(n,{id:`${o}-total-scan`,operation:h,variant:`total-scan`,bindings:[{name:`blockTotals`,view:L,type:`f32`,access:`read_write`}],invocationCount:g,workgroupSize:g,guardIndex:!1,declarations:B,body:`let begin = min(localInvocationIndex * BLOCK_CHUNK, BLOCK_COUNT);
  let end = min(begin + BLOCK_CHUNK, BLOCK_COUNT);
  var running = IDENTITY;
  for (var block = begin; block < end; block++) {
    running = combine(running, blockTotals[blockTotalsOffset + block]);
    blockTotals[blockTotalsOffset + block] = running;
  }
  scanValues[localInvocationIndex] = running;
  scanWorkgroup(localInvocationIndex);
  var prefix = IDENTITY;
  if (localInvocationIndex > 0u) {
    prefix = scanValues[localInvocationIndex - 1u];
  }
  for (var block = begin; block < end; block++) {
    blockTotals[blockTotalsOffset + block] = combine(blockTotals[blockTotalsOffset + block], prefix);
  }`})),b.push(a(n,{id:`${o}-scatter`,operation:h,variant:`scatter`,bindings:[{name:`sortedRows`,view:N,type:`u32`,access:`read`},{name:`family`,view:E,type:`u32`,access:`read`},{name:`scanned`,view:I,type:`f32`,access:`read`},{name:`blockTotals`,view:L,type:`f32`,access:`read`},{name:`adjusted`,view:s.output.adjustedPValues,type:`f32`,access:`read_write`}],invocationCount:y,declarations:`${m}
const ROWS: u32 = ${y}u;
const REVERSE: bool = ${z};
fn combine(left: f32, right: f32) -> f32 {
  return ${R?`max(left, right)`:`min(left, right)`};
}`,body:`let j = select(index, ROWS - 1u - index, REVERSE);
  var value = scanned[scannedOffset + j];
  let block = j / ${g}u;
  if (block > 0u) {
    value = combine(value, blockTotals[blockTotalsOffset + block - 1u]);
  }
  adjusted[adjustedOffset + sortedRows[sortedRowsOffset + index]] =
    select(getNaN(), min(value, 1.0), index < family[familyOffset]);`})),b}};export{y as n,x as r,C as t};