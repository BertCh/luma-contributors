import{t as e}from"./common-CshkX-d_.js";import{o as t,u as n}from"./raster-band-utils-BcTxc8_Y.js";import{r,s as i}from"./kernel-C3tfQOn4.js";import{t as a}from"./gpu-scan-B4NNTfF9.js";import{o,t as s}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as c,f as l,h as u,i as d,m as f,n as p,o as m,p as h,r as g,s as _,u as v}from"./raster-algebra-input-5ZxymrFJ.js";import{f as y}from"./gpu-contributor-utils-BIwLQON5.js";import{a as b}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as x}from"./scaled-quotient-wgsl-UDyhQag6.js";import{n as S,r as C}from"./raster-fractional-tile-kernel-D2BjD3g1.js";import{i as w,t as T}from"./wgsl-common-DfyXiYB0.js";import{n as E}from"./gpu-raster-grid-C_izEAl0.js";import{t as D}from"./define-run-B-brkCe4.js";import{i as O}from"./inputs-CJKKIRKK.js";import{d as k,l as A,s as j,u as M}from"./pointer-overlay-Qh8qflcA.js";import{ia as N}from"./entry-preview-D70TdSbK.js";import{t as P}from"./bands-CVDIl0aE.js";var F=1<<24;function I(t,n){if(!Number.isFinite(t))throw Error(`getGPURasterStretchParameterValues window ${n} must be finite`);return e(Math.floor(t),0,F)}function L(e={},n=new Float32Array(16)){t(`getGPURasterStretchParameterValues`,n,16),n.fill(0,0,16);let r=e.window??[0,0,16777216,16777216];n[0]=I(r[0],`column0`),n[1]=I(r[1],`row0`),n[2]=I(r[2],`column1`),n[3]=I(r[3],`row1`);let i=e.domain??`auto`;if(i!==`auto`){let[e,t]=i;if(!Number.isFinite(e)||!Number.isFinite(t)||e>t)throw Error(`getGPURasterStretchParameterValues domain must be finite with min <= max`);if(!Number.isFinite(Math.fround(t-e))&&Number.isFinite(t-e))throw Error(`getGPURasterStretchParameterValues domain range must fit in float32`);n[4]=1,n[5]=e,n[6]=t}let a=e.mode??`linear`;if(a!==`linear`&&a!==`percentile`&&a!==`equalize`)throw Error(`getGPURasterStretchParameterValues mode ${String(a)} is not supported`);n[7]=a===`linear`?0:a===`percentile`?1:2;let[o,s]=e.percentiles??[2,98];if(!Number.isFinite(o)||!Number.isFinite(s)||o<0||s>100||o>s)throw Error(`getGPURasterStretchParameterValues percentiles must satisfy 0 <= low <= high <= 100`);n[8]=o/100,n[9]=s/100;let c=e.gamma??1;if(!Number.isFinite(c)||c<=0)throw Error(`getGPURasterStretchParameterValues gamma must be positive and finite`);n[10]=c;let l=e.sigmoidContrast??0;if(!Number.isFinite(l)||l<0||l>80)throw Error(`getGPURasterStretchParameterValues sigmoidContrast must be in [0, 80]`);n[11]=l;let u=e.sigmoidMidpoint??.5;if(!Number.isFinite(u)||u<=0||u>=1)throw Error(`getGPURasterStretchParameterValues sigmoidMidpoint must be in (0, 1)`);n[12]=u;let d=e.paletteInterpolation??`nearest`;if(d!==`nearest`&&d!==`linear`)throw Error(`getGPURasterStretchParameterValues paletteInterpolation ${String(d)} is not supported`);return n[13]=+(d===`linear`),n}var R=`GPURasterStretch`,z=1024,B=256,V=65536,H=256,U=16,W=`
${x}
fn getStretchDomainFraction(value: f32, lower: f32, upper: f32) -> f32 {
  if (!(upper > lower) || value <= lower) { return 0.0; }
  if (value >= upper) { return 1.0; }
  let exponent = frexp(max(abs(lower), abs(upper))).exp;
  let scaledLower = ldexp(lower, -exponent);
  let scaledUpper = ldexp(upper, -exponent);
  let scaledValue = ldexp(value, -exponent);
  return getScaledQuotient(scaledValue - scaledLower, scaledUpper - scaledLower, 1.0, 0);
}
fn getStretchDomainStep(lower: f32, upper: f32, divisor: f32) -> f32 {
  if (!(upper > lower)) { return 0.0; }
  let exponent = frexp(max(abs(lower), abs(upper))).exp;
  let scaledLower = ldexp(lower, -exponent);
  let scaledUpper = ldexp(upper, -exponent);
  return getScaledQuotient(scaledUpper - scaledLower, divisor, 1.0, exponent);
}
fn getStretchDomainValue(fraction: f32, lower: f32, upper: f32) -> f32 {
  if (fraction <= 0.0) { return lower; }
  if (fraction >= 1.0) { return upper; }
  let exponent = frexp(max(abs(lower), abs(upper))).exp;
  let scaledLower = ldexp(lower, -exponent);
  let scaledUpper = ldexp(upper, -exponent);
  return ldexp(mix(scaledLower, scaledUpper, fraction), exponent);
}
`,G=class{id;props;binCount;lutSize;constructor(e){this.id=e.id??`raster-stretch`,this.props=e;let{id:t}=this,{output:r}=e,i=n(t,e.grid);for(let[n,r]of[[`width`,e.grid.width],[`height`,e.grid.height]])y(t,`grid ${n}`,r,1,F);if(this.binCount=e.binCount??z,this.lutSize=e.lutSize??B,l(t,`binCount`,this.binCount,16,65536),l(t,`lutSize`,this.lutSize,2,65536),f(t,e.noDataValue),m(t,`values`,e.values,e.grid,i),u(t,`mask`,e.mask,`uint32`,i),u(t,`regionMask`,e.regionMask,`uint32`,i),u(t,`parameters`,e.parameters,`float32`,16),!e.parameters)throw Error(`${t} needs parameters`);if(u(t,`palette`,e.palette,`uint32`,1),e.palette&&e.palette.length>V)throw Error(`${t} palette must hold at most ${V} colors`);if(!r||Object.values(r).every(e=>e===void 0))throw Error(`${t} needs at least one output`);if((r.colors||r.lutColors)&&!e.palette)throw Error(`${t} output.colors and output.lutColors require palette`);if(u(t,`output.stretched`,r.stretched,`float32`,i),u(t,`output.colors`,r.colors,`uint32`,i),u(t,`output.lut`,r.lut,`float32`,this.lutSize),u(t,`output.lutColors`,r.lutColors,`uint32`,this.lutSize),u(t,`output.histogram`,r.histogram,`uint32`,this.binCount),r.histogram&&r.histogram.length!==this.binCount)throw Error(`${t} output.histogram must hold exactly binCount (${this.binCount}) rows`);u(t,`output.statistics`,r.statistics,`float32`,8),u(t,`output.validCount`,r.validCount,`uint32`,1);let a=r.colors||r.lutColors?e.palette:void 0,o=c(e.values)?void 0:e.values;C(t,r,[...p(e.values,i),e.mask,e.regionMask,e.parameters,a],[{writes:[r.stretched,r.colors],reads:[o,e.mask,e.parameters,r.colors?a:void 0]},{writes:[r.lut,r.lutColors],reads:[e.parameters,r.lutColors?a:void 0]},{writes:[r.statistics],reads:[e.parameters]},{writes:[r.validCount],reads:[e.parameters]},{writes:[r.histogram],reads:[o,e.mask,e.regionMask,e.parameters]}])}getCommandNodes(t){let{id:n,props:l,binCount:u,lutSize:f}=this,{output:p}=l,{width:m,height:y}=l.grid,x=m*y;_(t,n,[l.values],l.grid);let C=[l.mask,l.regionMask,l.parameters,...p.colors||p.lutColors?[l.palette]:[],...Object.values(p)];b(t,n,Object.fromEntries(C.map((e,t)=>[`view.${t}`,e])));for(let e of C)e&&S(t.device,n,e.buffer.id,i(e).size,`reduce the packed view length`,e.buffer.byteLength);for(let[e,r]of[[`canonical band`,c(l.values)?x*4:0],[`histogram/CDF`,u*4],[`statistics state`,40],[`statistics keys`,16]])S(t.device,n,e,r,`reduce grid dimensions or binCount`);h(n,t,[...d(l.values),l.mask,l.regionMask,l.parameters,p.colors||p.lutColors?l.palette:void 0,p.stretched,p.colors,p.lut,p.lutColors,p.histogram,p.statistics,p.validCount]);let E=g(t,`${n}-input`,l.values,l.grid),D=[...E.nodes],O=r(t,`${n}-keys`,`uint32`,4),k=r(t,`${n}-state`,`float32`,10),A=p.histogram??r(t,`${n}-histogram`,`uint32`,u),j=r(t,`${n}-cdf`,`uint32`,u),M=!!(p.stretched||p.colors),N=!!(p.lut||p.lutColors),P=!!(p.colors||p.lutColors),F=P?l.palette.length:0,I=u*4<=t.device.limits.maxComputeWorkgroupStorageSize,L=e(Math.ceil(u/32),8,64),z=[{name:`values`,view:E.values,type:`f32`,access:`read`}];l.mask&&z.push({name:`validity`,view:l.mask,type:`u32`,access:`read`});let B=[...z];l.regionMask&&B.push({name:`regionMask`,view:l.regionMask,type:`u32`,access:`read`});let V={name:`params`,view:l.parameters,type:`f32`,access:`read`},G=`const WIDTH: u32 = ${m}u;
const HEIGHT: u32 = ${y}u;
const BIN_COUNT: u32 = ${u}u;
const LUT_SIZE: u32 = ${f}u;
const PALETTE_SIZE: u32 = ${F}u;`,J=`${G}
${v(c(l.values)?void 0:l.noDataValue)}
fn readCellValue(index: u32, value: ptr<function, f32>) -> bool {
  let sample = values[valuesOffset + index];
  *value = sample;
  var isValid = !isNoDataValue(sample);
  ${l.mask?`isValid = isValid && validity[validityOffset + index] != 0u;`:``}
  return isValid;
}`,Y=`${J}
${W}
// Statistics include cells inside the window, in the region mask, valid and finite.
fn readStatisticsValue(index: u32, value: ptr<function, f32>) -> bool {
  let column = index % WIDTH;
  let row = index / WIDTH;
  let column0 = min(u32(params[paramsOffset]), WIDTH);
  let row0 = min(u32(params[paramsOffset + 1u]), HEIGHT);
  let column1 = min(u32(params[paramsOffset + 2u]), WIDTH);
  let row1 = min(u32(params[paramsOffset + 3u]), HEIGHT);
  if (column < column0 || column >= column1 || row < row0 || row >= row1) {
    return false;
  }
  ${l.regionMask?`if (regionMask[regionMaskOffset + index] == 0u) { return false; }`:``}
  var sample = 0.0;
  if (!readCellValue(index, &sample)) {
    return false;
  }
  *value = sample;
  return isFiniteValue(sample);
}`;D.push(o(t,{id:`${n}-clear-keys`,operation:R,variant:`clear-keys`,bindings:[{name:`keys`,view:O,type:`u32`,access:`read_write`}],invocationCount:1,body:`keys[keysOffset] = 0xffffffffu;
  keys[keysOffset + 1u] = 0u;
  keys[keysOffset + 2u] = 0u;
  keys[keysOffset + 3u] = 0u;`}),o(t,{id:`${n}-reduce`,operation:R,variant:`reduce`,bindings:[...B,V,{name:`keys`,view:O,type:`atomic<u32>`,access:`read_write`}],invocationCount:Math.ceil(x/U),guardIndex:!1,workgroupSize:H,declarations:`${Y}
const CELL_COUNT: u32 = ${x}u;
const CELLS_PER_THREAD: u32 = ${U}u;
var<workgroup> tileMinimum: array<u32, ${H}>;
var<workgroup> tileMaximum: array<u32, ${H}>;
var<workgroup> tileCount: array<u32, ${H}>;
${w}`,body:`var minimumKey = 0xffffffffu;
  var maximumKey = 0u;
  var includedCount = 0u;
  let tileStart = (index - localInvocationIndex) * CELLS_PER_THREAD;
  for (var step = 0u; step < CELLS_PER_THREAD; step++) {
    let cell = tileStart + step * ${H}u + localInvocationIndex;
    var value = 0.0;
    if (cell < CELL_COUNT) {
      if (readStatisticsValue(cell, &value)) {
        let key = getOrderedKey(value);
        minimumKey = min(minimumKey, key);
        maximumKey = max(maximumKey, key);
        includedCount++;
      }
    }
  }
  tileMinimum[localInvocationIndex] = minimumKey;
  tileMaximum[localInvocationIndex] = maximumKey;
  tileCount[localInvocationIndex] = includedCount;
  workgroupBarrier();
  for (var stride = ${H/2}u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      tileMinimum[localInvocationIndex] = min(tileMinimum[localInvocationIndex], tileMinimum[localInvocationIndex + stride]);
      tileMaximum[localInvocationIndex] = max(tileMaximum[localInvocationIndex], tileMaximum[localInvocationIndex + stride]);
      tileCount[localInvocationIndex] = tileCount[localInvocationIndex] + tileCount[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u && tileCount[0] > 0u) {
    atomicAdd(&keys[keysOffset + 2u], tileCount[0]);
    if (params[paramsOffset + 4u] == 0.0) {
      atomicMin(&keys[keysOffset], tileMinimum[0]);
      atomicMax(&keys[keysOffset + 1u], tileMaximum[0]);
    }
  }`}),o(t,{id:`${n}-domain`,operation:R,variant:`domain`,bindings:[V,{name:`keys`,view:O,type:`u32`,access:`read`},{name:`state`,view:k,type:`f32`,access:`read_write`},...p.validCount?[{name:`validCountOut`,view:p.validCount,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:`const BIN_COUNT: u32 = ${u}u;
${T}
${W}`,body:`let count = keys[keysOffset + 2u];
  var domainMinimum = 0.0;
  var domainMaximum = 0.0;
  if (params[paramsOffset + 4u] != 0.0) {
    domainMinimum = params[paramsOffset + 5u];
    domainMaximum = params[paramsOffset + 6u];
  } else if (count > 0u) {
    domainMinimum = decodeOrderedKey(keys[keysOffset]);
    domainMaximum = decodeOrderedKey(keys[keysOffset + 1u]);
  }
  state[stateOffset] = domainMinimum;
  state[stateOffset + 1u] = domainMaximum;
  // Do not store the reciprocal bin scale: it can overflow for normal tiny domains.
  state[stateOffset + 2u] = 0.0;
  state[stateOffset + 3u] = getStretchDomainStep(domainMinimum, domainMaximum, f32(BIN_COUNT));
  state[stateOffset + 4u] = domainMinimum;
  state[stateOffset + 5u] = domainMaximum;
  state[stateOffset + 6u] = 0.0;
  state[stateOffset + 7u] = 0.0;
  ${p.validCount?`validCountOut[validCountOutOffset] = count;`:``}`}),s(t,{id:`${n}-clear-histogram`,operation:R,view:A,type:`u32`,value:`0u`}),o(t,{id:`${n}-histogram`,operation:R,variant:I?`histogram-private`:`histogram`,bindings:[...B,V,{name:`state`,view:k,type:`f32`,access:`read`},{name:`histogram`,view:A,type:`atomic<u32>`,access:`read_write`}],invocationCount:I?Math.ceil(x/L):x,guardIndex:!I,workgroupSize:I?H:void 0,declarations:I?`${Y}
const CELL_COUNT: u32 = ${x}u;
const CELLS_PER_THREAD: u32 = ${L}u;
var<workgroup> localHistogram: array<atomic<u32>, ${u}>;`:Y,body:I?`for (var bin = localInvocationIndex; bin < BIN_COUNT; bin += ${H}u) {
    atomicStore(&localHistogram[bin], 0u);
  }
  workgroupBarrier();
  let domainMinimum = state[stateOffset];
  let domainMaximum = state[stateOffset + 1u];
  let tileStart = (index - localInvocationIndex) * CELLS_PER_THREAD;
  for (var step = 0u; step < CELLS_PER_THREAD; step++) {
    let cell = tileStart + step * ${H}u + localInvocationIndex;
    var value = 0.0;
    if (cell < CELL_COUNT) {
      if (readStatisticsValue(cell, &value) && value >= domainMinimum && value <= domainMaximum) {
        let position = getStretchDomainFraction(value, domainMinimum, domainMaximum) * f32(BIN_COUNT);
        let bin = min(u32(position), BIN_COUNT - 1u);
        atomicAdd(&localHistogram[bin], 1u);
      }
    }
  }
  workgroupBarrier();
  for (var bin = localInvocationIndex; bin < BIN_COUNT; bin += ${H}u) {
    let partial = atomicLoad(&localHistogram[bin]);
    if (partial != 0u) {
      atomicAdd(&histogram[histogramOffset + bin], partial);
    }
  }`:`var value = 0.0;
  if (!readStatisticsValue(index, &value)) {
    return;
  }
  let domainMinimum = state[stateOffset];
  let domainMaximum = state[stateOffset + 1u];
  if (value < domainMinimum || value > domainMaximum) {
    return;
  }
  let position = getStretchDomainFraction(value, domainMinimum, domainMaximum) * f32(BIN_COUNT);
  let bin = min(u32(position), BIN_COUNT - 1u);
  atomicAdd(&histogram[histogramOffset + bin], 1u);`}),...new a({id:`${n}-cdf-scan`,input:A,output:j,mode:`inclusive`}).getCommandNodes(t));let X=[V,{name:`state`,view:k,type:`f32`,access:`read_write`},{name:`keys`,view:O,type:`u32`,access:`read`},{name:`cdf`,view:j,type:`u32`,access:`read`}];if(p.statistics&&X.push({name:`statisticsOut`,view:p.statistics,type:`f32`,access:`read_write`}),D.push(o(t,{id:`${n}-finalize`,operation:R,variant:`finalize`,bindings:X,invocationCount:1,declarations:`${G}
${W}
${v(void 0)}
// Smallest bin whose cumulative count exceeds rank. The last bin holds the total, so any rank
// below the total is found.
fn getRankBin(rank: u32) -> u32 {
  var low = 0u;
  var high = BIN_COUNT - 1u;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high) / 2u;
    if (cdf[cdfOffset + middle] > rank) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  return low;
}
// The rank rule of the published GPURasterGlobalPercentile: the bin that holds order statistic
// floor((total - 1) * fraction), reported at its centre. With an automatic domain the fractions 0
// and 1 are the exact observed minimum and maximum. Needs total > 0.
fn getPercentileValue(fraction: f32, total: u32) -> f32 {
  let domainMinimum = state[stateOffset];
  let domainMaximum = state[stateOffset + 1u];
  let isAutomatic = params[paramsOffset + 4u] == 0.0;
  if (isAutomatic && fraction <= 0.0) {
    return domainMinimum;
  }
  if (isAutomatic && fraction >= 1.0) {
    return domainMaximum;
  }
  if (!(domainMaximum > domainMinimum)) {
    return domainMinimum;
  }
  let rank = min(u32(floor(f32(total - 1u) * fraction)), total - 1u);
  return getStretchDomainValue((f32(getRankBin(rank)) + 0.5) / f32(BIN_COUNT), domainMinimum, domainMaximum);
}`,body:`let domainMinimum = state[stateOffset];
  let domainMaximum = state[stateOffset + 1u];
  let validCount = keys[keysOffset + 2u];
  let total = cdf[cdfOffset + BIN_COUNT - 1u];
  let mode = u32(params[paramsOffset + 7u]);
  var lo = domainMinimum;
  var hi = domainMaximum;
  if (mode != 0u && total == 0u) {
    hi = lo;
  } else if (mode == 1u) {
    lo = getPercentileValue(params[paramsOffset + 8u], total);
    hi = getPercentileValue(params[paramsOffset + 9u], total);
  }
  // Cumulative count of the first occupied bin (rank 0 lives there); 0 without data.
  var firstCount = 0u;
  if (total > 0u) {
    firstCount = cdf[cdfOffset + getRankBin(0u)];
  }
  state[stateOffset + 4u] = lo;
  state[stateOffset + 5u] = hi;
  state[stateOffset + 6u] = 0.0;
  state[stateOffset + 8u] = f32(firstCount & 0xffffu);
  state[stateOffset + 9u] = f32(firstCount >> 16u);
  ${p.statistics?`let isEmpty = params[paramsOffset + 4u] == 0.0 && validCount == 0u;
  let nan = getNaN();
  statisticsOut[statisticsOutOffset] = select(domainMinimum, nan, isEmpty);
  statisticsOut[statisticsOutOffset + 1u] = select(domainMaximum, nan, isEmpty);
  statisticsOut[statisticsOutOffset + 2u] = select(lo, nan, isEmpty);
  statisticsOut[statisticsOutOffset + 3u] = select(hi, nan, isEmpty);
  statisticsOut[statisticsOutOffset + 4u] = f32(validCount);
  statisticsOut[statisticsOutOffset + 5u] = select(state[stateOffset + 3u], nan, isEmpty);
  statisticsOut[statisticsOutOffset + 6u] = 0.0;
  statisticsOut[statisticsOutOffset + 7u] = 0.0;`:``}`})),!M&&!N)return D;let Z=`${K()}
${P?q():``}`,Q=[V,{name:`state`,view:k,type:`f32`,access:`read`},{name:`cdf`,view:j,type:`u32`,access:`read`}],$=P?[{name:`palette`,view:l.palette,type:`u32`,access:`read`}]:[];if(N){let e=[...Q];p.lut&&e.push({name:`lutOut`,view:p.lut,type:`f32`,access:`read_write`}),p.lutColors&&e.push({name:`lutColorsOut`,view:p.lutColors,type:`u32`,access:`read_write`}),e.push(...$),D.push(o(t,{id:`${n}-lut`,operation:R,variant:`lut`,bindings:e,invocationCount:f,declarations:`${G}
${Z}`,body:`let lo = state[stateOffset + 4u];
  let hi = state[stateOffset + 5u];
  let value = getStretchDomainValue((f32(index) + 0.5) / f32(LUT_SIZE), lo, hi);
  let t = getStretchedValue(value);
  ${p.lut?`lutOut[lutOutOffset + index] = t;`:``}
  ${p.lutColors?`lutColorsOut[lutColorsOutOffset + index] = getPaletteColor(t);`:``}`}))}if(M){let e=[...z,...Q];p.stretched&&e.push({name:`stretchedOut`,view:p.stretched,type:`f32`,access:`read_write`}),p.colors&&e.push({name:`colorsOut`,view:p.colors,type:`u32`,access:`read_write`}),e.push(...$),D.push(o(t,{id:`${n}-apply`,operation:R,variant:`apply`,bindings:e,invocationCount:x,declarations:`${J}
${Z}`,body:`var value = 0.0;
  if (!readCellValue(index, &value)) {
    ${p.stretched?`stretchedOut[stretchedOutOffset + index] = getNaN();`:``}
    ${p.colors?`colorsOut[colorsOutOffset + index] = 0u;`:``}
    return;
  }
  let t = getStretchedValue(value);
  ${p.stretched?`stretchedOut[stretchedOutOffset + index] = t;`:``}
  ${p.colors?`colorsOut[colorsOutOffset + index] = getPaletteColor(t);`:``}`}))}return D}};function K(){return`
${W}
fn applyCurve(unit: f32) -> f32 {
  var t = unit;
  let gamma = params[paramsOffset + 10u];
  if (gamma != 1.0 && t > 0.0 && t < 1.0) {
    t = pow(t, 1.0 / gamma);
  }
  let contrast = params[paramsOffset + 11u];
  if (contrast > 0.0) {
    let midpoint = params[paramsOffset + 12u];
    let lowEnd = 1.0 / (1.0 + exp(contrast * midpoint));
    let highEnd = 1.0 / (1.0 + exp(contrast * (midpoint - 1.0)));
    let denominator = highEnd - lowEnd;
    if (denominator > 0.0) {
      t = (1.0 / (1.0 + exp(contrast * (midpoint - t))) - lowEnd) / denominator;
    }
    t = clamp(t, 0.0, 1.0);
  }
  return t;
}

fn getStretchedValue(value: f32) -> f32 {
  let lo = state[stateOffset + 4u];
  let hi = state[stateOffset + 5u];
  if (hi <= lo) {
    return select(select(0.5, 1.0, value > hi), 0.0, value < lo);
  }
  var unit = 0.0;
  if (u32(params[paramsOffset + 7u]) == 2u) {
    // The rule of the published GPURasterContrast (and OpenCV equalizeHist):
    // (cdf[bin] - cdfFirst) / (total - cdfFirst), where cdfFirst is the cumulative count of the
    // first occupied bin. It depends only on ranks, so three equally frequent values map to 0,
    // 0.5 and 1. When every counted cell is in one bin the mapping is the linear one.
    let domainMinimum = state[stateOffset];
    let domainMaximum = state[stateOffset + 1u];
    let total = cdf[cdfOffset + BIN_COUNT - 1u];
    let firstCount = u32(state[stateOffset + 8u]) | (u32(state[stateOffset + 9u]) << 16u);
    let cumulativeWidth = total - min(total, firstCount);
    if (cumulativeWidth > 0u) {
      // The bin the histogram counted this value in; values outside the domain use an end bin.
      let position = getStretchDomainFraction(value, domainMinimum, domainMaximum) * f32(BIN_COUNT);
      let cumulative = cdf[cdfOffset + min(u32(position), BIN_COUNT - 1u)];
      unit = f32(cumulative - min(cumulative, firstCount)) / f32(cumulativeWidth);
    } else {
      unit = getStretchDomainFraction(value, domainMinimum, domainMaximum);
    }
  } else {
    unit = getStretchDomainFraction(value, lo, hi);
  }
  return applyCurve(unit);
}`}function q(){return`
fn unpackColor(color: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(color & 255u),
    f32((color >> 8u) & 255u),
    f32((color >> 16u) & 255u),
    f32((color >> 24u) & 255u)
  );
}

fn getPaletteColor(t: f32) -> u32 {
  if (params[paramsOffset + 13u] == 0.0 || PALETTE_SIZE == 1u) {
    let index = min(u32(t * f32(PALETTE_SIZE)), PALETTE_SIZE - 1u);
    return palette[paletteOffset + index];
  }
  let position = t * f32(PALETTE_SIZE - 1u);
  let lower = min(u32(position), PALETTE_SIZE - 2u);
  let fraction = position - f32(lower);
  let first = unpackColor(palette[paletteOffset + lower]);
  let second = unpackColor(palette[paletteOffset + lower + 1u]);
  let channels = vec4<u32>(clamp(floor(mix(first, second, fraction) + 0.5), vec4<f32>(0.0), vec4<f32>(255.0)));
  return channels.x | (channels.y << 8u) | (channels.z << 16u) | (channels.w << 24u);
}`}var J=[`linear`,`percentile`,`equalize`],Y=128,X=.16,Z=j(N,`percentileLow`),Q=j(N,`percentileHigh`),$=j(N,`gamma`),ee=D({parameters:N,caption:`A flat-looking raster in (left strip); the contrast-stretched raster and its curve against the identity line out.`,createInputs:({random:e,world:t})=>{let n=O(t,64),r=n.values.map(e=>.3+.25*e);for(let t=0;t<12;t++)r[Math.floor(e()*r.length)]=1;return{...n,dull:r}},build(e,t){let n=e.upload(`source`,`float32`,t.dull),r=e.allocate(`parameters`,`float32`,16),i=e.allocate(`stretched`,`float32`,t.dull.length),a=e.allocate(`lut`,`float32`,Y),o=e.upload(`identity`,`float32`,Float32Array.from({length:Y},(e,t)=>t/127));return e.graph.add(new G({values:n.view,grid:E({width:t.columns,height:t.rows}),parameters:r.view,binCount:256,lutSize:Y,output:{stretched:i.view,lut:a.view}})),{outputs:{source:n,stretched:i,lut:a,identity:o},update:e=>r.write(L({mode:J[Math.round(e.mode)],percentiles:[e.percentileLow,Math.max(e.percentileHigh,e.percentileLow)],gamma:e.gamma}))}},interaction:{kind:`level`,hint:`clip tails and gamma`,rest:()=>[A(Z,Z.default),1-A($,$.default,`logarithmic`)],resolve:(e,{world:t,values:n})=>{let r=M(Z,e.unitX),i=M($,1-e.unitY,`logarithmic`),a=k(Q,100-r);return{values:{percentileLow:r,percentileHigh:a,gamma:i},glyphs:[{shape:`crosshair`,at:[e.x,e.y]}],readout:`percentiles [${r}, ${a}] | gamma ${i.toFixed(1)}${n.mode===1?``:` (percentile mode only)`}`}}},draw:({inputs:e,outputs:t,world:n})=>{let r=n.width*X;return[P(e,t.source,{range:[0,1],clip:{side:`left`,x:r}}),P(e,t.stretched,{range:[0,1],ramp:`sequential`,clip:{side:`right`,x:r}}),{layer:`chart`,kind:`line`,context:!0,values:t.identity,range:[0,1],rectangle:[.62,.05,.96,.4]},{layer:`chart`,kind:`line`,values:t.lut,range:[0,1],rectangle:[.62,.05,.96,.4]}]}});export{ee as default};