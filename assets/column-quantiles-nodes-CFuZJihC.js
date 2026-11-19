import{t as e}from"./classification-kernel-node-C62eCEbJ.js";import{t}from"./common-CshkX-d_.js";import{r as n}from"./kernel-C3tfQOn4.js";import{c as r,f as i}from"./gpu-contributor-utils-BIwLQON5.js";import{t as a}from"./uint64-wgsl-DCuW6-R6.js";import{t as o}from"./column-classification-shared-BTIVlxdJ.js";import{n as s,t as c}from"./order-statistic-wgsl-BrLguFSK.js";import{a as l,t as u}from"./column-classification-rows-D2hiB8L9.js";var d={lower:0,higher:1,nearest:2,linear:3,midpoint:4};function f(e){return 4+e}function p(e,t){let n=e.quantileCount??e.quantiles.length;if(i(`GPUColumnQuantiles`,`quantileCount`,n,1,64),e.quantiles.length>n)throw Error(`GPUColumnQuantiles probabilities exceed quantileCount`);let r=f(n),a=t??new Float32Array(r);if(a.length<r)throw Error(`GPUColumnQuantiles parameter target must hold ${r} elements`);let o=e.interpolation??`linear`,s=d[o];if(s===void 0)throw Error(`GPUColumnQuantiles unknown interpolation ${String(o)}`);let[c,l]=e.filterRange??[0,1];a[0]=s,a[1]=c,a[2]=l,a[3]=0;for(let t=0;t<n;t++)a[4+t]=t<e.quantiles.length?e.quantiles[t]:NaN;return a}function m(e){return 2*e}var h=256,g=256,_=4,v=8,y=16,b=256;function x(i,d){let{id:f,operation:p,values:m,mask:x,parameters:S,quantileCount:C,output:w,gate:T}=d,{rationalProbabilities:E}=d;u(f,i,{values:m,mask:x,parameters:S,rationalProbabilities:E,gate:T?.view,"output.quantiles":w.quantiles,"output.validCount":w.validCount,"output.outputMask":w.outputMask,"output.filterBounds":w.filterBounds});let D=r(m),O=x?r(x):void 0,k=!!(w.outputMask||w.filterBounds),A=2*C+(k?2:0);l(i,f,{histogram:A*g,state:4*A+2});let j=n(i,`${f}-histogram`,`uint32`,A*g),M=4*A+2,N=n(i,`${f}-state`,`uint32`,M),P=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),F=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),I=T?[P(`gate`,T.view,`u32`)]:[],L=E?[P(`rationalProbabilities`,E,`u32`)]:[],R=T?`gate[gateOffset] == ${T.value>>>0}u`:`true`,z=`
const QUANTILE_COUNT: u32 = ${C}u;
const TARGET_COUNT: u32 = ${A}u;
const HEADER_LENGTH: u32 = 4u;
const STATE_PREFIX: u32 = 0u;
const STATE_RANK: u32 = ${A}u;
const STATE_SLOT: u32 = ${2*A}u;
const STATE_ACTIVE_PREFIX: u32 = ${3*A}u;
const STATE_ACTIVE_COUNT: u32 = ${4*A}u;
const STATE_VALID_COUNT: u32 = ${4*A+1}u;
${o}
${c}${E?`\n${a}\n${s}`:``}

fn isMissingValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u;
}
`,B=`
// Round half to even; the fraction h - floor(h) is exact in f32.
fn roundHalfEven(h: f32) -> f32 {
  let whole = floor(h);
  let fraction = h - whole;
  if (fraction < 0.5) {
    return whole;
  }
  if (fraction > 0.5) {
    return whole + 1.0;
  }
  return select(whole, whole + 1.0, (u32(whole) & 1u) == 1u);
}

// The product is correctly rounded in WGSL, so the CPU oracle matches it with Math.fround.
fn getQuantileHeight(validCount: u32, p: f32) -> f32 {
  return f32(validCount - 1u) * p;
}

fn getQuantileRanks(validCount: u32, p: f32) -> vec2<u32> {
  let h = getQuantileHeight(validCount, p);
  let lower = floor(h);
  let upper = ceil(h);
  let code = u32(params[paramsOffset]);
  var first = lower;
  var second = lower;
  if (code == 1u) {
    first = upper;
    second = upper;
  } else if (code == 2u) {
    first = roundHalfEven(h);
    second = first;
  } else if (code == 3u) {
    second = min(lower + 1.0, f32(validCount - 1u));
  } else if (code == 4u) {
    second = upper;
  }
  return vec2<u32>(u32(first), u32(second));
}${E?`

// The ranks of a probability numerator / denominator, exact at any row count.
fn getRationalQuantileRanks(validCount: u32, numerator: u32, denominator: u32) -> vec2<u32> {
  let rank = getRationalOrderStatisticRank(validCount, numerator, denominator);
  let upper = getRationalOrderStatisticUpperRank(rank);
  let code = u32(params[paramsOffset]);
  var first = rank.lower;
  var second = rank.lower;
  if (code == 1u) {
    first = upper;
    second = upper;
  } else if (code == 2u) {
    first = getRationalOrderStatisticNearestRank(rank, denominator);
    second = first;
  } else if (code == 3u) {
    second = min(rank.lower + 1u, validCount - 1u);
  } else if (code == 4u) {
    second = upper;
  }
  return vec2<u32>(first, second);
}`:``}

fn getTargetRank(targetIndex: u32, validCount: u32) -> u32 {
  if (validCount == 0u) {
    return 0u;
  }
  if (targetIndex < 2u * QUANTILE_COUNT) {
    ${E?`let slot = rationalProbabilitiesOffset + 2u * (targetIndex / 2u);
    let numerator = rationalProbabilities[slot];
    let denominator = rationalProbabilities[slot + 1u];
    if (!isRationalOrderStatisticProbability(numerator, denominator)) {
      return 0u;
    }
    let ranks = getRationalQuantileRanks(validCount, numerator, denominator);`:`let p = params[paramsOffset + HEADER_LENGTH + targetIndex / 2u];
    if (!isOrderStatisticProbability(p)) {
      return 0u;
    }
    let ranks = getQuantileRanks(validCount, p);`}
    return select(ranks.x, ranks.y, (targetIndex & 1u) == 1u);
  }
  let lastIndex = f32(validCount - 1u);
  if (targetIndex == 2u * QUANTILE_COUNT) {
    let lowerFraction = params[paramsOffset + 1u];
    let fraction = select(clamp(lowerFraction, 0.0, 1.0), 0.0, isNanBits(lowerFraction));
    return u32(clamp(floor(f32(validCount) * fraction), 0.0, lastIndex));
  }
  let upperFraction = params[paramsOffset + 2u];
  let fraction = select(clamp(upperFraction, 0.0, 1.0), 1.0, isNanBits(upperFraction));
  return u32(clamp(ceil(f32(validCount) * fraction) - 1.0, 0.0, lastIndex));
}
`,V=[];V.push(e(i,{id:`${f}-init`,operation:p,variant:`quantile-init`,bindings:[F(`histogram`,j),...I],invocationCount:j.length,body:`if (${R}) {
    histogram[histogramOffset + index] = 0u;
  }`}));for(let n=0;n<_;n++){let r=v*g;for(let a=0;a<D.length;a++){let o=D[a],s=O?.[a],c=o.length;if(c===0)continue;let l=t(Math.ceil(c/(h*y)),1,b);V.push(e(i,{id:D.length===1?`${f}-histogram-${n}`:`${f}-histogram-${n}-chunk-${a}`,operation:p,variant:`quantile-histogram-${n}`,parameterValues:[n],bindings:[P(`values`,o,`f32`),...s?[P(`rowMask`,s,`u32`)]:[],P(`state`,N,`u32`),F(`histogram`,j,`atomic<u32>`),...I],invocationCount:l*h,workgroupSize:h,workgroupCollective:!0,declarations:`${z}
const LOCAL_SLOT_COUNT: u32 = ${v}u;
const LOCAL_BIN_COUNT: u32 = ${r}u;
var<workgroup> localHistogram: array<atomic<u32>, ${r}>;`,body:`for (var bin = localInvocationIndex; bin < LOCAL_BIN_COUNT; bin = bin + ${h}u) {
    atomicStore(&localHistogram[bin], 0u);
  }
  workgroupBarrier();
  let isEnabled = ${R};
  if (isEnabled) {
    let phase = getChunkParameter(36u);
    let shift = 24u - 8u * phase;
    let activeCount = select(1u, state[stateOffset + STATE_ACTIVE_COUNT], phase > 0u);
    let rowCount = arrayLength(&values) - valuesOffset;
    for (var row = index; row < rowCount; row = row + getChunkParameter(1u)) {
      let value = values[valuesOffset + row];
      if (isMissingValue(value)) {
        continue;
      }
      ${s?`if (rowMask[rowMaskOffset + row] == 0u) {
        continue;
      }`:``}
      let key = getOrderStatisticKey(value);
      var slot = 0u;
      if (phase > 0u) {
        let prefixValue = key >> (shift + 8u);
        var low = 0u;
        var high = activeCount;
        while (low < high) {
          let middle = (low + high) / 2u;
          if (state[stateOffset + STATE_ACTIVE_PREFIX + middle] < prefixValue) { low = middle + 1u; }
          else { high = middle; }
        }
        if (low >= activeCount || state[stateOffset + STATE_ACTIVE_PREFIX + low] != prefixValue) { continue; }
        slot = low;
      }
      let digit = (key >> shift) & 255u;
      if (slot < LOCAL_SLOT_COUNT) {
        atomicAdd(&localHistogram[slot * 256u + digit], 1u);
      } else {
        atomicAdd(&histogram[histogramOffset + slot * 256u + digit], 1u);
      }
    }
  }
  workgroupBarrier();
  if (isEnabled) {
    for (var bin = localInvocationIndex; bin < LOCAL_BIN_COUNT; bin = bin + ${h}u) {
      let count = atomicLoad(&localHistogram[bin]);
      if (count != 0u && bin < ${A*g}u) {
        atomicAdd(&histogram[histogramOffset + bin], count);
      }
    }
  }`}))}V.push(e(i,{id:`${f}-select-${n}`,operation:p,variant:`quantile-select-${n}${E?`-rational`:``}`,parameterValues:[n],bindings:[F(`histogram`,j),F(`state`,N),P(`params`,S,`f32`),...L,...I],invocationCount:h,workgroupSize:h,workgroupCollective:!0,declarations:`${z}
${B}
var<workgroup> sharedPrefix: array<u32, ${A}>;
var<workgroup> sharedFirst: array<u32, ${A}>;
var<workgroup> sharedValidCount: u32;`,body:`let isEnabled = ${R};
  let phase = getChunkParameter(36u);
  let targetIndex = localInvocationIndex;
  if (phase == 0u && targetIndex == 0u) {
    var total = 0u;
    if (isEnabled) {
      for (var bin = 0u; bin < 256u; bin++) { total += histogram[histogramOffset + bin]; }
      state[stateOffset + STATE_VALID_COUNT] = total;
    }
    sharedValidCount = total;
  }
  workgroupBarrier();
  if (isEnabled && targetIndex < TARGET_COUNT) {
    var prefix = 0u;
    var remaining = 0u;
    var slot = 0u;
    if (phase == 0u) { remaining = getTargetRank(targetIndex, sharedValidCount); }
    else {
      prefix = state[stateOffset + STATE_PREFIX + targetIndex];
      remaining = state[stateOffset + STATE_RANK + targetIndex];
      slot = state[stateOffset + STATE_SLOT + targetIndex];
    }
    var digit = 255u;
    for (var bin = 0u; bin < 256u; bin++) {
      let count = histogram[histogramOffset + slot * 256u + bin];
      if (remaining < count) { digit = bin; break; }
      remaining -= count;
    }
    prefix = (prefix << 8u) | digit;
    sharedPrefix[targetIndex] = prefix;
    state[stateOffset + STATE_PREFIX + targetIndex] = prefix;
    state[stateOffset + STATE_RANK + targetIndex] = remaining;
  }
  workgroupBarrier();
  if (isEnabled) {
    for (var cell = targetIndex; cell < ${A*g}u; cell += ${h}u) {
      histogram[histogramOffset + cell] = 0u;
    }
  }
  if (phase < 3u && isEnabled && targetIndex < TARGET_COUNT) {
    var first = 1u;
    for (var other = 0u; other < targetIndex; other++) {
      if (sharedPrefix[other] == sharedPrefix[targetIndex]) { first = 0u; break; }
    }
    sharedFirst[targetIndex] = first;
  }
  workgroupBarrier();
  if (phase < 3u && isEnabled && targetIndex < TARGET_COUNT) {
    let prefix = sharedPrefix[targetIndex];
    var slot = 0u;
    for (var other = 0u; other < TARGET_COUNT; other++) {
      if (sharedFirst[other] == 1u && sharedPrefix[other] < prefix) { slot++; }
    }
    state[stateOffset + STATE_SLOT + targetIndex] = slot;
    if (sharedFirst[targetIndex] == 1u) { state[stateOffset + STATE_ACTIVE_PREFIX + slot] = prefix; }
    if (targetIndex == 0u) {
      var distinct = 0u;
      for (var other = 0u; other < TARGET_COUNT; other++) { distinct += sharedFirst[other]; }
      state[stateOffset + STATE_ACTIVE_COUNT] = distinct;
    }
  }`}))}let H=[P(`state`,N,`u32`),P(`params`,S,`f32`),...L,F(`quantilesOut`,w.quantiles,`f32`),F(`validCountOut`,w.validCount,`u32`),...w.filterBounds?[F(`filterBoundsOut`,w.filterBounds,`f32`)]:[],...I];if(V.push(e(i,{id:`${f}-finish`,operation:p,variant:E?`quantile-finish-rational`:`quantile-finish`,bindings:H,invocationCount:C+1,declarations:`${z}
${B}`,body:`if (!(${R})) {
    return;
  }
  let validCount = state[stateOffset + STATE_VALID_COUNT];
  if (index < QUANTILE_COUNT) {
    var result = getNaN();
    ${E?`let numerator = rationalProbabilities[rationalProbabilitiesOffset + 2u * index];
    let denominator = rationalProbabilities[rationalProbabilitiesOffset + 2u * index + 1u];
    if (validCount > 0u && isRationalOrderStatisticProbability(numerator, denominator)) {`:`let p = params[paramsOffset + HEADER_LENGTH + index];
    if (validCount > 0u && isOrderStatisticProbability(p)) {`}
      let first = decodeOrderedKey(state[stateOffset + STATE_PREFIX + 2u * index]);
      let second = decodeOrderedKey(state[stateOffset + STATE_PREFIX + 2u * index + 1u]);
      result = first;
      let code = u32(params[paramsOffset]);
      // validCount is below ${E?`2^31`:`2^24`}, so this is a runtime zero that blocks multiply-add fusion.
      let zero = validCount >> 31u;
      if (code == 3u) {
        ${E?`let rank = getRationalOrderStatisticRank(validCount, numerator, denominator);
        result = interpolateOrderStatistic(first, second, getRationalOrderStatisticFraction(rank.remainder, denominator), zero);`:`let h = getQuantileHeight(validCount, p);
        result = interpolateOrderStatistic(first, second, h - floor(h), zero);`}
      } else if (code == 4u) {
        result = getOrderStatisticMidpoint(first, second, zero);
      }
    }
    quantilesOut[quantilesOutOffset + index] = result;
  } else {
    validCountOut[validCountOutOffset] = validCount;
    ${w.filterBounds?`let hasRows = validCount > 0u;
    let lowerBound = decodeOrderedKey(state[stateOffset + STATE_PREFIX + 2u * QUANTILE_COUNT]);
    let upperBound = decodeOrderedKey(state[stateOffset + STATE_PREFIX + 2u * QUANTILE_COUNT + 1u]);
    filterBoundsOut[filterBoundsOutOffset] = select(getNaN(), lowerBound, hasRows);
    filterBoundsOut[filterBoundsOutOffset + 1u] = select(getNaN(), upperBound, hasRows);`:``}
  }`})),w.outputMask){let t=r(w.outputMask);for(let n=0;n<D.length;n++){let r=D[n],a=O?.[n],o=r.length;o!==0&&V.push(e(i,{id:D.length===1?`${f}-filter-mask`:`${f}-filter-mask-chunk-${n}`,operation:p,variant:`quantile-filter-mask`,bindings:[P(`values`,r,`f32`),...a?[P(`rowMask`,a,`u32`)]:[],P(`state`,N,`u32`),F(`filterMaskOut`,t[n],`u32`),...I],invocationCount:o,declarations:z,body:`if (!(${R})) {
    return;
  }
  var inside = 0u;
  let value = values[valuesOffset + index];
  if (state[stateOffset + STATE_VALID_COUNT] > 0u && !isMissingValue(value)) {
    ${a?`if (rowMask[rowMaskOffset + index] != 0u) {`:`{`}
      let key = getOrderStatisticKey(value);
      let lowerKey = state[stateOffset + STATE_PREFIX + 2u * QUANTILE_COUNT];
      let upperKey = state[stateOffset + STATE_PREFIX + 2u * QUANTILE_COUNT + 1u];
      inside = select(0u, 1u, key >= lowerKey && key <= upperKey);
    }
  }
  filterMaskOut[filterMaskOutOffset + index] = inside;`}))}}return V}export{m as i,f as n,p as r,x as t};