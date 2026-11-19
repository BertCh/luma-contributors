import{c as e,o as t,r as n,s as r,t as i}from"./kernel-C3tfQOn4.js";import{l as a,r as o}from"./gpu-command-graph-BdzdmV6u.js";import{Q as s}from"./dist-BusamGlx.js";import{t as c}from"./gpu-dispatch-utils-CbRzFAgT.js";import{E as l,S as u,T as d,b as f,c as p,f as m,p as h,u as g,v as _,w as v,x as y,y as b}from"./gpu-contributor-utils-BIwLQON5.js";import{a as x,i as S}from"./graph-chunk-dispatch-zBt8k2-o.js";import{i as C,n as w,r as T}from"./cell-key-wgsl-Cdgu1snt.js";import{a as E,o as D,t as O}from"./fixed-point-accumulator-Ce6W3HVY.js";import{t as k}from"./gpu-scan-B4NNTfF9.js";import{t as A}from"./gpu-sort-xRm8l5yn.js";import{t as j}from"./gpu-uint32-gather-C7ioZBaz.js";import{t as M}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{n as ee}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as N}from"./order-statistic-wgsl-BrLguFSK.js";import{n as P,r as F,t as I}from"./tiled-key-union-BSiaYSH6.js";var te=8,L=256,ne=2**24;function R(n,r){u(r.id,`kernel storage bindings`,r.bindings.length,te);let a=c(r.operation,Math.max(r.invocationCount,1),L,n.device.limits.maxComputeWorkgroupsPerDimension),l=new Uint32Array(16);l.set([r.invocationCount,a.x,a.y,r.rowOffset??0]),l[12]=r.halfWidth??0;let d=new Map([[`INVOCATION_COUNT`,`groupParameters[0].x`],[`GROUP_ROW_OFFSET`,`groupParameters[0].w`],[`GROUP_HALF_WIDTH`,`groupParameters[3].x`]]);for(let[t,n]of r.bindings.entries())l[4+t]=e(n.view),d.set(`${n.name}Offset`,`groupParameters[${1+Math.floor(t/4)}][${t%4}]`);let f=e=>e.replace(/\b[A-Za-z_][A-Za-z_0-9]*\b/g,e=>d.has(e)?`(${d.get(e)})`:e),p=`
${r.bindings.map((e,t)=>`@group(0) @binding(${t}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;`).join(`
`)}
@group(0) @binding(${r.bindings.length}) var<uniform> groupParameters: array<vec4u, 4>;
${f(r.declarations??``)}
@compute @workgroup_size(${L})
fn main(@builtin(local_invocation_index) localInvocationIndex: u32,
        @builtin(workgroup_id) workgroupId: vec3u) {
  ${r.bindings.map(e=>`_ = &${e.name};`).join(`
  `)}
  let workgroupIndex = (workgroupId.z * groupParameters[0].z + workgroupId.y) * groupParameters[0].y + workgroupId.x;
  if (workgroupIndex >= ${ne}u) { return; }
  let index = workgroupIndex * ${L}u + localInvocationIndex;
  ${r.guardIndex===!1?``:`if (index >= groupParameters[0].x) { return; }`}
  ${f(r.body)}
}`;return o({id:r.id,resources:r.bindings.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`})),workload:{operation:r.operation,...r.variant?{variant:r.variant}:{},commandCount:1,maximumInvocationCount:r.invocationCount,maximumWorkgroupCount:a.x*a.y*a.z},compile:({device:e})=>{let n=e.createBuffer({id:`${r.id}-parameters`,usage:s.UNIFORM|s.COPY_DST,data:l}),o=new i(e,{id:r.id,source:p,shaderLayout:{bindings:[...r.bindings.map((e,t)=>({name:e.name,type:e.access===`read`?`read-only-storage`:`storage`,group:0,location:t})),{name:`groupParameters`,type:`uniform`,group:0,location:r.bindings.length}]}});return{encode:({computePass:e,getBuffer:i})=>{let s={groupParameters:n};for(let e of r.bindings)s[e.name]=t(e.view,i);o.dispatch(e,{bindings:s,x:a.x,y:a.y,z:a.z})},destroy:()=>{o.destroy(),n.destroy()}}}})}var z=4294967295;function B(e,t,n=`u32`){return{name:e,view:t,type:n,access:`read`}}function V(e,t,n=`u32`){return{name:e,view:t,type:n,access:`read_write`}}function H(e,t){return{name:e,view:t,type:`atomic<u32>`,access:`read_write`}}var U=`
fn isFiniteBits(bits: u32) -> bool {
  return (bits & 0x7f800000u) != 0x7f800000u;
}

fn getValueKey(rawBits: u32) -> u32 {
  if (!isFiniteBits(rawBits)) {
    return 0xffffffffu;
  }
  var bits = rawBits;
  if ((bits & 0x7fffffffu) == 0u) {
    bits = 0u;
  }
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
`;function W(e,t){let{id:r,operation:i,structure:a,valueChunks:o,means:s,output:c}=t,{rowCount:l,positionGroups:u,permutation:d,offsets:f}=a,p=Array.from({length:4},(t,i)=>n(e,`${r}-moment-power-${i+1}`,`float32`,l)),m=[],h=n(e,`${r}-source-positions`,`uint32`,l);m.push(R(e,{id:`${r}-source-positions`,operation:i,variant:`source-positions`,bindings:[B(`permutation`,d),V(`sourcePositions`,h)],invocationCount:l,body:`sourcePositions[sourcePositionsOffset + permutation[permutationOffset + index]] = index;`}));for(let[t,{valueBits:n,rowOffset:c}]of o.entries())m.push(R(e,{id:o.length===1?`${r}-moment-terms`:`${r}-moment-terms-${t}`,operation:i,variant:`moment-terms`,bindings:[B(`rowGroups`,a.rowGroups),B(`sourcePositions`,h),B(`valueBits`,n),B(`means`,s,`f32`),...p.map((e,t)=>V(`power${t+1}`,e,`f32`))],invocationCount:n.length,rowOffset:c,declarations:U,body:`let sourceRow = index + GROUP_ROW_OFFSET;
  let position = sourcePositions[sourcePositionsOffset + sourceRow];
  let group = rowGroups[rowGroupsOffset + sourceRow];
  var deviation = 0.0;
  var scaledDeviation = 0.0;
  if (group != ${z}u) {
    let bits = valueBits[valueBitsOffset + index];
    if (isFiniteBits(bits)) {
      let value = bitcast<f32>(bits);
      let mean = means[meansOffset + 2u * group];
      let scaleExponent = i32(means[meansOffset + 2u * group + 1u]);
      deviation = value - mean;
      scaledDeviation = deviation;
      if (scaleExponent > 0) {
        // Scale before subtracting or cubing: either unscaled operation could overflow.
        scaledDeviation = ldexp(value, -scaleExponent) - ldexp(mean, -scaleExponent);
      }
    }
  }
  let squared = deviation * deviation;
  power1[power1Offset + position] = scaledDeviation;
  power2[power2Offset + position] = squared;
  power3[power3Offset + position] = scaledDeviation * scaledDeviation * scaledDeviation;
  power4[power4Offset + position] = squared * squared;`}));for(let t=1;t<l;t*=2)m.push(R(e,{id:`${r}-moment-merge-${t}`,operation:i,variant:`moment-merge`,bindings:[B(`positionGroups`,u),B(`offsets`,f),...p.map((e,t)=>V(`power${t+1}`,e,`f32`))],invocationCount:Math.ceil(l/(2*t)),halfWidth:t,body:`let blockBegin = index * GROUP_HALF_WIDTH * 2u;
  let midpoint = blockBegin + GROUP_HALF_WIDTH;
  if (midpoint >= arrayLength(&positionGroups) - positionGroupsOffset) {
    return;
  }
  let group = positionGroups[positionGroupsOffset + midpoint];
  if (group == ${z}u) {
    return;
  }
  let groupBegin = offsets[offsetsOffset + group];
  if (groupBegin >= midpoint) {
    return;
  }
  let root = max(groupBegin, blockBegin);
  ${p.map((e,t)=>{let n=`power${t+1}`;return`${n}[${n}Offset + root] += ${n}[${n}Offset + midpoint];`}).join(`
  `)}`}));return m.push(R(e,{id:`${r}-moments`,operation:i,variant:`moments`,bindings:[B(`positionGroups`,u),B(`means`,s,`f32`),...p.map((e,t)=>B(`power${t+1}`,e,`f32`)),V(`momentSums`,c,`f32`)],invocationCount:l,body:`let group = positionGroups[positionGroupsOffset + index];
  if (group == ${z}u) {
    return;
  }
  if (index > 0u && positionGroups[positionGroupsOffset + index - 1u] == group) {
    return;
  }
  let scaleExponent = i32(means[meansOffset + 2u * group + 1u]);
  momentSums[momentSumsOffset + 4u * group] = ${t.retainScaledOddMoments?`power1[power1Offset + index]`:`ldexp(power1[power1Offset + index], scaleExponent)`};
  momentSums[momentSumsOffset + 4u * group + 1u] = power2[power2Offset + index];
  // Three bounded exponent steps also handle a zero sum when 3 * scaleExponent exceeds 127.
  momentSums[momentSumsOffset + 4u * group + 2u] = ${t.retainScaledOddMoments?`power3[power3Offset + index]`:`ldexp(ldexp(ldexp(power3[power3Offset + index], scaleExponent), scaleExponent), scaleExponent)`};
  momentSums[momentSumsOffset + 4u * group + 3u] = power4[power4Offset + index];`})),m}var re=30;function G(e){return`
const SAMPLE_VARIANCE: bool = ${e===`sample`};

fn getMomentM2(count: u32, sums: vec4f) -> f32 {
  let n = f32(count);
  let delta = sums.x / n;
  var m2 = sums.y - sums.x * delta;
  if (m2 <= 1.9073486e-6 * sums.y) {
    m2 = 0.0;
  }
  return max(m2, 0.0);
}

fn getMomentVariance(count: u32, m2: f32) -> f32 {
  if (SAMPLE_VARIANCE) {
    return select(getNaN(), m2 / f32(count - 1u), count >= 2u);
  }
  return m2 / f32(count);
}

fn getMomentSkewness(count: u32, sums: vec4f, m2: f32) -> f32 {
  if (m2 == 0.0) {
    return getNaN();
  }
  let n = f32(count);
  let delta = sums.x / n;
  let m3 = sums.z - 3.0 * delta * sums.y + 2.0 * n * delta * delta * delta;
  let scale = m2 / n;
  return (m3 / n) / (scale * sqrt(scale));
}

fn getMomentKurtosis(count: u32, sums: vec4f, m2: f32) -> f32 {
  if (m2 == 0.0) {
    return getNaN();
  }
  let n = f32(count);
  let delta = sums.x / n;
  let delta2 = delta * delta;
  let m4 = sums.w - 4.0 * delta * sums.z + 6.0 * delta2 * sums.y - 3.0 * n * delta2 * delta2;
  let scale = m2 / n;
  return (m4 / n) / (scale * scale) - 3.0;
}
`}var ie=256,K=256;function q(e,t){let{id:r,operation:i,structure:a,valueChunks:o,needs:s,output:c}=t,{capacity:l,rowCount:u}=a,d=[],f=(t,i)=>n(e,`${r}-${t}`,`uint32`,i),p=(t,i)=>n(e,`${r}-${t}`,`float32`,i),m=f(`finite-counts`,l),h=s.sums||s.moments||s.zScore,g=!!c.sumValues||!!c.means||s.moments||s.zScore,_=h&&g?f(`sum-words`,3*l+1):void 0,v=c.sums,y=`
const SUM_SCALE_INDEX: u32 = ${3*l}u;
const FIXED_POINT_HEADROOM_BITS: u32 = ${D(u)}u;
${O}`,b=s.moments||s.zScore,x=s.minimum||b?f(`minimum-keys`,l):void 0,S=s.maximum||b?f(`maximum-keys`,l):void 0,C=!!c.means||b,k=b?p(`means`,2*l):void 0;if(d.push(R(e,{id:`${r}-clear`,operation:i,variant:`clear`,bindings:[V(`finiteCounts`,m),..._?[V(`sumWords`,_)]:[],...v?[V(`sums`,v)]:[],...x?[V(`minimumKeys`,x)]:[],...S?[V(`maximumKeys`,S)]:[]],invocationCount:l,body:`finiteCounts[finiteCountsOffset + index] = 0u;
  ${_?`for (var word = 0u; word < 3u; word++) {
    sumWords[sumWordsOffset + 3u * index + word] = 0u;
  }
  if (index == 0u) {
    sumWords[sumWordsOffset + ${3*l}u] = 0u;
  }`:``}
  ${v?`sums[sumsOffset + 2u * index] = 0u;
  sums[sumsOffset + 2u * index + 1u] = 0u;`:``}
  ${x?`minimumKeys[minimumKeysOffset + index] = 0xffffffffu;`:``}
  ${S?`maximumKeys[maximumKeysOffset + index] = 0u;`:``}`})),_)for(let[t,{valueBits:n,rowOffset:s,rowGroups:c}]of o.entries())d.push(R(e,{id:o.length===1?`${r}-sum-scale`:`${r}-sum-scale-${t}`,operation:i,variant:`sum-scale`,bindings:[B(`rowGroups`,c??a.rowGroups),B(`valueBits`,n),H(`sumWords`,_)],invocationCount:n.length,rowOffset:s,guardIndex:!1,declarations:`${y}
var<workgroup> workgroupExponentBits: atomic<u32>;`,body:`if (localInvocationIndex == 0u) {
    atomicStore(&workgroupExponentBits, 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT && rowGroups[rowGroupsOffset + index + GROUP_ROW_OFFSET] != ${z}u) {
    // 0 for zero and for non-finite values, which step 3 skips.
    let exponentBits = getFixedPointExponentBits(bitcast<f32>(valueBits[valueBitsOffset + index]));
    if (exponentBits != 0u) {
      atomicMax(&workgroupExponentBits, exponentBits);
    }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let workgroupMaximum = atomicLoad(&workgroupExponentBits);
    if (workgroupMaximum != 0u) {
      atomicMax(&sumWords[sumWordsOffset + SUM_SCALE_INDEX], workgroupMaximum);
    }
  }`}));let A=l<=ie,j=_?`${y}
${E(`sumWords`)}
fn getSumUnitExponent() -> i32 {
  return getFixedPointUnitExponent(
    atomicLoad(&sumWords[sumWordsOffset + SUM_SCALE_INDEX]),
    FIXED_POINT_HEADROOM_BITS
  );
}`:``;for(let[n,{valueBits:s,rowOffset:c,rowGroups:u}]of o.entries())d.push(R(e,{id:o.length===1?`${r}-accumulate`:`${r}-accumulate-${n}`,operation:i,variant:`accumulate`,bindings:[B(`rowGroups`,u??a.rowGroups),B(`valueBits`,s),H(`finiteCounts`,m),..._?[H(`sumWords`,_)]:[],...v?[H(`sums`,v)]:[],...x?[H(`minimumKeys`,x)]:[],...S?[H(`maximumKeys`,S)]:[]],invocationCount:s.length,rowOffset:c,...A?{guardIndex:!1,declarations:`${T(t.sumScale)}
${U}
${j}
const GROUP_CAPACITY: u32 = ${l}u;
var<workgroup> localCounts: array<atomic<u32>, ${l}>;
${_?`var<workgroup> localWords: array<atomic<u32>, ${3*l}>;
const localWordsOffset: u32 = 0u;
${E(`localWords`)}`:``}
${v?`var<workgroup> localSums: array<atomic<u32>, ${2*l}>;`:``}
${x?`var<workgroup> localMinimums: array<atomic<u32>, ${l}>;`:``}
${S?`var<workgroup> localMaximums: array<atomic<u32>, ${l}>;`:``}`,body:`for (var slot = localInvocationIndex; slot < GROUP_CAPACITY; slot += ${K}u) {
    atomicStore(&localCounts[slot], 0u);
    ${_?`for (var word = 0u; word < 3u; word++) {
      atomicStore(&localWords[3u * slot + word], 0u);
    }`:``}
    ${v?`atomicStore(&localSums[2u * slot], 0u);
    atomicStore(&localSums[2u * slot + 1u], 0u);`:``}
    ${x?`atomicStore(&localMinimums[slot], 0xffffffffu);`:``}
    ${S?`atomicStore(&localMaximums[slot], 0u);`:``}
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let group = rowGroups[rowGroupsOffset + index + GROUP_ROW_OFFSET];
    let bits = valueBits[valueBitsOffset + index];
    if (group != ${z}u && isFiniteBits(bits)) {
      atomicAdd(&localCounts[group], 1u);
      ${_?`addFixedPointToLocalWords(group, quantizeFixedPoint(bitcast<f32>(bits), getSumUnitExponent()));`:``}
      ${v?`let value = cellScaleValue(bitcast<f32>(bits));
      let previous = atomicAdd(&localSums[2u * group], value.y);
      let high = value.x + select(0u, 1u, previous + value.y < previous);
      if (high != 0u) {
        atomicAdd(&localSums[2u * group + 1u], high);
      }`:``}
      ${x?`atomicMin(&localMinimums[group], getValueKey(bits));`:``}
      ${S?`atomicMax(&localMaximums[group], getValueKey(bits));`:``}
    }
  }
  workgroupBarrier();
  for (var slot = localInvocationIndex; slot < GROUP_CAPACITY; slot += ${K}u) {
    let count = atomicLoad(&localCounts[slot]);
    if (count != 0u) {
      atomicAdd(&finiteCounts[finiteCountsOffset + slot], count);
      ${_?`addFixedPointToSumWords(slot, vec3<u32>(
        atomicLoad(&localWords[3u * slot]),
        atomicLoad(&localWords[3u * slot + 1u]),
        atomicLoad(&localWords[3u * slot + 2u])
      ));`:``}
      ${v?`let localLow = atomicLoad(&localSums[2u * slot]);
      let previous = atomicAdd(&sums[sumsOffset + 2u * slot], localLow);
      let high = atomicLoad(&localSums[2u * slot + 1u]) + select(0u, 1u, previous + localLow < previous);
      if (high != 0u) {
        atomicAdd(&sums[sumsOffset + 2u * slot + 1u], high);
      }`:``}
      ${x?`atomicMin(&minimumKeys[minimumKeysOffset + slot], atomicLoad(&localMinimums[slot]));`:``}
      ${S?`atomicMax(&maximumKeys[maximumKeysOffset + slot], atomicLoad(&localMaximums[slot]));`:``}
    }
  }`}:{declarations:`${T(t.sumScale)}
${U}
${j}`,body:`let group = rowGroups[rowGroupsOffset + index + GROUP_ROW_OFFSET];
  if (group == ${z}u) {
    return;
  }
  let bits = valueBits[valueBitsOffset + index];
  if (!isFiniteBits(bits)) {
    return;
  }
  atomicAdd(&finiteCounts[finiteCountsOffset + group], 1u);
  ${_?`addFixedPointToSumWords(group, quantizeFixedPoint(bitcast<f32>(bits), getSumUnitExponent()));`:``}
  ${v?`let value = cellScaleValue(bitcast<f32>(bits));
  let previous = atomicAdd(&sums[sumsOffset + 2u * group], value.y);
  let high = value.x + select(0u, 1u, previous + value.y < previous);
  if (high != 0u) {
    atomicAdd(&sums[sumsOffset + 2u * group + 1u], high);
  }`:``}
  ${x?`atomicMin(&minimumKeys[minimumKeysOffset + group], getValueKey(bits));`:``}
  ${S?`atomicMax(&maximumKeys[maximumKeysOffset + group], getValueKey(bits));`:``}`}}));if((c.counts||_)&&d.push(R(e,{id:`${r}-finish-sums`,operation:i,variant:`finish-sums`,bindings:[B(`finiteCounts`,m),..._?[B(`sumWords`,_)]:[],...c.counts?[V(`countsOut`,c.counts)]:[],...c.sumValues?[V(`sumValuesOut`,c.sumValues,`f32`)]:[],...c.means?[V(`meansOut`,c.means,`f32`)]:[],...k&&x&&S?[V(`meanScratch`,k,`f32`),B(`minimumKeys`,x),B(`maximumKeys`,S)]:[]],invocationCount:l,declarations:`${w}
${_?y:``}`,body:`let count = finiteCounts[finiteCountsOffset + index];
  ${c.counts?`countsOut[countsOutOffset + index] = count;`:``}
  ${_?`let base = sumWordsOffset + 3u * index;
  // The exact integer sum, rounded once to the nearest float32.
  let total = getFixedPointFloat(
    vec3<u32>(sumWords[base], sumWords[base + 1u], sumWords[base + 2u]),
    getFixedPointUnitExponent(sumWords[sumWordsOffset + SUM_SCALE_INDEX], FIXED_POINT_HEADROOM_BITS)
  );
  ${c.sumValues?`sumValuesOut[sumValuesOutOffset + index] = total;`:``}
  ${C?`let mean = select(getNaN(), total / f32(count), count > 0u);`:``}
  ${c.means?`meansOut[meansOutOffset + index] = mean;`:``}
  ${k?`meanScratch[meanScratchOffset + 2u * index] = mean;
  let magnitudeBits = max(
    bitcast<u32>(decodeOrderedKey(minimumKeys[minimumKeysOffset + index])) & 0x7fffffffu,
    bitcast<u32>(decodeOrderedKey(maximumKeys[maximumKeysOffset + index])) & 0x7fffffffu
  );
  let scaleExponent = max(i32(magnitudeBits >> 23u) - 127 - ${re}, 0);
  meanScratch[meanScratchOffset + 2u * index + 1u] = f32(select(0, scaleExponent, count > 0u));`:``}`:``}`})),(c.minimums||c.maximums)&&d.push(R(e,{id:`${r}-finish-extremes`,operation:i,variant:`finish-extremes`,bindings:[B(`finiteCounts`,m),...x?[B(`minimumKeys`,x)]:[],...S?[B(`maximumKeys`,S)]:[],...c.minimums?[V(`minimumsOut`,c.minimums,`f32`)]:[],...c.maximums?[V(`maximumsOut`,c.maximums,`f32`)]:[]],invocationCount:l,declarations:w,body:`let isEmpty = finiteCounts[finiteCountsOffset + index] == 0u;
  ${x&&c.minimums?`minimumsOut[minimumsOutOffset + index] = select(decodeOrderedKey(minimumKeys[minimumKeysOffset + index]), getNaN(), isEmpty);`:``}
  ${S&&c.maximums?`maximumsOut[maximumsOutOffset + index] = select(decodeOrderedKey(maximumKeys[maximumKeysOffset + index]), getNaN(), isEmpty);`:``}`})),!b||!k)return{nodes:d,finiteCounts:m};let M=p(`moment-sums`,4*l);if(d.push(...t.getMomentNodes?t.getMomentNodes(k,M):W(e,{id:r,operation:i,structure:a,valueChunks:o,means:k,output:M})),(c.variances||c.standardDeviations||c.skewness||c.kurtosis)&&d.push(R(e,{id:`${r}-finish-moments`,operation:i,variant:`finish-moments`,bindings:[B(`finiteCounts`,m),B(`momentSums`,M,`f32`),...c.variances?[V(`variancesOut`,c.variances,`f32`)]:[],...c.standardDeviations?[V(`deviationsOut`,c.standardDeviations,`f32`)]:[],...c.skewness?[V(`skewnessOut`,c.skewness,`f32`)]:[],...c.kurtosis?[V(`kurtosisOut`,c.kurtosis,`f32`)]:[]],invocationCount:l,declarations:`${w}
${G(t.variance)}`,body:`let count = finiteCounts[finiteCountsOffset + index];
  let nan = getNaN();
  var variance = nan;
  var skewness = nan;
  var kurtosis = nan;
  if (count > 0u) {
    let sums = vec4f(
      momentSums[momentSumsOffset + 4u * index],
      momentSums[momentSumsOffset + 4u * index + 1u],
      momentSums[momentSumsOffset + 4u * index + 2u],
      momentSums[momentSumsOffset + 4u * index + 3u]
    );
    let m2 = getMomentM2(count, sums);
    variance = getMomentVariance(count, m2);
    skewness = getMomentSkewness(count, sums, m2);
    kurtosis = getMomentKurtosis(count, sums, m2);
  }
  ${c.variances?`variancesOut[variancesOutOffset + index] = variance;`:``}
  ${c.standardDeviations?`deviationsOut[deviationsOutOffset + index] = select(nan, sqrt(variance), count > 0u && (!SAMPLE_VARIANCE || count >= 2u));`:``}
  ${c.skewness?`skewnessOut[skewnessOutOffset + index] = skewness;`:``}
  ${c.kurtosis?`kurtosisOut[kurtosisOutOffset + index] = kurtosis;`:``}`})),c.zScores)for(let[n,{valueBits:s,rowOffset:l,rowGroups:u}]of o.entries())d.push(R(e,{id:o.length===1?`${r}-z-scores`:`${r}-z-scores-${n}`,operation:i,variant:`z-scores`,bindings:[B(`rowGroups`,u??a.rowGroups),B(`valueBits`,s),B(`means`,k,`f32`),B(`finiteCounts`,m),B(`momentSums`,M,`f32`),V(`zScoresOut`,c.zScores,`f32`)],invocationCount:s.length,rowOffset:l,declarations:`${w}
${G(t.variance)}
fn isFiniteBits(bits: u32) -> bool {
  return (bits & 0x7f800000u) != 0x7f800000u;
}`,body:`let group = rowGroups[rowGroupsOffset + index + GROUP_ROW_OFFSET];
  let bits = valueBits[valueBitsOffset + index];
  var z = getNaN();
  if (group != ${z}u && isFiniteBits(bits)) {
    let count = finiteCounts[finiteCountsOffset + group];
    let sums = vec4f(
      momentSums[momentSumsOffset + 4u * group],
      momentSums[momentSumsOffset + 4u * group + 1u],
      momentSums[momentSumsOffset + 4u * group + 2u],
      momentSums[momentSumsOffset + 4u * group + 3u]
    );
    let m2 = getMomentM2(count, sums);
    let variance = select(0.0, getMomentVariance(count, m2), count >= 2u || !SAMPLE_VARIANCE);
    // Centre on the mean the variance uses: the shift mean plus delta = S1 / n. delta is at most
    // half a float32 step of the mean, so it survives only when it is subtracted from the
    // deviation, never when it is added to the mean. A shader compiler may rewrite
    // (v - mean) - delta as v - (mean + delta) and lose it (measured on the test GPU: values
    // 1e7 .. 1e7 + 3 came out 0.39 off). The deviation therefore passes through a bit operation
    // with a runtime zero the compiler cannot see (a group index is below 2^31), which fixes
    // its rounding before delta is subtracted.
    let delta = sums.x / f32(count);
    let runtimeZero = group >> 31u;
    let deviation = bitcast<f32>(
      bitcast<u32>(bitcast<f32>(bits) - means[meansOffset + 2u * group]) | runtimeZero
    );
    let centered = deviation - delta;
    z = select(centered / sqrt(variance), 0.0, m2 == 0.0 || !(variance > 0.0));
  }
  zScoresOut[zScoresOutOffset + index + GROUP_ROW_OFFSET] = z;`}));return{nodes:d,finiteCounts:m}}function J(e,t){let{id:n,keyLow:r,keyHigh:i,rowIds:a}=t,o=M(e,{id:n,operation:`GPUGroupStatistics`,indices:t.initialRows??a,indicesAreIdentity:!t.initialRows,keys:[...i?[{view:i}]:[],{view:r,keyBits:t.lowKeyBits??32}]}),s=i||t.initialRows?void 0:o.sortedPrimaryKeys;return{nodes:[...o.nodes],permutation:o.sortedIndices,sortedKeyLow:s}}function Y(e,t){let{id:r,operation:i,output:a}=t,o=t.keys.length,s=a.counts.length,c=t.keys.format===`uint32x2`,l=t.keyCount,u=l!==void 0,d=u?l:4294967295,f=u?Math.max(1,32-Math.clz32(l)):32,m=(t,i)=>n(e,`${r}-${t}`,`uint32`,i),h=[],g=m(`key-low`,o),_=c?m(`key-high`,o):void 0,v=m(`row-ids`,o),y=0,b=p(t.keys),x=t.mask?p(t.mask):void 0;for(let[t,n]of b.entries()){let a=x?.[t];n.length!==0&&(h.push(R(e,{id:b.length===1?`${r}-prepare`:`${r}-prepare-${t}`,operation:i,variant:`prepare`,bindings:[B(`keys`,n),...a?[B(`mask`,a)]:[],V(`keyLow`,g),..._?[V(`keyHigh`,_)]:[],V(`rowIds`,v)],invocationCount:n.length,rowOffset:y,body:`let sourceRow = index + GROUP_ROW_OFFSET;
  ${c?`var low = keys[keysOffset + 2u * index];
  var high = keys[keysOffset + 2u * index + 1u];
  var isValid = !(low == 0xffffffffu && high == 0xffffffffu);`:`var low = keys[keysOffset + index];
  var isValid = low != 0xffffffffu;`}
  ${a?`if (mask[maskOffset + index] == 0u) {
    isValid = false;
  }`:``}
  ${u?`if (low >= ${l}u) {\n    isValid = false;\n  }`:``}
  if (!isValid) {
    low = ${d}u;
    ${c?`high = 0xffffffffu;`:``}
  }
  keyLow[keyLowOffset + sourceRow] = low;
  ${c?`keyHigh[keyHighOffset + sourceRow] = high;`:``}
  rowIds[rowIdsOffset + sourceRow] = sourceRow;`})),y+=n.length)}let S=J(e,{id:`${r}-order`,keyLow:g,keyHigh:_,rowIds:v,lowKeyBits:f});h.push(...S.nodes);let C=S.permutation,w=S.sortedKeyLow;w||(w=m(`sorted-low`,o),h.push(...new j({id:`${r}-gather-sorted-low`,source:g,indices:C,output:w}).getCommandNodes(e)));let T=c?m(`sorted-high`,o):void 0;T&&_&&h.push(...new j({id:`${r}-gather-sorted-high`,source:_,indices:C,output:T}).getCommandNodes(e));let E=()=>[B(`sortedLow`,w),...T?[B(`sortedHigh`,T)]:[]],D=e=>c?`vec2u(sortedHigh[sortedHighOffset + ${e}], sortedLow[sortedLowOffset + ${e}])`:`vec2u(0u, sortedLow[sortedLowOffset + ${e}])`,O=`const TWO_WORDS: bool = ${c};
fn isInvalidKey(key: vec2u) -> bool {
  return key.y == ${d}u && (!TWO_WORDS || key.x == 0xffffffffu);
}`,A=m(`heads`,o),M=m(`valid-count`,1);h.push(R(e,{id:`${r}-heads`,operation:i,variant:`heads`,bindings:[...E(),V(`heads`,A),V(`validCount`,M)],invocationCount:o,declarations:`const ROW_COUNT: u32 = ${o}u;
${O}`,body:`let key = ${D(`index`)};
  let isValid = !isInvalidKey(key);
  var isHead = isValid;
  if (isValid && index > 0u) {
    isHead = any(key != ${D(`index - 1u`)});
  }
  heads[headsOffset + index] = select(0u, 1u, isHead);
  let isLastValid = isValid && (index + 1u == ROW_COUNT || isInvalidKey(${D(`index + 1u`)}));
  if (isLastValid) {
    validCount[validCountOffset] = index + 1u;
  } else if (index == 0u && !isValid) {
    validCount[validCountOffset] = 0u;
  }`}));let N=m(`offsets`,s+1),P=m(`group-total`,1),F=m(`position-groups`,o);if(u)h.push(R(e,{id:`${r}-dense-offsets`,operation:i,variant:`dense-offsets`,bindings:[...E(),B(`validCount`,M),V(`offsets`,N),V(`groupTotal`,P)],invocationCount:s+1,body:`var low = 0u;
  var high = validCount[validCountOffset];
  while (low < high) {
    let middle = (low + high) / 2u;
    if (sortedLow[sortedLowOffset + middle] < index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  offsets[offsetsOffset + index] = low;
  if (index == 0u) {
    groupTotal[groupTotalOffset] = ${s}u;
  }`}),R(e,{id:`${r}-dense-position-groups`,operation:i,variant:`dense-position-groups`,bindings:[...E(),B(`validCount`,M),V(`positionGroups`,F)],invocationCount:o,body:`positionGroups[positionGroupsOffset + index] =
    select(0xffffffffu, sortedLow[sortedLowOffset + index], index < validCount[validCountOffset]);`}));else{let t=m(`group-indices`,o);h.push(...new k({id:`${r}-scan`,input:A,output:t,mode:`exclusive`}).getCommandNodes(e)),h.push(R(e,{id:`${r}-offsets`,operation:i,variant:`offsets`,bindings:[B(`heads`,A),B(`groupIndices`,t),B(`validCount`,M),V(`offsets`,N),V(`groupTotal`,P),V(`positionGroups`,F)],invocationCount:o,declarations:`const ROW_COUNT: u32 = ${o}u;
const CAPACITY: u32 = ${s}u;`,body:`let isHead = heads[headsOffset + index] != 0u;
  let group = groupIndices[groupIndicesOffset + index] + select(0u, 1u, isHead) - 1u;
  let isValid = index < validCount[validCountOffset];
  positionGroups[positionGroupsOffset + index] = select(0xffffffffu, group, isValid && group < CAPACITY);
  if (isHead && group <= CAPACITY) {
    offsets[offsetsOffset + group] = index;
  }
  if (index == 0u) {
    let last = ROW_COUNT - 1u;
    let total = groupIndices[groupIndicesOffset + last] + heads[headsOffset + last];
    groupTotal[groupTotalOffset] = total;
    if (total <= CAPACITY) {
      offsets[offsetsOffset + total] = validCount[validCountOffset];
    }
  }`}))}let I=m(`row-groups`,o);return h.push(R(e,{id:`${r}-row-groups`,operation:i,variant:`row-groups`,bindings:[B(`permutation`,C),B(`positionGroups`,F),V(`rowGroups`,I)],invocationCount:o,body:`rowGroups[rowGroupsOffset + permutation[permutationOffset + index]] =
    positionGroups[positionGroupsOffset + index];`})),u?h.push(R(e,{id:`${r}-groups`,operation:i,variant:`dense-groups`,bindings:[B(`offsets`,N),V(`keysOut`,a.keys),V(`countsOut`,a.counts)],invocationCount:s,body:`keysOut[keysOutOffset + index] = index;
  countsOut[countsOutOffset + index] = offsets[offsetsOffset + index + 1u] - offsets[offsetsOffset + index];`})):h.push(R(e,{id:`${r}-groups`,operation:i,variant:`groups`,bindings:[B(`offsets`,N),B(`groupTotal`,P),...E(),V(`keysOut`,a.keys),V(`countsOut`,a.counts)],invocationCount:s,declarations:`const CAPACITY: u32 = ${s}u;
const GROUP_NONE: u32 = ${z}u;`,body:`if (index >= min(groupTotal[groupTotalOffset], CAPACITY)) {
    ${c?`keysOut[keysOutOffset + 2u * index] = 0xffffffffu;
    keysOut[keysOutOffset + 2u * index + 1u] = 0xffffffffu;`:`keysOut[keysOutOffset + index] = 0xffffffffu;`}
    countsOut[countsOutOffset + index] = 0u;
    return;
  }
  let begin = offsets[offsetsOffset + index];
  let key = ${D(`begin`)};
  ${c?`keysOut[keysOutOffset + 2u * index] = key.y;
  keysOut[keysOutOffset + 2u * index + 1u] = key.x;`:`keysOut[keysOutOffset + index] = key.y;`}
  countsOut[countsOutOffset + index] = offsets[offsetsOffset + index + 1u] - begin;`})),h.push(ee(e,{id:`${r}-publish`,operation:i,requiredCount:P,output:{ids:a.counts,status:a.status}})),{nodes:h,structure:{rowCount:o,capacity:s,twoWords:c,permutation:C,keyLow:g,lowKeyBits:f,keyHigh:_,rowIds:v,offsets:N,groupTotal:P,positionGroups:F,rowGroups:I}}}var ae=`
${N}

fn getQuantile(begin: u32, count: u32, p: f32) -> f32 {
  if (count == 0u || !isOrderStatisticProbability(p)) {
    return getNaN();
  }
  // Counts stay below 2^31, so this is a runtime zero that blocks multiply-add fusion.
  let zero = count >> 31u;
  let h = roundOrderStatisticStep(f32(count - 1u) * p, zero);
  let floorH = floor(h);
  let lowIndex = min(u32(floorH), count - 1u);
  let low = decodeOrderedKey(sortedKeys[sortedKeysOffset + begin + lowIndex]);
  if (lowIndex + 1u >= count) {
    return low;
  }
  let high = decodeOrderedKey(sortedKeys[sortedKeysOffset + begin + lowIndex + 1u]);
  return interpolateOrderStatistic(low, high, h - floorH, zero);
}
`;function oe(e,t){let{id:r,operation:i,structure:a,valueBits:o,finiteCounts:s,output:c}=t,{rowCount:l,capacity:u}=a,d=[],f=(t,i=l)=>n(e,`${r}-${t}`,`uint32`,i),p=f(`value-keys`);d.push(R(e,{id:`${r}-value-keys`,operation:i,variant:`value-keys`,bindings:[B(`valueBits`,o),V(`valueKeys`,p)],invocationCount:l,declarations:U,body:`valueKeys[valueKeysOffset + index] = getValueKey(valueBits[valueBitsOffset + index]);`}));let m=f(`rows-by-value`);d.push(...new A({id:`${r}-sort-value`,keys:p,values:a.rowIds,outputKeys:f(`sorted-value-scratch`),outputValues:m,keyBits:32}).getCommandNodes(e));let h=J(e,{id:`${r}-order`,keyLow:a.keyLow,keyHigh:a.keyHigh,rowIds:a.rowIds,initialRows:m,lowKeyBits:a.lowKeyBits});d.push(...h.nodes);let g=f(`sorted-keys`);d.push(...new j({id:`${r}-gather-sorted-keys`,source:p,indices:h.permutation,output:g}).getCommandNodes(e));let _=`${w}
${U}
${ae}`;if(c.medians&&d.push(R(e,{id:`${r}-medians`,operation:i,variant:`medians`,bindings:[B(`finiteCounts`,s),B(`offsets`,a.offsets),B(`sortedKeys`,g),V(`mediansOut`,c.medians,`f32`)],invocationCount:u,declarations:_,body:`let count = finiteCounts[finiteCountsOffset + index];
  var begin = 0u;
  if (count > 0u) {
    begin = offsets[offsetsOffset + index];
  }
  mediansOut[mediansOutOffset + index] = getQuantile(begin, count, 0.5);`})),c.percentiles&&t.percentiles){let n=t.percentiles.length;d.push(R(e,{id:`${r}-percentiles`,operation:i,variant:`percentiles`,bindings:[B(`finiteCounts`,s),B(`offsets`,a.offsets),B(`sortedKeys`,g),B(`fractions`,t.percentiles,`f32`),V(`percentilesOut`,c.percentiles,`f32`)],invocationCount:u*n,declarations:`${_}
const PERCENTILE_COUNT: u32 = ${n}u;`,body:`let group = index / PERCENTILE_COUNT;
  let count = finiteCounts[finiteCountsOffset + group];
  var begin = 0u;
  if (count > 0u) {
    begin = offsets[offsetsOffset + group];
  }
  percentilesOut[percentilesOutOffset + index] =
    getQuantile(begin, count, fractions[fractionsOffset + index % PERCENTILE_COUNT]);`}))}let v=!!c.modes,y=!!c.uniqueCounts;if(!v&&!y)return d;let b=f(`run-heads`);d.push(R(e,{id:`${r}-run-heads`,operation:i,variant:`run-heads`,bindings:[B(`sortedKeys`,g),B(`positionGroups`,a.positionGroups),B(`offsets`,a.offsets),V(`heads`,b)],invocationCount:l,body:`let group = positionGroups[positionGroupsOffset + index];
  let key = sortedKeys[sortedKeysOffset + index];
  var isHead = group != 0xffffffffu && key != 0xffffffffu;
  if (isHead && index != offsets[offsetsOffset + group]) {
    isHead = key != sortedKeys[sortedKeysOffset + index - 1u];
  }
  heads[headsOffset + index] = select(0u, 1u, isHead);`}));let x=f(`unique-counts`,u),S=v?f(`maximum-run-lengths`,u):void 0,C=v?f(`mode-runs`,u):void 0;if(d.push(R(e,{id:`${r}-run-clear`,operation:i,variant:`run-clear`,bindings:[V(`uniqueCounts`,x),...S?[V(`maximumLengths`,S)]:[],...C?[V(`modeRuns`,C)]:[]],invocationCount:u,body:`uniqueCounts[uniqueCountsOffset + index] = 0u;
  ${S?`maximumLengths[maximumLengthsOffset + index] = 0u;`:``}
  ${C?`modeRuns[modeRunsOffset + index] = 0xffffffffu;`:``}`})),!v)d.push(R(e,{id:`${r}-count-runs`,operation:i,variant:`count-runs`,bindings:[B(`heads`,b),B(`positionGroups`,a.positionGroups),H(`uniqueCounts`,x)],invocationCount:l,body:`if (heads[headsOffset + index] != 0u) {
    atomicAdd(&uniqueCounts[uniqueCountsOffset + positionGroups[positionGroupsOffset + index]], 1u);
  }`}));else if(S&&C){let t=f(`run-indices`);d.push(...new k({id:`${r}-run-scan`,input:b,output:t,mode:`exclusive`}).getCommandNodes(e));let n=f(`run-starts`);d.push(R(e,{id:`${r}-run-starts`,operation:i,variant:`run-starts`,bindings:[B(`heads`,b),B(`runIndices`,t),V(`runStarts`,n)],invocationCount:l,body:`if (heads[headsOffset + index] != 0u) {
    runStarts[runStartsOffset + runIndices[runIndicesOffset + index]] = index;
  }`}));let o=()=>[B(`heads`,b),B(`runIndices`,t),B(`runStarts`,n),B(`positionGroups`,a.positionGroups),B(`offsets`,a.offsets),B(`finiteCounts`,s)],p=`const ROW_COUNT: u32 = ${l}u;`,m=`let run = runIndices[runIndicesOffset + index];
  let group = positionGroups[positionGroupsOffset + index];
  let totalRuns = runIndices[runIndicesOffset + ROW_COUNT - 1u] + heads[headsOffset + ROW_COUNT - 1u];
  var nextStart = ROW_COUNT;
  if (run + 1u < totalRuns) {
    nextStart = runStarts[runStartsOffset + run + 1u];
  }
  let runEnd = min(nextStart, offsets[offsetsOffset + group] + finiteCounts[finiteCountsOffset + group]);
  let runLength = runEnd - index;`;d.push(R(e,{id:`${r}-run-lengths`,operation:i,variant:`run-lengths`,bindings:[...o(),H(`maximumLengths`,S),...y?[H(`uniqueCounts`,x)]:[]],invocationCount:l,declarations:p,body:`if (heads[headsOffset + index] == 0u) {
    return;
  }
  ${m}
  atomicMax(&maximumLengths[maximumLengthsOffset + group], runLength);
  ${y?`atomicAdd(&uniqueCounts[uniqueCountsOffset + group], 1u);`:``}`})),d.push(R(e,{id:`${r}-mode-runs`,operation:i,variant:`mode-runs`,bindings:[...o(),H(`maximumLengths`,S),H(`modeRuns`,C)],invocationCount:l,declarations:p,body:`if (heads[headsOffset + index] == 0u) {
    return;
  }
  ${m}
  if (runLength == atomicLoad(&maximumLengths[maximumLengthsOffset + group])) {
    atomicMin(&modeRuns[modeRunsOffset + group], run);
  }`})),d.push(R(e,{id:`${r}-finish-modes`,operation:i,variant:`finish-modes`,bindings:[B(`finiteCounts`,s),B(`modeRuns`,C),B(`runStarts`,n),B(`sortedKeys`,g),V(`modesOut`,c.modes,`f32`)],invocationCount:u,declarations:w,body:`var mode = getNaN();
  if (finiteCounts[finiteCountsOffset + index] > 0u) {
    mode = decodeOrderedKey(
      sortedKeys[sortedKeysOffset + runStarts[runStartsOffset + modeRuns[modeRunsOffset + index]]]
    );
  }
  modesOut[modesOutOffset + index] = mode;`}))}return c.uniqueCounts&&d.push(R(e,{id:`${r}-finish-unique`,operation:i,variant:`finish-unique`,bindings:[B(`uniqueCounts`,x),V(`uniqueOut`,c.uniqueCounts)],invocationCount:u,body:`uniqueOut[uniqueOutOffset + index] = uniqueCounts[uniqueCountsOffset + index];`})),d}function se(e,t,r,i,a,o){let s=[],c=t.output.keys.length,l=t=>{s.push(R(e,{...t,operation:`GPUGroupStatistics`}))},u=(t,a,o,s)=>{let c=n(e,`${r}-tile-${o}-${a}`,t.format,i),u=t.rowByteLength/4;l({id:`${r}-tile-${o}-${a}-pad`,invocationCount:i,bindings:[V(`destination`,c)],body:Array.from({length:u},(e,t)=>`destination[destinationOffset + ${u}u * index + ${t}u] = ${s}u;`).join(`
`)});let d=o*i,f=Math.min(d+i,t.length),m=0;for(let[n,i]of p(t).entries()){let t=m+i.length,s=Math.max(d,m),p=Math.min(f,t);if(s<p){let t=e.createDataView(i.buffer,{format:i.format,length:p-s,byteOffset:i.byteOffset+(s-m)*i.rowByteLength});l({id:`${r}-tile-${o}-${a}-gather-${n}`,invocationCount:p-s,rowOffset:s-d,bindings:[B(`source`,t),V(`destination`,c)],body:Array.from({length:u},(e,t)=>`destination[destinationOffset + ${u}u * (GROUP_ROW_OFFSET + index) + ${t}u] = source[sourceOffset + ${u}u * index + ${t}u];`).join(`
`)})}m=t}return c},d=Array.from({length:Math.ceil(t.keys.length/i)},(a,o)=>{let c=`${r}-tile-${o}`,d=u(t.keys,`keys`,o,4294967295),f=t.mask?u(t.mask,`mask`,o,0):void 0;t.keyCount!==void 0&&l({id:`${c}-dense-domain`,invocationCount:i,bindings:[V(`keys`,d)],body:`if (keys[keysOffset + index] >= ${t.keyCount}u) { keys[keysOffset + index] = 0xffffffffu; }`});let p={keys:n(e,`${c}-group-keys`,t.keys.format,i),counts:n(e,`${c}-group-counts`,`uint32`,i),status:{count:n(e,`${c}-count`,`uint32`,1),overflow:n(e,`${c}-overflow`,`uint32`,1)}},m=Y(e,{id:c,operation:`GPUGroupStatistics`,keys:d,mask:f,output:p});s.push(...m.nodes);let h=n(e,`${c}-union-keys`,`uint32x2`,i);l({id:`${c}-union-keys`,invocationCount:i,bindings:[B(`keys`,p.keys),B(`counts`,p.counts),V(`cells`,h)],body:`let occupied = counts[countsOffset + index] != 0u;
  cells[cellsOffset + 2u * index] = keys[keysOffset + ${t.keys.format===`uint32x2`?`2u * index`:`index`}];
  cells[cellsOffset + 2u * index + 1u] = ${t.keys.format===`uint32x2`?`keys[keysOffset + 2u * index + 1u]`:`select(0xffffffffu, 0u, occupied)`};`});let g=t.columns.map((e,t)=>u(e.values,`column-${t}`,o,2143289344));return{structure:m.structure,cells:h,counts:p.counts,values:g}}),f=n(e,`${r}-union-keys`,`uint32x2`,c),m;if(t.keyCount===void 0){let n=I(e,`${r}-union`,d.map(e=>({keys:e.cells,counts:e.counts,count:e.structure.groupTotal})),{keys:f,counts:t.output.counts},t.output.status);s.push(...n.nodes),m=n.ranks}else l({id:`${r}-dense-clear`,invocationCount:c,bindings:[V(`cells`,f),V(`counts`,t.output.counts)],body:`cells[cellsOffset + 2u * index] = index; cells[cellsOffset + 2u * index + 1u] = 0u; counts[countsOffset + index] = 0u;`}),m=d.map((a,o)=>{let s=n(e,`${r}-rank-${o}`,`uint32`,i);return l({id:`${r}-dense-merge-${o}`,invocationCount:i,bindings:[B(`keys`,a.cells),B(`tileCounts`,a.counts),V(`rank`,s),V(`counts`,t.output.counts)],body:`let key = keys[keysOffset + 2u * index];
  rank[rankOffset + index] = key;
  if (tileCounts[tileCountsOffset + index] != 0u) { counts[countsOffset + key] += tileCounts[tileCountsOffset + index]; }`}),s}),l({id:`${r}-dense-status`,invocationCount:1,bindings:[V(`count`,t.output.status.count),V(`overflow`,t.output.status.overflow),...t.output.status.requiredCount?[V(`requiredCount`,t.output.status.requiredCount)]:[]],body:`count[countOffset] = ${c}u; overflow[overflowOffset] = 0u; ${t.output.status.requiredCount?`requiredCount[requiredCountOffset] = ${c}u;`:``}`});l({id:`${r}-keys-publish`,invocationCount:c,bindings:[B(`cells`,f),V(`keys`,t.output.keys)],body:t.keys.format===`uint32x2`?`keys[keysOffset + 2u * index] = cells[cellsOffset + 2u * index]; keys[keysOffset + 2u * index + 1u] = cells[cellsOffset + 2u * index + 1u];`:`keys[keysOffset + index] = cells[cellsOffset + 2u * index];`});let h=d.map((t,a)=>{let o=n(e,`${r}-row-groups-${a}`,`uint32`,i);return l({id:`${r}-row-groups-${a}`,invocationCount:i,bindings:[B(`localGroups`,t.structure.rowGroups),B(`rank`,m[a]),V(`groups`,o)],body:`let localGroup = localGroups[localGroupsOffset + index];
  var group = 0xffffffffu;
  if (localGroup != 0xffffffffu) { group = rank[rankOffset + localGroup]; }
  groups[groupsOffset + index] = select(0xffffffffu, group, group < ${c}u);`}),o});for(let[l,u]of t.columns.entries()){let f=`${r}-column-${l}`,p=new Set(u.statistics),g=d.map((t,n)=>({valueBits:e.createDataView(t.values[l].buffer,{format:`uint32`,length:i}),rowOffset:0,rowGroups:h[n]}));s.push(...q(e,{id:f,operation:`GPUGroupStatistics`,structure:{...d[0].structure,rowCount:t.keys.length,capacity:c,rowGroups:h[0]},valueChunks:g,sumScale:a,variance:o,needs:{sums:p.has(`sum`)||p.has(`mean`),minimum:p.has(`minimum`),maximum:p.has(`maximum`),moments:[`variance`,`standardDeviation`,`skewness`,`kurtosis`].some(e=>p.has(e)),zScore:!1},output:u.output,getMomentNodes:(t,r)=>{let a=[],o=t=>a.push(R(e,{...t,operation:`GPUGroupStatistics`}));o({id:`${f}-moments-clear`,invocationCount:c,bindings:[V(`moments`,r,`f32`)],body:`for (var component = 0u; component < 4u; component++) { moments[momentsOffset + 4u * index + component] = 0.0; }`});for(let[s,l]of d.entries()){let u=n(e,`${f}-means-${s}`,`float32`,i*2),d=n(e,`${f}-moments-${s}`,`float32`,i*4);o({id:`${f}-means-${s}`,invocationCount:i,bindings:[B(`rank`,m[s]),B(`means`,t,`f32`),V(`localMeans`,u,`f32`)],body:`let group = rank[rankOffset + index];
  var mean = 0.0; var exponent = 0.0;
  if (group < ${c}u) { mean = means[meansOffset + 2u * group]; exponent = means[meansOffset + 2u * group + 1u]; }
  localMeans[localMeansOffset + 2u * index] = mean;
  localMeans[localMeansOffset + 2u * index + 1u] = exponent;`}),a.push(...W(e,{id:`${f}-tile-${s}`,operation:`GPUGroupStatistics`,structure:l.structure,valueChunks:[{valueBits:g[s].valueBits,rowOffset:0}],means:u,output:d,retainScaledOddMoments:!0})),o({id:`${f}-moments-merge-${s}`,invocationCount:i,bindings:[B(`rank`,m[s]),B(`counts`,l.counts),B(`localMoments`,d,`f32`),V(`moments`,r,`f32`)],body:`let group = rank[rankOffset + index];
  if (group < ${c}u && counts[countsOffset + index] != 0u) {
    for (var component = 0u; component < 4u; component++) {
      moments[momentsOffset + 4u * group + component] += localMoments[localMomentsOffset + 4u * index + component];
    }
  }`})}return o({id:`${f}-moments-restore`,invocationCount:c,bindings:[B(`means`,t,`f32`),V(`moments`,r,`f32`)],body:`let exponent = i32(means[meansOffset + 2u * index + 1u]);
  let first = momentsOffset + 4u * index;
  let third = first + 2u;
  moments[first] = ldexp(moments[first], exponent);
  moments[third] = ldexp(ldexp(ldexp(moments[third], exponent), exponent), exponent);`}),a}}).nodes)}return s}var X=`GPUGroupStatistics`,ce=1e3,le=64,Z=536870912,Q=4,ue=2**31,de=1073741823,fe=4294967295,pe=4294967295,me=512;function he(e){return Math.min(de,g(e,16))}function ge(e){return Math.min(pe,g(e,4))}var $={count:[`counts`],sum:[`sums`,`sumValues`],mean:[`means`],minimum:[`minimums`],maximum:[`maximums`],variance:[`variances`],standardDeviation:[`standardDeviations`],skewness:[`skewness`],kurtosis:[`kurtosis`],median:[`medians`],percentiles:[`percentiles`],mode:[`modes`],uniqueCount:[`uniqueCounts`],zScore:[`zScores`]},_e=[`sumValues`,`means`,`minimums`,`maximums`,`variances`,`standardDeviations`,`skewness`,`kurtosis`,`medians`,`percentiles`,`modes`,`zScores`],ve=class{id;scratchTileSize;usesTiledScratch;props;variance;sumScale;estimatedCommandNodeCount;constructor(e){this.id=e.id??`group-statistics`,this.props=e;let t=this.id;if(this.variance=e.variance??`sample`,this.variance!==`sample`&&this.variance!==`population`)throw Error(`${t} variance must be 'sample' or 'population'`);this.sumScale=e.sumScale??65536,C(t,this.sumScale);let{keys:n,mask:r,columns:i,output:a}=e;if(i.length>Q)throw Error(`${t} supports at most ${Q} columns`);for(let e of p(n))l(e,[`uint32`,`uint32x2`],`${t} keys`);d(t,n.format===`uint32`||n.format===`uint32x2`,`keys.format must be uint32 or uint32x2`);let o=n.length;if(m(t,`keys.length`,o,1,ue),r){for(let e of p(r))l(e,[`uint32`],`${t} mask`);if(r.length!==o)throw Error(`${t} mask must have the same length as keys`)}if(f(t,`mask`,n,r),e.keyCount!==void 0){if(!Number.isInteger(e.keyCount)||e.keyCount<1||e.keyCount>=4294967295)throw Error(`${t} keyCount must be an integer in [1, 2^32 - 2]`);if(n.format!==`uint32`)throw Error(`${t} keyCount needs uint32 keys`);if(a.keys.length!==e.keyCount)throw Error(`${t} output.keys must hold exactly keyCount rows`)}l(a.keys,[n.format],`${t} output.keys`);let s=a.keys.length;if(m(t,`output.keys.length`,s,1,de),l(a.counts,[`uint32`],`${t} output.counts`),a.counts.length!==s)throw Error(`${t} output.counts must have the same length as output.keys`);for(let e of[`count`,`overflow`,`requiredCount`]){let n=a.status[e];if(n&&(l(n,[`uint32`],`${t} output.status.${e}`),n.length<1))throw Error(`${t} output.status.${e} must contain one uint32 row`)}let c=0;for(let[n,r]of i.entries()){let i=`columns[${n}]`;for(let e of p(r.values))l(e,[`float32`],`${t} ${i}.values`);if(r.values.length!==o)throw Error(`${t} ${i}.values must have the same length as keys`);d(t,r.values.format===`float32`,`${i}.values.format must be float32`);let a=new Set(r.statistics);for(let e of a)if(!(e in $))throw Error(`${t} ${i} has unknown statistic '${e}'`);for(let e of a){let n=$[e];if(!n.some(e=>r.output[e]))throw Error(`${t} ${i} statistic '${e}' needs output.${n.join(` or output.`)}`)}for(let[e,n]of Object.entries(r.output)){if(!n)continue;let r=Object.entries($).find(([,t])=>t.includes(e));if(!r)throw Error(`${t} ${i} has unknown output.${e}`);if(!a.has(r[0]))throw Error(`${t} ${i} output.${e} requires statistic '${r[0]}'`)}let u=r.output;for(let e of[`counts`,`uniqueCounts`]){let n=u[e];n&&l(n,[`uint32`],`${t} ${i}.output.${e}`)}u.sums&&l(u.sums,[`uint32x2`],`${t} ${i}.output.sums`);for(let e of _e){let n=u[e];n&&l(n,[`float32`],`${t} ${i}.output.${e}`)}let f=[[`zScores`,o]];for(let e of[`counts`,`sums`,`sumValues`,`means`,`minimums`,`maximums`,`variances`,`standardDeviations`,`skewness`,`kurtosis`,`medians`,`modes`,`uniqueCounts`])f.push([e,s]);for(let[e,n]of f){let r=u[e];if(r&&r.length!==n)throw Error(`${t} ${i}.output.${e} must have ${e===`zScores`?`the same length as keys`:`the same length as output.keys`}`)}if(a.has(`percentiles`)){if(c++,!e.percentiles)throw Error(`${t} ${i} statistic 'percentiles' needs the percentiles view`);let n=u.percentiles;if(n&&n.length!==s*e.percentiles.length)throw Error(`${t} ${i}.output.percentiles must hold output.keys.length * percentiles.length rows`)}}if(e.percentiles&&(l(e.percentiles,[`float32`],`${t} percentiles`),m(t,`percentiles.length`,e.percentiles.length,1,fe),m(t,`output.keys.length * percentiles.length`,s*e.percentiles.length,1,pe),c===0))throw Error(`${t} percentiles is only used by the 'percentiles' statistic`);b(t,this.getOutputViews(),this.getInputViews()),this.validateOutputViews(),d(t,e.scratchMode===void 0||e.scratchMode===`auto`||e.scratchMode===`packed`,`scratchMode must be auto or packed`);let u=p(n)[0].buffer.graph.device.limits,g=Math.min(u.maxStorageBufferBindingSize,u.maxBufferSize);this.scratchTileSize=e.scratchTileSize??2**Math.floor(Math.log2(Math.max(1,Math.min(Z,g/16)))),m(t,`scratchTileSize`,this.scratchTileSize,1,Z),v(t,`scratchTileSize`,this.scratchTileSize);let _=i.some(e=>e.statistics.some(e=>[`median`,`percentiles`,`mode`,`uniqueCount`,`zScore`].includes(e)));this.usesTiledScratch=!_&&e.scratchMode!==`packed`&&(e.scratchTileSize===void 0?o*4>g:o>this.scratchTileSize);let y=Math.ceil(o/this.scratchTileSize);this.usesTiledScratch&&e.keyCount===void 0&&F(t,y*this.scratchTileSize),this.estimatedCommandNodeCount=2576+p(n).length+i.reduce((e,t)=>{let n=t.statistics.some(e=>[`median`,`percentiles`,`mode`,`uniqueCount`].includes(e)),r=t.statistics.some(e=>[`variance`,`standardDeviation`,`skewness`,`kurtosis`,`zScore`].includes(e));return e+32+5*p(t.values).length+(n?9*me:0)+(r?Math.ceil(Math.log2(o))+2:0)},0),this.usesTiledScratch&&(this.estimatedCommandNodeCount=(ce+le*i.length)*y+6*y*y+8*(p(n).length+i.reduce((e,t)=>e+p(t.values).length,0))),h(t,this.estimatedCommandNodeCount)}validateOutputViews(){let{id:e,props:t}=this,n=this.getOutputViews().flatMap(e=>e?[e.buffer.graph.createDataView(e.buffer,{format:`uint32`,length:e.length*e.rowByteLength/4,byteOffset:e.byteOffset})]:[]),r=n.reduce((e,t)=>e+t.length,0);S(e,{"output fields":new a({id:`${e}-output-fields`,name:`output fields`,format:`uint32`,length:r,valueLength:r,stride:1,byteStride:4,rowByteLength:4,data:n})}),S(e,{"output.keys":t.output.keys,"output.counts":t.output.counts}),S(e,{"output.status.count":t.output.status.count,"output.status.overflow":t.output.status.overflow,"output.status.requiredCount":t.output.status.requiredCount});for(let[n,r]of t.columns.entries()){let t=`columns[${n}].output`,i=r.output;for(let n of[[`counts`,`sumValues`,`means`],[`minimums`,`maximums`],[`variances`,`standardDeviations`,`skewness`,`kurtosis`]])S(e,Object.fromEntries(n.map(e=>[`${t}.${e}`,i[e]])))}}getInputViews(){let{props:e}=this;return[e.keys,e.mask,e.percentiles,...e.columns.map(e=>e.values)]}getOutputViews(){let{output:e}=this.props;return[e.keys,e.counts,e.status.count,e.status.overflow,e.status.requiredCount,...this.props.columns.flatMap(e=>Object.values(e.output))]}getCommandNodes(e){let{id:t,props:i}=this,a=[...this.getInputViews(),...this.getOutputViews()].flatMap(e=>e?p(e).filter(e=>e.length>0):[]);_(e.device,t,`keys.length row scratch`,(this.usesTiledScratch?this.scratchTileSize:i.keys.length)*4,`reduce keys.length (exact order statistics and zScore retain the packed row-scratch limit)`);for(let n of a)_(e.device,t,`input/output chunk`,r(n).size,`reduce the chunk length or output capacity`);y(t,e,[...this.getInputViews(),...this.getOutputViews()]),x(e,t,Object.fromEntries(a.map((e,t)=>[`bound field ${t}`,e])));for(let n of a)m(t,`input/output buffer byteLength`,n.buffer.byteLength,0,e.device.limits.maxBufferSize);let o=i.output.keys.length;u(t,`output.keys rows`,o,he(e.device)),i.percentiles&&u(t,`output.keys rows x percentiles`,o*i.percentiles.length,ge(e.device));let s=[[`group offsets`,o+1],[`group counts`,o]];for(let[e,t]of i.columns.entries()){let n=t.statistics.some(e=>[`variance`,`standardDeviation`,`skewness`,`kurtosis`,`zScore`].includes(e));(t.output.sumValues||t.output.means||n)&&s.push([`columns[${e}] fixed-point words`,3*o+1]),n&&s.push([`columns[${e}] mean scratch`,2*o],[`columns[${e}] moment sums`,4*o])}for(let[n,r]of s)_(e.device,t,n,r*4,`reduce output capacity`),m(t,`${n} byteLength`,r*4,0,e.device.limits.maxBufferSize);if(d(t,64<=e.device.limits.maxUniformBufferBindingSize,`dispatch parameters must fit maxUniformBufferBindingSize`),this.usesTiledScratch){for(let[n,r]of[[`tile moment scratch`,this.scratchTileSize*16],[`tile group offsets`,(this.scratchTileSize+1)*4],[`union keys`,o*8]])_(e.device,t,n,r,`reduce scratchTileSize or output capacity`),m(t,`${n} byteLength`,r,0,e.device.limits.maxBufferSize);return P(e,t),se(e,i,t,this.scratchTileSize,this.sumScale,this.variance)}let{nodes:c,structure:l}=Y(e,{id:t,operation:X,keys:i.keys,mask:i.mask,keyCount:i.keyCount,output:i.output}),f=[...c];for(let[r,a]of i.columns.entries()){let o=`${t}-c${r}`,s=new Set(a.statistics),c=a.output,u=0,d=p(a.values).flatMap(t=>{let n=u;return u+=t.length,t.length===0?[]:[{rowOffset:n,valueBits:e.createDataView(t.buffer,{format:`uint32`,length:t.length,byteOffset:t.byteOffset})}]}),m=[`variance`,`standardDeviation`,`skewness`,`kurtosis`].some(e=>s.has(e)),h=q(e,{id:o,operation:X,structure:l,valueChunks:d,needs:{sums:s.has(`sum`)||s.has(`mean`),minimum:s.has(`minimum`),maximum:s.has(`maximum`),moments:m,zScore:s.has(`zScore`)},variance:this.variance,sumScale:this.sumScale,output:c});if(f.push(...h.nodes),s.has(`median`)||s.has(`percentiles`)||s.has(`mode`)||s.has(`uniqueCount`)){let t=d.length===1?d[0].valueBits:n(e,`${o}-value-bits`,`uint32`,l.rowCount);if(d.length>1)for(let[n,r]of d.entries())f.push(R(e,{id:`${o}-gather-value-bits-${n}`,operation:X,variant:`gather-value-bits`,bindings:[B(`source`,r.valueBits),V(`valueBits`,t)],invocationCount:r.valueBits.length,rowOffset:r.rowOffset,body:`valueBits[valueBitsOffset + index + GROUP_ROW_OFFSET] = source[sourceOffset + index];`}));f.push(...oe(e,{id:o,operation:X,structure:l,valueBits:t,finiteCounts:h.finiteCounts,percentiles:i.percentiles,output:c}))}}return f}};export{ve as t};