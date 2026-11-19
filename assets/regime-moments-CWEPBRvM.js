import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-sort-xRm8l5yn.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as r}from"./sorted-segment-offsets-ChmOtA9F.js";import{a as i}from"./sorted-segment-sums-Dvmm6OdR.js";import{c as a}from"./wgsl-common-DfyXiYB0.js";import{t as o}from"./rounded-difference-wgsl-DgXXQtNj.js";var s=256;function c(e){let t=e+1;return t+t*(t+1)/2+t}function l(l,u){let{id:d,operation:f,predictorCount:p,regimeCount:m}=u,h=u.response.length,g=p+1,_=g*(g+1)/2,v=c(p),y=e(l,`${d}-regime-keys`,`uint32`,h),b=e(l,`${d}-regime-rows`,`uint32`,h),x=e(l,`${d}-sorted-regime-keys`,`uint32`,h),S=e(l,`${d}-sorted-regime-rows`,`uint32`,h),C=e(l,`${d}-regime-offsets`,`uint32`,m+1),w=e(l,`${d}-regime-moments`,`float32`,m*v);return{nodes:[n(l,{id:`${d}-regime-keys`,operation:f,variant:`regime-keys`,invocationCount:h,bindings:[{name:`predictors`,view:u.predictors,type:`f32`,access:`read`},{name:`response`,view:u.response,type:`f32`,access:`read`},{name:`regimes`,view:u.regimes,type:`u32`,access:`read`},...u.mask?[{name:`mask`,view:u.mask,type:`u32`,access:`read`}]:[],{name:`sortKeys`,view:y,type:`u32`,access:`read_write`},{name:`sortRows`,view:b,type:`u32`,access:`read_write`}],declarations:a,body:`var included = regimes[regimesOffset + index] < ${m}u &&
    isFiniteBits(response[responseOffset + index]);
  ${u.mask?`included = included && mask[maskOffset + index] != 0u;`:``}
  for (var column = 0u; column < ${p}u; column++) {
    included = included && isFiniteBits(predictors[predictorsOffset + index * ${p}u + column]);
  }
  sortKeys[sortKeysOffset + index] = select(${m}u, regimes[regimesOffset + index], included);
  sortRows[sortRowsOffset + index] = index;`}),...new t({id:`${d}-regime-sort`,keys:y,values:b,outputKeys:x,outputValues:S,keyBits:i(m)}).getCommandNodes(l),r(l,{id:`${d}-regime-offsets`,operation:f,variant:`regime-offsets`,segmentCount:m,sortedKeys:x,segmentOffsets:C}),n(l,{id:`${d}-regime-moments`,operation:f,variant:`regime-moments`,workgroupSize:s,invocationCount:m*s,guardIndex:!1,bindings:[{name:`predictors`,view:u.predictors,type:`f32`,access:`read`},{name:`response`,view:u.response,type:`f32`,access:`read`},{name:`sortedRows`,view:S,type:`u32`,access:`read`},{name:`segmentOffsets`,view:C,type:`u32`,access:`read`},{name:`moments`,view:w,type:`f32`,access:`read_write`}],declarations:`const K: u32 = ${p}u;
const D: u32 = ${g}u;
const MOMENT_COUNT: u32 = ${_}u;
const STRIDE: u32 = ${v}u;
const REGIME_COUNT: u32 = ${m}u;
${o}
var<workgroup> partialSums: array<f32, ${s}>;
var<workgroup> regimeMeans: array<f32, ${g}>;

fn getValue(row: u32, column: u32) -> f32 {
  if (column == K) {
    return response[responseOffset + row];
  }
  return predictors[predictorsOffset + row * K + column];
}

// Fixed binary tree over the workgroup (strides 128 to 1); every thread must call it.
fn reduceWorkgroup(localIndex: u32, value: f32) -> f32 {
  partialSums[localIndex] = value;
  workgroupBarrier();
  for (var stride = ${s/2}u; stride > 0u; stride = stride / 2u) {
    if (localIndex < stride) {
      partialSums[localIndex] = partialSums[localIndex] + partialSums[localIndex + stride];
    }
    workgroupBarrier();
  }
  let total = partialSums[0];
  workgroupBarrier();
  return total;
}`,body:`let regime = index / ${s}u;
  let isInRange = regime < REGIME_COUNT;
  var begin = 0u;
  var end = 0u;
  if (isInRange) {
    begin = segmentOffsets[segmentOffsetsOffset + regime];
    end = segmentOffsets[segmentOffsetsOffset + regime + 1u];
  }
  let count = f32(end - begin);
  let runtimeZero = segmentOffsets[segmentOffsetsOffset];
  var origins: array<f32, ${g}>;
  if (end > begin) {
    let originRow = sortedRows[sortedRowsOffset + begin];
    for (var column = 0u; column < D; column++) { origins[column] = getValue(originRow, column); }
  }
  var sums: array<f32, ${g}>;
  for (var slot = begin + localInvocationIndex; slot < end; slot += ${s}u) {
    let row = sortedRows[sortedRowsOffset + slot];
    for (var column = 0u; column < D; column++) {
      sums[column] += getRoundedDifference(getValue(row, column), origins[column], runtimeZero);
    }
  }
  for (var column = 0u; column < D; column++) {
    let total = reduceWorkgroup(localInvocationIndex, sums[column]);
    if (localInvocationIndex == 0u) {
      regimeMeans[column] = select(0.0, origins[column] + total / max(count, 1.0), end > begin);
    }
  }
  workgroupBarrier();
  var means: array<f32, ${g}>;
  for (var column = 0u; column < D; column++) {
    means[column] = regimeMeans[column];
  }
  var products: array<f32, ${_}>;
  var centered: array<f32, ${g}>;
  var deviationSums: array<f32, ${g}>;
  // The first segment starts at sorted row 0, so this buffer word is a zero the compiler cannot
  // know. It fixes each deviation as written: summed as (sum of values) - count * mean, the
  // deviation sums and the centered products would lose everything below a float32 step.
  for (var slot = begin + localInvocationIndex; slot < end; slot += ${s}u) {
    let row = sortedRows[sortedRowsOffset + slot];
    for (var column = 0u; column < D; column++) {
      centered[column] = getRoundedDifference(getValue(row, column), means[column], runtimeZero);
      deviationSums[column] = deviationSums[column] + centered[column];
    }
    var moment = 0u;
    for (var first = 0u; first < D; first++) {
      for (var second = first; second < D; second++) {
        products[moment] = products[moment] + centered[first] * centered[second];
        moment = moment + 1u;
      }
    }
  }
  let base = momentsOffset + min(regime, REGIME_COUNT - 1u) * STRIDE;
  if (isInRange && localInvocationIndex == 0u) {
    for (var column = 0u; column < D; column++) {
      moments[base + column] = means[column];
    }
  }
  for (var moment = 0u; moment < MOMENT_COUNT; moment++) {
    let total = reduceWorkgroup(localInvocationIndex, products[moment]);
    if (isInRange && localInvocationIndex == 0u) {
      moments[base + D + moment] = total;
    }
  }
  for (var column = 0u; column < D; column++) {
    let total = reduceWorkgroup(localInvocationIndex, deviationSums[column]);
    if (isInRange && localInvocationIndex == 0u) {
      moments[base + D + MOMENT_COUNT + column] = total;
    }
  }`})],views:{segmentOffsets:C,moments:w}}}export{c as n,l as t};