import{r as e}from"./kernel-C3tfQOn4.js";import{o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{v as n}from"./gpu-contributor-utils-BIwLQON5.js";import{r}from"./wgsl-common-DfyXiYB0.js";import{t as i}from"./addition-rounding-error-wgsl-DVwPAqKC.js";import{t as a}from"./multiplication-rounding-error-wgsl-P721A62W.js";import{n as o}from"./regression-row-tiles-7Xsw3Utq.js";import{a as s}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as c}from"./rounded-difference-wgsl-DgXXQtNj.js";var l=`${c}
${i}
${a}
fn addPanelCenters(first: vec2<f32>, second: vec2<f32>) -> vec2<f32> {
  let high = first.x + second.x;
  let low = first.y + second.y + getAdditionRoundingError(first.x, second.x, high);
  let combined = high + low;
  return vec2<f32>(combined, getAdditionRoundingError(high, low, combined));
}`;function u(e,t,n){let{tileCount:r}=o(t),i=e*(n+1);return{partialLength:i*r*2,meanLength:i*2+1,tileCount:r}}function d(i,a){let{entityCount:c,periodCount:d,predictorCount:f,id:p}=a,{partialLength:m,meanLength:h,tileCount:g}=u(c,d,f);s(i,p,{predictors:a.predictors,response:a.response,overflow:a.overflow,outputPredictors:a.output.predictors,outputResponse:a.output.response});let{tileRowCount:_}=o(d),v=2**-Math.ceil(Math.log2(d)),y=d*v;for(let[e,t]of[[`panel mean partials`,m],[`panel means`,h]])n(i.device,p,e,t*4,`reduce panel dimensions`);let b=e(i,`${p}-mean-partials`,`float32`,m),x=e(i,`${p}-means`,`float32`,h),S=[{name:`sourcePredictors`,view:a.predictors,type:`f32`,access:`read`},{name:`sourceResponse`,view:a.response,type:`f32`,access:`read`}],C=`${l}
fn readPanelValue(entity: u32, period: u32, component: u32) -> f32 {
  let row = period * ${c}u + entity;
  if (component == 0u) { return sourceResponse[sourceResponseOffset + row]; }
  return sourcePredictors[sourcePredictorsOffset + row * ${f}u + component - 1u];
}`;return[t(i,{id:`${p}-mean-initialize`,operation:`GPUSpatialPanelErrorMaximumLikelihood`,variant:`within-mean-initialize`,invocationCount:1,bindings:[{name:`means`,view:x,type:`f32`,access:`read_write`}],body:`means[meansOffset + ${h-1}u] = 0.0;`}),t(i,{id:`${p}-mean-tiles`,operation:`GPUSpatialPanelErrorMaximumLikelihood`,variant:`within-mean-tiles`,invocationCount:m/2,workgroupSize:64,bindings:[...S,{name:`means`,view:x,type:`f32`,access:`read`},{name:`partial`,view:b,type:`f32`,access:`read_write`}],declarations:C,body:`let tile = index % ${g}u;
let column = index / ${g}u;
let entity = column / ${f+1}u;
let component = column % ${f+1}u;
let runtimeZero = bitcast<u32>(means[meansOffset + ${h-1}u]);
var total = vec2<f32>(0.0);
for (var period = tile * ${_}u; period < min((tile + 1u) * ${_}u, ${d}u); period++) {
  let value = getRoundedDifference(readPanelValue(entity, period, component) * ${v}, 0.0, runtimeZero);
  total = addPanelCenters(total, vec2<f32>(value, 0.0));
}
partial[partialOffset + index * 2u] = total.x;
partial[partialOffset + index * 2u + 1u] = total.y;`}),t(i,{id:`${p}-mean-merge`,operation:`GPUSpatialPanelErrorMaximumLikelihood`,variant:`within-mean-merge`,invocationCount:(h-1)/2,workgroupSize:64,bindings:[{name:`partial`,view:b,type:`f32`,access:`read`},{name:`means`,view:x,type:`f32`,access:`read_write`}],declarations:l,body:`var levels: array<vec2<f32>, 32>;
var depth = 0u;
for (var tile = 0u; tile < ${g}u; tile++) {
  let start = partialOffset + (index * ${g}u + tile) * 2u;
  var centerBlock = vec2<f32>(partial[start], partial[start + 1u]);
  var carry = tile;
  for (var level = 0u; level < 32u; level++) {
    if ((carry & 1u) == 0u) { break; }
    depth--;
    centerBlock = addPanelCenters(levels[depth], centerBlock);
    carry >>= 1u;
  }
  levels[depth] = centerBlock;
  depth++;
}
var total = vec2<f32>(0.0);
for (var level = 0u; level < 32u; level++) {
  if (depth == 0u) { break; }
  depth--;
  total = addPanelCenters(levels[depth], total);
}
let high = total.x / ${y};
let product = high * ${y};
let runtimeZero = bitcast<u32>(means[meansOffset + ${h-1}u]);
let remainder = getRoundedDifference(total.x, product, runtimeZero)
  - getMultiplicationRoundingError(high, ${y}, product) + total.y;
means[meansOffset + index * 2u] = high;
means[meansOffset + index * 2u + 1u] = remainder / ${y};`}),t(i,{id:`${p}-within`,operation:`GPUSpatialPanelErrorMaximumLikelihood`,variant:`unit-demean`,invocationCount:c*d,workgroupSize:64,bindings:[...S,{name:`means`,view:x,type:`f32`,access:`read`},{name:`predictors`,view:a.output.predictors,type:`f32`,access:`read_write`},{name:`response`,view:a.output.response,type:`f32`,access:`read_write`},...a.overflow?[{name:`overflow`,view:a.overflow,type:`u32`,access:`read`}]:[]],declarations:`${C}\n${r}`,body:`let entity = index % ${c}u;
let period = index / ${c}u;
let runtimeZero = bitcast<u32>(means[meansOffset + ${h-1}u]);
for (var component = 0u; component < ${f+1}u; component++) {
  let center = meansOffset + (entity * ${f+1}u + component) * 2u;
  let spread = getRoundedDifference(readPanelValue(entity, period, component), means[center], runtimeZero);
  let centered = getRoundedDifference(spread, means[center + 1u], runtimeZero);
  if (component == 0u) { response[responseOffset + index] = centered; }
  else { predictors[predictorsOffset + index * ${f}u + component - 1u] = centered; }
}
${a.overflow?`if (overflow[overflowOffset] != 0u) { response[responseOffset + index] = getNaN(); }`:``}`})]}export{u as n,d as t};