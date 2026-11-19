import{n as e,t}from"./gpu-variogram-BZ1L8I4i.js";import{a as n,i as r,s as i}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as a}from"./construction-snapshot-DdkX9Pc0.js";import{E as o,S as s,T as c,f as l,p as u,v as d,x as f,y as p}from"./gpu-contributor-utils-BIwLQON5.js";import{c as m,o as h,r as g,t as _}from"./kernel-C3tfQOn4.js";import{c as v,o as y}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as ee,u as te}from"./wgsl-common-DfyXiYB0.js";import{t as b}from"./gpu-prediction-utils-DjZFSzGW.js";import{r as x}from"./variogram-model-l9c6-v8f.js";import{t as S}from"./define-run-B-brkCe4.js";import{r as C}from"./gpu-command-graph-BdzdmV6u.js";import{n as w,t as T}from"./gpu-dispatch-utils-CbRzFAgT.js";import{l as E,o as D,r as O,s as k,t as A,u as j}from"./distance-probe-BxIfRi4a.js";import{a as ne,h as re}from"./helpers-DPAd8TjN.js";import{oc as ie}from"./entry-preview-D70TdSbK.js";var M=`
struct FitSums {
  moments: vec4f,
  weightedCross: f32,
}
var<workgroup> fitSums: array<FitSums, 64>;
var<workgroup> sharedBest: vec4f;
var<workgroup> sharedBestStep: u32;
var<workgroup> sharedValid: u32;
var<workgroup> sharedLimits: vec2f;

fn reduceFitSums(lane: u32) -> FitSums {
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      fitSums[lane].moments += fitSums[lane + stride].moments;
      fitSums[lane].weightedCross += fitSums[lane + stride].weightedCross;
    }
    workgroupBarrier();
  }
  let total = workgroupUniformLoad(&fitSums[0]);
  workgroupBarrier();
  return total;
}

fn fitLinearCooperative(range: f32, lane: u32) -> vec4f {
  // Pass 0: total weight. Cressie weights are 1/max(fitted^2, 1e-20) and can reach 1e20, so the
  // normal-equation products weightSum * squareSum would overflow f32. Every weight is divided by
  // the total, which bounds the normalised weightSum by 1 and the determinant by 1.
  var totalWeight = 0.0;
  for (var bin = lane; bin < BIN_COUNT; bin += 64u) {
    if (usable(bin)) {
      totalWeight += weight(bin);
    }
  }
  fitSums[lane] = FitSums(vec4f(totalWeight, 0.0, 0.0, 0.0), 0.0);
  let weightTotal = reduceFitSums(lane).moments.x;
  let weightScale = select(1.0, 1.0 / weightTotal, weightTotal > 0.0 && weightTotal < 3.0e38);
  var moments = vec4f(0.0);
  var weightedCross = 0.0;
  for (var bin = lane; bin < BIN_COUNT; bin += 64u) {
    if (usable(bin)) {
      let binWeight = weight(bin) * weightScale;
      let binShape = shape(distances[distancesOffset + bin] / range);
      let value = semivariances[semivariancesOffset + bin];
      moments += vec4f(binWeight, binWeight * binShape, binWeight * binShape * binShape, binWeight * value);
      weightedCross += binWeight * binShape * value;
    }
  }
  fitSums[lane] = FitSums(moments, weightedCross);
  let sums = reduceFitSums(lane);
  let weightSum = sums.moments.x;
  let shapeSum = sums.moments.y;
  let squareSum = sums.moments.z;
  let valueSum = sums.moments.w;
  let crossSum = sums.weightedCross;
  let nuggetOnly = max(valueSum / weightSum, 0.0);
  let sillOnly = max(crossSum / max(squareSum, 1e-30), 0.0);
  let determinant = weightSum * squareSum - shapeSum * shapeSum;
  var nugget = 0.0;
  var sill = 0.0;
  var jointValid = false;
  if (determinant > 1e-6 * weightSum * squareSum) {
    nugget = (squareSum * valueSum - shapeSum * crossSum) / determinant;
    sill = (weightSum * crossSum - shapeSum * valueSum) / determinant;
    jointValid = nugget >= 0.0 && sill >= 0.0;
  }
  var errors = vec3f(0.0);
  for (var bin = lane; bin < BIN_COUNT; bin += 64u) {
    if (usable(bin)) {
      let binWeight = weight(bin) * weightScale;
      let binShape = shape(distances[distancesOffset + bin] / range);
      let value = semivariances[semivariancesOffset + bin];
      let differences = vec3f(value - nuggetOnly, value - sillOnly * binShape, value - nugget - sill * binShape);
      errors += binWeight * differences * differences;
    }
  }
  fitSums[lane] = FitSums(vec4f(errors, 0.0), 0.0);
  let residuals = reduceFitSums(lane).moments;
  var best = vec4f(nuggetOnly, 0.0, range, residuals.x);
  if (residuals.y < best.w) { best = vec4f(0.0, sillOnly, range, residuals.y); }
  if (jointValid && residuals.z < best.w) { best = vec4f(nugget, sill, range, residuals.z); }
  // Report the residual in the caller's unnormalised weights.
  best.w = best.w / weightScale;
  return best;
}
`,N=64,P=2,F={spherical:0,exponential:1,gaussian:2,"matern-3/2":3,"matern-5/2":4},I=2**-23,L=`1.8446743e19`;function R(e,t){let{id:r,props:a,models:o,rangeSteps:c,maximumIterations:l}=t;n(e,r,{distances:a.distances,semivariances:a.semivariances,pairCounts:a.pairCounts,...a.output}),s(r,`storage bindings`,8,e.device.limits.maxStorageBuffersPerShaderStage);let u=a.weighting===void 0||a.weighting===`cressie`?4:1,f=4+o.length*(c+1)*4;d(e.device,r,`search scratch`,f*4,`reduce rangeSteps or models length`),s(r,`search scratch byteLength`,f*4,e.device.limits.maxBufferSize),d(e.device,r,`round parameter slot`,8,`use a device with an eight-byte storage binding`),s(r,`round parameter buffer byteLength`,u*N*4,e.device.limits.maxBufferSize);let p=new Uint32Array(u*N);for(let e=0;e<u;e++)p.set([Number(e>0),Number(e===u-1)],e*N);let m=i(e,`${r}-round-parameters`,p),h=o.map(e=>F[e]),_=4+o.length*4,S=g(e,`${r}-search`,`float32`,_+o.length*c*4),C=[{name:`distances`,view:a.distances,type:`f32`,access:`read`},{name:`semivariances`,view:a.semivariances,type:`f32`,access:`read`},{name:`pairCounts`,view:a.pairCounts,type:`u32`,access:`read`},{name:`scratch`,view:S,type:`f32`,access:`read_write`},{name:`parameters`,view:a.output.parameters,type:`f32`,access:`read_write`},{name:`diagnostics`,view:a.output.diagnostics,type:`f32`,access:`read_write`},{name:`statuses`,view:a.output.status,type:`u32`,access:`read_write`}],w=`${b({},[`parameters`,`diagnostics`])}
${ee}
${te}
const BIN_COUNT: u32 = ${a.distances.length}u;
const MODEL_CODES: array<u32, ${o.length}> = array<u32, ${o.length}>(${h.map(e=>`${e}u`).join(`,`)});
fn readFit(offset: u32) -> vec4f {
  return vec4f(scratch[scratchOffset + offset], scratch[scratchOffset + offset + 1u], scratch[scratchOffset + offset + 2u], scratch[scratchOffset + offset + 3u]);
}
fn writeFit(offset: u32, value: vec4f) {
  scratch[scratchOffset + offset] = value.x;
  scratch[scratchOffset + offset + 1u] = value.y;
  scratch[scratchOffset + offset + 2u] = value.z;
  scratch[scratchOffset + offset + 3u] = value.w;
}
var<private> model: u32;
var<private> previous: vec4<f32>;
var<private> reweight: bool;
${x}
// The shared float32 shapes: they keep their digits for bins at a small fraction of a candidate
// range, where the direct 1 - exp(-x) forms cancel (interpolation-density-08).
fn shape(scaled: f32) -> f32 { return getVariogramModelShape(model, scaled); }
fn usable(bin: u32) -> bool { return pairCounts[pairCountsOffset + bin] > 0u && isFiniteValue(distances[distancesOffset + bin]) && distances[distancesOffset + bin] > 0.0 && isFiniteValue(semivariances[semivariancesOffset + bin]); }
fn weight(bin: u32) -> f32 {
  var value = ${a.weighting===`none`?`1.0`:`f32(pairCounts[pairCountsOffset + bin])`};
  if (reweight) { let fitted = previous.x + previous.y * shape(distances[distancesOffset + bin] / previous.z); value /= max(fitted * fitted, 1e-20); }
  return value;
}
fn residual(nugget: f32, sill: f32, range: f32) -> f32 {
  var value = 0.0;
  for (var bin = 0u; bin < BIN_COUNT; bin++) { if (usable(bin)) { let difference = semivariances[semivariancesOffset + bin] - nugget - sill * shape(distances[distancesOffset + bin] / range); value += weight(bin) * difference * difference; } }
  return value;
}
fn fitLinear(range: f32) -> vec4<f32> {
  // Cressie weights are N / max(fitted^2, 1e-20) and reach 1e20 for semivariances near 1e-8, so
  // the normal-equation products weightSum * squareSum would leave f32 and the joint nugget and
  // sill fit would be skipped. Every weight is divided by the total first, as the cooperative
  // refinement does; the residual below keeps the caller's unnormalised weights.
  var totalWeight = 0.0;
  for (var bin = 0u; bin < BIN_COUNT; bin++) { if (usable(bin)) { totalWeight += weight(bin); } }
  let weightScale = select(1.0, 1.0 / totalWeight, totalWeight > 0.0 && totalWeight < 3.0e38);
  var weightSum = 0.0; var shapeSum = 0.0; var squareSum = 0.0; var valueSum = 0.0; var crossSum = 0.0;
  for (var bin = 0u; bin < BIN_COUNT; bin++) {
    if (!usable(bin)) { continue; }
    let binWeight = weight(bin) * weightScale; let binShape = shape(distances[distancesOffset + bin] / range); let value = semivariances[semivariancesOffset + bin];
    weightSum += binWeight; shapeSum += binWeight * binShape; squareSum += binWeight * binShape * binShape; valueSum += binWeight * value; crossSum += binWeight * binShape * value;
  }
  let nuggetOnly = max(valueSum / weightSum, 0.0);
  let sillOnly = max(crossSum / max(squareSum, 1e-30), 0.0);
  var best = vec4<f32>(nuggetOnly, 0.0, range, residual(nuggetOnly, 0.0, range));
  let sillResidual = residual(0.0, sillOnly, range);
  if (sillResidual < best.w) { best = vec4<f32>(0.0, sillOnly, range, sillResidual); }
  let determinant = weightSum * squareSum - shapeSum * shapeSum;
  if (determinant > 1e-6 * weightSum * squareSum) {
    let nugget = (squareSum * valueSum - shapeSum * crossSum) / determinant;
    let sill = (weightSum * crossSum - shapeSum * valueSum) / determinant;
    if (nugget >= 0.0 && sill >= 0.0) { let error = residual(nugget, sill, range); if (error < best.w) { best = vec4<f32>(nugget, sill, range, error); } }
  }
  return best;
}
`,T=[...m.nodes,y(e,{id:`${r}-prepare`,operation:`GPUVariogramModelFit`,variant:`prepare-search`,invocationCount:1,workgroupSize:2,bindings:C,declarations:w,body:`
var minimumLag = 3.402823e38;
var maximumLag = 0.0;
var binCount = 0u;
var largestValue = 0.0;
var baseWeight = 0.0;
reweight = false;
for (var bin = 0u; bin < BIN_COUNT; bin++) {
  if (usable(bin)) {
    minimumLag = min(minimumLag, distances[distancesOffset + bin]);
    maximumLag = max(maximumLag, distances[distancesOffset + bin]);
    largestValue = max(largestValue, abs(semivariances[semivariancesOffset + bin]));
    baseWeight += weight(bin);
    binCount++;
  }
}
let valid = binCount >= 2u && maximumLag > minimumLag;
// The objective is a weighted sum of squared residuals in the caller's weights. A residual
// cannot be smaller than the float32 resolution of the largest semivariance (2^-23 of it), so
// when that resolution, squared and summed with the weights, is not finite, the objective of
// even an exact fit is a matter of rounding (0 or infinity) and candidates cannot be ranked.
// Such a frame is a numerical failure for every model. The test is written without the
// product so that it does not rely on overflow: resolution >= sqrt(FLT_MAX / weight).
let resolution = largestValue * ${v(I)};
let isRepresentable = !(resolution >= ${L} / sqrt(max(baseWeight, 1.0)));
writeFit(0u, vec4f(0.0, 0.0, f32(binCount), 0.0));
if (valid) {
  let lowest = log(minimumLag / 10.0);
  let highest = log(3.0 * maximumLag);
  scratch[scratchOffset] = lowest;
  scratch[scratchOffset + 1u] = (highest - lowest) / ${c-1}.0;
}
for (var modelIndex = 0u; modelIndex < ${o.length}u; modelIndex++) {
  write_parameters(modelIndex, vec4f(getNaN()));
  write_diagnostics(modelIndex, vec4f(getNaN()));
  statuses[statusesOffset + modelIndex] = select(1u, select(2u, 0u, isRepresentable), valid);
  writeFit(4u + modelIndex * 4u, vec4f(0.0, 0.0, 1.0, 3.402823e38));
}`})];for(let t=0;t<u;t++){let n=e.createDataView(m.parameters.buffer,{format:`uint32`,length:P,byteOffset:t*N*4}),i=[...C,{name:`roundParameters`,view:n,type:`u32`,access:`read`}];T.push(y(e,{id:`${r}-candidates-${t}`,operation:`GPUVariogramModelFit`,variant:`range-candidates`,invocationCount:o.length*c,workgroupSize:64,bindings:i,declarations:w,body:`
let modelIndex = index / ${c}u;
if (statuses[statusesOffset + modelIndex] != 0u) { return; }
model = MODEL_CODES[modelIndex];
previous = readFit(4u + modelIndex * 4u);
reweight = roundParameters[roundParametersOffset] != 0u;
let step = index % ${c}u;
let range = exp(scratch[scratchOffset] + scratch[scratchOffset + 1u] * f32(step));
writeFit(${_}u + index * 4u, fitLinear(range));`})),T.push(y(e,{id:`${r}-refine-${t}`,operation:`GPUVariogramModelFit`,variant:`refine-range`,invocationCount:o.length*64,workgroupSize:64,guardIndex:!1,bindings:i,declarations:w+M+`
var<workgroup> sharedPublishRound: u32;`,body:`
let modelIndex = workgroupIndex;
model = MODEL_CODES[modelIndex];
previous = readFit(4u + modelIndex * 4u);
reweight = roundParameters[roundParametersOffset] != 0u;
if (localInvocationIndex == 0u) {
  sharedPublishRound = roundParameters[roundParametersOffset + 1u];
  sharedValid = 0u;
  if (statuses[statusesOffset + modelIndex] == 0u) {
    var selected = previous;
    selected.w = 3.402823e38;
    var selectedStep = 0u;
    for (var step = 0u; step < ${c}u; step++) {
      let candidate = readFit(${_}u + (modelIndex * ${c}u + step) * 4u);
      if (candidate.w < selected.w) { selected = candidate; selectedStep = step; }
    }
    if (selected.w < 3.402823e38) {
      sharedBest = selected;
      sharedBestStep = selectedStep;
      sharedLimits = vec2f(scratch[scratchOffset], scratch[scratchOffset + 1u]);
      sharedValid = 1u;
    } else { statuses[statusesOffset + modelIndex] = 2u; }
  }
}
if (workgroupUniformLoad(&sharedValid) == 0u) { return; }
var best = workgroupUniformLoad(&sharedBest);
let bestStep = workgroupUniformLoad(&sharedBestStep);
let limits = workgroupUniformLoad(&sharedLimits);
let boundary = bestStep == 0u || bestStep == ${c-1}u;
let lowest = limits.x;
let stepSize = limits.y;
var left = lowest + stepSize * f32(select(bestStep - 1u, 0u, bestStep == 0u));
var rightBound = lowest + stepSize * f32(min(bestStep + 1u, ${c-1}u));
var inner = rightBound - 0.61803398875 * (rightBound - left);
var outer = left + 0.61803398875 * (rightBound - left);
var innerFit = fitLinearCooperative(exp(inner), localInvocationIndex);
var outerFit = fitLinearCooperative(exp(outer), localInvocationIndex);
for (var iteration = 0u; iteration < ${l}u; iteration++) {
  if (rightBound - left <= 1e-5) { break; }
  if (innerFit.w <= outerFit.w) {
    rightBound = outer; outer = inner; outerFit = innerFit;
    inner = rightBound - 0.61803398875 * (rightBound - left); innerFit = fitLinearCooperative(exp(inner), localInvocationIndex);
  } else {
    left = inner; inner = outer; innerFit = outerFit;
    outer = left + 0.61803398875 * (rightBound - left); outerFit = fitLinearCooperative(exp(outer), localInvocationIndex);
  }
}
if (innerFit.w < best.w) { best = innerFit; }
if (outerFit.w < best.w) { best = outerFit; }
if (localInvocationIndex == 0u) { writeFit(4u + modelIndex * 4u, best); }
if (workgroupUniformLoad(&sharedPublishRound) != 0u) {
let bracket = rightBound - left;
if (!isFiniteValue(best.w) || !isFiniteValue(best.x) || !isFiniteValue(best.y) || !isFiniteValue(best.z)) { if (localInvocationIndex == 0u) { statuses[statusesOffset + modelIndex] = 2u; } return; }
var partialUnweighted = 0.0;
for (var bin = localInvocationIndex; bin < BIN_COUNT; bin += 64u) {
  if (usable(bin)) {
    let difference = semivariances[semivariancesOffset + bin] - best.x - best.y * shape(distances[distancesOffset + bin] / best.z);
    partialUnweighted += difference * difference;
  }
}
fitSums[localInvocationIndex] = FitSums(vec4f(partialUnweighted, 0.0, 0.0, 0.0), 0.0);
let unweighted = reduceFitSums(localInvocationIndex).moments.x;
if (!isFiniteValue(unweighted) || !isFiniteValue(bracket)) { if (localInvocationIndex == 0u) { statuses[statusesOffset + modelIndex] = 2u; } return; }
if (localInvocationIndex == 0u) {
let binCount = scratch[scratchOffset + 2u];
write_parameters(modelIndex, best);
write_diagnostics(modelIndex, vec4f(binCount, unweighted, binCount * log(max(unweighted / binCount, 1e-30)) + 6.0, bracket));
statuses[statusesOffset + modelIndex] = select(0u, 3u, boundary || bracket > 1e-5);
}
}`}))}return T}var z=1024,B=64,V=5,H=4294967232,U=class{id;props;models;rangeSteps;maximumIterations;estimatedCommandNodeCount;constructor(e){let t=a(e,[`id`,`distances`,`semivariances`,`pairCounts`,`models`,`weighting`,`rangeSteps`,`maximumIterations`,`output`]);if(e=Object.freeze({...t,models:Object.freeze([...t.models??[`spherical`,`exponential`,`gaussian`]]),output:a(t.output,[`parameters`,`diagnostics`,`status`])}),this.props=e,this.id=e.id??`variogram-model-fit`,this.models=e.models??[`spherical`,`exponential`,`gaussian`],this.rangeSteps=e.rangeSteps??200,this.maximumIterations=e.maximumIterations??32,l(this.id,`distances length`,e.distances.length,0,H),l(this.id,`rangeSteps`,this.rangeSteps,3,z),l(this.id,`maximumIterations`,this.maximumIterations,0,B),l(this.id,`models length`,this.models.length,1,V),this.models.some(e=>![`spherical`,`exponential`,`gaussian`,`matern-3/2`,`matern-5/2`].includes(e))||![`pairs`,`none`,`cressie`].includes(e.weighting??`cressie`)||e.distances.length!==e.semivariances.length||e.distances.length!==e.pairCounts.length)throw Error(`${this.id} invalid model or bin topology`);for(let[t,n,r]of[[`distances`,e.distances,`float32`],[`semivariances`,e.semivariances,`float32`],[`pairCounts`,e.pairCounts,`uint32`],[`output.parameters`,e.output.parameters,`float32x4`],[`output.diagnostics`,e.output.diagnostics,`float32x4`],[`output.status`,e.output.status,`uint32`]])o(n,[r],`${this.id} ${t}`);for(let[t,n]of Object.entries(e.output))c(this.id,n.length>=this.models.length,`output.${t} length must be at least models length (${this.models.length})`);this.estimatedCommandNodeCount=2+2*((e.weighting??`cressie`)===`cressie`?4:1),u(this.id,this.estimatedCommandNodeCount),r(this.id,e.output),p(this.id,Object.values(e.output),[e.distances,e.semivariances,e.pairCounts])}getCommandNodes(e){let{id:t,props:n}=this;return f(t,e,[n.distances,n.semivariances,n.pairCounts,...Object.values(n.output)]),R(e,this)}},W=64,G=[`select(1.5 * q - 0.5 * q * q * q, 1.0, q >= 1.0)`,`1.0 - exp(-3.0 * q)`,`1.0 - exp(-3.0 * q * q)`],K=class{id;fitted;distances;curve;modelIndex;constructor(e,t,n,r,i){this.id=e,this.fitted=t,this.distances=n,this.curve=r,this.modelIndex=i}getCommandNodes(e){let{id:t,fitted:n,distances:r,curve:i,modelIndex:a}=this,o=T(t,i.length,W,e.device.limits.maxComputeWorkgroupsPerDimension),s=`
const fittedOffset: u32 = ${m(n)}u;
const distancesOffset: u32 = ${m(r)}u;
const curveOffset: u32 = ${m(i)}u;
@group(0) @binding(0) var<storage, read> fitted: array<vec4f>;
@group(0) @binding(1) var<storage, read> distances: array<f32>;
@group(0) @binding(2) var<storage, read_write> curve: array<f32>;
@compute @workgroup_size(${W})
fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${w(o,W)}
  if (index >= ${i.length}u) { return; }
  let model = fitted[fittedOffset + ${a}u];
  let distance = distances[distancesOffset + index];
  let q = distance / max(model.z, 1.0e-9);
  curve[curveOffset + index] = select(model.x + model.y * ${G[a]}, 0.0, distance <= 0.0);
}`;return[C({id:t,resources:[{buffer:n,usage:`storage-read`},{buffer:r,usage:`storage-read`},{buffer:i,usage:`storage-read-write`}],workload:{operation:`FittedCurve`,variant:String(a),commandCount:1,maximumWorkgroupCount:o.x*o.y*o.z,maximumInvocationCount:o.x*o.y*o.z*W,readByteLength:i.length*8,writeByteLength:i.length*4},compile:({device:e})=>{let a=new _(e,{id:t,source:s,shaderLayout:{bindings:[`fitted`,`distances`,`curve`].map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:t})=>{let s={fitted:h(n,t),distances:h(r,t),curve:h(i,t)};a.dispatch(e,{bindings:s,...o})},destroy:()=>a.destroy()}}})]}};function q(e,t,n,r,i){let a=e.allocate(t,`float32`,r.length);return e.graph.add(new K(`${t}-curve`,n.view,r.view,a.view,i)),a}var J=700,Y=20,X=48,Z=40,Q=A,$=[0,2.2],ae=.3,oe=[`cressie`,`pairs`,`none`],se=S({parameters:ie,caption:`Three fitted models over the empirical semivariance of log flash energy; the thin rules mark the spherical fit’s nugget, sill and range.`,createInputs:({random:e,world:t})=>{let n=ne(e,t.width*.5,J);return{...n,world:t,pairs:O(n.positions,Z)}},build(n,r,i){let a=n.upload(`positions`,`float32x2`,r.positions),o=n.allocate(`values`,`float32`,J),s=n.allocate(`parameters`,`float32`,8),c=n.allocate(`semivariances`,`float32`,Y),l=n.allocate(`pair-counts`,`uint32`,Y),u=n.allocate(`mean-distances`,`float32`,Y);n.graph.add(new t({positions:a.view,values:o.view,parameters:s.view,gridSize:[32,32],lagCount:Y,output:{semivariances:c.view,pairCounts:l.view,meanDistances:u.view}}));let d=n.allocate(`fitted`,`float32x4`,3);n.graph.add(new U({distances:u.view,semivariances:c.view,pairCounts:l.view,models:[`spherical`,`exponential`,`gaussian`],weighting:oe[i.weighting],output:{parameters:d.view,diagnostics:n.allocate(`diagnostics`,`float32x4`,3).view,status:n.allocate(`status`,`uint32`,3).view}}));let f=n.allocate(`curve-distances`,`float32`,X),p=n.allocate(`marks`,`float32x4`,3);return{outputs:{fitted:d,pairs:n.allocate(`pairs`,`float32x4`,Z),pairCounts:l,marks:p,positions:a,observed:o,semivariances:c,meanDistances:u,spherical:q(n,`spherical`,d,f,0),exponential:q(n,`exponential`,d,f,1),gaussian:q(n,`gaussian`,d,f,2)},update:t=>{o.write(re(r,t.noise)),f.write(Float32Array.from({length:X},(e,n)=>n/47*t.maximumDistance)),s.write(e({bounds:[0,0,r.windowWidth,1],maximumDistance:t.maximumDistance}))}}},rebuildOn:[`weighting`],interaction:{kind:`probe`,hint:`hover a distance`,rest:[k(ae),.5],resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=D(e.unitX),o=a*r.maximumDistance,s=(Q[0]+a*(Q[2]-Q[0]))*t.world.width,c=Q[1]*t.world.height,l=Q[3]*t.world.height,u=Math.round(a*47),d=i(n.semivariances)?.[E(a,Y)],f=[n.spherical,n.exponential,n.gaussian].map(e=>i(e)?.[u]),p=i(n.fitted);if(p){let[e,i,a]=p,o=Q[0]*t.world.width,s=Q[2]*t.world.width,l=o+Math.min(1,a/r.maximumDistance)*(s-o),u=j(Q,t.world,$,e),d=j(Q,t.world,$,e+i);n.marks.write(Float32Array.of(o,u,s,u,o,d,s,d,l,c,l,d))}let m=r.maximumDistance/Y,h=E(a,Y),g=t.pairs.find(h*m,(h+1)*m);n.pairs.write(g.segments);let _=i(n.pairCounts)?.[h],v=e=>e===void 0?`…`:e.toFixed(1);return{spotlight:null,glyphs:[{shape:`line`,from:[s,c],to:[s,l]},...f.flatMap((e,n)=>e===void 0?[]:[{shape:`dot`,at:[s,j(Q,t.world,$,e)]}])],readout:`h ${o.toFixed(2)} · ${_??`…`} pairs (sample of ${g.shown}) · data ${v(d)} · sph ${v(f[0])} exp ${v(f[1])} gau ${v(f[2])}`}}},draw:({outputs:e,values:t})=>[{layer:`dots`,context:!0,positions:e.positions,values:e.observed,range:[1,7],radius:2.8},{layer:`lines`,segments:e.pairs,count:Z,color:[1,.6,.2,.55]},{layer:`lines`,context:!0,segments:e.marks,count:3,color:[.3,.75,1,.55]},{layer:`chart`,kind:`scatter`,values:e.semivariances,xValues:e.meanDistances,xRange:[0,t.maximumDistance],range:$,rectangle:Q,radius:5,color:[.85,.85,.9,.9]},{layer:`chart`,kind:`line`,values:e.spherical,range:$,rectangle:Q,radius:2.4,color:[.3,.75,1,1]},{layer:`chart`,kind:`line`,values:e.exponential,range:$,rectangle:Q,radius:2.4,color:[1,.6,.2,1]},{layer:`chart`,kind:`line`,values:e.gaussian,range:$,rectangle:Q,radius:2.4,color:[.7,1,.7,1]}]});export{se as default};