import{r as e}from"./kernel-C3tfQOn4.js";import{o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{D as n,E as r,T as i,f as a,u as o,x as s,y as c}from"./gpu-contributor-utils-BIwLQON5.js";import{c as l,d as u,r as d,u as f}from"./wgsl-common-DfyXiYB0.js";import{t as p}from"./incomplete-gamma-wgsl-BipKCma-.js";import{n as m,r as h}from"./spatial-regression-solve-Vf-sQz9O.js";import{n as g,t as _}from"./regime-moments-CWEPBRvM.js";import{t as v}from"./rounded-difference-wgsl-DgXXQtNj.js";import{i as y,n as b,t as x}from"./regression-row-tiles-7Xsw3Utq.js";import{t as S}from"./define-run-B-brkCe4.js";import{c as C,r as w}from"./text-DYSgVV-n.js";import{Mn as T}from"./entry-preview-D70TdSbK.js";import{_ as E,a as D,h as O,l as k,m as A,p as j,s as M,u as N}from"./regression-CXg5hXjb.js";import{a as P}from"./weights-BRH93xop.js";var F=class{props;id;coefficientCount;coefficientOffsets;constructor(e){this.props=e,this.id=e.id??`restricted-regime-regression`,a(this.id,`predictorCount`,e.predictorCount,1,7),a(this.id,`regimeCount`,e.regimeCount,2,8),a(this.id,`unrestricted coefficients`,(e.predictorCount+1)*e.regimeCount,1,16),a(this.id,`rows`,e.response.length,1,x),i(this.id,e.common.length===e.predictorCount+1&&e.common.every(e=>typeof e==`boolean`),`common must hold one boolean for the intercept and one for each of predictorCount predictors`);let t=0;this.coefficientOffsets=e.common.map(n=>{let r=t;return t+=n?1:e.regimeCount,r}),this.coefficientCount=t;let o=e.response.length;for(let[i,a,s,c]of[[`predictors`,e.predictors,`float32`,o*e.predictorCount],[`response`,e.response,`float32`,o],[`regimes`,e.regimes,`uint32`,o],[`mask`,e.mask,`uint32`,o],[`output.coefficients`,e.output.coefficients,`float32`,t],[`output.covariance`,e.output.covariance,`float32`,t*t],[`output.fitted`,e.output.fitted,`float32`,o],[`output.chow`,e.output.chow,`float32`,3],[`output.stability`,e.output.stability,`float32`,(e.predictorCount+2)*3],[`output.status`,e.output.status,`uint32`,1]])a&&(r(a,[s],this.id),n(this.id,i,a,c));c(this.id,Object.values(e.output),[e.predictors,e.response,e.regimes,e.mask])}getCommandNodes(n){let{id:r,props:i,coefficientCount:c}=this,S=i.predictorCount,C=(S+1)*i.regimeCount,w=i.response.length;a(r,`rows`,w,1,Math.min(x,o(n.device,4*S)));let T=`${d}\n${l}
const OFFSETS = array<u32, ${S+1}>(${this.coefficientOffsets.map(e=>`${e}u`).join(`, `)});
const COMMON = array<bool, ${S+1}>(${i.common.join(`, `)});
fn included(row: u32) -> bool { if (regimes[regimesOffset + row] >= ${i.regimeCount}u || ${i.mask?`mask[maskOffset + row] == 0u || `:``}!isFiniteBits(response[responseOffset + row])) { return false; }
for (var column = 0u; column < ${S}u; column++) { if (!isFiniteBits(predictors[predictorsOffset + row * ${S}u + column])) { return false; } } return true; }
fn rawValue(row: u32, column: u32) -> f32 { if (column == 0u) { return 1.0; } return predictors[predictorsOffset + row * ${S}u + column - 1u]; }
`,E=g(S),D=i.regimeCount,O=S+2,k=2+c,A=k+c*c,j=A+C,M=j+3,N=M+D*O,P=N+D*O,F=P+c,I=F+c*c,L=I+c*c,R=L+c,z=1+2*(S+1),B=R+z,V=B+1,H=e(n,`${r}-workspace`,`float32`,V),U=b(w),W=`const D: u32 = ${S+1}u;
const STRIDE: u32 = ${E}u;
fn regimeRows(regime: u32) -> f32 { return f32(segmentOffsets[segmentOffsetsOffset + regime + 1u] - segmentOffsets[segmentOffsetsOffset + regime]); }
// Mean of a column (1..K predictors, K + 1 response) in one regime.
fn regimeMean(regime: u32, column: u32) -> f32 { return moments[momentsOffset + regime * STRIDE + column - 1u]; }
// Centered cross-product of two columns (1..K predictors, K + 1 response) in one regime.
fn centeredMoment(regime: u32, first: u32, second: u32) -> f32 {
  let low = min(first, second) - 1u;
  let high = max(first, second) - 1u;
  return moments[momentsOffset + regime * STRIDE + D + low * (2u * D - low + 1u) / 2u + (high - low)];
}
// Sum of the deviations of a column (1..K predictors, K + 1 response) from its float32 regime
// mean: what the float32 mean misses of the true mean, times the row count.
fn regimeDeviationSum(regime: u32, column: u32) -> f32 { return moments[momentsOffset + regime * STRIDE + D + D * (D + 1u) / 2u + column - 1u]; }
// Cross-product of two columns (0 the constant, 1..K predictors, K + 1 response) of one regime
// about the given shifts, exact for the stored float32 means: with offsets o = mean - shift,
// sum (a - sa)(b - sb) = C_ab + oa S_b + ob S_a + n oa ob, and sum (b - sb) = S_b + n ob.
fn momentAboutShifts(regime: u32, first: u32, second: u32, firstShift: f32, secondShift: f32) -> f32 {
  let count = regimeRows(regime);
  if (first == 0u && second == 0u) { return count; }
  if (first == 0u) { return regimeDeviationSum(regime, second) + count * (regimeMean(regime, second) - secondShift); }
  if (second == 0u) { return regimeDeviationSum(regime, first) + count * (regimeMean(regime, first) - firstShift); }
  let firstOffset = regimeMean(regime, first) - firstShift; let secondOffset = regimeMean(regime, second) - secondShift;
  return centeredMoment(regime, first, second) + firstOffset * regimeDeviationSum(regime, second) + secondOffset * regimeDeviationSum(regime, first) + count * firstOffset * secondOffset;
}
// Pooled mean of every column, the count-weighted mean of the regime means.
fn getPooledMeans() -> array<f32, ${S+2}> {
  var pooled: array<f32, ${S+2}>; var total = 0.0;
  for (var regime = 0u; regime < ${D}u; regime++) { let count = regimeRows(regime); total += count; for (var column = 1u; column < ${S+2}u; column++) { pooled[column] += count * regimeMean(regime, column); } }
  for (var column = 1u; column < ${S+2}u; column++) { pooled[column] /= max(total, 1.0); }
  return pooled;
}`,G=(e,t)=>`
fn destination_${e}(column: u32, regime: u32) -> u32 { return OFFSETS[column] + select(regime, 0u, COMMON[column]); }
// Cross-product about the pooled means of the columns (0 the constant, 1..K predictors, K + 1
// response) in one regime: the centered moment plus the product of the mean offsets.
fn pooledMoment_${e}(regime: u32, first: u32, second: u32, pooled: ptr<function, array<f32, ${S+2}>>) -> f32 {
  return momentAboutShifts(regime, first, second, (*pooled)[first], (*pooled)[second]);
}
// Cross-product of the design values of two source columns (the response is column K + 1, whose
// design value is its pooled deviation) in one regime.
fn designMoment_${e}(regime: u32, first: u32, second: u32, pooled: ptr<function, array<f32, ${S+2}>>, forms: ptr<function, array<f32, ${z}>>) -> f32 {
  let pivotColumn = u32((*forms)[0]);
  var firstIndices = array<u32, 3>(0u, first, pivotColumn); var secondIndices = array<u32, 3>(0u, second, pivotColumn);
  var firstWeights = array<f32, 3>(0.0, select(1.0, 0.0, first == 0u), 0.0); var secondWeights = array<f32, 3>(0.0, select(1.0, 0.0, second == 0u), 0.0);
  if (first <= ${S}u) { firstWeights[0] = (*forms)[1u + first]; firstWeights[2] = (*forms)[${S+2}u + first]; }
  if (second <= ${S}u) { secondWeights[0] = (*forms)[1u + second]; secondWeights[2] = (*forms)[${S+2}u + second]; }
  var value = 0.0;
  for (var left = 0u; left < 3u; left++) { if (firstWeights[left] == 0.0) { continue; }
    for (var other = 0u; other < 3u; other++) { if (secondWeights[other] == 0.0) { continue; }
      value += firstWeights[left] * secondWeights[other] * pooledMoment_${e}(regime, firstIndices[left], secondIndices[other], pooled); } }
  return value;
}
fn fit_${e}(estimate: ptr<function, array<f32, ${t}>>, levels: ptr<function, array<f32, ${D*O}>>, inverse: ptr<function, array<f32, ${t*t}>>, shiftedEstimate: ptr<function, array<f32, ${t}>>, shiftedInverse: ptr<function, array<f32, ${t*t}>>, transform: ptr<function, array<f32, ${t*t}>>, origin: ptr<function, array<f32, ${t}>>, forms: ptr<function, array<f32, ${z}>>) -> bool {
var pooled = getPooledMeans();
// Pivot: the varying-slope predictor whose pooled mean is largest against its pooled spread.
var pivot = 0u; var largestRatio = 1.0;
for (var column = 1u; column < ${S+1}u; column++) { if (COMMON[column]) { continue; }
  var squares = 0.0; var total = 0.0;
  for (var regime = 0u; regime < ${D}u; regime++) { squares += pooledMoment_${e}(regime, column, column, &pooled); total += regimeRows(regime); }
  if (!(squares > 0.0)) { continue; }
  let ratio = pooled[column] * pooled[column] * total / squares;
  if (ratio > largestRatio) { largestRatio = ratio; pivot = column; } }
for (var slot = 0u; slot < ${z}u; slot++) { (*forms)[slot] = 0.0; }
// With no pivot the pivot terms are all zero and column 1 only keeps the index in range.
(*forms)[0] = f32(max(pivot, 1u));
(*forms)[1] = select(1.0, 0.0, pivot != 0u); (*forms)[${S+2}u] = select(0.0, 1.0, pivot != 0u);
for (var column = 1u; column < ${S+1}u; column++) { if (COMMON[column]) { continue; }
  if (pivot == 0u || column == pivot) { (*forms)[1u + column] = pooled[column]; } else { (*forms)[${S+2}u + column] = -pooled[column] / pooled[pivot]; } }
var normal: array<f32, ${t*t}>; var right: array<f32, ${t}>;
for (var regime = 0u; regime < ${D}u; regime++) { for (var first = 0u; first < ${S+1}u; first++) { let row = destination_${e}(first, regime); right[row] += designMoment_${e}(regime, first, ${S+1}u, &pooled, forms);
 for (var second = 0u; second < ${S+1}u; second++) { normal[row * ${t}u + destination_${e}(second, regime)] += designMoment_${e}(regime, first, second, &pooled, forms); } } }
var scales: array<f32, ${t}>;
for (var column = 0u; column < ${t}u; column++) { let diagonal = normal[column * ${t+1}u]; if (!(diagonal > 0.0) || !isFiniteBits(diagonal)) { return false; } scales[column] = inverseSqrt(diagonal); right[column] *= scales[column]; }
for (var first = 0u; first < ${t}u; first++) { for (var second = 0u; second < ${t}u; second++) { normal[first * ${t}u + second] *= scales[first] * scales[second]; } }
if (!choleskyFactorLimited_${t}(&normal)) { return false; } choleskySolve_${t}(&normal, &right);
for (var column = 0u; column < ${t}u; column++) { (*shiftedEstimate)[column] = right[column] * scales[column]; (*origin)[column] = 0.0; var unit: array<f32, ${t}>; unit[column] = scales[column]; choleskySolve_${t}(&normal, &unit); for (var row = 0u; row < ${t}u; row++) { (*shiftedInverse)[row * ${t}u + column] = unit[row] * scales[row]; } }
// raw = T solution + t0. A slope is its own coefficient, and the pivot slope of a regime also
// collects the pivot terms of that regime's columns. The intercept collects the constant left by
// each form: 1 or -mean_p for the intercept slot, -mean_c for a common slope, and exactly 0 for a
// varying slope (mean_c - mean_c, or -mean_c + (mean_c / mean_p) mean_p).
for (var slot = 0u; slot < ${t*t}u; slot++) { (*transform)[slot] = 0.0; }
for (var column = 0u; column < ${t}u; column++) { (*transform)[column * ${t+1}u] = 1.0; }
let interceptSlot = OFFSETS[0]; (*origin)[interceptSlot] = pooled[${S+1}u];
(*transform)[interceptSlot * ${t+1}u] = select(1.0, -pooled[max(pivot, 1u)], pivot != 0u);
for (var column = 1u; column < ${S+1}u; column++) { if (COMMON[column]) { (*transform)[interceptSlot * ${t}u + OFFSETS[column]] = -pooled[column]; } }
if (pivot != 0u) { for (var regime = 0u; regime < ${D}u; regime++) { let pivotSlot = destination_${e}(pivot, regime);
  for (var column = 0u; column < ${S+1}u; column++) { if (column == pivot) { continue; } (*transform)[pivotSlot * ${t}u + destination_${e}(column, regime)] += (*forms)[${S+2}u + column]; } } }
for (var row = 0u; row < ${t}u; row++) { var value = (*origin)[row]; for (var column = 0u; column < ${t}u; column++) { value += (*transform)[row * ${t}u + column] * (*shiftedEstimate)[column]; } (*estimate)[row] = value; }
// Level records: every column of this model is taken about its pooled mean.
for (var regime = 0u; regime < ${D}u; regime++) { let level = regime * ${O}u; (*levels)[level] = (*shiftedEstimate)[interceptSlot]; for (var column = 1u; column < ${S+2}u; column++) { (*levels)[level + select(1u + column, 1u, column == ${S+1}u)] = pooled[column]; } }
// Inverse normal matrix of the raw design: T V T'.
var product: array<f32, ${t*t}>;
for (var row = 0u; row < ${t}u; row++) { for (var column = 0u; column < ${t}u; column++) { var value = 0.0; for (var inner = 0u; inner < ${t}u; inner++) { value += (*transform)[row * ${t}u + inner] * (*shiftedInverse)[inner * ${t}u + column]; } product[row * ${t}u + column] = value; } }
for (var row = 0u; row < ${t}u; row++) { for (var column = 0u; column < ${t}u; column++) { var value = 0.0; for (var inner = 0u; inner < ${t}u; inner++) { value += product[row * ${t}u + inner] * (*transform)[column * ${t}u + inner]; } (*inverse)[row * ${t}u + column] = value; } }
return true;
}`,K=(e,t,n)=>n&&i.common[0]?G(e,t):`
fn destination_${e}(column: u32, regime: u32) -> u32 { return ${n?`OFFSETS[column] + select(regime, 0u, COMMON[column])`:`column * ${D}u + regime`}; }
// Shift of a column (1..K predictors, K + 1 response) in one regime.
fn shift_${e}(regime: u32, column: u32, pooled: ptr<function, array<f32, ${S+2}>>) -> f32 {
  return regimeMean(regime, column);
}
// Cross-product about the shifts of design columns (0 intercept, 1..K predictors, K + 1 response)
// in one regime (momentAboutShifts with the shifts of this model).
fn shiftedMoment_${e}(regime: u32, first: u32, second: u32, pooled: ptr<function, array<f32, ${S+2}>>) -> f32 {
  var firstShift = 0.0; var secondShift = 0.0;
  if (first != 0u) { firstShift = shift_${e}(regime, first, pooled); }
  if (second != 0u) { secondShift = shift_${e}(regime, second, pooled); }
  return momentAboutShifts(regime, first, second, firstShift, secondShift);
}
fn fit_${e}(estimate: ptr<function, array<f32, ${t}>>, levels: ptr<function, array<f32, ${D*O}>>${n?`, inverse: ptr<function, array<f32, ${t*t}>>, shiftedEstimate: ptr<function, array<f32, ${t}>>, shiftedInverse: ptr<function, array<f32, ${t*t}>>, transform: ptr<function, array<f32, ${t*t}>>, origin: ptr<function, array<f32, ${t}>>, forms: ptr<function, array<f32, ${z}>>`:``}) -> bool {
var pooled = getPooledMeans();
${n?`// Design forms of the shifted basis: the intercept column is the constant 1, every predictor column its shifted value.
for (var slot = 0u; slot < ${z}u; slot++) { (*forms)[slot] = 0.0; } (*forms)[0] = 1.0; (*forms)[1] = 1.0;`:``}
var normal: array<f32, ${t*t}>; var right: array<f32, ${t}>;
for (var regime = 0u; regime < ${D}u; regime++) { for (var first = 0u; first < ${S+1}u; first++) { let row = destination_${e}(first, regime); right[row] += shiftedMoment_${e}(regime, first, ${S+1}u, &pooled);
 for (var second = 0u; second < ${S+1}u; second++) { normal[row * ${t}u + destination_${e}(second, regime)] += shiftedMoment_${e}(regime, first, second, &pooled); } } }
var scales: array<f32, ${t}>;
for (var column = 0u; column < ${t}u; column++) { let diagonal = normal[column * ${t+1}u]; if (!(diagonal > 0.0) || !isFiniteBits(diagonal)) { return false; } scales[column] = inverseSqrt(diagonal); right[column] *= scales[column]; }
for (var first = 0u; first < ${t}u; first++) { for (var second = 0u; second < ${t}u; second++) { normal[first * ${t}u + second] *= scales[first] * scales[second]; } }
if (!choleskyFactorLimited_${t}(&normal)) { return false; } choleskySolve_${t}(&normal, &right);
var solution: array<f32, ${t}>;
for (var column = 0u; column < ${t}u; column++) { solution[column] = right[column] * scales[column]; (*estimate)[column] = solution[column]; }
// Back to the raw design: slopes are unchanged, each intercept absorbs the shifts.
for (var regime = 0u; regime < ${D}u; regime++) {
  let interceptSlot = destination_${e}(0u, regime); let level = regime * ${O}u;
  let responseShift = shift_${e}(regime, ${S+1}u, &pooled);
  (*levels)[level] = solution[interceptSlot]; (*levels)[level + 1u] = responseShift;
  var intercept = solution[interceptSlot] + responseShift;
  for (var column = 1u; column < ${S+1}u; column++) { let shift = shift_${e}(regime, column, &pooled); (*levels)[level + 1u + column] = shift; intercept -= shift * solution[destination_${e}(column, regime)]; }
  (*estimate)[interceptSlot] = intercept;
}
${n?`for (var column = 0u; column < ${t}u; column++) { (*shiftedEstimate)[column] = solution[column]; (*origin)[column] = 0.0; var unit: array<f32, ${t}>; unit[column] = scales[column]; choleskySolve_${t}(&normal, &unit); for (var row = 0u; row < ${t}u; row++) { (*shiftedInverse)[row * ${t}u + column] = unit[row] * scales[row]; } }
// raw = T shifted + t0: T is the identity except that each intercept row carries minus the shifts.
for (var slot = 0u; slot < ${t*t}u; slot++) { (*transform)[slot] = 0.0; }
for (var column = 0u; column < ${t}u; column++) { (*transform)[column * ${t+1}u] = 1.0; }
for (var regime = 0u; regime < ${D}u; regime++) { let interceptSlot = destination_${e}(0u, regime); (*origin)[interceptSlot] = shift_${e}(regime, ${S+1}u, &pooled);
  for (var column = 1u; column < ${S+1}u; column++) { (*transform)[interceptSlot * ${t}u + destination_${e}(column, regime)] = -shift_${e}(regime, column, &pooled); } }
// Inverse normal matrix of the raw design: T V T'.
var product: array<f32, ${t*t}>;
for (var row = 0u; row < ${t}u; row++) { for (var column = 0u; column < ${t}u; column++) { var value = 0.0; for (var inner = 0u; inner < ${t}u; inner++) { value += (*transform)[row * ${t}u + inner] * (*shiftedInverse)[inner * ${t}u + column]; } product[row * ${t}u + column] = value; } }
for (var row = 0u; row < ${t}u; row++) { for (var column = 0u; column < ${t}u; column++) { var value = 0.0; for (var inner = 0u; inner < ${t}u; inner++) { value += product[row * ${t}u + inner] * (*transform)[column * ${t}u + inner]; } (*inverse)[row * ${t}u + column] = value; } }`:``}
return true;
}`,q=[{name:`predictors`,view:i.predictors,type:`f32`,access:`read`},{name:`response`,view:i.response,type:`f32`,access:`read`},{name:`regimes`,view:i.regimes,type:`u32`,access:`read`},...i.mask?[{name:`mask`,view:i.mask,type:`u32`,access:`read`}]:[]];s(r,n,[i.predictors,i.response,i.regimes,i.mask,...Object.values(i.output)]);let J=_(n,{id:r,operation:`GPURestrictedRegimeRegression`,predictors:i.predictors,predictorCount:S,response:i.response,regimes:i.regimes,regimeCount:D,mask:i.mask}),Y=[...J.nodes,t(n,{id:`${r}-fit`,operation:`GPURestrictedRegimeRegression`,variant:`nested-models`,invocationCount:1,bindings:[{name:`segmentOffsets`,view:J.views.segmentOffsets,type:`u32`,access:`read`},{name:`moments`,view:J.views.moments,type:`f32`,access:`read`},{name:`workspace`,view:H,type:`f32`,access:`read_write`}],declarations:`${d}\n${l}
const OFFSETS = array<u32, ${S+1}>(${this.coefficientOffsets.map(e=>`${e}u`).join(`, `)});
const COMMON = array<bool, ${S+1}>(${i.common.join(`, `)});
${W}\n${f}\n${m(c)}\n${h(c)}\n${C===c?``:`${m(C)}\n${h(C)}`}\n${K(`restricted`,c,!0)}\n${K(`unrestricted`,C,!1)}`,body:`for (var slot = 0u; slot < ${V}u; slot++) { workspace[workspaceOffset + slot] = getNaN(); }
workspace[workspaceOffset + ${B}u] = 0.0;
let count = segmentOffsets[segmentOffsetsOffset + ${D}u] - segmentOffsets[segmentOffsetsOffset];
workspace[workspaceOffset + 0u] = f32(count);
workspace[workspaceOffset + 1u] = 1.0; if (count <= ${C}u) { return; }
var estimate: array<f32, ${c}>; var inverse: array<f32, ${c*c}>; var levels: array<f32, ${D*O}>;
var shiftedEstimate: array<f32, ${c}>; var shiftedInverse: array<f32, ${c*c}>; var transform: array<f32, ${c*c}>; var origin: array<f32, ${c}>; var forms: array<f32, ${z}>;
workspace[workspaceOffset + 1u] = 2.0; if (!fit_restricted(&estimate, &levels, &inverse, &shiftedEstimate, &shiftedInverse, &transform, &origin, &forms)) { return; }
for (var slot = 0u; slot < ${z}u; slot++) { workspace[workspaceOffset + ${R}u + slot] = forms[slot]; }
for (var slot = 0u; slot < ${c}u; slot++) { workspace[workspaceOffset + 2u + slot] = estimate[slot]; workspace[workspaceOffset + ${P}u + slot] = shiftedEstimate[slot]; workspace[workspaceOffset + ${L}u + slot] = origin[slot]; }
for (var slot = 0u; slot < ${c*c}u; slot++) { workspace[workspaceOffset + ${k}u + slot] = inverse[slot]; workspace[workspaceOffset + ${F}u + slot] = shiftedInverse[slot]; workspace[workspaceOffset + ${I}u + slot] = transform[slot]; }
for (var slot = 0u; slot < ${D*O}u; slot++) { workspace[workspaceOffset + ${M}u + slot] = levels[slot]; }
var fullEstimate: array<f32, ${C}>; var fullLevels: array<f32, ${D*O}>;
workspace[workspaceOffset + 1u] = 3.0; if (!fit_unrestricted(&fullEstimate, &fullLevels)) { return; }
for (var slot = 0u; slot < ${C}u; slot++) { workspace[workspaceOffset + ${A}u + slot] = fullEstimate[slot]; }
for (var slot = 0u; slot < ${D*O}u; slot++) { workspace[workspaceOffset + ${N}u + slot] = fullLevels[slot]; }
workspace[workspaceOffset + 1u] = 0.0;`}),...y(n,{id:`${r}-residual-squares`,operation:`GPURestrictedRegimeRegression`,variant:`residual-squares`,rowCount:w,layout:U,sumCount:3,bindings:[...q,{name:`workspace`,view:H,type:`f32`,access:`read`}],declarations:`${T}
${v}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  if (!included(row)) { return; }
  let regime = regimes[regimesOffset + row];
  let restrictedLevel = workspaceOffset + ${M}u + regime * ${O}u;
  let unrestrictedLevel = workspaceOffset + ${N}u + regime * ${O}u;
  let observed = response[responseOffset + row];
  // Restricted model: the response about its shift minus, per source column, the shifted-basis
  // coefficient times the design value constant + deviation + pivotTerm * pivotDeviation (the
  // intercept column has no deviation of its own). The unrestricted model has regime-mean shifts
  // and a shifted intercept that carries what the float32 regime means miss.
  // Every difference of a value and its shift goes through getRoundedDifference: the compiler of
  // the test GPU otherwise regroups the chain as value - (shift + small terms) and the residual
  // comes out in float32 steps of the value.
  let runtimeZero = bitcast<u32>(workspace[workspaceOffset + ${B}u]);
  let formBase = workspaceOffset + ${R}u;
  let estimateBase = workspaceOffset + ${P}u;
  let pivotColumn = u32(workspace[formBase]);
  let pivotDeviation = getRoundedDifference(rawValue(row, pivotColumn), workspace[restrictedLevel + 1u + pivotColumn], runtimeZero);
  var restrictedResidual = getRoundedDifference(observed, workspace[restrictedLevel + 1u], runtimeZero);
  restrictedResidual -= workspace[estimateBase + OFFSETS[0] + select(regime, 0u, COMMON[0])] * (workspace[formBase + 1u] + workspace[formBase + ${S+2}u] * pivotDeviation);
  var unrestrictedResidual = getRoundedDifference(observed, workspace[unrestrictedLevel + 1u], runtimeZero);
  unrestrictedResidual -= workspace[unrestrictedLevel];
  for (var column = 1u; column < ${S+1}u; column++) {
    let value = rawValue(row, column);
    let designValue = workspace[formBase + 1u + column] + getRoundedDifference(value, workspace[restrictedLevel + 1u + column], runtimeZero) + workspace[formBase + ${S+2}u + column] * pivotDeviation;
    restrictedResidual -= designValue * workspace[estimateBase + OFFSETS[column] + select(regime, 0u, COMMON[column])];
    unrestrictedResidual -= getRoundedDifference(value, workspace[unrestrictedLevel + 1u + column], runtimeZero) * workspace[workspaceOffset + ${A}u + column * ${D}u + regime];
  }
  (*terms)[0] += restrictedResidual * restrictedResidual;
  (*terms)[1] += unrestrictedResidual * unrestrictedResidual;
  // SSE_restricted - SSE_unrestricted of nested least squares is the squared distance of the fits.
  let fitDifference = restrictedResidual - unrestrictedResidual;
  (*terms)[2] += fitDifference * fitDifference;
}`,partial:e(n,`${r}-residual-partials`,`float32`,U.tileCount*3),output:H,outputSlot:j}),t(n,{id:`${r}-finish`,operation:`GPURestrictedRegimeRegression`,variant:`finish`,invocationCount:1,bindings:[{name:`workspace`,view:H,type:`f32`,access:`read`},{name:`coefficients`,view:i.output.coefficients,type:`f32`,access:`read_write`},{name:`covariance`,view:i.output.covariance,type:`f32`,access:`read_write`},{name:`chow`,view:i.output.chow,type:`f32`,access:`read_write`},{name:`status`,view:i.output.status,type:`u32`,access:`read_write`}],declarations:`${d}\n${l}`,body:`for (var slot = 0u; slot < ${c}u; slot++) { coefficients[coefficientsOffset + slot] = getNaN(); }
for (var slot = 0u; slot < ${c*c}u; slot++) { covariance[covarianceOffset + slot] = getNaN(); }
for (var slot = 0u; slot < 3u; slot++) { chow[chowOffset + slot] = getNaN(); }
var code = u32(workspace[workspaceOffset + 1u]);
let squares = workspace[workspaceOffset + ${j}u]; let fullSquares = workspace[workspaceOffset + ${j+1}u]; let fitSquares = workspace[workspaceOffset + ${j+2}u];
if (code == 0u || code == 3u) { if (!isFiniteBits(squares)) { code = 2u; } }
status[statusOffset] = code; if (code == 1u || code == 2u) { return; }
let count = u32(workspace[workspaceOffset + 0u]);
let variance = squares / f32(count - ${c}u);
for (var slot = 0u; slot < ${c}u; slot++) { coefficients[coefficientsOffset + slot] = workspace[workspaceOffset + 2u + slot]; }
for (var slot = 0u; slot < ${c*c}u; slot++) { covariance[covarianceOffset + slot] = workspace[workspaceOffset + ${k}u + slot] * variance; }
if (code == 0u && !isFiniteBits(fullSquares)) { code = 3u; status[statusOffset] = 3u; }
if (code != 0u) { return; }
chow[chowOffset + 1u] = ${C-c}.0; chow[chowOffset + 2u] = f32(count - ${C}u);
${C>c?`if (fullSquares > 0.0) { chow[chowOffset] = fitSquares / ${C-c}.0 / (fullSquares/f32(count-${C}u)); }`:``}`}),t(n,{id:`${r}-predict`,operation:`GPURestrictedRegimeRegression`,variant:`predict`,invocationCount:w,bindings:[...q,{name:`coefficients`,view:i.output.coefficients,type:`f32`,access:`read`},{name:`fitted`,view:i.output.fitted,type:`f32`,access:`read_write`}],declarations:T,body:`var prediction = getNaN(); if (included(index)) { prediction = 0.0; let regime = regimes[regimesOffset+index]; for (var column=0u; column<${S+1}u; column++) { let coefficient = OFFSETS[column]+select(regime,0u,COMMON[column]); prediction += rawValue(index,column)*coefficients[coefficientsOffset+coefficient]; } } fitted[fittedOffset+index]=prediction;`})];if(i.output.stability){let e=i.common.flatMap((e,t)=>e?[]:[t]),a=Math.max(1,e.length*(i.regimeCount-1)),o=e.flatMap(e=>Array.from({length:i.regimeCount-1},()=>e)),s=e.flatMap(e=>Array.from({length:i.regimeCount-1},(t,n)=>this.coefficientOffsets[e]+n+1)),l=e.flatMap(e=>Array.from({length:i.regimeCount-1},()=>this.coefficientOffsets[e]));Y.push(t(n,{id:`${r}-stability`,operation:`GPURestrictedRegimeRegression`,variant:`wald-stability`,invocationCount:S+2,bindings:[{name:`workspace`,view:H,type:`f32`,access:`read`},{name:`status`,view:i.output.status,type:`u32`,access:`read`},{name:`stability`,view:i.output.stability,type:`f32`,access:`read_write`}],declarations:`${d}\n${u}\n${p}\n${f}\n${m(a)}
const COLUMNS = array<u32, ${a}>(${(o.length?o:[0]).map(e=>`${e}u`).join(`,`)});
const OFFSETS = array<u32, ${a}>(${(s.length?s:[0]).map(e=>`${e}u`).join(`,`)});
const BASES = array<u32, ${a}>(${(l.length?l:[0]).map(e=>`${e}u`).join(`,`)});`,body:`stability[stabilityOffset + index * 3u] = getNaN(); stability[stabilityOffset + index * 3u + 1u] = 0.0; stability[stabilityOffset + index * 3u + 2u] = getNaN();
if (status[statusOffset] != 0u) { return; }
var selected: array<u32, ${a}>; var count = 0u;
for (var contrast = 0u; contrast < ${o.length}u; contrast++) { if (index == ${S+1}u || COLUMNS[contrast] == index) { selected[count] = contrast; count++; } }
if (count == 0u) { return; }
var normal: array<f32, ${a*a}>; var right: array<f32, ${a}>; var difference: array<f32, ${a}>;
for (var first = 0u; first < ${a}u; first++) { normal[first * ${a+1}u] = 1.0; }
// Each contrast of raw coefficients is formed in the shifted basis (raw = T shifted + t0): the
// rows of T and the origins are differenced first, so the large shifts of two regimes cancel
// before they meet a coefficient or the covariance.
let variance = workspace[workspaceOffset + ${j}u] / (workspace[workspaceOffset + 0u] - ${c}.0);
var contrastRows: array<f32, ${a*c}>; var projected: array<f32, ${a*c}>;
for (var first = 0u; first < count; first++) { let positive = OFFSETS[selected[first]]; let negative = BASES[selected[first]];
  difference[first] = workspace[workspaceOffset + ${L}u + positive] - workspace[workspaceOffset + ${L}u + negative];
  for (var slot = 0u; slot < ${c}u; slot++) { contrastRows[first * ${c}u + slot] = workspace[workspaceOffset + ${I}u + positive * ${c}u + slot] - workspace[workspaceOffset + ${I}u + negative * ${c}u + slot]; } }
// Joint tests: an intercept contrast and a slope contrast of the same regime are almost parallel
// when the predictor has a large mean (the intercept row holds minus the shift of each regime).
// Adding the base regime's shift times the slope contrast to the intercept contrast spans the same
// hypothesis, leaves the statistic unchanged and keeps only the difference of the two shifts.
for (var first = 0u; first < count; first++) { if (COLUMNS[selected[first]] != 0u) { continue; } let regime = OFFSETS[selected[first]] - BASES[selected[first]];
  for (var second = 0u; second < count; second++) { let other = selected[second]; if (COLUMNS[other] == 0u || OFFSETS[other] - BASES[other] != regime) { continue; }
    let weight = contrastRows[first * ${c}u + BASES[other]];
    for (var slot = 0u; slot < ${c}u; slot++) { contrastRows[first * ${c}u + slot] += weight * contrastRows[second * ${c}u + slot]; } } }
for (var first = 0u; first < count; first++) { var value = difference[first];
  for (var slot = 0u; slot < ${c}u; slot++) { value += contrastRows[first * ${c}u + slot] * workspace[workspaceOffset + ${P}u + slot]; }
  difference[first] = value; right[first] = value; }
for (var first = 0u; first < count; first++) { for (var slot = 0u; slot < ${c}u; slot++) { var value = 0.0; for (var inner = 0u; inner < ${c}u; inner++) { value += contrastRows[first * ${c}u + inner] * workspace[workspaceOffset + ${F}u + inner * ${c}u + slot]; } projected[first * ${c}u + slot] = value; } }
for (var first = 0u; first < count; first++) { for (var second = 0u; second < count; second++) { var value = 0.0; for (var slot = 0u; slot < ${c}u; slot++) { value += projected[first * ${c}u + slot] * contrastRows[second * ${c}u + slot]; } normal[first * ${a}u + second] = value * variance; } }
if (!choleskyFactor_${a}(&normal)) { return; } choleskySolve_${a}(&normal, &right);
var statistic = 0.0; for (var contrast = 0u; contrast < count; contrast++) { statistic += difference[contrast] * right[contrast]; }
statistic = max(0.0, statistic); stability[stabilityOffset + index * 3u] = statistic; stability[stabilityOffset + index * 3u + 1u] = f32(count); stability[stabilityOffset + index * 3u + 2u] = getChiSquareSurvivalValue(statistic, f32(count));`}))}return Y}},I=[D+.08,.42,.97,.9],L=S({parameters:T,caption:`Restricted regimes: coefficients (right), Chow F below, fitted map (left).`,createInputs:({random:e,world:t})=>{let n=k(e,t);return{...n,boundary:n.mapWidth/2}},build(e,t,n){let r=t.tracts.count,i=[n.commonIntercept===1,n.commonFirst===1,n.commonSecond===1],a=i.reduce((e,t)=>e+(t?1:2),0),o=e.upload(`predictors`,`float32`,t.predictors),s=e.allocate(`response`,`float32`,r),c=e.allocate(`regimes`,`uint32`,r),l=e.allocate(`coefficients`,`float32`,a),u=e.allocate(`fitted`,`float32`,r),d=e.allocate(`chow`,`float32`,3);return e.graph.add(new F({predictors:o.view,predictorCount:2,response:s.view,regimes:c.view,regimeCount:2,common:i,output:{coefficients:l.view,covariance:e.allocate(`covariance`,`float32`,a**2).view,fitted:u.view,chow:d.view,status:e.allocate(`status`,`uint32`,1).view}})),{outputs:{mesh:E(e,t),fitted:u,coefficients:l,regimes:c,zeros:e.upload(`zeros`,`float32`,new Float32Array(64)),chowStatistic:P(e,`chow`,d,3,0,1)},update:e=>{c.write(Uint32Array.from({length:r},(e,n)=>t.tracts.centers[n*2]<t.boundary?0:1)),s.write(N(t,{noise:.4,coefficients:n=>t.regimes[n]===0?M:[M[0]+.6*e.contrast,M[1]-1.1*e.contrast,M[2]+.9*e.contrast]}))}}},rebuildOn:[`commonIntercept`,`commonFirst`,`commonSecond`],interaction:{kind:`level`,hint:`move the regime break`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,peek:r}){let i=Math.round((.15+.7*e.unitX)*t.mapWidth*50)/50,a=i!==t.boundary;t.boundary=i;let o=r(n.chowStatistic);return{rerun:a,glyphs:[{shape:`curtain`,x:i}],readout:`break at x ${(i/t.mapWidth).toFixed(2)} · Chow F ${o?o[0].toFixed(0):`-`}`}}},draw:({inputs:e,outputs:t,world:n})=>{let r=C({labels:[{text:`coefficients (a shared one is one bar)`,at:[(D+.08)*n.width,.9*n.height],anchor:`bottom-left`,offset:[0,3],role:`label`,priority:4},{text:`Chow F`,at:[(D+.08)*n.width,.34*n.height],anchor:`bottom-left`,offset:[0,3],role:`label`,priority:4}]});return[...A(e,t.mesh,t.fitted,{ramp:`sequential`,range:[-2,4]}),{layer:`polygons`,positions:t.mesh.vertices,indices:t.mesh.indices,owners:t.mesh.owners,values:t.regimes,triangleCount:e.tracts.triangleCount,ramp:`categorical`,alpha:.4},...j(t.coefficients,I,3),O(t.zeros,I),{layer:`chart`,kind:`bars`,values:t.chowStatistic,color:[.95,.6,.3,.95],range:[0,60],rectangle:[D+.08,.08,.97,.34]},{layer:`text`,...r},{layer:`text`,...w({corner:`top-left`,title:`regime tint`,entries:[{label:`west`,swatch:0},{label:`east`,swatch:1}]})}]}});export{L as default};