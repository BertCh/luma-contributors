import{t as e}from"./regression-output-dispatches-KKVTWkHP.js";import{i as t}from"./spatial-likelihood-allocations-BGEHbqkB.js";import{a as n}from"./graph-chunk-dispatch-zBt8k2-o.js";import{r}from"./kernel-C3tfQOn4.js";import{D as i,b as a,s as o,t as s,x as c}from"./local-weighted-least-squares-wgsl--tcHa6bd.js";import{a as l,o as u}from"./wgsl-kernel-nodes-DagJklH2.js";import{c as d,r as f,u as p}from"./wgsl-common-DfyXiYB0.js";import{E as m,S as h,T as g,f as _,p as v,v as y,x as b,y as x}from"./gpu-contributor-utils-BIwLQON5.js";import{a as S,i as C,n as w,o as T,r as E}from"./gpu-generalized-geographically-weighted-regression-CN_EA1t1.js";import{a as D,g as O,v as k}from"./spatial-context-DwZEd4EZ.js";import{n as A}from"./gwr-golden-section-nodes-BFRSkOqC.js";import{i as j,n as M}from"./regression-row-tiles-7Xsw3Utq.js";import{n as N}from"./spatial-regression-solve-Vf-sQz9O.js";import{t as P}from"./define-run-B-brkCe4.js";import{a as F}from"./pointer-overlay-Qh8qflcA.js";import{c as I,r as L}from"./text-DYSgVV-n.js";import{Rn as R}from"./entry-preview-D70TdSbK.js";import{_ as z,l as B,m as V,s as H,u as U}from"./regression-CXg5hXjb.js";import{a as W}from"./weights-BRH93xop.js";function ee(e,t,n,i){let{id:a,props:o}=t,s=o.inference,c=o.response.length,l=o.predictorCount+1,p=c*c*l,m=r(e,`${a}-response-smoother-first`,`float32`,p),h=r(e,`${a}-response-smoother-second`,`float32`,p),g=C({...o,parameters:n}),_=S(o,{sphereRadius:t.sphereRadius});return b(a,e,Object.values(s.output)),{initialize:()=>u(e,{id:`${a}-initialize-response-smoother`,operation:`GPUMultiscaleGeographicallyWeightedRegression`,variant:`initialize-full-inference`,invocationCount:p,bindings:[{name:`smoother`,view:m,type:`f32`,access:`read_write`}],body:`smoother[smootherOffset + index] = 0.0;`}),advance:(t,n)=>{let r=u(e,{id:`${a}-response-smoother-${t}-${n}`,operation:`GPUMultiscaleGeographicallyWeightedRegression`,variant:`full-backfit-smoother`,invocationCount:c*c,bindings:[...g,{name:`source`,view:m,type:`f32`,access:`read`},{name:`destination`,view:h,type:`f32`,access:`read_write`},{name:`state`,view:i,type:`u32`,access:`read`}],declarations:_,body:`let focus = index / ${c}u; let responseRow = index % ${c}u; let base = index * ${l}u; for (var component = 0u; component < ${l}u; component++) { destination[destinationOffset + base + component] = source[sourceOffset + base + component]; } if (state[stateOffset] != 3u || !included(focus) || !included(responseRow)) { return; }
let bandwidth = parameters[parametersOffset + ${n+1}u]; var denominator = 0.0; var numerator = 0.0;
for (var row = 0u; row < ${c}u; row++) { if (!included(row)) { continue; } let value = predictor(row, ${n}u); let weight = spatialWeight(focus,row,bandwidth); denominator += weight * value * value; var partial = select(0.0,1.0,row == responseRow); for (var other = 0u; other < ${l}u; other++) { if (other != ${n}u) { partial -= predictor(row,other) * source[sourceOffset + (row * ${c}u + responseRow) * ${l}u + other]; } } numerator += weight * value * partial; }
destination[destinationOffset + base + ${n}u] = select(getNaN(),numerator / denominator,denominator > 0.0);`});return[m,h]=[h,m],r},finish:()=>[u(e,{id:`${a}-full-inference-summary`,operation:`GPUMultiscaleGeographicallyWeightedRegression`,variant:`full-inference-summary`,invocationCount:1,bindings:[{name:`smoother`,view:m,type:`f32`,access:`read`},{name:`predictors`,view:o.predictors,type:`f32`,access:`read`},{name:`response`,view:o.response,type:`f32`,access:`read`},{name:`fitted`,view:o.output.fitted,type:`f32`,access:`read`},{name:`codes`,view:o.output.status.codes,type:`u32`,access:`read`},{name:`summary`,view:s.output.summary,type:`f32`,access:`read_write`},{name:`status`,view:s.output.status,type:`u32`,access:`read_write`}],declarations:`${f}\n${d}`,body:`status[statusOffset] = 1u; for (var slot = 0u; slot < 4u; slot++) { summary[summaryOffset + slot] = getNaN(); } var trace = 0.0; var traceSquared = 0.0; var residualSquares = 0.0; var count = 0u;
for (var focus = 0u; focus < ${c}u; focus++) { let code = codes[codesOffset + focus]; if (code == 1u) { continue; } if (code != 0u) { return; } count++; let residual = response[responseOffset + focus] - fitted[fittedOffset + focus]; residualSquares += residual * residual;
for (var responseRow = 0u; responseRow < ${c}u; responseRow++) { var influence = 0.0; for (var column = 0u; column < ${l}u; column++) { var value = 1.0; if (column > 0u) { value = predictors[predictorsOffset + focus * ${l-1}u + column - 1u]; } influence += value * smoother[smootherOffset + (focus * ${c}u + responseRow) * ${l}u + column]; } traceSquared += influence * influence; if (focus == responseRow) { trace += influence; } } }
let degrees = f32(count) - ${s.varianceConvention===`smoother-df`?`(2.0 * trace - traceSquared)`:`trace`}; let variance = residualSquares / degrees;
summary[summaryOffset] = trace; summary[summaryOffset + 1u] = traceSquared; summary[summaryOffset + 2u] = variance; summary[summaryOffset + 3u] = degrees; status[statusOffset] = select(2u,0u,count > 0u && degrees > 0.0 && variance >= 0.0 && isFiniteBits(variance) && isFiniteBits(traceSquared));`}),u(e,{id:`${a}-full-inference-publish`,operation:`GPUMultiscaleGeographicallyWeightedRegression`,variant:`full-inference-publish`,invocationCount:c*l,bindings:[{name:`smoother`,view:m,type:`f32`,access:`read`},{name:`predictors`,view:o.predictors,type:`f32`,access:`read`},{name:`codes`,view:o.output.status.codes,type:`u32`,access:`read`},{name:`summary`,view:s.output.summary,type:`f32`,access:`read`},{name:`status`,view:s.output.status,type:`u32`,access:`read`},{name:`standardErrors`,view:s.output.standardErrors,type:`f32`,access:`read_write`},{name:`effective`,view:s.output.effectiveParameters,type:`f32`,access:`read_write`}],declarations:f,body:`let focus = index / ${l}u; let column = index % ${l}u; standardErrors[standardErrorsOffset + index] = getNaN(); if (focus == 0u) { effective[effectiveOffset + column] = getNaN(); } if (status[statusOffset] != 0u) { return; }
if (codes[codesOffset + focus] == 0u) { var sum = 0.0; for (var responseRow = 0u; responseRow < ${c}u; responseRow++) { let derivative = smoother[smootherOffset + (focus * ${c}u + responseRow) * ${l}u + column]; sum += derivative * derivative; } standardErrors[standardErrorsOffset + index] = sqrt(max(0.0,sum * summary[summaryOffset + 2u])); }
if (focus == 0u) { var count = 0.0; for (var row = 0u; row < ${c}u; row++) { if (codes[codesOffset + row] != 0u) { continue; } var value = 1.0; if (column > 0u) { value = predictors[predictorsOffset + row * ${l-1}u + column - 1u]; } count += value * smoother[smootherOffset + (row * ${c}u + row) * ${l}u + column]; } effective[effectiveOffset + column] = count; }`})]}}var G=`GPUMultiscaleGeographicallyWeightedRegression`,K=4,q=7,J=3;function te(e,t,n,r){g(e,!t||n,`bandwidthSelection requires bandwidthCandidates`),g(e,t?.criterion===void 0||[`leave-one-out`,`AICc`].includes(t.criterion),`bandwidthSelection.criterion must be leave-one-out or AICc`),g(e,t?.searchMethod===void 0||[`finite-grid`,`golden-section`].includes(t.searchMethod),`bandwidthSelection.searchMethod must be finite-grid or golden-section`);let i=t?.searchMethod===`golden-section`;g(e,!i||r===2,`golden-section bandwidthCandidateCount must be 2 (lower and upper bounds)`),t?.maximumSearchIterations!==void 0&&(g(e,i,`maximumSearchIterations requires golden-section searchMethod`),_(e,`maximumSearchIterations`,t.maximumSearchIterations),h(e,`maximumSearchIterations`,t.maximumSearchIterations,64))}function ne(e,t,n){let r=n?.searchMethod===`golden-section`,i=r?3:t,a=n?.maximumSearchIterations??30,o=r?a+2:t;return{"bandwidth scores":e*i*K,"bandwidth partials":M(e).tileCount*i*K,"bandwidth totals":i*K,"bandwidth table":o*J,"bandwidth bounds":q,"bandwidth ladders":1+a*q,"bandwidth state":12,"bandwidth golden-section state":8,"bandwidth dispatch gate":12}}function re(e,t,n,i,a){let o=a?.searchMethod===`golden-section`,s=o?3:i,c=a?.maximumSearchIterations??30,l=o?c+2:i,u=M(n);y(e.device,t,`bandwidth scores`,n*s*K*4,`reduce rows or bandwidthCandidateCount`);let d=(n,i)=>r(e,`${t}-selection-${n}`,`float32`,i);return{golden:o,width:s,maximumIterations:c,tableRows:l,layout:u,scores:d(`scores`,n*s*K),partials:d(`partials`,u.tileCount*s*K),totals:d(`totals`,s*K),table:d(`table`,l*J),bounds:d(`bounds`,q),ladders:d(`ladders`,1+c*q),searchState:d(`state`,12)}}function ie(e,t){let{id:n,workspace:r,rows:i,column:o}=t,{golden:p,width:m,maximumIterations:h,tableRows:g,layout:_,scores:v,partials:y,totals:b,table:x,bounds:S,ladders:C,searchState:w}=r,T=t.selection?.criterion===`AICc`,E=2**-Math.ceil(Math.log2(i)/2),D=1/(E*E),O=(e,t,n=`f32`)=>({name:e,view:t,type:n,access:`read`}),k=(e,t)=>({name:e,view:t,type:`f32`,access:`read_write`}),M=[],N=`${f}\n${d}`;p&&M.push(u(e,{id:`${n}-bounds`,operation:G,variant:`selection-bounds`,invocationCount:q,bindings:[O(`candidates`,t.candidates),k(`bounds`,S),k(`ladders`,C)],declarations:N,body:`let lower = candidates[candidatesOffset + ${o*2}u];
let upper = candidates[candidatesOffset + ${o*2+1}u];
let valid = lower > 0.0 && upper >= lower && isFiniteBits(lower) && isFiniteBits(upper);
let field = index;
var value = 0.0;
  if (field == 2u) { value = 2.0; }
  if (field == 4u) { value = select(getNaN(), lower, valid); }
  if (field == 5u) { value = select(getNaN(), upper, valid); }
  bounds[boundsOffset + field] = value;
  ladders[laddersOffset + 1u + field] = value;
if (index == 0u) { ladders[laddersOffset] = 0.0; }`}));let P=p?A(e,{id:n,operation:G,rowCount:i,tileRows:_.tileRowCount,tileCount:_.tileCount,candidateWidth:m,maximumIterations:h,tableRowCount:g,callerBoundsOnly:!0,probeUpperBound:!0,runningState:t.state,dispatchShapes:[[i],[_.tileCount,64],[m],[m*K]],common:`${N}
const ROW_COUNT: u32 = ${i}u;
const TILE_ROWS: u32 = ${_.tileRowCount}u;
const TILE_COUNT: u32 = ${_.tileCount}u;
const TABLE_ROWS: u32 = ${g}u;
const TABLE_STRIDE: u32 = ${J}u;
const HEADER: u32 = 4u;
const PARAMETER_LENGTH: u32 = ${q}u;
const MAXIMUM_NEIGHBORS: u32 = 1u;
const P: u32 = 1u;
const SENTINEL: f32 = 3.402823e38;`,candidateDeclarations:``,focalRowWGSL:``,hasGrid:!1,candidateBindings:[],rowValid:t.state,ladders:C,parameters:S,candidateScratch:v,tileCandidates:y,candidateTable:x,searchState:w}):void 0,F=P?void 0:l(e,{id:`${n}-backfit-dispatch`,operation:G,shapes:[[i],[_.tileCount,64],[m],[m*K]]});P&&M.push(...P.setupNodes),F&&M.push(u(e,{id:`${n}-backfit-gate`,operation:G,variant:`selection-backfit-gate`,invocationCount:1,bindings:[O(`state`,t.state,`u32`),F.binding],body:`if (state[stateOffset] == 3u) { ${F.armWGSL} } else { ${F.disarmWGSL} }`}));for(let r=0;r<(p?h:1);r++){let c=`${n}-pass-${r}`,l=P?.gate??F;M.push(u(e,{id:`${c}-evaluate`,operation:G,variant:`selection-evaluate`,invocationCount:i,...l?.getGate(0),bindings:[O(`positions`,t.positions),O(`pairs`,t.pairs),O(`candidates`,p?C:t.candidates),k(`scores`,v),O(`state`,t.state,`u32`),...t.gridBindings],declarations:t.common,body:`for (var candidate = 0u; candidate < ${m}u; candidate++) {
  let slot = scoresOffset + (index * ${m}u + candidate) * 4u;
  for (var field = 0u; field < 4u; field++) { scores[slot + field] = 0.0; }
  ${p?`let candidateBlock = candidatesOffset + 1u + u32(candidates[candidatesOffset]) * ${q}u;
  if (candidate >= u32(candidates[candidateBlock + 2u])) { continue; }
  let bandwidth = candidates[candidateBlock + 4u + candidate];`:`let bandwidth = candidates[candidatesOffset + ${o*t.candidateCount}u + candidate];`}
  if (state[stateOffset] != 3u || !isActive(index)) { continue; }
  scores[slot + 2u] = 1.0;
  scores[slot + 3u] = 1.0;
  if (!(bandwidth > 0.0) || !isFiniteBits(bandwidth)) { continue; }
  var numerator = 0.0; var denominator = 0.0;
  let focus = index;
  let origin = getPosition(focus);
  ${s(`${T?``:`if (row == focus) { continue; }`}
    let weight = spatialWeight(focus, row, bandwidth);
    if (!(weight > 0.0)) { continue; }
    let value = pairs[pairsOffset + 2u * row + 1u];
    numerator += weight * value * pairs[pairsOffset + 2u * row];
    denominator += weight * value * value;`,t.hasGrid,`isActive(row)`)}
  if (!(denominator > 0.0) || !isFiniteBits(denominator)) { continue; }
  let value = pairs[pairsOffset + 2u * focus + 1u];
  // Form the coefficient first: value * numerator can overflow while both the coefficient
  // and the fitted contribution remain finite.
  let coefficient = numerator / denominator;
  let residual = pairs[pairsOffset + 2u * focus] - value * coefficient;
  if (!isFiniteBits(residual)) { continue; }
  ${T?`let scaledResidual = residual * f32(${E});
  scores[slot] = scaledResidual * scaledResidual;`:`scores[slot] = residual * residual;`}
  scores[slot + 1u] = ${T?`value * value / denominator`:`0.0`};
  scores[slot + 3u] = 0.0;
}`})),M.push(...j(e,{id:c,operation:G,variant:`selection-sum`,rowCount:i,layout:_,sumCount:m*K,bindings:[O(`scores`,v)],partial:y,output:b,tileGate:l?.getGate(1),mergeGate:l?.getGate(3),declarations:`fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  for (var term = 0u; term < TERM_COUNT; term++) { (*terms)[term] += scores[scoresOffset + row * TERM_COUNT + term]; }
}`})),M.push(u(e,{id:`${c}-score`,operation:G,variant:`selection-score`,invocationCount:m,...l?.getGate(2),bindings:[O(`totals`,b),O(`candidates`,p?C:t.candidates),k(`table`,x)],declarations:N,body:`${p?`let cursor = u32(candidates[candidatesOffset]);
let candidateBlock = candidatesOffset + 1u + cursor * ${q}u;
if (index >= u32(candidates[candidateBlock + 2u])) { return; }
let bandwidth = candidates[candidateBlock + 4u + index];
let tableRow = select(cursor + 2u, index, cursor == 0u);`:`let bandwidth = candidates[candidatesOffset + ${o*t.candidateCount}u + index];
let tableRow = index;`}
let base = totalsOffset + index * 4u;
let residualSquares = totals[base]; let trace = totals[base + 1u]; let count = totals[base + 2u];
var score = getNaN();
if (count > 0.0 && totals[base + 3u] == 0.0 && isFiniteBits(residualSquares)) {
  ${T?`let degrees = count - trace - 2.0;
  let variance = residualSquares * (f32(${D}) / count);
  if (degrees > 0.0 && isFiniteBits(trace) && isFiniteBits(variance)) {
    score = count * log(max(variance, ${a})) + count * log(6.283185307179586) + count * (count + trace) / degrees;
  }`:`score = residualSquares;`}
}
let slot = tableOffset + tableRow * 3u;
table[slot] = bandwidth; table[slot + 1u] = score; table[slot + 2u] = 0.0;`})),P&&M.push(P.createIterationNode(r+1))}return M.push(u(e,{id:`${n}-publish`,operation:G,variant:`selection-publish`,invocationCount:1,bindings:[O(`table`,x),...p?[O(`searchState`,w)]:[],k(`parameters`,t.parameters),O(`state`,t.state,`u32`),...p&&t.componentSearchIterationLimitReached?[{name:`componentSearchIterationLimitReached`,view:t.componentSearchIterationLimitReached,type:`u32`,access:`read_write`}]:[]],declarations:N,body:`if (state[stateOffset] != 3u) { return; }
var best = getNaN();
${p?`let row = i32(searchState[searchStateOffset + ${c.OPTIMUM_ROW}u]);
if (row >= 0 && row < ${g}) {
  let slot = tableOffset + u32(row) * 3u;
  if (isFiniteBits(table[slot + 1u])) { best = table[slot]; }
}`:`var bestScore = 3.402823e38;
for (var candidate = 0u; candidate < ${t.candidateCount}u; candidate++) {
  let slot = tableOffset + candidate * 3u;
  let score = table[slot + 1u];
  if (isFiniteBits(score) && score < bestScore) { bestScore = score; best = table[slot]; }
}`}
parameters[parametersOffset + ${o+1}u] = best;
${p&&t.componentSearchIterationLimitReached?`if (isFiniteBits(best) && searchState[searchStateOffset + ${c.CONVERGED}u] == 0.0) {
  componentSearchIterationLimitReached[componentSearchIterationLimitReachedOffset + ${o}u] = 1u;
}`:``}`})),M}function ae(e,t){let{id:n,predictorCount:i}=t,a=i+1,o=i*i+i,s=M(t.rowCount);y(e.device,n,`globalFitPartial`,Y(t.rowCount,i)*4,`reduce rowCount or predictorCount`);let c=r(e,`${n}-global-partial`,`float32`,Y(t.rowCount,i)),l=r(e,`${n}-global-sample`,`float32`,3),d=r(e,`${n}-global-deviations`,`float32`,a),f=r(e,`${n}-global-moments`,`float32`,o),m={name:`sample`,view:l,type:`f32`,access:`read`},h={name:`deviations`,view:d,type:`f32`,access:`read`},g=t.declarations.replace(/^const ROW_COUNT:.*$/m,``),_=`
// Bound response magnitudes below 4 before subtracting, including opposite finite extremes.
// Keep both powers normal and ldexp's exponent in [-126, 126], including a zero sample.
fn getGlobalResponseScale() -> f32 {
  let exponent = bitcast<u32>(sample[sampleOffset + 2u]) & 0x7f800000u;
  return bitcast<f32>(clamp(exponent, 0x00800000u, 0x7e800000u));
}
fn getGlobalValue(row: u32, column: u32) -> f32 {
  if (column == 0u) {
    let exponent = i32(bitcast<u32>(getGlobalResponseScale()) >> 23u) - 127;
    return ldexp(response[responseOffset + row], -exponent);
  }
  return predictor(row, column);
}
fn getGlobalReference(column: u32) -> f32 {
  return getGlobalValue(u32(max(sample[sampleOffset + 1u], 1.0)) - 1u, column);
}`,v=`${_}
fn getGlobalDeviation(column: u32) -> f32 {
  return deviations[deviationsOffset + column];
}
fn getGlobalMean(column: u32) -> f32 {
  return getGlobalReference(column) + getGlobalDeviation(column);
}`,b={operation:`GPUMultiscaleGeographicallyWeightedRegression`,rowCount:t.rowCount,layout:s,partial:c};return[...j(e,{...b,id:`${n}-global-sample`,variant:`global-sample`,sumCount:1,maximumCount:2,bindings:t.bindings,output:l,declarations:`${g}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  if (!included(row)) { return; }
  (*terms)[0] += 1.0;
  (*terms)[1] = max((*terms)[1], f32(row + 1u));
  (*terms)[2] = max((*terms)[2], abs(response[responseOffset + row]));
}`}),...j(e,{...b,id:`${n}-global-deviations`,variant:`global-deviations`,sumCount:a,bindings:[...t.bindings,m],output:d,declarations:`${g}
${_}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  if (!included(row)) { return; }
  for (var column = 0u; column < ${a}u; column++) {
    // Normalize before either the tile sum or its pairwise merge: a finite mean can have
    // unnormalized deviation partials larger than float32 (for example 64 * -1e37).
    (*terms)[column] += (getGlobalValue(row, column) - getGlobalReference(column)) / sample[sampleOffset];
  }
}`}),...j(e,{...b,id:`${n}-global-moments`,variant:`global-moments`,sumCount:o,bindings:[...t.bindings,m,h],output:f,declarations:`${g}
${v}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  if (!included(row)) { return; }
  let centeredResponse = (getGlobalValue(row, 0u) - getGlobalReference(0u)) - getGlobalDeviation(0u);
  for (var column = 0u; column < ${i}u; column++) {
    let first = (predictor(row, column + 1u) - getGlobalReference(column + 1u)) - getGlobalDeviation(column + 1u);
    (*terms)[column] += first * centeredResponse;
    for (var other = 0u; other <= column; other++) {
      (*terms)[${i}u + column * ${i}u + other] += first * ((predictor(row, other + 1u) - getGlobalReference(other + 1u)) - getGlobalDeviation(other + 1u));
    }
  }
}`}),u(e,{id:`${n}-global-fit`,operation:`GPUMultiscaleGeographicallyWeightedRegression`,variant:`global-fit`,invocationCount:1,bindings:[...t.bindings,m,h,{name:`moments`,view:f,type:`f32`,access:`read`},{name:`globalFit`,view:t.output,type:`f32`,access:`read_write`}],declarations:`${t.declarations}
${p}
${v}
${N(i,25)}`,body:`for (var column = 0u; column < ${a}u; column++) { globalFit[globalFitOffset + column] = 0.0; }
if (sample[sampleOffset] < ${a}.0) { return; }
var gram: array<f32, ${i*i}>;
var cross: array<f32, ${i}>;
for (var column = 0u; column < ${i}u; column++) {
  cross[column] = moments[momentsOffset + column];
  for (var other = 0u; other <= column; other++) {
    gram[column * ${i}u + other] = moments[momentsOffset + ${i}u + column * ${i}u + other];
  }
}
var equilibration: array<f32, ${i}>;
for (var column = 0u; column < ${i}u; column++) {
  let diagonal = gram[column * ${i+1}u];
  if (!(diagonal > 0.0) || !isFiniteBits(diagonal)) { return; }
  equilibration[column] = inverseSqrt(diagonal);
}
for (var column = 0u; column < ${i}u; column++) {
  cross[column] *= equilibration[column];
  for (var other = 0u; other <= column; other++) { gram[column * ${i}u + other] *= equilibration[column] * equilibration[other]; }
}
if (!choleskyFactor_${i}(&gram)) { return; }
choleskySolve_${i}(&gram, &cross);
var intercept = getGlobalMean(0u);
for (var column = 0u; column < ${i}u; column++) {
  let scaledSlope = cross[column] * equilibration[column];
  let slope = scaledSlope * getGlobalResponseScale();
  if (!isFiniteBits(slope)) { return; }
  globalFit[globalFitOffset + column + 1u] = slope;
  intercept -= scaledSlope * getGlobalMean(column + 1u);
}
intercept *= getGlobalResponseScale();
if (isFiniteBits(intercept)) { globalFit[globalFitOffset] = intercept; }`})]}function Y(e,t){return M(e).tileCount*Math.max(3,t*t+t)}var oe=256,X=256,se=256,ce=1024,Z={CONVERGED:0,EXCLUDED:1,SINGULAR:2,ITERATION_LIMIT:3,NUMERICAL_FAILURE:4};function le(e){switch(e){case Z.CONVERGED:return`converged`;case Z.EXCLUDED:return`excluded`;case Z.SINGULAR:return`singular`;case Z.ITERATION_LIMIT:return`iteration-limit`;case Z.NUMERICAL_FAILURE:return`numerical-failure`;default:return`unknown`}}var ue=class{id;estimatedCommandNodeCount;spatialContext;sphereRadius;props;constructor(t){this.id=t.id??`multiscale-geographically-weighted-regression`,this.spatialContext=k(this.id,t.spatialContext,{...D,measure:`distance`}),this.sphereRadius=this.spatialContext.metric===`great-circle`?O(this.spatialContext):null;let n=this.sphereRadius===null&&(t.kernel===`bisquare`||t.gaussianCutoff!==void 0),r=t.indexGridSize===!1||!n?void 0:t.indexGridSize??o(t.response.length);this.props={...t,indexGridSize:r},T(this.id,{...t,indexGridSize:t.indexGridSize||r},this.sphereRadius!==null,r?void 0:w),te(this.id,t.bandwidthSelection,!!t.bandwidthCandidates,t.bandwidthCandidateCount);let i=t.bandwidthCandidates?t.bandwidthSelection?.searchMethod===`golden-section`?3+5*(t.bandwidthSelection.maximumSearchIterations??30):6:0;if(this.estimatedCommandNodeCount=oe+(t.maximumIterations??32)*(3+(t.predictorCount+1)*(3+i)),v(this.id,this.estimatedCommandNodeCount),t.parameters.length<t.predictorCount+2)throw Error(`${this.id}: supply tolerance and one bandwidth per coefficient`);if(t.bandwidthCandidates){let e=t.bandwidthCandidateCount;if(!e||!Number.isInteger(e)||e>32)throw Error(`${this.id}: candidate count must be 1 to 32`);if(m(t.bandwidthCandidates,[`float32`],this.id),t.bandwidthCandidates.length!==(t.predictorCount+1)*e)throw Error(`${this.id}: candidate grid shape mismatch`)}else if(t.bandwidthCandidateCount!==void 0)throw Error(`${this.id}: candidate count requires candidates`);if(t.output.conditionalStandardErrors&&!t.noiseVariance)throw Error(`${this.id}: conditional inference requires known noise variance`);if(t.noiseVariance&&(m(t.noiseVariance,[`float32`],this.id),t.noiseVariance.length!==1))throw Error(`${this.id}: noise variance must be scalar`);for(let[e,n]of Object.entries(Q(t.output))){if(!n)continue;m(n,[[`coefficients`,`fitted`,`selectedBandwidths`,`conditionalStandardErrors`].includes(e)?`float32`:`uint32`],this.id);let r=e===`iterations`?1:e===`selectedBandwidths`||e===`componentSearchIterationLimitReached`?t.predictorCount+1:t.response.length*([`coefficients`,`conditionalStandardErrors`].includes(e)?t.predictorCount+1:1);if(n.length!==r)throw Error(`${this.id}: output length mismatch`)}if(t.inference){if(t.response.length>128||t.response.length**3*(t.predictorCount+1)*(t.maximumIterations??32)>268435456)throw Error(`${this.id}: full inference work budget exceeded`);if(![`residual-df`,`smoother-df`].includes(t.inference.varianceConvention??`residual-df`))throw Error(`${this.id}: unsupported variance convention`);for(let[e,n]of Object.entries(t.inference.output)){m(n,[e===`status`?`uint32`:`float32`],this.id);let r=e===`standardErrors`?t.response.length*(t.predictorCount+1):e===`effectiveParameters`?t.predictorCount+1:e===`summary`?4:1;if(n.length!==r)throw Error(`${this.id}: inference shape mismatch`)}}e(this.id,{...Q(t.output),...Object.fromEntries(Object.entries(t.inference?.output??{}).map(([e,t])=>[`inference ${e}`,t]))},[[`coefficients`,`fitted`,`status`,`iterations`],[`inference summary`,`inference status`],{read:[`fitted`,`status`],write:[`inference summary`,`inference status`]},{read:[`inference summary`,`inference status`,`status`],write:[`inference standardErrors`,`inference effectiveParameters`]}]),x(this.id,[...Object.values(Q(t.output)),...Object.values(t.inference?.output??{})],[...C(this.props).map(e=>e.view),t.bandwidthCandidates,t.output.conditionalStandardErrors?t.noiseVariance:void 0])}getCommandNodes(e){let{id:a,props:o}=this,c=C(o);n(e,a,{...Object.fromEntries(c.map(e=>[e.name,e.view])),bandwidthCandidates:o.bandwidthCandidates,noiseVariance:o.output.conditionalStandardErrors?o.noiseVariance:void 0,...Q(o.output),...Object.fromEntries(Object.entries(o.inference?.output??{}).map(([e,t])=>[`inference ${e}`,t]))});let l=o.response.length,p=o.predictorCount+1;t(e.device,a,{"coefficient surfaces":l*p,"working parameters":p+1,"iteration state":2,"component search flags":p,"convergence partials":3*Math.min(se,Math.max(1,Math.ceil(l*p/ce))),"row pairs":2*l,...Object.fromEntries(i(l,o.indexGridSize??null).map(([e,t])=>[e,t/4])),...o.inference?{"response smoother":l*l*p}:{"global fit":p,"global fit partials":Y(l,o.predictorCount)},...o.bandwidthCandidates?ne(l,o.bandwidthCandidateCount,o.bandwidthSelection):{}});let m=o.bandwidthCandidates?re(e,a,o.response.length,o.bandwidthCandidateCount,o.bandwidthSelection):void 0,h=o.response.length,g=o.predictorCount+1,_=o.maximumIterations??32,v=r(e,`${a}-first`,`float32`,h*g),y=r(e,`${a}-second`,`float32`,h*g),x=r(e,`${a}-previous`,`float32`,h*g),w=r(e,`${a}-state`,`uint32`,2),T=o.output.status.componentSearchIterationLimitReached??r(e,`${a}-component-search-limit`,`uint32`,g),D=r(e,`${a}-working-parameters`,`float32`,g+1),O=Math.min(se,Math.max(1,Math.ceil(h*g/ce))),k=r(e,`${a}-convergence-partials`,`uint32`,3*O),A=E(e,a,{...o,indexGridSize:o.indexGridSize||void 0},this.sphereRadius),j=A.props,M=A.grid!==void 0,N=C({...j,parameters:D}),P=S(j,{sphereRadius:this.sphereRadius}),F=A.grid?[{name:`cellOffsets`,view:A.grid.cellOffsets,type:`u32`,access:`read`},{name:`sortedIds`,view:A.grid.sortedIds,type:`u32`,access:`read`}]:[],I=`${S(j,{sphereRadius:this.sphereRadius,grid:j.indexGridSize,neighborhoodOnly:!0})}
fn isActive(row: u32) -> bool { return isFiniteBits(pairs[pairsOffset + 2u * row]); }`,L=M||o.bandwidthCandidates?r(e,`${a}-pairs`,`float32`,2*h):void 0,R=e=>`{\n${s(e,!0,`isActive(row)`)}\n}`;b(a,e,[...N.map(e=>e.view),...Object.values(Q(o.output)),o.parameters,o.bandwidthCandidates,o.output.conditionalStandardErrors?o.noiseVariance:void 0]);let z=o.inference?void 0:r(e,`${a}-global-fit`,`float32`,g),B=[...A.nodes];z&&B.push(...ae(e,{id:a,rowCount:h,predictorCount:o.predictorCount,bindings:N.filter(e=>e.name!==`parameters`),declarations:P,output:z})),B.push(u(e,{id:`${a}-initialize`,operation:`GPUMultiscaleGeographicallyWeightedRegression`,variant:`initialize`,invocationCount:h*g,bindings:[{name:`coefficients`,view:v,type:`f32`,access:`read_write`},{name:`state`,view:w,type:`u32`,access:`read_write`},{name:`componentSearchIterationLimitReached`,view:T,type:`u32`,access:`read_write`},...z?[{name:`globalFit`,view:z,type:`f32`,access:`read`}]:[]],body:`coefficients[coefficientsOffset + index] = ${z?`globalFit[globalFitOffset + index % ${g}u]`:`0.0`};
if (index < ${g}u) { componentSearchIterationLimitReached[componentSearchIterationLimitReachedOffset + index] = 0u; }
if (index == 0u) { state[stateOffset] = 3u; state[stateOffset + 1u] = 0u; }`})),B.push(u(e,{id:`${a}-parameters`,operation:`GPUMultiscaleGeographicallyWeightedRegression`,variant:`parameters`,invocationCount:g+1,bindings:[{name:`source`,view:o.parameters,type:`f32`,access:`read`},{name:`parameters`,view:D,type:`f32`,access:`read_write`}],body:`parameters[parametersOffset + index] = source[sourceOffset + index];`}));let V=o.inference?ee(e,this,D,w):void 0;V&&B.push(V.initialize());let H=v,U=y;for(let t=0;t<_;t++){B.push(u(e,{id:`${a}-snapshot-${t}`,operation:`GPUMultiscaleGeographicallyWeightedRegression`,variant:`snapshot`,invocationCount:h*g,bindings:[{name:`source`,view:H,type:`f32`,access:`read`},{name:`destination`,view:x,type:`f32`,access:`read_write`}],body:`destination[destinationOffset + index] = source[sourceOffset + index];`}));for(let n=0;n<g;n++)L&&B.push(u(e,{id:`${a}-prepare-${t}-${n}`,operation:`GPUMultiscaleGeographicallyWeightedRegression`,variant:`prepare-${n}`,invocationCount:h,bindings:[...N.filter(e=>e.name!==`parameters`),{name:`source`,view:H,type:`f32`,access:`read`},{name:`destination`,view:U,type:`f32`,access:`read_write`},{name:`pairs`,view:L,type:`f32`,access:`read_write`}],declarations:P,body:`var partial = response[responseOffset + index];
for (var other = 0u; other < ${g}u; other++) {
  destination[destinationOffset + index * ${g}u + other] = source[sourceOffset + index * ${g}u + other];
  if (other != ${n}u) { partial -= predictor(index, other) * source[sourceOffset + index * ${g}u + other]; }
}
pairs[pairsOffset + 2u * index] = select(getNaN(), partial, included(index));
pairs[pairsOffset + 2u * index + 1u] = predictor(index, ${n}u);`})),o.bandwidthCandidates&&L&&m&&B.push(...ie(e,{id:`${a}-selection-${t}-${n}`,positions:j.positions,pairs:L,gridBindings:F,common:I,hasGrid:M,rows:h,column:n,candidates:o.bandwidthCandidates,candidateCount:o.bandwidthCandidateCount,selection:o.bandwidthSelection,parameters:D,state:w,componentSearchIterationLimitReached:T,workspace:m})),B.push(M&&L?u(e,{id:`${a}-sweep-${t}-column-${n}`,operation:`GPUMultiscaleGeographicallyWeightedRegression`,variant:`column-${n}`,invocationCount:h,bindings:[{name:`positions`,view:j.positions,type:`f32`,access:`read`},{name:`pairs`,view:L,type:`f32`,access:`read`},{name:`parameters`,view:D,type:`f32`,access:`read`},{name:`destination`,view:U,type:`f32`,access:`read_write`},{name:`state`,view:w,type:`u32`,access:`read`},...F],declarations:I,body:`if (state[stateOffset] != 3u) { return; }
var coefficient = getNaN(); let bandwidth = parameters[parametersOffset + ${n+1}u];
if (isActive(index) && bandwidth > 0.0 && isFiniteBits(bandwidth)) {
  var numerator = 0.0; var denominator = 0.0;
  let origin = getPosition(index);
  ${R(`let weight = spatialWeight(index, row, bandwidth); if (!(weight > 0.0)) { continue; }
    let value = pairs[pairsOffset + 2u * row + 1u]; let partial = pairs[pairsOffset + 2u * row];
    numerator += weight * value * partial; denominator += weight * value * value;`)}
  if (denominator > 0.0 && isFiniteBits(denominator)) { coefficient = numerator / denominator; }
}
destination[destinationOffset + index * ${g}u + ${n}u] = coefficient;`}):u(e,{id:`${a}-sweep-${t}-column-${n}`,operation:`GPUMultiscaleGeographicallyWeightedRegression`,variant:`column-${n}`,invocationCount:h,bindings:[...N,{name:`source`,view:H,type:`f32`,access:`read`},{name:`destination`,view:U,type:`f32`,access:`read_write`},{name:`state`,view:w,type:`u32`,access:`read`}],declarations:P,body:`for (var column = 0u; column < ${g}u; column++) { destination[destinationOffset + index * ${g}u + column] = source[sourceOffset + index * ${g}u + column]; }
if (state[stateOffset] != 3u) { return; }
var coefficient = getNaN(); let bandwidth = parameters[parametersOffset + ${n+1}u];
if (included(index) && bandwidth > 0.0 && isFiniteBits(bandwidth)) {
  var numerator = 0.0; var denominator = 0.0;
  for (var row = 0u; row < ${h}u; row++) {
    if (!included(row)) { continue; }
    let weight = spatialWeight(index, row, bandwidth); if (!(weight > 0.0)) { continue; }
    var partial = response[responseOffset + row];
    for (var other = 0u; other < ${g}u; other++) {
      if (other != ${n}u) { partial -= predictor(row, other) * source[sourceOffset + row * ${g}u + other]; }
    }
    let value = predictor(row, ${n}u); numerator += weight * value * partial; denominator += weight * value * value;
  }
  if (denominator > 0.0 && isFiniteBits(denominator)) { coefficient = numerator / denominator; }
}
destination[destinationOffset + index * ${g}u + ${n}u] = coefficient;`})),V&&B.push(V.advance(t,n)),[H,U]=[U,H];B.push(u(e,{id:`${a}-convergence-${t}`,operation:`GPUMultiscaleGeographicallyWeightedRegression`,variant:`convergence`,invocationCount:O*X,workgroupSize:X,guardIndex:!1,bindings:[...N.filter(e=>e.name!==`parameters`),{name:`source`,view:H,type:`f32`,access:`read`},{name:`previous`,view:x,type:`f32`,access:`read`},{name:`partials`,view:k,type:`u32`,access:`read_write`},{name:`state`,view:w,type:`u32`,access:`read`}],declarations:`${P}
var<workgroup> sharedChange: array<f32, ${X}>;
var<workgroup> sharedKey: array<u32, ${X}>;
var<workgroup> sharedCount: array<u32, ${X}>;`,body:`let isRunning = state[stateOffset] == 3u;
var localChange = 0.0; var localKey = 0xffffffffu; var localCount = 0u;
let elementCount = select(0u, ${h*g}u, isRunning);
for (var element = index; element < elementCount; element += INVOCATION_COUNT) {
  let row = element / ${g}u; let column = element % ${g}u;
  if (!included(row)) { continue; }
  if (column == 0u) { localCount++; }
  let coefficient = source[sourceOffset + element];
  // Key bit 0 distinguishes a non-finite contribution (state 4) from a coefficient (state 2).
  if (!isFiniteBits(coefficient)) {
    let isInfinity = (bitcast<u32>(coefficient) & 0x7fffffffu) == 0x7f800000u;
    localKey = element * 2u + select(0u, 1u, isInfinity); break;
  }
  if (column == 0u) {
    var prediction = 0.0;
    var predictionInputsFinite = true;
    for (var predictionColumn = 0u; predictionColumn < ${g}u; predictionColumn++) {
      let predictionCoefficient = source[sourceOffset + row * ${g}u + predictionColumn];
      predictionInputsFinite = predictionInputsFinite && isFiniteBits(predictionCoefficient);
      prediction += predictor(row, predictionColumn) * predictionCoefficient;
    }
    if (predictionInputsFinite && !isFiniteBits(prediction)) { localKey = element * 2u + 1u; break; }
  }
  let contribution = predictor(row, column) * coefficient;
  let previousContribution = predictor(row, column) * previous[previousOffset + element];
  if (!isFiniteBits(contribution) || !isFiniteBits(previousContribution)) { localKey = element * 2u + 1u; break; }
  localChange = max(localChange, abs(contribution - previousContribution) / max(1.0, abs(previousContribution)));
}
sharedChange[localInvocationIndex] = localChange; sharedKey[localInvocationIndex] = localKey; sharedCount[localInvocationIndex] = localCount;
workgroupBarrier();
for (var stride = ${X/2}u; stride > 0u; stride = stride >> 1u) {
  if (localInvocationIndex < stride) {
    sharedChange[localInvocationIndex] = max(sharedChange[localInvocationIndex], sharedChange[localInvocationIndex + stride]);
    sharedKey[localInvocationIndex] = min(sharedKey[localInvocationIndex], sharedKey[localInvocationIndex + stride]);
    sharedCount[localInvocationIndex] += sharedCount[localInvocationIndex + stride];
  }
  workgroupBarrier();
}
if (localInvocationIndex == 0u) {
  let group = index / ${X}u;
  partials[partialsOffset + 3u * group] = bitcast<u32>(sharedChange[0]);
  partials[partialsOffset + 3u * group + 1u] = sharedKey[0];
  partials[partialsOffset + 3u * group + 2u] = sharedCount[0];
}`}),u(e,{id:`${a}-convergence-decide-${t}`,operation:`GPUMultiscaleGeographicallyWeightedRegression`,variant:`convergence-decide`,invocationCount:1,bindings:[{name:`parameters`,view:D,type:`f32`,access:`read`},{name:`partials`,view:k,type:`u32`,access:`read`},{name:`state`,view:w,type:`u32`,access:`read_write`}],declarations:d,body:`if (state[stateOffset] != 3u) { return; }
state[stateOffset + 1u] += 1u;
let tolerance = parameters[parametersOffset]; if (!(tolerance > 0.0) || !isFiniteBits(tolerance)) { state[stateOffset] = 4u; return; }
var maximumChange = 0.0; var firstKey = 0xffffffffu; var count = 0u;
for (var group = 0u; group < ${O}u; group++) {
  maximumChange = max(maximumChange, bitcast<f32>(partials[partialsOffset + 3u * group]));
  firstKey = min(firstKey, partials[partialsOffset + 3u * group + 1u]);
  count += partials[partialsOffset + 3u * group + 2u];
}
if (firstKey != 0xffffffffu) { state[stateOffset] = select(2u, 4u, (firstKey & 1u) == 1u); return; }
if (count == 0u) { state[stateOffset] = 1u; } else if (maximumChange <= tolerance) { state[stateOffset] = 0u; }`}))}let W=r(e,`${a}-output-state`,`uint32`,h);if(B.push(u(e,{id:`${a}-row-status`,operation:`GPUMultiscaleGeographicallyWeightedRegression`,variant:`row-status`,invocationCount:h,bindings:[...N,{name:`state`,view:w,type:`u32`,access:`read`},{name:`rowStatus`,view:W,type:`u32`,access:`read_write`}],declarations:P,body:`rowStatus[rowStatusOffset + index] = select(1u, state[stateOffset], included(index));`})),B.push(u(e,{id:`${a}-publish`,operation:`GPUMultiscaleGeographicallyWeightedRegression`,variant:`publish`,invocationCount:h,bindings:[{name:`source`,view:H,type:`f32`,access:`read`},{name:`rowStatus`,view:W,type:`u32`,access:`read`},{name:`predictors`,view:o.predictors,type:`f32`,access:`read`},{name:`state`,view:w,type:`u32`,access:`read`},{name:`coefficients`,view:o.output.coefficients,type:`f32`,access:`read_write`},{name:`fitted`,view:o.output.fitted,type:`f32`,access:`read_write`},{name:`status`,view:o.output.status.codes,type:`u32`,access:`read_write`},{name:`iterations`,view:o.output.status.iterationCount,type:`u32`,access:`read_write`}],declarations:f,body:`let code = rowStatus[rowStatusOffset + index]; var prediction = 0.0;
for (var column = 0u; column < ${g}u; column++) {
  let coefficient = select(getNaN(), source[sourceOffset + index * ${g}u + column], code == 0u || code == 3u);
  coefficients[coefficientsOffset + index * ${g}u + column] = coefficient;
  var value = 1.0; if (column > 0u) { value = predictors[predictorsOffset + index * ${o.predictorCount}u + column - 1u]; }
  prediction += value * coefficient;
}
fitted[fittedOffset + index] = prediction; status[statusOffset + index] = code;
if (index == 0u) { iterations[iterationsOffset] = state[stateOffset + 1u]; }`})),o.output.selectedBandwidths&&B.push(u(e,{id:`${a}-publish-bandwidths`,operation:`GPUMultiscaleGeographicallyWeightedRegression`,variant:`publish-bandwidths`,invocationCount:g,bindings:[{name:`parameters`,view:D,type:`f32`,access:`read`},{name:`bandwidths`,view:o.output.selectedBandwidths,type:`f32`,access:`read_write`}],body:`bandwidths[bandwidthsOffset + index] = parameters[parametersOffset + index + 1u];`})),M&&L&&o.output.conditionalStandardErrors&&o.noiseVariance)for(let t=0;t<g;t++)B.push(u(e,{id:`${a}-conditional-prepare-${t}`,operation:`GPUMultiscaleGeographicallyWeightedRegression`,variant:`conditional-prepare-${t}`,invocationCount:h,bindings:[...N.filter(e=>e.name!==`parameters`),{name:`pairs`,view:L,type:`f32`,access:`read_write`}],declarations:P,body:`pairs[pairsOffset + 2u * index] = select(getNaN(), 0.0, included(index));
pairs[pairsOffset + 2u * index + 1u] = predictor(index, ${t}u);`}),u(e,{id:`${a}-conditional-inference-${t}`,operation:`GPUMultiscaleGeographicallyWeightedRegression`,variant:`conditional-inference-${t}`,invocationCount:h,bindings:[{name:`positions`,view:j.positions,type:`f32`,access:`read`},{name:`pairs`,view:L,type:`f32`,access:`read`},{name:`parameters`,view:D,type:`f32`,access:`read`},{name:`noiseVariance`,view:o.noiseVariance,type:`f32`,access:`read`},{name:`state`,view:w,type:`u32`,access:`read`},{name:`standardErrors`,view:o.output.conditionalStandardErrors,type:`f32`,access:`read_write`},...F],declarations:I,body:`var standardError = getNaN(); let noise = noiseVariance[noiseVarianceOffset];
if (state[stateOffset] == 0u && isActive(index) && noise >= 0.0 && isFiniteBits(noise)) {
  let bandwidth = parameters[parametersOffset + ${t+1}u]; var numerator = 0.0; var denominator = 0.0;
  let origin = getPosition(index);
  ${R(`let weight = spatialWeight(index, row, bandwidth); if (!(weight > 0.0)) { continue; }
    let value = pairs[pairsOffset + 2u * row + 1u];
    denominator += weight * value * value; numerator += weight * weight * value * value;`)}
  if (denominator > 0.0) { let estimate = sqrt(noise * numerator) / denominator; if (isFiniteBits(estimate)) { standardError = estimate; } }
}
standardErrors[standardErrorsOffset + index * ${g}u + ${t}u] = standardError;`}));else o.output.conditionalStandardErrors&&o.noiseVariance&&B.push(u(e,{id:`${a}-conditional-inference`,operation:`GPUMultiscaleGeographicallyWeightedRegression`,variant:`conditional-inference`,invocationCount:h*g,bindings:[...N,{name:`noiseVariance`,view:o.noiseVariance,type:`f32`,access:`read`},{name:`state`,view:w,type:`u32`,access:`read`},{name:`standardErrors`,view:o.output.conditionalStandardErrors,type:`f32`,access:`read_write`}],declarations:P,body:`let focus = index / ${g}u; let column = index % ${g}u;
var standardError = getNaN(); let noise = noiseVariance[noiseVarianceOffset];
if (state[stateOffset] == 0u && included(focus) && noise >= 0.0 && isFiniteBits(noise)) {
  let bandwidth = parameters[parametersOffset + column + 1u]; var numerator = 0.0; var denominator = 0.0;
  for (var row = 0u; row < ${h}u; row++) {
    if (!included(row)) { continue; }
    let weight = spatialWeight(focus, row, bandwidth); let value = predictor(row, column);
    denominator += weight * value * value; numerator += weight * weight * value * value;
  }
  if (denominator > 0.0) { let estimate = sqrt(noise * numerator) / denominator; if (isFiniteBits(estimate)) { standardError = estimate; } }
}
standardErrors[standardErrorsOffset + index] = standardError;`}));return V&&B.push(...V.finish()),B}};function Q(e){let{status:t,...n}=e;return{...n,status:t.codes,iterations:t.iterationCount,componentSearchIterationLimitReached:t.componentSearchIterationLimitReached}}var de=2.5,fe=.32,$=1.04,pe=2,me=[`intercept`,`x1`,`x2`];function he(e){let{centers:t,count:n}=e.tracts,r=1/0,i=-1/0;for(let e=0;e<n;e++)r=Math.min(r,t[e*2+1]),i=Math.max(i,t[e*2+1]);return n=>[.5+pe*((t[n*2+1]-r)/(i-r||1)),.5+de*(t[n*2]/e.mapWidth),H[2]]}var ge=P({parameters:R,caption:`Three coefficient surfaces fitted at their own scales: an intercept rising south to north, a slope drifting west to east and a constant slope.`,createInputs:({random:e,world:t})=>B(e,t,8,fe),build(e,t){let n=t.tracts.count,r=e.upload(`positions`,`float32x2`,t.tracts.centers),i=e.upload(`predictors`,`float32`,t.predictors),a=e.allocate(`response`,`float32`,n),o=e.allocate(`parameters`,`float32`,4),s=e.allocate(`coefficients`,`float32`,n*3),c=e.allocate(`codes`,`uint32`,n);return e.graph.add(new ue({positions:r.view,predictors:i.view,predictorCount:2,response:a.view,kernel:`bisquare`,maximumIterations:50,parameters:o.view,output:{coefficients:s.view,fitted:e.allocate(`fitted`,`float32`,n).view,status:{codes:c.view,iterationCount:e.allocate(`iteration-count`,`uint32`,1).view}}})),{outputs:{codes:c,meshes:[z(e,t),z(e,t,t.mapWidth*$),z(e,t,t.mapWidth*$*2)],intercept:W(e,`intercept`,s,3,0,n),first:W(e,`first-slope`,s,3,1,n),second:W(e,`second-slope`,s,3,2,n)},update:e=>{a.write(U(t,{noise:.3,coefficients:he(t)})),o.write(Float32Array.of(.01,e.interceptBandwidth,e.firstBandwidth,e.secondBandwidth))}}},interaction:F(R,{kind:`pad`,hint:`intercept and x1 bandwidth`,x:`firstBandwidth`,y:`interceptBandwidth`,decorate:(e,t,{outputs:n,peek:r})=>{let i=r(n.codes),a=0,o=0,s=0,c=0;for(let e=0;i&&e<i.length;e++)switch(le(i[e])){case`converged`:a++;break;case`excluded`:o++;break;case`iteration-limit`:s++;break;default:c++}return{glyphs:[{shape:`crosshair`,at:[t.x,t.y]}],readout:`x1 bandwidth ${e.firstBandwidth} · intercept ${e.interceptBandwidth} · ${a} converged · ${o} excluded · ${s} iteration-limit estimate${s===1?``:`s`} displayed · ${c} failed`}}}),draw:({inputs:e,outputs:t,theme:n})=>{let r=-1/0;for(let t=0;t<e.tracts.count;t++)r=Math.max(r,e.tracts.centers[t*2+1]);let i=L({title:`coefficient`,corner:`bottom-left`,entries:[{label:`−3`,swatch:[...n.diverging[0]]},{label:`0`,swatch:[...n.diverging[2]]},{label:`+3`,swatch:[...n.diverging[5]]}]}),a=I(i,{labels:me.map((t,n)=>({text:t,at:[e.mapWidth*($*n+.5),r],offset:[0,22],role:`label`,priority:3}))});return[...V(e,t.meshes[0],t.intercept,{ramp:`diverging`,range:[-3,3]}),...V(e,t.meshes[1],t.first,{ramp:`diverging`,range:[-3,3]}),...V(e,t.meshes[2],t.second,{ramp:`diverging`,range:[-3,3]}),{layer:`text`,...a}]}});export{ge as default};