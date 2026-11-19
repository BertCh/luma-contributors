import{t as e}from"./regression-output-dispatches-KKVTWkHP.js";import{i as t}from"./spatial-likelihood-allocations-BGEHbqkB.js";import{a as n}from"./graph-chunk-dispatch-zBt8k2-o.js";import{o as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{c as i,r as a,u as o}from"./wgsl-common-DfyXiYB0.js";import{D as s,E as c,f as l,u,x as d,y as f}from"./gpu-contributor-utils-BIwLQON5.js";import{n as p,r as m}from"./spatial-regression-solve-Vf-sQz9O.js";import{n as h,t as g}from"./regime-moments-CWEPBRvM.js";import{t as _}from"./regression-row-tiles-7Xsw3Utq.js";import{t as v}from"./rounded-difference-wgsl-DgXXQtNj.js";import{n as y,t as b}from"./centered-regression-prediction-pXjnms-4.js";import{t as x}from"./define-run-B-brkCe4.js";import{Sn as S}from"./entry-preview-D70TdSbK.js";import{_ as C,c as w,d as T,l as E,m as D,s as O,u as k}from"./regression-CXg5hXjb.js";import{c as A,r as j}from"./probes-CVwGX4BJ.js";var M=11,N=4096,P=class{props;id;constructor(t){this.props=t,this.id=t.id??`spatial-regime-regression`;let n=t.response.length;l(this.id,`predictorCount`,t.predictorCount,1,M),l(this.id,`regimeCount`,t.regimeCount,1,N),l(this.id,`response rows`,n,1,_);for(let[e,r,i,a]of[[`predictors`,t.predictors,`float32`,n*t.predictorCount],[`response`,t.response,`float32`,n],[`regimes`,t.regimes,`uint32`,n],[`mask`,t.mask,`uint32`,n],[`output.coefficients`,t.output.coefficients,`float32`,t.regimeCount*(t.predictorCount+1)],[`output.fitted`,t.output.fitted,`float32`,n],[`output.status`,t.output.status,`uint32`,t.regimeCount],[`output.counts`,t.output.counts,`uint32`,t.regimeCount]])r&&(c(r,[i],this.id),s(this.id,e,r,a));e(this.id,t.output,[[`coefficients`,`status`,`counts`],{read:[`coefficients`,`status`],write:[`fitted`]}]),f(this.id,Object.values(t.output),[t.predictors,t.response,t.regimes,t.mask])}getCommandNodes(e){let{props:s,id:c}=this,l=s.response.length,f=s.predictorCount;if(n(e,c,{predictors:s.predictors,response:s.response,regimes:s.regimes,mask:s.mask,...s.output}),t(e.device,c,{"regime row sorting and radix/scan scratch":l,"regime offsets":s.regimeCount+1,"regime moments":s.regimeCount*h(f)}),l>u(e.device,4*f))throw Error(`${c}: invalid regime regression shape`);let _=[{name:`predictors`,view:s.predictors,type:`f32`,access:`read`},{name:`response`,view:s.response,type:`f32`,access:`read`},{name:`regimes`,view:s.regimes,type:`u32`,access:`read`},...s.mask?[{name:`mask`,view:s.mask,type:`u32`,access:`read`}]:[]];d(c,e,[..._.map(e=>e.view),...Object.values(s.output)]);let x=`${a}\n${i}
fn included(row: u32) -> bool {
  ${s.mask?`if (mask[maskOffset + row] == 0u) { return false; }`:``}
  if (!isFiniteBits(response[responseOffset + row]) || regimes[regimesOffset + row] >= ${s.regimeCount}u) { return false; }
  for (var column = 0u; column < ${f}u; column++) { if (!isFiniteBits(predictors[predictorsOffset + row * ${f}u + column])) { return false; } }
  return true;
}`,{nodes:S,views:C}=g(e,{id:c,operation:`GPUSpatialRegimeRegression`,predictors:s.predictors,predictorCount:f,response:s.response,regimes:s.regimes,regimeCount:s.regimeCount,mask:s.mask}),w=f+1,T=h(f);return[...S,r(e,{id:`${c}-fit`,operation:`GPUSpatialRegimeRegression`,variant:`fit`,invocationCount:s.regimeCount,bindings:[{name:`segmentOffsets`,view:C.segmentOffsets,type:`u32`,access:`read`},{name:`moments`,view:C.moments,type:`f32`,access:`read`},{name:`coefficients`,view:s.output.coefficients,type:`f32`,access:`read_write`},{name:`status`,view:s.output.status,type:`u32`,access:`read_write`},{name:`counts`,view:s.output.counts,type:`u32`,access:`read_write`}],declarations:`${a}\n${i}\n${o}\n${p(f)}\n${m(f)}
const D: u32 = ${w}u;
${v}
fn momentAt(base: u32, first: u32, second: u32, count: u32) -> f32 {
  let low = min(first, second);
  let high = max(first, second);
  let deviations = base + ${w+w*(w+1)/2}u;
  // Fix the quotient before multiplying: the product of two deviation sums can overflow
  // even when the corrected moment fits in f32. The first segment offset is a runtime zero.
  let secondCorrection = getRoundedDifference(moments[deviations + second] / max(f32(count), 1.0), 0.0, segmentOffsets[segmentOffsetsOffset]);
  return moments[base + D + low * (2u * D - low + 1u) / 2u + (high - low)]
    - moments[deviations + first] * secondCorrection;
}`,body:`let count = segmentOffsets[segmentOffsetsOffset + index + 1u] - segmentOffsets[segmentOffsetsOffset + index];
let base = momentsOffset + index * ${T}u;
counts[countsOffset + index] = count;
for (var column = 0u; column < ${f+1}u; column++) { coefficients[coefficientsOffset + index * ${f+1}u + column] = getNaN(); }
status[statusOffset + index] = 1u; if (count <= ${f}u) { return; }
var matrix: array<f32, ${f*f}>; var right: array<f32, ${f}>;
for (var column = 0u; column < ${f}u; column++) {
  right[column] = momentAt(base, column, ${f}u, count);
  for (var other = 0u; other <= column; other++) { matrix[column * ${f}u + other] = momentAt(base, column, other, count); }
}
let responseMean = moments[base + ${f}u];
status[statusOffset + index] = 2u; if (!choleskyFactorLimited_${f}(&matrix)) { return; }
choleskySolve_${f}(&matrix, &right);
let deviations = base + ${w+w*(w+1)/2}u;
var intercept = responseMean;
var correction = moments[deviations + ${f}u] / f32(count);
let runtimeZero = segmentOffsets[segmentOffsetsOffset];
for (var column = 0u; column < ${f}u; column++) {
  if (!isFiniteBits(right[column])) { return; }
  intercept = getRoundedDifference(intercept, moments[base + column] * right[column], runtimeZero);
  correction -= right[column] * (moments[deviations + column] / f32(count));
}
intercept += correction;
if (!isFiniteBits(intercept)) { return; }
coefficients[coefficientsOffset + index * ${f+1}u] = intercept;
for (var column = 0u; column < ${f}u; column++) { coefficients[coefficientsOffset + index * ${f+1}u + column + 1u] = right[column]; }
status[statusOffset + index] = 0u;`}),r(e,{id:`${c}-predict`,operation:`GPUSpatialRegimeRegression`,variant:`predict`,invocationCount:l,bindings:[..._,{name:`coefficients`,view:s.output.coefficients,type:`f32`,access:`read`},{name:`moments`,view:C.moments,type:`f32`,access:`read`},{name:`segmentOffsets`,view:C.segmentOffsets,type:`u32`,access:`read`},{name:`fitted`,view:s.output.fitted,type:`f32`,access:`read_write`}],declarations:`${x}\n${v}\n${b}`,body:`var value = getNaN();
if (included(index)) {
  let regime = regimes[regimesOffset + index];
  let base = coefficientsOffset + regime * ${f+1}u;
  if (!isFiniteBits(coefficients[base])) { fitted[fittedOffset + index] = value; return; }
  let momentBase = momentsOffset + regime * ${T}u;
  let deviations = momentBase + ${w+w*(w+1)/2}u;
  let count = f32(segmentOffsets[segmentOffsetsOffset + regime + 1u] - segmentOffsets[segmentOffsetsOffset + regime]);
  var correction = moments[deviations + ${f}u] / count;
  for (var column = 0u; column < ${f}u; column++) {
    correction -= coefficients[base + column + 1u] * moments[deviations + column] / count;
  }
  // The first sorted segment starts at zero; reading it keeps differences rounded on Metal.
  ${y({predictorCount:f,predictor:`predictors[predictorsOffset + index * `+f+`u + column]`,mean:`moments[momentBase + column]`,slope:`coefficients[base + column + 1u]`,runtimeZero:`segmentOffsets[segmentOffsetsOffset]`})}
  value = getCenteredRegressionPrediction(moments[momentBase + ${f}u], centeredContribution, correction, segmentOffsets[segmentOffsetsOffset]);
}
fitted[fittedOffset + index] = value;`})]}};function F(e,t){return[O[0]+.6*e*t,O[1]-1.1*e*t,O[2]+.9*e*t]}var I=x({parameters:S,caption:`Fitted values per regime and the per-regime coefficients against the truth.`,createInputs:({random:e,world:t})=>E(e,t),build(e,t,n){let r=t.tracts.count,i=n.regimeCount,a=e=>Math.min(i-1,Math.floor(t.tracts.centers[e*2]/t.mapWidth*i)),o=e.upload(`predictors`,`float32`,t.predictors),s=e.allocate(`response`,`float32`,r),c=e.upload(`regimes`,`uint32`,Uint32Array.from({length:r},(e,t)=>a(t))),l=e.allocate(`coefficients`,`float32`,i*3),u=e.allocate(`fitted`,`float32`,r);e.graph.add(new P({predictors:o.view,predictorCount:2,response:s.view,regimes:c.view,regimeCount:i,output:{coefficients:l.view,fitted:u.view,status:e.allocate(`status`,`uint32`,i).view,counts:e.allocate(`counts`,`uint32`,i).view}}));let d=w(e,l,1,Array.from({length:i*3},()=>0));return{outputs:{mesh:C(e,t),fitted:u,panel:d},update:e=>{s.write(k(t,{noise:e.noise,coefficients:t=>F(a(t),e.contrast)})),d.truth.write(Float32Array.from({length:i*3},(t,n)=>F(Math.floor(n/3),e.contrast)[n%3]))}}},rebuildOn:[`regimeCount`],interaction:{kind:`probe`,hint:`hover a regime`,rest:[.3,.5],resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=j(t.tracts,e.x,e.y),o=i(n.panel.estimates);if(a<0||!o)return{};let s=r.regimeCount,c=Math.min(s-1,Math.floor(t.tracts.centers[a*2]/t.mapWidth*s)),l=t.mapWidth/s,u=F(c,r.contrast)[1];return{focus:c,glyphs:[{shape:`box`,minimum:[c*l,.02],maximum:[(c+1)*l,.98]},A(t.tracts,a)],readout:`regime ${c+1} of ${s} · x1 slope ${o[c*3+1].toFixed(2)} (true ${u.toFixed(2)})`}}},draw:({inputs:e,outputs:t,theme:n})=>[...D(e,t.mesh,t.fitted,{ramp:`sequential`,range:[-2,4]}),...T(t.panel,{ink:n.ink})]});export{I as default};