import{E as e,f as t,v as n,y as r}from"./gpu-contributor-utils-BIwLQON5.js";import{a as i}from"./graph-chunk-dispatch-zBt8k2-o.js";import{c as a}from"./spatial-weights-CqWysstN.js";import{r as o}from"./kernel-C3tfQOn4.js";import{o as s}from"./wgsl-kernel-nodes-DagJklH2.js";import{c,r as l,u}from"./wgsl-common-DfyXiYB0.js";import{n as d,t as f}from"./gpu-ordinary-least-squares-B4HAVt2a.js";import{t as p}from"./gpu-spatial-error-gm-C0rDnQot.js";import{i as m,n as h,t as g}from"./regression-row-tiles-7Xsw3Utq.js";import{n as _,r as v}from"./spatial-regression-solve-Vf-sQz9O.js";import{t as y}from"./pairwise-sum-wgsl-iErcC5Or.js";import{t as b}from"./rounded-difference-wgsl-DgXXQtNj.js";import{i as x}from"./spatial-likelihood-allocations-BGEHbqkB.js";import{t as S}from"./regression-output-dispatches-KKVTWkHP.js";import{t as C}from"./define-run-B-brkCe4.js";import{c as w,n as T,o as E}from"./text-DYSgVV-n.js";import{jn as D}from"./entry-preview-D70TdSbK.js";import{_ as O,a as k,c as A,d as j,f as M,l as N,o as P}from"./regression-CXg5hXjb.js";import{c as F,r as I}from"./probes-CVwGX4BJ.js";import{t as L}from"./kernel-DfdJX5rr.js";function R(e,t,n){let r=e*(t+1),{tileRowCount:i,tileCount:a}=h(n),o=1+e*t,s=o+e,c=s+e*t,l=c+e*e,u=l+1,d=u+r,f=d+1,p=r*(r+1);return{tileRows:i,tileCount:a,meansStart:1,responsesStart:o,initialStart:s,precisionStart:c,errorScaleStart:l,centeredSolutionStart:u,runtimeZeroSlot:d,partialStart:f,partialStride:p,length:f+a*p}}function z(e,t,n){return R(e,t,n).length}function B(e,t){let{id:n,props:r}=t,i=r.equationCount,a=r.predictorCount,d=r.rowCount,f=a+1,p=i*f,{tileRows:m,tileCount:h,meansStart:g,responsesStart:x,initialStart:S,precisionStart:C,errorScaleStart:w,centeredSolutionStart:T,runtimeZeroSlot:E,partialStart:D,partialStride:O,length:k}=R(i,a,d),A={name:`scratch`,view:o(e,`${n}-statistics`,`float32`,k),type:`f32`,access:`read_write`},j={name:`status`,view:r.output.status,type:`u32`,access:`read_write`},M={name:`coefficients`,view:r.output.coefficients,type:`f32`,access:`read_write`},N={name:`covariance`,view:r.output.covariance,type:`f32`,access:`read_write`},P={name:`residualCovariance`,view:r.output.residualCovariance,type:`f32`,access:`read_write`},F=[{name:`predictors`,view:r.predictors,type:`f32`,access:`read`},{name:`responses`,view:r.responses,type:`f32`,access:`read`},...r.mask?[{name:`mask`,view:r.mask,type:`u32`,access:`read`}]:[],...r.initialResiduals?[{name:`initialResiduals`,view:r.initialResiduals,type:`f32`,access:`read`}]:[]],I=`${l}
${c}
${b}
const MEANS_START: u32 = ${g}u;
const RESPONSES_START: u32 = ${x}u;
const INITIAL_START: u32 = ${S}u;
const PRECISION_START: u32 = ${C}u;
const ERROR_SCALE_START: u32 = ${w}u;
const CENTERED_SOLUTION_START: u32 = ${T}u;
const RUNTIME_ZERO_SLOT: u32 = ${E}u;
const PARTIAL_START: u32 = ${D}u;
const PARTIAL_STRIDE: u32 = ${O}u;
// Fixed pairwise sum over the tiles (a float compensation is folded away by the shader compiler).
fn sumPartials(slot: u32) -> f32 {
  ${y({count:`${h}u`,name:`total`,term:`scratch[scratchOffset + PARTIAL_START + totalItem * PARTIAL_STRIDE + slot]`})}
  return total;
}`,L=`${I}
fn included(row: u32) -> bool {
  ${r.mask?`if (mask[maskOffset + row] == 0u) { return false; }`:``}
  for (var equation = 0u; equation < ${i}u; equation++) {
    ${r.initialResiduals?`if (!isFiniteBits(initialResiduals[initialResidualsOffset + equation * ${d}u + row])) { return false; }`:``}
    if (!isFiniteBits(responses[responsesOffset + equation * ${d}u + row])) { return false; }
    for (var column = 0u; column < ${a}u; column++) {
      if (!isFiniteBits(predictors[predictorsOffset + (equation * ${d}u + row) * ${a}u + column])) { return false; }
    }
  }
  return true;
}
fn predictor(row: u32, equation: u32, column: u32) -> f32 {
  if (column == 0u) { return 1.0; }
  return getRoundedDifference(
    predictors[predictorsOffset + (equation * ${d}u + row) * ${a}u + column - 1u],
    scratch[scratchOffset + MEANS_START + equation * ${a}u + column - 1u],
    bitcast<u32>(scratch[scratchOffset + RUNTIME_ZERO_SLOT]));
}
fn residual(row: u32, equation: u32) -> f32 {
  ${r.initialResiduals?`return initialResiduals[initialResidualsOffset + equation * ${d}u + row];`:`
  var value = getRoundedDifference(responses[responsesOffset + equation * ${d}u + row],
    scratch[scratchOffset + RESPONSES_START + equation],
    bitcast<u32>(scratch[scratchOffset + RUNTIME_ZERO_SLOT]));
  for (var column = 0u; column < ${a}u; column++) {
    value -= predictor(row, equation, column + 1u) * scratch[scratchOffset + INITIAL_START + equation * ${a}u + column];
  }
  return value;`}
}`,z=`${I}\n${u}\n${[...new Set([a,i,p])].map(e=>`${_(e)}\n${v(e)}`).join(`
`)}`,B=(t,r,i,a,o)=>s(e,{id:`${n}-${t}`,operation:`GPUSeeminglyUnrelatedRegression`,variant:t,invocationCount:r,workgroupSize:64,bindings:i,declarations:a,body:o});return[B(`tile-means`,h,[...F,A],L,`
var count = 0u;
var means: array<f32, ${p}>;
for (var row = index * ${m}u; row < min((index + 1u) * ${m}u, ${d}u); row++) {
  if (!included(row)) { continue; }
  count++;
  for (var equation = 0u; equation < ${i}u; equation++) {
    let responseSlot = ${i*a}u + equation;
    means[responseSlot] += (responses[responsesOffset + equation * ${d}u + row] - means[responseSlot]) / f32(count);
    for (var column = 0u; column < ${a}u; column++) {
      let slot = equation * ${a}u + column;
      means[slot] += (predictors[predictorsOffset + (equation * ${d}u + row) * ${a}u + column] - means[slot]) / f32(count);
    }
  }
}
let start = scratchOffset + PARTIAL_START + index * PARTIAL_STRIDE;
scratch[start] = f32(count);
for (var slot = 0u; slot < ${p}u; slot++) { scratch[start + 1u + slot] = means[slot]; }
`),B(`merge-means`,1,[A,j,M,N,P],I,`
status[statusOffset] = 1u;
scratch[scratchOffset + RUNTIME_ZERO_SLOT] = 0.0;
for (var slot = 0u; slot < ${p}u; slot++) { coefficients[coefficientsOffset + slot] = getNaN(); }
for (var slot = 0u; slot < ${p*p}u; slot++) { covariance[covarianceOffset + slot] = getNaN(); }
for (var slot = 0u; slot < ${i*i}u; slot++) { residualCovariance[residualCovarianceOffset + slot] = getNaN(); }
var count = 0u;
var means: array<f32, ${p}>;
for (var tile = 0u; tile < ${h}u; tile++) {
  let start = scratchOffset + PARTIAL_START + tile * PARTIAL_STRIDE;
  let tileCount = u32(scratch[start]);
  if (tileCount == 0u) { continue; }
  count += tileCount;
  for (var slot = 0u; slot < ${p}u; slot++) {
    means[slot] += (scratch[start + 1u + slot] - means[slot]) * (f32(tileCount) / f32(count));
  }
}
scratch[scratchOffset] = f32(count);
for (var slot = 0u; slot < ${p}u; slot++) { scratch[scratchOffset + MEANS_START + slot] = means[slot]; }
if (count > ${Math.max(f,i)}u) { status[statusOffset] = 0u; }
`),B(`tile-ols`,h*i,[...F,A,j],L,`
if (status[statusOffset] != 0u) { return; }
let tile = index / ${i}u;
let equation = index % ${i}u;
var normal: array<f32, ${a*a}>;
var right: array<f32, ${a}>;
for (var row = tile * ${m}u; row < min((tile + 1u) * ${m}u, ${d}u); row++) {
  if (!included(row)) { continue; }
  let response = responses[responsesOffset + equation * ${d}u + row] - scratch[scratchOffset + RESPONSES_START + equation];
  for (var first = 0u; first < ${a}u; first++) {
    let value = predictor(row, equation, first + 1u);
    right[first] += value * response;
    for (var second = 0u; second < ${a}u; second++) { normal[first * ${a}u + second] += value * predictor(row, equation, second + 1u); }
  }
}
let start = scratchOffset + PARTIAL_START + tile * PARTIAL_STRIDE + equation * ${a*(a+1)}u;
for (var slot = 0u; slot < ${a*a}u; slot++) { scratch[start + slot] = normal[slot]; }
for (var slot = 0u; slot < ${a}u; slot++) { scratch[start + ${a*a}u + slot] = right[slot]; }
`),B(`solve-ols`,1,[A,j],z,`
if (status[statusOffset] != 0u) { return; }
for (var equation = 0u; equation < ${i}u; equation++) {
var normal: array<f32, ${a*a}>;
var right: array<f32, ${a}>;
var scales: array<f32, ${a}>;
let start = equation * ${a*(a+1)}u;
for (var slot = 0u; slot < ${a*a}u; slot++) { normal[slot] = sumPartials(start + slot); }
for (var slot = 0u; slot < ${a}u; slot++) { right[slot] = sumPartials(start + ${a*a}u + slot); }
for(var column=0u;column<${a}u;column++){let diagonal=normal[column*${a+1}u];if(!(diagonal>0.0)){status[statusOffset]=2u;return;}scales[column]=inverseSqrt(diagonal);right[column]*=scales[column];}
for(var first=0u;first<${a}u;first++){for(var second=0u;second<${a}u;second++){normal[first*${a}u+second]*=scales[first]*scales[second];}}
if(!choleskyFactorLimited_${a}(&normal)){status[statusOffset]=2u;return;}choleskySolve_${a}(&normal,&right);for(var column=0u;column<${a}u;column++){scratch[scratchOffset+INITIAL_START+equation*${a}u+column]=right[column]*scales[column];}}

`),B(`tile-error-covariance`,h*i*i,[...F,A,j],L,`
if (status[statusOffset] != 0u) { return; }
let tile = index / ${i*i}u;
let slot = index % ${i*i}u;
var value = 0.0;
// Normalize during accumulation: the unnormalized sum can overflow for a finite covariance.
for (var row = tile * ${m}u; row < min((tile + 1u) * ${m}u, ${d}u); row++) {
  if (included(row)) { value += residual(row, slot / ${i}u) * residual(row, slot % ${i}u) / scratch[scratchOffset]; }
}
scratch[scratchOffset + PARTIAL_START + tile * PARTIAL_STRIDE + slot] = value;
`),B(`solve-error-covariance`,1,[A,j,P],z,`
if (status[statusOffset] != 0u) { return; }
var sigma: array<f32, ${i*i}>;
for (var slot = 0u; slot < ${i*i}u; slot++) { sigma[slot] = sumPartials(slot); }
for(var slot=0u;slot<${i*i}u;slot++){residualCovariance[residualCovarianceOffset+slot]=sigma[slot];}
// A common covariance scale cancels from coefficients and is restored to their covariance.
// Keep the precision and GLS products away from subnormal values for large error magnitudes.
var errorScale = 0.0;
for (var equation = 0u; equation < ${i}u; equation++) { errorScale = max(errorScale, sigma[equation * ${i+1}u]); }
if (!(errorScale > 0.0) || !isFiniteBits(errorScale)) { status[statusOffset] = 3u; return; }
scratch[scratchOffset + ERROR_SCALE_START] = errorScale;
for (var slot = 0u; slot < ${i*i}u; slot++) { sigma[slot] /= errorScale; }
if(!choleskyFactorLimited_${i}(&sigma)){status[statusOffset]=3u;return;}

for(var column=0u;column<${i}u;column++){var unit:array<f32,${i}>;unit[column]=1.0;choleskySolve_${i}(&sigma,&unit);for(var row=0u;row<${i}u;row++){scratch[scratchOffset+PRECISION_START+row*${i}u+column]=unit[row];}}

`),B(`tile-gls`,h*p,[...F,A,j],L,`
if (status[statusOffset] != 0u) { return; }
let tile = index / ${p}u;
let first = index % ${p}u;
let equation = first / ${f}u;
var normal: array<f32, ${p}>;
var right = 0.0;
for (var row = tile * ${m}u; row < min((tile + 1u) * ${m}u, ${d}u); row++) {
  if (!included(row)) { continue; }
  let value = predictor(row, equation, first % ${f}u);
  for (var other = 0u; other < ${i}u; other++) {
    right += scratch[scratchOffset + PRECISION_START + equation * ${i}u + other] * value
      * (responses[responsesOffset + other * ${d}u + row] - scratch[scratchOffset + RESPONSES_START + other]);
  }
  for (var second = 0u; second < ${p}u; second++) {
    normal[second] += scratch[scratchOffset + PRECISION_START + equation * ${i}u + second / ${f}u]
      * value * predictor(row, second / ${f}u, second % ${f}u);
  }
}
let start = scratchOffset + PARTIAL_START + tile * PARTIAL_STRIDE + first * ${p+1}u;
for (var second = 0u; second < ${p}u; second++) { scratch[start + second] = normal[second]; }
scratch[start + ${p}u] = right;
`),B(`solve-gls`,1,[A,j,M,N],z,`
if (status[statusOffset] != 0u) { return; }
var normal: array<f32, ${p*p}>;
var right: array<f32, ${p}>;
for (var first = 0u; first < ${p}u; first++) {
  right[first] = sumPartials(first * ${p+1}u + ${p}u);
  for (var second = 0u; second < ${p}u; second++) { normal[first * ${p}u + second] = sumPartials(first * ${p+1}u + second); }
}
var scales:array<f32,${p}>;for(var column=0u;column<${p}u;column++){let diagonal=normal[column*${p+1}u];if(!(diagonal>0.0)){status[statusOffset]=2u;return;}scales[column]=inverseSqrt(diagonal);right[column]*=scales[column];}
for(var first=0u;first<${p}u;first++){for(var second=0u;second<${p}u;second++){normal[first*${p}u+second]*=scales[first]*scales[second];}}
if(!choleskyFactorLimited_${p}(&normal)){status[statusOffset]=2u;return;}choleskySolve_${p}(&normal,&right);
for (var slot = 0u; slot < ${p}u; slot++) {
  scratch[scratchOffset + CENTERED_SOLUTION_START + slot] = right[slot] * scales[slot];
}
for(var equation=0u;equation<${i}u;equation++){var intercept=scratch[scratchOffset+RESPONSES_START+equation]+right[equation*${f}u]*scales[equation*${f}u];for(var column=0u;column<${a}u;column++){let slot=equation*${f}u+column+1u;let coefficient=right[slot]*scales[slot];coefficients[coefficientsOffset+slot]=coefficient;intercept-=scratch[scratchOffset+MEANS_START+equation*${a}u+column]*coefficient;}coefficients[coefficientsOffset+equation*${f}u]=intercept;}
for(var targetColumn=0u;targetColumn<${p}u;targetColumn++){var transform:array<f32,${p}>;transform[targetColumn]=1.0;if(targetColumn%${f}u==0u){for(var column=0u;column<${a}u;column++){transform[targetColumn+column+1u]=-scratch[scratchOffset+MEANS_START+(targetColumn/${f}u)*${a}u+column];}}for(var column=0u;column<${p}u;column++){transform[column]*=scales[column];}choleskySolve_${p}(&normal,&transform);for(var column=0u;column<${p}u;column++){transform[column]*=scales[column];}
for(var destination=0u;destination<${p}u;destination++){var value=transform[destination];if(destination%${f}u==0u){for(var column=0u;column<${a}u;column++){value-=scratch[scratchOffset+MEANS_START+(destination/${f}u)*${a}u+column]*transform[destination+column+1u];}}covariance[covarianceOffset+destination*${p}u+targetColumn]=value*scratch[scratchOffset+ERROR_SCALE_START];}}

`),B(`residuals`,d*i,[...F,A,j,{name:`residuals`,view:r.output.residuals,type:`f32`,access:`read_write`}],L,`
let equation = index / ${d}u;
let row = index % ${d}u;
var value = getNaN();
if (status[statusOffset] == 0u && included(row)) {
  // The raw intercept can lose the small centered intercept when predictor means are large.
  // Predict from the centered fit, rounding the first difference before subtracting small terms.
  value = getRoundedDifference(responses[responsesOffset + index],
    scratch[scratchOffset + RESPONSES_START + equation],
    bitcast<u32>(scratch[scratchOffset + RUNTIME_ZERO_SLOT]));
  value -= scratch[scratchOffset + CENTERED_SOLUTION_START + equation * ${f}u];
  for (var column = 0u; column < ${a}u; column++) {
    value -= predictor(row, equation, column + 1u)
      * scratch[scratchOffset + CENTERED_SOLUTION_START + equation * ${f}u + column + 1u];
  }
}
residuals[residualsOffset + index] = value;
`)]}function V(e,t,n){let{tileCount:r}=h(e),i=n+1,a=t*i,o=i*i+i+1;return{...d(e,n),"SUR filtered predictors":t*e*n,"SUR filtered responses/residuals":t*e,"SUR filtered coefficients":a,"SUR filtered covariance":a*a,"SUR unfiltered equation":e*(n+2),"SUR statistics scratch":z(t,n,e),"SUR weight partials":r,"SUR GM table":(n+2)*4,"SUR GM summary":13,"SUR OLS summary":22,"SUR GM workspace":n+4+i+16,"SUR GM mean partials":r*i,"SUR GM moment partials":r*7,"SUR GM moments":7,"SUR GM filtered partials":r*o,"SUR GM filtered moments":o,"SUR GM residual partials":r*5,"SUR GM residual totals":5}}function H(e,t){let{id:n,props:r}=t,i=r.spatial,a=r.rowCount,u=r.equationCount,d=r.predictorCount,m=d+1,h=u*m,g=(t,r,i)=>o(e,`${n}-${t}`,r,i),_=(t,n,r)=>e.createDataView(t.buffer,{format:t.format,byteOffset:t.byteOffset+n*4,length:r}),v=g(`filtered-predictors`,`float32`,u*a*d),y=g(`filtered-responses`,`float32`,u*a),b=g(`filtered-residuals`,`float32`,u*a),x=g(`filtered-coefficients`,`float32`,h),S=g(`filtered-covariance`,`float32`,h*h),C=g(`invalid-weights`,`float32`,1),w=U(e,{id:n,weights:i.weights,rowCount:a,output:C});for(let t=0;t<u;t++){let o=_(r.predictors,t*a*d,a*d),u=_(r.responses,t*a,a),h=g(`initial-residuals-${t}`,`float32`,a);w.push(...new f({id:`${n}-ols-${t}`,predictors:o,response:u,predictorCount:d,output:{coefficients:g(`ols-coefficients-${t}`,`float32`,m),standardErrors:g(`ols-errors-${t}`,`float32`,m),tStatistics:g(`ols-tests-${t}`,`float32`,m),summary:g(`ols-summary-${t}`,`float32`,22),status:{code:g(`ols-status-${t}`,`uint32`,1)},residuals:h}}).getCommandNodes(e));let x=g(`gm-table-${t}`,`float32`,(d+2)*4),S=_(i.output.status,t,1);w.push(...new p({id:`${n}-gm-${t}`,predictors:o,response:u,predictorCount:d,weights:i.weights,estimator:`kelejian-prucha`,output:{table:x,summary:g(`gm-summary-${t}`,`float32`,13),status:{code:S}}}).getCommandNodes(e)),w.push(s(e,{id:`${n}-parameter-${t}`,operation:`GPUSeeminglyUnrelatedRegression`,variant:`spatial-error-parameter`,invocationCount:1,bindings:[{name:`invalidWeights`,view:C,type:`f32`,access:`read`},{name:`table`,view:x,type:`f32`,access:`read`},{name:`status`,view:S,type:`u32`,access:`read_write`},{name:`parameters`,view:_(i.output.parameters,t,1),type:`f32`,access:`read_write`}],declarations:`${l}\n${c}`,body:`parameters[parametersOffset] = getNaN();
if (invalidWeights[invalidWeightsOffset] != 0.0) { status[statusOffset] = 4u; return; }
if (status[statusOffset] == 0u) { parameters[parametersOffset] = table[tableOffset + ${m*4}u]; }`}));let T=g(`unfiltered-${t}`,`float32`,a*(d+2));w.push(s(e,{id:`${n}-pack-${t}`,operation:`GPUSeeminglyUnrelatedRegression`,variant:`pack-spatial-system`,invocationCount:a,bindings:[{name:`predictors`,view:o,type:`f32`,access:`read`},{name:`response`,view:u,type:`f32`,access:`read`},{name:`residuals`,view:h,type:`f32`,access:`read`},{name:`packed`,view:T,type:`f32`,access:`read_write`}],body:`packed[packedOffset + index * ${d+2}u] = response[responseOffset + index]; packed[packedOffset + index * ${d+2}u + 1u] = residuals[residualsOffset + index]; for (var column = 0u; column < ${d}u; column++) { packed[packedOffset + index * ${d+2}u + column + 2u] = predictors[predictorsOffset + index * ${d}u + column]; }`})),w.push(s(e,{id:`${n}-filter-${t}`,operation:`GPUSeeminglyUnrelatedRegression`,variant:`spatial-error-filter`,invocationCount:a,bindings:[{name:`offsets`,view:i.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:i.weights.neighbors,type:`u32`,access:`read`},{name:`weights`,view:i.weights.weights,type:`f32`,access:`read`},{name:`packed`,view:T,type:`f32`,access:`read`},{name:`parameter`,view:_(i.output.parameters,t,1),type:`f32`,access:`read`},{name:`predictors`,view:_(v,t*a*d,a*d),type:`f32`,access:`read_write`},{name:`response`,view:_(y,t*a,a),type:`f32`,access:`read_write`},{name:`residuals`,view:_(b,t*a,a),type:`f32`,access:`read_write`}],body:`let parameterValue = parameter[parameterOffset]; for (var column = 0u; column < ${d+2}u; column++) { var value = packed[packedOffset + index * ${d+2}u + column]; for (var slot = offsets[offsetsOffset + index]; slot < min(offsets[offsetsOffset + index + 1u], ${i.weights.neighbors.length}u); slot++) { let neighbor = neighbors[neighborsOffset + slot]; if (neighbor < ${a}u) { value -= parameterValue * weights[weightsOffset + slot] * packed[packedOffset + neighbor * ${d+2}u + column]; } } if (column == 0u) { response[responseOffset + index] = value; } else if (column == 1u) { residuals[residualsOffset + index] = value; } else { predictors[predictorsOffset + index * ${d}u + column - 2u] = value; } }`}))}return w.push(...new Y({id:`${n}-filtered-system`,predictors:v,responses:y,initialResiduals:b,equationCount:u,predictorCount:d,rowCount:a,output:{coefficients:x,covariance:S,residualCovariance:r.output.residualCovariance,residuals:g(`filtered-final-residuals`,`float32`,a*u),status:r.output.status}}).getCommandNodes(e)),w.push(s(e,{id:`${n}-unfilter-coefficients`,operation:`GPUSeeminglyUnrelatedRegression`,variant:`restore-spatial-intercepts`,invocationCount:h*h,bindings:[{name:`parameters`,view:i.output.parameters,type:`f32`,access:`read`},{name:`sourceCoefficients`,view:x,type:`f32`,access:`read`},{name:`sourceCovariance`,view:S,type:`f32`,access:`read`},{name:`coefficients`,view:r.output.coefficients,type:`f32`,access:`read_write`},{name:`covariance`,view:r.output.covariance,type:`f32`,access:`read_write`}],body:`let first = index / ${h}u; let second = index % ${h}u; let firstScale = select(1.0, 1.0 / (1.0 - parameters[parametersOffset + first / ${m}u]), first % ${m}u == 0u); let secondScale = select(1.0, 1.0 / (1.0 - parameters[parametersOffset + second / ${m}u]), second % ${m}u == 0u); covariance[covarianceOffset + index] = sourceCovariance[sourceCovarianceOffset + index] * firstScale * secondScale; if (index < ${h}u) { coefficients[coefficientsOffset + index] = sourceCoefficients[sourceCoefficientsOffset + index] * secondScale; }`})),w.push(s(e,{id:`${n}-residuals`,operation:`GPUSeeminglyUnrelatedRegression`,variant:`structural-spatial-residuals`,invocationCount:u*a,bindings:[{name:`predictors`,view:r.predictors,type:`f32`,access:`read`},{name:`responses`,view:r.responses,type:`f32`,access:`read`},{name:`coefficients`,view:r.output.coefficients,type:`f32`,access:`read`},{name:`residuals`,view:r.output.residuals,type:`f32`,access:`read_write`}],body:`let equation = index / ${a}u; var residual = responses[responsesOffset + index] - coefficients[coefficientsOffset + equation * ${m}u]; for (var column = 0u; column < ${d}u; column++) { residual -= predictors[predictorsOffset + index * ${d}u + column] * coefficients[coefficientsOffset + equation * ${m}u + column + 1u]; } residuals[residualsOffset + index] = residual;`})),w}function U(e,t){i(e,t.id,{offsets:t.weights.offsets,neighbors:t.weights.neighbors,weights:t.weights.weights,overflow:t.weights.status?.overflow,output:t.output});let r=h(t.rowCount);return n(e.device,t.id,`weightValidationPartial`,r.tileCount*4,`reduce rowCount`),m(e,{id:`${t.id}-validate-weights`,operation:`GPUSeeminglyUnrelatedRegression`,variant:`validate-spatial-weights`,rowCount:t.rowCount,layout:r,sumCount:0,maximumCount:1,partial:o(e,`${t.id}-weight-validation-partial`,`float32`,r.tileCount),output:t.output,bindings:[{name:`offsets`,view:t.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:t.weights.neighbors,type:`u32`,access:`read`},{name:`weights`,view:t.weights.weights,type:`f32`,access:`read`},...t.weights.status?.overflow?[{name:`overflow`,view:t.weights.status.overflow,type:`u32`,access:`read`}]:[]],declarations:`${c}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  ${t.weights.status?.overflow?`if (overflow[overflowOffset] != 0u) { (*terms)[0] = 1.0; return; }`:``}
  let begin = offsets[offsetsOffset + row];
  let end = offsets[offsetsOffset + row + 1u];
  if (end < begin || end > ${t.weights.neighbors.length}u) { (*terms)[0] = 1.0; return; }
  var total = 0.0;
  for (var slot = begin; slot < end; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    let weight = weights[weightsOffset + slot];
    if (neighbor >= ROW_COUNT || neighbor == row || !isFiniteBits(weight) || weight < 0.0 || (slot > begin && neighbors[neighborsOffset + slot - 1u] >= neighbor)) { (*terms)[0] = 1.0; return; }
    total += weight;
  }
  if (abs(total - 1.0) > 1e-5) { (*terms)[0] = 1.0; }
}`})}var W=g,G=g,K=4,q=7,J=16,Y=class{props;id;constructor(n){this.props=n,this.id=n.id??`seemingly-unrelated-regression`;let i=n.equationCount,o=n.predictorCount,s=n.rowCount,c=i*(o+1);if(n.spatial){if(n.mask||n.initialResiduals||a(this.id,n.spatial.weights)!==s||n.spatial.output.parameters.length!==i||n.spatial.output.status.length!==i)throw Error(`${this.id}: spatial SUR requires complete aligned observations and equation outputs`);e(n.spatial.output.parameters,[`float32`],this.id),e(n.spatial.output.status,[`uint32`],this.id),r(this.id,[...Object.values(n.output),...Object.values(n.spatial.output)],[n.predictors,n.responses,n.spatial.weights.offsets,n.spatial.weights.neighbors,n.spatial.weights.weights,n.spatial.weights.status?.overflow])}t(this.id,`equationCount`,i,2,K),t(this.id,`predictorCount`,o,1,q),t(this.id,`system coefficients`,c,1,J),t(this.id,`rowCount`,s,1,n.spatial?G:W);for(let[t,r,a]of[[n.predictors,`float32`,i*s*o],[n.responses,`float32`,i*s],[n.mask,`uint32`,s],[n.initialResiduals,`float32`,s*i],[n.output.coefficients,`float32`,c],[n.output.covariance,`float32`,c*c],[n.output.residualCovariance,`float32`,i*i],[n.output.residuals,`float32`,i*s],[n.output.status,`uint32`,1]])if(t&&(e(t,[r],this.id),t.length!==a))throw Error(`${this.id}: shape mismatch`);S(this.id,{...n.output,spatialParameters:n.spatial?.output.parameters,spatialStatus:n.spatial?.output.status},n.spatial?[{read:[`spatialParameters`],write:[`coefficients`,`covariance`]},{read:[`coefficients`],write:[`residuals`]},[`residualCovariance`,`status`],[`spatialParameters`,`spatialStatus`]]:[[`coefficients`,`covariance`,`residualCovariance`,`status`],{read:[`status`],write:[`residuals`]}]),r(this.id,Object.values(n.output),[n.predictors,n.responses,n.mask,n.initialResiduals])}getCommandNodes(e){let{id:t,props:r}=this;return i(e,t,{predictors:r.predictors,responses:r.responses,mask:r.mask,initialResiduals:r.initialResiduals,...r.output,spatialParameters:r.spatial?.output.parameters,spatialStatus:r.spatial?.output.status,offsets:r.spatial?.weights.offsets,neighbors:r.spatial?.weights.neighbors,weights:r.spatial?.weights.weights,overflow:r.spatial?.weights.status?.overflow}),n(e.device,t,`statistics scratch`,z(r.equationCount,r.predictorCount,r.rowCount)*4,`reduce rowCount or the system coefficient count`),r.spatial?(x(e.device,t,V(r.rowCount,r.equationCount,r.predictorCount)),H(e,this)):B(e,this)}},X=[k+.08,.06,.97,.36],Z=[1,2,-1.5,.5,-1,2],Q=C({parameters:D,caption:`Two-equation SUR: coefficients (bars) against the truth (dots), residual covariance below.`,createInputs:({random:e,world:t})=>N(e,t),build(e,t){let n=t.tracts.count,r=e.upload(`predictors`,`float32`,Float32Array.from([...t.predictors,...t.predictors])),i=e.allocate(`responses`,`float32`,n*2),a=e.allocate(`coefficients`,`float32`,6),o=e.allocate(`residuals`,`float32`,n*2),s=e.allocate(`residual-covariance`,`float32`,4);e.graph.add(new Y({equationCount:2,predictorCount:2,rowCount:n,predictors:r.view,responses:i.view,output:{coefficients:a.view,covariance:e.allocate(`covariance`,`float32`,36).view,residualCovariance:s.view,residuals:o.view,status:e.allocate(`status`,`uint32`,1).view}}));let c=A(e,a,1,Z,[`1:int`,`1:x1`,`1:x2`,`2:int`,`2:x1`,`2:x2`]),l=e.allocate(`first-residuals`,`float32`,n),u=e.allocate(`second-residuals`,`float32`,n);return L(e,{id:`split-residuals`,invocationCount:n,bindings:[{name:`residuals`,buffer:o,access:`read`},{name:`first`,buffer:l,access:`read_write`},{name:`second`,buffer:u,access:`read_write`}],declarations:`const ROW_COUNT: u32 = ${n}u;`,body:`first[firstOffset + index] = residuals[residualsOffset + index];
  second[secondOffset + index] = residuals[residualsOffset + ROW_COUNT + index];`}),{outputs:{mesh:O(e,t),panel:c,residualCovariance:s,residualBuffer:o,firstResiduals:l,secondResiduals:u},update:e=>{let r=new Float32Array(n),a=new Float32Array(n),o=Math.sqrt(1-e.correlation**2);for(let i=0;i<n;i++){let s=t.normals[(i*7+3)%n],c=t.normals[i],l=Z[0]+Z[1]*t.predictors[i*2]+Z[2]*t.predictors[i*2+1],u=Z[3]+Z[4]*t.predictors[i*2]+Z[5]*t.predictors[i*2+1];r[i]=l+c*e.noise,a[i]=u+(e.correlation*c+o*s)*e.noise}i.write(Float32Array.from([...r,...a]))}}},interaction:{kind:`probe`,hint:`hover a tract`,rest:[.3,.5],resolve(e,{inputs:t,outputs:n,world:r,peek:i}){let a=I(t.tracts,e.x,e.y),o=i(n.firstResiduals),s=i(n.secondResiduals),c=i(n.residualCovariance);if(a<0||!o||!s)return{};let[l,u]=P,d=e=>Math.min(1,Math.max(0,(e-l)/(u-l))),[f,p,m,h]=X;return{focus:a,glyphs:[F(t.tracts,a),{shape:`ring`,at:[(f+d(o[a])*(m-f))*r.width,(p+d(s[a])*(h-p))*r.height],radius:.012}],readout:`tract ${a} · residuals ${o[a].toFixed(2)}, ${s[a].toFixed(2)} · sigma12 ${c?c[1].toFixed(2):`-`}`}}},draw:({inputs:e,outputs:t,world:n,theme:r})=>{let i=T({rectangle:E(n,X),y:{range:P,ticks:[-1,0,1],title:`residual of eq 2 (x: eq 1)`}}),a=w({labels:[{text:`residuals of equation 1`,at:[e.mapWidth/2,n.height],offset:[0,-12],role:`label`,priority:4}]});return[{layer:`text`,...i.grid},...M(e,t.mesh,t.firstResiduals),...j(t.panel,{rectangle:[k+.08,.42,.97,.9],world:n,ink:r.ink}),{layer:`chart`,kind:`scatter`,values:t.secondResiduals,xValues:t.firstResiduals,xRange:P,range:P,radius:2.6,color:[.55,.65,.95,.9],rectangle:X},{layer:`text`,...i},{layer:`text`,...a}]}});export{Q as default};