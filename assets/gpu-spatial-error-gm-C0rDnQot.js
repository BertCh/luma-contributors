import{t as e}from"./regression-output-dispatches-KKVTWkHP.js";import{a as t,c as n,n as r,o as i,t as a}from"./regression-binding-plans-DG-hiroU.js";import{f as o,i as s}from"./spatial-likelihood-allocations-BGEHbqkB.js";import{a as c}from"./graph-chunk-dispatch-zBt8k2-o.js";import{r as l}from"./kernel-C3tfQOn4.js";import{c as u,o as ee}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as d,S as f,T as p,f as m,x as te,y as h}from"./gpu-contributor-utils-BIwLQON5.js";import{a as g,s as _}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";import{i as ne,n as re,r as ie,t as v}from"./chunked-weights-kernels-BYYFtXmd.js";import{r as y,t as ae}from"./gpu-ordinary-least-squares-B4HAVt2a.js";import{n as b}from"./spatial-maximum-likelihood-wgsl-Gex6F7yK.js";import{n as x,t as oe}from"./gpu-spatial-weights-traces-BQQpelz9.js";import{i as se,n as ce,t as le}from"./spatial-regression-solve-Vf-sQz9O.js";import{n as S,r as ue,u as de}from"./wgsl-common-DfyXiYB0.js";import{t as fe}from"./pairwise-sum-wgsl-iErcC5Or.js";import{t as C}from"./rounded-difference-wgsl-DgXXQtNj.js";var pe=.99,me=`GPUSpatialErrorGM`,he=64,w=4096,T=4096,E=64,D=7,O=5,k=3,A=1024,j=40,M=2,N=class{id;props;rowCount;tileRowCount;tileCount;estimator;constructor(n){n=i(n,[`id`,`weights`,`predictors`,`response`,`predictorCount`,`estimator`,`traceMode`,`traceProbeCount`,`traceParameters`,`tileRowCount`,`output`],[`table`,`summary`,`residuals`,`status`],`spatial-error-gm`);let r=n.id??`spatial-error-gm`;this.id=r,this.props=n;let{predictorCount:a,output:s}=n;m(r,`predictorCount`,a,1,15);let c=n.estimator??`homoskedastic`;if(c!==`homoskedastic`&&c!==`kelejian-prucha`)throw Error(`${r} estimator must be 'homoskedastic' or 'kelejian-prucha'`);this.estimator=c,x(r,`traceParameters`,n.traceParameters);let l=_(r,n.weights);if(c===`homoskedastic`&&(p(r,n.traceMode===void 0||n.traceMode===`exact`||n.traceMode===`stochastic`,`traceMode must be 'exact' or 'stochastic'`),m(r,`traceProbeCount`,n.traceProbeCount??64),n.traceMode===`exact`&&f(r,`exact trace rows`,l,o)),c===`homoskedastic`&&g(n.weights))throw Error(`${r} homoskedastic estimator needs whole weights`);if(d(n.predictors,[`float32`],`${r} predictors`),d(n.response,[`float32`],`${r} response`),n.response.length!==l)throw Error(`${r} response length must equal the weights row count`);if(n.predictors.length!==l*a)throw Error(`${r} predictors length must equal the weights row count * predictorCount`);if(d(s.table,[`float32`],`${r} output.table`),s.table.length<(a+2)*4)throw Error(`${r} output.table must hold (predictorCount + 2) * 4 float32 values`);if(d(s.summary,[`float32`],`${r} output.summary`),s.summary.length<13)throw Error(`${r} output.summary must hold 13 float32 values`);if(d(s.status.code,[`uint32`],`${r} output.status.code`),s.status.code.length<1)throw Error(`${r} output.status.code must hold one uint32 row`);if(s.residuals&&(d(s.residuals,[`float32`],`${r} output.residuals`),s.residuals.length<l))throw Error(`${r} output.residuals must hold one row per input row`);n.tileRowCount!==void 0&&m(r,`tileRowCount`,n.tileRowCount),this.rowCount=l,this.tileRowCount=n.tileRowCount??Math.max(E,Math.ceil(l/w)),this.tileCount=v(n.weights,this.tileRowCount).tileCount,f(r,`tileCount`,this.tileCount,T),e(this.id,{...s,status:void 0,code:s.status.code},[[`table`,`summary`,`code`]]),h(r,[s.table,s.summary,s.status.code,s.residuals],[n.predictors,n.response,...Object.values(t(n.weights))])}getCommandNodes(e){let{id:i,props:o,rowCount:d,tileRowCount:f,tileCount:p,estimator:m}=this,h=m===`homoskedastic`,{weights:g,output:_,predictorCount:y}=o;c(e,i,{predictors:o.predictors,response:o.response,...t(g),traceParameters:h?o.traceParameters:void 0,table:_.table,summary:_.summary,code:_.status.code,residuals:_.residuals}),s(e.device,i,a({rowCount:d,predictorCount:y,edgeCount:r(g),tileRowCount:f,tileCount:p,homoskedastic:h,traceMode:o.traceMode,traceProbeCount:o.traceProbeCount})),te(i,e,[o.predictors,o.response,...Object.values(t(g)),_.table,_.summary,_.status.code,_.residuals]);let b=y+1,x=b*b+b+1,S=y+1,C=y+4+b,w=C+16,T=(t,n)=>l(e,`${i}-${t}`,`float32`,n),E=T(`ols-residuals`,d),A=l(e,`${i}-ols-status`,`uint32`,1),j=new ae({id:`${i}-ols`,predictors:o.predictors,response:o.response,predictorCount:y,tileRowCount:f,output:{coefficients:T(`ols-coefficients`,b),standardErrors:T(`ols-standard-errors`,b),tStatistics:T(`ols-t-statistics`,b),summary:T(`ols-summary`,22),status:{code:A},residuals:E}}).getCommandNodes(e),N=T(`partial-means`,p*S),L=T(`workspace`,w),R=T(`partial-weights-norm`,p),z=T(`lag-residuals`,d),B=T(`partial-moments`,p*D),V=T(`moments`,D),H=T(`partial-filtered-moments`,p*x),U=T(`filtered-moments`,x),be=T(`partial-residuals`,p*O),xe=T(`residual-sums`,O),W=_.residuals??T(`residuals`,d),G=`const ROWS: u32 = ${d}u;
const K: u32 = ${y}u;
const Q: u32 = ${b}u;
const TILE_ROWS: u32 = ${f}u;
const TILE_COUNT: u32 = ${p}u;
const W_YBAR: u32 = ${y}u;
const W_LAMBDA: u32 = ${y+1}u;
const W_OBJECTIVE: u32 = ${y+2}u;
const W_INTERCEPT: u32 = ${y+3}u;
const W_BETA: u32 = ${y+4}u;
const W_BOUND: u32 = ${C}u;
const W_LAMBDA_FINAL: u32 = ${C+1}u;
const W_OBJECTIVE_FINAL: u32 = ${C+2}u;
const W_APPROXIMATION: u32 = ${C+3}u;
const W_MOMENT_MATRIX: u32 = ${C+4}u;
const W_TRACE_TERMS: u32 = ${C+8}u;
const W_POWER_SUMS: u32 = ${C+12}u;
const W_RUNTIME_ZERO: u32 = ${C+15}u;`,K=e=>fe({count:`TILE_COUNT`,term:`partial[partialOffset + sumItem * ${e}u + index]`}),q=(e,t,n)=>({name:e,view:t,type:`f32`,access:n}),J=e=>[{name:`offsets`,view:e.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:e.neighbors,type:`u32`,access:`read`},q(`weights`,e.weights,`read`)],Se=e=>[...J(e),q(`design`,o.predictors,`read`),q(`response`,o.response,`read`),q(`workspace`,L,`read`)],Y=(t,n,r,a,o,s)=>ee(e,{id:`${i}-${t}`,operation:me,variant:t,bindings:n,invocationCount:r,declarations:a,body:o,...s?{workgroupSize:s}:{}}),Ce=v(g,f),X=(e,t,n,r)=>Ce.dispatches.map(i=>Y(`${e}${i.idSuffix}`,t(i.weights),i.chunkTileCount,`${ie(i,n)}${i.tileConstantsWGSL?`\n${i.tileConstantsWGSL}`:``}`,re(i,r),he)),we=`${G}
${ge()}`,Z=`${ue}

${de}`,Te=`${G}
${Z}

${ce(b)}`,Ee=(e,t,n,r)=>X(`moments-tiles${e}`,e=>[...J(e),q(`residual`,t,`read`),q(`lag`,n,`read`),q(`partial`,r,`read_write`)],G,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${D}>;
  for (var row = firstRow; row < lastRow; row++) {
    let u = residual[residualOffset + row];
    let wu = lag[lagOffset + row];
    var wwu = 0.0;
    var squareSum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      wwu += weight * lag[lagOffset + neighbor];
      squareSum += weight * weight;
    }
    sums[0] += u * u;
    sums[1] += u * wu;
    sums[2] += wu * wu;
    sums[3] += u * wwu;
    sums[4] += wu * wwu;
    sums[5] += wwu * wwu;
    sums[6] += squareSum;
  }
  for (var slot = 0u; slot < ${D}u; slot++) {
    partial[partialOffset + index * ${D}u + slot] = sums[slot];
  }`),De=(e,t,n)=>Ce.dispatches.map(r=>Y(`residual-lag${e}${r.idSuffix}`,[...J(r.weights),q(`residual`,t,`read`),q(`lag`,n,`read_write`)],r.rowCount,`${G}${r.constantsWGSL?`\n${r.constantsWGSL}`:``}`,ne(r,`var sum = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS) {
      continue;
    }
    sum += weights[weightsOffset + slot] * residual[residualOffset + neighbor];
  }
  lag[lagOffset + index] = sum;`))),Oe=(e,t,n)=>X(`filtered-moments-tiles${e}`,e=>[...Se(e),q(`partial`,n,`read_write`)],we,`let lambda = workspace[workspaceOffset + ${t}];
  let base = partialOffset + index * ${x}u;
  for (var slot = 0u; slot < ${x}u; slot++) {
    partial[base + slot] = 0.0;
  }
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  for (var row = firstRow; row < lastRow; row++) {
    var f: array<f32, ${b}>;
    var ys = 0.0;
    getFilteredRow(row, lambda, &f, &ys);
    for (var a = 0u; a < Q; a++) {
      for (var b = 0u; b < Q; b++) {
        partial[base + a * Q + b] += f[a] * f[b];
      }
      partial[base + Q * Q + a] += f[a] * ys;
    }
    partial[base + Q * Q + Q] += ys * ys;
  }`),ke=(e,t,n)=>Y(`filtered-moments-merge${e}`,[q(`partial`,t,`read`),q(`filteredMoments`,n,`read_write`)],x,G,`${K(x)}
  filteredMoments[filteredMomentsOffset + index] = sum;`),Q=[...j,...X(`means-tiles`,()=>[q(`design`,o.predictors,`read`),q(`response`,o.response,`read`),q(`partial`,N,`read_write`)],G,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${S}>;
  for (var row = firstRow; row < lastRow; row++) {
    for (var column = 0u; column < K; column++) {
      sums[column] = sums[column] + design[designOffset + row * K + column];
    }
    sums[K] = sums[K] + response[responseOffset + row];
  }
  for (var slot = 0u; slot < ${S}u; slot++) {
    partial[partialOffset + index * ${S}u + slot] = sums[slot];
  }`),Y(`means-merge`,[q(`partial`,N,`read`),q(`workspace`,L,`read_write`)],S,G,`${K(S)}
  workspace[workspaceOffset + index] = sum / f32(ROWS);
  if (index == 0u) {
    // A zero the compiler cannot see: the runtime zero of getRoundedDifference.
    workspace[workspaceOffset + W_RUNTIME_ZERO] = 0.0;
  }`),...X(`weights-norm-tiles`,e=>[...J(e),q(`partial`,R,`read_write`)],G,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var largest = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var rowSum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      if (neighbors[neighborsOffset + slot] < ROWS) {
        rowSum += abs(weights[weightsOffset + slot]);
      }
    }
    largest = max(largest, rowSum);
  }
  partial[partialOffset + index] = largest;`),Y(`weights-norm-merge`,[q(`partial`,R,`read`),q(`workspace`,L,`read_write`)],1,`${G}
${Z}`,`var largest = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    largest = max(largest, partial[partialOffset + tile]);
  }
  let valid = largest > 0.0 && isFiniteValue(largest);
  workspace[workspaceOffset + W_BOUND] = select(getNaN(), ${u(pe)} / largest, valid);`)],Ae=T(`trace-power-traces`,M),je=T(`trace-gram-traces`,4),Me=T(`trace-infinity-norm`,1),Ne=l(e,`${i}-trace-approximation`,`uint32`,1);h&&Q.push(...new oe({id:`${i}-traces`,weights:g,maximumPower:M,mode:o.traceMode,probeCount:o.traceProbeCount,parameters:o.traceParameters,includeGramTraces:!0,output:{powerTraces:Ae,powerSums:T(`trace-power-sums`,M),gramTraces:je,infinityNorm:Me,status:{approximation:Ne}}}).getCommandNodes(e)),Q.push(...De(``,E,z),...Ee(``,E,z,B),Y(`moments-merge`,[q(`partial`,B,`read`),q(`moments`,V,`read_write`)],D,G,`${K(D)}
  moments[momentsOffset + index] = sum;`),Y(`lambda-solve`,[q(`moments`,V,`read`),q(`workspace`,L,`read_write`)],1,h?`${G}
${Z}
${F(!1)}
${P()}`:`${G}
${Z}
${_e()}
${P()}`,h?`solveFirstStepLambda();`:`solveLambda();`),...Oe(``,`W_LAMBDA`,H),ke(``,H,U),Y(`filtered-solve`,[q(`filteredMoments`,U,`read`),q(`workspace`,L,`read_write`)],1,`${Te}
${I(b,`filteredMoments`,`solveFiltered`,!1)}`,`var beta: array<f32, ${b}>;
  var inverseDiagonal: array<f32, ${b}>;
  var interceptQuadratic = 0.0;
  var conditioning: array<f32, 3>;
  let ok = solveFiltered(&beta, &inverseDiagonal, &interceptQuadratic, &conditioning);
  var intercept = workspace[workspaceOffset + W_YBAR] + beta[0];
  for (var column = 0u; column < K; column++) {
    intercept -= workspace[workspaceOffset + column] * beta[1u + column];
  }
  workspace[workspaceOffset + W_INTERCEPT] = select(getNaN(), intercept, ok);
  for (var column = 0u; column < Q; column++) {
    workspace[workspaceOffset + W_BETA + column] = select(getNaN(), beta[column], ok);
  }`),...X(`residuals-tiles`,e=>[...Se(e),q(`residualsOut`,W,`read_write`),q(`partial`,be,`read_write`)],we,`let lambda = workspace[workspaceOffset + W_LAMBDA];
  let runtimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${O}>;
  for (var row = firstRow; row < lastRow; row++) {
    var f: array<f32, ${b}>;
    var ys = 0.0;
    getFilteredRow(row, lambda, &f, &ys);
    var filtered = ys;
    for (var column = 0u; column < Q; column++) {
      filtered -= f[column] * workspace[workspaceOffset + W_BETA + column];
    }
    // u = y - X b in the centered form (y - ybar) - a0 - (x - xbar) b, with a0 the intercept of
    // the centered model: y - (intercept + x b) would round every residual to a float32 step of
    // the response. Each difference from a mean is fixed with getRoundedDifference, because the
    // shader compiler of the test GPU otherwise regroups the chain around the mean.
    let centeredResponse = getRoundedDifference(response[responseOffset + row], workspace[workspaceOffset + W_YBAR], runtimeZero);
    var residual = centeredResponse - workspace[workspaceOffset + W_BETA];
    for (var column = 0u; column < K; column++) {
      residual -= getRoundedDifference(design[designOffset + row * K + column], workspace[workspaceOffset + column], runtimeZero)
        * workspace[workspaceOffset + W_BETA + 1u + column];
    }
    residualsOut[residualsOutOffset + row] = residual;
    sums[0] += filtered * filtered;
    sums[1] += residual;
    sums[2] += residual * residual;
    sums[3] += centeredResponse * residual;
    sums[4] += centeredResponse * centeredResponse;
  }
  for (var slot = 0u; slot < ${O}u; slot++) {
    partial[partialOffset + index * ${O}u + slot] = sums[slot];
  }`),Y(`residuals-merge`,[q(`partial`,be,`read`),q(`residualSums`,xe,`read_write`)],O,G,`${K(O)}
  residualSums[residualSumsOffset + index] = sum;`));let $=U;if(h){let e=T(`lag-residuals-fgls`,d),t=T(`partial-moments-fgls`,p*D),n=T(`moments-fgls`,D),r=T(`partial-power-sums-first`,p*k),i=T(`power-sums-first`,k),a=T(`partial-power-sums-final`,p*k),o=T(`partial-filtered-moments-final`,p*x),s=T(`filtered-moments-final`,x);$=s;let c=(t,n,r)=>X(`power-sums-tiles${t}`,()=>[q(`residual`,W,`read`),q(`lag`,e,`read`),q(`workspace`,L,`read`),q(`partial`,r,`read_write`)],G,`let lambda = workspace[workspaceOffset + ${n}];
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${k}>;
  for (var row = firstRow; row < lastRow; row++) {
    let filtered = residual[residualOffset + row] - lambda * lag[lagOffset + row];
    let square = filtered * filtered;
    sums[0] += square;
    sums[1] += square * filtered;
    sums[2] += square * square;
  }
  for (var slot = 0u; slot < ${k}u; slot++) {
    partial[partialOffset + index * ${k}u + slot] = sums[slot];
  }`);Q.push(...De(`-fgls`,W,e),...Ee(`-fgls`,W,e,t),Y(`moments-merge-fgls`,[q(`partial`,t,`read`),q(`moments`,n,`read_write`)],D,G,`${K(D)}
  moments[momentsOffset + index] = sum;`),...c(`-first`,`W_LAMBDA`,r),Y(`power-sums-merge-first`,[q(`partial`,r,`read`),q(`powers`,i,`read_write`)],k,G,`${K(k)}
  powers[powersOffset + index] = sum;`),Y(`lambda-solve-weighted`,[q(`moments`,n,`read`),q(`powers`,i,`read`),q(`powerTraces`,Ae,`read`),q(`gramTraces`,je,`read`),q(`infinityNorm`,Me,`read`),{name:`approximation`,view:Ne,type:`u32`,access:`read`},q(`workspace`,L,`read_write`)],1,`${G}
${Z}
${F(!0)}
${P()}`,`solveWeightedLambda();`),...c(`-final`,`W_LAMBDA_FINAL`,a),Y(`power-sums-merge-final`,[q(`partial`,a,`read`),q(`workspace`,L,`read_write`)],k,G,`${K(k)}
  workspace[workspaceOffset + W_POWER_SUMS + index] = sum;`),...Oe(`-final`,`W_LAMBDA_FINAL`,o),ke(`-final`,o,s))}return Q.push(Y(`finish`,[q(`filteredMoments`,U,`read`),...h?[q(`filteredMomentsFinal`,$,`read`)]:[],q(`workspace`,L,`read`),q(`residualSums`,xe,`read`),{name:`olsStatus`,view:A,type:`u32`,access:`read`},q(`tableOut`,_.table,`read_write`),q(`summaryOut`,_.summary,`read_write`),{name:`statusOut`,view:_.status.code,type:`u32`,access:`read_write`}],1,`${Te}
${se(b)}
${le(b-1)}
${h?`${I(b,`filteredMoments`,`solveFiltered`,!1)}
${I(b,`filteredMomentsFinal`,`solveFilteredFinal`,!0)}
${ve()}`:I(b,`filteredMoments`,`solveFilteredFinal`,!0)}
${ye(y,h)}`,`finish();`)),[...Q,...n(e,i,o.weights,o.output)]}};function ge(){return`
${C}
// Every difference of a value and its mean goes through getRoundedDifference: the shader
// compiler of the test GPU otherwise regroups (y - mean) - lambda * lag - f * b as
// y - (mean + ...), which rounds the filtered residual to a float32 step of the response.
fn getFilteredRow(row: u32, lambda: f32, f: ptr<function, array<f32, Q>>, ys: ptr<function, f32>) {
  let responseMean = workspace[workspaceOffset + W_YBAR];
  let runtimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  var rowSum = 0.0;
  var lagResponse = 0.0;
  var lagged: array<f32, K>;
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS) {
      continue;
    }
    let weight = weights[weightsOffset + slot];
    rowSum += weight;
    lagResponse += weight * getRoundedDifference(response[responseOffset + neighbor], responseMean, runtimeZero);
    for (var column = 0u; column < K; column++) {
      lagged[column] += weight * getRoundedDifference(design[designOffset + neighbor * K + column], workspace[workspaceOffset + column], runtimeZero);
    }
  }
  (*f)[0] = 1.0 - lambda * rowSum;
  for (var column = 0u; column < K; column++) {
    let centered = getRoundedDifference(design[designOffset + row * K + column], workspace[workspaceOffset + column], runtimeZero);
    (*f)[1u + column] = centered - lambda * lagged[column];
  }
  *ys = getRoundedDifference(response[responseOffset + row], responseMean, runtimeZero) - lambda * lagResponse;
}
`}function P(){return`
fn getObjective(lambda: f32) -> f32 { return -getMomentObjective(lambda); }
fn getDerivative(lambda: f32) -> f32 { return -getMomentSlope(lambda); }
${b(A,j)}
fn scanLambda(bound: f32) -> vec2<f32> {
  let result = searchParameter(bound);
  if (abs(result.parameter) <= 3.402823466e+38) {
    return vec2<f32>(result.parameter, getMomentObjective(result.parameter));
  }
  let startObjective = getMomentObjective(-bound);
  let endObjective = getMomentObjective(bound);
  if (endObjective < startObjective) {
    return vec2<f32>(bound, endObjective);
  }
  return vec2<f32>(-bound, startObjective);
}
`}function _e(){return`
var<private> momentA: vec3<f32>;
var<private> momentB: vec3<f32>;
var<private> momentG: vec3<f32>;
var<private> momentSigma: vec3<f32>;

// Moment residual with sigma2 at its constrained optimum: least squares in sigma2, clamped at zero.
fn getMomentResidual(lambda: f32) -> vec3<f32> {
  let free = momentG - momentA * lambda - momentB * (lambda * lambda);
  let sigma = max(dot(momentSigma, free) / dot(momentSigma, momentSigma), 0.0);
  return free - momentSigma * sigma;
}

fn getMomentObjective(lambda: f32) -> f32 {
  let residual = getMomentResidual(lambda);
  return dot(residual, residual);
}

// Half the derivative of the objective (envelope theorem: sigma2 is optimal or fixed at zero).
fn getMomentSlope(lambda: f32) -> f32 {
  return -dot(momentA + momentB * (2.0 * lambda), getMomentResidual(lambda));
}

fn solveLambda() {
  let n = f32(ROWS);
  let u2 = moments[momentsOffset];
  var finite = u2 > 0.0;
  for (var slot = 0u; slot < ${D}u; slot++) {
    finite = finite && isFiniteValue(moments[momentsOffset + slot]);
  }
  let bound = workspace[workspaceOffset + W_BOUND];
  if (!finite || !isFiniteValue(bound)) {
    workspace[workspaceOffset + W_LAMBDA] = getNaN();
    workspace[workspaceOffset + W_OBJECTIVE] = getNaN();
    return;
  }
  let uwu = moments[momentsOffset + 1u];
  let wu2 = moments[momentsOffset + 2u];
  let uwwu = moments[momentsOffset + 3u];
  let wuwwu = moments[momentsOffset + 4u];
  let wwu2 = moments[momentsOffset + 5u];
  let scale = 1.0 / u2;
  momentSigma = vec3<f32>(1.0, moments[momentsOffset + 6u] / n, 0.0);
  momentA = vec3<f32>(2.0 * uwu, 2.0 * wuwwu, uwwu + wu2) * scale;
  momentB = vec3<f32>(-wu2, -wwu2, -wuwwu) * scale;
  momentG = vec3<f32>(u2, wu2, uwu) * scale;
  let best = scanLambda(bound);
  workspace[workspaceOffset + W_LAMBDA] = best.x;
  workspace[workspaceOffset + W_OBJECTIVE] = best.y;
}
`}function F(e){return`
var<private> momentA: vec2<f32>;
var<private> momentB: vec2<f32>;
var<private> momentTarget: vec2<f32>;
var<private> momentWeight: mat2x2<f32>;

fn getMomentResidual(lambda: f32) -> vec2<f32> {
  return momentA * lambda + momentB * (lambda * lambda) - momentTarget;
}

fn getMomentObjective(lambda: f32) -> f32 {
  let residual = getMomentResidual(lambda);
  return dot(residual, momentWeight * residual);
}

// Half the derivative of the objective: (dr/dlambda)' M r.
fn getMomentSlope(lambda: f32) -> f32 {
  return dot(momentA + momentB * (2.0 * lambda), momentWeight * getMomentResidual(lambda));
}

// Fills momentA, momentB, momentTarget from the seven moment sums; false if they are not usable.
fn setMomentSystem() -> bool {
  let n = f32(ROWS);
  let u2 = moments[momentsOffset];
  var finite = u2 > 0.0;
  for (var slot = 0u; slot < ${D}u; slot++) {
    finite = finite && isFiniteValue(moments[momentsOffset + slot]);
  }
  if (!finite) {
    return false;
  }
  let uwu = moments[momentsOffset + 1u];
  let wu2 = moments[momentsOffset + 2u];
  let uwwu = moments[momentsOffset + 3u];
  let wuwwu = moments[momentsOffset + 4u];
  let wwu2 = moments[momentsOffset + 5u];
  let t = moments[momentsOffset + 6u];
  let ratio = t / n;
  let scale = 1.0 + ratio * ratio;
  momentA = vec2<f32>(2.0 * (wuwwu - ratio * uwu) / scale, uwwu + wu2) / n;
  momentB = vec2<f32>(-(wwu2 - ratio * wu2) / scale, -wuwwu) / n;
  momentTarget = vec2<f32>((wu2 - ratio * u2) / scale, uwu) / n;
  return true;
}
${e?`
var<private> traceTerms: vec4<f32>;

// psi (including the 1 / n) from sigma2 and the fourth moment mu4 of the filtered residuals.
fn getPsi(n: f32, sigma: f32, mu4: f32) -> mat2x2<f32> {
  let psi11 = 2.0 * sigma * sigma * traceTerms.x + (mu4 - 3.0 * sigma * sigma) * traceTerms.w;
  let psi12 = sigma * sigma * traceTerms.y;
  let psi22 = sigma * sigma * traceTerms.z;
  return mat2x2<f32>(psi11 / n, psi12 / n, psi12 / n, psi22 / n);
}

fn getPsiDeterminant(psi: mat2x2<f32>) -> f32 {
  return psi[0][0] * psi[1][1] - psi[0][1] * psi[1][0];
}

fn invertPsi(psi: mat2x2<f32>) -> mat2x2<f32> {
  return mat2x2<f32>(psi[1][1], -psi[0][1], -psi[1][0], psi[0][0]) * (1.0 / getPsiDeterminant(psi));
}

fn solveWeightedLambda() {
  let n = f32(ROWS);
  let nan = getNaN();
  workspace[workspaceOffset + W_LAMBDA_FINAL] = nan;
  workspace[workspaceOffset + W_OBJECTIVE_FINAL] = nan;
  workspace[workspaceOffset + W_APPROXIMATION] = f32(approximation[approximationOffset]);
  let bound = workspace[workspaceOffset + W_BOUND];
  var finite = isFiniteValue(bound);
  for (var slot = 0u; slot < ${k}u; slot++) {
    finite = finite && isFiniteValue(powers[powersOffset + slot]);
  }
  for (var slot = 0u; slot < 2u; slot++) {
    finite = finite && isFiniteValue(powerTraces[powerTracesOffset + slot]);
  }
  for (var slot = 0u; slot < 4u; slot++) {
    finite = finite && isFiniteValue(gramTraces[gramTracesOffset + slot]);
  }
  if (!finite || !setMomentSystem()) {
    return;
  }
  // powerTraces hold tr((W/s)^q): tr(W) = s c_1 and tr(W^2) = s^2 c_2.
  let norm = infinityNorm[infinityNormOffset];
  // Trace terms: tr(A1 A1), tr(A1 2 A2), tr(2 A2 A2) and diag(A1)'diag(A1).
  let t = moments[momentsOffset + 6u];
  let ratio = t / n;
  let scale = 1.0 + ratio * ratio;
  let scaleSquared = scale * scale;
  traceTerms = vec4<f32>(
    (gramTraces[gramTracesOffset + 2u] - t * ratio) / scaleSquared,
    (2.0 * gramTraces[gramTracesOffset + 1u] - 2.0 * ratio * powerTraces[powerTracesOffset] * norm) / scale,
    powerTraces[powerTracesOffset + 1u] * norm * norm + t,
    (gramTraces[gramTracesOffset + 3u] - t * ratio) / scaleSquared
  );
  // psi at lambda1 from the power sums of u_s = u1 - lambda1 W u1.
  let sigma = powers[powersOffset] / n;
  let psi = getPsi(n, sigma, powers[powersOffset + 2u] / n);
  let determinant = getPsiDeterminant(psi);
  if (!(determinant > 0.0) || !isFiniteValue(determinant)) {
    return;
  }
  var weight = invertPsi(psi);
  // Normalize the weight to unit trace: same minimizer, well-scaled objective.
  let weightScale = weight[0][0] + weight[1][1];
  weight = weight * (1.0 / weightScale);
  momentWeight = weight;
  let best = scanLambda(bound);
  workspace[workspaceOffset + W_LAMBDA_FINAL] = best.x;
  workspace[workspaceOffset + W_OBJECTIVE_FINAL] = best.y * weightScale;
  workspace[workspaceOffset + W_MOMENT_MATRIX] = momentA.x;
  workspace[workspaceOffset + W_MOMENT_MATRIX + 1u] = momentA.y;
  workspace[workspaceOffset + W_MOMENT_MATRIX + 2u] = momentB.x;
  workspace[workspaceOffset + W_MOMENT_MATRIX + 3u] = momentB.y;
  for (var slot = 0u; slot < 4u; slot++) {
    workspace[workspaceOffset + W_TRACE_TERMS + slot] = traceTerms[slot];
  }
}
`:`
fn solveFirstStepLambda() {
  let bound = workspace[workspaceOffset + W_BOUND];
  if (!isFiniteValue(bound) || !setMomentSystem()) {
    workspace[workspaceOffset + W_LAMBDA] = getNaN();
    workspace[workspaceOffset + W_OBJECTIVE] = getNaN();
    return;
  }
  momentWeight = mat2x2<f32>(1.0, 0.0, 0.0, 1.0);
  let best = scanLambda(bound);
  workspace[workspaceOffset + W_LAMBDA] = best.x;
  workspace[workspaceOffset + W_OBJECTIVE] = best.y;
}
`}
`}function I(e,t,n,r){let i=e=>`${t}[${t}Offset + ${e}]`;return`
fn ${n}(
  beta: ptr<function, array<f32, ${e}>>,
  inverseDiagonal: ptr<function, array<f32, ${e}>>,
  interceptQuadratic: ptr<function, f32>,
  conditioning: ptr<function, array<f32, 3>>
) -> bool {
  var scale: array<f32, ${e}>;
  for (var a = 0u; a < Q; a++) {
    let diagonal = ${i(`a * Q + a`)};
    if (!(diagonal > 0.0) || !isFiniteValue(diagonal)) {
      return false;
    }
    scale[a] = 1.0 / sqrt(diagonal);
  }
  var factor: array<f32, ${e*e}>;
  var rhs: array<f32, ${e}>;
  for (var a = 0u; a < Q; a++) {
    for (var b = 0u; b < Q; b++) {
      factor[a * Q + b] = ${i(`a * Q + b`)} * scale[a] * scale[b];
    }
    rhs[a] = ${i(`Q * Q + a`)} * scale[a];
  }
${r?`
  // Correlation condition number of the unit-diagonal matrix that is factored.
  var correlation = factor;
  let correlationRange = getSymmetricEigenvalueRange_${e}(&correlation);
  (*conditioning)[1] = correlationRange.y / max(correlationRange.x, 1.0e-30);
`:``}
  if (!choleskyFactor_${e}(&factor)) {
    return false;
  }
${r?`
  // Belsley condition number of the raw filtered design [f0, f_j + m_j f0] (f0 is the filtered
  // constant, not orthogonal to the centered columns), without forming its Gram matrix.
  // With the Cholesky factor L of the unit-diagonal Gram of [f0, f_1..f_K] the raw column j is
  // inPlane_j u0 + g_j sum_i L[j][i] e_i (u0 the unit f0, e_i orthonormal), so the helper's
  // alpha_j = inPlane_j / N_j, beta_j = g_j / N_j and factor = L without its first row and column.
  var belsleyFactor: array<f32, ${(e-1)*(e-1)}>;
  var belsleyAlpha: array<f32, ${e-1}>;
  var belsleyBeta: array<f32, ${e-1}>;
  for (var j = 0u; j < K; j++) {
    let column = 1.0 / scale[1u + j];
    let inPlane = workspace[workspaceOffset + j] / scale[0] + column * factor[(1u + j) * Q];
    var tail = 0.0;
    for (var i = 0u; i <= j; i++) {
      belsleyFactor[j * K + i] = factor[(1u + j) * Q + 1u + i];
      tail = tail + factor[(1u + j) * Q + 1u + i] * factor[(1u + j) * Q + 1u + i];
    }
    let rawNorm = sqrt(inPlane * inPlane + column * column * tail);
    belsleyAlpha[j] = inPlane / rawNorm;
    belsleyBeta[j] = column / rawNorm;
  }
  (*conditioning)[0] = getBelsleyConditionNumber_${e-1}(&belsleyFactor, &belsleyAlpha, &belsleyBeta);
  var inflation = 0.0;
  for (var b = 1u; b < Q; b++) {
    inflation = max(inflation, choleskyInverseDiagonal_${e}(&factor, b));
  }
  (*conditioning)[2] = inflation;
`:``}
  choleskySolve_${e}(&factor, &rhs);
  var meansScaled: array<f32, ${e}>;
  meansScaled[0] = scale[0];
  for (var b = 0u; b < K; b++) {
    meansScaled[1u + b] = -workspace[workspaceOffset + b] * scale[1u + b];
  }
  var solved = meansScaled;
  choleskySolve_${e}(&factor, &solved);
  var quadratic = 0.0;
  for (var b = 0u; b < Q; b++) {
    (*beta)[b] = rhs[b] * scale[b];
    (*inverseDiagonal)[b] = choleskyInverseDiagonal_${e}(&factor, b) * scale[b] * scale[b];
    quadratic += meansScaled[b] * solved[b];
  }
  *interceptQuadratic = quadratic;
  return true;
}
`}function ve(){return`
var<private> traceTerms: vec4<f32>;

fn getPsi(n: f32, sigma: f32, mu4: f32) -> mat2x2<f32> {
  let psi11 = 2.0 * sigma * sigma * traceTerms.x + (mu4 - 3.0 * sigma * sigma) * traceTerms.w;
  let psi12 = sigma * sigma * traceTerms.y;
  let psi22 = sigma * sigma * traceTerms.z;
  return mat2x2<f32>(psi11 / n, psi12 / n, psi12 / n, psi22 / n);
}

// Standard error of lambda, NaN if psi is not positive definite.
fn getLambdaStandardError(n: f32, lambda: f32, sigma: f32, mu4: f32) -> f32 {
  for (var slot = 0u; slot < 4u; slot++) {
    traceTerms[slot] = workspace[workspaceOffset + W_TRACE_TERMS + slot];
  }
  let psi = getPsi(n, sigma, mu4);
  let determinant = psi[0][0] * psi[1][1] - psi[0][1] * psi[1][0];
  if (!(determinant > 0.0) || !isFiniteValue(determinant)) {
    return getNaN();
  }
  let inverse = mat2x2<f32>(psi[1][1], -psi[0][1], -psi[1][0], psi[0][0]) * (1.0 / determinant);
  let j = vec2<f32>(
    workspace[workspaceOffset + W_MOMENT_MATRIX] + 2.0 * lambda * workspace[workspaceOffset + W_MOMENT_MATRIX + 2u],
    workspace[workspaceOffset + W_MOMENT_MATRIX + 1u] + 2.0 * lambda * workspace[workspaceOffset + W_MOMENT_MATRIX + 3u]
  );
  let quadratic = dot(j, inverse * j);
  return sqrt(1.0 / (quadratic * n));
}
`}function ye(e,t){let n=e=>`summaryOut[summaryOutOffset + ${e}u]`;return`
${S}

fn fail(status: u32) {
  let nan = getNaN();
  for (var slot = 0u; slot < ${(e+2)*4}u; slot++) {
    tableOut[tableOutOffset + slot] = nan;
  }
  for (var slot = 0u; slot < 13u; slot++) {
    summaryOut[summaryOutOffset + slot] = nan;
  }
  summaryOut[summaryOutOffset] = f32(ROWS);
  statusOut[statusOutOffset] = status;
}

fn finish() {
  let n = f32(ROWS);
  if (!(n > f32(Q + 1u))) {
    fail(2u);
    return;
  }
  if (olsStatus[olsStatusOffset] != 0u) {
    fail(select(1u, 2u, olsStatus[olsStatusOffset] == 2u));
    return;
  }
  let lambda = workspace[workspaceOffset + ${t?`W_LAMBDA_FINAL`:`W_LAMBDA`}];
  if (!isFiniteValue(lambda)) {
    fail(3u);
    return;
  }
  var beta: array<f32, ${e+1}>;
  var inverseDiagonal: array<f32, ${e+1}>;
  var interceptQuadratic = 0.0;
  var conditioning: array<f32, 3>;
${t?`
  // The reported betas are the FGLS fit at lambda1; their variance uses the system at lambda.
  var unusedInverse: array<f32, ${e+1}>;
  var unusedQuadratic = 0.0;
  if (!solveFiltered(&beta, &unusedInverse, &unusedQuadratic, &conditioning)) {
    fail(1u);
    return;
  }
  var finalBeta: array<f32, ${e+1}>;
  if (!solveFilteredFinal(&finalBeta, &inverseDiagonal, &interceptQuadratic, &conditioning)) {
    fail(1u);
    return;
  }
  let sigmaSquared = workspace[workspaceOffset + W_POWER_SUMS] / n;
  let fourthMoment = workspace[workspaceOffset + W_POWER_SUMS + 2u] / n;
  let lambdaStandardError = getLambdaStandardError(n, lambda, sigmaSquared, fourthMoment);`:`
  if (!solveFilteredFinal(&beta, &inverseDiagonal, &interceptQuadratic, &conditioning)) {
    fail(1u);
    return;
  }
  let sigmaSquared = residualSums[residualSumsOffset] / n;`}
  let filteredSquares = residualSums[residualSumsOffset];
  let residualTotal = residualSums[residualSumsOffset + 1u];
  let residualSquares = residualSums[residualSumsOffset + 2u];
  let responseResidual = residualSums[residualSumsOffset + 3u];
  let responseSquares = residualSums[residualSumsOffset + 4u];
  if (!isFiniteValue(filteredSquares) || !isFiniteValue(residualSquares) || !isFiniteValue(responseSquares)) {
    fail(3u);
    return;
  }
  for (var row = 0u; row < Q; row++) {
    // Row 0 is the intercept, rows 1..K the predictors.
    let coefficient = select(beta[row], workspace[workspaceOffset + W_INTERCEPT], row == 0u);
    let variance = sigmaSquared * select(inverseDiagonal[row], interceptQuadratic, row == 0u);
    let standardError = sqrt(variance);
    let z = coefficient / standardError;
    tableOut[tableOutOffset + row * 4u] = coefficient;
    tableOut[tableOutOffset + row * 4u + 1u] = standardError;
    tableOut[tableOutOffset + row * 4u + 2u] = z;
    tableOut[tableOutOffset + row * 4u + 3u] = erfcValue(abs(z) * 0.70710678);
  }
  let nan = getNaN();
${t?`
  let lambdaZ = lambda / lambdaStandardError;
  tableOut[tableOutOffset + Q * 4u] = lambda;
  tableOut[tableOutOffset + Q * 4u + 1u] = lambdaStandardError;
  tableOut[tableOutOffset + Q * 4u + 2u] = lambdaZ;
  tableOut[tableOutOffset + Q * 4u + 3u] = erfcValue(abs(lambdaZ) * 0.70710678);`:`
  tableOut[tableOutOffset + Q * 4u] = lambda;
  tableOut[tableOutOffset + Q * 4u + 1u] = nan;
  tableOut[tableOutOffset + Q * 4u + 2u] = nan;
  tableOut[tableOutOffset + Q * 4u + 3u] = nan;`}
  // Squared correlation of y and y - u: the response is centered, so Syy = responseSquares.
  let predictedVariance = responseSquares - 2.0 * responseResidual + residualSquares - residualTotal * residualTotal / n;
  let covariance = responseSquares - responseResidual;
  let bound = workspace[workspaceOffset + W_BOUND];
  summaryOut[summaryOutOffset] = n;
  ${n(1)} = lambda;
  ${n(2)} = sigmaSquared;
  ${n(3)} = covariance * covariance / (responseSquares * predictedVariance);
  ${n(4)} = residualSquares;
  ${n(5)} = workspace[workspaceOffset + ${t?`W_OBJECTIVE_FINAL`:`W_OBJECTIVE`}];
  ${n(6)} = conditioning[0];
  ${n(7)} = conditioning[1];
  ${n(8)} = conditioning[2];
  ${n(9)} =
    select(0.0, 1.0, conditioning[1] > ${u(y)});
  ${n(10)} = bound;
  ${n(11)} = ${t?`workspace[workspaceOffset + W_APPROXIMATION]`:`0.0`};
  ${n(12)} = select(0.0, 1.0, abs(lambda) >= bound * 0.9999);
  statusOut[statusOutOffset] = 0u;
}
`}export{N as t};