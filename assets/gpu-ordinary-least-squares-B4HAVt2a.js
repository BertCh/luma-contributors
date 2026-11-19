import{r as e}from"./kernel-C3tfQOn4.js";import{c as t,o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as r,f as i,y as a}from"./gpu-contributor-utils-BIwLQON5.js";import{a as o,i as s,n as c,t as l}from"./spatial-regression-solve-Vf-sQz9O.js";import{r as u,u as d}from"./wgsl-common-DfyXiYB0.js";import{t as f}from"./pairwise-sum-wgsl-iErcC5Or.js";import{t as p}from"./rounded-difference-wgsl-DgXXQtNj.js";import{n as m,t as h}from"./centered-regression-prediction-pXjnms-4.js";import{a as g}from"./graph-chunk-dispatch-zBt8k2-o.js";import{i as _}from"./spatial-likelihood-allocations-BGEHbqkB.js";import{t as v}from"./regression-output-dispatches-KKVTWkHP.js";var y=1e4;function b(e){if(!Number.isInteger(e)||e<1)throw Error(`Chi-square degrees of freedom must be a positive integer`);let t=e/2,n=e%2==0?1:.5,r=e%2==0?0:.5*Math.log(Math.PI);for(;n<t;)r+=Math.log(n),n+=1;return r}var x=`GPUOrdinaryLeastSquares`,S=64,C=4096,w=64,T=class{id;props;rowCount;tileRowCount;tileCount;constructor(e){this.id=e.id??`ordinary-least-squares`,this.props=e;let t=this.id,{predictorCount:n,output:o}=e;i(t,`predictorCount`,n,1,15),r(e.predictors,[`float32`],`${t} predictors`),r(e.response,[`float32`],`${t} response`);let s=e.response.length;if(s<1)throw Error(`${t} needs at least one row`);if(e.predictors.length!==s*n)throw Error(`${t} predictors length must equal response length * predictorCount`);if(e.mask&&(r(e.mask,[`uint32`],`${t} mask`),e.mask.length!==s))throw Error(`${t} mask length must equal response length`);if(e.parameters&&(r(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<1))throw Error(`${t} parameters must hold 1 float32 value`);let c=n+1;for(let e of[`coefficients`,`standardErrors`,`tStatistics`])if(r(o[e],[`float32`],`${t} output.${e}`),o[e].length<c)throw Error(`${t} output.${e} must hold predictorCount + 1 rows`);if(r(o.summary,[`float32`],`${t} output.summary`),o.summary.length<22)throw Error(`${t} output.summary must hold 22 float32 values`);if(r(o.status.code,[`uint32`],`${t} output.status.code`),o.status.code.length<1)throw Error(`${t} output.status.code must hold one uint32 row`);for(let e of[`residuals`,`fitted`,`predictions`]){let n=o[e];if(n&&(r(n,[`float32`],`${t} output.${e}`),n.length<s))throw Error(`${t} output.${e} must hold one row per input row`)}e.tileRowCount!==void 0&&i(t,`tileRowCount`,e.tileRowCount),this.rowCount=s,this.tileRowCount=e.tileRowCount??Math.max(w,Math.ceil(s/C)),this.tileCount=Math.ceil(s/this.tileRowCount),v(t,{coefficients:o.coefficients,standardErrors:o.standardErrors,tStatistics:o.tStatistics,summary:o.summary,code:o.status.code,residuals:o.residuals,fitted:o.fitted,predictions:o.predictions},[[`coefficients`,`standardErrors`,`tStatistics`,`summary`,`code`],[`residuals`,`fitted`,`predictions`]]),a(t,[o.coefficients,o.standardErrors,o.tStatistics,o.summary,o.status.code,o.residuals,o.fitted,o.predictions],[e.predictors,e.response,e.mask,e.parameters])}getCommandNodes(t){let{props:n,id:r,rowCount:i,tileRowCount:a,tileCount:o}=this,{output:u,predictorCount:d}=n;g(t,r,{predictors:n.predictors,response:n.response,mask:n.mask,parameters:n.parameters,coefficients:u.coefficients,standardErrors:u.standardErrors,tStatistics:u.tStatistics,summary:u.summary,code:u.status.code,residuals:u.residuals,fitted:u.fitted,predictions:u.predictions}),_(t.device,r,D(i,d,a));let p=d,h=O(p),v=(n,i)=>e(t,`${r}-${n}`,`uint32`,i),y=(n,i)=>e(t,`${r}-${n}`,`float32`,i),b=u.residuals??y(`residuals`,i),C=y(`squared-residuals`,i),w=[],T={id:r,operation:x,predictors:n.predictors,mask:n.mask,rowCount:i,k:p,tileRowCount:a,tileCount:o},U=v(`workspace`,h.length);w.push(...N(t,{...T,prefix:r,response:n.response,workspace:U,partialMeans:v(`partial-means`,o*k(p)),partialMoments:v(`partial-moments`,o*(h.momentCount+p+1))})),w.push(H(t,{id:`${r}-solve`,variant:`solve`,bindings:[...n.parameters?[R(`params`,n.parameters,`f32`)]:[],V(U)],invocationCount:1,declarations:`${j(h,!0)}
${c(p)}`,body:P(p,!!n.parameters)})),w.push(H(t,{id:`${r}-residuals`,variant:`residuals`,bindings:[R(`design`,n.predictors,`f32`),R(`response`,n.response,`f32`),...n.mask?[R(`rowMask`,n.mask,`u32`)]:[],B(U),z(`residualOut`,b,`f32`),...u.fitted?[z(`fittedOut`,u.fitted,`f32`)]:[],...u.predictions?[z(`predictionOut`,u.predictions,`f32`)]:[],z(`squaredOut`,C,`f32`)],invocationCount:i,declarations:`${j(h,!1)}
${M(i,p,a,!!n.mask)}`,body:`let nan = getNaN();
  residualOut[residualOutOffset + index] = nan;
  ${u.fitted?`fittedOut[fittedOutOffset + index] = nan;`:``}
  ${u.predictions?`predictionOut[predictionOutOffset + index] = nan;`:``}
  squaredOut[squaredOutOffset + index] = nan;
  if (ws[wsOffset + W_STATUS] != 0u) {
    return;
  }
  for (var column = 0u; column < K; column++) {
    if (!isFiniteValue(design[designOffset + index * K + column])) { return; }
  }
  // Each x - xbar is rounded as written, as in the moment pass: a compiler that regrouped the sum
  // into sum(b x) - sum(b xbar) would round every fitted value to a float32 step of b x.
  ${m({predictorCount:p,predictor:`design[designOffset + index * K + column]`,mean:`wsGet(W_MEANS + column)`,slope:`wsGet(W_SLOPES + column)`,runtimeZero:`ws[wsOffset + W_RUNTIME_ZERO]`})}
  let fit = centeredContribution;
  let meanResponse = wsGet(W_MEANS + K);
  // The stored means are float32: W_RESIDUAL_OFFSET is what they miss of the true means, carried
  // through the fit, so the residuals sum to zero as least squares with an intercept requires.
  let offset = wsGet(W_RESIDUAL_OFFSET);
  ${u.predictions?`predictionOut[predictionOutOffset + index] = getCenteredRegressionPrediction(meanResponse, fit, offset, ws[wsOffset + W_RUNTIME_ZERO]);`:``}
  if (!isRowValid(index)) { return; }
  // getRoundedDifference: without it the shader compiler of the test GPU adds the offset to the
  // mean first, where it is lost below the float32 step of the mean.
  let centeredResponse = getRoundedDifference(response[responseOffset + index], meanResponse, ws[wsOffset + W_RUNTIME_ZERO]);
  let residual = (centeredResponse - fit) - offset;
  residualOut[residualOutOffset + index] = residual;
  ${u.fitted?`fittedOut[fittedOutOffset + index] = getCenteredRegressionPrediction(meanResponse, fit, offset, ws[wsOffset + W_RUNTIME_ZERO]);`:``}
  squaredOut[squaredOutOffset + index] = residual * residual;`}));let W=v(`partial-residuals`,o*E);w.push(H(t,{id:`${r}-residual-tiles`,variant:`residual-tiles`,workgroupSize:S,bindings:[R(`residualIn`,b,`f32`),z(`partial`,W,`u32`)],invocationCount:o,declarations:`const ROW_COUNT: u32 = ${i}u;
const TILE_ROWS: u32 = ${a}u;
${A}`,body:`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var sumSquares = 0.0;
  var sumCubes = 0.0;
  var sumFourth = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    let e = residualIn[residualInOffset + row];
    if (!isFiniteValue(e)) {
      continue;
    }
    let e2 = e * e;
    sumSquares = sumSquares + e2;
    sumCubes = sumCubes + e2 * e;
    sumFourth = sumFourth + e2 * e2;
  }
  let base = partialOffset + index * ${E}u;
  partial[base] = bitcast<u32>(sumSquares);
  partial[base + 1u] = bitcast<u32>(sumCubes);
  partial[base + 2u] = bitcast<u32>(sumFourth);`})),w.push(H(t,{id:`${r}-residual-merge`,variant:`residual-merge`,bindings:[R(`partial`,W,`u32`),V(U)],invocationCount:E,declarations:`${j(h,!0)}
const TILE_COUNT: u32 = ${o}u;`,body:`${f({count:`TILE_COUNT`,term:`bitcast<f32>(partial[partialOffset + sumItem * ${E}u + index])`})}
  wsSet(W_RESIDUAL_SUMS + index, sum);`})),w.push(H(t,{id:`${r}-finish`,variant:`finish`,bindings:[B(U),z(`coefficientsOut`,u.coefficients,`f32`),z(`standardErrorsOut`,u.standardErrors,`f32`),z(`tStatisticsOut`,u.tStatistics,`f32`),z(`summaryOut`,u.summary,`f32`),z(`statusOut`,u.status.code,`u32`)],invocationCount:1,declarations:`${j(h,!1)}
${c(p)}
${s(p)}
${l(p)}`,body:F(p)}));let G=v(`bp-workspace`,h.length);return w.push(...N(t,{...T,prefix:`${r}-bp`,response:C,workspace:G,partialMeans:v(`bp-partial-means`,o*k(p)),partialMoments:v(`bp-partial-moments`,o*(h.momentCount+p+1))})),w.push(H(t,{id:`${r}-bp-solve`,variant:`solve`,bindings:[V(G)],invocationCount:1,declarations:`${j(h,!0)}
${c(p)}`,body:P(p,!1)})),w.push(H(t,{id:`${r}-bp-finish`,variant:`breusch-pagan`,bindings:[B(G),R(`mainWs`,U,`u32`),z(`summaryOut`,u.summary,`f32`)],invocationCount:1,declarations:`${j(h,!1)}
${I(p)}`,body:L(p)})),w}},E=3;function D(e,t,n=Math.max(64,Math.ceil(e/4096))){let r=O(t),i=Math.ceil(e/n);return{"OLS residuals/squared residuals":e,"OLS workspace":r.length,"OLS mean partials":i*k(t),"OLS moment partials":i*(r.momentCount+t+1),"OLS residual partials":i*E}}function O(e){let t=e+1,n=t*(t+1)/2,r=0+t,i=r+1,a=i+1,o=a+1,s=o+E,c=s+e,l=c+e,u=l+e*e,d=u+n,f=d+t;return{momentCount:n,means:0,count:r,status:i,lambda:a,residualSums:o,scale:s,slopes:c,factor:l,moments:u,deviationSums:d,residualOffset:f,runtimeZero:f+1,length:f+2}}function k(e){return 2*(e+1)+2}var A=`
${u}

${d}
`;function j(e,t){let n=e.slopes-e.scale;return`
const K: u32 = ${n}u;
const D: u32 = ${n+1}u;
const KK: u32 = ${n*n}u;
const W_MEANS: u32 = ${e.means}u;
const W_COUNT: u32 = ${e.count}u;
const W_STATUS: u32 = ${e.status}u;
const W_LAMBDA: u32 = ${e.lambda}u;
const W_RESIDUAL_SUMS: u32 = ${e.residualSums}u;
const W_SCALE: u32 = ${e.scale}u;
const W_SLOPES: u32 = ${e.slopes}u;
const W_FACTOR: u32 = ${e.factor}u;
const W_MOMENTS: u32 = ${e.moments}u;
const W_DEVIATION_SUMS: u32 = ${e.deviationSums}u;
const W_RESIDUAL_OFFSET: u32 = ${e.residualOffset}u;
const W_RUNTIME_ZERO: u32 = ${e.runtimeZero}u;
${p}
${h}
${A}
fn wsGet(slot: u32) -> f32 {
  return bitcast<f32>(ws[wsOffset + slot]);
}
${t?`fn wsSet(slot: u32, value: f32) {
  ws[wsOffset + slot] = bitcast<u32>(value);
}`:``}
// The stored float32 mean of column c misses the true mean by getMeanCorrection(c), which is
// exact to the rounding of a sum of small deviations.
fn getMeanCorrection(column: u32) -> f32 {
  return getRoundedDifference(wsGet(W_DEVIATION_SUMS + column) / max(f32(ws[wsOffset + W_COUNT]), 1.0), 0.0, ws[wsOffset + W_RUNTIME_ZERO]);
}
// Cross-product about the true means: the sum about the stored means less n times the product of
// the two mean corrections.
fn momentAt(first: u32, second: u32) -> f32 {
  let lo = min(first, second);
  let hi = max(first, second);
  return wsGet(W_MOMENTS + lo * (2u * D - lo + 1u) / 2u + (hi - lo))
    - wsGet(W_DEVIATION_SUMS + first) * getMeanCorrection(second);
}
`}function M(e,t,n,r){return`
const ROW_COUNT: u32 = ${e}u;
const TILE_ROWS: u32 = ${n}u;
const ROW_K: u32 = ${t}u;

fn isRowValid(row: u32) -> bool {
  ${r?`if (rowMask[rowMaskOffset + row] == 0u) {
    return false;
  }`:``}
  if (!isFiniteValue(response[responseOffset + row])) {
    return false;
  }
  for (var j = 0u; j < ROW_K; j++) {
    if (!isFiniteValue(design[designOffset + row * ROW_K + j])) {
      return false;
    }
  }
  return true;
}
`}function N(e,t){let{prefix:n,k:r,rowCount:i,tileRowCount:a,tileCount:o,workspace:s}=t,c=r+1,l=O(r),u=l.momentCount,d=k(r),m=!!t.mask,h=[R(`design`,t.predictors,`f32`),R(`response`,t.response,`f32`),...t.mask?[R(`rowMask`,t.mask,`u32`)]:[]],g=`${j(l,!1)}
${M(i,r,a,m)}`,_=`${j(l,!0)}
const TILE_COUNT: u32 = ${o}u;`,v=e=>`${f({count:`TILE_COUNT`,term:`bitcast<f32>(partial[partialOffset + sumItem * ${e} + index])`})}`;return[H(e,{id:`${n}-means-origins`,variant:`means-origins`,workgroupSize:S,bindings:[...h,z(`partial`,t.partialMeans,`u32`)],invocationCount:o,declarations:`${A}\n${M(i,r,a,m)}`,body:`let base = partialOffset + index * ${d}u;
  for (var column = 0u; column < ${c}u; column++) { partial[base + column] = bitcast<u32>(0.0); }
  // Separate dispatch: later difference bit casts read a zero the compiler cannot know.
  partial[base + ${2*c+1}u] = 0u;
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  for (var row = firstRow; row < lastRow; row++) {
    if (!isRowValid(row)) { continue; }
    for (var column = 0u; column < ROW_K; column++) {
      partial[base + column] = bitcast<u32>(design[designOffset + row * ROW_K + column]);
    }
    partial[base + ROW_K] = bitcast<u32>(response[responseOffset + row]);
    break;
  }`}),H(e,{id:`${n}-means-tiles`,variant:`means-tiles`,workgroupSize:S,bindings:[...h,z(`partial`,t.partialMeans,`u32`)],invocationCount:o,declarations:`${A}\n${p}
${M(i,r,a,m)}`,body:`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  let base = partialOffset + index * ${d}u;
  let runtimeZero = partial[base + ${2*c+1}u];
  var sums: array<f32, ${c}>;
  var count = 0u;
  for (var row = firstRow; row < lastRow; row++) {
    if (!isRowValid(row)) {
      continue;
    }
    count = count + 1u;
    for (var j = 0u; j < ROW_K; j++) {
      sums[j] += getRoundedDifference(design[designOffset + row * ROW_K + j], bitcast<f32>(partial[base + j]), runtimeZero);
    }
    sums[ROW_K] += getRoundedDifference(response[responseOffset + row], bitcast<f32>(partial[base + ROW_K]), runtimeZero);
  }
  for (var column = 0u; column < ${c}u; column++) {
    partial[base + ${c}u + column] = bitcast<u32>(sums[column]);
  }
  partial[base + ${2*c}u] = count;`}),H(e,{id:`${n}-means-merge`,variant:`means-merge`,bindings:[R(`partial`,t.partialMeans,`u32`),V(s)],invocationCount:c,declarations:_,body:`var rows = 0u;
  var referenceTile = 0u;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let tileRows = partial[partialOffset + tile * ${d}u + ${2*c}u];
    if (rows == 0u && tileRows > 0u) { referenceTile = tile; }
    rows += tileRows;
  }
  let origin = bitcast<f32>(partial[partialOffset + referenceTile * ${d}u + index]);
  let runtimeZero = partial[partialOffset + ${2*c+1}u];
  ${f({count:`TILE_COUNT`,setup:`let base = partialOffset + sumItem * ${d}u;
    let tileRows = partial[base + ${2*c}u];
    var contribution = 0.0;
    if (tileRows > 0u) {
      let shift = getRoundedDifference(bitcast<f32>(partial[base + index]), origin, runtimeZero);
      contribution = bitcast<f32>(partial[base + ${c}u + index]) + shift * f32(tileRows);
    }`,term:`contribution`})}
  wsSet(W_MEANS + index, select(0.0, origin + sum / max(f32(rows), 1.0), rows > 0u));
  if (index == 0u) {
    ws[wsOffset + W_COUNT] = rows;
    // A zero the compiler cannot see: the runtime zero of getRoundedDifference.
    ws[wsOffset + W_RUNTIME_ZERO] = 0u;
  }`}),H(e,{id:`${n}-moments-tiles`,variant:`moments-tiles`,workgroupSize:S,bindings:[...h,B(s),z(`partial`,t.partialMoments,`u32`)],invocationCount:o,declarations:g,body:`var means: array<f32, ${c}>;
  for (var column = 0u; column < D; column++) {
    means[column] = wsGet(W_MEANS + column);
  }
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  // Each deviation is rounded as written (getRoundedDifference): the shader compiler of the test
  // GPU otherwise sums the raw values of a tile and subtracts the mean once, which loses the
  // deviation sums to the float32 step of a tile total.
  let runtimeZero = ws[wsOffset + W_RUNTIME_ZERO];
  var accumulator: array<f32, ${u+c}>;
  var centered: array<f32, ${c}>;
  for (var row = firstRow; row < lastRow; row++) {
    if (!isRowValid(row)) {
      continue;
    }
    for (var j = 0u; j < ROW_K; j++) {
      centered[j] = getRoundedDifference(design[designOffset + row * ROW_K + j], means[j], runtimeZero);
    }
    centered[ROW_K] = getRoundedDifference(response[responseOffset + row], means[ROW_K], runtimeZero);
    var slot = 0u;
    for (var first = 0u; first < D; first++) {
      for (var second = first; second < D; second++) {
        accumulator[slot] = accumulator[slot] + centered[first] * centered[second];
        slot = slot + 1u;
      }
    }
    // The deviations from the stored float32 means do not sum to zero: their sums give the part
    // of each true mean that float32 cannot hold.
    for (var column = 0u; column < D; column++) {
      accumulator[${u}u + column] = accumulator[${u}u + column] + centered[column];
    }
  }
  for (var slot = 0u; slot < ${u+c}u; slot++) {
    partial[partialOffset + index * ${u+c}u + slot] = bitcast<u32>(accumulator[slot]);
  }`}),H(e,{id:`${n}-moments-merge`,variant:`moments-merge`,bindings:[R(`partial`,t.partialMoments,`u32`),V(s)],invocationCount:u+c,declarations:_,body:`${v(`${u+c}u`)}
  wsSet(W_MOMENTS + index, sum);`})]}function P(e,n){return`let rows = ws[wsOffset + W_COUNT];
  ${n?`let requested = params[paramsOffset];
  let lambda = select(0.0, requested, requested > 0.0 && isFiniteValue(requested));`:`let lambda = 0.0;`}
  wsSet(W_LAMBDA, lambda);
  if (rows <= K + 1u) {
    ws[wsOffset + W_STATUS] = 2u;
    return;
  }
  var a: array<f32, ${e*e}>;
  var rhs: array<f32, ${e}>;
  var scale: array<f32, ${e}>;
  for (var i = 0u; i < K; i++) {
    let variance = momentAt(i, i);
    if (!(variance > 0.0) || !isFiniteValue(variance)) {
      ws[wsOffset + W_STATUS] = 1u;
      return;
    }
    scale[i] = sqrt(variance);
  }
  for (var i = 0u; i < K; i++) {
    for (var j = 0u; j <= i; j++) {
      var entry = momentAt(i, j) / (scale[i] * scale[j]);
      if (i == j) {
        entry = entry + lambda / (scale[i] * scale[i]);
      }
      a[i * K + j] = entry;
    }
    rhs[i] = momentAt(i, K) / scale[i];
  }
  if (!choleskyFactor_${e}(&a)) {
    ws[wsOffset + W_STATUS] = 1u;
    return;
  }
  for (var i = 0u; i < K; i++) {
    if (!(a[i * K + i] >= ${t(o)})) {
      ws[wsOffset + W_STATUS] = 1u;
      return;
    }
  }
  for (var i = 0u; i < KK; i++) {
    wsSet(W_FACTOR + i, a[i]);
  }
  choleskySolve_${e}(&a, &rhs);
  var residualOffset = getMeanCorrection(K);
  for (var i = 0u; i < K; i++) {
    wsSet(W_SCALE + i, scale[i]);
    wsSet(W_SLOPES + i, rhs[i] / scale[i]);
    residualOffset = residualOffset - (rhs[i] / scale[i]) * getMeanCorrection(i);
  }
  wsSet(W_RESIDUAL_OFFSET, residualOffset);
  ws[wsOffset + W_STATUS] = 0u;`}function F(e){let n=e+1;return`let rows = ws[wsOffset + W_COUNT];
  let status = ws[wsOffset + W_STATUS];
  let nan = getNaN();
  statusOut[statusOutOffset] = status;
  if (status != 0u) {
    for (var i = 0u; i < ${n}u; i++) {
      coefficientsOut[coefficientsOutOffset + i] = nan;
      standardErrorsOut[standardErrorsOutOffset + i] = nan;
      tStatisticsOut[tStatisticsOutOffset + i] = nan;
    }
    for (var slot = 0u; slot < 22u; slot++) {
      summaryOut[summaryOutOffset + slot] = nan;
    }
    summaryOut[summaryOutOffset] = f32(rows);
    return;
  }
  let n = f32(rows);
  let freedom = n - ${n}.0;
  let residualSumSquares = wsGet(W_RESIDUAL_SUMS);
  let totalSumSquares = momentAt(K, K);
  var a: array<f32, ${e*e}>;
  for (var i = 0u; i < KK; i++) {
    a[i] = wsGet(W_FACTOR + i);
  }
  let sigmaSquared = residualSumSquares / freedom;
  var intercept = wsGet(W_MEANS + K);
  var u: array<f32, ${e}>;
  for (var i = 0u; i < K; i++) {
    intercept = getRoundedDifference(intercept, wsGet(W_SLOPES + i) * wsGet(W_MEANS + i), ws[wsOffset + W_RUNTIME_ZERO]);
    u[i] = wsGet(W_MEANS + i) / wsGet(W_SCALE + i);
  }
  intercept = intercept + wsGet(W_RESIDUAL_OFFSET);
  coefficientsOut[coefficientsOutOffset] = intercept;
  for (var i = 0u; i < K; i++) {
    coefficientsOut[coefficientsOutOffset + 1u + i] = wsGet(W_SLOPES + i);
  }
  // Intercept variance: sigma2 (1 / n + u' R^-1 u) with u = xbar / scale.
  choleskySolve_${e}(&a, &u);
  var quadratic = 0.0;
  for (var i = 0u; i < K; i++) {
    quadratic = quadratic + (wsGet(W_MEANS + i) / wsGet(W_SCALE + i)) * u[i];
  }
  standardErrorsOut[standardErrorsOutOffset] = sqrt(max(sigmaSquared * (1.0 / n + quadratic), 0.0));
  var maximumInflation = 0.0;
  for (var i = 0u; i < K; i++) {
    let scale = wsGet(W_SCALE + i);
    // Diagonal of the inverse correlation matrix is the variance inflation factor of column i.
    let correlationInverse = choleskyInverseDiagonal_${e}(&a, i);
    maximumInflation = max(maximumInflation, correlationInverse);
    let inverse = correlationInverse / (scale * scale);
    standardErrorsOut[standardErrorsOutOffset + 1u + i] = sqrt(max(sigmaSquared * inverse, 0.0));
  }
  for (var i = 0u; i < ${n}u; i++) {
    let standardError = standardErrorsOut[standardErrorsOutOffset + i];
    tStatisticsOut[tStatisticsOutOffset + i] = select(
      nan,
      coefficientsOut[coefficientsOutOffset + i] / standardError,
      standardError > 0.0
    );
  }
  let rSquared = select(nan, 1.0 - residualSumSquares / totalSumSquares, totalSumSquares > 0.0);
  let meanSquare = max(residualSumSquares / n, 1.17549435e-38);
  let logLikelihood = -0.5 * n * (1.8378770664093453 + log(meanSquare) + 1.0);
  var jarqueBera = nan;
  var jarqueBeraP = nan;
  var skewness = nan;
  var kurtosis = nan;
  let m2 = residualSumSquares / n;
  if (m2 > 0.0) {
    let m3 = wsGet(W_RESIDUAL_SUMS + 1u) / n;
    let m4 = wsGet(W_RESIDUAL_SUMS + 2u) / n;
    skewness = m3 / (m2 * sqrt(m2));
    kurtosis = m4 / (m2 * m2);
    let excess = kurtosis - 3.0;
    jarqueBera = n / 6.0 * (skewness * skewness + 0.25 * excess * excess);
    jarqueBeraP = exp(-0.5 * jarqueBera);
  }
  for (var slot = 0u; slot < 22u; slot++) {
    summaryOut[summaryOutOffset + slot] = nan;
  }
  summaryOut[summaryOutOffset + 0u] = n;
  summaryOut[summaryOutOffset + 1u] = rSquared;
  summaryOut[summaryOutOffset + 2u] = 1.0 - (1.0 - rSquared) * (n - 1.0) / freedom;
  summaryOut[summaryOutOffset + 3u] = sigmaSquared;
  summaryOut[summaryOutOffset + 4u] = logLikelihood;
  summaryOut[summaryOutOffset + 5u] = 2.0 * ${n}.0 - 2.0 * logLikelihood;
  summaryOut[summaryOutOffset + 6u] = ${n}.0 * log(n) - 2.0 * logLikelihood;
  summaryOut[summaryOutOffset + 7u] = jarqueBera;
  summaryOut[summaryOutOffset + 8u] = jarqueBeraP;
  summaryOut[summaryOutOffset + 11u] = residualSumSquares;
  summaryOut[summaryOutOffset + 12u] = totalSumSquares;
  summaryOut[summaryOutOffset + 13u] = skewness;
  summaryOut[summaryOutOffset + 14u] = kurtosis;
  summaryOut[summaryOutOffset + 15u] = wsGet(W_LAMBDA);

  // Conditioning. Belsley (spreg condition_index) of the raw [1, X] design, from the Cholesky
  // factor of the unpenalized predictor correlation matrix and the column means, never from the
  // raw Gram matrix (its smallest eigenvalue is below float32 resolution once the condition
  // number passes a few hundred). See getBelsleyConditionNumberWGSL.
  var unpenalized: array<f32, ${e*e}>;
  var alpha: array<f32, ${e}>;
  var beta: array<f32, ${e}>;
  for (var i = 0u; i < K; i++) {
    for (var j = 0u; j < K; j++) {
      unpenalized[i * K + j] = momentAt(i, j) / (wsGet(W_SCALE + i) * wsGet(W_SCALE + j));
    }
    let centered = momentAt(i, i);
    let meanI = wsGet(W_MEANS + i);
    let rawNorm = sqrt(n * meanI * meanI + centered);
    alpha[i] = sqrt(n) * meanI / rawNorm;
    beta[i] = sqrt(centered) / rawNorm;
  }
  var belsley = 3.0e38;
  if (choleskyFactor_${e}(&unpenalized)) {
    belsley = getBelsleyConditionNumber_${e}(&unpenalized, &alpha, &beta);
  }
  // The factored (optionally ridge-penalized) predictor correlation matrix.
  var correlation: array<f32, ${e*e}>;
  for (var i = 0u; i < K; i++) {
    for (var j = 0u; j < K; j++) {
      var entry = momentAt(i, j) / (wsGet(W_SCALE + i) * wsGet(W_SCALE + j));
      if (i == j) {
        entry = entry + wsGet(W_LAMBDA) / (wsGet(W_SCALE + i) * wsGet(W_SCALE + i));
      }
      correlation[i * K + j] = entry;
    }
  }
  let correlationRange = getSymmetricEigenvalueRange_${e}(&correlation);
  let correlationCondition = correlationRange.y / max(correlationRange.x, 1.0e-30);
  summaryOut[summaryOutOffset + 18u] =
    belsley;
  summaryOut[summaryOutOffset + 19u] =
    correlationCondition;
  summaryOut[summaryOutOffset + 20u] =
    maximumInflation;
  summaryOut[summaryOutOffset + 21u] =
    select(0.0, 1.0, correlationCondition > ${t(y)});`}function I(e){return`
const DEGREES_OF_FREEDOM: u32 = ${e}u;
const LOG_GAMMA: f32 = ${t(b(e))};

fn getChiSquareSurvival(value: f32) -> f32 {
  if (!(value > 0.0)) {
    return 1.0;
  }
  let a = f32(DEGREES_OF_FREEDOM) * 0.5;
  let x = value * 0.5;
  let logPrefactor = -x + a * log(x) - LOG_GAMMA;
  if (x < a + 1.0) {
    var term = 1.0 / a;
    var sum = term;
    for (var iteration = 1u; iteration <= 200u; iteration++) {
      term = term * x / (a + f32(iteration));
      sum = sum + term;
      if (term < sum * 1.0e-7) {
        break;
      }
    }
    return clamp(1.0 - sum * exp(logPrefactor), 0.0, 1.0);
  }
  let tiny = 1.0e-30;
  var b = x + 1.0 - a;
  var c = 1.0 / tiny;
  var d = 1.0 / b;
  var h = d;
  for (var iteration = 1u; iteration <= 200u; iteration++) {
    let fi = f32(iteration);
    let an = -fi * (fi - a);
    b = b + 2.0;
    d = an * d + b;
    if (abs(d) < tiny) {
      d = tiny;
    }
    c = b + an / c;
    if (abs(c) < tiny) {
      c = tiny;
    }
    d = 1.0 / d;
    let delta = d * c;
    h = h * delta;
    if (abs(delta - 1.0) < 1.0e-7) {
      break;
    }
  }
  return clamp(exp(logPrefactor) * h, 0.0, 1.0);
}
`}function L(e){return`let status = ws[wsOffset + W_STATUS];
  let nan = getNaN();
  var koenker = nan;
  var koenkerP = nan;
  var breusch = nan;
  var breuschP = nan;
  let totalSumSquares = momentAt(K, K);
  if (status == 0u && totalSumSquares > 0.0) {
    var explained = 0.0;
    for (var i = 0u; i < ${e}u; i++) {
      explained = explained + wsGet(W_SLOPES + i) * momentAt(i, K);
    }
    let rSquared = clamp(explained / totalSumSquares, 0.0, 1.0);
    let n = f32(ws[wsOffset + W_COUNT]);
    koenker = n * rSquared;
    koenkerP = getChiSquareSurvival(koenker);
    // Classic Breusch-Pagan: 0.5 R2 (sum e^4 / s2^2 - n) = 0.5 R2 n (kurtosis - 1), s2 = RSS / n.
    let residualSumSquares = bitcast<f32>(mainWs[mainWsOffset + W_RESIDUAL_SUMS]);
    let sumFourth = bitcast<f32>(mainWs[mainWsOffset + W_RESIDUAL_SUMS + 2u]);
    if (residualSumSquares > 0.0) {
      let kurtosis = sumFourth * n / (residualSumSquares * residualSumSquares);
      breusch = max(0.5 * rSquared * n * (kurtosis - 1.0), 0.0);
      breuschP = getChiSquareSurvival(breusch);
    }
  }
  summaryOut[summaryOutOffset + 9u] = koenker;
  summaryOut[summaryOutOffset + 10u] = koenkerP;
  summaryOut[summaryOutOffset + 16u] = breusch;
  summaryOut[summaryOutOffset + 17u] = breuschP;`}function R(e,t,n){return{name:e,view:t,type:n,access:`read`}}function z(e,t,n){return{name:e,view:t,type:n,access:`read_write`}}function B(e){return R(`ws`,e,`u32`)}function V(e){return z(`ws`,e,`u32`)}function H(e,t){return n(e,{operation:x,...t})}export{D as n,y as r,T as t};