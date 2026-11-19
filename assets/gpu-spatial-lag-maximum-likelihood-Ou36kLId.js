import{t as e}from"./regression-output-dispatches-KKVTWkHP.js";import{a as t,c as n,n as r,o as i,r as a}from"./regression-binding-plans-DG-hiroU.js";import{n as o,t as s}from"./regression-trace-layout-DEjyD3_7.js";import{a as c,h as l,i as u,l as d,r as ee,s as te,u as f}from"./spatial-likelihood-allocations-BGEHbqkB.js";import{n as ne,r as p,t as re}from"./gpu-ordinary-least-squares-B4HAVt2a.js";import{a as ie}from"./graph-chunk-dispatch-zBt8k2-o.js";import{r as ae}from"./kernel-C3tfQOn4.js";import{c as m}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as h,f as g,x as oe,y as _}from"./gpu-contributor-utils-BIwLQON5.js";import{n as se,r as ce,u as le}from"./wgsl-common-DfyXiYB0.js";import{t as ue}from"./addition-rounding-error-wgsl-DVwPAqKC.js";import{s as v}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";import{i as de,n as fe,t as pe}from"./chunked-weights-kernels-BYYFtXmd.js";import{n as me,t as he}from"./gpu-spatial-weights-traces-BQQpelz9.js";import{n as ge}from"./spatial-regression-solve-Vf-sQz9O.js";import{a as _e,i as ve,n as ye,o as be,r as xe}from"./spatial-maximum-likelihood-wgsl-Gex6F7yK.js";import{t as y}from"./pairwise-sum-wgsl-iErcC5Or.js";import{t as b}from"./rounded-difference-wgsl-DgXXQtNj.js";import{n as Se,r as x}from"./spatial-exact-log-determinant-CVAF5YqS.js";var Ce=`GPUSpatialLagMaximumLikelihood`,S=4096,we=64,C=64,Te=Math.log(2*Math.PI),w=.25,T=4;function E(e,t,n,r,i=`sum`){return y({count:r,name:i,term:`${e}[${e}Offset + ${t} + ${i}Item * ${n}]`})}var D=class{id;props;rowCount;maximumPower;tailExtrapolation;exactLogDeterminant;traceMode;tileRowCount;tileCount;traceTileRowCount;traceTileCount;constructor(n){n=i(n,[`id`,`weights`,`predictors`,`response`,`predictorCount`,`maximumPower`,`traceMode`,`tailExtrapolation`,`logDeterminant`,`eigenvalues`,`traceProbeCount`,`traceParameters`,`tileRowCount`,`output`],[`table`,`summary`,`residuals`,`status`,`powerTraces`,`powerSums`,`powerTraceStandardErrors`,`infinityNorm`],`spatial-lag-maximum-likelihood`);let r=n.id??`spatial-lag-maximum-likelihood`;this.id=r,this.props=n;let{predictorCount:a,output:o}=n;g(r,`predictorCount`,a,1,14);let c=n.maximumPower??128;g(r,`maximumPower`,c,2,256),this.maximumPower=c,this.tailExtrapolation=n.tailExtrapolation??!1,this.tailExtrapolation&&g(r,`maximumPower`,c,3,256);let l=v(r,n.weights);if(this.exactLogDeterminant=x(r,n,l),h(n.predictors,[`float32`],`${r} predictors`),h(n.response,[`float32`],`${r} response`),n.response.length!==l)throw Error(`${r} response length must equal the weights row count`);if(n.predictors.length!==l*a)throw Error(`${r} predictors length must equal the weights row count * predictorCount`);if(n.traceMode!==void 0&&n.traceMode!==`exact`&&n.traceMode!==`stochastic`)throw Error(`${r} traceMode must be 'exact' or 'stochastic'`);if(this.traceMode=n.traceMode??(l<=4096?`exact`:`stochastic`),this.traceMode===`exact`&&l>16384)throw Error(`${r} exact traces support at most 16384 rows; use traceMode 'stochastic'`);if(n.traceProbeCount!==void 0&&g(r,`traceProbeCount`,n.traceProbeCount),me(r,`traceParameters`,n.traceParameters),h(o.table,[`float32`],`${r} output.table`),o.table.length<(a+2)*4)throw Error(`${r} output.table must hold (predictorCount + 2) * 4 float32 values`);if(h(o.summary,[`float32`],`${r} output.summary`),o.summary.length<22)throw Error(`${r} output.summary must hold 22 float32 values`);if(h(o.status.code,[`uint32`],`${r} output.status.code`),o.status.code.length<1)throw Error(`${r} output.status.code must hold one uint32 row`);if(o.status.converged&&(h(o.status.converged,[`uint32`],`${r} output.status.converged`),o.status.converged.length<1))throw Error(`${r} output.status.converged must hold one uint32 row`);if(o.residuals&&(h(o.residuals,[`float32`],`${r} output.residuals`),o.residuals.length<l))throw Error(`${r} output.residuals must hold one row per input row`);if(o.powerTraces===void 0!=(o.powerSums===void 0)||o.powerTraces===void 0!=(o.infinityNorm===void 0))throw Error(`${r} output.powerTraces, output.powerSums and output.infinityNorm must be provided together`);for(let[e,t]of[[`powerTraces`,o.powerTraces],[`powerSums`,o.powerSums],[`infinityNorm`,o.infinityNorm],[`powerTraceStandardErrors`,o.powerTraceStandardErrors]])if(t&&(h(t,[`float32`],`${r} output.${e}`),t.length<(e===`infinityNorm`?1:c)))throw Error(`${r} output.${e} must hold maximumPower float32 values`);n.tileRowCount!==void 0&&g(r,`tileRowCount`,n.tileRowCount),this.rowCount=l,this.tileRowCount=n.tileRowCount??Math.max(we,Math.ceil(l/S)),this.tileCount=Math.ceil(l/this.tileRowCount);let u=s(r,n.weights,l);this.traceTileRowCount=u.tileRowCount,this.traceTileCount=u.tileCount,e(this.id,{...o,status:void 0,code:o.status.code,converged:o.status.converged},[[`table`,`summary`,`code`,`converged`],{read:[`infinityNorm`],write:[`powerTraces`,`powerTraceStandardErrors`]},{read:[...this.exactLogDeterminant?[]:[`powerTraces`],`infinityNorm`],write:[`table`,`summary`,`code`,`converged`]}]),_(r,[o.table,o.summary,o.status.code,o.status.converged,o.residuals,o.powerTraces,o.powerSums,o.infinityNorm,o.powerTraceStandardErrors],[n.predictors,n.response,...Object.values(t(n.weights))])}getCommandNodes(e){let{id:i,props:s,rowCount:p,tileRowCount:h,tileCount:g,maximumPower:_,tailExtrapolation:v}=this,me=v?o(_,`powerTraces`):``,y=this.exactLogDeterminant,x=Se(`eigenvalues`,p),S=y?x:`${l(_,`powerTraces`,`infinityNorm[infinityNormOffset]`,v)}${me}`,we=y?``:ve(_,`powerTraces`,`infinityNorm[infinityNormOffset]`,v),Te=y?x:`${ve(_,`powerTraces`,`infinityNorm[infinityNormOffset]`,v)}${me}`,{weights:w,output:T,predictorCount:D}=s;oe(i,e,[s.predictors,s.response,s.eigenvalues,...Object.values(t(w)),T.table,T.summary,T.status.code,T.status.converged,T.residuals,T.powerTraces,T.powerSums,T.infinityNorm,T.powerTraceStandardErrors]),ie(e,i,{predictors:s.predictors,response:s.response,eigenvalues:s.eigenvalues,traceParameters:s.traceParameters,...t(w),...T,status:void 0,code:T.status.code,converged:T.status.converged});let O=D+1,k=O+1,De=this.traceMode===`stochastic`,Oe=D*(D+1)/2,ke=D+1,Ae=D+2,je=D+3,Me=D+4,A=D+5,Ne=D+6,Pe=D+7,Fe=D+8,Ie=D+9,j=D+10,M=j+Oe,N=0,P={},F=(e,t=1)=>{P[e]=N,N+=t};F(`NORM`),F(`BOUND`),F(`PARAMETER`),F(`VALID`),F(`CANDIDATES`),F(`AT_BOUNDARY`),F(`SERIES_LIMITED`),F(`STATUS`),F(`BETA`,O),F(`MEANS`,D),F(`YBAR`),F(`CONDITIONING`,4),F(`TRACES`,3),F(`STANDARD_ERRORS`,O+1),F(`VARIANCE_DEFINED`),F(`WEIGHTED_MEAN`),F(`RUNTIME_ZERO`),F(`ROW_SUM`),F(`MOMENTS`,3),F(`PARAMETER_ZERO`);let Le=Object.entries(P).map(([e,t])=>`const W_${e}: u32 = ${t}u;`).join(`
`),Re=this.traceTileCount,ze=this.traceMode===`exact`?p:s.traceProbeCount??64;u(e.device,i,{...ne(p,D,h),...a({rowCount:p,edgeCount:r(w),maximumPower:_,tileCount:Re,mode:this.traceMode,probeCount:s.traceProbeCount}),...Object.fromEntries(Object.entries(ee(p,g,ze,1)).map(([e,t])=>[`information ${e}`,t])),"lag workspace":N,"lag row sums":2*p,"lag norm partials":2*pe(w,h).tileCount,"lag row scratch":p,"lag OLS outputs":Math.max(O,22),"lag mean partials":g*(D+1),"lag moment partials":g*3,"lag final partials":g*M,"lag final sums":M});let I=(t,n)=>ae(e,`${i}-${t}`,`float32`,n),Be=(t,n)=>ae(e,`${i}-${t}`,`uint32`,n),L=c(e,i,Ce),R=I(`workspace`,N),z=`const ROWS: u32 = ${p}u;
const K: u32 = ${D}u;
const P: u32 = ${O}u;
const TILE_ROWS: u32 = ${h}u;
const TILE_COUNT: u32 = ${g}u;
const MAXIMUM_POWER: u32 = ${_}u;
${Le}`,B=`${ce}
${le}`,V=d,H=e=>[f(`offsets`,e.offsets,`read`),f(`neighbors`,e.neighbors,`read`),V(`weights`,e.weights,`read`)],U=pe(w,h),W=U.tileCount,G=I(`lagged-response`,p),Ve=I(`weights-row-sums`,p*2),K=I(`partial-weights-norm`,W*2),q=[I(`ols-residuals-response`,p),I(`ols-residuals-lagged`,p)],J=[O,O].map((e,t)=>I(`ols-coefficients-${t}`,e)),Y=[0,1].map(e=>I(`ols-summary-${e}`,22)),He=[0,1].map(e=>Be(`ols-status-${e}`,1)),Ue=[0,1].map(t=>new re({id:`${i}-ols-${t===0?`response`:`lagged`}`,predictors:s.predictors,response:t===0?s.response:G,predictorCount:D,tileRowCount:h,output:{coefficients:J[t],standardErrors:I(`ols-standard-errors-${t}`,O),tStatistics:I(`ols-t-statistics-${t}`,O),summary:Y[t],status:{code:He[t]},residuals:q[t]}}).getCommandNodes(e)),X=T.powerTraces??I(`power-traces`,_),We=T.powerSums??I(`power-sums`,_),Z=T.infinityNorm??I(`traces-infinity-norm`,1),Ge=I(`partial-means`,g*(D+1)),Ke=I(`partial-moments`,g*3),qe=I(`partial-direct-moments`,g*2),Je=T.residuals??I(`residuals`,p),Ye=I(`reduced-prediction`,p),Xe=[I(`chain-a`,p),I(`chain-b`,p)],Ze=I(`weighted-prediction`,p),Qe=I(`partial-weighted-mean`,g),$e=I(`partial-final-sums`,g*M),et=I(`final-sums`,M),Q=[L(`means-tiles`,[V(`design`,s.predictors,`read`),V(`response`,s.response,`read`),V(`partial`,Ge,`read_write`)],g,z,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${D+1}>;
  for (var row = firstRow; row < lastRow; row++) {
    for (var column = 0u; column < K; column++) {
      sums[column] = sums[column] + design[designOffset + row * K + column];
    }
    sums[K] = sums[K] + response[responseOffset + row];
  }
  for (var slot = 0u; slot < ${D+1}u; slot++) {
    partial[partialOffset + index * ${D+1}u + slot] = sums[slot];
  }`,{workgroupSize:C}),L(`means-merge`,[V(`partial`,Ge,`read`),V(`workspace`,R,`read_write`)],D+1,z,`${E(`partial`,`index`,`${D+1}u`,`TILE_COUNT`)}
  workspace[workspaceOffset + W_MEANS + index] = sum / f32(ROWS);
  if (index == 0u) {
    // A zero the compiler cannot see: the runtime zero of getRoundedDifference.
    workspace[workspaceOffset + W_RUNTIME_ZERO] = 0.0;
  }`),...U.dispatches.map(e=>L(`lagged-response-tiles${e.idSuffix}`,[...H(e.weights),V(`response`,s.response,`read`),V(`workspace`,R,`read`),V(`lagged`,G,`read_write`),V(`rowSums`,Ve,`read_write`),V(`partialNorm`,K,`read_write`)],e.chunkTileCount,`${z}${e.tileConstantsWGSL?`\n${e.tileConstantsWGSL}`:``}
${b}
${ue}`,fe(e,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  let responseMean = workspace[workspaceOffset + W_YBAR];
  let runtimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  var largest = 0.0;
  var largestRowSum = -3.0e38;
  for (var row = firstRow; row < lastRow; row++) {
    var sum = 0.0;
    var absoluteSum = 0.0;
    var rowSum = 0.0;
    var rowSumError = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      sum += weight * getRoundedDifference(response[responseOffset + neighbor], responseMean, runtimeZero);
      absoluteSum += abs(weight);
      let nextRowSum = rowSum + weight;
      rowSumError += getAdditionRoundingError(rowSum, weight, nextRowSum);
      rowSum = nextRowSum;
    }
    lagged[laggedOffset + row] = sum;
    rowSums[rowSumsOffset + 2u * row] = rowSum;
    rowSums[rowSumsOffset + 2u * row + 1u] = rowSumError;
    largest = max(largest, absoluteSum);
    largestRowSum = max(largestRowSum, rowSum);
  }
  partialNorm[partialNormOffset + 2u * index] = largest;
  partialNorm[partialNormOffset + 2u * index + 1u] = largestRowSum;`),{workgroupSize:C})),L(`row-sum-merge`,[V(`partialNorm`,K,`read`),V(`workspace`,R,`read_write`)],1,z,`var largestRowSum = -3.0e38;
  for (var tile = 0u; tile < ${W}u; tile++) {
    largestRowSum = max(largestRowSum, partialNorm[partialNormOffset + 2u * tile + 1u]);
  }
  workspace[workspaceOffset + W_ROW_SUM] = largestRowSum;`),L(`lagged-response-shift`,[V(`rowSums`,Ve,`read`),V(`workspace`,R,`read`),V(`lagged`,G,`read_write`)],p,`${z}
${b}`,`let runtimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  let rowSumDeviation = getRoundedDifference(rowSums[rowSumsOffset + 2u * index], workspace[workspaceOffset + W_ROW_SUM], runtimeZero)
    + rowSums[rowSumsOffset + 2u * index + 1u];
  lagged[laggedOffset + index] = lagged[laggedOffset + index] + workspace[workspaceOffset + W_YBAR] * rowSumDeviation;`),...Ue[0],...Ue[1],...new he({id:`${i}-traces`,tileRowCount:this.traceTileRowCount,weights:w,maximumPower:_,mode:this.traceMode,probeCount:s.traceProbeCount,parameters:s.traceParameters,output:{powerTraces:X,powerSums:We,infinityNorm:Z,powerTraceStandardErrors:T.powerTraceStandardErrors}}).getCommandNodes(e),L(`moments-tiles`,[V(`residual0`,q[0],`read`),V(`residual1`,q[1],`read`),V(`partial`,Ke,`read_write`)],g,z,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sumA = 0.0;
  var sumB = 0.0;
  var sumC = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    let e0 = residual0[residual0Offset + row];
    let e1 = residual1[residual1Offset + row];
    sumA += e0 * e0;
    sumB += e0 * e1;
    sumC += e1 * e1;
  }
  partial[partialOffset + index * 3u] = sumA;
  partial[partialOffset + index * 3u + 1u] = sumB;
  partial[partialOffset + index * 3u + 2u] = sumC;`,{workgroupSize:C}),L(`moments-merge`,[V(`partialMoments`,Ke,`read`),V(`workspace`,R,`read_write`)],1,z,`${E(`partialMoments`,`0u`,`3u`,`TILE_COUNT`,`momentA`)}
  ${E(`partialMoments`,`1u`,`3u`,`TILE_COUNT`,`momentB`)}
  ${E(`partialMoments`,`2u`,`3u`,`TILE_COUNT`,`momentC`)}
  workspace[workspaceOffset + W_MOMENTS] = momentA;
  workspace[workspaceOffset + W_MOMENTS + 1u] = momentB;
  workspace[workspaceOffset + W_MOMENTS + 2u] = momentC;
  // e1'e1 = 0 (the lag column lies in the span of [1, X]): u'u does not depend on rho, and
  // rho0 is 0 by convention.
  workspace[workspaceOffset + W_PARAMETER_ZERO] = select(momentB / momentC, 0.0, momentC == 0.0);`),L(`direct-moments-tiles`,[V(`residual0`,q[0],`read`),V(`residual1`,q[1],`read`),V(`workspace`,R,`read`),V(`partial`,qe,`read_write`)],g,`${z}
${b}`,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  let parameterZero = workspace[workspaceOffset + W_PARAMETER_ZERO];
  let runtimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  var sumSquares = 0.0;
  var sumProducts = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    let e1 = residual1[residual1Offset + row];
    // Rounded as written, so the tile sums stay sums of the small differences.
    let residual = getRoundedDifference(residual0[residual0Offset + row], parameterZero * e1, runtimeZero);
    sumSquares += residual * residual;
    sumProducts += residual * e1;
  }
  partial[partialOffset + index * 2u] = sumSquares;
  partial[partialOffset + index * 2u + 1u] = sumProducts;`,{workgroupSize:C}),L(`rho-search`,[V(`partialNorm`,K,`read`),V(`partialDirect`,qe,`read`),y?d(`eigenvalues`,s.eigenvalues,`read`):V(`powerTraces`,X,`read`),V(`infinityNorm`,Z,`read`),V(`workspace`,R,`read_write`)],1,`${z}
${B}
${b}
const ROWS_F: f32 = ${p}.0;
${S}
${we}
// u(rho)'u(rho) = residualSquares + laggedSquares (rho - rho*)^2, a sum of two non-negative
// terms, with rho* = parameterZero + parameterCorrection its minimizer.
var<private> residualSquares: f32;
var<private> laggedSquares: f32;
var<private> parameterZero: f32;
var<private> parameterCorrection: f32;
var<private> searchRuntimeZero: u32;

fn getParameterOffset(rho: f32) -> f32 {
  return getRoundedDifference(rho, parameterZero, searchRuntimeZero) - parameterCorrection;
}

fn getSumOfSquares(rho: f32) -> f32 {
  let offset = getParameterOffset(rho);
  return residualSquares + laggedSquares * offset * offset;
}

fn getObjective(rho: f32) -> f32 {
  return -0.5 * ROWS_F * log(getSumOfSquares(rho) / ROWS_F) + getSpatialLogDeterminant(rho);
}

fn getDerivative(rho: f32) -> f32 {
  return -ROWS_F * laggedSquares * getParameterOffset(rho) / getSumOfSquares(rho) + getLogDeterminantDerivative(rho);
}
${ye()}`,`var largest = 0.0;
  for (var tile = 0u; tile < ${W}u; tile++) {
    largest = max(largest, partialNorm[partialNormOffset + 2u * tile]);
  }
  let validNorm = largest > 0.0 && isFiniteValue(largest);
  let seriesLimit = ${m(v||y?_e:be(_))};
  let stationarity = ${m(_e)};
  let bound = select(getNaN(), min(stationarity, seriesLimit) / largest, validNorm);
  workspace[workspaceOffset + W_NORM] = select(getNaN(), largest, validNorm);
  workspace[workspaceOffset + W_BOUND] = bound;
  workspace[workspaceOffset + W_SERIES_LIMITED] = select(0.0, 1.0, seriesLimit < stationarity);
  workspace[workspaceOffset + W_PARAMETER] = getNaN();
  workspace[workspaceOffset + W_VALID] = 0.0;
  workspace[workspaceOffset + W_CANDIDATES] = 0.0;
  workspace[workspaceOffset + W_AT_BOUNDARY] = 0.0;
  ${E(`partialDirect`,`0u`,`2u`,`TILE_COUNT`,`directSquares`)}
  ${E(`partialDirect`,`1u`,`2u`,`TILE_COUNT`,`directProducts`)}
  let momentA = workspace[workspaceOffset + W_MOMENTS];
  laggedSquares = workspace[workspaceOffset + W_MOMENTS + 2u];
  parameterZero = workspace[workspaceOffset + W_PARAMETER_ZERO];
  // e1'e1 = 0: a constant sum of squares e0'e0 (directSquares at rho0 = 0), no correction.
  parameterCorrection = select(directProducts / laggedSquares, 0.0, laggedSquares == 0.0);
  residualSquares = directSquares - directProducts * parameterCorrection;
  searchRuntimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  // The likelihood needs u'u > 0 on the whole search interval. S > 0 gives that for every rho.
  // S = 0 (the response residual is a multiple of the lag residual) gives it when the minimizer
  // rho* lies outside the interval, where u'u = e1'e1 (rho - rho*)^2 is positive throughout; S
  // is then a difference that rounds to zero either way, so a rounding-sized negative value
  // (below 1e-6 of r'r) is taken as zero. S = 0 with rho* inside the interval is a pole of the
  // likelihood: no candidate, status 3.
  let minimizer = parameterZero + parameterCorrection;
  let minimizerOutside = laggedSquares > 0.0 && abs(minimizer) > bound;
  let positiveSquares = residualSquares > 0.0 || (minimizerOutside && residualSquares >= -1.0e-6 * directSquares);
  residualSquares = max(residualSquares, 0.0);
  if (validNorm && isFiniteValue(momentA) && isFiniteValue(parameterZero) && isFiniteValue(parameterCorrection)
      && isFiniteValue(residualSquares) && isFiniteValue(laggedSquares) && momentA > 0.0 && laggedSquares >= 0.0 && positiveSquares) {
    let result = searchParameter(bound);
    workspace[workspaceOffset + W_PARAMETER] = result.parameter;
    workspace[workspaceOffset + W_VALID] = select(0.0, 1.0, isFiniteValue(result.parameter));
    workspace[workspaceOffset + W_CANDIDATES] = f32(result.localMaximumCount);
    workspace[workspaceOffset + W_AT_BOUNDARY] = select(0.0, 1.0, result.atBoundary);
  }`),L(`solve-coefficients`,[V(`olsCoefficients0`,J[0],`read`),V(`olsCoefficients1`,J[1],`read`),V(`olsSummary0`,Y[0],`read`),V(`olsSummary1`,Y[1],`read`),f(`olsStatus0`,He[0],`read`),f(`olsStatus1`,He[1],`read`),V(`workspace`,R,`read_write`)],1,`${z}
${B}`,`let rho = workspace[workspaceOffset + W_PARAMETER];
  var code = 0u;
  if (!(f32(ROWS) > f32(K + 2u))) {
    code = 2u;
  } else if (olsSummary0[olsSummary0Offset] != f32(ROWS) || olsSummary1[olsSummary1Offset] != f32(ROWS)) {
    // A non-finite input: the fits skipped a row. Tested before the fit statuses because a
    // non-finite response mean or row sum makes the whole lag column non-finite.
    code = 3u;
  } else if (olsStatus0[olsStatus0Offset] != 0u) {
    code = select(1u, 2u, olsStatus0[olsStatus0Offset] == 2u);
  } else if (olsStatus1[olsStatus1Offset] != 0u) {
    code = select(1u, 2u, olsStatus1[olsStatus1Offset] == 2u);
  } else if (workspace[workspaceOffset + W_VALID] != 1.0 || !isFiniteValue(rho)) {
    code = 3u;
  }
  workspace[workspaceOffset + W_STATUS] = f32(code);
  // The second fit is of the lag column v = W y - ybar s (see lagged-response-shift), so its
  // intercept misses the constant ybar s of the lag.
  let lagShift = workspace[workspaceOffset + W_YBAR] * workspace[workspaceOffset + W_ROW_SUM];
  for (var column = 0u; column < P; column++) {
    var laggedCoefficient = olsCoefficients1[olsCoefficients1Offset + column];
    if (column == 0u) {
      laggedCoefficient += lagShift;
    }
    workspace[workspaceOffset + W_BETA + column] = olsCoefficients0[olsCoefficients0Offset + column] - rho * laggedCoefficient;
  }
  for (var slot = 0u; slot < 4u; slot++) {
    workspace[workspaceOffset + W_CONDITIONING + slot] = olsSummary0[olsSummary0Offset + 18u + slot];
  }`),L(`residuals`,[V(`response`,s.response,`read`),V(`lagged`,G,`read`),V(`residual0`,q[0],`read`),V(`residual1`,q[1],`read`),V(`residualOut`,Je,`read_write`),V(`reduced`,Ye,`read_write`),V(`workspace`,R,`read`)],p,z,`let rho = workspace[workspaceOffset + W_PARAMETER];
  residualOut[residualOutOffset + index] = residual0[residual0Offset + index] - rho * residual1[residual1Offset + index];
  // X b = (y - e0) - rho (W y - e1): the OLS fitted values of y and W y, with W y the lag column
  // plus the constant it was formed about.
  let lagShift = workspace[workspaceOffset + W_YBAR] * workspace[workspaceOffset + W_ROW_SUM];
  reduced[reducedOffset + index] = (response[responseOffset + index] - residual0[residual0Offset + index])
    - rho * ((lagged[laggedOffset + index] + lagShift) - residual1[residual1Offset + index]);`)];for(let e=1;e<_;e++){let t=e===1;Q.push(...U.dispatches.map(n=>L(`inverse-series-${e}${n.idSuffix}`,[...H(n.weights),V(`reduced`,Ye,`read`),...t?[]:[V(`vectorIn`,Xe[e%2],`read`)],V(`vectorOut`,Xe[(e-1)%2],`read_write`),V(`workspace`,R,`read`)],n.rowCount,`${z}${n.constantsWGSL?`\n${n.constantsWGSL}`:``}`,de(n,`let rho = workspace[workspaceOffset + W_PARAMETER];
  var sum = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS) {
      continue;
    }
    sum += weights[weightsOffset + slot] * ${t?`reduced[reducedOffset + neighbor]`:`vectorIn[vectorInOffset + neighbor]`};
  }
  vectorOut[vectorOutOffset + index] = reduced[reducedOffset + index] + rho * sum;`))))}let tt=Xe[(_-2)%2];Q.push(...U.dispatches.map(e=>L(`weighted-prediction${e.idSuffix}`,[...H(e.weights),V(`vectorIn`,tt,`read`),V(`vectorOut`,Ze,`read_write`)],e.rowCount,`${z}${e.constantsWGSL?`\n${e.constantsWGSL}`:``}`,de(e,`var sum = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS) {
      continue;
    }
    sum += weights[weightsOffset + slot] * vectorIn[vectorInOffset + neighbor];
  }
  vectorOut[vectorOutOffset + index] = sum;`))),L(`weighted-mean-tiles`,[V(`weighted`,Ze,`read`),V(`partial`,Qe,`read_write`)],g,z,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sum = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    sum += weighted[weightedOffset + row];
  }
  partial[partialOffset + index] = sum;`,{workgroupSize:C}),L(`weighted-mean-merge`,[V(`partial`,Qe,`read`),V(`workspace`,R,`read_write`)],1,z,`${E(`partial`,`0u`,`1u`,`TILE_COUNT`)}
  workspace[workspaceOffset + W_WEIGHTED_MEAN] = sum / f32(ROWS);
  workspace[workspaceOffset + W_RUNTIME_ZERO] = 0.0;`),L(`final-moments-tiles`,[V(`design`,s.predictors,`read`),V(`response`,s.response,`read`),V(`residual`,Je,`read`),V(`residual1`,q[1],`read`),V(`prediction`,tt,`read`),V(`weighted`,Ze,`read`),V(`workspace`,R,`read`),V(`partial`,$e,`read_write`)],g,`${z}
${b}`,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  let ybar = workspace[workspaceOffset + W_YBAR];
  let weightedMean = workspace[workspaceOffset + W_WEIGHTED_MEAN];
  let runtimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  var sums: array<f32, ${M}>;
  var centered: array<f32, ${D}>;
  for (var row = firstRow; row < lastRow; row++) {
    // Each difference from a mean is rounded as written (getRoundedDifference), so the tile sums
    // below cannot be regrouped into sums of the raw values less a multiple of the mean.
    let y = getRoundedDifference(response[responseOffset + row], ybar, runtimeZero);
    let u = residual[residualOffset + row];
    let e = getRoundedDifference(prediction[predictionOffset + row], ybar, runtimeZero);
    // W A X b about its mean, rounded as written so the sums below stay sums of deviations.
    let wp = getRoundedDifference(weighted[weightedOffset + row], weightedMean, runtimeZero);
    sums[0] += wp;
    sums[${ke}] += wp * wp;
    sums[${Ae}] += y * y;
    sums[${je}] += y * u;
    sums[${Me}] += u;
    sums[${A}] += u * u;
    sums[${Ne}] += y * e;
    sums[${Pe}] += e * e;
    sums[${Fe}] += e;
    sums[${Ie}] += u * residual1[residual1Offset + row];
    for (var column = 0u; column < K; column++) {
      centered[column] = getRoundedDifference(design[designOffset + row * K + column], workspace[workspaceOffset + W_MEANS + column], runtimeZero);
      sums[1u + column] += centered[column] * wp;
    }
    for (var first = 0u; first < K; first++) {
      for (var second = 0u; second <= first; second++) {
        sums[${j}u + first * (first + 1u) / 2u + second] += centered[first] * centered[second];
      }
    }
  }
  for (var slot = 0u; slot < ${M}u; slot++) {
    partial[partialOffset + index * ${M}u + slot] = sums[slot];
  }`,{workgroupSize:C}),L(`final-merge`,[V(`partial`,$e,`read`),V(`finalSums`,et,`read_write`)],M,z,`${E(`partial`,`index`,`${M}u`,`TILE_COUNT`)}
  finalSums[finalSumsOffset + index] = sum;`));let $=te(e,{id:`${i}-tr3`,operation:Ce,weights:w,rowCount:p,maximumPower:_,mode:this.traceMode,probeCount:s.traceProbeCount??64,seedParameters:s.traceParameters,tileRowCount:h,tileCount:g,workspace:R,parameterSlot:P.PARAMETER});Q.push(...$.nodes);let nt=De?`1.0 / f32(${$.probeTotal}u)`:`1.0`;return Q.push(L(`variance`,[V(`finalSums`,et,`read`),V(`dots`,$.dots,`read`),y?d(`eigenvalues`,s.eigenvalues,`read`):V(`powerTraces`,X,`read`),V(`infinityNorm`,Z,`read`),V(`workspace`,R,`read_write`)],1,`${z}
${B}
${Te}
${ge(k)}`,`let n = f32(ROWS);
  let rho = workspace[workspaceOffset + W_PARAMETER];
  let sigma2 = finalSums[finalSumsOffset + ${A}u] / n;
  let nan = getNaN();
  for (var slot = 0u; slot < P + 1u; slot++) {
    workspace[workspaceOffset + W_STANDARD_ERRORS + slot] = nan;
  }
  workspace[workspaceOffset + W_VARIANCE_DEFINED] = 0.0;
  let traceOne = getTraceOne(rho);
  let traceTwo = getTraceTwo(rho);
  ${$.dotsSumWGSL}
  let traceThree = sum * ${nt};
  workspace[workspaceOffset + W_TRACES] = traceOne;
  workspace[workspaceOffset + W_TRACES + 1u] = traceTwo;
  workspace[workspaceOffset + W_TRACES + 2u] = traceThree;
  if (!(sigma2 > 0.0) || !isFiniteValue(sigma2) || !isFiniteValue(traceThree) || !isFiniteValue(rho)) {
    return;
  }
  // Information matrix on [intercept (centered), b (K), rho]; the sigma2 row is eliminated. The
  // rho column is W A X b about its mean (the sums of final-moments-tiles), which leaves the
  // variances of b and rho unchanged and moves the intercept, mapped back below.
  var a: array<f32, ${k*k}>;
  a[0] = n / sigma2;
  for (var first = 0u; first < K; first++) {
    for (var second = 0u; second <= first; second++) {
      let value = finalSums[finalSumsOffset + ${j}u + first * (first + 1u) / 2u + second] / sigma2;
      a[(1u + first) * ${k}u + 1u + second] = value;
      a[(1u + second) * ${k}u + 1u + first] = value;
    }
  }
  for (var column = 0u; column < P; column++) {
    let value = finalSums[finalSumsOffset + 0u + column] / sigma2;
    a[column * ${k}u + P] = value;
    a[P * ${k}u + column] = value;
  }
  a[P * ${k}u + P] = traceTwo + traceThree + finalSums[finalSumsOffset + ${ke}u] / sigma2 - 2.0 * traceOne * traceOne / n;
  var scale: array<f32, ${k}>;
  for (var column = 0u; column < ${k}u; column++) {
    let diagonal = a[column * ${k}u + column];
    if (!(diagonal > 0.0) || !isFiniteValue(diagonal)) {
      return;
    }
    scale[column] = sqrt(diagonal);
  }
  for (var row = 0u; row < ${k}u; row++) {
    for (var column = 0u; column < ${k}u; column++) {
      a[row * ${k}u + column] = a[row * ${k}u + column] / (scale[row] * scale[column]);
    }
  }
  if (!choleskyFactor_${k}(&a)) {
    return;
  }
  // Covariance V = D^-1 (scaled inverse) D^-1, one unit solve per column.
  var covariance: array<f32, ${k*k}>;
  for (var column = 0u; column < ${k}u; column++) {
    var unit: array<f32, ${k}>;
    unit[column] = 1.0;
    choleskySolve_${k}(&a, &unit);
    for (var row = 0u; row < ${k}u; row++) {
      covariance[row * ${k}u + column] = unit[row] / (scale[row] * scale[column]);
    }
  }
  // Intercept of the uncentered model: b0 = a0 - m'b - mean(W A X b) rho, with a0 the constant
  // of the model on the centered predictors and the centered W A X b column.
  var shift: array<f32, ${k}>;
  for (var column = 0u; column < K; column++) {
    shift[1u + column] = workspace[workspaceOffset + W_MEANS + column];
  }
  shift[P] = workspace[workspaceOffset + W_WEIGHTED_MEAN];
  var interceptVariance = covariance[0];
  for (var first = 1u; first < ${k}u; first++) {
    interceptVariance -= 2.0 * shift[first] * covariance[first];
    for (var second = 1u; second < ${k}u; second++) {
      interceptVariance += shift[first] * shift[second] * covariance[first * ${k}u + second];
    }
  }
  workspace[workspaceOffset + W_STANDARD_ERRORS] = sqrt(interceptVariance);
  for (var column = 1u; column < P; column++) {
    workspace[workspaceOffset + W_STANDARD_ERRORS + column] = sqrt(covariance[column * ${k}u + column]);
  }
  workspace[workspaceOffset + W_STANDARD_ERRORS + P] = sqrt(covariance[P * ${k}u + P]);
  workspace[workspaceOffset + W_VARIANCE_DEFINED] = 1.0;`),L(`finish`,[V(`finalSums`,et,`read`),y?d(`eigenvalues`,s.eigenvalues,`read`):V(`powerTraces`,X,`read`),V(`infinityNorm`,Z,`read`),V(`workspace`,R,`read`),V(`tableOut`,T.table,`read_write`),V(`summaryOut`,T.summary,`read_write`),f(`statusOut`,T.status.code,`read_write`),...T.status.converged?[f(`convergedOut`,T.status.converged,`read_write`)]:[]],1,`${z}
${B}
${se}
${S}
${xe()}
${Ee({predictorCount:D,maximumPower:_,stochastic:De,sumResponseSquares:Ae,sumResponseResidual:je,sumResidual:Me,sumResidualSquares:A,sumResponsePrediction:Ne,sumPredictionSquares:Pe,sumPrediction:Fe,sumResidualLagged:Ie,hasConverged:T.status.converged!==void 0,exact:y})}`,`finish();`)),[...Q,...n(e,i,s.weights,s.output)]}};function Ee(e){let{predictorCount:t}=e,n=t+2,r=e=>`summaryOut[summaryOutOffset + ${e}u]`,i=e=>`finalSums[finalSumsOffset + ${e}u]`;return`
fn fail(status: u32) {
  let nan = getNaN();
  for (var slot = 0u; slot < ${n*4}u; slot++) {
    tableOut[tableOutOffset + slot] = nan;
  }
  for (var slot = 0u; slot < 22u; slot++) {
    summaryOut[summaryOutOffset + slot] = nan;
  }
  ${r(0)} = f32(ROWS);
  statusOut[statusOutOffset] = status;
${e.hasConverged?`  convergedOut[convergedOutOffset] = 0u;`:``}
}

fn finish() {
  let code = u32(workspace[workspaceOffset + W_STATUS]);
  if (code != 0u) {
    fail(code);
    return;
  }
  let n = f32(ROWS);
  let rho = workspace[workspaceOffset + W_PARAMETER];
  let sigma2 = ${i(e.sumResidualSquares)} / n;
  if (!(sigma2 > 0.0) || !isFiniteValue(sigma2)) {
    fail(3u);
    return;
  }
  // The first-order condition of an interior estimate, from the published residuals: the
  // derivative of the concentrated log-likelihood at rho against the square root of its
  // curvature (see STATIONARITY_TOLERANCE). u'u and u'e1 are per-row sums of the published
  // residuals, not the sums the search used; the traces are those of the variance kernel at
  // rho. Only ratios of the sums are formed (u'e1 / u'u, e1'e1 / u'u), so the test does not
  // depend on the scale of the response: the products of two sums overflowed float32 for a
  // response of 1e10. The allowance for the float32 spacing of rho is the change of the
  // derivative over STATIONARITY_PARAMETER_STEPS steps of rho, curvature times that distance.
  if (workspace[workspaceOffset + W_AT_BOUNDARY] != 1.0) {
    let residualSquares = sigma2 * n;
    let laggedRatio = ${i(e.sumResidualLagged)} / residualSquares;
    let score = n * laggedRatio - workspace[workspaceOffset + W_TRACES];
    let curvature = n * (workspace[workspaceOffset + W_MOMENTS + 2u] / residualSquares - 2.0 * laggedRatio * laggedRatio)
      + workspace[workspaceOffset + W_TRACES + 1u];
    // One float32 step of rho, from its bit pattern.
    let parameterStep = bitcast<f32>(bitcast<u32>(abs(rho)) + 1u) - abs(rho);
    let allowance = sqrt(${m(w)} * curvature)
      + curvature * ${m(T)} * parameterStep;
    if (!(curvature > 0.0) || !(abs(score) <= allowance)) {
      fail(3u);
      return;
    }
  }
  let defined = workspace[workspaceOffset + W_VARIANCE_DEFINED] == 1.0;
  for (var row = 0u; row < P; row++) {
    writeTableRow(row, workspace[workspaceOffset + W_BETA + row], workspace[workspaceOffset + W_STANDARD_ERRORS + row]);
  }
  writeTableRow(P, rho, workspace[workspaceOffset + W_STANDARD_ERRORS + P]);
  let logDeterminant = getSpatialLogDeterminant(rho);
  let logLikelihood = -0.5 * n * log(sigma2) + logDeterminant - 0.5 * n * (${m(Te)} + 1.0);
  let parameterCount = f32(P + 1u);
  // Squared correlations of y with y - u and with (I - rho W)^-1 X b.
  let responseSquares = ${i(e.sumResponseSquares)};
  let responseResidual = ${i(e.sumResponseResidual)};
  let residualTotal = ${i(e.sumResidual)};
  let predictionCovariance = responseSquares - responseResidual;
  let predictionVariance = responseSquares - 2.0 * responseResidual + sigma2 * n - residualTotal * residualTotal / n;
  let spatialCovariance = ${i(e.sumResponsePrediction)};
  let spatialVariance = ${i(e.sumPredictionSquares)} - ${i(e.sumPrediction)} * ${i(e.sumPrediction)} / n;
  let conditionNumber = workspace[workspaceOffset + W_CONDITIONING];
  let correlationCondition = workspace[workspaceOffset + W_CONDITIONING + 1u];
  let bound = workspace[workspaceOffset + W_BOUND];
  let norm = workspace[workspaceOffset + W_NORM];
  let scaled = abs(rho) * norm;
  let atBoundary = workspace[workspaceOffset + W_AT_BOUNDARY] == 1.0;
  ${r(0)} = n;
  ${r(1)} = rho;
  ${r(2)} = sigma2;
  ${r(3)} = logLikelihood;
  ${r(4)} = 2.0 * parameterCount - 2.0 * logLikelihood;
  ${r(5)} = parameterCount * log(n) - 2.0 * logLikelihood;
  // Ratios first: the product of two sums of squares overflows float32 for a response of 1e10.
  ${r(6)} = (predictionCovariance / responseSquares) * (predictionCovariance / predictionVariance);
  ${r(7)} = (spatialCovariance / responseSquares) * (spatialCovariance / spatialVariance);
  ${r(8)} = sigma2 * n;
  ${r(9)} = conditionNumber;
  ${r(10)} = correlationCondition;
  ${r(11)} = workspace[workspaceOffset + W_CONDITIONING + 2u];
  ${r(12)} = select(0.0, 1.0, correlationCondition > ${m(p)});
  ${r(13)} = bound;
  ${r(14)} = ${e.stochastic?`1.0`:`0.0`};
  ${r(15)} = select(0.0, 1.0, !atBoundary);
  ${r(16)} = select(0.0, 1.0, atBoundary);
  ${r(17)} = workspace[workspaceOffset + W_SERIES_LIMITED];
  ${r(18)} = select(
    1.0e38,
    n * pow(scaled, ${e.maximumPower+1}.0) / (${e.maximumPower+1}.0 * (1.0 - scaled)),
    scaled < 1.0
  );
  ${e.exact?`${r(18)} = 0.0;`:``}
  ${r(19)} = select(0.0, 1.0, defined);
  ${r(20)} = workspace[workspaceOffset + W_CANDIDATES];
  ${r(21)} = logDeterminant;
  statusOut[statusOutOffset] = 0u;
${e.hasConverged?`  convergedOut[convergedOutOffset] = select(0u, 1u, !atBoundary);`:``}
}
`}export{D as t};