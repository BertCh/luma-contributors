import{r as e}from"./kernel-C3tfQOn4.js";import{c as t,o as n}from"./regression-binding-plans-DG-hiroU.js";import{n as r,t as i}from"./regression-trace-layout-DEjyD3_7.js";import{c as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as o,f as s,x as c,y as l}from"./gpu-contributor-utils-BIwLQON5.js";import{n as u,r as d,u as ee}from"./wgsl-common-DfyXiYB0.js";import{i as f,s as p}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";import{n as te,t as m}from"./chunked-weights-kernels-BYYFtXmd.js";import{r as h}from"./gpu-ordinary-least-squares-B4HAVt2a.js";import{n as g,t as ne}from"./gpu-spatial-weights-traces-BQQpelz9.js";import{a as re,h as ie,i as ae,l as _,s as oe,t as se,u as v}from"./spatial-likelihood-allocations-BGEHbqkB.js";import{a as ce,i as le,n as ue,o as de,r as fe,t as y}from"./spatial-maximum-likelihood-wgsl-Gex6F7yK.js";import{a as b,i as pe,n as x,t as me}from"./spatial-regression-solve-Vf-sQz9O.js";import{t as he}from"./rounded-difference-wgsl-DgXXQtNj.js";import{n as ge,r as S}from"./spatial-exact-log-determinant-CVAF5YqS.js";import{a as _e}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as C}from"./regression-output-dispatches-KKVTWkHP.js";var w=`GPUSpatialErrorMaximumLikelihood`,ve=4096,T=64,ye=64,E=Math.log(2*Math.PI),D=4,O=class{id;props;rowCount;maximumPower;tailExtrapolation;exactLogDeterminant;traceMode;tileRowCount;tileCount;traceTileRowCount;traceTileCount;constructor(e){e=n(e,[`id`,`weights`,`predictors`,`response`,`predictorCount`,`maximumPower`,`traceMode`,`tailExtrapolation`,`logDeterminant`,`eigenvalues`,`traceProbeCount`,`traceParameters`,`tileRowCount`,`output`],[`table`,`summary`,`residuals`,`status`,`powerTraceStandardErrors`],`spatial-error-maximum-likelihood`);let t=e.id??`spatial-error-maximum-likelihood`;this.id=t,this.props=e;let{predictorCount:r,output:a}=e;s(t,`predictorCount`,r,1,15);let c=e.maximumPower??128;s(t,`maximumPower`,c,2,256),this.maximumPower=c,this.tailExtrapolation=e.tailExtrapolation??!1,this.tailExtrapolation&&s(t,`maximumPower`,c,3,256);let u=p(t,e.weights);if(this.exactLogDeterminant=S(t,e,u),o(e.predictors,[`float32`],`${t} predictors`),o(e.response,[`float32`],`${t} response`),e.response.length!==u)throw Error(`${t} response length must equal the weights row count`);if(e.predictors.length!==u*r)throw Error(`${t} predictors length must equal the weights row count * predictorCount`);if(e.traceMode!==void 0&&e.traceMode!==`exact`&&e.traceMode!==`stochastic`)throw Error(`${t} traceMode must be 'exact' or 'stochastic'`);if(this.traceMode=e.traceMode??(u<=4096?`exact`:`stochastic`),this.traceMode===`exact`&&u>16384)throw Error(`${t} exact traces support at most 16384 rows; use traceMode 'stochastic'`);if(e.traceProbeCount!==void 0&&s(t,`traceProbeCount`,e.traceProbeCount),g(t,`traceParameters`,e.traceParameters),o(a.table,[`float32`],`${t} output.table`),a.table.length<(r+2)*4)throw Error(`${t} output.table must hold (predictorCount + 2) * 4 float32 values`);if(o(a.summary,[`float32`],`${t} output.summary`),a.summary.length<22)throw Error(`${t} output.summary must hold 22 float32 values`);if(o(a.status.code,[`uint32`],`${t} output.status.code`),a.status.code.length<1)throw Error(`${t} output.status.code must hold one uint32 row`);if(a.status.converged&&(o(a.status.converged,[`uint32`],`${t} output.status.converged`),a.status.converged.length<1))throw Error(`${t} output.status.converged must hold one uint32 row`);if(a.residuals&&(o(a.residuals,[`float32`],`${t} output.residuals`),a.residuals.length<u))throw Error(`${t} output.residuals must hold one row per input row`);e.tileRowCount!==void 0&&s(t,`tileRowCount`,e.tileRowCount),this.rowCount=u,this.tileRowCount=e.tileRowCount??Math.max(T,Math.ceil(u/ve)),this.tileCount=m(e.weights,this.tileRowCount).tileCount;let d=i(t,e.weights,u);this.traceTileRowCount=d.tileRowCount,this.traceTileCount=d.tileCount,C(t,{table:a.table,summary:a.summary,code:a.status.code,converged:a.status.converged,residuals:a.residuals,powerTraceStandardErrors:a.powerTraceStandardErrors},[[`table`,`summary`,`code`,`converged`]]),l(t,[a.table,a.summary,a.status.code,a.status.converged,a.residuals,a.powerTraceStandardErrors],[e.predictors,e.response,e.eigenvalues,e.traceParameters,...f(k(e.weights))])}getCommandNodes(n){let{id:i,props:o,rowCount:s,tileRowCount:l,tileCount:p,maximumPower:h,tailExtrapolation:g}=this;_e(n,i,{predictors:o.predictors,response:o.response,eigenvalues:o.eigenvalues,traceParameters:o.traceParameters,...Object.fromEntries(f(k(o.weights)).map((e,t)=>[`weights ${t}`,e])),table:o.output.table,summary:o.output.summary,residuals:o.output.residuals,code:o.output.status.code,converged:o.output.status.converged,powerTraceStandardErrors:o.output.powerTraceStandardErrors});let S=this.traceTileCount,C=m(o.weights,l),ve=Math.max(...C.dispatches.map(e=>e.weights.weights.length),0);ae(n.device,i,se({...o,rowCount:s,edgeCount:ve,tileCount:p,traceTileCount:S}));let T=g?r(h,`powerTraces`):``,E=this.exactLogDeterminant,O=ge(`eigenvalues`,s),Ce=E?O:`${ie(h,`powerTraces`,`infinityNorm[infinityNormOffset]`,g)}${T}`,we=E?``:le(h,`powerTraces`,`infinityNorm[infinityNormOffset]`,g),Te=E?O:`${le(h,`powerTraces`,`infinityNorm[infinityNormOffset]`,g)}${T}`,{output:A,predictorCount:j}=o,M=k(o.weights);c(i,n,[o.predictors,o.response,o.eigenvalues,...f(M),A.table,A.summary,A.status.code,A.status.converged,A.residuals,A.powerTraceStandardErrors]);let N=j+1,P=j+2,F=P*(P+1)/2,Ee=this.traceMode===`stochastic`,I=0,L={},R=(e,t=1)=>{L[e]=I,I+=t};R(`NORM`),R(`BOUND`),R(`PARAMETER`),R(`VALID`),R(`CANDIDATES`),R(`AT_BOUNDARY`),R(`SERIES_LIMITED`),R(`STATUS`),R(`MEANS`,j),R(`YBAR`),R(`OLS`,j),R(`RUNTIME_ZERO`),R(`BETA`,N),R(`SIGMA2`),R(`CONDITIONING`,4),R(`TRACES`,3),R(`STANDARD_ERRORS`,N+1),R(`VARIANCE_DEFINED`);let De=Object.entries(L).map(([e,t])=>`const W_${e}: u32 = ${t}u;`).join(`
`),z=(t,r)=>e(n,`${i}-${t}`,`float32`,r),B=re(n,i,w),V=z(`workspace`,I),H=`const ROWS: u32 = ${s}u;
const K: u32 = ${j}u;
const P: u32 = ${N}u;
const M: u32 = ${P}u;
const PACK: u32 = ${F}u;
const TILE_ROWS: u32 = ${l}u;
const TILE_COUNT: u32 = ${p}u;
${De}`,U=`${d}
${ee}`,W=_,Oe=e=>[v(`offsets`,e.offsets,`read`),v(`neighbors`,e.neighbors,`read`),W(`weights`,e.weights,`read`)],ke=m(M,l),G=(e,t,n,r)=>ke.dispatches.map(i=>B(`${e}${i.idSuffix}`,t(i.weights),i.chunkTileCount,`${n}${i.tileConstantsWGSL?`\n${i.tileConstantsWGSL}`:``}`,te(i,r),{workgroupSize:ye})),K=z(`power-traces`,h),Ae=z(`power-sums`,h),q=z(`traces-infinity-norm`,1),je=z(`partial-weights-norm`,p),Me=z(`partial-means`,p*(j+1)),J=j*(j+1)/2,Y=J+j,Ne=z(`partial-ols`,p*Y),Pe=z(`ols-sums`,Y),Fe=z(`partial-gram`,3*p*F),X=z(`gram`,3*F),Ie=A.residuals??z(`residuals`,s),Le=z(`partial-residuals`,p*D),Re=z(`residual-sums`,D),ze=`
${he}
fn getRow(row: u32, z: ptr<function, array<f32, ${P}>>) {
  // Each value minus its mean is fixed before it meets the small terms: the compiler of the test
  // GPU otherwise regroups the residual as y - (mean + fit), in float32 steps of the response.
  let runtimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  (*z)[0] = 1.0;
  for (var column = 0u; column < K; column++) {
    (*z)[1u + column] = getRoundedDifference(design[designOffset + row * K + column], workspace[workspaceOffset + W_MEANS + column], runtimeZero);
  }
  // The response column is its first-stage OLS residual (see the class documentation).
  var residual = getRoundedDifference(response[responseOffset + row], workspace[workspaceOffset + W_YBAR], runtimeZero);
  for (var column = 0u; column < K; column++) {
    residual -= (*z)[1u + column] * workspace[workspaceOffset + W_OLS + column];
  }
  (*z)[K + 1u] = residual;
}`,Z=e=>G(`gram-tiles-${e}`,e=>[...Oe(e),W(`design`,o.predictors,`read`),W(`response`,o.response,`read`),W(`workspace`,V,`read`),W(`partial`,Fe,`read_write`)],`${H}
${ze}`,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${F}>;
  for (var row = firstRow; row < lastRow; row++) {
    var z: array<f32, ${P}>;
    getRow(row, &z);
${e===0?`    for (var first = 0u; first < M; first++) {
      for (var second = 0u; second <= first; second++) {
        sums[first * (first + 1u) / 2u + second] += z[first] * z[second];
      }
    }`:`    var wz: array<f32, ${P}>;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      var zn: array<f32, ${P}>;
      getRow(neighbor, &zn);
      for (var column = 0u; column < M; column++) {
        wz[column] += weight * zn[column];
      }
    }
    for (var first = 0u; first < M; first++) {
      for (var second = 0u; second <= first; second++) {
        sums[first * (first + 1u) / 2u + second] += ${e===1?`z[first] * wz[second] + z[second] * wz[first]`:`wz[first] * wz[second]`};
      }
    }`}
  }
  for (var slot = 0u; slot < PACK; slot++) {
    partial[partialOffset + (${e}u * TILE_COUNT + index) * PACK + slot] = sums[slot];
  }`),Q=[...G(`weights-norm-tiles`,e=>[...Oe(e),W(`partial`,je,`read_write`)],H,`let firstRow = index * TILE_ROWS;
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
  partial[partialOffset + index] = largest;`),...G(`means-tiles`,()=>[W(`design`,o.predictors,`read`),W(`response`,o.response,`read`),W(`partial`,Me,`read_write`)],H,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${j+1}>;
  for (var row = firstRow; row < lastRow; row++) {
    for (var column = 0u; column < K; column++) {
      sums[column] = sums[column] + design[designOffset + row * K + column];
    }
    sums[K] = sums[K] + response[responseOffset + row];
  }
  for (var slot = 0u; slot < ${j+1}u; slot++) {
    partial[partialOffset + index * ${j+1}u + slot] = sums[slot];
  }`),B(`means-merge`,[W(`partial`,Me,`read`),W(`workspace`,V,`read_write`)],j+1,H,`${y(`partial`,`index`,`${j+1}u`,`TILE_COUNT`)}
  workspace[workspaceOffset + W_MEANS + index] = sum / f32(ROWS);`),...G(`ols-tiles`,()=>[W(`design`,o.predictors,`read`),W(`response`,o.response,`read`),W(`workspace`,V,`read`),W(`partial`,Ne,`read_write`)],H,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${Y}>;
  for (var row = firstRow; row < lastRow; row++) {
    var centered: array<f32, ${j}>;
    for (var column = 0u; column < K; column++) {
      centered[column] = design[designOffset + row * K + column] - workspace[workspaceOffset + W_MEANS + column];
    }
    let centeredResponse = response[responseOffset + row] - workspace[workspaceOffset + W_YBAR];
    for (var first = 0u; first < K; first++) {
      for (var second = 0u; second <= first; second++) {
        sums[first * (first + 1u) / 2u + second] += centered[first] * centered[second];
      }
      sums[${J}u + first] += centered[first] * centeredResponse;
    }
  }
  for (var slot = 0u; slot < ${Y}u; slot++) {
    partial[partialOffset + index * ${Y}u + slot] = sums[slot];
  }`),B(`ols-merge`,[W(`partial`,Ne,`read`),W(`olsSums`,Pe,`read_write`)],Y,H,`${y(`partial`,`index`,`${Y}u`,`TILE_COUNT`)}
  olsSums[olsSumsOffset + index] = sum;`),B(`ols-solve`,[W(`olsSums`,Pe,`read`),W(`workspace`,V,`read_write`)],1,`${H}
${U}
${x(j)}`,`workspace[workspaceOffset + W_RUNTIME_ZERO] = 0.0;
  var a: array<f32, ${j*j}>;
  var rhs: array<f32, ${j}>;
  var scale: array<f32, ${j}>;
  var valid = true;
  for (var column = 0u; column < K; column++) {
    let diagonal = olsSums[olsSumsOffset + column * (column + 1u) / 2u + column];
    valid = valid && diagonal > 0.0 && isFiniteValue(diagonal);
    scale[column] = sqrt(max(diagonal, 1.0e-30));
  }
  for (var row = 0u; row < K; row++) {
    for (var column = 0u; column <= row; column++) {
      let value = olsSums[olsSumsOffset + row * (row + 1u) / 2u + column] / (scale[row] * scale[column]);
      a[row * K + column] = value;
      a[column * K + row] = value;
    }
    rhs[row] = olsSums[olsSumsOffset + ${J}u + row] / scale[row];
  }
  valid = valid && choleskyFactor_${j}(&a);
  for (var column = 0u; column < K; column++) {
    valid = valid && a[column * K + column] >= ${a(b)};
  }
  if (valid) {
    choleskySolve_${j}(&a, &rhs);
  }
  for (var column = 0u; column < K; column++) {
    let slope = rhs[column] / scale[column];
    workspace[workspaceOffset + W_OLS + column] = select(0.0, slope, valid && isFiniteValue(slope));
  }`),...new ne({id:`${i}-traces`,tileRowCount:this.traceTileRowCount,weights:M,maximumPower:h,mode:this.traceMode,probeCount:o.traceProbeCount,parameters:o.traceParameters,output:{powerTraces:K,powerSums:Ae,infinityNorm:q,powerTraceStandardErrors:A.powerTraceStandardErrors}}).getCommandNodes(n),...Z(0),...Z(1),...Z(2),B(`gram-merge`,[W(`partial`,Fe,`read`),W(`gram`,X,`read_write`)],3*F,H,`let mode = index / PACK;
  let entry = index % PACK;
  ${y(`partial`,`mode * TILE_COUNT * PACK + entry`,`PACK`,`TILE_COUNT`)}
  gram[gramOffset + index] = sum;`),B(`lambda-search`,[W(`partialNorm`,je,`read`),W(`gram`,X,`read`),E?_(`eigenvalues`,o.eigenvalues,`read`):W(`powerTraces`,K,`read`),W(`infinityNorm`,q,`read`),W(`workspace`,V,`read_write`)],1,`${H}
${U}
const ROWS_F: f32 = ${s}.0;
${Ce}
${we}
${x(N)}
${be()}

// (ee, dee/dlambda) of the filtered regression at lambda; NaN when its system is not positive definite.
fn evaluate(lambda: f32) -> vec2<f32> {
  let nan = getNaN();
  var a: array<f32, ${N*N}>;
  var rhs: array<f32, ${N}>;
  var scale: array<f32, ${N}>;
  for (var column = 0u; column < P; column++) {
    let diagonal = getGramValue(column, column, lambda);
    if (!(diagonal > 0.0) || !isFiniteValue(diagonal)) {
      return vec2<f32>(nan, nan);
    }
    scale[column] = sqrt(diagonal);
  }
  for (var row = 0u; row < P; row++) {
    for (var column = 0u; column < P; column++) {
      a[row * P + column] = getGramValue(row, column, lambda) / (scale[row] * scale[column]);
    }
    rhs[row] = getGramValue(row, K + 1u, lambda) / scale[row];
  }
  if (!choleskyFactor_${N}(&a)) {
    return vec2<f32>(nan, nan);
  }
  choleskySolve_${N}(&a, &rhs);
  var errorSumOfSquares = getGramValue(K + 1u, K + 1u, lambda);
  var derivative = getGramDerivative(K + 1u, K + 1u, lambda);
  for (var row = 0u; row < P; row++) {
    let beta = rhs[row] / scale[row];
    errorSumOfSquares -= 2.0 * beta * getGramValue(row, K + 1u, lambda);
    derivative -= 2.0 * beta * getGramDerivative(row, K + 1u, lambda);
    for (var column = 0u; column < P; column++) {
      let other = rhs[column] / scale[column];
      errorSumOfSquares += beta * other * getGramValue(row, column, lambda);
      derivative += beta * other * getGramDerivative(row, column, lambda);
    }
  }
  return vec2<f32>(errorSumOfSquares, derivative);
}

fn getObjective(lambda: f32) -> f32 {
  return -0.5 * ROWS_F * log(evaluate(lambda).x / ROWS_F) + getSpatialLogDeterminant(lambda);
}

fn getDerivative(lambda: f32) -> f32 {
  let value = evaluate(lambda);
  return -0.5 * ROWS_F * value.y / value.x + getLogDeterminantDerivative(lambda);
}
${ue()}`,`var largest = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    largest = max(largest, partialNorm[partialNormOffset + tile]);
  }
  let validNorm = largest > 0.0 && isFiniteValue(largest);
  let seriesLimit = ${a(g||E?ce:de(h))};
  let stationarity = ${a(ce)};
  let bound = select(getNaN(), min(stationarity, seriesLimit) / largest, validNorm);
  workspace[workspaceOffset + W_NORM] = select(getNaN(), largest, validNorm);
  workspace[workspaceOffset + W_BOUND] = bound;
  workspace[workspaceOffset + W_SERIES_LIMITED] = select(0.0, 1.0, seriesLimit < stationarity);
  workspace[workspaceOffset + W_PARAMETER] = getNaN();
  workspace[workspaceOffset + W_VALID] = 0.0;
  workspace[workspaceOffset + W_CANDIDATES] = 0.0;
  workspace[workspaceOffset + W_AT_BOUNDARY] = 0.0;
  let count = gram[gramOffset];
  let responseSquares = gram[gramOffset + packedIndex(K + 1u, K + 1u)];
  if (validNorm && isFiniteValue(count) && isFiniteValue(responseSquares) && count > 0.0 && responseSquares > 0.0) {
    let result = searchParameter(bound);
    workspace[workspaceOffset + W_PARAMETER] = result.parameter;
    workspace[workspaceOffset + W_VALID] = select(0.0, 1.0, isFiniteValue(result.parameter));
    workspace[workspaceOffset + W_CANDIDATES] = f32(result.localMaximumCount);
    workspace[workspaceOffset + W_AT_BOUNDARY] = select(0.0, 1.0, result.atBoundary);
  }`),B(`filtered-solve`,[W(`gram`,X,`read`),W(`workspace`,V,`read_write`)],1,`${H}
${U}
${x(N)}
${pe(N)}
${me(j)}
${be()}`,xe(j,N)),...G(`residuals-tiles`,()=>[W(`design`,o.predictors,`read`),W(`response`,o.response,`read`),W(`workspace`,V,`read`),W(`residualOut`,Ie,`read_write`),W(`partial`,Le,`read_write`)],H,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  let ybar = workspace[workspaceOffset + W_YBAR];
  var sums: array<f32, ${D}>;
  for (var row = firstRow; row < lastRow; row++) {
    var fitted = workspace[workspaceOffset + W_BETA];
    for (var column = 0u; column < K; column++) {
      fitted += design[designOffset + row * K + column] * workspace[workspaceOffset + W_BETA + 1u + column];
    }
    let u = response[responseOffset + row] - fitted;
    let y = response[responseOffset + row] - ybar;
    residualOut[residualOutOffset + row] = u;
    sums[0] += u * u;
    sums[1] += u;
    sums[2] += y * u;
    sums[3] += y * y;
  }
  for (var slot = 0u; slot < ${D}u; slot++) {
    partial[partialOffset + index * ${D}u + slot] = sums[slot];
  }`),B(`residuals-merge`,[W(`partial`,Le,`read`),W(`residualSums`,Re,`read_write`)],D,H,`${y(`partial`,`index`,`${D}u`,`TILE_COUNT`)}
  residualSums[residualSumsOffset + index] = sum;`)],$=oe(n,{id:`${i}-tr3`,operation:w,weights:M,rowCount:s,maximumPower:h,mode:this.traceMode,probeCount:o.traceProbeCount??64,seedParameters:o.traceParameters,tileRowCount:l,tileCount:p,workspace:V,parameterSlot:L.PARAMETER});Q.push(...$.nodes);let Be=Ee?`1.0 / f32(${$.probeTotal}u)`:`1.0`;return Q.push(B(`variance`,[W(`dots`,$.dots,`read`),E?_(`eigenvalues`,o.eigenvalues,`read`):W(`powerTraces`,K,`read`),W(`infinityNorm`,q,`read`),W(`workspace`,V,`read_write`)],1,`${H}
${U}
${Te}`,`let n = f32(ROWS);
  let lambda = workspace[workspaceOffset + W_PARAMETER];
  let traceOne = getTraceOne(lambda);
  let traceTwo = getTraceTwo(lambda);
  ${$.dotsSumWGSL}
  let traceThree = sum * ${Be};
  workspace[workspaceOffset + W_TRACES] = traceOne;
  workspace[workspaceOffset + W_TRACES + 1u] = traceTwo;
  workspace[workspaceOffset + W_TRACES + 2u] = traceThree;
  let information = traceTwo + traceThree - 2.0 * traceOne * traceOne / n;
  if (information > 0.0 && isFiniteValue(information)) {
    workspace[workspaceOffset + W_STANDARD_ERRORS + P] = sqrt(1.0 / information);
  } else {
    workspace[workspaceOffset + W_STANDARD_ERRORS + P] = getNaN();
    workspace[workspaceOffset + W_VARIANCE_DEFINED] = 0.0;
  }`),B(`finish`,[W(`residualSums`,Re,`read`),E?_(`eigenvalues`,o.eigenvalues,`read`):W(`powerTraces`,K,`read`),W(`infinityNorm`,q,`read`),W(`workspace`,V,`read`),W(`tableOut`,A.table,`read_write`),W(`summaryOut`,A.summary,`read_write`),v(`statusOut`,A.status.code,`read_write`),...A.status.converged?[v(`convergedOut`,A.status.converged,`read_write`)]:[]],1,`${H}
${U}
${u}
${Ce}
${fe()}
${Se(j,h,Ee,A.status.converged!==void 0,E)}`,`finish();`)),[...Q,...t(n,i,o.weights,o.output)]}};function be(){return`
fn packedIndex(first: u32, second: u32) -> u32 {
  let high = max(first, second);
  let low = min(first, second);
  return high * (high + 1u) / 2u + low;
}

// Gram of Z - lambda W Z: G0 - lambda (G1 + G1') + lambda^2 G2, G0 G1 G2 stored back to back.
fn getGramValue(first: u32, second: u32, lambda: f32) -> f32 {
  let entry = packedIndex(first, second);
  return gram[gramOffset + entry] - lambda * gram[gramOffset + PACK + entry]
    + lambda * lambda * gram[gramOffset + 2u * PACK + entry];
}

fn getGramDerivative(first: u32, second: u32, lambda: f32) -> f32 {
  let entry = packedIndex(first, second);
  return -gram[gramOffset + PACK + entry] + 2.0 * lambda * gram[gramOffset + 2u * PACK + entry];
}
`}function xe(e,t){return`let n = f32(ROWS);
  let lambda = workspace[workspaceOffset + W_PARAMETER];
  let nan = getNaN();
  for (var slot = 0u; slot < P + 1u; slot++) {
    workspace[workspaceOffset + W_STANDARD_ERRORS + slot] = nan;
    if (slot < P) {
      workspace[workspaceOffset + W_BETA + slot] = nan;
    }
  }
  workspace[workspaceOffset + W_VARIANCE_DEFINED] = 0.0;
  workspace[workspaceOffset + W_SIGMA2] = nan;
  for (var slot = 0u; slot < 4u; slot++) {
    workspace[workspaceOffset + W_CONDITIONING + slot] = nan;
  }
  if (!(n > f32(K + 2u))) {
    workspace[workspaceOffset + W_STATUS] = 2.0;
    return;
  }
  var a: array<f32, ${t*t}>;
  var unit: array<f32, ${t*t}>;
  var rhs: array<f32, ${t}>;
  var scale: array<f32, ${t}>;
  if (workspace[workspaceOffset + W_VALID] != 1.0 || !isFiniteValue(lambda)) {
    // A rank-deficient design makes every filtered system indefinite, so the search finds no
    // maximum: classify that as singular (the unfiltered system at lambda = 0), else non-finite.
    var nonFinite = false;
    var singular = false;
    for (var column = 0u; column < P; column++) {
      let diagonal = getGramValue(column, column, 0.0);
      nonFinite = nonFinite || !isFiniteValue(diagonal);
      singular = singular || !(diagonal > 0.0);
      scale[column] = sqrt(max(diagonal, 1.0e-30));
    }
    for (var row = 0u; row < P; row++) {
      for (var column = 0u; column < P; column++) {
        let value = getGramValue(row, column, 0.0) / (scale[row] * scale[column]);
        nonFinite = nonFinite || !isFiniteValue(value);
        a[row * P + column] = value;
      }
    }
    singular = singular || !choleskyFactor_${t}(&a);
    for (var column = 0u; column < P; column++) {
      singular = singular || !(a[column * P + column] >= ${a(b)});
    }
    workspace[workspaceOffset + W_STATUS] = select(3.0, 1.0, singular && !nonFinite);
    return;
  }
  for (var column = 0u; column < P; column++) {
    let diagonal = getGramValue(column, column, lambda);
    if (!(diagonal > 0.0) || !isFiniteValue(diagonal)) {
      workspace[workspaceOffset + W_STATUS] = 1.0;
      return;
    }
    scale[column] = sqrt(diagonal);
  }
  for (var row = 0u; row < P; row++) {
    for (var column = 0u; column < P; column++) {
      let value = getGramValue(row, column, lambda) / (scale[row] * scale[column]);
      a[row * P + column] = value;
      unit[row * P + column] = value;
    }
    rhs[row] = getGramValue(row, K + 1u, lambda) / scale[row];
  }
  if (!choleskyFactor_${t}(&a)) {
    workspace[workspaceOffset + W_STATUS] = 1.0;
    return;
  }
  // The pivot limit of GPUOrdinaryLeastSquares on the unit-diagonal filtered system: a filtered
  // column within 1 - 1e-6 (squared cosine) of the span of the earlier ones is singular, not a fit.
  for (var column = 0u; column < P; column++) {
    if (!(a[column * P + column] >= ${a(b)})) {
      workspace[workspaceOffset + W_STATUS] = 1.0;
      return;
    }
  }
  choleskySolve_${t}(&a, &rhs);
  var filtered: array<f32, ${t}>;
  var errorSumOfSquares = getGramValue(K + 1u, K + 1u, lambda);
  for (var row = 0u; row < P; row++) {
    filtered[row] = rhs[row] / scale[row];
    errorSumOfSquares -= 2.0 * filtered[row] * getGramValue(row, K + 1u, lambda);
    for (var column = 0u; column < P; column++) {
      errorSumOfSquares += filtered[row] * (rhs[column] / scale[column]) * getGramValue(row, column, lambda);
    }
  }
  let sigma2 = errorSumOfSquares / n;
  if (!(sigma2 > 0.0) || !isFiniteValue(sigma2)) {
    workspace[workspaceOffset + W_STATUS] = 3.0;
    return;
  }
  workspace[workspaceOffset + W_STATUS] = 0.0;
  workspace[workspaceOffset + W_SIGMA2] = sigma2;
  // Raw coefficients: b = first-stage slopes + filtered slopes, intercept = a0 + ybar - m'b.
  var intercept = filtered[0] + workspace[workspaceOffset + W_YBAR];
  for (var column = 0u; column < K; column++) {
    let slope = workspace[workspaceOffset + W_OLS + column] + filtered[1u + column];
    intercept -= workspace[workspaceOffset + W_MEANS + column] * slope;
    workspace[workspaceOffset + W_BETA + 1u + column] = slope;
  }
  workspace[workspaceOffset + W_BETA] = intercept;
  // Covariance sigma2 (Xs'Xs)^-1 = sigma2 D^-1 (unit^-1) D^-1 from unit solves.
  var covariance: array<f32, ${t*t}>;
  var vifs = 0.0;
  for (var column = 0u; column < P; column++) {
    var unitVector: array<f32, ${t}>;
    unitVector[column] = 1.0;
    choleskySolve_${t}(&a, &unitVector);
    for (var row = 0u; row < P; row++) {
      covariance[row * P + column] = sigma2 * unitVector[row] / (scale[row] * scale[column]);
    }
    if (column > 0u) {
      vifs = max(vifs, unitVector[column]);
    }
  }
  var interceptVariance = covariance[0];
  for (var first = 0u; first < K; first++) {
    let meanFirst = workspace[workspaceOffset + W_MEANS + first];
    interceptVariance -= 2.0 * meanFirst * covariance[1u + first];
    for (var second = 0u; second < K; second++) {
      interceptVariance += meanFirst * workspace[workspaceOffset + W_MEANS + second] * covariance[(1u + first) * P + 1u + second];
    }
  }
  workspace[workspaceOffset + W_STANDARD_ERRORS] = sqrt(interceptVariance);
  for (var column = 1u; column < P; column++) {
    workspace[workspaceOffset + W_STANDARD_ERRORS + column] = sqrt(covariance[column * P + column]);
  }
  workspace[workspaceOffset + W_VARIANCE_DEFINED] = 1.0;
  // Conditioning: Belsley condition number of the raw filtered design [c, Xc_s + m c] (c is the
  // filtered constant, not orthogonal to the centered columns) without forming its Gram matrix.
  // With L the Cholesky factor of the unit-diagonal filtered Gram (held in a), the raw column j is
  // inPlane_j u0 + g_j sum_i L[j][i] e_i, so the helper's alpha_j = inPlane_j / N_j, beta_j =
  // g_j / N_j and factor = L without its first row and column.
  var belsleyFactor: array<f32, ${e*e}>;
  var belsleyAlpha: array<f32, ${e}>;
  var belsleyBeta: array<f32, ${e}>;
  for (var j = 0u; j < K; j++) {
    let column = scale[1u + j];
    let inPlane = workspace[workspaceOffset + W_MEANS + j] * scale[0] + column * a[(1u + j) * P];
    var tail = 0.0;
    for (var i = 0u; i <= j; i++) {
      belsleyFactor[j * K + i] = a[(1u + j) * P + 1u + i];
      tail = tail + a[(1u + j) * P + 1u + i] * a[(1u + j) * P + 1u + i];
    }
    let rawNorm = sqrt(inPlane * inPlane + column * column * tail);
    belsleyAlpha[j] = inPlane / rawNorm;
    belsleyBeta[j] = column / rawNorm;
  }
  let unitRange = getSymmetricEigenvalueRange_${t}(&unit);
  workspace[workspaceOffset + W_CONDITIONING] = getBelsleyConditionNumber_${e}(&belsleyFactor, &belsleyAlpha, &belsleyBeta);
  workspace[workspaceOffset + W_CONDITIONING + 1u] = unitRange.y / unitRange.x;
  workspace[workspaceOffset + W_CONDITIONING + 2u] = vifs;`}function Se(e,t,n,r,i){let o=e+2,s=e=>`summaryOut[summaryOutOffset + ${e}u]`;return`
fn fail(status: u32) {
  let nan = getNaN();
  for (var slot = 0u; slot < ${o*4}u; slot++) {
    tableOut[tableOutOffset + slot] = nan;
  }
  for (var slot = 0u; slot < 22u; slot++) {
    summaryOut[summaryOutOffset + slot] = nan;
  }
  ${s(0)} = f32(ROWS);
  statusOut[statusOutOffset] = status;
${r?`  convergedOut[convergedOutOffset] = 0u;`:``}
}

fn finish() {
  let code = u32(workspace[workspaceOffset + W_STATUS]);
  if (code != 0u) {
    fail(code);
    return;
  }
  let n = f32(ROWS);
  let lambda = workspace[workspaceOffset + W_PARAMETER];
  let sigma2 = workspace[workspaceOffset + W_SIGMA2];
  let residualSquares = residualSums[residualSumsOffset];
  let residualTotal = residualSums[residualSumsOffset + 1u];
  let responseResidual = residualSums[residualSumsOffset + 2u];
  let responseSquares = residualSums[residualSumsOffset + 3u];
  if (!isFiniteValue(residualSquares) || !isFiniteValue(responseSquares) || !isFiniteValue(responseResidual)) {
    fail(3u);
    return;
  }
  for (var row = 0u; row < P; row++) {
    writeTableRow(row, workspace[workspaceOffset + W_BETA + row], workspace[workspaceOffset + W_STANDARD_ERRORS + row]);
  }
  writeTableRow(P, lambda, workspace[workspaceOffset + W_STANDARD_ERRORS + P]);
  let logDeterminant = getSpatialLogDeterminant(lambda);
  let logLikelihood = -0.5 * n * log(sigma2) + logDeterminant - 0.5 * n * (${a(E)} + 1.0);
  // spreg ML_Error counts the intercept and the predictors only.
  let parameterCount = f32(P);
  let predictionCovariance = responseSquares - responseResidual;
  let predictionVariance = responseSquares - 2.0 * responseResidual + residualSquares - residualTotal * residualTotal / n;
  let correlationCondition = workspace[workspaceOffset + W_CONDITIONING + 1u];
  let bound = workspace[workspaceOffset + W_BOUND];
  let norm = workspace[workspaceOffset + W_NORM];
  let scaled = abs(lambda) * norm;
  let atBoundary = workspace[workspaceOffset + W_AT_BOUNDARY] == 1.0;
  ${s(0)} = n;
  ${s(1)} = lambda;
  ${s(2)} = sigma2;
  ${s(3)} = logLikelihood;
  ${s(4)} = 2.0 * parameterCount - 2.0 * logLikelihood;
  ${s(5)} = parameterCount * log(n) - 2.0 * logLikelihood;
  ${s(6)} = predictionCovariance * predictionCovariance / (responseSquares * predictionVariance);
  ${s(7)} = getNaN();
  ${s(8)} = residualSquares;
  ${s(9)} = workspace[workspaceOffset + W_CONDITIONING];
  ${s(10)} = correlationCondition;
  ${s(11)} = workspace[workspaceOffset + W_CONDITIONING + 2u];
  ${s(12)} = select(0.0, 1.0, correlationCondition > ${a(h)});
  ${s(13)} = bound;
  ${s(14)} = ${n?`1.0`:`0.0`};
  ${s(15)} = select(0.0, 1.0, !atBoundary);
  ${s(16)} = select(0.0, 1.0, atBoundary);
  ${s(17)} = workspace[workspaceOffset + W_SERIES_LIMITED];
  ${s(18)} = select(
    1.0e38,
    n * pow(scaled, ${t+1}.0) / (${t+1}.0 * (1.0 - scaled)),
    scaled < 1.0
  );
  ${i?`${s(18)} = 0.0;`:``}
  ${s(19)} = workspace[workspaceOffset + W_VARIANCE_DEFINED];
  ${s(20)} = workspace[workspaceOffset + W_CANDIDATES];
  ${s(21)} = logDeterminant;
  statusOut[statusOutOffset] = 0u;
${r?`  convergedOut[convergedOutOffset] = select(0u, 1u, !atBoundary);`:``}
}
`}function k(e){return`chunks`in e?{...e,chunks:e.chunks.map(e=>({...e,weights:{...e.weights,distances:void 0}}))}:{...e,distances:void 0}}export{O as t};