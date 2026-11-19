import{t as e}from"./regression-output-dispatches-KKVTWkHP.js";import{c as t,i as n,o as r,s as i}from"./regression-binding-plans-DG-hiroU.js";import{a,i as o,n as s,r as c,t as l}from"./gpu-instrumental-variables-covariance-DavpJqb4.js";import{i as u}from"./spatial-likelihood-allocations-BGEHbqkB.js";import{a as d}from"./graph-chunk-dispatch-zBt8k2-o.js";import{r as f}from"./kernel-C3tfQOn4.js";import{c as p,o as ee}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as m,S as h,T as g,f as _,p as v,x as te,y}from"./gpu-contributor-utils-BIwLQON5.js";import{c as b}from"./spatial-weights-CqWysstN.js";import{t as ne}from"./gpu-spatial-weights-transpose-BOTZ1Oz3.js";import{i as x,n as S,t as C}from"./spatial-regression-solve-Vf-sQz9O.js";import{r as w}from"./gpu-ordinary-least-squares-B4HAVt2a.js";import{n as T,r as E,u as D}from"./wgsl-common-DfyXiYB0.js";import{t as O}from"./pairwise-sum-wgsl-iErcC5Or.js";import{t as k}from"./rounded-difference-wgsl-DgXXQtNj.js";import{t as A}from"./addition-rounding-error-wgsl-DVwPAqKC.js";import{t as j}from"./multiplication-rounding-error-wgsl-P721A62W.js";function M(e){return Math.floor(16/(e+1))}var re=`GPUSpatialTwoStageLeastSquares`,N=512,P=64,F=4096,I=4096,L=64,R=4,z=5,B=1e-5,V=class{id;props;rowCount;tileRowCount;tileCount;instrumentOrder;estimatedCommandNodeCount;constructor(t){t=r(t,[`id`,`weights`,`predictors`,`response`,`predictorCount`,`tileRowCount`,`instrumentOrder`,`robust`,`output`],[`table`,`summary`,`residuals`,`status`],`spatial-two-stage-least-squares`);let n=t.id??`spatial-two-stage-least-squares`;this.id=n,this.props=t;let{predictorCount:s,output:l}=t,u=t.instrumentOrder??1;if(u!==1&&u!==2)throw Error(`${n} instrumentOrder must be 1 or 2`);let d=M(u);if(!Number.isInteger(s)||s<1||s>d)throw Error(`${n} predictorCount must be an integer in [1, ${d}] with instrumentOrder ${u}`);let f=b(n,t.weights);if(m(t.predictors,[`float32`],`${n} predictors`),m(t.response,[`float32`],`${n} response`),t.response.length!==f)throw Error(`${n} response length must equal the weights row count`);if(t.predictors.length!==f*s)throw Error(`${n} predictors length must equal the weights row count * predictorCount`);if(m(l.table,[`float32`],`${n} output.table`),l.table.length<(s+2)*4)throw Error(`${n} output.table must hold (predictorCount + 2) * 4 float32 values`);if(m(l.summary,[`float32`],`${n} output.summary`),l.summary.length<16)throw Error(`${n} output.summary must hold 16 float32 values`);if(m(l.status.code,[`uint32`],`${n} output.status.code`),l.status.code.length<1)throw Error(`${n} output.status.code must hold one uint32 row`);if(l.residuals&&(m(l.residuals,[`float32`],`${n} output.residuals`),l.residuals.length<f))throw Error(`${n} output.residuals must hold one row per input row`);if(t.robust){let e=t.robust;a(n,f,e),e.kernel&&g(n,b(n,e.kernel)===f,`robust.kernel row count must equal response.length`);let r=o(e.output.status);for(let[t,i,a,o]of[[`robust.output.covariance`,e.output.covariance,`float32`,(s+2)**2],[`robust.output.status.code`,r,`uint32`,1],...e.clusters?[[`robust.clusters`,e.clusters,`uint32`,f]]:[]])m(i,[a],`${n} ${t}`),g(n,i.length===o,`${t} length must equal ${o}`)}this.instrumentOrder=u,t.tileRowCount!==void 0&&_(n,`tileRowCount`,t.tileRowCount),this.estimatedCommandNodeCount=N+(t.robust?2+c(s+1,!!t.robust.clusters):0),v(n,this.estimatedCommandNodeCount),this.rowCount=f,this.tileRowCount=t.tileRowCount??Math.max(L,Math.ceil(f/F)),this.tileCount=Math.ceil(f/this.tileRowCount),h(n,`tileCount`,this.tileCount,I),e(this.id,{...l,status:void 0,code:l.status.code,covariance:t.robust?.output.covariance,covarianceStatus:t.robust?o(t.robust.output.status):void 0},[[`table`,`summary`,`code`],[`covariance`,`covarianceStatus`],{read:[`covariance`,`covarianceStatus`],write:[`table`]}]),y(n,[l.table,l.summary,l.status.code,l.residuals,t.robust?.output.covariance,t.robust?o(t.robust.output.status):void 0],[t.predictors,t.response,t.weights.offsets,t.weights.neighbors,t.weights.weights,...i(t.weights),t.robust?.kernel?.offsets,t.robust?.kernel?.neighbors,t.robust?.kernel?.weights,t.robust?.kernel?.status?.overflow,t.robust?.clusters])}getCommandNodes(e){let{id:r,props:a,rowCount:c,tileRowCount:p,tileCount:m,instrumentOrder:h}=this,{weights:g,output:_,predictorCount:v}=a;d(e,r,{predictors:a.predictors,response:a.response,offsets:g.offsets,neighbors:g.neighbors,weights:g.weights,...Object.fromEntries(i(g).map((e,t)=>[`weights loss ${t}`,e])),table:_.table,summary:_.summary,code:_.status.code,residuals:_.residuals,robustCovariance:a.robust?.output.covariance,robustStatus:a.robust?o(a.robust.output.status):void 0,robustClusters:a.robust?.clusters,robustKernelOffsets:a.robust?.kernel?.offsets,robustKernelNeighbors:a.robust?.kernel?.neighbors,robustKernelWeights:a.robust?.kernel?.weights,robustKernelOverflow:a.robust?.kernel?.status?.overflow}),u(e.device,r,n(c,v,g.neighbors.length,h,p)),a.robust&&u(e.device,r,{"robust design":c*(v+1),"robust instruments":c*(h+1)*v,...s(c,v+1,(h+1)*v,a.robust)}),te(r,e,[a.predictors,a.response,g.offsets,g.neighbors,g.weights,_.table,_.summary,_.status.code,_.residuals]);let y=(h+1)*v,b=v+1,x=y*y+y*b+y+b,S=x+b+1+(y-v),C=(h+1)*v+2,w=C+2*b+6,D=(t,n)=>f(e,`${r}-${t}`,`float32`,n),k=D(`partial-means`,m*C),A=D(`partial-moments`,m*S),j=D(`moments`,S),M=D(`workspace`,w),N=D(`partial-residuals`,m*R),F=D(`residual-sums`,R),I=D(`partial-refinement`,m*(y+1)),L=D(`refinement-sums`,y+1),B=z+b,V=D(`partial-weights-norm`,m*3),U=D(`partial-anselin-kelejian`,m*B),W=D(`anselin-kelejian-sums`,B),G=_.residuals??D(`residuals`,c),K={offsets:f(e,`${r}-transpose-offsets`,`uint32`,c+1),neighbors:f(e,`${r}-transpose-neighbors`,`uint32`,g.neighbors.length),weights:D(`transpose-weights`,g.neighbors.length)},ce=D(`transposed-lag-residual`,c),q=`const ROWS: u32 = ${c}u;
const K: u32 = ${v}u;
const ORDER: u32 = ${h}u;
const M: u32 = ${y}u;
const Q: u32 = ${b}u;
const TILE_ROWS: u32 = ${p}u;
const TILE_COUNT: u32 = ${m}u;
const MEAN_COUNT: u32 = ${C}u;
const MOMENTS: u32 = ${x}u;
const MOMENT_STRIDE: u32 = ${S}u;
const W_DELTA: u32 = ${C}u;
const W_NORM: u32 = ${C+b}u;
const W_RESIDUAL_OFFSET: u32 = ${C+b+1}u;
const W_RUNTIME_ZERO: u32 = ${C+b+2}u;
const W_LAG_CENTER: u32 = ${C+b+3}u;
const W_ROW_SUM: u32 = ${C+b+4}u;
const W_CORRECTION: u32 = ${C+b+5}u;
const W_CORRECTION_OFFSET: u32 = ${C+2*b+5}u;`,J=e=>O({count:`TILE_COUNT`,term:`partial[partialOffset + sumItem * ${e}u + index]`}),Y=(e,t,n)=>({name:e,view:t,type:`f32`,access:n}),X=[{name:`offsets`,view:g.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:g.neighbors,type:`u32`,access:`read`},Y(`weights`,g.weights,`read`),Y(`design`,a.predictors,`read`),Y(`response`,a.response,`read`)],Z=(t,n,i,a,o,s)=>ee(e,{id:`${r}-${t}`,operation:re,variant:t,bindings:n,invocationCount:i,declarations:a,body:o,...s?{workgroupSize:s}:{}}),Q=`${q}
${ie()}`,le=`${Q}
${ae()}`,ue=`${le}
${oe()}`,de=new ne({id:`${r}-transpose`,weights:g,output:K}).getCommandNodes(e),$=[Z(`weights-norm-tiles`,[X[0],X[1],X[2],X[4],Y(`partial`,V,`read_write`)],m,q,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var largest = 0.0;
  var largestRowSum = -3.0e38;
  var responseSum = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var absoluteSum = 0.0;
    var rowSum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      if (neighbors[neighborsOffset + slot] < ROWS) {
        absoluteSum += abs(weights[weightsOffset + slot]);
        rowSum += weights[weightsOffset + slot];
      }
    }
    largest = max(largest, absoluteSum);
    largestRowSum = max(largestRowSum, rowSum);
    responseSum += response[responseOffset + row];
  }
  partial[partialOffset + 3u * index] = largest;
  partial[partialOffset + 3u * index + 1u] = largestRowSum;
  partial[partialOffset + 3u * index + 2u] = responseSum;`,P),Z(`weights-norm-merge`,[Y(`partial`,V,`read`),Y(`workspace`,M,`read_write`)],1,q,`var largest = 0.0;
  var largestRowSum = -3.0e38;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    largest = max(largest, partial[partialOffset + 3u * tile]);
    largestRowSum = max(largestRowSum, partial[partialOffset + 3u * tile + 1u]);
  }
  ${O({count:`TILE_COUNT`,term:`partial[partialOffset + 3u * sumItem + 2u]`})}
  workspace[workspaceOffset + W_NORM] = largest;
  workspace[workspaceOffset + W_ROW_SUM] = largestRowSum;
  workspace[workspaceOffset + W_LAG_CENTER] = sum / f32(ROWS);
  // A zero the compiler cannot see: the runtime zero of getRoundedDifference.
  workspace[workspaceOffset + W_RUNTIME_ZERO] = 0.0;`),Z(`means-tiles`,[...X,Y(`workspace`,M,`read`),Y(`partial`,k,`read_write`)],m,Q,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${C}>;
  for (var row = firstRow; row < lastRow; row++) {
    var lagged: array<f32, ${v}>;
    var twiceLagged: array<f32, ${v}>;
    let lagResponse = getLags(row, &lagged, &twiceLagged, ORDER == 2u);
    for (var column = 0u; column < K; column++) {
      sums[column] = sums[column] + design[designOffset + row * K + column];
      sums[K + column] = sums[K + column] + lagged[column];
      if (ORDER == 2u) {
        sums[2u * K + 2u + column] = sums[2u * K + 2u + column] + twiceLagged[column];
      }
    }
    sums[2u * K] = sums[2u * K] + lagResponse;
    sums[2u * K + 1u] = sums[2u * K + 1u] + response[responseOffset + row];
  }
  for (var slot = 0u; slot < MEAN_COUNT; slot++) {
    partial[partialOffset + index * MEAN_COUNT + slot] = sums[slot];
  }`,P),Z(`means-merge`,[Y(`partial`,k,`read`),Y(`workspace`,M,`read_write`)],C,q,`${J(C)}
  workspace[workspaceOffset + index] = sum / f32(ROWS);
  if (index == 0u) {
    // A zero the compiler cannot see: the runtime zero of getRoundedDifference.
    workspace[workspaceOffset + W_RUNTIME_ZERO] = 0.0;
  }`),Z(`moments-tiles`,[...X,Y(`workspace`,M,`read`),Y(`partial`,A,`read_write`)],m,le,`let base = partialOffset + index * MOMENT_STRIDE;
  for (var slot = 0u; slot < MOMENT_STRIDE; slot++) {
    partial[base + slot] = 0.0;
  }
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  for (var row = firstRow; row < lastRow; row++) {
    var h: array<f32, ${y}>;
    var z: array<f32, ${b}>;
    var centeredResponse = 0.0;
    getCenteredRow(row, &h, &z, &centeredResponse, true);
    for (var a = 0u; a < M; a++) {
      for (var b = 0u; b < M; b++) {
        partial[base + a * M + b] += h[a] * h[b];
      }
      for (var b = 0u; b < Q; b++) {
        partial[base + M * M + a * Q + b] += h[a] * z[b];
      }
      partial[base + M * M + M * Q + a] += h[a] * centeredResponse;
    }
    for (var b = 0u; b < Q; b++) {
      partial[base + M * M + M * Q + M + b] += z[b] * z[b];
      partial[base + MOMENTS + b] += z[b];
    }
    partial[base + MOMENTS + Q] += centeredResponse;
    // The lagged instruments; the first K instruments are the first K regressors.
    for (var a = K; a < M; a++) {
      partial[base + MOMENTS + Q + 1u + (a - K)] += h[a];
    }
  }`,P),Z(`moments-merge`,[Y(`partial`,A,`read`),Y(`moments`,j,`read_write`)],S,q,`${J(S)}
  moments[momentsOffset + index] = sum;`),Z(`solve`,[Y(`moments`,j,`read`),Y(`workspace`,M,`read_write`)],1,`${q}
${H(y,b)}`,`var delta: array<f32, ${b}>;
  var inverseDiagonal: array<f32, ${b}>;
  var meanQuadratic = 0.0;
  var unusedVector: array<f32, ${b}>;
  var unusedQuadratic = 0.0;
  var unusedDiagnostics: array<f32, 6>;
  let ok = solveSystem(&delta, &inverseDiagonal, &meanQuadratic, &unusedVector, &unusedQuadratic, &unusedDiagnostics, false);
  // The stored means are float32, so the centered columns do not sum to zero. The residuals of a
  // model with an intercept do: this is the mean they would otherwise carry.
  var residualOffset = moments[momentsOffset + MOMENTS + Q];
  for (var column = 0u; column < Q; column++) {
    workspace[workspaceOffset + W_DELTA + column] = select(getNaN(), delta[column], ok);
    residualOffset -= moments[momentsOffset + MOMENTS + column] * delta[column];
  }
  workspace[workspaceOffset + W_RESIDUAL_OFFSET] = select(getNaN(), residualOffset / f32(ROWS), ok);`),Z(`refinement-tiles`,[...X,Y(`workspace`,M,`read`),Y(`partial`,I,`read_write`)],m,ue,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${y+1}>;
  for (var row = firstRow; row < lastRow; row++) {
    var h: array<f32, ${y}>;
    var z: array<f32, ${b}>;
    var centeredResponse = 0.0;
    getCenteredRow(row, &h, &z, &centeredResponse, true);
    let value = getStructuralResidual(&z, centeredResponse);
    let residual = value.x + value.y;
    for (var a = 0u; a < M; a++) {
      sums[a] += h[a] * residual;
    }
    sums[M] += residual;
  }
  for (var slot = 0u; slot < ${y+1}u; slot++) {
    partial[partialOffset + index * ${y+1}u + slot] = sums[slot];
  }`,P),Z(`refinement-merge`,[Y(`partial`,I,`read`),Y(`refinement`,L,`read_write`)],y+1,q,`${J(y+1)}
  refinement[refinementOffset + index] = sum;`),Z(`refine`,[Y(`moments`,j,`read`),Y(`refinement`,L,`read`),Y(`workspace`,M,`read_write`)],1,`${q}
${H(y,b,`refinement`)}`,`var correction: array<f32, ${b}>;
  var inverseDiagonal: array<f32, ${b}>;
  var meanQuadratic = 0.0;
  var unusedVector: array<f32, ${b}>;
  var unusedQuadratic = 0.0;
  var unusedDiagnostics: array<f32, 6>;
  let ok = solveSystem(&correction, &inverseDiagonal, &meanQuadratic, &unusedVector, &unusedQuadratic, &unusedDiagnostics, false);
  // The corrected residuals sum to zero: the mean of u less the mean of (Z - mean) correction.
  var offsetCorrection = refinement[refinementOffset + M];
  for (var column = 0u; column < Q; column++) {
    workspace[workspaceOffset + W_CORRECTION + column] = select(getNaN(), correction[column], ok);
    offsetCorrection -= moments[momentsOffset + MOMENTS + column] * correction[column];
  }
  workspace[workspaceOffset + W_CORRECTION_OFFSET] = select(getNaN(), offsetCorrection / f32(ROWS), ok);`),Z(`residuals-tiles`,[...X,Y(`workspace`,M,`read`),Y(`residualsOut`,G,`read_write`),Y(`partial`,N,`read_write`)],m,ue,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${R}>;
  for (var row = firstRow; row < lastRow; row++) {
    var h: array<f32, ${y}>;
    var z: array<f32, ${b}>;
    var centeredResponse = 0.0;
    getCenteredRow(row, &h, &z, &centeredResponse, false);
    // The residual at delta (a float32 value and its correction), then the refinement.
    let value = getStructuralResidual(&z, centeredResponse);
    var lowPart = value.y - workspace[workspaceOffset + W_CORRECTION_OFFSET];
    for (var b = 0u; b < Q; b++) {
      lowPart = lowPart - z[b] * workspace[workspaceOffset + W_CORRECTION + b];
    }
    let residual = value.x + lowPart;
    residualsOut[residualsOutOffset + row] = residual;
    sums[0] += residual * residual;
    sums[1] += residual;
    sums[2] += centeredResponse * residual;
    sums[3] += centeredResponse * centeredResponse;
  }
  for (var slot = 0u; slot < ${R}u; slot++) {
    partial[partialOffset + index * ${R}u + slot] = sums[slot];
  }`,P),Z(`residuals-merge`,[Y(`partial`,N,`read`),Y(`residualSums`,F,`read_write`)],R,q,`${J(R)}
  residualSums[residualSumsOffset + index] = sum;`),...de,Z(`anselin-kelejian-tiles`,[...X,Y(`workspace`,M,`read`),Y(`residuals`,G,`read`),Y(`partial`,U,`read_write`)],m,Q,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${B}>;
  for (var row = firstRow; row < lastRow; row++) {
    let residual = residuals[residualsOffset + row];
    var rowSum = 0.0;
    var lagResidual = 0.0;
    var squareSum = 0.0;
    var crossSum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      // Reverse weight w_ji by binary search of this row in the neighbor's sorted row.
      var reverse = 0.0;
      var low = offsets[offsetsOffset + neighbor];
      var high = offsets[offsetsOffset + neighbor + 1u];
      while (low < high) {
        let middle = (low + high) / 2u;
        let candidate = neighbors[neighborsOffset + middle];
        if (candidate == row) {
          reverse = weights[weightsOffset + middle];
          break;
        }
        if (candidate < row) {
          low = middle + 1u;
        } else {
          high = middle;
        }
      }
      let neighborResidual = residuals[residualsOffset + neighbor];
      rowSum += weight;
      lagResidual += weight * neighborResidual;
      squareSum += weight * weight;
      crossSum += weight * reverse;
    }
    sums[0] += residual * lagResidual;
    sums[1] += residual * rowSum;
    sums[2] += rowSum;
    sums[3] += squareSum;
    sums[4] += crossSum;
  }
  // The u'W z products (slots after the fixed ones) come from the next kernel.
  for (var slot = 0u; slot < ${z}u; slot++) {
    partial[partialOffset + index * ${B}u + slot] = sums[slot];
  }`,P),Z(`transposed-lag-residual`,[{name:`offsets`,view:K.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:K.neighbors,type:`u32`,access:`read`},Y(`weights`,K.weights,`read`),Y(`residuals`,G,`read`),Y(`lagOut`,ce,`read_write`)],c,q,`var sum = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    sum += weights[weightsOffset + slot] * residuals[residualsOffset + neighbors[neighborsOffset + slot]];
  }
  lagOut[lagOutOffset + index] = sum;`),Z(`anselin-kelejian-products`,[...X,Y(`workspace`,M,`read`),Y(`transposedLagResidual`,ce,`read`),Y(`partial`,U,`read_write`)],m,Q,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${b}>;
  for (var row = firstRow; row < lastRow; row++) {
    var lagged: array<f32, ${v}>;
    var twiceLagged: array<f32, ${v}>;
    let lagResponse = getLags(row, &lagged, &twiceLagged, false);
    let product = transposedLagResidual[transposedLagResidualOffset + row];
    // Differences from the means, rounded as written so the tile sums are not regrouped.
    let runtimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
    for (var column = 0u; column < K; column++) {
      let centered = getRoundedDifference(design[designOffset + row * K + column], workspace[workspaceOffset + column], runtimeZero);
      sums[column] += product * centered;
    }
    sums[K] += product * getRoundedDifference(lagResponse, workspace[workspaceOffset + 2u * K], runtimeZero);
  }
  for (var column = 0u; column < Q; column++) {
    partial[partialOffset + index * ${B}u + ${z}u + column] = sums[column];
  }`,P),Z(`anselin-kelejian-merge`,[Y(`partial`,U,`read`),Y(`akSums`,W,`read_write`)],B,q,`${J(B)}
  akSums[akSumsOffset + index] = sum;`),Z(`finish`,[Y(`moments`,j,`read`),Y(`workspace`,M,`read`),Y(`residualSums`,F,`read`),Y(`akSums`,W,`read`),Y(`tableOut`,_.table,`read_write`),Y(`summaryOut`,_.summary,`read_write`),{name:`statusOut`,view:_.status.code,type:`u32`,access:`read_write`}],1,`${q}
${H(y,b)}
${se(v)}`,`finish();`)];if(a.robust){let t=D(`robust-design`,c*b),n=D(`robust-instruments`,c*y);$.push(Z(`robust-inputs`,[...X,Y(`workspace`,M,`read`),Y(`robustDesign`,t,`read_write`),Y(`robustInstruments`,n,`read_write`)],c,Q,`var lagged: array<f32, ${v}>; var twiceLagged: array<f32, ${v}>;
let lagResponse = getLags(index, &lagged, &twiceLagged, ORDER == 2u);
for (var column = 0u; column < K; column++) {
 let value = design[designOffset + index * K + column];
 robustDesign[robustDesignOffset + index * Q + column] = value;
 robustInstruments[robustInstrumentsOffset + index * M + column] = value;
 robustInstruments[robustInstrumentsOffset + index * M + K + column] = lagged[column];
 if (ORDER == 2u) { robustInstruments[robustInstrumentsOffset + index * M + 2u * K + column] = twiceLagged[column]; }
}
// The raw lag W y: the lag column plus the constant it was formed about.
robustDesign[robustDesignOffset + index * Q + K] = lagResponse + workspace[workspaceOffset + W_LAG_CENTER] * workspace[workspaceOffset + W_ROW_SUM];`)),$.push(...new l({id:`${r}-robust`,design:t,coefficientCount:b,instruments:n,instrumentCount:y,residuals:G,...a.robust}).getCommandNodes(e)),$.push(Z(`robust-table`,[Y(`covariance`,a.robust.output.covariance,`read`),{name:`robustStatus`,view:o(a.robust.output.status),type:`u32`,access:`read`},Y(`table`,_.table,`read_write`)],b+1,`${E}\n${T}`,`let variance = covariance[covarianceOffset + index * ${b+2}u];
var standardError = getNaN();
if (robustStatus[robustStatusOffset] == 0u && variance >= 0.0) { standardError = sqrt(variance); }
let statistic = table[tableOffset + index * 4u] / standardError;
table[tableOffset + index * 4u + 1u] = standardError;
table[tableOffset + index * 4u + 2u] = statistic;
table[tableOffset + index * 4u + 3u] = erfcValue(abs(statistic) * 0.70710678);`))}return[...$,...t(e,r,a.weights,a.output,a.robust?.output)]}};function ie(){return`
${k}
${A}
// The parts of the lag column of the last getLags call: W (y - c), and s_i - s as a rounded
// difference and what the row sum dropped. getStructuralResidual reads them, because their sum
// times c is rounded to a float32 step of the response mean when the row sums differ.
var<private> lagCenteredPart: f32;
var<private> lagRowSumDeviationHigh: f32;
var<private> lagRowSumDeviationLow: f32;

fn getLags(
  row: u32,
  lagged: ptr<function, array<f32, K>>,
  twiceLagged: ptr<function, array<f32, K>>,
  needSecond: bool
) -> f32 {
  let lagCenter = workspace[workspaceOffset + W_LAG_CENTER];
  let lagRuntimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  var lagResponse = 0.0;
  var rowSum = 0.0;
  var rowSumError = 0.0;
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS) {
      continue;
    }
    let weight = weights[weightsOffset + slot];
    lagResponse += weight * getRoundedDifference(response[responseOffset + neighbor], lagCenter, lagRuntimeZero);
    let nextRowSum = rowSum + weight;
    rowSumError += getAdditionRoundingError(rowSum, weight, nextRowSum);
    rowSum = nextRowSum;
    for (var column = 0u; column < K; column++) {
      (*lagged)[column] += weight * design[designOffset + neighbor * K + column];
    }
    if (needSecond) {
      var inner: array<f32, K>;
      for (var second = offsets[offsetsOffset + neighbor]; second < offsets[offsetsOffset + neighbor + 1u]; second++) {
        let farther = neighbors[neighborsOffset + second];
        if (farther >= ROWS) {
          continue;
        }
        let farWeight = weights[weightsOffset + second];
        for (var column = 0u; column < K; column++) {
          inner[column] += farWeight * design[designOffset + farther * K + column];
        }
      }
      for (var column = 0u; column < K; column++) {
        (*twiceLagged)[column] += weight * inner[column];
      }
    }
  }
  lagCenteredPart = lagResponse;
  lagRowSumDeviationHigh = getRoundedDifference(rowSum, workspace[workspaceOffset + W_ROW_SUM], lagRuntimeZero);
  lagRowSumDeviationLow = rowSumError;
  return lagResponse + lagCenter * (lagRowSumDeviationHigh + lagRowSumDeviationLow);
}
`}function ae(){return`
// h = [x - mean(x), W x - mean(W x)], z = [x - mean(x), v - mean(v)] with v the lag column of
// getLags (W y less a constant, so its deviations are those of W y). Every difference of a
// value and its mean goes through getRoundedDifference: the shader compiler of the test GPU
// otherwise regroups (y - mean) - z delta - offset around the mean, where the small terms are
// lost below the float32 step of the response.
fn getCenteredRow(
  row: u32,
  h: ptr<function, array<f32, M>>,
  z: ptr<function, array<f32, Q>>,
  centeredResponse: ptr<function, f32>,
  needSecond: bool
) {
  let runtimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  var lagged: array<f32, K>;
  var twiceLagged: array<f32, K>;
  let lagResponse = getLags(row, &lagged, &twiceLagged, needSecond);
  for (var column = 0u; column < K; column++) {
    let centered = getRoundedDifference(design[designOffset + row * K + column], workspace[workspaceOffset + column], runtimeZero);
    (*h)[column] = centered;
    (*z)[column] = centered;
    (*h)[K + column] = getRoundedDifference(lagged[column], workspace[workspaceOffset + K + column], runtimeZero);
    if (ORDER == 2u && needSecond) {
      (*h)[2u * K + column] = getRoundedDifference(twiceLagged[column], workspace[workspaceOffset + 2u * K + 2u + column], runtimeZero);
    }
  }
  (*z)[K] = getRoundedDifference(lagResponse, workspace[workspaceOffset + 2u * K], runtimeZero);
  *centeredResponse = getRoundedDifference(response[responseOffset + row], workspace[workspaceOffset + 2u * K + 1u], runtimeZero);
}
`}function oe(){return`
${j}
// value - first * second, where value.x + value.y is the running value.
fn subtractProduct(value: vec2<f32>, first: f32, second: f32, runtimeZero: u32) -> vec2<f32> {
  let product = bitcast<f32>(bitcast<u32>(first * second) ^ runtimeZero);
  let next = value.x - product;
  return vec2<f32>(
    next,
    value.y + getAdditionRoundingError(value.x, -product, next) - getMultiplicationRoundingError(first, second, product)
  );
}

fn getStructuralResidual(z: ptr<function, array<f32, Q>>, centeredResponse: f32) -> vec2<f32> {
  let runtimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  var value = vec2<f32>(centeredResponse, 0.0);
  for (var b = 0u; b < K; b++) {
    value = subtractProduct(value, (*z)[b], workspace[workspaceOffset + W_DELTA + b], runtimeZero);
  }
  let lagCoefficient = workspace[workspaceOffset + W_DELTA + K];
  let lagCenter = workspace[workspaceOffset + W_LAG_CENTER];
  // The lag column times rho in its parts: W (y - c), then c (s_i - s) as c times the product of
  // the row sum deviation and rho, then the mean of the lag column, which the centered regressor
  // subtracts.
  value = subtractProduct(value, lagCenteredPart, lagCoefficient, runtimeZero);
  let scaledDeviation = bitcast<f32>(bitcast<u32>(lagRowSumDeviationHigh * lagCoefficient) ^ runtimeZero);
  let scaledDeviationError = getMultiplicationRoundingError(lagRowSumDeviationHigh, lagCoefficient, scaledDeviation);
  value = subtractProduct(value, lagCenter, scaledDeviation, runtimeZero);
  value = subtractProduct(value, -workspace[workspaceOffset + 2u * K], lagCoefficient, runtimeZero);
  let lowPart = value.y
    - lagCenter * (scaledDeviationError + lagRowSumDeviationLow * lagCoefficient)
    - workspace[workspaceOffset + W_RESIDUAL_OFFSET];
  return vec2<f32>(value.x, lowPart);
}
`}function H(e,t,n=`moments`){return`
${E}

${D}

// Mean of regressor column: a predictor, or W y, which is the lag column plus the constant it
// was formed about (see getLags).
fn regressorMean(column: u32) -> f32 {
  if (column == K) {
    return workspace[workspaceOffset + 2u * K] + workspace[workspaceOffset + W_LAG_CENTER] * workspace[workspaceOffset + W_ROW_SUM];
  }
  return workspace[workspaceOffset + column];
}

// The moment pass sums products of deviations from the stored float32 means, which miss the true
// means by up to a float32 step. Each accessor subtracts n times the product of the two mean
// remainders (the deviation sums over n), which gives the moment about the true means, as
// GPUOrdinaryLeastSquares does. Without it a constant predictor whose float32 mean is not the
// constant kept a sum of squares above zero and was fitted with status 0.
fn getInstrumentDeviationSum(a: u32) -> f32 {
  if (a < K) {
    return moments[momentsOffset + MOMENTS + a];
  }
  return moments[momentsOffset + MOMENTS + Q + 1u + (a - K)];
}

fn getInstrumentMoment(a: u32, c: u32) -> f32 {
  return moments[momentsOffset + a * M + c] - getInstrumentDeviationSum(a) * (getInstrumentDeviationSum(c) / f32(ROWS));
}

fn getCrossMoment(a: u32, b: u32) -> f32 {
  return moments[momentsOffset + M * M + a * Q + b] - getInstrumentDeviationSum(a) * (moments[momentsOffset + MOMENTS + b] / f32(ROWS));
}

fn getResponseMoment(a: u32) -> f32 {
  ${n===`refinement`?`return refinement[refinementOffset + a] - getInstrumentDeviationSum(a) * (refinement[refinementOffset + M] / f32(ROWS));`:`return moments[momentsOffset + M * M + M * Q + a] - getInstrumentDeviationSum(a) * (moments[momentsOffset + MOMENTS + Q] / f32(ROWS));`}
}

fn getRegressorSquares(b: u32) -> f32 {
  let deviationSum = moments[momentsOffset + MOMENTS + b];
  return moments[momentsOffset + M * M + M * Q + M + b] - deviationSum * (deviationSum / f32(ROWS));
}

// Whether a corrected sum of squares is above the rounding of the two sums it is the difference
// of: below, the column is constant at float32 resolution.
fn isColumnVarying(corrected: f32, uncorrected: f32) -> bool {
  return corrected > ${p(B)} * uncorrected && isFiniteValue(corrected);
}

${S(e)}
${t===e?``:S(t)}
${[...new Set([e,t])].map(e=>x(e)).join(`
`)}
${C(t)}

// With wantDiagnostics, diagnostics = [Belsley condition number of [1, PZ], correlation condition
// number of Z'PZ, largest variance inflation factor of Z'PZ, correlation condition number of the
// instruments, first-stage F of the excluded instruments, their partial R squared].
fn solveSystem(
  delta: ptr<function, array<f32, ${t}>>,
  inverseDiagonal: ptr<function, array<f32, ${t}>>,
  meanQuadratic: ptr<function, f32>,
  second: ptr<function, array<f32, ${t}>>,
  secondQuadratic: ptr<function, f32>,
  diagnostics: ptr<function, array<f32, 6>>,
  wantDiagnostics: bool
) -> bool {
  var instrumentScale: array<f32, ${e}>;
  var regressorScale: array<f32, ${t}>;
  for (var a = 0u; a < M; a++) {
    let diagonal = getInstrumentMoment(a, a);
    if (!isColumnVarying(diagonal, moments[momentsOffset + a * M + a])) {
      return false;
    }
    instrumentScale[a] = 1.0 / sqrt(diagonal);
  }
  for (var b = 0u; b < Q; b++) {
    let diagonal = getRegressorSquares(b);
    if (!isColumnVarying(diagonal, moments[momentsOffset + M * M + M * Q + M + b])) {
      return false;
    }
    regressorScale[b] = 1.0 / sqrt(diagonal);
  }
  var factor: array<f32, ${e*e}>;
  for (var a = 0u; a < M; a++) {
    for (var c = 0u; c < M; c++) {
      factor[a * M + c] = getInstrumentMoment(a, c) * instrumentScale[a] * instrumentScale[c];
    }
  }
  if (wantDiagnostics) {
    var instrumentCorrelation = factor;
    let instrumentRange = getSymmetricEigenvalueRange_${e}(&instrumentCorrelation);
    (*diagnostics)[3] = instrumentRange.y / max(instrumentRange.x, 1.0e-30);
  }
  if (!choleskyFactor_${e}(&factor)) {
    return false;
  }
  if (wantDiagnostics) {
    // First stage of W y on the instruments: explained sums of squares from the forward
    // substitution y = L^-1 (scaled H'z); the first K instruments are the included X.
    var reduced: array<f32, ${e}>;
    for (var a = 0u; a < M; a++) {
      reduced[a] = getCrossMoment(a, K) * instrumentScale[a];
    }
    var explainedFull = 0.0;
    var explainedRestricted = 0.0;
    for (var row = 0u; row < M; row++) {
      var value = reduced[row];
      for (var j = 0u; j < row; j++) {
        value = value - factor[row * M + j] * reduced[j];
      }
      value = value / factor[row * M + row];
      reduced[row] = value;
      explainedFull += value * value;
      if (row < K) {
        explainedRestricted += value * value;
      }
    }
    let total = getRegressorSquares(K);
    let excluded = explainedFull - explainedRestricted;
    (*diagnostics)[4] = (excluded / f32(M - K)) / ((total - explainedFull) / (f32(ROWS) - f32(M) - 1.0));
    (*diagnostics)[5] = excluded / (total - explainedRestricted);
  }
  // x1 = (H'H)^-1 H'Z column by column, x2 = (H'H)^-1 H'y (scaled instrument space).
  var projectedInstruments: array<f32, ${e*t}>;
  for (var b = 0u; b < Q; b++) {
    var rhs: array<f32, ${e}>;
    for (var a = 0u; a < M; a++) {
      rhs[a] = getCrossMoment(a, b) * instrumentScale[a] * regressorScale[b];
    }
    choleskySolve_${e}(&factor, &rhs);
    for (var a = 0u; a < M; a++) {
      projectedInstruments[a * Q + b] = rhs[a];
    }
  }
  var projectedResponse: array<f32, ${e}>;
  for (var a = 0u; a < M; a++) {
    projectedResponse[a] = getResponseMoment(a) * instrumentScale[a];
  }
  choleskySolve_${e}(&factor, &projectedResponse);
  var normal: array<f32, ${t*t}>;
  var rhs: array<f32, ${t}>;
  for (var b = 0u; b < Q; b++) {
    var rhsValue = 0.0;
    for (var a = 0u; a < M; a++) {
      rhsValue += getCrossMoment(a, b) * instrumentScale[a] * regressorScale[b] * projectedResponse[a];
    }
    rhs[b] = rhsValue;
    for (var c = 0u; c < Q; c++) {
      var sum = 0.0;
      for (var a = 0u; a < M; a++) {
        sum += getCrossMoment(a, b) * instrumentScale[a] * regressorScale[b] * projectedInstruments[a * Q + c];
      }
      normal[b * Q + c] = sum;
    }
  }
  var normalDiagonal: array<f32, ${t}>;
  if (wantDiagnostics) {
    for (var b = 0u; b < Q; b++) {
      normalDiagonal[b] = normal[b * Q + b];
    }
    // Correlation condition number of Z'PZ (unit diagonal) and Belsley's of the raw [1, PZ]
    // design with unit columns, from the Cholesky factor of the correlation matrix and the column
    // means (never the raw Gram matrix; see getBelsleyConditionNumberWGSL).
    var secondCorrelation: array<f32, ${t*t}>;
    var alpha: array<f32, ${t}>;
    var beta: array<f32, ${t}>;
    let sampleSize = f32(ROWS);
    for (var b = 0u; b < Q; b++) {
      let meanB = regressorMean(b);
      let centered = normal[b * Q + b] / (regressorScale[b] * regressorScale[b]);
      let rawNorm = sqrt(sampleSize * meanB * meanB + centered);
      alpha[b] = sqrt(sampleSize) * meanB / rawNorm;
      beta[b] = sqrt(centered) / rawNorm;
      for (var c = 0u; c < Q; c++) {
        secondCorrelation[b * Q + c] = select(normal[b * Q + c] / sqrt(normal[b * Q + b] * normal[c * Q + c]), 1.0, b == c);
      }
    }
    var correlationEigen = secondCorrelation;
    let correlationRange = getSymmetricEigenvalueRange_${t}(&correlationEigen);
    (*diagnostics)[1] = correlationRange.y / max(correlationRange.x, 1.0e-30);
    var belsley = 3.0e38;
    if (choleskyFactor_${t}(&secondCorrelation)) {
      belsley = getBelsleyConditionNumber_${t}(&secondCorrelation, &alpha, &beta);
    }
    (*diagnostics)[0] = belsley;
  }
  if (!choleskyFactor_${t}(&normal)) {
    return false;
  }
  if (wantDiagnostics) {
    var inflation = 0.0;
    for (var b = 0u; b < Q; b++) {
      inflation = max(inflation, choleskyInverseDiagonal_${t}(&normal, b) * normalDiagonal[b]);
    }
    (*diagnostics)[2] = inflation;
  }
  choleskySolve_${t}(&normal, &rhs);
  for (var b = 0u; b < Q; b++) {
    (*delta)[b] = rhs[b] * regressorScale[b];
    (*inverseDiagonal)[b] = choleskyInverseDiagonal_${t}(&normal, b) * regressorScale[b] * regressorScale[b];
  }
  // Quadratic form of the regressor means, in the scaled space.
  var meansScaled: array<f32, ${t}>;
  for (var b = 0u; b < Q; b++) {
    meansScaled[b] = regressorMean(b) * regressorScale[b];
  }
  var solved = meansScaled;
  choleskySolve_${t}(&normal, &solved);
  var quadratic = 0.0;
  for (var b = 0u; b < Q; b++) {
    quadratic += meansScaled[b] * solved[b];
  }
  *meanQuadratic = quadratic;
  var secondScaled: array<f32, ${t}>;
  for (var b = 0u; b < Q; b++) {
    secondScaled[b] = (*second)[b] * regressorScale[b];
  }
  var secondSolved = secondScaled;
  choleskySolve_${t}(&normal, &secondSolved);
  var secondSum = 0.0;
  for (var b = 0u; b < Q; b++) {
    secondSum += secondScaled[b] * secondSolved[b];
  }
  *secondQuadratic = secondSum;
  return true;
}
`}function se(e){return`
${T}

fn fail(status: u32) {
  let nan = getNaN();
  for (var slot = 0u; slot < ${(e+2)*4}u; slot++) {
    tableOut[tableOutOffset + slot] = nan;
  }
  for (var slot = 0u; slot < 16u; slot++) {
    summaryOut[summaryOutOffset + slot] = nan;
  }
  summaryOut[summaryOutOffset] = f32(ROWS);
  statusOut[statusOutOffset] = status;
}

fn finish() {
  let n = f32(ROWS);
  let degrees = n - f32(Q + 1u);
  if (!(degrees > 0.0)) {
    fail(2u);
    return;
  }
  let residualSquares = residualSums[residualSumsOffset];
  let residualTotal = residualSums[residualSumsOffset + 1u];
  let responseResidual = residualSums[residualSumsOffset + 2u];
  let responseSquares = residualSums[residualSumsOffset + 3u];
  if (!isFiniteValue(residualSquares) || !isFiniteValue(responseResidual) || !isFiniteValue(responseSquares)) {
    fail(select(3u, 1u, !isFiniteValue(workspace[workspaceOffset + W_DELTA])));
    return;
  }
  var delta: array<f32, ${e+1}>;
  var inverseDiagonal: array<f32, ${e+1}>;
  var meanQuadratic = 0.0;
  var productVector: array<f32, ${e+1}>;
  for (var b = 0u; b < Q; b++) {
    productVector[b] = akSums[akSumsOffset + ${z}u + b];
  }
  var productQuadratic = 0.0;
  var diagnostics: array<f32, 6>;
  if (!solveSystem(&delta, &inverseDiagonal, &meanQuadratic, &productVector, &productQuadratic, &diagnostics, true)) {
    fail(1u);
    return;
  }
  let sigmaSquared = residualSquares / n;
  // The published coefficients: the float32 nearest to the solve's delta plus its refinement
  // (the residuals, sigma2 and the statistics are those of the unrounded sum).
  for (var b = 0u; b < Q; b++) {
    delta[b] = workspace[workspaceOffset + W_DELTA + b] + workspace[workspaceOffset + W_CORRECTION + b];
  }
  // Intercept: ybar - mean(Z) delta. Its variance is sigma2 / n plus the delta part.
  var intercept = workspace[workspaceOffset + 2u * K + 1u];
  for (var b = 0u; b < Q; b++) {
    intercept -= regressorMean(b) * delta[b];
  }
  intercept += workspace[workspaceOffset + W_RESIDUAL_OFFSET] + workspace[workspaceOffset + W_CORRECTION_OFFSET];
  let interceptVariance = sigmaSquared * (1.0 / n + meanQuadratic);
  for (var row = 0u; row < Q + 1u; row++) {
    // Rows 0 is the intercept; rows 1..K are the predictors; row K + 1 is rho (delta index K).
    let index = max(row, 1u) - 1u;
    let coefficient = select(delta[index], intercept, row == 0u);
    let variance = select(sigmaSquared * inverseDiagonal[index], interceptVariance, row == 0u);
    let standardError = sqrt(variance);
    let z = coefficient / standardError;
    tableOut[tableOutOffset + row * 4u] = coefficient;
    tableOut[tableOutOffset + row * 4u + 1u] = standardError;
    tableOut[tableOutOffset + row * 4u + 2u] = z;
    tableOut[tableOutOffset + row * 4u + 3u] = erfcValue(abs(z) * 0.70710678);
  }
  // Squared correlation of y and y - u: the response is centered, so Syy = responseSquares.
  let predictedVariance = responseSquares - 2.0 * responseResidual + residualSquares - residualTotal * residualTotal / n;
  let covariance = responseSquares - responseResidual;
  summaryOut[summaryOutOffset] = n;
  summaryOut[summaryOutOffset + 1u] = sigmaSquared;
  summaryOut[summaryOutOffset + 2u] = residualSquares;
  summaryOut[summaryOutOffset + 3u] = covariance * covariance / (responseSquares * predictedVariance);
  // Anselin-Kelejian (1997) as in spreg akTest: Moran's I of u, corrected by the estimation variance.
  let residualLagProduct = akSums[akSumsOffset];
  let rowSumProduct = akSums[akSumsOffset + 1u];
  let weightsSum = akSums[akSumsOffset + 2u];
  let traceT = akSums[akSumsOffset + 3u] + akSums[akSumsOffset + 4u];
  // Moran's I divides by the total weight S0 and is undefined (NaN) when signed weights sum
  // to zero. The statistic n I^2 / phi^2, with phi^2 = (T + 4 a / sigma2) / ((S0 / n)^2 n),
  // does not depend on S0: it is (u'Wu / sigma2)^2 / (T + 4 a / sigma2), formed that way so
  // that a zero total does not turn it into infinity over infinity.
  let moranI = select((n * residualLagProduct) / (weightsSum * residualSquares), getNaN(), weightsSum == 0.0);
  let estimationTerm = rowSumProduct * rowSumProduct / n + productQuadratic;
  let residualLagRatio = residualLagProduct / sigmaSquared;
  let anselinKelejian = residualLagRatio * residualLagRatio / (traceT + 4.0 * estimationTerm / sigmaSquared);
  summaryOut[summaryOutOffset + 4u] = moranI;
  summaryOut[summaryOutOffset + 5u] = anselinKelejian;
  summaryOut[summaryOutOffset + 6u] = erfcValue(sqrt(max(anselinKelejian, 0.0) * 0.5));
  // Conditioning of the second-stage and instrument systems; a warning never changes the status.
  summaryOut[summaryOutOffset + 7u] = diagnostics[0];
  summaryOut[summaryOutOffset + 8u] = diagnostics[1];
  summaryOut[summaryOutOffset + 9u] = diagnostics[2];
  summaryOut[summaryOutOffset + 11u] = diagnostics[3];
  summaryOut[summaryOutOffset + 10u] =
    select(0.0, 1.0, diagnostics[1] > ${p(w)} || diagnostics[3] > ${p(w)});
  summaryOut[summaryOutOffset + 14u] = diagnostics[4];
  summaryOut[summaryOutOffset + 15u] = diagnostics[5];
  // rho is the last delta; admissible when |rho| ||W||_inf < 1 (NaN compares false: flagged).
  let weightsNorm = workspace[workspaceOffset + W_NORM];
  summaryOut[summaryOutOffset + 13u] = weightsNorm;
  summaryOut[summaryOutOffset + 12u] =
    select(1.0, 0.0, abs(delta[K]) * weightsNorm < 1.0);
  statusOut[statusOutOffset] = 0u;
}
`}export{M as n,V as t};