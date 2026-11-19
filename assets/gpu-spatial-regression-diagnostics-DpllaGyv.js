import{t as e}from"./construction-snapshot-DdkX9Pc0.js";import{r as t}from"./kernel-C3tfQOn4.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as r,S as i,f as a,u as o,x as s,y as c}from"./gpu-contributor-utils-BIwLQON5.js";import{c as l}from"./spatial-weights-CqWysstN.js";import{t as u}from"./gpu-spatial-weights-transpose-BOTZ1Oz3.js";import{n as d}from"./spatial-regression-solve-Vf-sQz9O.js";import{n as f,r as p,u as m}from"./wgsl-common-DfyXiYB0.js";import{t as h}from"./pairwise-sum-wgsl-iErcC5Or.js";var g=`GPUSpatialRegressionDiagnostics`,_=64,v=4096,y=64,b=7,x=1;function S(e){return o(e,4)}function C(e){return o(e,4)}var w=class{id;props;rowCount;tileRowCount;tileCount;constructor(t){let n=e(t,[`id`,`weights`,`predictors`,`response`,`residuals`,`mask`,`predictorCount`,`restandardizeRows`,`tileRowCount`,`output`]),i=e(n.weights,[`offsets`,`neighbors`,`weights`,`distances`,`spatialContext`,`information`,`status`]),o=e(n.output,[`tests`,`summary`,`status`]);t=Object.freeze({...n,weights:Object.freeze({...i,...i.status?{status:e(i.status,[`count`,`requiredCount`,`overflow`,`candidateOverflow`,`invalidCount`,`uncertainCount`,`iterationCount`,`converged`,`iterationLimitReached`,`approximation`])}:{}}),output:Object.freeze({...o,status:e(o.status,[`code`])})});let s=t.id??`spatial-regression-diagnostics`;this.id=s,this.props=t;let{predictorCount:u,output:d}=t;a(s,`predictorCount`,u,1,15);let f=l(s,t.weights);if(r(t.predictors,[`float32`],`${s} predictors`),r(t.response,[`float32`],`${s} response`),r(t.residuals,[`float32`],`${s} residuals`),t.response.length!==f)throw Error(`${s} response length must equal the weights row count`);if(t.residuals.length!==f)throw Error(`${s} residuals length must equal the weights row count`);if(t.predictors.length!==f*u)throw Error(`${s} predictors length must equal the weights row count * predictorCount`);if(t.mask&&(r(t.mask,[`uint32`],`${s} mask`),t.mask.length!==f))throw Error(`${s} mask length must equal the weights row count`);if(r(d.tests,[`float32`],`${s} output.tests`),d.tests.length<18)throw Error(`${s} output.tests must hold 18 float32 values`);if(r(d.summary,[`float32`],`${s} output.summary`),d.summary.length<13)throw Error(`${s} output.summary must hold 13 float32 values`);if(r(d.status.code,[`uint32`],`${s} output.status.code`),d.status.code.length<1)throw Error(`${s} output.status.code must hold one uint32 row`);t.tileRowCount!==void 0&&a(s,`tileRowCount`,t.tileRowCount),this.rowCount=f,this.tileRowCount=t.tileRowCount??Math.max(y,Math.ceil(f/v)),this.tileCount=Math.ceil(f/this.tileRowCount),c(s,[d.tests,d.summary,d.status.code],[t.predictors,t.response,t.residuals,t.mask,t.weights.offsets,t.weights.neighbors,t.weights.weights,t.weights.status?.overflow,t.weights.status?.candidateOverflow])}getCommandNodes(e){let{id:r,props:a,rowCount:o,tileRowCount:c,tileCount:l}=this,{weights:d,output:f,predictorCount:p}=a;s(r,e,[a.predictors,a.response,a.residuals,a.mask,d.offsets,d.neighbors,d.weights,d.status?.overflow,d.status?.candidateOverflow,f.tests,f.summary,f.status.code]);let v=p+1,y=5*v*v+b+v+x;i(r,`rows x (predictorCount + 3)`,o*(p+3),S(e.device)),i(r,`tiles x moments (rows, tileRowCount, predictorCount)`,l*y,C(e.device));let w=a.restandardizeRows===!0,D=(n,i)=>t(e,`${r}-${n}`,`float32`,i),O=D(`partial-means`,l*(p+1)),k=D(`means`,p+1),A=D(`partial-moments`,l*y),j=D(`moments`,y),M=D(`row-table`,o*(v+2)),N=D(`transposed-lag-table`,o*v),P=w?D(`row-scale`,o):void 0,F={offsets:t(e,`${r}-transpose-offsets`,`uint32`,o+1),neighbors:t(e,`${r}-transpose-neighbors`,`uint32`,d.neighbors.length),weights:D(`transpose-weights`,d.neighbors.length)},I=new u({id:`${r}-transpose`,weights:d,output:F}).getCommandNodes(e),L=`const ROWS: u32 = ${o}u;
const K: u32 = ${p}u;
const TABLE_WIDTH: u32 = ${v+2}u;
const P: u32 = ${v}u;
const TILE_ROWS: u32 = ${c}u;
const TILE_COUNT: u32 = ${l}u;`,R=`${L}
${m}
fn isFittedRow(row: u32) -> bool {
  ${a.mask?`if (rowMask[rowMaskOffset + row] == 0u) {
    return false;
  }`:``}
  if (!isFiniteValue(response[responseOffset + row]) ||
      !isFiniteValue(residuals[residualsOffset + row])) {
    return false;
  }
  for (var column = 0u; column < K; column++) {
    if (!isFiniteValue(design[designOffset + row * K + column])) {
      return false;
    }
  }
  return true;
}`,z=a.mask?[{name:`rowMask`,view:a.mask,type:`u32`,access:`read`}]:[],B=[{name:`design`,view:a.predictors,type:`f32`,access:`read`},{name:`response`,view:a.response,type:`f32`,access:`read`},{name:`residuals`,view:a.residuals,type:`f32`,access:`read`},...z],V=P?[{name:`rowScale`,view:P,type:`f32`,access:`read`}]:[],H=P?`fn getRowScale(row: u32) -> f32 {
  return rowScale[rowScaleOffset + row];
}`:`fn getRowScale(row: u32) -> f32 {
  return 1.0;
}`,U=e=>h({count:`TILE_COUNT`,term:`partial[partialOffset + sumItem * ${e}u + index]`}),W=[{name:`offsets`,view:d.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:d.neighbors,type:`u32`,access:`read`},{name:`weights`,view:d.weights,type:`f32`,access:`read`}],G={name:`table`,view:M,type:`f32`,access:`read`};return[n(e,{id:`${r}-means-tiles`,operation:g,variant:`means-tiles`,workgroupSize:_,bindings:[...B,{name:`partial`,view:O,type:`f32`,access:`read_write`}],invocationCount:l,declarations:R,body:`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${p}>;
  var count = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    if (!isFittedRow(row)) {
      continue;
    }
    count = count + 1.0;
    for (var column = 0u; column < K; column++) {
      sums[column] = sums[column] + design[designOffset + row * K + column];
    }
  }
  for (var column = 0u; column < K; column++) {
    partial[partialOffset + index * ${p+1}u + column] = sums[column];
  }
  partial[partialOffset + index * ${p+1}u + K] = count;`}),n(e,{id:`${r}-means-merge`,operation:g,variant:`means-merge`,bindings:[{name:`partial`,view:O,type:`f32`,access:`read`},{name:`means`,view:k,type:`f32`,access:`read_write`}],invocationCount:p+1,declarations:L,body:`${U(p+1)}
  var rows = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    rows = rows + partial[partialOffset + tile * ${p+1}u + K];
  }
  means[meansOffset + index] = select(sum, select(0.0, sum / rows, rows > 0.0), index < K);`}),n(e,{id:`${r}-row-table`,operation:g,variant:`row-table`,bindings:[...B,{name:`means`,view:k,type:`f32`,access:`read`},{name:`tableOut`,view:M,type:`f32`,access:`read_write`}],invocationCount:o,declarations:R,body:`let base = tableOutOffset + index * TABLE_WIDTH;
  if (!isFittedRow(index)) {
    for (var column = 0u; column < TABLE_WIDTH; column++) {
      tableOut[base + column] = 0.0;
    }
    return;
  }
  tableOut[base] = 1.0;
  for (var column = 0u; column < K; column++) {
    tableOut[base + 1u + column] =
      design[designOffset + index * K + column] - means[meansOffset + column];
  }
  tableOut[base + P] = response[responseOffset + index];
  tableOut[base + P + 1u] = residuals[residualsOffset + index];`}),...P?[n(e,{id:`${r}-row-scale`,operation:g,variant:`row-scale`,bindings:[...W,G,{name:`scaleOut`,view:P,type:`f32`,access:`read_write`}],invocationCount:o,declarations:L,body:`var total = 0.0;
  if (table[tableOffset + index * TABLE_WIDTH] > 0.5) {
    for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor < ROWS && table[tableOffset + neighbor * TABLE_WIDTH] > 0.5) {
        total += weights[weightsOffset + slot];
      }
    }
  }
  scaleOut[scaleOutOffset + index] = select(0.0, 1.0 / total, total > 0.0);`})]:[],...I,n(e,{id:`${r}-transposed-lag`,operation:g,variant:`transposed-lag`,bindings:[{name:`offsets`,view:F.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:F.neighbors,type:`u32`,access:`read`},{name:`weights`,view:F.weights,type:`f32`,access:`read`},G,...V,{name:`lagOut`,view:N,type:`f32`,access:`read_write`}],invocationCount:o,declarations:`${L}
${H}`,body:`var sums: array<f32, ${v}>;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    // The transposed link j -> i carries w_ji, scaled by the row scale of its source row j.
    let weight = weights[weightsOffset + slot] * getRowScale(neighbor);
    for (var column = 0u; column < P; column++) {
      sums[column] += weight * table[tableOffset + neighbor * TABLE_WIDTH + column];
    }
  }
  for (var column = 0u; column < P; column++) {
    lagOut[lagOutOffset + index * P + column] = sums[column];
  }`}),n(e,{id:`${r}-moments-tiles`,operation:g,variant:`moments-tiles`,workgroupSize:_,bindings:[...W,G,...V,{name:`transposedLag`,view:N,type:`f32`,access:`read`},{name:`partial`,view:A,type:`f32`,access:`read_write`}],invocationCount:l,declarations:`${L}
const PP: u32 = ${v*v}u;
const MOMENTS: u32 = ${y}u;
${H}
${T()}`,body:`let base = partialOffset + index * MOMENTS;
  for (var slot = 0u; slot < MOMENTS; slot++) {
    partial[base + slot] = 0.0;
  }
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  for (var row = firstRow; row < lastRow; row++) {
    accumulateRow(row, base);
  }`}),n(e,{id:`${r}-moments-merge`,operation:g,variant:`moments-merge`,bindings:[{name:`partial`,view:A,type:`f32`,access:`read`},{name:`moments`,view:j,type:`f32`,access:`read_write`}],invocationCount:y,declarations:L,body:`${U(y)}
  moments[momentsOffset + index] = sum;`}),n(e,{id:`${r}-finish`,operation:g,variant:`finish`,bindings:[{name:`moments`,view:j,type:`f32`,access:`read`},{name:`testsOut`,view:f.tests,type:`f32`,access:`read_write`},{name:`summaryOut`,view:f.summary,type:`f32`,access:`read_write`},{name:`statusOut`,view:f.status.code,type:`u32`,access:`read_write`},...d.status?.overflow?[{name:`weightsOverflow`,view:d.status.overflow,type:`u32`,access:`read`}]:[],...d.status?.candidateOverflow?[{name:`weightsCandidateOverflow`,view:d.status.candidateOverflow,type:`u32`,access:`read`}]:[]],invocationCount:1,declarations:`${L}
const PP: u32 = ${v*v}u;
${E(v)}`,body:`${d.status?.overflow?`if (weightsOverflow[weightsOverflowOffset] != 0u) { fail(3u, moments[momentsOffset + 5u * PP + 7u + P]); return; }`:``}
${d.status?.candidateOverflow?`if (weightsCandidateOverflow[weightsCandidateOverflowOffset] != 0u) { fail(3u, moments[momentsOffset + 5u * PP + 7u + P]); return; }`:``}
finish();`})]}};function T(){return`
fn tableValue(row: u32, column: u32) -> f32 {
  return table[tableOffset + row * TABLE_WIDTH + column];
}

fn accumulateRow(row: u32, base: u32) {
  // An excluded row (leading table entry 0) is outside the analysis.
  if (tableValue(row, 0u) < 0.5) {
    return;
  }
  let rowWeightScale = getRowScale(row);
  var z: array<f32, P>;
  var u: array<f32, P>;
  var v: array<f32, P>;
  for (var column = 0u; column < P; column++) {
    z[column] = tableValue(row, column);
    // (W'Z)_row, accumulated over the transposed CSR so one-way links are included.
    v[column] = transposedLag[transposedLagOffset + row * P + column];
  }
  var weightSum = 0.0;
  var squareSum = 0.0;
  var crossSum = 0.0;
  var lagResidual = 0.0;
  var lagResponse = 0.0;
  var lagFitted = 0.0;
  let residual = tableValue(row, P + 1u);
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    // Links to excluded rows are dropped from the restricted W.
    if (neighbor >= ROWS || tableValue(neighbor, 0u) < 0.5) {
      continue;
    }
    let weight = weights[weightsOffset + slot] * rowWeightScale;
    // Reverse weight w_ji by binary search of this row in the neighbor's sorted row; it is 0 for
    // a one-way link, which then contributes nothing to tr(WW).
    var reverse = 0.0;
    var low = offsets[offsetsOffset + neighbor];
    var high = offsets[offsetsOffset + neighbor + 1u];
    while (low < high) {
      let middle = (low + high) / 2u;
      let candidate = neighbors[neighborsOffset + middle];
      if (candidate == row) {
        reverse = weights[weightsOffset + middle] * getRowScale(neighbor);
        break;
      }
      if (candidate < row) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    weightSum += weight;
    squareSum += weight * weight;
    crossSum += weight * reverse;
    let neighborResponse = tableValue(neighbor, P);
    let neighborResidual = tableValue(neighbor, P + 1u);
    lagResponse += weight * neighborResponse;
    lagResidual += weight * neighborResidual;
    lagFitted += weight * (neighborResponse - neighborResidual);
    for (var column = 0u; column < P; column++) {
      u[column] += weight * tableValue(neighbor, column);
    }
  }
  for (var first = 0u; first < P; first++) {
    for (var second = 0u; second < P; second++) {
      let slot = first * P + second;
      partial[base + slot] += z[first] * z[second];
      partial[base + PP + slot] += z[first] * u[second];
      partial[base + 2u * PP + slot] += u[first] * u[second];
      partial[base + 3u * PP + slot] += v[first] * v[second];
      partial[base + 4u * PP + slot] += v[first] * u[second];
    }
  }
  let scalars = base + 5u * PP;
  partial[scalars] += weightSum;
  partial[scalars + 1u] += squareSum;
  partial[scalars + 2u] += crossSum;
  partial[scalars + 3u] += residual * residual;
  partial[scalars + 4u] += residual * lagResidual;
  partial[scalars + 5u] += residual * lagResponse;
  partial[scalars + 6u] += lagFitted * lagFitted;
  for (var column = 0u; column < P; column++) {
    partial[scalars + 7u + column] += z[column] * lagFitted;
  }
  partial[scalars + 7u + P] += 1.0;
}
`}function E(e){return`
var<private> columnScale: array<f32, ${e}>;

${p}

${m}

// Complementary error function, fractional error below 1.2e-7 (Numerical Recipes erfcc).
${f}

// Chi-square survival for the two degrees of freedom the LM tests use, in closed form:
// erfc(sqrt(x / 2)) for 1 and exp(-x / 2) for 2. Any other value is NaN, never the 2-degree
// form: a general caller uses getChiSquareSurvivalValue of analysis-core/kernels/incomplete-gamma-wgsl.
fn chiSquareSurvival(value: f32, degreesOfFreedom: u32) -> f32 {
  if (degreesOfFreedom == 1u) {
    return erfcValue(sqrt(max(value, 0.0) * 0.5));
  }
  if (degreesOfFreedom == 2u) {
    return exp(-max(value, 0.0) * 0.5);
  }
  return getNaN();
}

// Scaled (unit-diagonal Gram) entry of the p x p matrix stored at \`base\`.
fn scaledMoment(base: u32, row: u32, column: u32) -> f32 {
  return moments[momentsOffset + base + row * P + column] * columnScale[row] * columnScale[column];
}

${d(e)}

// result = (Z'Z)^-1 Q (or its transpose version) for the matrix Q stored at \`base\`.
fn solveMatrix(
  factor: ptr<function, array<f32, ${e*e}>>,
  base: u32,
  transposed: bool,
  result: ptr<function, array<f32, ${e*e}>>
) {
  for (var column = 0u; column < P; column++) {
    var rhs: array<f32, ${e}>;
    for (var row = 0u; row < P; row++) {
      rhs[row] = select(
        scaledMoment(base, row, column),
        scaledMoment(base, column, row),
        transposed
      );
    }
    choleskySolve_${e}(factor, &rhs);
    for (var row = 0u; row < P; row++) {
      (*result)[row * P + column] = rhs[row];
    }
  }
}

fn writeTest(row: u32, statistic: f32, degreesOfFreedom: u32) {
  testsOut[testsOutOffset + row * 3u] = statistic;
  testsOut[testsOutOffset + row * 3u + 1u] = f32(degreesOfFreedom);
  testsOut[testsOutOffset + row * 3u + 2u] = select(
    getNaN(),
    chiSquareSurvival(statistic, degreesOfFreedom),
    isFiniteValue(statistic)
  );
}

fn fail(status: u32, fittedRows: f32) {
  let nan = getNaN();
  for (var slot = 0u; slot < 18u; slot++) {
    testsOut[testsOutOffset + slot] = nan;
  }
  for (var slot = 0u; slot < 13u; slot++) {
    summaryOut[summaryOutOffset + slot] = nan;
  }
  summaryOut[summaryOutOffset] = fittedRows;
  summaryOut[summaryOutOffset + 12u] =
    f32(ROWS) - fittedRows;
  for (var row = 0u; row < 5u; row++) {
    testsOut[testsOutOffset + row * 3u + 1u] = select(1.0, 2.0, row == 4u);
  }
  testsOut[testsOutOffset + 16u] = 0.0;
  statusOut[statusOutOffset] = status;
}

fn finish() {
  let scalars = 5u * PP;
  // Rows analyzed: finite and unmasked. Everything below runs over those rows only.
  let n = moments[momentsOffset + scalars + 7u + P];
  let weightSum = moments[momentsOffset + scalars];
  let squareSum = moments[momentsOffset + scalars + 1u];
  let crossSum = moments[momentsOffset + scalars + 2u];
  let residualSquares = moments[momentsOffset + scalars + 3u];
  let residualLag = moments[momentsOffset + scalars + 4u];
  let responseLag = moments[momentsOffset + scalars + 5u];
  let fittedLagSquares = moments[momentsOffset + scalars + 6u];
  if (!(n > f32(P) + 1.0)) {
    fail(2u, select(0.0, n, isFiniteValue(n)));
    return;
  }
  if (!isFiniteValue(residualSquares) || !isFiniteValue(residualLag) ||
      !isFiniteValue(responseLag) || !isFiniteValue(fittedLagSquares) ||
      !isFiniteValue(squareSum) || !isFiniteValue(crossSum) || !(residualSquares > 0.0)) {
    fail(4u, n);
    return;
  }
  let traceT = squareSum + crossSum;
  if (!(weightSum > 0.0) || !(traceT > 0.0)) {
    fail(3u, n);
    return;
  }
  for (var column = 0u; column < P; column++) {
    let diagonal = moments[momentsOffset + column * P + column];
    if (!(diagonal > 0.0) || !isFiniteValue(diagonal)) {
      fail(1u, n);
      return;
    }
    columnScale[column] = 1.0 / sqrt(diagonal);
  }
  var factor: array<f32, ${e*e}>;
  for (var row = 0u; row < P; row++) {
    for (var column = 0u; column < P; column++) {
      factor[row * P + column] = scaledMoment(0u, row, column);
    }
  }
  if (!choleskyFactor_${e}(&factor)) {
    fail(1u, n);
    return;
  }
  var inverseC: array<f32, ${e*e}>;
  var inverseCTransposed: array<f32, ${e*e}>;
  var inverseD: array<f32, ${e*e}>;
  var inverseE: array<f32, ${e*e}>;
  var inverseF: array<f32, ${e*e}>;
  solveMatrix(&factor, PP, false, &inverseC);
  solveMatrix(&factor, PP, true, &inverseCTransposed);
  solveMatrix(&factor, 2u * PP, false, &inverseD);
  solveMatrix(&factor, 3u * PP, false, &inverseE);
  solveMatrix(&factor, 4u * PP, false, &inverseF);
  var traceC = 0.0;
  var traceD = 0.0;
  var traceE = 0.0;
  var traceF = 0.0;
  var traceCC = 0.0;
  var traceCCTransposed = 0.0;
  for (var row = 0u; row < P; row++) {
    traceC += inverseC[row * P + row];
    traceD += inverseD[row * P + row];
    traceE += inverseE[row * P + row];
    traceF += inverseF[row * P + row];
    for (var column = 0u; column < P; column++) {
      traceCC += inverseC[row * P + column] * inverseC[column * P + row];
      traceCCTransposed += inverseC[row * P + column] * inverseCTransposed[column * P + row];
    }
  }
  // (W Zb)' M (W Zb) = v'v - (Z'v)' (Z'Z)^-1 (Z'v) with v = W (y - e).
  var projected: array<f32, ${e}>;
  var scaledProduct: array<f32, ${e}>;
  for (var column = 0u; column < P; column++) {
    scaledProduct[column] = moments[momentsOffset + scalars + 7u + column] * columnScale[column];
    projected[column] = scaledProduct[column];
  }
  choleskySolve_${e}(&factor, &projected);
  var quadratic = 0.0;
  for (var column = 0u; column < P; column++) {
    quadratic += scaledProduct[column] * projected[column];
  }
  let fittedLagResidualSquares = max(fittedLagSquares - quadratic, 0.0);

  let sigmaSquared = residualSquares / n;
  let errorGradient = residualLag / sigmaSquared;
  let lagGradient = responseLag / sigmaSquared;
  let information = fittedLagResidualSquares / sigmaSquared + traceT;
  let lagStatistic = lagGradient * lagGradient / information;
  let errorStatistic = errorGradient * errorGradient / traceT;
  let robustDenominator = traceT - traceT * traceT / information;
  var robustError = getNaN();
  var robustLag = getNaN();
  var sarma = getNaN();
  if (robustDenominator > 0.0 && information > traceT) {
    let numerator = errorGradient - traceT * lagGradient / information;
    robustError = numerator * numerator / robustDenominator;
    let difference = lagGradient - errorGradient;
    robustLag = difference * difference / (information - traceT);
    sarma = lagStatistic + robustError;
  }
  writeTest(0u, lagStatistic, 1u);
  writeTest(1u, errorStatistic, 1u);
  writeTest(2u, robustLag, 1u);
  writeTest(3u, robustError, 1u);
  writeTest(4u, sarma, 2u);

  let degrees = n - f32(P);
  let scaleFactor = n / weightSum;
  let moranI = scaleFactor * residualLag / residualSquares;
  let moranExpectation = -scaleFactor * traceC / degrees;
  let traceMWMWt = squareSum - traceE - traceD + traceCCTransposed;
  let traceMWMW = crossSum - 2.0 * traceF + traceCC;
  let traceMW = -traceC;
  let moranVariance = scaleFactor * scaleFactor * (traceMWMWt + traceMWMW + traceMW * traceMW) /
    (degrees * (degrees + 2.0)) - moranExpectation * moranExpectation;
  var moranZ = getNaN();
  var moranP = getNaN();
  if (moranVariance > 0.0) {
    moranZ = (moranI - moranExpectation) / sqrt(moranVariance);
    moranP = erfcValue(abs(moranZ) * 0.70710678);
  }
  testsOut[testsOutOffset + 15u] = moranZ;
  testsOut[testsOutOffset + 16u] = 0.0;
  testsOut[testsOutOffset + 17u] = moranP;

  summaryOut[summaryOutOffset] = n;
  summaryOut[summaryOutOffset + 1u] = sigmaSquared;
  summaryOut[summaryOutOffset + 2u] = traceT;
  summaryOut[summaryOutOffset + 3u] = information;
  summaryOut[summaryOutOffset + 4u] = weightSum;
  summaryOut[summaryOutOffset + 5u] = moranI;
  summaryOut[summaryOutOffset + 6u] = moranExpectation;
  summaryOut[summaryOutOffset + 7u] = moranVariance;
  summaryOut[summaryOutOffset + 8u] = moranZ;
  summaryOut[summaryOutOffset + 9u] = moranP;
  summaryOut[summaryOutOffset + 10u] = residualLag;
  summaryOut[summaryOutOffset + 11u] = responseLag;
  summaryOut[summaryOutOffset + 12u] =
    f32(ROWS) - n;
  statusOut[statusOutOffset] = 0u;
}
`}export{w as t};