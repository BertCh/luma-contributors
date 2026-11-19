import{Q as e}from"./dist-BusamGlx.js";import{c as t,o as n,r,t as i}from"./kernel-C3tfQOn4.js";import{o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as o,f as s,p as c,x as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{a as d,g as ee,v as te}from"./spatial-context-DwZEd4EZ.js";import{A as ne,C as re,D as ie,E as ae,O as f,S as p,_ as m,a as h,b as oe,d as se,g,i as _,j as v,k as ce,l as le,n as ue,o as de,r as y,s as fe,t as b,u as pe,v as x,x as S}from"./local-weighted-least-squares-wgsl--tcHa6bd.js";import{r as C}from"./gpu-command-graph-BdzdmV6u.js";import{t as w}from"./gpu-dispatch-utils-CbRzFAgT.js";import{t as T}from"./gpu-reduction-tNgOVQ_C.js";import{c as E,n as D,r as O,u as k}from"./wgsl-common-DfyXiYB0.js";import{n as me}from"./gwr-golden-section-nodes-BFRSkOqC.js";import{n as he}from"./spatial-regression-solve-Vf-sQz9O.js";import{r as A}from"./local-collinearity-wgsl-CKjcBBZN.js";function j(e,r){let a=r.workgroupSize??256,o=w(r.operation,Math.max(r.invocationCount,1),a,e.device.limits.maxComputeWorkgroupsPerDimension),s=`${r.bindings.map((e,n)=>`const ${e.name}Offset: u32 = ${t(e.view)}u;
@group(0) @binding(${n}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;`).join(`
`)}
${r.declarations??``}
@compute @workgroup_size(${a})
fn main(@builtin(local_invocation_index) localInvocationIndex: u32,
        @builtin(workgroup_id) workgroupId: vec3u,
        @builtin(num_workgroups) workgroupCount: vec3u) {
  ${r.bindings.map(e=>`_ = &${e.name};`).join(`
  `)}
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  let index = workgroupIndex * ${a}u + localInvocationIndex;
  ${r.body}
}`;return C({id:r.id,condition:r.condition,resources:[...r.bindings.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`})),...r.extraResources??[]],workload:{operation:r.operation,...r.variant?{variant:r.variant}:{},commandCount:1,maximumInvocationCount:r.invocationCount,maximumWorkgroupCount:o.x*o.y*o.z},compile:({device:e})=>{let t=new i(e,{id:r.id,source:s,shaderLayout:{bindings:r.bindings.map((e,t)=>({name:e.name,type:e.access===`read`?`read-only-storage`:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:i})=>t.dispatch(e,{bindings:Object.fromEntries(r.bindings.map(e=>[e.name,n(e.view,()=>i(e.view))])),x:o.x,y:o.y,z:o.z}),destroy:()=>t.destroy()}}})}function ge(e,t){return[[`global reduction channels and shrinking published reduction scratch`,Math.ceil(e/64)*64*t*4],[`global reduction scalar slots`,t*64*4]]}function _e(e,t){let{id:n,tileCount:i,candidateWidth:o,source:s}=t,c=o*5,l=Math.ceil(i/64)*64,u=r(e,`${n}-reduction-channels`,`float32`,l*c),d=r(e,`${n}-reduction-totals`,`float32`,c*64),ee=Array.from({length:c},(t,n)=>({input:e.createDataView(u.buffer,{format:`float32`,length:i,byteOffset:n*l*4}),output:e.createDataView(d.buffer,{format:`float32`,length:1,byteOffset:n*64*4})}));return{totals:d,createNodes:t=>[a(e,{id:`${n}${t}-reduction-gather`,operation:`GPUGeographicallyWeightedRegression`,variant:`reduction-gather`,invocationCount:i*c,bindings:[{name:`sourceValues`,view:s,type:`f32`,access:`read`},{name:`channels`,view:u,type:`f32`,access:`read_write`}],declarations:`const TILE_COUNT: u32 = ${i}u;\nconst CHANNEL_COUNT: u32 = ${c}u;\nconst CHANNEL_STRIDE: u32 = ${l}u;`,body:`let tile = index / CHANNEL_COUNT;
  let channel = index % CHANNEL_COUNT;
  channels[channelsOffset + channel * CHANNEL_STRIDE + tile] = sourceValues[sourceValuesOffset + tile * CHANNEL_COUNT + channel];`}),...ee.flatMap((r,i)=>new T({id:`${n}${t}-sum-${i}`,...r,operation:`sum`}).getCommandNodes(e))]}}function ve(e,t){let{id:n,tileCount:i,statistics:o,summary:s}=t,c=[0,1,2].map(t=>r(e,`${n}-variance-channel-${t}`,`float32`,i)),l=[0,1,2].map(t=>r(e,`${n}-variance-total-${t}`,`float32`,1)),u=(e,t,n)=>({name:e,view:t,access:n,type:`f32`}),d=`const TILE_COUNT: u32 = ${i}u;`;return[a(e,{id:`${n}-variance-centre`,operation:`GPUGeographicallyWeightedRegression`,variant:`variance-centre`,invocationCount:i,bindings:[u(`statistics`,o,`read`),u(`counts`,c[0],`read_write`),u(`deviations`,c[1],`read_write`)],declarations:d,body:`let count = statistics[statisticsOffset + 3u * index];
  let anchor = statistics[statisticsOffset + 1u];
  counts[countsOffset + index] = count;
  deviations[deviationsOffset + index] = count * (statistics[statisticsOffset + 3u * index + 1u] - anchor);`}),...[0,1].flatMap(t=>new T({id:`${n}-variance-sum-${t}`,input:c[t],output:l[t],operation:`sum`}).getCommandNodes(e)),a(e,{id:`${n}-variance-squares`,operation:`GPUGeographicallyWeightedRegression`,variant:`variance-squares`,invocationCount:i,bindings:[u(`statistics`,o,`read`),u(`totalCount`,l[0],`read`),u(`totalDeviation`,l[1],`read`),u(`squares`,c[2],`read_write`)],declarations:d,body:`let count = statistics[statisticsOffset + 3u * index];
  let mean = statistics[statisticsOffset + 1u] + totalDeviation[totalDeviationOffset] / max(totalCount[totalCountOffset], 1.0);
  let delta = statistics[statisticsOffset + 3u * index + 1u] - mean;
  squares[squaresOffset + index] = statistics[statisticsOffset + 3u * index + 2u] + count * delta * delta;`}),...new T({id:`${n}-variance-sum-2`,input:c[2],output:l[2],operation:`sum`}).getCommandNodes(e),a(e,{id:`${n}-summarize`,operation:`GPUGeographicallyWeightedRegression`,variant:`summarize`,invocationCount:1,bindings:[u(`totalSquares`,l[2],`read`),u(`summary`,s,`read_write`)],declarations:O,body:`let squares = totalSquares[totalSquaresOffset];
  var value = getNaN();
  if (summary[summaryOffset + 5u] > 0.5 && squares > 0.0) { value = 1.0 - summary[summaryOffset] / squares; }
  summary[summaryOffset + 3u] = value;`})]}function ye(e,t){let{id:n,tileCount:i,tileCandidates:o}=t,s=[0,1,2].map(t=>r(e,`${n}-search-bound-channel-${t}`,`float32`,i)),c=[0,1,2].map(t=>r(e,`${n}-search-bound-total-${t}`,`float32`,1));return[a(e,{id:`${n}-search-bound-channels`,operation:`GPUGeographicallyWeightedRegression`,variant:`search-bound-channels`,invocationCount:i,bindings:[{name:`sourceValues`,view:o,type:`f32`,access:`read`},...s.map((e,t)=>({name:`channel${t}`,view:e,type:`f32`,access:`read_write`}))],body:s.map((e,t)=>`channel${t}[channel${t}Offset + index] = sourceValues[sourceValuesOffset + index * 3u + ${t}u];`).join(`
`)}),...s.flatMap((t,r)=>new T({id:`${n}-search-bound-${r}`,input:t,output:c[r],operation:r===0?`min`:r===1?`max`:`sum`}).getCommandNodes(e)),a(e,{id:`${n}-search-bound-record`,operation:`GPUGeographicallyWeightedRegression`,variant:`search-bound-record`,invocationCount:1,bindings:[...c.map((e,t)=>({name:`total${t}`,view:e,type:`f32`,access:`read`})),{name:`tileCandidates`,view:o,type:`f32`,access:`read_write`}],body:c.map((e,t)=>`tileCandidates[tileCandidatesOffset + ${t}u] = total${t}[total${t}Offset];`).join(`
`)})]}function be(e){let{coefficientCount:t,indexGridSize:n,hasConditionNumber:r,hasVariance:i,sphereRadius:a}=e,o=e.gaussianCutoff??null,s=!!n;return`
${s?``:k}
${he(t)}
${ue(t)}
${h(o)}
${r?A(t):``}
${n?_(n,o):``}
struct LocalFit {
  ok: bool,
  hat: f32,
  weightSum: f32,
  weightedResponse: f32,
  beta: array<f32, ${t}>,
  // diag((X'WX)^-1 X'W^2 X (X'WX)^-1) of the original (uncentered) coefficients; NaN when not
  // finite. Only filled when the squared-weight Gram matrix is accumulated.
  variance: array<f32, ${t}>
}

fn getPosition(row: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

${y(a)}

fn getWeight(distance: f32, bandwidth: f32) -> f32 {
  return getLocalKernelWeight(distance, bandwidth, getIsBisquare());
}

${n&&a===null?le():`// Keeps the maximumK smallest squared distances from row i (itself included) in ascending order.
fn selectNeighborDistances(i: u32, maximumK: u32, list: ptr<function, array<f32, MAXIMUM_NEIGHBORS>>) {
  for (var slot = 0u; slot < MAXIMUM_NEIGHBORS; slot++) {
    (*list)[slot] = SENTINEL;
  }
  let origin = getPosition(i);
  for (var row = 0u; row < ROW_COUNT; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    let squared = getSquaredDistance(origin, getPosition(row));
    if (squared < (*list)[maximumK - 1u]) {
      var slot = maximumK - 1u;
      loop {
        if (slot == 0u || (*list)[slot - 1u] <= squared) {
          break;
        }
        (*list)[slot] = (*list)[slot - 1u];
        slot = slot - 1u;
      }
      (*list)[slot] = squared;
    }
  }
}`}

// Positive bandwidth of a ladder value for the focal row of the neighbour list, or -1 when the
// value is invalid there (fixed: not positive; adaptive: k outside [2, MAXIMUM_NEIGHBORS] or more
// than the included rows).
fn getBandwidthForValue(value: f32, isAdaptive: bool, list: ptr<function, array<f32, MAXIMUM_NEIGHBORS>>) -> f32 {
  if (!isAdaptive) {
    return select(-1.0, value, value > 0.0 && isFiniteBits(value));
  }
  if (!(value >= 2.0) || value > f32(MAXIMUM_NEIGHBORS)) {
    return -1.0;
  }
  let squared = (*list)[u32(floor(value + 0.5)) - 1u];
  if (squared >= SENTINEL) {
    return -1.0;
  }
  return sqrt(squared) * ADAPTIVE_FACTOR;
}

fn fitLocation(i: u32, bandwidth: f32) -> LocalFit {
  var a: array<f32, ${t*t}>;
  var b: array<f32, ${t}>;
  var z: array<f32, ${t}>;
  ${i?`var squared: array<f32, ${t*t}>;`:``}
  var fit: LocalFit;
  fit.ok = false;
  fit.hat = 0.0;
  fit.weightSum = 0.0;
  fit.weightedResponse = 0.0;
  for (var column = 0u; column < P; column++) {
    fit.beta[column] = 0.0;
    fit.variance[column] = getNaN();
    b[column] = 0.0;
  }
  for (var entry = 0u; entry < P * P; entry++) {
    a[entry] = 0.0;
    ${i?`squared[entry] = 0.0;`:``}
  }
  let origin = getPosition(i);
  let originBase = predictorsOffset + i * PREDICTOR_COUNT;
  z[0] = 1.0;
  ${b(`let weight = getWeight(getDistance(origin, getPosition(row)), bandwidth);
    if (!(weight > 0.0)) {
      continue;
    }
    let rowBase = predictorsOffset + row * PREDICTOR_COUNT;
    for (var column = 0u; column < PREDICTOR_COUNT; column++) {
      z[column + 1u] = predictors[rowBase + column] - predictors[originBase + column];
    }
    let y = response[responseOffset + row];
    for (var r = 0u; r < P; r++) {
      let weighted = weight * z[r];
      b[r] = b[r] + weighted * y;
      for (var c = 0u; c <= r; c++) {
        a[r * P + c] = a[r * P + c] + weighted * z[c];
        ${i?`squared[r * P + c] = squared[r * P + c] + weighted * weight * z[c];`:``}
      }
    }`,s)}
  fit.weightSum = a[0];
  fit.weightedResponse = b[0];
  var scale: array<f32, ${t}>;
  if (!factorEquilibrated_${t}(&a, &scale)) {
    return fit;
  }
  let inverse00 = inverseDiagonalEquilibrated_${t}(&a, &scale, 0u);
  // b holds X'Wy of the centered design; the solve is in place and unscales the coefficients.
  solveEquilibrated_${t}(&a, &scale, &b);
  var finite = isFiniteBits(inverse00);
  for (var r = 0u; r < P; r++) {
    fit.beta[r] = b[r];
    finite = finite && isFiniteBits(fit.beta[r]);
  }
  fit.hat = inverse00;
  fit.ok = finite && fit.hat >= 0.0;
  ${i?`if (fit.ok) {
    // C C' = A^-1 B A^-1 with A = X'WX (centered design) and B = X'W^2 X. Column c of A^-1 comes
    // from the equilibrated factor: A^-1 e_c = s_c * (s o solve(e_c)). The original intercept is
    // gamma_0 - sum x_ic gamma_c, so its direction is t = (1, -x_i) and A^-1 t = s o solve(s o t);
    // variance_c = w' B w.
    for (var column = 0u; column < P; column++) {
      var w: array<f32, ${t}>;
      for (var r = 0u; r < P; r++) {
        w[r] = 0.0;
      }
      var columnScale = scale[column];
      if (column == 0u) {
        columnScale = 1.0;
        w[0] = scale[0];
        for (var r = 1u; r < P; r++) {
          w[r] = -scale[r] * predictors[originBase + r - 1u];
        }
      } else {
        w[column] = 1.0;
      }
      choleskySolve_${t}(&a, &w);
      for (var r = 0u; r < P; r++) {
        w[r] = scale[r] * w[r] * columnScale;
      }
      var quadratic = 0.0;
      for (var r = 0u; r < P; r++) {
        quadratic = quadratic + w[r] * w[r] * squared[r * P + r];
        for (var c = 0u; c < r; c++) {
          quadratic = quadratic + 2.0 * w[r] * w[c] * squared[r * P + c];
        }
      }
      fit.variance[column] = select(getNaN(), quadratic, quadratic >= 0.0 && isFiniteBits(quadratic));
    }
  }`:``}
  return fit;
}

// Weighted R^2 of the local fit over the neighborhood of row i: the residuals are those of local
// model i evaluated at every neighbor j (not each neighbor's own GWR residual).
fn getLocalR2(i: u32, bandwidth: f32, fit: LocalFit) -> f32 {
  let origin = getPosition(i);
  let originBase = predictorsOffset + i * PREDICTOR_COUNT;
  let weightedMean = fit.weightedResponse / fit.weightSum;
  var residualSum = 0.0;
  var totalSum = 0.0;
  ${b(`let weight = getWeight(getDistance(origin, getPosition(row)), bandwidth);
    if (!(weight > 0.0)) {
      continue;
    }
    let rowBase = predictorsOffset + row * PREDICTOR_COUNT;
    var prediction = fit.beta[0];
    for (var column = 0u; column < PREDICTOR_COUNT; column++) {
      prediction = prediction + fit.beta[column + 1u] * (predictors[rowBase + column] - predictors[originBase + column]);
    }
    let y = response[responseOffset + row];
    residualSum = residualSum + weight * (y - prediction) * (y - prediction);
    totalSum = totalSum + weight * (y - weightedMean) * (y - weightedMean);`,s)}
  return select(getNaN(), 1.0 - residualSum / totalSum, totalSum > 0.0);
}
${r?M(t,s):``}
`}function M(e,t){let n=e;return`
fn getLocalConditionNumber(i: u32, bandwidth: f32) -> f32 {
  var factor: array<f32, ${n*n}>;
  var designRow: array<f32, ${n}>;
  let origin = getPosition(i);
  ${b(`let weight = getWeight(getDistance(origin, getPosition(row)), bandwidth);
    if (!(weight > 0.0)) {
      continue;
    }
    let rowBase = predictorsOffset + row * PREDICTOR_COUNT;
    designRow[0] = weight;
    for (var column = 0u; column < PREDICTOR_COUNT; column++) {
      designRow[column + 1u] = weight * predictors[rowBase + column];
    }
    updateDesignFactor_${n}(&factor, &designRow);`,t)}
  // Unit column norms: the squared norm of column c of R is the weighted sum of squares of column c.
  for (var c = 0u; c < P; c++) {
    var squaredNorm = 0.0;
    for (var r = 0u; r <= c; r++) {
      squaredNorm = squaredNorm + factor[r * P + c] * factor[r * P + c];
    }
    if (!(squaredNorm > 0.0) || !isFiniteBits(squaredNorm)) {
      return getNaN();
    }
    let scale = inverseSqrt(squaredNorm);
    for (var r = 0u; r <= c; r++) {
      factor[r * P + c] = factor[r * P + c] * scale;
    }
  }
  var vectors: array<f32, ${n*n}>;
  var singularValues: array<f32, ${n}>;
  // The sweeps are bounded; an unconverged result is still the best available estimate.
  getDesignSingularValues_${n}(&factor, &vectors, &singularValues);
  var largest = 0.0;
  var smallest = 3.0e38;
  for (var c = 0u; c < P; c++) {
    largest = max(largest, singularValues[c]);
    smallest = min(smallest, singularValues[c]);
  }
  return largest / max(smallest, 1.0e-30);
}
`}var N=2e5,P=3e3,F=40;function I(){return`
const GWR_T_EPSILON: f32 = 3.0e-7;
const GWR_T_MINIMUM: f32 = 1.0e-30;
const GWR_T_SQRT_PI: f32 = 1.7724538509055159;
const GWR_T_MAXIMUM_DF: f32 = ${N}.0;

// log(1 + z) without the rounding of 1 + z (z small and positive).
fn gwrLog1p(z: f32) -> f32 {
  let u = 1.0 + z;
  if (u == 1.0) {
    return z;
  }
  return log(u) * z / (u - 1.0);
}

// Gamma(a + 1/2) / Gamma(a) for a > 0: recurrence up to a >= 10, then the asymptotic series.
fn gwrGammaHalfRatio(a: f32) -> f32 {
  var s = a;
  var factor = 1.0;
  for (var step = 0u; step < 12u; step++) {
    if (s >= 10.0) {
      break;
    }
    factor = factor * s / (s + 0.5);
    s = s + 1.0;
  }
  let inverse = 1.0 / s;
  let series = 1.0 - 0.125 * inverse +
    inverse * inverse * (1.0 / 128.0 + inverse * (5.0 / 1024.0 -
      inverse * (21.0 / 32768.0 + inverse * (399.0 / 262144.0 - inverse * (869.0 / 4194304.0)))));
  return factor * sqrt(s) * series;
}

// Lentz continued fraction of the incomplete beta function (Numerical Recipes betacf).
fn gwrBetaContinuedFraction(a: f32, b: f32, x: f32) -> f32 {
  let qab = a + b;
  let qap = a + 1.0;
  let qam = a - 1.0;
  var c = 1.0;
  var d = 1.0 - qab * x / qap;
  if (abs(d) < GWR_T_MINIMUM) {
    d = GWR_T_MINIMUM;
  }
  d = 1.0 / d;
  var h = d;
  for (var m = 1u; m <= ${P}u; m++) {
    let fm = f32(m);
    let m2 = 2.0 * fm;
    var aa = fm * (b - fm) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (abs(d) < GWR_T_MINIMUM) {
      d = GWR_T_MINIMUM;
    }
    c = 1.0 + aa / c;
    if (abs(c) < GWR_T_MINIMUM) {
      c = GWR_T_MINIMUM;
    }
    d = 1.0 / d;
    h = h * d * c;
    aa = -(a + fm) * (qab + fm) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (abs(d) < GWR_T_MINIMUM) {
      d = GWR_T_MINIMUM;
    }
    c = 1.0 + aa / c;
    if (abs(c) < GWR_T_MINIMUM) {
      c = GWR_T_MINIMUM;
    }
    d = 1.0 / d;
    let delta = d * c;
    h = h * delta;
    if (abs(delta - 1.0) < GWR_T_EPSILON) {
      break;
    }
  }
  return h;
}

// P(|T| >= t) for a Student t variate with df degrees of freedom.
fn gwrStudentTwoSidedTail(t: f32, degreesOfFreedom: f32) -> f32 {
  if (!(t > 0.0)) {
    return 1.0;
  }
  let df = min(degreesOfFreedom, GWR_T_MAXIMUM_DF);
  let a = 0.5 * df;
  let ratio = t * t / df;
  let x = 1.0 / (1.0 + ratio);
  let y = ratio / (1.0 + ratio);
  let front = exp(-a * gwrLog1p(ratio)) * sqrt(y) * gwrGammaHalfRatio(a) / GWR_T_SQRT_PI;
  if (x < (a + 1.0) / (a + 2.5)) {
    return front * gwrBetaContinuedFraction(a, 0.5, x) / a;
  }
  return 1.0 - 2.0 * front * gwrBetaContinuedFraction(0.5, a, y);
}

// The t >= 0 with P(|T| >= t) = tail.
fn gwrStudentCriticalValue(tail: f32, degreesOfFreedom: f32) -> f32 {
  if (!(tail > 0.0) || !(degreesOfFreedom >= 1.0) || tail > 3.0e38) {
    return getNaN();
  }
  if (tail >= 1.0) {
    return 0.0;
  }
  var low = 0.0;
  var high = 1.0;
  for (var step = 0u; step < 60u; step++) {
    if (gwrStudentTwoSidedTail(high, degreesOfFreedom) < tail) {
      break;
    }
    low = high;
    high = high * 2.0;
  }
  for (var step = 0u; step < ${F}u; step++) {
    let middle = 0.5 * (low + high);
    if (gwrStudentTwoSidedTail(middle, degreesOfFreedom) < tail) {
      high = middle;
    } else {
      low = middle;
    }
  }
  return 0.5 * (low + high);
}
`}var L=4096,R=1e-10,xe=32;function Se(){return`${I()}
${D}
const GWR_QUANTILE_MINIMUM_DF: f32 = ${L}.0;
const GWR_QUANTILE_MINIMUM_TAIL: f32 = ${R};
fn gwrLargeSampleCriticalValue(tail: f32, degreesOfFreedom: f32) -> f32 {
  if (!(degreesOfFreedom >= GWR_QUANTILE_MINIMUM_DF) ||
      !(tail >= GWR_QUANTILE_MINIMUM_TAIL) || !(tail < 1.0)) {
    return gwrStudentCriticalValue(tail, degreesOfFreedom);
  }
  var lower = 0.0;
  var upper = 8.0;
  for (var iteration = 0u; iteration < ${xe}u; iteration++) {
    let middle = 0.5 * (lower + upper);
    if (erfcValue(middle * 0.7071067811865476) < tail) {
      upper = middle;
    } else {
      lower = middle;
    }
  }
  let normal = 0.5 * (lower + upper);
  let squared = normal * normal;
  let inverse = 1.0 / min(degreesOfFreedom, GWR_T_MAXIMUM_DF);
  let first = normal * (squared + 1.0) / 4.0;
  let second = normal * (5.0 * squared * squared + 16.0 * squared + 3.0) / 96.0;
  let third = normal * (3.0 * squared * squared * squared + 19.0 * squared * squared + 17.0 * squared - 15.0) / 384.0;
  return normal + inverse * (first + inverse * (second + inverse * third));
}
`}var z=Object.freeze({...d,measure:`distance`}),B=`GPUGeographicallyWeightedRegression`,V=256,Ce=6,H=5,U=8,W=262144,we=1048576,G={SELECTED_INDEX:0,SELECTED_VALUE:1,KERNEL:2,MODE:3,SIGMA_SQUARED:4,CRITICAL_T:5,FOCUS_BASE:6},K=class{id;props;rowCount;coefficientCount;maximumBandwidthCount;maximumNeighborCount;indexGridSize;spatialContext;refinementPassCount;bandwidthSearch;maximumSearchIterations;gaussianCutoff;candidateWidth;candidateTableRowCount;focusRowCount;estimatedCommandNodeCount;constructor(e){this.id=e.id??`geographically-weighted-regression`,this.props=e;let t=this.id,{predictorCount:n}=e;f(t,e.output);let r=ae(e.output);s(t,`predictorCount`,n,1,11),this.coefficientCount=n+1,this.spatialContext=te(t,e.spatialContext,z),this.spatialContext.metric===`great-circle`&&v(t,`spatialContext.sphereRadius`,ee(this.spatialContext)),this.maximumBandwidthCount=e.maximumBandwidthCount??32,s(t,`maximumBandwidthCount`,this.maximumBandwidthCount,1,32),this.maximumNeighborCount=e.maximumNeighborCount??128,s(t,`maximumNeighborCount`,this.maximumNeighborCount,1,128);let i=e.refinement??!0;typeof i==`number`&&s(t,`refinement`,i,0,4);let a=e.bandwidthSearch??`ladder`;if(a!==`ladder`&&a!==`golden-section`)throw Error(`${t} bandwidthSearch must be 'ladder' or 'golden-section'`);this.bandwidthSearch=a;let l=a===`golden-section`;if(l){if(this.maximumBandwidthCount<2)throw Error(`${t} golden-section search needs maximumBandwidthCount >= 2`);if(this.maximumNeighborCount<2)throw Error(`${t} golden-section search needs maximumNeighborCount >= 2`)}this.maximumSearchIterations=l?e.maximumSearchIterations??30:0,l&&s(t,`maximumSearchIterations`,this.maximumSearchIterations,1,64);let{gaussianCutoff:d}=e;if(d!==void 0&&v(t,`gaussianCutoff`,d),e.gaussianMode!==void 0&&e.gaussianMode!==`exact`&&e.gaussianMode!==`truncated`)throw Error(`${t} gaussianMode must be exact or truncated`);if(e.gaussianMode===`exact`&&d!==void 0)throw Error(`${t} gaussianMode exact requires gaussianCutoff to be omitted`);if(this.refinementPassCount=this.maximumBandwidthCount<2||l?0:i===!0?2:i===!1?0:i,this.candidateWidth=l?2:this.maximumBandwidthCount,this.candidateTableRowCount=l?this.maximumSearchIterations+1:(1+this.refinementPassCount)*this.maximumBandwidthCount,o(e.positions,[`float32x2`],`${t} positions`),this.rowCount=e.positions.length,this.rowCount<1)throw Error(`${t} needs at least one row`);if(this.spatialContext.metric===`great-circle`){if(e.indexGridSize)throw Error(`${t} indexGridSize is planar-only; omit it for a great-circle context`);this.indexGridSize=null}else if(e.indexGridSize===!1)this.indexGridSize=null;else if(e.indexGridSize){let[n,r]=e.indexGridSize;if(!Number.isInteger(n)||!Number.isInteger(r)||n<1||r<1||n*r>=4294967295)throw Error(`${t} indexGridSize must be two positive integers with columns * rows < 2^32 - 1`);this.indexGridSize=[n,r]}else this.indexGridSize=fe(this.rowCount)??null;if(!this.indexGridSize&&this.rowCount>65536)throw Error(`${t} supports at most ${x} rows without a grid index`);s(t,`positions.length`,this.rowCount,1,this.indexGridSize?m:x),e.gaussianMode===`exact`&&s(t,`positions.length for gaussianMode exact`,this.rowCount,1,g),this.gaussianCutoff=d??(e.gaussianMode===`truncated`||this.rowCount>1048576?8:null),this.focusRowCount=Math.min(this.rowCount,W);let ie=Math.ceil(this.rowCount/this.focusRowCount),p=l?this.maximumSearchIterations:1+this.refinementPassCount;if(this.estimatedCommandNodeCount=2080+p*(5*ie+3+this.candidateWidth*H*8)+8*ie+32,c(t,this.estimatedCommandNodeCount),o(e.predictors,[`float32`],`${t} predictors`),e.predictors.length!==this.rowCount*n)throw Error(`${t} predictors length must equal rows * predictorCount`);if(o(e.response,[`float32`],`${t} response`),e.response.length!==this.rowCount)throw Error(`${t} response length must equal the row count`);if(e.mask&&(o(e.mask,[`uint32`],`${t} mask`),e.mask.length!==this.rowCount))throw Error(`${t} mask length must equal the row count`);o(e.parameters,[`float32`],`${t} parameters`);let h=re(this.maximumBandwidthCount);if(e.parameters.length<h)throw Error(`${t} parameters must hold ${h} float32 values`);let oe=this.candidateTableRowCount*7,se=[[`coefficients`,this.rowCount*this.coefficientCount],[`approximation`,1],[`standardErrors`,this.rowCount*this.coefficientCount],[`tValues`,this.rowCount*this.coefficientCount],[`significant`,this.rowCount*this.coefficientCount],[`localR2`,this.rowCount],[`fitted`,this.rowCount],[`residuals`,this.rowCount],[`hatDiagonal`,this.rowCount],[`localStatus`,this.rowCount],[`localConditionNumber`,this.rowCount],[`bandwidthScores`,this.maximumBandwidthCount],[`bandwidthSingularRowCounts`,this.maximumBandwidthCount],[`candidateTable`,oe],[`selectedBandwidth`,2],[`searchState`,12],[`summary`,12]];if(r.searchState&&!l)throw Error(`${t} output.searchState needs bandwidthSearch 'golden-section'`);let _=new Set([`localStatus`,`significant`,`bandwidthSingularRowCounts`,`approximation`]);for(let[n,i]of se){let a=r[n],s=n===`approximation`&&e.output.status?.approximation?`status.approximation`:n;if(!a){if(n===`coefficients`)throw Error(`${t} output.coefficients is required`)}else if(_.has(n)?o(a,[`uint32`],`${t} output.${s}`):o(a,[`float32`],`${t} output.${s}`),a.length<i)throw Error(`${t} output.${s} must hold at least ${i} values`)}ne(t,se.map(([e])=>r[e]),[{writes:[r.selectedBandwidth,r.summary],reads:[r.candidateTable,r.searchState]},{writes:[r.candidateTable,r.searchState]},{writes:[r.bandwidthScores,r.bandwidthSingularRowCounts],reads:[r.candidateTable]},{writes:[r.localR2,r.fitted,r.residuals,r.hatDiagonal,r.localStatus,r.localConditionNumber]},{writes:[r.standardErrors,r.tValues,r.significant]}]),u(t,se.map(([e])=>r[e]),[e.positions,e.predictors,e.response,e.mask,e.parameters])}hasInference(){let{output:e}=this.props;return!!(e.standardErrors||e.tValues||e.significant)}hasTValues(){let{output:e}=this.props;return!!(e.tValues||e.significant)}getRowStateWordCount(){let{rowCount:e,coefficientCount:t}=this;return e+this.focusRowCount*Ce+(this.hasInference()?this.focusRowCount*t:0)+(this.hasTValues()?this.focusRowCount*t:0)+U}getCommandNodes(t){let{props:n,id:i,rowCount:o,coefficientCount:s,maximumBandwidthCount:c,maximumNeighborCount:u,indexGridSize:d,refinementPassCount:te,maximumSearchIterations:ne,candidateWidth:f,candidateTableRowCount:m,gaussianCutoff:h,focusRowCount:g}=this,_=g<o,v=o>we,le=Math.ceil(o/g),ue=Math.ceil(g/V),y=this.bandwidthSearch===`golden-section`,{predictorCount:fe,positions:b,predictors:x,response:S,parameters:C}=n,w=ae(n.output);l(i,t,[b,x,S,n.mask,C,...Object.values(w)]),ce(t.device,i,[b,x,S,n.mask,C,...Object.values(w)],[[`row-state`,this.getRowStateWordCount()*4],[`selection`,8],[`approximation`,4],[`focus indirect dispatch`,24],...v?[...ge(Math.ceil(o/V),f*H),...ge(Math.ceil(o/V),3)]:[],[`summary`,48],...y?[[`search-state`,48],[`search gate`,36]]:[],[`candidate-scratch`,g*f*8],[`tile-candidates`,Math.ceil(o/V)*f*H*4],[`tile-statistics`,Math.ceil(o/V)*3*4],[`ladders`,(1+(y?ne:1+te)*re(c))*4],[`candidate-table`,m*7*4],...y?[[`search-bounds-scratch`,o*8]]:[],...ie(o,d)]);let T=y?ne:1+te,E=Math.ceil(o/V),D=this.hasInference(),O=this.hasTValues(),k=s,he=re(c),A=(e,n,a)=>r(t,`${i}-${e}`,n,Math.max(a,1)),M=A(`row-state`,`uint32`,this.getRowStateWordCount()),N=A(`candidate-scratch`,`float32`,g*f*2),P=A(`tile-candidates`,`float32`,E*f*H),F=v?_e(t,{id:i,tileCount:E,candidateWidth:f,source:P}):null,I=w.candidateTable??A(`candidate-table`,`float32`,m*7),L=A(`ladders`,`float32`,1+T*he),R=A(`tile-statistics`,`float32`,E*3),xe=w.selectedBandwidth??A(`selected-bandwidth`,`float32`,2),z=y?w.searchState??A(`search-state`,`float32`,12):null,U=w.summary??A(`summary`,`float32`,12),W=d?A(`grid-positions`,`float32x2`,o):null,K=d&&W?de(t,{id:i,operation:B,positions:W,rowCount:o,gridSize:d}):null,Oe=K?.cellOffsets??null,ke=K?.sortedIds??null,q=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),J=(e,t,n)=>({name:e,view:t,type:n,access:`read_write`}),Ae=o+g*Ce,je=Ae+(D?g*k:0),Me=je+(O?g*k:0),Ne=this.spatialContext.metric===`great-circle`,Pe=Ne?ee(this.spatialContext):null,Y=`const ROW_COUNT: u32 = ${o}u;
const FOCUS_ROWS: u32 = ${g}u;
const PREDICTOR_COUNT: u32 = ${fe}u;
const P: u32 = ${k}u;
const LADDER: u32 = ${f}u;
const TABLE_ROWS: u32 = ${m}u;
const MAXIMUM_NEIGHBORS: u32 = ${u}u;
const TILE_ROWS: u32 = ${V}u;
const TILE_COUNT: u32 = ${E}u;
const LOCAL_STRIDE: u32 = ${Ce}u;
const TILE_CANDIDATE_STRIDE: u32 = ${H}u;
const TABLE_STRIDE: u32 = 7u;
// Layout of the rowValid binding: validity flags, local results, standard errors, t values, tail.
const LOCALS_BASE: u32 = ${o}u;
const STANDARD_ERROR_BASE: u32 = ${Ae}u;
const T_VALUE_BASE: u32 = ${je}u;
const TAIL_BASE: u32 = ${Me}u;
const TAIL_FOCUS_BASE: u32 = ${G.FOCUS_BASE}u;
const TAIL_SELECTED_INDEX: u32 = ${G.SELECTED_INDEX}u;
const TAIL_SELECTED_VALUE: u32 = ${G.SELECTED_VALUE}u;
const TAIL_KERNEL: u32 = ${G.KERNEL}u;
const TAIL_MODE: u32 = ${G.MODE}u;
const TAIL_SIGMA_SQUARED: u32 = ${G.SIGMA_SQUARED}u;
const TAIL_CRITICAL_T: u32 = ${G.CRITICAL_T}u;
const HEADER: u32 = 4u;
const PARAMETER_LENGTH: u32 = ${he}u;
const SENTINEL: f32 = 3.0e38;
const ADAPTIVE_FACTOR: f32 = ${pe};
const MINIMUM_VARIANCE: f32 = ${oe};
const LN_TWO_PI: f32 = 1.8378770664093453;
${Ee}`,Fe=d?[q(`cellOffsets`,Oe,`u32`),q(`sortedIds`,ke,`u32`)]:[],Ie=n.mask,Le=d?`if (index >= getFocalCount()) {
    return;
  }
  let focal = getFocalRow(index);`:`let focal = index;`,X=`rowValid[rowValidOffset + TAIL_BASE + TAIL_FOCUS_BASE]`,Re=_?`let focal = ${X} + index;
  if (focal >= ROW_COUNT) { return; }`:Le,ze=`${Y}
fn getParameter(slot: u32) -> f32 {
  let passCursor = u32(params[paramsOffset]);
  return params[paramsOffset + 1u + passCursor * PARAMETER_LENGTH + slot];
}
fn getIsBisquare() -> bool {
  return getParameter(0u) > 0.5;
}
fn getIsAdaptive() -> bool {
  return getParameter(1u) > 0.5;
}
fn getCandidateCount() -> u32 {
  return min(u32(max(getParameter(2u), 0.0)), LADDER);
}`,Be=`${ze}
${be({coefficientCount:k,indexGridSize:d,hasConditionNumber:!1,hasVariance:!1,sphereRadius:Pe,gaussianCutoff:h})}
// Largest requested k among valid adaptive candidates, clamped to [1, MAXIMUM_NEIGHBORS].
fn getMaximumK(candidateCount: u32) -> u32 {
  var maximumK = 1u;
  for (var candidate = 0u; candidate < candidateCount; candidate++) {
    let k = getParameter(HEADER + candidate);
    if (k >= 1.0 && k <= f32(MAXIMUM_NEIGHBORS)) {
      maximumK = max(maximumK, u32(floor(k + 0.5)));
    }
  }
  return min(maximumK, MAXIMUM_NEIGHBORS);
}`,Ve=`${Y}
fn getIsBisquare() -> bool {
  return bitcast<f32>(rowValid[rowValidOffset + TAIL_BASE + TAIL_KERNEL]) > 0.5;
}
fn getIsAdaptive() -> bool {
  return bitcast<f32>(rowValid[rowValidOffset + TAIL_BASE + TAIL_MODE]) > 0.5;
}
${be({coefficientCount:k,indexGridSize:d,hasConditionNumber:!!w.localConditionNumber,hasVariance:D,sphereRadius:Pe,gaussianCutoff:h})}`,Z=[];Z.push(a(t,{id:`${i}-validate`,operation:B,variant:`validate`,bindings:[q(`positions`,b,`f32`),q(`predictors`,x,`f32`),q(`response`,S,`f32`),...Ie?[q(`rowMask`,Ie,`u32`)]:[],J(`rowValid`,M,`u32`)],invocationCount:o,declarations:Y,body:`var isValid = true;
  ${Ie?`isValid = rowMask[rowMaskOffset + index] != 0u;`:``}
  if (!isFiniteBits(positions[positionsOffset + 2u * index]) ||
    !isFiniteBits(positions[positionsOffset + 2u * index + 1u]) ||
    !isFiniteBits(response[responseOffset + index])) {
    isValid = false;
  }
  ${Ne?`if (!(abs(positions[positionsOffset + 2u * index + 1u]) <= 90.0)) {
    isValid = false;
  }`:``}
  for (var column = 0u; column < PREDICTOR_COUNT; column++) {
    if (!isFiniteBits(predictors[predictorsOffset + index * PREDICTOR_COUNT + column])) {
      isValid = false;
    }
  }
  rowValid[rowValidOffset + index] = select(0u, 1u, isValid);`})),K&&W&&Z.push(a(t,{id:`${i}-grid-positions`,operation:B,variant:`grid-positions`,bindings:[q(`positions`,b,`f32`),q(`rowValid`,M,`u32`),J(`gridPositions`,W,`f32`)],invocationCount:o,declarations:Y,body:`let included = rowValid[rowValidOffset + index] != 0u;
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  gridPositions[gridPositionsOffset + index * 2u] =
    select(invalid, positions[positionsOffset + index * 2u], included);
  gridPositions[gridPositionsOffset + index * 2u + 1u] =
    select(invalid, positions[positionsOffset + index * 2u + 1u], included);`}),...K.nodes),Z.push(a(t,{id:`${i}-ladder-init`,operation:B,variant:`ladder-init`,bindings:[q(`params`,C,`f32`),J(`ladders`,L,`f32`)],invocationCount:1,declarations:Y,body:`ladders[laddersOffset] = 0.0;
  for (var slot = 0u; slot < PARAMETER_LENGTH; slot++) {
    ladders[laddersOffset + 1u + slot] = params[paramsOffset + slot];
  }`}));let He=[q(`positions`,b,`f32`),q(`predictors`,x,`f32`),q(`response`,S,`f32`),q(`rowValid`,M,`u32`),q(`params`,L,`f32`),...Fe],Q=y&&z?me(t,{id:i,operation:B,rowCount:o,tileRows:V,tileCount:E,candidateWidth:f,maximumIterations:ne,tableRowCount:m,common:v?Y.replace(`const TILE_COUNT: u32 = ${E}u;`,`const TILE_COUNT: u32 = 1u;`):Y,candidateDeclarations:Be,focalRowWGSL:Le,dispatchShapes:[[g],[ue*f],[f]],hasGrid:!!d,candidateBindings:He,rowValid:M,ladders:L,parameters:C,candidateScratch:_?A(`search-bounds-scratch`,`float32`,o*2):N,tileCandidates:P,candidateTable:I,searchState:z}):null;Q&&(v?Z.push(...Q.setupNodes.slice(0,2),...ye(t,{id:i,tileCount:E,tileCandidates:P}),...Q.setupNodes.slice(2)):Z.push(...Q.setupNodes));let Ue=(e,n)=>a(t,{id:`${i}-${e}`,operation:B,variant:n?`focus-reset`:`focus-advance`,bindings:[J(`rowValid`,M,`u32`)],invocationCount:1,declarations:Y,body:`rowValid[rowValidOffset + TAIL_BASE + TAIL_FOCUS_BASE] = ${n?`0u`:`${X} + FOCUS_ROWS`};`}),$=_&&Q?r(t,`${i}-focus-dispatch`,`uint32`,6,e.STORAGE|e.INDIRECT):null,We=e=>({condition:{id:`${i}-focus-gate-${e}`,source:`gpu`,mode:`indirect`,buffer:$.buffer,byteOffset:$.byteOffset+e*12},extraResources:[{buffer:$,usage:`indirect`}]});for(let e=0;e<T;e++){let n=e===0?``:y?`-search-${e}`:`-refine-${e}`,r=Q&&e>0?Q.gate:null;Q&&e>0?Z.push(Q.createIterationNode(e)):e>0&&Z.push(a(t,{id:`${i}-refine-${e}-ladder`,operation:B,variant:`refine-ladder`,bindings:[J(`ladders`,L,`f32`),q(`table`,I,`f32`)],invocationCount:1,declarations:`${Y}
${De}`,body:`refineLadder();`})),_&&Z.push(Ue(`${n}-focus-reset`,!0));for(let e=0;e<le;e++){let s=_?`-batch-${e}`:``,c=Math.min(g,o-e*g),l=Math.ceil(c/V);r&&$&&Z.push(a(t,{id:`${i}${n}${s}-focus-dispatch`,operation:B,variant:`focus-dispatch`,invocationCount:1,bindings:[q(`rowValid`,M,`u32`),q(`searchDispatch`,Q.gate.view,`u32`),J(`focusDispatch`,$,`u32`)],declarations:Y,body:`let firstRow = ${X};
  let rows = min(FOCUS_ROWS, ROW_COUNT - min(firstRow, ROW_COUNT));
  let enabled = searchDispatch[searchDispatchOffset] > 0u;
  focusDispatch[focusDispatchOffset] = select(0u, (rows + 255u) / 256u, enabled);
  focusDispatch[focusDispatchOffset + 1u] = 1u;
  focusDispatch[focusDispatchOffset + 2u] = 1u;
  let tileInvocations = ((rows + TILE_ROWS - 1u) / TILE_ROWS) * LADDER;
  focusDispatch[focusDispatchOffset + 3u] = select(0u, (tileInvocations + 255u) / 256u, enabled);
  focusDispatch[focusDispatchOffset + 4u] = 1u;
  focusDispatch[focusDispatchOffset + 5u] = 1u;`})),Z.push((_?j:a)(t,{id:`${i}${n}${s}-candidates`,operation:B,variant:`candidates`,...r&&$?We(0):r?.getGate(0),bindings:[...He,J(`candidateScratch`,N,`f32`)],invocationCount:c,workgroupSize:_?256:void 0,declarations:Be,body:`${Re}
  let candidateCount = getCandidateCount();
  if (candidateCount == 0u || rowValid[rowValidOffset + focal] == 0u) {
    // Excluded rows are skipped by the tile sums, so their scratch is never read.
    return;
  }
  var neighbors: array<f32, MAXIMUM_NEIGHBORS>;
  let isAdaptive = getIsAdaptive();
  if (isAdaptive) {
    selectNeighborDistances(focal, getMaximumK(candidateCount), &neighbors);
  }
  for (var candidate = 0u; candidate < candidateCount; candidate++) {
    // Marker in the hat slot: -1 singular local system, -2 invalid bandwidth for this row.
    let slot = candidateScratchOffset + (candidate * FOCUS_ROWS + ${_?`index`:`focal`}) * 2u;
    candidateScratch[slot] = 0.0;
    candidateScratch[slot + 1u] = -1.0;
    let bandwidth = getBandwidthForValue(getParameter(HEADER + candidate), isAdaptive, &neighbors);
    if (bandwidth <= 0.0) {
      candidateScratch[slot + 1u] = -2.0;
      continue;
    }
    let fit = fitLocation(focal, bandwidth);
    if (!fit.ok) {
      continue;
    }
    let residual = response[responseOffset + focal] - fit.beta[0];
    candidateScratch[slot] = residual * residual;
    candidateScratch[slot + 1u] = fit.hat;
  }`})),Z.push((_?j:a)(t,{id:`${i}${n}${s}-tile-candidates`,operation:B,variant:`tile-candidates`,...r&&$?We(1):r?.getGate(1),bindings:[q(`candidateScratch`,N,`f32`),q(`rowValid`,M,`u32`),J(`tileCandidates`,P,`f32`)],invocationCount:l*f,workgroupSize:_?256:void 0,declarations:Y,body:`let tile = index / LADDER + ${_?`${X} / TILE_ROWS`:`0u`};
  if (tile >= TILE_COUNT) { return; }
  let candidate = index % LADDER;
  let firstRow = tile * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var residualSum = 0.0;
  var traceSum = 0.0;
  var singular = 0.0;
  var invalid = 0.0;
  var included = 0.0;
  for (var row = firstRow; row < endRow; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    included = included + 1.0;
    let slot = candidateScratchOffset + (candidate * FOCUS_ROWS + row - ${_?X:`0u`}) * 2u;
    let hat = candidateScratch[slot + 1u];
    if (hat < -1.5) {
      invalid = invalid + 1.0;
    } else if (hat < 0.0) {
      singular = singular + 1.0;
    } else {
      residualSum = residualSum + candidateScratch[slot];
      traceSum = traceSum + hat;
    }
  }
  let base = tileCandidatesOffset + (tile * LADDER + candidate) * TILE_CANDIDATE_STRIDE;
  tileCandidates[base] = residualSum;
  tileCandidates[base + 1u] = traceSum;
  tileCandidates[base + 2u] = singular;
  tileCandidates[base + 3u] = invalid;
  tileCandidates[base + 4u] = included;`})),_&&Z.push(Ue(`${n}${s}-focus-advance`,!1))}F&&Z.push(...F.createNodes(n)),Z.push(a(t,{id:`${i}${n}-select`,operation:B,variant:`select`,...r?.getGate(2),bindings:[q(`tileCandidates`,F?.totals??P,`f32`),q(`params`,L,`f32`),J(`table`,I,`f32`)],invocationCount:f,declarations:ze,body:`let candidate = index;
  let passCursor = u32(params[paramsOffset]);
  let nan = getNaN();
  ${y?`// One table row per evaluation, in order: pass 0 fills rows 0 and 1, pass p the row p + 1.
  // Rows of unused candidates keep the NaN fill of the search initialization.
  if (candidate >= getCandidateCount()) {
    return;
  }
  let base = tableOffset + select(passCursor + 1u, candidate, passCursor == 0u) * TABLE_STRIDE;`:`let base = tableOffset + (passCursor * LADDER + candidate) * TABLE_STRIDE;
  if (candidate >= getCandidateCount()) {
    for (var field = 0u; field < TABLE_STRIDE; field++) {
      table[base + field] = nan;
    }
    return;
  }`}
  ${F?`let reductionBase = tileCandidatesOffset + candidate * ${H*64}u;
  let residualSum = tileCandidates[reductionBase];
  let traceSum = tileCandidates[reductionBase + 64u];
  let singular = tileCandidates[reductionBase + 128u];
  let invalid = tileCandidates[reductionBase + 192u];
  let included = tileCandidates[reductionBase + 256u];`:`  var residualSum = 0.0;
  var traceSum = 0.0;
  var singular = 0.0;
  var invalid = 0.0;
  var included = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let tileBase = tileCandidatesOffset + (tile * LADDER + candidate) * TILE_CANDIDATE_STRIDE;
    residualSum = residualSum + tileCandidates[tileBase];
    traceSum = traceSum + tileCandidates[tileBase + 1u];
    singular = singular + tileCandidates[tileBase + 2u];
    invalid = invalid + tileCandidates[tileBase + 3u];
    included = included + tileCandidates[tileBase + 4u];
  }
`}
  // AICc over the non-singular rows; a row with an invalid bandwidth voids the candidate.
  let count = included - singular;
  let denominator = count - 2.0 - traceSum;
  var score = nan;
  if (invalid == 0.0 && count > 0.0 && denominator > 0.0) {
    let variance = max(residualSum / count, MINIMUM_VARIANCE);
    let candidateScore = count * log(variance) + count * LN_TWO_PI + count * (count + traceSum) / denominator;
    if (isFiniteBits(candidateScore)) {
      score = candidateScore;
    }
  }
  table[base] = getParameter(HEADER + candidate);
  table[base + 1u] = score;
  table[base + 2u] = singular;
  table[base + 3u] = residualSum;
  table[base + 4u] = traceSum;
  table[base + 5u] = count;
  table[base + 6u] = included;`}))}Q&&Z.push(Q.createIterationNode(T)),Z.push(a(t,{id:`${i}-choose`,operation:B,variant:`choose`,bindings:[q(`params`,C,`f32`),q(`table`,I,`f32`),J(`rowValid`,M,`u32`),J(`selection`,xe,`f32`),J(`summary`,U,`f32`),...z?[q(`state`,z,`f32`)]:[]],invocationCount:1,declarations:`${Y}
${Se()}
${z?Te:``}
const DEFAULT_SIGNIFICANCE_LEVEL: f32 = ${se};`,body:`let nan = getNaN();
  // Fewest singular rows first, then the smallest AICc; ties go to the lowest index.
  var hasBest = false;
  var bestIndex = 0u;
  var bestSingular = 0.0;
  var bestScore = 0.0;
  for (var candidate = 0u; candidate < TABLE_ROWS; candidate++) {
    let base = tableOffset + candidate * TABLE_STRIDE;
    let score = table[base + 1u];
    if (!isFiniteBits(score)) {
      continue;
    }
    let singular = table[base + 2u];
    if (!hasBest || singular < bestSingular || (singular == bestSingular && score < bestScore)) {
      hasBest = true;
      bestIndex = candidate;
      bestSingular = singular;
      bestScore = score;
    }
  }
  ${z?`// Golden section returns the winner of its last comparison (mgwr), when it is valid.
  let optimumRow = state[stateOffset + STATE_OPTIMUM_ROW];
  if (optimumRow >= 0.0) {
    let optimumBase = tableOffset + u32(optimumRow) * TABLE_STRIDE;
    let optimumScore = table[optimumBase + 1u];
    if (isFiniteBits(optimumScore)) {
      hasBest = true;
      bestIndex = u32(optimumRow);
      bestSingular = table[optimumBase + 2u];
      bestScore = optimumScore;
    }
  }`:``}
  let base = tableOffset + bestIndex * TABLE_STRIDE;
  let selectedValue = table[base];
  let residualSum = select(nan, table[base + 3u], hasBest);
  let traceSum = select(nan, table[base + 4u], hasBest);
  let count = table[base + 5u];
  let included = table[tableOffset + 6u];
  let sigmaSquared = select(nan, residualSum / (count - traceSum), hasBest);
  let requestedLevel = params[paramsOffset + 3u];
  let level = select(DEFAULT_SIGNIFICANCE_LEVEL, requestedLevel, requestedLevel > 0.0 && requestedLevel < 1.0);
  let adjustedAlpha = select(nan, level * f32(P) / traceSum, hasBest && traceSum > 0.0);
  let criticalT = select(nan, gwrLargeSampleCriticalValue(adjustedAlpha, count - 1.0), hasBest && traceSum > 0.0);
  let logLikelihood = select(
    nan,
    -0.5 * count * (LN_TWO_PI + log(max(residualSum / count, MINIMUM_VARIANCE)) + 1.0),
    hasBest
  );
  rowValid[rowValidOffset + TAIL_BASE + TAIL_FOCUS_BASE] = 0u;
  selection[selectionOffset] = f32(bestIndex);
  selection[selectionOffset + 1u] = selectedValue;
  rowValid[rowValidOffset + TAIL_BASE + TAIL_SELECTED_INDEX] = bestIndex;
  rowValid[rowValidOffset + TAIL_BASE + TAIL_SELECTED_VALUE] = bitcast<u32>(selectedValue);
  rowValid[rowValidOffset + TAIL_BASE + TAIL_KERNEL] = bitcast<u32>(params[paramsOffset]);
  rowValid[rowValidOffset + TAIL_BASE + TAIL_MODE] = bitcast<u32>(params[paramsOffset + 1u]);
  rowValid[rowValidOffset + TAIL_BASE + TAIL_SIGMA_SQUARED] = bitcast<u32>(sigmaSquared);
  rowValid[rowValidOffset + TAIL_BASE + TAIL_CRITICAL_T] = bitcast<u32>(criticalT);
  summary[summaryOffset] = residualSum;
  summary[summaryOffset + 1u] = traceSum;
  summary[summaryOffset + 2u] = select(nan, bestScore, hasBest);
  summary[summaryOffset + 3u] = nan;
  summary[summaryOffset + 4u] = select(included, count, hasBest);
  summary[summaryOffset + 5u] = select(0.0, 1.0, hasBest);
  summary[summaryOffset + 6u] = select(nan, bestSingular, hasBest);
  summary[summaryOffset + 7u] = sigmaSquared;
  summary[summaryOffset + 8u] = adjustedAlpha;
  summary[summaryOffset + 9u] = criticalT;
  summary[summaryOffset + 10u] = logLikelihood;
  summary[summaryOffset + 11u] = included;`}));let Ge=()=>{if(v?Z.push(...ve(t,{id:i,tileCount:E,statistics:R,summary:U})):Z.push(a(t,{id:`${i}-summarize`,operation:B,variant:`summarize`,bindings:[q(`tileStatistics`,R,`f32`),J(`summary`,U,`f32`)],invocationCount:1,declarations:Y,body:`var count = 0.0;
  var mean = 0.0;
  var m2 = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let tileCountValue = tileStatistics[tileStatisticsOffset + 3u * tile];
    if (tileCountValue == 0.0) {
      continue;
    }
    let tileMean = tileStatistics[tileStatisticsOffset + 3u * tile + 1u];
    let delta = tileMean - mean;
    let total = count + tileCountValue;
    mean = mean + delta * tileCountValue / total;
    m2 = m2 + tileStatistics[tileStatisticsOffset + 3u * tile + 2u] + delta * delta * count * tileCountValue / total;
    count = total;
  }
  let hasValid = summary[summaryOffset + 5u] > 0.5;
  summary[summaryOffset + 3u] = select(getNaN(), 1.0 - summary[summaryOffset] / m2, hasValid && m2 > 0.0);`})),w.bandwidthScores||w.bandwidthSingularRowCounts){let e=[q(`table`,I,`f32`)];w.bandwidthScores&&e.push(J(`scores`,w.bandwidthScores,`f32`)),w.bandwidthSingularRowCounts&&e.push(J(`singularRowCounts`,w.bandwidthSingularRowCounts,`u32`)),Z.push(a(t,{id:`${i}-publish-candidates`,operation:B,variant:`publish-candidates`,bindings:e,invocationCount:c,declarations:Y,body:`${y?`// The golden-section table can hold fewer rows than the ladder capacity.
  let isRow = index < TABLE_ROWS;
  let base = tableOffset + min(index, TABLE_ROWS - 1u) * TABLE_STRIDE;`:`let base = tableOffset + index * TABLE_STRIDE;`}
  ${w.bandwidthScores?`scores[scoresOffset + index] = ${y?`select(getNaN(), table[base + 1u], isRow)`:`table[base + 1u]`};`:``}
  ${w.bandwidthSingularRowCounts?`let singular = table[base + 2u];\n  singularRowCounts[singularRowCountsOffset + index] = select(0u, u32(singular), ${y?`isRow && `:``}singular >= 0.0 && singular < 4.0e9);`:``}`}))}};for(let e=0;e<le;e++){let n=_?`-batch-${e}`:``,r=Math.min(g,o-e*g);Z.push((_?j:a)(t,{id:`${i}${n}-fit`,operation:B,variant:`fit`,bindings:[q(`positions`,b,`f32`),q(`predictors`,x,`f32`),q(`response`,S,`f32`),J(`rowValid`,M,`u32`),...Fe,J(`coefficients`,w.coefficients,`f32`)],invocationCount:r,workgroupSize:_?256:void 0,declarations:Ve,body:`${Re}
  let localRow = ${_?`index`:`focal`};
  let nan = getNaN();
  let coefficientBase = coefficientsOffset + focal * P;
  let localBase = rowValidOffset + LOCALS_BASE + localRow * LOCAL_STRIDE;
  var status = ${p.SINGULAR}u;
  var localR2 = nan;
  var conditionNumber = nan;
  var fittedValue = nan;
  var residual = nan;
  var hat = nan;
  for (var column = 0u; column < P; column++) {
    coefficients[coefficientBase + column] = nan;
    ${D?`rowValid[rowValidOffset + STANDARD_ERROR_BASE + localRow * P + column] = bitcast<u32>(nan);`:``}
    ${O?`rowValid[rowValidOffset + T_VALUE_BASE + localRow * P + column] = bitcast<u32>(nan);`:``}
  }
  if (rowValid[rowValidOffset + focal] == 0u) {
    status = ${p.EXCLUDED}u;
  } else {
    let selectedValue = bitcast<f32>(rowValid[rowValidOffset + TAIL_BASE + TAIL_SELECTED_VALUE]);
    var neighbors: array<f32, MAXIMUM_NEIGHBORS>;
    let isAdaptive = getIsAdaptive();
    if (isAdaptive) {
      let maximumK = clamp(u32(max(floor(selectedValue + 0.5), 1.0)), 1u, MAXIMUM_NEIGHBORS);
      selectNeighborDistances(focal, maximumK, &neighbors);
    }
    let bandwidth = getBandwidthForValue(selectedValue, isAdaptive, &neighbors);
    if (bandwidth > 0.0) {
      let fit = fitLocation(focal, bandwidth);
      if (fit.ok) {
        status = ${p.OK}u;
        var intercept = fit.beta[0];
        for (var column = 1u; column < P; column++) {
          let slope = fit.beta[column];
          coefficients[coefficientBase + column] = slope;
          intercept = intercept - slope * predictors[predictorsOffset + focal * PREDICTOR_COUNT + column - 1u];
        }
        coefficients[coefficientBase] = intercept;
        fittedValue = fit.beta[0];
        residual = response[responseOffset + focal] - fittedValue;
        hat = fit.hat;
        localR2 = getLocalR2(focal, bandwidth, fit);
        ${w.localConditionNumber?`conditionNumber = getLocalConditionNumber(focal, bandwidth);`:``}
        ${D?`let sigmaSquared = bitcast<f32>(rowValid[rowValidOffset + TAIL_BASE + TAIL_SIGMA_SQUARED]);
        for (var column = 0u; column < P; column++) {
          var standardError = nan;
          let variance = fit.variance[column] * sigmaSquared;
          if (isFiniteBits(variance) && variance >= 0.0) {
            standardError = sqrt(variance);
          }
          rowValid[rowValidOffset + STANDARD_ERROR_BASE + localRow * P + column] = bitcast<u32>(standardError);
          ${O?`let coefficient = coefficients[coefficientBase + column];
          let tValue = select(nan, coefficient / standardError, isFiniteBits(standardError) && standardError > 0.0 && isFiniteBits(coefficient));
          rowValid[rowValidOffset + T_VALUE_BASE + localRow * P + column] = bitcast<u32>(tValue);`:``}
        }`:``}
      }
    }
  }
  rowValid[localBase] = bitcast<u32>(localR2);
  rowValid[localBase + 1u] = bitcast<u32>(fittedValue);
  rowValid[localBase + 2u] = bitcast<u32>(residual);
  rowValid[localBase + 3u] = bitcast<u32>(hat);
  rowValid[localBase + 4u] = status;
  rowValid[localBase + 5u] = bitcast<u32>(conditionNumber);`})),d&&!_&&Z.push((_?j:a)(t,{id:`${i}${n}-fit-excluded`,operation:B,variant:`fit-excluded`,bindings:[J(`rowValid`,M,`u32`),J(`coefficients`,w.coefficients,`f32`)],invocationCount:r,workgroupSize:_?256:void 0,declarations:Y,body:`if (rowValid[rowValidOffset + index] != 0u) {
    return;
  }
  let nan = getNaN();
  for (var column = 0u; column < P; column++) {
    coefficients[coefficientsOffset + index * P + column] = nan;
    ${D?`rowValid[rowValidOffset + STANDARD_ERROR_BASE + index * P + column] = bitcast<u32>(nan);`:``}
    ${O?`rowValid[rowValidOffset + T_VALUE_BASE + index * P + column] = bitcast<u32>(nan);`:``}
  }
  let localBase = rowValidOffset + LOCALS_BASE + index * LOCAL_STRIDE;
  for (var word = 0u; word < LOCAL_STRIDE; word++) {
    rowValid[localBase + word] = bitcast<u32>(nan);
  }
  rowValid[localBase + 4u] = ${p.EXCLUDED}u;`})),Z.push((_?j:a)(t,{id:`${i}${n}-tile-statistics`,operation:B,variant:`tile-statistics`,bindings:[q(`response`,S,`f32`),q(`rowValid`,M,`u32`),J(`tileStatistics`,R,`f32`)],invocationCount:Math.ceil(r/V),workgroupSize:_?256:void 0,declarations:Y,body:`let focusBase = ${_?X:`0u`};
  let firstRow = focusBase + index * TILE_ROWS;
  if (firstRow >= ROW_COUNT) { return; }
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var count = 0.0;
  var sum = 0.0;
  for (var row = firstRow; row < endRow; row++) {
    if (rowValid[rowValidOffset + LOCALS_BASE + (row - focusBase) * LOCAL_STRIDE + 4u] == ${p.OK}u) {
      count = count + 1.0;
      sum = sum + response[responseOffset + row];
    }
  }
  let mean = sum / max(count, 1.0);
  var m2 = 0.0;
  for (var row = firstRow; row < endRow; row++) {
    if (rowValid[rowValidOffset + LOCALS_BASE + (row - focusBase) * LOCAL_STRIDE + 4u] == ${p.OK}u) {
      let deviation = response[responseOffset + row] - mean;
      m2 = m2 + deviation * deviation;
    }
  }
  tileStatistics[tileStatisticsOffset + 3u * (firstRow / TILE_ROWS)] = count;
  tileStatistics[tileStatisticsOffset + 3u * (firstRow / TILE_ROWS) + 1u] = mean;
  tileStatistics[tileStatisticsOffset + 3u * (firstRow / TILE_ROWS) + 2u] = m2;`})),_||Ge();let s=[[`localR2`,w.localR2,`f32`,0],[`fitted`,w.fitted,`f32`,1],[`residuals`,w.residuals,`f32`,2],[`hatDiagonal`,w.hatDiagonal,`f32`,3],[`localStatus`,w.localStatus,`u32`,4],[`localConditionNumber`,w.localConditionNumber,`f32`,5]].filter(([,e])=>e);if(s.length>0&&Z.push((_?j:a)(t,{id:`${i}${n}-publish`,operation:B,variant:`publish`,bindings:[q(`rowValid`,M,`u32`),...s.map(([e,t,n])=>J(e,t,n))],invocationCount:r,workgroupSize:_?256:void 0,declarations:Y,body:`${_?`if (${X} + index >= ROW_COUNT) { return; }\n`:``}`+s.map(([e,,t,n])=>t===`f32`?`${e}[${e}Offset + ${_?`(${X} + index)`:`index`}] = bitcast<f32>(rowValid[rowValidOffset + LOCALS_BASE + index * LOCAL_STRIDE + ${n}u]);`:`${e}[${e}Offset + ${_?`(${X} + index)`:`index`}] = rowValid[rowValidOffset + LOCALS_BASE + index * LOCAL_STRIDE + ${n}u];`).join(`
  `)})),D){let e=[q(`rowValid`,M,`u32`)],o=[];w.standardErrors&&(e.push(J(`standardErrors`,w.standardErrors,`f32`)),o.push(`standardErrors[standardErrorsOffset + globalIndex] = bitcast<f32>(rowValid[rowValidOffset + STANDARD_ERROR_BASE + index]);`)),w.tValues&&(e.push(J(`tValues`,w.tValues,`f32`)),o.push(`tValues[tValuesOffset + globalIndex] = bitcast<f32>(rowValid[rowValidOffset + T_VALUE_BASE + index]);`)),w.significant&&(e.push(J(`significant`,w.significant,`u32`)),o.push(`let tValue = bitcast<f32>(rowValid[rowValidOffset + T_VALUE_BASE + index]);
  let criticalT = bitcast<f32>(rowValid[rowValidOffset + TAIL_BASE + TAIL_CRITICAL_T]);
  // NaN t values and a NaN critical value are never significant.
  significant[significantOffset + globalIndex] =
    select(0u, 1u, isFiniteBits(tValue) && isFiniteBits(criticalT) && abs(tValue) > criticalT);`)),Z.push((_?j:a)(t,{id:`${i}${n}-publish-inference`,operation:B,variant:`publish-inference`,bindings:e,invocationCount:r*k,workgroupSize:_?256:void 0,declarations:Y,body:`let globalIndex = ${_?`${X} * P + index`:`index`};\n  if (globalIndex >= ROW_COUNT * P) { return; }\n  `+o.join(`
  `)}))}_&&Z.push(Ue(`${n}-fit-focus-advance`,!1))}return _&&Ge(),w.approximation&&Z.push(a(t,{id:`${i}-approximation`,operation:B,variant:`approximation`,invocationCount:1,bindings:[q(`params`,C,`f32`),J(`approximation`,w.approximation,`u32`)],body:`approximation[approximationOffset] = ${h===null?`0u`:`select(1u, 0u, params[paramsOffset] > 0.5)`};`})),Z}},Te=Object.entries(S).map(([e,t])=>`const STATE_${e}: u32 = ${t}u;`).join(`
`),Ee=`
${O}

${E}
`,De=`
fn refineLadder() {
  let nan = getNaN();
  // Advance the GPU pass cursor: this node writes the ladder of pass passCursor.
  let passCursor = u32(ladders[laddersOffset]) + 1u;
  ladders[laddersOffset] = f32(passCursor);
  let callerBase = laddersOffset + 1u;
  let isAdaptive = ladders[callerBase + 1u] > 0.5;
  var hasBest = false;
  var bestValue = 0.0;
  var bestSingular = 0.0;
  var bestScore = 0.0;
  for (var candidate = 0u; candidate < passCursor * LADDER; candidate++) {
    let base = tableOffset + candidate * TABLE_STRIDE;
    let score = table[base + 1u];
    if (!isFiniteBits(score)) {
      continue;
    }
    let singular = table[base + 2u];
    if (!hasBest || singular < bestSingular || (singular == bestSingular && score < bestScore)) {
      hasBest = true;
      bestValue = table[base];
      bestSingular = singular;
      bestScore = score;
    }
  }
  // Adaptive ladders hold integer k, so values compare after rounding.
  if (isAdaptive) {
    bestValue = floor(bestValue + 0.5);
  }
  var lower = -SENTINEL;
  var upper = SENTINEL;
  for (var candidate = 0u; candidate < passCursor * LADDER; candidate++) {
    var value = table[tableOffset + candidate * TABLE_STRIDE];
    if (!isFiniteBits(value)) {
      continue;
    }
    if (isAdaptive) {
      value = floor(value + 0.5);
    }
    if (value < bestValue) {
      lower = max(lower, value);
    } else if (value > bestValue) {
      upper = min(upper, value);
    }
  }
  if (lower <= -SENTINEL) {
    lower = bestValue;
  }
  if (upper >= SENTINEL) {
    upper = bestValue;
  }
  let base = callerBase + passCursor * PARAMETER_LENGTH;
  for (var slot = 0u; slot < PARAMETER_LENGTH; slot++) {
    ladders[base + slot] = 0.0;
  }
  ladders[base] = ladders[callerBase];
  ladders[base + 1u] = ladders[callerBase + 1u];
  ladders[base + 3u] = ladders[callerBase + 3u];
  var count = 0u;
  if (hasBest) {
    if (isAdaptive) {
      let first = max(lower + 1.0, 2.0);
      let last = min(upper - 1.0, f32(MAXIMUM_NEIGHBORS));
      let isInside = first <= bestValue && bestValue <= last;
      let integers = max(last - first + 1.0 - select(0.0, 1.0, isInside), 0.0);
      if (integers > 0.0) {
        count = min(u32(integers), LADDER);
        for (var slot = 0u; slot < count; slot++) {
          var position = f32(slot);
          if (integers > f32(LADDER)) {
            position = floor((f32(slot) + 0.5) * integers / f32(LADDER));
          }
          var value = first + position;
          if (isInside && value >= bestValue) {
            value = value + 1.0;
          }
          ladders[base + HEADER + slot] = value;
        }
      }
    } else {
      let low = max(lower, 0.0);
      let span = upper - low;
      if (span > 1.0e-5 * max(abs(upper), 1.0e-30)) {
        count = LADDER;
        for (var slot = 0u; slot < LADDER; slot++) {
          ladders[base + HEADER + slot] = low + span * f32(slot + 1u) / f32(LADDER + 1u);
        }
      }
    }
  }
  ladders[base + 2u] = f32(count);
}
`;export{j as n,K as t};