import{T as e,f as t}from"./gpu-contributor-utils-BIwLQON5.js";import{t as n}from"./pairwise-sum-wgsl-iErcC5Or.js";var r=1024,i=.99;function a(e){let t=t=>t**(e+1)/((e+1)*(1-t));if(t(.99)<=1e-5)return i;let n=0,r=i;for(let e=0;e<60;e++){let e=.5*(n+r);t(e)<=1e-5?n=e:r=e}return n}function o(n,r){t(`spatial-maximum-likelihood-series`,`maximumPower`,n,2,256),e(`spatial-maximum-likelihood-series`,/^[A-Za-z_][A-Za-z0-9_]*$/.test(r),`bindingName must be a WGSL identifier`)}function s(e,t,n=`1.0`,r=!1){return o(e,t),`
fn getTraceOne(rho: f32) -> f32 {
  let norm = ${n};
  let t = rho * norm;
  var accumulator = 0.0;
  for (var step = 0u; step < ${e}u; step++) {
    let q = ${e}u - step;
    accumulator = accumulator * t + ${t}[${t}Offset + q - 1u];
  }
  return norm * (accumulator${r?` + getSeriesTailDerivative(t)`:``});
}

fn getLogDeterminantDerivative(rho: f32) -> f32 {
  return -getTraceOne(rho);
}

fn getTraceTwo(rho: f32) -> f32 {
  let norm = ${n};
  let t = rho * norm;
  var accumulator = 0.0;
  for (var step = 0u; step < ${e-1}u; step++) {
    let m = ${e-2}u - step;
    accumulator = accumulator * t + f32(m + 1u) * ${t}[${t}Offset + m + 1u];
  }
  return norm * norm * (accumulator${r?` + getSeriesTailSecond(t)`:``});
}
`}function c(e=r,t=32){return`
struct SearchResult {
  parameter: f32,
  localMaximumCount: u32,
  atBoundary: bool,
}

fn searchParameter(bound: f32) -> SearchResult {
  var best = getNaN();
  var bestObjective = -3.0e38;
  var bestAtBoundary = false;
  var count = 0u;
  let step = 2.0 * bound / ${e}.0;
  var previousParameter = -bound;
  var previous = getDerivative(previousParameter);
  if (previous <= 0.0) {
    let objective = getObjective(previousParameter);
    count = count + 1u;
    if (objective > bestObjective) {
      best = previousParameter;
      bestObjective = objective;
      bestAtBoundary = true;
    }
  }
  for (var index = 1u; index <= ${e}u; index++) {
    var parameter = bound;
    if (index < ${e}u) {
      parameter = -bound + f32(index) * step;
    }
    let current = getDerivative(parameter);
    if (previous > 0.0 && current <= 0.0) {
      var low = previousParameter;
      var high = parameter;
      for (var iteration = 0u; iteration < ${t}u; iteration++) {
        let middle = 0.5 * (low + high);
        if (getDerivative(middle) > 0.0) {
          low = middle;
        } else {
          high = middle;
        }
      }
      let root = 0.5 * (low + high);
      let objective = getObjective(root);
      count = count + 1u;
      if (objective > bestObjective) {
        best = root;
        bestObjective = objective;
        bestAtBoundary = false;
      }
    }
    previous = current;
    previousParameter = parameter;
  }
  if (previous > 0.0) {
    let objective = getObjective(bound);
    count = count + 1u;
    if (objective > bestObjective) {
      best = bound;
      bestObjective = objective;
      bestAtBoundary = true;
    }
  }
  return SearchResult(best, count, bestAtBoundary);
}
`}function l(e,t,r,i,a=`sum`){return n({count:i,name:a,term:`${e}[${e}Offset + ${t} + ${a}Item * ${r}]`})}function u(){return`
fn writeTableRow(row: u32, estimate: f32, standardError: f32) {
  let z = estimate / standardError;
  tableOut[tableOutOffset + row * 4u] = estimate;
  tableOut[tableOutOffset + row * 4u + 1u] = standardError;
  tableOut[tableOutOffset + row * 4u + 2u] = z;
  tableOut[tableOutOffset + row * 4u + 3u] = erfcValue(abs(z) * 0.70710678);
}
`}export{i as a,s as i,c as n,a as o,u as r,l as t};