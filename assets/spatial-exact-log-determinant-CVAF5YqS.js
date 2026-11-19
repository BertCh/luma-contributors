import{E as e}from"./gpu-contributor-utils-BIwLQON5.js";import{t}from"./pairwise-sum-wgsl-iErcC5Or.js";import{t as n}from"./multiplication-rounding-error-wgsl-P721A62W.js";var r=1024;function i(t,n,i){if(n.logDeterminant!==void 0&&![`series`,`exact`].includes(n.logDeterminant))throw Error(`${t} logDeterminant must be 'series' or 'exact'`);if(n.logDeterminant!==`exact`){if(n.eigenvalues)throw Error(`${t} eigenvalues need logDeterminant 'exact'`);return!1}if(i>1024)throw Error(`${t} logDeterminant 'exact' supports at most ${r} rows`);if(!n.eigenvalues)throw Error(`${t} logDeterminant 'exact' needs eigenvalues (getSpatialWeightsEigenvalues)`);if(e(n.eigenvalues,[`float32`],`${t} eigenvalues`),n.eigenvalues.length!==i)throw Error(`${t} eigenvalues must hold one value per row`);return!0}function a(e,r){let i=n=>`${t({count:`${r}u`,name:`total`,setup:`let lambda = ${e}[${e}Offset + totalItem];
    let ratio = lambda / (1.0 - rho * lambda);`,term:`(${n})`})}
  return total;`;return`
${n}
fn getSpatialLogAbsoluteFactor(rho: f32, eigenvalue: f32) -> f32 {
  let product = rho * eigenvalue;
  if (product == 1.0) {
    // Rebalance exact powers of two so even very unequal operands are normal and bounded.
    let first = frexp(rho);
    let second = frexp(eigenvalue);
    let residual = getMultiplicationRoundingError(
      ldexp(first.fract, first.exp + second.exp), second.fract, product);
    if (residual != 0.0) { return log(abs(residual)); }
    // Nonfinite constant expressions are shader-creation errors in WGSL.
    var negativeInfinityBits = 0xff800000u;
    return bitcast<f32>(negativeInfinityBits);
  }
  // The log magnitude is representable even when finite operands overflow on multiplication.
  if ((bitcast<u32>(product) & 0x7fffffffu) == 0x7f800000u) {
    return log(abs(rho)) + log(abs(eigenvalue));
  }
  // WGSL has no log1p. Near zero this series retains terms lost by subtracting from 1.
  // At |product| < 0.01 the omitted term is below 2.1e-11.
  if (abs(product) < 0.01) {
    return -product * (1.0 + product * (0.5 + product * (1.0 / 3.0 + product * 0.25)));
  }
  return log(abs(1.0 - product));
}

fn getSpatialLogDeterminant(rho: f32) -> f32 {
  ${i(`getSpatialLogAbsoluteFactor(rho, lambda)`)}
}

fn getTraceOne(rho: f32) -> f32 {
  ${i(`ratio`)}
}

fn getLogDeterminantDerivative(rho: f32) -> f32 {
  return -getTraceOne(rho);
}

fn getTraceTwo(rho: f32) -> f32 {
  ${i(`ratio * ratio`)}
}
`}export{a as n,i as r,r as t};