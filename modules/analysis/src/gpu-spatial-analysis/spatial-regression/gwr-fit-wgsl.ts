// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getRingNeighborDistancesWGSL} from './gwr-grid-index';
import {getCholeskyWGSL} from './spatial-regression-solve';
import {getDesignSingularValueWGSL} from './local-collinearity-wgsl';
import {
  getCandidateLoopWGSL,
  getEquilibratedCholeskyWGSL,
  getLocalDistanceWGSL,
  getLocalGridSupportWGSL,
  getLocalKernelWeightWGSL
} from './local-weighted-least-squares-wgsl';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {LOCAL_REGRESSION_INTERCEPT_WGSL} from './local-regression-intercept-wgsl';
import {getPairArithmeticWGSL} from './local-regression-pairs-wgsl';
import {ROUNDED_DIFFERENCE_WGSL} from './rounded-difference-wgsl';

/** Options of {@link getGWRFitWGSL}. @internal */
export type GWRFitWGSLOptions = {
  /** Design columns including the intercept, `P`. */
  coefficientCount: number;
  /** Grid index dimensions, or null for the all-rows scan. Planar distances only. */
  indexGridSize: readonly [number, number] | null;
  /** Declare `getLocalConditionNumber`. */
  hasConditionNumber: boolean;
  /** Also accumulate the squared-weight Gram matrix and return `variance` (local standard errors). */
  hasVariance: boolean;
  /** Retain the low words of the scalar-design Gram matrix and response moments. */
  preciseScalarFit?: boolean;
  /** Distances are great-circle `geodesicHaversineDistance(a, b, radius)`; otherwise planar. */
  sphereRadius: number | null;
  /**
   * Gaussian weights of rows farther than `gaussianCutoff * bandwidth` are exactly 0, which also
   * lets the Gaussian kernel use the grid index. Null or omitted is the exact unbounded Gaussian.
   */
  gaussianCutoff?: number | null;
};

/**
 * WGSL of the local weighted least squares shared by the candidate and final-fit kernels.
 *
 * The including kernel must declare, before this text: the constants `ROW_COUNT`,
 * `PREDICTOR_COUNT`, `P`, `MAXIMUM_NEIGHBORS`, `SENTINEL`, `ADAPTIVE_FACTOR`; the bindings
 * `positions`, `predictors`, `response`, `rowValid` (and `cellOffsets`, `sortedIds` with a grid);
 * `getNaN()` and `isFiniteBits()`; and the two functions `getIsBisquare() -> bool` and
 * `getIsAdaptive() -> bool` that read the kernel and bandwidth mode from wherever the kernel keeps
 * them (the parameter buffer for the candidate passes, the row-state tail for the final fit).
 *
 * Declares: `LocalFit`, `getPosition`, `getDistance`, `getWeight`, `selectNeighborDistances`,
 * `getBandwidthForValue`, `fitLocation`, `getLocalR2`, optionally `getLocalConditionNumber`, and
 * the Cholesky and eigenvalue helpers, and `isFiniteValue` (which the shared Cholesky needs; with a
 * grid it comes from the grid lookup), so the kernel must not include `IS_FINITE_VALUE_WGSL` again.
 *
 * @internal
 */
export function getGWRFitWGSL(options: GWRFitWGSLOptions): string {
  const {
    coefficientCount: P,
    indexGridSize,
    hasConditionNumber,
    hasVariance,
    sphereRadius
  } = options;
  const gaussianCutoff = options.gaussianCutoff ?? null;
  const hasGrid = Boolean(indexGridSize);
  const preciseScalarFit = options.preciseScalarFit && P === 2;
  const fitWeight =
    preciseScalarFit && sphereRadius === null
      ? 'getFitWeight(origin, getPosition(row), bandwidth)'
      : 'getWeight(getDistance(origin, getPosition(row)), bandwidth)';
  return /* wgsl */ `
${ROUNDED_DIFFERENCE_WGSL}
${LOCAL_REGRESSION_INTERCEPT_WGSL}
${preciseScalarFit ? getPairArithmeticWGSL() : ''}
${hasGrid ? '' : IS_FINITE_VALUE_WGSL}
${getCholeskyWGSL(P)}
${getEquilibratedCholeskyWGSL(P)}
${getLocalKernelWeightWGSL(gaussianCutoff)}
${hasConditionNumber ? getDesignSingularValueWGSL(P) : ''}
${indexGridSize ? getLocalGridSupportWGSL(indexGridSize, gaussianCutoff) : ''}
struct LocalFit {
  ok: bool,
  hat: f32,
  weightSum: f32,
  weightedResponse: f32,
  // beta[0] is the fitted value minus response[i]; predictor columns are centered on row i.
  beta: array<f32, ${P}>,
  // diag((X'WX)^-1 X'W^2 X (X'WX)^-1) of the original (uncentered) coefficients; NaN when not
  // finite. Only filled when the squared-weight Gram matrix is accumulated.
  variance: array<f32, ${P}>
}

fn getPosition(row: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

${getLocalDistanceWGSL(sphereRadius)}

fn getWeight(distance: f32, bandwidth: f32) -> f32 {
  return getLocalKernelWeight(distance, bandwidth, getIsBisquare());
}

${
  preciseScalarFit && sphereRadius === null
    ? `fn getFitWeight(origin: vec2f, point: vec2f, bandwidth: f32) -> f32 {
  if (getIsBisquare() && bandwidth >= 1.0e-15 && bandwidth <= 1.0e15) {
    let delta = point - origin;
    // Most grid candidates can be rejected before forming either low word.
    if (dot(delta, delta) > (bandwidth * bandwidth) * 1.000001) { return 0.0; }
    let norm = addRegressionPairs(multiplyRegressionPair(vec2f(delta.x, 0.0), delta.x),
      multiplyRegressionPair(vec2f(delta.y, 0.0), delta.y));
    let bandSquared = multiplyRegressionPair(vec2f(bandwidth, 0.0), bandwidth);
    if (norm.x > bandSquared.x) { return 0.0; }
    let gap = addRegressionPairs(bandSquared, -norm);
    if (!(gap.x > 0.0)) { return 0.0; }
    let ratio = divideRegressionPair(gap, bandSquared);
    return multiplyRegressionPairs(ratio, ratio).x;
  }
  return getWeight(getDistance(origin, point), bandwidth);
}`
    : ''
}

${
  indexGridSize && sphereRadius === null
    ? getRingNeighborDistancesWGSL()
    : `// Keeps the maximumK smallest squared distances from row i (itself included) in ascending order.
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
}`
}

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
  var a: array<f32, ${P * P}>;
  var b: array<f32, ${P}>;
  var z: array<f32, ${P}>;
  ${preciseScalarFit ? 'var moments: array<vec2f, 5>;' : ''}
  ${hasVariance ? `var squared: array<f32, ${P * P}>;` : ''}
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
    ${hasVariance ? 'squared[entry] = 0.0;' : ''}
  }
  let origin = getPosition(i);
  let originBase = predictorsOffset + i * PREDICTOR_COUNT;
  z[0] = 1.0;
  ${getCandidateLoopWGSL(
    `let weight = ${fitWeight};
    if (!(weight > 0.0)) {
      continue;
    }
    let rowBase = predictorsOffset + row * PREDICTOR_COUNT;
    for (var column = 0u; column < PREDICTOR_COUNT; column++) {
      z[column + 1u] = predictors[rowBase + column] - predictors[originBase + column];
    }
    let y = response[responseOffset + row];
    // The validation pass writes one for retained rows. This independent buffer load
    // preserves the rounded anchor difference under GPU fast math.
    let runtimeZero = rowValid[rowValidOffset + i] - 1u;
    let anchoredResponse = getRoundedDifference(y, response[responseOffset + i], runtimeZero);
    ${preciseScalarFit ? '' : 'fit.weightedResponse += weight * anchoredResponse;'}
    ${
      preciseScalarFit
        ? `let weightedX = multiplyRegressionPair(vec2f(weight, 0.0), z[1]);
    moments[0] = addRegressionPairs(moments[0], vec2f(weight, 0.0));
    moments[1] = addRegressionPairs(moments[1], weightedX);
    moments[2] = addRegressionPairs(moments[2], multiplyRegressionPair(weightedX, z[1]));
    moments[3] = addRegressionPairs(moments[3], multiplyRegressionPair(vec2f(weight, 0.0), anchoredResponse));
    moments[4] = addRegressionPairs(moments[4], multiplyRegressionPair(weightedX, anchoredResponse));`
        : ''
    }
    ${
      !preciseScalarFit || hasVariance
        ? `for (var r = 0u; r < P; r++) {
      let weighted = weight * z[r];
      ${preciseScalarFit ? '' : 'b[r] = b[r] + weighted * anchoredResponse;'}
      for (var c = 0u; c <= r; c++) {
        ${preciseScalarFit ? '' : 'a[r * P + c] = a[r * P + c] + weighted * z[c];'}
        ${hasVariance ? 'squared[r * P + c] = squared[r * P + c] + weighted * weight * z[c];' : ''}
      }
    }`
        : ''
    }`,
    hasGrid
  )}
  ${
    preciseScalarFit
      ? `fit.weightedResponse = moments[3].x;
  a[0] = moments[0].x;
  a[2] = moments[1].x;
  a[3] = moments[2].x;
  b[0] = moments[3].x;
  b[1] = moments[4].x;`
      : ''
  }
  fit.weightSum = a[0];
  var scale: array<f32, ${P}>;
  if (!factorEquilibrated_${P}(&a, &scale)) {
    return fit;
  }
  var inverse00 = inverseDiagonalEquilibrated_${P}(&a, &scale, 0u);
  // b holds X'Wy of the centered design; the solve is in place and unscales the coefficients.
  solveEquilibrated_${P}(&a, &scale, &b);
  ${
    preciseScalarFit
      ? `// Keep the usual scaled-factor validation. Refine the scalar fit from the two
  // moment words wherever the reciprocals stay normal.
  if (moments[2].x >= 1.1754943508222875e-38 && moments[2].x <= 8.507059173023462e37) {
    let ratio = divideRegressionPair(moments[1], moments[2]);
    let schur = addRegressionPairs(moments[0], -multiplyRegressionPairs(ratio, moments[1]));
    if (schur.x >= 1.1754943508222875e-38) {
      let intercept = divideRegressionPair(addRegressionPairs(moments[3], -multiplyRegressionPairs(ratio, moments[4])), schur);
      let slope = divideRegressionPair(addRegressionPairs(moments[4], -multiplyRegressionPairs(moments[1], intercept)), moments[2]);
      let hat = divideRegressionPair(vec2f(1.0, 0.0), schur).x;
      if (isFiniteBits(intercept.x) && isFiniteBits(slope.x) && isFiniteBits(hat) && hat > 0.0) {
        b[0] = intercept.x;
        b[1] = slope.x;
        inverse00 = hat;
      }
    }
  }`
      : ''
  }
  var finite = isFiniteBits(inverse00);
  for (var r = 0u; r < P; r++) {
    fit.beta[r] = b[r];
    finite = finite && isFiniteBits(fit.beta[r]);
  }
  fit.hat = inverse00;
  fit.ok = finite && fit.hat >= 0.0;
  ${
    hasVariance
      ? `if (fit.ok) {
    // C C' = A^-1 B A^-1 with A = X'WX (centered design) and B = X'W^2 X. Column c of A^-1 comes
    // from the equilibrated factor: A^-1 e_c = s_c * (s o solve(e_c)). The original intercept is
    // gamma_0 - sum x_ic gamma_c, so its direction is t = (1, -x_i) and A^-1 t = s o solve(s o t);
    // variance_c = w' B w.
    for (var column = 0u; column < P; column++) {
      var w: array<f32, ${P}>;
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
      choleskySolve_${P}(&a, &w);
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
  }`
      : ''
  }
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
  ${getCandidateLoopWGSL(
    `let weight = ${fitWeight};
    if (!(weight > 0.0)) {
      continue;
    }
    let rowBase = predictorsOffset + row * PREDICTOR_COUNT;
    var prediction = fit.beta[0];
    for (var column = 0u; column < PREDICTOR_COUNT; column++) {
      prediction = prediction + fit.beta[column + 1u] * (predictors[rowBase + column] - predictors[originBase + column]);
    }
    let y = response[responseOffset + row];
    let runtimeZero = rowValid[rowValidOffset + i] - 1u;
    let anchoredResponse = getRoundedDifference(y, response[responseOffset + i], runtimeZero);
    let residual = anchoredResponse - prediction;
    residualSum = residualSum + weight * residual * residual;
    let deviation = anchoredResponse - weightedMean;
    totalSum = totalSum + weight * deviation * deviation;`,
    hasGrid
  )}
  return select(getNaN(), 1.0 - residualSum / totalSum, totalSum > 0.0);
}
${hasConditionNumber ? getConditionNumberWGSL(P, hasGrid) : ''}
`;
}

/**
 * `getLocalConditionNumber`: mgwr `local_collinearity` condition number. With weights `w_j`
 * (kernel weights, not their square roots, exactly as mgwr's `xw = x * wi`) and the raw design
 * `x_j = (1, X_j)`, it is the ratio of the extreme singular values of the column-normalized
 * `diag(w) x`.
 *
 * The weighted rows are folded into an upper-triangular factor `R` by Givens rotations
 * (streaming QR, `updateDesignFactor_P` of `GPULocalCollinearity`), so `R'R` is the weighted
 * moment without ever being formed. Scaling the columns of `R` to unit norm gives a factor of the
 * column-normalized design, and one-sided Jacobi (`getDesignSingularValues_P`) gives its singular
 * values to about `epsilon * condition`. The former route (the Gram matrix `C` and its extreme
 * eigenvalues) needs `1 - cosine` of two columns from float32 values near 1: it was 17% off for
 * a predictor of 1500 +- 3 and meaningless for a projected northing. A rank-deficient design
 * returns a very large number rather than NaN.
 */
function getConditionNumberWGSL(coefficientCount: number, hasGrid: boolean): string {
  const P = coefficientCount;
  return /* wgsl */ `
fn getLocalConditionNumber(i: u32, bandwidth: f32) -> f32 {
  var factor: array<f32, ${P * P}>;
  var designRow: array<f32, ${P}>;
  let origin = getPosition(i);
  ${getCandidateLoopWGSL(
    `let weight = getWeight(getDistance(origin, getPosition(row)), bandwidth);
    if (!(weight > 0.0)) {
      continue;
    }
    let rowBase = predictorsOffset + row * PREDICTOR_COUNT;
    designRow[0] = weight;
    for (var column = 0u; column < PREDICTOR_COUNT; column++) {
      designRow[column + 1u] = weight * predictors[rowBase + column];
    }
    updateDesignFactor_${P}(&factor, &designRow);`,
    hasGrid
  )}
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
  var vectors: array<f32, ${P * P}>;
  var singularValues: array<f32, ${P}>;
  // The sweeps are bounded; an unconverged result is still the best available estimate.
  getDesignSingularValues_${P}(&factor, &vectors, &singularValues);
  var largest = 0.0;
  var smallest = 3.0e38;
  for (var c = 0u; c < P; c++) {
    largest = max(largest, singularValues[c]);
    smallest = min(smallest, singularValues[c]);
  }
  return largest / max(smallest, 1.0e-30);
}
`;
}
