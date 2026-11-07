// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Largest predictor count (including the intercept) the small dense solvers support. */
export const SPATIAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT = 16;

/**
 * Returns WGSL for an in-place Cholesky factorization and solve of a symmetric positive definite
 * `p x p` system held in function-local arrays, with `p` fixed at compile time.
 *
 * Generated functions (suffix `_${p}`):
 * - `choleskyFactor_p(a: ptr<function, array<f32, p*p>>) -> bool` overwrites the lower triangle
 *   (row-major `a[row * p + column]`) with `L` such that `A = L L^T`. Returns false when a pivot is
 *   not strictly positive or not finite (singular, indefinite, or ill-conditioned system).
 * - `choleskySolve_p(a, b: ptr<function, array<f32, p>>)` solves `L L^T x = b` in place in `b`
 *   using a factor from `choleskyFactor_p`.
 * - `choleskyInverseDiagonal_p(a, column: u32) -> f32` returns `(A^-1)[column][column]`, used for
 *   coefficient standard errors and hat-matrix terms.
 *
 * The loop order is fixed, so results are deterministic for a given adapter. The CPU reference
 * (test/reference/spatial-regression-solve-cpu.ts) uses the same operation order with `Math.fround`
 * rounding.
 *
 * `maximumSize` raises the default limit of {@link SPATIAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT} for
 * callers that keep the matrices in private memory at larger sizes (local regressions).
 */
export function getCholeskyWGSL(
  predictorCount: number,
  maximumSize = SPATIAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT
): string {
  validatePredictorCount(predictorCount, maximumSize);
  const p = predictorCount;
  const size = p * p;
  return /* wgsl */ `
fn choleskyFactor_${p}(a: ptr<function, array<f32, ${size}>>) -> bool {
  for (var column = 0u; column < ${p}u; column++) {
    var diagonal = (*a)[column * ${p}u + column];
    for (var k = 0u; k < column; k++) {
      let value = (*a)[column * ${p}u + k];
      diagonal = diagonal - value * value;
    }
    if (!(diagonal > 0.0) || diagonal > 3.0e38) {
      return false;
    }
    let pivot = sqrt(diagonal);
    (*a)[column * ${p}u + column] = pivot;
    for (var row = column + 1u; row < ${p}u; row++) {
      var value = (*a)[row * ${p}u + column];
      for (var k = 0u; k < column; k++) {
        value = value - (*a)[row * ${p}u + k] * (*a)[column * ${p}u + k];
      }
      (*a)[row * ${p}u + column] = value / pivot;
    }
  }
  return true;
}

fn choleskySolve_${p}(a: ptr<function, array<f32, ${size}>>, b: ptr<function, array<f32, ${p}>>) {
  for (var row = 0u; row < ${p}u; row++) {
    var value = (*b)[row];
    for (var k = 0u; k < row; k++) {
      value = value - (*a)[row * ${p}u + k] * (*b)[k];
    }
    (*b)[row] = value / (*a)[row * ${p}u + row];
  }
  for (var offset = 0u; offset < ${p}u; offset++) {
    let row = ${p - 1}u - offset;
    var value = (*b)[row];
    for (var k = row + 1u; k < ${p}u; k++) {
      value = value - (*a)[k * ${p}u + row] * (*b)[k];
    }
    (*b)[row] = value / (*a)[row * ${p}u + row];
  }
}

fn choleskyInverseDiagonal_${p}(a: ptr<function, array<f32, ${size}>>, column: u32) -> f32 {
  var unit: array<f32, ${p}>;
  for (var row = 0u; row < ${p}u; row++) {
    unit[row] = select(0.0, 1.0, row == column);
  }
  choleskySolve_${p}(a, &unit);
  return unit[column];
}
`;
}

/** Cyclic Jacobi sweeps of {@link getSymmetricEigenvalueRangeWGSL} and its CPU mirror. */
export const SYMMETRIC_EIGENVALUE_SWEEP_COUNT = 16;

/**
 * Returns WGSL for the extreme eigenvalues of a symmetric `p x p` matrix held in a function-local
 * array, with `p` fixed at compile time.
 *
 * Generated function (suffix `_${p}`):
 * - `getSymmetricEigenvalueRange_p(a: ptr<function, array<f32, p*p>>) -> vec2<f32>` returns
 *   `(smallest, largest)` eigenvalue. `a` must hold BOTH triangles (row-major `a[row * p + column]`)
 *   and is destroyed (it ends near-diagonal). Cyclic Jacobi rotations, at most
 *   `SYMMETRIC_EIGENVALUE_SWEEP_COUNT` sweeps, stopping when the off-diagonal mass falls below
 *   `1e-12` of the diagonal mass.
 *
 * Callers derive condition numbers from it: on a unit-diagonal (correlation-scaled) Gram matrix
 * `largest / smallest` is the 2-norm condition number of the Gram matrix and its square root the
 * condition number of the scaled design (Belsley's `condition_index`). The CPU reference lives in
 * test/reference/spatial-regression-solve-cpu.ts.
 */
export function getSymmetricEigenvalueRangeWGSL(predictorCount: number): string {
  validatePredictorCount(predictorCount);
  const p = predictorCount;
  return /* wgsl */ `
fn getSymmetricEigenvalueRange_${p}(a: ptr<function, array<f32, ${p * p}>>) -> vec2<f32> {
  for (var sweep = 0u; sweep < ${SYMMETRIC_EIGENVALUE_SWEEP_COUNT}u; sweep++) {
    var offDiagonal = 0.0;
    var diagonal = 0.0;
    for (var row = 0u; row < ${p}u; row++) {
      diagonal = diagonal + (*a)[row * ${p}u + row] * (*a)[row * ${p}u + row];
      for (var column = row + 1u; column < ${p}u; column++) {
        offDiagonal = offDiagonal + (*a)[row * ${p}u + column] * (*a)[row * ${p}u + column];
      }
    }
    // Both masses are sums of squares, so the f32 noise floor is about eps^2 (1.4e-14): a 1e-14
    // threshold is unreachable. 1e-12 on squared sums is about 1e-6 relative on the entries.
    if (offDiagonal <= 1.0e-12 * diagonal) {
      break;
    }
    for (var first = 0u; first + 1u < ${p}u; first++) {
      for (var second = first + 1u; second < ${p}u; second++) {
        let pivot = (*a)[first * ${p}u + second];
        if (abs(pivot) < 1.0e-30) {
          continue;
        }
        let theta = ((*a)[second * ${p}u + second] - (*a)[first * ${p}u + first]) / (2.0 * pivot);
        let tangent = select(-1.0, 1.0, theta >= 0.0) / (abs(theta) + sqrt(theta * theta + 1.0));
        let cosine = 1.0 / sqrt(tangent * tangent + 1.0);
        let sine = tangent * cosine;
        for (var k = 0u; k < ${p}u; k++) {
          let left = (*a)[k * ${p}u + first];
          let right = (*a)[k * ${p}u + second];
          (*a)[k * ${p}u + first] = cosine * left - sine * right;
          (*a)[k * ${p}u + second] = sine * left + cosine * right;
        }
        for (var k = 0u; k < ${p}u; k++) {
          let top = (*a)[first * ${p}u + k];
          let bottom = (*a)[second * ${p}u + k];
          (*a)[first * ${p}u + k] = cosine * top - sine * bottom;
          (*a)[second * ${p}u + k] = sine * top + cosine * bottom;
        }
      }
    }
  }
  var smallest = (*a)[0];
  var largest = (*a)[0];
  for (var row = 1u; row < ${p}u; row++) {
    smallest = min(smallest, (*a)[row * ${p}u + row]);
    largest = max(largest, (*a)[row * ${p}u + row]);
  }
  return vec2<f32>(smallest, largest);
}
`;
}

/**
 * Returns WGSL for the Belsley condition number (spreg / GeoDa `condition_index`) of a design
 * `[1, X]` with `k` predictor columns, computed WITHOUT forming the raw Gram matrix.
 *
 * Forming `X'X` with unit diagonal and taking `sqrt(lambdaMax / lambdaMin)` needs `lambdaMin` to
 * relative precision `1 / conditionNumber^2`, which float32 cannot hold beyond a condition number
 * of a few hundred. Instead, with column means `m_j`, centered sums of squares `C_jj` and the lower
 * Cholesky factor `L` of the predictor correlation matrix, each unit-length raw column is
 * `alpha_j u0 + beta_j z_j` with `u0 = 1 / sqrt(n)`, unit centered `z_j`, `alpha_j = sqrt(n) m_j /
 * sqrt(n m_j^2 + C_jj)` and `beta_j = sqrt(C_jj) / sqrt(n m_j^2 + C_jj)`. The scaled design then has
 * the singular values of the `(k + 1) x (k + 1)` matrix `B` with `B[0][0] = 1`, `B[0][j] = alpha_j`
 * and `B[1 + i][j] = beta_j L[j][i]` (`i <= j`), which one-sided (Hestenes) Jacobi rotations
 * orthogonalize to about `float32 epsilon * conditionNumber` relative accuracy.
 *
 * Generated function (suffix `_${k}`):
 * `getBelsleyConditionNumber_k(factor: ptr<function, array<f32, k*k>>, alpha: ptr<function,
 * array<f32, k>>, beta: ptr<function, array<f32, k>>) -> f32`, `factor` holding `L` row-major in
 * its lower triangle (as `choleskyFactor_k` leaves it). CPU reference:
 * test/reference/spatial-regression-solve-cpu.ts.
 */
export function getBelsleyConditionNumberWGSL(predictorCount: number): string {
  validatePredictorCount(predictorCount + 1);
  const k = predictorCount;
  const p = k + 1;
  return /* wgsl */ `
fn getBelsleyConditionNumber_${k}(
  factor: ptr<function, array<f32, ${k * k}>>,
  alpha: ptr<function, array<f32, ${k}>>,
  beta: ptr<function, array<f32, ${k}>>
) -> f32 {
  // Column-major B: b[column * ${p}u + row].
  var b: array<f32, ${p * p}>;
  b[0] = 1.0;
  for (var column = 1u; column < ${p}u; column++) {
    let j = column - 1u;
    b[column * ${p}u] = (*alpha)[j];
    for (var i = 0u; i <= j; i++) {
      b[column * ${p}u + 1u + i] = (*beta)[j] * (*factor)[j * ${k}u + i];
    }
  }
  for (var sweep = 0u; sweep < ${SYMMETRIC_EIGENVALUE_SWEEP_COUNT}u; sweep++) {
    var rotated = false;
    for (var first = 0u; first + 1u < ${p}u; first++) {
      for (var second = first + 1u; second < ${p}u; second++) {
        var normFirst = 0.0;
        var normSecond = 0.0;
        var product = 0.0;
        for (var row = 0u; row < ${p}u; row++) {
          let x = b[first * ${p}u + row];
          let y = b[second * ${p}u + row];
          normFirst = normFirst + x * x;
          normSecond = normSecond + y * y;
          product = product + x * y;
        }
        if (!(abs(product) > 1.0e-7 * sqrt(normFirst * normSecond))) {
          continue;
        }
        rotated = true;
        let zeta = (normSecond - normFirst) / (2.0 * product);
        let tangent = select(-1.0, 1.0, zeta >= 0.0) / (abs(zeta) + sqrt(1.0 + zeta * zeta));
        let cosine = 1.0 / sqrt(1.0 + tangent * tangent);
        let sine = cosine * tangent;
        for (var row = 0u; row < ${p}u; row++) {
          let x = b[first * ${p}u + row];
          let y = b[second * ${p}u + row];
          b[first * ${p}u + row] = cosine * x - sine * y;
          b[second * ${p}u + row] = sine * x + cosine * y;
        }
      }
    }
    if (!rotated) {
      break;
    }
  }
  var largest = 0.0;
  var smallest = 3.0e38;
  for (var column = 0u; column < ${p}u; column++) {
    var norm = 0.0;
    for (var row = 0u; row < ${p}u; row++) {
      norm = norm + b[column * ${p}u + row] * b[column * ${p}u + row];
    }
    largest = max(largest, sqrt(norm));
    smallest = min(smallest, sqrt(norm));
  }
  return largest / smallest;
}
`;
}

function validatePredictorCount(
  predictorCount: number,
  maximumSize = SPATIAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT
): void {
  if (!Number.isInteger(predictorCount) || predictorCount < 1 || predictorCount > maximumSize) {
    throw new Error(`Spatial regression predictor count must be an integer in [1, ${maximumSize}]`);
  }
}
