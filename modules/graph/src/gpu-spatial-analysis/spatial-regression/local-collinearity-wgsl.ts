// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Largest condition number reported for numerically singular local designs. @internal */
export const LOCAL_COLLINEARITY_CONDITION_CAP = 3.0e38;

/** Singular values with a ratio below this are numerically rank deficient in float32. @internal */
export const LOCAL_COLLINEARITY_SINGULAR_RATIO = 1.0e-7;

/**
 * WGSL for singular values of a weighted local design without forming its normal equations.
 *
 * `updateDesignFactor_n` folds one scaled design row into an upper-triangular factor with Givens
 * rotations (streaming QR), and `getDesignSingularValues_n` runs one-sided Jacobi on that factor.
 * Squaring the design (a Gram matrix and an eigen solve) only resolves condition numbers up to a
 * few thousand in float32; this path is accurate to about `epsilon * condition`. The right
 * singular vectors feed the Belsley variance decomposition. @internal
 */
export function getDesignSingularValueWGSL(size: number): string {
  const square = size * size;
  return /* wgsl */ `
fn updateDesignFactor_${size}(factor: ptr<function, array<f32, ${square}>>, row: ptr<function, array<f32, ${size}>>) {
  for (var column = 0u; column < ${size}u; column++) {
    let value = (*row)[column];
    if (value == 0.0) {
      continue;
    }
    let diagonal = (*factor)[column * ${size}u + column];
    let radius = sqrt(diagonal * diagonal + value * value);
    let cosine = diagonal / radius;
    let sine = value / radius;
    for (var entry = column; entry < ${size}u; entry++) {
      let held = (*factor)[column * ${size}u + entry];
      let incoming = (*row)[entry];
      (*factor)[column * ${size}u + entry] = cosine * held + sine * incoming;
      (*row)[entry] = cosine * incoming - sine * held;
    }
  }
}

// Diagonal entry of (R^T R)^-1 = R^-1 R^-T, the variance inflation factor when the design columns
// have unit norm. R is upper triangular and must be nonsingular.
fn getDesignFactorInverseDiagonal_${size}(factor: ptr<function, array<f32, ${square}>>, column: u32) -> f32 {
  var total = 0.0;
  for (var last = column; last < ${size}u; last++) {
    // Solve R x = e_target by back substitution; only x[column] is accumulated.
    var solution: array<f32, ${size}>;
    solution[last] = 1.0 / (*factor)[last * ${size}u + last];
    for (var offset = 1u; offset <= last; offset++) {
      let row = last - offset;
      var value = 0.0;
      for (var other = row + 1u; other <= last; other++) {
        value += (*factor)[row * ${size}u + other] * solution[other];
      }
      solution[row] = -value / (*factor)[row * ${size}u + row];
    }
    total += solution[column] * solution[column];
  }
  return total;
}

// One-sided Jacobi on the factor columns. vectors is column-major (right singular vectors);
// returns false when the sweeps are exhausted.
fn getDesignSingularValues_${size}(
  factor: ptr<function, array<f32, ${square}>>,
  vectors: ptr<function, array<f32, ${square}>>,
  values: ptr<function, array<f32, ${size}>>
) -> bool {
  var columns: array<f32, ${square}>;
  for (var column = 0u; column < ${size}u; column++) {
    for (var row = 0u; row < ${size}u; row++) {
      columns[column * ${size}u + row] = (*factor)[row * ${size}u + column];
      (*vectors)[column * ${size}u + row] = select(0.0, 1.0, row == column);
    }
  }
  var converged = false;
  for (var sweep = 0u; sweep < 64u; sweep++) {
    var rotated = false;
    for (var first = 0u; first + 1u < ${size}u; first++) {
      for (var second = first + 1u; second < ${size}u; second++) {
        var normFirst = 0.0;
        var normSecond = 0.0;
        var product = 0.0;
        for (var row = 0u; row < ${size}u; row++) {
          let x = columns[first * ${size}u + row];
          let y = columns[second * ${size}u + row];
          normFirst += x * x;
          normSecond += y * y;
          product += x * y;
        }
        if (!(abs(product) > 1.0e-6 * sqrt(normFirst * normSecond))) {
          continue;
        }
        rotated = true;
        let zeta = (normSecond - normFirst) / (2.0 * product);
        let tangent = select(-1.0, 1.0, zeta >= 0.0) / (abs(zeta) + sqrt(1.0 + zeta * zeta));
        let cosine = inverseSqrt(1.0 + tangent * tangent);
        let sine = cosine * tangent;
        for (var row = 0u; row < ${size}u; row++) {
          let x = columns[first * ${size}u + row];
          let y = columns[second * ${size}u + row];
          columns[first * ${size}u + row] = cosine * x - sine * y;
          columns[second * ${size}u + row] = sine * x + cosine * y;
          let vectorFirst = (*vectors)[first * ${size}u + row];
          let vectorSecond = (*vectors)[second * ${size}u + row];
          (*vectors)[first * ${size}u + row] = cosine * vectorFirst - sine * vectorSecond;
          (*vectors)[second * ${size}u + row] = sine * vectorFirst + cosine * vectorSecond;
        }
      }
    }
    if (!rotated) {
      converged = true;
      break;
    }
  }
  for (var column = 0u; column < ${size}u; column++) {
    var norm = 0.0;
    for (var row = 0u; row < ${size}u; row++) {
      norm += columns[column * ${size}u + row] * columns[column * ${size}u + row];
    }
    (*values)[column] = sqrt(norm);
  }
  return converged;
}
`;
}
