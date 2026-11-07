// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CPU reference mirrors of the WGSL small dense solvers. Test-only; not part of the package API.

import {SYMMETRIC_EIGENVALUE_SWEEP_COUNT} from '../../src/gpu-spatial-analysis/spatial-regression/spatial-regression-solve';

/**
 * CPU mirror of `choleskyFactor_p`: factors row-major `matrix` (length `p * p`) in place with f32
 * rounding after every operation. Returns false for a non-positive or non-finite pivot.
 */
export function factorCholeskyOnCPU(matrix: Float32Array, predictorCount: number): boolean {
  const p = predictorCount;
  const round = Math.fround;
  for (let column = 0; column < p; column++) {
    let diagonal = matrix[column * p + column];
    for (let k = 0; k < column; k++) {
      const value = matrix[column * p + k];
      diagonal = round(diagonal - round(value * value));
    }
    if (!(diagonal > 0) || diagonal > 3.0e38) {
      return false;
    }
    const pivot = round(Math.sqrt(diagonal));
    matrix[column * p + column] = pivot;
    for (let row = column + 1; row < p; row++) {
      let value = matrix[row * p + column];
      for (let k = 0; k < column; k++) {
        value = round(value - round(matrix[row * p + k] * matrix[column * p + k]));
      }
      matrix[row * p + column] = round(value / pivot);
    }
  }
  return true;
}

/** CPU mirror of `choleskySolve_p`: solves in place in `rightHandSide` with f32 rounding. */
export function solveCholeskyOnCPU(
  factor: Float32Array,
  rightHandSide: Float32Array,
  predictorCount: number
): void {
  const p = predictorCount;
  const round = Math.fround;
  for (let row = 0; row < p; row++) {
    let value = rightHandSide[row];
    for (let k = 0; k < row; k++) {
      value = round(value - round(factor[row * p + k] * rightHandSide[k]));
    }
    rightHandSide[row] = round(value / factor[row * p + row]);
  }
  for (let row = p - 1; row >= 0; row--) {
    let value = rightHandSide[row];
    for (let k = row + 1; k < p; k++) {
      value = round(value - round(factor[k * p + row] * rightHandSide[k]));
    }
    rightHandSide[row] = round(value / factor[row * p + row]);
  }
}

/**
 * Float64 CPU mirror of `getSymmetricEigenvalueRange_p`: the same rotation order and stopping
 * rule. `matrix` holds both triangles row-major and is destroyed. Returns `[smallest, largest]`.
 */
export function getSymmetricEigenvalueRangeOnCPU(
  matrix: Float64Array | number[],
  predictorCount: number
): [number, number] {
  const p = predictorCount;
  for (let sweep = 0; sweep < SYMMETRIC_EIGENVALUE_SWEEP_COUNT; sweep++) {
    let offDiagonal = 0;
    let diagonal = 0;
    for (let row = 0; row < p; row++) {
      diagonal += matrix[row * p + row] ** 2;
      for (let column = row + 1; column < p; column++) {
        offDiagonal += matrix[row * p + column] ** 2;
      }
    }
    // The f64 oracle keeps the tight 1e-14 stop (reachable in f64); the f32 WGSL uses 1e-12.
    if (offDiagonal <= 1e-14 * diagonal) {
      break;
    }
    for (let first = 0; first + 1 < p; first++) {
      for (let second = first + 1; second < p; second++) {
        const pivot = matrix[first * p + second];
        if (Math.abs(pivot) < 1e-30) {
          continue;
        }
        const theta = (matrix[second * p + second] - matrix[first * p + first]) / (2 * pivot);
        const tangent = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const cosine = 1 / Math.sqrt(tangent * tangent + 1);
        const sine = tangent * cosine;
        for (let k = 0; k < p; k++) {
          const left = matrix[k * p + first];
          const right = matrix[k * p + second];
          matrix[k * p + first] = cosine * left - sine * right;
          matrix[k * p + second] = sine * left + cosine * right;
        }
        for (let k = 0; k < p; k++) {
          const top = matrix[first * p + k];
          const bottom = matrix[second * p + k];
          matrix[first * p + k] = cosine * top - sine * bottom;
          matrix[second * p + k] = sine * top + cosine * bottom;
        }
      }
    }
  }
  let smallest = matrix[0];
  let largest = matrix[0];
  for (let row = 1; row < p; row++) {
    smallest = Math.min(smallest, matrix[row * p + row]);
    largest = Math.max(largest, matrix[row * p + row]);
  }
  return [smallest, largest];
}

/**
 * Float64 mirror of `getBelsleyConditionNumber_k` (same construction and rotation order).
 * `factor` is the row-major lower Cholesky factor of the `k x k` predictor correlation matrix.
 */
export function getBelsleyConditionNumberOnCPU(
  factor: ArrayLike<number>,
  alpha: ArrayLike<number>,
  beta: ArrayLike<number>,
  predictorCount: number
): number {
  const k = predictorCount;
  const p = k + 1;
  const b = new Float64Array(p * p);
  b[0] = 1;
  for (let column = 1; column < p; column++) {
    const j = column - 1;
    b[column * p] = alpha[j];
    for (let i = 0; i <= j; i++) {
      b[column * p + 1 + i] = beta[j] * factor[j * k + i];
    }
  }
  for (let sweep = 0; sweep < SYMMETRIC_EIGENVALUE_SWEEP_COUNT; sweep++) {
    let rotated = false;
    for (let first = 0; first + 1 < p; first++) {
      for (let second = first + 1; second < p; second++) {
        let normFirst = 0;
        let normSecond = 0;
        let product = 0;
        for (let row = 0; row < p; row++) {
          const x = b[first * p + row];
          const y = b[second * p + row];
          normFirst += x * x;
          normSecond += y * y;
          product += x * y;
        }
        if (!(Math.abs(product) > 1e-15 * Math.sqrt(normFirst * normSecond))) {
          continue;
        }
        rotated = true;
        const zeta = (normSecond - normFirst) / (2 * product);
        const tangent = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const cosine = 1 / Math.sqrt(1 + tangent * tangent);
        const sine = cosine * tangent;
        for (let row = 0; row < p; row++) {
          const x = b[first * p + row];
          const y = b[second * p + row];
          b[first * p + row] = cosine * x - sine * y;
          b[second * p + row] = sine * x + cosine * y;
        }
      }
    }
    if (!rotated) {
      break;
    }
  }
  let largest = 0;
  let smallest = Infinity;
  for (let column = 0; column < p; column++) {
    let norm = 0;
    for (let row = 0; row < p; row++) {
      norm += b[column * p + row] ** 2;
    }
    largest = Math.max(largest, Math.sqrt(norm));
    smallest = Math.min(smallest, Math.sqrt(norm));
  }
  return largest / smallest;
}
