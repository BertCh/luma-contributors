// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {DENSE_SOLVE_MAXIMUM_CAPACITY} from './dense-solve-wgsl';

/**
 * Returns WGSL for the eigen decomposition `A = V diag(values) V^T` of a symmetric matrix held in
 * a function-local array, by cyclic Jacobi rotations (contributor contract rule 11; the companion
 * of `dense-solve-wgsl.ts`). Suffix `_N` is the capacity, a compile-time constant; the size is a
 * runtime `size <= capacity`.
 *
 * - `struct SymmetricEigen_N { values: array<f32, N>, size: u32, status: u32, sweeps: u32 }`
 * - `getSymmetricEigen_N(matrix, vectors, size, maximumSweeps, tolerance) -> SymmetricEigen_N`
 *   reads the lower triangle of `matrix` (row-major, stride `N`) and destroys `matrix`. It writes
 *   the eigenvalues in ascending order to `values` and the matching unit eigenvectors to the
 *   columns of `vectors` (`vectors[row * N + column]` is component `row` of vector `column`).
 *   A sweep visits every pair `lower < upper` once. The iteration stops when the off-diagonal Frobenius
 *   norm is at most `tolerance` times the diagonal norm (checked before each sweep; `1e-6` is the
 *   criterion of the spatial-regression condition numbers) or the matrix is diagonal. The test
 *   runs on entries scaled by a power of two (exact) chosen from the largest input entry, so it
 *   is invariant to the matrix scale: entries near `1e-25` or `1e20` iterate like entries near 1
 *   (subnormal-only input, which a GPU may flush to zero, is the exception).
 *   Status (`DENSE_SOLVE_STATUS`): `OK` (0), `NONFINITE` (1, a lower-triangle entry within `size`
 *   is NaN or infinite; `matrix`, `vectors` and `values` are left untouched), or `NOT_CONVERGED` (4, `maximumSweeps` sweeps ran; values and vectors
 *   are the last iterate, sorted). `sweeps` is the number of sweeps run.
 *
 * Rotation angles use the small root `t = sign(theta) / (|theta| + sqrt(theta^2 + 1))`, with
 * `t = 1 / (2 theta)` once `theta^2` would overflow, so no `atan2` is evaluated. Requires
 * `isFiniteValue` from `wgsl-common.ts`.
 *
 * The CPU float64 reference is in `test/utils/symmetric-eigen-wgsl.spec.ts`.
 *
 * @param capacity Largest matrix size.
 * @internal
 */
export function getSymmetricEigenWGSL(capacity: number): string {
  if (!Number.isInteger(capacity) || capacity < 1 || capacity > DENSE_SOLVE_MAXIMUM_CAPACITY) {
    throw new Error(
      `getSymmetricEigenWGSL capacity must be an integer in [1, ${DENSE_SOLVE_MAXIMUM_CAPACITY}], got ${capacity}`
    );
  }
  const n = capacity;
  return /* wgsl */ `
struct SymmetricEigen_${n} {
  values: array<f32, ${n}>,
  size: u32,
  status: u32,
  sweeps: u32,
}

fn getSymmetricEigen_${n}(
  matrix: ptr<function, array<f32, ${n * n}>>,
  vectors: ptr<function, array<f32, ${n * n}>>,
  size: u32,
  maximumSweeps: u32,
  tolerance: f32
) -> SymmetricEigen_${n} {
  var result: SymmetricEigen_${n};
  result.size = min(size, ${n}u);
  result.status = 0u;
  result.sweeps = 0u;
  let count = result.size;
  // Validate first, so a NONFINITE return leaves matrix and vectors untouched.
  var largest = 0.0;
  for (var row = 0u; row < count; row++) {
    for (var column = 0u; column <= row; column++) {
      let value = (*matrix)[row * ${n}u + column];
      if (!isFiniteValue(value)) {
        result.status = 1u;
        return result;
      }
      largest = max(largest, abs(value));
    }
  }
  for (var row = 0u; row < count; row++) {
    for (var column = 0u; column <= row; column++) {
      (*matrix)[column * ${n}u + row] = (*matrix)[row * ${n}u + column];
      (*vectors)[row * ${n}u + column] = select(0.0, 1.0, row == column);
      (*vectors)[column * ${n}u + row] = select(0.0, 1.0, row == column);
    }
  }
  // The stopping test runs on entries scaled by the power of two that brings the largest input
  // entry into [0.5, 1): exact, so well-scaled input stops exactly as unscaled, and the squares
  // neither underflow (entries below about 1e-19) nor overflow (above about 1.8e19). Rotations
  // keep the Frobenius norm, so scaled entries stay at most the size.
  let scaleExponent = -frexp(largest).exp;
  for (var sweep = 0u; sweep <= maximumSweeps; sweep++) {
    var offDiagonal = 0.0;
    var diagonal = 0.0;
    for (var row = 0u; row < count; row++) {
      let diagonalValue = ldexp((*matrix)[row * ${n}u + row], scaleExponent);
      diagonal = diagonal + diagonalValue * diagonalValue;
      for (var column = row + 1u; column < count; column++) {
        let offDiagonalValue = ldexp((*matrix)[row * ${n}u + column], scaleExponent);
        offDiagonal = offDiagonal + offDiagonalValue * offDiagonalValue;
      }
    }
    if (offDiagonal == 0.0 || offDiagonal <= tolerance * tolerance * diagonal) {
      break;
    }
    if (sweep == maximumSweeps) {
      result.status = 4u;
      break;
    }
    result.sweeps = sweep + 1u;
    for (var lower = 0u; lower + 1u < count; lower++) {
      for (var upper = lower + 1u; upper < count; upper++) {
        let coupling = (*matrix)[lower * ${n}u + upper];
        if (coupling == 0.0) {
          continue;
        }
        let theta = ((*matrix)[upper * ${n}u + upper] - (*matrix)[lower * ${n}u + lower]) / (2.0 * coupling);
        let direction = select(-1.0, 1.0, theta >= 0.0);
        let tangent = select(direction / (abs(theta) + sqrt(theta * theta + 1.0)), 0.5 / theta, abs(theta) > 1.0e18);
        let cosine = 1.0 / sqrt(tangent * tangent + 1.0);
        let sine = tangent * cosine;
        for (var k = 0u; k < count; k++) {
          let lowerValue = (*matrix)[k * ${n}u + lower];
          let upperValue = (*matrix)[k * ${n}u + upper];
          (*matrix)[k * ${n}u + lower] = cosine * lowerValue - sine * upperValue;
          (*matrix)[k * ${n}u + upper] = sine * lowerValue + cosine * upperValue;
        }
        for (var k = 0u; k < count; k++) {
          let lowerValue = (*matrix)[lower * ${n}u + k];
          let upperValue = (*matrix)[upper * ${n}u + k];
          (*matrix)[lower * ${n}u + k] = cosine * lowerValue - sine * upperValue;
          (*matrix)[upper * ${n}u + k] = sine * lowerValue + cosine * upperValue;
        }
        (*matrix)[lower * ${n}u + upper] = 0.0;
        (*matrix)[upper * ${n}u + lower] = 0.0;
        for (var k = 0u; k < count; k++) {
          let lowerValue = (*vectors)[k * ${n}u + lower];
          let upperValue = (*vectors)[k * ${n}u + upper];
          (*vectors)[k * ${n}u + lower] = cosine * lowerValue - sine * upperValue;
          (*vectors)[k * ${n}u + upper] = sine * lowerValue + cosine * upperValue;
        }
      }
    }
  }
  for (var index = 0u; index < count; index++) {
    result.values[index] = (*matrix)[index * ${n}u + index];
  }
  // Selection sort, ascending, moving the vector columns with their values.
  for (var index = 0u; index + 1u < count; index++) {
    var smallest = index;
    for (var other = index + 1u; other < count; other++) {
      if (result.values[other] < result.values[smallest]) {
        smallest = other;
      }
    }
    if (smallest != index) {
      let held = result.values[index];
      result.values[index] = result.values[smallest];
      result.values[smallest] = held;
      for (var k = 0u; k < count; k++) {
        let component = (*vectors)[k * ${n}u + index];
        (*vectors)[k * ${n}u + index] = (*vectors)[k * ${n}u + smallest];
        (*vectors)[k * ${n}u + smallest] = component;
      }
    }
  }
  return result;
}
`;
}

/**
 * Smallest eigenpair of a finite symmetric 2 by 2 matrix with entries `(first, cross, second)`.
 * Returns `SmallestSymmetricEigen2 {value, vector}`. Repeated eigenvalues choose `(1, 0)`;
 * otherwise the unit vector has nonnegative first component (the sign at an axis is arbitrary).
 *
 * The rationalized Jacobi tangent avoids half-trace subtraction. Scaling is used only to find
 * that tangent, not to store the diagonals: `min(first, second) - abs(cross) * tangent` retains a
 * small normal negative eigenvalue beside a much larger positive diagonal even when their ratio
 * underflows float32. Unit vectors are formed from `(1, tangent)`, never from tiny curvatures.
 * Subnormal inputs/results can still be flushed by the GPU. Requires finite input entries.
 * @internal
 */
export const SMALLEST_SYMMETRIC_EIGEN_2_WGSL = /* wgsl */ `
struct SmallestSymmetricEigen2 {
  value: f32,
  vector: vec2<f32>,
}

fn getSmallestSymmetricEigen2(first: f32, cross: f32, second: f32) -> SmallestSymmetricEigen2 {
  if (cross == 0.0) {
    return SmallestSymmetricEigen2(min(first, second),
      select(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0), second < first));
  }
  let exponent = frexp(max(abs(first), max(abs(cross), abs(second)))).exp;
  let scaledFirst = ldexp(first, -exponent);
  let scaledSecond = ldexp(second, -exponent);
  let scaledCross = abs(ldexp(cross, -exponent));
  let halfGap = 0.5 * abs(scaledFirst - scaledSecond);
  var tangent = 1.0;
  if (halfGap > 0.0) {
    // Either the gap or coupling is large enough to square safely. If the gap is zero,
    // the tangent is exactly one, including when scaledCross underflows.
    let radius = length(vec2<f32>(halfGap, scaledCross));
    tangent = scaledCross / (halfGap + radius);
  }
  let correction = abs(cross) * tangent;
  let crossSign = select(-1.0, 1.0, cross > 0.0);
  let direction = select(vec2<f32>(1.0, -crossSign * tangent),
    vec2<f32>(tangent, -crossSign), second < first);
  return SmallestSymmetricEigen2(min(first, second) - correction,
    direction / sqrt(1.0 + tangent * tangent));
}
`;
