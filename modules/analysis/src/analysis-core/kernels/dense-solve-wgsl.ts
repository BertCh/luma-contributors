// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Small dense solvers for one invocation (contributor contract rule 11): a Cholesky factorization
 * for symmetric positive definite systems and an LU factorization with partial pivoting for
 * general ones. Both hold the matrix in a function-local array, so the capacity is a compile-time
 * constant and each generator takes it as an argument; the size actually solved is a runtime
 * `size <= capacity`, which lets one module serve neighborhoods of varying size.
 *
 * Storage: row-major `matrix[row * capacity + column]`. Cholesky reads only the lower triangle
 * (`column <= row`); LU reads the leading `size x size` block.
 *
 * Failure is an explicit status, never NaN propagation. Every factorization returns a struct whose
 * `status` is one of {@link DENSE_SOLVE_STATUS}:
 * - `OK` (0): factored; the factor and every solve are usable.
 * - `NONFINITE` (1): an input entry or an intermediate pivot is NaN or infinite. Nothing is usable.
 * - `SINGULAR` (2): a pivot is not positive (Cholesky: not positive definite) or at or below the
 *   pivot tolerance (LU). Nothing is usable.
 * - `ILL_CONDITIONED` (3): factored, but the condition estimate (`smallestPivotRatio`, or
 *   `pivotRatio` for LU) is below the caller's limit. The factor and the solves are still computed
 *   and finite, so callers that only need a flag can use the solution; callers that reject such
 *   systems test `status != 0u`.
 *
 * Status codes are written as literals in the generated WGSL; include
 * {@link DENSE_SOLVE_STATUS_WGSL} once if the kernel wants the named constants.
 *
 * Both generators need `isFiniteValue` from `wgsl-common.ts` (`IS_FINITE_VALUE_WGSL`) in the same
 * module. Including the output of one generator twice with the same capacity fails
 * compilation, as for every declaration string.
 *
 * Numerics are float32 with a fixed loop order, so results are deterministic on one adapter. The
 * headless spec `test/utils/dense-solve-wgsl.spec.ts` checks solutions, determinants, inverses and
 * every status against a float64 CPU reference.
 *
 * @internal
 */

/** Status codes returned by the dense factorizations. @internal */
export const DENSE_SOLVE_STATUS = {
  OK: 0,
  NONFINITE: 1,
  SINGULAR: 2,
  ILL_CONDITIONED: 3,
  NOT_CONVERGED: 4
} as const;

/**
 * Named WGSL constants for {@link DENSE_SOLVE_STATUS}. Optional; include once per module.
 * @internal
 */
export const DENSE_SOLVE_STATUS_WGSL = /* wgsl */ `
const DENSE_SOLVE_OK: u32 = 0u;
const DENSE_SOLVE_NONFINITE: u32 = 1u;
const DENSE_SOLVE_SINGULAR: u32 = 2u;
const DENSE_SOLVE_ILL_CONDITIONED: u32 = 3u;
const DENSE_SOLVE_NOT_CONVERGED: u32 = 4u;
`;

/**
 * Largest capacity of the generated solvers. The `capacity x capacity` f32 matrix lives in private
 * memory (16 KiB at 64), far above what GPUs keep in registers; larger systems belong to a
 * cooperative workgroup algorithm, not to one invocation.
 * @internal
 */
export const DENSE_SOLVE_MAXIMUM_CAPACITY = 64;

function validateDenseSolveCapacity(generator: string, capacity: number): void {
  if (!Number.isInteger(capacity) || capacity < 1 || capacity > DENSE_SOLVE_MAXIMUM_CAPACITY) {
    throw new Error(
      `${generator} capacity must be an integer in [1, ${DENSE_SOLVE_MAXIMUM_CAPACITY}], got ${capacity}`
    );
  }
}

/**
 * Returns WGSL for a Cholesky factorization `A = L L^T` of a symmetric positive definite system,
 * with optional symmetric diagonal equilibration and a relative pivot test. Suffix `_N` is the
 * capacity.
 *
 * - `struct DenseCholesky_N { scale: array<f32, N>, size: u32, status: u32, smallestPivotRatio: f32 }`
 * - `factorDenseCholesky_N(matrix, size, equilibrate, minimumPivotRatio) -> DenseCholesky_N`
 *   overwrites the lower triangle of `matrix` with `L`. With `equilibrate` it factors `S A S` with
 *   `S = diag(1 / sqrt(A_jj))` (the correlation matrix of a Gram matrix), which makes the factor
 *   independent of column scale; otherwise `scale` is all ones and the arithmetic is exactly that
 *   of the unequilibrated algorithm. The pivot ratio of column `j` is the Schur complement
 *   `A_jj - sum_k L_jk^2` over `A_jj` (both of the possibly equilibrated matrix): `1 - R^2` of column
 *   `j` regressed on the earlier columns, the reciprocal of its variance inflation factor.
 *   `smallestPivotRatio` is the smallest one. Status `SINGULAR` when a diagonal or pivot is not
 *   positive, `NONFINITE` when an entry of the lower triangle or a pivot is not finite, and
 *   `ILL_CONDITIONED` when `smallestPivotRatio < minimumPivotRatio` (pass 0 to disable). The OLS
 *   limit `GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE` (a factor pivot of `1e-3` on the
 *   equilibrated matrix) is `minimumPivotRatio = 1e-6`.
 * - `solveDenseCholesky_N(matrix, factorization, vector) -> bool` solves `A x = vector` in place.
 *   Returns false when a solution entry is not finite. Call it once per right-hand side.
 * - `getDenseCholeskyInverseDiagonal_N(matrix, factorization, column) -> f32` is `(A^-1)_cc`
 *   (standard errors, leverage).
 * - `invertDenseCholesky_N(matrix, factorization, inverse) -> bool` writes the full `A^-1`
 *   (row-major, stride `N`) into `inverse`.
 * - `getDenseCholeskyLogDeterminant_N(matrix, factorization) -> f32` is `log det A`.
 *
 * `factorization` is a `ptr<function, DenseCholesky_N>`. Requires `isFiniteValue`.
 *
 * @param capacity Largest system size, a compile-time constant.
 * @internal
 */
export function getDenseCholeskyWGSL(capacity: number): string {
  validateDenseSolveCapacity('getDenseCholeskyWGSL', capacity);
  const n = capacity;
  return /* wgsl */ `
struct DenseCholesky_${n} {
  scale: array<f32, ${n}>,
  size: u32,
  status: u32,
  smallestPivotRatio: f32,
}

fn factorDenseCholesky_${n}(
  matrix: ptr<function, array<f32, ${n * n}>>,
  size: u32,
  equilibrate: bool,
  minimumPivotRatio: f32
) -> DenseCholesky_${n} {
  var result: DenseCholesky_${n};
  result.size = min(size, ${n}u);
  result.status = 0u;
  result.smallestPivotRatio = 1.0;
  let count = result.size;
  for (var row = 0u; row < count; row++) {
    for (var column = 0u; column <= row; column++) {
      if (!isFiniteValue((*matrix)[row * ${n}u + column])) {
        result.status = 1u;
        return result;
      }
    }
  }
  for (var column = 0u; column < count; column++) {
    let diagonal = (*matrix)[column * ${n}u + column];
    if (!(diagonal > 0.0)) {
      result.status = 2u;
      return result;
    }
    result.scale[column] = select(1.0, 1.0 / sqrt(diagonal), equilibrate);
  }
  if (equilibrate) {
    for (var row = 0u; row < count; row++) {
      for (var column = 0u; column <= row; column++) {
        (*matrix)[row * ${n}u + column] = (*matrix)[row * ${n}u + column] * result.scale[row] * result.scale[column];
      }
    }
  }
  for (var column = 0u; column < count; column++) {
    let original = (*matrix)[column * ${n}u + column];
    var diagonal = original;
    for (var k = 0u; k < column; k++) {
      let value = (*matrix)[column * ${n}u + k];
      diagonal = diagonal - value * value;
    }
    if (!isFiniteValue(diagonal)) {
      result.status = 1u;
      return result;
    }
    if (!(diagonal > 0.0) || !(original > 0.0)) {
      result.status = 2u;
      result.smallestPivotRatio = 0.0;
      return result;
    }
    result.smallestPivotRatio = min(result.smallestPivotRatio, diagonal / original);
    let pivot = sqrt(diagonal);
    (*matrix)[column * ${n}u + column] = pivot;
    for (var row = column + 1u; row < count; row++) {
      var value = (*matrix)[row * ${n}u + column];
      for (var k = 0u; k < column; k++) {
        value = value - (*matrix)[row * ${n}u + k] * (*matrix)[column * ${n}u + k];
      }
      (*matrix)[row * ${n}u + column] = value / pivot;
    }
  }
  if (result.smallestPivotRatio < minimumPivotRatio) {
    result.status = 3u;
  }
  return result;
}

fn solveDenseCholesky_${n}(
  matrix: ptr<function, array<f32, ${n * n}>>,
  factorization: ptr<function, DenseCholesky_${n}>,
  vector: ptr<function, array<f32, ${n}>>
) -> bool {
  let count = (*factorization).size;
  for (var row = 0u; row < count; row++) {
    (*vector)[row] = (*vector)[row] * (*factorization).scale[row];
  }
  for (var row = 0u; row < count; row++) {
    var value = (*vector)[row];
    for (var k = 0u; k < row; k++) {
      value = value - (*matrix)[row * ${n}u + k] * (*vector)[k];
    }
    (*vector)[row] = value / (*matrix)[row * ${n}u + row];
  }
  for (var offset = 0u; offset < count; offset++) {
    let row = count - 1u - offset;
    var value = (*vector)[row];
    for (var k = row + 1u; k < count; k++) {
      value = value - (*matrix)[k * ${n}u + row] * (*vector)[k];
    }
    (*vector)[row] = value / (*matrix)[row * ${n}u + row];
  }
  var finite = true;
  for (var row = 0u; row < count; row++) {
    (*vector)[row] = (*vector)[row] * (*factorization).scale[row];
    finite = finite && isFiniteValue((*vector)[row]);
  }
  return finite;
}

fn getDenseCholeskyInverseDiagonal_${n}(
  matrix: ptr<function, array<f32, ${n * n}>>,
  factorization: ptr<function, DenseCholesky_${n}>,
  column: u32
) -> f32 {
  var unit: array<f32, ${n}>;
  unit[column] = 1.0;
  solveDenseCholesky_${n}(matrix, factorization, &unit);
  return unit[column];
}

fn invertDenseCholesky_${n}(
  matrix: ptr<function, array<f32, ${n * n}>>,
  factorization: ptr<function, DenseCholesky_${n}>,
  inverse: ptr<function, array<f32, ${n * n}>>
) -> bool {
  let count = (*factorization).size;
  var finite = true;
  for (var column = 0u; column < count; column++) {
    var unit: array<f32, ${n}>;
    unit[column] = 1.0;
    finite = solveDenseCholesky_${n}(matrix, factorization, &unit) && finite;
    for (var row = 0u; row < count; row++) {
      (*inverse)[row * ${n}u + column] = unit[row];
    }
  }
  return finite;
}

fn getDenseCholeskyLogDeterminant_${n}(
  matrix: ptr<function, array<f32, ${n * n}>>,
  factorization: ptr<function, DenseCholesky_${n}>
) -> f32 {
  var total = 0.0;
  for (var row = 0u; row < (*factorization).size; row++) {
    total = total + 2.0 * (log((*matrix)[row * ${n}u + row]) - log((*factorization).scale[row]));
  }
  return total;
}
`;
}

/**
 * Returns WGSL for an LU factorization with partial (row) pivoting, `P R A C = L U`, of a general
 * square system, with optional Ruiz equilibration (`R`, `C` diagonal). Suffix `_N` is the capacity.
 *
 * - `struct DenseLU_N { rowScale, columnScale: array<f32, N>, pivots: array<u32, N>, size, status:
 *   u32, pivotRatio, logAbsDeterminant, determinantSign: f32 }`
 * - `factorDenseLU_N(matrix, size, equilibrationPasses, pivotTolerance, minimumPivotRatio) ->
 *   DenseLU_N` overwrites the leading block of `matrix` with the unit-lower `L` (below the diagonal)
 *   and `U`. `equilibrationPasses` Ruiz passes scale every row and column of the current matrix by
 *   `1 / sqrt` of its largest magnitude at the same time, so a symmetric matrix keeps symmetric
 *   scales (`R = C`); 0 disables scaling, and 4 is what the local kriging solver uses. The tests run
 *   on the scaled matrix, so they are relative and scale invariant: a pivot at or below
 *   `pivotTolerance` times the largest scaled entry is `SINGULAR` (pass 0 to reject exact zeros
 *   only); `pivotRatio` (smallest over largest pivot magnitude, which tracks `1 / cond` to within
 *   a small factor) below `minimumPivotRatio` is `ILL_CONDITIONED`. A zero row or column during
 *   equilibration is `SINGULAR`. `logAbsDeterminant` and `determinantSign` give `det A` of the
 *   unscaled matrix.
 * - `solveDenseLU_N(matrix, factorization, vector) -> bool` solves `A x = vector` in place. Returns
 *   false when a solution entry is not finite. Call it once per right-hand side.
 * - `invertDenseLU_N(matrix, factorization, inverse) -> bool` writes the full `A^-1` (row-major,
 *   stride `N`).
 *
 * `factorization` is a `ptr<function, DenseLU_N>`. Requires `isFiniteValue`.
 *
 * @param capacity Largest system size, a compile-time constant.
 * @internal
 */
export function getDenseLUWGSL(capacity: number): string {
  validateDenseSolveCapacity('getDenseLUWGSL', capacity);
  const n = capacity;
  return /* wgsl */ `
struct DenseLU_${n} {
  rowScale: array<f32, ${n}>,
  columnScale: array<f32, ${n}>,
  pivots: array<u32, ${n}>,
  size: u32,
  status: u32,
  pivotRatio: f32,
  logAbsDeterminant: f32,
  determinantSign: f32,
}

fn factorDenseLU_${n}(
  matrix: ptr<function, array<f32, ${n * n}>>,
  size: u32,
  equilibrationPasses: u32,
  pivotTolerance: f32,
  minimumPivotRatio: f32
) -> DenseLU_${n} {
  var result: DenseLU_${n};
  result.size = min(size, ${n}u);
  result.status = 0u;
  result.pivotRatio = 0.0;
  result.logAbsDeterminant = 0.0;
  result.determinantSign = 1.0;
  let count = result.size;
  for (var row = 0u; row < count; row++) {
    result.rowScale[row] = 1.0;
    result.columnScale[row] = 1.0;
    result.pivots[row] = row;
    for (var column = 0u; column < count; column++) {
      if (!isFiniteValue((*matrix)[row * ${n}u + column])) {
        result.status = 1u;
        return result;
      }
    }
  }
  for (var equilibrationPass = 0u; equilibrationPass < equilibrationPasses; equilibrationPass++) {
    var rowMaximum: array<f32, ${n}>;
    var columnMaximum: array<f32, ${n}>;
    for (var row = 0u; row < count; row++) {
      for (var column = 0u; column < count; column++) {
        let magnitude = abs((*matrix)[row * ${n}u + column]) * result.rowScale[row] * result.columnScale[column];
        rowMaximum[row] = max(rowMaximum[row], magnitude);
        columnMaximum[column] = max(columnMaximum[column], magnitude);
      }
    }
    for (var index = 0u; index < count; index++) {
      if (!(rowMaximum[index] > 0.0) || !(columnMaximum[index] > 0.0)) {
        result.status = 2u;
        return result;
      }
      result.rowScale[index] = result.rowScale[index] / sqrt(rowMaximum[index]);
      result.columnScale[index] = result.columnScale[index] / sqrt(columnMaximum[index]);
    }
  }
  var largestEntry = 0.0;
  for (var row = 0u; row < count; row++) {
    for (var column = 0u; column < count; column++) {
      let value = (*matrix)[row * ${n}u + column] * result.rowScale[row] * result.columnScale[column];
      (*matrix)[row * ${n}u + column] = value;
      largestEntry = max(largestEntry, abs(value));
    }
  }
  var smallestPivot = 3.402823e38;
  var largestPivot = 0.0;
  for (var column = 0u; column < count; column++) {
    var pivotRow = column;
    for (var row = column + 1u; row < count; row++) {
      if (abs((*matrix)[row * ${n}u + column]) > abs((*matrix)[pivotRow * ${n}u + column])) {
        pivotRow = row;
      }
    }
    let pivotMagnitude = abs((*matrix)[pivotRow * ${n}u + column]);
    if (!isFiniteValue(pivotMagnitude)) {
      result.status = 1u;
      return result;
    }
    if (!(pivotMagnitude > pivotTolerance * largestEntry) || !(pivotMagnitude > 0.0)) {
      result.status = 2u;
      return result;
    }
    result.pivots[column] = pivotRow;
    if (pivotRow != column) {
      result.determinantSign = -result.determinantSign;
      for (var entry = 0u; entry < count; entry++) {
        let held = (*matrix)[column * ${n}u + entry];
        (*matrix)[column * ${n}u + entry] = (*matrix)[pivotRow * ${n}u + entry];
        (*matrix)[pivotRow * ${n}u + entry] = held;
      }
    }
    let pivot = (*matrix)[column * ${n}u + column];
    smallestPivot = min(smallestPivot, pivotMagnitude);
    largestPivot = max(largestPivot, pivotMagnitude);
    result.logAbsDeterminant = result.logAbsDeterminant + log(pivotMagnitude);
    if (pivot < 0.0) {
      result.determinantSign = -result.determinantSign;
    }
    for (var row = column + 1u; row < count; row++) {
      let factor = (*matrix)[row * ${n}u + column] / pivot;
      (*matrix)[row * ${n}u + column] = factor;
      for (var entry = column + 1u; entry < count; entry++) {
        (*matrix)[row * ${n}u + entry] = (*matrix)[row * ${n}u + entry] - factor * (*matrix)[column * ${n}u + entry];
      }
    }
  }
  for (var index = 0u; index < count; index++) {
    result.logAbsDeterminant = result.logAbsDeterminant - log(result.rowScale[index]) - log(result.columnScale[index]);
  }
  result.pivotRatio = select(1.0, smallestPivot / largestPivot, count > 0u);
  if (result.pivotRatio < minimumPivotRatio) {
    result.status = 3u;
  }
  return result;
}

fn solveDenseLU_${n}(
  matrix: ptr<function, array<f32, ${n * n}>>,
  factorization: ptr<function, DenseLU_${n}>,
  vector: ptr<function, array<f32, ${n}>>
) -> bool {
  let count = (*factorization).size;
  for (var row = 0u; row < count; row++) {
    (*vector)[row] = (*vector)[row] * (*factorization).rowScale[row];
  }
  for (var column = 0u; column < count; column++) {
    let pivotRow = (*factorization).pivots[column];
    if (pivotRow != column) {
      let held = (*vector)[column];
      (*vector)[column] = (*vector)[pivotRow];
      (*vector)[pivotRow] = held;
    }
  }
  for (var row = 0u; row < count; row++) {
    var value = (*vector)[row];
    for (var k = 0u; k < row; k++) {
      value = value - (*matrix)[row * ${n}u + k] * (*vector)[k];
    }
    (*vector)[row] = value;
  }
  var finite = true;
  for (var offset = 0u; offset < count; offset++) {
    let row = count - 1u - offset;
    var value = (*vector)[row];
    for (var k = row + 1u; k < count; k++) {
      value = value - (*matrix)[row * ${n}u + k] * (*vector)[k];
    }
    (*vector)[row] = value / (*matrix)[row * ${n}u + row];
  }
  for (var row = 0u; row < count; row++) {
    (*vector)[row] = (*vector)[row] * (*factorization).columnScale[row];
    finite = finite && isFiniteValue((*vector)[row]);
  }
  return finite;
}

fn invertDenseLU_${n}(
  matrix: ptr<function, array<f32, ${n * n}>>,
  factorization: ptr<function, DenseLU_${n}>,
  inverse: ptr<function, array<f32, ${n * n}>>
) -> bool {
  let count = (*factorization).size;
  var finite = true;
  for (var column = 0u; column < count; column++) {
    var unit: array<f32, ${n}>;
    unit[column] = 1.0;
    finite = solveDenseLU_${n}(matrix, factorization, &unit) && finite;
    for (var row = 0u; row < count; row++) {
      (*inverse)[row * ${n}u + column] = unit[row];
    }
  }
  return finite;
}
`;
}
