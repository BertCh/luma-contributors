// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Solves `matrix x = rightHandSide` in place with Gauss-Jordan elimination and partial pivoting.
 * Each right-hand side row holds one or more solution columns and receives the solution. The
 * pivot tolerance scales with the largest matrix entry, so squared-meter semivariances and
 * unit-scale normal equations are judged singular consistently. Returns false when singular.
 */
export function solveDenseLinearSystem(
  matrix: (number[] | Float64Array)[],
  rightHandSide: (number[] | Float64Array)[]
): boolean {
  const size = matrix.length;
  let scale = 1e-300;
  for (const row of matrix) for (const value of row) scale = Math.max(scale, Math.abs(value));
  for (let pivot = 0; pivot < size; pivot++) {
    let best = pivot;
    for (let row = pivot + 1; row < size; row++)
      if (Math.abs(matrix[row][pivot]) > Math.abs(matrix[best][pivot])) best = row;
    if (!(Math.abs(matrix[best][pivot]) >= 1e-12 * scale)) return false;
    [matrix[pivot], matrix[best]] = [matrix[best], matrix[pivot]];
    [rightHandSide[pivot], rightHandSide[best]] = [rightHandSide[best], rightHandSide[pivot]];
    const divisor = matrix[pivot][pivot];
    for (let column = 0; column < size; column++) matrix[pivot][column] /= divisor;
    for (let column = 0; column < rightHandSide[pivot].length; column++)
      rightHandSide[pivot][column] /= divisor;
    for (let row = 0; row < size; row++) {
      if (row === pivot) continue;
      const factor = matrix[row][pivot];
      if (!factor) continue;
      for (let column = 0; column < size; column++)
        matrix[row][column] -= factor * matrix[pivot][column];
      for (let column = 0; column < rightHandSide[row].length; column++)
        rightHandSide[row][column] -= factor * rightHandSide[pivot][column];
    }
  }
  return true;
}
