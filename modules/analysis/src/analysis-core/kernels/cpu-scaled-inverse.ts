// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Inverts a square row-major matrix in binary64 using global scaling and partial pivoting.
 * Copies the input. Returns undefined for a zero/nonfinite scale, an exactly zero/nonfinite
 * elimination pivot, or a nonfinite inverse. No relative pivot cutoff is imposed: callers that
 * need the prediction kernels' rejection thresholds use solveCPUEquilibratedSystem instead.
 *
 * @throws Error naming matrix when its rows do not form a nonempty square matrix.
 * @internal
 */
export function invertCPUScaledMatrix(
  matrix: readonly (readonly number[])[]
): number[][] | undefined {
  const size = matrix.length;
  if (size === 0 || matrix.some(row => row.length !== size)) {
    throw new Error('matrix must be a nonempty square row-major matrix');
  }
  let scale = 0;
  for (const row of matrix) for (const value of row) scale = Math.max(scale, Math.abs(value));
  if (!(scale > 0) || !Number.isFinite(scale)) return undefined;
  const left = matrix.map(row => row.map(value => value / scale));
  const right = Array.from({length: size}, (_, row) =>
    Array.from({length: size}, (_, column) => Number(row === column))
  );
  for (let column = 0; column < size; column++) {
    let pivot = column;
    for (let row = column + 1; row < size; row++)
      if (Math.abs(left[row][column]) > Math.abs(left[pivot][column])) pivot = row;
    const divisor = left[pivot][column];
    if (divisor === 0 || !Number.isFinite(divisor)) return undefined;
    [left[column], left[pivot]] = [left[pivot], left[column]];
    [right[column], right[pivot]] = [right[pivot], right[column]];
    for (let component = 0; component < size; component++) {
      left[column][component] /= divisor;
      right[column][component] /= divisor;
    }
    for (let row = 0; row < size; row++) {
      if (row === column) continue;
      const factor = left[row][column];
      for (let component = 0; component < size; component++) {
        left[row][component] -= factor * left[column][component];
        right[row][component] -= factor * right[column][component];
      }
    }
  }
  const inverse = right.map(row => row.map(value => value / scale));
  return inverse.every(row => row.every(Number.isFinite)) ? inverse : undefined;
}
