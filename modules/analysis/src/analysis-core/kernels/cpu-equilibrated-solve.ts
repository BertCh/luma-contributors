// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Relative elimination-pivot test used by the prediction kernel solver. */
const SYSTEM_PIVOT_TOLERANCE = 1e-6;

/**
 * Binary64 counterpart of the prediction kernels' equilibrated partial-pivot solve. Copies its
 * arguments. The four ordered Ruiz passes and relative rejection rules define the pivot diagnostic.
 * @internal
 */
export function solveCPUEquilibratedSystem(
  coefficients: Float64Array,
  values: Float64Array,
  minimumPivotRatio: number
): {solution?: Float64Array; pivotRatio: number} {
  const size = values.length;
  if (coefficients.length !== size * size) {
    throw new Error('coefficients length must equal values.length squared');
  }
  const matrix = coefficients.slice();
  const right = values.slice();
  const scales = new Float64Array(size).fill(1);
  const solution = new Float64Array(size);
  for (let pass = 0; pass < 4; pass++) {
    for (let row = 0; row < size; row++) {
      let maximum = 0;
      for (let column = 0; column < size; column++) {
        maximum = Math.max(
          maximum,
          Math.abs(matrix[row * size + column]) * scales[row] * scales[column]
        );
      }
      if (!(maximum > 0) || !Number.isFinite(maximum)) return {pivotRatio: 0};
      scales[row] /= Math.sqrt(maximum);
    }
  }
  let largestEntry = 0;
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      matrix[row * size + column] *= scales[row] * scales[column];
      largestEntry = Math.max(largestEntry, Math.abs(matrix[row * size + column]));
    }
    right[row] *= scales[row];
  }
  let smallestPivot = Infinity;
  let largestPivot = 0;
  for (let column = 0; column < size; column++) {
    let pivotRow = column;
    for (let row = column + 1; row < size; row++) {
      if (Math.abs(matrix[row * size + column]) > Math.abs(matrix[pivotRow * size + column])) {
        pivotRow = row;
      }
    }
    const magnitude = Math.abs(matrix[pivotRow * size + column]);
    if (!(magnitude > SYSTEM_PIVOT_TOLERANCE * largestEntry)) return {pivotRatio: 0};
    smallestPivot = Math.min(smallestPivot, magnitude);
    largestPivot = Math.max(largestPivot, magnitude);
    for (let entry = column; entry < size; entry++) {
      const held = matrix[column * size + entry];
      matrix[column * size + entry] = matrix[pivotRow * size + entry];
      matrix[pivotRow * size + entry] = held;
    }
    const held = right[column];
    right[column] = right[pivotRow];
    right[pivotRow] = held;
    for (let row = column + 1; row < size; row++) {
      const factor = matrix[row * size + column] / matrix[column * size + column];
      for (let entry = column + 1; entry < size; entry++) {
        matrix[row * size + entry] -= factor * matrix[column * size + entry];
      }
      right[row] -= factor * right[column];
    }
  }
  const pivotRatio = size === 0 ? 1 : smallestPivot / largestPivot;
  if (!(pivotRatio >= minimumPivotRatio)) return {pivotRatio};
  for (let row = size - 1; row >= 0; row--) {
    let value = right[row];
    for (let column = row + 1; column < size; column++) {
      value -= matrix[row * size + column] * solution[column];
    }
    solution[row] = value / matrix[row * size + row];
    if (!Number.isFinite(solution[row])) return {pivotRatio};
  }
  for (let row = 0; row < size; row++) solution[row] *= scales[row];
  return {solution, pivotRatio};
}
