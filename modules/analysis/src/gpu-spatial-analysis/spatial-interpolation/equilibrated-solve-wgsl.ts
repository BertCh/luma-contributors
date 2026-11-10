// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {KRIGING_PIVOT_RATIO_TOLERANCE} from './kriging-parameters';

/**
 * Shared bounded dense solver of the local prediction kernels (kriging and RBF).
 *
 * The caller fills `matrix` (row stride `SYSTEM_CAPACITY`) and `right` for a system of `size`
 * rows. Kriging and RBF systems mix covariance or radial entries of arbitrary magnitude (a sill
 * of 1e-4 or 1e6, cubed projected distances) with unit-scale drift or polynomial rows. The solver
 * therefore symmetrically equilibrates the system with a few Ruiz passes, applies the pivot tests
 * to the scaled system (so they are relative and scale invariant), eliminates with partial
 * pivoting and unscales the solution into `solution`. `pivotRatio` is the smallest over the
 * largest pivot of the equilibrated system.
 */
export function getEquilibratedSolveWGSL(capacity: number): string {
  return /* wgsl */ `
const SYSTEM_CAPACITY: u32 = ${capacity}u;
const SYSTEM_PIVOT_TOLERANCE: f32 = 1e-6;
const SYSTEM_PIVOT_RATIO_TOLERANCE: f32 = ${getWGSLFloatLiteral(KRIGING_PIVOT_RATIO_TOLERANCE)};
var<private> matrix: array<f32, ${capacity * capacity}>;
var<private> right: array<f32, ${capacity}>;
var<private> solution: array<f32, ${capacity}>;
var<private> systemScale: array<f32, ${capacity}>;
var<private> pivotRatio: f32;
fn solveSystem(size: u32) -> bool {
  pivotRatio = 0.0;
  for (var row = 0u; row < size; row++) { systemScale[row] = 1.0; }
  for (var equilibrationPass = 0u; equilibrationPass < 4u; equilibrationPass++) {
    for (var row = 0u; row < size; row++) {
      var rowMaximum = 0.0;
      for (var column = 0u; column < size; column++) {
        rowMaximum = max(rowMaximum, abs(matrix[row * SYSTEM_CAPACITY + column]) * systemScale[row] * systemScale[column]);
      }
      if (!(rowMaximum > 0.0) || !isFiniteValue(rowMaximum)) { return false; }
      systemScale[row] *= inverseSqrt(rowMaximum);
    }
  }
  var largestEntry = 0.0;
  for (var row = 0u; row < size; row++) {
    for (var column = 0u; column < size; column++) {
      matrix[row * SYSTEM_CAPACITY + column] *= systemScale[row] * systemScale[column];
      largestEntry = max(largestEntry, abs(matrix[row * SYSTEM_CAPACITY + column]));
    }
    right[row] *= systemScale[row];
  }
  var smallestPivot = 3.402823e38;
  var largestPivot = 0.0;
  for (var column = 0u; column < size; column++) {
    var pivotRow = column;
    for (var row = column + 1u; row < size; row++) {
      if (abs(matrix[row * SYSTEM_CAPACITY + column]) > abs(matrix[pivotRow * SYSTEM_CAPACITY + column])) { pivotRow = row; }
    }
    let pivotMagnitude = abs(matrix[pivotRow * SYSTEM_CAPACITY + column]);
    if (!(pivotMagnitude > SYSTEM_PIVOT_TOLERANCE * largestEntry)) { return false; }
    smallestPivot = min(smallestPivot, pivotMagnitude);
    largestPivot = max(largestPivot, pivotMagnitude);
    for (var entry = column; entry < size; entry++) {
      let held = matrix[column * SYSTEM_CAPACITY + entry];
      matrix[column * SYSTEM_CAPACITY + entry] = matrix[pivotRow * SYSTEM_CAPACITY + entry];
      matrix[pivotRow * SYSTEM_CAPACITY + entry] = held;
    }
    let heldRight = right[column];
    right[column] = right[pivotRow];
    right[pivotRow] = heldRight;
    for (var row = column + 1u; row < size; row++) {
      let factor = matrix[row * SYSTEM_CAPACITY + column] / matrix[column * SYSTEM_CAPACITY + column];
      for (var entry = column + 1u; entry < size; entry++) {
        matrix[row * SYSTEM_CAPACITY + entry] -= factor * matrix[column * SYSTEM_CAPACITY + entry];
      }
      right[row] -= factor * right[column];
    }
  }
  pivotRatio = smallestPivot / largestPivot;
  if (!(pivotRatio >= SYSTEM_PIVOT_RATIO_TOLERANCE)) { return false; }
  for (var reverse = 0u; reverse < size; reverse++) {
    let row = size - reverse - 1u;
    var value = right[row];
    for (var column = row + 1u; column < size; column++) {
      value -= matrix[row * SYSTEM_CAPACITY + column] * solution[column];
    }
    solution[row] = value / matrix[row * SYSTEM_CAPACITY + row];
    if (!isFiniteValue(solution[row])) { return false; }
  }
  // Undo the equilibration: x = S y.
  for (var row = 0u; row < size; row++) { solution[row] *= systemScale[row]; }
  return true;
}
`;
}
