// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getDenseLUWGSL} from '../../analysis-core/kernels/dense-solve-wgsl';
import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {KRIGING_PIVOT_RATIO_TOLERANCE} from './kriging-parameters';

/**
 * Prediction adapter for the analysis-core bounded dense LU solver.
 *
 * The caller fills `matrix` (row stride `SYSTEM_CAPACITY`) and `right` for a system of `size`
 * rows. Kriging and RBF systems mix covariance or radial entries of arbitrary magnitude (a sill
 * of 1e-4 or 1e6, cubed projected distances) with unit-scale drift or polynomial rows. The solver
 * therefore symmetrically equilibrates the system with a few Ruiz passes, applies the pivot tests
 * to the scaled system (so they are relative and scale invariant), eliminates with partial
 * pivoting and unscales the solution into `solution`. Sequential symmetric Ruiz scaling stays
 * here to preserve the conditioning decisions of existing prediction kernels; factorization and
 * substitution come from analysis-core. `pivotRatio` is the smallest over the
 * largest pivot of the equilibrated system.
 */
export function getEquilibratedSolveWGSL(capacity: number): string {
  return /* wgsl */ `
${getDenseLUWGSL(capacity)}
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
  // The core API takes function-local arrays; prediction kernels fill private arrays.
  var factorMatrix: array<f32, ${capacity * capacity}>;
  var factorRight: array<f32, ${capacity}>;
  for (var row = 0u; row < size; row++) {
    for (var column = 0u; column < size; column++) {
      factorMatrix[row * SYSTEM_CAPACITY + column] = matrix[row * SYSTEM_CAPACITY + column] * (systemScale[row] * systemScale[column]);
    }
    factorRight[row] = right[row] * systemScale[row];
  }
  // Scaling already happened above: disable core equilibration to retain the prediction policy.
  var factorization = factorDenseLU_${capacity}(&factorMatrix, size, 0u, SYSTEM_PIVOT_TOLERANCE, SYSTEM_PIVOT_RATIO_TOLERANCE);
  pivotRatio = factorization.pivotRatio;
  if (factorization.status != 0u) { return false; }
  if (!solveDenseLU_${capacity}(&factorMatrix, &factorization, &factorRight)) { return false; }
  for (var row = 0u; row < size; row++) { solution[row] = factorRight[row]; }
  // Undo the equilibration: x = S y.
  for (var row = 0u; row < size; row++) {
    solution[row] *= systemScale[row];
    if (!isFiniteValue(solution[row])) { return false; }
  }
  return true;
}
`;
}
