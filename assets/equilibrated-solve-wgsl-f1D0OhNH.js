import{n as e}from"./dense-solve-wgsl-0xzjbd8V.js";import{c as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as n}from"./kriging-parameters-DGogQ0zZ.js";function r(r){return`
${e(r)}
const SYSTEM_CAPACITY: u32 = ${r}u;
const SYSTEM_PIVOT_TOLERANCE: f32 = 1e-6;
const SYSTEM_PIVOT_RATIO_TOLERANCE: f32 = ${t(n)};
var<private> matrix: array<f32, ${r*r}>;
var<private> right: array<f32, ${r}>;
var<private> solution: array<f32, ${r}>;
var<private> systemScale: array<f32, ${r}>;
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
  var factorMatrix: array<f32, ${r*r}>;
  var factorRight: array<f32, ${r}>;
  for (var row = 0u; row < size; row++) {
    for (var column = 0u; column < size; column++) {
      factorMatrix[row * SYSTEM_CAPACITY + column] = matrix[row * SYSTEM_CAPACITY + column] * (systemScale[row] * systemScale[column]);
    }
    factorRight[row] = right[row] * systemScale[row];
  }
  // Scaling already happened above: disable core equilibration to retain the prediction policy.
  var factorization = factorDenseLU_${r}(&factorMatrix, size, 0u, SYSTEM_PIVOT_TOLERANCE, SYSTEM_PIVOT_RATIO_TOLERANCE);
  pivotRatio = factorization.pivotRatio;
  if (factorization.status != 0u) { return false; }
  if (!solveDenseLU_${r}(&factorMatrix, &factorization, &factorRight)) { return false; }
  for (var row = 0u; row < size; row++) { solution[row] = factorRight[row]; }
  // Undo the equilibration: x = S y.
  for (var row = 0u; row < size; row++) {
    solution[row] *= systemScale[row];
    if (!isFiniteValue(solution[row])) { return false; }
  }
  return true;
}
`}export{r as t};