// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Two residual-correction solves reduce the forward error of the float32 Cholesky solve. */
export const SUR_GLS_REFINEMENT_COUNT = 2;

/** Requires the public fp64arithmetic shader module, isFiniteBits, and choleskySolve_size. */
export function getSURGLSRefinementWGSL(
  size: number,
  refinementCount = SUR_GLS_REFINEMENT_COUNT
): string {
  return /* wgsl */ `
fn refineSURSolution_${size}(
  original: ptr<function, array<f32, ${size * size}>>,
  right: ptr<function, array<f32, ${size}>>,
  factor: ptr<function, array<f32, ${size * size}>>,
  scales: ptr<function, array<f32, ${size}>>,
  solution: ptr<function, array<f32, ${size}>>
) {
  for (var refinement = 0u; refinement < ${refinementCount}u; refinement++) {
    var correction: array<f32, ${size}>;
    for (var row = 0u; row < ${size}u; row++) {
      var remainder = vec2f((*right)[row], 0.0);
      for (var column = 0u; column < ${size}u; column++) {
        remainder = sub_fp64(remainder, mul_fp64(vec2f((*original)[row * ${size}u + column], 0.0), vec2f((*solution)[column], 0.0)));
      }
      correction[row] = (remainder.x + remainder.y) * (*scales)[row];
      if (!isFiniteBits(correction[row])) { return; }
    }
    choleskySolve_${size}(factor, &correction);
    for (var column = 0u; column < ${size}u; column++) {
      let corrected = sum_fp64(vec2f((*solution)[column], 0.0), mul_fp64(vec2f(correction[column], 0.0), vec2f((*scales)[column], 0.0)));
      correction[column] = corrected.x + corrected.y;
      if (!isFiniteBits(correction[column])) { return; }
    }
    for (var column = 0u; column < ${size}u; column++) { (*solution)[column] = correction[column]; }
  }
}
`;
}
