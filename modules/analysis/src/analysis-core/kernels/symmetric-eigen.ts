// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Float64 cyclic Jacobi decomposition of a small symmetric row-major matrix. The returned
 * eigenvectors are columns, with unsorted eigenvalues. Input is not modified. Scale first to
 * keep rotations finite. Shared CPU companion of symmetric-eigen-wgsl.ts.
 * @internal
 */
export function getSymmetricEigenOnCPU(
  input: Float64Array,
  size: number
): {
  values: Float64Array;
  vectors: Float64Array;
} {
  const matrix = input.slice();
  const vectors = new Float64Array(size * size);
  let scale = 0;
  for (const value of matrix) scale = Math.max(scale, Math.abs(value));
  for (let row = 0; row < size; row++) vectors[row * size + row] = 1;
  if (scale > 0) {
    for (let index = 0; index < matrix.length; index++) matrix[index] /= scale;
  }
  for (let sweep = 0; sweep < 64; sweep++) {
    let largestCoupling = 0;
    for (let lower = 0; lower < size; lower++) {
      for (let upper = lower + 1; upper < size; upper++) {
        const coupling = matrix[lower * size + upper];
        largestCoupling = Math.max(largestCoupling, Math.abs(coupling));
        if (Math.abs(coupling) <= Number.EPSILON) continue;
        const difference =
          (matrix[upper * size + upper] - matrix[lower * size + lower]) / (2 * coupling);
        const tangent =
          (difference >= 0 ? 1 : -1) / (Math.abs(difference) + Math.hypot(1, difference));
        const cosine = 1 / Math.hypot(1, tangent);
        const sine = tangent * cosine;
        for (let row = 0; row < size; row++) {
          const first = matrix[row * size + lower];
          const second = matrix[row * size + upper];
          matrix[row * size + lower] = cosine * first - sine * second;
          matrix[row * size + upper] = sine * first + cosine * second;
          const firstVector = vectors[row * size + lower];
          const secondVector = vectors[row * size + upper];
          vectors[row * size + lower] = cosine * firstVector - sine * secondVector;
          vectors[row * size + upper] = sine * firstVector + cosine * secondVector;
        }
        for (let column = 0; column < size; column++) {
          const first = matrix[lower * size + column];
          const second = matrix[upper * size + column];
          matrix[lower * size + column] = cosine * first - sine * second;
          matrix[upper * size + column] = sine * first + cosine * second;
        }
        matrix[lower * size + upper] = 0;
        matrix[upper * size + lower] = 0;
      }
    }
    if (largestCoupling <= Number.EPSILON) break;
  }
  return {
    values: Float64Array.from({length: size}, (_, index) => matrix[index * size + index] * scale),
    vectors
  };
}
