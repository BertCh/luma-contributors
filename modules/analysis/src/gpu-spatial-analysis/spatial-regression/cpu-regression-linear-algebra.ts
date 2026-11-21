// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Factors a symmetric positive-definite correlation matrix in float64. Rejects pivots at the
 * float64 rounding floor, rather than the shader's float32 conditioning threshold.
 * @internal
 */
export function factorCPURegressionCorrelation(
  matrix: readonly number[][]
): number[][] | undefined {
  const size = matrix.length;
  const factor = Array.from({length: size}, () => new Array<number>(size).fill(0));
  for (let column = 0; column < size; column++) {
    let diagonal = matrix[column][column];
    for (let previous = 0; previous < column; previous++) {
      diagonal -= factor[column][previous] ** 2;
    }
    if (!(diagonal > 64 * Number.EPSILON * matrix[column][column]) || !Number.isFinite(diagonal)) {
      return undefined;
    }
    factor[column][column] = Math.sqrt(diagonal);
    for (let row = column + 1; row < size; row++) {
      let entry = matrix[row][column];
      for (let previous = 0; previous < column; previous++) {
        entry -= factor[row][previous] * factor[column][previous];
      }
      factor[row][column] = entry / factor[column][column];
    }
  }
  return factor;
}

/** Solves from a lower Cholesky factor without mutating either input. @internal */
export function solveCPURegressionFactor(
  factor: readonly number[][],
  values: readonly number[]
): number[] {
  const solution = Array.from(values);
  for (let row = 0; row < solution.length; row++) {
    for (let column = 0; column < row; column++) {
      solution[row] -= factor[row][column] * solution[column];
    }
    solution[row] /= factor[row][row];
  }
  for (let row = solution.length - 1; row >= 0; row--) {
    for (let column = row + 1; column < solution.length; column++) {
      solution[row] -= factor[column][row] * solution[column];
    }
    solution[row] /= factor[row][row];
  }
  return solution;
}

/**
 * One-sided Jacobi singular values of a small rectangular matrix. Scales before taking products
 * and uses hypot norms with products of unit columns so small squared norms cannot underflow,
 * and orthogonalizes the shorter dimension (transposing wide inputs). Stops at float64
 * orthogonality, numerical stagnation, or max(32, 4 * columnCount) sweeps.
 * @internal
 */
export function getCPURegressionSingularValueRange(matrix: readonly number[][]): [number, number] {
  const workingRows =
    matrix[0].length > matrix.length
      ? matrix[0].map((_, column) => matrix.map(row => row[column]))
      : matrix;
  const columnCount = workingRows[0].length;
  let maximumEntry = 0;
  for (const row of matrix) {
    for (const entry of row) maximumEntry = Math.max(maximumEntry, Math.abs(entry));
  }
  if (maximumEntry === 0) return [0, 0];
  const working = workingRows.map(row => row.map(entry => entry / maximumEntry));
  let changed = true;
  const maximumSweeps = Math.max(32, 4 * columnCount);
  for (let sweep = 0; changed && sweep < maximumSweeps; sweep++) {
    changed = false;
    for (let first = 0; first < columnCount; first++) {
      for (let second = first + 1; second < columnCount; second++) {
        let firstNorm = 0;
        let secondNorm = 0;
        let product = 0;
        for (const row of working) {
          firstNorm = Math.hypot(firstNorm, row[first]);
          secondNorm = Math.hypot(secondNorm, row[second]);
        }
        if (!(firstNorm > 0) || !(secondNorm > 0)) continue;
        // Unit-column products retain a small column even when its squared norm underflows.
        for (const row of working) product += (row[first] / firstNorm) * (row[second] / secondNorm);
        if (Math.abs(product) <= 16 * Number.EPSILON) continue;
        const largest = Math.max(firstNorm, secondNorm);
        const firstRelative = firstNorm / largest;
        const secondRelative = secondNorm / largest;
        const difference = (secondRelative - firstRelative) * (secondRelative + firstRelative);
        const twiceProduct = 2 * firstRelative * secondRelative * product;
        const tangent =
          twiceProduct /
          (difference + (difference >= 0 ? 1 : -1) * Math.hypot(difference, twiceProduct));
        const cosine = 1 / Math.hypot(1, tangent);
        const sine = tangent * cosine;
        if (sine === 0 || !Number.isFinite(sine)) continue;
        let rotated = false;
        for (const row of working) {
          const firstEntry = row[first];
          const secondEntry = row[second];
          row[first] = cosine * firstEntry - sine * secondEntry;
          row[second] = sine * firstEntry + cosine * secondEntry;
          rotated ||= row[first] !== firstEntry || row[second] !== secondEntry;
        }
        changed ||= rotated;
      }
    }
  }
  const norms = Array.from({length: columnCount}, (_, column) => {
    let norm = 0;
    for (const row of working) norm = Math.hypot(norm, row[column]);
    return norm * maximumEntry;
  });
  return [Math.min(...norms), Math.max(...norms)];
}

/**
 * Norm-scaled Householder least squares, including a square root of the inverse normal matrix.
 * It never forms normal equations. A root row dot product gives the covariance entry.
 * @internal
 */
export function fitCPURegressionWithQR(
  design: readonly number[][],
  response: readonly number[]
):
  | {
      estimate: number[];
      inverseRoot: number[][];
    }
  | undefined {
  const rowCount = design.length;
  const columnCount = design[0].length;
  if (rowCount < columnCount) return undefined;
  const scales = new Array<number>(columnCount).fill(0);
  for (const row of design) {
    for (let column = 0; column < columnCount; column++) {
      scales[column] = Math.hypot(scales[column], row[column]);
    }
  }
  if (scales.some(scale => !(scale > 0) || !Number.isFinite(scale))) return undefined;
  const working = design.map(row => row.map((value, column) => value / scales[column]));
  const transformed = Array.from(response);
  for (let column = 0; column < columnCount; column++) {
    const reflector = working.slice(column).map(row => row[column]);
    let norm = 0;
    for (const value of reflector) norm = Math.hypot(norm, value);
    if (!(norm > 64 * Number.EPSILON)) return undefined;
    reflector[0] += reflector[0] >= 0 ? norm : -norm;
    let reflectorNorm = 0;
    for (const value of reflector) reflectorNorm = Math.hypot(reflectorNorm, value);
    for (let row = 0; row < reflector.length; row++) reflector[row] /= reflectorNorm;
    for (let target = column; target < columnCount; target++) {
      let projection = 0;
      for (let row = column; row < rowCount; row++)
        projection += reflector[row - column] * working[row][target];
      for (let row = column; row < rowCount; row++)
        working[row][target] -= 2 * reflector[row - column] * projection;
    }
    let projection = 0;
    for (let row = column; row < rowCount; row++)
      projection += reflector[row - column] * transformed[row];
    for (let row = column; row < rowCount; row++)
      transformed[row] -= 2 * reflector[row - column] * projection;
  }
  const solveUpper = (right: readonly number[]) => {
    const solution = Array.from(right);
    for (let row = columnCount - 1; row >= 0; row--) {
      for (let column = row + 1; column < columnCount; column++)
        solution[row] -= working[row][column] * solution[column];
      solution[row] /= working[row][row];
    }
    return solution.map((value, row) => value / scales[row]);
  };
  const estimate = solveUpper(transformed.slice(0, columnCount));
  const inverseRoot = Array.from({length: columnCount}, () =>
    new Array<number>(columnCount).fill(0)
  );
  for (let column = 0; column < columnCount; column++) {
    const solution = solveUpper(
      Array.from({length: columnCount}, (_, row) => (row === column ? 1 : 0))
    );
    for (let row = 0; row < columnCount; row++) inverseRoot[row][column] = solution[row];
  }
  // Refine against the original rows. This recovers exactly representable fits that the
  // Householder transformations alone can leave a few ulps away from zero residual.
  let bestSquares = Infinity;
  let bestEstimate = estimate.slice();
  while (true) {
    const residuals = design.map(
      (row, position) =>
        response[position] - row.reduce((sum, value, column) => sum + value * estimate[column], 0)
    );
    const squares = residuals.reduce((sum, value) => sum + value * value, 0);
    if (!(squares < bestSquares)) {
      estimate.splice(0, columnCount, ...bestEstimate);
      break;
    }
    bestSquares = squares;
    bestEstimate = estimate.slice();
    if (squares === 0) break;
    const projected = new Array<number>(columnCount).fill(0);
    for (let row = 0; row < rowCount; row++) {
      for (let column = 0; column < columnCount; column++) {
        let basis = 0;
        for (let inner = 0; inner < columnCount; inner++)
          basis += design[row][inner] * inverseRoot[inner][column];
        projected[column] += basis * residuals[row];
      }
    }
    for (let row = 0; row < columnCount; row++) {
      estimate[row] += inverseRoot[row].reduce(
        (sum, value, column) => sum + value * projected[column],
        0
      );
    }
  }
  return {estimate, inverseRoot};
}
