// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Deterministic uniform generator for reproducible scenes. */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state + 0.5) / 4294967296;
  };
}

export type LocalGlmScene = {
  positions: Float32Array;
  predictors: Float32Array;
  response: Float32Array;
  predictorCount: number;
  rowCount: number;
};

/**
 * A Poisson scene on a `[0, extent]^2` square with standard-normal-ish predictors and slowly
 * varying coefficients. Positions are float32 values so the GPU and the reference see the same input.
 */
export function createPoissonScene(
  rowCount: number,
  predictorCount: number,
  seed: number,
  extent = 100,
  isGeographic = false
): LocalGlmScene {
  const random = createRandom(seed);
  const positions = new Float32Array(rowCount * 2);
  const predictors = new Float32Array(rowCount * predictorCount);
  const response = new Float32Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    const x = random() * extent;
    const y = random() * extent;
    // Geographic scenes put the square at longitude [10, 10 + extent / 10], latitude [40, ...].
    positions[2 * row] = isGeographic ? 10 + x / 10 : x;
    positions[2 * row + 1] = isGeographic ? 40 + y / 10 : y;
    let linear = 0.2 + 0.3 * (x / extent);
    for (let column = 0; column < predictorCount; column++) {
      const value = (random() + random() + random() - 1.5) * 0.8;
      predictors[row * predictorCount + column] = value;
      linear += (0.4 / (1 + column)) * (1 + (y / extent) * (column % 2 ? -1 : 1) * 0.3) * value;
    }
    const rate = Math.exp(linear);
    // Inverse-CDF Poisson draw.
    const uniform = random();
    let count = 0;
    let probability = Math.exp(-rate);
    let cumulative = probability;
    while (uniform > cumulative && count < 200) {
      count++;
      probability *= rate / count;
      cumulative += probability;
    }
    response[row] = count;
  }
  return {positions, predictors, response, predictorCount, rowCount};
}

const EARTH_RADIUS = 6371008.8;

export function getHaversineDistance(
  longitudeA: number,
  latitudeA: number,
  longitudeB: number,
  latitudeB: number,
  radius = EARTH_RADIUS
): number {
  const toRadians = Math.PI / 180;
  const deltaLatitude = (latitudeB - latitudeA) * toRadians;
  const deltaLongitude = (longitudeB - longitudeA) * toRadians;
  const a =
    Math.sin(deltaLatitude / 2) ** 2 +
    Math.cos(latitudeA * toRadians) *
      Math.cos(latitudeB * toRadians) *
      Math.sin(deltaLongitude / 2) ** 2;
  return 2 * radius * Math.asin(Math.min(1, Math.sqrt(a)));
}

/**
 * Float64 local Poisson (log link) maximum likelihood at one focal row with bisquare weights and a
 * fixed bandwidth, by Newton iterations on the raw design `(1, x)`. Returns the coefficients
 * (intercept first), or null when the local information matrix is singular.
 */
export function fitLocalPoissonOnCPU(
  scene: LocalGlmScene,
  focus: number,
  bandwidth: number,
  getDistance: (focus: number, row: number) => number
): number[] | null {
  const {predictors, response, predictorCount, rowCount} = scene;
  const columns = predictorCount + 1;
  const weights: number[] = [];
  const rows: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    const distance = getDistance(focus, row);
    if (distance < bandwidth) {
      const ratio = distance / bandwidth;
      const weight = (1 - ratio * ratio) ** 2;
      if (weight > 0) {
        rows.push(row);
        weights.push(weight);
      }
    }
  }
  const design = (row: number, column: number) =>
    column === 0 ? 1 : predictors[row * predictorCount + column - 1];
  const beta = new Array(columns).fill(0);
  for (let iteration = 0; iteration < 100; iteration++) {
    const information = Array.from({length: columns}, () => new Array(columns).fill(0));
    const gradient = new Array(columns).fill(0);
    rows.forEach((row, index) => {
      let linear = 0;
      for (let column = 0; column < columns; column++) linear += beta[column] * design(row, column);
      const mean = Math.exp(linear);
      for (let column = 0; column < columns; column++) {
        gradient[column] += weights[index] * design(row, column) * (response[row] - mean);
        for (let other = 0; other < columns; other++) {
          information[column][other] +=
            weights[index] * mean * design(row, column) * design(row, other);
        }
      }
    });
    const step = solveLinearSystem(information, gradient);
    if (!step) return null;
    let change = 0;
    for (let column = 0; column < columns; column++) {
      beta[column] += step[column];
      change = Math.max(change, Math.abs(step[column]));
    }
    if (change < 1e-10) return beta;
  }
  return beta;
}

/** Gaussian elimination with partial pivoting; null when singular. */
export function solveLinearSystem(matrix: number[][], vector: number[]): number[] | null {
  const n = vector.length;
  const a = matrix.map((row, index) => [...row, vector[index]]);
  for (let column = 0; column < n; column++) {
    let pivot = column;
    for (let row = column + 1; row < n; row++) {
      if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row;
    }
    if (Math.abs(a[pivot][column]) < 1e-12) return null;
    [a[column], a[pivot]] = [a[pivot], a[column]];
    for (let row = column + 1; row < n; row++) {
      const factor = a[row][column] / a[column][column];
      for (let k = column; k <= n; k++) a[row][k] -= factor * a[column][k];
    }
  }
  const result = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    let sum = a[row][n];
    for (let k = row + 1; k < n; k++) sum -= a[row][k] * result[k];
    result[row] = sum / a[row][row];
  }
  return result;
}

/** A Gaussian MGWR scene: response from slowly varying coefficients plus small noise. */
export function createGaussianScene(
  rowCount: number,
  predictorCount: number,
  seed: number,
  extent = 100,
  isGeographic = false
): LocalGlmScene {
  const scene = createPoissonScene(rowCount, predictorCount, seed, extent, isGeographic);
  const random = createRandom(seed + 1000);
  for (let row = 0; row < rowCount; row++) {
    const x = isGeographic ? (scene.positions[2 * row] - 10) * 10 : scene.positions[2 * row];
    const y = isGeographic
      ? (scene.positions[2 * row + 1] - 40) * 10
      : scene.positions[2 * row + 1];
    let value = 1 + x / extent;
    for (let column = 0; column < predictorCount; column++) {
      const slope = 0.5 / (1 + column) + (column % 2 ? y : x) / (extent * 4);
      value += slope * scene.predictors[row * predictorCount + column];
    }
    scene.response[row] = value + (random() - 0.5) * 0.1;
  }
  return scene;
}

/**
 * Float64 MGWR backfitting (bisquare, fixed per-column bandwidths): start from the global least
 * squares fit, then per sweep and per column regress the partial residual on that column with its
 * own kernel at every focus, replacing the column surface at once. Returns row-major
 * `rowCount * (predictorCount + 1)` coefficients (intercept first).
 */
export function fitMultiscaleOnCPU(
  scene: LocalGlmScene,
  bandwidths: readonly number[],
  getDistance: (focus: number, row: number) => number,
  tolerance: number,
  maximumIterations: number
): Float64Array | null {
  return (
    fitMultiscaleBackfitOnCPU(scene, bandwidths, getDistance, tolerance, maximumIterations)
      ?.coefficients ?? null
  );
}

/**
 * {@link fitMultiscaleOnCPU} that also reports the completed sweep count and whether the
 * convergence criterion (largest relative change of any coefficient contribution) was met.
 */
export function fitMultiscaleBackfitOnCPU(
  scene: LocalGlmScene,
  bandwidths: readonly number[],
  getDistance: (focus: number, row: number) => number,
  tolerance: number,
  maximumIterations: number
): {coefficients: Float64Array; iterationCount: number; converged: boolean} | null {
  const {predictors, response, predictorCount, rowCount} = scene;
  const columns = predictorCount + 1;
  const design = (row: number, column: number) =>
    column === 0 ? 1 : predictors[row * predictorCount + column - 1];
  const normal = Array.from({length: columns}, () => new Array(columns).fill(0));
  const cross = new Array(columns).fill(0);
  for (let row = 0; row < rowCount; row++) {
    for (let a = 0; a < columns; a++) {
      cross[a] += design(row, a) * response[row];
      for (let b = 0; b < columns; b++) normal[a][b] += design(row, a) * design(row, b);
    }
  }
  const global = solveLinearSystem(normal, cross);
  if (!global) return null;
  let current = new Float64Array(rowCount * columns);
  for (let row = 0; row < rowCount; row++) {
    for (let column = 0; column < columns; column++)
      current[row * columns + column] = global[column];
  }
  let iterationCount = 0;
  let converged = false;
  for (let iteration = 0; iteration < maximumIterations; iteration++) {
    iterationCount++;
    const previous = Float64Array.from(current);
    for (let column = 0; column < columns; column++) {
      const next = Float64Array.from(current);
      const partial = new Float64Array(rowCount);
      for (let row = 0; row < rowCount; row++) {
        let value = response[row];
        for (let other = 0; other < columns; other++) {
          if (other !== column) value -= design(row, other) * current[row * columns + other];
        }
        partial[row] = value;
      }
      const bandwidth = bandwidths[column];
      for (let focus = 0; focus < rowCount; focus++) {
        let numerator = 0;
        let denominator = 0;
        for (let row = 0; row < rowCount; row++) {
          const distance = getDistance(focus, row);
          if (distance >= bandwidth) continue;
          const ratio = distance / bandwidth;
          const weight = (1 - ratio * ratio) ** 2;
          const value = design(row, column);
          numerator += weight * value * partial[row];
          denominator += weight * value * value;
        }
        next[focus * columns + column] = numerator / denominator;
      }
      current = next;
    }
    let change = 0;
    for (let row = 0; row < rowCount; row++) {
      for (let column = 0; column < columns; column++) {
        const contribution = design(row, column) * current[row * columns + column];
        const before = design(row, column) * previous[row * columns + column];
        change = Math.max(change, Math.abs(contribution - before) / Math.max(1, Math.abs(before)));
      }
    }
    if (change <= tolerance) {
      converged = true;
      break;
    }
  }
  return {coefficients: current, iterationCount, converged};
}
