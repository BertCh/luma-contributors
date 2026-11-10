// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Binary64 CPU reference of the Vecchia (nearest-neighbour Gaussian process) log-likelihood,
// written from the formulas of Datta et al. 2016 and Guinness 2018, independently of the WGSL under
// test. Same arithmetic as dev-docs/research/spatiotemporal-review-2026-11-09/experiments/vecchia-float32.mjs:
// a Cholesky of the (k+1) by (k+1) covariance with the site last. GpGp is R-only and stays pending.

export type ReferenceVecchiaKernel = 'exponential' | 'matern-3/2' | 'matern-5/2' | 'gaussian';

export type ReferenceVecchiaCovariance = {
  model: ReferenceVecchiaKernel;
  /** Effective range of `GPUPredictionCovariance`. */
  range: number;
  sill: number;
  nugget?: number;
};

/** Argument of `(1 + a) exp(-a) = exp(-3)` (and the 5/2 analogue), solved here by bisection. */
function solveEffectiveRangeArgument(correlation: (argument: number) => number): number {
  let low = 0;
  let high = 100;
  for (let step = 0; step < 200; step++) {
    const middle = (low + high) / 2;
    if (correlation(middle) > Math.exp(-3)) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return (low + high) / 2;
}

export const MATERN_3_2_ARGUMENT = solveEffectiveRangeArgument(a => (1 + a) * Math.exp(-a));
export const MATERN_5_2_ARGUMENT = solveEffectiveRangeArgument(
  a => (1 + a + (a * a) / 3) * Math.exp(-a)
);

/** Correlation of the effective-range parameterization at `distance / range = scaled`. */
export function getReferenceCorrelation(model: ReferenceVecchiaKernel, scaled: number): number {
  switch (model) {
    case 'exponential':
      return Math.exp(-3 * scaled);
    case 'matern-3/2': {
      const a = MATERN_3_2_ARGUMENT * scaled;
      return (1 + a) * Math.exp(-a);
    }
    case 'matern-5/2': {
      const a = MATERN_5_2_ARGUMENT * scaled;
      return (1 + a + (a * a) / 3) * Math.exp(-a);
    }
    case 'gaussian':
      return Math.exp(-3 * scaled * scaled);
    default:
      throw new Error('Unknown reference kernel');
  }
}

/**
 * Effective range that gives the experiment's correlation at `range` (its own parameterization):
 * `exp(-h / rho)`, `Matern(sqrt(3) h / rho)`, `exp(-(h / rho)^2)`.
 */
export function getEffectiveRangeForExperiment(
  model: ReferenceVecchiaKernel,
  experimentRange: number
): number {
  switch (model) {
    case 'exponential':
      return 3 * experimentRange;
    case 'matern-3/2':
      return (MATERN_3_2_ARGUMENT * experimentRange) / Math.sqrt(3);
    case 'matern-5/2':
      return (MATERN_5_2_ARGUMENT * experimentRange) / Math.sqrt(5);
    case 'gaussian':
      return Math.sqrt(3) * experimentRange;
    default:
      throw new Error('Unknown reference kernel');
  }
}

export type ReferenceVecchiaResult = {
  logLikelihood: number;
  failedSiteCount: number;
  /** Conditional variance per site, NaN where the factorization failed. */
  conditionalVariances: Float64Array;
  /** Conditional mean per site. */
  conditionalMeans: Float64Array;
};

/** Sets up `C` of the site-last system; returns the conditioning-set size. */
function assembleSystem(
  matrix: Float64Array,
  capacity: number,
  positions: ArrayLike<number>,
  offsets: ArrayLike<number>,
  neighbors: ArrayLike<number>,
  site: number,
  covariance: ReferenceVecchiaCovariance
): number {
  const begin = offsets[site];
  const count = offsets[site + 1] - begin;
  const size = count + 1;
  const rowOf = (slot: number): number => (slot < count ? neighbors[begin + slot] : site);
  const {model, range, sill, nugget = 0} = covariance;
  for (let row = 0; row < size; row++) {
    for (let column = 0; column <= row; column++) {
      const first = rowOf(row);
      const second = rowOf(column);
      const offsetX = positions[first * 2] - positions[second * 2];
      const offsetY = positions[first * 2 + 1] - positions[second * 2 + 1];
      const separation = Math.sqrt(offsetX * offsetX + offsetY * offsetY);
      matrix[row * capacity + column] =
        sill * getReferenceCorrelation(model, separation / range) +
        (first === second || (offsetX === 0 && offsetY === 0) ? nugget : 0);
    }
  }
  return count;
}

/** Binary64 Vecchia log-likelihood over given conditioning sets. */
export function evaluateReferenceVecchiaLogLikelihood(
  values: ArrayLike<number>,
  positions: ArrayLike<number>,
  offsets: ArrayLike<number>,
  neighbors: ArrayLike<number>,
  covariance: ReferenceVecchiaCovariance
): ReferenceVecchiaResult {
  const rowCount = offsets.length - 1;
  let maximumCount = 0;
  for (let site = 0; site < rowCount; site++) {
    maximumCount = Math.max(maximumCount, offsets[site + 1] - offsets[site]);
  }
  const capacity = maximumCount + 1;
  const matrix = new Float64Array(capacity * capacity);
  const forward = new Float64Array(capacity);
  const conditionalVariances = new Float64Array(rowCount).fill(Number.NaN);
  const conditionalMeans = new Float64Array(rowCount).fill(Number.NaN);
  let logLikelihood = 0;
  let failedSiteCount = 0;
  for (let site = 0; site < rowCount; site++) {
    const count = assembleSystem(matrix, capacity, positions, offsets, neighbors, site, covariance);
    const size = count + 1;
    let factored = true;
    for (let column = 0; column < size && factored; column++) {
      let diagonal = matrix[column * capacity + column];
      for (let index = 0; index < column; index++) {
        diagonal -= matrix[column * capacity + index] ** 2;
      }
      if (!(diagonal > 0)) {
        factored = false;
        break;
      }
      const pivot = Math.sqrt(diagonal);
      matrix[column * capacity + column] = pivot;
      for (let row = column + 1; row < size; row++) {
        let value = matrix[row * capacity + column];
        for (let index = 0; index < column; index++) {
          value -= matrix[row * capacity + index] * matrix[column * capacity + index];
        }
        matrix[row * capacity + column] = value / pivot;
      }
    }
    if (!factored) {
      failedSiteCount++;
      continue;
    }
    // Forward solve with the conditioning values, then the site last; only the last entry is needed.
    let mean = 0;
    for (let row = 0; row < count; row++) {
      let value = values[neighbors[offsets[site] + row]];
      for (let index = 0; index < row; index++) {
        value -= matrix[row * capacity + index] * forward[index];
      }
      forward[row] = value / matrix[row * capacity + row];
      mean += matrix[count * capacity + row] * forward[row];
    }
    const conditionalDeviation = matrix[count * capacity + count];
    const standardized = (values[site] - mean) / conditionalDeviation;
    logLikelihood +=
      -0.5 *
      (Math.log(2 * Math.PI) + 2 * Math.log(conditionalDeviation) + standardized * standardized);
    conditionalVariances[site] = conditionalDeviation * conditionalDeviation;
    conditionalMeans[site] = mean;
  }
  return {
    logLikelihood: failedSiteCount > 0 ? Number.NaN : logLikelihood,
    failedSiteCount,
    conditionalVariances,
    conditionalMeans
  };
}

/** Sequential binary64 simulation from the Vecchia model, by site in `rowCount` row order. */
export function simulateReferenceVecchiaField(
  gaussian: () => number,
  positions: ArrayLike<number>,
  offsets: ArrayLike<number>,
  neighbors: ArrayLike<number>,
  covariance: ReferenceVecchiaCovariance
): Float64Array {
  const rowCount = offsets.length - 1;
  let maximumCount = 0;
  for (let site = 0; site < rowCount; site++) {
    maximumCount = Math.max(maximumCount, offsets[site + 1] - offsets[site]);
  }
  const capacity = maximumCount + 1;
  const matrix = new Float64Array(capacity * capacity);
  const forward = new Float64Array(capacity);
  const values = new Float64Array(rowCount);
  for (let site = 0; site < rowCount; site++) {
    const count = assembleSystem(matrix, capacity, positions, offsets, neighbors, site, covariance);
    const size = count + 1;
    for (let column = 0; column < size; column++) {
      let diagonal = matrix[column * capacity + column];
      for (let index = 0; index < column; index++) {
        diagonal -= matrix[column * capacity + index] ** 2;
      }
      const pivot = Math.sqrt(Math.max(diagonal, 1e-300));
      matrix[column * capacity + column] = pivot;
      for (let row = column + 1; row < size; row++) {
        let value = matrix[row * capacity + column];
        for (let index = 0; index < column; index++) {
          value -= matrix[row * capacity + index] * matrix[column * capacity + index];
        }
        matrix[row * capacity + column] = value / pivot;
      }
    }
    let mean = 0;
    for (let row = 0; row < count; row++) {
      let value = values[neighbors[offsets[site] + row]];
      for (let index = 0; index < row; index++) {
        value -= matrix[row * capacity + index] * forward[index];
      }
      forward[row] = value / matrix[row * capacity + row];
      mean += matrix[count * capacity + row] * forward[row];
    }
    values[site] = mean + matrix[count * capacity + count] * gaussian();
  }
  return values;
}

/** `mulberry32` of the float32 experiment. */
export function createExperimentRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = state;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal draw consuming two uniforms, the experiment's method. */
export function createExperimentGaussian(random: () => number): () => number {
  return () => {
    const u = Math.max(random(), 1e-300);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
  };
}

export type ReferenceVecchiaFisher = {
  logLikelihood: number;
  /** Score in `(log sill, log effective range, log nugget)`. */
  score: number[];
  /** Upper triangle `I_00, I_01, I_02, I_11, I_12, I_22`. */
  information: number[];
};

/** Derivative of `correlation(h / range)` with respect to `log range`, for the fitted kernels. */
export function getReferenceCorrelationRangeDerivative(
  model: ReferenceVecchiaKernel,
  scaled: number
): number {
  if (model === 'exponential') {
    const a = 3 * scaled;
    return a * Math.exp(-a);
  }
  if (model === 'matern-3/2') {
    const a = MATERN_3_2_ARGUMENT * scaled;
    return a * a * Math.exp(-a);
  }
  throw new Error('No range derivative for this reference kernel');
}

/** Binary64 Vecchia score and expected information (Guinness 2019), one independent Cholesky per site. */
export function evaluateReferenceVecchiaFisher(
  values: ArrayLike<number>,
  positions: ArrayLike<number>,
  offsets: ArrayLike<number>,
  neighbors: ArrayLike<number>,
  covariance: ReferenceVecchiaCovariance
): ReferenceVecchiaFisher {
  const rowCount = offsets.length - 1;
  const {model, range, sill, nugget = 0} = covariance;
  const score = [0, 0, 0];
  const information = [0, 0, 0, 0, 0, 0];
  let logLikelihood = 0;
  const parameterDerivative = (parameter: number, first: number, second: number): number => {
    const offsetX = positions[first * 2] - positions[second * 2];
    const offsetY = positions[first * 2 + 1] - positions[second * 2 + 1];
    const separation = Math.sqrt(offsetX * offsetX + offsetY * offsetY);
    if (parameter === 0) {
      return sill * getReferenceCorrelation(model, separation / range);
    }
    if (parameter === 1) {
      return sill * getReferenceCorrelationRangeDerivative(model, separation / range);
    }
    return first === second || (offsetX === 0 && offsetY === 0) ? nugget : 0;
  };
  const covarianceAt = (first: number, second: number): number => {
    const offsetX = positions[first * 2] - positions[second * 2];
    const offsetY = positions[first * 2 + 1] - positions[second * 2 + 1];
    return (
      sill * getReferenceCorrelation(model, Math.hypot(offsetX, offsetY) / range) +
      (first === second || (offsetX === 0 && offsetY === 0) ? nugget : 0)
    );
  };
  for (let site = 0; site < rowCount; site++) {
    const begin = offsets[site];
    const count = offsets[site + 1] - begin;
    const rows: number[] = [];
    for (let slot = 0; slot < count; slot++) {
      rows.push(neighbors[begin + slot]);
    }
    const lower = new Float64Array(count * count);
    for (let row = 0; row < count; row++) {
      for (let column = 0; column <= row; column++) {
        lower[row * count + column] = covarianceAt(rows[row], rows[column]);
      }
    }
    for (let column = 0; column < count; column++) {
      let diagonal = lower[column * count + column];
      for (let index = 0; index < column; index++) {
        diagonal -= lower[column * count + index] ** 2;
      }
      const pivot = Math.sqrt(diagonal);
      lower[column * count + column] = pivot;
      for (let row = column + 1; row < count; row++) {
        let value = lower[row * count + column];
        for (let index = 0; index < column; index++) {
          value -= lower[row * count + index] * lower[column * count + index];
        }
        lower[row * count + column] = value / pivot;
      }
    }
    const solve = (right: Float64Array): Float64Array => {
      const solution = Float64Array.from(right);
      for (let row = 0; row < count; row++) {
        for (let column = 0; column < row; column++) {
          solution[row] -= lower[row * count + column] * solution[column];
        }
        solution[row] /= lower[row * count + row];
      }
      for (let row = count - 1; row >= 0; row--) {
        for (let column = row + 1; column < count; column++) {
          solution[row] -= lower[column * count + row] * solution[column];
        }
        solution[row] /= lower[row * count + row];
      }
      return solution;
    };
    const cross = Float64Array.from(rows, row => covarianceAt(row, site));
    const weights = solve(cross);
    let variance = covarianceAt(site, site);
    let prediction = 0;
    for (let row = 0; row < count; row++) {
      variance -= weights[row] * cross[row];
      prediction += weights[row] * values[rows[row]];
    }
    const residual = values[site] - prediction;
    logLikelihood +=
      -0.5 * (Math.log(2 * Math.PI) + Math.log(variance) + (residual * residual) / variance);
    const derivativeVariance = [0, 0, 0];
    const derivativeResidual = [0, 0, 0];
    const vectors: Float64Array[] = [];
    const solved: Float64Array[] = [];
    for (let parameter = 0; parameter < 3; parameter++) {
      const vector = new Float64Array(count);
      let sumDerivative = parameterDerivative(parameter, site, site);
      for (let row = 0; row < count; row++) {
        const derivativeCross = parameterDerivative(parameter, rows[row], site);
        let value = derivativeCross;
        for (let column = 0; column < count; column++) {
          value -= parameterDerivative(parameter, rows[row], rows[column]) * weights[column];
        }
        vector[row] = value;
        sumDerivative -= (derivativeCross + value) * weights[row];
      }
      derivativeVariance[parameter] = sumDerivative;
      const solution = solve(vector);
      vectors.push(vector);
      solved.push(solution);
      let total = 0;
      for (let row = 0; row < count; row++) {
        total += solution[row] * values[rows[row]];
      }
      derivativeResidual[parameter] = -total;
    }
    const standardized = (residual * residual) / variance;
    for (let first = 0; first < 3; first++) {
      score[first] +=
        -0.5 * (derivativeVariance[first] / variance) * (1 - standardized) -
        (residual * derivativeResidual[first]) / variance;
    }
    let cell = 0;
    for (let first = 0; first < 3; first++) {
      for (let second = first; second < 3; second++) {
        let quadratic = 0;
        for (let row = 0; row < count; row++) {
          quadratic += vectors[first][row] * solved[second][row];
        }
        information[cell++] +=
          (0.5 * derivativeVariance[first] * derivativeVariance[second]) / (variance * variance) +
          quadratic / variance;
      }
    }
  }
  return {logLikelihood, score, information};
}

/** Generic reference over an arbitrary row-pair covariance (the space-time case). */
export function evaluateReferenceConditionalLogLikelihood(
  values: ArrayLike<number>,
  offsets: ArrayLike<number>,
  neighbors: ArrayLike<number>,
  getCovariance: (firstRow: number, secondRow: number) => number
): number {
  const rowCount = offsets.length - 1;
  let logLikelihood = 0;
  for (let site = 0; site < rowCount; site++) {
    const begin = offsets[site];
    const count = offsets[site + 1] - begin;
    const rowAt = (slot: number): number => (slot < count ? neighbors[begin + slot] : site);
    const size = count + 1;
    const lower = new Float64Array(size * size);
    for (let row = 0; row < size; row++) {
      for (let column = 0; column <= row; column++) {
        lower[row * size + column] = getCovariance(rowAt(row), rowAt(column));
      }
    }
    for (let column = 0; column < size; column++) {
      let diagonal = lower[column * size + column];
      for (let index = 0; index < column; index++) {
        diagonal -= lower[column * size + index] ** 2;
      }
      const pivot = Math.sqrt(diagonal);
      lower[column * size + column] = pivot;
      for (let row = column + 1; row < size; row++) {
        let value = lower[row * size + column];
        for (let index = 0; index < column; index++) {
          value -= lower[row * size + index] * lower[column * size + index];
        }
        lower[row * size + column] = value / pivot;
      }
    }
    let mean = 0;
    const forward = new Float64Array(count);
    for (let row = 0; row < count; row++) {
      let value = values[rowAt(row)];
      for (let index = 0; index < row; index++) {
        value -= lower[row * size + index] * forward[index];
      }
      forward[row] = value / lower[row * size + row];
      mean += lower[count * size + row] * forward[row];
    }
    const deviation = lower[count * size + count];
    const standardized = (values[site] - mean) / deviation;
    logLikelihood +=
      -0.5 * (Math.log(2 * Math.PI) + 2 * Math.log(deviation) + standardized * standardized);
  }
  return logLikelihood;
}

/** Sequential binary64 simulation over an arbitrary row-pair covariance, rows in Vecchia order. */
export function simulateReferenceConditionalField(
  gaussian: () => number,
  offsets: ArrayLike<number>,
  neighbors: ArrayLike<number>,
  getCovariance: (firstRow: number, secondRow: number) => number
): Float64Array {
  const rowCount = offsets.length - 1;
  const values = new Float64Array(rowCount);
  for (let site = 0; site < rowCount; site++) {
    const begin = offsets[site];
    const count = offsets[site + 1] - begin;
    const rowAt = (slot: number): number => (slot < count ? neighbors[begin + slot] : site);
    const size = count + 1;
    const lower = new Float64Array(size * size);
    for (let row = 0; row < size; row++) {
      for (let column = 0; column <= row; column++) {
        lower[row * size + column] = getCovariance(rowAt(row), rowAt(column));
      }
    }
    for (let column = 0; column < size; column++) {
      let diagonal = lower[column * size + column];
      for (let index = 0; index < column; index++) {
        diagonal -= lower[column * size + index] ** 2;
      }
      const pivot = Math.sqrt(Math.max(diagonal, 1e-300));
      lower[column * size + column] = pivot;
      for (let row = column + 1; row < size; row++) {
        let value = lower[row * size + column];
        for (let index = 0; index < column; index++) {
          value -= lower[row * size + index] * lower[column * size + index];
        }
        lower[row * size + column] = value / pivot;
      }
    }
    let mean = 0;
    const forward = new Float64Array(count);
    for (let row = 0; row < count; row++) {
      let value = values[rowAt(row)];
      for (let index = 0; index < row; index++) {
        value -= lower[row * size + index] * forward[index];
      }
      forward[row] = value / lower[row * size + row];
      mean += lower[count * size + row] * forward[row];
    }
    values[site] = mean + lower[count * size + count] * gaussian();
  }
  return values;
}
