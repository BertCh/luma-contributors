// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getVecchiaNuggetFloor, type GPUVecchiaCovariance} from './vecchia-parameters';

/** One evaluation of the Vecchia likelihood and its derivatives at a covariance. */
export type VecchiaFisherEvaluation = {
  logLikelihood: number;
  /** Score with respect to `(log sill, log effective range, log nugget)`. */
  score: ArrayLike<number>;
  /** Upper triangle `I_00, I_01, I_02, I_11, I_12, I_22` of the expected information. */
  information: ArrayLike<number>;
  failedSiteCount: number;
};

/** Options of {@link fitVecchiaCovarianceFisherScoring}. */
export type VecchiaFisherFitOptions = {
  initial: GPUVecchiaCovariance;
  /** Runs one GPU or CPU evaluation; a `failedSiteCount > 0` or a non-finite value rejects the step. */
  evaluate: (
    covariance: GPUVecchiaCovariance
  ) => Promise<VecchiaFisherEvaluation> | VecchiaFisherEvaluation;
  /** Estimate the nugget. Default true; a zero nugget stays zero and fixed. */
  fitNugget?: boolean;
  /** Most Fisher iterations. Default 30. */
  maximumIterations?: number;
  /** Stops at Newton decrement `g' I^-1 g` below this. Default `1e-3`. */
  convergenceTolerance?: number;
  /**
   * Largest log-likelihood decrease a step may show and still be accepted, so float32 noise in the
   * total does not stall the line search. Default `0.05`, the size of the differences the float32
   * study reports at 10,000 sites.
   */
  logLikelihoodNoise?: number;
};

/** Result of {@link fitVecchiaCovarianceFisherScoring}. */
export type VecchiaFisherFit = {
  covariance: GPUVecchiaCovariance;
  logLikelihood: number;
  iterationCount: number;
  /** One when the Newton decrement fell below the tolerance. */
  converged: boolean;
  /** One when the budget ran out first. */
  iterationLimitReached: boolean;
  /** True when the first evaluation already failed. */
  failed: boolean;
  /** True when the nugget stopped at `getVecchiaNuggetFloor`. */
  atNuggetFloor: boolean;
  /** Log-likelihood and Newton decrement after each accepted iteration. */
  history: {logLikelihood: number; decrement: number}[];
};

const MAXIMUM_LINE_SEARCH_HALVINGS = 6;
/** Stops a single step from moving a log parameter by more than a factor of about 4.5. */
const MAXIMUM_PARAMETER_STEP = 1.5;

function isUsable(evaluation: VecchiaFisherEvaluation): boolean {
  return evaluation.failedSiteCount === 0 && Number.isFinite(evaluation.logLikelihood);
}

/** Solves a small dense symmetric system restricted to `active`, by Gaussian elimination. */
function solveActive(
  information: ArrayLike<number>,
  score: ArrayLike<number>,
  active: readonly number[]
): number[] {
  const matrixIndex = (row: number, column: number): number => {
    const first = Math.min(row, column);
    const second = Math.max(row, column);
    return first === 0 ? second : first === 1 ? 2 + second : 5;
  };
  const size = active.length;
  const matrix = active.map(row => active.map(column => information[matrixIndex(row, column)]));
  const right = active.map(row => score[row]);
  for (let row = 0; row < size; row++) {
    matrix[row][row] *= 1 + 1e-9;
  }
  for (let pivot = 0; pivot < size; pivot++) {
    let best = pivot;
    for (let row = pivot + 1; row < size; row++) {
      if (Math.abs(matrix[row][pivot]) > Math.abs(matrix[best][pivot])) {
        best = row;
      }
    }
    [matrix[pivot], matrix[best]] = [matrix[best], matrix[pivot]];
    [right[pivot], right[best]] = [right[best], right[pivot]];
    for (let row = pivot + 1; row < size; row++) {
      const factor = matrix[row][pivot] / matrix[pivot][pivot];
      for (let column = pivot; column < size; column++) {
        matrix[row][column] -= factor * matrix[pivot][column];
      }
      right[row] -= factor * right[pivot];
    }
  }
  const solution = new Array<number>(size).fill(0);
  for (let row = size - 1; row >= 0; row--) {
    let value = right[row];
    for (let column = row + 1; column < size; column++) {
      value -= matrix[row][column] * solution[column];
    }
    solution[row] = value / matrix[row][row];
  }
  return solution;
}

/**
 * Fits sill, effective range and optionally nugget of a Vecchia likelihood by Fisher scoring
 * (Guinness 2019) in log parameters, with a CPU-driven outer loop and a halving line search. Each
 * step reads one evaluation back. Nugget floors from `getVecchiaNuggetFloor` are enforced by clamping
 * the proposal and dropping the nugget from the system while the score pushes against the floor.
 */
export async function fitVecchiaCovarianceFisherScoring(
  options: VecchiaFisherFitOptions
): Promise<VecchiaFisherFit> {
  const {initial, evaluate} = options;
  const maximumIterations = options.maximumIterations ?? 30;
  const tolerance = options.convergenceTolerance ?? 1e-3;
  const noise = options.logLikelihoodNoise ?? 0.05;
  const estimatesNugget = (options.fitNugget ?? true) && (initial.nugget ?? 0) > 0;
  let theta = [
    Math.log(initial.sill),
    Math.log(initial.range),
    Math.log(Math.max(initial.nugget ?? 0, Number.MIN_VALUE))
  ];
  const toCovariance = (values: readonly number[]): GPUVecchiaCovariance => {
    const sill = Math.exp(values[0]);
    const proposedNugget = estimatesNugget ? Math.exp(values[2]) : (initial.nugget ?? 0);
    return {
      model: initial.model,
      sill,
      range: Math.exp(values[1]),
      nugget: Math.max(proposedNugget, getVecchiaNuggetFloor({model: initial.model, sill}))
    };
  };
  let current = await evaluate(toCovariance(theta));
  const history: {logLikelihood: number; decrement: number}[] = [];
  if (!isUsable(current)) {
    return {
      covariance: toCovariance(theta),
      logLikelihood: Number.NaN,
      iterationCount: 0,
      converged: false,
      iterationLimitReached: false,
      failed: true,
      atNuggetFloor: false,
      history
    };
  }
  let iteration = 0;
  let converged = false;
  for (; iteration < maximumIterations; iteration++) {
    const atFloor = (): boolean => {
      const covariance = toCovariance(theta);
      return (
        covariance.nugget! <=
        getVecchiaNuggetFloor({model: initial.model, sill: covariance.sill}) * (1 + 1e-9)
      );
    };
    const nuggetFrozen = estimatesNugget && atFloor() && current.score[2] < 0;
    const active = [0, 1];
    if (estimatesNugget && !nuggetFrozen) {
      active.push(2);
    }
    // Held at the floor, the nugget moves with the sill, so log sill also carries its score and
    // information, keeping the step a true Fisher step for the constrained problem.
    const score = Array.from(current.score);
    const information = Array.from(current.information);
    if (nuggetFrozen) {
      score[0] += current.score[2];
      information[0] += 2 * current.information[2] + current.information[5];
      information[1] += current.information[4];
    }
    const step = solveActive(information, score, active);
    const decrement = step.reduce((sum, value, row) => sum + value * score[active[row]], 0);
    if (decrement < tolerance) {
      converged = true;
      history.push({logLikelihood: current.logLikelihood, decrement});
      break;
    }
    const largest = Math.max(...step.map(Math.abs));
    const damp = largest > MAXIMUM_PARAMETER_STEP ? MAXIMUM_PARAMETER_STEP / largest : 1;
    let scale = damp;
    let accepted = false;
    for (let halving = 0; halving < MAXIMUM_LINE_SEARCH_HALVINGS && !accepted; halving++) {
      const trial = theta.slice();
      active.forEach((parameter, row) => {
        trial[parameter] += scale * step[row];
      });
      if (estimatesNugget) {
        const floor = getVecchiaNuggetFloor({model: initial.model, sill: Math.exp(trial[0])});
        if (floor > 0) {
          trial[2] = Math.max(trial[2], Math.log(floor));
        }
      }
      const evaluation = await evaluate(toCovariance(trial));
      if (isUsable(evaluation) && evaluation.logLikelihood >= current.logLikelihood - noise) {
        theta = trial;
        current = evaluation;
        accepted = true;
      } else {
        scale /= 2;
      }
    }
    history.push({logLikelihood: current.logLikelihood, decrement});
    if (!accepted) {
      break;
    }
  }
  const covariance = toCovariance(theta);
  return {
    covariance,
    logLikelihood: current.logLikelihood,
    iterationCount: iteration,
    converged,
    iterationLimitReached: !converged && iteration >= maximumIterations,
    failed: false,
    atNuggetFloor:
      covariance.nugget! <=
        getVecchiaNuggetFloor({model: initial.model, sill: covariance.sill}) * (1 + 1e-9) &&
      getVecchiaNuggetFloor({model: initial.model, sill: covariance.sill}) > 0,
    history
  };
}

/** Options of {@link maximizeVecchiaLogLikelihood}. */
export type VecchiaNelderMeadOptions = {
  /** Starting point. */
  initial: readonly number[];
  /** Starting simplex offset per coordinate. Defaults to 0.5. */
  step?: number;
  /** Log-likelihood to maximize; a non-finite value is treated as very poor. */
  evaluate: (values: readonly number[]) => Promise<number> | number;
  /** Most evaluations. Default 200. */
  maximumEvaluations?: number;
  /** Stops when the simplex spread in value falls below this. Default 1e-2. */
  tolerance?: number;
};

/** Result of {@link maximizeVecchiaLogLikelihood}. */
export type VecchiaNelderMeadResult = {
  values: number[];
  logLikelihood: number;
  evaluationCount: number;
  /** One when the spread criterion was met. */
  converged: boolean;
  /** One when the evaluation budget ran out first. */
  iterationLimitReached: boolean;
};

/**
 * CPU-driven Nelder-Mead over free per-frame parameters (often log-transformed), for models without
 * a Fisher kernel, such as the space-time likelihood. Each evaluation is one GPU run.
 */
export async function maximizeVecchiaLogLikelihood(
  options: VecchiaNelderMeadOptions
): Promise<VecchiaNelderMeadResult> {
  const {initial, evaluate} = options;
  const maximumEvaluations = options.maximumEvaluations ?? 200;
  const tolerance = options.tolerance ?? 1e-2;
  const step = options.step ?? 0.5;
  const dimension = initial.length;
  let evaluationCount = 0;
  const score = async (point: readonly number[]): Promise<number> => {
    evaluationCount++;
    const value = await evaluate(point);
    return Number.isFinite(value) ? value : -Infinity;
  };
  const simplex: {point: number[]; value: number}[] = [];
  simplex.push({point: initial.slice(), value: await score(initial)});
  for (let coordinate = 0; coordinate < dimension; coordinate++) {
    const point = initial.slice();
    point[coordinate] += step;
    simplex.push({point, value: await score(point)});
  }
  let converged = false;
  while (evaluationCount < maximumEvaluations) {
    simplex.sort((first, second) => second.value - first.value);
    if (Math.abs(simplex[0].value - simplex[dimension].value) < tolerance) {
      converged = true;
      break;
    }
    const centroid = new Array<number>(dimension).fill(0);
    for (let vertex = 0; vertex < dimension; vertex++) {
      for (let coordinate = 0; coordinate < dimension; coordinate++) {
        centroid[coordinate] += simplex[vertex].point[coordinate] / dimension;
      }
    }
    const worst = simplex[dimension];
    const along = (factor: number): number[] =>
      centroid.map((value, coordinate) => value + factor * (value - worst.point[coordinate]));
    const reflected = along(1);
    const reflectedValue = await score(reflected);
    if (reflectedValue >= simplex[0].value) {
      const expanded = along(2);
      const expandedValue = await score(expanded);
      simplex[dimension] =
        expandedValue > reflectedValue
          ? {point: expanded, value: expandedValue}
          : {point: reflected, value: reflectedValue};
    } else if (reflectedValue >= simplex[dimension - 1].value) {
      simplex[dimension] = {point: reflected, value: reflectedValue};
    } else {
      const contracted = along(reflectedValue >= worst.value ? 0.5 : -0.5);
      const contractedValue = await score(contracted);
      if (contractedValue > Math.max(reflectedValue, worst.value)) {
        simplex[dimension] = {point: contracted, value: contractedValue};
      } else {
        for (let vertex = 1; vertex <= dimension; vertex++) {
          const point = simplex[vertex].point.map(
            (value, coordinate) => (value + simplex[0].point[coordinate]) / 2
          );
          simplex[vertex] = {point, value: await score(point)};
        }
      }
    }
  }
  simplex.sort((first, second) => second.value - first.value);
  return {
    values: simplex[0].point,
    logLikelihood: simplex[0].value,
    evaluationCount,
    converged,
    iterationLimitReached: !converged
  };
}
