// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  fitVariogramModel,
  getVariogramModelShape,
  type VariogramModelType,
  type VariogramModelWeighting
} from './variogram-model';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** One marginal or joint structure: `gamma(lag) = nugget + sill * shape(lag / range)` for lag > 0. */
export type SpaceTimeVariogramStructure = {
  /** Family of the structure. */
  model: VariogramModelType;
  /** Nugget at least 0. Applies when this structure's lag is positive. */
  nugget: number;
  /** Partial sill, at least 0. */
  sill: number;
  /** Effective range, positive, in the lag units of the structure. */
  range: number;
};

/**
 * Space-time semivariogram models, using the gstat `vgmST` definitions (the gstat 2.1-6 vignette,
 * Graler, Pebesma and Heuvelink), plus the Gneiting non-separable class:
 *
 * - `separable`: `gamma = sill * (gs + gt - gs gt)`, on structures whose nugget plus sill is 1.
 * - `product-sum`: `gamma = (k St + 1) gs + (k Ss + 1) gt - k gs gt` on absolute structures, where
 *   `Ss`, `St` are the total sills; `C = k Cs Ct + Cs + Ct`.
 * - `metric`: `gamma = gj(sqrt(h^2 + (anisotropy u)^2))`.
 * - `sum-metric`: `gamma = gs(h) + gt(u) + gj(sqrt(h^2 + (anisotropy u)^2))`.
 * - `gneiting`: `C = nugget [h = u = 0] + sill psi^-tau exp(-c h^(2 gamma) psi^(-beta gamma))`
 *   with `psi = a u^(2 alpha) + 1` (Gneiting 2002), where `c = 3 / spatialRange^(2 gamma)` and
 *   `a = (exp(3 / tau) - 1) / temporalRange^(2 alpha)`, so the correlation is `exp(-3)` at
 *   `(spatialRange, 0)` and at `(0, temporalRange)`.
 *
 * `anisotropy` is spatial units per unit of time (gstat `stAni`).
 */
export type SpaceTimeVariogramModel =
  | {
      type: 'separable';
      sill: number;
      spatial: SpaceTimeVariogramStructure;
      temporal: SpaceTimeVariogramStructure;
    }
  | {
      type: 'product-sum';
      k: number;
      spatial: SpaceTimeVariogramStructure;
      temporal: SpaceTimeVariogramStructure;
    }
  | {type: 'metric'; joint: SpaceTimeVariogramStructure; anisotropy: number}
  | {
      type: 'sum-metric';
      spatial: SpaceTimeVariogramStructure;
      temporal: SpaceTimeVariogramStructure;
      joint: SpaceTimeVariogramStructure;
      anisotropy: number;
    }
  | {
      type: 'gneiting';
      nugget: number;
      sill: number;
      spatialRange: number;
      temporalRange: number;
      /** Temporal smoothness, in (0, 1]. */
      alpha: number;
      /** Spatial smoothness, in (0, 1]. */
      gamma: number;
      /** Space-time interaction, in [0, 1]; 0 is separable. */
      beta: number;
      /** Temporal exponent, at least `beta` in two spatial dimensions. */
      tau: number;
    };

/** Space-time model families of {@link SpaceTimeVariogramModel}. */
export type SpaceTimeVariogramModelType = SpaceTimeVariogramModel['type'];

/** Gneiting `c` of a model, as stored in the kriging parameter buffer. */
export function getGneitingSpatialScale(spatialRange: number, gamma: number): number {
  return 3 / spatialRange ** (2 * gamma);
}

/** Gneiting `a` of a model, as stored in the kriging parameter buffer. */
export function getGneitingTemporalScale(
  temporalRange: number,
  alpha: number,
  tau: number
): number {
  return (Math.exp(3 / tau) - 1) / temporalRange ** (2 * alpha);
}

function evaluateStructure(structure: SpaceTimeVariogramStructure, lag: number): number {
  return lag <= 0
    ? 0
    : structure.nugget +
        structure.sill * getVariogramModelShape(structure.model, lag / structure.range);
}

/**
 * Evaluates a space-time semivariogram at spatial lag `distance` and time lag `timeLag`
 * (`0` at the origin).
 */
export function evaluateSpaceTimeVariogramModel(
  model: SpaceTimeVariogramModel,
  distance: number,
  timeLag: number
): number {
  if (distance <= 0 && timeLag <= 0) {
    return 0;
  }
  switch (model.type) {
    case 'separable': {
      const spatial = evaluateStructure(model.spatial, distance);
      const temporal = evaluateStructure(model.temporal, timeLag);
      return model.sill * (spatial + temporal - spatial * temporal);
    }
    case 'product-sum': {
      const spatial = evaluateStructure(model.spatial, distance);
      const temporal = evaluateStructure(model.temporal, timeLag);
      const spatialTotal = model.spatial.nugget + model.spatial.sill;
      const temporalTotal = model.temporal.nugget + model.temporal.sill;
      return (
        (model.k * temporalTotal + 1) * spatial +
        (model.k * spatialTotal + 1) * temporal -
        model.k * spatial * temporal
      );
    }
    case 'metric':
      return evaluateStructure(model.joint, Math.hypot(distance, model.anisotropy * timeLag));
    case 'sum-metric':
      return (
        evaluateStructure(model.spatial, distance) +
        evaluateStructure(model.temporal, timeLag) +
        evaluateStructure(model.joint, Math.hypot(distance, model.anisotropy * timeLag))
      );
    case 'gneiting':
      return (
        model.nugget + model.sill * (1 - evaluateGneitingCorrelation(model, distance, timeLag))
      );
    default:
      throw new Error('Unknown space-time variogram model');
  }
}

function evaluateGneitingCorrelation(
  model: Extract<SpaceTimeVariogramModel, {type: 'gneiting'}>,
  distance: number,
  timeLag: number
): number {
  const c = getGneitingSpatialScale(model.spatialRange, model.gamma);
  const a = getGneitingTemporalScale(model.temporalRange, model.alpha, model.tau);
  const psi = a * timeLag ** (2 * model.alpha) + 1;
  return (
    psi ** -model.tau *
    Math.exp(-c * distance ** (2 * model.gamma) * psi ** (-model.beta * model.gamma))
  );
}

/** Empirical bins of a space-time surface, for example from `GPUSpaceTimeVariogram`. */
export type SpaceTimeVariogramInput = {
  /** Representative spatial lag per bin. */
  distances: ArrayLike<number>;
  /** Representative time lag per bin. */
  timeLags: ArrayLike<number>;
  /** Empirical semivariance per bin. */
  semivariances: ArrayLike<number>;
  /** Pair count per bin; bins without pairs, and the zero-zero bin, are skipped. */
  pairCounts: ArrayLike<number>;
};

/**
 * Representative lags of the spatial-major `GPUSpaceTimeVariogram` bins: lag 0 for class 0 of an
 * axis and the class midpoint otherwise.
 */
export function getGPUSpaceTimeVariogramBinLags(props: {
  lagCount: number;
  temporalLagCount: number;
  maximumDistance: number;
  maximumTimeLag: number;
}): {distances: Float64Array; timeLags: Float64Array} {
  const {lagCount, temporalLagCount, maximumDistance, maximumTimeLag} = props;
  const binCount = (lagCount + 1) * (temporalLagCount + 1);
  const distances = new Float64Array(binCount);
  const timeLags = new Float64Array(binCount);
  for (let spatialClass = 0; spatialClass <= lagCount; spatialClass++) {
    for (let temporalClass = 0; temporalClass <= temporalLagCount; temporalClass++) {
      const bin = spatialClass * (temporalLagCount + 1) + temporalClass;
      distances[bin] = spatialClass === 0 ? 0 : ((spatialClass - 0.5) * maximumDistance) / lagCount;
      timeLags[bin] =
        temporalClass === 0 ? 0 : ((temporalClass - 0.5) * maximumTimeLag) / temporalLagCount;
    }
  }
  return {distances, timeLags};
}

/** Options of {@link fitSpaceTimeVariogramModel}. */
export type SpaceTimeVariogramFitOptions = {
  /** Coordinate cycles per weighting pass, integer in [1, 24]. Defaults to 24. */
  maximumCycles?: number;
  /** Space-time family. */
  type: SpaceTimeVariogramModelType;
  /** Family of the spatial, temporal and joint structures. Defaults to `'exponential'`. */
  structure?: VariogramModelType;
  /** Least-squares weighting, as in `fitVariogramModel`. Defaults to `'cressie'`. */
  weighting?: VariogramModelWeighting;
  /** Fixed Gneiting smoothness and temporal exponent. Default to 0.5, 0.5 and 1. */
  gneiting?: {alpha?: number; gamma?: number; tau?: number};
};

/** Fitted space-time model. */
export type SpaceTimeVariogramFit = {
  model: SpaceTimeVariogramModel;
  /** Weighted sum of squared residuals. */
  residual: number;
  /** True when a nonlinear parameter stopped at the edge of its search ladder. */
  atBoundary: boolean;
  /** Actual completed coordinate cycles in the final weighting pass, at most maximumCycles. */
  cycleCount: number;
  /** Final pass met the relative residual improvement criterion (1e-9), even at a boundary. */
  converged: boolean;
};

type SearchParameter = {low: number; high: number; scale: 'log' | 'linear'; steps: number};
type LinearEvaluation = {residual: number; coefficients: Float64Array};

const RANGE_STEPS = 48;
const FRACTION_STEPS = 21;
const COORDINATE_CYCLES = 24;
const GOLDEN_ITERATIONS = 30;
const CRESSIE_PASSES = 3;
const MAXIMUM_FRACTION = 0.95;

/** Lawson-Hanson NNLS on normal equations, which stay well conditioned for these few columns. */
function solveNonNegative(
  normal: Float64Array,
  right: Float64Array,
  columns: number
): Float64Array {
  const solution = new Float64Array(columns);
  const passive = new Array<boolean>(columns).fill(false);
  for (let round = 0; round < 4 * columns + 8; round++) {
    let bestColumn = -1;
    let bestGradient = 1e-14;
    for (let column = 0; column < columns; column++) {
      if (passive[column]) {
        continue;
      }
      let gradient = right[column];
      for (let other = 0; other < columns; other++) {
        gradient -= normal[column * columns + other] * solution[other];
      }
      if (gradient > bestGradient) {
        bestGradient = gradient;
        bestColumn = column;
      }
    }
    if (bestColumn < 0) {
      break;
    }
    passive[bestColumn] = true;
    for (let inner = 0; inner < 2 * columns + 4; inner++) {
      const active = passive.flatMap((isPassive, index) => (isPassive ? [index] : []));
      const candidate = solveDense(normal, right, columns, active);
      if (!candidate) {
        passive[bestColumn] = false;
        break;
      }
      if (active.every(column => candidate[column] > 0)) {
        solution.set(candidate);
        break;
      }
      let step = 1;
      for (const column of active) {
        if (candidate[column] <= 0) {
          const denominator = solution[column] - candidate[column];
          step = Math.min(step, denominator > 0 ? solution[column] / denominator : 0);
        }
      }
      for (const column of active) {
        solution[column] += step * (candidate[column] - solution[column]);
        if (solution[column] <= 1e-14) {
          solution[column] = 0;
          passive[column] = false;
        }
      }
    }
  }
  return solution;
}

/** Solves the normal equations restricted to `active`; `undefined` when singular. */
function solveDense(
  normal: Float64Array,
  right: Float64Array,
  columns: number,
  active: readonly number[]
): Float64Array | undefined {
  const size = active.length;
  const matrix = new Float64Array(size * size);
  const vector = new Float64Array(size);
  active.forEach((row, rowIndex) => {
    active.forEach((column, columnIndex) => {
      matrix[rowIndex * size + columnIndex] = normal[row * columns + column];
    });
    vector[rowIndex] = right[row];
  });
  for (let pivot = 0; pivot < size; pivot++) {
    let bestRow = pivot;
    for (let row = pivot + 1; row < size; row++) {
      if (Math.abs(matrix[row * size + pivot]) > Math.abs(matrix[bestRow * size + pivot])) {
        bestRow = row;
      }
    }
    if (Math.abs(matrix[bestRow * size + pivot]) < 1e-30) {
      return undefined;
    }
    for (let column = 0; column < size; column++) {
      const held = matrix[pivot * size + column];
      matrix[pivot * size + column] = matrix[bestRow * size + column];
      matrix[bestRow * size + column] = held;
    }
    const heldValue = vector[pivot];
    vector[pivot] = vector[bestRow];
    vector[bestRow] = heldValue;
    for (let row = pivot + 1; row < size; row++) {
      const factor = matrix[row * size + pivot] / matrix[pivot * size + pivot];
      for (let column = pivot; column < size; column++) {
        matrix[row * size + column] -= factor * matrix[pivot * size + column];
      }
      vector[row] -= factor * vector[pivot];
    }
  }
  const candidate = new Float64Array(columns);
  for (let row = size - 1; row >= 0; row--) {
    let value = vector[row];
    for (let column = row + 1; column < size; column++) {
      value -= matrix[row * size + column] * candidate[active[column]];
    }
    candidate[active[row]] = value / matrix[row * size + row];
  }
  return candidate;
}

/** Bins kept for fitting, with weights and the linear design rebuilt per nonlinear candidate. */
type FitBins = {
  distances: number[];
  timeLags: number[];
  semivariances: number[];
  pairCounts: number[];
};

/** Weighted NNLS on `columns` given a per-bin design writer. */
function evaluateLinearDesign(
  bins: FitBins,
  weights: Float64Array,
  columns: number,
  writeDesign: (bin: number, row: Float64Array) => void
): LinearEvaluation {
  const count = bins.distances.length;
  const design = new Float64Array(count * columns);
  const row = new Float64Array(columns);
  const normal = new Float64Array(columns * columns);
  const right = new Float64Array(columns);
  for (let bin = 0; bin < count; bin++) {
    writeDesign(bin, row);
    design.set(row, bin * columns);
    const weight = weights[bin];
    for (let first = 0; first < columns; first++) {
      right[first] += weight * row[first] * bins.semivariances[bin];
      for (let second = 0; second < columns; second++) {
        normal[first * columns + second] += weight * row[first] * row[second];
      }
    }
  }
  const coefficients = solveNonNegative(normal, right, columns);
  let residual = 0;
  for (let bin = 0; bin < count; bin++) {
    let fitted = 0;
    for (let column = 0; column < columns; column++) {
      fitted += design[bin * columns + column] * coefficients[column];
    }
    const difference = bins.semivariances[bin] - fitted;
    residual += weights[bin] * difference * difference;
  }
  return {residual, coefficients};
}

/** Coordinate search with a ladder and golden-section refinement of each nonlinear parameter. */
function searchCoordinates(
  parameters: readonly SearchParameter[],
  start: readonly number[],
  evaluate: (values: readonly number[]) => LinearEvaluation,
  maximumCycles: number
): {
  values: number[];
  evaluation: LinearEvaluation;
  atBoundary: boolean;
  cycleCount: number;
  converged: boolean;
} {
  const values = start.slice();
  let best = evaluate(values);
  const ratio = (Math.sqrt(5) - 1) / 2;
  let cycleCount = 0;
  let converged = false;
  for (let cycle = 0; cycle < maximumCycles; cycle++) {
    cycleCount++;
    const beforeCycle = best.residual;
    parameters.forEach((parameter, parameterIndex) => {
      const position = (step: number): number => {
        const fraction = step / (parameter.steps - 1);
        return parameter.scale === 'log'
          ? Math.exp(Math.log(parameter.low) + fraction * Math.log(parameter.high / parameter.low))
          : parameter.low + fraction * (parameter.high - parameter.low);
      };
      const candidateAt = (value: number) => {
        const trial = values.slice();
        trial[parameterIndex] = value;
        return evaluate(trial);
      };
      let bestStep = 0;
      let bestStepResidual = Infinity;
      for (let step = 0; step < parameter.steps; step++) {
        const evaluation = candidateAt(position(step));
        if (evaluation.residual < bestStepResidual) {
          bestStepResidual = evaluation.residual;
          bestStep = step;
        }
      }
      if (bestStepResidual < best.residual) {
        best = candidateAt(position(bestStep));
        values[parameterIndex] = position(bestStep);
      }
      const toSearch = (value: number): number =>
        parameter.scale === 'log' ? Math.log(value) : value;
      const fromSearch = (value: number): number =>
        parameter.scale === 'log' ? Math.exp(value) : value;
      let left = toSearch(position(Math.max(bestStep - 1, 0)));
      let right = toSearch(position(Math.min(bestStep + 1, parameter.steps - 1)));
      let inner = right - ratio * (right - left);
      let outer = left + ratio * (right - left);
      let innerEvaluation = candidateAt(fromSearch(inner));
      let outerEvaluation = candidateAt(fromSearch(outer));
      for (let iteration = 0; iteration < GOLDEN_ITERATIONS; iteration++) {
        if (innerEvaluation.residual <= outerEvaluation.residual) {
          right = outer;
          outer = inner;
          outerEvaluation = innerEvaluation;
          inner = right - ratio * (right - left);
          innerEvaluation = candidateAt(fromSearch(inner));
        } else {
          left = inner;
          inner = outer;
          innerEvaluation = outerEvaluation;
          outer = left + ratio * (right - left);
          outerEvaluation = candidateAt(fromSearch(outer));
        }
      }
      for (const [value, evaluation] of [
        [inner, innerEvaluation],
        [outer, outerEvaluation]
      ] as const) {
        if (evaluation.residual < best.residual) {
          best = evaluation;
          values[parameterIndex] = fromSearch(value);
        }
      }
    });
    if (best.residual >= beforeCycle * (1 - 1e-9)) {
      converged = Number.isFinite(best.residual);
      break;
    }
  }
  const atBoundary = parameters.some((parameter, parameterIndex) => {
    const span =
      parameter.scale === 'log' ? parameter.high / parameter.low : parameter.high - parameter.low;
    const value = values[parameterIndex];
    return parameter.scale === 'log'
      ? value <= parameter.low * span ** 1e-3 || value >= parameter.high / span ** 1e-3
      : value <= parameter.low + span * 1e-3 || value >= parameter.high - span * 1e-3;
  });
  return {values, evaluation: best, atBoundary, cycleCount, converged};
}

/**
 * Fits a space-time model to empirical bins by weighted least squares, on the CPU. Bins are the
 * read-back output of `GPUSpaceTimeVariogram` (at most 409), so this stays on the CPU like
 * `fitVariogramModel`, and it uses the same ladder: each nonlinear parameter (ranges, anisotropy,
 * nugget fractions, the Gneiting `beta`) gets a ladder and golden-section refinement in coordinate
 * cycles, and the remaining nugget and sill terms are the exact non-negative weighted linear
 * least-squares solution for those values. The margins (spatial lag at zero time lag, temporal lag
 * at zero spatial lag) start the ranges, through `fitVariogramModel`. Cressie weighting repeats the
 * search three times. The bin at zero spatial and temporal lag is ignored.
 *
 * @throws If the bin arrays differ in length or fewer than two bins are usable.
 */
export function fitSpaceTimeVariogramModel(
  input: SpaceTimeVariogramInput,
  options: SpaceTimeVariogramFitOptions
): SpaceTimeVariogramFit {
  const maximumCycles = options.maximumCycles ?? COORDINATE_CYCLES;
  validateCapacity(
    'fitSpaceTimeVariogramModel',
    'maximumCycles',
    maximumCycles,
    1,
    COORDINATE_CYCLES
  );
  const weighting = options.weighting ?? 'cressie';
  validateRequirement(
    'fitSpaceTimeVariogramModel',
    weighting === 'cressie' || weighting === 'pairs' || weighting === 'none',
    `weighting must be one of cressie, pairs, none; received ${String(weighting)}`
  );
  const structure = options.structure ?? 'exponential';
  const bins: FitBins = {distances: [], timeLags: [], semivariances: [], pairCounts: []};
  const length = input.distances.length;
  if (
    input.timeLags.length !== length ||
    input.semivariances.length !== length ||
    input.pairCounts.length !== length
  ) {
    throw new Error(
      `fitSpaceTimeVariogramModel distances, timeLags, semivariances and pairCounts must have equal lengths, got ${length}, ${input.timeLags.length}, ${input.semivariances.length} and ${input.pairCounts.length}`
    );
  }
  for (let bin = 0; bin < length; bin++) {
    if (
      Number.isFinite(input.distances[bin]) &&
      Number.isFinite(input.timeLags[bin]) &&
      Number.isFinite(input.semivariances[bin]) &&
      input.pairCounts[bin] > 0 &&
      (input.distances[bin] > 0 || input.timeLags[bin] > 0)
    ) {
      bins.distances.push(input.distances[bin]);
      bins.timeLags.push(input.timeLags[bin]);
      bins.semivariances.push(input.semivariances[bin]);
      bins.pairCounts.push(input.pairCounts[bin]);
    }
  }
  if (bins.distances.length < 2) {
    throw new Error('Space-time variogram fitting needs at least two non-empty bins');
  }
  const specification = getSearchSpecification(bins, options.type, structure, options.gneiting);
  let weights = Float64Array.from(bins.pairCounts, count => (weighting === 'none' ? 1 : count));
  let fit = searchCoordinates(
    specification.parameters,
    specification.start,
    values =>
      evaluateLinearDesign(bins, weights, specification.columns, (bin, row) =>
        specification.writeDesign(values, bin, row)
      ),
    maximumCycles
  );
  // The exact zero field needs no reweighting: a vanishing fitted ordinate would produce
  // infinite Cressie weights and destroy the already exact zero objective.
  if (weighting === 'cressie' && !bins.semivariances.every(value => value === 0)) {
    for (let pass = 0; pass < CRESSIE_PASSES; pass++) {
      const previous = specification.getModel(fit.values, fit.evaluation.coefficients);
      weights = Float64Array.from(bins.pairCounts, (count, bin) => {
        const fitted = evaluateSpaceTimeVariogramModel(
          previous,
          bins.distances[bin],
          bins.timeLags[bin]
        );
        return count / Math.max(fitted * fitted, Number.MIN_VALUE);
      });
      fit = searchCoordinates(
        specification.parameters,
        fit.values,
        values =>
          evaluateLinearDesign(bins, weights, specification.columns, (bin, row) =>
            specification.writeDesign(values, bin, row)
          ),
        maximumCycles
      );
    }
  }
  return {
    model: specification.getModel(fit.values, fit.evaluation.coefficients),
    residual: fit.evaluation.residual,
    atBoundary: fit.atBoundary,
    cycleCount: fit.cycleCount,
    converged: fit.converged
  };
}

type SearchSpecification = {
  parameters: SearchParameter[];
  start: number[];
  columns: number;
  writeDesign: (values: readonly number[], bin: number, row: Float64Array) => void;
  getModel: (values: readonly number[], coefficients: Float64Array) => SpaceTimeVariogramModel;
};

function logBounds(positiveValues: readonly number[]): {low: number; high: number; start: number} {
  const minimum = Math.min(...positiveValues);
  const maximum = Math.max(...positiveValues);
  return {low: minimum / 10, high: maximum * 3, start: Math.sqrt(minimum * maximum)};
}

function fitMarginRange(
  distances: readonly number[],
  semivariances: readonly number[],
  pairCounts: readonly number[],
  structure: VariogramModelType,
  fallback: number
): number {
  if (distances.length < 2) {
    return fallback;
  }
  const fit = fitVariogramModel({distances, semivariances, pairCounts}, {model: structure});
  return fit.sill > 0 && fit.range > 0 ? fit.range : fallback;
}

/** Normalized shape of a structure with fraction `nuggetFraction` of its total as nugget. */
function getStandardizedStructure(
  structure: VariogramModelType,
  nuggetFraction: number,
  lag: number,
  range: number
): number {
  return lag <= 0
    ? 0
    : nuggetFraction + (1 - nuggetFraction) * getVariogramModelShape(structure, lag / range);
}

function getSearchSpecification(
  bins: FitBins,
  type: SpaceTimeVariogramModelType,
  structure: VariogramModelType,
  gneiting: SpaceTimeVariogramFitOptions['gneiting']
): SearchSpecification {
  const positiveDistances = bins.distances.filter(distance => distance > 0);
  const positiveTimes = bins.timeLags.filter(timeLag => timeLag > 0);
  if (!positiveDistances.length || !positiveTimes.length) {
    throw new Error('Space-time variogram fitting needs positive spatial and temporal lags');
  }
  const spatial = logBounds(positiveDistances);
  const temporal = logBounds(positiveTimes);
  const marginSelection = (keep: (bin: number) => boolean, lags: readonly number[]) => {
    const indices = bins.distances.map((_, bin) => bin).filter(keep);
    return {
      lags: indices.map(bin => lags[bin]),
      semivariances: indices.map(bin => bins.semivariances[bin]),
      pairCounts: indices.map(bin => bins.pairCounts[bin])
    };
  };
  const spatialMargin = marginSelection(bin => bins.timeLags[bin] === 0, bins.distances);
  const temporalMargin = marginSelection(bin => bins.distances[bin] === 0, bins.timeLags);
  const spatialStart = fitMarginRange(
    spatialMargin.lags,
    spatialMargin.semivariances,
    spatialMargin.pairCounts,
    structure,
    spatial.start
  );
  const temporalStart = fitMarginRange(
    temporalMargin.lags,
    temporalMargin.semivariances,
    temporalMargin.pairCounts,
    structure,
    temporal.start
  );
  const log = (low: number, high: number): SearchParameter => ({
    low,
    high,
    scale: 'log',
    steps: RANGE_STEPS
  });
  const fraction = (): SearchParameter => ({
    low: 0,
    high: MAXIMUM_FRACTION,
    scale: 'linear',
    steps: FRACTION_STEPS
  });
  const anisotropyLow = spatial.low / temporal.high;
  const anisotropyHigh = spatial.high / temporal.low;
  const anisotropy = log(anisotropyLow, anisotropyHigh);
  const clampToBounds = (value: number, parameter: SearchParameter): number =>
    Math.min(Math.max(value, parameter.low), parameter.high);
  const spatialParameter = log(spatial.low, spatial.high);
  const temporalParameter = log(temporal.low, temporal.high);
  const anisotropyStart = clampToBounds(spatialStart / temporalStart, anisotropy);
  const metricLag = (bin: number, anisotropyValue: number): number =>
    Math.hypot(bins.distances[bin], anisotropyValue * bins.timeLags[bin]);
  const jointParameter = log(spatial.low, Math.max(...bins.distances) * 3 + anisotropyHigh);
  const makeStructure = (
    nugget: number,
    sill: number,
    range: number
  ): SpaceTimeVariogramStructure => ({
    model: structure,
    nugget,
    sill,
    range
  });

  switch (type) {
    case 'separable':
      return {
        parameters: [spatialParameter, temporalParameter, fraction(), fraction()],
        start: [spatialStart, temporalStart, 0, 0],
        columns: 1,
        writeDesign: ([spatialRange, temporalRange, spatialNugget, temporalNugget], bin, row) => {
          const gs = getStandardizedStructure(
            structure,
            spatialNugget,
            bins.distances[bin],
            spatialRange
          );
          const gt = getStandardizedStructure(
            structure,
            temporalNugget,
            bins.timeLags[bin],
            temporalRange
          );
          row[0] = gs + gt - gs * gt;
        },
        getModel: ([spatialRange, temporalRange, spatialNugget, temporalNugget], [sill]) => ({
          type: 'separable',
          sill,
          spatial: makeStructure(spatialNugget, 1 - spatialNugget, spatialRange),
          temporal: makeStructure(temporalNugget, 1 - temporalNugget, temporalRange)
        })
      };
    case 'product-sum':
      return {
        parameters: [spatialParameter, temporalParameter, fraction(), fraction()],
        start: [spatialStart, temporalStart, 0, 0],
        columns: 3,
        writeDesign: ([spatialRange, temporalRange, spatialNugget, temporalNugget], bin, row) => {
          const gs = getStandardizedStructure(
            structure,
            spatialNugget,
            bins.distances[bin],
            spatialRange
          );
          const gt = getStandardizedStructure(
            structure,
            temporalNugget,
            bins.timeLags[bin],
            temporalRange
          );
          row[0] = gs;
          row[1] = gt;
          row[2] = gs + gt - gs * gt;
        },
        getModel: (
          [spatialRange, temporalRange, spatialNugget, temporalNugget],
          [spatialTotal, temporalTotal, interaction]
        ) => ({
          type: 'product-sum',
          k:
            spatialTotal > 0 && temporalTotal > 0
              ? interaction / (spatialTotal * temporalTotal)
              : 0,
          spatial: makeStructure(
            spatialNugget * spatialTotal,
            (1 - spatialNugget) * spatialTotal,
            spatialRange
          ),
          temporal: makeStructure(
            temporalNugget * temporalTotal,
            (1 - temporalNugget) * temporalTotal,
            temporalRange
          )
        })
      };
    case 'metric':
      return {
        parameters: [jointParameter, anisotropy],
        start: [clampToBounds(spatialStart, jointParameter), anisotropyStart],
        columns: 2,
        writeDesign: ([range, anisotropyValue], bin, row) => {
          row[0] = 1;
          row[1] = getVariogramModelShape(structure, metricLag(bin, anisotropyValue) / range);
        },
        getModel: ([range, anisotropyValue], [nugget, sill]) => ({
          type: 'metric',
          joint: makeStructure(nugget, sill, range),
          anisotropy: anisotropyValue
        })
      };
    case 'sum-metric':
      return {
        parameters: [spatialParameter, temporalParameter, jointParameter, anisotropy],
        start: [
          spatialStart,
          temporalStart,
          clampToBounds(spatialStart, jointParameter),
          anisotropyStart
        ],
        columns: 6,
        writeDesign: ([spatialRange, temporalRange, jointRange, anisotropyValue], bin, row) => {
          const distance = bins.distances[bin];
          const timeLag = bins.timeLags[bin];
          row[0] = distance > 0 ? 1 : 0;
          row[1] = distance > 0 ? getVariogramModelShape(structure, distance / spatialRange) : 0;
          row[2] = timeLag > 0 ? 1 : 0;
          row[3] = timeLag > 0 ? getVariogramModelShape(structure, timeLag / temporalRange) : 0;
          row[4] = 1;
          row[5] = getVariogramModelShape(structure, metricLag(bin, anisotropyValue) / jointRange);
        },
        getModel: (
          [spatialRange, temporalRange, jointRange, anisotropyValue],
          [spatialNugget, spatialSill, temporalNugget, temporalSill, jointNugget, jointSill]
        ) => ({
          type: 'sum-metric',
          spatial: makeStructure(spatialNugget, spatialSill, spatialRange),
          temporal: makeStructure(temporalNugget, temporalSill, temporalRange),
          joint: makeStructure(jointNugget, jointSill, jointRange),
          anisotropy: anisotropyValue
        })
      };
    case 'gneiting': {
      const alpha = gneiting?.alpha ?? 0.5;
      const gamma = gneiting?.gamma ?? 0.5;
      const tau = gneiting?.tau ?? 1;
      return {
        parameters: [
          spatialParameter,
          temporalParameter,
          {low: 0, high: 1, scale: 'linear', steps: 11}
        ],
        start: [spatialStart, temporalStart, 0.5],
        columns: 2,
        writeDesign: ([spatialRange, temporalRange, beta], bin, row) => {
          row[0] = 1;
          row[1] =
            1 -
            evaluateGneitingCorrelation(
              {
                type: 'gneiting',
                nugget: 0,
                sill: 1,
                spatialRange,
                temporalRange,
                alpha,
                gamma,
                beta,
                tau
              },
              bins.distances[bin],
              bins.timeLags[bin]
            );
        },
        getModel: ([spatialRange, temporalRange, beta], [nugget, sill]) => ({
          type: 'gneiting',
          nugget,
          sill,
          spatialRange,
          temporalRange,
          alpha,
          gamma,
          beta,
          tau
        })
      };
    }
    default:
      throw new Error(`Unknown space-time variogram model ${String(type)}`);
  }
}
