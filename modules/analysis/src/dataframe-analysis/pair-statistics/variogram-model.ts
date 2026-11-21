// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Variogram model families supported by {@link fitVariogramModel}. */
export type VariogramModelType =
  | 'spherical'
  | 'exponential'
  | 'gaussian'
  | 'matern-3/2'
  | 'matern-5/2';

/**
 * Argument factor `a = factor * h / range` of the closed-form Matern 3/2 correlation
 * `(1 + a) exp(-a)`, so that it equals `exp(-3)` at `h = range`, the exponential and Gaussian
 * convention. Solved in double precision from `a - ln(1 + a) = 3`.
 *
 * @internal
 */
export const MATERN_3_2_RANGE_FACTOR = 4.749031386012701;

/**
 * Argument factor of the closed-form Matern 5/2 correlation `(1 + a + a^2 / 3) exp(-a)`, so that
 * it equals `exp(-3)` at `h = range`. Solved in double precision from
 * `a - ln(1 + a + a^2 / 3) = 3`.
 *
 * @internal
 */
export const MATERN_5_2_RANGE_FACTOR = 5.924462618951797;

/**
 * Least-squares weighting of {@link fitVariogramModel}:
 * - `'cressie'`: `N_h / gamma_model(h)^2` (Cressie 1985), iterated three times.
 * - `'pairs'`: `N_h`.
 * - `'none'`: ordinary least squares.
 */
export type VariogramModelWeighting = 'cressie' | 'pairs' | 'none';

/**
 * A fitted variogram model `gamma(h) = nugget + sill * shape(h / range)` for `h > 0` and
 * `gamma(0) = 0`.
 */
export type VariogramModel = {
  /** Model family. */
  model: VariogramModelType;
  /** Nugget `c0 >= 0`, the discontinuity at the origin. */
  nugget: number;
  /** Partial sill `c >= 0`; the total sill is `nugget + sill`. */
  sill: number;
  /**
   * Effective (practical) range: the spherical model reaches the sill at `range`, the exponential,
   * Gaussian and Matern models reach 95% of it there (gstat ranges `range / 3` and
   * `range / sqrt(3)`).
   */
  range: number;
  /** Weighted sum of squared residuals at the fitted parameters. */
  residual: number;
};

/** Empirical variogram bins, for example read back from `GPUVariogram`. */
export type VariogramModelInput = {
  /** Representative lag per bin, usually the mean pair distance. */
  distances: ArrayLike<number>;
  /** Empirical semivariance per bin. */
  semivariances: ArrayLike<number>;
  /** Pair count per bin. Bins with no pairs or non-finite values are skipped. */
  pairCounts: ArrayLike<number>;
};

/** Options of {@link fitVariogramModel}. */
export type VariogramModelOptions = {
  /** Model family. */
  model: VariogramModelType;
  /** Least-squares weighting. Defaults to `'cressie'`. */
  weighting?: VariogramModelWeighting;
  /** Log-range ladder length, 3–1024. Defaults to 200. */
  rangeSteps?: number;
  /** Golden-section refinements per weighting pass, 0–64. Defaults to 60 on CPU. */
  maximumIterations?: number;
};

const RANGE_LADDER_LENGTH = 200;
const GOLDEN_SECTION_ITERATIONS = 60;
/** Match the GPU's bounded per-model candidate table. */
const MAXIMUM_RANGE_STEPS = 1024;
/** Match the ordered cooperative-refinement budget accepted by the GPU contributor. */
const MAXIMUM_ITERATIONS = 64;
const CRESSIE_PASSES = 3;

/** CPU fit with the measured final-pass search diagnostics. */
export type VariogramModelFitDiagnostics = {
  model: VariogramModel;
  /** Width of the final bracket in log(range), as in the operation twin. */
  logRangeBracket: number;
  /** Best ladder candidate was at a search-domain endpoint. */
  atBoundary: boolean;
  /** Interior optimum with final log-range bracket at most 1e-5. */
  converged: boolean;
};

/** @internal Validate explicit CPU search budgets, including zero, before fitting. */
export function getVariogramModelSearchOptions(
  options: Pick<VariogramModelOptions, 'rangeSteps' | 'maximumIterations'>,
  id: string = 'fitVariogramModel'
): {rangeSteps: number; maximumIterations: number} {
  const rangeSteps = options.rangeSteps ?? RANGE_LADDER_LENGTH;
  const maximumIterations = options.maximumIterations ?? GOLDEN_SECTION_ITERATIONS;
  validateCapacity(id, 'rangeSteps', rangeSteps, 3, MAXIMUM_RANGE_STEPS);
  validateCapacity(id, 'maximumIterations', maximumIterations, 0, MAXIMUM_ITERATIONS);
  return {rangeSteps, maximumIterations};
}

/** Below 0.5 the direct Matérn correlations cancel; the positive exponential remainder is stable. */
const MATERN_CPU_SERIES_ARGUMENT_LIMIT = 0.5;
/** At argument 0.5 the first omitted remainder term, a^17 / 19!, is below 7e-23. */
const MATERN_CPU_SERIES_MAXIMUM_ORDER = 18;

/**
 * e^-a * a^2 * (sum_{order=2}^infinity a^(order-2)/order! - quadraticCoefficient).
 * The coefficient is 0 for Matérn 3/2 and 1/3 for 5/2. The remainder starts at 1/2,
 * so neither subtraction loses the quadratic leading term of the semivariogram.
 */
function getMaternSmallArgumentShape(argument: number, quadraticCoefficient: number): number {
  let term = 0.5;
  let remainder = term;
  for (let order = 3; order <= MATERN_CPU_SERIES_MAXIMUM_ORDER; order++) {
    term *= argument / order;
    remainder += term;
  }
  return argument * argument * Math.exp(-argument) * (remainder - quadraticCoefficient);
}

/**
 * Returns the normalized shape `shape(h / range)` in `[0, 1]` of a model family.
 * Exponential and Gaussian shapes use expm1 to retain small lag/range values without
 * subtracting a correlation rounded near one. Matérn shapes use a positive exponential
 * remainder series below argument 0.5 to preserve their quadratic leading term.
 *
 * @param model Model family.
 * @param scaledDistance `h / range`, non-negative.
 */
export function getVariogramModelShape(model: VariogramModelType, scaledDistance: number): number {
  switch (model) {
    case 'spherical':
      return scaledDistance >= 1 ? 1 : 1.5 * scaledDistance - 0.5 * scaledDistance ** 3;
    case 'exponential':
      return -Math.expm1(-3 * scaledDistance);
    case 'gaussian':
      return -Math.expm1(-3 * scaledDistance * scaledDistance);
    case 'matern-3/2': {
      const argument = MATERN_3_2_RANGE_FACTOR * scaledDistance;
      if (argument < MATERN_CPU_SERIES_ARGUMENT_LIMIT)
        return getMaternSmallArgumentShape(argument, 0);
      return 1 - (1 + argument) * Math.exp(-argument);
    }
    case 'matern-5/2': {
      const argument = MATERN_5_2_RANGE_FACTOR * scaledDistance;
      if (argument < MATERN_CPU_SERIES_ARGUMENT_LIMIT)
        return getMaternSmallArgumentShape(argument, 1 / 3);
      return 1 - (1 + argument + (argument * argument) / 3) * Math.exp(-argument);
    }
    default:
      throw new Error(`Unknown variogram model ${String(model)}`);
  }
}

/**
 * WGSL mirror of {@link getVariogramModelShape} in float32, the one semivariogram-form definition
 * of the model shapes for GPU kernels (`GPUKriging`, `GPUVariogramModelFit`):
 * `getVariogramModelShape(modelCode: u32, scaled: f32) -> f32` with model codes 0 spherical,
 * 1 exponential, 2 Gaussian, 3 Matern 3/2, and anything else Matern 5/2.
 *
 * The shape is `1 - correlation`. For a lag that is a small fraction of the range the
 * correlation is within a few float32 steps of 1 and the difference cancels (at `3 * scaled =
 * 3e-8` the direct `1 - exp(-x)` is 0 or 6e-8; the Gaussian shape at 1e-4 of the range comes out
 * twice too large). Below an argument of 0.1 every exponential family therefore switches to its
 * alternating series, whose first omitted term is below 1e-7 of the result. The covariance-form
 * counterpart (correlation, not shape) is `getPredictionCovarianceWGSL` in
 * `gpu-spatial-analysis/spatial-interpolation/prediction-covariance.ts`; it needs the series only
 * for the Matern products and must not be derived from this one, because `1 - shape` would lose
 * the small correlations of long lags.
 *
 * @internal
 */
export const VARIOGRAM_MODEL_SHAPE_WGSL = /* wgsl */ `
// 1 - exp(-argument) for argument >= 0, by the series
// argument - argument^2/2 + argument^3/6 - argument^4/24 + argument^5/120 below 0.1.
fn getVariogramExponentialShape(argument: f32) -> f32 {
  if (argument < 0.1) {
    return argument * (1.0 - argument * (0.5 - argument * ((1.0 / 6.0) - argument * ((1.0 / 24.0) - argument * (1.0 / 120.0)))));
  }
  return 1.0 - exp(-argument);
}
fn getVariogramModelShape(modelCode: u32, scaled: f32) -> f32 {
  if (modelCode == 0u) {
    return select(1.5 * scaled - 0.5 * scaled * scaled * scaled, 1.0, scaled >= 1.0);
  }
  if (modelCode == 1u) { return getVariogramExponentialShape(3.0 * scaled); }
  if (modelCode == 2u) { return getVariogramExponentialShape(3.0 * scaled * scaled); }
  if (modelCode == 3u) {
    // 1 - (1 + a) exp(-a) = a^2/2 - a^3/3 + a^4/8 - a^5/30 + a^6/144 - ...
    let argument = ${MATERN_3_2_RANGE_FACTOR} * scaled;
    if (argument < 0.1) {
      return argument * argument * (0.5 - argument * ((1.0 / 3.0) - argument * (0.125 - argument * ((1.0 / 30.0) - argument * (1.0 / 144.0)))));
    }
    return 1.0 - (1.0 + argument) * exp(-argument);
  }
  // 1 - (1 + a + a^2/3) exp(-a) = a^2/6 - a^4/24 + a^5/45 - a^6/144 + ...
  let argument = ${MATERN_5_2_RANGE_FACTOR} * scaled;
  if (argument < 0.1) {
    return argument * argument * ((1.0 / 6.0) - argument * argument * ((1.0 / 24.0) - argument * ((1.0 / 45.0) - argument * (1.0 / 144.0))));
  }
  return 1.0 - (1.0 + argument + argument * argument / 3.0) * exp(-argument);
}
`;

/**
 * Evaluates a fitted variogram model at lag `distance` (`0` at the origin).
 *
 * @param model Fitted model.
 * @param distance Non-negative lag.
 */
export function evaluateVariogramModel(
  model: Pick<VariogramModel, 'model' | 'nugget' | 'sill' | 'range'>,
  distance: number
): number {
  if (distance <= 0) {
    return 0;
  }
  return model.nugget + model.sill * getVariogramModelShape(model.model, distance / model.range);
}

/**
 * Fits a spherical, exponential, Gaussian or Matern model to empirical variogram bins by weighted least
 * squares, on the CPU.
 *
 * For each range on a log ladder (200 points by default) over `[minLag / 10, 3 * maxLag]`, the
 * nugget and partial sill are the closed-form non-negative weighted linear least-squares solution,
 * and the best range is refined by golden-section search between its ladder neighbors. Cressie
 * weighting repeats the search three times with weights from the previous model. Every pass
 * honors explicit rangeSteps and maximumIterations, including zero; omitted CPU refinement
 * budgets retain 60 steps.
 *
 * @param input Empirical bins.
 * @param options Model family, weighting and search budgets.
 * @returns The fitted model.
 * @throws If the bin arrays differ in length, fewer than two bins are usable, the model or
 *   weighting is unknown, or a budget is invalid.
 */
export function fitVariogramModel(
  input: VariogramModelInput,
  options: VariogramModelOptions
): VariogramModel {
  return fitVariogramModelWithDiagnostics(input, options).model;
}

/** Fits a model and returns the measured final-pass bracket and boundary diagnostics. */
export function fitVariogramModelWithDiagnostics(
  input: VariogramModelInput,
  options: VariogramModelOptions
): VariogramModelFitDiagnostics {
  validateRequirement(
    'fitVariogramModel',
    !('diagnostics' in options),
    'diagnostics was removed. Use fitVariogramModelWithDiagnostics to return diagnostics'
  );
  const searchOptions = getVariogramModelSearchOptions(options);
  const {model} = options;
  const weighting = options.weighting ?? 'cressie';
  getVariogramModelShape(model, 0);
  if (weighting !== 'cressie' && weighting !== 'pairs' && weighting !== 'none') {
    throw new Error(`Unknown variogram weighting ${String(weighting)}`);
  }
  const length = input.distances.length;
  if (input.semivariances.length !== length || input.pairCounts.length !== length) {
    throw new Error(
      `fitVariogramModel distances, semivariances and pairCounts must have equal lengths, got ${length}, ${input.semivariances.length} and ${input.pairCounts.length}`
    );
  }
  const distances: number[] = [];
  const semivariances: number[] = [];
  const pairCounts: number[] = [];
  let minimumLag = Infinity;
  let maximumLag = 0;
  for (let bin = 0; bin < length; bin++) {
    const distance = input.distances[bin];
    const semivariance = input.semivariances[bin];
    const pairCount = input.pairCounts[bin];
    if (
      Number.isFinite(distance) &&
      Number.isFinite(semivariance) &&
      distance > 0 &&
      pairCount > 0
    ) {
      distances.push(distance);
      semivariances.push(semivariance);
      pairCounts.push(pairCount);
      minimumLag = Math.min(minimumLag, distance);
      maximumLag = Math.max(maximumLag, distance);
    }
  }
  if (distances.length < 2) {
    throw new Error('Variogram model fitting needs at least two non-empty bins');
  }
  const rangeSearch = {
    ...searchOptions,
    lowest: Math.log(minimumLag / 10),
    highest: Math.log(3 * maximumLag)
  };
  let weights = pairCounts.map(count => (weighting === 'none' ? 1 : count));
  let best = fitRange(model, distances, semivariances, weights, rangeSearch);
  // Zero observations already have the exact zero fit. Reweighting would divide pair counts
  // by Number.MIN_VALUE, overflow, and turn its zero objective into 0 * Infinity = NaN.
  if (weighting === 'cressie' && !semivariances.every(value => value === 0)) {
    for (let pass = 0; pass < CRESSIE_PASSES; pass++) {
      const previous = best.fit;
      weights = pairCounts.map((count, bin) => {
        const fitted = evaluateVariogramModel({...previous, model}, distances[bin]);
        return count / Math.max(fitted * fitted, Number.MIN_VALUE);
      });
      best = fitRange(model, distances, semivariances, weights, rangeSearch);
    }
  }
  return {
    model: {model, ...best.fit},
    logRangeBracket: best.logRangeBracket,
    atBoundary: best.atBoundary,
    converged: !best.atBoundary && best.logRangeBracket <= 1e-5
  };
}

type LinearFit = {nugget: number; sill: number; range: number; residual: number};

function fitRange(
  model: VariogramModelType,
  distances: number[],
  semivariances: number[],
  weights: number[],
  options: {rangeSteps: number; maximumIterations: number; lowest: number; highest: number}
): {fit: LinearFit; logRangeBracket: number; atBoundary: boolean} {
  const {lowest, highest} = options;
  const ladder: LinearFit[] = [];
  for (let step = 0; step < options.rangeSteps; step++) {
    const range = Math.exp(lowest + ((highest - lowest) * step) / (options.rangeSteps - 1));
    ladder.push(fitLinear(model, range, distances, semivariances, weights));
  }
  let bestStep = 0;
  for (let step = 1; step < ladder.length; step++) {
    if (ladder[step].residual < ladder[bestStep].residual) {
      bestStep = step;
    }
  }
  let left = Math.log(ladder[Math.max(bestStep - 1, 0)].range);
  let right = Math.log(ladder[Math.min(bestStep + 1, ladder.length - 1)].range);
  const ratio = (Math.sqrt(5) - 1) / 2;
  let inner = right - ratio * (right - left);
  let outer = left + ratio * (right - left);
  let innerFit = fitLinear(model, Math.exp(inner), distances, semivariances, weights);
  let outerFit = fitLinear(model, Math.exp(outer), distances, semivariances, weights);
  for (let iteration = 0; iteration < options.maximumIterations; iteration++) {
    if (innerFit.residual <= outerFit.residual) {
      right = outer;
      outer = inner;
      outerFit = innerFit;
      inner = right - ratio * (right - left);
      innerFit = fitLinear(model, Math.exp(inner), distances, semivariances, weights);
    } else {
      left = inner;
      inner = outer;
      innerFit = outerFit;
      outer = left + ratio * (right - left);
      outerFit = fitLinear(model, Math.exp(outer), distances, semivariances, weights);
    }
  }
  let best = ladder[bestStep];
  for (const candidate of [innerFit, outerFit]) {
    if (candidate.residual < best.residual) {
      best = candidate;
    }
  }
  return {
    fit: best,
    logRangeBracket: right - left,
    atBoundary: bestStep === 0 || bestStep === ladder.length - 1
  };
}

/** Non-negative weighted least squares of `gamma ~ nugget + sill * shape` for a fixed range. */
function fitLinear(
  model: VariogramModelType,
  range: number,
  distances: number[],
  semivariances: number[],
  weights: number[]
): LinearFit {
  let weightSum = 0;
  let shapeSum = 0;
  let shapeSquareSum = 0;
  let valueSum = 0;
  let crossSum = 0;
  const shapes = distances.map(distance => getVariogramModelShape(model, distance / range));
  for (let bin = 0; bin < distances.length; bin++) {
    const weight = weights[bin];
    const shape = shapes[bin];
    weightSum += weight;
    shapeSum += weight * shape;
    shapeSquareSum += weight * shape * shape;
    valueSum += weight * semivariances[bin];
    crossSum += weight * shape * semivariances[bin];
  }
  const determinant = weightSum * shapeSquareSum - shapeSum * shapeSum;
  const fullRank = determinant > 1e-300;
  if (fullRank) {
    const nugget = (shapeSquareSum * valueSum - shapeSum * crossSum) / determinant;
    const sill = (weightSum * crossSum - shapeSum * valueSum) / determinant;
    if (nugget >= 0 && sill >= 0) {
      return {
        nugget,
        sill,
        range,
        residual: getResidual(nugget, sill, shapes, semivariances, weights)
      };
    }
  }
  // Evaluate both one-parameter fits when the joint fit is singular or infeasible.
  const nugget = weightSum > 0 ? Math.max(valueSum / weightSum, 0) : 0;
  const sill = shapeSquareSum > 0 ? Math.max(crossSum / shapeSquareSum, 0) : 0;
  const nuggetResidual = getResidual(nugget, 0, shapes, semivariances, weights);
  const sillResidual = getResidual(0, sill, shapes, semivariances, weights);
  let useSill = sillResidual < nuggetResidual;
  if (fullRank && sillResidual === nuggetResidual) {
    // Keep the legacy sill-first rule on a true tie. Rounded residuals can hide a difference.
    useSill =
      !Number.isFinite(sillResidual) ||
      (nugget === 0 && sill === 0) ||
      doesSillFitMatchOrBeatNugget(shapes, semivariances, weights);
  }
  return useSill
    ? {nugget: 0, sill, range, residual: sillResidual}
    : {nugget, sill: 0, range, residual: nuggetResidual};
}

/** Compares the exact constrained one-parameter objectives on the stored binary64 inputs. */
function doesSillFitMatchOrBeatNugget(
  shapes: number[],
  semivariances: number[],
  weights: number[]
): boolean {
  const word = new DataView(new ArrayBuffer(8));
  const significandMask = (1n << 52n) - 1n;
  function toInteger(value: number): bigint {
    // All finite binary64 values are integer multiples of 2^-1074.
    word.setFloat64(0, value);
    const bits = word.getBigUint64(0);
    const exponent = (bits >> 52n) & 0x7ffn;
    const significand = bits & significandMask;
    const magnitude =
      exponent === 0n ? significand : (significand | (1n << 52n)) << (exponent - 1n);
    return bits >> 63n ? -magnitude : magnitude;
  }
  let weightSum = 0n;
  let valueSum = 0n;
  let shapeSquareSum = 0n;
  let crossSum = 0n;
  for (let bin = 0; bin < shapes.length; bin++) {
    const weight = toInteger(weights[bin]);
    const shape = toInteger(shapes[bin]);
    const weightedValue = weight * toInteger(semivariances[bin]);
    weightSum += weight;
    valueSum += weightedValue;
    shapeSquareSum += weight * shape * shape;
    crossSum += weightedValue * shape;
  }
  const nuggetProjection = valueSum > 0n ? valueSum : 0n;
  const sillProjection = crossSum > 0n ? crossSum : 0n;
  // Each fit subtracts its squared nonnegative projection divided by its basis norm
  // from the same total squared error. The common dyadic scale cancels.
  return (
    sillProjection * sillProjection * weightSum >=
    nuggetProjection * nuggetProjection * shapeSquareSum
  );
}

function getResidual(
  nugget: number,
  sill: number,
  shapes: number[],
  semivariances: number[],
  weights: number[]
): number {
  let residual = 0;
  for (let bin = 0; bin < shapes.length; bin++) {
    const difference = semivariances[bin] - nugget - sill * shapes[bin];
    residual += weights[bin] * difference * difference;
  }
  return residual;
}
