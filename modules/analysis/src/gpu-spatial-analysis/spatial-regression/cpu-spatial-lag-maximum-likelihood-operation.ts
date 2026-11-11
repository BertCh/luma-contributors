// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {GPU_REGRESSION_PRECISION_CONDITION_LIMIT} from '../../analysis-core/contracts/model-result';
import {
  factorCPURegressionCorrelation,
  fitCPURegressionWithQR,
  solveCPURegressionFactor
} from './cpu-regression-linear-algebra';
import {validateCPURegressionWeights, type CPURegressionWeights} from './cpu-regression-weights';
import {
  getCPUErrorConditioning,
  getCPUErrorDot,
  getCPUErrorMean,
  multiplyCPUErrorWeights
} from './cpu-spatial-error-regression';
import {getCPUSpatialErrorLikelihoodCandidates} from './cpu-spatial-error-likelihood-polynomial';
import {
  GPUSpatialLagMaximumLikelihoodOperation,
  type SpatialLagMaximumLikelihoodOperationProps
} from './gpu-spatial-lag-maximum-likelihood-operation';
import {getChiSquareSurvival} from './ordinary-least-squares-statistics';
import * as likelihood from './spatial-maximum-likelihood-parameters';
import {getSpatialExactLogDeterminant} from './spatial-exact-log-determinant';
import {getSpatialSeriesTailModel, SPATIAL_SERIES_TAIL_MINIMUM_POWER} from './spatial-series-tail';
import {
  GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT,
  GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT,
  GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH,
  getSpatialLogDeterminantFromTraces,
  getSpatialLogDeterminantTruncationBound,
  getSpatialWeightsTracesProbeValue
} from './spatial-weights-traces-parameters';

/** Packed float32 observations and whole CSR; traceParameters is the contributor's seed view. */
export type CPUSpatialLagMaximumLikelihoodInput = {
  weights: CPURegressionWeights;
  predictors: Float32Array;
  response: Float32Array;
  eigenvalues?: Float32Array;
  traceParameters?: Uint32Array;
};

/** Requested estimator, retaining the finite trace-series definition. GPU-only caps are dropped. */
export type CPUSpatialLagMaximumLikelihoodOptions = Pick<
  SpatialLagMaximumLikelihoodOperationProps,
  | 'predictorCount'
  | 'maximumPower'
  | 'traceMode'
  | 'traceProbeCount'
  | 'tailExtrapolation'
  | 'logDeterminant'
>;

/** Exactly sized outputs in the contributor's layouts; optional graph outputs are always returned. */
export type CPUSpatialLagMaximumLikelihoodResult = {
  table: Float32Array;
  summary: Float32Array;
  residuals: Float32Array;
  powerTraces: Float32Array;
  powerSums: Float32Array;
  infinityNorm: Float32Array;
  powerTraceStandardErrors: Float32Array;
  status: Uint32Array;
  converged: Uint32Array;
};

function validateOptions(id: string, options: CPUSpatialLagMaximumLikelihoodOptions): void {
  validateCapacity(id, 'predictorCount', options.predictorCount);
  validateCapacity(
    id,
    'maximumPower',
    options.maximumPower ?? likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_DEFAULT_MAXIMUM_POWER,
    options.tailExtrapolation
      ? SPATIAL_SERIES_TAIL_MINIMUM_POWER
      : likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_MINIMUM_POWER
  );
  validateCapacity(
    id,
    'traceProbeCount',
    options.traceProbeCount ?? GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT
  );
  validateRequirement(
    id,
    options.traceMode === undefined ||
      options.traceMode === 'exact' ||
      options.traceMode === 'stochastic',
    "traceMode must be 'exact' or 'stochastic'"
  );
  validateRequirement(
    id,
    options.logDeterminant === undefined ||
      options.logDeterminant === 'series' ||
      options.logDeterminant === 'exact',
    "logDeterminant must be 'series' or 'exact'"
  );
}

// Certify exact dependencies of the original float32 data: QR can leave positive roundoff
// at a genuine likelihood pole. Every float32 times 2^149 is an exact binary64 integer.
function certifyResidualDependency(
  input: CPUSpatialLagMaximumLikelihoodInput,
  predictorCount: number
): {pole?: number; constantLag: boolean} | undefined {
  const scale = 1n << 149n;
  const getInteger = (value: number) => BigInt(value * 2 ** 149);
  const response = Array.from(input.response, getInteger);
  const width = predictorCount + 1;
  const working = response.map((value, row) => {
    let lagged = 0n;
    for (let slot = input.weights.offsets[row]; slot < input.weights.offsets[row + 1]; slot++) {
      const neighbor = input.weights.neighbors[slot];
      if (neighbor < response.length)
        lagged += getInteger(input.weights.weights[slot]) * response[neighbor];
    }
    // All columns have the common denominator 2^298, including exact products W*y.
    return [
      scale * scale,
      ...Array.from(
        {length: predictorCount},
        (_, column) => getInteger(input.predictors[row * predictorCount + column]) * scale
      ),
      value * scale,
      lagged
    ];
  });
  let previousPivot = 1n;
  // Rectangular fraction-free elimination, solely for an exact rank certificate.
  for (let column = 0; column <= width; column++) {
    const pivotRow = working.findIndex((row, index) => index >= column && row[column] !== 0n);
    if (pivotRow < 0) return column === width ? {pole: 0, constantLag: false} : undefined;
    [working[column], working[pivotRow]] = [working[pivotRow], working[column]];
    const pivot = working[column][column];
    for (let row = column + 1; row < working.length; row++) {
      for (let target = column + 1; target <= width + 1; target++) {
        working[row][target] =
          (working[row][target] * pivot - working[row][column] * working[column][target]) /
          previousPivot;
      }
      working[row][column] = 0n;
    }
    previousPivot = pivot;
  }
  if (working.slice(width + 1).some(row => row[width + 1] !== 0n)) return undefined;
  const numerator = working[width][width];
  const denominator = working[width][width + 1];
  if (denominator === 0n) return {constantLag: true};
  // Scale integer significands separately, avoiding Infinity/Infinity for large minors.
  const getMantissa = (value: bigint) => {
    const magnitude = value < 0n ? -value : value;
    const shift = Math.max(magnitude.toString(2).length - 53, 0);
    return {value: Number(magnitude >> BigInt(shift)) * (value < 0n ? -1 : 1), shift};
  };
  const first = getMantissa(numerator);
  const second = getMantissa(denominator);
  return {
    pole: (first.value / second.value) * 2 ** (first.shift - second.shift),
    constantLag: false
  };
}

/**
 * Fits y = rho W y + X b + e with centered float64 QR. The finite trace-series likelihood
 * (or the supplied eigenvalue likelihood) is maximized over the contributor's norm-scaled bound.
 * The information matrix, residuals, trace outputs and all 22 summary fields use that same fit.
 * Pseudo-R squared uses the contributor's residual moments, including NaN for zero fitted variance.
 *
 * What differs from the GPU contributor:
 * - Float64 centered QR, direct residuals, sparse chains and scaled information replace float32
 *   moments/Cholesky; outputs round once. Ordinary GPU comparisons use 5e-3 relative tolerance.
 *   QR rank uses 64 * Number.EPSILON, so some GPU status-1 designs succeed. Conditioning uses
 *   converged float64 singular values. Subnormals are retained.
 * - The existing integer-polynomial/Bernstein helper isolates all stationary maxima to binary64
 *   resolution, removing the 1024-interval scan and 32-step bisection caps. Equal objectives choose
 *   the lowest rho; an identically flat profile chooses the lower boundary. The local maximum
 *   count can exceed the GPU's. This exact search is intended for small reference-tier fits.
 * - GPU-only predictor, power and exact-row limits are dropped; tileRowCount has no numerical
 *   effect. Whole CSR is supported; optional chunked weights are omitted from the operation.
 * - The requested Q-term determinant/trace series, geometric tail model, public tail correction
 *   stopping rule and Q-term inverse series for prediction and variance are retained. The default
 *   mode threshold, Philox seed/probe order, sampling errors and approximation flag are retained.
 *   P-values use the public float64 chi-square survival helper, including its stopping rule.
 * - Stationarity is checked from unrounded float64 residuals with a binary64 spacing allowance.
 *   A finite rejected iterate is published with status 3 and converged 0; the GPU clears its table
 *   and summary in this branch, retaining only residuals. A zero-RSS pole has no finite maximizer
 *   and returns status 3 with NaN fit rows. S = 0 outside the bound and e1'e1 = 0 remain admissible.
 *   An exact integer rank certificate on the original float32 data distinguishes true poles
 *   from positive residual minima, even when the QR leaves a positive rounding residual.
 * - Non-finite data/eigenvalues/active weights return status 3 (too few rows takes precedence).
 *   Non-finite active weights give NaN trace outputs. Unused capacity and out-of-range neighbor
 *   sentinels are ignored; empty slot arrays are accepted by the pure function. Zero-link weights
 *   return status 3, while their trace infinityNorm is 1, matching the trace contributor fallback.
 *   Finite positive norms at/above 3e38 stay normalized in float64; the GPU trace pass falls back
 *   to 1 there. Norms above float32 range round to infinity in the infinityNorm output only.
 *
 * @throws If weights, predictors, response, eigenvalues or traceParameters arrays are inconsistent.
 */
export function computeSpatialLagMaximumLikelihoodOnCPU(
  input: CPUSpatialLagMaximumLikelihoodInput,
  options: CPUSpatialLagMaximumLikelihoodOptions
): CPUSpatialLagMaximumLikelihoodResult {
  const id = 'computeSpatialLagMaximumLikelihoodOnCPU';
  validateOptions(id, options);
  const rowCount = validateCPURegressionWeights(id, input.weights);
  const {predictorCount} = options;
  const width = predictorCount + 1;
  const maximumPower =
    options.maximumPower ?? likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_DEFAULT_MAXIMUM_POWER;
  const exact = options.logDeterminant === 'exact';
  if (input.response.length !== rowCount)
    throw new Error(`${id} response length must equal weights rowCount`);
  if (input.predictors.length !== rowCount * predictorCount)
    throw new Error(`${id} predictors length must equal response length * predictorCount`);
  if (exact !== (input.eigenvalues !== undefined))
    throw new Error(`${id} eigenvalues are required exactly when logDeterminant is exact`);
  if (input.eigenvalues && input.eigenvalues.length !== rowCount)
    throw new Error(`${id} eigenvalues length must equal response length`);
  if (
    input.traceParameters &&
    input.traceParameters.length < GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH
  )
    throw new Error(`${id} traceParameters must hold the packed seed`);
  const stochastic =
    (options.traceMode ??
      (rowCount <= GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT ? 'exact' : 'stochastic')) ===
    'stochastic';
  const probeCount = stochastic
    ? (options.traceProbeCount ?? GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT)
    : rowCount;
  const [seed = 0] = input.traceParameters ?? [];
  const result: CPUSpatialLagMaximumLikelihoodResult = {
    table: new Float32Array(
      (predictorCount + 2) * likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE
    ).fill(NaN),
    summary: new Float32Array(likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH).fill(NaN),
    residuals: new Float32Array(rowCount).fill(NaN),
    powerTraces: new Float32Array(maximumPower),
    powerSums: new Float32Array(maximumPower),
    infinityNorm: new Float32Array(1),
    powerTraceStandardErrors: new Float32Array(maximumPower),
    status: new Uint32Array(1),
    converged: new Uint32Array(1)
  };
  result.summary[likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_ROW_COUNT] = rowCount;
  const fail = (code: number) => {
    result.status[0] = code;
    return result;
  };
  let norm = 0;
  let finiteWeights = true;
  const rowSums = new Array<number>(rowCount).fill(0);
  for (let row = 0; row < rowCount; row++) {
    let absoluteSum = 0;
    for (let slot = input.weights.offsets[row]; slot < input.weights.offsets[row + 1]; slot++) {
      if (input.weights.neighbors[slot] >= rowCount) continue;
      const weight = input.weights.weights[slot];
      finiteWeights &&= Number.isFinite(weight);
      rowSums[row] += weight;
      absoluteSum += Math.abs(weight);
    }
    norm = Math.max(norm, absoluteSum);
  }
  const traceNorm = finiteWeights ? (norm > 0 ? norm : 1) : NaN;
  result.infinityNorm[0] = traceNorm;
  const multiply = (values: number[]) =>
    multiplyCPUErrorWeights(input.weights, values).map(value => value / traceNorm);
  const getProbe = (probe: number) =>
    Array.from({length: rowCount}, (_, row) =>
      stochastic ? getSpatialWeightsTracesProbeValue(seed, probe, row) : Number(row === probe)
    );
  // Keep traces in binary64 for fitting; the public trace twin returns already-rounded outputs.
  const traces = new Array<number>(maximumPower).fill(0);
  const centeredSquares = new Array<number>(maximumPower).fill(0);
  let ones = new Array<number>(rowCount).fill(1);
  for (let power = 0; power < maximumPower; power++) {
    ones = multiply(ones);
    result.powerSums[power] = ones.reduce((sum, value) => sum + value, 0);
  }
  for (let probe = 0; probe < probeCount; probe++) {
    const initial = getProbe(probe);
    let vector = initial;
    for (let power = 0; power < maximumPower; power++) {
      vector = multiply(vector);
      const value = stochastic ? getCPUErrorDot(initial, vector) : vector[probe];
      if (stochastic) {
        const difference = value - traces[power];
        traces[power] += difference / (probe + 1);
        centeredSquares[power] += difference * (value - traces[power]);
      } else traces[power] += value;
    }
  }
  result.powerTraces.set(traces);
  result.powerTraceStandardErrors.set(
    centeredSquares.map(value =>
      stochastic
        ? probeCount < 2
          ? NaN
          : Math.sqrt(Math.max(value, 0) / (probeCount - 1) / probeCount)
        : 0
    )
  );
  if (!finiteWeights) {
    result.powerTraces.fill(NaN);
    result.powerSums.fill(NaN);
    result.powerTraceStandardErrors.fill(NaN);
  }
  if (rowCount <= predictorCount + 2)
    return fail(likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_TOO_FEW_ROWS);
  if (
    !finiteWeights ||
    !Array.from(input.predictors).every(Number.isFinite) ||
    !Array.from(input.response).every(Number.isFinite) ||
    (input.eigenvalues && !Array.from(input.eigenvalues).every(Number.isFinite))
  )
    return fail(likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_NON_FINITE);
  const means = Array.from({length: predictorCount}, (_, column) =>
    getCPUErrorMean(
      Array.from({length: rowCount}, (_, row) => input.predictors[row * predictorCount + column])
    )
  );
  const responseMean = getCPUErrorMean(Array.from(input.response));
  const design = Array.from({length: rowCount}, (_, row) => [
    1,
    ...means.map((mean, column) => input.predictors[row * predictorCount + column] - mean)
  ]);
  const response = Array.from(input.response, value => value - responseMean);
  // Same mean-shift identity as lagged-response-shift, in the normalized parameter t = rho*s.
  const rowSumReference = rowSums.reduce((largest, value) => Math.max(largest, value), -Infinity);
  const lagged = multiply(response).map(
    (value, row) => value + responseMean * ((rowSums[row] - rowSumReference) / traceNorm)
  );
  const first = fitCPURegressionWithQR(design, response);
  const second = fitCPURegressionWithQR(design, lagged);
  if (!first || !second) return fail(likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_SINGULAR);
  if (!(norm > 0) || !Number.isFinite(norm) || !traces.every(Number.isFinite))
    return fail(likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_NON_FINITE);
  const residualZero = response.map(
    (value, row) => value - getCPUErrorDot(design[row], first.estimate)
  );
  const residualOne = lagged.map(
    (value, row) => value - getCPUErrorDot(design[row], second.estimate)
  );
  const dependency = certifyResidualDependency(input, predictorCount);
  if (dependency?.constantLag) residualOne.fill(0);
  const initialSquares = getCPUErrorDot(residualZero, residualZero);
  const laggedSquares = getCPUErrorDot(residualOne, residualOne);
  if (!(initialSquares > 0))
    return fail(likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_NON_FINITE);
  const limit =
    exact || options.tailExtrapolation
      ? likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BOUND_FRACTION
      : likelihood.getSpatialMaximumLikelihoodSeriesLimit(maximumPower);
  if (dependency?.pole !== undefined && Math.abs(dependency.pole * norm) <= limit)
    return fail(likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_NON_FINITE);
  const eigenvalues = input.eigenvalues
    ? Array.from(input.eigenvalues, value => value / norm)
    : undefined;
  const tail = !exact && options.tailExtrapolation ? getSpatialSeriesTailModel(traces) : undefined;
  const logDeterminant = (parameter: number) =>
    eigenvalues
      ? getSpatialExactLogDeterminant(parameter, eigenvalues)
      : getSpatialLogDeterminantFromTraces(parameter, traces, 1, options.tailExtrapolation);
  // Paired signs make the constant column exactly orthogonal in the integer Gram determinant.
  // Thus N/D is exactly sum(e0-t*e1)^2 up to an irrelevant factor, with no rounded moments.
  const candidateRows = residualZero.flatMap(value => [
    [1, value],
    [1, -value]
  ]);
  const candidateLags = residualOne.flatMap(value => [
    [0, value],
    [0, -value]
  ]);
  const candidates = getCPUSpatialErrorLikelihoodCandidates(
    candidateRows,
    candidateLags,
    traces,
    eigenvalues,
    tail,
    rowCount,
    limit
  );
  let bestObjective = -Infinity;
  let parameter = NaN;
  let atBoundary = false;
  for (const candidate of candidates) {
    const errors = residualZero.map((value, row) => value - candidate.parameter * residualOne[row]);
    const objective =
      -0.5 * rowCount * Math.log(getCPUErrorDot(errors, errors) / rowCount) +
      logDeterminant(candidate.parameter);
    if (Number.isFinite(objective) && objective > bestObjective) {
      bestObjective = objective;
      parameter = candidate.parameter;
      atBoundary = candidate.atBoundary;
    }
  }
  if (!Number.isFinite(parameter))
    return fail(likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_NON_FINITE);
  const residuals = residualZero.map((value, row) => value - parameter * residualOne[row]);
  const residualSquares = getCPUErrorDot(residuals, residuals);
  const sigmaSquared = residualSquares / rowCount;
  const centeredCoefficients = first.estimate.map(
    (value, column) => value - parameter * second.estimate[column]
  );
  const lagShift = (responseMean * rowSumReference) / norm;
  const coefficients = centeredCoefficients.slice();
  coefficients[0] +=
    responseMean - parameter * lagShift - getCPUErrorDot(means, coefficients.slice(1));
  // Q terms, as specified by maximumPower; retain the unrounded centered fit throughout.
  const reduced = design.map(
    row => responseMean - parameter * lagShift + getCPUErrorDot(row, centeredCoefficients)
  );
  let spatialPrediction = reduced;
  for (let power = 1; power < maximumPower; power++)
    spatialPrediction = multiply(spatialPrediction).map(
      (value, row) => reduced[row] + parameter * value
    );
  const weightedPrediction = multiply(spatialPrediction);
  const weightedMean = getCPUErrorMean(weightedPrediction);
  const centeredWeighted = weightedPrediction.map(value => value - weightedMean);
  const traceOne = eigenvalues
    ? eigenvalues.reduce((sum, value) => sum + value / (1 - parameter * value), 0)
    : likelihood.getSpatialMaximumLikelihoodTraceOne(
        parameter,
        traces,
        1,
        options.tailExtrapolation
      );
  const traceTwo = eigenvalues
    ? eigenvalues.reduce((sum, value) => sum + (value / (1 - parameter * value)) ** 2, 0)
    : likelihood.getSpatialMaximumLikelihoodTraceTwo(
        parameter,
        traces,
        1,
        options.tailExtrapolation
      );
  let traceThree = 0;
  for (let probe = 0; probe < probeCount; probe++) {
    const initial = getProbe(probe);
    let vector = initial;
    for (let power = 1; power < maximumPower; power++)
      vector = multiply(vector).map((value, row) => initial[row] + parameter * value);
    const transformed = multiply(vector);
    traceThree += getCPUErrorDot(transformed, transformed);
  }
  if (stochastic) traceThree /= probeCount;
  const size = width + 1;
  const information = Array.from({length: size}, () => new Array<number>(size).fill(0));
  for (let row = 0; row < rowCount; row++) {
    for (let firstColumn = 0; firstColumn < width; firstColumn++) {
      for (let secondColumn = 0; secondColumn < width; secondColumn++)
        information[firstColumn][secondColumn] +=
          (design[row][firstColumn] * design[row][secondColumn]) / sigmaSquared;
      information[firstColumn][width] +=
        (design[row][firstColumn] * centeredWeighted[row]) / sigmaSquared;
    }
    information[width][width] += centeredWeighted[row] ** 2 / sigmaSquared;
  }
  information[width][width] += traceTwo + traceThree - (2 * traceOne ** 2) / rowCount;
  for (let column = 0; column < width; column++)
    information[width][column] = information[column][width];
  const scales = information.map((row, column) => Math.sqrt(row[column]));
  const factor = scales.every(value => value > 0 && Number.isFinite(value))
    ? factorCPURegressionCorrelation(
        information.map((row, firstColumn) =>
          row.map((value, secondColumn) => value / scales[firstColumn] / scales[secondColumn])
        )
      )
    : undefined;
  const standardErrors = new Array<number>(size).fill(NaN);
  if (factor) {
    // C^-1 L = L^-T gives a covariance root without differencing large variances.
    const inverseRoot = factor.map((_, column) =>
      solveCPURegressionFactor(
        factor,
        factor.map(row => row[column])
      ).map((value, row) => value / scales[row])
    );
    const shift = [1, ...means.map(value => -value), -weightedMean];
    standardErrors[0] = Math.hypot(...inverseRoot.map(column => getCPUErrorDot(column, shift)));
    for (let column = 1; column < size; column++)
      standardErrors[column] = Math.hypot(...inverseRoot.map(rootColumn => rootColumn[column]));
    standardErrors[width] /= norm;
  }
  coefficients.push(parameter / norm);
  coefficients.forEach((coefficient, row) => {
    const statistic = coefficient / standardErrors[row];
    result.table.set(
      [coefficient, standardErrors[row], statistic, getChiSquareSurvival(statistic * statistic, 1)],
      row * likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE
    );
  });
  result.residuals.set(residuals);
  const correlationSquared = (prediction: number[]) => {
    const mean = getCPUErrorMean(prediction);
    const centered = prediction.map(value => value - mean);
    const covariance = getCPUErrorDot(response, centered);
    return (
      (covariance / getCPUErrorDot(response, response)) *
      (covariance / getCPUErrorDot(centered, centered))
    );
  };
  // Preserve the contributor's moment definition: a constant fit has undefined correlation,
  // even if QR roundoff leaves tiny nonconstant reconstructed fitted rows.
  const responseSquares = getCPUErrorDot(response, response);
  const responseResidual = getCPUErrorDot(response, residuals);
  const residualTotal = residuals.reduce((sum, value) => sum + value, 0);
  const predictionCovariance = responseSquares - responseResidual;
  const predictionVariance =
    responseSquares -
    2 * responseResidual +
    sigmaSquared * rowCount -
    (residualTotal * residualTotal) / rowCount;
  const conditioning = getCPUErrorConditioning(design, means);
  const logLikelihood = bestObjective - 0.5 * rowCount * (Math.log(2 * Math.PI) + 1);
  const laggedRatio = getCPUErrorDot(residuals, residualOne) / residualSquares;
  const score = rowCount * laggedRatio - traceOne;
  const curvature = rowCount * (laggedSquares / residualSquares - 2 * laggedRatio ** 2) + traceTwo;
  const stationary =
    atBoundary ||
    (curvature > 0 &&
      Math.abs(score) <=
        Math.sqrt(0.25 * curvature) + curvature * 4 * Number.EPSILON * Math.abs(parameter));
  const converged = stationary && !atBoundary;
  result.status[0] = stationary ? 0 : likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_NON_FINITE;
  result.converged[0] = Number(converged);
  result.summary.set([
    rowCount,
    parameter / norm,
    sigmaSquared,
    logLikelihood,
    2 * size - 2 * logLikelihood,
    size * Math.log(rowCount) - 2 * logLikelihood,
    (predictionCovariance / responseSquares) * (predictionCovariance / predictionVariance),
    correlationSquared(spatialPrediction),
    residualSquares,
    ...conditioning,
    Number(conditioning[1] > GPU_REGRESSION_PRECISION_CONDITION_LIMIT),
    limit / norm,
    Number(stochastic),
    Number(converged),
    Number(atBoundary),
    Number(limit < likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BOUND_FRACTION),
    exact ? 0 : getSpatialLogDeterminantTruncationBound(parameter, maximumPower, rowCount),
    Number(Boolean(factor)),
    candidates.length,
    logDeterminant(parameter)
  ]);
  return result;
}

/** CPU whole-CSR twin, with GPU-only size/search caps dropped and identical graph values. */
export class CPUSpatialLagMaximumLikelihoodOperation extends GPUCPUOperation {
  constructor(props: SpatialLagMaximumLikelihoodOperationProps) {
    const id = props.id ?? 'spatial-lag-maximum-likelihood';
    validateOptions(id, props);
    validateCapacity(id, 'rowCount', props.rowCount);
    if (props.tileRowCount !== undefined) validateCapacity(id, 'tileRowCount', props.tileRowCount);
    validateRequirement(
      id,
      (props.logDeterminant === 'exact') === (props.eigenvalues !== undefined),
      'eigenvalues are required exactly when logDeterminant is exact'
    );
    validateRequirement(
      id,
      (props.output.powerTraces === undefined) === (props.output.powerSums === undefined) &&
        (props.output.powerTraces === undefined) === (props.output.infinityNorm === undefined),
      'output.powerTraces, output.powerSums and output.infinityNorm must be provided together'
    );
    const operation = new GPUSpatialLagMaximumLikelihoodOperation(props);
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries(
        Object.values(operation.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      execute({inputs}) {
        const response = getCPUOperationFloat32Rows(id, inputs, 'response');
        if (response.length !== props.rowCount)
          throw new Error(`${id} response length must equal rowCount`);
        const result = computeSpatialLagMaximumLikelihoodOnCPU(
          {
            weights: {
              offsets: getCPUOperationUint32Rows(id, inputs, 'offsets'),
              neighbors: getCPUOperationUint32Rows(id, inputs, 'neighbors'),
              weights: getCPUOperationFloat32Rows(id, inputs, 'weights')
            },
            predictors: getCPUOperationFloat32Rows(id, inputs, 'predictors'),
            response,
            eigenvalues:
              props.eigenvalues !== undefined
                ? getCPUOperationFloat32Rows(id, inputs, 'eigenvalues')
                : undefined,
            traceParameters:
              props.traceParameters !== undefined
                ? getCPUOperationUint32Rows(id, inputs, 'traceParameters')
                : undefined
          },
          props
        );
        return {
          [props.output.table]: result.table,
          [props.output.summary]: result.summary,
          [props.output.status.code]: result.status,
          ...(props.output.status.converged !== undefined
            ? {[props.output.status.converged]: result.converged}
            : {}),
          ...Object.fromEntries(
            (
              [
                'residuals',
                'powerTraces',
                'powerSums',
                'infinityNorm',
                'powerTraceStandardErrors'
              ] as const
            )
              .filter(port => props.output[port] !== undefined)
              .map(port => [props.output[port]!, result[port]])
          )
        };
      }
    });
  }
}
