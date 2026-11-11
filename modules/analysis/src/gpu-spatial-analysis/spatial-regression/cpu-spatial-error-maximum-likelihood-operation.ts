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
import {fitCPURegressionWithQR} from './cpu-regression-linear-algebra';
import {validateCPURegressionWeights, type CPURegressionWeights} from './cpu-regression-weights';
import {
  getCPUErrorConditioning,
  getCPUErrorDot,
  getCPUErrorMean
} from './cpu-spatial-error-regression';
import {
  getCPUSpatialErrorLikelihoodCandidates,
  getCPUSpatialErrorResponseRelation
} from './cpu-spatial-error-likelihood-polynomial';
import {
  GPUSpatialErrorMaximumLikelihoodOperation,
  type SpatialErrorMaximumLikelihoodOperationProps
} from './gpu-spatial-error-maximum-likelihood-operation';
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

/** Whole CSR and packed float32 data, with optional eigenvalues and packed Philox seed. */
export type CPUSpatialErrorMaximumLikelihoodInput = {
  weights: CPURegressionWeights;
  predictors: Float32Array;
  response: Float32Array;
  eigenvalues?: Float32Array;
  traceParameters?: Uint32Array;
};

/** Estimator settings; GPU-only predictor, power, exact-row and tile limits are dropped. */
export type CPUSpatialErrorMaximumLikelihoodOptions = Pick<
  SpatialErrorMaximumLikelihoodOperationProps,
  | 'predictorCount'
  | 'maximumPower'
  | 'traceMode'
  | 'traceProbeCount'
  | 'tailExtrapolation'
  | 'logDeterminant'
>;

/** Exactly sized scalar vector outputs, including optional-operation outputs unconditionally. */
export type CPUSpatialErrorMaximumLikelihoodResult = {
  table: Float32Array;
  summary: Float32Array;
  residuals: Float32Array;
  powerTraceStandardErrors: Float32Array;
  status: Uint32Array;
  converged: Uint32Array;
};

function validateOptions(id: string, options: CPUSpatialErrorMaximumLikelihoodOptions): void {
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

/**
 * Fits y = X b + u, u = lambda W u + e, using centered float64 QR and direct residuals.
 * The likelihood is the requested trace-series (including optional geometric tail) or supplied
 * eigenvalue likelihood. Its rational profile derivative is isolated without a sampling grid.
 *
 * What differs from the GPU contributor:
 * - Float64 centered QR, residuals and probe chains replace float32 moments/Cholesky; arrays round
 *   once at output. Well-conditioned GPU comparisons use 5e-3 relative tolerance. Residuals and
 *   pseudo R squared use the centered fit; squared correlation uses ratios, avoiding raw
 *   intercept cancellation and products of large sums of squares.
 *   Predictions come directly from the centered slopes; zero prediction variance gives NaN.
 *   Exact zero-moment certificates suppress QR noise for an orthogonal response.
 * - Integer polynomial determinants of the float64 filtered rows and Bernstein root isolation
 *   find every positive-to-negative stationary crossing to binary64 resolution, without the
 *   GPU's 1024-interval scan or 32-bisection limit. Equal objectives choose the lowest lambda;
 *   an identically flat profile chooses the lower boundary. localMaximumCount can exceed the GPU's.
 *   Weights are normalized before multiplication so a scalar identity retains its flat profile;
 *   the GPU's float32 search can choose another point on that profile.
 * - GPU-only predictor, power and exact-row caps are dropped; tileRowCount has no numerical effect.
 *   QR rank uses 64 * Number.EPSILON; some GPU status-1 fits succeed. Conditioning uses converged
 *   float64 singular values. Polynomial isolation is intended for small reference-tier fits.
 * - The requested finite trace series, geometric tail model, public tail correction and Q-term
 *   inverse series for the variance are retained, including on the eigenvalue path. The tail
 *   correction calls the public helper with its existing stopping rule. Stochastic mode retains
 *   the default mode threshold, Philox seed/probe order, standard errors and approximation flag.
 * - P-values call the public float64 chi-square survival helper (its iteration cap remains).
 *   Subnormals are retained. Non-finite data/eigenvalues give status 3 (status 2 takes precedence
 *   for too few rows); non-finite active weights make stochastic trace errors NaN (exact-mode
 *   trace errors remain zero). Inactive capacity and out-of-range neighbor sentinels are ignored.
 *   Empty slot arrays are accepted by the pure
 *   function. Whole CSR is supported; optional chunked weights are omitted from the operation.
 * - Finite float64 norms and moments are retained when the GPU's float32 norm/moments overflow;
 *   some GPU status-3 fits succeed here. Final float32 outputs may underflow or overflow.
 * - An exact augmented-Gram certificate rejects an exactly fitted response with status 3 and
 *   NaN fit outputs: zero variance has no finite Gaussian maximum. The GPU can instead publish
 *   a tiny positive variance created by rounding. No small nonzero residual is thresholded away.
 *
 * @throws If weights, predictors, response, eigenvalues or traceParameters arrays are inconsistent.
 */
export function computeSpatialErrorMaximumLikelihoodOnCPU(
  input: CPUSpatialErrorMaximumLikelihoodInput,
  options: CPUSpatialErrorMaximumLikelihoodOptions
): CPUSpatialErrorMaximumLikelihoodResult {
  const id = 'computeSpatialErrorMaximumLikelihoodOnCPU';
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
  const seed = input.traceParameters?.[0] ?? 0;
  const result: CPUSpatialErrorMaximumLikelihoodResult = {
    table: new Float32Array(
      (predictorCount + 2) * likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE
    ).fill(NaN),
    summary: new Float32Array(likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH).fill(NaN),
    residuals: new Float32Array(rowCount).fill(NaN),
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
  for (let row = 0; row < rowCount; row++) {
    let sum = 0;
    for (let slot = input.weights.offsets[row]; slot < input.weights.offsets[row + 1]; slot++)
      if (input.weights.neighbors[slot] < rowCount) sum += Math.abs(input.weights.weights[slot]);
    norm = Math.max(norm, sum);
  }
  // The shared raw-weight multiplier divides only after the sum. Normalizing each weight first
  // preserves W/s=I exactly for scalar identities, without rounding a normalized CSR to f32.
  const normalizedWeights = Array.from(
    input.weights.weights,
    value => value / (norm > 0 ? norm : 1)
  );
  const multiply = (values: number[]) =>
    Array.from({length: rowCount}, (_, row) => {
      let total = 0;
      for (let slot = input.weights.offsets[row]; slot < input.weights.offsets[row + 1]; slot++) {
        const neighbor = input.weights.neighbors[slot];
        if (neighbor < rowCount) total += normalizedWeights[slot] * values[neighbor];
      }
      return total;
    });
  const getProbe = (probe: number) =>
    Array.from({length: rowCount}, (_, row) =>
      stochastic ? getSpatialWeightsTracesProbeValue(seed, probe, row) : Number(row === probe)
    );
  const traces = new Array<number>(maximumPower).fill(0);
  const centeredSquares = new Array<number>(maximumPower).fill(0);
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
  result.powerTraceStandardErrors.set(
    centeredSquares.map(value =>
      stochastic
        ? probeCount < 2
          ? NaN
          : Math.sqrt(Math.max(value, 0) / (probeCount - 1) / probeCount)
        : 0
    )
  );
  if (rowCount <= width + 1)
    return fail(likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_TOO_FEW_ROWS);
  if (
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
  const initialFit = fitCPURegressionWithQR(design, response);
  if (!initialFit) return fail(likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_SINGULAR);
  // Certify the original f32 rows, before their float64 means/subtractions can round away a
  // tiny genuine residual or introduce one into an exact affine response.
  const rawDesign = Array.from({length: rowCount}, (_, row) => [
    1,
    ...input.predictors.subarray(row * predictorCount, (row + 1) * predictorCount)
  ]);
  if (getCPUSpatialErrorResponseRelation(rawDesign, Array.from(input.response)).exactFit)
    return fail(likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_NON_FINITE);
  const initialRelation = getCPUSpatialErrorResponseRelation(design, response);
  if (initialRelation.orthogonal) initialFit.estimate.fill(0);
  if (!(norm > 0) || !Number.isFinite(norm) || !traces.every(Number.isFinite))
    return fail(likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_NON_FINITE);
  const initialResiduals = response.map(
    (value, row) => value - getCPUErrorDot(design[row], initialFit.estimate)
  );
  if (!(getCPUErrorDot(initialResiduals, initialResiduals) > 0))
    return fail(likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_NON_FINITE);
  const augmented = design.map((row, position) => [...row, initialResiduals[position]]);
  const laggedColumns = augmented[0].map((_, column) =>
    multiply(augmented.map(row => row[column]))
  );
  const lagged = augmented.map((row, position) =>
    row.map((_, column) => laggedColumns[column][position])
  );
  const eigenvalues = input.eigenvalues
    ? Array.from(input.eigenvalues, value => value / norm)
    : undefined;
  const tail = !exact && options.tailExtrapolation ? getSpatialSeriesTailModel(traces) : undefined;
  const limit =
    exact || options.tailExtrapolation
      ? likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BOUND_FRACTION
      : likelihood.getSpatialMaximumLikelihoodSeriesLimit(maximumPower);
  const logDeterminant = (parameter: number) =>
    eigenvalues
      ? getSpatialExactLogDeterminant(parameter, eigenvalues)
      : getSpatialLogDeterminantFromTraces(parameter, traces, 1, options.tailExtrapolation);
  const candidates = getCPUSpatialErrorLikelihoodCandidates(
    augmented,
    lagged,
    traces,
    eigenvalues,
    tail,
    rowCount,
    limit
  );
  const evaluate = (parameter: number) => {
    const filtered = design.map((row, position) =>
      row.map((value, column) => value - parameter * lagged[position][column])
    );
    const filteredResponse = initialResiduals.map(
      (value, row) => value - parameter * lagged[row][width]
    );
    const fit = fitCPURegressionWithQR(filtered, filteredResponse);
    if (!fit) return undefined;
    const relation = getCPUSpatialErrorResponseRelation(filtered, filteredResponse);
    if (relation.exactFit) return undefined;
    if (relation.orthogonal) fit.estimate.fill(0);
    const errors = filteredResponse.map(
      (value, row) => value - getCPUErrorDot(filtered[row], fit.estimate)
    );
    const sigmaSquared = getCPUErrorDot(errors, errors) / rowCount;
    const objective = -0.5 * rowCount * Math.log(sigmaSquared) + logDeterminant(parameter);
    return {fit, filtered, sigmaSquared, objective};
  };
  let best: ReturnType<typeof evaluate>;
  let parameter = NaN;
  let atBoundary = false;
  for (const candidate of candidates) {
    const fit = evaluate(candidate.parameter);
    if (fit && Number.isFinite(fit.objective) && (!best || fit.objective > best.objective)) {
      best = fit;
      parameter = candidate.parameter;
      atBoundary = candidate.atBoundary;
    }
  }
  if (!best) return fail(likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_NON_FINITE);
  const {fit, filtered, sigmaSquared} = best;
  const centeredCoefficients = fit.estimate.map(
    (value, column) => value + initialFit.estimate[column]
  );
  const coefficients = centeredCoefficients.slice();
  coefficients[0] += responseMean - getCPUErrorDot(means, coefficients.slice(1));
  const residuals = initialResiduals.map(
    (value, row) => value - getCPUErrorDot(design[row], fit.estimate)
  );
  const inverseRoot = fit.inverseRoot.map(row => row.slice());
  for (let column = 0; column < predictorCount; column++) {
    for (let target = 0; target < width; target++)
      inverseRoot[0][target] -= means[column] * fit.inverseRoot[column + 1][target];
  }
  const standardErrors = inverseRoot.map(row => Math.sqrt(sigmaSquared) * Math.hypot(...row));
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
  const information = traceTwo + traceThree - (2 * traceOne ** 2) / rowCount;
  const varianceDefined = information > 0 && Number.isFinite(information);
  standardErrors.push(varianceDefined ? Math.sqrt(1 / information) / norm : NaN);
  coefficients.push(parameter / norm);
  coefficients.forEach((coefficient, row) => {
    const statistic = coefficient / standardErrors[row];
    result.table.set(
      [coefficient, standardErrors[row], statistic, getChiSquareSurvival(statistic * statistic, 1)],
      row * likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE
    );
  });
  result.residuals.set(residuals);
  const residualSquares = getCPUErrorDot(residuals, residuals);
  const slopes = centeredCoefficients.slice(1);
  const predicted = design.map(row => getCPUErrorDot(row.slice(1), slopes));
  const predictionMean = getCPUErrorMean(predicted);
  const centeredPrediction = predicted.map(value => value - predictionMean);
  const responseSquares = getCPUErrorDot(response, response);
  const predictionSquares = getCPUErrorDot(centeredPrediction, centeredPrediction);
  const correlation =
    responseSquares > 0 && predictionSquares > 0
      ? getCPUErrorDot(response, centeredPrediction) /
        Math.sqrt(responseSquares) /
        Math.sqrt(predictionSquares)
      : NaN;
  const conditioning = getCPUErrorConditioning(filtered, means);
  const determinant = logDeterminant(parameter);
  const logLikelihood = best.objective - 0.5 * rowCount * (Math.log(2 * Math.PI) + 1);
  result.summary.set([
    rowCount,
    parameter / norm,
    sigmaSquared,
    logLikelihood,
    2 * width - 2 * logLikelihood,
    width * Math.log(rowCount) - 2 * logLikelihood,
    correlation ** 2,
    NaN,
    residualSquares,
    ...conditioning,
    Number(conditioning[1] > GPU_REGRESSION_PRECISION_CONDITION_LIMIT),
    limit / norm,
    Number(stochastic),
    Number(!atBoundary),
    Number(atBoundary),
    Number(limit < likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BOUND_FRACTION),
    exact ? 0 : getSpatialLogDeterminantTruncationBound(parameter, maximumPower, rowCount),
    Number(varianceDefined),
    candidates.length,
    determinant
  ]);
  result.converged[0] = Number(!atBoundary);
  return result;
}

/** CPU whole-CSR operation; GPU-only size/search limits are dropped, with identical graph ports. */
export class CPUSpatialErrorMaximumLikelihoodOperation extends GPUCPUOperation {
  constructor(props: SpatialErrorMaximumLikelihoodOperationProps) {
    const id = props.id ?? 'spatial-error-maximum-likelihood';
    validateOptions(id, props);
    validateCapacity(id, 'rowCount', props.rowCount);
    if (props.tileRowCount !== undefined) validateCapacity(id, 'tileRowCount', props.tileRowCount);
    validateRequirement(
      id,
      (props.logDeterminant === 'exact') === (props.eigenvalues !== undefined),
      'eigenvalues are required exactly when logDeterminant is exact'
    );
    const operation = new GPUSpatialErrorMaximumLikelihoodOperation(props);
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
        const result = computeSpatialErrorMaximumLikelihoodOnCPU(
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
          ...(props.output.residuals !== undefined
            ? {[props.output.residuals]: result.residuals}
            : {}),
          ...(props.output.powerTraceStandardErrors !== undefined
            ? {[props.output.powerTraceStandardErrors]: result.powerTraceStandardErrors}
            : {})
        };
      }
    });
  }
}
