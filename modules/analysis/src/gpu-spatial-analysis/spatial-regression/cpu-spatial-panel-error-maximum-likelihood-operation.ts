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
import {fitCPURegressionWithQR} from './cpu-regression-linear-algebra';
import {
  validateCPURegressionWeights,
  hasCPURegressionWeightsLoss,
  type CPURegressionWeights,
  type CPURegressionWeightsStatus
} from './cpu-regression-weights';
import {
  getCPUErrorConditioning,
  getCPUErrorDot,
  getCPUErrorMean
} from './cpu-spatial-error-regression';
import {getChiSquareSurvival} from './ordinary-least-squares-statistics';
import * as likelihood from './spatial-maximum-likelihood-parameters';
import {getSpatialExactLogDeterminant} from './spatial-exact-log-determinant';
import {
  getSpatialLogDeterminantFromTraces,
  getSpatialLogDeterminantTruncationBound,
  getSpatialWeightsTracesProbeValue
} from './spatial-weights-traces-parameters';
import {SPATIAL_SERIES_TAIL_MINIMUM_POWER} from './spatial-series-tail';
import {getCPUSpatialErrorResponseRelation} from './cpu-spatial-error-likelihood-polynomial';
import {hasSpatialErrorScalarWeights} from './spatial-error-scalar-profile';
import {
  GPUSpatialPanelErrorMaximumLikelihoodOperation,
  snapshotPanelErrorOperationProps,
  type SpatialPanelErrorMaximumLikelihoodOperationProps
} from './gpu-spatial-panel-error-maximum-likelihood-operation';
import type {CPUSpatialErrorMaximumLikelihoodResult} from './cpu-spatial-error-maximum-likelihood-operation';

/** Balanced period-major input; eigenvalues describe W (N values) or the expanded N*T block graph. */
export type CPUSpatialPanelErrorMaximumLikelihoodInput = {
  weights: CPURegressionWeights & {
    status?: CPURegressionWeightsStatus;
  };
  predictors: Float32Array;
  response: Float32Array;
  eigenvalues?: Float32Array;
  traceParameters?: Uint32Array;
};
export type CPUSpatialPanelErrorMaximumLikelihoodOptions = Pick<
  SpatialPanelErrorMaximumLikelihoodOperationProps,
  | 'periodCount'
  | 'effects'
  | 'predictorCount'
  | 'maximumPower'
  | 'traceMode'
  | 'traceProbeCount'
  | 'tailExtrapolation'
  | 'logDeterminant'
>;
export type CPUSpatialPanelErrorMaximumLikelihoodResult = CPUSpatialErrorMaximumLikelihoodResult & {
  diagnosticTests: Float32Array;
  diagnosticStatus: Uint32Array;
};

function validateOptions(id: string, options: CPUSpatialPanelErrorMaximumLikelihoodOptions): void {
  validateCapacity(id, 'periodCount', options.periodCount, 2);
  validateCapacity(id, 'predictorCount', options.predictorCount);
  validateCapacity(
    id,
    'maximumPower',
    options.maximumPower ?? 128,
    options.tailExtrapolation ? SPATIAL_SERIES_TAIL_MINIMUM_POWER : 2
  );
  validateCapacity(id, 'traceProbeCount', options.traceProbeCount ?? 64);
  validateRequirement(
    id,
    ['unit', 'time', 'twoWay', 'random'].includes(options.effects ?? 'unit'),
    "effects must be 'unit', 'time', 'twoWay' or 'random'"
  );
  validateRequirement(
    id,
    options.effects !== 'random',
    "effects 'random' with independent unit effects needs a second covariance determinant; use a dense random-effects spatial-error estimator"
  );
  validateRequirement(
    id,
    options.traceMode === undefined || ['exact', 'stochastic'].includes(options.traceMode),
    "traceMode must be 'exact' or 'stochastic'"
  );
  validateRequirement(
    id,
    options.logDeterminant === undefined || ['exact', 'series'].includes(options.logDeterminant),
    "logDeterminant must be 'exact' or 'series'"
  );
}

/** Exact dyadic rank after eliminating the selected balanced-panel nuisance effects. */
function hasPanelPredictorRank(
  input: CPUSpatialPanelErrorMaximumLikelihoodInput,
  periods: number,
  columns: number,
  effects: string
): boolean {
  const entities = input.response.length / periods;
  const bits = new Uint32Array(
    input.predictors.buffer,
    input.predictors.byteOffset,
    input.predictors.length
  );
  const integer = (row: number, column: number) => {
    const word = bits[row * columns + column];
    const exponent = (word >>> 23) & 255;
    const magnitude =
      BigInt((word & 0x7fffff) | (exponent ? 0x800000 : 0)) << BigInt(Math.max(0, exponent - 1));
    return word >>> 31 ? -magnitude : magnitude;
  };
  const contrasts = Array.from({length: input.response.length}, (_, row) =>
    Array.from({length: columns}, (_, column) => {
      const entity = row % entities,
        periodStart = row - entity;
      if (effects === 'time') return integer(row, column) - integer(periodStart, column);
      if (effects === 'twoWay')
        return (
          integer(row, column) -
          integer(entity, column) -
          integer(periodStart, column) +
          integer(0, column)
        );
      return integer(row, column) - integer(entity, column);
    })
  );
  let rank = 0;
  for (let column = 0; column < columns; column++) {
    const pivot = contrasts.findIndex((row, index) => index >= rank && row[column] !== 0n);
    if (pivot < 0) return false;
    [contrasts[rank], contrasts[pivot]] = [contrasts[pivot], contrasts[rank]];
    for (let row = rank + 1; row < contrasts.length; row++) {
      const factor = contrasts[row][column];
      if (!factor) continue;
      const divisor = contrasts[rank][column];
      for (let other = column; other < columns; other++)
        contrasts[row][other] = contrasts[row][other] * divisor - contrasts[rank][other] * factor;
      // Normalize each exact integer row to bound growth; no numeric rank tolerance.
      const gcd = (first: bigint, second: bigint): bigint => {
        while (second) {
          const remainder = first % second;
          first = second;
          second = remainder;
        }
        return first < 0n ? -first : first;
      };
      const common = contrasts[row].reduce(gcd, 0n);
      if (common) contrasts[row] = contrasts[row].map(value => value / common);
    }
    rank++;
  }
  return true;
}

/**
 * Profiled balanced fixed-effects error ML, divisor N*T, with conditional expected-information
 * standard errors and pooled OLS LM diagnostics. All arithmetic is float64 until publication.
 *
 * What differs from the GPU contributor:
 * - Float64 within transforms and QR replace float32 tiled transforms and Cholesky; rank uses
 *   the shared QR floor. GPU-only row/predictor/power/period-contrast limits are dropped.
 * - The same 1024-interval stationary-crossing search uses direct filtered residuals and
 *   48 bisections instead of float32 moment polynomials and 32 bisections. It can miss extrema
 *   narrower than one interval, as can the GPU search. No unbounded optimization is claimed.
 * - Initial rank deficiency returns status 1; no filtered rescue is attempted. No mask or
 *   missing panels: any nonfinite observation returns status 3. Overflow suppresses the fit
 *   and diagnostics (status 3 and 1 respectively). Per-frame values remain ordinary arrays.
 * - Shared float64 chi-square survival and conditioning helpers retain their own conventions.
 * - An exact original-data certificate rejects a zero-variance pooled OLS fit with diagnostic
 *   status 2; roundoff in a QR residual cannot supply the positive variance required by LM tests.
 *   The Gram certificate also covers multivariate/effect-space exact fits; the GPU's original-
 *   input witness covers y=a+b*x_j on normal/zero coordinates certified by integer collinearity.
 * - Trace standard errors remain available when exact-fit rejection returns status 3: they
 *   describe W and its probes independently of whether y has a positive-variance fit.
 */
export function computeSpatialPanelErrorMaximumLikelihoodOnCPU(
  input: CPUSpatialPanelErrorMaximumLikelihoodInput,
  options: CPUSpatialPanelErrorMaximumLikelihoodOptions
): CPUSpatialPanelErrorMaximumLikelihoodResult {
  const id = 'computeSpatialPanelErrorMaximumLikelihoodOnCPU';
  validateOptions(id, options);
  const entities = validateCPURegressionWeights(id, input.weights, 'handled-by-caller');
  const {periodCount, predictorCount} = options;
  const rows = entities * periodCount;
  validateRequirement(
    id,
    input.response.length === rows,
    'response length must equal entities * periodCount'
  );
  validateRequirement(
    id,
    input.predictors.length === rows * predictorCount,
    'predictors length must equal response length * predictorCount'
  );
  const exact = options.logDeterminant === 'exact';
  validateRequirement(
    id,
    exact === (input.eigenvalues !== undefined),
    'eigenvalues are required exactly when logDeterminant is exact'
  );
  if (input.eigenvalues)
    validateRequirement(
      id,
      input.eigenvalues.length === entities || input.eigenvalues.length === rows,
      `eigenvalues length ${input.eigenvalues.length} must equal entityCount ${entities} or total panel rows ${rows}`
    );
  if (input.traceParameters)
    validateRequirement(
      id,
      input.traceParameters.length >= 1,
      'traceParameters must hold the packed seed'
    );
  if (input.weights.status?.overflow)
    validateRequirement(
      id,
      input.weights.status.overflow.length >= 1,
      'weights.status.overflow must hold one word'
    );
  if (input.weights.status?.candidateOverflow)
    validateRequirement(
      id,
      input.weights.status.candidateOverflow.length >= 1,
      'weights.status.candidateOverflow must hold one word'
    );
  const spectrumRepetitions = input.eigenvalues?.length === entities ? periodCount : 1;
  const maximumPower = options.maximumPower ?? 128;
  const stochastic =
    (options.traceMode ?? (rows <= 4096 ? 'exact' : 'stochastic')) === 'stochastic';
  const probes = stochastic ? (options.traceProbeCount ?? 64) : rows;
  const periodEffects = options.effects === 'time' || options.effects === 'twoWay';
  const width = predictorCount + 1 + (periodEffects ? periodCount - 1 : 0);
  const result: CPUSpatialPanelErrorMaximumLikelihoodResult = {
    table: new Float32Array((predictorCount + 2) * 4).fill(NaN),
    summary: new Float32Array(22).fill(NaN),
    residuals: new Float32Array(rows).fill(NaN),
    powerTraceStandardErrors: new Float32Array(maximumPower),
    status: Uint32Array.of(0),
    converged: Uint32Array.of(0),
    diagnosticTests: new Float32Array(6).fill(NaN),
    diagnosticStatus: Uint32Array.of(1)
  };
  result.summary[0] = rows;
  const fail = (code: number) => {
    result.status[0] = code;
    return result;
  };
  if (hasCPURegressionWeightsLoss(id, input.weights)) {
    result.summary.fill(NaN);
    result.powerTraceStandardErrors.fill(NaN);
    return fail(3);
  }
  const multiply = (values: readonly number[]) =>
    Array.from({length: rows}, (_, row) => {
      const entity = row % entities;
      const start = row - entity;
      let total = 0;
      for (
        let slot = input.weights.offsets[entity];
        slot < input.weights.offsets[entity + 1];
        slot++
      ) {
        const neighbor = input.weights.neighbors[slot];
        if (neighbor < entities) total += input.weights.weights[slot] * values[start + neighbor];
      }
      return total;
    });
  let norm = 0;
  for (let entity = 0; entity < entities; entity++) {
    let total = 0;
    for (
      let slot = input.weights.offsets[entity];
      slot < input.weights.offsets[entity + 1];
      slot++
    ) {
      if (input.weights.neighbors[slot] < entities) total += Math.abs(input.weights.weights[slot]);
    }
    norm = Math.max(norm, total);
  }
  if (!Number.isFinite(norm)) return fail(3);
  const getProbe = (probe: number) =>
    Array.from({length: rows}, (_, row) =>
      stochastic
        ? getSpatialWeightsTracesProbeValue(input.traceParameters?.[0] ?? 0, probe, row)
        : Number(row === probe)
    );
  const traces = new Array<number>(maximumPower).fill(0);
  const traceSquares = traces.slice();
  for (let probe = 0; probe < probes; probe++) {
    const initialProbe = getProbe(probe);
    let vector = initialProbe;
    for (let power = 0; power < maximumPower; power++) {
      vector = multiply(vector).map(value => value / norm);
      const value = stochastic ? getCPUErrorDot(initialProbe, vector) : vector[probe];
      if (stochastic) {
        const difference = value - traces[power];
        traces[power] += difference / (probe + 1);
        traceSquares[power] += difference * (value - traces[power]);
      } else traces[power] += value;
    }
  }
  result.powerTraceStandardErrors.set(
    traceSquares.map(value =>
      stochastic ? (probes < 2 ? NaN : Math.sqrt(Math.max(0, value) / (probes - 1) / probes)) : 0
    )
  );
  computeDiagnostics(input, predictorCount, multiply, result);
  if (rows <= width + 1) return fail(2);
  if (
    ![...input.response, ...input.predictors, ...(input.eigenvalues ?? [])].every(Number.isFinite)
  )
    return fail(3);

  const panel = Array.from({length: rows}, (_, row) => [
    input.response[row],
    ...input.predictors.subarray(row * predictorCount, (row + 1) * predictorCount)
  ]);
  if (!hasPanelPredictorRank(input, periodCount, predictorCount, options.effects ?? 'unit'))
    return fail(1);
  const demean = (time: boolean) => {
    const groupCount = time ? periodCount : entities;
    const observationCount = time ? entities : periodCount;
    for (let group = 0; group < groupCount; group++) {
      for (let column = 0; column <= predictorCount; column++) {
        const indices = Array.from({length: observationCount}, (_, observation) =>
          time ? group * entities + observation : observation * entities + group
        );
        const mean = getCPUErrorMean(indices.map(row => panel[row][column]));
        for (const row of indices) panel[row][column] -= mean;
      }
    }
  };
  if (options.effects !== 'time') demean(false);
  if (periodEffects) demean(true);
  const raw = panel.map((row, index) => [
    ...row.slice(1),
    ...(periodEffects
      ? Array.from({length: periodCount - 1}, (_, period) =>
          Number(Math.floor(index / entities) === period)
        )
      : [])
  ]);
  const means = raw[0].map((_, column) => getCPUErrorMean(raw.map(row => row[column])));
  const design = raw.map(row => [1, ...row.map((value, column) => value - means[column])]);
  const responseMean = getCPUErrorMean(panel.map(row => row[0]));
  const response = panel.map(row => row[0] - responseMean);
  const initial = fitCPURegressionWithQR(design, response);
  if (!initial) return fail(1);
  const initiallyOrthogonal = getCPUSpatialErrorResponseRelation(design, response).orthogonal;
  if (initiallyOrthogonal) initial.estimate.fill(0);
  if (!(norm > 0)) return fail(3);
  // Certify the original dyadic float32 observations, before the within means introduce
  // float64 rounding. Otherwise an exact affine panel publishes a rounding-sized variance.
  const originalDesign = Array.from({length: rows}, (_, row) => [
    1,
    ...input.predictors.subarray(row * predictorCount, (row + 1) * predictorCount),
    ...(options.effects !== 'time'
      ? Array.from({length: entities - 1}, (_, entity) => Number(row % entities === entity))
      : []),
    ...(periodEffects
      ? Array.from({length: periodCount - 1}, (_, period) =>
          Number(Math.floor(row / entities) === period)
        )
      : [])
  ]);
  const exactFit = getCPUSpatialErrorResponseRelation(
    originalDesign,
    Array.from(input.response)
  ).exactFit;
  const initialResidual = response.map(
    (value, row) => value - getCPUErrorDot(design[row], initial.estimate)
  );
  const laggedColumns = design[0].map((_, column) => multiply(design.map(row => row[column])));
  const laggedResponse = multiply(initialResidual);
  // Trace uncertainty describes W, not the existence of a positive-variance response fit.
  // Preserve this independently meaningful output on an exact-fit failure, as the GPU does.
  if (exactFit) return fail(3);
  const limit =
    exact || options.tailExtrapolation
      ? 0.99
      : likelihood.getSpatialMaximumLikelihoodSeriesLimit(maximumPower);
  const bound = limit / norm;
  const determinant = (parameter: number) =>
    exact
      ? spectrumRepetitions * getSpatialExactLogDeterminant(parameter, input.eigenvalues!)
      : getSpatialLogDeterminantFromTraces(parameter, traces, norm, options.tailExtrapolation);
  const traceOne = (parameter: number) =>
    exact
      ? spectrumRepetitions *
        Array.from(input.eigenvalues!).reduce(
          (total, value) => total + value / (1 - parameter * value),
          0
        )
      : likelihood.getSpatialMaximumLikelihoodTraceOne(
          parameter,
          traces,
          norm,
          options.tailExtrapolation
        );
  const evaluate = (parameter: number) => {
    const filtered = design.map((row, index) =>
      row.map((value, column) => value - parameter * laggedColumns[column][index])
    );
    const filteredResponse = initialResidual.map(
      (value, row) => value - parameter * laggedResponse[row]
    );
    const fit = fitCPURegressionWithQR(filtered, filteredResponse);
    if (!fit) return undefined;
    if (
      initiallyOrthogonal &&
      getCPUSpatialErrorResponseRelation(filtered, filteredResponse).orthogonal
    )
      fit.estimate.fill(0);
    const residual = initialResidual.map(
      (value, row) => value - getCPUErrorDot(design[row], fit.estimate)
    );
    const lagged = multiply(residual);
    const errors = residual.map((value, row) => value - parameter * lagged[row]);
    const squares = getCPUErrorDot(errors, errors);
    if (!(squares > 0) || !Number.isFinite(squares)) return undefined;
    return {
      fit,
      filtered,
      residual,
      sigmaSquared: squares / rows,
      objective: (-rows / 2) * Math.log(squares / rows) + determinant(parameter),
      derivative: (rows * getCPUErrorDot(errors, lagged)) / squares - traceOne(parameter)
    };
  };
  const candidates: {parameter: number; atBoundary: boolean}[] = [];
  const flatProfile =
    exact &&
    Array.from(input.eigenvalues!).every(value => value === input.eigenvalues![0]) &&
    hasSpatialErrorScalarWeights(input.weights, entities, input.eigenvalues![0]);
  if (flatProfile) candidates.push({parameter: -bound, atBoundary: true});
  else {
    let previousParameter = -bound;
    let previousDerivative = evaluate(-bound)?.derivative ?? NaN;
    const lowerInward = -bound + bound * Number.EPSILON;
    const lowerInwardDerivative = evaluate(lowerInward)?.derivative ?? NaN;
    if (previousDerivative < 0 || (previousDerivative === 0 && lowerInwardDerivative <= 0))
      candidates.push({parameter: -bound, atBoundary: true});
    if (previousDerivative === 0 && lowerInwardDerivative > 0) {
      previousDerivative = lowerInwardDerivative;
      previousParameter = lowerInward;
    }
    for (let interval = 1; interval <= 1024; interval++) {
      const parameter = -bound + (2 * bound * interval) / 1024;
      const derivative = evaluate(parameter)?.derivative ?? NaN;
      if (previousDerivative > 0 && derivative <= 0) {
        let lower = previousParameter,
          upper = parameter;
        for (let iteration = 0; iteration < 48; iteration++) {
          const middle = (lower + upper) / 2;
          if ((evaluate(middle)?.derivative ?? NaN) > 0) lower = middle;
          else upper = middle;
        }
        const root =
          interval === 1024 && derivative === 0 && upper === bound ? bound : (lower + upper) / 2;
        candidates.push({parameter: root, atBoundary: root === -bound || root === bound});
      }
      previousParameter = parameter;
      previousDerivative = derivative;
    }
    const upperInwardDerivative = evaluate(bound - bound * Number.EPSILON)?.derivative ?? NaN;
    if (
      (previousDerivative > 0 || (previousDerivative === 0 && upperInwardDerivative > 0)) &&
      !candidates.some(candidate => candidate.parameter === bound)
    )
      candidates.push({parameter: bound, atBoundary: true});
  }
  let best: ReturnType<typeof evaluate>;
  let parameter = NaN,
    atBoundary = false;
  for (const candidate of candidates) {
    const fit = evaluate(candidate.parameter);
    if (fit && (!best || fit.objective > best.objective)) {
      best = fit;
      parameter = candidate.parameter;
      atBoundary = candidate.atBoundary;
    }
  }
  if (!best) return fail(3);
  const coefficients = initial.estimate.map((value, column) => value + best!.fit.estimate[column]);
  let traceThree = 0;
  for (let probe = 0; probe < probes; probe++) {
    const initialProbe = getProbe(probe);
    let vector = initialProbe;
    for (let power = 1; power < maximumPower; power++)
      vector = multiply(vector).map((value, row) => initialProbe[row] + parameter * value);
    const transformed = multiply(vector);
    traceThree += getCPUErrorDot(transformed, transformed);
  }
  if (stochastic) traceThree /= probes;
  const traceTwo = exact
    ? spectrumRepetitions *
      Array.from(input.eigenvalues!).reduce(
        (total, value) => total + (value / (1 - parameter * value)) ** 2,
        0
      )
    : likelihood.getSpatialMaximumLikelihoodTraceTwo(
        parameter,
        traces,
        norm,
        options.tailExtrapolation
      );
  const information = traceTwo + traceThree - (2 * traceOne(parameter) ** 2) / rows;
  const varianceDefined = !flatProfile && information > 0 && Number.isFinite(information);
  for (let column = 1; column <= predictorCount + 1; column++) {
    const estimate = column <= predictorCount ? coefficients[column] : parameter;
    const error =
      column <= predictorCount
        ? Math.sqrt(best.sigmaSquared) * Math.hypot(...best.fit.inverseRoot[column])
        : varianceDefined
          ? Math.sqrt(1 / information)
          : NaN;
    const statistic = estimate / error;
    result.table.set(
      [estimate, error, statistic, getChiSquareSurvival(statistic ** 2, 1)],
      column * 4
    );
  }
  result.residuals.set(best.residual);
  const fitted = design.map(row => getCPUErrorDot(row.slice(1), coefficients.slice(1)));
  const fittedMean = getCPUErrorMean(fitted);
  const centeredFit = fitted.map(value => value - fittedMean);
  const correlation =
    getCPUErrorDot(response, centeredFit) /
    Math.sqrt(getCPUErrorDot(response, response)) /
    Math.sqrt(getCPUErrorDot(centeredFit, centeredFit));
  const conditioning = getCPUErrorConditioning(best.filtered, means);
  result.summary.set([
    rows,
    parameter,
    best.sigmaSquared,
    best.objective - (rows / 2) * (Math.log(2 * Math.PI) + 1),
    NaN,
    NaN,
    correlation ** 2,
    NaN,
    getCPUErrorDot(best.residual, best.residual),
    ...conditioning,
    Number(conditioning[1] > 1e4),
    bound,
    Number(stochastic),
    Number(!atBoundary),
    Number(atBoundary),
    Number(limit < 0.99),
    exact
      ? 0
      : getSpatialLogDeterminantTruncationBound(parameter, maximumPower, rows, norm) +
        Math.abs(
          determinant(parameter) - getSpatialLogDeterminantFromTraces(parameter, traces, norm)
        ),
    Number(varianceDefined),
    candidates.length,
    determinant(parameter)
  ]);
  result.converged[0] = Number(!atBoundary);
  return result;
}

function computeDiagnostics(
  input: CPUSpatialPanelErrorMaximumLikelihoodInput,
  predictorCount: number,
  multiply: (values: readonly number[]) => number[],
  result: CPUSpatialPanelErrorMaximumLikelihoodResult
): void {
  const rows = input.response.length;
  if (
    ![...input.response, ...input.predictors].every(Number.isFinite) ||
    rows <= predictorCount + 1
  )
    return;
  const means = Array.from({length: predictorCount}, (_, column) =>
    getCPUErrorMean(
      Array.from({length: rows}, (_, row) => input.predictors[row * predictorCount + column])
    )
  );
  const responseMean = getCPUErrorMean(Array.from(input.response));
  const design = Array.from({length: rows}, (_, row) => [
    1,
    ...means.map((mean, column) => input.predictors[row * predictorCount + column] - mean)
  ]);
  const response = Array.from(input.response, value => value - responseMean);
  const fit = fitCPURegressionWithQR(design, response);
  if (!fit) return;
  const originalDesign = Array.from({length: rows}, (_, row) => [
    1,
    ...input.predictors.subarray(row * predictorCount, (row + 1) * predictorCount)
  ]);
  const responseRelation = getCPUSpatialErrorResponseRelation(
    originalDesign,
    Array.from(input.response)
  );
  if (responseRelation.exactFit || responseRelation.orthogonal) {
    // An exact pooled fit has no error variance; an exactly zero pooled fitted vector has
    // no robust-test information. Neither can be identified by positive QR roundoff.
    result.diagnosticStatus[0] = 2;
    return;
  }
  if (getCPUSpatialErrorResponseRelation(design, response).orthogonal) fit.estimate.fill(0);
  const centeredFitted = design.map(row => getCPUErrorDot(row, fit.estimate));
  const residual = response.map((value, row) => value - getCPUErrorDot(design[row], fit.estimate));
  const rowSums = multiply(new Array<number>(rows).fill(1));
  // Remove only a constant from W fitted. Varying W1 still carries the response mean.
  const lagged = multiply(centeredFitted).map(
    (value, row) => value + responseMean * (rowSums[row] - rowSums[0])
  );
  const laggedFit = lagged.map(value => value - lagged[0]);
  const projection = fitCPURegressionWithQR(design, laggedFit);
  if (!projection) return;
  result.diagnosticStatus[0] = 2;
  const projected = laggedFit.map(
    (value, row) => value - getCPUErrorDot(design[row], projection.estimate)
  );
  const variance = getCPUErrorDot(residual, residual) / (rows - predictorCount - 1);
  const entities = input.weights.offsets.length - 1;
  let trace = 0;
  for (let entity = 0; entity < entities; entity++) {
    for (
      let slot = input.weights.offsets[entity];
      slot < input.weights.offsets[entity + 1];
      slot++
    ) {
      const neighbor = input.weights.neighbors[slot],
        weight = input.weights.weights[slot];
      if (
        neighbor >= entities ||
        neighbor === entity ||
        !Number.isFinite(weight) ||
        (slot > input.weights.offsets[entity] && input.weights.neighbors[slot - 1] >= neighbor)
      )
        return;
      let reciprocal = 0;
      for (
        let other = input.weights.offsets[neighbor];
        other < input.weights.offsets[neighbor + 1];
        other++
      )
        if (input.weights.neighbors[other] === entity) reciprocal = input.weights.weights[other];
      trace += weight * (weight + reciprocal) * (rows / entities);
    }
  }
  const projectedSquares = getCPUErrorDot(projected, projected);
  const laggedFitMean = getCPUErrorMean(laggedFit);
  const centeredLaggedFit = laggedFit.map(value => value - laggedFitMean);
  const laggedFitSquares = getCPUErrorDot(centeredLaggedFit, centeredLaggedFit);
  // Match the diagnostic's f32 norm-relative rank floor; QR roundoff is not information.
  if (!(projectedSquares / laggedFitSquares > 1e-12)) return;
  const projectedInformation = projectedSquares / variance;
  const residualLag = getCPUErrorDot(residual, multiply(residual)) / variance;
  const lagContrast = getCPUErrorDot(residual, projected) / variance;
  if (
    !(variance > 0 && trace > 0 && projectedInformation > 0) ||
    ![projectedInformation, residualLag, lagContrast].every(Number.isFinite)
  )
    return;
  const information = trace + projectedInformation;
  const errorContrast =
    (projectedInformation / information) * residualLag - (trace / information) * lagContrast;
  const tests = [
    residualLag ** 2 / trace,
    lagContrast ** 2 / projectedInformation,
    errorContrast ** 2 / (trace * (projectedInformation / information))
  ];
  if (!tests.every(value => value >= 0 && Number.isFinite(value))) return;
  result.diagnosticTests.set(tests.flatMap(value => [value, getChiSquareSurvival(value, 1)]));
  result.diagnosticStatus[0] = 0;
}

/** CPU operation with the contributor's ports; GPU-only dimensional caps are dropped. */
export class CPUSpatialPanelErrorMaximumLikelihoodOperation extends GPUCPUOperation {
  constructor(props: SpatialPanelErrorMaximumLikelihoodOperationProps) {
    props = snapshotPanelErrorOperationProps(props);
    const id = props.id ?? 'spatial-panel-error-maximum-likelihood';
    validateOptions(id, props);
    validateCapacity(id, 'entityCount', props.entityCount);
    if (props.tileRowCount !== undefined) validateCapacity(id, 'tileRowCount', props.tileRowCount);
    validateRequirement(
      id,
      (props.logDeterminant === 'exact') === (props.eigenvalues !== undefined),
      'eigenvalues are required exactly when logDeterminant is exact'
    );
    const operation = new GPUSpatialPanelErrorMaximumLikelihoodOperation(props);
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
        validateRequirement(
          id,
          getCPUOperationUint32Rows(id, inputs, 'offsets').length === props.entityCount + 1,
          'entityCount must equal weights.offsets.length - 1'
        );
        const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(
          {
            predictors: getCPUOperationFloat32Rows(id, inputs, 'predictors'),
            response: getCPUOperationFloat32Rows(id, inputs, 'response'),
            weights: {
              offsets: getCPUOperationUint32Rows(id, inputs, 'offsets'),
              neighbors: getCPUOperationUint32Rows(id, inputs, 'neighbors'),
              weights: getCPUOperationFloat32Rows(id, inputs, 'weights'),
              status: props.weights.status
                ? {
                    complete:
                      props.weights.status.complete !== undefined
                        ? getCPUOperationUint32Rows(id, inputs, 'complete')
                        : undefined,
                    count: props.weights.status.count
                      ? getCPUOperationUint32Rows(id, inputs, 'count')
                      : undefined,
                    requiredCount: props.weights.status.requiredCount
                      ? getCPUOperationUint32Rows(id, inputs, 'requiredCount')
                      : undefined,
                    overflow:
                      props.weights.status.overflow !== undefined
                        ? getCPUOperationUint32Rows(id, inputs, 'overflow')
                        : undefined,
                    candidateOverflow:
                      props.weights.status.candidateOverflow !== undefined
                        ? getCPUOperationUint32Rows(id, inputs, 'candidateOverflow')
                        : undefined
                  }
                : undefined
            },
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
        validateRequirement(
          id,
          result.residuals.length === props.entityCount * props.periodCount,
          'entityCount must match weights row count'
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
            : {}),
          ...(props.diagnostics
            ? {
                [props.diagnostics.tests]: result.diagnosticTests,
                [props.diagnostics.status]: result.diagnosticStatus
              }
            : {})
        };
      }
    });
  }
}
