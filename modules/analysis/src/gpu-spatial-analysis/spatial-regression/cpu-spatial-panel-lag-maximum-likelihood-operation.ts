// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {invertCPUScaledMatrix} from '../../analysis-core/kernels/cpu-scaled-inverse';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  factorCPURegressionCorrelation,
  fitCPURegressionWithQR,
  solveCPURegressionFactor
} from './cpu-regression-linear-algebra';
import {type CPURegressionWeights, validateCPURegressionWeights} from './cpu-regression-weights';
import {getCPUErrorDot, getCPUErrorMean} from './cpu-spatial-error-regression';
import {getChiSquareSurvival} from './ordinary-least-squares-statistics';
import {
  GPUSpatialPanelLagMaximumLikelihoodOperation,
  type SpatialPanelLagMaximumLikelihoodOperationProps
} from './gpu-spatial-panel-lag-maximum-likelihood-operation';

// These bounds define the contributor's constrained estimator and boundary diagnostics.
const MINIMUM_PHI = 1e-8;
const RELATIVE_PHI = 1e-2;
const RHO_BOUND = 0.999;
const GOLDEN_RATIO = (Math.sqrt(5) - 1) / 2;
// The shader's finite objective failure sentinel, including its decimal rounding.
const INVALID_OBJECTIVE = 3.402823e38;

/** Period-major complete panel and whole entity CSR. Unused slot capacity is ignored. */
export type CPUSpatialPanelLagMaximumLikelihoodInput = {
  weights: CPURegressionWeights &
    Pick<
      SpatialPanelLagMaximumLikelihoodOperationProps['weights'],
      'information' | 'spatialContext'
    > & {
      distances?: Float32Array;
      status?: Partial<
        Record<
          keyof NonNullable<SpatialPanelLagMaximumLikelihoodOperationProps['weights']['status']>,
          Uint32Array
        >
      >;
    };
  /** Compatibility alias for weights.status.overflow, used only when that field is absent. */
  overflow?: Uint32Array;
  predictors: Float32Array;
  response: Float32Array;
};
/** Estimator settings. GPU iteration budgets are accepted but not used. */
export type CPUSpatialPanelLagMaximumLikelihoodOptions = Pick<
  SpatialPanelLagMaximumLikelihoodOperationProps,
  | 'periodCount'
  | 'predictorCount'
  | 'effects'
  | 'maximumIterations'
  | 'maximumSearchIterations'
  | 'tolerance'
> & {hausman?: boolean};
/** Scalar vector layouts of the contributor, including its optional Hausman outputs. */
export type CPUSpatialPanelLagMaximumLikelihoodResult = {
  coefficients: Float32Array;
  covariance: Float32Array;
  summary: Float32Array;
  status: Uint32Array;
  fitted: Float32Array;
  residuals: Float32Array;
  hausman?: {tests: Float32Array; status: Uint32Array};
};
type PanelFit = {
  coefficients: number[];
  covariance: number[][];
  summary: number[];
  status: number;
  fitted: number[];
  residuals: number[];
};

function validateOptions(id: string, options: CPUSpatialPanelLagMaximumLikelihoodOptions): void {
  validateCapacity(id, 'periodCount', options.periodCount, 2);
  validateCapacity(id, 'predictorCount', options.predictorCount);
  validateCapacity(id, 'maximumIterations', options.maximumIterations ?? 64);
  validateCapacity(id, 'maximumSearchIterations', options.maximumSearchIterations ?? 32);
  validateFiniteNumber(id, 'tolerance', options.tolerance ?? 1e-5, 0);
  validateRequirement(id, (options.tolerance ?? 1e-5) > 0, 'tolerance must be positive');
  validateRequirement(
    id,
    ['random', 'unit'].includes(options.effects ?? 'random'),
    "effects must be 'random' or 'unit'"
  );
  validateRequirement(
    id,
    !(options.hausman && options.effects === 'unit'),
    'hausman requires random effects'
  );
}

function validateWeightsInformation(
  id: string,
  weights: Pick<SpatialPanelLagMaximumLikelihoodOperationProps['weights'], 'information'>,
  entityCount: number
): void {
  const information = weights.information;
  if (!information) return;
  validateRequirement(
    id,
    information.rowCount === entityCount,
    'weights.information.rowCount must equal entityCount'
  );
  validateRequirement(
    id,
    ['unknown', 'raw', 'binary', 'row', 'double', 'variance', 'kernel', 'symmetric'].includes(
      information.transform
    ),
    'weights.information.transform must be a known transform state'
  );
  validateRequirement(
    id,
    [true, false, 'unknown'].includes(information.symmetric),
    "weights.information.symmetric must be a boolean or 'unknown'"
  );
}

function invertPositiveMatrix(matrix: number[][]): number[][] | undefined {
  const scales = matrix.map((row, index) => Math.sqrt(row[index]));
  if (scales.some(value => !(value > 0) || !Number.isFinite(value))) return undefined;
  const factor = factorCPURegressionCorrelation(
    matrix.map((row, first) => row.map((value, second) => value / scales[first] / scales[second]))
  );
  if (!factor) return undefined;
  return scales.map((scale, row) =>
    solveCPURegressionFactor(
      factor,
      scales.map((_, column) => Number(row === column))
    ).map((value, column) => value / scale / scales[column])
  );
}

// Partial-pivoted elimination sums log pivots; it never multiplies a large determinant.
function getLogDeterminant(matrix: number[][]): number {
  const working = matrix.map(row => row.slice());
  let result = 0;
  for (let column = 0; column < matrix.length; column++) {
    let pivot = column;
    for (let row = column + 1; row < matrix.length; row++) {
      if (Math.abs(working[row][column]) > Math.abs(working[pivot][column])) pivot = row;
    }
    [working[column], working[pivot]] = [working[pivot], working[column]];
    const diagonal = working[column][column];
    if (diagonal === 0 || !Number.isFinite(diagonal)) return -Infinity;
    result += Math.log(Math.abs(diagonal));
    for (let row = column + 1; row < matrix.length; row++) {
      const multiplier = working[row][column] / diagonal;
      for (let other = column + 1; other < matrix.length; other++)
        working[row][other] -= multiplier * working[column][other];
    }
  }
  return result;
}

// Only used after the binary64 residuals are all zero: verify their exact signs, so rounded
// cancellation cannot turn a very small positive variance into a certified zero-variance pole.
function hasExactPanelFit(design: number[][], target: number[], estimate: number[]): boolean {
  const storage = new DataView(new ArrayBuffer(8));
  const getDyadic = (value: number): {units: bigint; exponent: number} => {
    storage.setFloat64(0, value);
    const bits = storage.getBigUint64(0);
    const exponent = Number((bits >> 52n) & 2047n);
    let units = (bits & ((1n << 52n) - 1n)) | (exponent === 0 ? 0n : 1n << 52n);
    if (bits >> 63n) units = -units;
    return {units, exponent: exponent === 0 ? -1074 : exponent - 1075};
  };
  const coefficients = estimate.map(getDyadic);
  return design.every((row, position) => {
    const expected = getDyadic(target[position]);
    const terms = row.map((value, column) => {
      const entry = getDyadic(value);
      return {
        units: entry.units * coefficients[column].units,
        exponent: entry.exponent + coefficients[column].exponent
      };
    });
    terms.push({units: -expected.units, exponent: expected.exponent});
    const nonzero = terms.filter(term => term.units !== 0n);
    const minimumExponent = Math.min(...nonzero.map(term => term.exponent));
    return (
      nonzero.reduce(
        (sum, term) => sum + (term.units << BigInt(term.exponent - minimumExponent)),
        0n
      ) === 0n
    );
  });
}

function minimizeProfile(objective: (parameter: number) => number): {
  parameter: number;
  width: number;
} {
  let lower = -RHO_BOUND;
  let upper = RHO_BOUND;
  let left = upper - GOLDEN_RATIO * (upper - lower);
  let right = lower + GOLDEN_RATIO * (upper - lower);
  let leftScore = objective(left);
  let rightScore = objective(right);
  while (upper - lower > 16 * Number.EPSILON) {
    if (leftScore <= rightScore) {
      upper = right;
      right = left;
      rightScore = leftScore;
      left = upper - GOLDEN_RATIO * (upper - lower);
      leftScore = objective(left);
    } else {
      lower = left;
      left = right;
      leftScore = rightScore;
      right = lower + GOLDEN_RATIO * (upper - lower);
      rightScore = objective(right);
    }
  }
  let parameter = leftScore <= rightScore ? left : right;
  // A true constrained boundary maximum is included, unlike the GPU's interior golden points.
  for (const endpoint of [-RHO_BOUND, RHO_BOUND]) {
    if (objective(endpoint) < objective(parameter)) parameter = endpoint;
  }
  return {parameter, width: upper - lower};
}

/**
 * Balanced random/unit-effects lag ML with centered float64 QR, direct residual squares and
 * dense log determinants. Conditional covariance and spreg's reported likelihood convention
 * (without N log(phi), and with the extra FE constant) follow the current contributor.
 *
 * What differs from the GPU contributor:
 * - Float64 centering, entity means, QR, lags and residuals replace float32 packed moments and
 *   Cholesky. Outputs round once. Ordinary comparisons use 5e-3 relative error, covariance
 *   scaled by diagonal geometric means; near-zero fitted/residual rows use a response-scale floor.
 * - Entity, period and predictor shader caps are dropped. Numerical rank uses the shared
 *   norm-scaled QR / correlation Cholesky float64 floors instead of the GPU's 1e-3 pivot floor.
 * - maximumIterations and maximumSearchIterations are ignored. Golden section continues to
 *   binary64 resolution, including interval endpoints; alternation continues to the same relative
 *   phi tolerance or a repeated complete state (status 3). Sweep counts, phi change and bracket
 *   width describe this CPU search and can differ. This retains the GPU's bounded golden-section
 *   profiling method, not a guarantee of a global maximum on a multimodal profile.
 * - The rho interval, phi floor 1e-8, boundary bits, and status-3 publication at that floor are
 *   retained. Covariance is withheld for status 3. Variance that rounds to nonpositive or
 *   nonfinite float32 is status 2, with covariance/fitted/residuals withheld. Hausman keeps
 *   RE-minus-FE covariance and df=predictorCount+2, and uses the public chi-square helper.
 *   On representability failures the CPU retains its float64 iterate and likelihood before
 *   output rounding; the GPU's earlier float32 overflow/underflow can change those diagnostics.
 * - A float64 profile score/curvature check rejects nonmaximal interior candidates as status 3
 *   with the iterate retained. Exact response copies of a design column are detected independently
 *   of QR. Other zero-variance poles with admissible rho are detected when the augmented QR
 *   estimate passes an exact dyadic residual certificate; this is not a complete symbolic rank
 *   test. Certified poles return status 2 rather than reporting a finite Gaussian maximum.
 * - Weight normalization acceptance uses serial float32 addition and the shader's float32
 *   1e-5 predicate. Statistical lags and fitting still use the original float32 weights in float64.
 * - Non-finite observations and invalid active CSR rows return status 1; malformed row bounds
 *   also return status 1, as on the GPU. Array shape mismatches throw. Empty CSR slot arrays
 *   are accepted by the pure function and produce status 1. Subnormals are retained.
 * - Optional weights distances and status arrays retain shape validation; information metadata
 *   retains contributor validation. Only the first overflow row affects fitting. Other status
 *   values and distances have no numerical effect, matching the contributor.
 *
 * @throws If predictors, response, weights array lengths or overflow are inconsistent.
 */
export function computeSpatialPanelLagMaximumLikelihoodOnCPU(
  input: CPUSpatialPanelLagMaximumLikelihoodInput,
  options: CPUSpatialPanelLagMaximumLikelihoodOptions
): CPUSpatialPanelLagMaximumLikelihoodResult {
  const id = 'computeSpatialPanelLagMaximumLikelihoodOnCPU';
  validateOptions(id, options);
  const {periodCount, predictorCount} = options;
  const entityCount = input.weights.offsets.length - 1;
  validateCapacity(id, 'weights.offsets entityCount', entityCount, 2);
  const rowCount = entityCount * periodCount;
  const columnCount = predictorCount + 1;
  const covarianceWidth = columnCount + 1;
  if (input.response.length !== rowCount)
    throw new Error(`${id} response length must equal entityCount * periodCount`);
  if (input.predictors.length !== rowCount * predictorCount)
    throw new Error(`${id} predictors length must equal response length * predictorCount`);
  if (input.weights.neighbors.length !== input.weights.weights.length)
    throw new Error(`${id} weights.weights length must equal weights.neighbors length`);
  if (input.weights.distances && input.weights.distances.length !== input.weights.neighbors.length)
    throw new Error(`${id} weights.distances length must equal weights.neighbors length`);
  if (input.overflow && input.overflow.length < 1)
    throw new Error(`${id} overflow must hold at least one value`);
  for (const [field, values] of Object.entries(input.weights.status ?? {})) {
    if (values && values.length < 1)
      throw new Error(`${id} weights.status.${field} must hold at least one value`);
  }
  validateWeightsInformation(id, input.weights, entityCount);
  const emptyFit = (): PanelFit => ({
    coefficients: new Array<number>(columnCount + 2).fill(NaN),
    covariance: Array.from({length: covarianceWidth}, () =>
      new Array<number>(covarianceWidth).fill(NaN)
    ),
    summary: new Array<number>(6).fill(NaN),
    status: 1,
    fitted: new Array<number>(rowCount).fill(NaN),
    residuals: new Array<number>(rowCount).fill(NaN)
  });
  const finish = (
    fit: PanelFit,
    comparison?: PanelFit
  ): CPUSpatialPanelLagMaximumLikelihoodResult => {
    const result: CPUSpatialPanelLagMaximumLikelihoodResult = {
      coefficients: Float32Array.from(fit.coefficients),
      covariance: Float32Array.from(fit.covariance.flat()),
      summary: Float32Array.from(fit.summary),
      status: Uint32Array.of(fit.status),
      fitted: Float32Array.from(fit.fitted),
      residuals: Float32Array.from(fit.residuals)
    };
    if (options.hausman) {
      result.hausman = {tests: new Float32Array(3).fill(NaN), status: Uint32Array.of(1)};
      if (fit.status === 0 && comparison?.status === 0) {
        result.hausman.status[0] = 2;
        const difference = fit.covariance
          .slice(1)
          .map((row, first) =>
            row
              .slice(1)
              .map((value, second) => value - comparison.covariance[first + 1][second + 1])
          );
        const inverse = invertPositiveMatrix(difference);
        if (inverse) {
          const contrast = comparison.coefficients
            .slice(1, columnCount + 1)
            .map((value, column) => value - fit.coefficients[column + 1]);
          const statistic = getCPUErrorDot(
            contrast,
            inverse.map(row => getCPUErrorDot(row, contrast))
          );
          result.hausman.tests.set([statistic, covarianceWidth, NaN]);
          if (statistic >= 0 && Number.isFinite(statistic)) {
            result.hausman.tests[2] = getChiSquareSurvival(statistic, covarianceWidth);
            result.hausman.status[0] = 0;
          }
        }
      }
    }
    return result;
  };
  if (
    (input.weights.status?.overflow ?? input.overflow)?.[0] ||
    !input.response.every(Number.isFinite) ||
    !input.predictors.every(Number.isFinite)
  )
    return finish(emptyFit());
  const dense = Array.from({length: entityCount}, () => new Array<number>(entityCount).fill(0));
  const {offsets, neighbors, weights} = input.weights;
  for (let entity = 0; entity < entityCount; entity++) {
    const begin = offsets[entity];
    const end = offsets[entity + 1];
    if (end < begin || end > neighbors.length) return finish(emptyFit());
    let total = 0;
    for (let slot = begin; slot < end; slot++) {
      const neighbor = neighbors[slot];
      const weight = weights[slot];
      if (
        neighbor >= entityCount ||
        neighbor === entity ||
        !(weight >= 0) ||
        !Number.isFinite(weight) ||
        (slot > begin && neighbors[slot - 1] >= neighbor)
      )
        return finish(emptyFit());
      dense[entity][neighbor] = weight;
      // Input acceptance is a contributor definition, not a statistical reduction.
      total = Math.fround(total + weight);
    }
    if (Math.abs(Math.fround(total - 1)) > Math.fround(1e-5)) return finish(emptyFit());
  }
  // The GPU reads each row's explicit begin/end; a nonzero first offset skips capacity.
  if (offsets[0] === 0) validateCPURegressionWeights(id, input.weights);
  const response = Array.from(input.response);
  const centers = [
    getCPUErrorMean(response),
    ...Array.from({length: predictorCount}, (_, column) =>
      getCPUErrorMean(
        Array.from({length: rowCount}, (_, row) => input.predictors[row * predictorCount + column])
      )
    )
  ];
  const raw = response.map((value, row) => {
    const entity = row % entityCount;
    const start = row - entity;
    return [
      1,
      ...centers
        .slice(1)
        .map((center, column) => input.predictors[row * predictorCount + column] - center),
      value - centers[0],
      getCPUErrorDot(dense[entity], response.slice(start, start + entityCount)) - centers[0]
    ];
  });
  const means = Array.from({length: entityCount}, (_, entity) =>
    raw[0].map((_, column) =>
      getCPUErrorMean(
        Array.from({length: periodCount}, (_, period) => raw[period * entityCount + entity][column])
      )
    )
  );
  const deviations = raw.map((row, index) =>
    row.map((value, column) => value - means[index % entityCount][column])
  );
  const datum = (row: number, column: number, phi: number) =>
    deviations[row][column] + phi * means[row % entityCount][column];
  const spatialMatrix = (rho: number) =>
    dense.map((row, entity) =>
      row.map((value, neighbor) => Number(entity === neighbor) - rho * value)
    );
  const fitModel = (unit: boolean): PanelFit => {
    const fit = emptyFit();
    fit.status = 2;
    const firstColumn = unit ? 1 : 0;
    const activeCount = columnCount - firstColumn;
    const designAt = (phi: number) =>
      raw.map((_, row) =>
        Array.from({length: activeCount}, (_, column) => datum(row, column + firstColumn, phi))
      );
    const pooled = fitCPURegressionWithQR(
      designAt(1),
      raw.map(row => row[columnCount])
    );
    if (!pooled) return fit;
    let beta = pooled.estimate;
    let rho = 0.1;
    let phi = unit ? 0 : 0.1;
    let completed = 0;
    let change = 0;
    let belowLimit = false;
    let width = 2;
    let design: number[][] = [];
    let firstResidual: number[] = [];
    let lagResidual: number[] = [];
    let variance = 0;
    let objective = (_parameter: number) => INVALID_OBJECTIVE;
    const states = new Set<string>();
    while (true) {
      if (!unit) {
        let within = 0;
        let between = 0;
        for (let row = 0; row < rowCount; row++) {
          const entity = row % entityCount;
          const deviation =
            deviations[row][columnCount] -
            rho * deviations[row][columnCount + 1] -
            getCPUErrorDot(deviations[row].slice(0, columnCount), beta);
          const mean =
            means[entity][columnCount] -
            rho * means[entity][columnCount + 1] -
            getCPUErrorDot(means[entity].slice(0, columnCount), beta);
          within += deviation ** 2;
          between += mean ** 2;
        }
        const estimate = between > 0 ? Math.sqrt(within / between / (periodCount - 1)) : 1;
        belowLimit = estimate < MINIMUM_PHI;
        const nextPhi = Math.max(MINIMUM_PHI, Math.min(estimate, 1));
        change = Math.abs(nextPhi - phi);
        phi = nextPhi;
      }
      design = designAt(phi);
      const target = raw.map((_, row) => datum(row, columnCount, phi));
      const lag = raw.map((_, row) => datum(row, columnCount + 1, phi));
      const firstFit = fitCPURegressionWithQR(design, target);
      const lagFit = fitCPURegressionWithQR(design, lag);
      if (!firstFit || !lagFit) {
        fit.status = 2;
        break;
      }
      firstResidual = target.map(
        (value, row) => value - getCPUErrorDot(design[row], firstFit.estimate)
      );
      lagResidual = lag.map((value, row) => value - getCPUErrorDot(design[row], lagFit.estimate));
      const getVariance = (parameter: number) =>
        firstResidual.reduce(
          (sum, value, row) => sum + (value - parameter * lagResidual[row]) ** 2,
          0
        ) / rowCount;
      objective = parameter => {
        const value = getVariance(parameter);
        return value > 0
          ? (rowCount / 2) * Math.log(value) -
              periodCount * getLogDeterminant(spatialMatrix(parameter))
          : INVALID_OBJECTIVE;
      };
      const search = minimizeProfile(objective);
      rho = search.parameter;
      width = search.width;
      beta = firstFit.estimate.map((value, column) => value - rho * lagFit.estimate[column]);
      variance = getVariance(rho);
      completed++;
      // A zero-RSS pole is not a finite ML solution. Refining the augmented system against
      // original rows can establish exact zeros even when two separately projected residuals
      // differ by a few binary64 ulps.
      const augmentedDesign = design.map((row, position) => [...row, lag[position]]);
      const augmentedFit = fitCPURegressionWithQR(augmentedDesign, target);
      const copiedColumn = augmentedDesign[0].findIndex((_, column) =>
        target.every((value, row) => value === augmentedDesign[row][column])
      );
      const exactCandidate =
        copiedColumn >= 0
          ? augmentedDesign[0].map((_, column) => Number(column === copiedColumn))
          : augmentedFit?.estimate;
      if (
        exactCandidate &&
        Math.abs(exactCandidate[activeCount]) <= RHO_BOUND &&
        target.every(
          (value, row) => value === getCPUErrorDot(augmentedDesign[row], exactCandidate)
        ) &&
        hasExactPanelFit(augmentedDesign, target, exactCandidate)
      ) {
        beta = exactCandidate.slice(0, activeCount);
        rho = exactCandidate[activeCount];
        variance = 0;
        objective = () => INVALID_OBJECTIVE;
        fit.status = 2;
        break;
      }
      const state = [phi, rho, ...beta].join(',');
      const converged =
        unit || change <= (options.tolerance ?? 1e-5) * Math.min(1, phi / RELATIVE_PHI);
      fit.status = converged && !belowLimit ? 0 : 3;
      if (converged || states.has(state)) break;
      states.add(state);
    }
    for (let column = 0; column < activeCount; column++)
      fit.coefficients[column + firstColumn] = beta[column];
    fit.coefficients[columnCount] = rho;
    fit.coefficients[columnCount + 1] = phi;
    if (!unit)
      fit.coefficients[0] =
        beta[0] + (1 - rho) * centers[0] - getCPUErrorDot(centers.slice(1), beta.slice(1));
    const publishedVariance = Math.fround(variance);
    if (!(publishedVariance > 0) || !Number.isFinite(publishedVariance)) fit.status = 2;
    fit.summary = [
      variance,
      -objective(rho) -
        (rowCount / 2) * (1 + Math.log(2 * Math.PI)) -
        (unit ? (rowCount / 2) * Math.log(rowCount) : 0),
      completed,
      change,
      width,
      Number(Math.abs(rho) > 0.9989) +
        (!unit && (phi >= 0.99999 || belowLimit) ? 2 : 0) +
        (belowLimit ? 4 : 0)
    ];
    if (fit.status !== 0 && fit.status !== 3) return fit;
    // Evaluate in centered coordinates before rounding any coefficient.
    for (let row = 0; row < rowCount; row++) {
      const predicted = rho * datum(row, columnCount + 1, phi) + getCPUErrorDot(design[row], beta);
      fit.fitted[row] = predicted + phi * centers[0];
      fit.residuals[row] = datum(row, columnCount, phi) - predicted;
    }
    if (fit.status !== 0) return fit;
    const inverse = invertCPUScaledMatrix(spatialMatrix(rho));
    if (!inverse) {
      fit.status = 2;
      fit.fitted.fill(NaN);
      fit.residuals.fill(NaN);
      return fit;
    }
    const product = dense.map(row =>
      row.map((_, column) =>
        row.reduce((sum, value, inner) => sum + value * inverse[inner][column], 0)
      )
    );
    let traceSquare = 0;
    let traceProduct = 0;
    for (let row = 0; row < entityCount; row++)
      for (let column = 0; column < entityCount; column++) {
        traceSquare += product[row][column] * product[column][row];
        traceProduct += product[row][column] ** 2;
      }
    const residualSquares = rowCount * variance;
    const residualLagProduct = firstResidual.reduce(
      (sum, value, row) => sum + (value - rho * lagResidual[row]) * lagResidual[row],
      0
    );
    const lagSquares = getCPUErrorDot(lagResidual, lagResidual);
    const ratio = residualLagProduct / residualSquares;
    const trace = product.reduce((sum, row, index) => sum + row[index], 0);
    const score = rowCount * ratio - periodCount * trace;
    const curvature =
      rowCount * (lagSquares / residualSquares - 2 * ratio ** 2) + periodCount * traceSquare;
    const allowance =
      1e-5 * Math.sqrt(Math.abs(curvature)) +
      64 *
        Number.EPSILON *
        (Math.abs(rowCount * ratio) + Math.abs(periodCount * trace) + Math.abs(curvature));
    const atLowerMaximum = rho === -RHO_BOUND && score <= allowance;
    const atUpperMaximum = rho === RHO_BOUND && score >= -allowance;
    if (!atLowerMaximum && !atUpperMaximum && !(curvature > 0 && Math.abs(score) <= allowance)) {
      fit.status = 3;
      return fit;
    }
    const informationSize = activeCount + 1;
    const information = Array.from({length: informationSize}, () =>
      new Array<number>(informationSize).fill(0)
    );
    for (let row = 0; row < rowCount; row++) {
      const entity = row % entityCount;
      const start = row - entity;
      const lagPrediction = product[entity].reduce(
        (sum, value, neighbor) => sum + value * getCPUErrorDot(design[start + neighbor], beta),
        0
      );
      const informationRow = [...design[row], lagPrediction];
      for (let first = 0; first < informationSize; first++)
        for (let second = 0; second < informationSize; second++)
          information[first][second] += (informationRow[first] * informationRow[second]) / variance;
    }
    information[activeCount][activeCount] += periodCount * (traceSquare + traceProduct);
    const centeredCovariance = invertPositiveMatrix(information);
    if (!centeredCovariance) {
      fit.status = 2;
      fit.fitted.fill(NaN);
      fit.residuals.fill(NaN);
      return fit;
    }
    const transform = information.map((_, row) =>
      information.map((_, column) => Number(row === column))
    );
    if (!unit) transform[0] = [1, ...centers.slice(1).map(value => -value), -centers[0]];
    for (let row = 0; row < informationSize; row++)
      for (let column = 0; column < informationSize; column++) {
        fit.covariance[row + firstColumn][column + firstColumn] = transform[row].reduce(
          (sum, value, first) =>
            sum + value * getCPUErrorDot(centeredCovariance[first], transform[column]),
          0
        );
      }
    return fit;
  };
  return finish(fitModel(options.effects === 'unit'), options.hausman ? fitModel(true) : undefined);
}

/** CPU twin; shader size/search caps are dropped. Prefer float64 for these small dense fits. */
export class CPUSpatialPanelLagMaximumLikelihoodOperation extends GPUCPUOperation {
  constructor(props: SpatialPanelLagMaximumLikelihoodOperationProps) {
    const id = props.id ?? 'spatial-panel-lag-maximum-likelihood';
    validateOptions(id, {...props, hausman: Boolean(props.hausman)});
    validateCapacity(id, 'entityCount', props.entityCount, 2);
    validateWeightsInformation(id, props.weights, props.entityCount);
    const operation = new GPUSpatialPanelLagMaximumLikelihoodOperation(props);
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
        const offsets = getCPUOperationUint32Rows(id, inputs, 'offsets');
        if (offsets.length !== props.entityCount + 1)
          throw new Error(`${id} weights.offsets length must equal entityCount + 1`);
        const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(
          {
            weights: {
              offsets,
              neighbors: getCPUOperationUint32Rows(id, inputs, 'neighbors'),
              weights: getCPUOperationFloat32Rows(id, inputs, 'weights'),
              distances:
                props.weights.distances !== undefined
                  ? getCPUOperationFloat32Rows(id, inputs, 'distances')
                  : undefined,
              information: props.weights.information,
              spatialContext: props.weights.spatialContext,
              status: props.weights.status
                ? Object.fromEntries(
                    Object.entries(props.weights.status)
                      .filter(([, value]) => value !== undefined)
                      .map(([field]) => [
                        field,
                        getCPUOperationUint32Rows(id, inputs, `weightsStatus_${field}`)
                      ])
                  )
                : undefined
            },
            overflow:
              props.weights.overflow && !props.weights.status?.overflow
                ? getCPUOperationUint32Rows(id, inputs, 'overflow')
                : undefined,
            predictors: getCPUOperationFloat32Rows(id, inputs, 'predictors'),
            response: getCPUOperationFloat32Rows(id, inputs, 'response')
          },
          {...props, hausman: Boolean(props.hausman)}
        );
        return {
          [props.output.coefficients]: result.coefficients,
          [props.output.covariance]: result.covariance,
          [props.output.summary]: result.summary,
          [props.output.status]: result.status,
          [props.output.fitted]: result.fitted,
          [props.output.residuals]: result.residuals,
          ...(props.hausman
            ? {
                [props.hausman.tests]: result.hausman!.tests,
                [props.hausman.status]: result.hausman!.status
              }
            : {})
        };
      }
    });
  }
}
