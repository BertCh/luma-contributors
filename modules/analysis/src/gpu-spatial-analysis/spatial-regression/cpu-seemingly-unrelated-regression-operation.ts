// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotSURIterationOutput} from './sur-iteration-output';
import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUSeeminglyUnrelatedRegressionOperation,
  type SeeminglyUnrelatedRegressionOperationProps
} from './gpu-seemingly-unrelated-regression-operation';
import {
  factorCPURegressionCorrelation,
  fitCPURegressionWithQR
} from './cpu-regression-linear-algebra';
import {
  validateCompleteRegressionWeights,
  type CPURegressionWeights
} from './cpu-regression-weights';
import {GPU_SPATIAL_ERROR_GM_LAMBDA_BOUND_FRACTION} from './spatial-error-gm-parameters';

/** Equation-major float32 rows, with common complete-case selection. */
export type CPUSeeminglyUnrelatedRegressionInput = {
  predictors: Float32Array;
  responses: Float32Array;
  mask?: Uint32Array;
  initialResiduals?: Float32Array;
  parameters?: Float32Array;
  spatial?: {weights: CPURegressionWeights};
};
/** Shape of the aligned equation-major observations. */
export type CPUSeeminglyUnrelatedRegressionOptions = Pick<
  SeeminglyUnrelatedRegressionOperationProps,
  'equationCount' | 'predictorCount' | 'rowCount' | 'iteration'
>;
/** Same scalar vector output layouts as the contributor, including optional spatial outputs. */
export type CPUSeeminglyUnrelatedRegressionResult = {
  coefficients: Float32Array;
  covariance: Float32Array;
  residualCovariance: Float32Array;
  residuals: Float32Array;
  status: Uint32Array;
  iterationCount: Uint32Array;
  spatial?: {parameters: Float32Array; status: Uint32Array};
};

type SURFloat64Input = {
  predictors: Float32Array | Float64Array;
  responses: Float32Array | Float64Array;
  mask?: Uint32Array;
  initialResiduals?: Float32Array | Float64Array;
  parameters?: Float32Array;
};
type SURFloat64Result = {
  coefficients: Float64Array;
  covariance: Float64Array;
  residualCovariance: Float64Array;
  residuals: Float64Array;
  status: Uint32Array;
  iterationCount: Uint32Array;
  spatial?: {parameters: Float64Array; status: Uint32Array};
};

function validateOptions(id: string, options: CPUSeeminglyUnrelatedRegressionOptions): void {
  validateCapacity(id, 'equationCount', options.equationCount, 2);
  validateCapacity(id, 'predictorCount', options.predictorCount);
  validateCapacity(id, 'rowCount', options.rowCount);
  if (options.iteration) {
    validateCapacity(id, 'iteration.maximumIterations', options.iteration.maximumIterations, 1);
    validateRequirement(
      id,
      Number.isFinite(options.iteration.tolerance ?? 1e-5) &&
        (options.iteration.tolerance ?? 1e-5) > 0,
      'iteration.tolerance must be a positive finite JavaScript number'
    );
  }
}

/**
 * Feasible GLS: equation OLS, joint residual covariance (divisor complete-case n),
 * Cholesky whitening and a Householder QR system fit. All work is float64 until output.
 * Optional iteration repeats covariance and GLS with exactly the caller's budget and tolerance.
 * The first GLS counts as round 1; status 4 retains the last fit on budget exhaustion. Both
 * covariance outputs are those used by the last round, not a post-fit covariance refresh.
 *
 * What differs from the GPU contributor:
 * - GPU tolerance must round to a normal positive float32, avoiding zero after rounding or
 *   subnormal flushing. CPU convenience tolerance accepts every positive finite JavaScript
 *   number, including smaller tolerances and values above the float32 range, in float64.
 *   Optional per-frame parameters retain the shared GPU float32 validity/layout convention.
 * - Spatial mode retains the contributor's OLS -> Kelejian-Prucha GM -> filtered joint GLS ->
 *   restored intercept/structural-residual sequence, entirely in float64. Its profiled quartic
 *   moment objective is minimized over both variance branches using all stationary points and
 *   branch boundaries, without the GPU's 1024-interval scan or 40-step refinement cap. Ties select
 *   the lowest lambda. Invalid spatial weights retain equation status 4 and NaN parameters.
 * - Float64 centered QR replaces float32 tile reductions and normal-equation solves. Residuals
 *   use centered coefficients before rounding. Compare well-conditioned outputs at 5e-3 relative
 *   tolerance; covariance error is scaled by the diagonal geometric mean.
 * - GPU equation, predictor, 16-system-coefficient and float32 row-count envelopes are dropped.
 *   Design rank uses norm-scaled QR pivots > 64 * Number.EPSILON; normalized error covariance
 *   squared Cholesky pivots use 64 * Number.EPSILON times the corresponding diagonal. GPU 1e-3
 *   Cholesky-pivot failures may therefore succeed on the CPU.
 * - Subnormals are retained. In nonspatial mode a non-finite value in any equation, including
 *   supplied initial residuals, excludes the aligned row. Spatial non-finite equation data returns
 *   equation status 3, except where insufficient-row status 2 or invalid-weight status 4 takes
 *   precedence. Error covariance is retained on one-step joint status 3; an iterated singular
 *   covariance clears numeric outputs to NaN, preserving the completed-round count.
 * - Spatial CSR shape mismatches throw by argument name. Invalid row offsets, links, ordering,
 *   or row sums covered by the GPU's weight-status check return equation status 4 instead.
 *   Status-bearing spatial weights are refused because these forms cannot consume loss ports.
 *
 * @throws If predictors, responses, mask, initialResiduals or spatial CSR arrays are inconsistent.
 */
export function computeSeeminglyUnrelatedRegressionOnCPU(
  input: CPUSeeminglyUnrelatedRegressionInput,
  options: CPUSeeminglyUnrelatedRegressionOptions
): CPUSeeminglyUnrelatedRegressionResult {
  const result = input.spatial
    ? computeSpatialSUR(input, options)
    : computeOrdinarySUR(input, options);
  return {
    coefficients: Float32Array.from(result.coefficients),
    covariance: Float32Array.from(result.covariance),
    residualCovariance: Float32Array.from(result.residualCovariance),
    residuals: Float32Array.from(result.residuals),
    status: result.status,
    iterationCount: result.iterationCount,
    ...(result.spatial
      ? {
          spatial: {
            parameters: Float32Array.from(result.spatial.parameters),
            status: result.spatial.status
          }
        }
      : {})
  };
}

/** Residuals below the arithmetic error floor of their response/fit are numerically zero.
 * The bound is relative to centered values, never an absolute variance cutoff.
 */
function computeSURResidual(
  response: number,
  design: readonly number[],
  coefficients: readonly number[]
): number {
  const prediction = design.reduce((sum, value, column) => sum + value * coefficients[column], 0);
  const magnitude =
    Math.abs(response) +
    design.reduce((sum, value, column) => sum + Math.abs(value * coefficients[column]), 0);
  const residual = response - prediction;
  return Math.abs(residual) <= 8 * design.length * Number.EPSILON * magnitude ? 0 : residual;
}

function computeOrdinarySUR(
  input: SURFloat64Input,
  options: CPUSeeminglyUnrelatedRegressionOptions
): SURFloat64Result {
  const id = 'computeSeeminglyUnrelatedRegressionOnCPU';
  validateOptions(id, options);
  const {equationCount, predictorCount, rowCount} = options;
  const coefficientWidth = predictorCount + 1;
  const coefficientCount = equationCount * coefficientWidth;
  if (input.predictors.length !== equationCount * rowCount * predictorCount)
    throw new Error(`${id} predictors length must equal equationCount * rowCount * predictorCount`);
  if (input.responses.length !== equationCount * rowCount)
    throw new Error(`${id} responses length must equal equationCount * rowCount`);
  if (input.mask && input.mask.length !== rowCount)
    throw new Error(`${id} mask length must equal rowCount`);
  if (input.initialResiduals && input.initialResiduals.length !== equationCount * rowCount)
    throw new Error(`${id} initialResiduals length must equal equationCount * rowCount`);
  validateRequirement(
    id,
    !input.parameters || Boolean(options.iteration),
    'parameters requires iteration; omit parameters for one-step SUR'
  );
  if (input.parameters && input.parameters.length !== 1)
    throw new Error(`${id} parameters length must equal 1`);
  const result: SURFloat64Result = {
    coefficients: new Float64Array(coefficientCount).fill(NaN),
    covariance: new Float64Array(coefficientCount ** 2).fill(NaN),
    residualCovariance: new Float64Array(equationCount ** 2).fill(NaN),
    residuals: new Float64Array(equationCount * rowCount).fill(NaN),
    status: Uint32Array.of(1),
    iterationCount: Uint32Array.of(0)
  };
  const tolerance = input.parameters?.[0] ?? options.iteration?.tolerance ?? 1e-5;
  if (input.parameters && (!Number.isFinite(tolerance) || tolerance < 2 ** -126)) {
    result.status[0] = 5;
    return result;
  }
  const usedRows: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    if (input.mask && input.mask[row] === 0) continue;
    let included = true;
    for (let equation = 0; equation < equationCount; equation++) {
      const index = equation * rowCount + row;
      included &&=
        Number.isFinite(input.responses[index]) &&
        (!input.initialResiduals || Number.isFinite(input.initialResiduals[index]));
      for (let column = 0; column < predictorCount; column++)
        included &&= Number.isFinite(input.predictors[index * predictorCount + column]);
    }
    if (included) usedRows.push(row);
  }
  const usedCount = usedRows.length;
  if (usedCount <= Math.max(coefficientWidth, equationCount)) return result;
  const means = Array.from({length: equationCount}, (_, equation) => {
    const origin = Array.from(
      {length: predictorCount},
      (_, column) => input.predictors[(equation * rowCount + usedRows[0]) * predictorCount + column]
    );
    return origin.map(
      (value, column) =>
        value +
        usedRows.reduce(
          (sum, row) =>
            sum + (input.predictors[(equation * rowCount + row) * predictorCount + column] - value),
          0
        ) /
          usedCount
    );
  });
  const responseMeans = Array.from({length: equationCount}, (_, equation) => {
    const origin = input.responses[equation * rowCount + usedRows[0]];
    return (
      origin +
      usedRows.reduce(
        (sum, row) => sum + (input.responses[equation * rowCount + row] - origin),
        0
      ) /
        usedCount
    );
  });
  const designs = means.map((equationMeans, equation) =>
    usedRows.map(row => [
      1,
      ...equationMeans.map(
        (mean, column) =>
          input.predictors[(equation * rowCount + row) * predictorCount + column] - mean
      )
    ])
  );
  const responses = responseMeans.map((mean, equation) =>
    usedRows.map(row => input.responses[equation * rowCount + row] - mean)
  );
  const initialResiduals: number[][] = [];
  let previousEstimates: number[] = [];
  const dot = (first: readonly number[], second: readonly number[]) =>
    first.reduce((sum, value, column) => sum + value * second[column], 0);
  for (let equation = 0; equation < equationCount; equation++) {
    const fit = fitCPURegressionWithQR(designs[equation], responses[equation]);
    if (!fit) {
      result.status[0] = 2;
      return result;
    }
    initialResiduals.push(
      usedRows.map((row, position) =>
        input.initialResiduals
          ? input.initialResiduals[equation * rowCount + row]
          : computeSURResidual(
              responses[equation][position],
              designs[equation][position],
              fit.estimate
            )
      )
    );
    previousEstimates.push(
      responseMeans[equation] + fit.estimate[0] - dot(means[equation], fit.estimate.slice(1)),
      ...fit.estimate.slice(1)
    );
  }
  for (let iteration = 0; iteration < (options.iteration?.maximumIterations ?? 1); iteration++) {
    const errorCovariance = initialResiduals.map(first =>
      initialResiduals.map(second => dot(first, second) / usedCount)
    );
    result.residualCovariance.set(errorCovariance.flat());
    const errorScale = Math.max(...errorCovariance.map((row, equation) => row[equation]));
    const factor =
      errorScale > 0
        ? factorCPURegressionCorrelation(
            errorCovariance.map(row => row.map(value => value / errorScale))
          )
        : undefined;
    if (!factor) {
      result.status[0] = 3;
      if (options.iteration) {
        result.coefficients.fill(NaN);
        result.covariance.fill(NaN);
        result.residualCovariance.fill(NaN);
        result.residuals.fill(NaN);
      }
      return result;
    }
    const whitenedDesign: number[][] = [];
    const whitenedResponse: number[] = [];
    for (let position = 0; position < usedCount; position++) {
      const block: number[][] = [];
      const blockResponse: number[] = [];
      for (let equation = 0; equation < equationCount; equation++) {
        const values = new Array<number>(coefficientCount).fill(0);
        values.splice(
          equation * coefficientWidth,
          coefficientWidth,
          ...designs[equation][position]
        );
        let value = responses[equation][position];
        for (let previous = 0; previous < equation; previous++) {
          value -= factor[equation][previous] * blockResponse[previous];
          for (let column = 0; column < coefficientCount; column++)
            values[column] -= factor[equation][previous] * block[previous][column];
        }
        block.push(values.map(entry => entry / factor[equation][equation]));
        blockResponse.push(value / factor[equation][equation]);
      }
      whitenedDesign.push(...block);
      whitenedResponse.push(...blockResponse);
    }
    const fit = fitCPURegressionWithQR(whitenedDesign, whitenedResponse);
    if (!fit) {
      result.status[0] = 2;
      if (options.iteration) {
        result.coefficients.fill(NaN);
        result.covariance.fill(NaN);
        result.residualCovariance.fill(NaN);
        result.residuals.fill(NaN);
      }
      return result;
    }
    const inverseRoot = fit.inverseRoot.map(row => row.slice());
    const estimates = fit.estimate.slice();
    for (let equation = 0; equation < equationCount; equation++) {
      const intercept = equation * coefficientWidth;
      estimates[intercept] += responseMeans[equation];
      for (let column = 0; column < predictorCount; column++) {
        estimates[intercept] -= means[equation][column] * fit.estimate[intercept + column + 1];
        for (let target = 0; target < coefficientCount; target++)
          inverseRoot[intercept][target] -=
            means[equation][column] * fit.inverseRoot[intercept + column + 1][target];
      }
      usedRows.forEach((row, position) => {
        result.residuals[equation * rowCount + row] = computeSURResidual(
          responses[equation][position],
          designs[equation][position],
          fit.estimate.slice(intercept, intercept + coefficientWidth)
        );
      });
    }
    result.coefficients.set(estimates);
    result.covariance.set(
      inverseRoot.flatMap(first => inverseRoot.map(second => errorScale * dot(first, second)))
    );
    result.status[0] = 0;
    result.iterationCount[0]++;
    if (!options.iteration) break;
    const change = Math.max(
      ...estimates.map(
        (value, index) =>
          Math.abs(value - previousEstimates[index]) /
          Math.max(1, Math.abs(previousEstimates[index]))
      )
    );
    // A supplied covariance may reproduce OLS without matching the fitted residual covariance.
    if ((!input.initialResiduals || iteration > 0) && change < tolerance) break;
    result.status[0] = 4;
    previousEstimates = estimates;
    for (let equation = 0; equation < equationCount; equation++) {
      initialResiduals[equation] = usedRows.map(row => result.residuals[equation * rowCount + row]);
    }
  }
  return result;
}

function getPolynomialCandidates(
  coefficients: readonly number[],
  lower: number,
  upper: number
): number[] {
  const polynomial = Array.from(coefficients);
  while (polynomial.length > 1 && polynomial[polynomial.length - 1] === 0) polynomial.pop();
  const scale = Math.max(...polynomial.map(Math.abs));
  if (!(scale > 0) || polynomial.length === 1) return [lower, upper];
  for (let index = 0; index < polynomial.length; index++) polynomial[index] /= scale;
  if (polynomial.length === 2) {
    const root = -polynomial[0] / polynomial[1];
    return root > lower && root < upper ? [lower, root, upper] : [lower, upper];
  }
  const derivative = polynomial.slice(1).map((value, index) => value * (index + 1));
  const partitions = getPolynomialCandidates(derivative, lower, upper).sort(
    (first, second) => first - second
  );
  const evaluate = (value: number) =>
    polynomial.reduceRight((sum, coefficient) => sum * value + coefficient, 0);
  const candidates = partitions.slice();
  for (let interval = 1; interval < partitions.length; interval++) {
    let left = partitions[interval - 1];
    let right = partitions[interval];
    let leftValue = evaluate(left);
    const rightValue = evaluate(right);
    if (leftValue === 0 || rightValue === 0 || Math.sign(leftValue) === Math.sign(rightValue))
      continue;
    while (true) {
      const middle = left + (right - left) / 2;
      if (middle === left || middle === right) {
        candidates.push(Math.abs(evaluate(left)) <= Math.abs(evaluate(right)) ? left : right);
        break;
      }
      const value = evaluate(middle);
      if (value === 0) {
        candidates.push(middle);
        break;
      }
      if (Math.sign(value) === Math.sign(leftValue)) {
        left = middle;
        leftValue = value;
      } else right = middle;
    }
  }
  return [...new Set(candidates)];
}

function minimizeSpatialMomentObjective(
  residuals: readonly number[],
  multiply: (values: readonly number[]) => number[],
  trace: number,
  bound: number
): number {
  const dot = (first: readonly number[], second: readonly number[]) =>
    first.reduce((sum, value, row) => sum + value * second[row], 0);
  const lagged = multiply(residuals);
  const twiceLagged = multiply(lagged);
  const squares = dot(residuals, residuals);
  if (!(squares > 0) || !Number.isFinite(squares)) return NaN;
  const lagSquares = dot(lagged, lagged);
  const lagProduct = dot(residuals, lagged);
  const lagTwiceProduct = dot(lagged, twiceLagged);
  const coefficients = [
    [1, lagSquares / squares, lagProduct / squares],
    [
      (-2 * lagProduct) / squares,
      (-2 * lagTwiceProduct) / squares,
      -(dot(residuals, twiceLagged) + lagSquares) / squares
    ],
    [lagSquares / squares, dot(twiceLagged, twiceLagged) / squares, lagTwiceProduct / squares]
  ];
  if (!coefficients.flat().every(Number.isFinite)) return NaN;
  const varianceColumn = [1, trace / residuals.length, 0];
  const varianceNorm = dot(varianceColumn, varianceColumn);
  const variancePolynomial = coefficients.map(row => dot(row, varianceColumn) / varianceNorm);
  const projected = coefficients.map((row, degree) =>
    row.map((value, index) => value - varianceColumn[index] * variancePolynomial[degree])
  );
  const candidates = getPolynomialCandidates(variancePolynomial, -bound, bound);
  // Both branches are quartics. Their cubic derivatives and the variance=0 boundaries
  // contain every possible minimum; extra derivative partition points are harmless candidates.
  for (const branch of [coefficients, projected]) {
    const derivative = new Array<number>(4).fill(0);
    for (let first = 0; first < 3; first++) {
      for (let second = 0; second < 3; second++) {
        const degree = first + second;
        if (degree > 0) derivative[degree - 1] += degree * dot(branch[first], branch[second]);
      }
    }
    candidates.push(...getPolynomialCandidates(derivative, -bound, bound));
  }
  let best = -bound;
  let bestObjective = Infinity;
  for (const parameter of candidates.sort((first, second) => first - second)) {
    const free = coefficients[0].map(
      (value, index) =>
        value + parameter * (coefficients[1][index] + parameter * coefficients[2][index])
    );
    const variance = Math.max(dot(free, varianceColumn) / varianceNorm, 0);
    const objective = free.reduce(
      (sum, value, index) => sum + (value - variance * varianceColumn[index]) ** 2,
      0
    );
    if (objective < bestObjective) {
      bestObjective = objective;
      best = parameter;
    }
  }
  return Number.isFinite(bestObjective) ? best : NaN;
}

function computeSpatialSUR(
  input: CPUSeeminglyUnrelatedRegressionInput,
  options: CPUSeeminglyUnrelatedRegressionOptions
): SURFloat64Result {
  const id = 'computeSeeminglyUnrelatedRegressionOnCPU';
  validateOptions(id, options);
  validateCompleteRegressionWeights(id, input.spatial!.weights);
  const {rowCount, equationCount, predictorCount} = options;
  validateRequirement(
    id,
    !input.mask && !input.initialResiduals,
    'spatial requires no mask or initialResiduals'
  );
  if (input.predictors.length !== equationCount * rowCount * predictorCount)
    throw new Error(`${id} predictors length must equal equationCount * rowCount * predictorCount`);
  if (input.responses.length !== equationCount * rowCount)
    throw new Error(`${id} responses length must equal equationCount * rowCount`);
  const {offsets, neighbors, weights} = input.spatial!.weights;
  if (offsets.length !== rowCount + 1)
    throw new Error(`${id} spatial.weights.offsets length must equal rowCount + 1`);
  if (weights.length !== neighbors.length)
    throw new Error(
      `${id} spatial.weights.weights length must equal spatial.weights.neighbors length`
    );
  let validWeights = offsets[0] === 0;
  let norm = 0;
  let trace = 0;
  const rowTotals = new Float64Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    const begin = offsets[row];
    const end = offsets[row + 1];
    if (end < begin || end > neighbors.length) validWeights = false;
    let total = 0;
    let absoluteTotal = 0;
    let rowSquares = 0;
    for (let slot = begin; slot < Math.min(end, neighbors.length); slot++) {
      const neighbor = neighbors[slot];
      const weight = weights[slot];
      if (
        neighbor >= rowCount ||
        neighbor === row ||
        !Number.isFinite(weight) ||
        weight < 0 ||
        (slot > begin && neighbors[slot - 1] >= neighbor)
      )
        validWeights = false;
      total += weight;
      if (neighbor < rowCount) {
        absoluteTotal += Math.abs(weight);
        rowSquares += weight * weight;
      }
    }
    if (Math.abs(total - 1) > 1e-5) validWeights = false;
    rowTotals[row] = total > 0 && Number.isFinite(total) ? total : 1;
    trace += rowSquares / rowTotals[row] ** 2;
    norm = Math.max(norm, absoluteTotal / rowTotals[row]);
  }
  const multiply = (values: readonly number[]) =>
    Array.from({length: rowCount}, (_, row) => {
      let value = 0;
      for (let slot = offsets[row]; slot < Math.min(offsets[row + 1], neighbors.length); slot++) {
        if (neighbors[slot] < rowCount)
          value += (weights[slot] / rowTotals[row]) * values[neighbors[slot]];
      }
      return value;
    });
  const parameters = new Float64Array(equationCount).fill(NaN);
  const statuses = new Uint32Array(equationCount);
  const filteredPredictors = new Float64Array(input.predictors.length);
  const filteredResponses = new Float64Array(input.responses.length);
  const filteredResiduals = new Float64Array(input.responses.length);
  for (let equation = 0; equation < equationCount; equation++) {
    const response = Array.from(
      input.responses.subarray(equation * rowCount, (equation + 1) * rowCount)
    );
    const design = Array.from({length: rowCount}, (_, row) => [
      1,
      ...input.predictors.subarray(
        (equation * rowCount + row) * predictorCount,
        (equation * rowCount + row + 1) * predictorCount
      )
    ]);
    const origin = response[0];
    const shiftedResponse = response.map(value => value - origin);
    const finite = response.every(Number.isFinite) && design.flat().every(Number.isFinite);
    const fit =
      finite && rowCount > predictorCount + 1
        ? fitCPURegressionWithQR(design, shiftedResponse)
        : undefined;
    const residuals = fit
      ? design.map((row, position) =>
          computeSURResidual(shiftedResponse[position], row, fit.estimate)
        )
      : new Array<number>(rowCount).fill(NaN);
    let status = !validWeights
      ? 4
      : rowCount <= predictorCount + 2
        ? 2
        : !finite
          ? 3
          : !fit
            ? 1
            : 0;
    let parameter =
      status === 0
        ? minimizeSpatialMomentObjective(
            residuals,
            multiply,
            trace,
            GPU_SPATIAL_ERROR_GM_LAMBDA_BOUND_FRACTION / norm
          )
        : NaN;
    if (status === 0 && !Number.isFinite(parameter)) status = 3;
    if (status === 0) {
      const laggedColumns = design[0].map((_, column) => multiply(design.map(row => row[column])));
      const filteredDesign = design.map((row, position) =>
        row.map((value, column) => value - parameter * laggedColumns[column][position])
      );
      const laggedResponse = multiply(shiftedResponse);
      if (
        !fitCPURegressionWithQR(
          filteredDesign,
          shiftedResponse.map((value, row) => value - parameter * laggedResponse[row])
        )
      )
        status = 1;
    }
    if (!validWeights) status = 4;
    if (status !== 0) parameter = NaN;
    parameters[equation] = parameter;
    statuses[equation] = status;
    const packed = [
      shiftedResponse,
      residuals,
      ...design[0].slice(1).map((_, column) => design.map(row => row[column + 1]))
    ];
    for (let column = 0; column < packed.length; column++) {
      for (let row = 0; row < rowCount; row++) {
        // An island has no neighbor terms; invalid lambda must still suppress its row.
        let value = Number.isFinite(parameter) ? packed[column][row] : NaN;
        // Keep the shader's ordered filter, including defined NaN propagation after status 4.
        for (let slot = offsets[row]; slot < Math.min(offsets[row + 1], neighbors.length); slot++) {
          if (neighbors[slot] < rowCount)
            value -= parameter * (weights[slot] / rowTotals[row]) * packed[column][neighbors[slot]];
        }
        if (column === 0) filteredResponses[equation * rowCount + row] = value;
        else if (column === 1) filteredResiduals[equation * rowCount + row] = value;
        else filteredPredictors[(equation * rowCount + row) * predictorCount + column - 2] = value;
      }
    }
  }
  const result = computeOrdinarySUR(
    {
      predictors: filteredPredictors,
      responses: filteredResponses,
      initialResiduals: filteredResiduals,
      parameters: input.parameters
    },
    options
  );
  const width = predictorCount + 1;
  const size = equationCount * width;
  const scales = Array.from({length: size}, (_, index) =>
    index % width === 0 ? 1 / (1 - parameters[Math.floor(index / width)]) : 1
  );
  for (let first = 0; first < size; first++) {
    result.coefficients[first] *= scales[first];
    if (first % width === 0)
      result.coefficients[first] += input.responses[Math.floor(first / width) * rowCount];
    for (let second = 0; second < size; second++)
      result.covariance[first * size + second] *= scales[first] * scales[second];
  }
  for (let equation = 0; equation < equationCount; equation++) {
    for (let row = 0; row < rowCount; row++) {
      let residual =
        input.responses[equation * rowCount + row] - result.coefficients[equation * width];
      for (let column = 0; column < predictorCount; column++)
        residual -=
          input.predictors[(equation * rowCount + row) * predictorCount + column] *
          result.coefficients[equation * width + column + 1];
      result.residuals[equation * rowCount + row] = residual;
    }
  }
  result.spatial = {parameters, status: statuses};
  return result;
}

/** CPU spatial/nonspatial SUR twin, without shader size caps. */
export class CPUSeeminglyUnrelatedRegressionOperation extends GPUCPUOperation {
  constructor(props: SeeminglyUnrelatedRegressionOperationProps) {
    const id = props.id ?? 'seemingly-unrelated-regression';
    props = {
      ...props,
      iteration: props.iteration ? {...props.iteration} : undefined,
      output: snapshotSURIterationOutput(id, props.output),
      spatial: props.spatial
        ? {...props.spatial, weights: {...props.spatial.weights}, output: {...props.spatial.output}}
        : undefined
    };
    validateOptions(id, props);
    validateRequirement(
      id,
      !props.parameters || Boolean(props.iteration),
      'parameters requires iteration; omit parameters for one-step SUR'
    );
    validateRequirement(
      id,
      !props.spatial || (!props.mask && !props.initialResiduals),
      'spatial requires no mask or initialResiduals'
    );
    const operation = new GPUSeeminglyUnrelatedRegressionOperation(props);
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
        const result = computeSeeminglyUnrelatedRegressionOnCPU(
          {
            predictors: getCPUOperationFloat32Rows(id, inputs, 'predictors'),
            responses: getCPUOperationFloat32Rows(id, inputs, 'responses'),
            mask: props.mask ? getCPUOperationUint32Rows(id, inputs, 'mask') : undefined,
            initialResiduals: props.initialResiduals
              ? getCPUOperationFloat32Rows(id, inputs, 'initialResiduals')
              : undefined,
            parameters: props.parameters
              ? getCPUOperationFloat32Rows(id, inputs, 'parameters')
              : undefined,
            spatial: props.spatial
              ? {
                  weights: {
                    offsets: getCPUOperationUint32Rows(id, inputs, 'spatialOffsets'),
                    neighbors: getCPUOperationUint32Rows(id, inputs, 'spatialNeighbors'),
                    weights: getCPUOperationFloat32Rows(id, inputs, 'spatialWeights')
                  }
                }
              : undefined
          },
          props
        );
        return {
          [props.output.coefficients]: result.coefficients,
          [props.output.covariance]: result.covariance,
          [props.output.residualCovariance]: result.residualCovariance,
          [props.output.residuals]: result.residuals,
          [props.output.status]: result.status,
          ...(props.output.iteration?.status.iterationCount
            ? {[props.output.iteration?.status.iterationCount]: result.iterationCount}
            : {}),
          ...(props.spatial
            ? {
                [props.spatial.output.parameters]: result.spatial!.parameters,
                [props.spatial.output.status]: result.spatial!.status
              }
            : {})
        };
      }
    });
  }
}
