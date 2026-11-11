// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
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
  GPUProbitRegressionOperation,
  type ProbitRegressionOperationProps
} from './gpu-probit-regression-operation';
import {
  factorCPURegressionCorrelation,
  solveCPURegressionFactor
} from './cpu-regression-linear-algebra';
import type {CPURegressionWeights} from './cpu-regression-weights';
import {getCPUErrorDot as getDotProduct, getCPUErrorMean} from './cpu-spatial-error-regression';
import {getChiSquareSurvival} from './ordinary-least-squares-statistics';

/** Packed complete-case observations, with optional whole-CSR residual diagnostics. */
export type CPUProbitRegressionInput = {
  predictors: Float32Array;
  response: Float32Array;
  mask?: Uint32Array;
  spatialDiagnostics?: {weights: CPURegressionWeights; overflow?: Uint32Array};
};
/** Fisher scoring settings. The caller's iteration budget remains a stopping rule. */
export type CPUProbitRegressionOptions = Pick<
  ProbitRegressionOperationProps,
  'predictorCount' | 'maximumIterations' | 'tolerance'
>;
/** Scalar vectors in the contributor's layouts; diagnostic outputs exist only when requested. */
export type CPUProbitRegressionResult = {
  coefficients: Float32Array;
  standardErrors: Float32Array;
  fitted: Float32Array;
  marginalEffects: Float32Array;
  summary: Float32Array;
  status: Uint32Array;
  spatialDiagnostics?: {tests: Float32Array; status: Uint32Array};
};

function validateOptions(id: string, options: CPUProbitRegressionOptions): void {
  validateCapacity(id, 'predictorCount', options.predictorCount);
  validateCapacity(id, 'maximumIterations', options.maximumIterations ?? 64);
  validateFiniteNumber(id, 'tolerance', options.tolerance ?? 1e-5, 0);
  validateRequirement(id, (options.tolerance ?? 1e-5) > 0, 'tolerance must be positive');
}

function getDensity(value: number): number {
  return Math.exp(-0.5 * value * value) / Math.sqrt(2 * Math.PI);
}

function getProbability(value: number): number {
  if (!Number.isFinite(value)) return value > 0 ? 1 : value < 0 ? 0 : NaN;
  if (Math.abs(value) > 38) return value > 0 ? 1 : 0;
  const tail = 0.5 * getChiSquareSurvival(value * value, 1);
  return value < 0 ? tail : 1 - tail;
}

// Laplace's continued fraction of the inverse Mills ratio avoids underflow in the far tail.
function getMillsRatio(value: number): number {
  if (value >= -30) return getDensity(value) / getProbability(value);
  let remainder = 0;
  for (let term = 64; term >= 1; term--) remainder = term / (-value + remainder);
  return -value + remainder;
}

function getLogProbability(value: number): number {
  if (value < -30)
    return -0.5 * value * value - 0.5 * Math.log(2 * Math.PI) - Math.log(getMillsRatio(value));
  return value > 0 ? Math.log1p(-getProbability(-value)) : Math.log(getProbability(value));
}

// An exact dyadic expansion certifies the real affine direction on the raw float32 inputs.
type Dyadic = {significand: bigint; exponent: number};
function getDyadic(value: number): Dyadic {
  if (value === 0) return {significand: 0n, exponent: 0};
  const storage = new DataView(new ArrayBuffer(8));
  storage.setFloat64(0, value);
  const bits = storage.getBigUint64(0);
  const biasedExponent = Number((bits >> 52n) & 2047n);
  const significand = (bits & ((1n << 52n) - 1n)) | (biasedExponent ? 1n << 52n : 0n);
  return {
    significand: bits >> 63n ? -significand : significand,
    exponent: (biasedExponent || 1) - 1075
  };
}
function getSum(terms: Dyadic[]): Dyadic {
  const exponent = Math.min(...terms.map(term => term.exponent));
  return {
    exponent,
    significand: terms.reduce(
      (sum, term) => sum + (term.significand << BigInt(term.exponent - exponent)),
      0n
    )
  };
}
function getProduct(first: Dyadic, second: Dyadic): Dyadic {
  return {
    significand: first.significand * second.significand,
    exponent: first.exponent + second.exponent
  };
}
function certifyDirection(
  raw: number[][],
  sides: number[],
  means: number[],
  scales: number[],
  direction: number[]
): boolean {
  if (direction.some(value => !Number.isFinite(value))) return false;
  const intercept = getDyadic(direction[0]);
  const slopes = scales.map((scale, column) =>
    getProduct(getDyadic(scale), getDyadic(direction[column + 1]))
  );
  const origin = means.map(value => getDyadic(-value));
  let movesRight = false;
  for (let row = 0; row < raw.length; row++) {
    const terms = [
      intercept,
      ...slopes.map((slope, column) =>
        getProduct(slope, getSum([getDyadic(raw[row][column]), origin[column]]))
      )
    ];
    const signedMove = getSum(terms).significand * BigInt(sides[row]);
    if (signedMove < 0n) return false;
    if (signedMove > 0n) movesRight = true;
  }
  return movesRight;
}

function certifyColumn(raw: number[][], response: number[]): boolean {
  for (let column = 0; column < raw[0].length; column++) {
    const minimum = [Infinity, Infinity];
    const maximum = [-Infinity, -Infinity];
    raw.forEach((row, position) => {
      const observed = response[position];
      minimum[observed] = Math.min(minimum[observed], row[column]);
      maximum[observed] = Math.max(maximum[observed], row[column]);
    });
    // A threshold between the classes, including a shared support value, is an exact certificate.
    if (
      (maximum[0] <= minimum[1] && minimum[0] < maximum[1]) ||
      (maximum[1] <= minimum[0] && minimum[1] < maximum[0])
    )
      return true;
    // Quasi-separation at either extreme, with arbitrary outcomes on the tied rows.
    for (const threshold of [Math.min(...minimum), Math.max(...maximum)]) {
      for (const sign of [-1, 1]) {
        let movesRight = false;
        let movesWrong = false;
        raw.forEach((row, position) => {
          const move = sign * (2 * response[position] - 1) * Math.sign(row[column] - threshold);
          movesRight ||= move > 0;
          movesWrong ||= move < 0;
        });
        if (movesRight && !movesWrong) return true;
      }
    }
  }
  return false;
}

/**
 * Binary probit by centered/scaled Fisher scoring and monotone likelihood backtracking.
 * Only status 0 publishes expected-information errors. Status 3 publishes the last iterate;
 * statuses 1, 2 and 4 leave coefficients, predictions and effects NaN. Complete cases are shared
 * by all outputs; nonbinary complete responses return status 4 before recording a row count.
 *
 * What differs from the GPU contributor:
 * - Float64 means, information, centered predictions, densities and diagnostics are rounded once
 *   on output. Well-conditioned comparisons use 5e-4 relative tolerance (iterations are discrete).
 * - Shader row/predictor/iteration caps and the 24-halving cap are dropped. The requested iteration
 *   budget and tolerance remain; backtracking stops when the candidate no longer changes.
 * - Stable float64 log tails and inverse Mills ratios remove the wrong-side bound of 12 and the
 *   float32 likelihood noise allowance. A bound-pinned GPU status 3 can converge here.
 * - Singular pivots use the shared float64 floor, 64 * Number.EPSILON of the information diagonal,
 *   instead of the GPU's 1e-3 pivot limit. Some GPU status 2 fits succeed here.
 * - Single-column separation is tested over the entire class supports before iteration. General
 *   Fisher directions (including the GPU's cleaned candidate) are certified with exact dyadic
 *   arithmetic on raw inputs. Certificates may be found sooner; an unclassified, nearly separating
 *   direction blocks status 0. The candidate search is still incomplete for multiple columns.
 * - Diagnostic probabilities stay float64 until output, with the same [1e-7, 1-1e-7] clamp.
 *   P-values call the public chi-square survival helper. Subnormals are retained.
 *
 * @throws If predictors, mask or diagnostic arrays have inconsistent lengths, naming the argument.
 */
export function computeProbitRegressionOnCPU(
  input: CPUProbitRegressionInput,
  options: CPUProbitRegressionOptions
): CPUProbitRegressionResult {
  const id = 'computeProbitRegressionOnCPU';
  validateOptions(id, options);
  const predictorCount = options.predictorCount;
  const coefficientCount = predictorCount + 1;
  const rowCount = input.response.length;
  if (input.predictors.length !== rowCount * predictorCount)
    throw new Error(`${id} predictors length must equal response length * predictorCount`);
  if (input.mask && input.mask.length !== rowCount)
    throw new Error(`${id} mask length must equal response length`);
  if (input.spatialDiagnostics) {
    if (input.spatialDiagnostics.weights.offsets.length !== rowCount + 1)
      throw new Error(
        `${id} spatialDiagnostics.weights.offsets length must equal response length + 1`
      );
    if (
      input.spatialDiagnostics.weights.weights.length !==
      input.spatialDiagnostics.weights.neighbors.length
    )
      throw new Error(
        `${id} spatialDiagnostics.weights.weights length must equal neighbors length`
      );
    if (input.spatialDiagnostics.overflow && input.spatialDiagnostics.overflow.length !== 1)
      throw new Error(`${id} spatialDiagnostics.overflow must hold one value`);
  }
  const result: CPUProbitRegressionResult = {
    coefficients: new Float32Array(coefficientCount).fill(NaN),
    standardErrors: new Float32Array(coefficientCount).fill(NaN),
    fitted: new Float32Array(rowCount).fill(NaN),
    marginalEffects: new Float32Array(predictorCount).fill(NaN),
    summary: new Float32Array(4).fill(NaN),
    status: new Uint32Array([1]),
    ...(input.spatialDiagnostics
      ? {spatialDiagnostics: {tests: new Float32Array(6).fill(NaN), status: new Uint32Array([1])}}
      : {})
  };
  const usedRows: number[] = [];
  const raw: number[][] = [];
  const response: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    if (input.mask && input.mask[row] === 0) continue;
    const values = Array.from(
      input.predictors.subarray(row * predictorCount, (row + 1) * predictorCount)
    );
    if (!Number.isFinite(input.response[row]) || values.some(value => !Number.isFinite(value)))
      continue;
    usedRows.push(row);
    raw.push(values);
    response.push(input.response[row]);
  }
  if (response.some(value => value !== 0 && value !== 1)) {
    result.status[0] = 4;
    return result;
  }
  const count = raw.length;
  result.summary[0] = count;
  result.summary[3] = 0;
  if (count <= coefficientCount) return result;
  if (response.every(value => value === response[0])) {
    result.status[0] = 4;
    return result;
  }
  const means = raw[0].map((_, column) => getCPUErrorMean(raw.map(row => row[column])));
  const scales = means.map((mean, column) =>
    Math.sqrt(count / raw.reduce((sum, row) => sum + (row[column] - mean) ** 2, 0))
  );
  if (scales.some(value => !Number.isFinite(value))) {
    result.status[0] = 2;
    return result;
  }
  const design = raw.map(row => [
    1,
    ...row.map((value, column) => (value - means[column]) * scales[column])
  ]);
  const sides = response.map(value => 2 * value - 1);
  let separating = certifyColumn(raw, response);
  let undecided = false;
  let estimate = new Array<number>(coefficientCount).fill(0);
  let likelihood = -count * Math.log(2);
  let iterations = 0;
  let factor: number[][] | undefined;
  result.status[0] = 3;
  for (let iteration = 0; iteration <= (options.maximumIterations ?? 64); iteration++) {
    const gradient = estimate.map(() => 0);
    const information = estimate.map(() => estimate.map(() => 0));
    let hasOverflow = false;
    for (let row = 0; row < count; row++) {
      const linear = getDotProduct(design[row], estimate);
      if (!Number.isFinite(linear)) {
        hasOverflow = true;
        break;
      }
      const ratio = getMillsRatio(sides[row] * linear);
      const weight = ratio * getMillsRatio(-sides[row] * linear);
      for (let first = 0; first < coefficientCount; first++) {
        gradient[first] += sides[row] * ratio * design[row][first];
        for (let second = 0; second < coefficientCount; second++)
          information[first][second] += weight * design[row][first] * design[row][second];
      }
    }
    if (hasOverflow) {
      result.status[0] = 3;
      break;
    }
    factor = factorCPURegressionCorrelation(information);
    if (!factor) {
      result.status[0] = 2;
      break;
    }
    if (result.status[0] === 0 || iteration === (options.maximumIterations ?? 64)) break;
    const direction = solveCPURegressionFactor(factor, gradient);
    const extents = scales.map((_, column) =>
      raw.reduce(
        (extent, row) => Math.max(extent, Math.abs((row[column] - means[column]) * scales[column])),
        0
      )
    );
    const contributions = extents.map((extent, column) => extent * Math.abs(direction[column + 1]));
    const noiseLevel = 1e-4 * Math.max(...contributions);
    const candidate = direction.map((value, column) =>
      column === 0 || contributions[column - 1] >= noiseLevel ? value : 0
    );
    const corners = [-1, 1].map(
      sign =>
        candidate[0] +
        scales.reduce((sum, scale, column) => {
          const extreme = raw.reduce(
            (value, row) =>
              sign < 0 ? Math.min(value, row[column]) : Math.max(value, row[column]),
            sign < 0 ? Infinity : -Infinity
          );
          return sum + candidate[column + 1] * (extreme - means[column]) * scale;
        }, 0)
    );
    if (Math.abs(corners[0]) <= noiseLevel && Math.abs(corners[0]) <= Math.abs(corners[1]))
      candidate[0] -= corners[0];
    else if (Math.abs(corners[1]) <= noiseLevel) candidate[0] -= corners[1];
    separating ||=
      certifyDirection(raw, sides, means, scales, direction) ||
      certifyDirection(raw, sides, means, scales, candidate);
    let movesRight = false;
    let movesWrong = false;
    let ambiguous = false;
    for (let row = 0; row < count; row++) {
      const move = sides[row] * getDotProduct(design[row], candidate);
      const magnitude = design[row].reduce(
        (sum, value, column) => sum + Math.abs(value * candidate[column]),
        0
      );
      const bound = (predictorCount + 5) * Number.EPSILON * magnitude * 4;
      if (move > bound) movesRight = true;
      else if (move < -bound) movesWrong = true;
      else if (magnitude !== 0) ambiguous = true;
    }
    undecided ||= movesRight && !movesWrong && ambiguous;
    let scale = 1;
    let accepted = false;
    let nextEstimate = estimate;
    while (scale > 0) {
      nextEstimate = estimate.map((value, column) => value + scale * direction[column]);
      if (nextEstimate.every((value, column) => value === estimate[column])) {
        accepted =
          Math.max(...direction.map(Math.abs)) <=
          (options.tolerance ?? 1e-5) * (1 + Math.max(...estimate.map(Math.abs)));
        break;
      }
      const nextLikelihood = design.reduce(
        (sum, row, position) =>
          sum + getLogProbability(sides[position] * getDotProduct(row, nextEstimate)),
        0
      );
      if (
        Number.isFinite(nextLikelihood) &&
        nextLikelihood >= likelihood - 16 * Number.EPSILON * (1 + Math.abs(likelihood))
      ) {
        likelihood = nextLikelihood;
        accepted = true;
        break;
      }
      scale *= 0.5;
    }
    if (!accepted) break;
    estimate = nextEstimate;
    iterations++;
    const largestStep = Math.max(...direction.map(value => Math.abs(scale * value)));
    const magnitude = Math.max(...estimate.map(Math.abs));
    if (largestStep <= (options.tolerance ?? 1e-5) * (1 + magnitude)) {
      // A tiny backtracked step alone is not a maximum: require the unshortened score step too.
      if (Math.max(...direction.map(Math.abs)) > (options.tolerance ?? 1e-5) * (1 + magnitude))
        break;
      if (!separating && !undecided) result.status[0] = 0;
      else break;
    }
  }
  if (separating) result.status[0] = 4;
  if (result.status[0] !== 0 && result.status[0] !== 3) return result;
  const slopes = scales.map((scale, column) => scale * estimate[column + 1]);
  result.coefficients.set([estimate[0] - getDotProduct(means, slopes), ...slopes]);
  result.summary.set([count, likelihood, -2 * likelihood, iterations]);
  const probabilities = design.map(row => {
    const linear = getDotProduct(row, estimate);
    return Number.isFinite(linear)
      ? Math.min(1 - 1e-7, Math.max(1e-7, getProbability(linear)))
      : NaN;
  });
  const densities = design.map(row => getDensity(getDotProduct(row, estimate)));
  probabilities.forEach((value, row) => {
    result.fitted[usedRows[row]] = value;
  });
  const meanDensity = densities.reduce((sum, value) => sum + value, 0) / count;
  result.marginalEffects.set(slopes.map(value => value * meanDensity));
  if (result.status[0] === 0 && factor) {
    const transform = [1, ...means.map((mean, column) => -mean * scales[column])];
    result.standardErrors[0] = Math.sqrt(
      Math.max(0, getDotProduct(transform, solveCPURegressionFactor(factor, transform)))
    );
    scales.forEach((scale, column) => {
      const basis = estimate.map((_, position) => (position === column + 1 ? 1 : 0));
      result.standardErrors[column + 1] =
        scale * Math.sqrt(solveCPURegressionFactor(factor!, basis)[column + 1]);
    });
  }
  if (input.spatialDiagnostics && result.spatialDiagnostics && result.status[0] === 0) {
    computeDiagnostics(
      input.spatialDiagnostics,
      usedRows,
      response,
      probabilities,
      densities,
      result.spatialDiagnostics
    );
  }
  return result;
}

function computeDiagnostics(
  input: NonNullable<CPUProbitRegressionInput['spatialDiagnostics']>,
  usedRows: number[],
  response: number[],
  probabilities: number[],
  densities: number[],
  result: NonNullable<CPUProbitRegressionResult['spatialDiagnostics']>
): void {
  const {offsets, neighbors, weights} = input.weights;
  result.status[0] = 2;
  if (input.overflow?.[0]) return;
  const positions = new Map(usedRows.map((row, position) => [row, position]));
  const variance = probabilities.map(value => value * (1 - value));
  const naive = response.map((value, row) => value - probabilities[row]);
  const generalized = naive.map((value, row) => (value * densities[row]) / variance[row]);
  const standardized = naive.map((value, row) => value / Math.sqrt(variance[row]));
  let trace = 0;
  let varianceTrace = 0;
  let naiveProduct = 0;
  let generalizedProduct = 0;
  let standardizedProduct = 0;
  for (let position = 0; position < usedRows.length; position++) {
    const row = usedRows[position];
    if (offsets[row] > offsets[row + 1] || offsets[row + 1] > neighbors.length) return;
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
      const neighbor = neighbors[slot];
      const weight = weights[slot];
      if (
        neighbor >= offsets.length - 1 ||
        neighbor === row ||
        !Number.isFinite(weight) ||
        (slot > offsets[row] && neighbors[slot - 1] >= neighbor)
      )
        return;
      const otherPosition = positions.get(neighbor);
      if (otherPosition === undefined) continue;
      let reciprocal = 0;
      for (
        let other = offsets[neighbor];
        other < Math.min(offsets[neighbor + 1], neighbors.length);
        other++
      )
        if (neighbors[other] === row) reciprocal = weights[other];
      const contribution = weight * (weight + reciprocal);
      trace += contribution;
      varianceTrace += contribution * variance[position] * variance[otherPosition];
      naiveProduct += weight * naive[position] * naive[otherPosition];
      generalizedProduct += weight * generalized[position] * generalized[otherPosition];
      standardizedProduct += weight * standardized[position] * standardized[otherPosition];
    }
  }
  result.status[0] = 3;
  const information =
    densities.reduce((sum, value, row) => sum + (value * value) / variance[row], 0) /
    usedRows.length;
  if (!(trace > 0 && varianceTrace > 0 && information > 0)) return;
  const pinkse = (generalizedProduct / information) ** 2 / trace;
  const kelejian = naiveProduct / Math.sqrt(varianceTrace);
  const slade = standardizedProduct ** 2 / trace;
  result.tests.set([
    pinkse,
    getChiSquareSurvival(pinkse, 1),
    kelejian,
    getChiSquareSurvival(kelejian ** 2, 1),
    slade,
    getChiSquareSurvival(slade, 1)
  ]);
  result.status[0] = 0;
}

/** CPU twin; no GPU shape caps. The caller's iteration budget and output layouts are retained. */
export class CPUProbitRegressionOperation extends GPUCPUOperation {
  constructor(props: ProbitRegressionOperationProps) {
    const id = props.id ?? 'probit-regression';
    validateOptions(id, props);
    validateCapacity(id, 'rowCount', props.rowCount);
    const operation = new GPUProbitRegressionOperation(props);
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
        const result = computeProbitRegressionOnCPU(
          {
            predictors: getCPUOperationFloat32Rows(id, inputs, 'predictors'),
            response,
            mask: props.mask ? getCPUOperationUint32Rows(id, inputs, 'mask') : undefined,
            spatialDiagnostics: props.spatialDiagnostics
              ? {
                  weights: {
                    offsets: getCPUOperationUint32Rows(id, inputs, 'offsets'),
                    neighbors: getCPUOperationUint32Rows(id, inputs, 'neighbors'),
                    weights: getCPUOperationFloat32Rows(id, inputs, 'weights')
                  },
                  overflow: props.spatialDiagnostics.weights.overflow
                    ? getCPUOperationUint32Rows(id, inputs, 'overflow')
                    : undefined
                }
              : undefined
          },
          props
        );
        return {
          ...Object.fromEntries(
            Object.entries(props.output).map(([name, value]) => [
              value,
              result[name as keyof typeof props.output]
            ])
          ),
          ...(props.spatialDiagnostics && result.spatialDiagnostics
            ? {
                [props.spatialDiagnostics.output.tests]: result.spatialDiagnostics.tests,
                [props.spatialDiagnostics.output.status]: result.spatialDiagnostics.status
              }
            : {})
        };
      }
    });
  }
}
