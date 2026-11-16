// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {solveCPUEquilibratedSystem} from '../../analysis-core/kernels/cpu-equilibrated-solve';
import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {getCPUVecchiaCorrelation} from '../spatial-interpolation/cpu-vecchia-evaluation';
import {KRIGING_PIVOT_RATIO_TOLERANCE} from '../spatial-interpolation/kriging-parameters';
import {getPredictionCovarianceModelCode} from '../spatial-interpolation/prediction-covariance';
import type {GPUNestedInterpolationCandidate} from './gpu-nested-interpolation-validation';
import {
  GPUNestedInterpolationValidationOperation,
  type NestedInterpolationValidationOperationProps
} from './gpu-nested-interpolation-validation-operation';
import {computeSpatialCrossValidationOnCPU} from './cpu-spatial-cross-validation-operation';

const NONE = 0xffffffff;
/** Counts in scores and metrics are float32; every row count must remain exactly representable. */
export const CPU_NESTED_INTERPOLATION_VALIDATION_MAXIMUM_ROW_COUNT = 2 ** 24;

/** Packed inputs; outer masks are fold-major and nonzero means included. */
export type CPUNestedInterpolationValidationInput = {
  positions: Float32Array;
  response: Float32Array;
  training: Uint32Array;
  testing: Uint32Array;
  innerBlocks: Uint32Array;
};
/** Numeric model and split settings, with no GPU work caps. */
export type CPUNestedInterpolationValidationOptions = Pick<
  NestedInterpolationValidationOperationProps,
  | 'id'
  | 'foldCount'
  | 'innerFoldCount'
  | 'innerBufferDistance'
  | 'seed'
  | 'neighborCount'
  | 'candidates'
>;
/** Same five arrays and scalar row layouts as GPUNestedInterpolationValidation. */
export type CPUNestedInterpolationValidationResult = {
  selected: Uint32Array;
  scores: Float32Array;
  predictions: Float32Array;
  metrics: Float32Array;
  status: Uint32Array;
};

function getRBFModel(candidate: Extract<GPUNestedInterpolationCandidate, {mode: 'rbf'}>) {
  const kernel = candidate.kernel ?? 'thin-plate-spline';
  const degree =
    candidate.degree ??
    (kernel === 'quintic'
      ? 2
      : ['thin-plate-spline', 'cubic'].includes(kernel)
        ? 1
        : ['linear', 'multiquadric'].includes(kernel)
          ? 0
          : -1);
  const polynomialCount = degree < 0 ? 0 : degree === 0 ? 1 : degree === 1 ? 3 : 6;
  const homogeneousDegree =
    kernel === 'thin-plate-spline'
      ? 2
      : kernel === 'cubic'
        ? 3
        : kernel === 'quintic'
          ? 5
          : kernel === 'linear'
            ? 1
            : 0;
  return {kernel, degree, polynomialCount, homogeneousDegree};
}

function validateFiniteOption(id: string, name: string, value: number, minimum = 0): void {
  // validateFiniteNumber tests float32 representability; these options are evaluated in float64.
  validateRequirement(
    id,
    Number.isFinite(value) && value >= minimum,
    `${name} must be finite and at least ${minimum}`
  );
}

function validateOptions(options: CPUNestedInterpolationValidationOptions, rowCount: number): void {
  const id = options.id ?? 'nested-interpolation-validation';
  validateCapacity(
    id,
    'rowCount',
    rowCount,
    3,
    CPU_NESTED_INTERPOLATION_VALIDATION_MAXIMUM_ROW_COUNT
  );
  validateCapacity(id, 'foldCount', options.foldCount, 2);
  validateCapacity(id, 'innerFoldCount', options.innerFoldCount ?? 3, 2);
  validateCapacity(id, 'neighborCount', options.neighborCount ?? 16, 2);
  validateCapacity(id, 'candidates.length', options.candidates.length, 1);
  validateCapacity(id, 'seed', options.seed ?? 0, 0);
  validateFiniteOption(id, 'innerBufferDistance', options.innerBufferDistance ?? 0);
  for (const [name, length] of [
    ['predictions length', rowCount * options.foldCount],
    ['inner masks length', rowCount * (options.innerFoldCount ?? 3)],
    ['scores length', options.foldCount * options.candidates.length * 3],
    ['metrics length', options.foldCount * 6]
  ] as const)
    validateMaximumCount(id, name, length, NONE);
  options.candidates.forEach((candidate, index) => {
    const name = `candidates[${index}]`;
    validateRequirement(
      id,
      ['ordinary-kriging', 'rbf'].includes(candidate.mode),
      `${name}.mode must be ordinary-kriging or rbf`
    );
    if (candidate.mode === 'ordinary-kriging') {
      const covariance = candidate.covariance;
      validateRequirement(
        id,
        ['spherical', 'exponential', 'gaussian', 'matern', 'matern-3/2', 'matern-5/2'].includes(
          covariance.model
        ),
        `${name}.covariance.model must name a supported covariance`
      );
      validateRequirement(
        id,
        covariance.model !== 'matern' || [0.5, 1.5, 2.5].includes(covariance.smoothness!),
        `${name}.covariance.smoothness must be 0.5, 1.5 or 2.5 for matern`
      );
      validateFiniteOption(id, `${name}.covariance.range`, covariance.range);
      validateFiniteOption(id, `${name}.covariance.sill`, covariance.sill);
      validateFiniteOption(id, `${name}.covariance.nugget`, covariance.nugget ?? 0);
      validateRequirement(
        id,
        covariance.range > 0 && covariance.sill + (covariance.nugget ?? 0) > 0,
        `${name}.covariance.range and total variance must be positive`
      );
      validateFiniteOption(
        id,
        `${name}.covariance total variance`,
        covariance.sill + (covariance.nugget ?? 0)
      );
    } else {
      const {kernel, degree} = getRBFModel(candidate);
      validateRequirement(
        id,
        [
          'thin-plate-spline',
          'cubic',
          'gaussian',
          'linear',
          'quintic',
          'multiquadric',
          'inverse-multiquadric',
          'inverse-quadratic'
        ].includes(kernel),
        `${name}.kernel must name a supported RBF kernel`
      );
      validateRequirement(
        id,
        [-1, 0, 1, 2].includes(degree) &&
          (!['thin-plate-spline', 'cubic'].includes(kernel) || degree >= 1) &&
          (!['linear', 'multiquadric'].includes(kernel) || degree >= 0) &&
          (kernel !== 'quintic' || degree >= 2),
        `${name}.degree must satisfy the kernel polynomial degree (-1 through 2)`
      );
      validateFiniteOption(id, `${name}.epsilon`, candidate.epsilon);
      validateRequirement(id, candidate.epsilon > 0, `${name}.epsilon must be positive`);
      validateFiniteOption(id, `${name}.smoothing`, candidate.smoothing ?? 0);
    }
  });
}

function getRadialValue(kernel: string, separation: number): number {
  switch (kernel) {
    case 'thin-plate-spline':
      return separation > 0 ? separation ** 2 * Math.log(separation) : 0;
    case 'cubic':
      return separation ** 3;
    case 'gaussian':
      return Math.exp(-(separation ** 2));
    case 'quintic':
      return -(separation ** 5);
    case 'multiquadric':
      return -Math.hypot(1, separation);
    case 'inverse-multiquadric':
      return 1 / Math.hypot(1, separation);
    case 'inverse-quadratic':
      return 1 / (1 + separation ** 2);
    default:
      return -separation;
  }
}

function predictRow(
  input: CPUNestedInterpolationValidationInput,
  mask: Uint32Array,
  query: number,
  candidate: GPUNestedInterpolationCandidate,
  neighborCount: number
): number {
  const {positions, response} = input;
  const distance = (first: number, second: number) =>
    Math.hypot(
      positions[first * 2] - positions[second * 2],
      positions[first * 2 + 1] - positions[second * 2 + 1]
    );
  if (!Number.isFinite(positions[query * 2]) || !Number.isFinite(positions[query * 2 + 1]))
    return NaN;
  const neighbors = Array.from({length: response.length}, (_, row) => row)
    .filter(
      row =>
        mask[row] !== 0 &&
        Number.isFinite(response[row]) &&
        Number.isFinite(positions[row * 2]) &&
        Number.isFinite(positions[row * 2 + 1])
    )
    .sort((first, second) => distance(first, query) - distance(second, query) || first - second)
    .slice(0, neighborCount);
  const retained = neighbors.length;
  if (retained === 0) return NaN;
  if (candidate.mode === 'ordinary-kriging') {
    const {covariance} = candidate;
    const varianceScale = covariance.sill + (covariance.nugget ?? 0);
    const modelCode = getPredictionCovarianceModelCode(covariance);
    const evaluateCovariance = (first: number, second: number) =>
      (covariance.sill / varianceScale) *
        getCPUVecchiaCorrelation(modelCode, distance(first, second) / covariance.range) +
      (positions[first * 2] === positions[second * 2] &&
      positions[first * 2 + 1] === positions[second * 2 + 1]
        ? (covariance.nugget ?? 0) / varianceScale
        : 0);
    const size = retained + 1;
    const matrix = new Float64Array(size * size);
    const right = new Float64Array(size);
    neighbors.forEach((sample, row) => {
      neighbors.forEach((other, column) => {
        matrix[row * size + column] = evaluateCovariance(sample, other);
      });
      matrix[row * size + retained] = matrix[retained * size + row] = 1;
      right[row] = evaluateCovariance(sample, query);
    });
    right[retained] = 1;
    const {solution} = solveCPUEquilibratedSystem(matrix, right, KRIGING_PIVOT_RATIO_TOLERANCE);
    if (!solution) return NaN;
    let prediction = 0;
    let variance = 1;
    neighbors.forEach((sample, row) => {
      prediction += solution[row] * response[sample];
      variance -= solution[row] * right[row];
    });
    variance -= solution[retained];
    return Number.isFinite(prediction) &&
      Number.isFinite(variance) &&
      variance >= -1e-4 * Math.max(1, 1e-20 / varianceScale)
      ? prediction
      : NaN;
  }
  const {kernel, polynomialCount, homogeneousDegree} = getRBFModel(candidate);
  if (retained < polynomialCount) return NaN;
  const coordinateScale = Math.max(distance(neighbors[retained - 1], query), 1e-20);
  const radial = (first: number, second: number) =>
    getRadialValue(
      kernel,
      (homogeneousDegree ? distance(first, second) / coordinateScale : distance(first, second)) *
        candidate.epsilon
    );
  let smoothing = candidate.smoothing ?? 0;
  // Repeated division avoids overflowing scale^degree when smoothing/scale^degree is finite.
  for (let power = 0; power < homogeneousDegree; power++) smoothing /= coordinateScale;
  const size = retained + polynomialCount;
  const matrix = new Float64Array(size * size);
  const right = new Float64Array(size);
  neighbors.forEach((sample, row) => {
    neighbors.forEach((other, column) => {
      matrix[row * size + column] = radial(sample, other);
    });
    matrix[row * size + row] += smoothing;
    right[row] = response[sample];
    const positionX = (positions[sample * 2] - positions[query * 2]) / coordinateScale;
    const positionY = (positions[sample * 2 + 1] - positions[query * 2 + 1]) / coordinateScale;
    const polynomial = [
      1,
      positionX,
      positionY,
      positionX ** 2,
      positionX * positionY,
      positionY ** 2
    ];
    for (let column = 0; column < polynomialCount; column++)
      matrix[row * size + retained + column] = matrix[(retained + column) * size + row] =
        polynomial[column];
  });
  const {solution} = solveCPUEquilibratedSystem(matrix, right, KRIGING_PIVOT_RATIO_TOLERANCE);
  if (!solution) return NaN;
  let prediction = polynomialCount ? solution[retained] : 0;
  neighbors.forEach((sample, row) => {
    prediction += solution[row] * radial(query, sample);
  });
  return Number.isFinite(prediction) ? prediction : NaN;
}

/**
 * Selects the first candidate with minimum complete inner held-out MSE, refits on the complete
 * outer training mask, and publishes only outer testing predictions. Overlap rejects the entire
 * outer fold. Scores are [MSE, scored, missing]; metrics are [scored, missing, MSE, MAE, bias, R²].
 * Status bits are 1 overlap, 2 no selected candidate, 4 missing held-out predictions, 8 empty
 * held-out set, 16 nonfinite metric arithmetic. R² is NaN for constant observed values.
 *
 * What differs from the GPU contributor:
 * - All fits, scores and metrics use float64, rounding only the five returned arrays. Near score,
 *   neighbor or rank thresholds selection/status can differ; compare finite outputs with relative
 *   tolerance 5e-4 and absolute tolerance 2e-5 for values near zero.
 * - GPU work caps (4096 tiled / 256 reference rows, 16 outer folds/candidates, 8 inner folds, 32 neighbors) are removed.
 *   The neighborCount hyperparameter still selects the nearest requested number, with row-index
 *   ties. uint32 output-length limits and the 2^24 exact float32 count limit remain.
 * - Matern uses the exact closed form rather than the GPU's short-lag truncated Taylor series;
 *   thin-plate-spline uses log(r) without the GPU's 1e-30 floor. The GPU's equilibrated pivot
 *   and variance rejection tolerances are retained.
 * - Float64 permits scores/distances beyond float32 range, without the GPU's finite best-score
 *   sentinel or 1e18 buffer cap. Status 16 concerns float64 arithmetic; final float32 overflow
 *   can produce infinity. Nonfinite coordinates/responses are omitted as on GPU; a nonfinite
 *   solve produces a missing prediction. Subnormals are preserved rather than flushed to zero.
 *
 * @throws If positions, response, training, testing or innerBlocks have inconsistent lengths,
 * or a numeric option or candidate model is invalid.
 */
export function computeNestedInterpolationValidationOnCPU(
  input: CPUNestedInterpolationValidationInput,
  options: CPUNestedInterpolationValidationOptions
): CPUNestedInterpolationValidationResult {
  const rowCount = input.response.length;
  if (input.positions.length !== rowCount * 2)
    throw new Error('positions length must equal response length * 2');
  if (input.innerBlocks.length !== rowCount)
    throw new Error('innerBlocks length must equal response length');
  for (const name of ['training', 'testing'] as const)
    if (input[name].length !== rowCount * options.foldCount)
      throw new Error(`${name} length must equal response length * foldCount`);
  validateOptions(options, rowCount);
  const {foldCount, candidates} = options;
  const innerFoldCount = options.innerFoldCount ?? 3;
  const neighborCount = options.neighborCount ?? 16;
  const selected = new Uint32Array(foldCount).fill(NONE);
  const scores = new Float64Array(foldCount * candidates.length * 3).fill(NaN);
  const predictions = new Float64Array(foldCount * rowCount).fill(NaN);
  const metrics = new Float64Array(foldCount * 6).fill(NaN);
  const status = new Uint32Array(foldCount);
  for (let fold = 0; fold < foldCount; fold++) {
    const training = input.training.subarray(fold * rowCount, (fold + 1) * rowCount);
    const testing = input.testing.subarray(fold * rowCount, (fold + 1) * rowCount);
    const overlap = training.some((value, row) => value !== 0 && testing[row] !== 0);
    status[fold] = Number(overlap);
    const mask = Uint32Array.from(training, (value, row) =>
      Number(value !== 0 && !overlap && Number.isFinite(input.response[row]))
    );
    const inner = computeSpatialCrossValidationOnCPU(
      {positions: input.positions, blocks: input.innerBlocks, mask},
      {
        id: options.id,
        foldCount: innerFoldCount,
        seed: options.seed,
        bufferDistance: options.innerBufferDistance
      }
    );
    let best = Infinity;
    candidates.forEach((candidate, candidateIndex) => {
      let squares = 0;
      let count = 0;
      let missing = 0;
      for (let split = 0; split < innerFoldCount; split++) {
        const innerTraining = inner.training.subarray(split * rowCount, (split + 1) * rowCount);
        for (let row = 0; row < rowCount; row++) {
          if (inner.testing[split * rowCount + row] === 0) continue;
          const predicted = predictRow(input, innerTraining, row, candidate, neighborCount);
          if (!Number.isFinite(predicted) || !Number.isFinite(input.response[row])) {
            missing++;
            continue;
          }
          squares += (predicted - input.response[row]) ** 2;
          count++;
        }
      }
      const score = count > 0 && missing === 0 ? squares / count : NaN;
      scores.set([score, count, missing], (fold * candidates.length + candidateIndex) * 3);
      if (Number.isFinite(score) && score < best) {
        best = score;
        selected[fold] = candidateIndex;
      }
    });
    if (selected[fold] === NONE) status[fold] |= 2;
    else
      for (let row = 0; row < rowCount; row++)
        if (testing[row] !== 0)
          predictions[fold * rowCount + row] = predictRow(
            input,
            mask,
            row,
            candidates[selected[fold]],
            neighborCount
          );
    let count = 0;
    let missing = 0;
    let squares = 0;
    let absolute = 0;
    let bias = 0;
    let mean = 0;
    let total = 0;
    for (let row = 0; row < rowCount; row++) {
      if (testing[row] === 0) continue;
      const predicted = predictions[fold * rowCount + row];
      const observed = input.response[row];
      if (!Number.isFinite(predicted) || !Number.isFinite(observed)) {
        missing++;
        continue;
      }
      count++;
      const error = predicted - observed;
      squares += error * error;
      absolute += Math.abs(error);
      bias += error;
      const difference = observed - mean;
      mean += difference / count;
      total += difference * (observed - mean);
    }
    const coefficientOfDetermination = 1 - squares / total;
    const numericFailure =
      ![squares, absolute, bias, mean, total].every(Number.isFinite) ||
      (total > 0 && !Number.isFinite(coefficientOfDetermination));
    metrics.set(
      [
        count,
        missing,
        count > 0 && !numericFailure ? squares / count : NaN,
        count > 0 && !numericFailure ? absolute / count : NaN,
        count > 0 && !numericFailure ? bias / count : NaN,
        total > 0 && !numericFailure ? coefficientOfDetermination : NaN
      ],
      fold * 6
    );
    status[fold] |=
      (missing > 0 ? 4 : 0) | (count + missing === 0 ? 8 : 0) | (numericFailure ? 16 : 0);
  }
  return {
    selected,
    scores: Float32Array.from(scores),
    predictions: Float32Array.from(predictions),
    metrics: Float32Array.from(metrics),
    status
  };
}

/** CPU twin without GPU work caps; prefer it for the serial nested search. */
export class CPUNestedInterpolationValidationOperation extends GPUCPUOperation {
  constructor(props: NestedInterpolationValidationOperationProps) {
    const id = props.id ?? 'nested-interpolation-validation';
    validateOptions(props, props.rowCount);
    const operation = new GPUNestedInterpolationValidationOperation(props);
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
        const result = computeNestedInterpolationValidationOnCPU(
          {
            positions: getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2'),
            response,
            training: getCPUOperationUint32Rows(id, inputs, 'training'),
            testing: getCPUOperationUint32Rows(id, inputs, 'testing'),
            innerBlocks: getCPUOperationUint32Rows(id, inputs, 'innerBlocks')
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUNestedInterpolationValidationResult]
          ])
        );
      }
    });
  }
}
