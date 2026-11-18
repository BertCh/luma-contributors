// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotRegressionOperationProps} from './regression-construction-snapshot';

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPU_REGRESSION_PRECISION_CONDITION_LIMIT} from '../../analysis-core/contracts/model-result';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUSpatialTwoStageLeastSquaresOperation,
  type SpatialTwoStageLeastSquaresOperationProps
} from './gpu-spatial-two-stage-least-squares-operation';
import {
  validateInstrumentalVariablesCovarianceOptions,
  INSTRUMENTAL_VARIABLES_COVARIANCE_LEVERAGE_TOLERANCE,
  INSTRUMENTAL_VARIABLES_COVARIANCE_PSD_TOLERANCE,
  type GPUInstrumentalVariablesCovarianceMethod
} from './instrumental-variables-covariance-parameters';
import {
  factorCPURegressionCorrelation,
  fitCPURegressionWithQR
} from './cpu-regression-linear-algebra';
import {
  validateCPURegressionWeights,
  validateCompleteRegressionWeights,
  type CPURegressionWeights
} from './cpu-regression-weights';
import {
  getCPUErrorMean,
  getCPUErrorDot,
  getCPUErrorConditioning,
  multiplyCPUErrorWeights
} from './cpu-spatial-error-regression';
import {getChiSquareSurvival} from './ordinary-least-squares-statistics';
import {
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE
} from './spatial-two-stage-least-squares-parameters';

/** Packed observations; no automatic case deletion. Invalid neighbor sentinels are skipped. */
export type CPUSpatialTwoStageLeastSquaresInput = {
  weights: CPURegressionWeights;
  predictors: Float32Array;
  response: Float32Array;
  robust?: {
    method: GPUInstrumentalVariablesCovarianceMethod;
    kernel?: CPURegressionWeights;
    overflow?: Uint32Array;
    clusters?: Uint32Array;
    clusterCount?: number;
  };
};
/** Instrument order is a model definition and remains 1 or 2. */
export type CPUSpatialTwoStageLeastSquaresOptions = Pick<
  SpatialTwoStageLeastSquaresOperationProps,
  'predictorCount' | 'instrumentOrder'
>;
/** Exact scalar-vector layouts of the contributor, including all classical diagnostics. */
export type CPUSpatialTwoStageLeastSquaresResult = {
  status: Uint32Array;
  table: Float32Array;
  summary: Float32Array;
  residuals: Float32Array;
  /** Covariance status: 0 success, 1 finite rank failure, 2 insufficient rows,
   * 3 non-finite failure, 4 invalid HAC kernel, 5 invalid leverage, 6 invalid clusters, 7 non-PSD covariance. Failure covariance entries are NaN. */
  robust?: {covariance: Float32Array; status: Uint32Array};
};

/** Certifies a covariance directly, without running the CPU model fit. @internal */
export function getCPUInstrumentalVariablesCovarianceStatus(
  covariance: readonly number[][]
): number {
  // CPU twins do not inherit the GPU width cap. Avoid flattening the matrix or passing its
  // entries as function arguments: a 402-by-402 covariance already exceeds Node's stack limit.
  let magnitude = 0;
  for (const row of covariance) {
    for (const value of row) magnitude = Math.max(magnitude, Math.abs(value));
  }
  if (!Number.isFinite(magnitude)) return 3;
  if (magnitude > 0) {
    const shifted = covariance.map((row, first) =>
      row.map(
        (value, second) =>
          value / magnitude +
          (first === second ? INSTRUMENTAL_VARIABLES_COVARIANCE_PSD_TOLERANCE : 0)
      )
    );
    if (!factorCPURegressionCorrelation(shifted)) return 7;
  }
  return 0;
}

function validateOptions(id: string, options: CPUSpatialTwoStageLeastSquaresOptions): void {
  validateCapacity(id, 'predictorCount', options.predictorCount);
  validateRequirement(
    id,
    options.instrumentOrder === undefined ||
      options.instrumentOrder === 1 ||
      options.instrumentOrder === 2,
    'instrumentOrder must be 1 or 2'
  );
}

function getInverse(root: number[][]): number[][] {
  return root.map(first => root.map(second => getCPUErrorDot(first, second)));
}
function multiplyMatrixRows(matrix: number[][], values: number[]): number[] {
  return matrix.map(row => getCPUErrorDot(row, values));
}

function isKernelValid(kernel: CPURegressionWeights, overflow?: Uint32Array): boolean {
  if (overflow?.[0]) return false;
  const rowCount = kernel.offsets.length - 1;
  for (let row = 0; row < rowCount; row++) {
    if (
      kernel.offsets[row] > kernel.offsets[row + 1] ||
      kernel.offsets[row + 1] > kernel.neighbors.length
    )
      return false;
    let diagonal = 0;
    for (let slot = kernel.offsets[row]; slot < kernel.offsets[row + 1]; slot++) {
      const neighbor = kernel.neighbors[slot];
      const weight = kernel.weights[slot];
      if (neighbor >= rowCount || !Number.isFinite(weight) || weight < 0) return false;
      if (
        kernel.offsets[neighbor] > kernel.offsets[neighbor + 1] ||
        kernel.offsets[neighbor + 1] > kernel.neighbors.length
      )
        return false;
      if (neighbor === row) diagonal += weight;
      let forward = 0;
      let reverse = 0;
      for (let other = kernel.offsets[row]; other < kernel.offsets[row + 1]; other++)
        if (kernel.neighbors[other] === neighbor) forward += kernel.weights[other];
      for (let other = kernel.offsets[neighbor]; other < kernel.offsets[neighbor + 1]; other++)
        if (kernel.neighbors[other] === row) reverse += kernel.weights[other];
      if (Math.abs(reverse - forward) > 1e-5 * Math.max(1, Math.abs(forward))) return false;
    }
    if (Math.abs(diagonal - 1) > 1e-5) return false;
  }
  return true;
}

/**
 * Spatial lag IV with instruments [1,X,WX(,W²X)], using centered Householder QR projections.
 * The structural residuals and every diagnostic use the float64 estimate before publication.
 * Moran's I is NaN at zero total weight; the cancelled Anselin-Kelejian statistic remains defined.
 *
 * What differs from the GPU contributor:
 * - Float64 sparse lags, centered QR projection, solves and residuals replace float32 lags,
 *   normal equations and one refinement step. Rounding occurs once per returned output. Compare
 *   ordinary well-conditioned fits at 5e-3 relative tolerance (5e-4 for table/residuals).
 * - Shader predictor/instrument caps, tile budget and float32 row-count limits are dropped.
 *   Instrument order remains 1 or 2. Rank uses norm-scaled QR pivots > 64 * Number.EPSILON.
 * - The ordinary comparison tolerances do not bound ill-conditioned instrument systems. For
 *   adjacent-pair weights and x=[-1,-1+epsilon,1,1,-1,-1,1,1-epsilon], epsilon=2^-24, the
 *   instrument correlation condition number is about 2.25e15. The GPU accepted the fit with
 *   status 0 but its intercept differed by 2.63% from the exact IV intercept (4.5 after rounding).
 *   The CPU retains the float64 QR estimate; GPU status 0 with a precision warning does not
 *   imply agreement of coefficients, covariance, residuals or derived diagnostics in this case.
 * - Conditioning uses the shared float64 Jacobi singular values without the GPU sweep cap.
 *   Probability calculations call the public chi-square survival helper, retaining its cap.
 * - HC0, HC1 and HAC use float64 influence rows and structural residuals, avoiding the GPU's
 *   second rounding of raw lag columns and residuals before robust covariance. Kernel symmetry
 *   and diagonal tolerances remain 1e-5; PSD remains a caller precondition.
 * - HC2/HC3 leverage and the shifted PSD test use float64 with the same 1e-6 and 1e-5
 *   tolerances as the GPU. CLUSTER sums scores in input order, rather than the GPU segment tree.
 * - Non-finite active data/weights return solver status 1, matching the current kernel's failed
 *   solve branch (despite its status-3 documentation); skipped neighbor sentinels ignore weights.
 *   Subnormals are retained. Inconsistent model CSR arrays throw rather than reading outside
 *   storage; malformed diagnostic/HAC row bounds return their defined failure statuses.
 * - Insufficient-row fits return NaN residuals. The GPU publishes its residual pass before
 *   selecting status 2 and does not clear those rows; a full-rank saturated fit can therefore
 *   leave finite residuals (zero to rounding for y=X) despite its failed table and summary.
 *
 * @throws If predictors, weights or robust kernel arrays are inconsistent, naming the argument.
 */
export function computeSpatialTwoStageLeastSquaresOnCPU(
  input: CPUSpatialTwoStageLeastSquaresInput,
  options: CPUSpatialTwoStageLeastSquaresOptions
): CPUSpatialTwoStageLeastSquaresResult {
  const id = 'computeSpatialTwoStageLeastSquaresOnCPU';
  validateOptions(id, options);
  const rowCount = input.response.length;
  const predictorCount = options.predictorCount;
  const coefficientCount = predictorCount + 2;
  const instrumentCount = ((options.instrumentOrder ?? 1) + 1) * predictorCount;
  if (validateCPURegressionWeights(id, input.weights) !== rowCount)
    throw new Error(`${id} weights.offsets length must equal response length + 1`);
  if (input.predictors.length !== rowCount * predictorCount)
    throw new Error(`${id} predictors length must equal response length * predictorCount`);
  if (input.robust) {
    validateInstrumentalVariablesCovarianceOptions(id, rowCount, input.robust);
    if (input.robust.clusters && input.robust.clusters.length !== rowCount)
      throw new Error(`${id} robust.clusters length must equal response length`);
    if (input.robust.kernel && input.robust.kernel.offsets.length !== rowCount + 1)
      throw new Error(`${id} robust.kernel.offsets length must equal response length + 1`);
    if (
      input.robust.kernel &&
      input.robust.kernel.weights.length !== input.robust.kernel.neighbors.length
    )
      throw new Error(`${id} robust.kernel.weights length must equal neighbors length`);
    if (input.robust.overflow && input.robust.overflow.length !== 1)
      throw new Error(`${id} robust.overflow must hold one value`);
  }
  const result: CPUSpatialTwoStageLeastSquaresResult = {
    status: new Uint32Array([1]),
    table: new Float32Array(
      coefficientCount * GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE
    ).fill(NaN),
    summary: new Float32Array(GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH).fill(NaN),
    residuals: new Float32Array(rowCount).fill(NaN),
    ...(input.robust
      ? {
          robust: {
            covariance: new Float32Array(coefficientCount ** 2).fill(NaN),
            status: new Uint32Array([3])
          }
        }
      : {})
  };
  result.summary[0] = rowCount;
  const invalidKernel =
    input.robust?.kernel && !isKernelValid(input.robust.kernel, input.robust.overflow);
  if (rowCount <= coefficientCount) {
    result.status[0] = 2;
    if (result.robust) result.robust.status[0] = invalidKernel ? 4 : 2;
    return result;
  }
  if (result.robust && invalidKernel) result.robust.status[0] = 4;
  const predictors = Array.from({length: rowCount}, (_, row) =>
    Array.from(input.predictors.subarray(row * predictorCount, (row + 1) * predictorCount))
  );
  const response = Array.from(input.response);
  const lagResponse = multiplyCPUErrorWeights(input.weights, response);
  const lagPredictors = Array.from({length: predictorCount}, (_, column) =>
    multiplyCPUErrorWeights(
      input.weights,
      predictors.map(row => row[column])
    )
  );
  const twiceLagPredictors =
    options.instrumentOrder === 2
      ? lagPredictors.map(column => multiplyCPUErrorWeights(input.weights, column))
      : [];
  const rawDesign = predictors.map((row, position) => [...row, lagResponse[position]]);
  const rawInstruments = predictors.map((row, position) => [
    ...row,
    ...lagPredictors.map(column => column[position]),
    ...twiceLagPredictors.map(column => column[position])
  ]);
  if (
    rawDesign.some(row => row.some(value => !Number.isFinite(value))) ||
    rawInstruments.some(row => row.some(value => !Number.isFinite(value))) ||
    response.some(value => !Number.isFinite(value))
  )
    return result;
  if (result.robust && !invalidKernel) result.robust.status[0] = 1;
  const means = rawDesign[0].map((_, column) => getCPUErrorMean(rawDesign.map(row => row[column])));
  const instrumentMeans = rawInstruments[0].map((_, column) =>
    getCPUErrorMean(rawInstruments.map(row => row[column]))
  );
  const responseMean = getCPUErrorMean(response);
  const centeredResponse = response.map(value => value - responseMean);
  const design = rawDesign.map(row => [1, ...row.map((value, column) => value - means[column])]);
  const instruments = rawInstruments.map(row => [
    1,
    ...row.map((value, column) => value - instrumentMeans[column])
  ]);
  // Solve H gamma = Wy directly with Householder QR. Forming (H'H)^-1 from the inverse
  // root and applying it to H'Wy would square H's condition number. The exogenous columns
  // and intercept are already in H, so their projections are exactly the original columns.
  const projectionFit = fitCPURegressionWithQR(
    instruments,
    design.map(row => row[coefficientCount - 1])
  );
  if (!projectionFit) return result;
  const projected = design.map((row, position) => [
    ...row.slice(0, -1),
    getCPUErrorDot(instruments[position], projectionFit.estimate)
  ]);
  const fit = fitCPURegressionWithQR(projected, centeredResponse);
  if (!fit) return result;
  const inverse = getInverse(fit.inverseRoot);
  const residuals = design.map(
    (row, position) => centeredResponse[position] - getCPUErrorDot(row, fit.estimate)
  );
  const residualSquares = getCPUErrorDot(residuals, residuals);
  const sigmaSquared = residualSquares / rowCount;
  const coefficients = [
    responseMean + fit.estimate[0] - getCPUErrorDot(means, fit.estimate.slice(1)),
    ...fit.estimate.slice(1)
  ];
  const transform = [1, ...means.map(value => -value)];
  const rawInverse = inverse.map(row => row.slice());
  const interceptCross = multiplyMatrixRows(inverse, transform);
  rawInverse[0] = interceptCross.slice();
  for (let column = 0; column < coefficientCount; column++)
    rawInverse[column][0] = interceptCross[column];
  rawInverse[0][0] = getCPUErrorDot(transform, interceptCross);
  const standardErrors = coefficients.map((_, column) =>
    Math.sqrt(sigmaSquared * rawInverse[column][column])
  );
  const secondConditioning = getCPUErrorConditioning(projected, means);
  const instrumentConditioning = getCPUErrorConditioning(instruments, instrumentMeans)[1];
  const restricted = fitCPURegressionWithQR(
    design.map(row => row.slice(0, -1)),
    lagResponse
  )!;
  const full = fitCPURegressionWithQR(instruments, lagResponse)!;
  const fullSquares = instruments.reduce(
    (sum, row, position) => sum + (lagResponse[position] - getCPUErrorDot(row, full.estimate)) ** 2,
    0
  );
  const restrictedSquares = design.reduce(
    (sum, row, position) =>
      sum + (lagResponse[position] - getCPUErrorDot(row.slice(0, -1), restricted.estimate)) ** 2,
    0
  );
  const responseSquares = getCPUErrorDot(centeredResponse, centeredResponse);
  const predicted = centeredResponse.map((value, row) => value - residuals[row]);
  const predictionMean = getCPUErrorMean(predicted);
  const centeredPredicted = predicted.map(value => value - predictionMean);
  const covariance = getCPUErrorDot(centeredResponse, centeredPredicted);
  const lagResiduals = multiplyCPUErrorWeights(input.weights, residuals);
  const residualLagProduct = getCPUErrorDot(residuals, lagResiduals);
  let totalWeight = 0;
  let trace = 0;
  let infinityNorm = 0;
  const {offsets, neighbors, weights} = input.weights;
  for (let row = 0; row < rowCount; row++) {
    let absoluteSum = 0;
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
      const neighbor = neighbors[slot];
      const weight = weights[slot];
      if (neighbor >= rowCount) continue;
      totalWeight += weight;
      absoluteSum += Math.abs(weight);
      // Matches the contributor's sorted-CSR lookup; callers supply sorted unique rows.
      let lower = offsets[neighbor];
      let upper = offsets[neighbor + 1];
      let reciprocal = 0;
      while (lower < upper) {
        const middle = lower + Math.floor((upper - lower) / 2);
        if (neighbors[middle] === row) {
          reciprocal = weights[middle];
          break;
        }
        if (neighbors[middle] < row) lower = middle + 1;
        else upper = middle;
      }
      trace += weight * (weight + reciprocal);
    }
    infinityNorm = Math.max(infinityNorm, absoluteSum);
  }
  const weightedDesign = design[0].map((_, column) =>
    multiplyCPUErrorWeights(
      input.weights,
      design.map(row => row[column])
    )
  );
  const residualWeightedDesign = weightedDesign.map(column => getCPUErrorDot(residuals, column));
  const estimationTerm = getCPUErrorDot(
    residualWeightedDesign,
    multiplyMatrixRows(inverse, residualWeightedDesign)
  );
  const anselinKelejian =
    (residualLagProduct / sigmaSquared) ** 2 / (trace + (4 * estimationTerm) / sigmaSquared);
  result.summary.set([
    rowCount,
    sigmaSquared,
    residualSquares,
    covariance ** 2 / (responseSquares * getCPUErrorDot(centeredPredicted, centeredPredicted)),
    totalWeight === 0 ? NaN : (rowCount * residualLagProduct) / (totalWeight * residualSquares),
    anselinKelejian,
    getChiSquareSurvival(anselinKelejian, 1),
    ...secondConditioning,
    secondConditioning[1] > GPU_REGRESSION_PRECISION_CONDITION_LIMIT ||
    instrumentConditioning > GPU_REGRESSION_PRECISION_CONDITION_LIMIT
      ? 1
      : 0,
    instrumentConditioning,
    Math.abs(coefficients[coefficientCount - 1]) * infinityNorm < 1 ? 0 : 1,
    infinityNorm,
    (restrictedSquares - fullSquares) /
      (instrumentCount - predictorCount) /
      (fullSquares / (rowCount - instrumentCount - 1)),
    (restrictedSquares - fullSquares) / restrictedSquares
  ]);
  result.status[0] = 0;
  result.residuals.set(residuals);
  if (input.robust && result.robust && !invalidKernel) {
    const influence = projected.map(row => multiplyMatrixRows(inverse, row));
    let robustCode = 0;
    let scores = influence.map((row, position) => {
      let residual = residuals[position];
      if (input.robust!.method === 'HC2' || input.robust!.method === 'HC3') {
        const leverage = getCPUErrorDot(projected[position], row);
        const denominator = 1 - leverage;
        if (
          !Number.isFinite(leverage) ||
          leverage < -INSTRUMENTAL_VARIABLES_COVARIANCE_LEVERAGE_TOLERANCE ||
          denominator <= INSTRUMENTAL_VARIABLES_COVARIANCE_LEVERAGE_TOLERANCE
        )
          robustCode = 5;
        residual /= input.robust!.method === 'HC2' ? Math.sqrt(denominator) : denominator;
      }
      return [
        residual * getCPUErrorDot(transform, row),
        ...row.slice(1).map(value => residual * value)
      ];
    });
    let correction = input.robust.method === 'HC1' ? rowCount / (rowCount - coefficientCount) : 1;
    if (input.robust.clusters) {
      const grouped = new Map<number, number[]>();
      for (let position = 0; position < rowCount; position++) {
        const key = input.robust.clusters[position];
        if (key >= input.robust.clusterCount!) {
          robustCode = 6;
          continue;
        }
        const total = grouped.get(key) ?? new Array<number>(coefficientCount).fill(0);
        scores[position].forEach((value, column) => {
          total[column] += value;
        });
        grouped.set(key, total);
      }
      const groups = grouped.size;
      if (groups < 2) robustCode = 6;
      correction =
        ((groups / Math.max(groups - 1, 1)) * (rowCount - 1)) / (rowCount - coefficientCount);
      scores = Array.from(grouped.values());
      if (!scores.length) scores = [new Array<number>(coefficientCount).fill(0)];
    }
    const smoothed = input.robust.kernel
      ? scores[0].map((_, column) =>
          multiplyCPUErrorWeights(
            input.robust!.kernel!,
            scores.map(row => row[column])
          )
        )
      : scores[0].map((_, column) => scores.map(row => row[column]));
    const robustCovariance = scores[0].map((_, first) =>
      scores[0].map(
        (_, second) =>
          correction *
          scores.reduce((sum, row, position) => sum + row[first] * smoothed[second][position], 0)
      )
    );
    // Publish the symmetric sandwich and certify it at the same scale/tolerance as the GPU.
    for (let first = 0; first < coefficientCount; first++) {
      for (let second = first + 1; second < coefficientCount; second++) {
        const value = 0.5 * robustCovariance[first][second] + 0.5 * robustCovariance[second][first];
        robustCovariance[first][second] = value;
        robustCovariance[second][first] = value;
      }
    }
    if (!robustCode) robustCode = getCPUInstrumentalVariablesCovarianceStatus(robustCovariance);
    result.robust.covariance.set(robustCovariance.flat());
    if (!robustCode && result.robust.covariance.some(value => !Number.isFinite(value)))
      robustCode = 3;
    result.robust.status[0] = robustCode;
    if (result.robust.status[0] !== 0) result.robust.covariance.fill(NaN);
    standardErrors.splice(
      0,
      coefficientCount,
      ...robustCovariance.map((row, column) =>
        row[column] >= 0 && result.robust!.status[0] === 0 ? Math.sqrt(row[column]) : NaN
      )
    );
  } else if (result.robust) standardErrors.fill(NaN);
  coefficients.forEach((value, column) => {
    const statistic = value / standardErrors[column];
    result.table.set(
      [
        value,
        standardErrors[column],
        statistic,
        Number.isFinite(statistic)
          ? getChiSquareSurvival(statistic ** 2, 1)
          : Number.isNaN(statistic)
            ? NaN
            : 0
      ],
      column * GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE
    );
  });
  return result;
}

/** CPU IV twin, without shader matrix/row caps; same classical and optional robust outputs. */
export class CPUSpatialTwoStageLeastSquaresOperation extends GPUCPUOperation {
  constructor(props: SpatialTwoStageLeastSquaresOperationProps) {
    props = snapshotRegressionOperationProps(props);
    validateCompleteRegressionWeights(props.id ?? 'spatial-two-stage-least-squares', props.weights);
    const id = props.id ?? 'spatial-two-stage-least-squares';
    validateOptions(id, props);
    validateCapacity(id, 'rowCount', props.rowCount);
    if (props.tileRowCount !== undefined) validateCapacity(id, 'tileRowCount', props.tileRowCount);
    if (props.robust)
      validateInstrumentalVariablesCovarianceOptions(id, props.rowCount, props.robust);
    const operation = new GPUSpatialTwoStageLeastSquaresOperation(props);
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
        const result = computeSpatialTwoStageLeastSquaresOnCPU(
          {
            predictors: getCPUOperationFloat32Rows(id, inputs, 'predictors'),
            response,
            weights: {
              offsets: getCPUOperationUint32Rows(id, inputs, 'offsets'),
              neighbors: getCPUOperationUint32Rows(id, inputs, 'neighbors'),
              weights: getCPUOperationFloat32Rows(id, inputs, 'weights')
            },
            robust: props.robust
              ? {
                  method: props.robust.method,
                  clusters: props.robust.clusters
                    ? getCPUOperationUint32Rows(id, inputs, 'clusters')
                    : undefined,
                  clusterCount: props.robust.clusterCount,
                  kernel: props.robust.kernel
                    ? {
                        offsets: getCPUOperationUint32Rows(id, inputs, 'kernelOffsets'),
                        neighbors: getCPUOperationUint32Rows(id, inputs, 'kernelNeighbors'),
                        weights: getCPUOperationFloat32Rows(id, inputs, 'kernelWeights')
                      }
                    : undefined,
                  overflow: props.robust.kernel?.overflow
                    ? getCPUOperationUint32Rows(id, inputs, 'kernelOverflow')
                    : undefined
                }
              : undefined
          },
          props
        );
        return {
          [props.output.status.code]: result.status,
          [props.output.table]: result.table,
          [props.output.summary]: result.summary,
          ...(props.output.residuals ? {[props.output.residuals]: result.residuals} : {}),
          ...(props.robust && result.robust
            ? {
                [props.robust.output.covariance]: result.robust.covariance,
                [props.robust.output.status]: result.robust.status
              }
            : {})
        };
      }
    });
  }
}
