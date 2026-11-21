// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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
  GPUOrdinaryLeastSquaresOperation,
  snapshotOrdinaryLeastSquaresOperationProps,
  type OrdinaryLeastSquaresOperationProps
} from './gpu-ordinary-least-squares-operation';
import {
  resolveOrdinaryLeastSquaresPredictorCount,
  GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_STATUS_SINGULAR,
  GPU_ORDINARY_LEAST_SQUARES_STATUS_TOO_FEW_ROWS
} from './ordinary-least-squares-parameters';
import {getChiSquareSurvival} from './ordinary-least-squares-statistics';
import {
  factorCPURegressionCorrelation,
  solveCPURegressionFactor,
  getCPURegressionSingularValueRange
} from './cpu-regression-linear-algebra';
import {computeCenteredRegressionContribution} from './centered-regression-prediction';

/** Exact float32 input rows; parameters use getGPUOrdinaryLeastSquaresParameterValues. */
export type CPUOrdinaryLeastSquaresInput = {
  predictors: Float32Array;
  response: Float32Array;
  mask?: Uint32Array;
  parameters?: Float32Array;
};

/** Same row layouts as the contributor, including optional columns (always returned here). */
export type CPUOrdinaryLeastSquaresResult = {
  status: Uint32Array;
  coefficients: Float32Array;
  standardErrors: Float32Array;
  tStatistics: Float32Array;
  summary: Float32Array;
  residuals: Float32Array;
  fitted: Float32Array;
  /** Centered predictions for finite predictor rows, independent of the mask or response validity. */
  predictions: Float32Array;
};

type CenteredFit = {
  means: number[];
  moments: number[][];
  scales: number[];
  correlation: number[][];
  factor: number[][];
  inverse: number[][];
  slopes: number[];
};

function fitCenteredRows(
  predictors: number[][],
  response: number[],
  ridgeLambda: number
): CenteredFit | undefined {
  const predictorCount = predictors[0].length;
  const means = new Array<number>(predictorCount + 1).fill(0);
  // Shift first so small deviations from a large mean survive the summation.
  const origin = [...predictors[0], response[0]];
  for (let row = 0; row < response.length; row++) {
    for (let column = 0; column < predictorCount; column++) {
      means[column] += predictors[row][column] - origin[column];
    }
    means[predictorCount] += response[row] - origin[predictorCount];
  }
  for (let column = 0; column < means.length; column++) {
    means[column] = origin[column] + means[column] / response.length;
  }
  const moments = means.map(() => means.map(() => 0));
  for (let row = 0; row < response.length; row++) {
    const centered = [...predictors[row], response[row]].map(
      (entry, column) => entry - means[column]
    );
    for (let first = 0; first < means.length; first++) {
      for (let second = 0; second <= first; second++) {
        moments[first][second] += centered[first] * centered[second];
      }
    }
  }
  for (let first = 0; first < means.length; first++) {
    for (let second = 0; second < first; second++) moments[second][first] = moments[first][second];
  }
  const scales = predictors[0].map((_, column) => Math.sqrt(moments[column][column]));
  if (scales.some(scale => !(scale > 0) || !Number.isFinite(scale))) return undefined;
  const correlation = scales.map((scale, first) =>
    scales.map(
      (secondScale, second) =>
        moments[first][second] / (scale * secondScale) +
        (first === second ? ridgeLambda / scale ** 2 : 0)
    )
  );
  const factor = factorCPURegressionCorrelation(correlation);
  if (!factor) return undefined;
  const slopes = solveCPURegressionFactor(
    factor,
    scales.map((scale, column) => moments[column][predictorCount] / scale)
  ).map((entry, column) => entry / scales[column]);
  const inverse = scales.map((_, column) =>
    solveCPURegressionFactor(
      factor,
      scales.map((_, row) => (row === column ? 1 : 0))
    )
  );
  return {means, moments, scales, correlation, factor, inverse, slopes};
}

/**
 * Centered OLS (or slope-only ridge) and all 22 report slots in float64, rounded once on return.
 * Non-finite predictor/response rows and mask=0 rows are excluded, as in the contributor.
 * Predictions still cover those rows when their predictors are finite; fitted values remain NaN.
 *
 * What differs from the GPU contributor:
 * - Float64 moments, solves, residuals and diagnostics replace float32 tile reductions and mean
 *   corrections. tileRowCount is ignored. Comparisons on well-conditioned inputs use 5e-3 relative
 *   tolerance for diagnostics and 5e-4 for coefficients and row outputs.
 * - The shader-only 15-predictor cap is dropped. Numerical singularity uses a squared Cholesky
 *   pivot floor of 64 * Number.EPSILON relative to that column's correlation diagonal, instead of
 *   the GPU's pivot < 1e-3 rejection; some GPU status-1 fits succeed here.
 * - Conditioning uses one-sided Jacobi singular values until float64 orthogonality or numerical
 *   stagnation, with a separate bounded host sweep count. A singular unpenalized design in a
 *   successful ridge fit reports Belsley 3e38,
 *   matching the GPU sentinel. The precision-warning threshold still describes the GPU matrix.
 * - The public float64 chi-square function uses a converged shape-dependent host evaluation;
 *   the GPU retains its 200-iteration budget for at most 15 degrees of freedom.
 * - Subnormals are retained. Non-finite parameter lambda means no ridge, as on the GPU.
 *
 * @throws If predictors, mask or parameters have inconsistent lengths, naming the argument.
 */
export function computeOrdinaryLeastSquaresOnCPU(
  input: CPUOrdinaryLeastSquaresInput,
  explicitPredictorCount?: number
): CPUOrdinaryLeastSquaresResult {
  const id = 'computeOrdinaryLeastSquaresOnCPU';
  const rowCount = input.response.length;
  const predictorCount = resolveOrdinaryLeastSquaresPredictorCount(
    id,
    input.predictors.length,
    rowCount,
    explicitPredictorCount
  );
  if (input.predictors.length !== rowCount * predictorCount)
    throw new Error(`${id} predictors length must equal response length * predictorCount`);
  if (input.mask && input.mask.length !== rowCount)
    throw new Error(`${id} mask length must equal response length`);
  if (input.parameters && input.parameters.length < GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH)
    throw new Error(`${id} parameters must hold the packed ridge lambda`);
  const [requestedRidge = 0] = input.parameters ?? [];
  const ridgeLambda = Number.isFinite(requestedRidge) && requestedRidge > 0 ? requestedRidge : 0;
  const usedRows: number[] = [];
  const predictors: number[][] = [];
  const response: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    if (input.mask && input.mask[row] === 0) continue;
    const values = Array.from(
      input.predictors.subarray(row * predictorCount, (row + 1) * predictorCount)
    );
    if (!Number.isFinite(input.response[row]) || values.some(value => !Number.isFinite(value)))
      continue;
    usedRows.push(row);
    predictors.push(values);
    response.push(input.response[row]);
  }
  const coefficientCount = predictorCount + 1;
  const result: CPUOrdinaryLeastSquaresResult = {
    status: new Uint32Array(1),
    coefficients: new Float32Array(coefficientCount).fill(NaN),
    standardErrors: new Float32Array(coefficientCount).fill(NaN),
    tStatistics: new Float32Array(coefficientCount).fill(NaN),
    summary: new Float32Array(GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH).fill(NaN),
    residuals: new Float32Array(rowCount).fill(NaN),
    fitted: new Float32Array(rowCount).fill(NaN),
    predictions: new Float32Array(rowCount).fill(NaN)
  };
  const usedCount = usedRows.length;
  result.summary[0] = usedCount;
  if (usedCount <= coefficientCount) {
    result.status[0] = GPU_ORDINARY_LEAST_SQUARES_STATUS_TOO_FEW_ROWS;
    return result;
  }
  const fit = fitCenteredRows(predictors, response, ridgeLambda);
  if (!fit) {
    result.status[0] = GPU_ORDINARY_LEAST_SQUARES_STATUS_SINGULAR;
    return result;
  }
  const residuals = response.map((value, row) => {
    const centeredFitted = computeCenteredRegressionContribution(
      predictors[row],
      fit.means,
      fit.slopes
    );
    result.fitted[usedRows[row]] = fit.means[predictorCount] + centeredFitted;
    return value - fit.means[predictorCount] - centeredFitted;
  });
  for (let row = 0; row < rowCount; row++) {
    const values = input.predictors.subarray(row * predictorCount, (row + 1) * predictorCount);
    if (values.every(Number.isFinite)) {
      result.predictions[row] =
        fit.means[predictorCount] +
        computeCenteredRegressionContribution(values, fit.means, fit.slopes);
    }
  }
  const residualSumSquares = residuals.reduce((sum, value) => sum + value ** 2, 0);
  const sumCubes = residuals.reduce((sum, value) => sum + value ** 3, 0);
  const sumFourth = residuals.reduce((sum, value) => sum + value ** 4, 0);
  residuals.forEach((value, row) => {
    result.residuals[usedRows[row]] = value;
  });
  const freedom = usedCount - coefficientCount;
  const sigmaSquared = residualSumSquares / freedom;
  const coefficients = [
    fit.means[predictorCount] -
      fit.slopes.reduce((sum, slope, column) => sum + slope * fit.means[column], 0),
    ...fit.slopes
  ];
  const normalizedMeans = fit.scales.map((scale, column) => fit.means[column] / scale);
  const solvedMeans = solveCPURegressionFactor(fit.factor, normalizedMeans);
  const interceptVariance =
    1 / usedCount +
    normalizedMeans.reduce((sum, value, column) => sum + value * solvedMeans[column], 0);
  const standardErrors = [
    Math.sqrt(Math.max(sigmaSquared * interceptVariance, 0)),
    ...fit.scales.map((scale, column) =>
      Math.sqrt(Math.max((sigmaSquared * fit.inverse[column][column]) / scale ** 2, 0))
    )
  ];
  result.coefficients.set(coefficients);
  result.standardErrors.set(standardErrors);
  result.tStatistics.set(
    coefficients.map((value, column) =>
      standardErrors[column] > 0 ? value / standardErrors[column] : NaN
    )
  );
  const totalSumSquares = fit.moments[predictorCount][predictorCount];
  const rSquared = totalSumSquares > 0 ? 1 - residualSumSquares / totalSumSquares : NaN;
  const meanSquare = residualSumSquares / usedCount;
  const logLikelihood =
    -0.5 *
    usedCount *
    (Math.log(2 * Math.PI) + Math.log(meanSquare === 0 ? 1.17549435e-38 : meanSquare) + 1);
  const skewness = meanSquare > 0 ? sumCubes / usedCount / meanSquare ** 1.5 : NaN;
  const kurtosis = meanSquare > 0 ? sumFourth / usedCount / meanSquare ** 2 : NaN;
  const jarqueBera = (usedCount / 6) * (skewness ** 2 + (kurtosis - 3) ** 2 / 4);
  const auxiliary = fitCenteredRows(
    predictors,
    residuals.map(value => value ** 2),
    0
  );
  let koenkerBassett = NaN;
  let breuschPagan = NaN;
  if (auxiliary) {
    const explained = auxiliary.slopes.reduce(
      (sum, slope, column) => sum + slope * auxiliary.moments[column][predictorCount],
      0
    );
    if (auxiliary.moments[predictorCount][predictorCount] > 0) {
      const auxiliaryRSquared = Math.min(
        1,
        Math.max(0, explained / auxiliary.moments[predictorCount][predictorCount])
      );
      koenkerBassett = usedCount * auxiliaryRSquared;
    }
    if (residualSumSquares > 0)
      breuschPagan = Math.max((0.5 * (explained / meanSquare)) / meanSquare, 0);
  }
  const [minimumSingular, maximumSingular] = getCPURegressionSingularValueRange(fit.factor);
  const correlationCondition = (maximumSingular / minimumSingular) ** 2;
  const unpenalized = fit.scales.map((scale, first) =>
    fit.scales.map((secondScale, second) => fit.moments[first][second] / (scale * secondScale))
  );
  const unpenalizedFactor = factorCPURegressionCorrelation(unpenalized);
  let belsley = 3e38;
  if (unpenalizedFactor) {
    // QR-equivalent reduced raw design: first row carries the means, lower rows the centered factor.
    const rawNorms = fit.scales.map((scale, column) =>
      Math.hypot(Math.sqrt(usedCount) * fit.means[column], scale)
    );
    const reduced = Array.from({length: coefficientCount}, () =>
      new Array<number>(coefficientCount).fill(0)
    );
    reduced[0][0] = 1;
    for (let column = 0; column < predictorCount; column++) {
      reduced[0][column + 1] = (Math.sqrt(usedCount) * fit.means[column]) / rawNorms[column];
      for (let row = 0; row <= column; row++) {
        reduced[row + 1][column + 1] =
          (unpenalizedFactor[column][row] * fit.scales[column]) / rawNorms[column];
      }
    }
    const [minimum, maximum] = getCPURegressionSingularValueRange(reduced);
    belsley = maximum / minimum;
  }
  result.summary.set([
    usedCount,
    rSquared,
    1 - ((1 - rSquared) * (usedCount - 1)) / freedom,
    sigmaSquared,
    logLikelihood,
    2 * coefficientCount - 2 * logLikelihood,
    coefficientCount * Math.log(usedCount) - 2 * logLikelihood,
    jarqueBera,
    Math.exp(-jarqueBera / 2),
    koenkerBassett,
    getChiSquareSurvival(koenkerBassett, predictorCount),
    residualSumSquares,
    totalSumSquares,
    skewness,
    kurtosis,
    ridgeLambda,
    breuschPagan,
    getChiSquareSurvival(breuschPagan, predictorCount),
    belsley,
    correlationCondition,
    Math.max(...fit.inverse.map((row, column) => row[column])),
    correlationCondition > GPU_REGRESSION_PRECISION_CONDITION_LIMIT ? 1 : 0
  ]);
  return result;
}

/** CPU twin; drops the shader predictor cap and tile budget, retaining every output layout. */
export class CPUOrdinaryLeastSquaresOperation extends GPUCPUOperation {
  constructor(props: OrdinaryLeastSquaresOperationProps) {
    props = snapshotOrdinaryLeastSquaresOperationProps(props);
    const id = props.id ?? 'ordinary-least-squares';
    validateCapacity(id, 'rowCount', props.rowCount);
    validateCapacity(id, 'predictorCount', props.predictorCount);
    if (props.tileRowCount !== undefined) validateCapacity(id, 'tileRowCount', props.tileRowCount);
    // Reuse only the adapter's port declarations and graph alias checks, never its GPU contributor.
    const operation = new GPUOrdinaryLeastSquaresOperation(props);
    validateRequirement(
      id,
      props.predictorCount + 1 <= 0xffffffff,
      'predictorCount + 1 must fit uint32'
    );
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
        const result = computeOrdinaryLeastSquaresOnCPU(
          {
            predictors: getCPUOperationFloat32Rows(id, inputs, 'predictors'),
            response,
            mask: props.mask ? getCPUOperationUint32Rows(id, inputs, 'mask') : undefined,
            parameters: props.parameters
              ? getCPUOperationFloat32Rows(id, inputs, 'parameters')
              : undefined
          },
          props.predictorCount
        );
        return {
          [props.output.status.code]: result.status,
          [props.output.coefficients]: result.coefficients,
          [props.output.standardErrors]: result.standardErrors,
          [props.output.tStatistics]: result.tStatistics,
          [props.output.summary]: result.summary,
          ...(props.output.residuals ? {[props.output.residuals]: result.residuals} : {}),
          ...(props.output.fitted ? {[props.output.fitted]: result.fitted} : {}),
          ...(props.output.predictions ? {[props.output.predictions]: result.predictions} : {})
        };
      }
    });
  }
}
