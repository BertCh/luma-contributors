// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotRegressionOperationProps} from './regression-construction-snapshot';

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
import {
  validateCPURegressionWeights,
  validateCompleteRegressionWeights,
  type CPURegressionWeights
} from './cpu-regression-weights';
import {
  getCPUErrorConditioning,
  getCPUErrorDot,
  getCPUErrorMean,
  getCPUErrorPolynomialCandidates,
  multiplyCPUErrorWeights
} from './cpu-spatial-error-regression';
import {
  GPUSpatialErrorGMOperation,
  type SpatialErrorGMOperationProps
} from './gpu-spatial-error-gm-operation';
import {
  GPU_SPATIAL_ERROR_GM_DEFAULT_ESTIMATOR,
  GPU_SPATIAL_ERROR_GM_LAMBDA_BOUND_FRACTION,
  GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH,
  GPU_SPATIAL_ERROR_GM_TABLE_STRIDE
} from './spatial-error-gm-parameters';
import {
  GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT,
  GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT,
  GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH,
  getSpatialWeightsTracesProbeValue
} from './spatial-weights-traces-parameters';
import {getChiSquareSurvival} from './ordinary-least-squares-statistics';

/** Whole CSR, float32 design/response and the optional packed Philox trace seed. */
export type CPUSpatialErrorGMInput = {
  weights: CPURegressionWeights;
  predictors: Float32Array;
  response: Float32Array;
  traceParameters?: Uint32Array;
};
/** Estimator settings; shader-only predictor, trace-size and tile budgets are dropped. */
export type CPUSpatialErrorGMOptions = Pick<
  SpatialErrorGMOperationProps,
  'predictorCount' | 'estimator' | 'traceMode' | 'traceProbeCount'
>;
/** Scalar float32 table/summary/residual rows and the one-word solver code. */
export type CPUSpatialErrorGMResult = {
  table: Float32Array;
  summary: Float32Array;
  residuals: Float32Array;
  status: Uint32Array;
};

function validateOptions(id: string, options: CPUSpatialErrorGMOptions): void {
  validateCapacity(id, 'predictorCount', options.predictorCount);
  validateRequirement(
    id,
    options.estimator === undefined ||
      options.estimator === 'homoskedastic' ||
      options.estimator === 'kelejian-prucha',
    "estimator must be 'homoskedastic' or 'kelejian-prucha'"
  );
  if ((options.estimator ?? GPU_SPATIAL_ERROR_GM_DEFAULT_ESTIMATOR) === 'homoskedastic') {
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
  }
}

function getMomentPolynomial(
  residuals: number[],
  multiply: (values: number[]) => number[],
  trace: number,
  bound: number,
  homoskedastic: boolean
): number[][] | undefined {
  const lagged = multiply(residuals);
  const twiceLagged = multiply(lagged);
  const squares = getCPUErrorDot(residuals, residuals);
  if (!(squares > 0)) return undefined;
  const cross = getCPUErrorDot(residuals, lagged);
  const lagSquares = getCPUErrorDot(lagged, lagged);
  const twiceCross = getCPUErrorDot(residuals, twiceLagged);
  const lagTwiceCross = getCPUErrorDot(lagged, twiceLagged);
  const twiceSquares = getCPUErrorDot(twiceLagged, twiceLagged);
  const count = residuals.length;
  const ratio = trace / count;
  const scale = 1 + ratio * ratio;
  // Polynomial in t=lambda/bound. This keeps the search on [-1,1], even for raw weights.
  const polynomial = homoskedastic
    ? [
        [-(lagSquares - ratio * squares) / scale / count, -cross / count],
        [
          ((2 * (lagTwiceCross - ratio * cross)) / scale / count) * bound,
          ((twiceCross + lagSquares) / count) * bound
        ],
        [
          (-(twiceSquares - ratio * lagSquares) / scale / count) * bound * bound,
          (-lagTwiceCross / count) * bound * bound
        ]
      ]
    : [
        [1, lagSquares / squares, cross / squares],
        [
          ((-2 * cross) / squares) * bound,
          ((-2 * lagTwiceCross) / squares) * bound,
          (-(twiceCross + lagSquares) / squares) * bound
        ],
        [
          (lagSquares / squares) * bound * bound,
          (twiceSquares / squares) * bound * bound,
          (lagTwiceCross / squares) * bound * bound
        ]
      ];
  return polynomial.flat().every(Number.isFinite) ? polynomial : undefined;
}

function minimizeMomentPolynomial(
  polynomial: number[][],
  weight?: number[][],
  varianceColumn?: number[]
): {parameter: number; objective: number} {
  const product = (first: number[], second: number[]) =>
    weight
      ? first.reduce((total, value, row) => total + value * getCPUErrorDot(weight[row], second), 0)
      : getCPUErrorDot(first, second);
  const branches = [polynomial];
  const candidates = [-1, 1];
  if (varianceColumn) {
    const varianceNorm = getCPUErrorDot(varianceColumn, varianceColumn);
    const variance = polynomial.map(row => getCPUErrorDot(row, varianceColumn) / varianceNorm);
    candidates.push(...getCPUErrorPolynomialCandidates(variance));
    branches.push(
      polynomial.map((row, degree) =>
        row.map((value, index) => value - variance[degree] * varianceColumn[index])
      )
    );
  }
  for (const branch of branches) {
    const derivative = [0, 0, 0, 0];
    for (let first = 0; first < 3; first++) {
      for (let second = 0; second < 3; second++) {
        const degree = first + second;
        if (degree > 0) derivative[degree - 1] += degree * product(branch[first], branch[second]);
      }
    }
    candidates.push(...getCPUErrorPolynomialCandidates(derivative));
  }
  let parameter = -1;
  let objective = Infinity;
  for (const candidate of candidates.sort((first, second) => first - second)) {
    const residual = polynomial[0].map(
      (value, index) =>
        value + candidate * (polynomial[1][index] + candidate * polynomial[2][index])
    );
    if (varianceColumn) {
      const variance = Math.max(
        getCPUErrorDot(residual, varianceColumn) / getCPUErrorDot(varianceColumn, varianceColumn),
        0
      );
      for (let index = 0; index < residual.length; index++)
        residual[index] -= variance * varianceColumn[index];
    }
    const value = product(residual, residual);
    if (value < objective) {
      parameter = candidate;
      objective = value;
    }
  }
  return {parameter, objective};
}

function getTraceTerms(
  weights: CPURegressionWeights,
  multiply: (values: number[]) => number[],
  stochastic: boolean,
  probeCount: number,
  seed: number,
  trace: number
): number[] {
  const count = weights.offsets.length - 1;
  let firstPower = 0;
  let secondPower = 0;
  let gramCross = 0;
  let gramSquares = 0;
  const columnSquares = new Array<number>(count).fill(0);
  for (let row = 0; row < count; row++) {
    for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
      const neighbor = weights.neighbors[slot];
      if (neighbor < count) columnSquares[neighbor] += weights.weights[slot] ** 2;
    }
  }
  const probes = stochastic ? probeCount : count;
  for (let probe = 0; probe < probes; probe++) {
    const initial = Array.from({length: count}, (_, row) =>
      stochastic ? getSpatialWeightsTracesProbeValue(seed, probe, row) : Number(row === probe)
    );
    const lagged = multiply(initial);
    const twiceLagged = multiply(lagged);
    firstPower += stochastic ? getCPUErrorDot(initial, lagged) : lagged[probe];
    secondPower += stochastic ? getCPUErrorDot(initial, twiceLagged) : twiceLagged[probe];
    gramCross += getCPUErrorDot(lagged, twiceLagged);
    const transposed = new Array<number>(count).fill(0);
    for (let row = 0; row < count; row++) {
      for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
        const neighbor = weights.neighbors[slot];
        if (neighbor < count) transposed[neighbor] += weights.weights[slot] * lagged[row];
      }
    }
    gramSquares += getCPUErrorDot(transposed, transposed);
  }
  const divisor = stochastic ? probes : 1;
  const ratio = trace / count;
  const scale = 1 + ratio * ratio;
  return [
    (gramSquares / divisor - trace * ratio) / scale ** 2,
    ((2 * gramCross) / divisor - (2 * ratio * firstPower) / divisor) / scale,
    secondPower / divisor + trace,
    (getCPUErrorDot(columnSquares, columnSquares) - trace * ratio) / scale ** 2
  ];
}

function getMomentVariance(traceTerms: number[], filtered: number[]): number[][] {
  const count = filtered.length;
  const sigmaSquared = getCPUErrorDot(filtered, filtered) / count;
  const fourthMoment = filtered.reduce((sum, value) => sum + value ** 4, 0) / count;
  const cross = (sigmaSquared ** 2 * traceTerms[1]) / count;
  return [
    [
      (2 * sigmaSquared ** 2 * traceTerms[0] +
        (fourthMoment - 3 * sigmaSquared ** 2) * traceTerms[3]) /
        count,
      cross
    ],
    [cross, (sigmaSquared ** 2 * traceTerms[2]) / count]
  ];
}

function invertMomentVariance(matrix: number[][]): number[][] | undefined {
  // Scale before taking the determinant, so a finite inverse does not overflow or underflow.
  const scale = Math.max(...matrix.flat().map(Math.abs));
  if (!(scale > 0) || !Number.isFinite(scale)) return undefined;
  const first = matrix[0][0] / scale;
  const cross = matrix[0][1] / scale;
  const second = matrix[1][1] / scale;
  const determinant = first * second - cross * cross;
  if (!(determinant > 0)) return undefined;
  return [
    [second / determinant / scale, -cross / determinant / scale],
    [-cross / determinant / scale, first / determinant / scale]
  ];
}

/**
 * Fits both spatial error GM estimators in float64, retaining the homoskedastic first-step betas
 * and residuals while evaluating their covariance at the second-step lambda.
 *
 * What differs from the GPU contributor:
 * - Float64 centered QR, residuals, moments and traces replace float32 reductions/normal solves;
 *   only final arrays are rounded. Well-conditioned comparisons use 5e-3 relative tolerance.
 * - Both piecewise-quartic moment objectives are minimized over every stationary point and
 *   variance-zero boundary, to float64 resolution. The GPU's 1024-interval scan and 40 bisections
 *   can miss narrow basins. Exact ties still choose the lowest lambda.
 * - Shader-only predictor and explicit-exact trace row caps are dropped; tileRowCount is ignored.
 *   QR rank uses 64 * Number.EPSILON, so some GPU status-1 fits succeed here. Conditioning uses
 *   converged float64 singular values, without a GPU sweep cap.
 * - Stochastic traces retain the GPU's default mode, registered Philox seed/probe order and
 *   approximation flag. They are not silently replaced with exact traces. Kelejian-Prucha
 *   ignores traceMode and traceProbeCount, including otherwise invalid settings.
 * - P-values call the public float64 chi-square survival helper (its 200-iteration cap remains).
 *   Subnormals are retained; non-finite active weights or data produce failure codes. The pure
 *   function accepts empty slot arrays; GPU weights require positive capacity.
 * - As in the kernels, residuals are the first filtered fit's rows even if the final report fails;
 *   the finish kernel clears table/summary only. Whole CSR is supported; chunked weights are omitted.
 *
 * @throws If weights, predictors, response or traceParameters arrays are inconsistent.
 */
export function computeSpatialErrorGMOnCPU(
  input: CPUSpatialErrorGMInput,
  options: CPUSpatialErrorGMOptions
): CPUSpatialErrorGMResult {
  const id = 'computeSpatialErrorGMOnCPU';
  validateOptions(id, options);
  const count = validateCPURegressionWeights(id, input.weights);
  const {predictorCount} = options;
  const width = predictorCount + 1;
  if (input.response.length !== count)
    throw new Error(`${id} response length must equal weights rowCount`);
  if (input.predictors.length !== count * predictorCount)
    throw new Error(`${id} predictors length must equal response length * predictorCount`);
  if (
    input.traceParameters &&
    input.traceParameters.length < GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH
  )
    throw new Error(`${id} traceParameters must hold the packed seed`);
  const result: CPUSpatialErrorGMResult = {
    table: new Float32Array((predictorCount + 2) * GPU_SPATIAL_ERROR_GM_TABLE_STRIDE).fill(NaN),
    summary: new Float32Array(GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH).fill(NaN),
    residuals: new Float32Array(count).fill(NaN),
    status: new Uint32Array(1)
  };
  result.summary[0] = count;
  const fail = (code: number) => {
    result.status[0] = code;
    return result;
  };
  const means = Array.from({length: predictorCount}, (_, column) =>
    getCPUErrorMean(
      Array.from({length: count}, (_, row) => input.predictors[row * predictorCount + column])
    )
  );
  const responseMean = getCPUErrorMean(Array.from(input.response));
  const design = Array.from({length: count}, (_, row) => [
    1,
    ...means.map((mean, column) => input.predictors[row * predictorCount + column] - mean)
  ]);
  const response = Array.from(input.response, value => value - responseMean);
  // OLS filters non-finite rows, but GM's own means/moments use every row.
  const finiteRows = Array.from({length: count}, (_, row) => row).filter(
    row =>
      Number.isFinite(input.response[row]) &&
      Array.from(input.predictors.subarray(row * predictorCount, (row + 1) * predictorCount)).every(
        Number.isFinite
      )
  );
  if (finiteRows.length <= width) return fail(2);
  if (finiteRows.length !== count) {
    const finiteDesign = finiteRows.map(row => [
      1,
      ...input.predictors.subarray(row * predictorCount, (row + 1) * predictorCount)
    ]);
    const finiteResponse = finiteRows.map(row => input.response[row]);
    return fail(
      count <= width + 1 ? 2 : fitCPURegressionWithQR(finiteDesign, finiteResponse) ? 3 : 1
    );
  }
  const initialFit = fitCPURegressionWithQR(design, response);
  if (!initialFit) return fail(count <= width + 1 ? 2 : 1);
  const initialResiduals = response.map(
    (value, row) => value - getCPUErrorDot(design[row], initialFit.estimate)
  );
  const multiply = (values: number[]) => multiplyCPUErrorWeights(input.weights, values);
  let norm = 0;
  let trace = 0;
  for (let row = 0; row < count; row++) {
    let rowSum = 0;
    for (let slot = input.weights.offsets[row]; slot < input.weights.offsets[row + 1]; slot++) {
      if (input.weights.neighbors[slot] < count) {
        rowSum += Math.abs(input.weights.weights[slot]);
        trace += input.weights.weights[slot] ** 2;
      }
    }
    norm = Math.max(norm, rowSum);
  }
  if (!(norm > 0) || !Number.isFinite(norm) || !Number.isFinite(trace))
    return fail(count <= width + 1 ? 2 : 3);
  const bound = GPU_SPATIAL_ERROR_GM_LAMBDA_BOUND_FRACTION / norm;
  const homoskedastic =
    (options.estimator ?? GPU_SPATIAL_ERROR_GM_DEFAULT_ESTIMATOR) === 'homoskedastic';
  let polynomial = getMomentPolynomial(initialResiduals, multiply, trace, bound, homoskedastic);
  if (!polynomial) return fail(count <= width + 1 ? 2 : 3);
  const firstMinimum = minimizeMomentPolynomial(
    polynomial,
    undefined,
    homoskedastic ? undefined : [1, trace / count, 0]
  );
  const firstLambda = firstMinimum.parameter * bound;
  const laggedDesign = design[0].map((_, column) => multiply(design.map(row => row[column])));
  const laggedResponse = multiply(response);
  const filterDesign = (lambda: number) =>
    design.map((row, position) =>
      row.map((value, column) => value - lambda * laggedDesign[column][position])
    );
  const firstDesign = filterDesign(firstLambda);
  const firstFit = fitCPURegressionWithQR(
    firstDesign,
    response.map((value, row) => value - firstLambda * laggedResponse[row])
  );
  if (!firstFit) return fail(count <= width + 1 ? 2 : 1);
  const residuals = response.map(
    (value, row) => value - getCPUErrorDot(design[row], firstFit.estimate)
  );
  result.residuals.set(residuals);
  if (count <= width + 1) return fail(2);
  const laggedResiduals = multiply(residuals);
  let lambda = firstLambda;
  let objective = firstMinimum.objective;
  let lambdaStandardError = NaN;
  let finalFit = firstFit;
  let finalDesign = firstDesign;
  let sigmaSquared =
    getCPUErrorDot(
      residuals.map((value, row) => value - lambda * laggedResiduals[row]),
      residuals.map((value, row) => value - lambda * laggedResiduals[row])
    ) / count;
  const stochastic =
    homoskedastic &&
    (options.traceMode ??
      (count <= GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT ? 'exact' : 'stochastic')) ===
      'stochastic';
  if (homoskedastic) {
    polynomial = getMomentPolynomial(residuals, multiply, trace, bound, true);
    if (!polynomial) return fail(3);
    const traceTerms = getTraceTerms(
      input.weights,
      multiply,
      stochastic,
      options.traceProbeCount ?? GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT,
      input.traceParameters?.[0] ?? 0,
      trace
    );
    const firstFiltered = residuals.map((value, row) => value - firstLambda * laggedResiduals[row]);
    const weight = invertMomentVariance(getMomentVariance(traceTerms, firstFiltered));
    if (!weight) return fail(3);
    const minimum = minimizeMomentPolynomial(polynomial, weight);
    lambda = minimum.parameter * bound;
    objective = minimum.objective;
    const filtered = residuals.map((value, row) => value - lambda * laggedResiduals[row]);
    sigmaSquared = getCPUErrorDot(filtered, filtered) / count;
    const finalWeight = invertMomentVariance(getMomentVariance(traceTerms, filtered));
    if (finalWeight) {
      const jacobian = polynomial[1].map(
        (value, index) => (value + 2 * minimum.parameter * polynomial![2][index]) / bound
      );
      const information = getCPUErrorDot(
        jacobian,
        finalWeight.map(row => getCPUErrorDot(row, jacobian))
      );
      lambdaStandardError = Math.sqrt(1 / information / count);
    }
    finalDesign = filterDesign(lambda);
    const fit = fitCPURegressionWithQR(
      finalDesign,
      response.map((value, row) => value - lambda * laggedResponse[row])
    );
    if (!fit) return fail(1);
    finalFit = fit;
  }
  if (![lambda, objective, sigmaSquared].every(Number.isFinite)) return fail(3);
  const coefficients = firstFit.estimate.slice();
  coefficients[0] += responseMean - getCPUErrorDot(means, coefficients.slice(1));
  const inverseRoot = finalFit.inverseRoot.map(row => row.slice());
  for (let column = 0; column < predictorCount; column++) {
    for (let target = 0; target < width; target++)
      inverseRoot[0][target] -= means[column] * finalFit.inverseRoot[column + 1][target];
  }
  const standardErrors = inverseRoot.map(row => Math.sqrt(sigmaSquared * getCPUErrorDot(row, row)));
  coefficients.push(lambda);
  standardErrors.push(lambdaStandardError);
  coefficients.forEach((coefficient, row) => {
    const statistic = coefficient / standardErrors[row];
    result.table.set(
      [
        coefficient,
        standardErrors[row],
        statistic,
        Math.abs(statistic) === Infinity ? 0 : getChiSquareSurvival(statistic ** 2, 1)
      ],
      row * GPU_SPATIAL_ERROR_GM_TABLE_STRIDE
    );
  });
  const residualSquares = getCPUErrorDot(residuals, residuals);
  const residualTotal = residuals.reduce((sum, value) => sum + value, 0);
  const responseSquares = getCPUErrorDot(response, response);
  const responseResidual = getCPUErrorDot(response, residuals);
  const predictedVariance =
    responseSquares - 2 * responseResidual + residualSquares - residualTotal ** 2 / count;
  const conditioning = getCPUErrorConditioning(finalDesign, means);
  result.summary.set([
    count,
    lambda,
    sigmaSquared,
    (responseSquares - responseResidual) ** 2 / responseSquares / predictedVariance,
    residualSquares,
    objective,
    ...conditioning,
    Number(conditioning[1] > GPU_REGRESSION_PRECISION_CONDITION_LIMIT),
    bound,
    Number(stochastic),
    Number(Math.abs(lambda) >= bound * 0.9999)
  ]);
  return result;
}

/** CPU whole-CSR twin, without shader predictor/trace-size caps; tileRowCount is ignored. */
export class CPUSpatialErrorGMOperation extends GPUCPUOperation {
  constructor(props: SpatialErrorGMOperationProps) {
    props = snapshotRegressionOperationProps(props);
    validateCompleteRegressionWeights(props.id ?? 'spatial-error-gm', props.weights);
    const id = props.id ?? 'spatial-error-gm';
    validateOptions(id, props);
    validateCapacity(id, 'rowCount', props.rowCount);
    if (props.tileRowCount !== undefined) validateCapacity(id, 'tileRowCount', props.tileRowCount);
    const operation = new GPUSpatialErrorGMOperation(props);
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
        const result = computeSpatialErrorGMOnCPU(
          {
            weights: {
              offsets: getCPUOperationUint32Rows(id, inputs, 'offsets'),
              neighbors: getCPUOperationUint32Rows(id, inputs, 'neighbors'),
              weights: getCPUOperationFloat32Rows(id, inputs, 'weights')
            },
            predictors: getCPUOperationFloat32Rows(id, inputs, 'predictors'),
            response,
            traceParameters: props.traceParameters
              ? getCPUOperationUint32Rows(id, inputs, 'traceParameters')
              : undefined
          },
          props
        );
        return {
          [props.output.status.code]: result.status,
          [props.output.table]: result.table,
          [props.output.summary]: result.summary,
          ...(props.output.residuals ? {[props.output.residuals]: result.residuals} : {})
        };
      }
    });
  }
}
