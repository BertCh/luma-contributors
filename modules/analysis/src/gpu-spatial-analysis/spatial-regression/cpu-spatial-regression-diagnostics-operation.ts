// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUSpatialRegressionDiagnosticsOperation,
  snapshotDiagnosticsOperationProps,
  type SpatialRegressionDiagnosticsOperationProps
} from './gpu-spatial-regression-diagnostics-operation';
import {validateCPURegressionWeights, type CPURegressionWeights} from './cpu-regression-weights';
import {fitCPURegressionWithQR} from './cpu-regression-linear-algebra';
import {getChiSquareSurvival} from './ordinary-least-squares-statistics';
import {
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH
} from './spatial-regression-diagnostics-parameters';

/** Supplied OLS residuals are consumed as-is, with common finite/unmasked row selection. */
export type CPUSpatialRegressionDiagnosticsInput = {
  weights: CPURegressionWeights & {
    status?: {overflow?: Uint32Array; candidateOverflow?: Uint32Array};
  };
  predictors: Float32Array;
  response: Float32Array;
  residuals: Float32Array;
  mask?: Uint32Array;
};
/** Predictor width and optional row restandardization after exclusion. */
export type CPUSpatialRegressionDiagnosticsOptions = Pick<
  SpatialRegressionDiagnosticsOperationProps,
  'predictorCount' | 'restandardizeRows'
>;
/** All six tests, thirteen summary slots and one solver code. */
export type CPUSpatialRegressionDiagnosticsResult = {
  tests: Float32Array;
  summary: Float32Array;
  status: Uint32Array;
};

/**
 * Spatial LM tests and residual Moran inference in float64, using the supplied residuals. Sparse
 * products of an orthonormal design give the low-rank trace corrections without an n-by-n matrix.
 *
 * What differs from the GPU contributor:
 * - Float64 QR and sparse products replace float32 centered Gram solves and reductions.
 *   Information is computed by directly summing squared residuals of W(y-e) against the design,
 *   avoiding the GPU's subtraction of two large squares. Well-conditioned comparisons use
 *   relative tolerance 5e-3, with row counts and test degrees compared exactly.
 * - Shader-only 15-predictor and device binding-size limits are dropped; tileRowCount is ignored.
 *   Design rank uses norm-scaled QR pivots > 64 * Number.EPSILON, so some GPU status-1 designs
 *   succeed. Failure precedence and all status values/NaN layouts are retained.
 * - P-values call the public float64 chi-square survival helper instead of the shader erfc
 *   approximation. Its shared 200-iteration cap remains. Subnormals are retained.
 * - Sorted unique CSR neighbors are checked on the CPU. Invalid neighbor IDs are dropped,
 *   unused capacity is ignored, and non-finite active weights produce status 4. The pure function
 *   accepts empty slot arrays, whereas the GPU needs positive slot capacity.
 * - As in the kernel, Moran's trace correction assumes zero diagonal weights; self-links should
 *   not be supplied (the GPUSpatialWeights self-join invariant).
 *
 * @throws If weights, predictors, response, residuals or mask arrays are inconsistent.
 */
export function computeSpatialRegressionDiagnosticsOnCPU(
  input: CPUSpatialRegressionDiagnosticsInput,
  options: CPUSpatialRegressionDiagnosticsOptions
): CPUSpatialRegressionDiagnosticsResult {
  const id = 'computeSpatialRegressionDiagnosticsOnCPU';
  validateCapacity(id, 'predictorCount', options.predictorCount);
  // This twin publishes status 3 and suppresses diagnostics for either weights loss flag below.
  const rowCount = validateCPURegressionWeights(id, input.weights, 'handled-by-caller');
  const {predictorCount} = options;
  const width = predictorCount + 1;
  if (input.response.length !== rowCount)
    throw new Error(`${id} response length must equal weights rowCount`);
  if (input.residuals.length !== rowCount)
    throw new Error(`${id} residuals length must equal weights rowCount`);
  if (input.predictors.length !== rowCount * predictorCount)
    throw new Error(`${id} predictors length must equal rowCount * predictorCount`);
  if (input.mask && input.mask.length !== rowCount)
    throw new Error(`${id} mask length must equal weights rowCount`);
  const {offsets, neighbors, weights} = input.weights;
  for (let row = 0; row < rowCount; row++) {
    for (let slot = offsets[row] + 1; slot < offsets[row + 1]; slot++) {
      if (neighbors[slot] <= neighbors[slot - 1])
        throw new Error(`${id} weights.neighbors must be sorted and unique within each row`);
    }
  }
  const usedRows: number[] = [];
  const positions = new Int32Array(rowCount).fill(-1);
  for (let row = 0; row < rowCount; row++) {
    if (
      (input.mask && input.mask[row] === 0) ||
      !Number.isFinite(input.response[row]) ||
      !Number.isFinite(input.residuals[row])
    )
      continue;
    let finite = true;
    for (let column = 0; column < predictorCount; column++)
      finite &&= Number.isFinite(input.predictors[row * predictorCount + column]);
    if (finite) {
      positions[row] = usedRows.length;
      usedRows.push(row);
    }
  }
  const usedCount = usedRows.length;
  const result: CPUSpatialRegressionDiagnosticsResult = {
    tests: new Float32Array(GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH).fill(NaN),
    summary: new Float32Array(GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH).fill(NaN),
    status: new Uint32Array(1)
  };
  result.summary[0] = usedCount;
  result.summary[12] = rowCount - usedCount;
  for (let test = 0; test < 6; test++)
    result.tests[test * 3 + 1] = test === 5 ? 0 : test === 4 ? 2 : 1;
  const fail = (code: number) => {
    result.status[0] = code;
    return result;
  };
  if (input.weights.status?.overflow?.[0] || input.weights.status?.candidateOverflow?.[0])
    return fail(3);
  if (usedCount <= width + 1) return fail(2);
  const active = usedRows.map(row => {
    const entries: [number, number][] = [];
    let sum = 0;
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
      const neighbor = neighbors[slot];
      if (neighbor < rowCount && positions[neighbor] >= 0) {
        entries.push([positions[neighbor], weights[slot]]);
        sum += weights[slot];
      }
    }
    const scale = options.restandardizeRows ? (sum > 0 ? 1 / sum : 0) : 1;
    return entries.map(([neighbor, weight]) => [neighbor, weight * scale] as const);
  });
  const response = usedRows.map(row => input.response[row]);
  const residuals = usedRows.map(row => input.residuals[row]);
  const multiply = (vector: readonly number[]) =>
    active.map(entries =>
      entries.reduce((sum, [neighbor, weight]) => sum + weight * vector[neighbor], 0)
    );
  const dot = (first: readonly number[], second: readonly number[]) =>
    first.reduce((sum, value, row) => sum + value * second[row], 0);
  const residualSquares = dot(residuals, residuals);
  const residualLag = dot(residuals, multiply(residuals));
  const responseLag = dot(residuals, multiply(response));
  const fittedLag = multiply(response.map((value, row) => value - residuals[row]));
  let weightSum = 0;
  let squareSum = 0;
  let crossSum = 0;
  active.forEach((entries, row) => {
    for (const [neighbor, weight] of entries) {
      weightSum += weight;
      squareSum += weight ** 2;
      const reverse = active[neighbor].find(([target]) => target === row)?.[1] ?? 0;
      crossSum += weight * reverse;
    }
  });
  if (
    ![
      residualSquares,
      residualLag,
      responseLag,
      dot(fittedLag, fittedLag),
      squareSum,
      crossSum
    ].every(Number.isFinite) ||
    !(residualSquares > 0)
  )
    return fail(4);
  const trace = squareSum + crossSum;
  if (!(weightSum > 0) || !(trace > 0)) return fail(3);
  const means = Array.from({length: predictorCount}, (_, column) => {
    const origin = input.predictors[usedRows[0] * predictorCount + column];
    return (
      origin +
      usedRows.reduce(
        (sum, row) => sum + input.predictors[row * predictorCount + column] - origin,
        0
      ) /
        usedCount
    );
  });
  const design = usedRows.map(row => [
    1,
    ...means.map((mean, column) => input.predictors[row * predictorCount + column] - mean)
  ]);
  const fit = fitCPURegressionWithQR(design, new Array<number>(usedCount).fill(0));
  if (!fit) return fail(1);
  // design * inverseRoot = orthonormal Q, since inverseRoot is (scaled R)^-1.
  const basis = design.map(row =>
    Array.from({length: width}, (_, column) =>
      row.reduce((sum, value, inner) => sum + value * fit.inverseRoot[inner][column], 0)
    )
  );
  const lagBasis = active.map(entries =>
    Array.from({length: width}, (_, column) =>
      entries.reduce((sum, [neighbor, weight]) => sum + weight * basis[neighbor][column], 0)
    )
  );
  const transposedBasis = basis.map(() => new Array<number>(width).fill(0));
  active.forEach((entries, row) => {
    for (const [neighbor, weight] of entries) {
      for (let column = 0; column < width; column++)
        transposedBasis[neighbor][column] += weight * basis[row][column];
    }
  });
  const correction = Array.from({length: width}, (_, first) =>
    Array.from({length: width}, (_, second) =>
      basis.reduce((sum, row, position) => sum + row[first] * lagBasis[position][second], 0)
    )
  );
  const projection = Array.from({length: width}, (_, column) =>
    basis.reduce((sum, row, position) => sum + row[column] * fittedLag[position], 0)
  );
  const fittedLagResidual = fittedLag.map((value, row) => value - dot(basis[row], projection));
  const sigmaSquared = residualSquares / usedCount;
  const information = dot(fittedLagResidual, fittedLagResidual) / sigmaSquared + trace;
  const errorGradient = residualLag / sigmaSquared;
  const lagGradient = responseLag / sigmaSquared;
  const lagStatistic = lagGradient ** 2 / information;
  const errorStatistic = errorGradient ** 2 / trace;
  const robustDenominator = trace - trace ** 2 / information;
  const robustError =
    robustDenominator > 0 && information > trace
      ? (errorGradient - (trace * lagGradient) / information) ** 2 / robustDenominator
      : NaN;
  const robustLag =
    robustDenominator > 0 && information > trace
      ? (lagGradient - errorGradient) ** 2 / (information - trace)
      : NaN;
  const statistics = [
    lagStatistic,
    errorStatistic,
    robustLag,
    robustError,
    lagStatistic + robustError
  ];
  statistics.forEach((statistic, test) => {
    result.tests[test * 3] = statistic;
    result.tests[test * 3 + 2] = Number.isFinite(statistic)
      ? getChiSquareSurvival(statistic, test === 4 ? 2 : 1)
      : NaN;
  });
  const traceCorrection = correction.reduce((sum, row, column) => sum + row[column], 0);
  const traceCorrectionSquared = correction.reduce(
    (sum, row, first) =>
      sum + row.reduce((total, value, second) => total + value * correction[second][first], 0),
    0
  );
  const correctionNormSquared = correction.reduce((sum, row) => sum + dot(row, row), 0);
  const lagNormSquared = lagBasis.reduce((sum, row) => sum + dot(row, row), 0);
  const transposedNormSquared = transposedBasis.reduce((sum, row) => sum + dot(row, row), 0);
  const crossCorrection = transposedBasis.reduce(
    (sum, row, position) => sum + dot(row, lagBasis[position]),
    0
  );
  const degrees = usedCount - width;
  const scale = usedCount / weightSum;
  const moranI = (scale * residualLag) / residualSquares;
  const moranExpectation = (-scale * traceCorrection) / degrees;
  const moranVariance =
    (scale ** 2 *
      (squareSum -
        lagNormSquared -
        transposedNormSquared +
        correctionNormSquared +
        crossSum -
        2 * crossCorrection +
        traceCorrectionSquared +
        traceCorrection ** 2)) /
      (degrees * (degrees + 2)) -
    moranExpectation ** 2;
  const moranZ = moranVariance > 0 ? (moranI - moranExpectation) / Math.sqrt(moranVariance) : NaN;
  const moranP = moranVariance > 0 ? getChiSquareSurvival(moranZ ** 2, 1) : NaN;
  result.tests.set([moranZ, 0, moranP], 15);
  result.summary.set([
    usedCount,
    sigmaSquared,
    trace,
    information,
    weightSum,
    moranI,
    moranExpectation,
    moranVariance,
    moranZ,
    moranP,
    residualLag,
    responseLag,
    rowCount - usedCount
  ]);
  return result;
}

/** CPU diagnostic report; supplied residuals and all output/status layouts are preserved. */
export class CPUSpatialRegressionDiagnosticsOperation extends GPUCPUOperation {
  constructor(props: SpatialRegressionDiagnosticsOperationProps) {
    props = snapshotDiagnosticsOperationProps(props);
    const id = props.id ?? 'spatial-regression-diagnostics';
    validateCapacity(id, 'rowCount', props.rowCount);
    validateCapacity(id, 'predictorCount', props.predictorCount);
    if (props.tileRowCount !== undefined) validateCapacity(id, 'tileRowCount', props.tileRowCount);
    const operation = new GPUSpatialRegressionDiagnosticsOperation(props);
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
        const result = computeSpatialRegressionDiagnosticsOnCPU(
          {
            weights: {
              offsets: getCPUOperationUint32Rows(id, inputs, 'offsets'),
              neighbors: getCPUOperationUint32Rows(id, inputs, 'neighbors'),
              weights: getCPUOperationFloat32Rows(id, inputs, 'weights'),
              status: {
                overflow: props.weights.status?.overflow
                  ? getCPUOperationUint32Rows(id, inputs, 'weightsOverflow')
                  : undefined,
                candidateOverflow: props.weights.status?.candidateOverflow
                  ? getCPUOperationUint32Rows(id, inputs, 'weightsCandidateOverflow')
                  : undefined
              }
            },
            predictors: getCPUOperationFloat32Rows(id, inputs, 'predictors'),
            response,
            residuals: getCPUOperationFloat32Rows(id, inputs, 'residuals'),
            mask: props.mask ? getCPUOperationUint32Rows(id, inputs, 'mask') : undefined
          },
          props
        );
        return {
          [props.output.tests]: result.tests,
          [props.output.summary]: result.summary,
          [props.output.status.code]: result.status
        };
      }
    });
  }
}
