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
import {
  GPURestrictedRegimeRegressionOperation,
  type RestrictedRegimeRegressionOperationProps
} from './gpu-restricted-regime-regression-operation';
import {fitCPURegressionWithQR} from './cpu-regression-linear-algebra';
import {getChiSquareSurvival} from './ordinary-least-squares-statistics';

/** Complete-case observations; regimes outside [0, regimeCount) are excluded. */
export type CPURestrictedRegimeRegressionInput = {
  predictors: Float32Array;
  response: Float32Array;
  regimes: Uint32Array;
  mask?: Uint32Array;
};
/** Model restrictions, intercept first. */
export type CPURestrictedRegimeRegressionOptions = Pick<
  RestrictedRegimeRegressionOperationProps,
  'predictorCount' | 'regimeCount' | 'common'
>;
/** Scalar vector layouts of the GPU contributor. */
export type CPURestrictedRegimeRegressionResult = {
  coefficients: Float32Array;
  covariance: Float32Array;
  fitted: Float32Array;
  chow: Float32Array;
  stability: Float32Array;
  status: Uint32Array;
};

function validateOptions(id: string, options: CPURestrictedRegimeRegressionOptions): void {
  validateCapacity(id, 'predictorCount', options.predictorCount);
  validateCapacity(id, 'regimeCount', options.regimeCount, 2);
  validateRequirement(
    id,
    options.common.length === options.predictorCount + 1 &&
      options.common.every(value => typeof value === 'boolean'),
    'common must hold one boolean for the intercept and each predictor'
  );
}

/**
 * Fits both nested models by norm-scaled Householder QR in float64. The coefficient order is
 * source-column-major, shared once or repeated by regime. Returns pooled restricted covariance,
 * Chow [F, numerator df, denominator df], and per-column plus joint Wald [chi-square, df, p].
 *
 * What differs from the GPU contributor:
 * - Float64 QR replaces shifted float32 normal equations; response origins are per regime when
 *   intercepts vary, and pooled otherwise. The unrestricted model always uses per-regime origins.
 *   Compare well-conditioned outputs at 5e-3 relative tolerance (covariances at 5e-4
 *   of their diagonal geometric mean). Fitted rows are evaluated before output rounding.
 * - Shader-only predictor, regime, 16-coefficient and 2^24-row limits are dropped. Numerical rank
 *   uses a norm-scaled QR pivot floor of 64 * Number.EPSILON, rather than GPU Cholesky 1e-3;
 *   some GPU status-2/3 fits succeed here. Empty regimes still give status 3 when the restricted
 *   model fits, with its coefficients/covariance/fitted retained and Chow/stability unavailable.
 * - Wald covariance is factored through a QR square root, avoiding raw covariance cancellation.
 *   The public float64 chi-square survival function supplies p-values (shared 200-iteration cap).
 *   A statistic exceeding float32 range rounds to Infinity; its float64 p-value may round to zero.
 * - Subnormals are retained; non-finite observations are excluded exactly as by the contributor.
 *
 * @throws If predictors, regimes or mask lengths disagree with response, naming the argument.
 */
export function computeRestrictedRegimeRegressionOnCPU(
  input: CPURestrictedRegimeRegressionInput,
  options: CPURestrictedRegimeRegressionOptions
): CPURestrictedRegimeRegressionResult {
  const id = 'computeRestrictedRegimeRegressionOnCPU';
  validateOptions(id, options);
  const {predictorCount, regimeCount, common} = options;
  const rowCount = input.response.length;
  if (input.predictors.length !== rowCount * predictorCount)
    throw new Error(`${id} predictors length must equal response length * predictorCount`);
  if (input.regimes.length !== rowCount)
    throw new Error(`${id} regimes length must equal response length`);
  if (input.mask && input.mask.length !== rowCount)
    throw new Error(`${id} mask length must equal response length`);
  let coefficientCount = 0;
  const offsets = common.map(shared => {
    const offset = coefficientCount;
    coefficientCount += shared ? 1 : regimeCount;
    return offset;
  });
  const unrestrictedCount = (predictorCount + 1) * regimeCount;
  const result: CPURestrictedRegimeRegressionResult = {
    coefficients: new Float32Array(coefficientCount).fill(NaN),
    covariance: new Float32Array(coefficientCount ** 2).fill(NaN),
    fitted: new Float32Array(rowCount).fill(NaN),
    chow: new Float32Array(3).fill(NaN),
    stability: new Float32Array((predictorCount + 2) * 3).fill(NaN),
    status: Uint32Array.of(1)
  };
  for (let column = 0; column < predictorCount + 2; column++) result.stability[column * 3 + 1] = 0;
  const usedRows: number[] = [];
  const restrictedDesign: number[][] = [];
  const unrestrictedDesign: number[][] = [];
  const observed: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    if (
      input.regimes[row] >= regimeCount ||
      (input.mask && input.mask[row] === 0) ||
      !Number.isFinite(input.response[row])
    )
      continue;
    const values = [
      1,
      ...input.predictors.subarray(row * predictorCount, (row + 1) * predictorCount)
    ];
    if (values.some(value => !Number.isFinite(value))) continue;
    const regime = input.regimes[row];
    const restrictedRow = new Array<number>(coefficientCount).fill(0);
    const unrestrictedRow = new Array<number>(unrestrictedCount).fill(0);
    values.forEach((value, column) => {
      restrictedRow[offsets[column] + (common[column] ? 0 : regime)] = value;
      unrestrictedRow[column * regimeCount + regime] = value;
    });
    usedRows.push(row);
    restrictedDesign.push(restrictedRow);
    unrestrictedDesign.push(unrestrictedRow);
    observed.push(input.response[row]);
  }
  if (usedRows.length <= unrestrictedCount) return result;
  const regimeOrigins = new Array<number>(regimeCount).fill(NaN);
  usedRows.forEach((row, position) => {
    const regime = input.regimes[row];
    if (Number.isNaN(regimeOrigins[regime])) regimeOrigins[regime] = observed[position];
  });
  const restrictedOrigins = usedRows.map(row =>
    common[0] ? observed[0] : regimeOrigins[input.regimes[row]]
  );
  const unrestrictedOrigins = usedRows.map(row => regimeOrigins[input.regimes[row]]);
  const shiftedResponse = observed.map((value, row) => value - restrictedOrigins[row]);
  const unrestrictedResponse = observed.map((value, row) => value - unrestrictedOrigins[row]);
  const restricted = fitCPURegressionWithQR(restrictedDesign, shiftedResponse);
  if (!restricted) {
    result.status[0] = 2;
    return result;
  }
  const unrestricted = fitCPURegressionWithQR(unrestrictedDesign, unrestrictedResponse);
  result.status[0] = unrestricted ? 0 : 3;
  const dot = (first: readonly number[], second: readonly number[]) =>
    first.reduce((sum, value, column) => sum + value * second[column], 0);
  const restrictedFits = restrictedDesign.map(row => dot(row, restricted.estimate));
  const residualSquares = restrictedFits.reduce(
    (sum, value, row) => sum + (shiftedResponse[row] - value) ** 2,
    0
  );
  const variance = residualSquares / (usedRows.length - coefficientCount);
  const coefficients = restricted.estimate.slice();
  for (let regime = 0; regime < (common[0] ? 1 : regimeCount); regime++)
    coefficients[regime] += common[0] ? observed[0] : regimeOrigins[regime];
  result.coefficients.set(coefficients);
  for (let first = 0; first < coefficientCount; first++) {
    for (let second = 0; second < coefficientCount; second++)
      result.covariance[first * coefficientCount + second] =
        variance * dot(restricted.inverseRoot[first], restricted.inverseRoot[second]);
  }
  restrictedFits.forEach((value, row) => {
    result.fitted[usedRows[row]] = restrictedOrigins[row] + value;
  });
  if (!unrestricted) return result;
  const unrestrictedFits = unrestrictedDesign.map(row => dot(row, unrestricted.estimate));
  const fullSquares = unrestrictedFits.reduce(
    (sum, value, row) => sum + (unrestrictedResponse[row] - value) ** 2,
    0
  );
  const fitSquares = restrictedFits.reduce(
    (sum, value, row) =>
      sum +
      (shiftedResponse[row] - value - (unrestrictedResponse[row] - unrestrictedFits[row])) ** 2,
    0
  );
  const restrictionCount = unrestrictedCount - coefficientCount;
  result.chow.set([
    restrictionCount > 0 && fullSquares > 0
      ? fitSquares / restrictionCount / (fullSquares / (usedRows.length - unrestrictedCount))
      : NaN,
    restrictionCount,
    usedRows.length - unrestrictedCount
  ]);
  for (let test = 0; test < predictorCount + 2; test++) {
    const contrasts: [number, number][] = [];
    common.forEach((shared, column) => {
      if (!shared && (test === predictorCount + 1 || test === column)) {
        for (let regime = 1; regime < regimeCount; regime++)
          contrasts.push([offsets[column] + regime, offsets[column]]);
      }
    });
    if (contrasts.length === 0 || !(variance > 0)) continue;
    const differences = contrasts.map(([positive, negative]) => {
      const originDifference =
        !common[0] && positive < regimeCount
          ? regimeOrigins[positive] - regimeOrigins[negative]
          : 0;
      return originDifference + (restricted.estimate[positive] - restricted.estimate[negative]);
    });
    const contrastRoots = contrasts.map(([positive, negative]) =>
      restricted.inverseRoot[positive].map(
        (value, column) => value - restricted.inverseRoot[negative][column]
      )
    );
    const transposedRoots = Array.from({length: coefficientCount}, (_, row) =>
      contrastRoots.map(column => column[row])
    );
    const contrastFit = fitCPURegressionWithQR(
      transposedRoots,
      new Array<number>(coefficientCount).fill(0)
    );
    if (!contrastFit) continue;
    let statistic = 0;
    for (let column = 0; column < contrasts.length; column++) {
      let projection = 0;
      for (let row = 0; row < contrasts.length; row++)
        projection += differences[row] * contrastFit.inverseRoot[row][column];
      statistic += projection ** 2 / variance;
    }
    result.stability.set(
      [statistic, contrasts.length, getChiSquareSurvival(statistic, contrasts.length)],
      test * 3
    );
  }
  return result;
}

/** CPU twin with float64 rank decisions and no shader-size limits. */
export class CPURestrictedRegimeRegressionOperation extends GPUCPUOperation {
  constructor(props: RestrictedRegimeRegressionOperationProps) {
    const id = props.id ?? 'restricted-regime-regression';
    validateOptions(id, props);
    validateCapacity(id, 'rowCount', props.rowCount);
    const operation = new GPURestrictedRegimeRegressionOperation(props);
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
        const result = computeRestrictedRegimeRegressionOnCPU(
          {
            predictors: getCPUOperationFloat32Rows(id, inputs, 'predictors'),
            response,
            regimes: getCPUOperationUint32Rows(id, inputs, 'regimes'),
            mask: props.mask ? getCPUOperationUint32Rows(id, inputs, 'mask') : undefined
          },
          props
        );
        return {
          [props.output.coefficients]: result.coefficients,
          [props.output.covariance]: result.covariance,
          [props.output.fitted]: result.fitted,
          [props.output.chow]: result.chow,
          [props.output.status]: result.status,
          ...(props.output.stability ? {[props.output.stability]: result.stability} : {})
        };
      }
    });
  }
}
