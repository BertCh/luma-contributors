// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  snapshotRestrictedRegimeKernelStatus,
  snapshotRestrictedRegimeKernelInformation
} from './restricted-regime-kernel-snapshot';
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
import {
  factorCPURegressionCorrelation,
  fitCPURegressionWithQR,
  solveCPURegressionFactor
} from './cpu-regression-linear-algebra';
import {getCPUDyadicLeastSquaresResidualSquares} from './cpu-spatial-error-likelihood-polynomial';
import {getChiSquareSurvival} from './ordinary-least-squares-statistics';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';
import type {
  RestrictedRegimeRobustOptions,
  RestrictedRegimeRobustOutput
} from './restricted-regime-robust';

/** CPU arrays corresponding to the HAC kernel views, in the original observation row space. */
export type CPURestrictedRegimeKernel = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
  status?: Partial<Record<string, Uint32Array>>;
  information?: GPUSpatialWeights['information'];
};

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
> & {robust?: RestrictedRegimeRobustOptions<CPURestrictedRegimeKernel>};
/** Scalar vector layouts of the GPU contributor. */
export type CPURestrictedRegimeRegressionResult = {
  coefficients: Float32Array;
  covariance: Float32Array;
  fitted: Float32Array;
  chow: Float32Array;
  stability: Float32Array;
  status: Uint32Array;
  robust?: RestrictedRegimeRobustOutput<Float32Array, Uint32Array>;
};

function validateOptions(
  id: string,
  options: Pick<CPURestrictedRegimeRegressionOptions, 'predictorCount' | 'regimeCount' | 'common'>
): void {
  validateCapacity(id, 'predictorCount', options.predictorCount);
  validateCapacity(id, 'regimeCount', options.regimeCount, 2);
  validateRequirement(
    id,
    options.common.length === options.predictorCount + 1 &&
      options.common.every(value => typeof value === 'boolean'),
    'common must hold one boolean for the intercept and each predictor'
  );
}

function validateKernelInformation(
  id: string,
  information: GPUSpatialWeights['information'],
  rowCount: number
): void {
  validateRequirement(
    id,
    !information || information.rowCount === rowCount,
    'robust.kernelWeights.information.rowCount must equal response length'
  );
  validateRequirement(
    id,
    information?.transform !== 'row' &&
      information?.symmetric !== false &&
      information?.hasSelfWeights !== false,
    'robust.kernelWeights requires a symmetric unstandardized kernel with unit diagonal; use kernel weights with self weights'
  );
}

/**
 * Fits both nested models by norm-scaled Householder QR in float64. The coefficient order is
 * source-column-major, shared once or repeated by regime. Returns pooled restricted covariance,
 * Chow [F, numerator df, denominator df], and per-column plus joint Wald [chi-square, df, p].
 *
 * What differs from the GPU contributor:
 * - Float64 QR replaces shifted float32 normal equations; response origins are per regime when
 *   intercepts vary, and pooled otherwise. These origins are medians of the complete rows,
 *   independent of row order. When intercepts vary, predictor medians condition the restricted
 *   design before fitting and residual formation; coefficients and inverse roots are transformed
 *   back to the original basis. The unrestricted model uses per-regime response medians.
 *   Compare well-conditioned outputs at 5e-3 relative tolerance (covariances at 5e-4
 *   of their diagonal geometric mean). Fitted rows are evaluated before output rounding.
 * - Shader-only predictor, regime, 16-coefficient and 2^24-row limits are dropped. Numerical rank
 *   uses a norm-scaled QR pivot floor of 64 * Number.EPSILON, rather than GPU Cholesky 1e-3;
 *   some GPU status-2/3 fits succeed here. Empty regimes still give status 3 when the restricted
 *   model fits, with its coefficients/covariance/fitted retained and classical Chow/stability
 *   unavailable. Optional robust inference needs only the restricted fit and n > restricted k.
 * - Wald covariance is factored through a QR square root, avoiding raw covariance cancellation.
 *   The public float64 chi-square survival function supplies p-values (shared 200-iteration cap).
 *   A statistic exceeding float32 range rounds to Infinity; its float64 p-value may round to zero.
 * - Subnormals are retained; non-finite observations are excluded exactly as by the contributor.
 * - Optional robust accepts the same white/HC1 or HAC option with CPU kernel arrays. Scores and
 *   contrasts use float64 QR roots rather than the GPU's shifted inverse matrix; status meanings,
 *   original-row masking, unit kernel diagonal and suppression of invalid inference are identical.
 *   Every contrast factor pivot must exceed its propagated residual-formation and accumulation
 *   uncertainty before inference is published; the GPU uses its float32 tile/CSR error budget.
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
  const robust = options.robust;
  const rowCount = input.response.length;
  if (robust) {
    validateRequirement(
      id,
      robust.type === 'white' || robust.type === 'hac',
      "robust.type must be 'white' or 'hac'"
    );
    if (robust.type === 'white')
      validateRequirement(
        id,
        robust.smallSample === undefined || typeof robust.smallSample === 'boolean',
        'robust.smallSample must be boolean'
      );
    else {
      validateKernelInformation(id, robust.kernelWeights.information, rowCount);
      validateRequirement(
        id,
        robust.kernelWeights.offsets.length === rowCount + 1,
        'robust.kernelWeights.offsets length must equal response length + 1'
      );
      validateRequirement(
        id,
        robust.kernelWeights.neighbors.length === robust.kernelWeights.weights.length,
        'robust.kernelWeights.weights length must equal neighbors length'
      );
      for (const values of Object.values(robust.kernelWeights.status ?? {}))
        if (values)
          validateRequirement(
            id,
            values.length === 1,
            'robust.kernelWeights.status ports must have length 1'
          );
    }
  }
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
    status: Uint32Array.of(1),
    ...(robust
      ? {
          robust: {
            covariance: new Float32Array(coefficientCount ** 2).fill(NaN),
            chow: Float32Array.of(NaN, 0, NaN),
            stability: Float32Array.from({length: (predictorCount + 2) * 3}, (_, slot) =>
              slot % 3 === 1 ? 0 : NaN
            ),
            status: Uint32Array.of(1)
          }
        }
      : {})
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
  if (usedRows.length <= (robust ? coefficientCount : unrestrictedCount)) return result;
  const median = (values: number[]) => {
    values.sort((first, second) => first - second);
    return values[Math.floor(values.length / 2)];
  };
  const predictorOrigins = Array.from({length: predictorCount}, (_, column) =>
    median(usedRows.map(row => input.predictors[row * predictorCount + column]))
  );
  const regimeOrigins = Array.from({length: regimeCount}, (_, regime) => {
    const values = observed.filter((_, position) => input.regimes[usedRows[position]] === regime);
    return values.length ? median(values) : NaN;
  });
  const pooledOrigin = median(observed.slice());
  const restrictedOrigins = usedRows.map(row =>
    common[0] ? pooledOrigin : regimeOrigins[input.regimes[row]]
  );
  const unrestrictedOrigins = usedRows.map(row => regimeOrigins[input.regimes[row]]);
  const shiftedResponse = observed.map((value, row) => value - restrictedOrigins[row]);
  const unrestrictedResponse = observed.map((value, row) => value - unrestrictedOrigins[row]);
  const transform = Array.from({length: coefficientCount}, (_, row) =>
    Array.from({length: coefficientCount}, (_, column): number => (row === column ? 1 : 0))
  );
  if (!common[0]) {
    for (let column = 1; column <= predictorCount; column++) {
      for (let regime = 0; regime < regimeCount; regime++)
        transform[regime][offsets[column] + (common[column] ? 0 : regime)] =
          -predictorOrigins[column - 1];
      for (let position = 0; position < usedRows.length; position++)
        restrictedDesign[position][
          offsets[column] + (common[column] ? 0 : input.regimes[usedRows[position]])
        ] -= predictorOrigins[column - 1];
    }
  }
  const restricted = fitCPURegressionWithQR(restrictedDesign, shiftedResponse);
  if (!restricted) {
    result.status[0] = usedRows.length <= unrestrictedCount ? 1 : 2;
    return result;
  }
  const unrestricted =
    usedRows.length > unrestrictedCount
      ? fitCPURegressionWithQR(unrestrictedDesign, unrestrictedResponse)
      : undefined;
  result.status[0] = usedRows.length <= unrestrictedCount ? 1 : unrestricted ? 0 : 3;
  const dot = (first: readonly number[], second: readonly number[]) =>
    first.reduce((sum, value, column) => sum + value * second[column], 0);
  const rawEstimate = transform.map(row => dot(row, restricted.estimate));
  const rawInverseRoot = transform.map(row =>
    restricted.inverseRoot[0].map((_, column) =>
      row.reduce((sum, value, inner) => sum + value * restricted.inverseRoot[inner][column], 0)
    )
  );
  const positions = new Map(usedRows.map((row, position) => [row, position]));
  const restrictedFits = restrictedDesign.map(row => dot(row, restricted.estimate));
  const residualSquares = restrictedFits.reduce(
    (sum, value, row) => sum + (shiftedResponse[row] - value) ** 2,
    0
  );
  const variance = residualSquares / (usedRows.length - coefficientCount);
  const coefficients = rawEstimate.slice();
  for (let regime = 0; regime < (common[0] ? 1 : regimeCount); regime++)
    coefficients[regime] += common[0] ? pooledOrigin : regimeOrigins[regime];
  if (result.status[0] !== 1) result.coefficients.set(coefficients);
  for (let first = 0; first < coefficientCount; first++) {
    for (let second = 0; second < coefficientCount; second++)
      if (result.status[0] !== 1)
        result.covariance[first * coefficientCount + second] =
          variance * dot(rawInverseRoot[first], rawInverseRoot[second]);
  }
  restrictedFits.forEach((value, row) => {
    if (result.status[0] !== 1) result.fitted[usedRows[row]] = restrictedOrigins[row] + value;
  });
  const unrestrictedFits = unrestricted
    ? unrestrictedDesign.map(row => dot(row, unrestricted.estimate))
    : [];
  const unrestrictedResiduals = unrestrictedFits.map(
    (value, row) => unrestrictedResponse[row] - value
  );
  const fullSquares = unrestrictedResiduals.reduce((sum, value) => sum + value ** 2, 0);
  const fitSquares = restrictedFits.reduce(
    (sum, value, row) => sum + (shiftedResponse[row] - value - unrestrictedResiduals[row]) ** 2,
    0
  );
  const restrictionCount = unrestrictedCount - coefficientCount;
  if (unrestricted) {
    // Include the magnitude of every fitted term before cancellation. If roundoff could
    // explain all the residuals, recover the unrestricted RSS from the original dyadic data.
    const fittedScale = unrestrictedDesign.reduce((largest, row, position) => {
      const fittedMagnitude = row.reduce(
        (sum, value, column) => sum + Math.abs(value * unrestricted.estimate[column]),
        0
      );
      return Math.max(largest, fittedMagnitude + Math.abs(unrestrictedResponse[position]));
    }, 0);
    const roundingThreshold = 64 * Number.EPSILON * unrestrictedCount * fittedScale;
    const nearExact =
      restrictionCount > 0 &&
      unrestrictedResiduals.every(residual => Math.abs(residual) <= roundingThreshold);
    const unrestrictedSquares = nearExact
      ? getCPUDyadicLeastSquaresResidualSquares(unrestrictedDesign, observed)
      : fullSquares;
    result.chow.set([
      restrictionCount > 0 && unrestrictedSquares > 0
        ? fitSquares /
          restrictionCount /
          (unrestrictedSquares / (usedRows.length - unrestrictedCount))
        : NaN,
      restrictionCount,
      usedRows.length - unrestrictedCount
    ]);
  }
  if (robust && result.robust) {
    const output = result.robust;
    const dot = (first: readonly number[], second: readonly number[]) =>
      first.reduce((sum, value, column) => sum + value * second[column], 0);
    const residualErrors = restrictedDesign.map(
      (row, position) =>
        64 *
        Number.EPSILON *
        coefficientCount *
        (Math.abs(shiftedResponse[position]) +
          row.reduce(
            (sum, value, column) => sum + Math.abs(value * restricted.estimate[column]),
            0
          ))
    );
    const scoreInfluences: number[][] = [];
    const scores = restrictedDesign.map((row, position) => {
      const residual = shiftedResponse[position] - restrictedFits[position];
      const projection = restricted.inverseRoot[0].map((_, column) =>
        dot(
          row,
          restricted.inverseRoot.map(values => values[column])
        )
      );
      const influences = rawInverseRoot.map(root => dot(root, projection));
      scoreInfluences.push(influences);
      return influences.map(value => value * residual);
    });
    const smoothed = scores.map(row => row.slice());
    let invalidKernel = false;
    if (robust.type === 'hac') {
      const kernel = robust.kernelWeights;
      invalidKernel = [
        'overflow',
        'candidateOverflow',
        'iterationLimitReached',
        'iterationLimit',
        'invalidInput',
        'invalidCount',
        'unresolvedCount',
        'approximation'
      ].some(name => (kernel.status?.[name]?.[0] ?? 0) !== 0);
      invalidKernel ||= Boolean(
        (kernel.status?.['converged'] && kernel.status['converged'][0] === 0) ||
          (kernel.status?.['complete'] && kernel.status['complete'][0] === 0)
      );
      const retainedCount = kernel.offsets[rowCount];
      const accessibleCount = Math.min(
        retainedCount,
        kernel.status?.['count']?.[0] ?? retainedCount,
        kernel.neighbors.length
      );
      invalidKernel ||= (kernel.status?.['requiredCount']?.[0] ?? 0) > accessibleCount;
      invalidKernel ||=
        retainedCount > accessibleCount ||
        (kernel.status?.['count'] !== undefined && kernel.status['count'][0] !== retainedCount);
      invalidKernel ||=
        kernel.offsets[0] !== 0 ||
        kernel.offsets.some(
          (offset, row) => offset > accessibleCount || (row > 0 && offset < kernel.offsets[row - 1])
        );
      // Validate the caller's complete CSR, including masked rows; masking never repairs a kernel.
      for (let row = 0; row < rowCount; row++) {
        if (invalidKernel) break;
        const begin = kernel.offsets[row];
        const end = kernel.offsets[row + 1];
        let diagonal = false;
        if (begin > end || end > kernel.neighbors.length) {
          invalidKernel = true;
          continue;
        }
        for (let slot = begin; slot < end; slot++) {
          const neighbor = kernel.neighbors[slot];
          const weight = kernel.weights[slot];
          if (
            neighbor >= rowCount ||
            !Number.isFinite(weight) ||
            weight < 0 ||
            (slot > begin && neighbor <= kernel.neighbors[slot - 1])
          ) {
            invalidKernel = true;
            continue;
          }
          if (neighbor === row && weight === 1) diagonal = true;
          let low = kernel.offsets[neighbor];
          let high = kernel.offsets[neighbor + 1];
          const otherEnd = high;
          while (low < high) {
            const middle = low + Math.floor((high - low) / 2);
            if (kernel.neighbors[middle] < row) low = middle + 1;
            else high = middle;
          }
          if (low >= otherEnd || kernel.neighbors[low] !== row || kernel.weights[low] !== weight)
            invalidKernel = true;
        }
        if (!diagonal) invalidKernel = true;
      }
      usedRows.forEach((row, position) => {
        smoothed[position].fill(0);
        if (invalidKernel) return;
        for (let slot = kernel.offsets[row]; slot < kernel.offsets[row + 1]; slot++) {
          const neighbor = positions.get(kernel.neighbors[slot]);
          if (neighbor === undefined) continue;
          for (let column = 0; column < coefficientCount; column++) {
            smoothed[position][column] += kernel.weights[slot] * scores[neighbor][column];
          }
        }
      });
    }
    output.status[0] = invalidKernel ? 2 : 0;
    if (!invalidKernel) {
      const correction =
        robust.type === 'white' && robust.smallSample
          ? usedRows.length / (usedRows.length - coefficientCount)
          : 1;
      const covariance = Array.from({length: coefficientCount}, (_, first) =>
        Array.from(
          {length: coefficientCount},
          (_, second) =>
            correction *
            scores.reduce(
              (sum, row, position) =>
                sum +
                0.5 *
                  (row[first] * smoothed[position][second] +
                    row[second] * smoothed[position][first]),
              0
            )
        )
      );
      output.covariance.set(covariance.flat());
      if (!output.covariance.every(Number.isFinite)) output.status[0] = 3;
      for (let test = 0; test < predictorCount + 2 && output.status[0] === 0; test++) {
        const contrasts: [number, number][] = [];
        const contrastColumns: number[] = [];
        common.forEach((shared, column) => {
          if (!shared && (test === predictorCount + 1 || test === column))
            for (let regime = 1; regime < regimeCount; regime++) {
              contrasts.push([offsets[column] + regime, offsets[column]]);
              contrastColumns.push(column);
            }
        });
        if (!contrasts.length) continue;
        const differences = contrasts.map(
          ([positive, negative]) => coefficients[positive] - coefficients[negative]
        );
        // Project the scores before multiplying: common large predictor shifts cancel here.
        const contrastScores = scores.map(row =>
          contrasts.map(([positive, negative]) => row[positive] - row[negative])
        );
        const contrastInfluences = scoreInfluences.map(row =>
          contrasts.map(([positive, negative]) => row[positive] - row[negative])
        );
        const contrastErrors = scores.map(() => contrasts.map(() => 0));
        const contrastSmoothedErrors = scores.map(() => contrasts.map(() => 0));
        const contrastSmoothed = smoothed.map(row =>
          contrasts.map(([positive, negative]) => row[positive] - row[negative])
        );
        // An invertible row operation preserves the joint hypothesis. Rebase the intercept
        // contrasts at a predictor origin before forming covariance or normalizing its diagonal.
        contrasts.forEach(([positive, negative], first) => {
          if (contrastColumns[first] !== 0) return;
          contrasts.forEach(([otherPositive, otherNegative], second) => {
            const column = contrastColumns[second];
            if (column === 0 || positive - negative !== otherPositive - otherNegative) return;
            // A median stays near the bulk of the design even with a high-leverage first row.
            // Its value depends on the multiset, never on the order of complete rows.
            const weight = predictorOrigins[column - 1];
            differences[first] += weight * differences[second];
            for (let row = 0; row < usedRows.length; row++) {
              contrastScores[row][first] += weight * contrastScores[row][second];
              contrastSmoothed[row][first] += weight * contrastSmoothed[row][second];
              contrastInfluences[row][first] += weight * contrastInfluences[row][second];
            }
          });
        });
        // Form residual-error projections in the chosen basis, before absolute values.
        // Otherwise a large predictor shift would turn an exact cancellation into a huge bound.
        for (let row = 0; row < usedRows.length; row++)
          for (let column = 0; column < contrasts.length; column++)
            contrastErrors[row][column] =
              Math.abs(contrastInfluences[row][column]) * residualErrors[row];
        for (let row = 0; row < usedRows.length; row++) {
          contrastSmoothedErrors[row].fill(0);
          if (robust.type === 'white') contrastSmoothedErrors[row] = contrastErrors[row].slice();
          else {
            const kernel = robust.kernelWeights;
            for (
              let slot = kernel.offsets[usedRows[row]];
              slot < kernel.offsets[usedRows[row] + 1];
              slot++
            ) {
              const neighbor = positions.get(kernel.neighbors[slot]);
              if (neighbor === undefined) continue;
              for (let column = 0; column < contrasts.length; column++)
                contrastSmoothedErrors[row][column] +=
                  kernel.weights[slot] * contrastErrors[neighbor][column];
            }
          }
        }
        const matrix = contrasts.map((_, first) =>
          contrasts.map(
            (_, second) =>
              correction *
              contrastScores.reduce(
                (sum, row, position) =>
                  sum +
                  0.5 *
                    (row[first] * contrastSmoothed[position][second] +
                      row[second] * contrastSmoothed[position][first]),
                0
              )
          )
        );
        // Bound cancellation in the signed meat BEFORE normalizing its diagonal to one.
        // A unit-diagonal factorization cannot distinguish numerical noise from information.
        const accumulationScales = contrasts.map(
          (_, column) =>
            contrastScores.reduce((sum, row, position) => {
              if (robust.type !== 'hac') return sum + row[column] ** 2;
              const kernel = robust.kernelWeights;
              let absoluteNeighborSum = 0;
              for (
                let slot = kernel.offsets[usedRows[position]];
                slot < kernel.offsets[usedRows[position] + 1];
                slot++
              ) {
                const neighbor = positions.get(kernel.neighbors[slot]);
                if (neighbor !== undefined)
                  absoluteNeighborSum +=
                    kernel.weights[slot] * Math.abs(contrastScores[neighbor][column]);
              }
              return sum + Math.abs(row[column]) * absoluteNeighborSum;
            }, 0) * correction
        );
        const scales = matrix.map((row, column) =>
          row[column] > 64 * Number.EPSILON * usedRows.length * accumulationScales[column]
            ? Math.sqrt(row[column])
            : NaN
        );
        const factor = scales.every(value => value > 0 && Number.isFinite(value))
          ? factorCPURegressionCorrelation(
              matrix.map((row, first) =>
                row.map((value, second) => value / scales[first] / scales[second])
              )
            )
          : undefined;
        // Certify every Schur pivot against the uncertainty in its original quadratic form.
        // Checking marginal diagonals alone admits rank-one meat with two positive marginals.
        let certified = Boolean(factor);
        if (factor) {
          const errors = contrasts.map((_, first) =>
            contrasts.map((_, second) => {
              const residualError =
                contrastScores.reduce(
                  (sum, row, position) =>
                    sum +
                    contrastErrors[position][first] *
                      (Math.abs(contrastSmoothed[position][second]) +
                        contrastSmoothedErrors[position][second]) +
                    Math.abs(row[first]) * contrastSmoothedErrors[position][second],
                  0
                ) * correction;
              return (
                (64 *
                  Number.EPSILON *
                  usedRows.length *
                  Math.sqrt(accumulationScales[first] * accumulationScales[second]) +
                  residualError) /
                scales[first] /
                scales[second]
              );
            })
          );
          const elimination = contrasts.map((_, row) =>
            contrasts.map((_, column): number => (row === column ? 1 : 0))
          );
          for (let row = 0; row < contrasts.length; row++) {
            for (let previous = 0; previous < row; previous++) {
              const weight = factor[row][previous] / factor[previous][previous];
              for (let column = 0; column < contrasts.length; column++)
                elimination[row][column] -= weight * elimination[previous][column];
            }
            const uncertainty = elimination[row].reduce(
              (sum, left, first) =>
                sum +
                elimination[row].reduce(
                  (inner, right, second) => inner + Math.abs(left * right) * errors[first][second],
                  0
                ),
              0
            );
            certified &&= factor[row][row] ** 2 > uncertainty;
          }
        }
        if (!factor || !certified) {
          output.status[0] = 3;
          break;
        }
        const right = differences.map((value, column) => value / scales[column]);
        const statistic = dot(right, solveCPURegressionFactor(factor, right));
        if (!Number.isFinite(statistic) || statistic < 0) {
          output.status[0] = 3;
          break;
        }
        output.stability.set(
          [statistic, contrasts.length, getChiSquareSurvival(statistic, contrasts.length)],
          test * 3
        );
      }
      if (output.status[0] === 0)
        output.chow.set(output.stability.subarray((predictorCount + 1) * 3));
      else {
        output.covariance.fill(NaN);
        output.stability.set(
          Float32Array.from({length: output.stability.length}, (_, slot) =>
            slot % 3 === 1 ? 0 : NaN
          )
        );
      }
    }
  }
  if (!unrestricted) return result;
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
      return originDifference + (rawEstimate[positive] - rawEstimate[negative]);
    });
    const contrastRoots = contrasts.map(([positive, negative]) =>
      rawInverseRoot[positive].map((value, column) => value - rawInverseRoot[negative][column])
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
    props = {
      ...props,
      common: [...props.common],
      output: {...props.output, robust: props.output.robust ? {...props.output.robust} : undefined},
      robust: props.robust ? {...props.robust} : undefined
    };
    if (props.robust?.type === 'hac') {
      const {offsets, neighbors, weights, status, information} = props.robust.kernelWeights;
      props.robust = {
        ...props.robust,
        kernelWeights: {
          offsets,
          neighbors,
          weights,
          status: status && snapshotRestrictedRegimeKernelStatus(status),
          information: snapshotRestrictedRegimeKernelInformation(information)
        }
      };
    }
    const id = props.id ?? 'restricted-regime-regression';
    if (props.robust?.type === 'hac')
      validateKernelInformation(id, props.robust.kernelWeights.information, props.rowCount);
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
          {
            ...props,
            robust:
              props.robust?.type === 'hac'
                ? {
                    type: 'hac',
                    kernelWeights: {
                      offsets: getCPUOperationUint32Rows(id, inputs, 'kernelOffsets'),
                      neighbors: getCPUOperationUint32Rows(id, inputs, 'kernelNeighbors'),
                      weights: getCPUOperationFloat32Rows(id, inputs, 'kernelWeights'),
                      information: props.robust.kernelWeights.information,
                      status: Object.fromEntries(
                        Object.keys(props.robust.kernelWeights.status ?? {}).map(name => [
                          name,
                          getCPUOperationUint32Rows(id, inputs, `kernel_${name}`)
                        ])
                      )
                    }
                  }
                : props.robust
          }
        );
        return {
          [props.output.coefficients]: result.coefficients,
          [props.output.covariance]: result.covariance,
          [props.output.fitted]: result.fitted,
          [props.output.chow]: result.chow,
          [props.output.status]: result.status,
          ...(props.output.stability ? {[props.output.stability]: result.stability} : {}),
          ...(props.output.robust && result.robust
            ? {
                [props.output.robust.covariance]: result.robust.covariance,
                [props.output.robust.chow]: result.robust.chow,
                [props.output.robust.stability]: result.robust.stability,
                [props.output.robust.status]: result.robust.status
              }
            : {})
        };
      }
    });
  }
}
