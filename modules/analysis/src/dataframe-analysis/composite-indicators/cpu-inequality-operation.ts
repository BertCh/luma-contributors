// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {validateOperationGraphNames} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  getInequalityOperationOutputs,
  type InequalityOperationProps
} from './gpu-inequality-operation';
import {
  ATKINSON_WEIGHT_NORMALIZATION_THRESHOLD,
  GPU_INEQUALITY_GLOBAL_SUMMARY,
  GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH,
  GPU_INEQUALITY_PARAMETER_LENGTH
} from './inequality-parameters';

/** Packed input arrays, with parameters from getGPUInequalityParameterValues. */
export type CPUInequalityInput = {
  values: Float32Array;
  zoneIds: Uint32Array;
  zoneCount: number;
  weights?: Float32Array;
  mask?: Uint32Array;
  parameters: Float32Array;
  lorenzKnotCount?: number;
};

/** Every supported output, at its exact logical length. */
export type CPUInequalityResult = Record<
  'theilT' | 'theilL' | 'atkinson' | 'hoover' | 'palma' | 'mean' | 'lorenzKnots' | 'globalSummary',
  Float32Array
> & {
  /** Weighted Gini coefficient per zone. */
  weightedGini: Float32Array;
  counts: Uint32Array;
};
type IncomeEntry = {value: number; weight: number};

function computeIncomeIndices(
  entries: IncomeEntry[],
  epsilon: number,
  palmaTop: number,
  palmaBottom: number,
  knotCount: number
) {
  entries.sort((left, right) => left.value - right.value);
  let totalWeight = 0;
  let totalIncome = 0;
  // Build suffixes directly: totalWeight - cumulativeWeight can erase a small final weight.
  // Traverse them from the opposite end while retaining the original forward totals order.
  const suffixWeights = new Float64Array(entries.length + 1);
  for (let row = 0; row < entries.length; row++) {
    const entry = entries[row];
    totalWeight += entry.weight;
    totalIncome += entry.weight * entry.value;
    const suffixRow = entries.length - 1 - row;
    suffixWeights[suffixRow] = entries[suffixRow].weight + suffixWeights[suffixRow + 1];
  }
  const result = {
    gini: NaN,
    theilT: NaN,
    theilL: NaN,
    atkinson: NaN,
    hoover: NaN,
    palma: NaN,
    mean: entries.length ? totalIncome / totalWeight : NaN,
    totalWeight,
    totalIncome,
    knots: new Float64Array(knotCount).fill(NaN)
  };
  if (!(totalIncome > 0)) {
    return result;
  }
  const mean = result.mean;
  const normalizeLogWeights =
    epsilon === 1 && totalWeight > ATKINSON_WEIGHT_NORMALIZATION_THRESHOLD;
  let cumulativeWeight = 0;
  let cumulativeIncome = 0;
  let weightedGapTotal = 0;
  let theilTotal = 0;
  let logarithmTotal = 0;
  let absoluteTotal = 0;
  let hasZero = false;
  const populationShares = [0];
  const incomeShares = [0];
  for (let row = 0; row < entries.length; row++) {
    const {value, weight} = entries[row];
    cumulativeWeight += weight;
    cumulativeIncome += weight * value;
    if (row + 1 < entries.length)
      weightedGapTotal +=
        (entries[row + 1].value - value) * cumulativeWeight * suffixWeights[row + 1];
    populationShares.push(cumulativeWeight / totalWeight);
    incomeShares.push(cumulativeIncome / totalIncome);
    absoluteTotal += weight * Math.abs(value - mean);
    if (value > 0) {
      const logarithm = Math.log(value) - Math.log(mean);
      theilTotal += weight * (value / mean) * logarithm;
      const logWeight = normalizeLogWeights ? weight / totalWeight : weight;
      logarithmTotal += logWeight * logarithm;
    } else {
      hasZero = true;
    }
  }
  const evaluateLorenz = (share: number): number => {
    if (share <= 0) return 0;
    if (share >= 1) return 1;
    let lower = 0;
    let upper = entries.length;
    while (lower + 1 < upper) {
      const middle = Math.floor((lower + upper) / 2);
      if (populationShares[middle] < share) lower = middle;
      else upper = middle;
    }
    const width = populationShares[upper] - populationShares[lower];
    return width > 0
      ? incomeShares[lower] +
          ((share - populationShares[lower]) / width) * (incomeShares[upper] - incomeShares[lower])
      : incomeShares[upper];
  };
  result.gini = weightedGapTotal / (totalWeight * totalIncome);
  result.theilT = theilTotal / totalWeight;
  const weightedLogMean = normalizeLogWeights ? logarithmTotal : logarithmTotal / totalWeight;
  result.theilL = hasZero ? NaN : -weightedLogMean;
  const power = 1 - epsilon;
  if (hasZero && epsilon >= 1) {
    // Continuous extension of the geometric/negative-power mean at positive-weight zero income.
    result.atkinson = 1;
  } else if (epsilon === 0) {
    result.atkinson = 0;
  } else if (epsilon === 1) {
    result.atkinson = -Math.expm1(weightedLogMean);
  } else if (Math.abs(power) <= 0.01 && totalWeight > ATKINSON_WEIGHT_NORMALIZATION_THRESHOLD) {
    // With large weights near epsilon one, center before adding and dividing the logarithm by power.
    let centeredMean = 0;
    const logMean = Math.log(mean);
    for (const {value, weight} of entries) {
      const centeredTerm = value === 0 ? -1 : Math.expm1(power * (Math.log(value) - logMean));
      centeredMean += (weight / totalWeight) * centeredTerm;
    }
    result.atkinson = -Math.expm1(Math.log1p(Math.max(-1, centeredMean)) / power);
  } else if (epsilon < 1 || !hasZero) {
    // Log-sum-exp prevents an intermediate power overflow at large epsilon or income ratios.
    let largestTerm = -Infinity;
    for (const {value, weight} of entries) {
      if (value > 0)
        largestTerm = Math.max(
          largestTerm,
          Math.log(weight) + power * (Math.log(value) - Math.log(mean))
        );
    }
    let scaledTotal = 0;
    for (const {value, weight} of entries) {
      if (value > 0)
        scaledTotal += Math.exp(
          Math.log(weight) + power * (Math.log(value) - Math.log(mean)) - largestTerm
        );
    }
    const atkinson = -Math.expm1(
      (largestTerm + Math.log(scaledTotal) - Math.log(totalWeight)) / power
    );
    result.atkinson = Number.isFinite(Math.fround(atkinson)) ? atkinson : NaN;
  }
  result.hoover = absoluteTotal / (2 * totalIncome);
  const bottomShare = evaluateLorenz(palmaBottom);
  result.palma = bottomShare > 0 ? (1 - evaluateLorenz(1 - palmaTop)) / bottomShare : NaN;
  for (let knot = 0; knot < knotCount; knot++)
    result.knots[knot] = evaluateLorenz(knot / (knotCount - 1));
  return result;
}

/**
 * Computes weighted zone indices, Lorenz knots and pooled Theil decomposition in float64.
 * Filters and undefined outputs match GPUInequality, including NaN knots for zero income.
 *
 * What differs from the GPU contributor:
 * - Float64 sums and transcendental functions replace float32; compare ordinary inputs at 2e-5.
 * - Gini sums sorted income gaps times the population weights on each side, preserving small
 *   weighted inequality without subtracting a rounded Lorenz area from one.
 * - Atkinson uses log-sum-exp or a centered power mean, so finite results survive intermediate
 *   powers that overflow on GPU.
 * - The GPU scratch-binding zone limit and 1025-knot cap are dropped; uint32 layout limits remain.
 * - Equal incomes retain input order; this cannot change the exact weighted statistics.
 * - Empty values and zoneIds are accepted: counts and pooled totals are zero; indices, mean and
 *   Lorenz knots are NaN. The GPU contributor requires at least one input row.
 * - Subnormal incomes and weights may be flushed to zero on the GPU and are kept by the twin.
 *
 * @throws If zoneIds, weights or mask lengths disagree with values, parameters is too short,
 * or the zone or knot count is invalid. Argument names are included in errors.
 */
export function computeInequalityOnCPU(input: CPUInequalityInput): CPUInequalityResult {
  const {values, zoneIds, weights, mask, parameters, zoneCount} = input;
  validateCapacity('computeInequalityOnCPU', 'zoneCount', zoneCount, 1, 0xfffffffe);
  const knotCount = input.lorenzKnotCount ?? 11;
  validateCapacity('computeInequalityOnCPU', 'lorenzKnotCount', knotCount, 2);
  for (const [name, rows] of [
    ['zoneIds', zoneIds],
    ['weights', weights],
    ['mask', mask]
  ] as const) {
    if (rows && rows.length !== values.length)
      throw new Error(`computeInequalityOnCPU ${name} length must equal values length`);
  }
  if (parameters.length < GPU_INEQUALITY_PARAMETER_LENGTH)
    throw new Error('computeInequalityOnCPU parameters is too short');
  // The packer documents [epsilon, palmaTopShare, palmaBottomShare, reserved]; no slot constants exist.
  const [epsilon, palmaTop, palmaBottom] = parameters;
  const zones: IncomeEntry[][] = Array.from({length: zoneCount}, () => []);
  const pooled: IncomeEntry[] = [];
  for (let row = 0; row < values.length; row++) {
    const value = values[row];
    const weight = weights ? weights[row] : 1;
    if (
      !Number.isFinite(value) ||
      value < 0 ||
      zoneIds[row] >= zoneCount ||
      (mask && mask[row] === 0) ||
      !Number.isFinite(weight) ||
      weight <= 0
    )
      continue;
    const entry = {value: value === 0 ? 0 : value, weight};
    zones[zoneIds[row]].push(entry);
    pooled.push(entry);
  }
  const zoneResults = zones.map(entries =>
    computeIncomeIndices(entries, epsilon, palmaTop, palmaBottom, knotCount)
  );
  const result: CPUInequalityResult = {
    weightedGini: new Float32Array(zoneCount),
    theilT: new Float32Array(zoneCount),
    theilL: new Float32Array(zoneCount),
    atkinson: new Float32Array(zoneCount),
    hoover: new Float32Array(zoneCount),
    palma: new Float32Array(zoneCount),
    mean: new Float32Array(zoneCount),
    counts: Uint32Array.from(zones, entries => entries.length),
    lorenzKnots: new Float32Array(zoneCount * knotCount),
    globalSummary: new Float32Array(GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH)
  };
  zoneResults.forEach((zoneResult, zone) => {
    result.weightedGini[zone] = zoneResult.gini;
    for (const name of ['theilT', 'theilL', 'atkinson', 'hoover', 'palma', 'mean'] as const)
      result[name][zone] = zoneResult[name];
    result.lorenzKnots.set(zoneResult.knots, zone * knotCount);
  });
  const pooledResult = computeIncomeIndices(pooled, epsilon, palmaTop, palmaBottom, 2);
  let between = 0;
  let within = 0;
  for (const zoneResult of zoneResults) {
    if (zoneResult.totalIncome > 0) {
      const incomeShare = zoneResult.totalIncome / pooledResult.totalIncome;
      const populationShare = zoneResult.totalWeight / pooledResult.totalWeight;
      between += incomeShare * Math.log(incomeShare / populationShare);
      within += incomeShare * zoneResult.theilT;
    }
  }
  const summary = GPU_INEQUALITY_GLOBAL_SUMMARY;
  result.globalSummary[summary.TOTAL_THEIL_T] = pooledResult.theilT;
  result.globalSummary[summary.BETWEEN_THEIL_T] = pooledResult.totalIncome > 0 ? between : NaN;
  result.globalSummary[summary.WITHIN_THEIL_T] = pooledResult.totalIncome > 0 ? within : NaN;
  result.globalSummary[summary.GINI] = pooledResult.gini;
  result.globalSummary[summary.COUNT] = pooled.length;
  result.globalSummary[summary.MEAN] = pooledResult.mean;
  result.globalSummary[summary.TOTAL_WEIGHT] = pooledResult.totalWeight;
  result.globalSummary[summary.TOTAL_INCOME] = pooledResult.totalIncome;
  return result;
}

/** CPU twin with no GPU scratch-size or knot-budget limits. Optional outputs stay optional. */
export class CPUInequalityOperation extends GPUCPUOperation {
  constructor(props: InequalityOperationProps) {
    const id = props.id ?? 'inequality';
    const {zoneCount, lorenzKnotCount} = props;
    const hasWeights = props.weights !== undefined;
    const hasMask = props.mask !== undefined;
    validateCapacity(id, 'zoneCount', props.zoneCount, 1, 0xfffffffe);
    validateCapacity(id, 'lorenzKnotCount', props.lorenzKnotCount ?? 11, 2);
    const outputs = getInequalityOperationOutputs(props);
    validateCapacity(
      id,
      'zoneCount x lorenzKnotCount',
      props.zoneCount * (props.lorenzKnotCount ?? 11)
    );
    const inputs = {
      values: props.values,
      zoneIds: props.zoneIds,
      parameters: props.parameters,
      ...(props.weights !== undefined ? {weights: props.weights} : {}),
      ...(props.mask !== undefined ? {mask: props.mask} : {})
    };
    validateOperationGraphNames(
      id,
      inputs,
      Object.fromEntries(Object.entries(outputs).map(([name, output]) => [name, output.value]))
    );
    super({
      id,
      inputs,
      outputs: Object.fromEntries(
        Object.values(outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      execute({inputs}) {
        const result = computeInequalityOnCPU({
          values: getCPUOperationFloat32Rows(id, inputs, 'values'),
          zoneIds: getCPUOperationUint32Rows(id, inputs, 'zoneIds'),
          parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters'),
          weights: hasWeights ? getCPUOperationFloat32Rows(id, inputs, 'weights') : undefined,
          mask: hasMask ? getCPUOperationUint32Rows(id, inputs, 'mask') : undefined,
          zoneCount,
          lorenzKnotCount
        });
        return Object.fromEntries(
          Object.entries(outputs).map(([name, output]) => [
            output.value,
            result[name === 'gini' ? 'weightedGini' : (name as keyof CPUInequalityResult)]
          ])
        );
      }
    });
  }
}
