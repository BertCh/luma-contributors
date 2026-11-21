// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {getPointPatternRejectionBudget} from './point-pattern-rejection-budget';
import {
  getPointPatternDyadicInteger,
  roundPointPatternDyadicStatistic
} from './point-pattern-exact-statistic-cpu';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  snapshotEnvelopeConstruction,
  resolveEnvelopeSimulationCount,
  resolveEnvelopeStatistic
} from './point-pattern-envelope-construction';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUPointPatternEnvelopeOperation,
  type PointPatternEnvelopeOperationProps
} from './gpu-point-pattern-envelope-operation';

/** Replicate-major simulation curves and their fixed null reference. */
export type CPUPointPatternEnvelopeInput = {
  observed: Float32Array;
  simulations: Float32Array;
  reference?: Float32Array;
  integrationWeights?: Float32Array;
};
/** Inference settings, including the contributor's required exchangeability metadata. */
export type CPUPointPatternEnvelopeOptions = Pick<
  PointPatternEnvelopeOperationProps,
  'simulationCount' | 'rank' | 'statistic' | 'significanceLevel' | 'nullModel' | 'fittedParameters'
>;
/** Outputs in the contributor's scalar-vector layouts. */
export type CPUPointPatternEnvelopeResult = {
  lower: Float32Array;
  upper: Float32Array;
  statistics: Float32Array;
  status: Uint32Array;
  pValueInterval: Float32Array;
};

/** Packed simulations must fit a uint32-indexed vector. */
export const CPU_POINT_PATTERN_ENVELOPE_MAXIMUM_VALUE_COUNT = 0xffffffff;

function validateOptions(
  id: string,
  curveRowCount: number,
  options: CPUPointPatternEnvelopeOptions & {simulationCount: number}
): void {
  validateCapacity(id, 'curveRowCount', curveRowCount);
  validateCapacity(id, 'simulationCount', options.simulationCount, 2);
  validateRequirement(
    id,
    curveRowCount * options.simulationCount <= CPU_POINT_PATTERN_ENVELOPE_MAXIMUM_VALUE_COUNT,
    'curveRowCount * simulationCount must fit uint32'
  );
  const rank = options.rank ?? 1;
  validateRequirement(
    id,
    Number.isInteger(rank) && rank >= 1 && 2 * rank <= options.simulationCount,
    'rank must be an integer with 1 <= rank and 2 * rank <= simulationCount'
  );
  validateRequirement(id, options.nullModel.trim().length > 0, 'nullModel must be non-empty');
  validateRequirement(
    id,
    ['none', 'fixed-external', 'refit-each-replicate'].includes(options.fittedParameters),
    'fittedParameters must be none, fixed-external or refit-each-replicate'
  );
  validateRequirement(
    id,
    options.statistic === undefined ||
      ['maximum-deviation', 'integrated-squared-deviation', 'extreme-rank'].includes(
        options.statistic
      ),
    'statistic must name a supported envelope statistic'
  );
  if (options.statistic === 'extreme-rank')
    validateRequirement(
      id,
      (options.significanceLevel ?? 0.05) > 0 && (options.significanceLevel ?? 0.05) < 1,
      'significanceLevel must lie in (0, 1)'
    );
}

/**
 * Pointwise simulation envelopes and global deviation or exchangeable extreme-midrank tests.
 * All curves participate, ties are conservative, and no invalid replicate is discarded.
 *
 * What differs from the GPU contributor:
 * - Both forms compare maximum and integrated deviations with exact dyadic
 *   integer arithmetic on the stored float32 inputs. Displayed statistics remain
 *   float32; use 2e-6 relative tolerance for ordinary CPU/GPU display comparisons.
 * - GPU-only 4096-row, 4096-simulation and quadratic rank-comparison caps are dropped;
 *   packed simulation length must still fit uint32. Sorting computes exact order statistics.
 * - Finite float64 statistics that overflow float32 round to Infinity with status=0, whereas
 *   the GPU's float32 intermediate overflow invalidates every inferential result.
 * - Extreme-rank mode ignores reference and integrationWeights values, exactly as the kernel
 *   does. Maximum deviation ignores integrationWeights; integrated deviation requires positive finite weights.
 *
 * @throws If simulations, reference or integrationWeights have inconsistent lengths.
 */
export function computePointPatternEnvelopeOnCPU(
  input: CPUPointPatternEnvelopeInput,
  options: CPUPointPatternEnvelopeOptions
): CPUPointPatternEnvelopeResult {
  const count = input.observed.length;
  const id = 'point-pattern-envelope';
  const capturedOptions = snapshotConstructionFields(options, [
    'simulationCount',
    'rank',
    'statistic',
    'significanceLevel',
    'nullModel',
    'fittedParameters'
  ]);
  options = {
    ...capturedOptions,
    statistic: resolveEnvelopeStatistic(id, capturedOptions.statistic)
  };
  validateRequirement(id, count > 0, `observed.length must be at least 1; received ${count}`);
  const simulationCount = resolveEnvelopeSimulationCount(
    'point-pattern-envelope',
    count,
    input.simulations.length,
    options.simulationCount
  );
  const resolvedOptions = {...options, simulationCount};
  validateOptions('point-pattern-envelope', count, resolvedOptions);
  validateRequirement(
    id,
    options.statistic === 'extreme-rank' || input.reference?.length === count,
    `reference.length must equal observed.length ${count}; received ${input.reference?.length}`
  );
  validateRequirement(
    id,
    options.statistic !== 'integrated-squared-deviation' ||
      input.integrationWeights?.length === count,
    `integrationWeights.length must equal observed.length ${count}; received ${input.integrationWeights?.length}`
  );
  const curves = simulationCount + 1;
  const result: CPUPointPatternEnvelopeResult = {
    lower: new Float32Array(count).fill(NaN),
    upper: new Float32Array(count).fill(NaN),
    statistics: Float32Array.of(NaN, NaN, 1 / curves, simulationCount),
    status: Uint32Array.of(1),
    pValueInterval: new Float32Array(2).fill(NaN)
  };
  if (!input.observed.every(Number.isFinite) || !input.simulations.every(Number.isFinite))
    return result;
  const columns = Array.from({length: count}, (_, column) =>
    Array.from({length: curves}, (_, curve) => ({
      curve,
      value:
        curve === simulationCount
          ? input.observed[column]
          : input.simulations[curve * count + column]
    })).sort((first, second) => first.value - second.value || first.curve - second.curve)
  );
  if (options.statistic === 'extreme-rank') {
    const extremes = new Float64Array(curves).fill(curves);
    for (const column of columns) {
      let start = 0;
      while (start < curves) {
        let end = start + 1;
        while (end < curves && column[end].value === column[start].value) end++;
        const midrank = Math.min(start, curves - end) + (end - start + 1) / 2;
        for (let index = start; index < end; index++)
          extremes[column[index].curve] = Math.min(extremes[column[index].curve], midrank);
        start = end;
      }
    }
    const observed = extremes[simulationCount];
    const strict = extremes.filter(rank => rank < observed).length;
    const inclusive = extremes.filter(rank => rank <= observed).length;
    let criticalRank = 1;
    for (const candidate of new Set(extremes))
      if (
        extremes.filter(rank => rank < candidate).length <=
        getPointPatternRejectionBudget(curves, options.significanceLevel ?? 0.05)
      )
        criticalRank = Math.max(criticalRank, candidate);
    result.statistics[0] = observed;
    result.statistics[1] = inclusive / curves;
    result.pValueInterval.set([strict / curves, inclusive / curves]);
    columns.forEach((column, index) => {
      const retained = column.filter(entry => extremes[entry.curve] >= criticalRank);
      result.lower[index] = retained[0]?.value ?? NaN;
      result.upper[index] = retained[retained.length - 1]?.value ?? NaN;
    });
  } else {
    if (
      !input.reference!.every(Number.isFinite) ||
      (options.statistic === 'integrated-squared-deviation' &&
        !input.integrationWeights!.every(weight => Number.isFinite(weight) && weight > 0))
    )
      return result;
    const exactStatistics: bigint[] = [];
    const integrated = options.statistic === 'integrated-squared-deviation';
    const referenceIntegers = Array.from(input.reference!, getPointPatternDyadicInteger);
    const weightIntegers = integrated
      ? Array.from(input.integrationWeights!, getPointPatternDyadicInteger)
      : [];
    for (let curve = 0; curve < curves; curve++) {
      let exactStatistic = 0n;
      for (let column = 0; column < count; column++) {
        const value =
          curve === simulationCount
            ? input.observed[column]
            : input.simulations[curve * count + column];
        const difference = getPointPatternDyadicInteger(value) - referenceIntegers[column];
        if (integrated) {
          exactStatistic += weightIntegers[column] * difference * difference;
        } else {
          const magnitude = difference < 0n ? -difference : difference;
          if (magnitude > exactStatistic) exactStatistic = magnitude;
        }
      }
      exactStatistics.push(exactStatistic);
    }
    const observed = exactStatistics[simulationCount];
    // The observed curve supplies the add-one count.
    const inclusiveCount = exactStatistics.filter(value => value >= observed).length;
    // Differences are in units of 2^-149; the weighted squared products are in units of 2^-447.
    result.statistics[0] = integrated
      ? roundPointPatternDyadicStatistic(observed)
      : Number(observed) * 2 ** -149;
    result.statistics[1] = inclusiveCount / curves;
    const rank = options.rank ?? 1;
    columns.forEach((column, index) => {
      const simulations = column.filter(entry => entry.curve !== simulationCount);
      result.lower[index] = simulations[rank - 1].value;
      result.upper[index] = simulations[simulationCount - rank].value;
    });
  }
  result.status[0] = 0;
  return result;
}

/** CPU twin, without GPU-only curve-size or comparison-budget limits. */
export class CPUPointPatternEnvelopeOperation extends GPUCPUOperation {
  constructor(props: PointPatternEnvelopeOperationProps) {
    props = snapshotEnvelopeConstruction(props);
    const operation = new GPUPointPatternEnvelopeOperation(props);
    const {id} = operation;
    validateOptions(id, props.curveRowCount, {
      ...props,
      simulationCount: props.simulationCount ?? Math.max(2, 2 * (props.rank ?? 1))
    });
    validateRequirement(
      id,
      props.output.pValueInterval === undefined || props.statistic === 'extreme-rank',
      'output.pValueInterval needs statistic extreme-rank'
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
        const observed = getCPUOperationFloat32Rows(id, inputs, 'observed');
        if (observed.length !== props.curveRowCount)
          throw new Error(`${id} observed length must equal curveRowCount`);
        const result = computePointPatternEnvelopeOnCPU(
          {
            observed,
            simulations: getCPUOperationFloat32Rows(id, inputs, 'simulations'),
            reference:
              props.statistic === 'extreme-rank'
                ? undefined
                : getCPUOperationFloat32Rows(id, inputs, 'reference'),
            integrationWeights:
              props.statistic === 'integrated-squared-deviation'
                ? getCPUOperationFloat32Rows(id, inputs, 'integrationWeights')
                : undefined
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUPointPatternEnvelopeResult]
          ])
        );
      }
    });
  }
}
