// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
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
  reference: Float32Array;
  integrationWeights: Float32Array;
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
  options: CPUPointPatternEnvelopeOptions
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
 * - Float64 deviations and sums are rounded only at output; use 2e-6 relative tolerance on
 *   ordinary float outputs. Near arithmetic ties, exceedance counts and p-values can differ.
 * - GPU-only 4096-row, 4096-simulation and quadratic rank-comparison caps are dropped;
 *   packed simulation length must still fit uint32. Sorting computes exact order statistics.
 * - Finite float64 statistics that overflow float32 round to Infinity with status=0, whereas
 *   the GPU's float32 intermediate overflow invalidates every inferential result.
 * - Extreme-rank mode ignores reference and integrationWeights values, exactly as the kernel
 *   does (their shapes are still required). Other modes reject nonfinite/nonpositive weights.
 *
 * @throws If simulations, reference or integrationWeights have inconsistent lengths.
 */
export function computePointPatternEnvelopeOnCPU(
  input: CPUPointPatternEnvelopeInput,
  options: CPUPointPatternEnvelopeOptions
): CPUPointPatternEnvelopeResult {
  const count = input.observed.length;
  if (count === 0) throw new Error('observed must contain at least one ordinate');
  validateOptions('point-pattern-envelope', count, options);
  if (input.simulations.length !== count * options.simulationCount)
    throw new Error('simulations length must equal observed.length * simulationCount');
  if (input.reference.length !== count)
    throw new Error('reference length must equal observed.length');
  if (input.integrationWeights.length !== count)
    throw new Error('integrationWeights length must equal observed.length');
  const curves = options.simulationCount + 1;
  const result: CPUPointPatternEnvelopeResult = {
    lower: new Float32Array(count).fill(NaN),
    upper: new Float32Array(count).fill(NaN),
    statistics: Float32Array.of(NaN, NaN, 1 / curves, options.simulationCount),
    status: Uint32Array.of(1),
    pValueInterval: new Float32Array(2).fill(NaN)
  };
  if (!input.observed.every(Number.isFinite) || !input.simulations.every(Number.isFinite))
    return result;
  const columns = Array.from({length: count}, (_, column) =>
    Array.from({length: curves}, (_, curve) => ({
      curve,
      value:
        curve === options.simulationCount
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
    const observed = extremes[options.simulationCount];
    const strict = extremes.filter(rank => rank < observed).length;
    const inclusive = extremes.filter(rank => rank <= observed).length;
    let criticalRank = 1;
    for (let candidate = 1; candidate <= Math.ceil(curves / 2); candidate++)
      if (
        extremes.filter(rank => rank < candidate).length <=
        Math.floor((options.significanceLevel ?? 0.05) * curves)
      )
        criticalRank = candidate;
    result.statistics[0] = observed;
    result.statistics[1] = inclusive / curves;
    result.pValueInterval.set([strict / curves, inclusive / curves]);
    columns.forEach((column, index) => {
      result.lower[index] = column[criticalRank - 1].value;
      result.upper[index] = column[curves - criticalRank].value;
    });
  } else {
    if (
      !input.reference.every(Number.isFinite) ||
      !input.integrationWeights.every(weight => Number.isFinite(weight) && weight > 0)
    )
      return result;
    const statistics = new Float64Array(curves);
    for (let curve = 0; curve < curves; curve++) {
      let statistic = 0;
      for (let column = 0; column < count; column++) {
        const value =
          curve === options.simulationCount
            ? input.observed[column]
            : input.simulations[curve * count + column];
        const deviation = Math.abs(value - input.reference[column]);
        statistic =
          options.statistic === 'integrated-squared-deviation'
            ? statistic + input.integrationWeights[column] * deviation ** 2
            : Math.max(statistic, deviation);
      }
      statistics[curve] = statistic;
    }
    const observed = statistics[options.simulationCount];
    const exceedances = statistics
      .subarray(0, options.simulationCount)
      .filter(value => value >= observed).length;
    result.statistics[0] = observed;
    result.statistics[1] = (exceedances + 1) / curves;
    const rank = options.rank ?? 1;
    columns.forEach((column, index) => {
      const simulations = column.filter(entry => entry.curve !== options.simulationCount);
      result.lower[index] = simulations[rank - 1].value;
      result.upper[index] = simulations[options.simulationCount - rank].value;
    });
  }
  result.status[0] = 0;
  return result;
}

/** CPU twin, without GPU-only curve-size or comparison-budget limits. */
export class CPUPointPatternEnvelopeOperation extends GPUCPUOperation {
  constructor(props: PointPatternEnvelopeOperationProps) {
    const operation = new GPUPointPatternEnvelopeOperation(props);
    const {id} = operation;
    validateOptions(id, props.curveRowCount, props);
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
            reference: getCPUOperationFloat32Rows(id, inputs, 'reference'),
            integrationWeights: getCPUOperationFloat32Rows(id, inputs, 'integrationWeights')
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
