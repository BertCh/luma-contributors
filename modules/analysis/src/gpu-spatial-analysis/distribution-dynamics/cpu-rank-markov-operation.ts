// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {GPURankMarkovOperation, type RankMarkovOperationProps} from './gpu-rank-markov-operation';
import {computeTransitionDiagnosticsOnCPU} from './cpu-transition-diagnostics-operation';

/** Largest population whose dense square transition matrix fits a uint32 element count. */
export const CPU_RANK_MARKOV_MAXIMUM_ROW_COUNT = 65535;

/** Rank vectors and the total opportunities counted by ignored must fit uint32. */
export const CPU_RANK_MARKOV_MAXIMUM_VALUE_COUNT = 0xffffffff;

/** Options of computeRankMarkovOnCPU. GPU-only population and period caps are dropped. */
export type CPURankMarkovOptions = {
  periodCount: number;
  mode?: 'full-rank' | 'geographic-rank';
  tieOrder?: 'giddy' | 'row-id';
};

/** Every supported output, in the contributor's layouts. */
export type CPURankMarkovResult = {
  ranks: Uint32Array;
  counts: Uint32Array;
  probabilities: Float32Array;
  rowTotals: Uint32Array;
  ignored: Uint32Array;
  mobility: Float32Array;
  steadyStates: Float32Array;
  stateSparsity: Uint32Array;
  converged: Uint32Array;
};

function validateOptions(id: string, rowCount: number, options: CPURankMarkovOptions): void {
  // The dense output and exact transition counters must fit their uint32 layouts.
  validateCapacity(id, 'rowCount', rowCount, 1, CPU_RANK_MARKOV_MAXIMUM_ROW_COUNT);
  validateCapacity(id, 'periodCount', options.periodCount, 2);
  validateMaximumCount(
    id,
    'rowCount * periodCount',
    rowCount * options.periodCount,
    CPU_RANK_MARKOV_MAXIMUM_VALUE_COUNT
  );
  validateRequirement(
    id,
    rowCount * (options.periodCount - 1) <= CPU_RANK_MARKOV_MAXIMUM_VALUE_COUNT,
    'rowCount * (periodCount - 1) must not exceed uint32'
  );
  validateRequirement(
    id,
    options.mode === undefined || ['full-rank', 'geographic-rank'].includes(options.mode),
    'mode must be full-rank or geographic-rank'
  );
  validateRequirement(
    id,
    options.tieOrder === undefined || ['giddy', 'row-id'].includes(options.tieOrder),
    'tieOrder must be giddy or row-id'
  );
}

/**
 * Complete-case ordinal-rank Markov transitions, including all unconditional diagnostics.
 *
 * What differs from the GPU contributor:
 * - Arithmetic uses float64 and rounds once at the outputs; use 2e-6 relative tolerance for floats.
 * - The GPU-only 64-location and 65536-period caps are dropped. Dense output sizes and total
 *   transition opportunities must fit uint32.
 * - Sorting replaces pairwise ranking. Tie order and missing ranks are unchanged.
 * - Diagnostics call the public float64 computeTransitionDiagnosticsOnCPU, including its
 *   rescaled GTH stationary solver. Empty chains keep converged=1 and NaN steady states.
 *
 * @throws If values is not a balanced panel or mask does not have exactly one word per location.
 */
export function computeRankMarkovOnCPU(
  values: Float32Array,
  options: CPURankMarkovOptions,
  mask?: Uint32Array
): CPURankMarkovResult {
  const rowCount = values.length / options.periodCount;
  if (!Number.isInteger(rowCount) || rowCount < 1)
    throw new Error('values must be a nonempty balanced panel');
  if (mask && mask.length !== rowCount)
    throw new Error('mask length must equal the location count');
  validateOptions('rank-markov', rowCount, options);
  const included: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    if (mask !== undefined && mask[row] === 0) continue;
    let complete = true;
    for (let period = 0; period < options.periodCount; period++) {
      if (!Number.isFinite(values[period * rowCount + row])) {
        complete = false;
        break;
      }
    }
    if (complete) included.push(row);
  }
  const result: CPURankMarkovResult = {
    ranks: new Uint32Array(values.length).fill(0xffffffff),
    counts: new Uint32Array(rowCount ** 2),
    probabilities: new Float32Array(rowCount ** 2),
    rowTotals: new Uint32Array(rowCount),
    ignored: Uint32Array.of((rowCount - included.length) * (options.periodCount - 1)),
    mobility: new Float32Array(options.periodCount - 1).fill(NaN),
    steadyStates: new Float32Array(rowCount).fill(NaN),
    stateSparsity: new Uint32Array(2),
    converged: Uint32Array.of(1)
  };
  const occupants = new Uint32Array(values.length).fill(0xffffffff);
  for (let period = 0; period < options.periodCount; period++) {
    const base = period * rowCount;
    const order = [...included].sort(
      (first, second) =>
        values[base + second] - values[base + first] ||
        (options.tieOrder === 'row-id' ? first - second : second - first)
    );
    order.forEach((row, rank) => {
      result.ranks[base + row] = rank;
      occupants[base + rank] = row;
    });
  }
  const classes = options.mode === 'geographic-rank' ? occupants : result.ranks;
  for (let period = 0; period + 1 < options.periodCount; period++) {
    let displacement = 0;
    for (let row = 0; row < rowCount; row++) {
      const source = classes[period * rowCount + row];
      const target = classes[(period + 1) * rowCount + row];
      if (source < rowCount && target < rowCount) result.counts[source * rowCount + target]++;
    }
    for (const row of included)
      displacement += Math.abs(
        result.ranks[period * rowCount + row] - result.ranks[(period + 1) * rowCount + row]
      );
    if (included.length > 1) result.mobility[period] = displacement / included.length;
  }
  for (let source = 0; source < rowCount; source++) {
    let total = 0;
    for (let target = 0; target < rowCount; target++)
      total += result.counts[source * rowCount + target];
    result.rowTotals[source] = total;
    for (let target = 0; target < rowCount; target++)
      result.probabilities[source * rowCount + target] =
        total === 0 ? 0 : result.counts[source * rowCount + target] / total;
  }
  const diagnostics = computeTransitionDiagnosticsOnCPU(result.counts, {classCount: rowCount});
  result.steadyStates = diagnostics.steadyStates;
  result.stateSparsity = diagnostics.stateSparsity;
  result.converged = diagnostics.converged;
  return result;
}

/** CPU twin, retaining uint32 layout limits and dropping GPU-only size caps. */
export class CPURankMarkovOperation extends GPUCPUOperation {
  constructor(props: RankMarkovOperationProps) {
    const operation = new GPURankMarkovOperation(props);
    const {id} = operation;
    validateOptions(id, props.rowCount, props);
    validateRequirement(
      id,
      props.output.status === undefined || props.output.steadyStates !== undefined,
      'output.status.converged needs output.steadyStates'
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
        const values = getCPUOperationFloat32Rows(id, inputs, 'values');
        if (values.length !== props.rowCount * props.periodCount)
          throw new Error(`${id} values length must match rowCount * periodCount`);
        const result = computeRankMarkovOnCPU(
          values,
          props,
          props.mask === undefined ? undefined : getCPUOperationUint32Rows(id, inputs, 'mask')
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPURankMarkovResult]
          ])
        );
      }
    });
  }
}
