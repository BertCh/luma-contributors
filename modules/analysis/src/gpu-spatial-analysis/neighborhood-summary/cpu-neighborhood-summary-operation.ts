// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPU_NEIGHBORHOOD_SUMMARY_NO_MODE,
  type GPUNeighborhoodSummaryStatistic
} from './gpu-neighborhood-summary';
import {
  GPUNeighborhoodSummaryOperation,
  type NeighborhoodSummaryOperationProps
} from './gpu-neighborhood-summary-operation';

export type CPUNeighborhoodSummaryInput = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
  weightsOverflow?: Uint32Array;
  values?: Float32Array | Uint32Array;
  categories?: Uint32Array;
  mask?: Uint32Array;
};

export type CPUNeighborhoodSummaryOptions = {
  statistics?: readonly GPUNeighborhoodSummaryStatistic[];
  includeFocal?: boolean;
  focalWeight?: number;
  /** Re-standardize active slots independently for numeric and categorical summaries. */
  standardized?: boolean;
};

export type CPUNeighborhoodSummaryResult = {
  table: Float32Array;
  modes: Uint32Array;
  entropy: Float32Array;
  overflow: Uint32Array;
};

const STATISTICS: readonly GPUNeighborhoodSummaryStatistic[] = [
  'count',
  'weightSum',
  'sum',
  'mean',
  'min',
  'max',
  'standardDeviation',
  'median'
];

function validateSummaryOptions(id: string, options: CPUNeighborhoodSummaryOptions): void {
  const statistics = options.statistics ?? [];
  validateRequirement(
    id,
    statistics.every(statistic => STATISTICS.includes(statistic)) &&
      new Set(statistics).size === statistics.length,
    'statistics must contain distinct supported statistics'
  );
  validateFiniteNumber(id, 'focalWeight', options.focalWeight ?? 1, 0);
}

/**
 * Computes numeric summaries by sorting and categorical summaries with a frequency map. Members
 * and row-standardization follow the kernel, including zero-weight focal members. CSR offsets
 * are clamped to capacity and reversed ranges are empty, as explicitly defined by the kernel.
 *
 * What differs from the GPU contributor:
 * - Arithmetic is float64 with one output rounding (relative tolerance 3e-5 for conditioned
 *   positive outputs). Uint32 values first undergo the contributor's float32 conversion.
 * - Medians have no member cap; maximumNeighbors is ignored and output overflow is always zero.
 *   Truncated input weights still invalidate included rows, and masked rows still have count zero.
 * - Min/max use the actual members even beyond the GPU's initial extrema of +/-3e38. Medians
 *   and weighted sums retain finite float64 intermediates where GPU float32 may overflow.
 * - Category 0xffffffff is treated as an ordinary category, matching the current kernel. Ties
 *   use the lowest category. Subnormal values and weights are retained rather than flushed.
 *
 * @throws If aligned arrays disagree in length, or a required numeric column is absent.
 */
export function computeNeighborhoodSummaryOnCPU(
  input: CPUNeighborhoodSummaryInput,
  options: CPUNeighborhoodSummaryOptions = {}
): CPUNeighborhoodSummaryResult {
  validateSummaryOptions('computeNeighborhoodSummaryOnCPU', options);
  const {offsets, neighbors, weights, values, categories, mask, weightsOverflow} = input;
  if (offsets.length < 2) throw new Error('offsets must hold at least two values');
  const rows = offsets.length - 1;
  if (weights.length !== neighbors.length)
    throw new Error('weights length must equal neighbors length');
  for (const [name, array] of [
    ['values', values],
    ['categories', categories],
    ['mask', mask]
  ] as const) {
    if (array && array.length !== rows)
      throw new Error(`${name} length must equal offsets.length - 1`);
  }
  if (weightsOverflow && weightsOverflow.length < 1)
    throw new Error('weightsOverflow must hold one value');
  const statistics = options.statistics ?? [];
  if (statistics.length && !values) throw new Error('values is required for statistics');
  const table = new Float32Array(rows * statistics.length).fill(NaN);
  const modes = new Uint32Array(rows).fill(GPU_NEIGHBORHOOD_SUMMARY_NO_MODE);
  const entropy = new Float32Array(rows).fill(NaN);
  const readValue = (row: number) =>
    values instanceof Uint32Array ? Math.fround(values[row]) : values![row];
  for (let row = 0; row < rows; row++) {
    if (mask && mask[row] === 0) {
      const countColumn = statistics.indexOf('count');
      if (countColumn >= 0) table[row * statistics.length + countColumn] = 0;
      continue;
    }
    if (weightsOverflow && weightsOverflow[0] !== 0) continue;
    const begin = Math.min(offsets[row], neighbors.length);
    const end = Math.max(begin, Math.min(offsets[row + 1], neighbors.length));
    const getMembers = (numeric: boolean) => {
      const members: {row: number; weight: number}[] = [];
      let slotWeightSum = 0;
      for (let slot = begin; slot < end; slot++) {
        const neighbor = neighbors[slot];
        if (
          neighbor >= rows ||
          neighbor === row ||
          !(weights[slot] > 0) ||
          (mask && mask[neighbor] === 0) ||
          (numeric && !Number.isFinite(readValue(neighbor)))
        )
          continue;
        members.push({row: neighbor, weight: weights[slot]});
        slotWeightSum += weights[slot];
      }
      if (options.standardized) {
        const scale = slotWeightSum > 0 ? 1 / slotWeightSum : 0;
        for (const member of members) member.weight *= scale;
      }
      if (options.includeFocal && (!numeric || Number.isFinite(readValue(row))))
        members.unshift({row, weight: options.focalWeight ?? 1});
      return members;
    };
    if (values && statistics.length) {
      const members = getMembers(true);
      let weightSum = 0;
      let weightedSum = 0;
      const sorted: number[] = [];
      for (const member of members) {
        weightSum += member.weight;
        weightedSum += member.weight * readValue(member.row);
        sorted.push(readValue(member.row));
      }
      sorted.sort((first, second) => first - second);
      const mean = weightSum > 0 ? weightedSum / weightSum : NaN;
      let squares = 0;
      for (const member of members) squares += member.weight * (readValue(member.row) - mean) ** 2;
      const count = members.length;
      const middle = Math.floor(count / 2);
      const summary: Record<GPUNeighborhoodSummaryStatistic, number> = {
        count,
        weightSum,
        sum: weightedSum,
        mean,
        min: count ? sorted[0] : NaN,
        max: count ? sorted[count - 1] : NaN,
        standardDeviation: weightSum > 0 ? Math.sqrt(squares / weightSum) : NaN,
        median: count
          ? count % 2
            ? sorted[middle]
            : (sorted[middle - 1] + sorted[middle]) / 2
          : NaN
      };
      for (let column = 0; column < statistics.length; column++)
        table[row * statistics.length + column] = summary[statistics[column]];
    }
    if (categories) {
      const frequencies = new Map<number, number>();
      let weightSum = 0;
      for (const member of getMembers(false)) {
        const category = categories[member.row];
        frequencies.set(category, (frequencies.get(category) ?? 0) + member.weight);
        weightSum += member.weight;
      }
      let bestFrequency = -1;
      let total = 0;
      for (const [category, frequency] of frequencies) {
        if (frequency > bestFrequency || (frequency === bestFrequency && category < modes[row])) {
          modes[row] = category;
          bestFrequency = frequency;
        }
        if (frequency > 0) {
          const share = frequency / weightSum;
          total -= share * Math.log(share);
        }
      }
      entropy[row] = weightSum > 0 ? total : NaN;
    }
  }
  return {table, modes, entropy, overflow: Uint32Array.of(0)};
}

/** CPU twin with uncapped medians. Optional unused spatial-weights ports are omitted. */
export class CPUNeighborhoodSummaryOperation extends GPUCPUOperation {
  constructor(props: NeighborhoodSummaryOperationProps) {
    const id = props.id ?? 'neighborhood-summary';
    const statistics = props.statistics ?? [];
    validateSummaryOptions(id, props);
    validateCapacity(id, 'rowCount', props.rowCount, 1);
    validateCapacity(id, 'maximumNeighbors', props.maximumNeighbors ?? 32, 1);
    validateCapacity(id, 'rowCount * statistics.length', props.rowCount * statistics.length, 0);
    const numeric = statistics.length > 0;
    const categorical = props.output.modes !== undefined || props.output.entropy !== undefined;
    validateRequirement(id, numeric || categorical, 'output must request a statistic');
    validateRequirement(
      id,
      numeric === (props.values !== undefined) && numeric === (props.output.table !== undefined),
      'statistics requires values and output.table'
    );
    validateRequirement(
      id,
      !categorical || props.categories !== undefined,
      'categories is required for modes or entropy'
    );
    validateRequirement(
      id,
      !statistics.includes('median') || props.output.status !== undefined,
      'median requires output.status.overflow'
    );
    validateRequirement(
      id,
      props.valuesFormat === undefined || ['float32', 'uint32'].includes(props.valuesFormat),
      'valuesFormat must be float32 or uint32'
    );
    if (props.weights.information)
      validateRequirement(
        id,
        props.weights.information.rowCount === props.rowCount,
        'weights.information.rowCount must equal rowCount'
      );
    new GPUNeighborhoodSummaryOperation(props);
    const outputs = {
      ...(props.output.table !== undefined
        ? {
            [props.output.table]: gpuGraph.vector('float32', {
              length: props.rowCount * statistics.length
            })
          }
        : {}),
      ...(props.output.modes !== undefined
        ? {[props.output.modes]: gpuGraph.vector('uint32', {length: props.rowCount})}
        : {}),
      ...(props.output.entropy !== undefined
        ? {[props.output.entropy]: gpuGraph.vector('float32', {length: props.rowCount})}
        : {}),
      ...(props.output.status !== undefined
        ? {[props.output.status.overflow]: gpuGraph.vector('uint32', {length: 1})}
        : {})
    };
    super({
      id,
      inputs: {
        offsets: props.weights.offsets,
        neighbors: props.weights.neighbors,
        weights: props.weights.weights,
        ...(props.weights.status?.overflow !== undefined
          ? {weightsOverflow: props.weights.status.overflow}
          : {}),
        ...(props.values !== undefined ? {values: props.values} : {}),
        ...(props.categories !== undefined ? {categories: props.categories} : {}),
        ...(props.mask !== undefined ? {mask: props.mask} : {})
      },
      outputs,
      execute({inputs}) {
        const offsets = getCPUOperationUint32Rows(id, inputs, 'offsets');
        if (offsets.length !== props.rowCount + 1)
          throw new Error(`${id} offsets length must equal rowCount + 1`);
        const result = computeNeighborhoodSummaryOnCPU(
          {
            offsets,
            neighbors: getCPUOperationUint32Rows(id, inputs, 'neighbors'),
            weights: getCPUOperationFloat32Rows(id, inputs, 'weights'),
            weightsOverflow:
              props.weights.status?.overflow !== undefined
                ? getCPUOperationUint32Rows(id, inputs, 'weightsOverflow')
                : undefined,
            values:
              props.values !== undefined
                ? props.valuesFormat === 'uint32'
                  ? getCPUOperationUint32Rows(id, inputs, 'values')
                  : getCPUOperationFloat32Rows(id, inputs, 'values')
                : undefined,
            categories:
              props.categories !== undefined
                ? getCPUOperationUint32Rows(id, inputs, 'categories')
                : undefined,
            mask:
              props.mask !== undefined ? getCPUOperationUint32Rows(id, inputs, 'mask') : undefined
          },
          {...props, standardized: props.weights.information?.transform === 'row'}
        );
        return {
          ...(props.output.table !== undefined ? {[props.output.table]: result.table} : {}),
          ...(props.output.modes !== undefined ? {[props.output.modes]: result.modes} : {}),
          ...(props.output.entropy !== undefined ? {[props.output.entropy]: result.entropy} : {}),
          ...(props.output.status !== undefined
            ? {[props.output.status.overflow]: result.overflow}
            : {})
        };
      }
    });
  }
}
