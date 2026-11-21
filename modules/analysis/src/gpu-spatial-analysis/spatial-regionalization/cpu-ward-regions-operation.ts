// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  type RegionalizationWeightsStatus,
  validateRegionalizationWeightsStatus
} from './adjacency-loss-suppression';
import {hashUint32} from '../../analysis-core/kernels/hash-uint32';
import {getCPURegionTable} from './cpu-region-inputs';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPU_WARD_REGIONS_PARAMETER_LENGTH,
  GPU_WARD_STATUS_LAYOUT,
  GPU_WARD_ROUND_COUNT_STATUS_WORD
} from './gpu-ward-regions';
import {
  GPUWardRegionsOperation,
  snapshotWardOperationProps,
  type WardRegionsOperationProps
} from './gpu-ward-regions-operation';

export type CPUWardRegionsOptions = Pick<
  WardRegionsOperationProps,
  'columnCount' | 'standardize' | 'targetRegionCount' | 'mergeSchedule' | 'statusLength'
> & {id?: string; maximumIterations?: number};
export type CPUWardRegionsResult = {
  labels: Uint32Array;
  objective: Float32Array;
  status: Uint32Array;
};
type WardPair = {first: number; second: number; increase: number; key: number};

/**
 * Completes spatial Ward agglomeration, using the undirected union of the CSR pattern, the
 * seed-keyed total pair order, and either sequential or mutual-nearest merging. Non-finite
 * feature entries and out-of-range neighbor slots each add one to invalidCount and prevent merges.
 * Status contains the four GPU_WARD_STATUS_LAYOUT words, optionally followed by the mutual-nearest round
 * count. A disconnected final round counts as an attempted round. The fifth word is unused for
 * sequential schedules and is returned as zero; callers should use four words in that mode.
 * Optional input.status loss scalars suppress the partition as on GPU: labels are 0xffffffff,
 * objective is NaN and iteration/convergence fields are zero. Loss is not an invalid feature.
 *
 * What differs from the GPU contributor:
 * - Uses float64 column standardization, region sums, costs and objective, with one output rounding.
 *   Objective comparisons allow 1e-4 relative error; near-tied costs can change the partition.
 * - Honors maximumIterations; omitted in the pure function it defaults to rows minus the target.
 *   iterationCount is the completed merge count; the budget counts rounds for either schedule.
 * - Drops the eight-column private-array limit and GPU storage/command scheduling limits.
 * - Finite inputs whose GPU merge costs overflow remain mergeable; the float32 objective can
 *   overflow when the completed float64 objective is rounded at the output.
 */
export function computeWardRegionsOnCPU(
  input: {
    offsets: Uint32Array;
    neighbors: Uint32Array;
    values: Float32Array;
    parameters: Uint32Array;
    status?: RegionalizationWeightsStatus<Uint32Array>;
  },
  options: CPUWardRegionsOptions
): CPUWardRegionsResult {
  const id = options.id ?? 'computeWardRegionsOnCPU';
  validateRegionalizationWeightsStatus(id, input.status);
  const {offsets, neighbors, values, parameters} = input;
  const rowCount = offsets.length - 1;
  const columnCount = options.columnCount ?? 1;
  const mergeSchedule = options.mergeSchedule ?? 'sequential';
  const statusLength = options.statusLength ?? GPU_WARD_STATUS_LAYOUT.length;
  const maximumIterations = options.maximumIterations ?? rowCount - options.targetRegionCount;
  validateCapacity(id, 'maximumIterations', maximumIterations, 0);
  validateCapacity(id, 'rowCount', rowCount);
  validateCapacity(id, 'columnCount', columnCount);
  validateCapacity(
    id,
    'statusLength',
    statusLength,
    GPU_WARD_STATUS_LAYOUT.length,
    GPU_WARD_ROUND_COUNT_STATUS_WORD + 1
  );
  validateCapacity(id, 'targetRegionCount', options.targetRegionCount, 1, rowCount);
  validateRequirement(
    id,
    ['sequential', 'mutual-nearest'].includes(mergeSchedule),
    "mergeSchedule must be 'sequential' or 'mutual-nearest'",
    {
      code: 'INVALID_OPTION',
      path: 'mergeSchedule',
      received: mergeSchedule,
      expected: ['sequential', 'mutual-nearest']
    }
  );
  if (values.length !== rowCount * columnCount)
    validateRequirement(
      id,
      false,
      `values length must equal rowCount * columnCount; received values.length=${values.length}`,
      {
        code: 'VIEW_LENGTH',
        path: 'values.length',
        received: values.length,
        expected: 'values length must equal rowCount * columnCount'
      }
    );
  if (parameters.length < GPU_WARD_REGIONS_PARAMETER_LENGTH)
    validateRequirement(
      id,
      false,
      `parameters is too short; received parameters.length=${parameters.length}`,
      {
        code: 'VIEW_LENGTH',
        path: 'parameters.length',
        received: parameters.length,
        expected: 'parameters is too short'
      }
    );
  validateRequirement(id, offsets[0] === 0, 'offsets must start at 0', {
    code: 'REQUIREMENT',
    path: 'offsets[0]',
    received: offsets[0],
    expected: 0
  });
  for (let row = 0; row < rowCount; row++) {
    validateRequirement(
      id,
      offsets[row] <= offsets[row + 1] && offsets[row + 1] <= neighbors.length,
      'offsets must be monotone and fit neighbors length',
      {
        code: 'REQUIREMENT',
        path: `offsets[${row + 1}]`,
        received: offsets[row + 1],
        expected: {minimum: offsets[row], maximum: neighbors.length}
      }
    );
  }
  const labels = Uint32Array.from({length: rowCount}, (_, row) => row);
  const status = new Uint32Array(statusLength);
  for (const value of values)
    if (!Number.isFinite(value)) status[GPU_WARD_STATUS_LAYOUT.invalidCount]++;
  for (let slot = 0; slot < offsets[rowCount]; slot++)
    if (neighbors[slot] >= rowCount) status[GPU_WARD_STATUS_LAYOUT.invalidCount]++;
  const objective = new Float32Array(1);
  for (const [name, values] of Object.entries(input.status ?? {})) {
    if (values) validateCapacity(id, `status.${name}.length`, values.length, 1);
  }
  if (
    (input.status?.overflow?.[0] ?? 0) !== 0 ||
    (input.status?.candidateOverflow?.[0] ?? 0) !== 0
  ) {
    labels.fill(0xffffffff);
    objective[0] = NaN;
    return {labels, objective, status};
  }
  if (status[GPU_WARD_STATUS_LAYOUT.invalidCount] > 0) {
    if (
      mergeSchedule === 'mutual-nearest' &&
      statusLength > GPU_WARD_ROUND_COUNT_STATUS_WORD &&
      rowCount > options.targetRegionCount &&
      maximumIterations > 0
    )
      status[GPU_WARD_ROUND_COUNT_STATUS_WORD] = 1;
    return {labels, objective, status};
  }
  const table = getCPURegionTable(values, columnCount, options.standardize);
  const origins = options.standardize === false ? table : undefined;
  const sums = origins ? new Float64Array(rowCount * columnCount) : table;
  const sizes = new Float64Array(rowCount).fill(1);
  const [seed] = parameters; // Packer layout: [seed], GPU_WARD_REGIONS_PARAMETER_LENGTH words.
  const comparePairs = (first: WardPair, second: WardPair) =>
    first.increase - second.increase ||
    first.key - second.key ||
    first.first - second.first ||
    first.second - second.second;
  let remaining = rowCount;
  let totalObjective = 0;
  let roundCount = 0;
  let exhausted = false;
  while (remaining > options.targetRegionCount && roundCount < maximumIterations) {
    roundCount++;
    if (mergeSchedule === 'mutual-nearest' && statusLength > GPU_WARD_ROUND_COUNT_STATUS_WORD)
      status[GPU_WARD_ROUND_COUNT_STATUS_WORD]++;
    const pairs = new Map<string, WardPair>();
    // Original edges induce the exact region adjacency, in either input direction.
    for (let row = 0; row < rowCount; row++) {
      for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
        const first = Math.min(labels[row], labels[neighbors[slot]]);
        const second = Math.max(labels[row], labels[neighbors[slot]]);
        const name = `${first},${second}`;
        if (first === second || pairs.has(name)) continue;
        let increase = 0;
        for (let column = 0; column < columnCount; column++) {
          const offsetDifference =
            sums[first * columnCount + column] / sizes[first] -
            sums[second * columnCount + column] / sizes[second];
          const difference = origins
            ? origins[first * columnCount + column] -
              origins[second * columnCount + column] +
              offsetDifference
            : offsetDifference;
          increase +=
            ((sizes[first] * sizes[second]) / (sizes[first] + sizes[second])) *
            difference *
            difference;
        }
        pairs.set(name, {
          first,
          second,
          increase,
          key: hashUint32(Math.imul(first, 0x9e3779b9) ^ second ^ seed)
        });
      }
    }
    const orderedPairs = [...pairs.values()].sort(comparePairs);
    if (orderedPairs.length === 0) {
      exhausted = true;
      break;
    }
    let selected: WardPair[];
    if (mergeSchedule === 'sequential') selected = [orderedPairs[0]];
    else {
      const nearest = new Map<number, WardPair>();
      // First occurrence in the total order is each region's unique nearest pair.
      for (const pair of orderedPairs) {
        if (!nearest.has(pair.first)) nearest.set(pair.first, pair);
        if (!nearest.has(pair.second)) nearest.set(pair.second, pair);
      }
      selected = orderedPairs
        .filter(pair => nearest.get(pair.first) === pair && nearest.get(pair.second) === pair)
        .slice(0, remaining - options.targetRegionCount);
    }
    for (const pair of selected) {
      totalObjective += pair.increase;
      const secondCount = sizes[pair.second];
      sizes[pair.first] += sizes[pair.second];
      sizes[pair.second] = 0;
      for (let column = 0; column < columnCount; column++) {
        const rebasedSecond = origins
          ? sums[pair.second * columnCount + column] +
            secondCount *
              (origins[pair.second * columnCount + column] -
                origins[pair.first * columnCount + column])
          : sums[pair.second * columnCount + column];
        sums[pair.first * columnCount + column] += rebasedSecond;
      }
      for (let row = 0; row < rowCount; row++)
        if (labels[row] === pair.second) labels[row] = pair.first;
      remaining--;
      status[GPU_WARD_STATUS_LAYOUT.iterationCount]++;
    }
  }
  objective[0] = totalObjective;
  status[GPU_WARD_STATUS_LAYOUT.converged] = Number(remaining === options.targetRegionCount);
  status[GPU_WARD_STATUS_LAYOUT.iterationLimitReached] = Number(
    remaining > options.targetRegionCount && !exhausted && roundCount === maximumIterations
  );
  return {labels, objective, status};
}

/** CPU twin; honors the chosen schedule and round budget without the GPU column limit. */
export class CPUWardRegionsOperation extends GPUCPUOperation {
  constructor(props: WardRegionsOperationProps) {
    props = snapshotWardOperationProps(props);
    const id = props.id ?? 'ward-regions';
    validateCapacity(id, 'rowCount', props.rowCount);
    validateCapacity(id, 'columnCount', props.columnCount ?? 1);
    validateCapacity(
      id,
      'statusLength',
      props.statusLength ?? GPU_WARD_STATUS_LAYOUT.length,
      GPU_WARD_STATUS_LAYOUT.length,
      GPU_WARD_ROUND_COUNT_STATUS_WORD + 1
    );
    validateCapacity(id, 'targetRegionCount', props.targetRegionCount, 1, props.rowCount);
    validateCapacity(id, 'maximumIterations', props.maximumIterations, 0);
    validateRequirement(
      id,
      ['sequential', 'mutual-nearest'].includes(props.mergeSchedule ?? 'sequential'),
      "mergeSchedule must be 'sequential' or 'mutual-nearest'",
      {
        code: 'INVALID_OPTION',
        path: 'mergeSchedule',
        received: props.mergeSchedule,
        expected: ['sequential', 'mutual-nearest']
      }
    );
    // Reuse the graph-port shape and disjointness checks; this allocates no GPU resources.
    const operation = new GPUWardRegionsOperation(props);
    super({
      id,
      inputs: operation.inputs,
      outputs: {
        [props.output.labels]: gpuGraph.vector('uint32', {length: props.rowCount}),
        [props.output.objective]: gpuGraph.vector('float32', {length: 1}),
        [props.output.status]: gpuGraph.vector('uint32', {
          length: props.statusLength ?? GPU_WARD_STATUS_LAYOUT.length
        })
      },
      execute({inputs}) {
        const offsets = getCPUOperationUint32Rows(id, inputs, 'offsets');
        const neighbors = getCPUOperationUint32Rows(id, inputs, 'neighbors');
        const weights = getCPUOperationFloat32Rows(id, inputs, 'weights');
        if (offsets.length !== props.rowCount + 1)
          validateRequirement(
            id,
            false,
            `offsets length must equal rowCount + 1; received offsets.length=${offsets.length}`,
            {
              code: 'VIEW_LENGTH',
              path: 'offsets.length',
              received: offsets.length,
              expected: 'offsets length must equal rowCount + 1'
            }
          );
        if (weights.length !== neighbors.length)
          validateRequirement(
            id,
            false,
            `weights length must equal neighbors length; received weights.length=${weights.length}, neighbors.length=${neighbors.length}`,
            {
              code: 'VIEW_LENGTH',
              path: 'weights.length',
              received: weights.length,
              expected: 'weights length must equal neighbors length'
            }
          );
        const result = computeWardRegionsOnCPU(
          {
            offsets,
            neighbors,
            values: getCPUOperationFloat32Rows(id, inputs, 'values'),
            parameters: getCPUOperationUint32Rows(id, inputs, 'parameters'),
            status: {
              overflow: props.weights.status?.overflow
                ? getCPUOperationUint32Rows(id, inputs, 'weightsOverflow')
                : undefined,
              candidateOverflow: props.weights.status?.candidateOverflow
                ? getCPUOperationUint32Rows(id, inputs, 'weightsCandidateOverflow')
                : undefined
            }
          },
          {...props, id}
        );
        return {
          [props.output.labels]: result.labels,
          [props.output.objective]: result.objective,
          [props.output.status]: result.status
        };
      }
    });
  }
}
