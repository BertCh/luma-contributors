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
import {fillShufflePermutation} from '../../analysis-core/kernels/permutation-shuffle';
import {PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';
import {
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  GPU_SPACE_TIME_SUMMARY,
  GPU_SPACE_TIME_SUMMARY_LENGTH
} from './space-time-parameters';
import {
  GPUMantelTestOperation,
  snapshotMantelOperationProps,
  type MantelTestOperationProps
} from './gpu-mantel-test-operation';
import {resolveGPUSpaceTimeTemporalContext} from './space-time-bands';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

/** CSR slots past the final offset are unused capacity; weights do not affect inference. */
export type CPUMantelTestInput = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
  distances: Float32Array;
  times: Float32Array;
  /** Packed by getGPUSpaceTimeParameterValues. */
  parameters: Uint32Array;
  /** Per-frame loss words from the pair producer. Either nonzero word suppresses inference. */
  status?: {overflow?: Uint32Array; candidateOverflow?: Uint32Array};
};
/** Compile-time inference settings. */
export type CPUMantelTestOptions = Pick<
  MantelTestOperationProps,
  | 'maximumPermutations'
  | 'stopAfterExceedances'
  | 'distanceTransform'
  | 'timeTransform'
  | 'temporalContext'
>;
/** Every output, with unused statistics capacity zeroed. */
export type CPUMantelTestResult = {
  statistics: Float32Array;
  summary: Float32Array;
  /** Attempted statistics prefix, excluding observed slot 0. Zero for undefined observations. */
  permutationCount: Uint32Array;
  /** Defined null correlations used by conditional inference. */
  usablePermutationCount: Uint32Array;
};

/** One statistics row is reserved for the observed value in a uint32-sized vector. */
export const CPU_MANTEL_TEST_MAXIMUM_PERMUTATIONS = 0xfffffffe;

function validateOptions(id: string, rowCount: number, options: CPUMantelTestOptions): void {
  resolveGPUSpaceTimeTemporalContext(id, options.temporalContext);
  validateCapacity(id, 'rowCount', rowCount, 2);
  validateCapacity(
    id,
    'maximumPermutations',
    options.maximumPermutations,
    1,
    CPU_MANTEL_TEST_MAXIMUM_PERMUTATIONS
  );
  if (options.stopAfterExceedances !== undefined)
    validateCapacity(
      id,
      'stopAfterExceedances',
      options.stopAfterExceedances,
      1,
      options.maximumPermutations
    );
  for (const [name, transform] of [
    ['distanceTransform', options.distanceTransform],
    ['timeTransform', options.timeTransform]
  ] as const)
    if (transform) {
      validateFiniteNumber(id, `${name}.constant`, transform.constant);
      validateFiniteNumber(id, `${name}.power`, transform.power);
    }
}

function transformValue(
  value: number,
  transform: CPUMantelTestOptions['distanceTransform']
): number {
  return transform ? (value + transform.constant) ** transform.power : value;
}

/**
 * Mantel correlation of unique listed unordered pairs, with the registered Philox swap-or-not
 * stream and optional Besag-Clifford stopping in permutation order.
 *
 * What differs from the GPU contributor:
 * - Correlations retain float64 residuals through threshold comparisons and centered null moments;
 *   only the published statistics and summary words are rounded to float32.
 *   Correlations usually agree within 1e-4 absolute. Near ties can change counts and stopping.
 * - GPU-only row and permutation caps are dropped; uint32 indexing/output lengths remain.
 * - Scaled centered vectors avoid intermediate overflow for finite transformed data.
 * - Nonfinite times, selected distances or transformed pair values throw an Error naming the
 *   argument. These inputs have no defined GPU result. Invalid neighbor IDs, self entries and
 *   duplicates are ignored as in the GPU pair mask; malformed CSR offsets throw.
 * - Transform props use their supplied binary64 numbers, rather than WGSL float32 literals.
 *
 * @throws For inconsistent arrays, invalid offsets or undefined numeric input as described above.
 */
export function computeMantelTestOnCPU(
  input: CPUMantelTestInput,
  options: CPUMantelTestOptions
): CPUMantelTestResult {
  const {offsets, neighbors, weights, distances, times, parameters} = input;
  const rows = times.length;
  validateOptions('mantel-test', rows, options);
  const inputStatus = input.status;
  const status =
    inputStatus && snapshotConstructionFields(inputStatus, ['overflow', 'candidateOverflow']);
  for (const name of ['overflow', 'candidateOverflow'] as const) {
    const flag = status?.[name];
    if (flag) validateCapacity('mantel-test', `status.${name}.length`, flag.length, 1);
  }
  validateRequirement(
    'mantel-test',
    offsets.length === rows + 1,
    `offsets.length must equal times.length + 1 (${rows + 1}); received ${offsets.length}`
  );
  validateRequirement(
    'mantel-test',
    weights.length === neighbors.length,
    `weights.length must equal neighbors.length (${neighbors.length}); received ${weights.length}`
  );
  validateRequirement(
    'mantel-test',
    distances.length === neighbors.length,
    `distances.length must equal neighbors.length (${neighbors.length}); received ${distances.length}`
  );
  validateCapacity(
    'mantel-test',
    'parameters.length',
    parameters.length,
    GPU_SPACE_TIME_PARAMETER_LENGTH
  );
  if ((status?.overflow?.[0] ?? 0) !== 0 || (status?.candidateOverflow?.[0] ?? 0) !== 0)
    return {
      statistics: new Float32Array(options.maximumPermutations + 1).fill(NaN),
      summary: new Float32Array(GPU_SPACE_TIME_SUMMARY_LENGTH).fill(NaN),
      permutationCount: Uint32Array.of(0),
      usablePermutationCount: Uint32Array.of(0)
    };
  validateRequirement(
    'mantel-test',
    offsets[0] === 0,
    `offsets[0] must be zero; received ${offsets[0]}`
  );
  validateRequirement(
    'mantel-test',
    times.every(Number.isFinite),
    'times must contain finite values; received a nonfinite value'
  );
  for (let row = 0; row < rows; row++)
    validateRequirement(
      'mantel-test',
      offsets[row + 1] >= offsets[row] && offsets[row + 1] <= neighbors.length,
      `offsets[${row + 1}] must be nondecreasing and within neighbors capacity ${neighbors.length}; received ${offsets[row + 1]}`
    );
  // Ascending row visitation makes the lower-row entry win even for asymmetric lists.
  const seen = new Set<bigint>();
  const firstRows: number[] = [];
  const secondRows: number[] = [];
  const spatialValues: number[] = [];
  for (let row = 0; row < rows; row++)
    for (let entry = offsets[row]; entry < offsets[row + 1]; entry++) {
      const other = neighbors[entry];
      if (other >= rows || other === row) continue;
      const key = BigInt(Math.min(row, other)) * BigInt(rows) + BigInt(Math.max(row, other));
      if (seen.has(key)) continue;
      seen.add(key);
      const distance = transformValue(distances[entry], options.distanceTransform);
      validateRequirement(
        'mantel-test',
        Number.isFinite(distances[entry]) && Number.isFinite(distance),
        `distances and distanceTransform must give finite selected pair values; received ${distance} at entry ${entry}`
      );
      firstRows.push(row);
      secondRows.push(other);
      spatialValues.push(distance);
    }
  const count = firstRows.length;
  // Power-of-two scaling preserves integer moment identities, including exact zero covariance.
  // Center on a data anchor before taking moments; never introduce a rounded fractional mean.
  const centerAndScale = (values: number[]) => {
    const maximum = values.reduce((scale, value) => Math.max(scale, Math.abs(value)), 0);
    const exponent = Math.max(-1022, Math.min(1023, Math.floor(Math.log2(maximum || 1))));
    const scale = 2 ** exponent;
    const anchor = (values[0] ?? 0) / scale;
    return values.map(value => value / scale - anchor);
  };
  const spatial = centerAndScale(spatialValues);
  const sumSpatial = spatial.reduce((sum, value) => sum + value, 0);
  const spatialVariance =
    count * spatial.reduce((sum, value) => sum + value ** 2, 0) - sumSpatial ** 2;
  // Destructure the packer's [seedLow, seedHigh, permutationCount, thresholdBits] layout.
  const [seedLow, seedHigh, requestedPermutationCount] = parameters;
  const permutations = Math.max(
    1,
    Math.min(requestedPermutationCount, options.maximumPermutations)
  );
  const permutation = new Uint32Array(rows);
  const preciseStatistics = new Float64Array(options.maximumPermutations + 1);
  let used = permutations;
  let stopped = false;
  let greater = 0;
  let lesser = 0;
  for (let slot = 0; slot <= permutations; slot++) {
    if (slot > 0)
      fillShufflePermutation(
        permutation,
        rows,
        [seedLow, seedHigh],
        slot,
        PHILOX_PURPOSES.mantelTestShuffle.id
      );
    const temporal = firstRows.map((row, index) =>
      transformValue(
        Math.abs(
          times[slot === 0 ? row : permutation[row]] -
            times[slot === 0 ? secondRows[index] : permutation[secondRows[index]]]
        ),
        options.timeTransform
      )
    );
    validateRequirement(
      'mantel-test',
      temporal.every(Number.isFinite),
      'times and timeTransform must give finite selected pair values; received a nonfinite transformed gap'
    );
    const scaled = centerAndScale(temporal);
    let covariance = 0;
    let variance = 0;
    let sumTemporal = 0;
    for (let pair = 0; pair < count; pair++) {
      covariance += spatial[pair] * scaled[pair];
      variance += scaled[pair] ** 2;
      sumTemporal += scaled[pair];
    }
    covariance = count * covariance - sumSpatial * sumTemporal;
    variance = count * variance - sumTemporal ** 2;
    const correlation =
      count === 2
        ? spatialValues[0] === spatialValues[1] || temporal[0] === temporal[1]
          ? NaN
          : (spatialValues[1] > spatialValues[0] ? 1 : -1) * (temporal[1] > temporal[0] ? 1 : -1)
        : count > 1 && spatialVariance > 0 && variance > 0
          ? covariance === 0
            ? 0
            : Math.max(
                -1,
                Math.min(1, covariance / Math.sqrt(spatialVariance) / Math.sqrt(variance))
              )
          : NaN;
    preciseStatistics[slot] = correlation;
    if (slot === 0) {
      if (!Number.isFinite(correlation)) {
        const summary = new Float32Array(GPU_SPACE_TIME_SUMMARY_LENGTH).fill(NaN);
        summary[GPU_SPACE_TIME_SUMMARY.pairCount] = count;
        summary[GPU_SPACE_TIME_SUMMARY.timeClosePairs] = 0;
        summary[GPU_SPACE_TIME_SUMMARY.expected] = 0;
        return {
          statistics: new Float32Array(options.maximumPermutations + 1).fill(NaN),
          summary,
          permutationCount: Uint32Array.of(0),
          usablePermutationCount: Uint32Array.of(0)
        };
      }
      continue;
    }
    if (correlation >= preciseStatistics[0]) greater++;
    if (correlation <= preciseStatistics[0]) lesser++;
    if (options.stopAfterExceedances !== undefined && greater === options.stopAfterExceedances) {
      used = slot;
      stopped = true;
      break;
    }
  }
  // Undefined null correlations keep their slot as NaN but are absent from both tails,
  // moments and the pseudo-count denominator. Stopping still follows draw order.
  const usableStatistics = Array.from(preciseStatistics.subarray(1, used + 1)).filter(
    Number.isFinite
  );
  const attemptedCount = used;
  used = usableStatistics.length;
  const firstNullStatistic = usableStatistics[0] ?? NaN;
  let sumDeviation = 0;
  for (const value of usableStatistics) sumDeviation += value - firstNullStatistic;
  const meanDeviation = sumDeviation / used;
  const mean = firstNullStatistic + meanDeviation;
  let squared = 0;
  for (const value of usableStatistics)
    squared += (value - firstNullStatistic - meanDeviation) ** 2;
  const variance = used > 1 ? squared / (used - 1) : used === 1 ? 0 : NaN;
  const greaterProbability =
    used === 0
      ? NaN
      : stopped
        ? Math.min(1, options.stopAfterExceedances! / used)
        : (greater + 1) / (used + 1);
  const lesserProbability = used === 0 ? NaN : (lesser + 1) / (used + 1);
  const summary = new Float32Array(GPU_SPACE_TIME_SUMMARY_LENGTH);
  summary[GPU_SPACE_TIME_SUMMARY.observed] = preciseStatistics[0];
  summary[GPU_SPACE_TIME_SUMMARY.pairCount] = count;
  summary[GPU_SPACE_TIME_SUMMARY.permutationMean] = mean;
  summary[GPU_SPACE_TIME_SUMMARY.permutationVariance] = variance;
  summary[GPU_SPACE_TIME_SUMMARY.greaterCount] = used === 0 ? NaN : greater;
  summary[GPU_SPACE_TIME_SUMMARY.lesserCount] = used === 0 ? NaN : lesser;
  summary[GPU_SPACE_TIME_SUMMARY.pseudoPGreater] = greaterProbability;
  summary[GPU_SPACE_TIME_SUMMARY.pseudoPLesser] = lesserProbability;
  summary[GPU_SPACE_TIME_SUMMARY.pseudoPTwoSided] = Math.min(
    1,
    2 * Math.min(greaterProbability, lesserProbability)
  );
  summary[GPU_SPACE_TIME_SUMMARY.zScore] =
    variance > 0
      ? (preciseStatistics[0] - firstNullStatistic - meanDeviation) / Math.sqrt(variance)
      : NaN;
  return {
    statistics: Float32Array.from(preciseStatistics),
    summary,
    permutationCount: Uint32Array.of(attemptedCount),
    usablePermutationCount: Uint32Array.of(used)
  };
}

/** CPU twin; no GPU-only population, partial-table or permutation-budget limits. */
export class CPUMantelTestOperation extends GPUCPUOperation {
  constructor(props: MantelTestOperationProps) {
    props = snapshotMantelOperationProps(props);
    const operation = new GPUMantelTestOperation(props);
    const {id} = operation;
    validateOptions(id, props.rowCount, props);
    validateRequirement(
      id,
      props.stopAfterExceedances === undefined || props.output.permutationCount !== undefined,
      'output.permutationCount is required when stopAfterExceedances is supplied'
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
        const times = getCPUOperationFloat32Rows(id, inputs, 'times');
        validateRequirement(
          id,
          times.length === props.rowCount,
          `times.length must equal rowCount ${props.rowCount}; received ${times.length}`
        );
        const result = computeMantelTestOnCPU(
          {
            times,
            offsets: getCPUOperationUint32Rows(id, inputs, 'offsets'),
            neighbors: getCPUOperationUint32Rows(id, inputs, 'neighbors'),
            weights: getCPUOperationFloat32Rows(id, inputs, 'weights'),
            distances: getCPUOperationFloat32Rows(id, inputs, 'distances'),
            parameters: getCPUOperationUint32Rows(id, inputs, 'parameters'),
            status: {
              overflow:
                props.pairs.status?.overflow !== undefined
                  ? getCPUOperationUint32Rows(id, inputs, 'pairsOverflow')
                  : undefined,
              candidateOverflow:
                props.pairs.status?.candidateOverflow !== undefined
                  ? getCPUOperationUint32Rows(id, inputs, 'pairsCandidateOverflow')
                  : undefined
            }
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUMantelTestResult]
          ])
        );
      }
    });
  }
}
