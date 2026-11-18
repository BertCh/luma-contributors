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
  'maximumPermutations' | 'stopAfterExceedances' | 'distanceTransform' | 'timeTransform'
>;
/** Every output, with unused statistics capacity zeroed. */
export type CPUMantelTestResult = {
  statistics: Float32Array;
  summary: Float32Array;
  permutationCount: Uint32Array;
};

/** One statistics row is reserved for the observed value in a uint32-sized vector. */
export const CPU_MANTEL_TEST_MAXIMUM_PERMUTATIONS = 0xfffffffe;

function validateOptions(id: string, rowCount: number, options: CPUMantelTestOptions): void {
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
 * - Correlations, threshold comparisons and summary moments use float64; only outputs round.
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
  if (offsets.length !== rows + 1 || offsets[0] !== 0)
    throw new Error('offsets must have times.length + 1 words and start at zero');
  if (weights.length !== neighbors.length)
    throw new Error('weights length must equal neighbors.length');
  if (distances.length !== neighbors.length)
    throw new Error('distances length must equal neighbors.length');
  if (parameters.length < GPU_SPACE_TIME_PARAMETER_LENGTH)
    throw new Error('parameters must hold getGPUSpaceTimeParameterValues layout');
  if (!times.every(Number.isFinite)) throw new Error('times must be finite');
  for (let row = 0; row < rows; row++)
    if (offsets[row + 1] < offsets[row] || offsets[row + 1] > neighbors.length)
      throw new Error('offsets must be nondecreasing and within neighbors capacity');
  if ((status?.overflow?.[0] ?? 0) !== 0 || (status?.candidateOverflow?.[0] ?? 0) !== 0)
    return {
      statistics: new Float32Array(options.maximumPermutations + 1).fill(NaN),
      summary: new Float32Array(GPU_SPACE_TIME_SUMMARY_LENGTH).fill(NaN),
      permutationCount: Uint32Array.of(0)
    };
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
      if (!Number.isFinite(distances[entry]) || !Number.isFinite(distance))
        throw new Error('distances and distanceTransform must give finite selected pair values');
      firstRows.push(row);
      secondRows.push(other);
      spatialValues.push(distance);
    }
  const count = firstRows.length;
  const spatialScale =
    spatialValues.reduce((scale, value) => Math.max(scale, Math.abs(value)), 0) || 1;
  const spatial = spatialValues.map(value => value / spatialScale);
  const spatialMean = spatial.reduce((sum, value) => sum + value, 0) / Math.max(1, count);
  const spatialCentered = spatial.map(value => value - spatialMean);
  const spatialNorm = Math.sqrt(spatialCentered.reduce((sum, value) => sum + value ** 2, 0));
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
    if (!temporal.every(Number.isFinite))
      throw new Error('times and timeTransform must give finite selected pair values');
    const temporalScale =
      temporal.reduce((scale, value) => Math.max(scale, Math.abs(value)), 0) || 1;
    const scaled = temporal.map(value => value / temporalScale);
    const mean = scaled.reduce((sum, value) => sum + value, 0) / Math.max(1, count);
    let covariance = 0;
    let variance = 0;
    for (let pair = 0; pair < count; pair++) {
      const centered = scaled[pair] - mean;
      covariance += spatialCentered[pair] * centered;
      variance += centered ** 2;
    }
    const correlation =
      count > 1 && spatialNorm > 0 && variance > 0
        ? Math.max(-1, Math.min(1, covariance / spatialNorm / Math.sqrt(variance)))
        : 0;
    preciseStatistics[slot] = correlation;
    if (slot === 0) continue;
    if (correlation >= preciseStatistics[0]) greater++;
    if (correlation <= preciseStatistics[0]) lesser++;
    if (options.stopAfterExceedances !== undefined && greater === options.stopAfterExceedances) {
      used = slot;
      stopped = true;
      break;
    }
  }
  let mean = 0;
  for (let slot = 1; slot <= used; slot++) mean += preciseStatistics[slot] / used;
  let squared = 0;
  for (let slot = 1; slot <= used; slot++) squared += (preciseStatistics[slot] - mean) ** 2;
  const variance = used > 1 ? squared / (used - 1) : 0;
  const greaterProbability = stopped
    ? Math.min(1, options.stopAfterExceedances! / used)
    : (greater + 1) / (used + 1);
  const lesserProbability = (lesser + 1) / (used + 1);
  const summary = new Float32Array(GPU_SPACE_TIME_SUMMARY_LENGTH);
  summary[GPU_SPACE_TIME_SUMMARY.observed] = preciseStatistics[0];
  summary[GPU_SPACE_TIME_SUMMARY.pairCount] = count;
  summary[GPU_SPACE_TIME_SUMMARY.permutationMean] = mean;
  summary[GPU_SPACE_TIME_SUMMARY.permutationVariance] = variance;
  summary[GPU_SPACE_TIME_SUMMARY.greaterCount] = greater;
  summary[GPU_SPACE_TIME_SUMMARY.lesserCount] = lesser;
  summary[GPU_SPACE_TIME_SUMMARY.pseudoPGreater] = greaterProbability;
  summary[GPU_SPACE_TIME_SUMMARY.pseudoPLesser] = lesserProbability;
  summary[GPU_SPACE_TIME_SUMMARY.pseudoPTwoSided] = Math.min(
    1,
    2 * Math.min(greaterProbability, lesserProbability)
  );
  summary[GPU_SPACE_TIME_SUMMARY.zScore] =
    variance > 0 ? (preciseStatistics[0] - mean) / Math.sqrt(variance) : NaN;
  return {
    statistics: Float32Array.from(preciseStatistics),
    summary,
    permutationCount: Uint32Array.of(used)
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
      (props.stopAfterExceedances !== undefined) === (props.output.permutationCount !== undefined),
      'output.permutationCount is required exactly when stopAfterExceedances is supplied'
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
        if (times.length !== props.rowCount)
          throw new Error(`${id} times length must equal rowCount`);
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
