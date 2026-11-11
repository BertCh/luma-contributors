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
  GPUSpatialWeightsTracesOperation,
  type SpatialWeightsTracesOperationProps
} from './gpu-spatial-weights-traces-operation';
import {
  GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT,
  GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT,
  GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH,
  getSpatialWeightsTracesProbeValue
} from './spatial-weights-traces-parameters';
import {validateCPURegressionWeights, type CPURegressionWeights} from './cpu-regression-weights';

/** CSR input and the optional seed view packed by getGPUSpatialWeightsTracesParameterValues. */
export type CPUSpatialWeightsTracesInput = {
  weights:
    | CPURegressionWeights
    | {
        rowCount: number;
        chunks: readonly {rowStart: number; rowCount: number; weights: CPURegressionWeights}[];
      };
  parameters?: Uint32Array;
};
/** Requested method and series length. */
export type CPUSpatialWeightsTracesOptions = Pick<
  SpatialWeightsTracesOperationProps,
  'maximumPower' | 'mode' | 'probeCount' | 'includeGramTraces'
>;
/** Scalar vector layouts of GPUSpatialWeightsTraces. */
export type CPUSpatialWeightsTracesResult = {
  powerTraces: Float32Array;
  powerTraceStandardErrors: Float32Array;
  powerSums: Float32Array;
  gramTraces?: Float32Array;
  infinityNorm: Float32Array;
  approximation: Uint32Array;
};

function validateOptions(id: string, options: CPUSpatialWeightsTracesOptions): void {
  validateCapacity(id, 'maximumPower', options.maximumPower);
  validateCapacity(
    id,
    'probeCount',
    options.probeCount ?? GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT
  );
  validateRequirement(
    id,
    options.mode === undefined || options.mode === 'exact' || options.mode === 'stochastic',
    "mode must be 'exact' or 'stochastic'"
  );
}

/**
 * Sparse power chains in float64. Exact mode sums unit-probe diagonals; stochastic mode replays
 * the registered Philox Rademacher probes in the GPU order. Sums always use the ones chain.
 * Optional Gram moments are unscaled, as on the GPU, including per-slot square sums.
 *
 * What differs from the GPU contributor:
 * - Float64 chains/reductions replace float32 arithmetic, with one rounding per output. Compare
 *   nonzero outputs with relative tolerance 2e-4; integer approximation and exact-mode errors agree
 *   exactly. Stochastic standard errors use float64 online centered moments.
 * - The shader power cap and explicit-exact row cap are dropped. The same mode default switches
 *   to stochastic above 4096 rows. tileRowCount and probeBatchSize have no computational effect.
 * - Whole and chunked CSR retain separate arrays; no implicit packing occurs. As on the GPU,
 *   Gram traces require whole CSR. Unused metadata ports are omitted.
 * - A finite positive infinity norm is retained even at/above 3e38, where the GPU falls back to 1.
 *   Empty weights scale by 1. A non-finite active weight makes the norm NaN and all floating
 *   results NaN; the GPU does not define portable NaN-reduction behavior. Subnormals are retained.
 * - Empty slot arrays are accepted by the pure function; GPU weights require positive capacity.
 *
 * @throws If weights.offsets, weights.weights or parameters are inconsistent, naming the argument.
 */
export function computeSpatialWeightsTracesOnCPU(
  input: CPUSpatialWeightsTracesInput,
  options: CPUSpatialWeightsTracesOptions
): CPUSpatialWeightsTracesResult {
  const id = 'computeSpatialWeightsTracesOnCPU';
  validateOptions(id, options);
  const chunked = 'chunks' in input.weights;
  const rowCount =
    'chunks' in input.weights
      ? input.weights.rowCount
      : validateCPURegressionWeights(id, input.weights);
  validateCapacity(id, 'weights.rowCount', rowCount);
  const chunks =
    'chunks' in input.weights
      ? input.weights.chunks
      : [{rowStart: 0, rowCount, weights: input.weights}];
  validateRequirement(
    id,
    !chunked || !options.includeGramTraces,
    'weights chunks do not support Gram traces'
  );
  let nextRow = 0;
  const rowChunks: (typeof chunks)[number][] = [];
  for (const [index, chunk] of chunks.entries()) {
    validateCapacity(id, `weights.chunks[${index}].rowCount`, chunk.rowCount);
    validateRequirement(
      id,
      chunk.rowStart === nextRow,
      `weights.chunks[${index}].rowStart must equal ${nextRow}`
    );
    validateRequirement(
      id,
      validateCPURegressionWeights(`${id} weights.chunks[${index}]`, chunk.weights) ===
        chunk.rowCount,
      `weights.chunks[${index}].offsets length must equal rowCount + 1`
    );
    for (let row = 0; row < chunk.rowCount; row++) rowChunks.push(chunk);
    nextRow += chunk.rowCount;
  }
  validateRequirement(id, nextRow === rowCount, 'weights.chunks must cover weights.rowCount rows');
  const visitRow = (row: number, visit: (neighbor: number, weight: number) => void) => {
    const chunk = rowChunks[row];
    const localRow = row - chunk.rowStart;
    const {offsets, neighbors, weights} = chunk.weights;
    for (let slot = offsets[localRow]; slot < offsets[localRow + 1]; slot++) {
      if (neighbors[slot] < rowCount) visit(neighbors[slot], weights[slot]);
    }
  };
  if (input.parameters && input.parameters.length < GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH)
    throw new Error(`${id} parameters must hold the packed seed`);
  const [seed = 0] = input.parameters ?? [];
  const mode =
    options.mode ??
    (rowCount <= GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT ? 'exact' : 'stochastic');
  const stochastic = mode === 'stochastic';
  const probeCount = stochastic
    ? (options.probeCount ?? GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT)
    : rowCount;
  let infinityNorm = 0;
  let hasNonFinite = false;
  for (let row = 0; row < rowCount; row++) {
    let rowSum = 0;
    visitRow(row, (_neighbor, weight) => {
      rowSum += Math.abs(weight);
      hasNonFinite ||= !Number.isFinite(weight);
    });
    infinityNorm = Math.max(infinityNorm, rowSum);
  }
  infinityNorm = hasNonFinite ? NaN : infinityNorm > 0 ? infinityNorm : 1;
  const powerTraces = new Float64Array(options.maximumPower);
  const centeredSquares = new Float64Array(options.maximumPower);
  const powerSums = new Float64Array(options.maximumPower);
  let gramPair = 0;
  let gramSquare = 0;
  const multiply = (vector: Float64Array) => {
    const result = new Float64Array(rowCount);
    for (let row = 0; row < rowCount; row++) {
      visitRow(row, (neighbor, weight) => {
        result[row] += (weight / infinityNorm) * vector[neighbor];
      });
    }
    return result;
  };
  const dot = (first: Float64Array, second: Float64Array) =>
    first.reduce((sum, value, row) => sum + value * second[row], 0);
  let ones = new Float64Array(rowCount).fill(1);
  for (let power = 0; power < options.maximumPower; power++) {
    ones = multiply(ones);
    powerSums[power] = ones.reduce((sum, value) => sum + value, 0);
  }
  for (let probe = 0; probe < probeCount; probe++) {
    const initial = new Float64Array(rowCount);
    if (stochastic) {
      for (let row = 0; row < rowCount; row++)
        initial[row] = getSpatialWeightsTracesProbeValue(seed, probe, row);
    } else initial[probe] = 1;
    let vector = initial;
    for (
      let power = 0;
      power < Math.max(options.maximumPower, options.includeGramTraces ? 2 : 0);
      power++
    ) {
      const next = multiply(vector);
      if (power < options.maximumPower) {
        const value = stochastic ? dot(initial, next) : next[probe];
        if (stochastic) {
          const difference = value - powerTraces[power];
          powerTraces[power] += difference / (probe + 1);
          centeredSquares[power] += difference * (value - powerTraces[power]);
        } else powerTraces[power] += value;
      }
      if (options.includeGramTraces && power === 1) {
        gramPair += dot(vector, next);
        const transposed = new Float64Array(rowCount);
        for (let row = 0; row < rowCount; row++) {
          visitRow(row, (neighbor, weight) => {
            transposed[neighbor] += weight * vector[row];
          });
        }
        gramSquare += dot(transposed, transposed);
      }
      vector = next;
    }
  }
  let gramTraces: Float32Array | undefined;
  if (options.includeGramTraces) {
    let squares = 0;
    const columnSquares = new Float64Array(rowCount);
    for (let row = 0; row < rowCount; row++) {
      visitRow(row, (neighbor, weight) => {
        squares += weight ** 2;
        columnSquares[neighbor] += weight ** 2;
      });
    }
    const divisor = stochastic ? probeCount : 1;
    gramTraces = Float32Array.of(
      squares,
      (gramPair / divisor) * infinityNorm ** 3,
      (gramSquare / divisor) * infinityNorm ** 2,
      dot(columnSquares, columnSquares)
    );
  }
  const powerTraceStandardErrors = Float32Array.from(centeredSquares, value =>
    stochastic
      ? probeCount < 2
        ? NaN
        : Math.sqrt(Math.max(value, 0) / (probeCount - 1) / probeCount)
      : 0
  );
  const result = {
    powerTraces: Float32Array.from(powerTraces),
    powerTraceStandardErrors,
    powerSums: Float32Array.from(powerSums),
    gramTraces,
    infinityNorm: Float32Array.of(infinityNorm),
    approximation: Uint32Array.of(stochastic ? 1 : 0)
  };
  if (hasNonFinite) {
    result.powerTraces.fill(NaN);
    result.powerTraceStandardErrors.fill(NaN);
    result.powerSums.fill(NaN);
    result.gramTraces?.fill(NaN);
  }
  return result;
}

/** CPU whole/chunked CSR twin; shader iteration and exact-size caps are dropped. */
export class CPUSpatialWeightsTracesOperation extends GPUCPUOperation {
  constructor(props: SpatialWeightsTracesOperationProps) {
    const id = props.id ?? 'spatial-weights-traces';
    const includeGramTraces = props.includeGramTraces ?? props.output.gramTraces !== undefined;
    validateOptions(id, props);
    validateCapacity(id, 'rowCount', props.rowCount);
    if ('chunks' in props.weights) {
      validateRequirement(id, !includeGramTraces, 'weights chunks do not support Gram traces');
      let nextRow = 0;
      for (const [index, chunk] of props.weights.chunks.entries()) {
        validateCapacity(id, `weights.chunks[${index}].rowCount`, chunk.rowCount);
        validateRequirement(
          id,
          chunk.rowStart === nextRow,
          `weights.chunks[${index}].rowStart must equal ${nextRow}`
        );
        nextRow += chunk.rowCount;
      }
      validateRequirement(
        id,
        nextRow === props.rowCount && nextRow === props.weights.rowCount,
        'weights.chunks must cover rowCount rows'
      );
    }
    if (props.tileRowCount !== undefined) validateCapacity(id, 'tileRowCount', props.tileRowCount);
    if (props.probeBatchSize !== undefined)
      validateCapacity(id, 'probeBatchSize', props.probeBatchSize);
    validateRequirement(
      id,
      includeGramTraces === (props.output.gramTraces !== undefined),
      'output.gramTraces is required exactly when includeGramTraces is true'
    );
    const operation = new GPUSpatialWeightsTracesOperation(props);
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
        const weights =
          'chunks' in props.weights
            ? {
                rowCount: props.weights.rowCount,
                chunks: props.weights.chunks.map((chunk, index) => ({
                  rowStart: chunk.rowStart,
                  rowCount: chunk.rowCount,
                  weights: {
                    offsets: getCPUOperationUint32Rows(id, inputs, `chunk${index}Offsets`),
                    neighbors: getCPUOperationUint32Rows(id, inputs, `chunk${index}Neighbors`),
                    weights: getCPUOperationFloat32Rows(id, inputs, `chunk${index}Weights`)
                  }
                }))
              }
            : {
                offsets: getCPUOperationUint32Rows(id, inputs, 'offsets'),
                neighbors: getCPUOperationUint32Rows(id, inputs, 'neighbors'),
                weights: getCPUOperationFloat32Rows(id, inputs, 'weights')
              };
        if (
          ('chunks' in weights ? weights.rowCount : weights.offsets.length - 1) !== props.rowCount
        )
          throw new Error(`${id} weights row count must equal rowCount`);
        const result = computeSpatialWeightsTracesOnCPU(
          {
            weights,
            parameters: props.parameters
              ? getCPUOperationUint32Rows(id, inputs, 'parameters')
              : undefined
          },
          {...props, includeGramTraces}
        );
        return {
          [props.output.powerTraces]: result.powerTraces,
          [props.output.powerSums]: result.powerSums,
          ...(props.output.powerTraceStandardErrors
            ? {[props.output.powerTraceStandardErrors]: result.powerTraceStandardErrors}
            : {}),
          ...(props.output.gramTraces ? {[props.output.gramTraces]: result.gramTraces!} : {}),
          ...(props.output.infinityNorm ? {[props.output.infinityNorm]: result.infinityNorm} : {}),
          ...(props.output.status?.approximation
            ? {[props.output.status.approximation]: result.approximation}
            : {})
        };
      }
    });
  }
}
