// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {hashUint32} from '../../analysis-core/kernels/hash-uint32';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {validateSpatialWeightsOperationInformation} from '../spatial-weights/gpu-spatial-weights-algebra-operation';
import type {GPUSpatialWeightsInformation} from '../spatial-weights/spatial-weights';
import {GPU_AZP_REGIONS_PARAMETER_LENGTH, GPU_AZP_STATUS} from './gpu-azp-regions';
import {GPUAZPRegionsOperation, type AZPRegionsOperationProps} from './gpu-azp-regions-operation';
import {
  countCPURegionInvalid,
  getCPURegionAdjacency,
  getCPURegionObjective,
  getCPURegionTable,
  validateCPURegionInput,
  type CPURegionInput
} from './cpu-region-inputs';

/** Packed CSR and initial partition; extensive arrays are required together. */
export type CPUAZPRegionsInput = CPURegionInput & {
  initialLabels: Uint32Array;
  extensiveValues?: Float32Array;
  extensiveThreshold?: Float32Array;
};
/** Explicit CPU round budget; defaults to 256 when the pure function is called without one. */
export type CPUAZPRegionsOptions = Pick<
  AZPRegionsOperationProps,
  'columnCount' | 'standardize' | 'regionCapacity' | 'minimumRegionSize'
> & {
  maximumIterations?: number;
  information?: GPUSpatialWeightsInformation;
};
/** Same graph outputs, with successful batched rounds in status.iterationCount. */
export type CPUAZPRegionsResult = {
  labels: Uint32Array;
  objective: Float32Array;
  status: Uint32Array;
};
type Move = {row: number; source: number; destination: number; gain: number; key: number};

function validateOptions(id: string, options: CPUAZPRegionsOptions, rowCount: number): void {
  validateCapacity(id, 'columnCount', options.columnCount ?? 1);
  validateCapacity(id, 'regionCapacity', options.regionCapacity, 2, rowCount);
  validateCapacity(id, 'minimumRegionSize', options.minimumRegionSize ?? 1);
  validateCapacity(id, 'maximumIterations', options.maximumIterations ?? 256);
}

// Full traversal is used only to validate initial region connectivity.
function visitRegion(
  adjacency: number[][],
  labels: Uint32Array,
  root: number,
  excluded: number
): Set<number> {
  const visited = new Set([root]);
  const queue = [root];
  for (let cursor = 0; cursor < queue.length; cursor++)
    for (const neighbor of adjacency[queue[cursor]])
      if (neighbor !== excluded && labels[neighbor] === labels[root] && !visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push(neighbor);
      }
  return visited;
}

// Preserve the GPU's two-hop admissibility rule, without its private-array or degree caps.
function isRegionRemovalSafe(adjacency: number[][], labels: Uint32Array, row: number): boolean {
  const source = labels[row];
  const ring = new Set(
    adjacency[row].filter(neighbor => neighbor !== row && labels[neighbor] === source)
  );
  if (ring.size < 2) return ring.size === 1;
  const localRows = new Set(ring);
  for (const member of ring)
    for (const neighbor of adjacency[member])
      if (neighbor !== row && labels[neighbor] === source) localRows.add(neighbor);
  const root = ring.values().next().value!;
  const visited = new Set([root]);
  const queue = [root];
  for (let cursor = 0; cursor < queue.length; cursor++)
    for (const neighbor of adjacency[queue[cursor]])
      if (localRows.has(neighbor) && !visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push(neighbor);
      }
  return [...ring].every(member => visited.has(member));
}

/**
 * Internal float64 table entry point shared with Max-P; avoids rounding a standardized table
 * before refinement. Extra initial connected components contribute one invalidCount each.
 * @internal
 */
export function computeAZPRegionsTableOnCPU(
  adjacency: number[][],
  table: Float64Array,
  initialLabels: Uint32Array,
  regionCapacity: number,
  columnCount: number,
  seed: number,
  minimumRegionSize: number,
  extensiveValues?: Float32Array,
  extensiveThreshold = 0,
  initialInvalidCount = 0,
  maximumIterations = Infinity
): {labels: Uint32Array; iterationCount: number; invalidCount: number; converged: boolean} {
  const labels = initialLabels.slice();
  const rowCount = labels.length;
  let invalidCount = initialInvalidCount;
  const visited = new Set<number>();
  const seenRegions = new Set<number>();
  for (let row = 0; row < rowCount; row++) {
    if (labels[row] >= regionCapacity) {
      invalidCount++;
      continue;
    }
    if (visited.has(row)) continue;
    if (seenRegions.has(labels[row])) invalidCount++;
    seenRegions.add(labels[row]);
    for (const member of visitRegion(adjacency, labels, row, -1)) visited.add(member);
  }
  let iterationCount = 0;
  let converged = false;
  if (invalidCount > 0) return {labels, iterationCount, invalidCount, converged};
  while (iterationCount < maximumIterations) {
    const sizes = new Float64Array(regionCapacity);
    const extensiveSums = new Float64Array(regionCapacity);
    const sums = new Float64Array(regionCapacity * columnCount);
    const origins = new Float64Array(regionCapacity * columnCount);
    for (let row = 0; row < rowCount; row++) {
      const region = labels[row];
      for (let column = 0; column < columnCount; column++) {
        const index = region * columnCount + column;
        if (sizes[region] === 0) origins[index] = table[row * columnCount + column];
        sums[index] += table[row * columnCount + column] - origins[index];
      }
      sizes[region]++;
      if (extensiveValues) extensiveSums[region] += extensiveValues[row];
    }
    const moves: Move[] = [];
    for (let row = 0; row < rowCount; row++) {
      const source = labels[row];
      const sourceSize = sizes[source];
      if (
        sourceSize <= minimumRegionSize ||
        (extensiveValues && extensiveSums[source] - extensiveValues[row] < extensiveThreshold)
      )
        continue;
      let bestDelta = Infinity;
      let bestScale = 0;
      let bestDestination = regionCapacity;
      for (const neighbor of adjacency[row]) {
        const destination = labels[neighbor];
        if (destination === source) continue;
        const destinationSize = sizes[destination];
        let delta = 0;
        let scale = 0;
        for (let column = 0; column < columnCount; column++) {
          const value = table[row * columnCount + column];
          const sourceIndex = source * columnCount + column;
          const destinationIndex = destination * columnCount + column;
          const sourceDifference = value - origins[sourceIndex] - sums[sourceIndex] / sourceSize;
          const destinationDifference =
            value - origins[destinationIndex] - sums[destinationIndex] / destinationSize;
          // Stable removal/insertion SSD identities, rather than subtracting squared moments.
          delta +=
            (destinationSize / (destinationSize + 1)) * destinationDifference ** 2 -
            (sourceSize / (sourceSize - 1)) * sourceDifference ** 2;
          const sourceSum = origins[sourceIndex] * sourceSize + sums[sourceIndex];
          const destinationSum =
            origins[destinationIndex] * destinationSize + sums[destinationIndex];
          scale +=
            sourceSum ** 2 / sourceSize +
            destinationSum ** 2 / destinationSize +
            (sourceSum - value) ** 2 / (sourceSize - 1) +
            (destinationSum + value) ** 2 / (destinationSize + 1);
        }
        if (delta < bestDelta || (delta === bestDelta && destination < bestDestination)) {
          bestDelta = delta;
          bestScale = scale;
          bestDestination = destination;
        }
      }
      // Retain the contributor's explicit relative gain rule, in float64.
      if (bestDestination === regionCapacity || !(bestDelta < -4e-6 * bestScale)) continue;
      if (!isRegionRemovalSafe(adjacency, labels, row)) continue;
      moves.push({
        row,
        source,
        destination: bestDestination,
        gain: -bestDelta,
        key: hashUint32(row ^ seed)
      });
    }
    const claimGains = new Float64Array(regionCapacity);
    for (const move of moves) {
      claimGains[move.source] = Math.max(claimGains[move.source], move.gain);
      claimGains[move.destination] = Math.max(claimGains[move.destination], move.gain);
    }
    const contenders = moves.filter(
      move => claimGains[move.source] === move.gain && claimGains[move.destination] === move.gain
    );
    const claimKeys = new Uint32Array(regionCapacity);
    for (const move of contenders) {
      claimKeys[move.source] = Math.max(claimKeys[move.source], move.key);
      claimKeys[move.destination] = Math.max(claimKeys[move.destination], move.key);
    }
    const selected = contenders.filter(
      move => claimKeys[move.source] === move.key && claimKeys[move.destination] === move.key
    );
    if (selected.length === 0) {
      converged = true;
      break;
    }
    for (const move of selected) labels[move.row] = move.destination;
    iterationCount++;
  }
  return {labels, iterationCount, invalidCount, converged};
}

/**
 * Greedy AZP with independent batched moves, two-hop source-region removal tests, seeded row
 * ties and lowest destination-label ties. Retains the GPU's explicit relative gain threshold
 * (gain > 4e-6 times the four squared-moment terms), minimum size and extensive constraints.
 *
 * What differs from the GPU contributor:
 * - Float64 standardization, centered sums and stable removal/insertion gains replace float32
 *   moments; objective rounds only at output. Relative objective tolerance is 1e-4; near ties
 *   and gain-threshold boundaries can change moves, labels and iterationCount.
 * - Preserves two-hop admissibility, dropping only the 128-row and 64-degree neighborhood caps.
 * - Extensive sums and departing-value subtraction also use float64. An extensive-threshold
 *   decision can differ when a small remainder is lost from the GPU's float32 region sum.
 *   For [1, 16777216] at threshold 1, the twin permits the second row to leave; the GPU may not.
 * - Honors maximumIterations (default 256 for the pure function). A final successful batch at
 *   the limit is not proof of convergence; a no-move round must execute to set converged.
 * - Drops GPU column, storage and command-node caps. Empty slot arrays are accepted. Invalid
 *   labels, nonfinite features, negative/nonfinite extensive values or threshold and bad neighbor
 *   IDs keep labels, publish zero objective and set invalidCount, including extra initial components.
 * - Finite float64 gains remain eligible above float32 range; objective can round to Infinity.
 *   Full initial connectivity has no GPU hook-round cutoff, so it never reports an unfinished
 *   component search as invalid input.
 *
 * @throws Error naming inconsistent arrays, including malformed CSR offsets.
 */
export function computeAZPRegionsOnCPU(
  input: CPUAZPRegionsInput,
  options: CPUAZPRegionsOptions
): CPUAZPRegionsResult {
  const columnCount = options.columnCount ?? 1;
  const rowCount = validateCPURegionInput(input, columnCount, GPU_AZP_REGIONS_PARAMETER_LENGTH);
  validateOptions('computeAZPRegionsOnCPU', options, rowCount);
  validateSpatialWeightsOperationInformation(
    'computeAZPRegionsOnCPU',
    'weights',
    options.information,
    rowCount
  );
  if (input.initialLabels.length !== rowCount)
    throw new Error('initialLabels length must equal rowCount');
  if ((input.extensiveValues === undefined) !== (input.extensiveThreshold === undefined))
    throw new Error('extensiveValues and extensiveThreshold are required together');
  if (input.extensiveValues && input.extensiveValues.length !== rowCount)
    throw new Error('extensiveValues length must equal rowCount');
  if (input.extensiveThreshold && input.extensiveThreshold.length < 1)
    throw new Error('extensiveThreshold must hold one value');
  const adjacency = getCPURegionAdjacency(input, options.information);
  const table = getCPURegionTable(input.values, columnCount, options.standardize);
  let invalidCount = countCPURegionInvalid(input);
  if (input.extensiveValues && input.extensiveThreshold)
    for (const value of [...input.extensiveValues, input.extensiveThreshold[0]])
      if (!Number.isFinite(value) || value < 0) invalidCount++;
  const [seed] = input.parameters; // getGPUAZPRegionsParameterValues: [seed].
  const result = computeAZPRegionsTableOnCPU(
    adjacency,
    table,
    input.initialLabels,
    options.regionCapacity,
    columnCount,
    seed,
    options.minimumRegionSize ?? 1,
    input.extensiveValues,
    input.extensiveThreshold?.[0],
    invalidCount,
    options.maximumIterations ?? 256
  );
  const status = new Uint32Array(GPU_AZP_STATUS.length);
  status[GPU_AZP_STATUS.iterationCount] = result.iterationCount;
  status[GPU_AZP_STATUS.converged] = Number(result.invalidCount === 0 && result.converged);
  status[GPU_AZP_STATUS.iterationLimitReached] = Number(
    result.invalidCount === 0 && !result.converged
  );
  status[GPU_AZP_STATUS.invalidCount] = result.invalidCount;
  return {
    labels: result.labels,
    objective: Float32Array.of(
      result.invalidCount > 0 ? 0 : getCPURegionObjective(table, result.labels, columnCount)
    ),
    status
  };
}

/** CPU twin, honoring the round budget without GPU storage limits. */
export class CPUAZPRegionsOperation extends GPUCPUOperation {
  constructor(props: AZPRegionsOperationProps) {
    props = Object.freeze({...props});
    const id = props.id ?? 'azp-regions';
    validateCapacity(id, 'rowCount', props.rowCount, 2);
    validateRequirement(
      id,
      typeof props.maximumIterations === 'number',
      'maximumIterations must be a positive integer'
    );
    validateOptions(id, props, props.rowCount);
    validateRequirement(
      id,
      (props.extensiveValues === undefined) === (props.extensiveThreshold === undefined),
      'extensiveValues and extensiveThreshold are required together'
    );
    validateSpatialWeightsOperationInformation(
      id,
      'weights',
      props.weights.information,
      props.rowCount
    );
    const operation = new GPUAZPRegionsOperation(props);
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
        const offsets = getCPUOperationUint32Rows(id, inputs, 'offsets');
        if (offsets.length !== props.rowCount + 1)
          throw new Error(`${id} offsets length must equal rowCount + 1`);
        const result = computeAZPRegionsOnCPU(
          {
            offsets,
            neighbors: getCPUOperationUint32Rows(id, inputs, 'neighbors'),
            weights: getCPUOperationFloat32Rows(id, inputs, 'weights'),
            distances:
              props.weights.distances === undefined
                ? undefined
                : getCPUOperationFloat32Rows(id, inputs, 'distances'),
            values: getCPUOperationFloat32Rows(id, inputs, 'values'),
            initialLabels: getCPUOperationUint32Rows(id, inputs, 'initialLabels'),
            parameters: getCPUOperationUint32Rows(id, inputs, 'parameters'),
            extensiveValues:
              props.extensiveValues === undefined
                ? undefined
                : getCPUOperationFloat32Rows(id, inputs, 'extensiveValues'),
            extensiveThreshold:
              props.extensiveThreshold === undefined
                ? undefined
                : getCPUOperationFloat32Rows(id, inputs, 'extensiveThreshold')
          },
          {...props, information: props.weights.information}
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
