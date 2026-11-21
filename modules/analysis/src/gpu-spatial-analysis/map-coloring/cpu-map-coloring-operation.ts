// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {hashUint32} from '../../analysis-core/kernels/hash-uint32';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPU_MAP_COLORING_PARAMETER_LENGTH,
  GPU_MAP_COLORING_UNCOLORED,
  type GPUMapColoringPriority
} from './gpu-map-coloring';
import {
  GPUMapColoringOperation,
  getMapColoringOperationInputs,
  snapshotMapColoringOperationProps,
  type MapColoringOperationProps
} from './gpu-map-coloring-operation';

export type CPUMapColoringResult = {
  colors: Uint32Array;
  colorCount: Uint32Array;
  conflictCount: Uint32Array;
  converged: Uint32Array;
  iterationCount: Uint32Array;
};

/**
 * Colors an unchunked CSR in descending (degree, seeded hash), then ascending row order.
 * Self slots are ignored in color selection but count toward the degree, as on the GPU.
 *
 * What differs from the GPU contributor:
 * - Without an explicit round budget, completes one priority-ordered greedy sweep.
 * - With a budget, visits rows in ascending order per round, retaining uncolored rows when
 *   a higher-priority neighbor is still uncolored. GPU scheduling can finish different rows
 *   within a round; both forms bind the supplied maximum and report actual convergence.
 * - The 1024-round GPU limit is dropped. Chunked CSR is not supported by either operation.
 * - On asymmetric adjacency the unbudgeted CPU sweep follows priority order; GPU scheduling can
 *   produce different colors and conflicts. Symmetric adjacency gives the same complete coloring.
 */
export function computeMapColoringOnCPU(
  offsets: Uint32Array,
  neighbors: Uint32Array,
  parameters: Uint32Array,
  priority: GPUMapColoringPriority = 'largest-degree-first',
  maximumIterations?: number
): CPUMapColoringResult {
  const rowCount = offsets.length - 1;
  if (rowCount < 1 || offsets[0] !== 0) {
    throw new Error(
      'computeMapColoringOnCPU offsets must start at 0 and describe at least one row'
    );
  }
  if (parameters.length < GPU_MAP_COLORING_PARAMETER_LENGTH) {
    throw new Error('computeMapColoringOnCPU parameters is too short');
  }
  validateRequirement(
    'computeMapColoringOnCPU',
    ['largest-degree-first', 'random'].includes(priority),
    "priority must be 'largest-degree-first' or 'random'"
  );
  if (maximumIterations !== undefined)
    validateCapacity('computeMapColoringOnCPU', 'maximumIterations', maximumIterations);
  for (let row = 0; row < rowCount; row++) {
    if (offsets[row] > offsets[row + 1] || offsets[row + 1] > neighbors.length) {
      throw new Error('computeMapColoringOnCPU offsets must be monotone and fit neighbors length');
    }
  }
  for (let slot = 0; slot < offsets[rowCount]; slot++) {
    if (neighbors[slot] >= rowCount)
      throw new Error('computeMapColoringOnCPU neighbors must contain valid row IDs');
  }
  // The packer layout is [seed], with length GPU_MAP_COLORING_PARAMETER_LENGTH.
  const [seed] = parameters;
  const hashes = Uint32Array.from({length: rowCount}, (_, row) => hashUint32(row ^ seed));
  const order = Array.from({length: rowCount}, (_, row) => row).sort((first, second) => {
    const degreeDifference =
      priority === 'largest-degree-first'
        ? offsets[second + 1] - offsets[second] - (offsets[first + 1] - offsets[first])
        : 0;
    return degreeDifference || hashes[second] - hashes[first] || first - second;
  });
  const colors = new Uint32Array(rowCount).fill(GPU_MAP_COLORING_UNCOLORED);
  const ranks = new Uint32Array(rowCount);
  order.forEach((row, rank) => {
    ranks[row] = rank;
  });
  let colorCount = 0;
  let iterationCount = 0;
  let converged = false;
  for (let round = 0; round < (maximumIterations ?? 1); round++) {
    iterationCount++;
    for (let rowSlot = 0; rowSlot < rowCount; rowSlot++) {
      const row = maximumIterations === undefined ? order[rowSlot] : rowSlot;
      if (colors[row] !== GPU_MAP_COLORING_UNCOLORED) continue;
      const usedColors = new Set<number>();
      let blocked = false;
      for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
        const other = neighbors[slot];
        if (other === row) continue;
        if (colors[other] === GPU_MAP_COLORING_UNCOLORED) {
          if (ranks[other] < ranks[row]) blocked = true;
        } else usedColors.add(colors[other]);
      }
      if (blocked) continue;
      let color = 0;
      while (usedColors.has(color)) color++;
      colors[row] = color;
      colorCount = Math.max(colorCount, color + 1);
    }
    converged = colors.every(color => color !== GPU_MAP_COLORING_UNCOLORED);
    if (converged) break;
  }
  let conflictCount = 0;
  for (let row = 0; row < rowCount; row++) {
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
      const other = neighbors[slot];
      if (
        other > row &&
        colors[row] !== GPU_MAP_COLORING_UNCOLORED &&
        colors[row] === colors[other]
      )
        conflictCount++;
    }
  }
  return {
    colors,
    colorCount: Uint32Array.of(colorCount),
    conflictCount: Uint32Array.of(conflictCount),
    converged: Uint32Array.of(Number(converged)),
    iterationCount: Uint32Array.of(iterationCount)
  };
}

/** CPU twin with ascending-row bounded rounds, or a complete greedy sweep without a budget. */
export class CPUMapColoringOperation extends GPUCPUOperation {
  constructor(props: MapColoringOperationProps) {
    props = snapshotMapColoringOperationProps(props);
    const id = props.id ?? 'map-coloring';
    const maximumIterations = props.maximumIterations;
    const rowCount = props.rowCount;
    validateCapacity(id, 'rowCount', props.rowCount);
    if (props.maximumIterations !== undefined)
      validateCapacity(id, 'maximumIterations', props.maximumIterations);
    const priority = props.priority ?? 'largest-degree-first';
    validateRequirement(
      id,
      ['largest-degree-first', 'random'].includes(priority),
      "priority must be 'largest-degree-first' or 'random'"
    );
    const {output} = props;
    const scalarShape = {length: 1};
    // Reuse the graph-port shape and disjointness checks; this allocates no GPU resources.
    new GPUMapColoringOperation({...props, maximumIterations: undefined});
    super({
      id,
      inputs: getMapColoringOperationInputs(props),
      outputs: {
        [output.colors]: gpuGraph.vector('uint32', {length: props.rowCount}),
        ...(output.colorCount ? {[output.colorCount]: gpuGraph.vector('uint32', scalarShape)} : {}),
        ...(output.conflictCount
          ? {[output.conflictCount]: gpuGraph.vector('uint32', scalarShape)}
          : {}),
        ...(output.status?.converged
          ? {[output.status.converged]: gpuGraph.vector('uint32', scalarShape)}
          : {}),
        ...(output.status?.iterationCount
          ? {[output.status.iterationCount]: gpuGraph.vector('uint32', scalarShape)}
          : {})
      },
      execute({inputs}) {
        const offsets = getCPUOperationUint32Rows(id, inputs, 'offsets');
        const neighbors = getCPUOperationUint32Rows(id, inputs, 'neighbors');
        const weights = getCPUOperationFloat32Rows(id, inputs, 'weights');
        if (offsets.length !== rowCount + 1)
          throw new Error(`${id} offsets length must equal rowCount + 1`);
        if (weights.length !== neighbors.length)
          throw new Error(`${id} weights length must equal neighbors length`);
        const result = computeMapColoringOnCPU(
          offsets,
          neighbors,
          getCPUOperationUint32Rows(id, inputs, 'parameters'),
          priority,
          maximumIterations
        );
        const incomplete = (['overflow', 'candidateOverflow'] as const).some(name => {
          if (!props.weights.status?.[name]) return false;
          const words = getCPUOperationUint32Rows(id, inputs, name);
          validateRequirement(
            id,
            words.length >= 1,
            `weights.status.${name} must contain one uint32 word; received ${words.length}`
          );
          return words[0] !== 0;
        });
        if (incomplete) {
          result.colors.fill(GPU_MAP_COLORING_UNCOLORED);
          result.colorCount.fill(0);
          result.conflictCount.fill(0);
          result.converged.fill(0);
        }
        return {
          [output.colors]: result.colors,
          ...(output.colorCount ? {[output.colorCount]: result.colorCount} : {}),
          ...(output.conflictCount ? {[output.conflictCount]: result.conflictCount} : {}),
          ...(output.status?.converged ? {[output.status.converged]: result.converged} : {}),
          ...(output.status?.iterationCount
            ? {[output.status.iterationCount]: result.iterationCount}
            : {})
        };
      }
    });
  }
}
