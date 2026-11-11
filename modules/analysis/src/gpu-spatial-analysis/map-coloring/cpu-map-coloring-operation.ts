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
 * - Completes sequential greedy coloring without a round budget; never leaves an uncolored row.
 * - `converged` is always 1; `iterationCount` is 1 completed greedy sweep. GPU rounds depend on
 *   invocation scheduling and are not comparable to this diagnostic.
 * - The 1024-round GPU limit is dropped. Chunked CSR is not supported by either operation.
 * - On asymmetric adjacency the CPU follows the documented priority order; GPU scheduling can
 *   produce different colors and conflicts. Symmetric adjacency gives the same complete coloring.
 */
export function computeMapColoringOnCPU(
  offsets: Uint32Array,
  neighbors: Uint32Array,
  parameters: Uint32Array,
  priority: GPUMapColoringPriority = 'largest-degree-first'
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
  let colorCount = 0;
  for (const row of order) {
    const usedColors = new Set<number>();
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
      const other = neighbors[slot];
      if (other !== row && colors[other] !== GPU_MAP_COLORING_UNCOLORED)
        usedColors.add(colors[other]);
    }
    let color = 0;
    while (usedColors.has(color)) color++;
    colors[row] = color;
    colorCount = Math.max(colorCount, color + 1);
  }
  let conflictCount = 0;
  for (let row = 0; row < rowCount; row++) {
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
      const other = neighbors[slot];
      if (other > row && colors[row] === colors[other]) conflictCount++;
    }
  }
  return {
    colors,
    colorCount: Uint32Array.of(colorCount),
    conflictCount: Uint32Array.of(conflictCount),
    converged: Uint32Array.of(1),
    iterationCount: Uint32Array.of(1)
  };
}

/** CPU twin. Optional iterationCount counts one CPU sweep, not GPU scheduling rounds. */
export class CPUMapColoringOperation extends GPUCPUOperation {
  constructor(props: MapColoringOperationProps) {
    const id = props.id ?? 'map-coloring';
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
    new GPUMapColoringOperation(props);
    super({
      id,
      inputs: {...props.weights, parameters: props.parameters},
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
        if (offsets.length !== props.rowCount + 1)
          throw new Error(`${id} offsets length must equal rowCount + 1`);
        if (weights.length !== neighbors.length)
          throw new Error(`${id} weights length must equal neighbors length`);
        const result = computeMapColoringOnCPU(
          offsets,
          neighbors,
          getCPUOperationUint32Rows(id, inputs, 'parameters'),
          priority
        );
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
