// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationUint32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import type {GPUTerrainFlowDirectionEncoding} from './terrain-flow-tracing';
import {
  getTerrainDrainageDirection,
  getTerrainDrainageReceiver,
  TERRAIN_DRAINAGE_NONE,
  TERRAIN_DRAINAGE_UNRESOLVED,
  validateTerrainDrainageOptions
} from './cpu-terrain-drainage-utils';
import {
  GPUTerrainWatershedsOperation,
  type TerrainWatershedsOperationProps
} from './gpu-terrain-watersheds-operation';

export type CPUTerrainWatershedsResult = {
  labels: Uint32Array;
  converged: Uint32Array;
  iterationLimitReached: Uint32Array;
  iterationCount: Uint32Array;
};

/**
 * Labels each path at its nearest inclusive pour point, or at its terminal cell without points.
 * Memoized walks preserve duplicate-point priority, invalid words and unknown destinations.
 *
 * What differs from the GPU contributor:
 * - Walks complete without maximumIterations or the GPU's 1024-round cap. iterationCount is 0
 *   (no GPU rounds); iterationLimitReached is always 0. converged is 0 only for receiver cycles.
 * - Cycles and paths into them are explicitly unresolved, as documented by the contributor;
 *   in-place GPU pointer jumping can instead collapse a cycle to an arbitrary cell.
 * - No float outputs or tolerance differences; resolved labels and point ties are exact integers.
 */
export function computeTerrainWatershedsOnCPU(
  flowDirections: Uint32Array,
  options: {grid: GPURasterMetadata; directionEncoding?: GPUTerrainFlowDirectionEncoding},
  pourPoints?: Uint32Array
): CPUTerrainWatershedsResult {
  const {grid, directionEncoding} = options;
  const count = validateTerrainDrainageOptions(
    'computeTerrainWatershedsOnCPU',
    grid,
    directionEncoding
  );
  if (flowDirections.length !== count)
    throw new Error('flowDirections must contain one word per grid cell');
  const markers = new Uint32Array(count).fill(TERRAIN_DRAINAGE_NONE);
  pourPoints?.forEach((cell, point) => {
    if (cell < count && flowDirections[cell] !== TERRAIN_DRAINAGE_NONE && point < markers[cell])
      markers[cell] = point;
  });
  const labels = new Uint32Array(count).fill(TERRAIN_DRAINAGE_NONE);
  const state = new Uint8Array(count);
  let hasCycle = false;
  for (let start = 0; start < count; start++) {
    if (state[start] === 2) continue;
    const path: number[] = [];
    let cell = start;
    let label = TERRAIN_DRAINAGE_NONE;
    while (true) {
      if (state[cell] === 2) {
        label = labels[cell];
        break;
      }
      if (state[cell] === 1) {
        label = TERRAIN_DRAINAGE_UNRESOLVED;
        hasCycle = true;
        break;
      }
      path.push(cell);
      state[cell] = 1;
      const word = flowDirections[cell];
      if (word === TERRAIN_DRAINAGE_NONE) break;
      if (pourPoints && markers[cell] !== TERRAIN_DRAINAGE_NONE) {
        label = markers[cell];
        break;
      }
      if (word === TERRAIN_DRAINAGE_UNRESOLVED) {
        label = TERRAIN_DRAINAGE_UNRESOLVED;
        break;
      }
      const receiver = getTerrainDrainageReceiver(
        cell,
        getTerrainDrainageDirection(word, directionEncoding),
        grid.width,
        grid.height
      );
      if (receiver === TERRAIN_DRAINAGE_NONE) {
        label = pourPoints ? TERRAIN_DRAINAGE_NONE : cell;
        break;
      }
      cell = receiver;
    }
    for (const member of path) {
      labels[member] = label;
      state[member] = 2;
    }
  }
  return {
    labels,
    converged: new Uint32Array([Number(!hasCycle)]),
    iterationLimitReached: new Uint32Array(1),
    iterationCount: new Uint32Array(1)
  };
}

/** CPU watershed twin. GPU iteration caps are dropped; cycles remain unresolved. */
export class CPUTerrainWatershedsOperation extends GPUCPUOperation {
  constructor(props: TerrainWatershedsOperationProps) {
    const operation = new GPUTerrainWatershedsOperation(props);
    const {id} = operation;
    validateTerrainDrainageOptions(id, props.grid, props.directionEncoding);
    if (props.maximumIterations !== undefined)
      validateCapacity(
        id,
        'maximumIterations',
        props.maximumIterations,
        1,
        Number.MAX_SAFE_INTEGER
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
        const result = computeTerrainWatershedsOnCPU(
          getCPUOperationUint32Rows(id, inputs, 'flowDirections'),
          props,
          props.pourPoints !== undefined
            ? getCPUOperationUint32Rows(id, inputs, 'pourPoints')
            : undefined
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUTerrainWatershedsResult]
          ])
        );
      }
    });
  }
}
