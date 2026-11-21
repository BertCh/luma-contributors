// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotRasterGrid} from '../../raster-analysis/raster-grid/raster-construction-snapshot';
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
  snapshotTerrainWatershedsOperationProps,
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
 * - Without maximumIterations, walks complete and iterationCount is 0 (no pointer rounds).
 *   An explicit budget instead bounds synchronous pointer rounds, including confirmation;
 *   iterationLimitReached is one if the last permitted round still changed a pointer.
 *   Explicit budgets are in [1,1024], as on the GPU contributor.
 * - Both forms reject cycles and unmarked unresolved directions, even when their pointers
 *   are stable. An inclusive pour point can intercept either path and restore convergence.
 * - No float outputs or tolerance differences; resolved labels and point ties are exact integers.
 */
export function computeTerrainWatershedsOnCPU(
  flowDirections: Uint32Array,
  options: {
    grid: GPURasterMetadata;
    directionEncoding?: GPUTerrainFlowDirectionEncoding;
    maximumIterations?: number;
  },
  pourPoints?: Uint32Array
): CPUTerrainWatershedsResult {
  const {
    grid: originalGrid,
    directionEncoding,
    maximumIterations
  } = snapshotConstructionFields(options, ['grid', 'directionEncoding', 'maximumIterations']);
  const grid = snapshotRasterGrid(originalGrid);
  const count = validateTerrainDrainageOptions(
    'computeTerrainWatershedsOnCPU',
    grid,
    directionEncoding
  );
  if (flowDirections.length !== count)
    throw new Error('flowDirections must contain one word per grid cell');
  if (maximumIterations !== undefined)
    validateCapacity(
      'computeTerrainWatershedsOnCPU',
      'maximumIterations',
      maximumIterations,
      1,
      1024
    );
  const markers = new Uint32Array(count).fill(TERRAIN_DRAINAGE_NONE);
  pourPoints?.forEach((cell, point) => {
    if (cell < count && flowDirections[cell] !== TERRAIN_DRAINAGE_NONE && point < markers[cell])
      markers[cell] = point;
  });
  const labels = new Uint32Array(count).fill(TERRAIN_DRAINAGE_NONE);
  const state = new Uint8Array(count);
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
  let iterationCount = 0;
  let changed = false;
  if (maximumIterations !== undefined) {
    let pointers = Uint32Array.from(flowDirections, (word, cell) => {
      if (
        word === TERRAIN_DRAINAGE_NONE ||
        word === TERRAIN_DRAINAGE_UNRESOLVED ||
        (pourPoints && markers[cell] !== TERRAIN_DRAINAGE_NONE)
      )
        return cell;
      const receiver = getTerrainDrainageReceiver(
        cell,
        getTerrainDrainageDirection(word, directionEncoding),
        grid.width,
        grid.height
      );
      return receiver === TERRAIN_DRAINAGE_NONE ? cell : receiver;
    });
    // Read one snapshot per round, as explicitly budgeted GPU watershed rounds do.
    // Both forms must spend an unchanged round before reporting convergence.
    for (; iterationCount < maximumIterations; ) {
      const nextPointers = pointers.slice();
      changed = false;
      for (let cell = 0; cell < count; cell++) {
        const next = pointers[pointers[cell]];
        nextPointers[cell] = next;
        changed ||= next !== pointers[cell];
      }
      pointers = nextPointers;
      iterationCount++;
      if (!changed) break;
    }
    for (let cell = 0; cell < count; cell++) {
      const reached = pointers[cell];
      if (flowDirections[cell] !== TERRAIN_DRAINAGE_NONE && pointers[reached] !== reached)
        labels[cell] = TERRAIN_DRAINAGE_UNRESOLVED;
    }
  }
  return {
    labels,
    converged: new Uint32Array([Number(!changed && !labels.includes(TERRAIN_DRAINAGE_UNRESOLVED))]),
    iterationLimitReached: new Uint32Array([Number(changed)]),
    iterationCount: new Uint32Array([iterationCount])
  };
}

/** CPU watershed twin. Explicit budgets bound pointer rounds; cycles remain unresolved. */
export class CPUTerrainWatershedsOperation extends GPUCPUOperation {
  constructor(props: TerrainWatershedsOperationProps) {
    props = snapshotTerrainWatershedsOperationProps(props);
    const operation = new GPUTerrainWatershedsOperation(props);
    const {id} = operation;
    validateTerrainDrainageOptions(id, props.grid, props.directionEncoding);
    if (props.maximumIterations !== undefined)
      validateCapacity(id, 'maximumIterations', props.maximumIterations, 1, 1024);
    const options = {
      grid: {...props.grid, affine: [...props.grid.affine] as typeof props.grid.affine},
      directionEncoding: props.directionEncoding,
      maximumIterations: props.maximumIterations
    };
    const hasPourPoints = props.pourPoints !== undefined;
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
          options,
          hasPourPoints ? getCPUOperationUint32Rows(id, inputs, 'pourPoints') : undefined
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
