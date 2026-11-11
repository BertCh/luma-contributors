// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationUint32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {
  calibrateRasterOperationBand,
  readRasterOperationBand,
  validateRasterOperationBand,
  type CPURasterOperationBand
} from '../raster-grid/raster-operation-band';
import type {RasterWatershedSegmentationOperationProps} from './gpu-raster-watershed-segmentation-operation';

export type CPURasterWatershedSegmentationOptions = Omit<
  RasterWatershedSegmentationOperationProps,
  'id' | 'input' | 'markers' | 'output'
>;
export type CPURasterWatershedSegmentationResult = {
  labels: Uint32Array;
  converged: Uint32Array;
  iterationLimitReached: Uint32Array;
};

/** Four canonical forest edge IDs per cell must fit below the reserved uint32 sentinel. */
const MAXIMUM_FOREST_CELL_COUNT = 0x3fffffff;

/**
 * Marker-controlled FIFO minimax priority flooding, or a rooted minimum-spanning forest. FIFO
 * ties follow discovery age, starting with row-major markers and row-major neighbour offsets.
 * Forest edges order by maximum endpoint elevation and then canonical edge ID.
 *
 * What differs from the GPU contributor:
 * - Calibration and costs use float64 without intermediate float32 rounding; nearly equal
 *   calibrated elevations can therefore order differently. Calibrated elevations beyond float32
 *   range can participate instead of becoming GPU nodata.
 * - Flooding and forest merging run to completion, ignoring `maximumSteps`/`maximumIterations`.
 *   `converged` is always one and `iterationLimitReached` zero; no unresolved labels are emitted.
 * - GPU FIFO cell and forest round limits are dropped. The forest's uint32 edge-ID cell bound
 *   remains so canonical edge identities retain their documented representation.
 *
 * @throws If band values/validity do not exactly match the grid, markers do not cover it, or options are invalid.
 */
export function computeRasterWatershedSegmentationOnCPU(
  input: CPURasterOperationBand,
  markers: Uint32Array,
  options: CPURasterWatershedSegmentationOptions
): CPURasterWatershedSegmentationResult {
  const id = 'computeRasterWatershedSegmentationOnCPU';
  validateRasterOperationBand(id, input, options.grid);
  validateWatershedOptions(id, options);
  const count = options.grid.width * options.grid.height;
  if (markers.length < count) throw new Error(`${id} markers must cover grid cells`);
  const elevations = calibrateRasterOperationBand(input, count);
  const labels =
    options.method === 'minimum-spanning-forest'
      ? computeWatershedForest(elevations, markers, options)
      : computeWatershedFlood(elevations, markers, options);
  return {labels, converged: new Uint32Array([1]), iterationLimitReached: new Uint32Array([0])};
}

function computeWatershedFlood(
  elevations: Float64Array,
  markers: Uint32Array,
  options: CPURasterWatershedSegmentationOptions
): Uint32Array {
  const count = elevations.length;
  const labels = new Uint32Array(count);
  const costs = new Float64Array(count).fill(Infinity);
  const ages = new Float64Array(count);
  const heap: number[] = [];
  let age = 0;
  const precedes = (first: number, second: number) =>
    costs[first] < costs[second] || (costs[first] === costs[second] && ages[first] < ages[second]);
  const insertCell = (cell: number) => {
    let slot = heap.length;
    heap.push(cell);
    while (slot > 0) {
      const parent = Math.floor((slot - 1) / 2);
      if (!precedes(cell, heap[parent])) break;
      heap[slot] = heap[parent];
      slot = parent;
    }
    heap[slot] = cell;
  };
  for (let cell = 0; cell < count; cell++) {
    if (Number.isFinite(elevations[cell]) && markers[cell] > 0 && markers[cell] < 0xfffffffe) {
      labels[cell] = markers[cell];
      costs[cell] = elevations[cell];
      ages[cell] = age++;
      insertCell(cell);
    }
  }
  while (heap.length > 0) {
    const best = heap[0];
    const replacement = heap.pop()!;
    if (heap.length > 0) {
      let slot = 0;
      while (slot * 2 + 1 < heap.length) {
        let child = slot * 2 + 1;
        if (child + 1 < heap.length && precedes(heap[child + 1], heap[child])) child++;
        if (!precedes(heap[child], replacement)) break;
        heap[slot] = heap[child];
        slot = child;
      }
      heap[slot] = replacement;
    }
    for (let rowOffset = -1; rowOffset <= 1; rowOffset++) {
      for (let columnOffset = -1; columnOffset <= 1; columnOffset++) {
        if (
          (rowOffset === 0 && columnOffset === 0) ||
          ((options.connectivity ?? 4) === 4 && Math.abs(rowOffset) + Math.abs(columnOffset) !== 1)
        )
          continue;
        const column = (best % options.grid.width) + columnOffset;
        const row = Math.floor(best / options.grid.width) + rowOffset;
        if (column < 0 || column >= options.grid.width || row < 0 || row >= options.grid.height)
          continue;
        const neighbor = row * options.grid.width + column;
        if (
          !Number.isFinite(elevations[neighbor]) ||
          markers[neighbor] !== 0 ||
          labels[neighbor] !== 0
        )
          continue;
        costs[neighbor] = Math.max(costs[best], elevations[neighbor]);
        labels[neighbor] = labels[best];
        ages[neighbor] = age++;
        insertCell(neighbor);
      }
    }
  }
  return labels;
}

function computeWatershedForest(
  elevations: Float64Array,
  markers: Uint32Array,
  options: CPURasterWatershedSegmentationOptions
): Uint32Array {
  const count = elevations.length;
  const parents = Uint32Array.from({length: count}, (_, cell) => cell);
  const labels = Uint32Array.from(markers.subarray(0, count));
  const usable = (cell: number) => Number.isFinite(elevations[cell]) && markers[cell] < 0xfffffffe;
  const findRoot = (cell: number): number => {
    while (parents[cell] !== cell) {
      parents[cell] = parents[parents[cell]];
      cell = parents[cell];
    }
    return cell;
  };
  const edges: {source: number; target: number; cost: number; identity: number}[] = [];
  const columnOffsets = [1, -1, 0, 1];
  const rowOffsets = [0, 1, 1, 1];
  for (let cell = 0; cell < count; cell++) {
    if (!usable(cell)) continue;
    for (let direction = 0; direction < 4; direction++) {
      if ((options.connectivity ?? 4) === 4 && direction % 2 === 1) continue;
      const column = (cell % options.grid.width) + columnOffsets[direction];
      const row = Math.floor(cell / options.grid.width) + rowOffsets[direction];
      if (column < 0 || column >= options.grid.width || row >= options.grid.height) continue;
      const target = row * options.grid.width + column;
      if (usable(target))
        edges.push({
          source: cell,
          target,
          cost: Math.max(elevations[cell], elevations[target]),
          identity: cell * 4 + direction
        });
    }
  }
  edges.sort((first, second) => first.cost - second.cost || first.identity - second.identity);
  for (const edge of edges) {
    const first = findRoot(edge.source);
    const second = findRoot(edge.target);
    if (first === second || (labels[first] !== 0 && labels[second] !== 0)) continue;
    parents[second] = first;
    if (labels[first] === 0) labels[first] = labels[second];
  }
  return Uint32Array.from({length: count}, (_, cell) =>
    usable(cell) ? labels[findRoot(cell)] : 0
  );
}

function validateWatershedOptions(
  id: string,
  options: CPURasterWatershedSegmentationOptions
): void {
  validateRequirement(
    id,
    ['fifo', 'minimum-spanning-forest'].includes(options.method ?? 'fifo'),
    "method must be 'fifo' or 'minimum-spanning-forest'"
  );
  validateRequirement(
    id,
    options.connectivity === undefined || options.connectivity === 4 || options.connectivity === 8,
    'connectivity must be 4 or 8'
  );
  if (options.method === 'minimum-spanning-forest') {
    // Four canonical IDs per cell must fit below the reserved uint32 sentinel.
    validateCapacity(
      id,
      'forest grid cells',
      options.grid.width * options.grid.height,
      1,
      MAXIMUM_FOREST_CELL_COUNT
    );
    validateRequirement(
      id,
      options.maximumSteps === undefined,
      'maximumSteps applies to fifo only'
    );
    if (options.maximumIterations !== undefined)
      validateCapacity(id, 'maximumIterations', options.maximumIterations);
  } else {
    validateRequirement(
      id,
      options.maximumIterations === undefined,
      'maximumIterations applies to minimum-spanning-forest only'
    );
    if (options.maximumSteps !== undefined)
      validateCapacity(id, 'maximumSteps', options.maximumSteps);
  }
}

/** CPU twin with complete flooding/merging and no GPU FIFO size or round caps. */
export class CPURasterWatershedSegmentationOperation extends GPUCPUOperation {
  constructor(props: RasterWatershedSegmentationOperationProps) {
    const id = props.id ?? 'raster-watershed-segmentation';
    validateRasterOperationBand(id, props.input, props.grid);
    validateWatershedOptions(id, props);
    const iterationLimitReached = props.output.status.iterationLimitReached;
    super({
      id,
      inputs: {
        values: props.input.storage.values,
        markers: props.markers,
        ...(props.input.validity !== undefined ? {validity: props.input.validity} : {})
      },
      outputs: {
        [props.output.labels]: gpuGraph.vector('uint32', {
          length: props.grid.width * props.grid.height
        }),
        [props.output.status.converged]: gpuGraph.vector('uint32', {length: 1}),
        ...(iterationLimitReached !== undefined
          ? {[iterationLimitReached]: gpuGraph.vector('uint32', {length: 1})}
          : {})
      },
      execute({inputs}) {
        const result = computeRasterWatershedSegmentationOnCPU(
          readRasterOperationBand(id, inputs, props.input),
          getCPUOperationUint32Rows(id, inputs, 'markers'),
          props
        );
        return {
          [props.output.labels]: result.labels,
          [props.output.status.converged]: result.converged,
          ...(iterationLimitReached !== undefined
            ? {[iterationLimitReached]: result.iterationLimitReached}
            : {})
        };
      }
    });
  }
}
