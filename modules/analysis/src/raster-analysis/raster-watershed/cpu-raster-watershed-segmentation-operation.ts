// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRasterOperationArray} from '../raster-grid/raster-operation-band';
import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotRasterGrid} from '../raster-grid/raster-construction-snapshot';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationUint32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {
  calibrateRasterOperationBand,
  readRasterOperationBand,
  snapshotRasterOperationBand,
  validateRasterOperationOutputNames,
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
 * - Explicit `maximumSteps` bounds FIFO pops and `maximumIterations` bounds synchronous rooted
 *   Boruvka contractions, matching the GPU budgets. An incomplete run marks usable non-seeds
 *   unresolved (0xfffffffe), with `converged=0` and `iterationLimitReached=1`.
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
  validateRasterOperationArray(id, 'markers', markers, 'Uint32Array');
  validateRasterOperationBand(id, input, options.grid);
  validateWatershedOptions(id, options);
  const count = options.grid.width * options.grid.height;
  if (markers.length < count) throw new Error(`${id} markers must cover grid cells`);
  const elevations = calibrateRasterOperationBand(input, count);
  return options.method === 'minimum-spanning-forest'
    ? computeWatershedForest(elevations, markers, options)
    : computeWatershedFlood(elevations, markers, options);
}

function computeWatershedFlood(
  elevations: Float64Array,
  markers: Uint32Array,
  options: CPURasterWatershedSegmentationOptions
): CPURasterWatershedSegmentationResult {
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
  const maximumSteps = options.maximumSteps ?? count;
  let processedCount = 0;
  while (heap.length > 0 && processedCount < maximumSteps) {
    processedCount++;
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
  const converged = heap.length === 0;
  if (!converged) {
    for (let cell = 0; cell < count; cell++) {
      if (Number.isFinite(elevations[cell]) && markers[cell] === 0) labels[cell] = 0xfffffffe;
    }
  }
  return createWatershedResult(labels, converged);
}

function computeWatershedForest(
  elevations: Float64Array,
  markers: Uint32Array,
  options: CPURasterWatershedSegmentationOptions
): CPURasterWatershedSegmentationResult {
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
  if (options.maximumIterations !== undefined) {
    return computeBoundedWatershedForest(
      edges,
      markers.subarray(0, count),
      usable,
      options.maximumIterations
    );
  }
  for (const edge of edges) {
    const first = findRoot(edge.source);
    const second = findRoot(edge.target);
    if (first === second || (labels[first] !== 0 && labels[second] !== 0)) continue;
    parents[second] = first;
    if (labels[first] === 0) labels[first] = labels[second];
  }
  return createWatershedResult(
    Uint32Array.from({length: count}, (_, cell) => (usable(cell) ? labels[findRoot(cell)] : 0)),
    true
  );
}

/** Synchronous rooted contractions preserve the meaning of an explicit GPU round budget. */
function computeBoundedWatershedForest(
  edges: readonly {source: number; target: number}[],
  markers: Uint32Array,
  usable: (cell: number) => boolean,
  maximumIterations: number
): CPURasterWatershedSegmentationResult {
  const count = markers.length;
  let components = Uint32Array.from({length: count}, (_, cell) => cell);
  for (let iteration = 0; iteration < maximumIterations; iteration++) {
    const selectedEdges = new Float64Array(count).fill(-1);
    for (const [edgeIndex, edge] of edges.entries()) {
      const first = components[edge.source];
      const second = components[edge.target];
      if (first === second) continue;
      // Edges arrive in total (cost, canonical ID) order, so the first is the minimum.
      if (markers[first] === 0 && selectedEdges[first] === -1) selectedEdges[first] = edgeIndex;
      if (markers[second] === 0 && selectedEdges[second] === -1) selectedEdges[second] = edgeIndex;
    }
    const parents = Uint32Array.from({length: count}, (_, cell) => cell);
    let changed = false;
    for (let component = 0; component < count; component++) {
      const edgeIndex = selectedEdges[component];
      if (edgeIndex === -1) continue;
      changed = true;
      const edge = edges[edgeIndex];
      const first = components[edge.source];
      const second = components[edge.target];
      const neighbor = first === component ? second : first;
      parents[component] =
        selectedEdges[neighbor] === edgeIndex && component < neighbor ? component : neighbor;
    }
    if (!changed) break;
    components = Uint32Array.from(components, component => {
      let root = component;
      while (parents[root] !== root) root = parents[root];
      // Compress the shared parent forest while resolving it. Subsequent components can
      // reuse the root instead of traversing the same flat-row chain from scratch.
      let current = component;
      while (parents[current] !== root) {
        const parent = parents[current];
        parents[current] = root;
        current = parent;
      }
      return root;
    });
  }
  const converged = !edges.some(edge => {
    const first = components[edge.source];
    const second = components[edge.target];
    return first !== second && (markers[first] === 0 || markers[second] === 0);
  });
  const labels = Uint32Array.from({length: count}, (_, cell) => {
    if (!usable(cell)) return 0;
    if (!converged && markers[cell] === 0) return 0xfffffffe;
    return markers[components[cell]];
  });
  return createWatershedResult(labels, converged);
}

function createWatershedResult(
  labels: Uint32Array,
  converged: boolean
): CPURasterWatershedSegmentationResult {
  return {
    labels,
    converged: Uint32Array.of(Number(converged)),
    iterationLimitReached: Uint32Array.of(Number(!converged))
  };
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

/** CPU twin with binding caller budgets and no GPU FIFO size or command-node caps. */
export class CPURasterWatershedSegmentationOperation extends GPUCPUOperation {
  constructor(props: RasterWatershedSegmentationOperationProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'grid',
      'input',
      'markers',
      'connectivity',
      'method',
      'maximumSteps',
      'maximumIterations',
      'output'
    ]);
    const output = snapshotConstructionFields(props.output, ['labels', 'status']);
    props = Object.freeze({
      ...props,
      grid: snapshotRasterGrid(props.grid),
      input: snapshotRasterOperationBand(props.input),
      output: Object.freeze({
        ...output,
        status: snapshotConstructionFields(output.status, ['converged', 'iterationLimitReached'])
      })
    });
    const id = props.id ?? 'raster-watershed-segmentation';
    validateRasterOperationOutputNames(
      id,
      {
        'output.labels': props.output.labels,
        'output.status.converged': props.output.status.converged
      },
      {'output.status.iterationLimitReached': props.output.status.iterationLimitReached},
      {
        'input.storage.values': props.input.storage.values,
        ...(props.input.validity !== undefined ? {'input.validity': props.input.validity} : {}),
        markers: props.markers
      }
    );
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
