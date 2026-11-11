// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {GPU_SKATER_NO_CUT, GPU_SKATER_REGIONS_PARAMETER_LENGTH} from './gpu-skater-regions';
import {GPU_MINIMUM_SPANNING_TREE_STATUS} from './gpu-spatial-weights-minimum-spanning-tree';
import {validateSpatialWeightsOperationInformation} from '../spatial-weights/gpu-spatial-weights-algebra-operation';
import {
  GPUSkaterRegionsOperation,
  type SkaterRegionsOperationProps
} from './gpu-skater-regions-operation';

/** Packed inputs; only topology is used from weights, whose arrays must align with neighbors. */
export type CPUSkaterRegionsInput = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
  distances?: Float32Array;
  treeEdgeFlags: Uint32Array;
  componentLabels: Uint32Array;
  values: Float32Array;
  parameters: Uint32Array;
  treeStatus?: Uint32Array;
};
/** CPU dimensions. maximumRegionCount retains the cut-log layout and per-frame target clamp. */
export type CPUSkaterRegionsOptions = {columnCount?: number; maximumRegionCount: number};
/** Same row labels, region scalar and fixed-size cut log as GPUSkaterRegions. */
export type CPUSkaterRegionsResult = {
  labels: Uint32Array;
  regionCount: Uint32Array;
  cutEdges: Uint32Array;
  cutGains: Float32Array;
};

function validateOptions(id: string, options: CPUSkaterRegionsOptions): void {
  validateCapacity(id, 'columnCount', options.columnCount ?? 1);
  validateCapacity(id, 'maximumRegionCount', options.maximumRegionCount, 2);
}
type TreeEdge = {first: number; second: number; slot: number; child: number; live: boolean};

/**
 * Greedy minimum-spanning-forest cuts maximizing the between-side sum of squares. Each candidate
 * traverses its current region afresh; labels remain the vertex closest to the original root.
 * Zero-gain cuts are allowed, minimum sizes are inclusive, and exact ties use the lowest edge
 * slot (the lesser of its two CSR slots). The parameter layout is the two uint32 words packed
 * by getGPUSkaterRegionsParameterValues: regionCount, minimumRegionSize.
 *
 * What differs from the GPU contributor:
 * - Direct float64 side sums replace compensated float32 tour sums and repeated subtraction.
 *   Gains round once at output; near ties can change the cut sequence, since GPU gain keys are
 *   float32. A finite float64 gain above float32 range is eligible and publishes Infinity.
 * - The 16-column and GPU storage-binding limits are dropped. maximumRegionCount remains for
 *   the output log shape and target clamp, rather than an additional CPU iteration budget.
 *   Empty input slot arrays are accepted.
 * - Malformed CSR arrays, non-forest tree flags and inconsistent componentLabels throw errors
 *   naming the input. The GPU assumes a symmetric CSR and valid upstream forest outputs.
 * - Nonfinite values without treeStatus suppress cuts in their whole initial tree, as the GPU's
 *   nonfinite region sums do. With nonzero treeStatus.invalidCount no tree is cut.
 *
 * @throws Error naming inconsistent arrays or a malformed forest.
 */
export function computeSkaterRegionsOnCPU(
  input: CPUSkaterRegionsInput,
  options: CPUSkaterRegionsOptions
): CPUSkaterRegionsResult {
  validateOptions('skater-regions', options);
  const {offsets, neighbors, values, componentLabels, treeEdgeFlags, parameters} = input;
  const rowCount = offsets.length - 1;
  const columnCount = options.columnCount ?? 1;
  if (
    rowCount < 1 ||
    offsets[0] !== 0 ||
    offsets.some(
      (offset, row) => offset > neighbors.length || (row > 0 && offset < offsets[row - 1])
    )
  )
    throw new Error(
      'offsets must be nondecreasing CSR offsets within neighbors length, starting at zero'
    );
  if (input.weights.length !== neighbors.length)
    throw new Error('weights length must equal neighbors length');
  if (input.distances && input.distances.length !== neighbors.length)
    throw new Error('distances length must equal neighbors length');
  if (treeEdgeFlags.length !== neighbors.length)
    throw new Error('treeEdgeFlags length must equal neighbors length');
  if (componentLabels.length !== rowCount)
    throw new Error('componentLabels length must equal row count');
  if (values.length !== rowCount * columnCount)
    throw new Error('values length must equal row count * columnCount');
  if (parameters.length < GPU_SKATER_REGIONS_PARAMETER_LENGTH)
    throw new Error('parameters must hold the getGPUSkaterRegionsParameterValues layout');
  if (input.treeStatus && input.treeStatus.length < GPU_MINIMUM_SPANNING_TREE_STATUS.length)
    throw new Error('treeStatus must hold GPU_MINIMUM_SPANNING_TREE_STATUS.length words');
  const rowSlots = Array.from({length: rowCount}, () => new Map<number, number>());
  for (let row = 0; row < rowCount; row++)
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
      const neighbor = neighbors[slot];
      if (neighbor >= rowCount || (slot > offsets[row] && neighbor <= neighbors[slot - 1]))
        throw new Error('neighbors must be valid strictly ascending row IDs');
      rowSlots[row].set(neighbor, slot);
    }
  const edges: TreeEdge[] = [];
  const adjacency: number[][] = Array.from({length: rowCount}, () => []);
  for (let row = 0; row < rowCount; row++)
    for (const [neighbor, slot] of rowSlots[row]) {
      const reverse = rowSlots[neighbor].get(row);
      if (reverse === undefined) throw new Error('neighbors must list reciprocal links');
      if (slot <= reverse && treeEdgeFlags[slot] !== 0) {
        if (neighbor === row) throw new Error('treeEdgeFlags must not select self links');
        adjacency[row].push(edges.length);
        adjacency[neighbor].push(edges.length);
        edges.push({first: row, second: neighbor, slot, child: GPU_SKATER_NO_CUT, live: true});
      }
    }
  const discovered = new Uint8Array(rowCount);
  const treeHasInvalidValues = new Uint8Array(rowCount);
  let regionCount = 0;
  for (let root = 0; root < rowCount; root++) {
    if (discovered[root]) continue;
    regionCount++;
    const queue = [root];
    discovered[root] = 1;
    const parentEdges = new Map<number, number>();
    for (let cursor = 0; cursor < queue.length; cursor++) {
      const row = queue[cursor];
      if (componentLabels[row] !== root)
        throw new Error('componentLabels must name the smallest row of each flagged tree');
      for (let column = 0; column < columnCount; column++)
        if (!Number.isFinite(values[row * columnCount + column])) treeHasInvalidValues[root] = 1;
      for (const edgeIndex of adjacency[row]) {
        if (parentEdges.get(row) === edgeIndex) continue;
        const edge = edges[edgeIndex];
        const neighbor = edge.first === row ? edge.second : edge.first;
        if (discovered[neighbor]) throw new Error('treeEdgeFlags must select an acyclic forest');
        edge.child = neighbor;
        discovered[neighbor] = 1;
        parentEdges.set(neighbor, edgeIndex);
        queue.push(neighbor);
      }
    }
  }
  const labels = componentLabels.slice();
  const stepCount = options.maximumRegionCount - 1;
  const result: CPUSkaterRegionsResult = {
    labels,
    regionCount: Uint32Array.of(regionCount),
    cutEdges: new Uint32Array(stepCount).fill(GPU_SKATER_NO_CUT),
    cutGains: new Float32Array(stepCount)
  };
  const [requestedRegionCount, requestedMinimumSize] = parameters;
  const target = input.treeStatus?.[GPU_MINIMUM_SPANNING_TREE_STATUS.invalidCount]
    ? 0
    : Math.min(requestedRegionCount, options.maximumRegionCount);
  const minimumSize = Math.max(requestedMinimumSize, 1);
  for (let step = 0; step < stepCount && regionCount < target; step++) {
    let bestGain = -Infinity;
    let bestEdge: TreeEdge | undefined;
    let bestSide: number[] = [];
    for (const edge of edges) {
      if (!edge.live || treeHasInvalidValues[componentLabels[edge.child]]) continue;
      const region = labels[edge.child];
      const side = [edge.child];
      const included = new Set(side);
      for (let cursor = 0; cursor < side.length; cursor++)
        for (const edgeIndex of adjacency[side[cursor]]) {
          const link = edges[edgeIndex];
          if (!link.live || link === edge) continue;
          const neighbor = link.first === side[cursor] ? link.second : link.first;
          if (labels[neighbor] === region && !included.has(neighbor)) {
            included.add(neighbor);
            side.push(neighbor);
          }
        }
      const remainder: number[] = [];
      for (let row = 0; row < rowCount; row++)
        if (labels[row] === region && !included.has(row)) remainder.push(row);
      if (side.length < minimumSize || remainder.length < minimumSize) continue;
      let squaredDifference = 0;
      for (let column = 0; column < columnCount; column++) {
        const origin = values[side[0] * columnCount + column];
        let sideSum = 0;
        let remainderSum = 0;
        for (const row of side) sideSum += values[row * columnCount + column] - origin;
        for (const row of remainder) remainderSum += values[row * columnCount + column] - origin;
        const difference = sideSum / side.length - remainderSum / remainder.length;
        squaredDifference += difference * difference;
      }
      const gain =
        (side.length * remainder.length * squaredDifference) / (side.length + remainder.length);
      if (
        Number.isFinite(gain) &&
        (gain > bestGain || (gain === bestGain && edge.slot < bestEdge!.slot))
      ) {
        bestGain = gain;
        bestEdge = edge;
        bestSide = side;
      }
    }
    if (!bestEdge) break;
    bestEdge.live = false;
    for (const row of bestSide) labels[row] = bestEdge.child;
    result.cutEdges[step] = bestEdge.slot;
    result.cutGains[step] = bestGain;
    regionCount++;
  }
  result.regionCount[0] = regionCount;
  return result;
}

/** CPU twin, with fresh float64 side sums and no GPU column or storage-binding cap. */
export class CPUSkaterRegionsOperation extends GPUCPUOperation {
  constructor(props: SkaterRegionsOperationProps) {
    const id = props.id ?? 'skater-regions';
    validateOptions(id, props);
    validateCapacity(id, 'rowCount', props.rowCount);
    validateCapacity(id, 'edgeCount', props.edgeCount, 0);
    validateSpatialWeightsOperationInformation(
      id,
      'weights',
      props.weights.information,
      props.rowCount
    );
    const operation = new GPUSkaterRegionsOperation(props);
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
        const neighbors = getCPUOperationUint32Rows(id, inputs, 'neighbors');
        if (offsets.length !== props.rowCount + 1)
          throw new Error(`${id} offsets length must equal rowCount + 1`);
        if (neighbors.length !== props.edgeCount)
          throw new Error(`${id} neighbors length must equal edgeCount`);
        const result = computeSkaterRegionsOnCPU(
          {
            offsets,
            neighbors,
            weights: getCPUOperationFloat32Rows(id, inputs, 'weights'),
            distances:
              props.weights.distances === undefined
                ? undefined
                : getCPUOperationFloat32Rows(id, inputs, 'distances'),
            treeEdgeFlags: getCPUOperationUint32Rows(id, inputs, 'treeEdgeFlags'),
            componentLabels: getCPUOperationUint32Rows(id, inputs, 'componentLabels'),
            values: getCPUOperationFloat32Rows(id, inputs, 'values'),
            parameters: getCPUOperationUint32Rows(id, inputs, 'parameters'),
            treeStatus:
              props.treeStatus === undefined
                ? undefined
                : getCPUOperationUint32Rows(id, inputs, 'treeStatus')
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUSkaterRegionsResult]
          ])
        );
      }
    });
  }
}
