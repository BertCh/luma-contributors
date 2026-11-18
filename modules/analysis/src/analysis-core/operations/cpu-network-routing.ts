// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCapacity} from '../kernels/gpu-contributor-utils';

const NONE = 0xffffffff;

/** Packed nonnegative network search inputs. Offset values have defined GPU clamping. @internal */
export type CPUNetworkRoutingInput = {
  topology: {offsets: Uint32Array; neighbors: Uint32Array; weights: Float32Array};
  startNodes: Uint32Array;
  startCosts?: Float32Array;
  startCount?: number | Uint32Array;
  costLimit?: Float32Array;
  costBreaks?: Float32Array;
  activeIterations?: Uint32Array;
};

/** Checks topology lengths; offset values have a defined clamped GPU interpretation. @internal */
export function validateNetworkRoutingArrays(topology: CPUNetworkRoutingInput['topology']): number {
  if (topology.offsets.length < 2)
    throw new Error('topology.offsets must contain at least two rows');
  if (topology.weights.length !== topology.neighbors.length)
    throw new Error('topology.weights length must equal topology.neighbors length');
  return topology.offsets.length - 1;
}

/**
 * Nonnegative shortest paths in float64. Unbounded searches use a binary min heap; an explicit
 * maximumHops uses a synchronous frontier and converges only when that frontier empties.
 * `id` names the calling function in messages. @internal
 */
export function computeNetworkRoutingDistances(
  id: string,
  input: CPUNetworkRoutingInput,
  laneCount = 1,
  maximumHops = Infinity
): {
  distances: Float64Array;
  scannedEdges: number;
  reachedEdges: number;
  converged: boolean;
  hopCount: number;
} {
  const nodeCount = validateNetworkRoutingArrays(input.topology);
  validateCapacity(id, 'laneCount', laneCount);
  if (input.startCosts && input.startCosts.length !== input.startNodes.length)
    throw new Error('startCosts length must equal startNodes length');
  for (const [name, values] of [
    ['costLimit', input.costLimit],
    ['activeIterations', input.activeIterations],
    ['startCount', typeof input.startCount === 'object' ? input.startCount : undefined]
  ] as const) {
    if (values && values.length !== 1) throw new Error(`${name} must contain exactly one row`);
  }
  if (typeof input.startCount === 'number')
    validateCapacity(id, 'startCount', input.startCount, 0, input.startNodes.length);
  const startCount = Math.min(
    typeof input.startCount === 'number'
      ? input.startCount
      : (input.startCount?.[0] ?? input.startNodes.length),
    input.startNodes.length
  );
  const costLimit = input.costLimit?.[0] ?? Infinity;
  const distances = new Float64Array(nodeCount * laneCount).fill(Infinity);
  const heap: {node: number; cost: number}[] = [];
  const pushEntry = (node: number, cost: number) => {
    const entry = {node, cost};
    let position = heap.length;
    heap.push(entry);
    while (position > 0) {
      const parent = Math.floor((position - 1) / 2);
      if (heap[parent].cost <= cost) break;
      heap[position] = heap[parent];
      position = parent;
    }
    heap[position] = entry;
  };
  for (let start = 0; start < startCount; start++) {
    const node = input.startNodes[start];
    const cost = input.startCosts?.[start] ?? 0;
    if (
      node < distances.length &&
      Number.isFinite(cost) &&
      cost >= 0 &&
      !(cost > costLimit) &&
      cost < distances[node]
    ) {
      distances[node] = cost === 0 ? 0 : cost;
      pushEntry(node, distances[node]);
    }
  }
  let scannedEdges = 0;
  let reachedEdges = 0;
  const {offsets, neighbors, weights} = input.topology;
  if (maximumHops !== Infinity) {
    validateCapacity(id, 'maximumHops', maximumHops, 0);
    let frontier = new Set(heap.map(entry => entry.node));
    let hopCount = 0;
    // A synchronous frontier consumes at most one edge per hop, independent of CSR row order.
    // Never compute labels beyond the caller's explicit round × local-hop allowance.
    while (frontier.size && hopCount < maximumHops) {
      const sources = Array.from(frontier, source => ({source, cost: distances[source]}));
      const next = new Set<number>();
      for (const {source, cost} of sources) {
        const node = source % nodeCount;
        const laneBase = source - node;
        for (
          let edge = Math.min(offsets[node], neighbors.length);
          edge < Math.min(offsets[node + 1], neighbors.length);
          edge++
        ) {
          scannedEdges++;
          const target = neighbors[edge];
          if (target >= nodeCount) continue;
          reachedEdges++;
          const candidate = cost + weights[edge];
          if (
            weights[edge] >= 0 &&
            Number.isFinite(candidate) &&
            !(candidate > costLimit) &&
            candidate < distances[laneBase + target]
          ) {
            distances[laneBase + target] = candidate;
            next.add(laneBase + target);
          }
        }
      }
      frontier = next;
      hopCount++;
    }
    // reachedEdges is the degree sum of final finite labels, not repeated relaxation visits.
    reachedEdges = 0;
    for (let source = 0; source < distances.length; source++) {
      if (!Number.isFinite(distances[source])) continue;
      const node = source % nodeCount;
      for (
        let edge = Math.min(offsets[node], neighbors.length);
        edge < Math.min(offsets[node + 1], neighbors.length);
        edge++
      )
        if (neighbors[edge] < nodeCount) reachedEdges++;
    }
    return {distances, scannedEdges, reachedEdges, converged: frontier.size === 0, hopCount};
  }
  while (heap.length) {
    const entry = heap[0];
    const last = heap.pop()!;
    if (heap.length) {
      let position = 0;
      while (position * 2 + 1 < heap.length) {
        let child = position * 2 + 1;
        if (child + 1 < heap.length && heap[child + 1].cost < heap[child].cost) child++;
        if (last.cost <= heap[child].cost) break;
        heap[position] = heap[child];
        position = child;
      }
      heap[position] = last;
    }
    if (entry.cost !== distances[entry.node]) continue;
    const node = entry.node % nodeCount;
    const laneBase = entry.node - node;
    for (
      let edge = Math.min(offsets[node], neighbors.length);
      edge < Math.min(offsets[node + 1], neighbors.length);
      edge++
    ) {
      scannedEdges++;
      const target = neighbors[edge];
      if (target >= nodeCount) continue;
      reachedEdges++;
      if (!(weights[edge] >= 0)) continue;
      const candidate = entry.cost + weights[edge];
      if (
        Number.isFinite(candidate) &&
        !(candidate > costLimit) &&
        candidate < distances[laneBase + target]
      ) {
        distances[laneBase + target] = candidate;
        pushEntry(laneBase + target, candidate);
      }
    }
  }
  return {distances, scannedEdges, reachedEdges, converged: true, hopCount: 0};
}

/**
 * Strict parents and breadth-first tie levels, with the GPU's slot-indexed tie flags.
 * Overlapping CSR rows reuse flags without rechecking source costs or labels, as on the GPU.
 * @internal
 */
export function computeNetworkRoutingPredecessors(
  input: CPUNetworkRoutingInput,
  distances: Float64Array,
  labels?: Uint32Array,
  maximumTieHops = Infinity
): Uint32Array {
  const {offsets, neighbors, weights} = input.topology;
  const nodeCount = offsets.length - 1;
  const predecessors = new Uint32Array(nodeCount).fill(NONE);
  const levels = new Float64Array(nodeCount).fill(Infinity);
  const tieEdges = new Uint8Array(neighbors.length);
  const visitEdges = (visit: (source: number, target: number, edge: number) => void) => {
    for (let source = 0; source < nodeCount; source++) {
      if (!Number.isFinite(distances[source])) continue;
      for (
        let edge = Math.min(offsets[source], neighbors.length);
        edge < Math.min(offsets[source + 1], neighbors.length);
        edge++
      ) {
        const target = neighbors[edge];
        if (
          target < nodeCount &&
          weights[edge] >= 0 &&
          Number.isFinite(distances[target]) &&
          distances[source] + weights[edge] === distances[target] &&
          (!labels || labels[source] === labels[target])
        )
          visit(source, target, edge);
      }
    }
  };
  visitEdges((source, target, edge) => {
    if (distances[source] < distances[target]) {
      predecessors[target] = Math.min(predecessors[target], source);
      levels[target] = 0;
    } else if (distances[source] === distances[target]) {
      tieEdges[edge] = 1;
    }
  });
  const startCount = Math.min(
    typeof input.startCount === 'number'
      ? input.startCount
      : (input.startCount?.[0] ?? input.startNodes.length),
    input.startNodes.length
  );
  for (let start = 0; start < startCount; start++) {
    const node = input.startNodes[start];
    const cost = input.startCosts?.[start] ?? 0;
    if (
      node < nodeCount &&
      Number.isFinite(cost) &&
      cost >= 0 &&
      !(cost > (input.costLimit?.[0] ?? Infinity)) &&
      cost === distances[node] &&
      (!labels || labels[node] === start)
    )
      levels[node] = 0;
  }
  const ties: number[][] = Array.from({length: nodeCount}, () => []);
  // The tie passes read the slot flags, not the source-specific tightness/label conditions.
  for (let source = 0; source < nodeCount; source++) {
    for (
      let edge = Math.min(offsets[source], neighbors.length);
      edge < Math.min(offsets[source + 1], neighbors.length);
      edge++
    ) {
      if (tieEdges[edge]) ties[source].push(neighbors[edge]);
    }
  }
  const queue: number[] = [];
  for (let node = 0; node < nodeCount; node++) if (levels[node] === 0) queue.push(node);
  for (let cursor = 0; cursor < queue.length; cursor++) {
    const source = queue[cursor];
    if (levels[source] >= maximumTieHops) continue;
    for (const target of ties[source])
      if (levels[target] === Infinity) {
        levels[target] = levels[source] + 1;
        queue.push(target);
      }
  }
  for (let source = 0; source < nodeCount; source++)
    for (const target of ties[source]) {
      if (levels[source] + 1 === levels[target] && Number.isFinite(levels[target]))
        predecessors[target] = Math.min(predecessors[target], source);
    }
  return predecessors;
}
