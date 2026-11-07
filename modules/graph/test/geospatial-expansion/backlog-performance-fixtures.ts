// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Version the generator and reference contract whenever the workload changes. */
export const BACKLOG_BENCHMARK_VERSION = 1;
export const BACKLOG_BENCHMARK_SEED = 0x5eed2026;

function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

export function createPairWorkload(count: number, seed = BACKLOG_BENCHMARK_SEED) {
  const random = createRandom(seed);
  // Integer coordinates and noninteger bin boundaries avoid ambiguous float32 distance edges.
  const positions = Float32Array.from({length: count * 2}, () => Math.floor(random() * 64));
  const marks = Float32Array.from({length: count}, () => 1 + Math.floor(random() * 8));
  return {positions, marks, radiusCount: 4, maximumRadius: 80.375, updatedMaximumRadius: 41.375};
}

/** Direct double-precision ordered-pair definition, not an established CPU library benchmark. */
export function calculatePairReference(
  workload: ReturnType<typeof createPairWorkload>,
  maximumRadius: number
) {
  const {positions, marks, radiusCount} = workload;
  const pairCounts = new Array<number>(radiusCount).fill(0);
  const cumulative = new Array<number>(radiusCount).fill(0);
  const shellWeights = new Array<number>(radiusCount).fill(0);
  const products = new Array<number>(radiusCount).fill(0);
  const mean = marks.reduce((sum, value) => sum + value, 0) / marks.length;
  for (let first = 0; first < marks.length; first++) {
    for (let second = 0; second < marks.length; second++) {
      if (first === second) continue;
      const horizontal = Math.abs(positions[first * 2] - positions[second * 2]);
      const vertical = Math.abs(positions[first * 2 + 1] - positions[second * 2 + 1]);
      const distance = Math.hypot(horizontal, vertical);
      const weight = 4096 / ((64 - horizontal) * (64 - vertical));
      for (let radius = 0; radius < radiusCount; radius++) {
        const upper = (maximumRadius * (radius + 1)) / radiusCount;
        const lower = (maximumRadius * radius) / radiusCount;
        if (distance > upper) continue;
        pairCounts[radius]++;
        cumulative[radius] += weight;
        if (distance > lower || radius === 0) {
          shellWeights[radius] += weight;
          products[radius] += weight * marks[first] * marks[second];
        }
      }
    }
  }
  const scale = 4096 / marks.length ** 2;
  const statistics = cumulative.flatMap((weight, radius) => {
    const upper = (maximumRadius * (radius + 1)) / radiusCount;
    const lower = (maximumRadius * radius) / radiusCount;
    const estimate = weight * scale;
    return [
      estimate,
      Math.sqrt(estimate / Math.PI),
      shellWeights[radius]
        ? (shellWeights[radius] * scale) / (Math.PI * (upper ** 2 - lower ** 2))
        : NaN,
      shellWeights[radius] ? products[radius] / shellWeights[radius] / mean ** 2 : NaN
    ];
  });
  return {statistics, pairCounts, status: new Array<number>(radiusCount).fill(0)};
}

export function createWatershedWorkload(width: number, seed = BACKLOG_BENCHMARK_SEED) {
  const random = createRandom(seed);
  const values = Float32Array.from({length: width * width}, () => Math.floor(random() * 32));
  const markers = new Uint32Array(values.length);
  markers[0] = 1;
  markers[width - 1] = 2;
  markers[values.length - 1] = 3;
  const updatedValues = Float32Array.from(values, value => 31 - value);
  return {width, values, markers, updatedValues};
}

/** Priority-queue reference; unlike the GPU implementation it never scans the full cell queue. */
export function calculateWatershedReference(
  workload: ReturnType<typeof createWatershedWorkload>,
  values: Float32Array
) {
  type Entry = {cell: number; cost: number; age: number};
  const {width, markers} = workload;
  const labels = Array.from(markers);
  const costs = new Float64Array(values.length).fill(Infinity);
  const visited = new Uint8Array(values.length);
  const heap: Entry[] = [];
  const precedes = (left: Entry, right: Entry) =>
    left.cost < right.cost || (left.cost === right.cost && left.age < right.age);
  const push = (entry: Entry) => {
    let child = heap.length;
    heap.push(entry);
    while (child > 0) {
      const parent = Math.floor((child - 1) / 2);
      if (!precedes(entry, heap[parent])) break;
      heap[child] = heap[parent];
      child = parent;
    }
    heap[child] = entry;
  };
  const pop = (): Entry => {
    const first = heap[0];
    const last = heap.pop()!;
    if (heap.length) {
      let parent = 0;
      while (parent * 2 + 1 < heap.length) {
        let child = parent * 2 + 1;
        if (child + 1 < heap.length && precedes(heap[child + 1], heap[child])) child++;
        if (!precedes(heap[child], last)) break;
        heap[parent] = heap[child];
        parent = child;
      }
      heap[parent] = last;
    }
    return first;
  };
  let age = 0;
  markers.forEach((marker, cell) => {
    if (marker) {
      costs[cell] = values[cell];
      push({cell, cost: values[cell], age: age++});
    }
  });
  while (heap.length) {
    const {cell, cost} = pop();
    if (visited[cell] || cost !== costs[cell]) continue;
    visited[cell] = 1;
    const column = cell % width;
    const row = Math.floor(cell / width);
    // Row-major four-neighbor discovery is part of the FIFO tie contract.
    for (const [horizontal, vertical] of [
      [0, -1],
      [-1, 0],
      [1, 0],
      [0, 1]
    ]) {
      const nextColumn = column + horizontal;
      const nextRow = row + vertical;
      if (nextColumn < 0 || nextColumn >= width || nextRow < 0 || nextRow >= width) continue;
      const neighbor = nextRow * width + nextColumn;
      const candidate = Math.max(cost, values[neighbor]);
      if (visited[neighbor] || markers[neighbor] || candidate >= costs[neighbor]) continue;
      costs[neighbor] = candidate;
      labels[neighbor] = labels[cell];
      push({cell: neighbor, cost: candidate, age: age++});
    }
  }
  return {labels, converged: [1]};
}

export function createNetworkWorkload(count: number, seed = BACKLOG_BENCHMARK_SEED) {
  const random = createRandom(seed);
  const componentSize = 8;
  const offsets = [0];
  const neighbors: number[] = [];
  for (let vertex = 0; vertex < count; vertex++) {
    const base = Math.floor(vertex / componentSize) * componentSize;
    neighbors.push(base + ((vertex + 1) % componentSize));
    neighbors.push(base + Math.floor(random() * componentSize));
    if (vertex % componentSize === componentSize - 1 && vertex + 1 < count)
      neighbors.push(vertex + 1);
    // Last self edge becomes the return edge in the update, closing all components into a cycle.
    if (vertex === count - 1) neighbors.push(vertex);
    offsets.push(neighbors.length);
  }
  const updatedNeighbors = Uint32Array.from(neighbors);
  updatedNeighbors[updatedNeighbors.length - 1] = 0;
  return {
    offsets: Uint32Array.from(offsets),
    neighbors: Uint32Array.from(neighbors),
    updatedNeighbors
  };
}

/** Tarjan SCC reference, independent of the GPU all-source reachability matrix. */
export function calculateNetworkReference(offsets: Uint32Array, neighbors: Uint32Array) {
  const count = offsets.length - 1;
  const discovery = new Int32Array(count).fill(-1);
  const low = new Int32Array(count);
  const active = new Uint8Array(count);
  const stack: number[] = [];
  const labels = new Array<number>(count);
  let clock = 0;
  function visit(vertex: number) {
    discovery[vertex] = low[vertex] = clock++;
    active[vertex] = 1;
    stack.push(vertex);
    for (let edge = offsets[vertex]; edge < offsets[vertex + 1]; edge++) {
      const neighbor = neighbors[edge];
      if (discovery[neighbor] === -1) {
        visit(neighbor);
        low[vertex] = Math.min(low[vertex], low[neighbor]);
      } else if (active[neighbor]) low[vertex] = Math.min(low[vertex], discovery[neighbor]);
    }
    if (low[vertex] !== discovery[vertex]) return;
    const members: number[] = [];
    let member: number;
    do {
      member = stack.pop()!;
      active[member] = 0;
      members.push(member);
    } while (member !== vertex);
    const label = Math.min(...members);
    for (const member of members) labels[member] = label;
  }
  for (let vertex = 0; vertex < count; vertex++) if (discovery[vertex] === -1) visit(vertex);
  return {labels};
}
