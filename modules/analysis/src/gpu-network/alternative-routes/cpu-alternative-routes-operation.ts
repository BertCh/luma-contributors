// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import type {CPUNetworkReachabilityInput} from '../network-reachability/cpu-network-reachability-operation';
import {validateNetworkRoutingArrays} from '../../analysis-core/operations/cpu-network-routing';
import {
  GPUAlternativeRoutesOperation,
  type AlternativeRoutesOperationProps
} from './gpu-alternative-routes-operation';

/** Queries on a packed directed network; parallel CSR edges remain distinct. */
export type CPUAlternativeRoutesInput = {
  topology: CPUNetworkReachabilityInput['topology'];
  sources: Uint32Array;
  targets: Uint32Array;
};
/** Number of routes retained per query. Path padding is the topology node count. */
export type CPUAlternativeRoutesOptions = {routeCount: number};
/** Padded path matrices, per-route lengths/costs, and per-query counts/certification. */
export type CPUAlternativeRoutesResult = {
  vertices: Uint32Array;
  edges: Uint32Array;
  lengths: Uint32Array;
  costs: Float32Array;
  counts: Uint32Array;
  complete: Uint32Array;
};

type RoutePrefix = {cost: number; vertices: number[]; edges: number[]};

/** Orders equal-cost prefixes by lexicographic CSR edge IDs, including the shorter-prefix rule. */
function compareRoutePrefixes(first: RoutePrefix, second: RoutePrefix): number {
  if (first.cost !== second.cost) return first.cost < second.cost ? -1 : 1;
  for (let slot = 0; slot < Math.min(first.edges.length, second.edges.length); slot++) {
    if (first.edges[slot] !== second.edges[slot]) return first.edges[slot] - second.edges[slot];
  }
  return first.edges.length - second.edges.length;
}

/**
 * Finds the exact top-k simple edge paths with an unbounded binary min heap. Repeated vertices
 * are prohibited, parallel edges are distinct, and equal costs use lexicographic CSR edge IDs.
 *
 * What differs from the GPU contributor:
 * - Float64 path costs are rounded once on output; near ties, subnormals and paths whose float32
 *   sums overflow can change route order and membership.
 * - Node, route, expansion, frontier and device limits imposed for GPU execution are dropped.
 *   The search certifies the requested top-k or exhausts all paths; complete is always 1.
 * - This exact search remains exponential in the worst case and can require substantial CPU memory.
 * - Invalid endpoints are empty certified queries; invalid neighbors/weights are impassable,
 *   and offset values retain the GPU's per-row clamping interpretation.
 *
 * @throws Error naming topology, sources or targets for inconsistent array lengths.
 */
export function computeAlternativeRoutesOnCPU(
  input: CPUAlternativeRoutesInput,
  options: CPUAlternativeRoutesOptions
): CPUAlternativeRoutesResult {
  const nodeCount = validateNetworkRoutingArrays(input.topology);
  if (!input.sources.length) throw new Error('sources must contain at least one query');
  if (input.targets.length !== input.sources.length)
    throw new Error('targets length must equal sources length');
  validateCapacity('computeAlternativeRoutesOnCPU', 'routeCount', options.routeCount);
  const queryCount = input.sources.length;
  const routeCount = options.routeCount;
  validateCapacity(
    'computeAlternativeRoutesOnCPU',
    'queryCount * routeCount * nodeCount',
    queryCount * routeCount * nodeCount
  );
  const result: CPUAlternativeRoutesResult = {
    vertices: new Uint32Array(queryCount * routeCount * nodeCount).fill(0xffffffff),
    edges: new Uint32Array(queryCount * routeCount * nodeCount).fill(0xffffffff),
    lengths: new Uint32Array(queryCount * routeCount),
    costs: new Float32Array(queryCount * routeCount).fill(Infinity),
    counts: new Uint32Array(queryCount),
    complete: new Uint32Array(queryCount).fill(1)
  };
  const {offsets, neighbors, weights} = input.topology;
  for (let query = 0; query < queryCount; query++) {
    const source = input.sources[query];
    const destination = input.targets[query];
    if (source >= nodeCount || destination >= nodeCount) continue;
    const heap: RoutePrefix[] = [{cost: 0, vertices: [source], edges: []}];
    const pushPrefix = (prefix: RoutePrefix) => {
      let position = heap.length;
      heap.push(prefix);
      while (position > 0) {
        const parent = Math.floor((position - 1) / 2);
        if (compareRoutePrefixes(heap[parent], prefix) <= 0) break;
        heap[position] = heap[parent];
        position = parent;
      }
      heap[position] = prefix;
    };
    while (heap.length && result.counts[query] < routeCount) {
      const prefix = heap[0];
      const last = heap.pop()!;
      if (heap.length) {
        let position = 0;
        while (position * 2 + 1 < heap.length) {
          let child = position * 2 + 1;
          if (child + 1 < heap.length && compareRoutePrefixes(heap[child + 1], heap[child]) < 0)
            child++;
          if (compareRoutePrefixes(last, heap[child]) <= 0) break;
          heap[position] = heap[child];
          position = child;
        }
        heap[position] = last;
      }
      const vertex = prefix.vertices[prefix.vertices.length - 1];
      if (vertex === destination) {
        const route = query * routeCount + result.counts[query]++;
        result.lengths[route] = prefix.vertices.length;
        result.costs[route] = prefix.cost;
        result.vertices.set(prefix.vertices, route * nodeCount);
        result.edges.set(prefix.edges, route * nodeCount);
        continue;
      }
      for (
        let edge = Math.min(offsets[vertex], neighbors.length);
        edge < Math.min(offsets[vertex + 1], neighbors.length);
        edge++
      ) {
        const neighbor = neighbors[edge];
        const weight = weights[edge];
        if (
          neighbor >= nodeCount ||
          !(weight >= 0) ||
          !Number.isFinite(weight) ||
          prefix.vertices.includes(neighbor)
        )
          continue;
        pushPrefix({
          cost: prefix.cost + weight,
          vertices: [...prefix.vertices, neighbor],
          edges: [...prefix.edges, edge]
        });
      }
    }
  }
  return result;
}

/** Exact CPU twin; the GPU search budgets are ignored and output dimensions remain caller-owned. */
export class CPUAlternativeRoutesOperation extends GPUCPUOperation {
  constructor(props: AlternativeRoutesOperationProps) {
    const id = props.id ?? 'alternative-routes';
    validateCapacity(id, 'nodeCount', props.nodeCount);
    validateCapacity(id, 'queryCount', props.queryCount);
    validateCapacity(id, 'routeCount', props.routeCount);
    validateCapacity(
      id,
      'queryCount * routeCount * nodeCount',
      props.queryCount * props.routeCount * props.nodeCount
    );
    const operation = new GPUAlternativeRoutesOperation(props);
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
        const topology = {
          offsets: getCPUOperationUint32Rows(id, inputs, 'offsets'),
          neighbors: getCPUOperationUint32Rows(id, inputs, 'neighbors'),
          weights: getCPUOperationFloat32Rows(id, inputs, 'weights')
        };
        const sources = getCPUOperationUint32Rows(id, inputs, 'sources');
        const targets = getCPUOperationUint32Rows(id, inputs, 'targets');
        if (props.topology.nodePositions !== undefined) {
          const nodePositions = getCPUOperationFloat32Rows(
            id,
            inputs,
            'nodePositions',
            'float32x2'
          );
          if (nodePositions.length !== props.nodeCount * 2)
            throw new Error(`${id} topology.nodePositions must contain one float32x2 row per node`);
        }
        if (topology.offsets.length !== props.nodeCount + 1)
          throw new Error(`${id} topology.offsets length must equal nodeCount + 1`);
        if (sources.length !== props.queryCount)
          throw new Error(`${id} sources length must equal queryCount`);
        const result = computeAlternativeRoutesOnCPU(
          {topology, sources, targets},
          {routeCount: props.routeCount}
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUAlternativeRoutesResult]
          ])
        );
      }
    });
  }
}
