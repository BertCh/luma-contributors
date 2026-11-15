// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  computeNetworkRoutingDistances,
  computeNetworkRoutingPredecessors,
  validateNetworkRoutingArrays,
  type CPUNetworkRoutingInput
} from '../../analysis-core/operations/cpu-network-routing';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUKShortestPathsOperation,
  type KShortestPathsOperationProps
} from './gpu-k-shortest-paths-operation';

export type CPUKShortestPathsInput = {
  topology: CPUNetworkRoutingInput['topology'];
  sources: Uint32Array;
  targets: Uint32Array;
};
export type CPUKShortestPathsOptions = {
  routeCount: number;
  maximumPathLength?: number;
  candidateCapacity?: number;
};
/** Same padded layouts, per-query counts and batch status scalars as GPUKShortestPaths. */
export type CPUKShortestPathsResult = {
  vertices: Uint32Array;
  edges: Uint32Array;
  lengths: Uint32Array;
  costs: Float32Array;
  counts: Uint32Array;
  converged: Uint32Array;
  iterationLimitReached: Uint32Array;
  candidateOverflow: Uint32Array;
  iterationCount: Uint32Array;
};

type Route = {vertices: number[]; edges: number[]; cost: number};

function compareRoutes(first: Route, second: Route): number {
  if (first.cost !== second.cost) return first.cost - second.cost;
  for (let slot = 0; slot < Math.min(first.edges.length, second.edges.length); slot++) {
    if (first.edges[slot] !== second.edges[slot]) return first.edges[slot] - second.edges[slot];
  }
  return first.edges.length - second.edges.length;
}

/**
 * Yen's algorithm using the shared float64 Dijkstra and acyclic predecessor helpers.
 *
 * What differs from the GPU contributor:
 * - Float64 costs, rounded once on output, can change near-tie membership and overflow behavior.
 * - GPU node, route, command, device and shortest-search round limits are dropped. Every shortest
 *   search runs to convergence. Path length and candidate budgets retain the same output/status meaning.
 * - Equal-cost paths use the same predecessor policy but float64 tightness, so ties may differ.
 *
 * @throws Error naming topology, sources or targets for malformed packed arrays.
 */
export function computeKShortestPathsOnCPU(
  input: CPUKShortestPathsInput,
  options: CPUKShortestPathsOptions
): CPUKShortestPathsResult {
  const id = 'computeKShortestPathsOnCPU';
  const nodeCount = validateNetworkRoutingArrays(input.topology);
  const {offsets, neighbors, weights} = input.topology;
  for (let row = 0; row < offsets.length; row++) {
    if (offsets[row] > neighbors.length || (row > 0 && offsets[row] < offsets[row - 1])) {
      throw new Error('topology.offsets must be nondecreasing and within neighbors.length');
    }
  }
  validateCapacity(id, 'sources.length', input.sources.length);
  validateViewLength(id, 'targets', input.targets, input.sources.length);
  const pathLength = options.maximumPathLength ?? 64;
  const candidateCapacity = options.candidateCapacity ?? 256;
  validateCapacity(id, 'routeCount', options.routeCount);
  validateCapacity(id, 'maximumPathLength', pathLength);
  validateCapacity(id, 'candidateCapacity', candidateCapacity);
  const routeRows = input.sources.length * options.routeCount;
  validateCapacity(id, 'queryCount * routeCount * maximumPathLength', routeRows * pathLength);
  const result: CPUKShortestPathsResult = {
    vertices: new Uint32Array(routeRows * pathLength).fill(0xffffffff),
    edges: new Uint32Array(routeRows * pathLength).fill(0xffffffff),
    lengths: new Uint32Array(routeRows),
    costs: new Float32Array(routeRows).fill(Infinity),
    counts: new Uint32Array(input.sources.length),
    converged: Uint32Array.of(1),
    iterationLimitReached: Uint32Array.of(0),
    candidateOverflow: Uint32Array.of(0),
    iterationCount: Uint32Array.of(0)
  };
  for (let query = 0; query < input.sources.length; query++) {
    const source = input.sources[query];
    const destination = input.targets[query];
    if (source >= nodeCount || destination >= nodeCount) continue;
    const accepted: Route[] = [];
    const candidates: Route[] = [];
    for (let round = 0; round < options.routeCount; round++) {
      result.iterationCount[0] = Math.max(result.iterationCount[0], round + 1);
      const previous = accepted[round - 1];
      const spurCount = previous ? previous.vertices.length - 1 : 1;
      let stopped = false;
      for (let spur = 0; spur < spurCount; spur++) {
        const rootVertices = previous?.vertices.slice(0, spur) ?? [];
        const rootEdges = previous?.edges.slice(0, spur) ?? [];
        const spurSource = previous?.vertices[spur] ?? source;
        const bannedVertices = new Set(rootVertices);
        const bannedEdges = new Set<number>();
        for (const route of accepted) {
          if (
            route.edges.length > spur &&
            rootEdges.every((edge, slot) => route.edges[slot] === edge)
          )
            bannedEdges.add(route.edges[spur]);
        }
        const filteredWeights = weights.slice();
        for (let vertex = 0; vertex < nodeCount; vertex++) {
          for (let edge = offsets[vertex]; edge < offsets[vertex + 1]; edge++) {
            if (
              bannedVertices.has(vertex) ||
              bannedVertices.has(neighbors[edge]) ||
              bannedEdges.has(edge)
            )
              filteredWeights[edge] = -1;
          }
        }
        const search: CPUNetworkRoutingInput = {
          topology: {offsets, neighbors, weights: filteredWeights},
          startNodes: Uint32Array.of(spurSource)
        };
        const {distances} = computeNetworkRoutingDistances(id, search);
        if (!Number.isFinite(distances[destination])) continue;
        const predecessors = computeNetworkRoutingPredecessors(search, distances);
        const suffixVertices: number[] = [];
        const suffixEdges: number[] = [];
        let vertex = destination;
        while (true) {
          if (suffixVertices.length + spur >= pathLength) {
            stopped = true;
            break;
          }
          suffixVertices.push(vertex);
          if (vertex === spurSource) break;
          const parent = predecessors[vertex];
          if (parent === 0xffffffff) {
            stopped = true;
            break;
          }
          let chosen = -1;
          for (let edge = offsets[parent]; edge < offsets[parent + 1]; edge++) {
            if (
              neighbors[edge] === vertex &&
              filteredWeights[edge] >= 0 &&
              distances[parent] + filteredWeights[edge] === distances[vertex]
            ) {
              chosen = edge;
              break;
            }
          }
          if (chosen < 0) {
            stopped = true;
            break;
          }
          suffixEdges.push(chosen);
          vertex = parent;
        }
        if (stopped) {
          result.iterationLimitReached[0] = 1;
          break;
        }
        const route: Route = {
          vertices: [...rootVertices, ...suffixVertices.reverse()],
          edges: [...rootEdges, ...suffixEdges.reverse()],
          cost: rootEdges.reduce((cost, edge) => cost + weights[edge], 0) + distances[destination]
        };
        const key = route.edges.join(',');
        if ([...accepted, ...candidates].some(other => other.edges.join(',') === key)) continue;
        if (candidates.length === candidateCapacity) {
          result.candidateOverflow[0] = 1;
          stopped = true;
          break;
        }
        candidates.push(route);
      }
      if (stopped) {
        result.converged[0] = 0;
        break;
      }
      candidates.sort(compareRoutes);
      const route = candidates.shift();
      if (!route) break;
      accepted.push(route);
      const row = query * options.routeCount + round;
      result.vertices.set(route.vertices, row * pathLength);
      result.edges.set(route.edges, row * pathLength);
      result.lengths[row] = route.vertices.length;
      result.costs[row] = route.cost;
      result.counts[query]++;
    }
  }
  return result;
}

/** CPU twin; prefer it for small networks to avoid the GPU submission overhead of Yen rounds. */
export class CPUKShortestPathsOperation extends GPUCPUOperation {
  constructor(props: KShortestPathsOperationProps) {
    const id = props.id ?? 'k-shortest-paths';
    validateCapacity(id, 'queryCount', props.queryCount);
    validateCapacity(id, 'routeCount', props.routeCount);
    validateCapacity(id, 'maximumPathLength', props.maximumPathLength ?? 64);
    validateCapacity(id, 'candidateCapacity', props.candidateCapacity ?? 256);
    const operation = new GPUKShortestPathsOperation(props);
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
        const input: CPUKShortestPathsInput = {
          topology: {
            offsets: getCPUOperationUint32Rows(id, inputs, 'offsets'),
            neighbors: getCPUOperationUint32Rows(id, inputs, 'neighbors'),
            weights: getCPUOperationFloat32Rows(id, inputs, 'weights')
          },
          sources: getCPUOperationUint32Rows(id, inputs, 'sources'),
          targets: getCPUOperationUint32Rows(id, inputs, 'targets')
        };
        if (props.topology.nodePositions !== undefined) {
          validateViewLength(
            id,
            'topology.nodePositions',
            getCPUOperationFloat32Rows(id, inputs, 'nodePositions', 'float32x2'),
            (input.topology.offsets.length - 1) * 2
          );
        }
        validateViewLength(id, 'sources', input.sources, props.queryCount);
        const result = computeKShortestPathsOnCPU(input, props);
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUKShortestPathsResult]
          ])
        );
      }
    });
  }
}
