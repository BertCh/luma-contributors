// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {computeBudgetedNetworkDistances} from '../network-accessibility/network-budgeted-search';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUNetworkCentralityOperation,
  type NetworkCentralityOperationProps
} from './gpu-network-centrality-operation';

/** Packed weighted CSR, with the same clamped offset semantics as the contributor. */
type CentralityTopology = {offsets: Uint32Array; neighbors: Uint32Array; weights: Float32Array};

export type CPUNetworkCentralityInput = {
  topology: CentralityTopology;
  /** Exact transpose, including repeated edges and their weights. */
  reverseTopology?: CentralityTopology;
  pivotNodes: Uint32Array;
  costLimit?: Float32Array;
};

export type CPUNetworkCentralityOptions = {
  maximumIterations?: number;
  localIterations?: number;
  betweennessScale?: number;
  sampling?: 'exact' | 'sampled';
  /** False for closeness-only calls: zero-weight cycles then do not affect convergence. */
  computeBetweenness?: boolean;
};

export type CPUNetworkCentralityResult = {
  betweenness: Float32Array;
  edgeBetweenness: Float32Array;
  closeness: Float32Array;
  harmonicCloseness: Float32Array;
  reachedCounts: Uint32Array;
  approximation: Uint32Array;
  converged: Uint32Array;
  zeroWeightCycle: Uint32Array;
  iterationLimitReached: Uint32Array;
};

/**
 * Computes outward radius-bounded centrality using Dijkstra and topological Brandes frontiers.
 * Repeated pivots and parallel slots count separately; endpoints and self-loops are excluded.
 * Negative/NaN weights and invalid neighbors are impassable. A negative radius rejects even the
 * pivot; a NaN radius imposes no bound, matching the GPU comparison. Offset bounds are
 * clamped to the edge capacity; a reversed interval is empty, as on the GPU.
 *
 * What differs from the GPU contributor:
 * - Distances, tight-edge tests and dependencies use float64, rounded only at output. Small
 *   positive edges absorbed by float32 can therefore participate in shortest paths on CPU.
 *   Compare well-conditioned centralities with relative tolerance 2e-5.
 * - Explicit maximumIterations bounds search rounds (localIterations defaults to 16, with
 *   256-entry local queues), and independently bounds each dependency phase to one edge per
 *   round. Omitted maximumIterations runs to completion. GPU scheduling can change partial
 *   results for competing discoveries; command-node and lane-storage limits are dropped.
 * - Path counts use scaled float64 mantissas and exponents to avoid overflow; sums are independent
 *   of laneCount. Zero-cycle partial node and edge dependencies retain the GPU frontier convention.
 * - Float32 distance/sum overflow is avoided internally. Subnormals are retained; undefined GPU
 *   arithmetic at extreme magnitudes can differ.
 *
 * @throws If topology arrays, reverseTopology node count or costLimit length are inconsistent.
 */
export function computeNetworkCentralityOnCPU(
  input: CPUNetworkCentralityInput,
  options: CPUNetworkCentralityOptions = {}
): CPUNetworkCentralityResult {
  options = snapshotConstructionFields(options, [
    'maximumIterations',
    'localIterations',
    'betweennessScale',
    'sampling',
    'computeBetweenness'
  ]);
  if (options.maximumIterations !== undefined)
    validateCapacity(
      'computeNetworkCentralityOnCPU',
      'maximumIterations',
      options.maximumIterations
    );
  const localIterations = options.localIterations ?? 16;
  validateCapacity('computeNetworkCentralityOnCPU', 'localIterations', localIterations);
  const maximumRounds = options.maximumIterations ?? Infinity;
  const {topology, pivotNodes, costLimit} = input;
  validateRequirement(
    'computeNetworkCentralityOnCPU',
    options.computeBetweenness === false || Boolean(input.reverseTopology),
    'reverseTopology is required for betweenness; supply the transpose or set computeBetweenness: false'
  );
  const reverseTopology = input.reverseTopology ?? topology;
  const nodeCount = topology.offsets.length - 1;
  for (const [name, network] of [
    ['topology', topology],
    ['reverseTopology', reverseTopology]
  ] as const) {
    if (network.offsets.length !== nodeCount + 1 || nodeCount < 1)
      throw new Error(`${name}.offsets must contain nodeCount + 1 rows with nodeCount >= 1`);
    if (network.weights.length !== network.neighbors.length)
      throw new Error(`${name}.weights length must equal ${name}.neighbors length`);
  }
  if (costLimit && costLimit.length !== 1)
    throw new Error('costLimit must contain exactly one row');
  validateRequirement(
    'computeNetworkCentralityOnCPU',
    options.sampling === undefined ||
      options.sampling === 'exact' ||
      options.sampling === 'sampled',
    "sampling must be 'exact' or 'sampled'"
  );
  const scale = options.betweennessScale ?? 1;
  validateFiniteNumber('computeNetworkCentralityOnCPU', 'betweennessScale', scale);
  const nodeTotals = new Float64Array(nodeCount);
  const edgeTotals = new Float64Array(topology.neighbors.length);
  const closeness = new Float32Array(pivotNodes.length);
  const harmonicCloseness = new Float32Array(pivotNodes.length);
  const reachedCounts = new Uint32Array(pivotNodes.length);
  let anyCycle = false;
  let anyLimitReached = false;
  for (let pivotRow = 0; pivotRow < pivotNodes.length; pivotRow++) {
    const pivot = pivotNodes[pivotRow];
    if (pivot >= nodeCount || (costLimit && 0 > costLimit[0])) continue;
    const distances = new Float64Array(nodeCount).fill(Infinity);
    if (options.maximumIterations !== undefined) {
      const search = computeBudgetedNetworkDistances(
        {topology, startNodes: Uint32Array.of(pivot), costLimit},
        options.maximumIterations,
        localIterations
      );
      distances.set(search.distances);
      anyLimitReached ||= !search.converged;
    } else {
      distances[pivot] = 0;
      const heap: [number, number][] = [];
      pushDistance(heap, [0, pivot]);
      while (heap.length) {
        const [distance, source] = popDistance(heap);
        if (distance !== distances[source]) continue;
        for (
          let edge = Math.min(topology.offsets[source], topology.neighbors.length);
          edge < Math.min(topology.offsets[source + 1], topology.neighbors.length);
          edge++
        ) {
          const target = topology.neighbors[edge];
          const weight = topology.weights[edge];
          if (target >= nodeCount || !(weight >= 0)) continue;
          const candidate = distance + weight;
          if (costLimit && candidate > costLimit[0]) continue;
          if (candidate < distances[target]) {
            distances[target] = candidate;
            pushDistance(heap, [candidate, target]);
          }
        }
      }
    }
    let costSum = 0;
    let harmonicSum = 0;
    for (const distance of distances) {
      if (Number.isFinite(distance)) {
        reachedCounts[pivotRow]++;
        costSum += distance;
        if (distance > 0) harmonicSum += 1 / distance;
      }
    }
    closeness[pivotRow] = costSum > 0 ? (reachedCounts[pivotRow] - 1) / costSum : 0;
    harmonicCloseness[pivotRow] = harmonicSum;
    if (options.computeBetweenness === false) continue;

    const successors: {target: number; edge: number}[][] = Array.from(
      {length: nodeCount},
      () => []
    );
    const predecessors: number[][] = Array.from({length: nodeCount}, () => []);
    const isTight = (source: number, target: number, weight: number) =>
      source !== target &&
      target < nodeCount &&
      source < nodeCount &&
      weight >= 0 &&
      Number.isFinite(distances[target]) &&
      distances[source] + weight === distances[target] &&
      (distances[source] < distances[target] || weight === 0);
    for (let source = 0; source < nodeCount; source++) {
      for (
        let edge = Math.min(topology.offsets[source], topology.neighbors.length);
        edge < Math.min(topology.offsets[source + 1], topology.neighbors.length);
        edge++
      ) {
        const target = topology.neighbors[edge];
        if (isTight(source, target, topology.weights[edge]))
          successors[source].push({target, edge});
      }
      for (
        let edge = Math.min(reverseTopology.offsets[source], reverseTopology.neighbors.length);
        edge < Math.min(reverseTopology.offsets[source + 1], reverseTopology.neighbors.length);
        edge++
      ) {
        const predecessor = reverseTopology.neighbors[edge];
        if (isTight(predecessor, source, reverseTopology.weights[edge]))
          predecessors[source].push(predecessor);
      }
    }
    const remaining = predecessors.map(rows => rows.length);
    const mantissas = new Float64Array(nodeCount);
    const exponents = new Float64Array(nodeCount);
    let queue: number[] = [];
    for (let node = 0; node < nodeCount; node++) {
      if (remaining[node] === 0 && Number.isFinite(distances[node])) {
        queue.push(node);
        mantissas[node] = 1;
      }
    }
    for (let round = 0; round < maximumRounds && queue.length; round++) {
      const next: number[] = [];
      for (const node of queue) {
        if (predecessors[node].length) {
          let largestExponent = -Infinity;
          for (const predecessor of predecessors[node])
            largestExponent = Math.max(largestExponent, exponents[predecessor]);
          let sum = 0;
          for (const predecessor of predecessors[node])
            sum += mantissas[predecessor] * 2 ** (exponents[predecessor] - largestExponent);
          const shift = Math.floor(Math.log2(sum));
          mantissas[node] = sum / 2 ** shift;
          exponents[node] = largestExponent + shift;
        }
        for (const {target} of successors[node]) if (--remaining[target] === 0) next.push(target);
      }
      queue = next;
    }
    const sigmaTruncated = queue.length > 0;
    anyLimitReached ||= sigmaTruncated;
    anyCycle ||=
      !sigmaTruncated &&
      remaining.some((count, node) => count > 0 && Number.isFinite(distances[node]));
    // The GPU's delta frontier starts at every tight sink, including sinks after a cycle.
    const pending = successors.map(rows => rows.length);
    const delta = new Float64Array(nodeCount);
    let sinks: number[] = [];
    for (let node = 0; node < nodeCount; node++)
      if (pending[node] === 0 && Number.isFinite(distances[node])) sinks.push(node);
    const getRatio = (source: number, target: number) =>
      mantissas[target] > 0
        ? (mantissas[source] / mantissas[target]) * 2 ** (exponents[source] - exponents[target])
        : 0;
    for (let round = 0; round < maximumRounds && sinks.length; round++) {
      const next: number[] = [];
      for (const node of sinks) {
        for (const {target} of successors[node])
          delta[node] += getRatio(node, target) * (1 + delta[target]);
        for (const predecessor of predecessors[node])
          if (--pending[predecessor] === 0) next.push(predecessor);
      }
      sinks = next;
    }
    anyLimitReached ||= sinks.length > 0;
    for (let node = 0; node < nodeCount; node++) {
      if (node !== pivot) nodeTotals[node] += scale * delta[node];
      for (const {target, edge} of successors[node])
        edgeTotals[edge] += scale * getRatio(node, target) * (1 + delta[target]);
    }
  }
  return {
    betweenness: Float32Array.from(nodeTotals),
    edgeBetweenness: Float32Array.from(edgeTotals),
    closeness,
    harmonicCloseness,
    reachedCounts,
    approximation: Uint32Array.of(Number(options.sampling === 'sampled')),
    converged: Uint32Array.of(Number(!anyCycle && !anyLimitReached)),
    zeroWeightCycle: Uint32Array.of(Number(anyCycle)),
    iterationLimitReached: Uint32Array.of(Number(anyLimitReached))
  };
}

function pushDistance(heap: [number, number][], entry: [number, number]): void {
  let cursor = heap.length;
  heap.push(entry);
  while (cursor > 0) {
    const parent = Math.floor((cursor - 1) / 2);
    if (heap[parent][0] <= entry[0]) break;
    heap[cursor] = heap[parent];
    cursor = parent;
  }
  heap[cursor] = entry;
}

function popDistance(heap: [number, number][]): [number, number] {
  const result = heap[0];
  const last = heap.pop()!;
  if (heap.length) {
    let cursor = 0;
    while (cursor * 2 + 1 < heap.length) {
      let child = cursor * 2 + 1;
      if (child + 1 < heap.length && heap[child + 1][0] < heap[child][0]) child++;
      if (last[0] <= heap[child][0]) break;
      heap[cursor] = heap[child];
      cursor = child;
    }
    heap[cursor] = last;
  }
  return result;
}

/** CPU centrality twin retaining explicit search/dependency budgets without GPU storage caps. */
export class CPUNetworkCentralityOperation extends GPUCPUOperation {
  constructor(props: NetworkCentralityOperationProps) {
    const id = props.id ?? 'network-centrality';
    validateCapacity(id, 'nodeCount', props.nodeCount);
    validateCapacity(id, 'edgeCount', props.edgeCount, 0);
    if (props.reverseEdgeCount !== undefined)
      validateCapacity(id, 'reverseEdgeCount', props.reverseEdgeCount, 0);
    validateCapacity(id, 'pivotCount', props.pivotCount);
    validateFiniteNumber(id, 'betweennessScale', props.betweennessScale ?? 1);
    validateRequirement(
      id,
      Object.entries(props.output).some(
        ([port, value]) => port !== 'status' && value !== undefined
      ),
      'output must request a centrality value'
    );
    validateRequirement(
      id,
      props.output.status?.zeroWeightCycle === undefined ||
        props.output.betweenness !== undefined ||
        props.output.edgeBetweenness !== undefined,
      'output.status.zeroWeightCycle needs a betweenness output'
    );
    validateRequirement(
      id,
      props.sampling === undefined || props.sampling === 'exact' || props.sampling === 'sampled',
      "sampling must be 'exact' or 'sampled'"
    );
    const operation = new GPUNetworkCentralityOperation(props);
    if (props.maximumIterations !== undefined)
      validateCapacity(id, 'maximumIterations', props.maximumIterations);
    validateCapacity(id, 'localIterations', props.localIterations ?? 16);
    const options = Object.freeze({
      maximumIterations: props.maximumIterations,
      localIterations: props.localIterations,
      betweennessScale: props.betweennessScale,
      sampling: props.sampling,
      computeBetweenness:
        props.output.betweenness !== undefined || props.output.edgeBetweenness !== undefined
    });
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
        const getTopology = (
          prefix: 'topology' | 'reverseTopology',
          edgeCount: number
        ): CentralityTopology => {
          const topology = {
            offsets: getCPUOperationUint32Rows(id, inputs, `${prefix}.offsets`),
            neighbors: getCPUOperationUint32Rows(id, inputs, `${prefix}.neighbors`),
            weights: getCPUOperationFloat32Rows(id, inputs, `${prefix}.weights`)
          };
          if (
            topology.offsets.length !== props.nodeCount + 1 ||
            topology.neighbors.length !== edgeCount
          )
            throw new Error(`${id} ${prefix} arrays must match nodeCount and edgeCount`);
          if (
            props[prefix]?.nodePositions !== undefined &&
            getCPUOperationFloat32Rows(id, inputs, `${prefix}.nodePositions`, 'float32x2')
              .length !==
              props.nodeCount * 2
          )
            throw new Error(`${id} ${prefix}.nodePositions length must equal nodeCount * 2`);
          return topology;
        };
        const pivotNodes = getCPUOperationUint32Rows(id, inputs, 'pivotNodes');
        if (pivotNodes.length !== props.pivotCount)
          throw new Error(`${id} pivotNodes length must equal pivotCount`);
        const result = computeNetworkCentralityOnCPU(
          {
            topology: getTopology('topology', props.edgeCount),
            reverseTopology: props.reverseTopology
              ? getTopology(
                  'reverseTopology',
                  props.reverseEdgeCount ??
                    getCPUOperationUint32Rows(id, inputs, 'reverseTopology.neighbors').length
                )
              : undefined,
            pivotNodes,
            costLimit:
              props.costLimit === undefined
                ? undefined
                : getCPUOperationFloat32Rows(id, inputs, 'costLimit')
          },
          options
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUNetworkCentralityResult]
          ])
        );
      }
    });
  }
}
