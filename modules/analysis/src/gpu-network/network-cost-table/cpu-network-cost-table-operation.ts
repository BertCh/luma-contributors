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
import {GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS} from '../network-reachability/gpu-network-reachability';
import {
  validateNetworkRoutingArrays,
  resolveNetworkRoutingCostPrecision,
  addNetworkRoutingCosts,
  type CPUNetworkRoutingCostPrecision
} from '../../analysis-core/operations/cpu-network-routing';
import {
  GPUNetworkCostTableOperation,
  snapshotNetworkCostTableOperationProps,
  type NetworkCostTableOperationProps
} from './gpu-network-cost-table-operation';

/** Largest node count whose nodeCount + 1 offset rows fit the uint32 graph shape limit. */
export const NETWORK_COST_TABLE_MAXIMUM_NODE_COUNT = 0xfffffffe;
/** Same default local-hop budget as the GPU table. CPU rounds are uncapped unless supplied. */
const DEFAULT_LOCAL_ITERATIONS = 16;
/** Mirrors the GPU workgroup-local frontier queue, which spills excess entries to the next round. */
const LOCAL_FRONTIER_CAPACITY = 256;

/** Packed CSR and the same one-word cutoff view as the GPU contributor. */
export type CPUNetworkCostTableInput = {
  topology: CPUNetworkReachabilityInput['topology'];
  costLimit: Float32Array;
  /** Default float32-per-edge, preserving inclusive table membership for GPU consumers. */
  costPrecision?: CPUNetworkRoutingCostPrecision;
};
/** Entry capacity, independent of the required count. */
export type CPUNetworkCostTableOptions = {
  capacity: number;
  maximumIterations?: number;
  localIterations?: number;
};
/** CSR outputs and one-word statuses. There is no separate count port in GPUNetworkCostTable. */
export type CPUNetworkCostTableResult = {
  /** Captured cutoff; bind this retained value as the matcher routeTable.costLimit. */
  builtCostLimit: Float32Array;
  offsets: Uint32Array;
  targets: Uint32Array;
  costs: Float32Array;
  overflow: Uint32Array;
  requiredCount: Uint32Array;
  converged: Uint32Array;
  iterationLimitReached: Uint32Array;
  complete: Uint32Array;
};

/**
 * Builds cutoff all-pairs CSR rows with ascending targets, including finite diagonal zeros.
 *
 * What differs from the GPU contributor:
 * - costPrecision defaults to float32-per-edge, rounding each addition before the inclusive
 *   cutoff to preserve membership for mixed CPU-table/GPU-matcher graphs. Explicit float64
 *   uses wider paths and can change membership. NaN costLimit disables the cutoff, as on GPU.
 * - An explicit round budget applies; omitting maximumIterations searches to completion.
 *   localIterations defaults to 16. CPU workgroups visit
 *   source rows independently in a deterministic order; GPU scheduling and lane packing can change
 *   partial incumbents and convergence near the budget boundary. Completed paths agree up to rounding.
 *   GPU lane, command-node and device limits are dropped. Capacity still truncates rows.
 * - Unused target/cost capacity is zero-initialized; the GPU does not write unused capacity.
 * - A finite float64 path outside the finite float32 output range is omitted from the CSR and
 *   clears complete, even if the search converged and final capacity fits. requiredCount counts
 *   representable entries; convergence and capacity overflow keep their independent meanings.
 *
 * @throws Error naming an inconsistent topology or costLimit argument.
 */
export function computeNetworkCostTableOnCPU(
  input: CPUNetworkCostTableInput,
  options: CPUNetworkCostTableOptions
): CPUNetworkCostTableResult {
  const requestedCostPrecision = input.costPrecision;
  const costPrecision = resolveNetworkRoutingCostPrecision(
    'computeNetworkCostTableOnCPU',
    requestedCostPrecision === undefined ? 'float32-per-edge' : requestedCostPrecision
  );
  validateCapacity('computeNetworkCostTableOnCPU', 'capacity', options.capacity);
  const maximumIterations = options.maximumIterations ?? Infinity;
  const localIterations = options.localIterations ?? DEFAULT_LOCAL_ITERATIONS;
  if (options.maximumIterations !== undefined) {
    validateCapacity('computeNetworkCostTableOnCPU', 'maximumIterations', maximumIterations, 1);
  }
  validateCapacity(
    'computeNetworkCostTableOnCPU',
    'localIterations',
    localIterations,
    1,
    GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS
  );
  const nodeCount = validateNetworkRoutingArrays(input.topology);
  if (input.costLimit.length !== 1) throw new Error('costLimit must contain exactly one row');
  const builtCostLimit = Float32Array.of(input.costLimit[0]);
  input = {topology: input.topology, costLimit: builtCostLimit, costPrecision};
  const offsets = new Uint32Array(nodeCount + 1);
  const targets = new Uint32Array(options.capacity);
  const costs = new Float32Array(options.capacity);
  let requiredCount = 0;
  let converged = true;
  let representable = true;
  for (let source = 0; source < nodeCount; source++) {
    offsets[source] = Math.min(requiredCount, options.capacity);
    const search = computeBudgetedCosts(input, source, maximumIterations, localIterations);
    const {distances} = search;
    converged &&= search.converged;
    for (let target = 0; target < nodeCount; target++) {
      if (!Number.isFinite(distances[target]) || distances[target] < 0) continue;
      const storedCost = Math.fround(distances[target]);
      if (!Number.isFinite(storedCost)) {
        representable = false;
        continue;
      }
      if (requiredCount < options.capacity) {
        targets[requiredCount] = target;
        costs[requiredCount] = storedCost;
      }
      requiredCount = Math.min(requiredCount + 1, 0xffffffff);
    }
  }
  offsets[nodeCount] = Math.min(requiredCount, options.capacity);
  return {
    builtCostLimit,
    offsets,
    targets,
    costs,
    requiredCount: Uint32Array.of(requiredCount),
    overflow: Uint32Array.of(Number(requiredCount > options.capacity)),
    converged: Uint32Array.of(Number(converged)),
    iterationLimitReached: Uint32Array.of(Number(!converged)),
    complete: Uint32Array.of(
      Number(converged && representable && requiredCount <= options.capacity)
    )
  };
}

/** Deterministic CPU scheduling of the bounded, 256-entry local frontier queues. */
function computeBudgetedCosts(
  input: CPUNetworkCostTableInput,
  source: number,
  maximumIterations: number,
  localIterations: number
): {distances: Float64Array; converged: boolean} {
  const {offsets, neighbors, weights} = input.topology;
  const distances = new Float64Array(offsets.length - 1).fill(Infinity);
  let frontier: number[] = [];
  if (!(0 > input.costLimit[0])) {
    distances[source] = 0;
    frontier.push(source);
  }
  for (let round = 0; round < maximumIterations && frontier.length; round++) {
    const nextFrontier = new Set<number>();
    for (let chunk = 0; chunk < frontier.length; chunk += LOCAL_FRONTIER_CAPACITY) {
      let localFrontier = frontier.slice(chunk, chunk + LOCAL_FRONTIER_CAPACITY);
      for (let hop = 0; hop < localIterations && localFrontier.length; hop++) {
        const nextLocalFrontier: number[] = [];
        const sources = localFrontier.map(node => ({node, sourceCost: distances[node]}));
        for (const {node, sourceCost} of sources) {
          for (
            let edge = Math.min(offsets[node], neighbors.length);
            edge < Math.min(offsets[node + 1], neighbors.length);
            edge++
          ) {
            const target = neighbors[edge];
            const candidate = addNetworkRoutingCosts(
              sourceCost,
              weights[edge],
              input.costPrecision!
            );
            if (
              target < distances.length &&
              weights[edge] >= 0 &&
              Number.isFinite(candidate) &&
              !(candidate > input.costLimit[0]) &&
              candidate < distances[target]
            ) {
              distances[target] = candidate;
              if (hop + 1 < localIterations && nextLocalFrontier.length < LOCAL_FRONTIER_CAPACITY) {
                nextLocalFrontier.push(target);
              } else {
                nextFrontier.add(target);
              }
            }
          }
        }
        localFrontier = nextLocalFrontier;
      }
    }
    frontier = Array.from(nextFrontier);
  }
  return {distances, converged: frontier.length === 0};
}

/** CPU twin, retaining the CSR capacity and iteration contracts, without GPU device limits. */
export class CPUNetworkCostTableOperation extends GPUCPUOperation {
  constructor(props: NetworkCostTableOperationProps) {
    props = snapshotNetworkCostTableOperationProps(props);
    const id = props.id ?? 'network-cost-table';
    const costPrecision = resolveNetworkRoutingCostPrecision(
      id,
      props.costPrecision === undefined ? 'float32-per-edge' : props.costPrecision
    );
    validateCapacity(id, 'nodeCount', props.nodeCount, 1, NETWORK_COST_TABLE_MAXIMUM_NODE_COUNT);
    validateCapacity(id, 'capacity', props.capacity);
    // The graph operation validates iteration props once for both execution forms.
    const operation = new GPUNetworkCostTableOperation({...props, costPrecision: undefined});
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
        const result = computeNetworkCostTableOnCPU(
          {topology, costLimit: getCPUOperationFloat32Rows(id, inputs, 'costLimit'), costPrecision},
          {
            capacity: props.capacity,
            maximumIterations: props.maximumIterations,
            localIterations: props.localIterations
          }
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[(port === 'outputOffsets' ? 'offsets' : port) as keyof CPUNetworkCostTableResult]
          ])
        );
      }
    });
  }
}
