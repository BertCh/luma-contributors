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
import {
  computeNetworkRoutingDistances,
  computeNetworkRoutingPredecessors
} from '../../analysis-core/operations/cpu-network-routing';
import {
  GPUNetworkServiceAreasOperation,
  type NetworkServiceAreasOperationProps
} from './gpu-network-service-areas-operation';

/** Packed network and facilities; facility IDs are their original rows. */
export type CPUNetworkServiceAreasInput = {
  topology: CPUNetworkReachabilityInput['topology'];
  facilities: Uint32Array;
  facilityCosts?: Float32Array;
  facilityCount?: number | Uint32Array;
  costLimit?: Float32Array;
  activeIterations?: Uint32Array;
};
/** Every optional port is returned in the same layout as the GPU contributor. */
export type CPUNetworkServiceAreasResult = {
  assignments: Uint32Array;
  costs: Float32Array;
  predecessors: Uint32Array;
  facilityNodeCounts: Uint32Array;
  facilityCostSums: Float32Array;
  converged: Uint32Array;
};

/**
 * Assigns minimum-cost facilities, breaking ties by facility row, and builds same-label predecessors.
 * Overlapping CSR rows reuse slot-indexed tight/tie flags exactly as on the GPU; assignments and
 * predecessors can then follow edges that are not tight for that source or have a different label.
 *
 * What differs from the GPU contributor:
 * - Float64 shortest paths, tight-edge tests and facility sums precede one float32 output rounding.
 *   Near ties, cutoffs, subnormals and float32-overflow paths can change assignments and parents.
 * - Cost, label and tie phases finish exactly. GPU round, local-hop, activeIterations and device
 *   limits are dropped; converged is always 1.
 * - Unused facility aggregates are zero, including a zero-length facility input when aggregates
 *   are omitted from the graph operation. NaN costLimit disables the cutoff, like the GPU.
 *
 * @throws Error naming inconsistent topology, facilities, facilityCosts, facilityCount or scalar arrays.
 */
export function computeNetworkServiceAreasOnCPU(
  input: CPUNetworkServiceAreasInput
): CPUNetworkServiceAreasResult {
  if (input.facilityCosts && input.facilityCosts.length !== input.facilities.length)
    throw new Error('facilityCosts length must equal facilities length');
  if (typeof input.facilityCount === 'object' && input.facilityCount.length !== 1)
    throw new Error('facilityCount must contain exactly one row');
  if (typeof input.facilityCount === 'number')
    validateCapacity(
      'computeNetworkServiceAreasOnCPU',
      'facilityCount',
      input.facilityCount,
      0,
      input.facilities.length
    );
  const routingInput: CPUNetworkReachabilityInput = {
    topology: input.topology,
    startNodes: input.facilities,
    startCosts: input.facilityCosts,
    startCount: input.facilityCount,
    costLimit: input.costLimit,
    activeIterations: input.activeIterations
  };
  const {distances} = computeNetworkRoutingDistances(
    'computeNetworkServiceAreasOnCPU',
    routingInput
  );
  const nodeCount = distances.length;
  const assignments = new Uint32Array(nodeCount).fill(0xffffffff);
  const facilityCount = Math.min(
    typeof input.facilityCount === 'number'
      ? input.facilityCount
      : (input.facilityCount?.[0] ?? input.facilities.length),
    input.facilities.length
  );
  const queue: number[] = [];
  for (let facility = 0; facility < facilityCount; facility++) {
    const node = input.facilities[facility];
    const cost = input.facilityCosts?.[facility] ?? 0;
    if (
      node < nodeCount &&
      Number.isFinite(cost) &&
      cost >= 0 &&
      cost === distances[node] &&
      facility < assignments[node]
    ) {
      assignments[node] = facility;
      queue.push(node);
    }
  }
  const {offsets, neighbors, weights} = input.topology;
  const tightEdges = new Uint8Array(neighbors.length);
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
        distances[source] + weights[edge] === distances[target]
      )
        tightEdges[edge] = 1;
    }
  }
  for (let cursor = 0; cursor < queue.length; cursor++) {
    const source = queue[cursor];
    for (
      let edge = Math.min(offsets[source], neighbors.length);
      edge < Math.min(offsets[source + 1], neighbors.length);
      edge++
    ) {
      const target = neighbors[edge];
      if (tightEdges[edge] && assignments[source] < assignments[target]) {
        assignments[target] = assignments[source];
        queue.push(target);
      }
    }
  }
  const predecessors = computeNetworkRoutingPredecessors(routingInput, distances, assignments);
  const facilityNodeCounts = new Uint32Array(input.facilities.length);
  const sums = new Float64Array(input.facilities.length);
  for (let node = 0; node < nodeCount; node++) {
    const facility = assignments[node];
    if (facility < input.facilities.length) {
      facilityNodeCounts[facility]++;
      sums[facility] += distances[node];
    }
  }
  return {
    assignments,
    costs: Float32Array.from(distances),
    predecessors,
    facilityNodeCounts,
    facilityCostSums: Float32Array.from(sums),
    converged: Uint32Array.of(1)
  };
}

/** Exact CPU twin; GPU-only phase budgets do not limit computation. */
export class CPUNetworkServiceAreasOperation extends GPUCPUOperation {
  constructor(props: NetworkServiceAreasOperationProps) {
    const id = props.id ?? 'network-service-areas';
    validateCapacity(id, 'nodeCount', props.nodeCount);
    validateCapacity(id, 'facilityCapacity', props.facilityCapacity, 0);
    if (typeof props.facilityCount === 'number')
      validateCapacity(id, 'facilityCount', props.facilityCount, 0, props.facilityCapacity);
    const operation = new GPUNetworkServiceAreasOperation(props);
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
        const facilities = getCPUOperationUint32Rows(id, inputs, 'facilities');
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
        if (facilities.length !== props.facilityCapacity)
          throw new Error(`${id} facilities length must equal facilityCapacity`);
        const result = computeNetworkServiceAreasOnCPU({
          topology,
          facilities,
          facilityCosts:
            props.facilityCosts === undefined
              ? undefined
              : getCPUOperationFloat32Rows(id, inputs, 'facilityCosts'),
          facilityCount:
            typeof props.facilityCount === 'string'
              ? getCPUOperationUint32Rows(id, inputs, 'facilityCount')
              : props.facilityCount,
          costLimit:
            props.costLimit === undefined
              ? undefined
              : getCPUOperationFloat32Rows(id, inputs, 'costLimit'),
          activeIterations:
            props.activeIterations === undefined
              ? undefined
              : getCPUOperationUint32Rows(id, inputs, 'activeIterations')
        });
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUNetworkServiceAreasResult]
          ])
        );
      }
    });
  }
}
