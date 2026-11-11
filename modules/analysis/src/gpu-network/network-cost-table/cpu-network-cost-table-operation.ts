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
  validateNetworkRoutingArrays
} from '../../analysis-core/operations/cpu-network-routing';
import {
  GPUNetworkCostTableOperation,
  type NetworkCostTableOperationProps
} from './gpu-network-cost-table-operation';

/** Largest node count whose nodeCount + 1 offset rows fit the uint32 graph shape limit. */
export const NETWORK_COST_TABLE_MAXIMUM_NODE_COUNT = 0xfffffffe;

/** Packed CSR and the same one-word cutoff view as the GPU contributor. */
export type CPUNetworkCostTableInput = {
  topology: CPUNetworkReachabilityInput['topology'];
  costLimit: Float32Array;
};
/** Entry capacity, independent of the required count. */
export type CPUNetworkCostTableOptions = {capacity: number};
/** CSR outputs and one-word statuses. There is no separate count port in GPUNetworkCostTable. */
export type CPUNetworkCostTableResult = {
  offsets: Uint32Array;
  targets: Uint32Array;
  costs: Float32Array;
  overflow: Uint32Array;
  requiredCount: Uint32Array;
  converged: Uint32Array;
};

/**
 * Builds cutoff all-pairs CSR rows with ascending targets, including finite diagonal zeros.
 *
 * What differs from the GPU contributor:
 * - Float64 shortest paths are rounded once at output; cutoff decisions and float32-overflow paths
 *   can differ. NaN costLimit disables the cutoff, as in the GPU comparisons.
 * - Searches finish exactly; GPU lane, round, command-node and device limits are dropped.
 *   converged is always 1. Output capacity still truncates rows and requiredCount saturates at uint32.
 * - Unused target/cost capacity is zero-initialized; the GPU does not write unused capacity.
 *
 * @throws Error naming an inconsistent topology or costLimit argument.
 */
export function computeNetworkCostTableOnCPU(
  input: CPUNetworkCostTableInput,
  options: CPUNetworkCostTableOptions
): CPUNetworkCostTableResult {
  validateCapacity('computeNetworkCostTableOnCPU', 'capacity', options.capacity);
  const nodeCount = validateNetworkRoutingArrays(input.topology);
  if (input.costLimit.length !== 1) throw new Error('costLimit must contain exactly one row');
  const offsets = new Uint32Array(nodeCount + 1);
  const targets = new Uint32Array(options.capacity);
  const costs = new Float32Array(options.capacity);
  let requiredCount = 0;
  for (let source = 0; source < nodeCount; source++) {
    offsets[source] = Math.min(requiredCount, options.capacity);
    const {distances} = computeNetworkRoutingDistances('computeNetworkCostTableOnCPU', {
      topology: input.topology,
      startNodes: Uint32Array.of(source),
      costLimit: input.costLimit
    });
    for (let target = 0; target < nodeCount; target++) {
      if (!Number.isFinite(distances[target]) || distances[target] < 0) continue;
      if (requiredCount < options.capacity) {
        targets[requiredCount] = target;
        costs[requiredCount] = distances[target];
      }
      requiredCount = Math.min(requiredCount + 1, 0xffffffff);
    }
  }
  offsets[nodeCount] = Math.min(requiredCount, options.capacity);
  return {
    offsets,
    targets,
    costs,
    requiredCount: Uint32Array.of(requiredCount),
    overflow: Uint32Array.of(Number(requiredCount > options.capacity)),
    converged: Uint32Array.of(1)
  };
}

/** CPU twin, retaining the CSR capacity contract and dropping GPU execution budgets. */
export class CPUNetworkCostTableOperation extends GPUCPUOperation {
  constructor(props: NetworkCostTableOperationProps) {
    const id = props.id ?? 'network-cost-table';
    validateCapacity(id, 'nodeCount', props.nodeCount, 1, NETWORK_COST_TABLE_MAXIMUM_NODE_COUNT);
    validateCapacity(id, 'capacity', props.capacity);
    const operation = new GPUNetworkCostTableOperation(props);
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
          {topology, costLimit: getCPUOperationFloat32Rows(id, inputs, 'costLimit')},
          {capacity: props.capacity}
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
