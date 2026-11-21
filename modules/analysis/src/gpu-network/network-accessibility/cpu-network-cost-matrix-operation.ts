// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {computeBudgetedNetworkDistances} from './network-budgeted-search';
import {
  validateNetworkCostMatrixIterations,
  snapshotNetworkCostMatrixOperationProps
} from './network-cost-matrix-construction';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import type {CPUNetworkReachabilityInput} from '../network-reachability/cpu-network-reachability-operation';
import {
  computeNetworkRoutingDistances,
  resolveNetworkRoutingCostPrecision,
  validateNetworkRoutingArrays
} from '../../analysis-core/operations/cpu-network-routing';
import {
  GPUNetworkCostMatrixOperation,
  type NetworkCostMatrixOperationProps
} from './gpu-network-cost-matrix-operation';

/** Starts may share rows; invalid rows and targets are ignored exactly as on GPU. */
export type CPUNetworkCostMatrixInput = Omit<
  CPUNetworkReachabilityInput,
  'startCount' | 'activeIterations' | 'costBreaks'
> & {startRows?: Uint32Array; targetNodes?: Uint32Array};
/** Dimensions independent of the input view lengths. */
export type CPUNetworkCostMatrixOptions = {
  rowCount: number;
  startsPerRow?: number;
  maximumIterations?: number;
  localIterations?: number;
};
/** Row-major costs and one-word status outputs. */
export type CPUNetworkCostMatrixResult = {
  costs: Float32Array;
  converged: Uint32Array;
  iterationLimitReached: Uint32Array;
};

/**
 * Computes a multi-start shortest-path search per row, gathering targets when requested.
 *
 * What differs from the GPU contributor:
 * - costPrecision defaults to float64 with one output rounding; float32-per-edge instead rounds
 *   each addition before cutoff/label comparisons. Near ties, subnormals and overflowing paths
 *   can differ from GPU arithmetic. NaN costLimit disables the cutoff, matching the GPU rule.
 * - Omitted maximumIterations searches to completion. An explicit round budget permits at most
 *   maximumIterations rounds with localIterations hops and a 256-slot local queue per chunk
 *   (localIterations defaults to 16). Overflow waits for the next round. GPU workgroup
 *   ordering can change partial labels with competing discoveries. Converged is 1 only
 *   when every row's frontier empties; iterationLimitReached is its complement.
 * - GPU lane, command-node and device limits are dropped.
 * - CPU memory is one network search plus the output, without lane-expanded scratch.
 *
 * @throws Error naming inconsistent startCosts, startRows, targetNodes, costLimit or topology arrays.
 */
export function computeNetworkCostMatrixOnCPU(
  input: CPUNetworkCostMatrixInput,
  options: CPUNetworkCostMatrixOptions
): CPUNetworkCostMatrixResult {
  const id = 'computeNetworkCostMatrixOnCPU';
  const costPrecision = resolveNetworkRoutingCostPrecision(id, input.costPrecision);
  options = snapshotConstructionFields(options, [
    'rowCount',
    'startsPerRow',
    'maximumIterations',
    'localIterations'
  ]);
  validateCapacity(id, 'rowCount', options.rowCount);
  const startsPerRow = options.startsPerRow ?? 1;
  validateCapacity(id, 'startsPerRow', startsPerRow);
  validateNetworkCostMatrixIterations(id, options);
  const localIterations = options.localIterations ?? 16;
  if (input.startRows && input.startRows.length !== input.startNodes.length)
    throw new Error('startRows length must equal startNodes length');
  if (input.startCosts && input.startCosts.length !== input.startNodes.length)
    throw new Error('startCosts length must equal startNodes length');
  if (input.costLimit && input.costLimit.length !== 1)
    throw new Error('costLimit must contain exactly one row');
  if (input.targetNodes && !input.targetNodes.length)
    throw new Error('targetNodes must contain at least one row');
  const nodeCount = validateNetworkRoutingArrays(input.topology);
  const columnCount = input.targetNodes?.length ?? nodeCount;
  validateCapacity(id, 'rowCount * columnCount', options.rowCount * columnCount);
  const costs = new Float32Array(options.rowCount * columnCount).fill(Infinity);
  const starts: number[][] = Array.from({length: options.rowCount}, () => []);
  for (let start = 0; start < input.startNodes.length; start++) {
    const row = input.startRows?.[start] ?? Math.floor(start / startsPerRow);
    if (row < options.rowCount) starts[row].push(start);
  }
  let converged = true;
  for (let row = 0; row < options.rowCount; row++) {
    const searchInput = {
      costPrecision,
      topology: input.topology,
      startNodes: Uint32Array.from(starts[row], start => input.startNodes[start]),
      startCosts: input.startCosts
        ? Float32Array.from(starts[row], start => input.startCosts![start])
        : undefined,
      costLimit: input.costLimit
    };
    const search =
      options.maximumIterations === undefined
        ? computeNetworkRoutingDistances(id, searchInput)
        : computeBudgetedNetworkDistances(searchInput, options.maximumIterations, localIterations);
    const {distances} = search;
    converged &&= search.converged;
    for (let column = 0; column < columnCount; column++) {
      const target = input.targetNodes?.[column] ?? column;
      if (target < nodeCount) costs[row * columnCount + column] = distances[target];
    }
  }
  return {
    costs,
    converged: Uint32Array.of(Number(converged)),
    iterationLimitReached: Uint32Array.of(Number(!converged))
  };
}

/** CPU twin; explicit round/local-hop budgets bind, while GPU lane/device limits do not. */
export class CPUNetworkCostMatrixOperation extends GPUCPUOperation {
  constructor(props: NetworkCostMatrixOperationProps) {
    props = snapshotNetworkCostMatrixOperationProps(props);
    const id = props.id ?? 'network-cost-matrix';
    const costPrecision = resolveNetworkRoutingCostPrecision(id, props.costPrecision);
    validateCapacity(id, 'nodeCount', props.nodeCount);
    validateCapacity(id, 'rowCount', props.rowCount);
    validateCapacity(id, 'startsPerRow', props.startsPerRow ?? 1);
    validateNetworkCostMatrixIterations(id, props);
    if (props.targetNodes !== undefined)
      validateCapacity(id, 'targetCount', props.targetCount ?? 0);
    const operation = new GPUNetworkCostMatrixOperation({...props, costPrecision: undefined});
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
        const targetNodes =
          props.targetNodes === undefined
            ? undefined
            : getCPUOperationUint32Rows(id, inputs, 'targetNodes');
        if (targetNodes && targetNodes.length !== props.targetCount)
          throw new Error(`${id} targetNodes length must equal targetCount`);
        const result = computeNetworkCostMatrixOnCPU(
          {
            costPrecision,
            topology,
            startNodes: getCPUOperationUint32Rows(id, inputs, 'startNodes'),
            startCosts:
              props.startCosts === undefined
                ? undefined
                : getCPUOperationFloat32Rows(id, inputs, 'startCosts'),
            startRows:
              props.startRows === undefined
                ? undefined
                : getCPUOperationUint32Rows(id, inputs, 'startRows'),
            costLimit:
              props.costLimit === undefined
                ? undefined
                : getCPUOperationFloat32Rows(id, inputs, 'costLimit'),
            targetNodes
          },
          {
            rowCount: props.rowCount,
            startsPerRow: props.startsPerRow,
            maximumIterations: props.maximumIterations,
            localIterations: props.localIterations
          }
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUNetworkCostMatrixResult]
          ])
        );
      }
    });
  }
}
