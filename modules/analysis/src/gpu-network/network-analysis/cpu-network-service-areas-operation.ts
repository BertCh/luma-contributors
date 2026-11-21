// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import type {CPUNetworkReachabilityInput} from '../network-reachability/cpu-network-reachability-operation';
import {
  computeNetworkRoutingDistances,
  resolveNetworkRoutingCostPrecision,
  addNetworkRoutingCosts,
  type CPUNetworkRoutingCostPrecision,
  computeNetworkRoutingPredecessorsWithStatus
} from '../../analysis-core/operations/cpu-network-routing';
import {
  GPUNetworkServiceAreasOperation,
  snapshotNetworkServiceAreasOperationProps,
  type NetworkServiceAreasOperationProps
} from './gpu-network-service-areas-operation';

/** Packed network and facilities; facility IDs are their original rows. */
export type CPUNetworkServiceAreasInput = {
  topology: CPUNetworkReachabilityInput['topology'];
  facilities: Uint32Array;
  facilityCosts?: Float32Array;
  facilityCount?: number | Uint32Array;
  costLimit?: Float32Array;
  /** Default float64; applies to costs, facility labels and predecessor tightness. */
  costPrecision?: CPUNetworkRoutingCostPrecision;
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
  iterationLimitReached: Uint32Array;
};

/** Explicit phase budgets bind CPU frontiers; omitted budgets keep exact CPU traversal. */
export type CPUNetworkServiceAreasOptions = {
  maximumIterations?: number;
  maximumLabelIterations?: number;
  maximumTieIterations?: number;
  localIterations?: number;
  /** Operation lowering runs the tie phase only when predecessors are requested. */
  computePredecessors?: boolean;
};

/**
 * Propagates the smallest facility label over tight shortest-path edges and builds tight,
 * same-label predecessors. For overlapping CSR rows, the label phase shares tight flags for
 * overlapping slots, as on the GPU. Predecessor selection checks each source individually.
 *
 * What differs from the GPU contributor:
 * - costPrecision defaults to float64; float32-per-edge rounds every path addition before
 *   cutoff, facility-label and predecessor tightness tests. Facility sums use float64 before
 *   output rounding. Subnormals and overflowing paths can still differ from GPU arithmetic.
 * - Explicit cost, label and tie budgets bound synchronous CPU frontiers; activeIterations
 *   clamps cost and label rounds. Omitted budgets keep exact traversal. GPU scheduling can
 *   change partial labels near a budget boundary; device limits are dropped.
 * - Unused facility aggregates are zero, including a zero-length facility input when aggregates
 *   are omitted from the graph operation. NaN costLimit disables the cutoff, like the GPU.
 *
 * @throws Error naming inconsistent topology, facilities, facilityCosts, facilityCount or scalar arrays.
 */
export function computeNetworkServiceAreasOnCPU(
  input: CPUNetworkServiceAreasInput,
  options: CPUNetworkServiceAreasOptions = {}
): CPUNetworkServiceAreasResult {
  const costPrecision = resolveNetworkRoutingCostPrecision(
    'computeNetworkServiceAreasOnCPU',
    input.costPrecision
  );
  options = snapshotConstructionFields(options, [
    'maximumIterations',
    'maximumLabelIterations',
    'maximumTieIterations',
    'localIterations',
    'computePredecessors'
  ]);
  for (const name of [
    'maximumIterations',
    'maximumLabelIterations',
    'maximumTieIterations',
    'localIterations'
  ] as const)
    if (options[name] !== undefined)
      validateCapacity('computeNetworkServiceAreasOnCPU', name, options[name]);
  const localIterations = options.localIterations ?? 16;
  const boundedCosts =
    options.maximumIterations !== undefined || input.activeIterations !== undefined;
  const costRounds = Math.min(
    options.maximumIterations ?? 64,
    input.activeIterations?.[0] ?? Infinity
  );
  const boundedLabels = boundedCosts || options.maximumLabelIterations !== undefined;
  const labelRounds = Math.min(
    options.maximumLabelIterations ?? Math.min(options.maximumIterations ?? 64, 32),
    input.activeIterations?.[0] ?? Infinity
  );
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
    costPrecision,
    topology: input.topology,
    startNodes: input.facilities,
    startCosts: input.facilityCosts,
    startCount: input.facilityCount,
    costLimit: input.costLimit,
    activeIterations: input.activeIterations
  };
  const {distances, converged: costsConverged} = computeNetworkRoutingDistances(
    'computeNetworkServiceAreasOnCPU',
    routingInput,
    1,
    // A nonnegative shortest path has at most nodeCount - 1 edges. One
    // additional hop consumes the final frontier, regardless of the round budget.
    boundedCosts
      ? Math.min(costRounds * localIterations, input.topology.offsets.length - 1)
      : Infinity
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
        addNetworkRoutingCosts(distances[source], weights[edge], costPrecision) ===
          distances[target]
      )
        tightEdges[edge] = 1;
    }
  }
  let frontier = queue;
  let labelHops = 0;
  const maximumLabelHops = boundedLabels ? labelRounds * localIterations : Infinity;
  while (frontier.length && labelHops < maximumLabelHops) {
    const sources = frontier.map(source => ({source, label: assignments[source]}));
    const next = new Set<number>();
    for (const {source, label} of sources) {
      for (
        let edge = Math.min(offsets[source], neighbors.length);
        edge < Math.min(offsets[source + 1], neighbors.length);
        edge++
      ) {
        const target = neighbors[edge];
        if (tightEdges[edge] && label < assignments[target]) {
          assignments[target] = label;
          next.add(target);
        }
      }
    }
    frontier = Array.from(next);
    labelHops++;
  }
  const predecessorResult =
    options.computePredecessors === false
      ? {predecessors: new Uint32Array(nodeCount).fill(0xffffffff), converged: true}
      : computeNetworkRoutingPredecessorsWithStatus(
          routingInput,
          distances,
          assignments,
          options.maximumTieIterations !== undefined
            ? options.maximumTieIterations * localIterations
            : Infinity
        );
  const converged =
    costsConverged &&
    frontier.length === 0 &&
    predecessorResult.converged &&
    (!boundedCosts || costRounds > 0) &&
    (!boundedLabels || labelRounds > 0);
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
    predecessors: predecessorResult.predecessors,
    facilityNodeCounts,
    facilityCostSums: Float32Array.from(sums),
    converged: Uint32Array.of(Number(converged)),
    iterationLimitReached: Uint32Array.of(Number(!converged))
  };
}

/** CPU twin retaining explicit cost, label and tie budgets. */
export class CPUNetworkServiceAreasOperation extends GPUCPUOperation {
  constructor(props: NetworkServiceAreasOperationProps) {
    props = snapshotNetworkServiceAreasOperationProps(props);
    const id = props.id ?? 'network-service-areas';
    const costPrecision = resolveNetworkRoutingCostPrecision(id, props.costPrecision);
    const options = snapshotConstructionFields(props, [
      'maximumIterations',
      'maximumLabelIterations',
      'maximumTieIterations',
      'localIterations'
    ]);
    for (const name of [
      'maximumIterations',
      'maximumLabelIterations',
      'maximumTieIterations',
      'localIterations'
    ] as const)
      if (options[name] !== undefined) validateCapacity(id, name, options[name]);
    const computePredecessors = props.output.predecessors !== undefined;
    validateCapacity(id, 'nodeCount', props.nodeCount);
    validateCapacity(id, 'facilityCapacity', props.facilityCapacity, 0);
    if (typeof props.facilityCount === 'number')
      validateCapacity(id, 'facilityCount', props.facilityCount, 0, props.facilityCapacity);
    const operation = new GPUNetworkServiceAreasOperation({...props, costPrecision: undefined});
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
        const result = computeNetworkServiceAreasOnCPU(
          {
            costPrecision,
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
          },
          {...options, computePredecessors}
        );
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
