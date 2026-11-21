// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUNetworkCostTable, validateNetworkCostTableIterations} from './gpu-network-cost-table';
import {
  resolveNetworkRoutingCostPrecision,
  type CPUNetworkRoutingCostPrecision
} from '../../analysis-core/operations/cpu-network-routing';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Graph-value properties with explicit node count and entry capacity. */
export type NetworkCostTableOperationProps = {
  id?: string;
  nodeCount: number;
  capacity: number;
  topology: {offsets: string; neighbors: string; weights: string; nodePositions?: string};
  costLimit: string;
  /** CPU default float32-per-edge; GPU supports only float32-per-edge. */
  costPrecision?: CPUNetworkRoutingCostPrecision;
  laneCount?: number;
  maximumIterations?: number;
  localIterations?: number;
  output: {
    builtCostLimit?: string;
    offsets: string;
    targets: string;
    costs: string;
    status?: {
      complete?: string;
      overflow?: string;
      requiredCount?: string;
      converged?: string;
      iterationLimitReached?: string;
    };
  };
};

/** Captures the graph-value names and budgets used by either operation backend. @internal */
export function snapshotNetworkCostTableOperationProps(
  props: NetworkCostTableOperationProps
): NetworkCostTableOperationProps {
  const topology = props.topology;
  const output = props.output;
  const status = output.status;
  return Object.freeze({
    id: props.id,
    nodeCount: props.nodeCount,
    capacity: props.capacity,
    costLimit: props.costLimit,
    costPrecision: props.costPrecision,
    laneCount: props.laneCount,
    maximumIterations: props.maximumIterations,
    localIterations: props.localIterations,
    topology: Object.freeze({
      offsets: topology.offsets,
      neighbors: topology.neighbors,
      weights: topology.weights,
      nodePositions: topology.nodePositions
    }),
    output: Object.freeze({
      builtCostLimit: output.builtCostLimit,
      offsets: output.offsets,
      targets: output.targets,
      costs: output.costs,
      status: status
        ? Object.freeze({
            complete: status.complete,
            overflow: status.overflow,
            requiredCount: status.requiredCount,
            converged: status.converged,
            iterationLimitReached: status.iterationLimitReached
          })
        : undefined
    })
  });
}
/** Vector operation form. Count is the last CSR offset, as in the contributor. */
export class GPUNetworkCostTableOperation extends GPUContributorOperation {
  constructor(props: NetworkCostTableOperationProps) {
    props = snapshotNetworkCostTableOperationProps(props);
    const costPrecision = resolveNetworkRoutingCostPrecision(
      props.id ?? 'network-cost-table',
      props.costPrecision === undefined ? 'float32-per-edge' : props.costPrecision
    );
    validateRequirement(
      props.id ?? 'network-cost-table',
      costPrecision === 'float32-per-edge',
      `costPrecision must be float32-per-edge on GPU; received ${costPrecision}; use CPUNetworkCostTableOperation for float64`
    );
    validateNetworkCostTableIterations(props.id ?? 'network-cost-table', props);
    const inputs: Record<string, string> = {
      offsets: props.topology.offsets,
      neighbors: props.topology.neighbors,
      weights: props.topology.weights,
      costLimit: props.costLimit
    };
    if (props.topology.nodePositions !== undefined)
      inputs['nodePositions'] = props.topology.nodePositions;
    const outputs: Record<string, GPUContributorOperationOutput> = {
      outputOffsets: {value: props.output.offsets, format: 'uint32', length: props.nodeCount + 1},
      targets: {value: props.output.targets, format: 'uint32', length: props.capacity},
      costs: {value: props.output.costs, format: 'float32', length: props.capacity}
    };
    if (props.output.builtCostLimit !== undefined)
      outputs['builtCostLimit'] = {
        value: props.output.builtCostLimit,
        format: 'float32',
        length: 1
      };
    for (const [port, value] of Object.entries(props.output.status ?? {}))
      if (value !== undefined) outputs[port] = {value, format: 'uint32', length: 1};
    super({
      id: props.id ?? 'network-cost-table',
      inputs,
      outputs,
      createContributor: getView =>
        new GPUNetworkCostTable({
          id: props.id,
          topology: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights'),
            nodePositions:
              props.topology.nodePositions === undefined
                ? undefined
                : getView<'float32x2'>('nodePositions')
          },
          costLimit: getView<'float32'>('costLimit'),
          laneCount: props.laneCount,
          maximumIterations: props.maximumIterations,
          localIterations: props.localIterations,
          output: {
            builtCostLimit:
              props.output.builtCostLimit === undefined
                ? undefined
                : getView<'float32'>('builtCostLimit'),
            offsets: getView<'uint32'>('outputOffsets'),
            targets: getView<'uint32'>('targets'),
            costs: getView<'float32'>('costs'),
            status: Object.fromEntries(
              Object.entries(props.output.status ?? {})
                .filter(([, value]) => value !== undefined)
                .map(([port]) => [port, getView<'uint32'>(port)])
            )
          }
        })
    });
  }
}
