// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUNetworkCostTable} from './gpu-network-cost-table';

/** Graph-value properties with explicit node count and entry capacity. */
export type NetworkCostTableOperationProps = {
  id?: string;
  nodeCount: number;
  capacity: number;
  topology: {offsets: string; neighbors: string; weights: string; nodePositions?: string};
  costLimit: string;
  laneCount?: number;
  maximumIterations?: number;
  localIterations?: number;
  output: {
    offsets: string;
    targets: string;
    costs: string;
    status?: {
      overflow?: string;
      requiredCount?: string;
      converged?: string;
      iterationLimitReached?: string;
    };
  };
};
/** Vector operation form. Count is the last CSR offset, as in the contributor. */
export class GPUNetworkCostTableOperation extends GPUContributorOperation {
  constructor(props: NetworkCostTableOperationProps) {
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
