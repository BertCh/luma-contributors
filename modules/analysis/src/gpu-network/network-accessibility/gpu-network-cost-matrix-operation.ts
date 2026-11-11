// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUNetworkCostMatrix} from './gpu-network-cost-matrix';

/** Graph-value properties; matrix dimensions are explicit compile-time counts. */
export type NetworkCostMatrixOperationProps = {
  id?: string;
  nodeCount: number;
  rowCount: number;
  targetCount?: number;
  topology: {offsets: string; neighbors: string; weights: string; nodePositions?: string};
  startNodes: string;
  startCosts?: string;
  startRows?: string;
  startsPerRow?: number;
  costLimit?: string;
  targetNodes?: string;
  laneCount?: number;
  maximumIterations?: number;
  localIterations?: number;
  output: {costs: string; status?: {converged?: string; iterationLimitReached?: string}};
};

/** Vector operation form. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUNetworkCostMatrixOperation extends GPUContributorOperation {
  constructor(props: NetworkCostMatrixOperationProps) {
    const inputs: Record<string, string> = {
      offsets: props.topology.offsets,
      neighbors: props.topology.neighbors,
      weights: props.topology.weights,
      startNodes: props.startNodes
    };
    if (props.topology.nodePositions !== undefined)
      inputs['nodePositions'] = props.topology.nodePositions;
    for (const port of ['startCosts', 'startRows', 'costLimit', 'targetNodes'] as const)
      if (props[port] !== undefined) inputs[port] = props[port];
    const outputs: Record<string, GPUContributorOperationOutput> = {
      costs: {
        value: props.output.costs,
        format: 'float32',
        length:
          props.rowCount *
          (props.targetNodes === undefined ? props.nodeCount : (props.targetCount ?? 0))
      }
    };
    for (const [port, value] of Object.entries(props.output.status ?? {}))
      if (value !== undefined) outputs[port] = {value, format: 'uint32', length: 1};
    super({
      id: props.id ?? 'network-cost-matrix',
      inputs,
      outputs,
      createContributor: getView =>
        new GPUNetworkCostMatrix({
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
          startNodes: getView<'uint32'>('startNodes'),
          startCosts: props.startCosts === undefined ? undefined : getView<'float32'>('startCosts'),
          startRows: props.startRows === undefined ? undefined : getView<'uint32'>('startRows'),
          startsPerRow: props.startsPerRow,
          costLimit: props.costLimit === undefined ? undefined : getView<'float32'>('costLimit'),
          targetNodes:
            props.targetNodes === undefined ? undefined : getView<'uint32'>('targetNodes'),
          laneCount: props.laneCount,
          maximumIterations: props.maximumIterations,
          localIterations: props.localIterations,
          output: {
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
