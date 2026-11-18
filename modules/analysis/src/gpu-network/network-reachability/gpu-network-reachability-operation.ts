// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUNetworkReachability} from './gpu-network-reachability';
import {snapshotNetworkReachabilityConstructionProps} from './network-reachability-construction';

/** Graph values and explicit output dimensions shared by the GPU and CPU operations. */
export type NetworkReachabilityOperationProps = {
  id?: string;
  nodeCount: number;
  laneCount?: number;
  topology: {offsets: string; neighbors: string; weights: string; nodePositions?: string};
  startNodes: string;
  startCosts?: string;
  startCount?: number | string;
  activeIterations?: string;
  costLimit?: string;
  costBreaks?: string;
  costBreakCount?: number;
  maximumIterations?: number;
  localIterations?: number;
  maximumTieIterations?: number;
  output: {
    costs: string;
    predecessors?: string;
    bands?: string;
    bandCounts?: string;
    unresolvedCount?: string;
    status?: {
      converged?: string;
      iterationCount?: string;
      iterationLimitReached?: string;
      relaxationCount?: string;
      reachedEdgeCount?: string;
    };
  };
};

/** Vector operation form. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUNetworkReachabilityOperation extends GPUContributorOperation {
  constructor(props: NetworkReachabilityOperationProps) {
    props = snapshotNetworkReachabilityConstructionProps(props);
    const inputs: Record<string, string> = {
      offsets: props.topology.offsets,
      neighbors: props.topology.neighbors,
      weights: props.topology.weights,
      startNodes: props.startNodes
    };
    if (props.topology.nodePositions !== undefined)
      inputs['nodePositions'] = props.topology.nodePositions;
    for (const port of ['startCosts', 'activeIterations', 'costLimit', 'costBreaks'] as const) {
      if (props[port] !== undefined) inputs[port] = props[port];
    }
    if (typeof props.startCount === 'string') inputs['startCount'] = props.startCount;
    const outputs: Record<string, GPUContributorOperationOutput> = {};
    const length = props.nodeCount * (props.laneCount ?? 1);
    for (const [port, value] of Object.entries({...props.output, ...props.output.status})) {
      if (typeof value !== 'string') continue;
      outputs[port] = {
        value,
        format: port === 'costs' ? 'float32' : 'uint32',
        length: ['costs', 'predecessors', 'bands'].includes(port)
          ? length
          : port === 'bandCounts'
            ? (props.costBreakCount ?? 0)
            : ['relaxationCount', 'reachedEdgeCount'].includes(port)
              ? 2
              : 1
      };
    }
    super({
      id: props.id ?? 'network-reachability',
      inputs,
      outputs,
      createContributor: getView =>
        new GPUNetworkReachability({
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
          startCount:
            typeof props.startCount === 'string'
              ? getView<'uint32'>('startCount')
              : props.startCount,
          activeIterations:
            props.activeIterations === undefined
              ? undefined
              : getView<'uint32'>('activeIterations'),
          costLimit: props.costLimit === undefined ? undefined : getView<'float32'>('costLimit'),
          costBreaks: props.costBreaks === undefined ? undefined : getView<'float32'>('costBreaks'),
          laneCount: props.laneCount,
          maximumIterations: props.maximumIterations,
          localIterations: props.localIterations,
          maximumTieIterations: props.maximumTieIterations,
          output: {
            costs: getView<'float32'>('costs'),
            predecessors:
              props.output.predecessors === undefined
                ? undefined
                : getView<'uint32'>('predecessors'),
            bands: props.output.bands === undefined ? undefined : getView<'uint32'>('bands'),
            bandCounts:
              props.output.bandCounts === undefined ? undefined : getView<'uint32'>('bandCounts'),
            unresolvedCount:
              props.output.unresolvedCount === undefined
                ? undefined
                : getView<'uint32'>('unresolvedCount'),
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
