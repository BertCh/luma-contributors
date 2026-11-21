// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  validateRequirement,
  validateCapacity
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUNetworkCentrality, type GPUNetworkCentralityProps} from './gpu-network-centrality';

/** Graph value names replacing views; counts formerly inferred from views are explicit. */
export type NetworkCentralityOperationProps = Pick<
  GPUNetworkCentralityProps,
  'id' | 'laneCount' | 'maximumIterations' | 'localIterations' | 'betweennessScale' | 'sampling'
> & {
  nodeCount: number;
  edgeCount: number;
  reverseEdgeCount?: number;
  pivotCount: number;
  topology: {offsets: string; neighbors: string; weights: string; nodePositions?: string};
  reverseTopology?: {offsets: string; neighbors: string; weights: string; nodePositions?: string};
  pivotNodes: string;
  costLimit?: string;
  output: {
    betweenness?: string;
    edgeBetweenness?: string;
    closeness?: string;
    harmonicCloseness?: string;
    reachedCounts?: string;
    status?: {
      converged?: string;
      zeroWeightCycle?: string;
      iterationLimitReached?: string;
      approximation?: string;
    };
  };
};

/** GPU centrality operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUNetworkCentralityOperation extends GPUContributorOperation {
  constructor(props: NetworkCentralityOperationProps) {
    validateRequirement(
      props.id ?? 'network-centrality',
      !(props.output.betweenness || props.output.edgeBetweenness) || Boolean(props.reverseTopology),
      'reverseTopology is required for betweenness'
    );
    if (props.reverseEdgeCount !== undefined)
      validateCapacity(
        props.id ?? 'network-centrality',
        'reverseEdgeCount',
        props.reverseEdgeCount,
        0
      );
    const inputs: Record<string, string> = {pivotNodes: props.pivotNodes};
    for (const prefix of ['topology', 'reverseTopology'] as const) {
      for (const [field, value] of Object.entries(props[prefix] ?? {})) {
        if (value !== undefined) inputs[`${prefix}.${field}`] = value;
      }
    }
    if (props.costLimit !== undefined) inputs['costLimit'] = props.costLimit;
    const outputs: Record<string, GPUContributorOperationOutput> = {};
    for (const port of [
      'betweenness',
      'edgeBetweenness',
      'closeness',
      'harmonicCloseness',
      'reachedCounts'
    ] as const) {
      const value = props.output[port];
      if (value !== undefined) {
        outputs[port] = {
          value,
          format: port === 'reachedCounts' ? 'uint32' : 'float32',
          length:
            port === 'betweenness'
              ? props.nodeCount
              : port === 'edgeBetweenness'
                ? props.edgeCount
                : props.pivotCount
        };
      }
    }
    for (const [port, value] of Object.entries(props.output.status ?? {})) {
      if (value !== undefined) outputs[port] = {value, format: 'uint32', length: 1};
    }
    super({
      id: props.id ?? 'network-centrality',
      inputs,
      outputs,
      createContributor: getView => {
        if (props.reverseTopology && props.reverseEdgeCount !== undefined) {
          const actual = getView<'uint32'>('reverseTopology.neighbors').length;
          validateRequirement(
            props.id ?? 'network-centrality',
            actual === props.reverseEdgeCount,
            `reverseEdgeCount must equal reverseTopology.neighbors.length ${actual}; received ${props.reverseEdgeCount}`
          );
        }
        return new GPUNetworkCentrality({
          ...props,
          topology: {
            offsets: getView<'uint32'>('topology.offsets'),
            neighbors: getView<'uint32'>('topology.neighbors'),
            weights: getView<'float32'>('topology.weights'),
            nodePositions:
              props.topology.nodePositions !== undefined
                ? getView<'float32x2'>('topology.nodePositions')
                : undefined
          },
          reverseTopology: props.reverseTopology
            ? {
                offsets: getView<'uint32'>('reverseTopology.offsets'),
                neighbors: getView<'uint32'>('reverseTopology.neighbors'),
                weights: getView<'float32'>('reverseTopology.weights'),
                nodePositions:
                  props.reverseTopology.nodePositions !== undefined
                    ? getView<'float32x2'>('reverseTopology.nodePositions')
                    : undefined
              }
            : undefined,
          pivotNodes: getView<'uint32'>('pivotNodes'),
          costLimit: props.costLimit === undefined ? undefined : getView<'float32'>('costLimit'),
          output: {
            betweenness:
              props.output.betweenness !== undefined
                ? getView<'float32'>('betweenness')
                : undefined,
            edgeBetweenness:
              props.output.edgeBetweenness !== undefined
                ? getView<'float32'>('edgeBetweenness')
                : undefined,
            closeness:
              props.output.closeness !== undefined ? getView<'float32'>('closeness') : undefined,
            harmonicCloseness:
              props.output.harmonicCloseness !== undefined
                ? getView<'float32'>('harmonicCloseness')
                : undefined,
            reachedCounts:
              props.output.reachedCounts !== undefined
                ? getView<'uint32'>('reachedCounts')
                : undefined,
            status: {
              approximation:
                props.output.status?.approximation !== undefined
                  ? getView<'uint32'>('approximation')
                  : undefined,
              converged:
                props.output.status?.converged !== undefined
                  ? getView<'uint32'>('converged')
                  : undefined,
              iterationLimitReached:
                props.output.status?.iterationLimitReached !== undefined
                  ? getView<'uint32'>('iterationLimitReached')
                  : undefined,
              zeroWeightCycle:
                props.output.status?.zeroWeightCycle !== undefined
                  ? getView<'uint32'>('zeroWeightCycle')
                  : undefined
            }
          }
        });
      }
    });
  }
}
