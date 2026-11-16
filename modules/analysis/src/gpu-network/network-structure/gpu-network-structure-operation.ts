// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUNetworkStructure, type GPUNetworkStructureProps} from './gpu-network-structure';

/** Graph-value properties shared by the structure operations. Counts are compile-time. */
export type NetworkStructureOperationProps = {
  id?: string;
  nodeCount: number;
  edgeCount: number;
  topology: {offsets: string; neighbors: string; weights?: string; nodePositions?: string};
  reverseEdgeSlots?: string;
  output: {
    strongComponents?: string;
    bridges?: string;
    articulationPoints?: string;
    biconnectedComponents?: string;
  };
};

/** GPU-only execution controls extend the original props shared with the CPU twin. */
export type GPUNetworkStructureOperationProps = NetworkStructureOperationProps & {
  mode?: GPUNetworkStructureProps['mode'];
  strongComponentsMaximumIterations?: number;
  /** Caller-selected graph value names for the three uint32 SCC status scalars. */
  strongComponentsStatus?: {
    converged: string;
    iterationCount: string;
    iterationLimitReached: string;
  };
};

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUNetworkStructureOperation extends GPUContributorOperation {
  constructor(props: GPUNetworkStructureOperationProps) {
    const inputs: Record<string, string> = {
      offsets: props.topology.offsets,
      neighbors: props.topology.neighbors
    };
    if (props.topology.weights !== undefined) inputs['weights'] = props.topology.weights;
    if (props.topology.nodePositions !== undefined)
      inputs['nodePositions'] = props.topology.nodePositions;
    if (props.reverseEdgeSlots !== undefined) inputs['reverseEdgeSlots'] = props.reverseEdgeSlots;
    const outputs: Record<string, GPUContributorOperationOutput> = {};
    for (const [port, value] of Object.entries(props.output)) {
      if (value !== undefined)
        outputs[port] = {
          value,
          format: 'uint32',
          length:
            port === 'strongComponents' || port === 'articulationPoints'
              ? props.nodeCount
              : props.edgeCount
        };
    }
    if (props.strongComponentsStatus) {
      for (const [port, value] of Object.entries(props.strongComponentsStatus)) {
        outputs[port] = {value, format: 'uint32', length: 1};
      }
    }
    super({
      id: props.id ?? 'network-structure',
      inputs,
      outputs,
      createContributor: getView =>
        new GPUNetworkStructure({
          id: props.id,
          mode: props.mode,
          strongComponentsMaximumIterations: props.strongComponentsMaximumIterations,
          strongComponentsStatus: props.strongComponentsStatus
            ? {
                converged: getView<'uint32'>('converged'),
                iterationCount: getView<'uint32'>('iterationCount'),
                iterationLimitReached: getView<'uint32'>('iterationLimitReached')
              }
            : undefined,
          topology: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights:
              props.topology.weights === undefined ? undefined : getView<'float32'>('weights'),
            nodePositions:
              props.topology.nodePositions === undefined
                ? undefined
                : getView<'float32x2'>('nodePositions')
          },
          reverseEdgeSlots:
            props.reverseEdgeSlots === undefined
              ? undefined
              : getView<'uint32'>('reverseEdgeSlots'),
          output: {
            strongComponents:
              props.output.strongComponents === undefined
                ? undefined
                : getView<'uint32'>('strongComponents'),
            bridges: props.output.bridges === undefined ? undefined : getView<'uint32'>('bridges'),
            articulationPoints:
              props.output.articulationPoints === undefined
                ? undefined
                : getView<'uint32'>('articulationPoints'),
            biconnectedComponents:
              props.output.biconnectedComponents === undefined
                ? undefined
                : getView<'uint32'>('biconnectedComponents')
          }
        })
    });
  }
}
