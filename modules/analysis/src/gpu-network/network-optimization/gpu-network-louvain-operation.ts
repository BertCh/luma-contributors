// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUNetworkLouvain, type GPUNetworkLouvainProps} from './gpu-network-louvain';

/** Shared operation properties: graph value names in place of packed views. */
export type NetworkLouvainOperationProps = Pick<
  GPUNetworkLouvainProps,
  'id' | 'nodeCount' | 'directed' | 'seed' | 'maximumLevels' | 'maximumIterations'
> & {
  /** float32: resolution and minimum modularity improvement. */
  parameters: string;
  output: {
    communities: string;
    objectiveTrace: string;
    summary: string;
    status?: {
      iterationCount?: string;
      converged?: string;
      iterationLimitReached?: string;
      invalidCount?: string;
      approximation?: string;
    };
  };
} & (
    | {weights: string; topology?: never}
    | {weights?: never; topology: {offsets: string; neighbors: string; weights: string}}
  );

/** GPU Louvain operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUNetworkLouvainOperation extends GPUContributorOperation {
  constructor(props: NetworkLouvainOperationProps) {
    super({
      id: props.id ?? 'network-louvain',
      inputs: {
        ...(props.topology
          ? {
              offsets: props.topology.offsets,
              neighbors: props.topology.neighbors,
              weights: props.topology.weights
            }
          : {weights: props.weights!}),
        parameters: props.parameters
      },
      outputs: {
        communities: {value: props.output.communities, format: 'uint32', length: props.nodeCount},
        objectiveTrace: {
          value: props.output.objectiveTrace,
          format: 'float32',
          length: 1 + (props.maximumLevels ?? 8) * (props.maximumIterations ?? 16)
        },
        summary: {value: props.output.summary, format: 'uint32', length: 4},
        ...Object.fromEntries(
          Object.entries(props.output.status ?? {}).map(([name, value]) => [
            name,
            {value, format: 'uint32' as const, length: 1}
          ])
        )
      },
      createContributor: getView =>
        new GPUNetworkLouvain({
          ...props,
          ...(props.topology
            ? {
                topology: {
                  offsets: getView<'uint32'>('offsets'),
                  neighbors: getView<'uint32'>('neighbors'),
                  weights: getView<'float32'>('weights')
                },
                weights: undefined
              }
            : {weights: getView<'float32'>('weights'), topology: undefined}),
          parameters: getView<'float32'>('parameters'),
          output: {
            communities: getView<'uint32'>('communities'),
            objectiveTrace: getView<'float32'>('objectiveTrace'),
            summary: getView<'uint32'>('summary'),
            status: Object.fromEntries(
              Object.keys(props.output.status ?? {}).map(name => [name, getView<'uint32'>(name)])
            )
          }
        })
    });
  }
}
