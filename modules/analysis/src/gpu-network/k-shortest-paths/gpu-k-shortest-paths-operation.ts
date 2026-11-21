// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUKShortestPaths, type GPUKShortestPathsProps} from './gpu-k-shortest-paths';
import {getGPUKShortestPathsOutputLayout} from './k-shortest-paths-output';

/** Graph-value form of GPUKShortestPaths. counts has queryCount rows; status names map to batch scalars. */
export type KShortestPathsOperationProps = Omit<
  GPUKShortestPathsProps,
  'topology' | 'sources' | 'targets' | 'output'
> & {
  queryCount: number;
  topology: {offsets: string; neighbors: string; weights: string; nodePositions?: string};
  sources: string;
  targets: string;
  output: {
    vertices: string;
    edges: string;
    lengths: string;
    costs: string;
    counts: string;
    status: {
      converged: string;
      iterationLimitReached: string;
      candidateOverflow: string;
      iterationCount: string;
    };
  };
};

/** Vector graph operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUKShortestPathsOperation extends GPUContributorOperation {
  constructor(props: KShortestPathsOperationProps) {
    const id = props.id ?? 'k-shortest-paths';
    const queryCount = props.queryCount;
    const routeCount = props.routeCount;
    const maximumPathLength = props.maximumPathLength ?? 64;
    const candidateCapacity = props.candidateCapacity;
    const maximumIterations = props.maximumIterations;
    const outputs: Record<string, GPUContributorOperationOutput> = {};
    const {status: statusLayout, ...columnLayout} = getGPUKShortestPathsOutputLayout({
      id,
      queryCount,
      routeCount,
      maximumPathLength
    });
    const outputByName = {...props.output, ...props.output.status};
    for (const [port, descriptor] of Object.entries({...columnLayout, ...statusLayout})) {
      outputs[port] = {
        value: outputByName[port as keyof typeof outputByName] as string,
        ...descriptor
      };
    }
    super({
      emptyOutputPorts: ['vertices', 'edges', 'lengths', 'costs', 'counts', 'complete'],
      id,
      inputs: {
        offsets: props.topology.offsets,
        neighbors: props.topology.neighbors,
        weights: props.topology.weights,
        ...(props.topology.nodePositions === undefined
          ? {}
          : {nodePositions: props.topology.nodePositions}),
        sources: props.sources,
        targets: props.targets
      },
      outputs,
      createContributor: getView =>
        new GPUKShortestPaths({
          id,
          routeCount,
          maximumPathLength,
          candidateCapacity,
          maximumIterations,
          topology: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights'),
            nodePositions:
              props.topology.nodePositions === undefined
                ? undefined
                : getView<'float32x2'>('nodePositions')
          },
          sources: getView<'uint32'>('sources'),
          targets: getView<'uint32'>('targets'),
          output: {
            vertices: getView<'uint32'>('vertices'),
            edges: getView<'uint32'>('edges'),
            lengths: getView<'uint32'>('lengths'),
            costs: getView<'float32'>('costs'),
            counts: getView<'uint32'>('counts'),
            status: {
              converged: getView<'uint32'>('converged'),
              iterationLimitReached: getView<'uint32'>('iterationLimitReached'),
              candidateOverflow: getView<'uint32'>('candidateOverflow'),
              iterationCount: getView<'uint32'>('iterationCount')
            }
          }
        })
    });
  }
}
