// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getNetworkOptimizationStatusOutputs,
  type NetworkOptimizationStatusValues
} from './network-optimization-status';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUNetworkFlow,
  validateNetworkFlowOptions,
  type GPUNetworkFlowProps
} from './gpu-network-flow';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateNetworkFlowPathShape, snapshotNetworkFlowPaths} from './network-flow-paths';

/** Shared operation properties, with graph value names replacing packed views. */
export type NetworkFlowOperationProps = Pick<
  GPUNetworkFlowProps,
  | 'id'
  | 'nodeCount'
  | 'solver'
  | 'mode'
  | 'maximumCostPhases'
  | 'maximumAugmentations'
  | 'maximumRelaxationRounds'
  | 'maximumRelabels'
> & {
  /** Number of original directed edges; zero declares an empty edge-flow output. */
  edgeCount: number;
  /** Number of fixed-width path rows when output.paths is supplied (1..1024).
   * Decomposition requires nodeCount <= 128 and edgeCount <= 1024, including super arcs. */
  pathCapacity?: number;
  edges: string;
  capacities: string;
  costs?: string;
  /** Optional CSR connectivity, aligned with the original edge order. Topology weights are
   * unused by the solver; original integral costs remain in costs. */
  topology?: {offsets: string; neighbors: string};
  /** uint32: source, sink, required flow (0xffffffff maximizes). */
  parameters: string;
  output: {
    flow: string;
    cut: string;
    summary: string;
    exactCost?: string;
    paths?: {
      edgeRows: string;
      lengths: string;
      volumes: string;
      status: {
        count: string;
        requiredCount: string;
        overflow: string;
        invalidCount: string;
        converged: string;
        iterationLimitReached: string;
      };
    };
    status?: NetworkOptimizationStatusValues;
  };
};

/** Capture the declared shape and named port maps for both operation forms. @internal */
export function snapshotNetworkFlowOperationProps(
  props: NetworkFlowOperationProps
): NetworkFlowOperationProps {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'nodeCount',
    'edgeCount',
    'pathCapacity',
    'solver',
    'mode',
    'maximumCostPhases',
    'maximumAugmentations',
    'maximumRelaxationRounds',
    'maximumRelabels',
    'edges',
    'capacities',
    'costs',
    'parameters',
    'topology',
    'output'
  ]);
  const output = snapshotConstructionFields(snapshot.output, [
    'flow',
    'cut',
    'summary',
    'exactCost',
    'paths',
    'status'
  ]);
  return Object.freeze({
    ...snapshot,
    topology:
      snapshot.topology && snapshotConstructionFields(snapshot.topology, ['offsets', 'neighbors']),
    output: Object.freeze({
      ...output,
      paths: snapshotNetworkFlowPaths(output.paths, snapshot.id ?? 'network-flow'),
      status:
        output.status &&
        snapshotConstructionFields(
          output.status,
          Array.from(
            new Set([
              ...Object.keys(output.status),
              'invalidCount',
              'iterationCount',
              'converged',
              'iterationLimitReached'
            ])
          ) as (keyof NetworkOptimizationStatusValues)[]
        )
    })
  });
}

/** GPU flow operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUNetworkFlowOperation extends GPUContributorOperation {
  constructor(props: NetworkFlowOperationProps) {
    props = snapshotNetworkFlowOperationProps(props);
    validateNetworkFlowOptions(
      props.id ?? 'network-flow',
      props,
      props.edgeCount,
      props.costs !== undefined
    );
    validateRequirement(
      props.id ?? 'network-flow',
      Boolean(props.output.paths) === (props.pathCapacity !== undefined),
      'pathCapacity and output.paths must be supplied together'
    );
    if (props.pathCapacity !== undefined)
      validateNetworkFlowPathShape(
        props.id ?? 'network-flow',
        props.nodeCount,
        props.edgeCount,
        props.pathCapacity
      );
    super({
      id: props.id ?? 'network-flow',
      emptyOutputPorts: ['flow'],
      inputs: {
        edges: props.edges,
        capacities: props.capacities,
        parameters: props.parameters,
        ...(props.costs === undefined ? {} : {costs: props.costs}),
        ...(props.topology
          ? {topologyOffsets: props.topology.offsets, topologyNeighbors: props.topology.neighbors}
          : {})
      },
      outputs: {
        flow: {value: props.output.flow, format: 'uint32', length: props.edgeCount},
        cut: {value: props.output.cut, format: 'uint32', length: props.nodeCount},
        summary: {value: props.output.summary, format: 'uint32', length: 8},
        ...(props.output.paths
          ? {
              pathEdgeRows: {
                value: props.output.paths.edgeRows,
                format: 'uint32' as const,
                length: props.pathCapacity! * (props.nodeCount - 1)
              },
              pathLengths: {
                value: props.output.paths.lengths,
                format: 'uint32' as const,
                length: props.pathCapacity!
              },
              pathVolumes: {
                value: props.output.paths.volumes,
                format: 'uint32' as const,
                length: props.pathCapacity!
              },
              pathCount: {
                value: props.output.paths.status.count,
                format: 'uint32' as const,
                length: 1
              },
              pathRequiredCount: {
                value: props.output.paths.status.requiredCount,
                format: 'uint32' as const,
                length: 1
              },
              pathOverflow: {
                value: props.output.paths.status.overflow,
                format: 'uint32' as const,
                length: 1
              },
              pathInvalidCount: {
                value: props.output.paths.status.invalidCount,
                format: 'uint32' as const,
                length: 1
              },
              pathConverged: {
                value: props.output.paths.status.converged,
                format: 'uint32' as const,
                length: 1
              },
              pathIterationLimitReached: {
                value: props.output.paths.status.iterationLimitReached,
                format: 'uint32' as const,
                length: 1
              }
            }
          : {}),
        ...getNetworkOptimizationStatusOutputs(props.id ?? 'network-flow', props.output.status),
        ...(props.output.exactCost === undefined
          ? {}
          : {exactCost: {value: props.output.exactCost, format: 'uint32x2' as const, length: 1}})
      },
      createContributor: getView =>
        new GPUNetworkFlow({
          ...props,
          edges: getView<'uint32x2'>('edges'),
          capacities: getView<'uint32'>('capacities'),
          parameters: getView<'uint32'>('parameters'),
          costs: props.costs === undefined ? undefined : getView<'uint32'>('costs'),
          topology: props.topology
            ? {
                offsets: getView<'uint32'>('topologyOffsets'),
                neighbors: getView<'uint32'>('topologyNeighbors')
              }
            : undefined,
          output: {
            flow: getView<'uint32'>('flow'),
            cut: getView<'uint32'>('cut'),
            summary: getView<'uint32'>('summary'),
            paths: props.output.paths
              ? {
                  edgeRows: getView<'uint32'>('pathEdgeRows'),
                  lengths: getView<'uint32'>('pathLengths'),
                  volumes: getView<'uint32'>('pathVolumes'),
                  status: {
                    count: getView<'uint32'>('pathCount'),
                    requiredCount: getView<'uint32'>('pathRequiredCount'),
                    overflow: getView<'uint32'>('pathOverflow'),
                    invalidCount: getView<'uint32'>('pathInvalidCount'),
                    converged: getView<'uint32'>('pathConverged'),
                    iterationLimitReached: getView<'uint32'>('pathIterationLimitReached')
                  }
                }
              : undefined,
            status: Object.fromEntries(
              Object.entries(props.output.status ?? {}).flatMap(([name, value]) =>
                value === undefined ? [] : [[name, getView<'uint32'>(name)]]
              )
            ),
            exactCost:
              props.output.exactCost === undefined ? undefined : getView<'uint32x2'>('exactCost')
          }
        })
    });
  }
}
