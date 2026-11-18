// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUNetworkPathExtraction} from './gpu-network-path-extraction';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Graph names and caller-selected capacity of one compact path output. */
export type NetworkPathExtractionCompactOperationOutput = {
  capacity: number;
  ids: string;
  status: {count: string; overflow: string; requiredCount: string; candidateOverflow?: string};
  lineOffsets?: string;
};

export type NetworkPathExtractionNodeOperationOutput =
  NetworkPathExtractionCompactOperationOutput & {
    nodeRows?: string;
    pathCosts?: string;
    pathFound?: string;
  };

export type NetworkPathExtractionEdgeOperationOutput =
  NetworkPathExtractionCompactOperationOutput & {
    edgeRows?: string;
  };

/** Shared graph-value properties; every number and optional-port presence is compile-time. */
export type NetworkPathExtractionOperationProps = {
  id?: string;
  nodeCount: number;
  targetCapacity: number;
  predecessors: string;
  costs: string;
  targets: string;
  /** Numeric compile-time active count, or uint32 one-row graph value. */
  targetCount?: number | string;
  predecessorsComplete?: string;
  maximumPathLength?: number;
  nodeIds?: string;
  output: NetworkPathExtractionNodeOperationOutput;
  edges?: {
    edgeCount: number;
    topology: {offsets: string; neighbors: string; weights: string; nodePositions?: string};
    edgeIds?: string;
    output: NetworkPathExtractionEdgeOperationOutput;
  };
};

/** Capture role-specific ports and their nested maps for both backends. @internal */
export function snapshotNetworkPathExtractionOperationProps(
  props: NetworkPathExtractionOperationProps
): NetworkPathExtractionOperationProps {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'nodeCount',
    'targetCapacity',
    'predecessors',
    'costs',
    'targets',
    'targetCount',
    'predecessorsComplete',
    'maximumPathLength',
    'nodeIds',
    'output',
    'edges'
  ]);
  const id = snapshot.id ?? 'network-path-extraction';
  validateRequirement(
    id,
    !('edgeRows' in snapshot.output),
    'output.edgeRows is only supported under edges.output'
  );
  const output = snapshotConstructionFields(snapshot.output, [
    'capacity',
    'ids',
    'nodeRows',
    'status',
    'lineOffsets',
    'pathCosts',
    'pathFound'
  ]);
  const edges =
    snapshot.edges &&
    snapshotConstructionFields(snapshot.edges, ['edgeCount', 'topology', 'edgeIds', 'output']);
  if (edges)
    validateRequirement(
      id,
      !('nodeRows' in edges.output),
      'edges.output.nodeRows is only supported under output'
    );
  const edgeOutput =
    edges &&
    snapshotConstructionFields(edges.output, [
      'capacity',
      'ids',
      'edgeRows',
      'status',
      'lineOffsets'
    ]);
  const statusFields = ['count', 'overflow', 'requiredCount', 'candidateOverflow'] as const;
  return Object.freeze({
    ...snapshot,
    output: Object.freeze({
      ...output,
      status: snapshotConstructionFields(output.status, statusFields)
    }),
    edges:
      edges && edgeOutput
        ? Object.freeze({
            ...edges,
            topology: snapshotConstructionFields(edges.topology, [
              'offsets',
              'neighbors',
              'weights',
              'nodePositions'
            ]),
            output: Object.freeze({
              ...edgeOutput,
              status: snapshotConstructionFields(edgeOutput.status, statusFields)
            })
          })
        : undefined
  });
}

/**
 * GPU path extraction as a vector operation, including optional edge extraction. Compile with
 * GPU_CONTRIBUTOR_OPERATION_LOWERING. Input contents are per-frame. Output capacities must be
 * positive; omit zero-length optional outputs when targetCapacity is zero (executor limit).
 */
export class GPUNetworkPathExtractionOperation extends GPUContributorOperation {
  constructor(props: NetworkPathExtractionOperationProps) {
    props = snapshotNetworkPathExtractionOperationProps(props);
    const inputs: Record<string, string> = {
      predecessors: props.predecessors,
      costs: props.costs,
      targets: props.targets
    };
    if (typeof props.targetCount === 'string') inputs['targetCount'] = props.targetCount;
    if (props.predecessorsComplete !== undefined)
      inputs['predecessorsComplete'] = props.predecessorsComplete;
    if (props.nodeIds !== undefined) inputs['nodeIds'] = props.nodeIds;
    const outputs: Record<string, GPUContributorOperationOutput> = {};
    const addCompactOutput = (
      output: NetworkPathExtractionCompactOperationOutput,
      prefix: string
    ) => {
      outputs[`${prefix}ids`] = {value: output.ids, format: 'uint32', length: output.capacity};
      for (const field of ['count', 'overflow', 'requiredCount'] as const)
        outputs[`${prefix}${field}`] = {value: output.status[field], format: 'uint32', length: 1};
      if (output.status.candidateOverflow !== undefined)
        outputs[`${prefix}candidateOverflow`] = {
          value: output.status.candidateOverflow,
          format: 'uint32',
          length: 1
        };
      if (output.lineOffsets !== undefined)
        outputs[`${prefix}lineOffsets`] = {
          value: output.lineOffsets,
          format: 'uint32',
          length: props.targetCapacity + 1
        };
    };
    addCompactOutput(props.output, '');
    if (props.output.nodeRows !== undefined)
      outputs['nodeRows'] = {
        value: props.output.nodeRows,
        format: 'uint32',
        length: props.output.capacity
      };
    if (props.output.pathCosts !== undefined)
      outputs['pathCosts'] = {
        value: props.output.pathCosts,
        format: 'float32',
        length: props.targetCapacity
      };
    if (props.output.pathFound !== undefined)
      outputs['pathFound'] = {
        value: props.output.pathFound,
        format: 'uint32',
        length: props.targetCapacity
      };
    if (props.edges) {
      inputs['offsets'] = props.edges.topology.offsets;
      inputs['neighbors'] = props.edges.topology.neighbors;
      inputs['weights'] = props.edges.topology.weights;
      if (props.edges.topology.nodePositions !== undefined)
        inputs['nodePositions'] = props.edges.topology.nodePositions;
      if (props.edges.edgeIds !== undefined) inputs['edgeIds'] = props.edges.edgeIds;
      addCompactOutput(props.edges.output, 'edge-');
      if (props.edges.output.edgeRows !== undefined)
        outputs['edge-edgeRows'] = {
          value: props.edges.output.edgeRows,
          format: 'uint32',
          length: props.edges.output.capacity
        };
    }
    super({
      id: props.id ?? 'network-path-extraction',
      inputs,
      outputs,
      createContributor: getView => {
        const getCompactOutput = (
          output: NetworkPathExtractionCompactOperationOutput,
          prefix: string
        ) => ({
          ids: getView<'uint32'>(`${prefix}ids`),
          status: {
            count: getView<'uint32'>(`${prefix}count`),
            overflow: getView<'uint32'>(`${prefix}overflow`),
            candidateOverflow:
              output.status.candidateOverflow === undefined
                ? undefined
                : getView<'uint32'>(`${prefix}candidateOverflow`),
            requiredCount: getView<'uint32'>(`${prefix}requiredCount`)
          },
          lineOffsets:
            output.lineOffsets === undefined ? undefined : getView<'uint32'>(`${prefix}lineOffsets`)
        });
        return new GPUNetworkPathExtraction({
          id: props.id,
          predecessors: getView<'uint32'>('predecessors'),
          costs: getView<'float32'>('costs'),
          targets: getView<'uint32'>('targets'),
          targetCount:
            typeof props.targetCount === 'string'
              ? getView<'uint32'>('targetCount')
              : props.targetCount,
          predecessorsComplete:
            props.predecessorsComplete === undefined
              ? undefined
              : getView<'uint32'>('predecessorsComplete'),
          maximumPathLength: props.maximumPathLength,
          nodeIds: props.nodeIds === undefined ? undefined : getView<'uint32'>('nodeIds'),
          output: {
            ...getCompactOutput(props.output, ''),
            nodeRows:
              props.output.nodeRows === undefined ? undefined : getView<'uint32'>('nodeRows'),
            pathCosts:
              props.output.pathCosts === undefined ? undefined : getView<'float32'>('pathCosts'),
            pathFound:
              props.output.pathFound === undefined ? undefined : getView<'uint32'>('pathFound')
          },
          edges:
            props.edges === undefined
              ? undefined
              : {
                  topology: {
                    offsets: getView<'uint32'>('offsets'),
                    neighbors: getView<'uint32'>('neighbors'),
                    weights: getView<'float32'>('weights'),
                    nodePositions:
                      props.edges.topology.nodePositions === undefined
                        ? undefined
                        : getView<'float32x2'>('nodePositions')
                  },
                  edgeIds:
                    props.edges.edgeIds === undefined ? undefined : getView<'uint32'>('edgeIds'),
                  output: {
                    ...getCompactOutput(props.edges.output, 'edge-'),
                    edgeRows:
                      props.edges.output.edgeRows === undefined
                        ? undefined
                        : getView<'uint32'>('edge-edgeRows')
                  }
                }
        });
      }
    });
  }
}
