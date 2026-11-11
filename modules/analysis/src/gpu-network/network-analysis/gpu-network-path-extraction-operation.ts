// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUNetworkPathExtraction} from './gpu-network-path-extraction';

/** Graph names and caller-selected capacity of one compact path output. */
export type NetworkPathExtractionCompactOperationOutput = {
  capacity: number;
  ids: string;
  status: {count: string; overflow: string; requiredCount: string};
  lineOffsets?: string;
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
  output: NetworkPathExtractionCompactOperationOutput & {pathCosts?: string; pathFound?: string};
  edges?: {
    edgeCount: number;
    topology: {offsets: string; neighbors: string; weights: string; nodePositions?: string};
    edgeIds?: string;
    output: NetworkPathExtractionCompactOperationOutput;
  };
};

/**
 * GPU path extraction as a vector operation, including optional edge extraction. Compile with
 * GPU_CONTRIBUTOR_OPERATION_LOWERING. Input contents are per-frame. Output capacities must be
 * positive; omit zero-length optional outputs when targetCapacity is zero (executor limit).
 */
export class GPUNetworkPathExtractionOperation extends GPUContributorOperation {
  constructor(props: NetworkPathExtractionOperationProps) {
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
      if (output.lineOffsets !== undefined)
        outputs[`${prefix}lineOffsets`] = {
          value: output.lineOffsets,
          format: 'uint32',
          length: props.targetCapacity + 1
        };
    };
    addCompactOutput(props.output, '');
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
                  output: getCompactOutput(props.edges.output, 'edge-')
                }
        });
      }
    });
  }
}
