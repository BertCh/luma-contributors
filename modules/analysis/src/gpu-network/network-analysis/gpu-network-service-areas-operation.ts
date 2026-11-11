// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUNetworkServiceAreas} from './gpu-network-service-areas';

/** Graph-value properties; node and facility capacity counts are compile-time. */
export type NetworkServiceAreasOperationProps = {
  id?: string;
  nodeCount: number;
  facilityCapacity: number;
  topology: {offsets: string; neighbors: string; weights: string; nodePositions?: string};
  facilities: string;
  facilityCosts?: string;
  facilityCount?: number | string;
  costLimit?: string;
  maximumIterations?: number;
  maximumLabelIterations?: number;
  localIterations?: number;
  activeIterations?: string;
  maximumTieIterations?: number;
  output: {
    assignments: string;
    costs?: string;
    predecessors?: string;
    facilityNodeCounts?: string;
    facilityCostSums?: string;
    status?: {converged?: string};
  };
};
/** Vector operation form. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUNetworkServiceAreasOperation extends GPUContributorOperation {
  constructor(props: NetworkServiceAreasOperationProps) {
    const inputs: Record<string, string> = {
      offsets: props.topology.offsets,
      neighbors: props.topology.neighbors,
      weights: props.topology.weights,
      facilities: props.facilities
    };
    if (props.topology.nodePositions !== undefined)
      inputs['nodePositions'] = props.topology.nodePositions;
    for (const port of ['facilityCosts', 'costLimit', 'activeIterations'] as const)
      if (props[port] !== undefined) inputs[port] = props[port];
    if (typeof props.facilityCount === 'string') inputs['facilityCount'] = props.facilityCount;
    const outputs: Record<string, GPUContributorOperationOutput> = {};
    for (const [port, value] of Object.entries({...props.output, ...props.output.status})) {
      if (typeof value !== 'string') continue;
      outputs[port] = {
        value,
        format: ['costs', 'facilityCostSums'].includes(port) ? 'float32' : 'uint32',
        length:
          port === 'converged'
            ? 1
            : ['facilityNodeCounts', 'facilityCostSums'].includes(port)
              ? props.facilityCapacity
              : props.nodeCount
      };
    }
    super({
      id: props.id ?? 'network-service-areas',
      inputs,
      outputs,
      createContributor: getView =>
        new GPUNetworkServiceAreas({
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
          facilities: getView<'uint32'>('facilities'),
          facilityCosts:
            props.facilityCosts === undefined ? undefined : getView<'float32'>('facilityCosts'),
          facilityCount:
            typeof props.facilityCount === 'string'
              ? getView<'uint32'>('facilityCount')
              : props.facilityCount,
          costLimit: props.costLimit === undefined ? undefined : getView<'float32'>('costLimit'),
          activeIterations:
            props.activeIterations === undefined
              ? undefined
              : getView<'uint32'>('activeIterations'),
          maximumIterations: props.maximumIterations,
          maximumLabelIterations: props.maximumLabelIterations,
          maximumTieIterations: props.maximumTieIterations,
          localIterations: props.localIterations,
          output: {
            assignments: getView<'uint32'>('assignments'),
            costs: props.output.costs === undefined ? undefined : getView<'float32'>('costs'),
            predecessors:
              props.output.predecessors === undefined
                ? undefined
                : getView<'uint32'>('predecessors'),
            facilityNodeCounts:
              props.output.facilityNodeCounts === undefined
                ? undefined
                : getView<'uint32'>('facilityNodeCounts'),
            facilityCostSums:
              props.output.facilityCostSums === undefined
                ? undefined
                : getView<'float32'>('facilityCostSums'),
            status: {
              converged:
                props.output.status?.converged === undefined
                  ? undefined
                  : getView<'uint32'>('converged')
            }
          }
        })
    });
  }
}
