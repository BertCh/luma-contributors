// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUNetworkServiceAreas} from './gpu-network-service-areas';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  resolveNetworkRoutingCostPrecision,
  type CPUNetworkRoutingCostPrecision
} from '../../analysis-core/operations/cpu-network-routing';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

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
  /** CPU default float64; GPU supports only float32-per-edge (its default). */
  costPrecision?: CPUNetworkRoutingCostPrecision;
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
    status?: {converged?: string; iterationLimitReached?: string};
  };
};

/** Captures graph bindings and phase budgets once for both operation backends. @internal */
export function snapshotNetworkServiceAreasOperationProps(
  props: NetworkServiceAreasOperationProps
): NetworkServiceAreasOperationProps {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'nodeCount',
    'facilityCapacity',
    'topology',
    'facilities',
    'facilityCosts',
    'facilityCount',
    'costLimit',
    'costPrecision',
    'maximumIterations',
    'maximumLabelIterations',
    'localIterations',
    'activeIterations',
    'maximumTieIterations',
    'output'
  ]);
  const output = snapshotConstructionFields(snapshot.output, [
    'assignments',
    'costs',
    'predecessors',
    'facilityNodeCounts',
    'facilityCostSums',
    'status'
  ]);
  return Object.freeze({
    ...snapshot,
    topology: snapshotConstructionFields(snapshot.topology, [
      'offsets',
      'neighbors',
      'weights',
      'nodePositions'
    ]),
    output: Object.freeze({
      ...output,
      status:
        output.status === undefined
          ? undefined
          : snapshotConstructionFields(output.status, ['converged', 'iterationLimitReached'])
    })
  });
}

/** Vector operation form. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUNetworkServiceAreasOperation extends GPUContributorOperation {
  constructor(props: NetworkServiceAreasOperationProps) {
    props = snapshotNetworkServiceAreasOperationProps(props);
    const costPrecision = resolveNetworkRoutingCostPrecision(
      props.id ?? 'network-service-areas',
      props.costPrecision === undefined ? 'float32-per-edge' : props.costPrecision
    );
    validateRequirement(
      props.id ?? 'network-service-areas',
      costPrecision === 'float32-per-edge',
      `costPrecision must be float32-per-edge on GPU; received ${costPrecision}; use CPUNetworkServiceAreasOperation for float64`
    );
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
        length: ['converged', 'iterationLimitReached'].includes(port)
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
                  : getView<'uint32'>('converged'),
              iterationLimitReached:
                props.output.status?.iterationLimitReached === undefined
                  ? undefined
                  : getView<'uint32'>('iterationLimitReached')
            }
          }
        })
    });
  }
}
