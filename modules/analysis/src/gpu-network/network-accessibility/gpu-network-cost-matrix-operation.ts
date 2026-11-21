// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUNetworkCostMatrix} from './gpu-network-cost-matrix';
import {
  resolveNetworkRoutingCostPrecision,
  type CPUNetworkRoutingCostPrecision
} from '../../analysis-core/operations/cpu-network-routing';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateNetworkCostMatrixIterations,
  snapshotNetworkCostMatrixOperationProps
} from './network-cost-matrix-construction';

/** Named graph-value status outputs shared by legacy and GPU operation props. */
export type NetworkCostMatrixOperationStatus = {
  converged?: string;
  iterationLimitReached?: string;
  sourceComplete?: string;
};

/** Graph-value properties; matrix dimensions are explicit compile-time counts. */
export type NetworkCostMatrixOperationProps = {
  id?: string;
  nodeCount: number;
  rowCount: number;
  targetCount?: number;
  topology: {offsets: string; neighbors: string; weights: string; nodePositions?: string};
  startNodes: string;
  startCosts?: string;
  startRows?: string;
  startsPerRow?: number;
  costLimit?: string;
  /** CPU default float64; GPU supports only float32-per-edge (its default). */
  costPrecision?: CPUNetworkRoutingCostPrecision;
  targetNodes?: string;
  laneCount?: number;
  maximumIterations?: number;
  localIterations?: number;
  output: {costs: string; status?: NetworkCostMatrixOperationStatus};
};

/** GPU operation inputs retaining a noded producer's live cardinality and build evidence. */
export type GPUNetworkCostMatrixOperationProps = NetworkCostMatrixOperationProps & {
  direction?: 'start-to-target' | 'target-to-start';
  nodedTopology?: {
    nodeCount: string;
    pieceCount: string;
    overflow?: string;
    uncertainCount?: string;
    piecesOverflow: string;
    candidateOverflow?: string;
  };
};

/** Vector operation form. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUNetworkCostMatrixOperation extends GPUContributorOperation {
  constructor(props: GPUNetworkCostMatrixOperationProps) {
    props = snapshotNetworkCostMatrixOperationProps(props, 'gpu');
    const id = props.id ?? 'network-cost-matrix';
    validateRequirement(
      id,
      props.direction === undefined ||
        props.direction === 'start-to-target' ||
        props.direction === 'target-to-start',
      `direction must be 'start-to-target' or 'target-to-start'; received ${String(props.direction)}`
    );
    validateRequirement(
      id,
      !props.nodedTopology || Boolean(props.output.status?.sourceComplete),
      `nodedTopology requires output.status.sourceComplete to retain source build evidence; received ${String(props.output.status?.sourceComplete)}`
    );
    const costPrecision = resolveNetworkRoutingCostPrecision(
      props.id ?? 'network-cost-matrix',
      props.costPrecision === undefined ? 'float32-per-edge' : props.costPrecision
    );
    validateRequirement(
      props.id ?? 'network-cost-matrix',
      costPrecision === 'float32-per-edge',
      `costPrecision must be float32-per-edge on GPU; received ${costPrecision}; use CPUNetworkCostMatrixOperation for float64`
    );
    validateNetworkCostMatrixIterations(props.id ?? 'network-cost-matrix', props);
    const inputs: Record<string, string> = {
      offsets: props.topology.offsets,
      neighbors: props.topology.neighbors,
      weights: props.topology.weights,
      startNodes: props.startNodes
    };
    if (props.topology.nodePositions !== undefined)
      inputs['nodePositions'] = props.topology.nodePositions;
    for (const port of ['startCosts', 'startRows', 'costLimit', 'targetNodes'] as const)
      if (props[port] !== undefined) inputs[port] = props[port];
    for (const [port, value] of Object.entries(props.nodedTopology ?? {}))
      if (value !== undefined) inputs[`noded-${port}`] = value;
    const outputs: Record<string, GPUContributorOperationOutput> = {
      costs: {
        value: props.output.costs,
        format: 'float32',
        length:
          props.rowCount *
          (props.targetNodes === undefined ? props.nodeCount : (props.targetCount ?? 0))
      }
    };
    for (const [port, value] of Object.entries(props.output.status ?? {}))
      if (value !== undefined) outputs[port] = {value, format: 'uint32', length: 1};
    super({
      id: props.id ?? 'network-cost-matrix',
      inputs,
      outputs,
      createContributor: getView =>
        new GPUNetworkCostMatrix({
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
          direction: props.direction,
          nodedTopology: props.nodedTopology
            ? {
                topology: {
                  offsets: getView<'uint32'>('offsets'),
                  neighbors: getView<'uint32'>('neighbors'),
                  weights: getView<'float32'>('weights')
                },
                nodes: {status: {count: getView<'uint32'>('noded-nodeCount')}},
                pieces: {
                  status: {
                    count: getView<'uint32'>('noded-pieceCount'),
                    overflow: getView<'uint32'>('noded-piecesOverflow'),
                    candidateOverflow:
                      props.nodedTopology.candidateOverflow === undefined
                        ? undefined
                        : getView<'uint32'>('noded-candidateOverflow')
                  }
                },
                status: {
                  overflow:
                    props.nodedTopology.overflow === undefined
                      ? undefined
                      : getView<'uint32'>('noded-overflow'),
                  uncertainCount:
                    props.nodedTopology.uncertainCount === undefined
                      ? undefined
                      : getView<'uint32'>('noded-uncertainCount')
                }
              }
            : undefined,
          startCosts: props.startCosts === undefined ? undefined : getView<'float32'>('startCosts'),
          startRows: props.startRows === undefined ? undefined : getView<'uint32'>('startRows'),
          startsPerRow: props.startsPerRow,
          costLimit: props.costLimit === undefined ? undefined : getView<'float32'>('costLimit'),
          targetNodes:
            props.targetNodes === undefined ? undefined : getView<'uint32'>('targetNodes'),
          laneCount: props.laneCount,
          maximumIterations: props.maximumIterations,
          localIterations: props.localIterations,
          output: {
            costs: getView<'float32'>('costs'),
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
