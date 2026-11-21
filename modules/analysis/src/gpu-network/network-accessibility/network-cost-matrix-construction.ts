// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import type {GPUNetworkCostMatrixOperationProps} from './gpu-network-cost-matrix-operation';
import {
  GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
  GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS
} from '../network-reachability/gpu-network-reachability';

/** Shared public matrix budget limits, enforced before either backend builds work. @internal */
export function validateNetworkCostMatrixIterations(
  id: string,
  props: {maximumIterations?: number; localIterations?: number}
): void {
  validateCapacity(
    id,
    'maximumIterations',
    props.maximumIterations ?? 64,
    1,
    GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS
  );
  validateCapacity(
    id,
    'localIterations',
    props.localIterations ?? 16,
    1,
    GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS
  );
}

/** Read graph names and nested getter-backed objects once for both operation backends. @internal */
export function snapshotNetworkCostMatrixOperationProps(
  props: GPUNetworkCostMatrixOperationProps,
  backend: 'cpu' | 'gpu' = 'cpu'
): GPUNetworkCostMatrixOperationProps {
  const captured = snapshotConstructionFields(props, [
    'id',
    'nodeCount',
    'rowCount',
    'targetCount',
    'topology',
    'nodedTopology',
    'direction',
    'startNodes',
    'startCosts',
    'startRows',
    'startsPerRow',
    'costLimit',
    'costPrecision',
    'targetNodes',
    'laneCount',
    'maximumIterations',
    'localIterations',
    'output'
  ]);
  const output = snapshotConstructionFields(captured.output, ['costs', 'status']);
  const status = output.status;
  if (backend === 'cpu') {
    const id = captured.id ?? 'cpu-network-cost-matrix';
    validateRequirement(
      id,
      !captured.nodedTopology,
      'nodedTopology is currently supported by the GPU operation only; use GPUNetworkCostMatrixOperation'
    );
    validateRequirement(
      id,
      status?.sourceComplete === undefined,
      'output.status.sourceComplete is currently supported by the GPU operation only; use GPUNetworkCostMatrixOperation'
    );
  }
  return Object.freeze({
    ...captured,
    nodedTopology:
      captured.nodedTopology &&
      snapshotConstructionFields(captured.nodedTopology, [
        'nodeCount',
        'pieceCount',
        'overflow',
        'uncertainCount',
        'piecesOverflow',
        'candidateOverflow'
      ]),
    topology: snapshotConstructionFields(captured.topology, [
      'offsets',
      'neighbors',
      'weights',
      'nodePositions'
    ]),
    output: Object.freeze({
      ...output,
      status: status
        ? snapshotConstructionFields(status, [
            'converged',
            'iterationLimitReached',
            'sourceComplete'
          ])
        : undefined
    })
  });
}
