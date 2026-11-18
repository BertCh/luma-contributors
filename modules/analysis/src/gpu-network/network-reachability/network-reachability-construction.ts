// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import type {GPUNetworkReachabilityProps} from './gpu-network-reachability';
import type {NetworkReachabilityOperationProps} from './gpu-network-reachability-operation';

/** Capture named compile-time metadata and port names; view contents remain live. @internal */
export function snapshotNetworkReachabilityConstructionProps<
  Props extends GPUNetworkReachabilityProps | NetworkReachabilityOperationProps
>(props: Props): Props {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'topology',
    'startNodes',
    'startCosts',
    'startCount',
    'maximumIterations',
    'localIterations',
    'maximumTieIterations',
    'activeIterations',
    'costLimit',
    'laneCount',
    'costBreaks',
    'output',
    'nodeCount',
    'costBreakCount'
  ] as (keyof Props)[]);
  const output = snapshotConstructionFields(snapshot.output, [
    'costs',
    'predecessors',
    'bands',
    'bandCounts',
    'unresolvedCount',
    'status'
  ]);
  return Object.freeze({
    ...snapshot,
    topology:
      snapshot.topology &&
      snapshotConstructionFields(snapshot.topology, [
        'offsets',
        'neighbors',
        'weights',
        'nodePositions'
      ]),
    output: Object.freeze({
      ...output,
      ...(output.status
        ? {
            status: snapshotConstructionFields(output.status, [
              'converged',
              'iterationCount',
              'iterationLimitReached',
              'relaxationCount',
              'reachedEdgeCount'
            ])
          }
        : {})
    })
  }) as Props;
}
