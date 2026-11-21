// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import type {GPUNetworkCoarseningProps} from './gpu-network-coarsening';

/** Capture topology descriptors and nested writer selections, retaining graph views. @internal */
export function snapshotCoarseningConstructionProps(
  props: GPUNetworkCoarseningProps
): GPUNetworkCoarseningProps {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'topology',
    'directed',
    'nodeMask',
    'edgeMask',
    'labels',
    'nodeValues',
    'groupCapacity',
    'fixedPointScale',
    'output'
  ]);
  const output = snapshotConstructionFields(snapshot.output, [
    'groupNodeCount',
    'groupIntraEdgeCount',
    'groupIntraWeight',
    'groupCentroid',
    'groupBounds',
    'groupValueSum',
    'edges',
    'edgeTargetGroups',
    'edgeCounts',
    'edgeWeights',
    'summary'
  ]);
  const edges = snapshotConstructionFields(output.edges, ['ids', 'status']);
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
      edges: Object.freeze({
        ...edges,
        status: snapshotConstructionFields(edges.status, [
          'count',
          'requiredCount',
          'overflow',
          'candidateOverflow'
        ])
      })
    })
  });
}
