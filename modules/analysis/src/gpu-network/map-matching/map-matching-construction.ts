// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import type {GPUMapMatchingProps} from './gpu-map-matching';
import type {MapMatchingOperationProps} from './gpu-map-matching-operation';

/** Capture the shared model, ports and nested metadata before either backend validates it. @internal */
export function snapshotMapMatchingConstructionProps<
  Props extends Partial<GPUMapMatchingProps> | MapMatchingOperationProps
>(props: Props): Props {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'spatialContext',
    'positions',
    'trackOffsets',
    'parameters',
    'topology',
    'candidateCount',
    'cellSize',
    'bounds',
    'entryCapacity',
    'routeNodeBudget',
    'minimumFixSpacingSigmas',
    'traversedEdgeCapacity',
    'routeTable',
    'output',
    'pointCount',
    'trackCount',
    'nodeCount',
    'edgeCount'
  ] as (keyof Props)[]);
  const bounds =
    snapshot.bounds && snapshotConstructionFields(snapshot.bounds, ['minimum', 'maximum']);
  const routeTable =
    snapshot.routeTable &&
    snapshotConstructionFields(snapshot.routeTable, [
      'offsets',
      'targets',
      'costs',
      'costLimit',
      'status',
      'entryCount'
    ] as (keyof typeof snapshot.routeTable)[]);
  const output =
    snapshot.output &&
    snapshotConstructionFields(snapshot.output, [
      'matchedEdges',
      'matchedFractions',
      'matchedOffsets',
      'snapDistances',
      'snappedPositions',
      'breaks',
      'trackLogLikelihoods',
      'matchedCount',
      'breakCount',
      'status',
      'traversedEdges'
    ]);
  const traversedEdges =
    output?.traversedEdges &&
    snapshotConstructionFields(output.traversedEdges, ['edges', 'counts', 'status']);
  return Object.freeze({
    ...snapshot,
    ...(snapshot.spatialContext
      ? {spatialContext: snapshotConstructionSpatialContext(snapshot.spatialContext)}
      : {}),
    ...(snapshot.topology
      ? {
          topology: snapshotConstructionFields(snapshot.topology, [
            'offsets',
            'neighbors',
            'weights',
            'nodePositions'
          ])
        }
      : {}),
    ...(bounds
      ? {
          bounds: Object.freeze({
            ...bounds,
            minimum: Object.freeze([...bounds.minimum]),
            maximum: Object.freeze([...bounds.maximum])
          })
        }
      : {}),
    ...(routeTable
      ? {
          routeTable: Object.freeze({
            ...routeTable,
            ...(routeTable.status
              ? {status: snapshotConstructionFields(routeTable.status, ['complete'])}
              : {})
          })
        }
      : {}),
    ...(output
      ? {
          output: Object.freeze({
            ...output,
            ...(output.status
              ? {
                  status: snapshotConstructionFields(output.status, [
                    'overflow',
                    'candidateOverflow',
                    'routeOverflow'
                  ])
                }
              : {}),
            ...(traversedEdges
              ? {
                  traversedEdges: Object.freeze({
                    ...traversedEdges,
                    ...(traversedEdges.status
                      ? {
                          status: snapshotConstructionFields(traversedEdges.status, [
                            'overflow',
                            'candidateOverflow'
                          ])
                        }
                      : {})
                  })
                }
              : {})
          })
        }
      : {})
  }) as Props;
}
