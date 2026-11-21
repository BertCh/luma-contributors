// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import type {S2CapCoveringOperationProps} from './gpu-s2-cap-covering-operation';
import type {S2PolygonCoveringOperationProps} from './gpu-s2-polygon-covering-operation';

/** Capture declared fields even when a caller provides prototype getters. @internal */
export function snapshotS2CapOperationProps(
  props: S2CapCoveringOperationProps
): S2CapCoveringOperationProps {
  const captured = snapshotConstructionFields(props, [
    'id',
    'caps',
    'capCount',
    'level',
    'mode',
    'capacity',
    'tolerance',
    'refinement',
    'frontierCapacity',
    'spatialContext',
    'output'
  ]);
  return Object.freeze({
    ...captured,
    output: snapshotConstructionFields(captured.output, ['cells', 'counts', 'status']),
    ...(captured.spatialContext
      ? {spatialContext: snapshotConstructionSpatialContext(captured.spatialContext)}
      : {})
  });
}

/** Copy the pole and metadata while preserving resource identities and input names. @internal */
export function snapshotS2PolygonOperationProps(
  props: S2PolygonCoveringOperationProps
): S2PolygonCoveringOperationProps {
  const captured = snapshotConstructionFields(props, [
    'id',
    'positions',
    'ringOffsets',
    'positionCount',
    'ringOffsetCount',
    'minimumLevel',
    'maximumLevel',
    'mode',
    'capacity',
    'maximumVisitedCells',
    'tolerance',
    'refinement',
    'hemisphereCenter',
    'complement',
    'spatialContext',
    'output'
  ]);
  return Object.freeze({
    ...captured,
    output: snapshotConstructionFields(captured.output, [
      'cells',
      'counts',
      'status',
      'visitedCount'
    ]),
    ...(captured.hemisphereCenter
      ? {
          hemisphereCenter: Object.freeze([...captured.hemisphereCenter]) as readonly [
            number,
            number,
            number
          ]
        }
      : {}),
    ...(captured.spatialContext
      ? {spatialContext: snapshotConstructionSpatialContext(captured.spatialContext)}
      : {})
  });
}
