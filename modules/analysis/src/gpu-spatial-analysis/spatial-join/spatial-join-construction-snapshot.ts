// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCompactPairPort} from '../../analysis-core/contracts/index';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';

/** @internal Capture geometry descriptors, including multipart fields that validation rejects. */
export function snapshotJoinGeometry<Geometry extends {kind: string}>(
  geometry: Geometry
): Geometry {
  return snapshotConstructionFields(geometry, [
    'kind',
    'positions',
    'sourceIds',
    'lineOffsets',
    'featureOffsets',
    'polygonOffsets',
    'ringOffsets'
  ] as readonly (keyof Geometry)[]);
}

/** @internal Preserve declared bounded-result and join-specific scalar destinations. */
export function snapshotJoinStatus<Status extends object>(status: Status): Status {
  return snapshotConstructionFields(status, [
    'count',
    'requiredCount',
    'overflow',
    'candidateOverflow',
    'uncertainCount',
    'invalidCount',
    'iterationCount',
    'converged',
    'iterationLimitReached',
    'approximation',
    'candidateCount',
    'candidateTruncatedCount',
    'antimeridianUncertainCount',
    'poleUncertainCount'
  ] as unknown as readonly (keyof Status)[]);
}

/** @internal Keep row views and status bindings paired after construction. */
export function snapshotJoinPairs(pairs: GPUCompactPairPort): GPUCompactPairPort {
  const snapshot = snapshotConstructionFields(pairs, [
    'leftRows',
    'rightRows',
    'status',
    'leftIds',
    'rightIds'
  ] as readonly (keyof GPUCompactPairPort)[]);
  return Object.freeze({...snapshot, status: snapshotJoinStatus(snapshot.status)});
}

/** @internal Capture CSR fields and validated metadata without cloning any graph view. */
export function snapshotJoinWeights<Weights extends GPUSpatialWeights>(weights: Weights): Weights {
  const snapshot = snapshotConstructionFields(weights, [
    'offsets',
    'neighbors',
    'weights',
    'distances',
    'status',
    'spatialContext',
    'information'
  ]);
  const information =
    snapshot.information &&
    snapshotConstructionFields(snapshot.information, [
      'rowCount',
      'transform',
      'symmetric',
      'hasSelfWeights',
      'provenance'
    ]);
  return Object.freeze({
    ...snapshot,
    ...(snapshot.status ? {status: snapshotJoinStatus(snapshot.status)} : {}),
    ...(snapshot.spatialContext
      ? {spatialContext: snapshotConstructionSpatialContext(snapshot.spatialContext)}
      : {}),
    ...(information
      ? {
          information: Object.freeze({
            ...information,
            ...(information.provenance
              ? {
                  provenance: snapshotConstructionFields(information.provenance, [
                    'source',
                    'criterion',
                    'k',
                    'radius',
                    'order'
                  ])
                }
              : {})
          })
        }
      : {})
  });
}
