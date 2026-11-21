// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialPredicateJoinOperationProps} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';
import type {CPUSpatialJoinGeometry} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-join';

/** Independent Shapely pair enumeration: predicate-output-oracles.oracle.py. */
export function createPredicateOracleGeometries(): {
  left: CPUSpatialJoinGeometry;
  right: CPUSpatialJoinGeometry;
} {
  return {
    left: {kind: 'points', positions: Float32Array.of(1, 1, 2, 1, 5, 5, 6, 6)},
    right: {
      kind: 'polygons',
      positions: Float32Array.of(0, 0, 2, 0, 2, 2, 0, 2, 1, 0, 3, 0, 3, 2, 1, 2),
      featureOffsets: Uint32Array.of(0, 1, 2),
      polygonOffsets: Uint32Array.of(0, 1, 2),
      ringOffsets: Uint32Array.of(0, 4, 8)
    }
  };
}

export function getPredicateOracleOperationProps(): SpatialPredicateJoinOperationProps {
  return {
    left: {kind: 'points', positions: {value: 'leftPositions', length: 4}},
    right: {
      kind: 'polygons',
      positions: {value: 'rightPositions', length: 8},
      featureOffsets: {value: 'rightFeatureOffsets', length: 3},
      polygonOffsets: {value: 'rightPolygonOffsets', length: 3},
      ringOffsets: {value: 'rightRingOffsets', length: 3}
    },
    predicate: 'intersects',
    candidateCapacity: 4,
    pairCapacity: 4,
    output: {
      leftRows: 'leftRows',
      rightRows: 'rightRows',
      count: 'count',
      requiredCount: 'requiredCount',
      overflow: 'overflow',
      candidateOverflow: 'candidateOverflow',
      candidateTruncated: 'candidateTruncated'
    }
  };
}

export function getPredicateOracleOperationInputs() {
  const {left, right} = createPredicateOracleGeometries();
  return {
    leftPositions: {format: 'float32x2' as const, length: 4, values: left.positions},
    rightPositions: {format: 'float32x2' as const, length: 8, values: right.positions},
    rightFeatureOffsets: {format: 'uint32' as const, length: 3, values: right.featureOffsets!},
    rightPolygonOffsets: {format: 'uint32' as const, length: 3, values: right.polygonOffsets!},
    rightRingOffsets: {format: 'uint32' as const, length: 3, values: right.ringOffsets!}
  };
}
