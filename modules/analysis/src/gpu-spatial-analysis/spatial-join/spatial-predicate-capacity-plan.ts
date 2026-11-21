// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getDefaultSpatialJoinCandidateCapacity} from './spatial-join-capacity-plan';
import {
  planSpatialJoinCandidatePages,
  SPATIAL_JOIN_PAGE_QUERY_WORD_COUNT
} from './spatial-join-candidate-pages';

/** Signed-compatible cutoff below the saturated uint32 requirement sentinel. */
const GPU_SPATIAL_PREDICATE_JOIN_MAXIMUM_CANDIDATES = 0x7ffffffe;

/** @internal Construction-time global candidate cutoff, shared by every candidate scratch page. */
export function resolveSpatialPredicateCandidateCapacity(
  id: string,
  leftCount: number,
  rightCount: number,
  capacity?: number
): number {
  const resolved = capacity ?? getDefaultSpatialJoinCandidateCapacity(leftCount, rightCount);
  validateCapacity(
    id,
    'candidateCapacity',
    resolved,
    1,
    GPU_SPATIAL_PREDICATE_JOIN_MAXIMUM_CANDIDATES
  );
  return resolved;
}

/** @internal Device-bounded page selection and cooperative-dispatch address validation. */
export function getSpatialPredicateCandidatePagePlan(
  device: Device,
  id: string,
  capacity: number,
  tileCapacity: number | undefined,
  robustPointPolygon: boolean,
  hasDistanceView: boolean,
  cooperativeLaneCount: number
) {
  const plan = planSpatialJoinCandidatePages(
    device,
    id,
    capacity,
    tileCapacity,
    robustPointPolygon ? 16 : 8,
    robustPointPolygon ? 8 : hasDistanceView ? 8 : 0,
    1,
    cooperativeLaneCount
  );
  validateCapacity(id, 'candidate tile invocations', plan.capacity * cooperativeLaneCount);
  return plan;
}

/** @internal Logical binding requirements, before graph declarations or persistent allocations. */
export function getSpatialPredicateScratchByteLengths(props: {
  leftCount: number;
  rightCount: number;
  tileCapacity: number;
  pairCapacity: number;
  leafCapacity: number;
  prepared: boolean;
  hasDistanceView: boolean;
  hasWeightsDistances: boolean;
  hasWeights: boolean;
  hasUnmatched: boolean;
  hasCandidateTruncatedCount: boolean;
  robustPointPolygon: boolean;
  liveGeographicRingOffsetCount: number;
}): Record<string, number> {
  const {leftCount, rightCount, tileCapacity, pairCapacity, leafCapacity} = props;
  return {
    state: 6 * 4,
    'candidate dispatch': 3 * 4,
    'candidate pairs': (tileCapacity + (props.hasDistanceView ? 1 : 0)) * 8,
    'candidate flags/offsets/matrices': (tileCapacity + 1) * 4,
    'candidate probe counts/offsets/scan': (leftCount + SPATIAL_JOIN_PAGE_QUERY_WORD_COUNT) * 4,
    'left feature bounds/rings': leftCount * 8,
    'right feature bounds/rings': props.prepared ? 0 : rightCount * 8,
    'BVH node bounds/children': props.prepared ? 0 : (2 * leafCapacity - 1) * 8,
    'BVH leaf IDs': props.prepared ? 0 : leafCapacity * 4,
    'matched pair IDs': pairCapacity * 4,
    'distance pairs': props.hasWeightsDistances ? pairCapacity * 8 : 0,
    'left match counts': props.hasWeights || props.hasUnmatched ? (leftCount + 1) * 4 : 0,
    'anti flags/offsets': props.hasUnmatched ? leftCount * 4 : 0,
    'candidate truncated':
      props.hasUnmatched || props.hasCandidateTruncatedCount ? leftCount * 4 : 0,
    'robust pair points': props.robustPointPolygon ? (2 * tileCapacity + 1) * 8 : 0,
    'robust geometry offsets': props.robustPointPolygon ? (2 * tileCapacity + 2) * 4 : 0,
    'live geographic ring offsets': props.liveGeographicRingOffsetCount * 4
  };
}
