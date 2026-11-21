// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  validateCapacity,
  validatePowerOfTwo,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPUPointInPolygonJoin,
  type GPUPointInPolygonBoundaryRule
} from './gpu-point-in-polygon-join';
import type {FeatureJoinGeometryValues} from './feature-join-operation';
import {FEATURE_JOIN_MAXIMUM_PAIR_CAPACITY} from './cpu-feature-join';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  snapshotPointJoinGeometry,
  getPointJoinFeatureCount,
  getPointJoinGeometryInputs,
  resolvePointJoinGeometry
} from './point-join-operation-utils';

/** Packed planar adapter. Output names separate input feature rows from polygon application IDs. */
export type PointInPolygonJoinOperationProps = {
  id?: string;
  points: FeatureJoinGeometryValues;
  polygons: FeatureJoinGeometryValues;
  candidateCapacity: number;
  candidateTileCapacity?: number;
  leafCapacity?: number;
  matchCapacity: number;
  boundaryRule?: GPUPointInPolygonBoundaryRule;
  output: Record<
    | 'pointFeatureIds'
    | 'candidateTruncated'
    | 'ids'
    | 'count'
    | 'requiredCount'
    | 'matchOverflow'
    | 'matchCandidateOverflow'
    | 'overflow'
    | 'candidateOverflow'
    | 'candidateCount'
    | 'uncertainCount',
    string
  > & {
    pointMatchCounts?: string;
    featurePolygonMembership?: string;
    pointFeatureRows?: string;
    polygonAssignmentCounts?: string;
    polygonMembershipCounts?: string;
  };
};
/** Both operation forms use unsorted, unprepared bounds and the contributor's boundary rule. @internal */
export function createPointInPolygonJoinOperation(
  input: PointInPolygonJoinOperationProps
): GPUContributorOperation {
  const id = input.id ?? 'point-in-polygon-join-operation';
  const points = snapshotPointJoinGeometry(id, input.points),
    polygons = snapshotPointJoinGeometry(id, input.polygons);
  validateRequirement(
    id,
    points.kind === 'points' && polygons.kind === 'polygons',
    'points and polygons must have their named geometry kinds'
  );
  const {candidateCapacity, candidateTileCapacity, leafCapacity, matchCapacity} = input;
  const boundaryRule = input.boundaryRule ?? 'half-open';
  validateRequirement(
    id,
    ['open', 'closed', 'half-open'].includes(boundaryRule),
    'boundaryRule must be open, closed or half-open'
  );
  validateCapacity(
    id,
    'candidateCapacity',
    candidateCapacity,
    1,
    FEATURE_JOIN_MAXIMUM_PAIR_CAPACITY
  );
  if (candidateTileCapacity !== undefined)
    validateCapacity(id, 'candidateTileCapacity', candidateTileCapacity, 1, candidateCapacity);
  validateCapacity(id, 'matchCapacity', matchCapacity, 0);
  if (leafCapacity !== undefined) validatePowerOfTwo(id, 'leafCapacity', leafCapacity);
  const queryCount = getPointJoinFeatureCount(points),
    polygonCount = getPointJoinFeatureCount(polygons);
  const requestedOutput = input.output;
  validateRequirement(
    id,
    !('featureCounts' in requestedOutput),
    'output.featureCounts was removed; use output.polygonAssignmentCounts or output.polygonMembershipCounts'
  );
  const output = snapshotConstructionFields(requestedOutput, [
    'pointFeatureIds',
    'candidateTruncated',
    'ids',
    'count',
    'requiredCount',
    'matchOverflow',
    'matchCandidateOverflow',
    'overflow',
    'candidateOverflow',
    'candidateCount',
    'uncertainCount',
    'pointMatchCounts',
    'featurePolygonMembership',
    'pointFeatureRows',
    'polygonAssignmentCounts',
    'polygonMembershipCounts'
  ]);
  return new GPUContributorOperation({
    id,
    inputs: {
      ...getPointJoinGeometryInputs('points', points),
      ...getPointJoinGeometryInputs('polygons', polygons)
    },
    outputs: Object.fromEntries(
      Object.entries(output)
        .filter(([, value]) => value !== undefined)
        .map(([port, value]) => [
          port,
          {
            value: value!,
            format: 'uint32' as const,
            length:
              port === 'ids'
                ? matchCapacity
                : port === 'pointFeatureIds' ||
                    port === 'pointFeatureRows' ||
                    port === 'candidateTruncated' ||
                    port === 'pointMatchCounts'
                  ? queryCount
                  : port === 'polygonAssignmentCounts' || port === 'polygonMembershipCounts'
                    ? polygonCount
                    : port === 'featurePolygonMembership'
                      ? queryCount * polygonCount
                      : 1
          }
        ])
    ),
    emptyOutputPorts: [
      'ids',
      'pointFeatureIds',
      'pointFeatureRows',
      'candidateTruncated',
      'polygonAssignmentCounts',
      'polygonMembershipCounts',
      'featurePolygonMembership',
      'pointMatchCounts'
    ],
    createContributor: getView =>
      new GPUPointInPolygonJoin({
        id,
        points: resolvePointJoinGeometry(id, 'points', points, getView) as never,
        polygons: resolvePointJoinGeometry(id, 'polygons', polygons, getView) as never,
        candidateCapacity,
        candidateTileCapacity,
        leafCapacity,
        boundaryRule,
        spatialSort: false,
        querySort: false,
        edgeSlabs: false,
        output: {
          ...(output.pointMatchCounts
            ? {pointMatchCounts: getView<'uint32'>('pointMatchCounts')}
            : {}),
          pointFeatureIds: getView('pointFeatureIds'),
          ...(output.pointFeatureRows
            ? {pointFeatureRows: getView<'uint32'>('pointFeatureRows')}
            : {}),
          ...(output.polygonAssignmentCounts
            ? {polygonAssignmentCounts: getView<'uint32'>('polygonAssignmentCounts')}
            : {}),
          ...(output.polygonMembershipCounts
            ? {polygonMembershipCounts: getView<'uint32'>('polygonMembershipCounts')}
            : {}),
          featurePolygonMembership: output.featurePolygonMembership
            ? getView('featurePolygonMembership')
            : undefined,
          candidateTruncated: getView<'uint32'>('candidateTruncated'),
          matches: {
            ids: getView('ids'),
            status: {
              count: getView('count'),
              requiredCount: getView('requiredCount'),
              overflow: getView('matchOverflow'),
              candidateOverflow: getView('matchCandidateOverflow')
            }
          },
          status: {
            overflow: getView('overflow'),
            candidateOverflow: getView('candidateOverflow'),
            candidateCount: getView('candidateCount'),
            uncertainCount: getView('uncertainCount')
          }
        }
      })
  });
}
