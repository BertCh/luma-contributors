// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationViewGetter
} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  validateCapacity,
  validatePowerOfTwo,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUBufferSelection} from './gpu-buffer-selection';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import type {FeatureJoinGeometryValues, FeatureJoinValue} from './feature-join-operation';
import {
  snapshotPointJoinGeometry,
  getPointJoinFeatureCount,
  getPointJoinGeometryInputs,
  resolvePointJoinGeometry
} from './point-join-operation-utils';

/** Internal packed planar adapter. Both GPU and CPU factories take this identical contract. */
export type BufferSelectionOperationProps = {
  id?: string;
  points: FeatureJoinGeometryValues;
  features:
    | FeatureJoinGeometryValues
    | {kind: 'segments'; starts: FeatureJoinValue; ends: FeatureJoinValue};
  /** Named one-row float32 inclusive planar radius, in coordinate units. */
  radius: FeatureJoinValue;
  candidateCapacity: number;
  leafCapacity?: number;
  matchCapacity: number;
  output: Record<
    | 'outputMask'
    | 'distances'
    | 'nearestFeatureRows'
    | 'candidateTruncated'
    | 'ids'
    | 'count'
    | 'requiredCount'
    | 'matchOverflow'
    | 'matchCandidateOverflow'
    | 'overflow'
    | 'candidateOverflow'
    | 'drawInstanceCount',
    string
  >;
};
/** Captures descriptors, including inherited getters, at construction. @internal */
export function resolveBufferSelectionRadiusDescriptor(
  id: string,
  input: Pick<BufferSelectionOperationProps, 'radius'>
): FeatureJoinValue {
  validateRequirement(id, !('distance' in input), 'distance was removed; use radius');
  const radius = input.radius;
  validateRequirement(id, Boolean(radius), 'radius descriptor is required');
  const result = snapshotConstructionFields(radius, ['value', 'length']);
  validateRequirement(
    id,
    result.length === 1,
    `radius descriptor length must equal one float32 row; received ${result.length}`
  );
  return result;
}
/** Resolves an endpoint view against its captured named descriptor. @internal */
function resolveSegmentEndpoint(
  id: string,
  port: string,
  descriptor: FeatureJoinValue,
  getView: GPUContributorOperationViewGetter
) {
  const view = getView(port, 'float32x2');
  validateRequirement(
    id,
    view.length === descriptor.length,
    `${port} length must match declared length ${descriptor.length}; received ${view.length}`
  );
  return view;
}

/** Packed planar point/segment-reference adapter, preserving feature rows. @internal */
export function createBufferSelectionOperation(
  input: BufferSelectionOperationProps
): GPUContributorOperation {
  const id = input.id ?? 'buffer-selection-operation';
  const points = snapshotPointJoinGeometry(id, input.points);
  const originalFeatures = input.features;
  const features =
    originalFeatures.kind === 'segments'
      ? Object.freeze({
          kind: 'segments' as const,
          starts: snapshotConstructionFields(originalFeatures.starts, ['value', 'length']),
          ends: snapshotConstructionFields(originalFeatures.ends, ['value', 'length'])
        })
      : snapshotPointJoinGeometry(id, originalFeatures);
  validateRequirement(
    id,
    points.kind === 'points' &&
      (features.kind === 'segments' || (features.kind === 'points' && !features.featureOffsets)),
    'points must be points and features must contain points or explicit segments; use GPUNearestFeatureJoin for other reference geometries'
  );
  if (features.kind === 'segments') {
    validateCapacity(id, 'features.starts.length', features.starts.length, 0);
    validateCapacity(id, 'features.ends.length', features.ends.length, 0);
    validateRequirement(
      id,
      features.starts.length === features.ends.length,
      'features.starts and features.ends must have equal lengths'
    );
  }
  const distance = resolveBufferSelectionRadiusDescriptor(id, input);
  const {candidateCapacity, leafCapacity, matchCapacity} = input;
  validateCapacity(id, 'candidateCapacity', candidateCapacity);
  validateCapacity(id, 'matchCapacity', matchCapacity);
  if (leafCapacity !== undefined) validatePowerOfTwo(id, 'leafCapacity', leafCapacity);
  const queryCount = getPointJoinFeatureCount(points);
  validateCapacity(id, 'point feature count', queryCount);
  const output = snapshotConstructionFields(input.output, [
    'outputMask',
    'distances',
    'nearestFeatureRows',
    'candidateTruncated',
    'ids',
    'count',
    'requiredCount',
    'matchOverflow',
    'matchCandidateOverflow',
    'overflow',
    'candidateOverflow',
    'drawInstanceCount'
  ]);
  return new GPUContributorOperation({
    id,
    inputs: {
      ...getPointJoinGeometryInputs('points', points),
      ...(features.kind === 'segments'
        ? {featureStarts: features.starts.value, featureEnds: features.ends.value}
        : getPointJoinGeometryInputs('features', features)),
      distance: distance.value
    },
    outputs: Object.fromEntries(
      Object.entries(output).map(([port, value]) => [
        port,
        {
          value,
          format: port === 'distances' ? ('float32' as const) : ('uint32' as const),
          length:
            port === 'ids'
              ? matchCapacity
              : ['outputMask', 'distances', 'nearestFeatureRows', 'candidateTruncated'].includes(
                    port
                  )
                ? queryCount
                : 1
        }
      ])
    ),
    createContributor: getView =>
      new GPUBufferSelection({
        id,
        points: resolvePointJoinGeometry(id, 'points', points, getView) as never,
        features:
          features.kind === 'segments'
            ? {
                kind: 'segments',
                starts: resolveSegmentEndpoint(id, 'featureStarts', features.starts, getView),
                ends: resolveSegmentEndpoint(id, 'featureEnds', features.ends, getView)
              }
            : (resolvePointJoinGeometry(id, 'features', features, getView) as never),
        radius: getView<'float32'>('distance'),
        candidateCapacity,
        leafCapacity,
        spatialSort: false,
        querySort: false,
        output: {
          outputMask: getView('outputMask'),
          distances: getView('distances'),
          nearestFeatureRows: getView('nearestFeatureRows'),
          candidateTruncated: getView('candidateTruncated'),
          drawInstanceCount: getView('drawInstanceCount'),
          matches: {
            ids: getView('ids'),
            status: {
              count: getView('count'),
              requiredCount: getView('requiredCount'),
              overflow: getView('matchOverflow'),
              candidateOverflow: getView('matchCandidateOverflow')
            }
          },
          status: {overflow: getView('overflow'), candidateOverflow: getView('candidateOverflow')}
        }
      })
  });
}
