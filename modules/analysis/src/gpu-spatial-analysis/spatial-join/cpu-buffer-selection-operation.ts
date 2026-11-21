// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {getCPUPointJoinOrientation} from './cpu-point-join-geometry';
import {createCPUFeatureDistanceRank} from './cpu-nearest-distance-rank';
import {
  validateCapacity,
  validatePowerOfTwo,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  isPointWithinSegmentDistance,
  getPointSegmentDistance,
  type CPUSpatialJoinGeometry
} from './cpu-feature-join';
import {
  createBufferSelectionOperation,
  resolveBufferSelectionRadiusDescriptor,
  type BufferSelectionOperationProps
} from './gpu-buffer-selection-operation';
import {
  createPointJoinCPUOperation,
  getCPUPointJoinOffsets,
  readPointJoinGeometry,
  snapshotPointJoinGeometry
} from './point-join-operation-utils';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

/** CPU operation descriptors accept the same radius as the pure CPU and GPU contributors. */
export type CPUBufferSelectionOperationProps = BufferSelectionOperationProps;

/** Planar point-reference inputs; feature source IDs do not affect nearest row identities. */
export type CPUBufferSelectionInput = {
  points: CPUSpatialJoinGeometry;
  features: CPUSpatialJoinGeometry | {kind: 'segments'; starts: Float32Array; ends: Float32Array};
  /** Inclusive planar radius in coordinate units. */
  radius: number;
};
/** Same budgets as the packed planar GPU adapter. */
export type CPUBufferSelectionOptions = Pick<
  BufferSelectionOperationProps,
  'candidateCapacity' | 'leafCapacity' | 'matchCapacity'
>;
/**
 * Selects a feature when any point part is within the inclusive distance of a reference point.
 * What differs from the GPU contributor:
 * - Distances use float64 internally, rounded once at output; last-ulp radius decisions can differ.
 * - This packed planar adapter supports point and segment references; use GPUBufferSelection directly for great-circle references.
 * - Unused compact ID slots are zero. GPU callers must only read the prefix selected by count.
 * - Invalid offset layouts throw on CPU. Nonfinite points are ignored, like GPU query probes.
 */
export function computeBufferSelectionOnCPU(
  input: CPUBufferSelectionInput,
  options: CPUBufferSelectionOptions
): Record<keyof BufferSelectionOperationProps['output'], Uint32Array | Float32Array> {
  const id = 'CPUBufferSelection';
  validateRequirement(id, !('distance' in input), 'distance was removed; use radius');
  validateRequirement(id, input.radius !== undefined, 'radius is required');
  const offsets = getCPUPointJoinOffsets(id, input.points);
  const starts =
    input.features.kind === 'segments' ? input.features.starts : input.features.positions;
  const ends = input.features.kind === 'segments' ? input.features.ends : starts;
  validateRequirement(
    id,
    (input.features.kind === 'segments' ||
      (input.features.kind === 'points' && !input.features.featureOffsets)) &&
      starts.length % 2 === 0 &&
      ends.length === starts.length,
    'features must contain equal-length starts/ends XY segment arrays or individual point rows'
  );
  validateCapacity(id, 'candidateCapacity', options.candidateCapacity);
  validateCapacity(id, 'matchCapacity', options.matchCapacity);
  if (options.leafCapacity !== undefined)
    validatePowerOfTwo(id, 'leafCapacity', options.leafCapacity);
  const featureCount = starts.length / 2;
  const leafCapacity = options.leafCapacity ?? 2 ** Math.ceil(Math.log2(Math.max(1, featureCount)));
  const queryCount = offsets.length - 1;
  validateCapacity(id, 'point feature count', queryCount);
  const outputMask = new Uint32Array(queryCount),
    distances = new Float32Array(queryCount).fill(-1),
    nearestFeatureRows = new Uint32Array(queryCount).fill(0xffffffff),
    candidateTruncated = new Uint32Array(queryCount);
  const ids = new Uint32Array(options.matchCapacity);
  let requiredCount = 0,
    candidateCount = 0,
    candidateOverflow = Number(featureCount > leafCapacity);
  const radius = Math.fround(input.radius);
  const getDistanceRank = createCPUFeatureDistanceRank();
  for (let query = 0; query < queryCount; query++) {
    let bestDistance = Infinity,
      bestRank = Infinity,
      bestRow = 0xffffffff;
    let truncated = featureCount > leafCapacity;
    for (let part = offsets[query]; part < offsets[query + 1]; part++) {
      const x = input.points.positions[2 * part],
        y = input.points.positions[2 * part + 1];
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(radius) || radius < 0)
        continue;
      for (let feature = 0; feature < Math.min(featureCount, leafCapacity); feature++) {
        const referenceX = starts[2 * feature],
          referenceY = starts[2 * feature + 1],
          endX = ends[2 * feature],
          endY = ends[2 * feature + 1];
        if (![referenceX, referenceY, endX, endY].every(Number.isFinite)) continue;
        if (!input.points.featureOffsets) {
          if (
            x < Math.min(referenceX, endX) - radius ||
            x > Math.max(referenceX, endX) + radius ||
            y < Math.min(referenceY, endY) - radius ||
            y > Math.max(referenceY, endY) + radius
          )
            continue;
          if (candidateCount++ >= options.candidateCapacity) {
            truncated = true;
            candidateOverflow = 1;
            continue;
          }
        }
        const onSegment =
          x >= Math.min(referenceX, endX) &&
          x <= Math.max(referenceX, endX) &&
          y >= Math.min(referenceY, endY) &&
          y <= Math.max(referenceY, endY) &&
          getCPUPointJoinOrientation(referenceX, referenceY, endX, endY, x, y) === 0;
        const distance = onSegment
          ? 0
          : getPointSegmentDistance([x, y], [referenceX, referenceY], [endX, endY]);
        const distanceRank = getDistanceRank(distance);
        if (
          (onSegment ||
            (radius > 0 &&
              isPointWithinSegmentDistance(
                [x, y],
                [referenceX, referenceY],
                [endX, endY],
                radius
              ))) &&
          (distanceRank < bestRank || (distanceRank === bestRank && feature < bestRow))
        ) {
          bestDistance = distance;
          bestRank = distanceRank;
          bestRow = feature;
        }
      }
    }
    candidateTruncated[query] = Number(truncated);
    if (bestRow !== 0xffffffff) {
      nearestFeatureRows[query] = bestRow;
      distances[query] = bestDistance;
      outputMask[query] = 1;
      if (requiredCount < options.matchCapacity)
        ids[requiredCount] = input.points.sourceIds?.[query] ?? query;
      requiredCount++;
    }
  }
  const count = Math.min(requiredCount, options.matchCapacity);
  return {
    outputMask,
    distances,
    nearestFeatureRows,
    candidateTruncated,
    ids,
    count: Uint32Array.of(count),
    requiredCount: Uint32Array.of(requiredCount),
    matchOverflow: Uint32Array.of(Number(requiredCount > options.matchCapacity)),
    matchCandidateOverflow: Uint32Array.of(candidateOverflow),
    overflow: Uint32Array.of(candidateOverflow),
    candidateOverflow: Uint32Array.of(candidateOverflow),
    drawInstanceCount: Uint32Array.of(count)
  };
}
/** CPU twin of the packed planar adapter. @internal */
export function createCPUBufferSelectionOperation(input: CPUBufferSelectionOperationProps) {
  const id = input.id ?? 'buffer-selection-operation';
  const radius = resolveBufferSelectionRadiusDescriptor(id, input);
  const points = snapshotPointJoinGeometry(id, input.points);
  const sourceFeatures = input.features;
  const features =
    sourceFeatures.kind === 'segments'
      ? Object.freeze({
          kind: 'segments' as const,
          starts: snapshotConstructionFields(sourceFeatures.starts, ['value', 'length']),
          ends: snapshotConstructionFields(sourceFeatures.ends, ['value', 'length'])
        })
      : snapshotPointJoinGeometry(id, sourceFeatures);
  const options = Object.freeze({
    candidateCapacity: input.candidateCapacity,
    leafCapacity: input.leafCapacity,
    matchCapacity: input.matchCapacity
  });
  const operation = createBufferSelectionOperation({
    id,
    points,
    features,
    radius,
    ...options,
    output: input.output
  });
  return createPointJoinCPUOperation(operation, inputs => {
    const distance = getCPUOperationFloat32Rows(operation.id, inputs, 'distance');
    validateRequirement(operation.id, distance.length === 1, 'distance needs one float32 row');
    const featureArrays =
      features.kind === 'segments'
        ? {
            kind: 'segments' as const,
            starts: getCPUOperationFloat32Rows(operation.id, inputs, 'featureStarts', 'float32x2'),
            ends: getCPUOperationFloat32Rows(operation.id, inputs, 'featureEnds', 'float32x2')
          }
        : readPointJoinGeometry(operation.id, 'features', features, inputs);
    if (features.kind === 'segments')
      validateRequirement(
        operation.id,
        featureArrays.kind === 'segments' &&
          featureArrays.starts.length === features.starts.length * 2 &&
          featureArrays.ends.length === features.ends.length * 2,
        'featureStarts and featureEnds must match declared lengths'
      );
    return computeBufferSelectionOnCPU(
      {
        points: readPointJoinGeometry(operation.id, 'points', points, inputs),
        features: featureArrays,
        radius: distance[0]
      },
      options
    );
  });
}
