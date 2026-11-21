// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validatePowerOfTwo,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {FEATURE_JOIN_MAXIMUM_PAIR_CAPACITY, type CPUSpatialJoinGeometry} from './cpu-feature-join';
import {
  createPointInPolygonJoinOperation,
  type PointInPolygonJoinOperationProps
} from './gpu-point-in-polygon-join-operation';
import {
  createPointJoinCPUOperation,
  getCPUPointJoinOffsets,
  readPointJoinGeometry,
  snapshotPointJoinGeometry
} from './point-join-operation-utils';
import {readCPUPointJoinPolygons, classifyCPUPointJoinPolygon} from './cpu-point-join-geometry';

/** Same per-part candidate cutoff, per-feature compact capacity and boundary rule as the adapter. */
export type CPUPointInPolygonJoinOptions = Pick<
  PointInPolygonJoinOperationProps,
  'candidateCapacity' | 'candidateTileCapacity' | 'leafCapacity' | 'matchCapacity' | 'boundaryRule'
>;
/** Sparse assignments, polygon reductions and compact-result diagnostics. */
export type CPUPointInPolygonCountsResult = Record<
  Exclude<keyof PointInPolygonJoinOperationProps['output'], 'featurePolygonMembership'>,
  Uint32Array
> & {
  pointFeatureRows: Uint32Array;
  polygonAssignmentCounts: Uint32Array;
  polygonMembershipCounts: Uint32Array;
  candidateBoundsTestCount: Uint32Array;
};
/** Dense feature-major membership adds 4 * pointFeatureCount * polygonFeatureCount bytes. */
export type CPUPointInPolygonJoinResult = CPUPointInPolygonCountsResult & {
  featurePolygonMembership: Uint32Array;
};
/**
 * Joins packed points/multipoints to polygons using exact binary64 integer orientation signs.
 * What differs from the GPU contributor:
 * - Exact CPU predicates have no dynamic-range uncertainty; GPU may
 *   refuse a candidate whose edge orientation spans more than 200 binary orders of magnitude.
 * - Malformed layouts throw. Nonfinite polygon candidates increment uncertainCount and do not
 *   match; results for other polygons survive, as on the GPU.
 * - Unused compact slots are zero, and are outside the readable count prefix.
 * - Unprepared planar geometry only; sorted/prepared GPU traversals remain contributor options.
 * Part candidates are visited in input-part order, then polygon row order, exactly as the
 * unsorted complete binary BVH. A cutoff flags the owner even when an earlier part matched.
 */
export function computePointInPolygonJoinOnCPU(
  input: {points: CPUSpatialJoinGeometry; polygons: CPUSpatialJoinGeometry},
  options: CPUPointInPolygonJoinOptions
): CPUPointInPolygonJoinResult {
  return computePointInPolygonRowsOnCPU(input, options, true) as CPUPointInPolygonJoinResult;
}

/** Counts/assignments from sparse matches without a feature-by-polygon membership matrix. */
export function computePointInPolygonCountsOnCPU(
  input: {points: CPUSpatialJoinGeometry; polygons: CPUSpatialJoinGeometry},
  options: CPUPointInPolygonJoinOptions
): CPUPointInPolygonCountsResult {
  return computePointInPolygonRowsOnCPU(input, options, false);
}

function computePointInPolygonRowsOnCPU(
  input: {points: CPUSpatialJoinGeometry; polygons: CPUSpatialJoinGeometry},
  options: CPUPointInPolygonJoinOptions,
  retainMembership: boolean
): CPUPointInPolygonCountsResult & {featurePolygonMembership?: Uint32Array} {
  const id = 'CPUPointInPolygonJoin';
  // The existing point and polygon readers each validate their hierarchy once.
  const offsets = getCPUPointJoinOffsets(id, input.points),
    polygons = readCPUPointJoinPolygons(id, input.polygons, true);
  validateCapacity(
    id,
    'candidateCapacity',
    options.candidateCapacity,
    1,
    FEATURE_JOIN_MAXIMUM_PAIR_CAPACITY
  );
  validateCapacity(id, 'matchCapacity', options.matchCapacity, 0);
  if (options.candidateTileCapacity !== undefined)
    validateCapacity(
      id,
      'candidateTileCapacity',
      options.candidateTileCapacity,
      1,
      options.candidateCapacity
    );
  if (options.leafCapacity !== undefined)
    validatePowerOfTwo(id, 'leafCapacity', options.leafCapacity);
  const rule = options.boundaryRule ?? 'half-open';
  validateRequirement(
    id,
    ['open', 'closed', 'half-open'].includes(rule),
    'boundaryRule must be open, closed or half-open'
  );
  const queryCount = offsets.length - 1,
    polygonCount = polygons.length;
  const leafCapacity = options.leafCapacity ?? 2 ** Math.ceil(Math.log2(Math.max(1, polygonCount)));
  const indexedCount = Math.min(leafCapacity, polygonCount);
  const boundsLeafCount = 2 ** Math.ceil(Math.log2(Math.max(1, indexedCount)));
  const boundsTree = Array.from({length: 2 * boundsLeafCount}, () => [
    Infinity,
    Infinity,
    -Infinity,
    -Infinity
  ]);
  const orderedRows = Array.from({length: indexedCount}, (_, row) => row).sort(
    (first, second) => polygons[first].bounds[0] - polygons[second].bounds[0] || first - second
  );
  for (let leaf = 0; leaf < indexedCount; leaf++)
    boundsTree[boundsLeafCount + leaf] = [...polygons[orderedRows[leaf]].bounds];
  for (let node = boundsLeafCount - 1; node > 0; node--) {
    const left = boundsTree[node * 2],
      right = boundsTree[node * 2 + 1];
    boundsTree[node] = [
      Math.min(left[0], right[0]),
      Math.min(left[1], right[1]),
      Math.max(left[2], right[2]),
      Math.max(left[3], right[3])
    ];
  }
  let candidateBoundsTestCount = 0;
  function* queryBounds(x: number, y: number): Generator<number> {
    const stack = [1];
    const candidates: number[] = [];
    while (stack.length) {
      const node = stack.pop()!,
        bounds = boundsTree[node];
      candidateBoundsTestCount++;
      if (x < bounds[0] || x > bounds[2] || y < bounds[1] || y > bounds[3]) continue;
      if (node >= boundsLeafCount) {
        if (node - boundsLeafCount < indexedCount)
          candidates.push(orderedRows[node - boundsLeafCount]);
      } else {
        // LIFO visits original polygon rows in ascending order, preserving capacity cutoffs.
        stack.push(node * 2 + 1, node * 2);
      }
    }
    // Restore source row order before applying the global candidate cutoff.
    yield* candidates.sort((first, second) => first - second);
  }
  const pointMatchCounts = new Uint32Array(queryCount),
    featurePolygonMembership = new Uint32Array(retainMembership ? queryCount * polygonCount : 0),
    polygonAssignmentCounts = new Uint32Array(polygonCount),
    polygonMembershipCounts = new Uint32Array(polygonCount),
    pointFeatureRows = new Uint32Array(queryCount).fill(0xffffffff),
    pointFeatureIds = new Uint32Array(queryCount).fill(0xffffffff),
    candidateTruncated = new Uint32Array(queryCount),
    ids = new Uint32Array(options.matchCapacity);
  let candidateCount = 0,
    requiredCount = 0,
    uncertainCount = 0;
  for (let feature = 0; feature < queryCount; feature++) {
    const membership = new Set<number>();
    for (let part = offsets[feature]; part < offsets[feature + 1]; part++) {
      if (polygonCount > leafCapacity) candidateTruncated[feature] = 1;
      const x = input.points.positions[2 * part],
        y = input.points.positions[2 * part + 1];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      for (const polygon of queryBounds(x, y)) {
        if (candidateCount++ >= options.candidateCapacity) {
          candidateTruncated[feature] = 1;
          continue;
        }
        if (polygons[polygon].uncertain) {
          uncertainCount++;
          continue;
        }
        if (classifyCPUPointJoinPolygon(x, y, polygons[polygon], rule)) membership.add(polygon);
      }
    }
    let first = 0xffffffff;
    for (const polygon of membership) {
      first = Math.min(first, polygon);
      pointMatchCounts[feature]++;
      polygonMembershipCounts[polygon]++;
      if (retainMembership) featurePolygonMembership[feature * polygonCount + polygon] = 1;
    }
    if (first !== 0xffffffff) {
      polygonAssignmentCounts[first]++;
      pointFeatureRows[feature] = first;
      pointFeatureIds[feature] = input.polygons.sourceIds?.[first] ?? first;
      if (requiredCount < options.matchCapacity)
        ids[requiredCount] = input.points.sourceIds?.[feature] ?? feature;
      requiredCount++;
    }
  }
  const candidateOverflow = Number(
      polygonCount > leafCapacity || candidateCount > options.candidateCapacity
    ),
    matchOverflow = Number(requiredCount > options.matchCapacity);
  return {
    pointMatchCounts,
    pointFeatureIds,
    pointFeatureRows,
    polygonAssignmentCounts,
    polygonMembershipCounts,
    ...(retainMembership ? {featurePolygonMembership} : {}),
    candidateTruncated,
    ids,
    count: Uint32Array.of(Math.min(requiredCount, options.matchCapacity)),
    requiredCount: Uint32Array.of(requiredCount),
    matchOverflow: Uint32Array.of(matchOverflow),
    matchCandidateOverflow: Uint32Array.of(candidateOverflow),
    overflow: Uint32Array.of(Number(Boolean(candidateOverflow || matchOverflow))),
    candidateOverflow: Uint32Array.of(candidateOverflow),
    candidateCount: Uint32Array.of(Math.min(candidateCount, 0xffffffff)),
    candidateBoundsTestCount: Uint32Array.of(Math.min(candidateBoundsTestCount, 0xffffffff)),
    uncertainCount: Uint32Array.of(uncertainCount)
  };
}
/** CPU operation using precisely the GPU adapter's named value contracts. @internal */
export function createCPUPointInPolygonJoinOperation(input: PointInPolygonJoinOperationProps) {
  const id = input.id ?? 'point-in-polygon-join-operation';
  const points = snapshotPointJoinGeometry(id, input.points),
    polygons = snapshotPointJoinGeometry(id, input.polygons);
  const options = Object.freeze({
    candidateCapacity: input.candidateCapacity,
    candidateTileCapacity: input.candidateTileCapacity,
    leafCapacity: input.leafCapacity,
    matchCapacity: input.matchCapacity,
    boundaryRule: input.boundaryRule
  });
  const operation = createPointInPolygonJoinOperation({
    id,
    points,
    polygons,
    ...options,
    output: input.output
  });
  const computeJoin = operation.outputs['featurePolygonMembership']
    ? computePointInPolygonJoinOnCPU
    : computePointInPolygonCountsOnCPU;
  return createPointJoinCPUOperation(operation, inputs =>
    computeJoin(
      {
        points: readPointJoinGeometry(operation.id, 'points', points, inputs),
        polygons: readPointJoinGeometry(operation.id, 'polygons', polygons, inputs)
      },
      options
    )
  );
}
