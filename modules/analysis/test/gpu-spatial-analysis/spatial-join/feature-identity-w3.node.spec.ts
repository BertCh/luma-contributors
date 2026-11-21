// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computePointInPolygonJoinOnCPU,
  computePointInPolygonCountsOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-point-in-polygon-join-operation';
import {computeBufferSelectionOnCPU} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-buffer-selection-operation';
import {computeNearestFeatureJoinOnCPU} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-join';

const polygons = {
  kind: 'polygons' as const,
  positions: Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1),
  featureOffsets: Uint32Array.of(0, 1, 2),
  polygonOffsets: Uint32Array.of(0, 1, 2),
  ringOffsets: Uint32Array.of(0, 4, 8),
  sourceIds: Uint32Array.of(99, 1)
};
for (const multipart of [false, true]) {
  it(`UGE-01 explicit counts and GEO-114 rows, multipart=${multipart}`, () => {
    // One point belongs to two identical squares; lowest polygon row owns it.
    const result = computePointInPolygonJoinOnCPU(
      {
        points: {
          kind: 'points',
          positions: Float32Array.of(0.5, 0.5),
          ...(multipart ? {featureOffsets: Uint32Array.of(0, 1)} : {})
        },
        polygons
      },
      {candidateCapacity: 8, matchCapacity: 1}
    );
    expect(Array.from(result['polygonAssignmentCounts'])).toEqual([1, 0]);
    expect(Array.from(result['polygonMembershipCounts'])).toEqual([1, 1]);
    expect(Array.from(result['pointFeatureRows'])).toEqual([0]);
    expect(Array.from(result.pointFeatureIds)).toEqual([99]);
  });
}
it('UGE-15 counts-only has no dense membership, including zero candidates', () => {
  const result = computePointInPolygonCountsOnCPU(
    {
      points: {
        kind: 'points',
        positions: Float32Array.of(10, 10)
      },
      polygons
    },
    {candidateCapacity: 1, matchCapacity: 0}
  );
  expect('featurePolygonMembership' in result).toBe(false);
  expect(Array.from(result.candidateCount)).toEqual([0]);
  expect(Array.from(result.candidateBoundsTestCount)).toEqual([1]);
  expect(Array.from(result['polygonMembershipCounts'])).toEqual([0, 0]);
});
it('UGE-15 multipart sparse matches deduplicate polygons and retain loss diagnostics', () => {
  const result = computePointInPolygonCountsOnCPU(
    {
      points: {
        kind: 'points',
        positions: Float32Array.of(0.5, 0.5, 0.5, 0.5),
        featureOffsets: Uint32Array.of(0, 2)
      },
      polygons
    },
    {candidateCapacity: 8, matchCapacity: 1}
  );
  expect(Array.from(result.polygonMembershipCounts)).toEqual([1, 1]);
  expect(Array.from(result.pointMatchCounts)).toEqual([2]);
  expect(Array.from(result.polygonAssignmentCounts)).toEqual([1, 0]);
  const truncated = computePointInPolygonCountsOnCPU(
    {points: {kind: 'points', positions: Float32Array.of(0.5, 0.5)}, polygons},
    {candidateCapacity: 1, matchCapacity: 1}
  );
  expect(Array.from(truncated.matchOverflow)).toEqual([0]);
  expect(Array.from(truncated.candidateOverflow)).toEqual([1]);
  expect(Array.from(truncated.candidateTruncated)).toEqual([1]);
});
it('UGE-06 radius selects inclusively and rejects retired distance', () => {
  const input = {
    points: {kind: 'points' as const, positions: Float32Array.of(0, 0)},
    features: {kind: 'points' as const, positions: Float32Array.of(0, 0)},
    radius: 0
  };
  expect(
    Array.from(
      computeBufferSelectionOnCPU(input, {candidateCapacity: 1, matchCapacity: 1}).outputMask
    )
  ).toEqual([1]);
  const legacy = {...input, distance: 0};
  expect(() =>
    computeBufferSelectionOnCPU(legacy, {candidateCapacity: 1, matchCapacity: 1})
  ).toThrow(/distance.*radius/);
});
for (const reverse of [false, true]) {
  it(`UGE-06 certified radius threshold, reverse=${reverse}`, () => {
    const features = {
      kind: 'segments' as const,
      starts: Float32Array.from(reverse ? [1, 1] : [-(2 ** 30), -(2 ** 30)]),
      ends: Float32Array.from(reverse ? [-(2 ** 30), -(2 ** 30)] : [1, 1])
    };
    const compute = (y: number, distance: number) =>
      computeBufferSelectionOnCPU(
        {
          points: {kind: 'points', positions: Float32Array.of(0.5, y)},
          features,
          radius: distance
        },
        {candidateCapacity: 4, matchCapacity: 1}
      );
    expect(Array.from(compute(0.5 + 2 ** -24, 0).outputMask)).toEqual([0]);
    expect(Array.from(compute(0.5, 0).outputMask)).toEqual([1]);
    // Perpendicular distance = 2^-24 / sqrt(2), strictly between 2^-25 and 2^-24.
    expect(Array.from(compute(0.5 + 2 ** -24, 2 ** -25).outputMask)).toEqual([0]);
    expect(Array.from(compute(0.5 + 2 ** -24, 2 ** -24).outputMask)).toEqual([1]);
    const selectedDistance = compute(0.5 + 2 ** -24, 2 ** -24).distances[0];
    expect(Number.isFinite(selectedDistance)).toBe(true);
    // float32(2^-24 / sqrt(2)), the independently derived perpendicular distance.
    expect(selectedDistance).toBe(4.214684778958144e-8);
  });
}
it('NAM-03 one-neighbor ties retain lowest row independently of source IDs', () => {
  const result = computeNearestFeatureJoinOnCPU(
    {kind: 'points', positions: Float32Array.of(0, 0)},
    {kind: 'points', positions: Float32Array.of(-1, 0, 1, 0), sourceIds: Uint32Array.of(99, 1)}
  );
  expect(Array.from(result.neighborRows)).toEqual([0]);
  expect(Array.from(result.neighborIds)).toEqual([99]);
  expect(Number.isFinite(result.neighborDistances[0])).toBe(true);
  expect(Array.from(result.neighborDistances)).toEqual([1]);
});
