// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeRatioConcaveHullOnCPU} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-ratio-concave-hull-operation';
import {computeVectorStraightSkeletonOnCPU} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-vector-straight-skeleton-operation';

it('CPU ratio status is complete for an unremovable simplex and empty collinear hull', () => {
  // One triangle has three boundary vertices; removing it would discard the
  // sites. Collinear sites have no triangle. Neither requires any removal work.
  for (const frame of [
    {positions: [0, 0, 1, 0, 0, 1], validity: 1},
    {positions: [0, 0, 1, 0, 2, 0], validity: 0}
  ]) {
    const result = computeRatioConcaveHullOnCPU(Float32Array.from(frame.positions), {
      ratio: 0,
      maximumIterations: 0
    });
    expect(Array.from(result.triangleValidity)).toEqual([frame.validity]);
    expect(Array.from(result.converged)).toEqual([1]);
  }
});
it('CPU ratio validity identifies the removed longest-edge triangle at the exact work budget', () => {
  // Independent SciPy triangulation and exterior squared lengths 36,16,52:
  // the only winning removal is {1,2,3}. See the adjacent committed .oracle.py.
  for (const maximumIterations of [0, 1]) {
    const result = computeRatioConcaveHullOnCPU(Float32Array.of(0, 0, 6, 0, 0, 4, 1, 1), {
      ratio: 0,
      maximumIterations
    });
    expect(result.triangles.length).toBe(9);
    expect(result.triangleValidity.length).toBe(3);
    const byIdentity = Array.from(result.triangleValidity, (validity, row) => ({
      identity: Array.from(result.triangles.slice(3 * row, 3 * row + 3))
        .sort((first, second) => first - second)
        .join(','),
      validity
    })).sort((first, second) => first.identity.localeCompare(second.identity));
    expect(byIdentity).toEqual([
      {identity: '0,1,3', validity: 1},
      {identity: '0,2,3', validity: 1},
      {identity: '1,2,3', validity: maximumIterations === 0 ? 1 : 0}
    ]);
    expect(Array.from(result.converged)).toEqual([maximumIterations === 0 ? 0 : 1]);
  }
});
it('CPU vector skeleton zero-event status preserves the four-vertex initial snapshot', () => {
  const result = computeVectorStraightSkeletonOnCPU(
    {
      positions: Float32Array.of(0, 0, 4, 0, 4, 2, 0, 2),
      ringOffsets: Uint32Array.of(0, 4),
      polygonOffsets: Uint32Array.of(0, 1),
      featureOffsets: Uint32Array.of(0, 1)
    },
    {maximumVertices: 4, maximumEvents: 0, segmentCapacity: 8, offsetDistance: 0}
  );
  expect(Array.from(result.status)).toEqual([0, 0, 4, 4]);
});
