// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeDelaunayTessellationOnCPU} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/cpu-delaunay-tessellation-operation';

// Independent oracle: singleton repeated four times has three duplicates;
// a convex quadrilateral has exactly two triangles; one non-finite site cancels.
// CPU iterationCount counts distinct sites rather than GPU insertion/flip rounds.
it('CPU Delaunay statuses match singleton, square and invalid input cardinalities', () => {
  for (const frame of [
    {positions: [2, 2, 2, 2, 2, 2, 2, 2], status: [0, 0, 0, 0, 3, 1, 0], iterations: 1},
    {positions: [0, 0, 1, 0, 1, 1, 0, 1], status: [1, 2, 1, 0, 0, 1, 0], iterations: 4},
    {positions: [NaN, 0, 1, 0, 2, 0, 3, 0], status: [0, 0, 0, 1, 0, 0, 0], iterations: 0}
  ]) {
    const result = computeDelaunayTessellationOnCPU(
      {positions: Float32Array.from(frame.positions)},
      {capacity: 1}
    );
    expect([
      result.count[0],
      result.requiredCount[0],
      result.overflow[0],
      result.invalidCount[0],
      result.duplicateCount[0],
      result.converged[0],
      result.iterationLimitReached[0]
    ]).toEqual(frame.status);
    expect(Array.from(result.iterationCount)).toEqual([frame.iterations]);
  }
});
