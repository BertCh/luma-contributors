// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeSpatialPredicateJoinOnCPU} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-join';

it('zero-length line within distinguishes the outer endpoint from its interior', () => {
  // OGC within needs an interior/interior intersection: (0,0) is a boundary point,
  // whereas (0.5,0) is in the relative interior of the right line.
  const result = computeSpatialPredicateJoinOnCPU(
    {
      kind: 'lines',
      positions: Float32Array.of(0, 0, 0, 0, 0.5, 0, 0.5, 0),
      lineOffsets: Uint32Array.of(0, 2, 4)
    },
    {kind: 'lines', positions: Float32Array.of(0, 0, 1, 0), lineOffsets: Uint32Array.of(0, 2)},
    {predicate: 'within', candidateCapacity: 2, pairCapacity: 2}
  );
  expect(result.status).toEqual({count: 1, requiredCount: 1, overflow: 0, candidateOverflow: 0});
  expect(result.leftRows[0]).toBe(1);
  expect(result.rightRows[0]).toBe(0);
});
