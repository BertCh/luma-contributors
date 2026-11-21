// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeSegmentSnapOnCPU} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-segment-snap-operation';

// Two interior collinear references insert into the one edge: four vertices,
// valid complete geometry, but grid capacity 1 must report its lost candidates.
it('CPU snap status matches the complete-fallback candidate cardinality', () => {
  for (const candidateCapacity of [1, 16]) {
    const result = computeSegmentSnapOnCPU(
      {
        positions: Float32Array.of(0, 0, 3, 0),
        lineOffsets: Uint32Array.of(0, 2),
        referencePositions: Float32Array.of(1, 0, 2, 0)
      },
      {tolerance: 0, maximumVerticesPerLine: 4, candidateCapacity, searchStructure: 'grid'}
    );
    expect(Array.from(result.counts)).toEqual([4, 4, 0, 0]);
    expect(Array.from(result.validity)).toEqual([1]);
    expect(Array.from(result.candidateOverflow)).toEqual([candidateCapacity === 1 ? 1 : 0]);
  }
});
