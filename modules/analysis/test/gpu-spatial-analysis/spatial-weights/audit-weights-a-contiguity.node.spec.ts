// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeContiguityWeightsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-contiguity-weights-operation';
import {
  AUDIT_CONTIGUITY_EXPECTED,
  createAuditContiguityInput
} from './audit-weights-a-contiguity-fixture';

it('audits queen and rook against literal pairwise-set oracles, including empty features and affine copies', () => {
  for (const [scale, shift] of [
    [1, 0],
    [8, 1048576]
  ]) {
    for (const criterion of ['queen', 'rook'] as const) {
      const expected = AUDIT_CONTIGUITY_EXPECTED[criterion];
      for (const capacity of [5, expected.neighbors.length]) {
        for (const snapTolerance of [0, scale / 4]) {
          const actual = computeContiguityWeightsOnCPU(createAuditContiguityInput(scale, shift), {
            criterion,
            capacity,
            pairCapacity: 128,
            snapTolerance
          });
          expect(Array.from(actual.offsets)).toEqual(
            expected.offsets.map(value => Math.min(capacity, value))
          );
          expect(Array.from(actual.neighbors)).toEqual(expected.neighbors.slice(0, capacity));
          expect(Array.from(actual.weights)).toEqual(Array(capacity).fill(1));
          expect(Array.from(actual.status.requiredCount)).toEqual([expected.neighbors.length]);
          expect(Array.from(actual.status.overflow)).toEqual([
            Number(capacity < expected.neighbors.length)
          ]);
          expect(Array.from(actual.status.candidateOverflow)).toEqual([0]);
        }
      }
    }
  }
});
