// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {computeRankMarkovOnCPU} from '../../../src/gpu-spatial-analysis/distribution-dynamics/cpu-rank-markov-operation';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {
  AUDIT_RANK_CASES,
  AUDIT_RANK_EMPTY,
  AUDIT_RANK_MASK,
  AUDIT_RANK_VALUES
} from './audit-rank-markov-fixture';

it('audits every rank Markov CPU output with SciPy ordinal ranks and linear stationary solve', () => {
  for (const {mode, tieOrder, expected} of AUDIT_RANK_CASES) {
    for (const affine of [false, true]) {
      const actual = computeRankMarkovOnCPU(
        Float32Array.from(AUDIT_RANK_VALUES, value => (affine ? 2 * value + 3 : value)),
        {periodCount: 4, mode, tieOrder},
        Uint32Array.from(AUDIT_RANK_MASK)
      );
      for (const port of Object.keys(expected) as (keyof typeof expected)[]) {
        expectNumericArrayClose(actual[port], expected[port], {
          absoluteTolerance: actual[port] instanceof Float32Array ? 1e-7 : 0,
          label: `${mode}, ${tieOrder}, ${port}`
        });
      }
    }
  }
});

it('audits all empty-panel rank outputs against the hand-derived exclusion counts', () => {
  for (const mode of ['full-rank', 'geographic-rank'] as const) {
    const actual = computeRankMarkovOnCPU(
      Float32Array.from(AUDIT_RANK_VALUES),
      {periodCount: 4, mode},
      new Uint32Array(8)
    );
    for (const port of Object.keys(AUDIT_RANK_EMPTY) as (keyof typeof AUDIT_RANK_EMPTY)[]) {
      expectNumericArrayClose(actual[port], AUDIT_RANK_EMPTY[port], {label: port});
    }
  }
});
