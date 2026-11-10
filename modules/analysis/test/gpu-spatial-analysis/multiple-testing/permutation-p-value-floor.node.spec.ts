// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getMultipleTestingAdjustedPValues,
  getPermutationRejectionFloor
} from '../../../src/gpu-spatial-analysis/multiple-testing/index';

it('getPermutationRejectionFloor reproduces the review figures at 999 permutations', () => {
  const base = {permutationCount: 999, alpha: 0.05};
  expect(getPermutationRejectionFloor({...base, rowCount: 10000}).pValueFloor).toBe(0.002);

  // Benjamini-Hochberg with 10,000 rows needs 400 rows at the floor.
  const hochberg = getPermutationRejectionFloor({
    ...base,
    rowCount: 10000,
    method: 'benjamini-hochberg'
  });
  expect(hochberg).toEqual({pValueFloor: 0.002, canReject: true, minimumRowsAtFloor: 400});

  // Bonferroni and Holm cannot reject above 25 rows.
  for (const method of ['bonferroni', 'holm'] as const) {
    expect(getPermutationRejectionFloor({...base, rowCount: 25, method})).toMatchObject({
      canReject: true,
      maximumRowCount: 25
    });
    expect(getPermutationRejectionFloor({...base, rowCount: 26, method})).toMatchObject({
      canReject: false,
      minimumRowsAtFloor: Infinity,
      maximumRowCount: 25
    });
  }
});

it('getPermutationRejectionFloor agrees with the CPU adjustment', () => {
  const base = {permutationCount: 999, alpha: 0.05};
  for (const method of [
    'none',
    'bonferroni',
    'holm',
    'benjamini-hochberg',
    'benjamini-yekutieli'
  ] as const) {
    for (const rowCount of [1, 10, 25, 26, 400, 2000]) {
      const floor = getPermutationRejectionFloor({...base, rowCount, method});
      // Best case for rejection: `minimumRowsAtFloor` rows at the floor, the rest at 1.
      const atFloor = Math.min(floor.minimumRowsAtFloor, rowCount);
      const pValues = Array.from({length: rowCount}, (_, row) => (row < atFloor ? 0.002 : 1));
      const adjusted = getMultipleTestingAdjustedPValues(pValues, method);
      expect(
        adjusted.some(value => value <= base.alpha),
        `${method} ${rowCount}`
      ).toBe(floor.canReject);
      if (floor.canReject && atFloor > 1) {
        // One row fewer at the floor cannot reject.
        const fewer = pValues.map((value, row) => (row === atFloor - 1 ? 1 : value));
        expect(
          getMultipleTestingAdjustedPValues(fewer, method).some(value => value <= base.alpha),
          `${method} ${rowCount} minus one`
        ).toBe(false);
      }
    }
  }
});

it('getPermutationRejectionFloor uses the tail convention', () => {
  expect(
    getPermutationRejectionFloor({permutationCount: 999, rowCount: 1, alpha: 0.05}).pValueFloor
  ).toBe(0.002);
  for (const alternative of ['greater', 'lesser', 'folded', 'directed'] as const) {
    expect(
      getPermutationRejectionFloor({permutationCount: 999, alternative, rowCount: 1, alpha: 0.05})
        .pValueFloor
    ).toBe(0.001);
  }
  // 19 permutations: the one-sided floor 0.05 just rejects at alpha 0.05, two-sided cannot.
  expect(
    getPermutationRejectionFloor({
      permutationCount: 19,
      alternative: 'greater',
      rowCount: 1,
      alpha: 0.05
    }).canReject
  ).toBe(true);
  expect(
    getPermutationRejectionFloor({permutationCount: 19, rowCount: 1, alpha: 0.05}).canReject
  ).toBe(false);
});
