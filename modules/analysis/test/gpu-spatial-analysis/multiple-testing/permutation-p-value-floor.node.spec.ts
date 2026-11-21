// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getMultipleTestingAdjustedPValues,
  getPermutationRejectionFloor
} from '../../../src/gpu-spatial-analysis/multiple-testing/index';

it('W5-13 resolves the H73/128 boundary in both floor and inverse budget calculations', () => {
  // Python Fraction: H73/128 exceeds the first binary64 cutoff by 3.91e-18;
  // the next representable cutoff exceeds H73/128. Neither boolean uses a rounded H73.
  for (const [alpha, expected, requiredPermutations] of [
    [0.03808209987270469, false, 256],
    [0.038082099872704695, true, 255]
  ] as const) {
    expect(
      getPermutationRejectionFloor({
        permutationCount: 255,
        rowCount: 73,
        method: 'benjamini-yekutieli',
        alpha,
        contributorBudget: {maximumPermutations: 255}
      })
    ).toMatchObject({
      canReject: expected,
      minimumRowsAtFloor: expected ? 73 : Infinity,
      requiredPermutations,
      attainableWithinBudget: expected
    });
  }
});

it('W5-13 resolves larger harmonic boundaries without a data-sized sum', () => {
  // Exact Fraction sums in scratch/boundaries.py bracket H_m/128 by adjacent binary64 values.
  for (const [rowCount, lower, upper] of [
    [129, 0.04250702367635678, 0.04250702367635679],
    [1000, 0.058480241098049564, 0.05848024109804957]
  ]) {
    for (const [alpha, expected] of [
      [lower, false],
      [upper, true]
    ] as const) {
      expect(
        getPermutationRejectionFloor({
          permutationCount: 255,
          rowCount,
          method: 'benjamini-yekutieli',
          alpha
        }).canReject
      ).toBe(expected);
    }
  }
  expect(
    getPermutationRejectionFloor({
      permutationCount: 255,
      rowCount: 73,
      method: 'benjamini-yekutieli',
      alpha: Number.MIN_VALUE
    })
  ).toMatchObject({canReject: false, minimumRowsAtFloor: Infinity, requiredPermutations: Infinity});
});

it('B6 uses the same forward host cutoff for required draws and budget attainability', () => {
  const props = {
    permutationCount: 39,
    rowCount: 3,
    alpha: 0.15,
    method: 'bonferroni' as const,
    contributorBudget: {maximumPermutations: 39}
  };
  expect(getPermutationRejectionFloor(props)).toMatchObject({
    requiredPermutations: 40,
    attainableWithinBudget: false,
    canReject: false
  });
  expect(
    getPermutationRejectionFloor({
      ...props,
      permutationCount: 40,
      contributorBudget: {maximumPermutations: 40}
    })
  ).toMatchObject({requiredPermutations: 40, attainableWithinBudget: true, canReject: true});
});

it('NUM-05 separates sidedness from the p-value counting construction', () => {
  const base = {permutationCount: 19, rowCount: 1, alpha: 0.05, alternative: 'two-sided' as const};
  expect(
    getPermutationRejectionFloor({...base, pValueConstruction: 'upper-tail-count'})
  ).toMatchObject({pValueFloor: 0.05, canReject: true});
  expect(
    getPermutationRejectionFloor({...base, pValueConstruction: 'doubled-minimum-tail'})
  ).toMatchObject({pValueFloor: 0.1, canReject: false});
});

it('UST-14 reports the required draws and whether a contributor budget can attain them', () => {
  // 2 / (P + 1) <= .05 / 50000 gives P >= 1,999,999 exactly.
  const result = getPermutationRejectionFloor({
    permutationCount: 2 ** 20,
    rowCount: 50000,
    alpha: 0.05,
    method: 'bonferroni',
    contributorBudget: {maximumPermutations: 2 ** 20}
  });
  expect(result).toMatchObject({
    requiredPermutations: 1999999,
    attainableWithinBudget: false,
    maximumRowCount: 26214
  });
  expect(
    getPermutationRejectionFloor({
      permutationCount: 9,
      rowCount: 50000,
      alpha: 0.05,
      method: 'bonferroni',
      contributorBudget: {maximumPermutations: 2 ** 20, globalRowBlockCount: 25}
    })
  ).toMatchObject({
    maximumSupportedPermutations: 671087,
    requiredPermutations: 1999999,
    attainableWithinBudget: false
  });
  expect(() =>
    getPermutationRejectionFloor({
      permutationCount: 9,
      rowCount: 1,
      alpha: 0.05,
      contributorBudget: {maximumPermutations: 2 ** 20 + 1}
    })
  ).toThrow(/maximumPermutations/);
});

it('review 2 R4 BY cannot reject just below H68/128 and alpha one needs zero floor rows', () => {
  // Exact Fraction sum(1/k,k=1..68)/128 exceeds alpha by 7.38e-18.
  const alpha = 0.037531705388509126;
  expect(
    getMultipleTestingAdjustedPValues(new Array(68).fill(1 / 128), 'benjamini-yekutieli').some(
      value => value <= alpha
    )
  ).toBe(false);
  expect(
    getPermutationRejectionFloor({
      permutationCount: 255,
      rowCount: 68,
      method: 'benjamini-yekutieli',
      alpha
    })
  ).toMatchObject({canReject: false, minimumRowsAtFloor: Infinity});
  for (const method of [
    'none',
    'bonferroni',
    'holm',
    'benjamini-hochberg',
    'benjamini-yekutieli'
  ] as const) {
    expect(
      getPermutationRejectionFloor({permutationCount: 99, rowCount: 3, method, alpha: 1})
        .minimumRowsAtFloor
    ).toBe(0);
  }
});

it('R4 respects host alpha and the adjustment cap at one without a comparison slack', () => {
  for (const method of ['bonferroni', 'holm', 'benjamini-yekutieli'] as const) {
    expect(Array.from(getMultipleTestingAdjustedPValues([1, 1], method))).toEqual([1, 1]);
    expect(
      getPermutationRejectionFloor({permutationCount: 1, rowCount: 2, alpha: 1, method})
    ).toMatchObject({canReject: true, minimumRowsAtFloor: 0});
    if (method !== 'benjamini-yekutieli')
      expect(
        getPermutationRejectionFloor({permutationCount: 1, rowCount: 2, alpha: 1, method})
          .maximumRowCount
      ).toBe(Infinity);
  }
  expect(
    getPermutationRejectionFloor({
      permutationCount: 19,
      alternative: 'greater',
      rowCount: 1,
      alpha: 0.04999999999999
    }).canReject
  ).toBe(false);
  expect(() =>
    getPermutationRejectionFloor({permutationCount: 19, rowCount: 1, alpha: 1.00000001})
  ).toThrow(/alpha/);
  expect(
    getPermutationRejectionFloor({permutationCount: 19, rowCount: 1, alpha: 1e-50}).canReject
  ).toBe(false);
});

it('R5 floor occupancy is sufficient with all other p-values one, not necessary in general', () => {
  const floor = getPermutationRejectionFloor({
    permutationCount: 99,
    rowCount: 8,
    alpha: 0.05,
    method: 'benjamini-hochberg'
  });
  expect(floor).toMatchObject({pValueFloor: 0.02, canReject: true, minimumRowsAtFloor: 4});
  // Closed form: at rank eight, .04 * 8 / 8 = .04; none sits at the .02 floor.
  expect(
    Array.from(getMultipleTestingAdjustedPValues(new Array(8).fill(0.04), 'benjamini-hochberg'))
  ).toEqual(new Array(8).fill(0.04));
  expect(
    getMultipleTestingAdjustedPValues([0.02, 0.02, 0.02, 1, 1, 1, 1, 1], 'benjamini-hochberg')[0]
  ).toBeGreaterThan(0.05);
});

it('getPermutationRejectionFloor reproduces the review figures at 999 permutations', () => {
  const base = {permutationCount: 999, alpha: 0.05};
  expect(getPermutationRejectionFloor({...base, rowCount: 10000}).pValueFloor).toBe(0.002);

  // Benjamini-Hochberg with 10,000 rows needs 400 rows at the floor.
  const hochberg = getPermutationRejectionFloor({
    ...base,
    rowCount: 10000,
    method: 'benjamini-hochberg'
  });
  expect(hochberg).toMatchObject({pValueFloor: 0.002, canReject: true, minimumRowsAtFloor: 400});

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

it('getPermutationRejectionFloor refuses invalid counts, levels and alternatives before reporting a floor', () => {
  const valid = {permutationCount: 99, rowCount: 8, alpha: 0.05};
  for (const permutationCount of [-1, 0, 1.5, Number.NaN, 2 ** 31]) {
    expect(() => getPermutationRejectionFloor({...valid, permutationCount})).toThrow(
      /permutationCount/
    );
  }
  for (const rowCount of [-1, 1.5, Number.NaN, 2 ** 32]) {
    expect(() => getPermutationRejectionFloor({...valid, rowCount})).toThrow(/rowCount/);
  }
  for (const alpha of [0, -0.1, 1.1, Number.NaN, Infinity]) {
    expect(() => getPermutationRejectionFloor({...valid, alpha})).toThrow(/alpha/);
  }
  expect(() =>
    getPermutationRejectionFloor({...valid, alternative: 'invalid' as 'greater'})
  ).toThrow(/alternative/);
  // Empty families are valid but cannot reject; alpha=1 is a valid level.
  expect(getPermutationRejectionFloor({...valid, rowCount: 0}).canReject).toBe(false);
  expect(getPermutationRejectionFloor({...valid, alpha: 1}).canReject).toBe(true);
});
