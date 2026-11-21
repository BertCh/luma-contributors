// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export const AUDIT_RANK_VALUES = [
  8,
  6,
  6,
  4,
  2,
  0,
  50,
  9,
  0,
  8,
  6,
  6,
  4,
  2,
  40,
  8,
  2,
  0,
  8,
  6,
  6,
  4,
  30,
  NaN,
  9,
  1,
  7,
  3,
  7,
  5,
  20,
  7
];
export const AUDIT_RANK_MASK = [1, 1, 1, 1, 1, 1, 0, 1];
const NO_RANK = 0xffffffff;
const GIDDY_RANKS = [
  0,
  2,
  1,
  3,
  4,
  5,
  NO_RANK,
  NO_RANK,
  5,
  0,
  2,
  1,
  3,
  4,
  NO_RANK,
  NO_RANK,
  4,
  5,
  0,
  2,
  1,
  3,
  NO_RANK,
  NO_RANK,
  0,
  5,
  2,
  4,
  1,
  3,
  NO_RANK,
  NO_RANK
];
const ROW_ID_RANKS = [
  0,
  1,
  2,
  3,
  4,
  5,
  NO_RANK,
  NO_RANK,
  5,
  0,
  1,
  2,
  3,
  4,
  NO_RANK,
  NO_RANK,
  4,
  5,
  0,
  1,
  2,
  3,
  NO_RANK,
  NO_RANK,
  0,
  5,
  1,
  4,
  2,
  3,
  NO_RANK,
  NO_RANK
];

/** Sparse literal count cells from audit-rank-markov.oracle.py; all omitted cells are zero. */
const GIDDY_COUNTS = [
  [2, 1],
  [5, 2],
  [9, 1],
  [10, 2],
  [16, 2],
  [20, 1],
  [25, 2],
  [27, 1],
  [32, 1],
  [35, 2],
  [44, 2],
  [45, 1]
];
const ROW_ID_COUNTS = [
  [1, 1],
  [5, 2],
  [8, 2],
  [12, 1],
  [17, 2],
  [18, 1],
  [26, 2],
  [27, 1],
  [32, 1],
  [35, 2],
  [44, 2],
  [45, 1]
];
const GEOGRAPHIC_COUNTS = [
  [1, 2],
  [3, 1],
  [9, 1],
  [10, 2],
  [16, 1],
  [19, 2],
  [26, 1],
  [28, 2],
  [36, 1],
  [37, 2],
  [40, 2],
  [45, 1]
];

export const AUDIT_RANK_CASES = (['full-rank', 'geographic-rank'] as const).flatMap(mode =>
  (['giddy', 'row-id'] as const).map(tieOrder => {
    const entries =
      mode === 'geographic-rank'
        ? GEOGRAPHIC_COUNTS
        : tieOrder === 'giddy'
          ? GIDDY_COUNTS
          : ROW_ID_COUNTS;
    const counts = new Array<number>(64).fill(0);
    const probabilities = new Array<number>(64).fill(0);
    for (const [cell, count] of entries) {
      counts[cell] = count;
      probabilities[cell] = count === 1 ? 0.3333333333333333 : 0.6666666666666666;
    }
    return {
      mode,
      tieOrder,
      expected: {
        ranks: tieOrder === 'giddy' ? GIDDY_RANKS : ROW_ID_RANKS,
        counts,
        probabilities,
        rowTotals: [3, 3, 3, 3, 3, 3, 0, 0],
        ignored: [6],
        mobility:
          tieOrder === 'giddy'
            ? [2, 2, 1.3333333333333333]
            : [1.6666666666666667, 1.6666666666666667, 1.3333333333333333],
        // Doubly stochastic irreducible active support: exactly 1/6, modulo solve roundoff.
        steadyStates: [
          0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666,
          0.16666666666666666, 0.16666666666666666, 0, 0
        ],
        stateSparsity: [6, 2],
        converged: [1]
      }
    };
  })
);

/** Hand derivation: no included location means every opportunity is ignored. */
export const AUDIT_RANK_EMPTY = {
  ranks: new Array<number>(32).fill(NO_RANK),
  counts: new Array<number>(64).fill(0),
  probabilities: new Array<number>(64).fill(0),
  rowTotals: new Array<number>(8).fill(0),
  ignored: [24],
  mobility: [NaN, NaN, NaN],
  steadyStates: new Array<number>(8).fill(NaN),
  stateSparsity: [0, 8],
  converged: [1]
};
