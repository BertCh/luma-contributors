// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Literal NumPy/SciPy expectations from audit-dynamics.oracle.py, never production output. */
export const AUDIT_FIT_VALUES = [-4, -1, -1, 2, 2, 8, 10, 20, 50, NaN, Infinity, 7];
export const AUDIT_FIT_CLASSES = [0, 0, 0, 1, 1, 1, 3, 3, 3, 0, 1, 0xffffffff];
export const AUDIT_FIT_MASK = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1];
export const AUDIT_FIT_EXPECTED = [
  {
    counts: [3, 3, 0, 2, 8],
    medians: [-1, 2, 0, 15, 2],
    absolute: [3, 6, 0, 10, 44],
    squared: [6, 24, 0, 50, 428],
    summary: [0.5681818181818181, 19, 44, 0.8130841121495327, 80]
  },
  {
    counts: [3, 3, 0, 2, 8],
    medians: [1, 7, 0, 33, 7],
    absolute: [6, 12, 0, 20, 88],
    squared: [24, 96, 0, 200, 1712],
    summary: [0.5681818181818181, 38, 88, 0.8130841121495327, 320]
  }
];

export const AUDIT_PANEL = [
  0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0xffffffff, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1,
  1, 0
];
export const AUDIT_CONDITIONS = [
  0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0
];
export const AUDIT_PANEL_MASK = [1, 1, 1, 1, 1, 1, 1, 0];
export const AUDIT_TRANSITIONS = [
  {
    periodLag: 1,
    counts: [4, 0, 0, 2, 3, 0, 0, 0, 0, 0, 5, 0, 3, 1, 0, 0, 0, 0],
    probabilities: [1, 0, 0, 0.4, 0.6, 0, 0, 0, 0, 0, 1, 0, 0.75, 0.25, 0, 0, 0, 0],
    rowTotals: [4, 5, 0, 5, 4, 0],
    ignored: [6],
    homogeneity: [10.1025, 0.0064013267846546006, 13.501818930460608, 0.0011698152305735054, 2],
    steadyStates: [0.5, 0.5, 0, NaN, NaN, NaN, 0.42857142857142855, 0.5714285714285714, 0],
    stateSparsity: [2, 2],
    converged: [0]
  },
  {
    periodLag: 2,
    counts: [1, 3, 0, 2, 1, 0, 0, 0, 0, 1, 2, 0, 3, 0, 0, 0, 0, 0],
    probabilities: [
      0.25, 0.75, 0, 0.6666666666666666, 0.3333333333333333, 0, 0, 0, 0, 0.3333333333333333,
      0.6666666666666666, 0, 1, 0, 0, 0, 0, 0
    ],
    rowTotals: [4, 3, 0, 3, 3, 0],
    ignored: [3],
    homogeneity: [
      1.2583333333333333, 0.5330358124868246, 1.6456575701010374, 0.4391875285380541, 2
    ],
    steadyStates: [
      0.5384615384615385, 0.4615384615384615, 0, 0.47058823529411764, 0.5294117647058824, 0, 0.6,
      0.4, 0
    ],
    stateSparsity: [4, 2],
    converged: [1]
  }
];

export const AUDIT_SUPPORT_CASES = [
  {counts: [1, 0, 1, 0], steadyStates: [NaN, NaN], stateSparsity: [2, 0], converged: [0]},
  {counts: [0, 1, 0, 1], steadyStates: [NaN, NaN], stateSparsity: [2, 0], converged: [0]},
  {counts: [0, 1, 1, 0], steadyStates: [0.5, 0.5], stateSparsity: [2, 0], converged: [1]},
  {counts: [4, 0, 0, 0], steadyStates: [1, 0], stateSparsity: [1, 1], converged: [1]}
];

/** Independent SciPy boundary fixture: 128 ordinary chains, 128 reducible identities, one empty chain. */
export const AUDIT_PARALLEL_DIAGNOSTICS = {
  counts: [
    ...Array.from({length: 128}, () => [1, 3, 0, 2, 2, 0, 0, 0, 0]).flat(),
    ...Array.from({length: 128}, () => [4, 0, 0, 0, 4, 0, 0, 0, 0]).flat(),
    ...new Array<number>(9).fill(0)
  ],
  // scipy chi2.sf(LR,510)=3.5516167371654106e-60, which underflows float32 to zero.
  homogeneity: [955.7333333333333, 1.6504487245806922e-29, 1220.9299869438228, 0, 510],
  // The pooled chain and 128 ordinary chains have pi=[.4,.6,0]; identities/empty are NaN.
  steadyStates: [
    ...Array.from({length: 129}, () => [0.4, 0.6, 0]).flat(),
    ...new Array<number>(129 * 3).fill(NaN)
  ],
  stateSparsity: [512, 259],
  converged: [0]
};
