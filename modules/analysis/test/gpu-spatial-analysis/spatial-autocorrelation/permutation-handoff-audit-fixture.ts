// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Dense-definition and independent Random123/rational oracle: permutation-handoff-audit.oracle.py. */
export const HANDOFF_REFERENCE = {
  localBivariateMoran: [
    0.6290306418729622,
    -0.10250869719411232,
    1.4607489350161011,
    0.006989229354143995,
    2.977411704865354,
    1.901070384327174,
    3.6297397779187954,
    NaN,
    NaN,
    NaN
  ],
  localLeeL: [
    1.345426650672725,
    -0.1630820182633605,
    0.7746395867509631,
    -0.1766721864519733,
    -4.4385489304011285,
    7.567005647419929,
    -2.1173482037859648,
    NaN,
    NaN,
    NaN
  ],
  localJoinCount: [0, 1, 2, 1, 0, 1, 1, NaN, NaN, NaN],
  BJC: [0, 0, 0, 0, 0, 1, 0, NaN, NaN, NaN],
  CLC: [0, 1, 0, 0, 0, 0, 0, NaN, NaN, NaN]
};
export const HANDOFF_BINARY_SECOND = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0];
export const HANDOFF_GLOBAL_LEE = 0.042944931476018314;
export const LOSH_HANDOFF_REFERENCE = [
  {
    power: 1,
    statistic: [
      1.3235294117647058, 0.9558823529411765, 0.29411764705882354, 1.3602941176470589,
      1.0661764705882353
    ],
    normalized: [
      1.6911764705882353, 0.14705882352941177, 0.22058823529411764, 0.4411764705882353, 2.5
    ]
  },
  {
    power: 2,
    statistic: [
      1.6724336793540946, 0.7756632064590542, 0.05767012687427912, 1.6796424452133796,
      0.8145905420991926
    ],
    normalized: [
      1.525374855824683, 0.011534025374855825, 0.025951557093425604, 0.10380622837370242,
      3.3333333333333335
    ]
  }
] as const;
export const LOSH_HANDOFF_TAILS = [
  {
    alternative: 'greater',
    exceedances: [7, 16, 19, 7, 2],
    probabilities: [0.4, 0.85, 1, 0.4, 0.15],
    significant: [1, 0, 0, 1, 1]
  },
  {
    alternative: 'lesser',
    exceedances: [14, 5, 4, 16, 19],
    probabilities: [0.75, 0.3, 0.25, 0.85, 1],
    significant: [0, 1, 1, 0, 0]
  },
  {
    alternative: 'two-sided',
    exceedances: [7, 5, 4, 7, 2],
    probabilities: [0.8, 0.6, 0.5, 0.8, 0.3],
    significant: [0, 1, 1, 0, 1]
  }
] as const;
