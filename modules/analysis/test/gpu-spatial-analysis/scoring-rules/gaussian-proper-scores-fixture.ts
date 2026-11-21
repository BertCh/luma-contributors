// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Two exponent ULPs near -33 contribute ~1.53e-5 after squaring; allow other f32 rounding too. */
export const GAUSSIAN_BRIER_TAIL_RELATIVE_TOLERANCE = 2e-5;

// Independent SciPy literals: /usr/bin/python3 gaussian-proper-scores.oracle.py
export const GAUSSIAN_BRIER_TAIL_EXPECTED = {
  means: [
    1.0000000150474662e30, -1.0000000150474662e30, 0, 0, 1.0000000150474662e30,
    -1.0000000150474662e30, 2, -2
  ],
  variances: [
    9.999999682655225e-21, 9.999999682655225e-21, 9.999999682655225e-21, 9.999999682655225e-21, 1,
    1, 1, 1
  ],
  observations: [1, 1, 0, 1, -1, -1, 0, 0],
  expected: [0, 1, 0.25, 0.25, 1, 0, 0.9550173046073012, 0.0005175685036595639]
} as const;

export const GAUSSIAN_PROPER_SCORES_EXPECTED = {
  base: {
    crps: [
      0.23369497725510913,
      1.9888480079549058,
      0.7263959108429516,
      2.561801939605749,
      1.4527918216859033,
      0,
      2,
      0,
      null,
      null,
      0.02336949746433642,
      null
    ],
    logScore: [
      0.9189385332046727,
      2.737085713764618,
      2.2257913526447273,
      2.9064397107616715,
      2.9189385332046727,
      null,
      null,
      null,
      null,
      null,
      -1.3836465709652441,
      null
    ],
    brierScore: [
      0.025171489600055125,
      0.4781203353511161,
      0.0005175685036595643,
      0.06375248151688427,
      0.9550173046073012,
      0,
      1,
      0,
      null,
      null,
      0.25,
      null
    ],
    intervalScore: [
      3.2897072394548257,
      6.579414478909651,
      5.19631736953235,
      9.869121718364477,
      10.3926347390647,
      0,
      39.99999940395356,
      0,
      null,
      null,
      0.32897072026894814,
      null
    ]
  },
  scaled: {
    crps: [
      0.46738995451021825,
      3.9776960159098116,
      1.4527918216859033,
      5.123603879211498,
      2.9055836433718065,
      0,
      4,
      0,
      null,
      null,
      0.04673899492867284,
      null
    ],
    logScore: [
      1.612085713764618,
      3.430232894324563,
      2.9189385332046727,
      3.5995868913216165,
      3.612085713764618,
      null,
      null,
      null,
      null,
      null,
      -0.6904993904052987,
      null
    ],
    brierScore: [
      0.025171489600055125,
      0.4781203353511161,
      0.0005175685036595643,
      0.06375248151688427,
      0.9550173046073012,
      0,
      1,
      0,
      null,
      null,
      0.25,
      null
    ],
    intervalScore: [
      6.579414478909653,
      13.158828957819303,
      10.392634739064686,
      19.73824343672895,
      20.785269478129372,
      0,
      79.99999880790712,
      0,
      null,
      null,
      0.6579414405378969,
      null
    ]
  },
  quantiles: [
    [2.350988701644575e-38, 12.949952701187202],
    [1.0000000031710769e-30, 11.523883577107101],
    [9.999999960041972e-13, 7.130506848721279],
    [0.00009999999747378752, 3.8905918925425738],
    [0.009999999776482582, 2.5758293112778565],
    [0.10000000149011612, 1.6448536197274128],
    [0.5, 0.6744897501960817],
    [0.9990000128746033, 0.0012532983294975893]
  ],
  erf: [
    [0, 0],
    [0.1, 0.1124629160182849],
    [0.5, 0.5204998778130465],
    [1, 0.8427007929497148],
    [2, 0.9953222650189527],
    [4, 0.9999999845827421],
    [5, 0.9999999999984626],
    [7, 1]
  ]
} as const;

export function createGaussianProperScoresInput(scaled = false) {
  const means = Float32Array.from([0, 0, 2, -1, 3, 0, 2, 1, 0, 0, 1, NaN]);
  const variances = Float32Array.from([1, 4, 0.25, 9, 1, 0, 0, 0, -1, NaN, 0.01, 1]);
  const observations = Float32Array.from([0, 3, 3, -5, 1, 0, 0, 1, 2, 0, 1, 2]);
  return scaled
    ? {
        means: means.map(value => 2 * value + 16),
        variances: variances.map(value => 4 * value),
        observations: observations.map(value => 2 * value + 16)
      }
    : {means, variances, observations};
}
