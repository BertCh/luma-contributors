// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Closed-form weighted category MLEs from generalized-gwr-audit.oracle.py (float64). */
export const GENERALIZED_AUDIT_MODELS = [
  {
    family: 'poisson',
    kernel: 'gaussian',
    bandwidthMode: 'fixed',
    sites: [
      [
        0.5372051165760972, 0.0259754962002971, 1.8893438894752617, 2.555328055262369,
        0.19705078467073697, 0.18510935175412305, 5, 5
      ],
      [
        0.5704548960003755, -0.05154395600710643, 2.110656110524738, 2.4446719447376313,
        0.19296473447810725, 0.18510935175412296, 5, 5
      ],
      [0.5547189562170504, -0.013428224342895145, 2, 2.5, 0.19364916731037074, 0, 5, 5]
    ],
    aicc: [21.0037449969809, 20.97149485067517],
    loo: [1.7102592757769646, 1.7203422094015142]
  },
  {
    family: 'poisson',
    kernel: 'gaussian',
    bandwidthMode: 'adaptive',
    sites: [
      [
        0.5133147687043792, 0.075796888711397, 1.7550860376057904, 2.622456981197105,
        0.20495363138459052, 0.20748566039903496, 5, 5
      ],
      [
        0.5873674028833384, -0.09629979347591727, 2.2449139623942096, 2.3775430188028945,
        0.19557628613782957, 0.207485660399035, 5, 5
      ],
      [0.5547189562170503, -0.0134282243428952, 2, 2.5, 0.19364916731037085, 0, 5, 5]
    ],
    aicc: [21.0037449969809, 20.97149485067517],
    loo: [1.7102592757769646, 1.7203422094015142]
  },
  {
    family: 'poisson',
    kernel: 'bisquare',
    bandwidthMode: 'fixed',
    sites: [
      [
        0.4515752140212351, 0.19015830088376515, 1.4716981132075473, 2.764150943396227,
        0.23557386747917022, 0.25471698113207547, 5, 5
      ],
      [
        0.6160844667109268, -0.18646341968786273, 2.5283018867924527, 2.2358490566037736,
        0.211935724987988, 0.25471698113207547, 5, 5
      ],
      [0.5547189562170503, -0.01342822434289509, 2, 2.5, 0.19364916731037074, 0, 5, 5]
    ],
    aicc: [23.527584018856253, 22.040337660725395],
    loo: [1.7953958564929575, 1.7005016482620754]
  },
  {
    family: 'poisson',
    kernel: 'bisquare',
    bandwidthMode: 'adaptive',
    sites: [
      [
        0.29930614466737815, 0.42430614386740206, 1.000000000799976, 2.9999999996000115,
        0.33333333310556257, 0.33333333320000413, 5, 5
      ],
      [
        0.6458797345806953, -0.3277325538207559, 2.9999999992000244, 2.000000000399988,
        0.2635231382402021, 0.333333333200004, 6, 6
      ],
      [0.5547189562170503, -0.01342822434289509, 2, 2.5, 0.19364916731037085, 0, 5, 5]
    ],
    aicc: [23.527584018856253, 22.040337660725395],
    loo: [1.7953958564929575, 1.7005016482620754]
  },
  {
    family: 'logistic',
    kernel: 'gaussian',
    bandwidthMode: 'fixed',
    sites: [
      [
        0, -0.17619577653147747, 0.48155731491254367, 0.5184426850874563, 0.5812698286169211,
        0.18510935175412305, 3, 3
      ],
      [
        0, -0.32380422346852256, 0.5184426850874564, 0.48155731491254383, 0.5812698286169208,
        0.185109351754123, 3, 3
      ],
      [0, -0.25, 0.5, 0.5, 0.5773502691896256, 0, 3, 3]
    ],
    aicc: [22.650958960001198, 22.42948460826682],
    loo: [1.8832573094250202, 1.866002740626784]
  },
  {
    family: 'logistic',
    kernel: 'gaussian',
    bandwidthMode: 'adaptive',
    sites: [
      [
        0, -0.08635983583375563, 0.4591810062676317, 0.5408189937323683, 0.5964044434917534,
        0.207485660399035, 3, 3
      ],
      [
        0, -0.41364016416624444, 0.5408189937323682, 0.4591810062676316, 0.5964044434917534,
        0.20748566039903502, 3, 3
      ],
      [0, -0.25, 0.5, 0.5, 0.5773502691896257, 0, 3, 3]
    ],
    aicc: [22.650958960001198, 22.42948460826682],
    loo: [1.8832573094250202, 1.866002740626784]
  },
  {
    family: 'logistic',
    kernel: 'bisquare',
    bandwidthMode: 'fixed',
    sites: [
      [
        0, 0.10591129365343499, 0.4119496855345912, 0.5880503144654088, 0.6633348044720851,
        0.2547169811320755, 3, 3
      ],
      [
        0, -0.605911293653435, 0.5880503144654088, 0.4119496855345912, 0.6633348044720851,
        0.2547169811320755, 3, 3
      ],
      [0, -0.25, 0.5, 0.5, 0.5773502691896258, 0, 3, 3]
    ],
    aicc: [26.74650886865732, 24.82507112574093],
    loo: [2.4792010549324695, 2.110186153168742]
  },
  {
    family: 'logistic',
    kernel: 'bisquare',
    bandwidthMode: 'adaptive',
    sites: [
      [
        0, 0.4431471799599633, 0.3333333334666626, 0.6666666665333373, 0.8660254033514387,
        0.33333333320000397, 4, 4
      ],
      [
        0, -0.9431471799599633, 0.6666666665333374, 0.3333333334666627, 0.8660254033514391,
        0.333333333200004, 4, 4
      ],
      [0, -0.25, 0.5, 0.5, 0.5773502691896256, 0, 3, 3]
    ],
    aicc: [26.74650886865732, 24.82507112574093],
    loo: [2.4792010549324695, 2.110186153168742]
  }
] as const;

export type GeneralizedAuditModel = (typeof GENERALIZED_AUDIT_MODELS)[number];
export type GeneralizedAuditReviewCase =
  | 'large-counts'
  | 'offset-shift'
  | 'exposure-shift-up'
  | 'exposure-shift-down'
  | 'extrapolation'
  | 'singular';

export function applyGeneralizedAuditReviewCase(
  fixture: ReturnType<typeof createGeneralizedAuditFixture>,
  reviewCase?: GeneralizedAuditReviewCase
) {
  if (reviewCase === 'large-counts') {
    fixture.response = fixture.response.map(
      (value, row) => value + 10000 + (row >= 6 && row < 12 ? 3 : 0)
    );
    fixture.offset = fixture.offset.map(value => value + Math.log(10000));
  } else if (reviewCase === 'offset-shift') {
    fixture.offset = fixture.offset.map(value => value - 14);
  } else if (reviewCase === 'exposure-shift-up' || reviewCase === 'exposure-shift-down') {
    fixture.offset = fixture.offset.map(
      value => value + (reviewCase === 'exposure-shift-up' ? 100 : -100)
    );
  } else if (reviewCase === 'extrapolation') {
    fixture.predictors[12] = 1000;
  } else if (reviewCase === 'singular') {
    fixture.predictors.fill(1);
  }
}

export function createGeneralizedAuditFixture(model: GeneralizedAuditModel, rescaled = false) {
  const predictors = [-1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1];
  const positions = [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 1, 1].flatMap(value =>
    rescaled ? [32 + 4 * value, -16] : [value, 0]
  );
  const offset = predictors.map(value =>
    model.family === 'poisson' ? 0.25 + 0.125 * value : 0.25 * value
  );
  const response =
    model.family === 'poisson'
      ? [0, 1, 2, 1, 3, 5, 2, 3, 4, 0, 2, 4, NaN, NaN]
      : [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, NaN, NaN];
  const mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0];
  const coefficients: number[] = [];
  const fitted: number[] = [];
  const standardErrors: number[] = [];
  const influence: number[] = [];
  const iterationCount: number[] = [];
  for (let row = 0; row < 14; row++) {
    const site = model.sites[Math.min(Math.floor(row / 6), 2)];
    const positive = predictors[row] === 1;
    coefficients.push(site[0], site[1] / (rescaled ? 2 : 1));
    fitted.push(site[positive ? 3 : 2]);
    standardErrors.push(site[4], site[4] / (rescaled ? 2 : 1));
    influence.push(site[5]);
    iterationCount.push(site[positive ? 7 : 6]);
  }
  return {
    positions,
    predictors: predictors.map(value => value * (rescaled ? 2 : 1)),
    response,
    offset,
    mask,
    parameters: [model.bandwidthMode === 'adaptive' ? 8 : rescaled ? 12 : 3, 1e-5],
    expected: {
      coefficients,
      fitted,
      standardErrors,
      influence,
      iterationCount,
      codes: Array<number>(14).fill(0)
    }
  };
}
