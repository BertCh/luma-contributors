// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Independent rational-moment/libm-sqrt literals from rate-standardization-audit.oracle.py. */
export const RATE_AUDIT_POPULATIONS = [10, 20, 40, 80, 160, 320, 1, 0, 10, 20];
export const RATE_AUDIT_MASK = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1];
const UNDER_EVENTS = [1, 2, 5, 9, 15, 34, 1000, 2, -1, NaN];
export const RATE_AUDIT_REFERENCE = [
  {
    name: 'negative prior',
    scale: 1,
    events: UNDER_EVENTS,
    rates: [
      -0.04874063593845721,
      -0.07255956013076786,
      0.492598449629783,
      0.39779353988536803,
      -0.4303489473092928,
      0.08224396186997113,
      NaN,
      NaN,
      NaN,
      NaN
    ],
    statistics: [6, 0.10476190476190476, 6.660997732426303e-5, -0.0009311224489795918, 630],
    negativeVarianceRows: 2
  },
  {
    name: 'replicated counts',
    scale: 2,
    events: UNDER_EVENTS,
    rates: [
      -0.06869040807568584,
      -0.10183030118679122,
      0.6832876903994197,
      0.5187564749039544,
      -0.6086053178377864,
      0.11631052629980886,
      NaN,
      NaN,
      NaN,
      NaN
    ],
    statistics: [6, 0.10476190476190476, 6.660997732426303e-5, -0.0004322562358276644, 1260],
    negativeVarianceRows: 2
  },
  {
    name: 'positive prior',
    scale: 1,
    events: [1, 20, 4, 40, 8, 150, 1000, 2, -1, NaN],
    rates: [
      -0.8801594470748222,
      2.5230699346575234,
      -1.066450436138267,
      0.6386271595903313,
      -1.3583631533754057,
      0.5186937064349271,
      NaN,
      NaN,
      NaN,
      NaN
    ],
    statistics: [6, 0.353968253968254, 0.051234252960443434, 0.04786312673217435, 630],
    negativeVarianceRows: 0
  },
  {
    name: 'no events',
    scale: 1,
    events: [0, 0, 0, 0, 0, 0, 1000, 2, -1, NaN],
    rates: [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
    statistics: [6, 0, 0, 0, 630],
    negativeVarianceRows: 0
  }
];
