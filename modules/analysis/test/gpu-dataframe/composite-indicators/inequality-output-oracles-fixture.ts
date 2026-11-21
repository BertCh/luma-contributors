// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUInequalityParameterValues} from '../../../src/dataframe-analysis/composite-indicators/inequality-parameters';

export const INEQUALITY_INPUT = {
  values: new Float32Array([1, 1, 3, 3, 1, 2, 6]),
  zoneIds: new Uint32Array([0, 0, 0, 0, 1, 1, 1]),
  parameters: getGPUInequalityParameterValues({
    epsilon: 1,
    palmaTopShare: 0.25,
    palmaBottomShare: 0.5
  }),
  zoneCount: 3,
  lorenzKnotCount: 5
};
// NumPy, inequality-output-oracles.oracle.py. Empty zone 2 independently implies count=0, NaN indices.
export const INEQUALITY_EXPECTED = {
  gini: [0.25, 0.37037037037037035, NaN],
  theilT: [0.13081203594113697, 0.2499267309416926, NaN],
  theilL: [0.14384103622589045, 0.2703100720721096, NaN],
  atkinson: [0.1339745962155613, 0.23685717163111208, NaN],
  hoover: [0.25, 0.3333333333333333, NaN],
  palma: [1.5, 2.25, NaN],
  mean: [2, 3, NaN],
  counts: [4, 3, 0],
  lorenzKnots: [
    0,
    0.125,
    0.25,
    0.625,
    1,
    0,
    0.08333333333333333,
    0.2222222222222222,
    0.5,
    1,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN
  ],
  globalSummary: [
    0.21437474079303134, 0.02050198396924728, 0.19387275682378408, 0.35294117647058826, 7,
    2.4285714285714284, 7, 17
  ]
};
