// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Independent dense NumPy/SciPy values from rr-weights-regression-review.oracle.py.
// Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/rr-weights-regression-review.oracle.py
export const REVIEW_REGRESSION_FIXTURE = {
  predictors: Float32Array.of(0, 1, 4, 2, 5, 3, 7, 6),
  response: Float32Array.of(1, 3, 2, 5, 4, 8, 7, 10),
  weights: {
    offsets: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8),
    neighbors: Uint32Array.of(1, 2, 3, 4, 5, 6, 7, 0),
    weights: Float32Array.of(1, 1, 1, 1, 1, 1, 1, 1)
  },
  twoStageCoefficients: [1.294032023289664, 0.6055312954876272, 0.3173216885007281],
  errorCoefficients: [1.1956192552901095, 1.086965927059969, -0.3865080026872177],
  twoStageResiduals: [
    -1.2459970887918486, 0.4657933042212523, -3.302765647743813, 1.2256186317321687,
    -2.860262008733626, 2.6681222707423577, -1.705967976710335, 4.755458515283845
  ],
  errorResiduals: [
    -0.19561925529010948, 0.7174148176499218, -3.5434829635299856, 1.6304488905899523,
    -2.630448890589955, 3.543482963529984, -1.804380744709892, 2.2825851823500765
  ],
  // Classic LM p-values 0.42292566/0.10407001 exceed 0.05, so the clean decision is OLS (0).
  specificationDecision: 0
};
