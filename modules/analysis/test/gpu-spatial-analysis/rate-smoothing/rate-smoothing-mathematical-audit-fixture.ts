// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Independent literals: /usr/bin/python3 rate-smoothing-mathematical-audit.oracle.py
export const RATE_AUDIT_EXPECTED = {
  global: {
    raw: [0, 2, null, 2, 0.5, 0, 0, null],
    standardized: [
      -0.9782458241220511,
      1.1287451816792897,
      null,
      1.1590011139293837,
      -0.45149807267171593,
      -0.9782458241220511,
      -0.9782458241220511,
      null
    ],
    smoothed: [
      0.09569648924122309,
      1.889580973952435,
      null,
      1.9417910447761193,
      0.5441676104190261,
      0.09569648924122309,
      0.09569648924122309,
      null
    ],
    excess: [0, 2.1538461538461537, null, 2.1538461538461537, 0.5384615384615384, 0, 0, null],
    summary: [
      6, 65, 70, 0.9285714285714286, 0.8877551020408164, 0.8081632653061226, 0.8081632653061226, 0
    ]
  },
  spatial: {
    spatialRate: [
      1,
      1.5,
      null,
      1.3333333333333333,
      0.16666666666666666,
      0.16666666666666666,
      0,
      null
    ],
    smoothed: [
      0.09999999999999998,
      1.9047619047619047,
      null,
      1.948717948717949,
      0.4000000000000001,
      0.04999999999999996,
      0,
      null
    ],
    priorMean: [
      1,
      1.5,
      null,
      1.3333333333333333,
      0.16666666666666666,
      0.16666666666666666,
      0,
      null
    ],
    priorVariance: [
      0.9,
      0.6375,
      null,
      0.8000000000000002,
      0.0388888888888889,
      0.0388888888888889,
      0,
      null
    ]
  }
} as const;

export function createRateAuditInput(populationScale = 1) {
  return {
    events: Float32Array.from([0, 20, 0, 40, 5, 0, 0, 10]),
    populations: Float32Array.from(
      [10, 10, 0, 20, 10, 10, 10, 10],
      value => value * populationScale
    ),
    mask: Uint32Array.from([1, 1, 1, 1, 1, 1, 1, 0]),
    offsets: [0, 2, 4, 6, 8, 10, 12, 12, 13],
    neighbors: [1, 2, 0, 3, 1, 4, 2, 5, 0, 5, 0, 4, 0]
  };
}
