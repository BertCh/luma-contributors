// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export const GEOGRAPHIC_INPUT = {
  positions: [-2, 0, 2, 0, 0, -1, 0, 1, 0, 0, 10, 0, 14, 0, 999, 999, NaN, 0],
  lineEnds: [0, 0, 2, 3, NaN, 0, 0, 1, -4, 0, 11, 0, 13, 0, 1000, 1000, 0, 0],
  weights: [1, 1, 1, 1, 4, 1, 1, 1, 1],
  groupIds: [0, 0, 0, 0, 0, 1, 1, 0, 0],
  mask: [1, 1, 1, 1, 1, 1, 1, 0, 1],
  groupCount: 3,
  vertexCount: 4,
  ellipseConvention: 'standard' as const
};

// geographic-distribution-audit.oracle.py: NumPy population covariance and eigh;
// exact symmetry/two-point midpoint medians, direct circular vector moments.
// Trigonometric residuals below 3e-16 are written as zero.
export const GEOGRAPHIC_EXPECTED = {
  counts: [5, 2, 0],
  weightSums: [8, 2, 0],
  meanCenters: [0, 0, 12, 0, NaN, NaN],
  medianCenters: [0, 0, 12, 0, NaN, NaN],
  medianConverged: [1, 1, 0],
  standardDistances: [1.118033988749895, 2, NaN],
  ellipses: [1.5707963267948966, 0.5, 1, 1.5707963267948966, 0, 2, NaN, NaN, NaN],
  directionalMeans: [2.819842099193151, 0.4729537233052701, 3.5, NaN, 1, 1, NaN, NaN, NaN],
  ellipseVertices: [
    0,
    0.5,
    -1,
    0,
    0,
    -0.5,
    1,
    0,
    12,
    0,
    10,
    0,
    12,
    0,
    14,
    0,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN
  ],
  circleVertices: [
    1.118033988749895,
    0,
    0,
    1.118033988749895,
    -1.118033988749895,
    0,
    0,
    -1.118033988749895,
    14,
    0,
    12,
    2,
    10,
    0,
    12,
    -2,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN
  ]
};

export const AXIAL_DIRECTIONAL_EXPECTED = [0, 1 / 3, 3.5, 0, 0, 1, NaN, NaN, NaN];

export const COINCIDENT_INPUT = {
  positions: [0, 0, 10, 0, -5, 0, 4, 7, 8, 8, 9, 1, 5, 2, 3, 2],
  weights: [3, 1, 2, 1, 1, 1, 1, 1],
  mask: [1, 1, 1, 0, 0, 0, 0, 0],
  groupCount: 1,
  maximumMedianIterations: 1,
  medianTolerance: 1e-6
};
