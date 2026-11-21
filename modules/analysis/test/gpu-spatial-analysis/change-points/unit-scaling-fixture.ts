// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Independent NumPy enumeration of all 128 partitions, rejecting segments shorter than 2
// (3 for slope). SSE uses direct residuals; slope uses centered OLS; Gaussian costs use
// n * (log(2*pi*max(RSS/n, 1e-11*globalVariance)) + 1). Penalty per interior boundary.
// Mean at x1/penalty 2 and x10/penalty 200 has two exact mathematical optima: no split
// and a split at 4. Last-place recurrence rounding may choose either.
// Inputs were rounded to float32 before binary64 evaluation. No dynamic program or CPU twin.
export const UNIT_SCALING_PARTITIONS = [
  {
    cost: 'mean',
    scale: 1,
    penalty: 2,
    minimum: 2,
    values: [
      0.0, 0.10000000149011612, -0.10000000149011612, 0.0, 1.0, 1.100000023841858,
      0.8999999761581421, 1.0
    ],
    expectedCost: 2.0400000101327906,
    points: []
  },
  {
    cost: 'mean',
    scale: 10,
    penalty: 2,
    minimum: 2,
    values: [0.0, 1.0, -1.0, 0.0, 10.0, 11.0, 9.0, 10.0],
    expectedCost: 6.0,
    points: [4]
  },
  {
    cost: 'mean',
    scale: 10,
    penalty: 200,
    minimum: 2,
    values: [0.0, 1.0, -1.0, 0.0, 10.0, 11.0, 9.0, 10.0],
    expectedCost: 204.0,
    points: []
  },
  {
    cost: 'slope',
    scale: 1,
    penalty: 2,
    minimum: 3,
    values: [
      0.0, 0.10000000149011612, -0.10000000149011612, 0.0, 1.0, 1.100000023841858,
      0.8999999761581421, 1.0
    ],
    expectedCost: 0.5914285909703811,
    points: []
  },
  {
    cost: 'slope',
    scale: 10,
    penalty: 2,
    minimum: 3,
    values: [0.0, 1.0, -1.0, 0.0, 10.0, 11.0, 9.0, 10.0],
    expectedCost: 5.6000000000000005,
    points: [4]
  },
  {
    cost: 'slope',
    scale: 10,
    penalty: 200,
    minimum: 3,
    values: [0.0, 1.0, -1.0, 0.0, 10.0, 11.0, 9.0, 10.0],
    expectedCost: 59.142857142857146,
    points: []
  },
  {
    cost: 'variance',
    scale: 1,
    penalty: 2,
    minimum: 2,
    values: [
      0.0, 0.10000000149011612, -0.10000000149011612, 0.0, 1.0, 1.100000023841858,
      0.8999999761581421, 1.0
    ],
    expectedCost: 11.77108270042151,
    points: []
  },
  {
    cost: 'variance',
    scale: 10,
    penalty: 2,
    minimum: 2,
    values: [0.0, 1.0, -1.0, 0.0, 10.0, 11.0, 9.0, 10.0],
    expectedCost: 48.61244414858981,
    points: []
  },
  {
    cost: 'variance',
    scale: 10,
    penalty: 200,
    minimum: 2,
    values: [0.0, 1.0, -1.0, 0.0, 10.0, 11.0, 9.0, 10.0],
    expectedCost: 48.61244414858981,
    points: []
  },
  {
    cost: 'mean-variance',
    scale: 1,
    penalty: 2,
    minimum: 2,
    values: [
      0.0, 0.10000000149011612, -0.10000000149011612, 0.0, 1.0, 1.100000023841858,
      0.8999999761581421, 1.0
    ],
    expectedCost: -19.228697819031396,
    points: [2, 4, 6]
  },
  {
    cost: 'mean-variance',
    scale: 10,
    penalty: 2,
    minimum: 2,
    values: [0.0, 1.0, -1.0, 0.0, 10.0, 11.0, 9.0, 10.0],
    expectedCost: 17.612661642315636,
    points: [2, 4, 6]
  },
  {
    cost: 'mean-variance',
    scale: 10,
    penalty: 200,
    minimum: 2,
    values: [0.0, 1.0, -1.0, 0.0, 10.0, 11.0, 9.0, 10.0],
    expectedCost: 48.61244414858981,
    points: []
  }
] as const;
