// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUNeighborhoodSummaryStatistic} from '../../../src/gpu-spatial-analysis/neighborhood-summary/gpu-neighborhood-summary';

export const STATISTICS: GPUNeighborhoodSummaryStatistic[] = [
  'count',
  'weightSum',
  'sum',
  'mean',
  'min',
  'max',
  'standardDeviation',
  'median'
];
export const INPUT = {
  offsets: Uint32Array.of(0, 6, 8, 8, 11, 13, 14, 15, 16),
  neighbors: Uint32Array.of(0, 1, 2, 3, 7, 99, 2, 4, 1, 4, 5, 1, 2, 6, 3, 0),
  weights: Float32Array.of(9, 1, 3, 4, 2, 1, 1, 1, 0, 2, 2, 1, 1, 1, 1, 1),
  values: Float32Array.of(10, 2, 6, NaN, 8, 4, -2, 99),
  categories: Uint32Array.of(5, 5, 3, 3, 7, 7, 3, 9),
  mask: Uint32Array.of(1, 1, 1, 1, 1, 1, 1, 0)
};

// NumPy literals from neighborhood-audit.oracle.py; each row lists all eight statistics.
export const RAW_EXPECTED = {
  table: [
    [2, 4, 20, 5, 2, 6, 1.7320508075688772, 4],
    [2, 2, 14, 7, 6, 8, 1, 7],
    [0, 0, 0, NaN, NaN, NaN, NaN, NaN],
    [2, 4, 24, 6, 4, 8, 2, 6],
    [2, 2, 8, 4, 2, 6, 2, 4],
    [1, 1, -2, -2, -2, -2, 0, -2],
    [0, 0, 0, NaN, NaN, NaN, NaN, NaN],
    [0, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
  ].flat(),
  modes: [3, 3, 4294967295, 7, 3, 3, 3, 4294967295],
  entropy: [0.37677016125643675, 0.6931471805599453, NaN, 0, 0.6931471805599453, 0, 0, NaN],
  overflow: [0]
};
export const STANDARDIZED_FOCAL_EXPECTED = {
  table: [
    [3, 3, 25, 8.333333333333334, 2, 10, 2.560381915956203, 6],
    [3, 3, 11, 3.6666666666666665, 2, 8, 2.4267032964268394, 6],
    [1, 2, 12, 6, 6, 6, 0, 6],
    [2, 1, 6, 6, 4, 8, 2, 6],
    [3, 3, 20, 6.666666666666667, 2, 8, 2.211083193570267, 6],
    [2, 3, 6, 2, -2, 4, 2.8284271247461903, 1],
    [1, 2, -4, -2, -2, -2, 0, -2],
    [0, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
  ].flat(),
  modes: [5, 5, 3, 3, 7, 7, 3, 4294967295],
  entropy: [
    0.6036372515003262,
    0.8675632284814612,
    0,
    0.6365141682948128,
    0.8675632284814612,
    0.6365141682948128,
    0,
    NaN
  ],
  overflow: [0]
};
export const INCOMPLETE_EXPECTED = {
  table: Array.from({length: 64}, (_, index) => (index === 56 ? 0 : NaN)),
  modes: new Array<number>(8).fill(4294967295),
  entropy: new Array<number>(8).fill(NaN),
  overflow: [0]
};
