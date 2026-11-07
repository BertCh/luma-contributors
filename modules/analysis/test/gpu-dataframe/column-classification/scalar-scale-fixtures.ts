// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUScalarScaleControls} from '../../../src/dataframe-analysis/column-classification/scalar-scale-parameters';

export type ScalarScaleFixture = {
  name: string;
  controls: GPUScalarScaleControls;
  values: (number | null)[];
  validity?: number[];
  expected: (number | null)[];
};

// Hand-written fixtures independently checked against pinned d3-scale 4.0.2 in the node suite.
export const scalarScaleFixtures: ScalarScaleFixture[] = [
  {
    name: 'native inverted fixed height extrapolates with nullable source chunks',
    controls: {scale: 'identity', domain: [10, 30], invert: true, nullValue: -99},
    values: [-10, 10, null, 20, 30, 50, 25],
    validity: [1, 1, 0, 1, 1, 1, 0],
    expected: [50, 30, null, 20, 10, -10, null]
  },
  {
    name: 'native inverted fixed height descending domain',
    controls: {scale: 'identity', domain: [30, 10], invert: true},
    values: [-10, 10, 20, 30, 50],
    expected: [50, 30, 20, 10, -10]
  },
  {
    name: 'native inverted fixed height constant domain',
    controls: {scale: 'identity', domain: [7, 7], invert: true},
    values: [-100, 0, 7, 100, null],
    expected: [7, 7, 7, 7, null]
  },
  {
    name: 'native inverted fixed height spans opposite large endpoints',
    controls: {
      scale: 'identity',
      domain: [-Math.fround(3e38), Math.fround(3e38)],
      invert: true
    },
    values: [-Math.fround(3e38), 0, Math.fround(3e38)],
    expected: [Math.fround(3e38), 0, -Math.fround(3e38)]
  },
  {
    name: 'native inverted fixed height large endpoints avoid sum overflow',
    controls: {
      scale: 'identity',
      domain: [Math.fround(2e38), Math.fround(3e38)],
      invert: true
    },
    values: [Math.fround(2e38), Math.fround(2.5e38), Math.fround(3e38)],
    expected: [
      Math.fround(3e38),
      Math.fround(2e38) + Math.fround(3e38) - Math.fround(2.5e38),
      Math.fround(2e38)
    ]
  },
  {
    name: 'descending fixed scale remains identity',
    controls: {scale: 'identity', domain: [100, -100]},
    values: [-200, 0, 50, 200],
    expected: [-200, 0, 50, 200]
  },
  {
    name: 'finite double output overflowing float32 is explicitly invalid',
    controls: {scale: 'linear', domain: [0, 1], range: [0, Math.fround(3e38)], nullValue: -99},
    values: [0, 1, 2],
    expected: [0, Math.fround(3e38), null]
  },
  {
    name: 'linear extrapolates',
    controls: {scale: 'linear', domain: [0, 10], range: [0, 5]},
    values: [-10, 0, 5, 10, 20],
    expected: [-5, 0, 2.5, 5, 10]
  },
  {
    name: 'descending domain',
    controls: {scale: 'linear', domain: [10, 0], range: [2, 12]},
    values: [-10, 0, 5, 10, 20],
    expected: [22, 12, 7, 2, -8]
  },
  {
    name: 'descending range',
    controls: {scale: 'linear', domain: [0, 10], range: [12, 2]},
    values: [0, 5, 10],
    expected: [12, 7, 2]
  },
  {
    name: 'constant domain midpoint',
    controls: {scale: 'linear', domain: [7, 7], range: [-8, 12]},
    values: [-10, 0, 7, 100],
    expected: [2, 2, 2, 2]
  },
  {
    name: 'signed sqrt crosses zero',
    controls: {scale: 'sqrt', domain: [-100, 100], range: [0, 10]},
    values: [-400, -100, -25, -0, 25, 100, 400],
    expected: [-5, 0, 2.5, 5, 7.5, 10, 15]
  },
  {
    name: 'descending signed sqrt',
    controls: {scale: 'sqrt', domain: [100, -100], range: [0, 10]},
    values: [-100, -25, 0, 25, 100],
    expected: [10, 7.5, 5, 2.5, 0]
  },
  {
    name: 'positive logarithm',
    controls: {scale: 'log', domain: [1, 100], range: [0, 10]},
    values: [0.25, 1, 10, 100, 1000],
    expected: [-3.010299956639812, 0, 5, 10, 15]
  },
  {
    name: 'negative logarithm',
    controls: {scale: 'log', domain: [-100, -1], range: [0, 10]},
    values: [-1000, -100, -10, -1, -0.25],
    expected: [-5, 0, 5, 10, 13.010299956639813]
  },
  {
    name: 'descending negative logarithm',
    controls: {scale: 'log', domain: [-1, -100], range: [0, 10]},
    values: [-1000, -100, -10, -1],
    expected: [15, 10, 5, 0]
  },
  {
    name: 'wrong-sign and zero log inputs are outside finite parity',
    controls: {scale: 'log', domain: [1, 100], range: [0, 10], nullValue: -99},
    values: [-2, 0, 1, 100],
    expected: [null, null, 0, 10]
  },
  {
    name: 'degenerate log ignores transform as native normalization is constant',
    controls: {scale: 'log', domain: [-1, -1], range: [4, 8]},
    values: [-100, -1, 0, 1, 100],
    expected: [6, 6, 6, 6, 6]
  },
  {
    name: 'native fixed scale identity',
    controls: {scale: 'identity', domain: [-100, 100]},
    values: [-200, -1, 0, 10, 200],
    expected: [-200, -1, 0, 10, 200]
  },
  {
    name: 'native fixed equal domain is constant',
    controls: {scale: 'identity', domain: [7, 7]},
    values: [-1, 0, 7, 99],
    expected: [7, 7, 7, 7]
  },
  {
    name: 'ordered categorical point domain',
    controls: {scale: 'point', domain: [8, -3, 42], range: [2, 10]},
    values: [-3, 42, 8, 0],
    expected: [6, 10, 2, null]
  },
  {
    name: 'point reversed range',
    controls: {scale: 'point', domain: [8, -3, 42], range: [10, 2]},
    values: [-3, 42, 8],
    expected: [6, 2, 10]
  },
  {
    name: 'one category point midpoint',
    controls: {scale: 'point', domain: [8], range: [2, 10]},
    values: [8, 0],
    expected: [6, null]
  },
  {
    name: 'empty point domain',
    controls: {scale: 'point', domain: [], range: [2, 10]},
    values: [8, 0],
    expected: [null, null]
  },
  {
    name: 'source validity and NaN',
    controls: {scale: 'linear', domain: [0, 10], range: [1, 11], nullValue: -99},
    values: [0, 5, null, NaN, Infinity],
    validity: [1, 0, 0, 1, 1],
    expected: [1, null, null, null, null]
  },
  {
    name: 'prepared float32 neighbors',
    controls: {scale: 'linear', domain: [1, 1 + 2 ** -22], range: [0, 8]},
    values: [1 - 2 ** -24, 1, 1 + 2 ** -23, 1 + 2 ** -22],
    expected: [-2, 0, 4, 8]
  },
  {
    name: 'large finite span avoids overflowing subtraction',
    controls: {scale: 'linear', domain: [-Math.fround(3e38), Math.fround(3e38)], range: [-1, 1]},
    values: [-Math.fround(3e38), 0, Math.fround(3e38)],
    expected: [-1, 0, 1]
  }
];
