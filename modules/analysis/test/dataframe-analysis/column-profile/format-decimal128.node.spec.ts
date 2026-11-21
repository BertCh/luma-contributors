// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {formatDecimal128} from '../../../src/dataframe-analysis/column-profile/wide-numeric-gpu-decimal-reference';

// Expected text is coefficient × 10^(−scale), written out by hand.
test.each([
  [-12345n, 2, '-123.45'],
  [12345n, 2, '123.45'],
  [5n, 3, '0.005'],
  [-5n, 3, '-0.005'],
  [0n, 2, '0.00'],
  [123n, 0, '123'],
  [-123n, 0, '-123'],
  [12n, -3, '12000'],
  [123n, -2, '12300'],
  [-123n, -2, '-12300'],
  [0n, -2, '0']
])('DPC-106: formatDecimal128(%s, %s) is %s', (unscaled, scale, expected) => {
  expect(formatDecimal128(unscaled, scale)).toBe(expected);
});
