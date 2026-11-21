// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {expectIndicesClose} from './terrain-indices-test-assertions';

it('S4 rejects extra or missing actual rows in every hydrologic output', () => {
  const expected = {
    descentTangent: Float64Array.of(1),
    specificCatchmentArea: Float64Array.of(1),
    wetnessIndex: Float64Array.of(0),
    streamPowerIndex: Float64Array.of(1)
  };
  const actual = {
    descentTangent: Float32Array.of(1),
    specificCatchmentArea: Float32Array.of(1),
    wetnessIndex: Float32Array.of(0),
    streamPowerIndex: Float32Array.of(1)
  };
  expect(() => expectIndicesClose('matched', actual, expected)).not.toThrow();
  expect(() =>
    expectIndicesClose('nan', {...actual, descentTangent: Float32Array.of(NaN)}, expected)
  ).toThrow();
  for (const name of [
    'descentTangent',
    'specificCatchmentArea',
    'wetnessIndex',
    'streamPowerIndex'
  ] as const) {
    expect(() =>
      expectIndicesClose(
        'extra',
        {...actual, [name]: Float32Array.of(actual[name][0], 99)},
        expected
      )
    ).toThrow();
    expect(() =>
      expectIndicesClose('missing', {...actual, [name]: new Float32Array(0)}, expected)
    ).toThrow();
  }
});
