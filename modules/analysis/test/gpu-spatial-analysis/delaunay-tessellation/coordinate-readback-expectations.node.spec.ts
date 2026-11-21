// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {expectFiniteCoordinatePrefix} from './coordinate-readback-expectations';

const EXPECTED_COORDINATES = [1, 1, 3, 3];
it('S3 accepts the complete two-row coordinate readback', () => {
  expect(() =>
    expectFiniteCoordinatePrefix(EXPECTED_COORDINATES, 2, 2, EXPECTED_COORDINATES, 1e-6)
  ).not.toThrow();
});
for (const length of [0, 2, 3]) {
  it(`S3 rejects a matching but truncated coordinate readback of ${length} words`, () => {
    expect(() =>
      expectFiniteCoordinatePrefix(
        EXPECTED_COORDINATES.slice(0, length),
        2,
        2,
        EXPECTED_COORDINATES,
        1e-6
      )
    ).toThrow();
  });
}
it('S3 rejects a changed published count and nonfinite coordinates', () => {
  expect(() =>
    expectFiniteCoordinatePrefix(EXPECTED_COORDINATES, 1, 2, EXPECTED_COORDINATES, 1e-6)
  ).toThrow();
  expect(() =>
    expectFiniteCoordinatePrefix([1, 1, NaN, 3], 2, 2, EXPECTED_COORDINATES, 1e-6)
  ).toThrow();
});
