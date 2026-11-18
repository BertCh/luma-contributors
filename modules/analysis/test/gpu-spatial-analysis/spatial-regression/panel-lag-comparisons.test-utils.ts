// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Matches the panel oracle contract: covariance error relative to marginal standard errors. */
export function expectPanelLagCovarianceClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  width: number
): void {
  expect(actual.length).toBe(width * width);
  expect(expected.length).toBe(width * width);
  for (let row = 0; row < width; row++) {
    for (let column = 0; column < width; column++) {
      const index = row * width + column;
      if (Number.isNaN(expected[index])) {
        expect(actual[index], `covariance ${row},${column}`).toBeNaN();
      } else {
        const scale = Math.sqrt(expected[row * width + row] * expected[column * width + column]);
        expect(Number.isFinite(actual[index]), `covariance ${row},${column} is finite`).toBe(true);
        expect(
          Math.abs(actual[index] - expected[index]),
          `covariance ${row},${column}`
        ).toBeLessThanOrEqual(0.005 * scale);
      }
    }
  }
}

/** Matches the established panel prediction tolerance, including values close to zero. */
export function expectPanelLagPredictionsClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  response: ArrayLike<number>
): void {
  expect(actual.length).toBe(expected.length);
  expect(expected.length).toBe(response.length);
  let responseSquares = 0;
  for (let row = 0; row < response.length; row++) responseSquares += response[row] * response[row];
  const responseScale = Math.sqrt(responseSquares / response.length);
  for (let row = 0; row < expected.length; row++) {
    expect(Number.isFinite(actual[row]), `prediction ${row} is finite`).toBe(true);
    expect(Math.abs(actual[row] - expected[row]), `prediction ${row}`).toBeLessThanOrEqual(
      2e-4 * Math.max(responseScale, Math.abs(expected[row]))
    );
  }
}
