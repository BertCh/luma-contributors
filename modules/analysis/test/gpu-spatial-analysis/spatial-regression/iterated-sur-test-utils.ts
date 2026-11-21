// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import fixtures from './iterated-sur-fixtures.json';

export type IteratedSURFixture = (typeof fixtures)[number];
export type IteratedSURExpectation = IteratedSURFixture['rounds'][number]['expected'];

/** Eight balanced +/-1 predictors, y=2+x in each equation, and distinct 2^64
 * first-stage residual impulses: Sigma=diag(2^125), X'X=8 I, Cov(beta)=2^122 I.
 * These are exact powers of two, independently derived without either implementation.
 */
export const SUR_OVERFLOW_EXPECTED: IteratedSURExpectation = {
  coefficients: [2, 1, 2, 1],
  covariance: [2 ** 122, 0, 0, 0, 0, 2 ** 122, 0, 0, 0, 0, 2 ** 122, 0, 0, 0, 0, 2 ** 122],
  residualCovariance: [2 ** 125, 0, 0, 2 ** 125],
  residuals: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  status: [0],
  iterationCount: [1]
};

/** For y=0, balanced +/-1 designs and distinct unit residual impulses: Sigma=I/8,
 * covariance=(8 I / (1/8))^-1=I/64; both OLS and GLS beta are exactly zero.
 */
export const SUR_ZERO_CHANGE_EXPECTED: IteratedSURExpectation = {
  coefficients: [0, 0, 0, 0],
  covariance: [1 / 64, 0, 0, 0, 0, 1 / 64, 0, 0, 0, 0, 1 / 64, 0, 0, 0, 0, 1 / 64],
  residualCovariance: [1 / 8, 0, 0, 1 / 8],
  residuals: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  status: [0],
  iterationCount: [1]
};

/** Every expected value is a NumPy literal; NaN checks cannot silently pass a numeric comparison. */
export function expectIteratedSURResult(
  actual: Record<keyof IteratedSURExpectation, ArrayLike<number>>,
  expected: IteratedSURExpectation,
  tolerance: number
): void {
  for (const name of ['status', 'iterationCount'] as const) {
    expect(Array.from(actual[name]), name).toEqual(expected[name]);
  }
  for (const name of ['coefficients', 'covariance', 'residualCovariance', 'residuals'] as const) {
    expect(actual[name].length, name).toBe(expected[name].length);
    expected[name].forEach((value, index) => {
      if (value === null) {
        expect(actual[name][index], `${name}[${index}]`).toBeNaN();
      } else {
        expect(Number.isFinite(actual[name][index]), `${name}[${index}] finite`).toBe(true);
        // Float32 normal equations and correlated-error precision amplify rounding.
        // Covariance uses diagonal geometric means, including near-zero off-diagonals.
        const size = expected.coefficients.length;
        const scale =
          name === 'covariance'
            ? Math.sqrt(
                expected.covariance[Math.floor(index / size) * (size + 1)] *
                  expected.covariance[(index % size) * (size + 1)]
              )
            : Math.max(1, Math.abs(value));
        expect(Math.abs(actual[name][index] - value), `${name}[${index}]`).toBeLessThanOrEqual(
          tolerance * scale
        );
      }
    });
  }
}
