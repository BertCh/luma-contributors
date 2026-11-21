// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeOrdinaryLeastSquaresOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-ordinary-least-squares-operation';
import {getChiSquareSurvival} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-statistics';
import {AUDIT_OLS_FIXTURES} from './audit-ols-fixtures';

for (const fixture of AUDIT_OLS_FIXTURES) {
  it(`OLS every output matches independent QR/SVD with ties and held-out rows: ${fixture.name}`, () => {
    const result = computeOrdinaryLeastSquaresOnCPU(
      {
        predictors: Float32Array.from(fixture.predictors),
        response: Float32Array.from(fixture.response),
        mask: Uint32Array.from(fixture.mask)
      },
      2
    );
    for (const name of Object.keys(fixture.expected) as (keyof typeof fixture.expected)[]) {
      expect(result[name].length).toBe(fixture.expected[name].length);
      fixture.expected[name].forEach((expected, index) => {
        const actual = result[name][index];
        if (Number.isNaN(expected)) expect(actual, `${name}[${index}]`).toBeNaN();
        else {
          expect(Number.isFinite(actual), `${name}[${index}]`).toBe(true);
          // One final float32 rounding, including values smaller than one.
          expect(Math.abs(actual - expected), `${name}[${index}]`).toBeLessThanOrEqual(
            1e-6 * Math.max(1, Math.abs(expected))
          );
        }
      });
    }
  });
}

it('chi-square survival has the exact positive-infinity limit', () => {
  // P(X > +infinity) = 0 for every chi-square distribution (also scipy.stats.chi2.sf).
  for (const freedom of [1, 2, 3, 15]) expect(getChiSquareSurvival(Infinity, freedom)).toBe(0);
});
