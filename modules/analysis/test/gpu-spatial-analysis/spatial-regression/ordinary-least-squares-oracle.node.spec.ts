// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {fitOrdinaryLeastSquaresOnCPU} from './ordinary-least-squares-oracle';
import {
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LOG_LIKELIHOOD,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_AIC,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BIC
} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-parameters';

// x = [-1,-1,1,1], y = [s,-s,s,-s] => intercept = slope = 0, RSS/n = s^2.
// /usr/bin/python3: -2*(log(2*pi)+1+2*exponent*log(2)). These literals are independent
// of both this test oracle's fit and the production CPU twin.
for (const [exponent, logLikelihood] of [
  [-70, 188.40545642396597],
  [-100, 271.58311809115946],
  [-141, 385.25925570299046]
]) {
  it(`UST-09 oracle preserves positive RSS likelihood at response scale 2^${exponent}`, () => {
    const scale = 2 ** exponent;
    const result = fitOrdinaryLeastSquaresOnCPU({
      predictors: Float32Array.of(-1, -1, 1, 1),
      response: Float32Array.of(scale, -scale, scale, -scale),
      predictorCount: 1
    });
    expect(result.status).toBe(0);
    expect(Array.from(result.coefficients)).toEqual([0, 0]);
    expect(Array.from(result.residuals)).toEqual([scale, -scale, scale, -scale]);
    for (const [slot, expected] of [
      [GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LOG_LIKELIHOOD, logLikelihood],
      [GPU_ORDINARY_LEAST_SQUARES_SUMMARY_AIC, 4 - 2 * logLikelihood],
      [GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BIC, 2 * Math.log(4) - 2 * logLikelihood]
    ]) {
      expect(Number.isFinite(result.summary[slot])).toBe(true);
      expect(result.summary[slot]).toBeCloseTo(expected, 10);
    }
  });
}

it('UST-09 oracle keeps the legacy floor only for exact zero RSS', () => {
  const result = fitOrdinaryLeastSquaresOnCPU({
    predictors: Float32Array.of(-1, -1, 1, 1),
    response: new Float32Array(4),
    predictorCount: 1
  });
  expect(result.status).toBe(0);
  expect(Array.from(result.residuals)).toEqual([0, 0, 0, 0]);
  // /usr/bin/python3: -2*(log(2*pi)+1+log(1.17549435e-38)).
  expect(Number.isFinite(result.summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LOG_LIKELIHOOD])).toBe(
    true
  );
  expect(result.summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LOG_LIKELIHOOD]).toBeCloseTo(
    168.99733536968657,
    10
  );
});
