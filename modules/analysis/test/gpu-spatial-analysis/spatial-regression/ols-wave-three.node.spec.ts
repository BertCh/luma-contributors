// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeOrdinaryLeastSquaresOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-ordinary-least-squares-operation';
import {getGPUOrdinaryLeastSquaresParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-parameters';
import {getGPUOrdinaryLeastSquaresModelResult} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-statistics';

function fit(scale = 1, ridgeLambda = 0) {
  return computeOrdinaryLeastSquaresOnCPU(
    {
      predictors: new Float32Array([0, 1, 2, 3]),
      response: Float32Array.from([1, 3, 4, 8], value => value * scale),
      parameters: getGPUOrdinaryLeastSquaresParameterValues({ridgeLambda})
    },
    1
  );
}

it('UST-09 retains positive small residual likelihoods and separates exact zero', () => {
  // Hand regression: intercept=.7, slope=2.2, RSS=1.8. Gaussian profile log L
  // = -n/2 * (log(2*pi) + log(RSS/n) + 1); scaling y by 1/10 adds n*log(10).
  const first = fit(1e-20);
  const second = fit(1e-21);
  expect(Number.isFinite(first.summary[4])).toBe(true);
  expect(first.summary[4]).toBeCloseTo(180.12807207656713, 4);
  expect(second.summary[4] - first.summary[4]).toBeCloseTo(4 * Math.log(10), 4);
  expect(second.summary[5] - first.summary[5]).toBeCloseTo(-8 * Math.log(10), 4);
  expect(second.summary[6] - first.summary[6]).toBeCloseTo(-8 * Math.log(10), 4);
  // Keep the established finite convention for exactly zero RSS, separately from positive RSS.
  expect(fit(0).summary[4]).toBeCloseTo(168.99733320922616, 4);
});

it('CMP-01 reproduces loss of training rows when a joined mask hides a truncated key', () => {
  const input = {
    predictors: new Float32Array([0, 1, 2, 3, 4]),
    response: new Float32Array([1, 3, 5, 7, 50])
  };
  expect(computeOrdinaryLeastSquaresOnCPU(input, 1).coefficients[1]).toBeCloseTo(10.2, 5);
  expect(
    computeOrdinaryLeastSquaresOnCPU({...input, mask: new Uint32Array([1, 1, 1, 1, 0])}, 1)
      .coefficients[1]
  ).toBe(2);
});

it('CMP-11 reproduces column concatenation versus row-major coefficient order', () => {
  const response = new Float32Array([1, 3, 4, 6, 8]);
  const correct = computeOrdinaryLeastSquaresOnCPU(
    {response, predictors: new Float32Array([0, 0, 1, 0, 0, 1, 1, 1, 2, 1])},
    2
  );
  expect(Array.from(correct.coefficients)).toEqual([1, 2, 3]);
  const wrong = computeOrdinaryLeastSquaresOnCPU(
    {response, predictors: new Float32Array([0, 1, 0, 1, 2, 0, 0, 1, 1, 1])},
    2
  );
  expect(wrong.coefficients[0]).toBeCloseTo(-16 / 3, 5);
});

it('UST-08 validates float32 ridge conversion before mutating a target', () => {
  expect(getGPUOrdinaryLeastSquaresParameterValues({ridgeLambda: 0})[0]).toBe(0);
  expect(getGPUOrdinaryLeastSquaresParameterValues({ridgeLambda: 1e-30})[0]).toBe(
    Math.fround(1e-30)
  );
  for (const ridgeLambda of [2 ** -126, 3.4028234663852886e38]) {
    expect(getGPUOrdinaryLeastSquaresParameterValues({ridgeLambda})[0]).toBe(
      Math.fround(ridgeLambda)
    );
  }
  for (const ridgeLambda of [1e40, 1e-60, 1e-45, Number.MIN_VALUE]) {
    const target = new Float32Array([7, 9]);
    expect(() => getGPUOrdinaryLeastSquaresParameterValues({ridgeLambda}, target)).toThrow(
      /ridgeLambda/
    );
    expect(Array.from(target)).toEqual([7, 9]);
  }
});

it('UST-06 labels applied ridge inference while retaining ordinary controls', () => {
  const penalized = getGPUOrdinaryLeastSquaresModelResult(fit(1, 5));
  expect(penalized.regularization).toEqual({kind: 'ridge', lambda: 5});
  expect(penalized.inferenceMethod).toBe('ridge-approximate');
  expect(penalized.coefficients[1].estimate).toBeCloseTo(1.1, 6);
  const ordinary = getGPUOrdinaryLeastSquaresModelResult(fit());
  expect(ordinary.regularization).toBeUndefined();
  expect(ordinary.inferenceMethod).toBe('ordinary-student-t');
});

import {getStudentTTwoSidedPValue} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-statistics';
import {
  getStudentTTwoSidedTail,
  getStudentTCriticalValue
} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-student-t';

it('W3-04 preserves integer OLS and effective GWR Student t domains', () => {
  // /usr/bin/python3: scipy.stats.t.sf(2.3, df)*2; literals independently evaluated with SciPy.
  for (const [degreesOfFreedom, expected] of [
    [1, 0.2610951741772454],
    [2, 0.14814814814814814],
    [200000, 0.021449244568912645]
  ]) {
    expect(Number.isFinite(getStudentTTwoSidedPValue(2.3, degreesOfFreedom))).toBe(true);
    expect(getStudentTTwoSidedPValue(2.3, degreesOfFreedom)).toBeCloseTo(expected, 11);
    expect(getStudentTTwoSidedTail(2.3, degreesOfFreedom)).toBeCloseTo(expected, 11);
  }
  for (const [degreesOfFreedom, expected] of [
    [2.5, 0.12247966442415527],
    [7.25, 0.05373828598794974]
  ]) {
    expect(getStudentTTwoSidedPValue(2.3, degreesOfFreedom)).toBeNaN();
    expect(getStudentTTwoSidedTail(2.3, degreesOfFreedom)).toBeCloseTo(expected, 11);
    expect(getStudentTCriticalValue(expected, degreesOfFreedom)).toBeCloseTo(2.3, 9);
  }
  expect(getStudentTTwoSidedTail(-2.3, 4)).toBe(1);
  expect(getStudentTTwoSidedPValue(-2.3, 2)).toBeCloseTo(0.14814814814814814, 12);
  // Cauchy tail = 2/pi * atan(1/t); this remains representable although t² overflows.
  for (const actual of [getStudentTTwoSidedTail(1e200, 1), getStudentTTwoSidedPValue(1e200, 1)]) {
    expect(Number.isFinite(actual)).toBe(true);
    expect(Math.abs(actual / 6.366197723675814e-201 - 1)).toBeLessThan(1e-12);
  }
  expect(getStudentTTwoSidedTail(Infinity, 2)).toBe(0);
  expect(getStudentTTwoSidedTail(NaN, 2)).toBeNaN();
  expect(getStudentTTwoSidedTail(2.3, 1e9)).toBe(getStudentTTwoSidedTail(2.3, 200000));
});

it('DEF-11 CPU view-based twin infers the same exact design width', () => {
  const input = {
    predictors: new Float32Array([0, 0, 1, 0, 0, 1, 1, 1, 2, 1]),
    response: new Float32Array([1, 3, 4, 6, 8])
  };
  expect(Array.from(computeOrdinaryLeastSquaresOnCPU(input).coefficients)).toEqual([1, 2, 3]);
  expect(() =>
    computeOrdinaryLeastSquaresOnCPU({...input, predictors: input.predictors.subarray(1)})
  ).toThrow(/predictorCount/);
  expect(() =>
    computeOrdinaryLeastSquaresOnCPU({
      predictors: new Float32Array(0),
      response: new Float32Array(0)
    })
  ).toThrow(/predictorCount/);
  expect(
    computeOrdinaryLeastSquaresOnCPU(
      {predictors: new Float32Array(0), response: new Float32Array(0)},
      1
    ).status[0]
  ).toBe(2);
});

it('earlier ridge fix rejects Metal-unsafe subnormal penalties before target mutation', () => {
  const target = new Float32Array([7]);
  expect(() => getGPUOrdinaryLeastSquaresParameterValues({ridgeLambda: 2 ** -127}, target)).toThrow(
    /ridgeLambda.*normal float32/
  );
  expect(target[0]).toBe(7);
  expect(getGPUOrdinaryLeastSquaresParameterValues({ridgeLambda: 2 ** -126})[0]).toBe(2 ** -126);
});

it('B5 normal float32 responses may have nonzero subnormal residuals', () => {
  const center = 2 ** -120;
  const deviation = 2 ** -141;
  const result = computeOrdinaryLeastSquaresOnCPU(
    {
      predictors: new Float32Array([-1, -1, 1, 1]),
      response: new Float32Array([
        center + deviation,
        center - deviation,
        center + deviation,
        center - deviation
      ])
    },
    1
  );
  // Mean=center, slope=0, RSS/n=2^-282. -2*(log(2*pi)+1-282*log(2)).
  expect(Array.from(result.residuals)).toEqual([deviation, -deviation, deviation, -deviation]);
  expect(Number.isFinite(result.summary[4])).toBe(true);
  expect(result.summary[4]).toBeCloseTo(385.25925570299046, 4);
});
