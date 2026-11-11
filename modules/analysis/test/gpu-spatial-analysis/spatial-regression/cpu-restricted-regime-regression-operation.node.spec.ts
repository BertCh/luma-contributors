// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeRestrictedRegimeRegressionOnCPU,
  CPURestrictedRegimeRegressionOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-restricted-regime-regression-operation';
import {
  GPURestrictedRegimeRegressionOperation,
  type RestrictedRegimeRegressionOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression-operation';
import fixtures from './restricted-regime-regression-operation-fixtures.json';
import {
  constructRegressionOperationContributor,
  expectRegressionRowsClose
} from './regression-operation-test-utils';

const PROPS: RestrictedRegimeRegressionOperationProps = {
  id: 'restricted-test',
  predictorCount: 2,
  regimeCount: 3,
  common: [false, false, true],
  rowCount: 60,
  predictors: 'predictors',
  response: 'response',
  regimes: 'regimes',
  mask: 'mask',
  output: {
    coefficients: 'coefficients',
    covariance: 'covariance',
    fitted: 'fitted',
    chow: 'chow',
    stability: 'stability',
    status: 'status'
  }
};

it('keeps a small regime response variation beside a 1e20 response origin', () => {
  const largeOrigin = Math.fround(1e20);
  const result = computeRestrictedRegimeRegressionOnCPU(
    {
      predictors: Float32Array.of(-1, 1, -1, 1, -1, 1, -1, 1),
      regimes: Uint32Array.of(0, 0, 0, 0, 1, 1, 1, 1),
      response: Float32Array.of(largeOrigin, largeOrigin, largeOrigin, largeOrigin, 0, 2, 0, 2)
    },
    {predictorCount: 1, regimeCount: 2, common: [false, true]}
  );
  // Each regime has zero-mean x. The pooled slope is (0 + 4)/(4 + 4)=1/2;
  // intercepts are the two response means, H and 1. RSS=2, variance=2/(8-3).
  expect(result.status[0]).toBe(0);
  expectRegressionRowsClose(result.coefficients, [largeOrigin, 1, 0.5], 1e-7);
  expectRegressionRowsClose(
    result.fitted,
    [largeOrigin, largeOrigin, largeOrigin, largeOrigin, 0.5, 1.5, 0.5, 1.5],
    1e-7
  );
  const expectedCovariance = [0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.05];
  result.covariance.forEach((value, index) =>
    expect(Math.abs(value - expectedCovariance[index])).toBeLessThan(1e-8)
  );
  expect(result.chow[0]).toBeNaN();
});

for (const fixture of fixtures) {
  it(`matches independent SciPy SVD fits and Wald tests: ${fixture.name}`, () => {
    const result = computeRestrictedRegimeRegressionOnCPU(
      {
        predictors: Float32Array.from(fixture.predictors),
        response: Float32Array.from(fixture.response),
        regimes: Uint32Array.from(fixture.regimes),
        mask: Uint32Array.from(fixture.mask)
      },
      fixture
    );
    expect(result.status[0]).toBe(0);
    for (const name of ['coefficients', 'fitted', 'chow', 'stability'] as const) {
      expectRegressionRowsClose(
        result[name],
        fixture.expected[name].map(value => (value === null ? NaN : value)),
        2e-5
      );
    }
    const count = result.coefficients.length;
    result.covariance.forEach((value, index) => {
      const scale = Math.sqrt(
        fixture.expected.covariance[Math.floor(index / count) * (count + 1)] *
          fixture.expected.covariance[(index % count) * (count + 1)]
      );
      expect(Math.abs(value - fixture.expected.covariance[index])).toBeLessThanOrEqual(
        2e-5 * scale
      );
    });
  });
}

it('preserves the restricted result with status 3 when an empty regime invalidates comparison', () => {
  const input = {
    predictors: Float32Array.of(0, 1, 2, 3, 4, 5, 6, 7),
    response: Float32Array.of(2, 4, 5, 8, 9, 11, 15, 16),
    regimes: new Uint32Array(8)
  };
  const options = {predictorCount: 1, regimeCount: 2, common: [true, true]};
  const result = computeRestrictedRegimeRegressionOnCPU(input, options);
  expect(result.status[0]).toBe(3);
  expect(result.coefficients.every(Number.isFinite)).toBe(true);
  expect(result.covariance.every(Number.isFinite)).toBe(true);
  expect(result.fitted.every(Number.isFinite)).toBe(true);
  expect(result.chow.every(Number.isNaN)).toBe(true);
  expect(Array.from(result.stability)).toEqual([NaN, 0, NaN, NaN, 0, NaN, NaN, 0, NaN]);
  expect(
    computeRestrictedRegimeRegressionOnCPU({...input, mask: new Uint32Array(8)}, options).status[0]
  ).toBe(1);
  expect(
    computeRestrictedRegimeRegressionOnCPU({...input, predictors: new Float32Array(8)}, options)
      .status[0]
  ).toBe(2);
});

it('checks inconsistent arrays, construction limits and equal output contracts', () => {
  const fixture = fixtures[0];
  const input = {
    predictors: Float32Array.from(fixture.predictors),
    response: Float32Array.from(fixture.response),
    regimes: Uint32Array.from(fixture.regimes)
  };
  for (const change of [
    {predictors: new Float32Array(119)},
    {regimes: new Uint32Array(59)},
    {mask: new Uint32Array(59)}
  ]) {
    expect(() => computeRestrictedRegimeRegressionOnCPU({...input, ...change}, fixture)).toThrow(
      new RegExp(Object.keys(change)[0])
    );
  }
  const shapes = {
    predictors: {format: 'float32' as const, length: 120},
    response: {format: 'float32' as const, length: 60},
    regimes: {format: 'uint32' as const, length: 60},
    mask: {format: 'uint32' as const, length: 60}
  };
  for (const change of [{predictorCount: 0}, {regimeCount: 1}, {common: [true]}]) {
    expect(() => new CPURestrictedRegimeRegressionOperation({...PROPS, ...change})).toThrow(
      /restricted-test/
    );
    expect(() =>
      constructRegressionOperationContributor(
        new GPURestrictedRegimeRegressionOperation({...PROPS, ...change}),
        shapes
      )
    ).toThrow(/restricted-test/);
  }
  for (const Operation of [
    CPURestrictedRegimeRegressionOperation,
    GPURestrictedRegimeRegressionOperation
  ]) {
    expect(() => new Operation({...PROPS, rowCount: 0})).toThrow(/restricted-test/);
    expect(() => new Operation({...PROPS, output: {...PROPS.output, chow: 'response'}})).toThrow(
      /restricted-test/
    );
  }
  expect(new CPURestrictedRegimeRegressionOperation(PROPS).contract.outputs).toEqual(
    new GPURestrictedRegimeRegressionOperation(PROPS).contract.outputs
  );
  expect(() =>
    constructRegressionOperationContributor(
      new GPURestrictedRegimeRegressionOperation(PROPS),
      shapes
    )
  ).not.toThrow();
});
