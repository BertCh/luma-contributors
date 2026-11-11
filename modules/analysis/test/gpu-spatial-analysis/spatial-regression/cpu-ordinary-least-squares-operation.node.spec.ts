// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeOrdinaryLeastSquaresOnCPU,
  CPUOrdinaryLeastSquaresOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-ordinary-least-squares-operation';
import {
  GPUOrdinaryLeastSquaresOperation,
  type OrdinaryLeastSquaresOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares-operation';
import {getGPUOrdinaryLeastSquaresParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-parameters';
import fixtures from './ordinary-least-squares-operation-fixtures.json';
import {
  constructRegressionOperationContributor,
  expectRegressionRowsClose
} from './regression-operation-test-utils';

const PROPS: OrdinaryLeastSquaresOperationProps = {
  id: 'ols-test',
  rowCount: 32,
  predictorCount: 2,
  predictors: 'predictors',
  response: 'response',
  mask: 'mask',
  parameters: 'parameters',
  output: {
    status: {code: 'code'},
    coefficients: 'coefficients',
    standardErrors: 'standardErrors',
    tStatistics: 'tStatistics',
    summary: 'summary',
    residuals: 'residuals',
    fitted: 'fitted'
  }
};

for (const fixture of fixtures) {
  it(`matches every output against independent SciPy QR/SVD: ${fixture.name}`, () => {
    const result = computeOrdinaryLeastSquaresOnCPU(
      {
        predictors: Float32Array.from(fixture.predictors),
        response: Float32Array.from(fixture.response),
        mask: Uint32Array.from(fixture.mask),
        parameters: getGPUOrdinaryLeastSquaresParameterValues({ridgeLambda: fixture.ridgeLambda})
      },
      2
    );
    expect(Array.from(result.status)).toEqual([0]);
    for (const name of [
      'coefficients',
      'standardErrors',
      'tStatistics',
      'summary',
      'residuals',
      'fitted'
    ] as const) {
      const expected = fixture.expected[name].map(value => (value === null ? NaN : value));
      expectRegressionRowsClose(result[name], expected, 3e-6);
    }
    // Moderate p-values ensure the auxiliary fit is actually exercised.
    expect(result.summary[10]).toBeGreaterThan(1e-4);
    expect(result.summary[10]).toBeLessThan(0.99);
  });
}

it('matches hand-computed constant response and failure layouts', () => {
  const result = computeOrdinaryLeastSquaresOnCPU(
    {predictors: Float32Array.of(-2, -1, 0, 1, 2), response: new Float32Array(5).fill(7)},
    1
  );
  expect(Array.from(result.coefficients)).toEqual([7, 0]);
  expect(result.residuals.every(value => value === 0)).toBe(true);
  expect(result.standardErrors.every(value => value === 0)).toBe(true);
  expect(result.tStatistics.every(Number.isNaN)).toBe(true);
  expect(result.summary[1]).toBeNaN();
  expect(result.summary[7]).toBeNaN();
  for (const [input, status, usedCount] of [
    [{predictors: new Float32Array(5), response: new Float32Array(5)}, 1, 5],
    [{predictors: Float32Array.of(1, 2), response: Float32Array.of(3, 4)}, 2, 2],
    [
      {
        predictors: Float32Array.of(NaN, 2, 3),
        response: Float32Array.of(3, Infinity, 4),
        mask: Uint32Array.of(1, 1, 0)
      },
      2,
      0
    ]
  ] as const) {
    const failed = computeOrdinaryLeastSquaresOnCPU(input, 1);
    expect(failed.status[0]).toBe(status);
    expect(failed.summary[0]).toBe(usedCount);
    expect(failed.summary.subarray(1).every(Number.isNaN)).toBe(true);
    for (const name of [
      'coefficients',
      'standardErrors',
      'tStatistics',
      'residuals',
      'fitted'
    ] as const)
      expect(failed[name].every(Number.isNaN)).toBe(true);
  }
});

it('rejects inconsistent arrays and validates both operations at construction', () => {
  const input = {predictors: new Float32Array(64), response: new Float32Array(32)};
  for (const change of [
    {predictors: new Float32Array(63)},
    {mask: new Uint32Array(31)},
    {parameters: new Float32Array()}
  ]) {
    expect(() => computeOrdinaryLeastSquaresOnCPU({...input, ...change}, 2)).toThrow(
      new RegExp(Object.keys(change)[0])
    );
  }
  const shapes = {
    predictors: {format: 'float32' as const, length: 64},
    response: {format: 'float32' as const, length: 32},
    mask: {format: 'uint32' as const, length: 32},
    parameters: {format: 'float32' as const, length: 1}
  };
  for (const change of [{predictorCount: 0}, {tileRowCount: 0}]) {
    expect(() => new CPUOrdinaryLeastSquaresOperation({...PROPS, ...change})).toThrow(/ols-test/);
    expect(() =>
      constructRegressionOperationContributor(
        new GPUOrdinaryLeastSquaresOperation({...PROPS, ...change}),
        shapes
      )
    ).toThrow(/ols-test/);
  }
  for (const Operation of [CPUOrdinaryLeastSquaresOperation, GPUOrdinaryLeastSquaresOperation]) {
    expect(() => new Operation({...PROPS, rowCount: 0})).toThrow(/ols-test/);
    expect(() => new Operation({...PROPS, output: {...PROPS.output, summary: 'response'}})).toThrow(
      /ols-test/
    );
  }
  expect(new CPUOrdinaryLeastSquaresOperation(PROPS).contract.outputs).toEqual(
    new GPUOrdinaryLeastSquaresOperation(PROPS).contract.outputs
  );
  expect(() =>
    constructRegressionOperationContributor(new GPUOrdinaryLeastSquaresOperation(PROPS), shapes)
  ).not.toThrow();
});
