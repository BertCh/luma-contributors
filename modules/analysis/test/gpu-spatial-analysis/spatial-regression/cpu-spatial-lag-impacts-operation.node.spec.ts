// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, expectTypeOf, it} from 'vitest';
import {
  computeSpatialLagImpactsOnCPU,
  CPUSpatialLagImpactsOperation,
  type CPUSpatialLagImpactsResult
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-lag-impacts-operation';
import {
  GPUSpatialLagImpactsOperation,
  type SpatialLagImpactsOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-impacts-operation';
import {
  constructRegressionOperationContributor,
  expectRegressionRowsClose
} from './regression-operation-test-utils';

const OPTIONS = {
  coefficientStride: 2,
  rhoIndex: 3,
  predictorCount: 2,
  rowCount: 2,
  maximumPower: 8
};

it('TYP-16 exposes the named lag impacts result type', () => {
  expectTypeOf<
    ReturnType<typeof computeSpatialLagImpactsOnCPU>
  >().toEqualTypeOf<CPUSpatialLagImpactsResult>();
});
const INPUT = {
  coefficients: Float32Array.of(99, 99, 2, 99, -3, 99, 0.25),
  // W = [[0, 2], [2, 0]], norm 2; W/norm swaps the rows.
  powerTraces: Float32Array.of(0, 2, 0, 2, 0, 2, 0, 2, 999),
  powerSums: Float32Array.of(2, 2, 2, 2, 2, 2, 2, 2, 999),
  infinityNorm: Float32Array.of(2)
};
const PROPS: SpatialLagImpactsOperationProps = {
  ...OPTIONS,
  id: 'impacts-test',
  coefficients: 'coefficients',
  traces: {powerTraces: 'traces', powerSums: 'sums', infinityNorm: 'norm'},
  output: {impacts: 'impacts'}
};

it('rejects an empty impacts destination before constructing descriptors (FRM-01)', () => {
  expect(() => new CPUSpatialLagImpactsOperation({...PROPS, output: {impacts: ''}})).toThrow(
    /impacts-test.*non-empty/
  );
});

it('FRM-01 rejects empty required and optional input names in both forms', () => {
  for (const Operation of [CPUSpatialLagImpactsOperation, GPUSpatialLagImpactsOperation]) {
    expect(() => new Operation({...PROPS, coefficients: ''})).toThrow(/impacts-test.*non-empty/);
    expect(() => new Operation({...PROPS, traces: {...PROPS.traces, powerTraces: ''}})).toThrow(
      /impacts-test.*non-empty/
    );
  }
  for (const Operation of [CPUSpatialLagImpactsOperation, GPUSpatialLagImpactsOperation]) {
    expect(() => new Operation({...PROPS, status: ''})).toThrow(/impacts-test.*status.*non-empty/);
    expect(new Operation(PROPS).inputs).not.toHaveProperty('status');
    expect(new Operation({...PROPS, status: 'fitStatus'}).inputs['status']).toBe('fitStatus');
  }
});

it('matches finite geometric-series closed forms for a scaled two-node weights matrix', () => {
  // The swap has even trace 2 and odd trace 0. No Horner recurrence in this oracle.
  const directMultiplier = (1 - 0.5 ** 10) / (1 - 0.5 ** 2);
  const totalMultiplier = (1 - 0.5 ** 9) / (1 - 0.5);
  const expected = [2, -3].flatMap(beta => [
    beta * directMultiplier,
    beta * (totalMultiplier - directMultiplier),
    beta * totalMultiplier
  ]);
  expectRegressionRowsClose(computeSpatialLagImpactsOnCPU(INPUT, OPTIONS).impacts, expected, 1e-7);
});

it('handles zero rho, negative rho, strict bounds and failed fits', () => {
  const coefficients = INPUT.coefficients.slice();
  coefficients[6] = 0;
  expect(
    Array.from(computeSpatialLagImpactsOnCPU({...INPUT, coefficients}, OPTIONS).impacts)
  ).toEqual([2, 0, 2, -3, 0, -3]);
  coefficients[6] = -0.25;
  const totalMultiplier = (1 - (-0.5) ** 9) / 1.5;
  const result = computeSpatialLagImpactsOnCPU({...INPUT, coefficients}, OPTIONS);
  expectRegressionRowsClose(result.impacts.subarray(2, 3), [2 * totalMultiplier], 1e-7);
  coefficients[6] = 0.5;
  expect(
    computeSpatialLagImpactsOnCPU({...INPUT, coefficients}, OPTIONS).impacts.every(Number.isNaN)
  ).toBe(true);
  expect(
    computeSpatialLagImpactsOnCPU({...INPUT, status: Uint32Array.of(2)}, OPTIONS).impacts.every(
      Number.isNaN
    )
  ).toBe(true);
  // The CPU accepts more than 256 terms; at rho=0 the closed form is beta.
  coefficients[6] = 0;
  expect(
    computeSpatialLagImpactsOnCPU(
      {
        ...INPUT,
        coefficients,
        powerTraces: new Float32Array(257),
        powerSums: new Float32Array(257)
      },
      {...OPTIONS, maximumPower: 257}
    ).impacts[0]
  ).toBe(2);
});

it('rejects inconsistent arrays by argument name', () => {
  for (const name of ['coefficients', 'powerTraces', 'powerSums', 'infinityNorm'] as const) {
    expect(() =>
      computeSpatialLagImpactsOnCPU({...INPUT, [name]: new Float32Array()}, OPTIONS)
    ).toThrow(new RegExp(name));
  }
  expect(() =>
    computeSpatialLagImpactsOnCPU({...INPUT, status: new Uint32Array()}, OPTIONS)
  ).toThrow(/status/);
});

it('checks CPU construction, GPU contributor construction and equal graph contracts', () => {
  const shapes = {
    coefficients: {format: 'float32' as const, length: 7},
    powerTraces: {format: 'float32' as const, length: 8},
    powerSums: {format: 'float32' as const, length: 8},
    infinityNorm: {format: 'float32' as const, length: 1}
  };
  for (const changes of [
    {coefficientStride: 0},
    {rhoIndex: 1},
    {maximumPower: 0},
    {rowCount: 0},
    {firstPredictorIndex: -1},
    {rhoBound: 0}
  ]) {
    expect(() => new CPUSpatialLagImpactsOperation({...PROPS, ...changes})).toThrow(/impacts-test/);
    expect(() =>
      constructRegressionOperationContributor(
        new GPUSpatialLagImpactsOperation({...PROPS, ...changes}),
        shapes
      )
    ).toThrow(/impacts-test/);
  }
  for (const Operation of [CPUSpatialLagImpactsOperation, GPUSpatialLagImpactsOperation]) {
    expect(() => new Operation({...PROPS, predictorCount: 0})).toThrow(/impacts-test/);
    expect(() => new Operation({...PROPS, output: {impacts: 'coefficients'}})).toThrow(
      /impacts-test/
    );
  }
  expect(new CPUSpatialLagImpactsOperation(PROPS).contract.outputs).toEqual(
    new GPUSpatialLagImpactsOperation(PROPS).contract.outputs
  );
  expect(() =>
    constructRegressionOperationContributor(new GPUSpatialLagImpactsOperation(PROPS), shapes)
  ).not.toThrow();
});
