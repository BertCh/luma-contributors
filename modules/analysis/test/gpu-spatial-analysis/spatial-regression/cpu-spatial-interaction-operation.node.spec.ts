// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeSpatialInteractionOnCPU,
  CPUSpatialInteractionOperation
} from '../../../src/gpu-spatial-analysis/spatial-interaction/cpu-spatial-interaction-operation';
import {
  GPUSpatialInteractionOperation,
  type SpatialInteractionOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-interaction/gpu-spatial-interaction-operation';
import {
  constructRegressionOperationContributor,
  expectRegressionRowsClose
} from './regression-operation-test-utils';

const INPUT = {
  origins: Uint32Array.of(0, 0, 1, 1),
  destinations: Uint32Array.of(0, 1, 0, 1),
  flows: Float32Array.of(10, 20, 30, 40),
  covariate: Float32Array.of(0, 1, 2, 0)
};
const COEFFICIENT = -Math.log(2);
const PROPS: SpatialInteractionOperationProps = {
  id: 'gravity-test',
  pairCount: 4,
  originCount: 2,
  destinationCount: 2,
  origins: 'origins',
  destinations: 'destinations',
  flows: 'flows',
  covariate: 'covariate',
  constraint: 'doubly',
  coefficients: [COEFFICIENT],
  tolerance: 1e-12,
  output: {
    predictions: 'predictions',
    summary: 'summary',
    candidates: 'candidates',
    status: 'status',
    originResiduals: 'originResiduals',
    destinationResiduals: 'destinationResiduals'
  }
};

for (const constraint of ['unconstrained', 'origin', 'destination', 'doubly'] as const) {
  it(`matches hand normalization or the 2x2 odds-ratio quadratic: ${constraint}`, () => {
    const result = computeSpatialInteractionOnCPU(INPUT, {...PROPS, constraint});
    const kernel = [1, 0.5, 0.25, 1];
    let expected: number[];
    if (constraint === 'unconstrained') expected = kernel.map(value => (value * 100) / 2.75);
    else if (constraint === 'origin') expected = [20, 10, 14, 56];
    else if (constraint === 'destination') expected = [32, 20, 8, 40];
    else {
      // With these margins all four cells are [x,30-x,40-x,30+x]. Their odds ratio
      // is exactly 8, so 7*x^2 - 590*x + 9600 = 0. The smaller root is feasible.
      const first = (590 - Math.sqrt(590 ** 2 - 4 * 7 * 9600)) / 14;
      expected = [first, 30 - first, 40 - first, 30 + first];
    }
    expect(result.status[0]).toBe(0);
    expectRegressionRowsClose(result.predictions, expected, 1e-7);
    const deviance =
      2 *
      expected.reduce(
        (sum, prediction, pair) =>
          sum +
          INPUT.flows[pair] * Math.log(INPUT.flows[pair] / prediction) -
          INPUT.flows[pair] +
          prediction,
        0
      );
    expectRegressionRowsClose(
      [result.summary[0], result.summary[1]],
      [COEFFICIENT, deviance],
      1e-7
    );
    expectRegressionRowsClose(
      result.candidates.subarray(0, 2),
      [result.summary[1], result.summary[2]],
      0
    );
    expect(result.candidates[2]).toBe(result.summary[3]);
    expect(result.candidates[3]).toBe(0);
    for (let group = 0; group < 2; group++) {
      expect(
        Math.abs(
          result.originResiduals[group] -
            Math.fround(
              INPUT.flows[group * 2] +
                INPUT.flows[group * 2 + 1] -
                expected[group * 2] -
                expected[group * 2 + 1]
            )
        )
      ).toBeLessThan(1e-9);
      expect(
        Math.abs(
          result.destinationResiduals[group] -
            Math.fround(
              INPUT.flows[group] + INPUT.flows[group + 2] - expected[group] - expected[group + 2]
            )
        )
      ).toBeLessThan(1e-9);
    }
    // For doubly constrained fitting, column-normalized cells are
    // [160*a/(4*a+1), 60*a/(a+2), 40/(4*a+1), 120/(a+2)]. After the first
    // sweep a=5/14; later sweeps apply a <- (78*a+30)/(140*a+133).
    // This scalar Mobius recurrence gives marginal error 2.795e-12 at sweep 16,
    // and 5.575e-13 at sweep 17: exactly 17 sweeps are required at tolerance 1e-12.
    expect(result.summary[3]).toBe(
      constraint === 'unconstrained' ? 0 : constraint === 'doubly' ? 17 : 1
    );
  });
}

it('selects a non-first grid candidate, resolves ties first, and profiles the bounded optimum', () => {
  const grid = computeSpatialInteractionOnCPU(INPUT, {
    ...PROPS,
    constraint: 'unconstrained',
    coefficients: [-2, 0, 2, 0]
  });
  expect(grid.summary[0]).toBe(0);
  expect(grid.summary[1]).toBe(grid.candidates[4]);
  expect(Array.from(grid.predictions)).toEqual([25, 25, 25, 25]);
  const tied = computeSpatialInteractionOnCPU(
    {...INPUT, covariate: new Float32Array(4)},
    {...PROPS, coefficients: [4, -7]}
  );
  expect(tied.summary[0]).toBe(4);
  // Nuisance-free 2x2 doubly constrained profile fits every observed count when its
  // odds ratio exp(-3*beta) equals 10*40/(20*30) = 2/3.
  const bounded = computeSpatialInteractionOnCPU(INPUT, {
    ...PROPS,
    coefficients: [-1, 1],
    calibration: 'bounded',
    coefficientTolerance: 1e-9,
    maximumCalibrationIterations: 1,
    maximumIterations: 1
  });
  expect(bounded.status[0]).toBe(0);
  expect(Math.abs(bounded.summary[0] - Math.log(1.5) / 3)).toBeLessThan(1e-7);
  expectRegressionRowsClose(bounded.predictions, INPUT.flows, 2e-7);
  expect(bounded.calibration![2]).toBeLessThanOrEqual(1e-9);
  expect(bounded.calibration![3]).toBe(1);
});

it('publishes a positive golden-section width consistent with rounded bracket endpoints', () => {
  const coefficientTolerance = 1e-3;
  const result = computeSpatialInteractionOnCPU(INPUT, {
    ...PROPS,
    coefficients: [-1, 1],
    calibration: 'bounded',
    coefficientTolerance,
    tolerance: 1e-5
  });
  const [lower, upper, width, converged] = result.calibration!;
  // Each successful refinement shrinks the width by phi. The first width <=1e-3
  // is 2*phi^16 = 0.0009062077075696455; phi^15 still gives width >1e-3.
  const ratio = (Math.sqrt(5) - 1) / 2;
  const expectedWidth = 2 * ratio ** 16;
  expect(2 * ratio ** 15).toBeGreaterThan(coefficientTolerance);
  expect(width).toBeGreaterThan(0);
  expect(width).toBeLessThanOrEqual(coefficientTolerance);
  expectRegressionRowsClose([width], [Math.fround(expectedWidth)], 1e-7);
  expect(upper).toBeGreaterThan(lower);
  const endpointRounding = 2 * 2 ** -23 * Math.max(Math.abs(lower), Math.abs(upper));
  expect(Math.abs(width - (upper - lower))).toBeLessThanOrEqual(endpointRounding);
  expect(converged).toBe(1);
});

it('rounds a finite float64 Poisson deviance beyond float32 range to infinity', () => {
  const flows = Float32Array.of(1e38, 0);
  const result = computeSpatialInteractionOnCPU(
    {
      origins: Uint32Array.of(0, 0),
      destinations: Uint32Array.of(0, 1),
      flows,
      covariate: Float32Array.of(0, 10)
    },
    {...PROPS, originCount: 1, constraint: 'unconstrained', coefficients: [1]}
  );
  // Normalization gives mu=[F/(1+exp(10)), F*exp(10)/(1+exp(10))]. The
  // linear terms cancel in total deviance, leaving 2*F*log(1+exp(10)).
  const denominator = 1 + Math.exp(10);
  const expectedDeviance = 2 * flows[0] * Math.log(denominator);
  expect(Number.isFinite(expectedDeviance)).toBe(true);
  expect(Math.fround(expectedDeviance)).toBe(Infinity);
  expect(result.status[0]).toBe(0);
  expectRegressionRowsClose(
    result.predictions,
    Float32Array.of(flows[0] / denominator, (flows[0] * Math.exp(10)) / denominator),
    1e-7
  );
  expect(Array.from(result.summary)).toEqual([1, Infinity, 0, 0]);
  expect(Array.from(result.candidates)).toEqual([Infinity, 0, 0, 0]);
});

it('preserves structural zeros, missing pairs, zero groups and each invalid/no-mass status', () => {
  const input = {
    origins: Uint32Array.of(0, 0, 1, 99),
    destinations: Uint32Array.of(0, 1, 1, 99),
    flows: Float32Array.of(4, 0, 0, NaN),
    covariate: Float32Array.of(0, NaN, 2, NaN),
    pairState: Uint32Array.of(1, 0, 1, 2)
  };
  const result = computeSpatialInteractionOnCPU(input, PROPS);
  expect(result.status[0]).toBe(0);
  expect(Array.from(result.predictions)).toEqual([4, 0, 0, NaN]);
  const noMass = computeSpatialInteractionOnCPU({...INPUT, flows: new Float32Array(4)}, PROPS);
  expect(noMass.status[0]).toBe(1);
  expect(Array.from(noMass.candidates)).toEqual([NaN, NaN, 0, 2]);
  for (const changes of [
    {flows: Float32Array.of(-1, 2, 3, 4)},
    {pairState: Uint32Array.of(3, 1, 1, 1)},
    {origins: Uint32Array.of(9, 0, 1, 1)},
    {covariate: Float32Array.of(NaN, 1, 2, 0)}
  ]) {
    const invalid = computeSpatialInteractionOnCPU({...INPUT, ...changes}, PROPS);
    expect(invalid.status[0]).toBe(2);
    expect(Array.from(invalid.candidates)).toEqual([NaN, NaN, 0, 3]);
    for (const port of [
      'predictions',
      'summary',
      'originResiduals',
      'destinationResiduals'
    ] as const)
      expect(invalid[port].every(Number.isNaN)).toBe(true);
  }
});

it('fits an exponent span of 1000 without rejection or intermediate underflow', () => {
  const result = computeSpatialInteractionOnCPU(
    {...INPUT, covariate: Float32Array.of(0, -1000, -1000, 0)},
    {...PROPS, constraint: 'origin', coefficients: [1]}
  );
  expect(result.status[0]).toBe(0);
  expect(Array.from(result.predictions)).toEqual([30, 0, 0, 70]);
  expect(result.summary[1]).toBeGreaterThan(90000);
  expect(Number.isFinite(result.summary[1])).toBe(true);
  expect(result.candidates[3]).toBe(0);
  const shifted = computeSpatialInteractionOnCPU(
    {...INPUT, covariate: new Float32Array(4).fill(1e30)},
    {...PROPS, constraint: 'unconstrained', coefficients: [1e200]}
  );
  expect(shifted.status[0]).toBe(0);
  expect(Array.from(shifted.predictions)).toEqual([25, 25, 25, 25]);
});

it('checks arrays and construction-time validation of both classes', () => {
  for (const changes of [
    {origins: new Uint32Array(3)},
    {destinations: new Uint32Array(3)},
    {covariate: new Float32Array(3)},
    {pairState: new Uint32Array(3)}
  ])
    expect(() => computeSpatialInteractionOnCPU({...INPUT, ...changes}, PROPS)).toThrow(
      new RegExp(Object.keys(changes)[0])
    );
  const shapes = {
    origins: {format: 'uint32' as const, length: 4},
    destinations: {format: 'uint32' as const, length: 4},
    flows: {format: 'float32' as const, length: 4},
    covariate: {format: 'float32' as const, length: 4}
  };
  for (const changes of [
    {originCount: 0},
    {destinationCount: 0},
    {maximumIterations: 0},
    {maximumCalibrationIterations: 0},
    {tolerance: 0},
    {coefficientTolerance: NaN},
    {constraint: 'invalid' as never}
  ]) {
    expect(() => new CPUSpatialInteractionOperation({...PROPS, ...changes})).toThrow(
      /gravity-test/
    );
    expect(() =>
      constructRegressionOperationContributor(
        new GPUSpatialInteractionOperation({...PROPS, ...changes}),
        shapes
      )
    ).toThrow(/gravity-test/);
  }
  for (const Operation of [GPUSpatialInteractionOperation, CPUSpatialInteractionOperation])
    expect(
      () => new Operation({...PROPS, output: {...PROPS.output, predictions: 'flows'}})
    ).toThrow(/gravity-test/);
  expect(new CPUSpatialInteractionOperation(PROPS).contract.outputs).toEqual(
    new GPUSpatialInteractionOperation(PROPS).contract.outputs
  );
  expect(() =>
    constructRegressionOperationContributor(new GPUSpatialInteractionOperation(PROPS), shapes)
  ).not.toThrow();
});
