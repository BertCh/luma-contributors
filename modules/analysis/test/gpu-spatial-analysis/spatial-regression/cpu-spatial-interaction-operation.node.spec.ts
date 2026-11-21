// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SpatialInteractionCycleDetector} from '../../../src/gpu-spatial-analysis/spatial-interaction/spatial-interaction-cycle-detector';

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
  // Omit iteration budgets to test the converged float64 optimum; explicit budgets
  // bind the CPU solve and exhaustion is covered by spatial-interaction-audit.node.spec.ts.
  const bounded = computeSpatialInteractionOnCPU(INPUT, {
    ...PROPS,
    coefficients: [-1, 1],
    calibration: 'bounded',
    coefficientTolerance: 1e-9
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

it('separates structurally absorbed coefficients from prediction success, independently of grid order', () => {
  for (const coefficients of [
    [0, 1],
    [1, 0],
    [-3, -1]
  ]) {
    const result = computeSpatialInteractionOnCPU(
      {
        origins: Uint32Array.of(0, 0, 1, 1),
        destinations: Uint32Array.of(0, 1, 0, 1),
        flows: Float32Array.of(1, 3, 2, 6),
        covariate: Float32Array.of(1, 1, 7, 7)
      },
      {...PROPS, constraint: 'origin', coefficients}
    );
    expect(result.status[0]).toBe(0);
    expect(result.coefficientStatus[0]).toBe(1);
    expectRegressionRowsClose(result.predictions, [2, 2, 4, 4], 1e-7);
    // 2 * (log(1/2) + 3*log(3/2) + 2*log(2/4) + 6*log(6/4)).
    expectRegressionRowsClose([result.summary[1]], [3.13948894], 1e-7);
  }
});

it('balances boundary support to default tolerance with bounded retained history', () => {
  const input = {
    origins: Uint32Array.of(0, 0, 1),
    destinations: Uint32Array.of(0, 1, 1),
    flows: Float32Array.of(1, 0, 1),
    covariate: Float32Array.of(1, 1, 1)
  };
  for (const transpose of [false, true])
    for (const maximumIterations of [10, 37, 100, 1000, undefined]) {
      const fixture = transpose
        ? {...input, origins: input.destinations, destinations: input.origins}
        : input;
      const result = computeSpatialInteractionOnCPU(fixture, {
        originCount: 2,
        destinationCount: 2,
        constraint: 'doubly',
        coefficients: [0],
        maximumIterations
      });
      const iterations = maximumIterations ?? 50000;
      // Exact IPF recurrence: after n sweeps means are [1,1/(2n+1),2n/(2n+1)].
      if (maximumIterations === undefined)
        expectRegressionRowsClose(
          result.predictions,
          transpose
            ? [(2 * iterations) / (2 * iterations + 1), 1 / (2 * iterations + 1), 1]
            : [1, 1 / (2 * iterations + 1), (2 * iterations) / (2 * iterations + 1)],
          1e-7
        );
      expectRegressionRowsClose([result.candidates[1]], [1 / (2 * iterations + 1)], 1e-7);
      expect(result.candidates[2]).toBe(iterations);
      expect(result.status[0]).toBe(maximumIterations === undefined ? 0 : 3);
    }
});

it('review B1: certifies additive margins on connected and disconnected observed support', () => {
  for (const coefficients of [
    [-1, 0, 1],
    [1, 0, -1]
  ])
    for (const disconnected of [false, true]) {
      const result = computeSpatialInteractionOnCPU(
        {
          origins: Uint32Array.from(disconnected ? [0, 0, 1, 1, 2, 2, 3, 3] : [0, 0, 1, 1]),
          destinations: Uint32Array.from(disconnected ? [0, 1, 0, 1, 2, 3, 2, 3] : [0, 1, 0, 1]),
          flows: Float32Array.from(disconnected ? [2, 3, 5, 7, 4, 6, 8, 10] : [2, 3, 5, 7]),
          covariate: Float32Array.from(disconnected ? [0, 1, 1, 2, 5, 7, 8, 10] : [0, 1, 1, 2])
        },
        {
          originCount: disconnected ? 4 : 2,
          destinationCount: disconnected ? 4 : 2,
          constraint: 'doubly',
          coefficients
        }
      );
      expect(result.status[0]).toBe(0);
      expect(result.coefficientStatus[0]).toBe(1);
      // Independence means in each connected 2x2 block are rowTotal*columnTotal/blockTotal.
      expectRegressionRowsClose(
        result.predictions,
        disconnected
          ? [35 / 17, 50 / 17, 84 / 17, 120 / 17, 120 / 28, 160 / 28, 216 / 28, 288 / 28]
          : [35 / 17, 50 / 17, 84 / 17, 120 / 17],
        1e-5
      );
    }
});

it('review B2: compares sampled deviances at their published float32 precision', () => {
  const result = computeSpatialInteractionOnCPU(
    {
      origins: Uint32Array.of(0, 0, 1, 1),
      destinations: Uint32Array.of(0, 1, 0, 1),
      flows: Float32Array.of(2, 3, 5, 7),
      covariate: Float32Array.of(0, 1, 1, 0)
    },
    {originCount: 2, destinationCount: 2, constraint: 'origin', coefficients: [1, 1 + 1e-8]}
  );
  expect(result.candidates[0]).toBe(result.candidates[4]);
  expect(result.coefficientStatus[0]).toBe(2);
});

it('review S2: counts reachable retained vector words and detects array/map history mutations', () => {
  const retainedWords = (root: unknown): number => {
    const visited = new Set<object>();
    const walk = (value: unknown): number => {
      if (typeof value === 'number') return 1;
      if (typeof value === 'string')
        return new TextEncoder().encode(value).byteLength / Float64Array.BYTES_PER_ELEMENT;
      if (!value || typeof value !== 'object' || visited.has(value)) return 0;
      visited.add(value);
      if (ArrayBuffer.isView(value)) return value.byteLength / Float64Array.BYTES_PER_ELEMENT;
      if (value instanceof Map)
        return [...value.entries()].reduce(
          (total, [key, child]) => total + walk(key) + walk(child),
          0
        );
      if (value instanceof Set)
        return [...value.values()].reduce((total, child) => total + walk(child), 0);
      if (Array.isArray(value) && value.every(element => typeof element === 'number'))
        return value.length;
      return Object.values(value).reduce<number>((total, child) => total + walk(child), 0);
    };
    return walk(root);
  };
  const detector = new SpatialInteractionCycleDetector([1, 0, 1]);
  expect(retainedWords(detector)).toBe(8);
  for (let iteration = 1; iteration <= 50000; iteration++) {
    detector.observe(
      [1, 1 / (2 * iteration + 1), (2 * iteration) / (2 * iteration + 1)],
      [0, 1, 2]
    );
    if ([10, 37, 100, 1000, 50000].includes(iteration)) expect(retainedWords(detector)).toBe(8);
  }
  // Six vector words plus two scalar counters stay fixed. Array, Map and serialized Set
  // mutation controls must all exceed this retained-payload bound.
  expect(
    retainedWords({
      detector,
      history: [
        [1, 0, 1],
        [1, 0.5, 0.5]
      ]
    })
  ).toBeGreaterThan(8);
  expect(
    retainedWords({detector, history: new Map([[0, Float64Array.of(1, 0, 1)]])})
  ).toBeGreaterThan(8);
  expect(retainedWords({detector, history: new Set(['1,0,1'])})).toBeGreaterThan(8);
});

it('review B1/S4: exact support certificates handle longer cycles, signed zero and subnormals', () => {
  const input = {
    origins: Uint32Array.of(0, 0, 1, 1, 2, 2),
    destinations: Uint32Array.of(0, 1, 1, 2, 2, 0),
    flows: Float32Array.of(1, 2, 3, 4, 5, 6),
    covariate: Float32Array.of(3, 5, 7, 9, 11, 7)
  };
  const options = {
    originCount: 3,
    destinationCount: 3,
    constraint: 'doubly' as const,
    coefficients: [0]
  };
  expect(computeSpatialInteractionOnCPU(input, options).coefficientStatus[0]).toBe(1);
  expect(
    computeSpatialInteractionOnCPU(
      {...input, covariate: Float32Array.of(3, 5, 7, 9, 11, 8)},
      options
    ).coefficientStatus[0]
  ).toBe(0);
  for (const [words, expected] of [
    [[0, 0x80000000], 1],
    [[0, 1], 2]
  ] as const) {
    const result = computeSpatialInteractionOnCPU(
      {
        origins: Uint32Array.of(0, 0),
        destinations: Uint32Array.of(0, 1),
        flows: Float32Array.of(2, 3),
        covariate: new Float32Array(Uint32Array.from(words).buffer)
      },
      {originCount: 1, destinationCount: 2, constraint: 'origin', coefficients: [-1, 1]}
    );
    expect(result.coefficientStatus[0]).toBe(expected);
  }
});
