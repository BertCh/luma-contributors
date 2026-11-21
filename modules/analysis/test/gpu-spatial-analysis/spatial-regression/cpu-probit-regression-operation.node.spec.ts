// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeProbitRegressionOnCPU,
  CPUProbitRegressionOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-probit-regression-operation';
import {
  GPUProbitRegressionOperation,
  type ProbitRegressionOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression-operation';
import {
  constructRegressionOperationContributor,
  expectRegressionRowsClose
} from './regression-operation-test-utils';
import fixture from './probit-regression-operation-fixtures.json';
import {
  createProbitConvergenceFixture,
  PROBIT_CONVERGENCE_SLOPE,
  PROBIT_CONVERGENCE_LOG_LIKELIHOOD,
  PROBIT_CONVERGENCE_STANDARD_ERROR,
  createProbitLargeCoefficientFixture,
  PROBIT_LARGE_COEFFICIENT_SLOPE
} from './probit-convergence-fixture';

const PROPS: ProbitRegressionOperationProps = {
  id: 'probit-test',
  rowCount: 96,
  predictorCount: 2,
  predictors: 'predictors',
  response: 'response',
  mask: 'mask',
  spatialDiagnostics: {
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    output: {tests: 'tests', status: 'diagnosticStatus'}
  },
  output: {
    coefficients: 'coefficients',
    standardErrors: 'standardErrors',
    fitted: 'fitted',
    marginalEffects: 'marginalEffects',
    summary: 'summary',
    status: 'status'
  }
};

function getInput() {
  return {
    predictors: Float32Array.from(fixture.predictors),
    response: Float32Array.from(fixture.response),
    mask: Uint32Array.from(fixture.mask),
    spatialDiagnostics: {
      weights: {
        offsets: Uint32Array.from(fixture.offsets),
        neighbors: Uint32Array.from(fixture.neighbors),
        weights: Float32Array.from(fixture.weights)
      }
    }
  };
}

it('matches independent SciPy BFGS and dense NumPy information and spatial diagnostics', () => {
  const result = computeProbitRegressionOnCPU(getInput(), {predictorCount: 2, tolerance: 1e-9});
  expect(result.status[0]).toBe(0);
  for (const name of ['coefficients', 'standardErrors', 'fitted', 'marginalEffects'] as const) {
    expectRegressionRowsClose(
      result[name],
      fixture.expected[name].map(value => (value === null ? NaN : value)),
      3e-6
    );
  }
  expectRegressionRowsClose(
    result.summary.subarray(0, 3),
    [93, fixture.expected.logLikelihood, -2 * fixture.expected.logLikelihood],
    2e-7
  );
  expect(result.spatialDiagnostics!.status[0]).toBe(0);
  expectRegressionRowsClose(result.spatialDiagnostics!.tests, fixture.expected.tests, 3e-6);
  expect(result.spatialDiagnostics!.tests[1]).toBeGreaterThan(0.01);
  expect(result.spatialDiagnostics!.tests[1]).toBeLessThan(0.99);
});

it('publishes a budget-limited iterate and no standard errors or diagnostics', () => {
  const result = computeProbitRegressionOnCPU(getInput(), {
    predictorCount: 2,
    maximumIterations: 1
  });
  expect(result.status[0]).toBe(3);
  expect(result.summary[3]).toBe(1);
  expect(result.coefficients.every(Number.isFinite)).toBe(true);
  expect(result.fitted[0]).toBeGreaterThan(0);
  expect(result.standardErrors.every(Number.isNaN)).toBe(true);
  expect(result.spatialDiagnostics!.status[0]).toBe(1);
  expect(result.spatialDiagnostics!.tests.every(Number.isNaN)).toBe(true);
});

it('retains centered predictions and effects under a large exactly representable predictor offset', () => {
  // Translation and scale equivariance is an independent invariant; both outcomes at every x
  // exclude separation. The raw float32 intercept cannot reconstruct these probabilities.
  const predictors: number[] = [];
  const response: number[] = [];
  [-2, -1, 0, 1, 2].forEach((value, group) => {
    for (let row = 0; row < 1000; row++) {
      predictors.push(value);
      response.push(row < [136, 345, 618, 841, 955][group] ? 1 : 0);
    }
  });
  const first = computeProbitRegressionOnCPU(
    {predictors: Float32Array.from(predictors), response: Float32Array.from(response)},
    {predictorCount: 1}
  );
  const shifted = computeProbitRegressionOnCPU(
    {
      predictors: Float32Array.from(predictors, value => 2 ** 24 + 2 * value),
      response: Float32Array.from(response)
    },
    {predictorCount: 1}
  );
  expect(first.status[0]).toBe(0);
  expect(shifted.status[0]).toBe(0);
  expectRegressionRowsClose(shifted.fitted, first.fitted, 1e-6);
  expectRegressionRowsClose(
    shifted.marginalEffects,
    Array.from(first.marginalEffects, value => value / 2),
    1e-6
  );
  expect(first.coefficients[0]).toBeCloseTo(0.29961284, 5);
  expect(first.coefficients[1]).toBeCloseTo(0.69865819, 5);
});

it('does not certify the closing-review nearly separated counterexample', () => {
  const predictors = Float32Array.from([
    ...new Array<number>(1000).fill(-1),
    ...new Array<number>(2000).fill(1),
    -1e-5,
    1e-5
  ]);
  const response = Float32Array.from([
    ...new Array<number>(1000).fill(0),
    ...new Array<number>(2000).fill(1),
    1,
    0
  ]);
  const result = computeProbitRegressionOnCPU({predictors, response}, {predictorCount: 1});
  expect(result.status[0]).toBe(0);
  expect(result.coefficients[1]).toBeCloseTo(6.02211527, 4);
  expect(result.summary[1]).toBeCloseTo(-1.38639304475, 5);
});

it('uses exact certificates at extrema, interior support values and for multiple columns', () => {
  for (const input of [
    {
      predictors: Float32Array.of(-1, -1, 0, 0, 1, 1),
      response: Float32Array.of(0, 0, 0, 1, 1, 1),
      predictorCount: 1
    },
    {
      predictors: Float32Array.of(-2, 1, 1, -2, -1, -1, 2, -1, -1, 2, 1, 1),
      response: Float32Array.of(0, 0, 0, 1, 1, 1),
      predictorCount: 2
    }
  ]) {
    const result = computeProbitRegressionOnCPU(input, {predictorCount: input.predictorCount});
    expect(result.status[0]).toBe(4);
    expect(result.coefficients.every(Number.isNaN)).toBe(true);
    expect(result.fitted.every(Number.isNaN)).toBe(true);
    expect(result.summary[3]).toBe(0);
  }
});

it('matches hand-computed zero-score fit and failure layouts', () => {
  const input = {
    predictors: Float32Array.of(-1, -1, 0, 0, 1, 1),
    response: Float32Array.of(0, 1, 0, 1, 0, 1)
  };
  const fit = computeProbitRegressionOnCPU(input, {predictorCount: 1});
  expect(fit.status[0]).toBe(0);
  expect(Array.from(fit.coefficients)).toEqual([0, 0]);
  expect(Array.from(fit.fitted)).toEqual(new Array(6).fill(0.5));
  expect(fit.standardErrors[0]).toBeCloseTo(Math.sqrt(Math.PI / 12), 6);
  expect(fit.standardErrors[1]).toBeCloseTo(Math.sqrt(Math.PI / 8), 6);
  expectRegressionRowsClose(fit.summary, [6, -6 * Math.log(2), 12 * Math.log(2), 1], 1e-7);
  expect(Array.from(fit.marginalEffects)).toEqual([0]);
  for (const [change, status, count] of [
    [{mask: new Uint32Array(6)}, 1, 0],
    [{predictors: new Float32Array(6)}, 2, 6],
    [{response: new Float32Array(6).fill(1)}, 4, 6],
    [{response: new Float32Array(6).fill(0.5)}, 4, NaN]
  ] as const) {
    const result = computeProbitRegressionOnCPU({...input, ...change}, {predictorCount: 1});
    expect(result.status[0]).toBe(status);
    expect(
      Number.isNaN(count) ? Number.isNaN(result.summary[0]) : result.summary[0] === count
    ).toBe(true);
    for (const name of ['coefficients', 'standardErrors', 'fitted', 'marginalEffects'] as const)
      expect(result[name].every(Number.isNaN)).toBe(true);
  }
});

it('returns the diagnostic weight and variance failure statuses', () => {
  const input = getInput();
  const malformedOffsets = input.spatialDiagnostics.weights.offsets.slice();
  malformedOffsets[1] = malformedOffsets[2] + 1;
  for (const [change, status] of [
    [{weights: {...input.spatialDiagnostics.weights, status: {overflow: Uint32Array.of(1)}}}, 2],
    [{weights: {...input.spatialDiagnostics.weights, offsets: malformedOffsets}}, 2],
    [
      {
        weights: {
          ...input.spatialDiagnostics.weights,
          weights: new Float32Array(fixture.weights.length)
        }
      },
      3
    ],
    [
      {
        weights: {
          ...input.spatialDiagnostics.weights,
          neighbors: new Uint32Array(fixture.neighbors.length).fill(0xffffffff)
        }
      },
      2
    ]
  ] as const) {
    const result = computeProbitRegressionOnCPU(
      {...input, spatialDiagnostics: {...input.spatialDiagnostics, ...change}},
      {predictorCount: 2}
    );
    expect(result.status[0]).toBe(0);
    expect(result.spatialDiagnostics!.status[0]).toBe(status);
    expect(result.spatialDiagnostics!.tests.every(Number.isNaN)).toBe(true);
  }
});

it('refuses negative diagnostic weights and a CSR that does not start at zero', () => {
  // GPUSpatialWeights requires nonnegative values and offsets[0] = 0. A signed matrix can
  // still have positive diagnostic denominators, so a variance check does not validate it.
  for (const change of [
    {weights: Float32Array.from(fixture.weights, value => -value)},
    {offsets: Uint32Array.from(fixture.offsets, (value, index) => (index === 0 ? 1 : value))}
  ]) {
    const input = getInput();
    Object.assign(input.spatialDiagnostics.weights, change);
    const result = computeProbitRegressionOnCPU(input, {predictorCount: 2});
    expect(result.spatialDiagnostics!.status[0]).toBe(2);
    expect(result.spatialDiagnostics!.tests.every(Number.isNaN)).toBe(true);
  }
});

it('propagates candidate loss and failed weights convergence into diagnostic status', () => {
  for (const status of [{candidateOverflow: Uint32Array.of(1)}, {converged: Uint32Array.of(0)}]) {
    const input = getInput();
    const result = computeProbitRegressionOnCPU(
      {...input, spatialDiagnostics: {weights: {...input.spatialDiagnostics.weights, status}}},
      {predictorCount: 2}
    );
    expect(result.status[0]).toBe(0);
    expect(result.spatialDiagnostics!.status[0]).toBe(2);
    expect(result.spatialDiagnostics!.tests.every(Number.isNaN)).toBe(true);
  }
});

it('fits beyond the shader predictor cap against an orthogonal closed form', () => {
  // Twelve Walsh columns, each design row observed once with each response: zero score,
  // full rank and information (2/pi) * 64 I. The GPU permits only eleven predictors.
  const predictors: number[] = [];
  const response: number[] = [];
  for (let row = 0; row < 32; row++) {
    for (const observed of [0, 1]) {
      for (let column = 1; column <= 12; column++) {
        let bits = row & column;
        let parity = 0;
        while (bits) {
          parity ^= bits & 1;
          bits >>>= 1;
        }
        predictors.push(parity ? -1 : 1);
      }
      response.push(observed);
    }
  }
  const result = computeProbitRegressionOnCPU(
    {predictors: Float32Array.from(predictors), response: Float32Array.from(response)},
    {predictorCount: 12, maximumIterations: 257}
  );
  expect(result.status[0]).toBe(0);
  expect(result.coefficients.every(value => value === 0)).toBe(true);
  expectRegressionRowsClose(
    result.standardErrors,
    new Array(13).fill(Math.sqrt(Math.PI / 128)),
    2e-7
  );
});

it('rejects malformed arrays and validates both operations at construction', () => {
  for (const change of [{predictors: new Float32Array(191)}, {mask: new Uint32Array(95)}]) {
    expect(() =>
      computeProbitRegressionOnCPU({...getInput(), ...change}, {predictorCount: 2})
    ).toThrow(new RegExp(Object.keys(change)[0]));
  }
  expect(() =>
    computeProbitRegressionOnCPU(
      {
        ...getInput(),
        spatialDiagnostics: {
          weights: {...getInput().spatialDiagnostics.weights, offsets: Uint32Array.of(0, 3, 2)}
        }
      },
      {predictorCount: 2}
    )
  ).toThrow(/offsets/);
  const shapes = {
    predictors: {format: 'float32' as const, length: 192},
    response: {format: 'float32' as const, length: 96},
    mask: {format: 'uint32' as const, length: 96},
    offsets: {format: 'uint32' as const, length: 97},
    neighbors: {format: 'uint32' as const, length: 192},
    weights: {format: 'float32' as const, length: 192}
  };
  for (const change of [
    {predictorCount: 0},
    {maximumIterations: 0},
    {tolerance: 0},
    {tolerance: NaN}
  ]) {
    expect(() => new CPUProbitRegressionOperation({...PROPS, ...change})).toThrow(/probit-test/);
    expect(() =>
      constructRegressionOperationContributor(
        new GPUProbitRegressionOperation({...PROPS, ...change}),
        shapes
      )
    ).toThrow(/probit-test/);
  }
  expect(
    () =>
      new CPUProbitRegressionOperation({...PROPS, output: {...PROPS.output, fitted: 'response'}})
  ).toThrow(/probit-test/);
  expect(
    () =>
      new GPUProbitRegressionOperation({...PROPS, output: {...PROPS.output, fitted: 'response'}})
  ).toThrow(/probit-test/);
  expect(
    () => new CPUProbitRegressionOperation({...PROPS, predictorCount: 12, maximumIterations: 257})
  ).not.toThrow();
});

for (const tolerance of [1e-4, 1e-3, 1e-2]) {
  it(`SSR-109 certifies the closed-form maximum even with loose tolerance ${tolerance}`, () => {
    const result = computeProbitRegressionOnCPU(createProbitConvergenceFixture(), {
      predictorCount: 1,
      tolerance
    });
    expect(result.status[0]).toBe(0);
    expect(result.coefficients.every(Number.isFinite)).toBe(true);
    expect(result.coefficients[0]).toBeCloseTo(0, 6);
    expect(result.coefficients[1]).toBeCloseTo(PROBIT_CONVERGENCE_SLOPE, 5);
    expect(Number.isFinite(result.summary[1])).toBe(true);
    expect(result.summary[1]).toBeCloseTo(PROBIT_CONVERGENCE_LOG_LIKELIHOOD, 5);
    expect(result.summary[3]).toBeGreaterThan(1);
    expect(result.standardErrors).toHaveLength(2);
    for (const value of result.standardErrors) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeCloseTo(PROBIT_CONVERGENCE_STANDARD_ERROR, 5);
    }
  });
}

it('SSR-109 loose tolerance cannot certify an exhausted fit or separated data', () => {
  const unfinished = computeProbitRegressionOnCPU(createProbitConvergenceFixture(), {
    predictorCount: 1,
    tolerance: 1e-2,
    maximumIterations: 1
  });
  expect(unfinished.status[0]).toBe(3);
  expect(unfinished.summary[3]).toBe(1);
  expect(unfinished.coefficients).toHaveLength(2);
  expect(unfinished.coefficients.every(Number.isFinite)).toBe(true);
  expect(unfinished.standardErrors).toHaveLength(2);
  expect(unfinished.standardErrors.every(Number.isNaN)).toBe(true);
  const separated = computeProbitRegressionOnCPU(
    {
      predictors: Float32Array.of(-1, -1, 1, 1),
      response: Float32Array.of(0, 0, 1, 1)
    },
    {predictorCount: 1, tolerance: 1e-2}
  );
  expect(separated.status[0]).toBe(4);
  expect(separated.coefficients).toHaveLength(2);
  expect(separated.coefficients.every(Number.isNaN)).toBe(true);
  expect(separated.standardErrors).toHaveLength(2);
  expect(separated.standardErrors.every(Number.isNaN)).toBe(true);
});

it('SSR-109 loose tolerance still matches the independent masked two-predictor review fixture', () => {
  const result = computeProbitRegressionOnCPU(getInput(), {predictorCount: 2, tolerance: 1e-2});
  expect(result.status[0]).toBe(0);
  expectRegressionRowsClose(result.coefficients, fixture.expected.coefficients, 5e-6);
});

for (const tolerance of [undefined, 1e-2]) {
  it(`SSR-109 one large coefficient cannot certify an unrelated one (tolerance ${tolerance})`, () => {
    const result = computeProbitRegressionOnCPU(createProbitLargeCoefficientFixture(), {
      predictorCount: 3,
      tolerance
    });
    expect(result.status[0]).toBe(0);
    expect(result.coefficients).toHaveLength(4);
    expect(Number.isFinite(result.coefficients[3])).toBe(true);
    // Before the per-coefficient certificate: status 0 after 4 iterations at 2.2803545.
    expect(Math.abs(result.coefficients[3] - PROBIT_LARGE_COEFFICIENT_SLOPE)).toBeLessThan(1e-4);
  });
}

it('SSR-109 accepts positive tolerances below the float32 range', () => {
  for (const tolerance of [1e-46, Number.MIN_VALUE]) {
    const result = computeProbitRegressionOnCPU(createProbitConvergenceFixture(), {
      predictorCount: 1,
      tolerance
    });
    expect([0, 3]).toContain(result.status[0]);
    expect(() => new CPUProbitRegressionOperation({...PROPS, tolerance})).not.toThrow();
  }
  const loose = computeProbitRegressionOnCPU(createProbitConvergenceFixture(), {
    predictorCount: 1,
    tolerance: 1e-2
  });
  expect(loose.status[0]).toBe(0);
  expect(loose.coefficients[1]).toBeCloseTo(PROBIT_CONVERGENCE_SLOPE, 5);
});

it('both operation forms reject an explicitly empty optional mask or weights overflow name', () => {
  const lag = {
    weights: {offsets: 'lagOffsets', neighbors: 'lagNeighbors', weights: 'lagWeights'},
    output: {coefficients: 'lagCoefficients', standardErrors: 'lagErrors', status: 'lagStatus'}
  };
  const changes: Partial<ProbitRegressionOperationProps>[] = [
    {mask: ''},
    {spatialLag: {...lag, weights: {...lag.weights, status: {overflow: ''}}}},
    {
      spatialDiagnostics: {
        ...PROPS.spatialDiagnostics!,
        weights: {...PROPS.spatialDiagnostics!.weights, status: {overflow: ''}}
      }
    }
  ];
  for (const change of changes) {
    for (const Operation of [CPUProbitRegressionOperation, GPUProbitRegressionOperation]) {
      expect(() => new Operation({...PROPS, ...change})).toThrow(/probit-test/);
    }
  }
  // Omission stays valid and declares no port; a supplied name declares one.
  const {mask: _mask, ...unmasked} = PROPS;
  expect(new GPUProbitRegressionOperation(unmasked).inputs).not.toHaveProperty('mask');
  expect(new GPUProbitRegressionOperation(PROPS).inputs).toHaveProperty('mask', 'mask');
  const withLag = new GPUProbitRegressionOperation({
    ...PROPS,
    spatialLag: {...lag, weights: {...lag.weights, status: {overflow: 'lagOverflow'}}}
  });
  expect(withLag.inputs).toHaveProperty('lagWeightoverflow', 'lagOverflow');
});

it('a subnormal response word is nonbinary (status 4); signed zero is a binary zero', () => {
  const input = createProbitConvergenceFixture();
  const response = Float32Array.from(input.response);
  response[0] = 2 ** -149;
  expect(new Uint32Array(response.buffer)[0]).toBe(1);
  const rejected = computeProbitRegressionOnCPU({...input, response}, {predictorCount: 1});
  expect(rejected.status[0]).toBe(4);
  expect(rejected.coefficients.every(Number.isNaN)).toBe(true);
  response[0] = -0;
  expect(computeProbitRegressionOnCPU({...input, response}, {predictorCount: 1}).status[0]).toBe(0);
});
