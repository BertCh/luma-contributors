// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeSpatialLagMaximumLikelihoodOnCPU,
  CPUSpatialLagMaximumLikelihoodOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-lag-maximum-likelihood-operation';
import {
  GPUSpatialLagMaximumLikelihoodOperation,
  type SpatialLagMaximumLikelihoodOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood-operation';
import {computeSpatialWeightsTracesOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-weights-traces-operation';
import {getGPUSpatialWeightsTracesParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters';
import {
  constructRegressionOperationContributor,
  expectRegressionRowsClose
} from './regression-operation-test-utils';
import fixtures from './spatial-lag-maximum-likelihood-operation-fixtures.json';

function getInput(fixture = fixtures[0]) {
  return {
    weights: {
      offsets: Uint32Array.from(fixture.weights.offsets),
      neighbors: Uint32Array.from(fixture.weights.neighbors),
      weights: Float32Array.from(fixture.weights.weights)
    },
    predictors: Float32Array.from(fixture.predictors),
    response: Float32Array.from(fixture.response),
    eigenvalues: fixture.eigenvalues ? Float32Array.from(fixture.eigenvalues) : undefined,
    traceParameters: getGPUSpatialWeightsTracesParameterValues({seed: 42})
  };
}

for (const fixture of fixtures.filter(value => value.name !== 'high-fit-64')) {
  it(`matches independent dense NumPy/SciPy outputs: ${fixture.name}`, () => {
    const result = computeSpatialLagMaximumLikelihoodOnCPU(getInput(fixture), {
      ...fixture.options,
      predictorCount: fixture.predictorCount,
      traceMode: fixture.options.traceMode as 'exact' | 'stochastic',
      logDeterminant: fixture.options.logDeterminant as 'exact' | 'series'
    });
    expect(Array.from(result.status)).toEqual([0]);
    for (const port of [
      'table',
      'summary',
      'residuals',
      'powerTraces',
      'powerSums',
      'infinityNorm',
      'powerTraceStandardErrors'
    ] as const)
      expectRegressionRowsClose(
        result[port],
        Float32Array.from(fixture.expected[port], value => value ?? NaN),
        5e-5
      );
    for (const slot of [0, 12, 14, 15, 16, 17, 19, 20])
      expect(result.summary[slot]).toBe(fixture.expected.summary[slot]);
    expect(result.converged[0]).toBe(fixture.expected.summary[15]);
    expect(result.table[15]).toBeGreaterThan(0.01);
    expect(result.table[15]).toBeLessThan(0.9);
  });
}

it("resolves the positive minimum of the review's 64-row near-perfect fit", () => {
  const fixture = fixtures.find(value => value.name === 'high-fit-64')!;
  const result = computeSpatialLagMaximumLikelihoodOnCPU(getInput(fixture), {
    predictorCount: 1,
    maximumPower: 64,
    logDeterminant: 'exact'
  });
  expect(result.status[0]).toBe(0);
  expect(result.converged[0]).toBe(1);
  expect(result.summary[2]).toBeGreaterThan(0);
  // The dense oracle uses fresh least-squares fits and Brent's envelope score, resolving
  // rho to 5e-16 despite a profile standard error below one float32 step.
  expectRegressionRowsClose(
    [result.summary[1], result.summary[2], result.summary[3], result.summary[8]],
    [
      fixture.expected.summary[1]!,
      fixture.expected.summary[2]!,
      fixture.expected.summary[3]!,
      fixture.expected.summary[8]!
    ],
    2e-6
  );
  expectRegressionRowsClose(
    [result.table[1], result.table[5], result.table[9]],
    [fixture.expected.table[1], fixture.expected.table[5], fixture.expected.table[9]],
    2e-6
  );
  expectRegressionRowsClose(result.residuals, fixture.expected.residuals, 2e-5);
});

it('keeps stationarity scale invariant for responses of order 1e10 and preserves large offsets', () => {
  const fixture = fixtures.find(value => value.name === 'high-fit-64')!;
  const input = getInput(fixture);
  const options = {predictorCount: 1, maximumPower: 64, logDeterminant: 'exact' as const};
  const ordinary = computeSpatialLagMaximumLikelihoodOnCPU(input, options);
  const scale = 2 ** 34;
  const scaled = computeSpatialLagMaximumLikelihoodOnCPU(
    {...input, response: Float32Array.from(input.response, value => scale * value)},
    options
  );
  const shifted = computeSpatialLagMaximumLikelihoodOnCPU(
    {...input, response: Float32Array.from(input.response, value => value + 10000)},
    options
  );
  expect(scaled.status[0]).toBe(0);
  expect(shifted.status[0]).toBe(0);
  expectRegressionRowsClose(
    [scaled.summary[1], scaled.summary[2] / scale ** 2],
    [ordinary.summary[1], ordinary.summary[2]],
    2e-6
  );
  expectRegressionRowsClose(
    Float32Array.from(scaled.residuals, value => value / scale),
    ordinary.residuals,
    2e-6
  );
  expectRegressionRowsClose(shifted.residuals, ordinary.residuals, 2e-6);
  expectRegressionRowsClose(
    [shifted.summary[1], shifted.summary[2]],
    [ordinary.summary[1], ordinary.summary[2]],
    2e-6
  );
  expect(Math.abs(shifted.table[0] - 10000 * (1 - ordinary.summary[1]))).toBeLessThan(0.001);
});

it('fits S=0 with its vertex outside the interval: the six-row closing-review closed form', () => {
  const rowCount = 6;
  const input = {
    weights: {
      offsets: Uint32Array.from({length: rowCount + 1}, (_, row) => 2 * row),
      neighbors: Uint32Array.from(
        {length: rowCount * 2},
        (_, slot) => (Math.floor(slot / 2) + (slot % 2 ? 1 : rowCount - 1)) % rowCount
      ),
      weights: new Float32Array(rowCount * 2).fill(0.5)
    },
    predictors: Float32Array.of(1, -1, 1, -1, 1, -1),
    response: Float32Array.of(2, 1, -1, -2, -1, 1),
    eigenvalues: Float32Array.of(-1, -0.5, -0.5, 0.5, 0.5, 1)
  };
  const result = computeSpatialLagMaximumLikelihoodOnCPU(input, {
    predictorCount: 1,
    logDeterminant: 'exact'
  });
  const parameter = (Math.sqrt(17) - 1) / 4;
  expect(result.status[0]).toBe(0);
  expect(result.converged[0]).toBe(1);
  expectRegressionRowsClose(
    [result.summary[1], result.summary[2], result.summary[3]],
    [parameter, 2 * (1 - parameter / 2) ** 2, -8.89478870858],
    2e-6
  );
  expectRegressionRowsClose(
    result.residuals,
    Float32Array.from(input.response, value => value * (1 - parameter / 2)),
    2e-6
  );
});

it("fits e1'e1=0: the reciprocal-pair closed form has rho zero and defined variance", () => {
  const result = computeSpatialLagMaximumLikelihoodOnCPU(
    {
      weights: {
        offsets: Uint32Array.from({length: 9}, (_, row) => row),
        neighbors: Uint32Array.of(4, 5, 6, 7, 0, 1, 2, 3),
        weights: new Float32Array(8).fill(1)
      },
      predictors: Float32Array.of(0, 0, 0, 0, -1, 1, -1, 1),
      response: Float32Array.of(-1, 1, -1, 1, 0, 0, 0, 0)
    },
    {predictorCount: 1}
  );
  expect(result.status[0]).toBe(0);
  expect(result.summary[1] === 0).toBe(true);
  expect(result.summary[2]).toBe(0.5);
  expect(result.summary[3]).toBeCloseTo(-8.5789195434, 5);
  // y is orthogonal to both design columns and rho=0, so the fitted response is constant.
  // Syy=Syu=Suu=4 and Su=0 give covariance=variance=0: pseudo-R squared is undefined.
  expect(result.summary[6]).toBeNaN();
  expectRegressionRowsClose(
    [result.table[1], result.table[5], result.table[9]],
    [0.25, Math.sqrt(0.125), 0.25],
    1e-7
  );
  expect(result.summary[19]).toBe(1);
});

it('does not certify a zero-RSS pole inside the admissible interval as a finite maximum', () => {
  const response = Float32Array.of(-3, 1, -1, 2, 4, -2, 0, -1);
  const neighbors = Uint32Array.of(4, 5, 6, 7, 0, 1, 2, 3);
  const result = computeSpatialLagMaximumLikelihoodOnCPU(
    {
      weights: {
        offsets: Uint32Array.from({length: 9}, (_, row) => row),
        neighbors,
        weights: new Float32Array(8).fill(1)
      },
      response,
      eigenvalues: Float32Array.of(-1, -1, -1, -1, 1, 1, 1, 1),
      predictors: Float32Array.from(
        response,
        (value, row) => value - 0.5 * response[neighbors[row]]
      )
    },
    {predictorCount: 1, logDeterminant: 'exact', maximumPower: 8}
  );
  expect(result.status[0]).toBe(3);
  expect(result.converged[0]).toBe(0);
});

it('preserves response/weight scaling, unused capacity, sentinel neighbors and the Philox seed', () => {
  const input = getInput();
  const options = {
    predictorCount: 2,
    maximumPower: 16,
    traceMode: 'stochastic' as const,
    traceProbeCount: 12
  };
  const ordinary = computeSpatialLagMaximumLikelihoodOnCPU(input, options);
  const scaled = computeSpatialLagMaximumLikelihoodOnCPU(
    {
      ...input,
      response: Float32Array.from(input.response, value => value * 2),
      weights: {
        ...input.weights,
        weights: Float32Array.from(input.weights.weights, value => value * 4)
      }
    },
    options
  );
  expectRegressionRowsClose(
    [scaled.summary[1] * 4, scaled.summary[2] / 4],
    [ordinary.summary[1], ordinary.summary[2]],
    1e-6
  );
  expectRegressionRowsClose(
    Float32Array.from(scaled.residuals, value => value / 2),
    ordinary.residuals,
    1e-6
  );
  expectRegressionRowsClose(scaled.powerTraces, ordinary.powerTraces, 0);
  const traces = computeSpatialWeightsTracesOnCPU(
    {weights: input.weights, parameters: input.traceParameters},
    {maximumPower: 16, mode: 'stochastic', probeCount: 12}
  );
  for (const port of [
    'powerTraces',
    'powerSums',
    'powerTraceStandardErrors',
    'infinityNorm'
  ] as const)
    expectRegressionRowsClose(ordinary[port], traces[port], 2e-7);
  const otherSeed = computeSpatialLagMaximumLikelihoodOnCPU(
    {...input, traceParameters: getGPUSpatialWeightsTracesParameterValues({seed: 43})},
    options
  );
  expect(otherSeed.summary[1]).not.toBe(ordinary.summary[1]);
  const unused = computeSpatialLagMaximumLikelihoodOnCPU(
    {
      ...input,
      weights: {
        ...input.weights,
        neighbors: Uint32Array.from([...input.weights.neighbors, 0xffffffff]),
        weights: Float32Array.from([...input.weights.weights, NaN])
      }
    },
    options
  );
  expectRegressionRowsClose(unused.table, ordinary.table, 0);
  const sentinelOffsets = input.weights.offsets.slice();
  sentinelOffsets[sentinelOffsets.length - 1]++;
  const sentinel = computeSpatialLagMaximumLikelihoodOnCPU(
    {
      ...input,
      weights: {
        ...input.weights,
        offsets: sentinelOffsets,
        neighbors: Uint32Array.from([...input.weights.neighbors, 0xffffffff]),
        weights: Float32Array.from([...input.weights.weights, NaN])
      }
    },
    options
  );
  expectRegressionRowsClose(sentinel.table, ordinary.table, 0);
});

it('reports too few rows, rank deficiency, zero links and non-finite inputs with their defined statuses', () => {
  const input = getInput();
  const options = {predictorCount: 2, maximumPower: 4};
  const singular = computeSpatialLagMaximumLikelihoodOnCPU(
    {...input, predictors: new Float32Array(input.predictors.length)},
    options
  );
  expect(singular.status[0]).toBe(1);
  for (const invalid of [
    {...input, response: Float32Array.from(input.response, (value, row) => (row ? value : NaN))},
    {
      ...input,
      weights: {...input.weights, weights: new Float32Array(input.weights.weights.length)}
    },
    {
      ...input,
      weights: {
        ...input.weights,
        weights: Float32Array.from(input.weights.weights, (value, slot) =>
          slot ? value : Infinity
        )
      }
    }
  ]) {
    const result = computeSpatialLagMaximumLikelihoodOnCPU(invalid, options);
    expect(result.status[0]).toBe(3);
    expect(result.converged[0]).toBe(0);
    expect(result.table.every(Number.isNaN)).toBe(true);
    expect(result.summary[0]).toBe(input.response.length);
    expect(result.summary.subarray(1).every(Number.isNaN)).toBe(true);
  }
  const empty = {
    offsets: Uint32Array.of(0, 0, 0, 0),
    neighbors: new Uint32Array(0),
    weights: new Float32Array(0)
  };
  const short = computeSpatialLagMaximumLikelihoodOnCPU(
    {weights: empty, predictors: Float32Array.of(1, 2, 3), response: Float32Array.of(NaN, 2, 3)},
    {predictorCount: 1}
  );
  expect(short.status[0]).toBe(2);
});

it('rejects inconsistent arrays and names the argument', () => {
  const input = getInput();
  const options = {predictorCount: 2, maximumPower: 4};
  for (const [replacement, argument] of [
    [{response: new Float32Array(1)}, 'response'],
    [{predictors: new Float32Array(1)}, 'predictors'],
    [{traceParameters: new Uint32Array(0)}, 'traceParameters'],
    [{eigenvalues: new Float32Array(input.response.length)}, 'eigenvalues'],
    [{weights: {...input.weights, offsets: Uint32Array.of(1, 2)}}, 'weights.offsets'],
    [{weights: {...input.weights, neighbors: new Uint32Array(1)}}, 'weights.weights'],
    [{weights: {...input.weights, offsets: Uint32Array.of(0, 2, 1)}}, 'weights.offsets']
  ] as const)
    expect(() =>
      computeSpatialLagMaximumLikelihoodOnCPU({...input, ...replacement}, options)
    ).toThrow(argument);
  expect(() =>
    computeSpatialLagMaximumLikelihoodOnCPU(
      {...input, eigenvalues: new Float32Array(1)},
      {...options, logDeterminant: 'exact'}
    )
  ).toThrow('eigenvalues');
});

function getProps(): SpatialLagMaximumLikelihoodOperationProps {
  return {
    id: 'lag-validation',
    rowCount: 8,
    predictorCount: 1,
    maximumPower: 8,
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    predictors: 'predictors',
    response: 'response',
    output: {
      status: {code: 'code', converged: 'converged'},
      table: 'table',
      summary: 'summary',
      residuals: 'residuals'
    }
  };
}

it('validates both operation forms when constructing the underlying contributor, without a browser', () => {
  const inputs = {
    offsets: {format: 'uint32' as const, length: 9},
    neighbors: {format: 'uint32' as const, length: 16},
    weights: {format: 'float32' as const, length: 16},
    predictors: {format: 'float32' as const, length: 8},
    response: {format: 'float32' as const, length: 8}
  };
  const cases: {change: Partial<SpatialLagMaximumLikelihoodOperationProps>; argument: string}[] = [
    {change: {predictorCount: 0}, argument: 'predictorCount'},
    {change: {maximumPower: 1}, argument: 'maximumPower'},
    {change: {maximumPower: 2, tailExtrapolation: true}, argument: 'maximumPower'},
    {change: {traceProbeCount: 0}, argument: 'traceProbeCount'},
    {change: {tileRowCount: 0}, argument: 'tileRowCount'},
    {change: {traceMode: 'other' as never}, argument: 'traceMode'},
    {change: {logDeterminant: 'other' as never}, argument: 'logDeterminant'},
    {change: {logDeterminant: 'exact'}, argument: 'eigenvalues'},
    {change: {output: {...getProps().output, powerTraces: 'traces'}}, argument: 'powerTraces'}
  ];
  for (const {change, argument} of cases) {
    const props = {...getProps(), ...change};
    expect(() => new CPUSpatialLagMaximumLikelihoodOperation(props)).toThrow(
      new RegExp(`^lag-validation.*${argument}`)
    );
    expect(() =>
      constructRegressionOperationContributor(
        new GPUSpatialLagMaximumLikelihoodOperation(props),
        inputs
      )
    ).toThrow(new RegExp(`^lag-validation.*${argument}`));
  }
  expect(() =>
    constructRegressionOperationContributor(
      new GPUSpatialLagMaximumLikelihoodOperation(getProps()),
      inputs
    )
  ).not.toThrow();
  expect(
    () =>
      new CPUSpatialLagMaximumLikelihoodOperation({
        ...getProps(),
        predictorCount: 15,
        maximumPower: 257,
        rowCount: 1025,
        logDeterminant: 'exact',
        eigenvalues: 'eigenvalues'
      })
  ).not.toThrow();
});
