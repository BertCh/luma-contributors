// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeSpatialErrorMaximumLikelihoodOnCPU,
  CPUSpatialErrorMaximumLikelihoodOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-maximum-likelihood-operation';
import {
  GPUSpatialErrorMaximumLikelihoodOperation,
  type SpatialErrorMaximumLikelihoodOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood-operation';
import {getCPUSpatialErrorLikelihoodCandidates} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-likelihood-polynomial';
import {getGPUSpatialWeightsTracesParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters';
import {
  constructRegressionOperationContributor,
  expectRegressionRowsClose
} from './regression-operation-test-utils';
import fixtures from './spatial-error-maximum-likelihood-operation-fixtures.json';

function createCompleteWeights(rowCount: number) {
  return {
    offsets: Uint32Array.from({length: rowCount + 1}, (_, row) => row * (rowCount - 1)),
    neighbors: Uint32Array.from({length: rowCount * (rowCount - 1)}, (_, slot) => {
      const row = Math.floor(slot / (rowCount - 1));
      const column = slot % (rowCount - 1);
      return column < row ? column : column + 1;
    }),
    weights: new Float32Array(rowCount * (rowCount - 1)).fill(1)
  };
}

it('reports undefined pseudo R squared for an exactly constant orthogonal prediction', () => {
  const result = computeSpatialErrorMaximumLikelihoodOnCPU(
    {
      weights: createCompleteWeights(4),
      predictors: Float32Array.of(-3, -1, 1, 3),
      response: Float32Array.of(0, 1, 1, 0),
      eigenvalues: Float32Array.of(-1, -1, -1, 3)
    },
    {predictorCount: 1, maximumPower: 16, logDeterminant: 'exact'}
  );
  // x'y_centered=0 and W acts as -I on both centered columns. Every filtered
  // regression therefore has b=(1/2,0), u=(-1/2,1/2,1/2,-1/2), and constant prediction.
  expect(result.status[0]).toBe(0);
  expect(result.table[0]).toBe(0.5);
  expect(result.table[4] === 0).toBe(true);
  expectRegressionRowsClose(result.residuals, [-0.5, 0.5, 0.5, -0.5], 0);
  expect(result.summary[6]).toBeNaN();
});

it('rejects an exactly linear response whose Gaussian likelihood has no finite maximum', () => {
  const values = Float32Array.of(0, 1, 2, 3, 4, 5, 6, 7);
  const result = computeSpatialErrorMaximumLikelihoodOnCPU(
    {
      weights: createCompleteWeights(8),
      predictors: values,
      response: values
    },
    {predictorCount: 1, maximumPower: 8}
  );
  // b=(0,1) fits every row exactly for every admissible lambda: sigma2=0, LL=+infinity.
  // The finite-positive-variance estimator must return code 3, not a rounding-created fit.
  expect(result.status[0]).toBe(3);
  expect(result.converged[0]).toBe(0);
  expect(result.summary[0]).toBe(8);
  expect(result.summary.subarray(1).every(Number.isNaN)).toBe(true);
  expect(result.table.every(Number.isNaN)).toBe(true);
  expect(result.residuals.every(Number.isNaN)).toBe(true);
});

it('retains a small nonzero residual instead of applying a perfect-fit tolerance', () => {
  const deviation = 2 ** -20;
  const result = computeSpatialErrorMaximumLikelihoodOnCPU(
    {
      weights: createCompleteWeights(8),
      predictors: Float32Array.of(0, 1, 2, 3, 4, 5, 6, 7),
      response: Float32Array.of(deviation, 1, 2, 3, 4, 5, 6, 7),
      eigenvalues: Float32Array.of(-1, -1, -1, -1, -1, -1, -1, 7)
    },
    {predictorCount: 1, maximumPower: 8, logDeterminant: 'exact'}
  );
  // The response perturbation's residual is deviation*[7,-4,-3,-2,-1,0,1,2]/12;
  // its SSE is 7*deviation^2/12. W acts as -I on these zero-sum residuals.
  expect(result.status[0]).toBe(0);
  expectRegressionRowsClose(
    [result.summary[1], result.summary[2], result.summary[8]],
    [-0.99 / 7, ((1 - 0.99 / 7) ** 2 * 7 * deviation ** 2) / 96, (7 * deviation ** 2) / 12],
    1e-7
  );
});

it('chooses the documented lower boundary for the flat W=3I profile', () => {
  const result = computeSpatialErrorMaximumLikelihoodOnCPU(
    {
      weights: {
        offsets: Uint32Array.of(0, 1, 2, 3, 4),
        neighbors: Uint32Array.of(0, 1, 2, 3),
        weights: Float32Array.of(3, 3, 3, 3)
      },
      predictors: Float32Array.of(0, 1, 2, 3),
      response: Float32Array.of(0, 1, 2, 4),
      eigenvalues: Float32Array.of(3, 3, 3, 3)
    },
    {predictorCount: 1, maximumPower: 16, logDeterminant: 'exact'}
  );
  // SSE=3/10, b=(-1/5,13/10). Filtering multiplies all rows by 1-3lambda,
  // and logdet=4 log(1-3lambda) cancels the profile's variance term identically.
  expect(result.status[0]).toBe(0);
  expect(result.summary[1]).toBe(Math.fround(-0.33));
  expect(result.summary[16]).toBe(1);
  expect(result.summary[20]).toBe(1);
  expectRegressionRowsClose(result.residuals, [0.2, -0.1, -0.4, 0.3], 1e-7);
  expectRegressionRowsClose(
    [result.table[0], result.table[4], result.summary[2], result.summary[3], result.summary[6]],
    [
      -0.2,
      1.3,
      0.075 * 1.99 ** 2,
      -2 * Math.log(0.075) - 2 * (Math.log(2 * Math.PI) + 1),
      169 / 175
    ],
    1e-7
  );
});

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

for (const fixture of fixtures) {
  it(`matches independent dense NumPy/SciPy outputs: ${fixture.name}`, () => {
    const result = computeSpatialErrorMaximumLikelihoodOnCPU(getInput(fixture), {
      ...fixture.options,
      predictorCount: fixture.predictorCount,
      traceMode: fixture.options.traceMode as 'exact' | 'stochastic',
      logDeterminant: fixture.options.logDeterminant as 'exact' | 'series'
    });
    expect(Array.from(result.status)).toEqual([0]);
    for (const port of ['table', 'summary', 'residuals', 'powerTraceStandardErrors'] as const)
      expectRegressionRowsClose(
        result[port],
        Float32Array.from(fixture.expected[port], value => value ?? NaN),
        2e-5
      );
    for (const slot of [0, 12, 14, 15, 16, 17, 19, 20])
      expect(result.summary[slot]).toBe(fixture.expected.summary[slot]);
    expect(result.converged[0]).toBe(fixture.expected.summary[15]);
    // This is the spatial parameter p-value, not a trivial far-tail-only comparison.
    expect(result.table[15]).toBeGreaterThan(0.01);
    expect(result.table[15]).toBeLessThan(0.9);
  });
}

it('isolates two stationary points inside one GPU scan interval, and recognizes a flat profile', () => {
  const response = [-1, 1, -1, 1];
  const orthogonal = [-1, -1, 1, 1];
  const perturbation = 1e-7;
  const slope = 1e6;
  const rows = response.map(value => [1, value]);
  const lagged = response.map((value, row) => [0, 2 * value + perturbation * orthogonal[row]]);
  // SSE/4=(1-2t)^2+epsilon^2*t^2. Adding logdet=-slope*t gives two
  // stationary points separated by 4e-6, both inside the GPU interval [0.499,0.501].
  const candidates = getCPUSpatialErrorLikelihoodCandidates(
    rows,
    lagged,
    [slope, 0],
    undefined,
    undefined,
    4,
    0.9
  );
  const quadratic = 4 + perturbation ** 2;
  const discriminant = Math.sqrt(16 - (4 * slope ** 2 * perturbation ** 2) / quadratic ** 2);
  const maximum =
    2 / quadratic - (2 * slope * perturbation ** 2) / quadratic ** 2 / (4 + discriminant);
  expect(candidates).toHaveLength(2);
  expect(candidates[0]).toEqual({parameter: -0.9, atBoundary: true});
  expect(Math.abs(candidates[1].parameter - maximum)).toBeLessThan(2e-15);
  expect(candidates[1].atBoundary).toBe(false);
  // W=I: SSE=(1-t)^2*SSE(0), logdet=n*log(1-t), hence a constant profile.
  expect(
    getCPUSpatialErrorLikelihoodCandidates(rows, rows, [], [1, 1, 1, 1], undefined, 4, 0.9)
  ).toEqual([{parameter: -0.9, atBoundary: true}]);
  // A stationary maximum at either endpoint is a boundary solution: only interior fits converge.
  for (const sign of [1, -1])
    expect(
      getCPUSpatialErrorLikelihoodCandidates(
        rows,
        response.map((value, row) => [0, sign * (2 * value + 2 * orthogonal[row])]),
        [0, 0],
        undefined,
        undefined,
        4,
        0.25
      )
    ).toEqual([{parameter: sign * 0.25, atBoundary: true}]);
});

it('matches a closed form on W=J-I, including the raw constant column and unfiltered residuals', () => {
  const input = {
    weights: {
      offsets: Uint32Array.of(0, 3, 6, 9, 12),
      neighbors: Uint32Array.of(1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2),
      weights: new Float32Array(12).fill(1)
    },
    predictors: Float32Array.of(-1, 0, 1, 2),
    response: Float32Array.of(0, 1, 4, 9),
    eigenvalues: Float32Array.of(-1, -1, -1, 3)
  };
  const result = computeSpatialErrorMaximumLikelihoodOnCPU(input, {
    predictorCount: 1,
    maximumPower: 16,
    logDeterminant: 'exact'
  });
  // OLS residual u=[1,-1,-1,1], Wu=-u. The concentrated likelihood is
  // log(1-3lambda)-log(1+lambda)+constant, strictly decreasing over the interval.
  expect(result.status[0]).toBe(0);
  expectRegressionRowsClose(
    [result.table[0], result.table[4], result.table[8]],
    [2, 3, -0.33],
    1e-7
  );
  expectRegressionRowsClose(result.residuals, [1, -1, -1, 1], 1e-7);
  expectRegressionRowsClose([result.summary[2]], [0.67 ** 2], 1e-7);
  expect(result.summary[8]).toBe(4);
  expect(result.summary[16]).toBe(1);
  expect(result.converged[0]).toBe(0);
  const longSeries = computeSpatialErrorMaximumLikelihoodOnCPU(input, {
    predictorCount: 1,
    maximumPower: 257,
    logDeterminant: 'exact'
  });
  expect(longSeries.status[0]).toBe(0);
  expect(longSeries.powerTraceStandardErrors).toHaveLength(257);
  expectRegressionRowsClose(
    [longSeries.table[0], longSeries.table[4], longSeries.table[8]],
    [2, 3, -0.33],
    1e-7
  );
  expect(longSeries.table[9]).not.toBe(result.table[9]);
});

it('uses the documented default series length and supports more than fifteen predictors', () => {
  const defaults = computeSpatialErrorMaximumLikelihoodOnCPU(getInput(), {predictorCount: 2});
  expect(defaults.status[0]).toBe(0);
  expect(defaults.powerTraceStandardErrors).toHaveLength(128);
  expectRegressionRowsClose(defaults.table, Float32Array.from(fixtures[0].expected.table), 2e-5);
  const rowCount = 32;
  const predictorCount = 16;
  const getHadamardEntry = (row: number, column: number) => {
    let parity = row & column;
    parity ^= parity >> 4;
    parity ^= parity >> 2;
    parity ^= parity >> 1;
    return (parity & 1) === 0 ? 1 : -1;
  };
  const predictors = Float32Array.from({length: rowCount * predictorCount}, (_, slot) =>
    getHadamardEntry(Math.floor(slot / predictorCount), (slot % predictorCount) + 1)
  );
  const response = Float32Array.from(
    {length: rowCount},
    (_, row) =>
      0.25 +
      Array.from(
        {length: predictorCount},
        (_, column) => (predictors[row * predictorCount + column] * (column + 1)) / 16
      ).reduce((sum, value) => sum + value, 0) +
      getHadamardEntry(row, 17) / 2
  );
  const result = computeSpatialErrorMaximumLikelihoodOnCPU(
    {
      weights: {
        offsets: Uint32Array.from({length: rowCount + 1}, (_, row) => row * (rowCount - 1)),
        neighbors: Uint32Array.from({length: rowCount * (rowCount - 1)}, (_, slot) => {
          const row = Math.floor(slot / (rowCount - 1));
          const column = slot % (rowCount - 1);
          return column < row ? column : column + 1;
        }),
        weights: new Float32Array(rowCount * (rowCount - 1)).fill(1)
      },
      predictors,
      response,
      eigenvalues: Float32Array.from({length: rowCount}, (_, row) =>
        row === 0 ? rowCount - 1 : -1
      )
    },
    {predictorCount, maximumPower: 2, logDeterminant: 'exact'}
  );
  // Orthogonal Hadamard columns: all slopes stay (column+1)/16 and u is column 17 / 2.
  // W=J-I again makes the profile strictly decreasing, independently of predictor count.
  expect(result.status[0]).toBe(0);
  expect(result.table[0]).toBe(0.25);
  for (let column = 0; column < predictorCount; column++)
    expect(result.table[(column + 1) * 4]).toBe((column + 1) / 16);
  expectRegressionRowsClose(
    [result.summary[1], result.summary[2]],
    [-0.99 / 31, (1 - 0.99 / 31) ** 2 / 4],
    1e-7
  );
});

it('preserves response and weights scaling, centered large offsets, unused slots and the Philox seed', () => {
  const input = getInput();
  const options = {
    predictorCount: 2,
    maximumPower: 16,
    traceMode: 'stochastic' as const,
    traceProbeCount: 12
  };
  const ordinary = computeSpatialErrorMaximumLikelihoodOnCPU(input, options);
  const scaled = computeSpatialErrorMaximumLikelihoodOnCPU(
    {
      ...input,
      response: Float32Array.from(input.response, value => 2 * value),
      weights: {
        ...input.weights,
        weights: Float32Array.from(input.weights.weights, value => 4 * value)
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
  expectRegressionRowsClose(scaled.powerTraceStandardErrors, ordinary.powerTraceStandardErrors, 0);
  const largeResponse = computeSpatialErrorMaximumLikelihoodOnCPU(
    {
      ...input,
      response: Float32Array.from(input.response, value => value * 2 ** 40)
    },
    options
  );
  expect(largeResponse.status[0]).toBe(0);
  expectRegressionRowsClose(
    [largeResponse.summary[1], largeResponse.summary[2] / 2 ** 80, largeResponse.summary[6]],
    [ordinary.summary[1], ordinary.summary[2], ordinary.summary[6]],
    1e-6
  );
  const largeWeights = computeSpatialErrorMaximumLikelihoodOnCPU(
    {
      ...input,
      weights: {
        ...input.weights,
        weights: Float32Array.from(input.weights.weights, value => value * 2 ** 128)
      }
    },
    options
  );
  expect(largeWeights.status[0]).toBe(0);
  expect(largeWeights.summary[1]).toBe(Math.fround(ordinary.summary[1] / 2 ** 128));
  expectRegressionRowsClose(
    [largeWeights.summary[2], largeWeights.summary[6]],
    [ordinary.summary[2], ordinary.summary[6]],
    1e-6
  );
  const otherSeed = computeSpatialErrorMaximumLikelihoodOnCPU(
    {...input, traceParameters: getGPUSpatialWeightsTracesParameterValues({seed: 43})},
    options
  );
  expect(otherSeed.summary[1]).not.toBe(ordinary.summary[1]);
  expectRegressionRowsClose(
    computeSpatialErrorMaximumLikelihoodOnCPU(input, options).table,
    ordinary.table,
    0
  );
  const padded = {
    ...input,
    weights: {
      ...input.weights,
      neighbors: Uint32Array.from([...input.weights.neighbors, 0xffffffff]),
      weights: Float32Array.from([...input.weights.weights, NaN])
    }
  };
  expectRegressionRowsClose(
    computeSpatialErrorMaximumLikelihoodOnCPU(padded, options).summary,
    ordinary.summary,
    0
  );
  const sentinelOffsets = padded.weights.offsets.slice();
  sentinelOffsets[sentinelOffsets.length - 1]++;
  expectRegressionRowsClose(
    computeSpatialErrorMaximumLikelihoodOnCPU(
      {
        ...padded,
        weights: {...padded.weights, offsets: sentinelOffsets}
      },
      options
    ).summary,
    ordinary.summary,
    0
  );
  // Quantized data are identical up to an exact 2^24 shift; centered fitting must retain the
  // residual variation while the raw intercept changes units.
  const quantized = {
    ...input,
    predictors: Float32Array.from(input.predictors, value => 2 * Math.round(value)),
    response: Float32Array.from(input.response, value => 2 * Math.round(value))
  };
  const shifted = {
    ...quantized,
    predictors: Float32Array.from(quantized.predictors, value => value + 2 ** 24),
    response: Float32Array.from(quantized.response, value => value + 2 ** 24)
  };
  const first = computeSpatialErrorMaximumLikelihoodOnCPU(quantized, options);
  const second = computeSpatialErrorMaximumLikelihoodOnCPU(shifted, options);
  expect(first.status[0]).toBe(0);
  expect(second.status[0]).toBe(0);
  expectRegressionRowsClose(
    [second.summary[1], second.summary[2]],
    [first.summary[1], first.summary[2]],
    1e-6
  );
  expectRegressionRowsClose(second.residuals, first.residuals, 1e-6);
});

it('reports status and NaN fields for singular, too-small, zero-link and non-finite inputs', () => {
  const input = getInput();
  const options = {predictorCount: 2, maximumPower: 8};
  const cases = [
    {input: {...input, predictors: new Float32Array(input.predictors.length)}, code: 1},
    {
      input: {
        ...input,
        weights: {...input.weights, weights: new Float32Array(input.weights.weights.length)}
      },
      code: 3
    },
    {
      input: {
        ...input,
        response: Float32Array.from(input.response, (value, row) => (row === 2 ? NaN : value))
      },
      code: 3
    },
    {
      input: {
        ...input,
        weights: {
          ...input.weights,
          weights: Float32Array.from(input.weights.weights, (value, slot) =>
            slot === 0 ? Infinity : value
          )
        }
      },
      code: 3
    },
    {
      input: {
        weights: {
          offsets: Uint32Array.of(0, 1, 2, 3),
          neighbors: Uint32Array.of(1, 2, 0),
          weights: Float32Array.of(1, 1, 1)
        },
        predictors: Float32Array.of(0, 1, 1, 0, 0, 2),
        response: Float32Array.of(1, 2, 4)
      },
      code: 2
    }
  ];
  for (const fixture of cases) {
    const result = computeSpatialErrorMaximumLikelihoodOnCPU(fixture.input, options);
    expect(result.status[0]).toBe(fixture.code);
    expect(result.converged[0]).toBe(0);
    expect(result.table.every(Number.isNaN)).toBe(true);
    expect(result.residuals.every(Number.isNaN)).toBe(true);
    expect(result.summary[0]).toBe(fixture.input.response.length);
    expect(result.summary.subarray(1).every(Number.isNaN)).toBe(true);
    expect(result.powerTraceStandardErrors.every(value => value === 0)).toBe(true);
  }
});

it('rejects inconsistent arrays, and supports valid empty CSR slots', () => {
  const input = getInput();
  for (const changes of [
    {predictors: new Float32Array(1)},
    {response: new Float32Array(1)},
    {traceParameters: new Uint32Array(0)},
    {eigenvalues: new Float32Array(1)}
  ])
    expect(() =>
      computeSpatialErrorMaximumLikelihoodOnCPU(
        {...input, ...changes},
        {predictorCount: 2, maximumPower: 8}
      )
    ).toThrow(new RegExp(Object.keys(changes)[0]));
  for (const weights of [
    {...input.weights, offsets: Uint32Array.of(0, 99999)},
    {...input.weights, offsets: Uint32Array.of(0, 2, 1)},
    {...input.weights, weights: new Float32Array(1)}
  ])
    expect(() =>
      computeSpatialErrorMaximumLikelihoodOnCPU({...input, weights}, {predictorCount: 2})
    ).toThrow(/weights/);
  expect(() =>
    computeSpatialErrorMaximumLikelihoodOnCPU(input, {predictorCount: 2, logDeterminant: 'exact'})
  ).toThrow(/eigenvalues/);
  expect(
    computeSpatialErrorMaximumLikelihoodOnCPU(
      {
        ...input,
        weights: {
          offsets: new Uint32Array(input.response.length + 1),
          neighbors: new Uint32Array(0),
          weights: new Float32Array(0)
        }
      },
      {predictorCount: 2, maximumPower: 8}
    ).status[0]
  ).toBe(3);
});

it('validates CPU construction and GPU contributor construction, with identical graph contracts', () => {
  const input = getInput();
  const props: SpatialErrorMaximumLikelihoodOperationProps = {
    id: 'error-ml-test',
    rowCount: input.response.length,
    predictorCount: 2,
    maximumPower: 8,
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    predictors: 'predictors',
    response: 'response',
    traceParameters: 'traceParameters',
    output: {
      status: {code: 'code', converged: 'converged'},
      table: 'table',
      summary: 'summary',
      residuals: 'residuals',
      powerTraceStandardErrors: 'errors'
    }
  };
  const shapes = {
    offsets: {format: 'uint32' as const, length: input.weights.offsets.length},
    neighbors: {format: 'uint32' as const, length: input.weights.neighbors.length},
    weights: {format: 'float32' as const, length: input.weights.weights.length},
    predictors: {format: 'float32' as const, length: input.predictors.length},
    response: {format: 'float32' as const, length: input.response.length},
    traceParameters: {format: 'uint32' as const, length: 1}
  };
  for (const changes of [
    {predictorCount: 0},
    {maximumPower: 1},
    {maximumPower: 2, tailExtrapolation: true},
    {traceProbeCount: 0},
    {tileRowCount: 0},
    {traceMode: 'invalid' as never},
    {logDeterminant: 'exact' as const}
  ]) {
    expect(() => new CPUSpatialErrorMaximumLikelihoodOperation({...props, ...changes})).toThrow(
      /error-ml-test/
    );
    expect(() =>
      constructRegressionOperationContributor(
        new GPUSpatialErrorMaximumLikelihoodOperation({...props, ...changes}),
        shapes
      )
    ).toThrow(/error-ml-test/);
  }
  expect(
    () =>
      new CPUSpatialErrorMaximumLikelihoodOperation({
        ...props,
        predictorCount: 16,
        maximumPower: 257
      })
  ).not.toThrow();
  for (const Operation of [
    CPUSpatialErrorMaximumLikelihoodOperation,
    GPUSpatialErrorMaximumLikelihoodOperation
  ])
    expect(() => new Operation({...props, output: {...props.output, table: 'response'}})).toThrow(
      /error-ml-test/
    );
  expect(new CPUSpatialErrorMaximumLikelihoodOperation(props).contract.outputs).toEqual(
    new GPUSpatialErrorMaximumLikelihoodOperation(props).contract.outputs
  );
  expect(() =>
    constructRegressionOperationContributor(
      new GPUSpatialErrorMaximumLikelihoodOperation(props),
      shapes
    )
  ).not.toThrow();
});
