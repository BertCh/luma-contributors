// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeSpatialTwoStageLeastSquaresOnCPU,
  CPUSpatialTwoStageLeastSquaresOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-two-stage-least-squares-operation';
import {
  GPUSpatialTwoStageLeastSquaresOperation,
  type SpatialTwoStageLeastSquaresOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares-operation';
import {
  constructRegressionOperationContributor,
  expectRegressionRowsClose
} from './regression-operation-test-utils';
import fixtures from './spatial-two-stage-least-squares-operation-fixtures.json';

function getInput(fixture = fixtures[0]) {
  return {
    predictors: Float32Array.from(fixture.predictors),
    response: Float32Array.from(fixture.response),
    weights: {
      offsets: Uint32Array.from(fixture.weights.offsets),
      neighbors: Uint32Array.from(fixture.weights.neighbors),
      weights: Float32Array.from(fixture.weights.weights)
    }
  };
}
const PROPS: SpatialTwoStageLeastSquaresOperationProps = {
  id: 'iv-test',
  rowCount: 48,
  predictorCount: 2,
  predictors: 'predictors',
  response: 'response',
  weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
  output: {status: {code: 'code'}, table: 'table', summary: 'summary', residuals: 'residuals'}
};

it('projects nearly coincident instruments without squaring their condition number', () => {
  const epsilon = 2 ** -24;
  const result = computeSpatialTwoStageLeastSquaresOnCPU(
    {
      predictors: Float32Array.of(-1, -1 + epsilon, 1, 1, -1, -1, 1, 1 - epsilon),
      response: Float32Array.of(1, 3, -1, 2, 7, 0, -2, 8),
      weights: {
        offsets: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8),
        neighbors: Uint32Array.of(1, 0, 3, 2, 5, 4, 7, 6),
        weights: new Float32Array(8).fill(1)
      }
    },
    {predictorCount: 1}
  );
  // s=(x+Wx)/2 and a=(x-Wx)/2 are orthogonal. Their squared norms are
  // 8-4*epsilon+epsilon² and epsilon²; s'y=-4-epsilon and a'y=-4*epsilon.
  // Set A=s'y/(s's), B=a'y/(a'a). The IV equations in these independent directions
  // give beta=2AB/(A+B), rho=(A-B)/(A+B), intercept=mean(y)*(1-rho).
  const expectedSlope = -1801439877791744 / 1801439810682881;
  expect(result.status[0]).toBe(0);
  expect(result.table[0]).toBe(4.5);
  expect(result.table[4]).toBe(Math.fround(expectedSlope));
  expect(result.table[8]).toBe(-1);
});

it('reports finite rank failure as robust status 1 and non-finite failure as status 3', () => {
  const input = {
    predictors: new Float32Array(8),
    response: Float32Array.of(1, 3, -1, 2, 7, 0, -2, 8),
    weights: {
      offsets: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8),
      neighbors: Uint32Array.of(1, 0, 3, 2, 5, 4, 7, 6),
      weights: new Float32Array(8).fill(1)
    },
    robust: {method: 'HC0' as const}
  };
  // Both centered instrument columns are zero: no projection, hence no covariance.
  const result = computeSpatialTwoStageLeastSquaresOnCPU(input, {predictorCount: 1});
  expect(result.status[0]).toBe(1);
  expect(result.robust!.status[0]).toBe(1);
  expect(result.robust!.covariance.every(Number.isNaN)).toBe(true);
  const nonFinite = computeSpatialTwoStageLeastSquaresOnCPU(
    {...input, response: new Float32Array(8).fill(NaN)},
    {predictorCount: 1}
  );
  expect(nonFinite.robust!.status[0]).toBe(3);
});

it('keeps insufficient-row residuals NaN even when the algebraic fit is exact', () => {
  const result = computeSpatialTwoStageLeastSquaresOnCPU(
    {
      predictors: Float32Array.of(-1, 0, 1),
      response: Float32Array.of(-1, 0, 1),
      weights: {
        offsets: Uint32Array.of(0, 1, 2, 3),
        neighbors: Uint32Array.of(1, 2, 0),
        weights: Float32Array.of(1, 1, 1)
      }
    },
    {predictorCount: 1}
  );
  // n=3 equals the three coefficients, so status 2. The exact IV fit beta=1, rho=0,
  // intercept=0 has residuals zero, but the CPU deliberately clears every failed-fit row.
  expect(result.status[0]).toBe(2);
  expect(Array.from(result.summary)).toEqual([3, ...new Array(15).fill(NaN)]);
  expect(result.table.every(Number.isNaN)).toBe(true);
  expect(result.residuals.every(Number.isNaN)).toBe(true);
});

for (const fixture of fixtures) {
  it(`matches independent dense SVD projector and NumPy covariance: ${fixture.name}, order ${fixture.instrumentOrder}`, () => {
    const result = computeSpatialTwoStageLeastSquaresOnCPU(getInput(fixture), {
      predictorCount: fixture.predictorCount,
      instrumentOrder: fixture.instrumentOrder as 1 | 2
    });
    expect(result.status[0]).toBe(0);
    expectRegressionRowsClose(result.table, fixture.table, 3e-6);
    expectRegressionRowsClose(
      result.summary,
      fixture.summary.map(value => (value === null ? NaN : value)),
      3e-6
    );
    expectRegressionRowsClose(result.residuals, fixture.residuals, 3e-6);
    expect(
      Math.abs(Array.from(result.residuals).reduce((sum, value) => sum + value, 0))
    ).toBeLessThan(1e-5);
    if (fixture.name === 'zero-total-weight') {
      expect(result.summary[4]).toBeNaN();
      expect(result.summary[5]).toBeCloseTo(0.9505670553, 6);
    }
  });
  for (const method of ['HC0', 'HC1', 'HAC'] as const) {
    it(`matches full ${method} sandwich, table and unchanged diagnostics: ${fixture.name}, order ${fixture.instrumentOrder}`, () => {
      const kernel = {
        offsets: Uint32Array.from(fixture.kernel.offsets),
        neighbors: Uint32Array.from(fixture.kernel.neighbors),
        weights: Float32Array.from(fixture.kernel.weights)
      };
      const result = computeSpatialTwoStageLeastSquaresOnCPU(
        {...getInput(fixture), robust: {method, kernel: method === 'HAC' ? kernel : undefined}},
        {predictorCount: fixture.predictorCount, instrumentOrder: fixture.instrumentOrder as 1 | 2}
      );
      expect(result.status[0]).toBe(0);
      expect(result.robust!.status[0]).toBe(0);
      expectRegressionRowsClose(result.robust!.covariance, fixture.robust[method].covariance, 3e-6);
      expectRegressionRowsClose(result.table, fixture.robust[method].table, 3e-6);
      expectRegressionRowsClose(
        result.summary,
        fixture.summary.map(value => (value === null ? NaN : value)),
        3e-6
      );
    });
  }
}

it('keeps high-fit residuals invariant under exactly representable response translations and scaling', () => {
  const input = getInput(fixtures[2]);
  input.weights.weights.fill(1);
  const first = computeSpatialTwoStageLeastSquaresOnCPU(input, {predictorCount: 1});
  const shifted = computeSpatialTwoStageLeastSquaresOnCPU(
    {...input, response: Float32Array.from(input.response, value => 10000 + value * 0.125)},
    {predictorCount: 1}
  );
  const scaled = computeSpatialTwoStageLeastSquaresOnCPU(
    {...input, response: Float32Array.from(input.response, value => value * 0.125)},
    {predictorCount: 1}
  );
  expect(shifted.status[0]).toBe(0);
  expectRegressionRowsClose(shifted.residuals, scaled.residuals, 1e-6);
  expect(shifted.table[8]).toBeCloseTo(first.table[8], 6);
  expectRegressionRowsClose(
    scaled.residuals,
    Array.from(first.residuals, value => value * 0.125),
    1e-6
  );
  expectRegressionRowsClose(scaled.summary.subarray(4, 7), first.summary.subarray(4, 7), 1e-6);
});

it('ignores invalid neighbor sentinels, rejects malformed CSR and uses failure layouts', () => {
  const input = getInput();
  const weights = {
    ...input.weights,
    neighbors: Uint32Array.from([...input.weights.neighbors, 0xffffffff]),
    weights: Float32Array.from([...input.weights.weights, NaN]),
    offsets: input.weights.offsets.slice()
  };
  weights.offsets[48]++;
  const original = computeSpatialTwoStageLeastSquaresOnCPU(input, {predictorCount: 2});
  const padded = computeSpatialTwoStageLeastSquaresOnCPU({...input, weights}, {predictorCount: 2});
  expectRegressionRowsClose(padded.table, original.table, 0);
  for (const change of [
    {predictors: new Float32Array(96)},
    {response: new Float32Array(48).fill(NaN)}
  ]) {
    const result = computeSpatialTwoStageLeastSquaresOnCPU(
      {...input, ...change},
      {predictorCount: 2}
    );
    expect(result.status[0]).toBe(1);
    expect(result.summary[0]).toBe(48);
    expect(result.summary.subarray(1).every(Number.isNaN)).toBe(true);
    expect(result.table.every(Number.isNaN)).toBe(true);
    expect(result.residuals.every(Number.isNaN)).toBe(true);
  }
  const insufficient = computeSpatialTwoStageLeastSquaresOnCPU(
    {
      predictors: Float32Array.of(1, 2, 3),
      response: Float32Array.of(0, 1, 2),
      weights: {
        offsets: Uint32Array.of(0, 1, 2, 3),
        neighbors: Uint32Array.of(1, 2, 0),
        weights: Float32Array.of(1, 1, 1)
      }
    },
    {predictorCount: 1}
  );
  expect(insufficient.status[0]).toBe(2);
  for (const change of [
    {predictors: new Float32Array(95)},
    {weights: {...input.weights, offsets: Uint32Array.of(0, 3, 2)}},
    {weights: {...input.weights, weights: new Float32Array(1)}}
  ]) {
    expect(() =>
      computeSpatialTwoStageLeastSquaresOnCPU({...input, ...change}, {predictorCount: 2})
    ).toThrow(/predictors|weights/);
  }
});

it('invalid HAC kernel changes only robust inference', () => {
  const fixture = fixtures[0];
  const input = getInput();
  const malformedOffsets = Uint32Array.from(fixture.kernel.offsets);
  malformedOffsets[1] = malformedOffsets[2] + 1;
  for (const change of [
    {overflow: Uint32Array.of(1)},
    {kernel: {...input.weights}},
    {
      kernel: {
        offsets: malformedOffsets,
        neighbors: Uint32Array.from(fixture.kernel.neighbors),
        weights: Float32Array.from(fixture.kernel.weights)
      }
    }
  ]) {
    const result = computeSpatialTwoStageLeastSquaresOnCPU(
      {
        ...input,
        robust: {
          method: 'HAC',
          kernel: {
            offsets: Uint32Array.from(fixture.kernel.offsets),
            neighbors: Uint32Array.from(fixture.kernel.neighbors),
            weights: Float32Array.from(fixture.kernel.weights)
          },
          ...change
        }
      },
      {predictorCount: 2}
    );
    expect(result.status[0]).toBe(0);
    expect(result.robust!.status[0]).toBe(4);
    expect(result.robust!.covariance.every(Number.isNaN)).toBe(true);
    for (let column = 0; column < 4; column++) {
      expect(result.table[column * 4]).toBeCloseTo(fixture.table[column * 4], 5);
      expect(result.table.subarray(column * 4 + 1, column * 4 + 4).every(Number.isNaN)).toBe(true);
    }
  }
});

it('validates both operation constructions and retains CPU support beyond shader caps', () => {
  const input = getInput();
  const shapes = {
    predictors: {format: 'float32' as const, length: 96},
    response: {format: 'float32' as const, length: 48},
    offsets: {format: 'uint32' as const, length: 49},
    neighbors: {format: 'uint32' as const, length: input.weights.neighbors.length},
    weights: {format: 'float32' as const, length: input.weights.weights.length}
  };
  for (const change of [{predictorCount: 0}, {instrumentOrder: 3 as 1}, {tileRowCount: 0}]) {
    expect(() => new CPUSpatialTwoStageLeastSquaresOperation({...PROPS, ...change})).toThrow(
      /iv-test/
    );
    expect(() =>
      constructRegressionOperationContributor(
        new GPUSpatialTwoStageLeastSquaresOperation({...PROPS, ...change}),
        shapes
      )
    ).toThrow(/iv-test/);
  }
  expect(
    () => new CPUSpatialTwoStageLeastSquaresOperation({...PROPS, predictorCount: 9})
  ).not.toThrow();
  for (const Operation of [
    CPUSpatialTwoStageLeastSquaresOperation,
    GPUSpatialTwoStageLeastSquaresOperation
  ])
    expect(
      () => new Operation({...PROPS, output: {...PROPS.output, residuals: 'response'}})
    ).toThrow(/iv-test/);
  expect(
    () =>
      new CPUSpatialTwoStageLeastSquaresOperation({
        ...PROPS,
        robust: {method: 'HAC', output: {covariance: 'covariance', status: 'robustStatus'}}
      })
  ).toThrow(/robust.kernel/);
});
