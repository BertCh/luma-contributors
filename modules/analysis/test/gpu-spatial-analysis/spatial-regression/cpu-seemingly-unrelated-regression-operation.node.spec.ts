// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeSeeminglyUnrelatedRegressionOnCPU,
  CPUSeeminglyUnrelatedRegressionOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-seemingly-unrelated-regression-operation';
import {
  GPUSeeminglyUnrelatedRegressionOperation,
  type SeeminglyUnrelatedRegressionOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression-operation';
import fixtures from './seemingly-unrelated-regression-operation-fixtures.json';
import spatialFixture from './seemingly-unrelated-regression-spatial-operation-fixtures.json';
import {GPUSeeminglyUnrelatedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression';
import {
  constructRegressionOperationContributor,
  expectRegressionRowsClose
} from './regression-operation-test-utils';

const PROPS: SeeminglyUnrelatedRegressionOperationProps = {
  id: 'sur-test',
  equationCount: 2,
  predictorCount: 2,
  rowCount: 40,
  predictors: 'predictors',
  responses: 'responses',
  mask: 'mask',
  initialResiduals: 'initialResiduals',
  output: {
    coefficients: 'coefficients',
    covariance: 'covariance',
    residualCovariance: 'residualCovariance',
    residuals: 'residuals',
    status: 'status'
  }
};

it('preserves spatial ports and reports self-links as spatial status 4', () => {
  const props: SeeminglyUnrelatedRegressionOperationProps = {
    ...PROPS,
    predictorCount: 1,
    rowCount: 6,
    mask: undefined,
    initialResiduals: undefined,
    spatial: {
      weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      output: {parameters: 'parameters', status: 'spatialStatus'}
    }
  };
  const result = computeSeeminglyUnrelatedRegressionOnCPU(
    {
      predictors: Float32Array.of(-2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, 3),
      responses: Float32Array.of(0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 2, 0),
      spatial: {
        weights: {
          offsets: Uint32Array.of(0, 1, 2, 3, 4, 5, 6),
          neighbors: Uint32Array.of(0, 1, 2, 3, 4, 5),
          weights: new Float32Array(6).fill(1)
        }
      }
    },
    props
  );
  // Every W=I row has a forbidden self-link: each equation has status 4 and lambda NaN.
  // Filtering each linked row with NaN excludes all six rows from the joint fit: status 1.
  expect(Array.from(result.spatial!.status)).toEqual([4, 4]);
  expect(result.spatial!.parameters.every(Number.isNaN)).toBe(true);
  expect(result.status[0]).toBe(1);
  for (const name of ['coefficients', 'covariance', 'residualCovariance', 'residuals'] as const)
    expect(result[name].every(Number.isNaN)).toBe(true);
  const operation = new GPUSeeminglyUnrelatedRegressionOperation(props);
  expect(operation.inputs).toMatchObject({
    spatialOffsets: 'offsets',
    spatialNeighbors: 'neighbors',
    spatialWeights: 'weights'
  });
  const contributor = constructRegressionOperationContributor(operation, {
    predictors: {format: 'float32', length: 12},
    responses: {format: 'float32', length: 12},
    spatialOffsets: {format: 'uint32', length: 7},
    spatialNeighbors: {format: 'uint32', length: 6},
    spatialWeights: {format: 'float32', length: 6}
  });
  expect(contributor).toBeInstanceOf(GPUSeeminglyUnrelatedRegression);
  expect((contributor as GPUSeeminglyUnrelatedRegression).props.spatial).toBeDefined();
  expect(new CPUSeeminglyUnrelatedRegressionOperation(props).contract.outputs).toEqual(
    operation.contract.outputs
  );
  for (const forbidden of [{mask: 'mask'}, {initialResiduals: 'initial'}]) {
    expect(() => new CPUSeeminglyUnrelatedRegressionOperation({...props, ...forbidden})).toThrow(
      /sur-test.*spatial/
    );
  }
});

it('matches independent SciPy spatial moments, filtered GLS and structural residuals', () => {
  const result = computeSeeminglyUnrelatedRegressionOnCPU(
    {
      predictors: Float32Array.from(spatialFixture.predictors),
      responses: Float32Array.from(spatialFixture.responses),
      spatial: {
        weights: {
          offsets: Uint32Array.from(spatialFixture.weights.offsets),
          neighbors: Uint32Array.from(spatialFixture.weights.neighbors),
          weights: Float32Array.from(spatialFixture.weights.weights)
        }
      }
    },
    spatialFixture
  );
  expect(result.status[0]).toBe(0);
  expect(Array.from(result.spatial!.status)).toEqual([0, 0]);
  expectRegressionRowsClose(result.spatial!.parameters, spatialFixture.expected.parameters, 2e-5);
  for (const name of ['coefficients', 'residualCovariance', 'residuals'] as const)
    expectRegressionRowsClose(result[name], spatialFixture.expected[name], 2e-5);
  const size = result.coefficients.length;
  result.covariance.forEach((value, index) => {
    const scale = Math.sqrt(
      spatialFixture.expected.covariance[Math.floor(index / size) * (size + 1)] *
        spatialFixture.expected.covariance[(index % size) * (size + 1)]
    );
    expect(Math.abs(value - spatialFixture.expected.covariance[index])).toBeLessThanOrEqual(
      2e-5 * scale
    );
  });
});

it('checks spatial CSR shape errors and retains the defined invalid-offset status', () => {
  const weights = {
    offsets: Uint32Array.from(spatialFixture.weights.offsets),
    neighbors: Uint32Array.from(spatialFixture.weights.neighbors),
    weights: Float32Array.from(spatialFixture.weights.weights)
  };
  const input = {
    predictors: Float32Array.from(spatialFixture.predictors),
    responses: Float32Array.from(spatialFixture.responses)
  };
  for (const [change, argument] of [
    [{offsets: weights.offsets.subarray(1)}, /spatial.weights.offsets/],
    [{weights: weights.weights.subarray(1)}, /spatial.weights.weights/],
    [{neighbors: weights.neighbors.subarray(1)}, /spatial.weights.neighbors/]
  ] as const) {
    expect(() =>
      computeSeeminglyUnrelatedRegressionOnCPU(
        {...input, spatial: {weights: {...weights, ...change}}},
        spatialFixture
      )
    ).toThrow(argument);
  }
  const offsets = weights.offsets.slice();
  offsets[2] = offsets[1] - 1;
  const invalid = computeSeeminglyUnrelatedRegressionOnCPU(
    {...input, spatial: {weights: {...weights, offsets}}},
    spatialFixture
  );
  expect(Array.from(invalid.spatial!.status)).toEqual([4, 4]);
  expect(invalid.spatial!.parameters.every(Number.isNaN)).toBe(true);
});

for (const fixture of fixtures) {
  it(`matches independent SciPy whitened SVD outputs: ${fixture.name}`, () => {
    const result = computeSeeminglyUnrelatedRegressionOnCPU(
      {
        predictors: Float32Array.from(fixture.predictors),
        responses: Float32Array.from(fixture.responses),
        mask: Uint32Array.from(fixture.mask),
        initialResiduals: fixture.initialResiduals
          ? Float32Array.from(fixture.initialResiduals)
          : undefined
      },
      fixture
    );
    expect(result.status[0]).toBe(0);
    for (const name of ['coefficients', 'residualCovariance', 'residuals'] as const)
      expectRegressionRowsClose(
        result[name],
        fixture.expected[name].map(value => (value === null ? NaN : value)),
        2e-6
      );
    const count = result.coefficients.length;
    result.covariance.forEach((value, index) => {
      const scale = Math.sqrt(
        fixture.expected.covariance[Math.floor(index / count) * (count + 1)] *
          fixture.expected.covariance[(index % count) * (count + 1)]
      );
      expect(Math.abs(value - fixture.expected.covariance[index])).toBeLessThanOrEqual(
        2e-6 * scale
      );
    });
    expect(Math.abs(result.residualCovariance[1])).toBeGreaterThan(0.05);
  });
}

it('returns the precise failure layouts, retaining a singular error covariance', () => {
  const fixture = fixtures[0];
  const input = {
    predictors: Float32Array.from(fixture.predictors),
    responses: Float32Array.from(fixture.responses)
  };
  const failedCovariance = computeSeeminglyUnrelatedRegressionOnCPU(
    {...input, initialResiduals: new Float32Array(80).fill(2)},
    fixture
  );
  expect(failedCovariance.status[0]).toBe(3);
  expect(Array.from(failedCovariance.residualCovariance)).toEqual([4, 4, 4, 4]);
  for (const name of ['coefficients', 'covariance', 'residuals'] as const)
    expect(failedCovariance[name].every(Number.isNaN)).toBe(true);
  const failedDesign = computeSeeminglyUnrelatedRegressionOnCPU(
    {...input, predictors: new Float32Array(160)},
    fixture
  );
  expect(failedDesign.status[0]).toBe(2);
  expect(failedDesign.residualCovariance.every(Number.isNaN)).toBe(true);
  const excluded = computeSeeminglyUnrelatedRegressionOnCPU(
    {...input, mask: new Uint32Array(40)},
    fixture
  );
  expect(excluded.status[0]).toBe(1);
  expect(excluded.residuals.every(Number.isNaN)).toBe(true);
});

it('excludes aligned rows if a supplied initial residual is nonfinite', () => {
  const fixture = fixtures[2];
  const initialResiduals = Float32Array.from(fixture.initialResiduals!);
  initialResiduals[41] = NaN;
  const result = computeSeeminglyUnrelatedRegressionOnCPU(
    {
      predictors: Float32Array.from(fixture.predictors),
      responses: Float32Array.from(fixture.responses),
      initialResiduals
    },
    fixture
  );
  expect(result.status[0]).toBe(0);
  expect(result.residuals[1]).toBeNaN();
  expect(result.residuals[41]).toBeNaN();
});

it('checks inconsistent arrays, both constructions and equal output contracts', () => {
  const input = {predictors: new Float32Array(160), responses: new Float32Array(80)};
  for (const change of [
    {predictors: new Float32Array(159)},
    {responses: new Float32Array(79)},
    {mask: new Uint32Array(39)},
    {initialResiduals: new Float32Array(79)}
  ])
    expect(() => computeSeeminglyUnrelatedRegressionOnCPU({...input, ...change}, PROPS)).toThrow(
      new RegExp(Object.keys(change)[0])
    );
  const shapes = {
    predictors: {format: 'float32' as const, length: 160},
    responses: {format: 'float32' as const, length: 80},
    mask: {format: 'uint32' as const, length: 40},
    initialResiduals: {format: 'float32' as const, length: 80}
  };
  for (const change of [{equationCount: 1}, {predictorCount: 0}, {rowCount: 0}]) {
    expect(() => new CPUSeeminglyUnrelatedRegressionOperation({...PROPS, ...change})).toThrow(
      /sur-test/
    );
    expect(() =>
      constructRegressionOperationContributor(
        new GPUSeeminglyUnrelatedRegressionOperation({...PROPS, ...change}),
        shapes
      )
    ).toThrow(/sur-test/);
  }
  for (const Operation of [
    CPUSeeminglyUnrelatedRegressionOperation,
    GPUSeeminglyUnrelatedRegressionOperation
  ])
    expect(
      () => new Operation({...PROPS, output: {...PROPS.output, covariance: 'responses'}})
    ).toThrow(/sur-test/);
  expect(new CPUSeeminglyUnrelatedRegressionOperation(PROPS).contract.outputs).toEqual(
    new GPUSeeminglyUnrelatedRegressionOperation(PROPS).contract.outputs
  );
  expect(() =>
    constructRegressionOperationContributor(
      new GPUSeeminglyUnrelatedRegressionOperation(PROPS),
      shapes
    )
  ).not.toThrow();
});
