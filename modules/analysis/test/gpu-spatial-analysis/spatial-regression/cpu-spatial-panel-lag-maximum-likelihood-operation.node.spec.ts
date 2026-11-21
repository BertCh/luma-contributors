// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUSpatialPanelLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood';
import {
  computeSpatialPanelLagMaximumLikelihoodOnCPU,
  CPUSpatialPanelLagMaximumLikelihoodOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-lag-maximum-likelihood-operation';
import {
  GPUSpatialPanelLagMaximumLikelihoodOperation,
  type SpatialPanelLagMaximumLikelihoodOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood-operation';
import {
  constructRegressionOperationContributor,
  expectRegressionRowsClose
} from './regression-operation-test-utils';
import fixtures from './spatial-panel-lag-maximum-likelihood-operation-fixtures.json';
import edges from './spatial-panel-lag-maximum-likelihood-operation-edge-fixtures.json';
import review from '../../oracles/fixtures/statistics-regression-panel-lag-conditioning.json';
import {
  createPanelAuditInput,
  expectPanelAuditRows,
  PANEL_AUDIT_EXPECTED
} from './panel-audit.test-utils';

it('review 2 R3 matching written/required counts retain the independent fit with spare capacity', () => {
  const input = createPanelAuditInput();
  const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(
    {
      ...input,
      weights: {
        ...input.weights,
        neighbors: Uint32Array.from([...input.weights.neighbors, 0, 0]),
        weights: Float32Array.from([...input.weights.weights, 0, 0]),
        status: {count: Uint32Array.of(8), requiredCount: Uint32Array.of(8)}
      }
    },
    {effects: 'random', periodCount: 4, predictorCount: 1, hausman: true}
  );
  expect(Array.from(result.status)).toEqual([0]);
  expect(Array.from(result.hausman!.status)).toEqual([0]);
  for (const port of ['coefficients', 'covariance', 'fitted', 'residuals'] as const)
    expectPanelAuditRows(result[port], PANEL_AUDIT_EXPECTED.random[port], 2e-6);
  expectPanelAuditRows(result.summary.slice(0, 2), PANEL_AUDIT_EXPECTED.random.summary, 2e-6);
  expectPanelAuditRows(result.hausman!.tests, PANEL_AUDIT_EXPECTED.random.hausman, 2e-6);
});

for (const countPresent of [false, true, 'spare', 'inflated'] as const)
  it(`R3 refuses required edge loss without overflow, written count present=${countPresent}`, () => {
    const input = createPanelAuditInput();
    if (countPresent === 'spare' || countPresent === 'inflated') {
      input.weights.neighbors = Uint32Array.from([...input.weights.neighbors, 0, 0]);
      input.weights.weights = Float32Array.from([...input.weights.weights, 0, 0]);
    }
    const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(
      {
        ...input,
        weights: {
          ...input.weights,
          status: {
            requiredCount: Uint32Array.of(9),
            ...(countPresent ? {count: Uint32Array.of(countPresent === 'inflated' ? 9 : 8)} : {})
          }
        }
      },
      {periodCount: 4, predictorCount: 1, effects: 'random', hausman: true}
    );
    expect(
      result.status[0],
      JSON.stringify({rho: result.coefficients[2], hausman: result.hausman?.status[0]})
    ).toBe(1);
    expect(result.hausman!.status[0]).toBe(1);
    for (const values of [
      result.coefficients,
      result.covariance,
      result.summary,
      result.fitted,
      result.residuals,
      result.hausman!.tests
    ])
      expect(Array.from(values).every(Number.isNaN)).toBe(true);
  });

function getInput(fixture = fixtures[0]) {
  return {
    weights: {
      offsets: Uint32Array.from(fixture.weights.offsets),
      neighbors: Uint32Array.from(fixture.weights.neighbors),
      weights: Float32Array.from(fixture.weights.weights)
    },
    predictors: Float32Array.from(fixture.predictors),
    response: Float32Array.from(fixture.response)
  };
}

for (const scene of edges.reviewCases) {
  it(`pins the hand-derived review result: ${scene.name}`, () => {
    const input = {
      weights: {
        offsets: Uint32Array.from(scene.weights.offsets),
        neighbors: Uint32Array.from(scene.weights.neighbors),
        weights: Float32Array.from(scene.weights.weights)
      },
      predictors: Float32Array.from(scene.predictors),
      response: Float32Array.from(scene.response)
    };
    const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(input, {...scene, effects: 'unit'});
    if (scene.name === 'normalizationBoundary') {
      // The exact sum is inside tolerance, but serial f32 addition gives 1+84*2^-23.
      expect(335 * 2 ** -25).toBeLessThan(1e-5);
      expect(1 + 84 * 2 ** -23 - 1).toBeGreaterThan(Math.fround(1e-5));
      expect(result.status[0]).toBe(1);
      for (const port of ['coefficients', 'summary'] as const)
        expect(result[port].every(Number.isNaN)).toBe(true);
    } else {
      // Paired weights: e1=0, rho=0 and variance=scale²/2. At scales 2^70 and
      // 2^-90 this rounds respectively to Infinity and zero, both status 2.
      // The other scenes have exact solutions [1,0,0] and [1,0.5], hence zero RSS.
      expect(result.status[0]).toBe(2);
      const expectedVariance = scene.name === 'varianceOverflow' ? Infinity : 0;
      if (scene.name === 'varianceOverflow' || scene.name === 'varianceUnderflow')
        expect(result.summary[0] === expectedVariance).toBe(true);
      if (scene.name === 'varianceOverflow' || scene.name === 'varianceUnderflow') {
        const scale = scene.name === 'varianceOverflow' ? 2 ** 70 : 2 ** -90;
        const expectedLikelihood = Math.fround(
          -4 * Math.log(scale ** 2 / 2) - 4 * (1 + Math.log(2 * Math.PI)) - 4 * Math.log(8)
        );
        expect(result.summary[1]).toBe(expectedLikelihood);
        expect(Math.abs(result.coefficients[1])).toBeLessThan(1e-6 * scale);
        expect(Math.abs(result.coefficients[2])).toBeLessThan(1e-7);
      }
      if (scene.name === 'responseCopy' || scene.name === 'originalPole')
        for (const port of ['coefficients', 'summary'] as const)
          expect(result[port].every(Number.isNaN)).toBe(true);
    }
    for (const port of ['covariance', 'fitted', 'residuals'] as const)
      expect(result[port].every(Number.isNaN)).toBe(true);
  });
}

for (const fixture of fixtures) {
  it(`matches independent nested SciPy likelihood/SVD/information: ${fixture.name}`, () => {
    const input = getInput(fixture);
    const unit = fixture.effects === 'unit';
    const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(input, {
      ...fixture,
      effects: fixture.effects as 'unit' | 'random',
      tolerance: 1e-11,
      hausman: !unit
    });
    expect(result.status[0]).toBe(0);
    for (const port of ['coefficients', 'covariance', 'fitted', 'residuals'] as const) {
      expectRegressionRowsClose(
        result[port],
        Float32Array.from(fixture.expected[port], value => value ?? NaN),
        3e-5
      );
    }
    expectRegressionRowsClose(
      result.summary.subarray(0, 2),
      [fixture.expected.variance, fixture.expected.logLikelihood],
      3e-6
    );
    expect(Number.isInteger(result.summary[2])).toBe(true);
    expect(result.summary[2]).toBeGreaterThanOrEqual(1);
    expect(result.summary[3]).toBeLessThanOrEqual(1e-11);
    expect(result.summary[4]).toBeLessThan(1e-13);
    expect(result.summary[5]).toBe(result.coefficients.at(-1) === 1 ? 2 : 0);
    const phi = result.coefficients.at(-1)!;
    // Independent publication invariant, in original uncentered response coordinates.
    for (let row = 0; row < input.response.length; row++) {
      const entity = row % fixture.entityCount;
      let mean = 0;
      for (let period = 0; period < fixture.periodCount; period++)
        mean += input.response[period * fixture.entityCount + entity] / fixture.periodCount;
      const expected = input.response[row] - (1 - phi) * mean;
      expect(Math.abs(result.fitted[row] + result.residuals[row] - expected)).toBeLessThan(1e-6);
    }
    if (fixture.expected.hausman) {
      expect(result.hausman!.status[0]).toBe(fixture.expected.hausman.status);
      expectRegressionRowsClose(
        result.hausman!.tests,
        Float32Array.from(fixture.expected.hausman.tests, value => value ?? NaN),
        3e-5
      );
    }
  });
}

it('retains centred prediction and covariance under large exactly representable shifts', () => {
  const fixture = fixtures[0];
  const input = getInput();
  input.predictors = Float32Array.from(input.predictors, value => 2 * Math.round(value));
  input.response = Float32Array.from(input.response, value => 2 * Math.round(value));
  const options = {
    periodCount: fixture.periodCount,
    predictorCount: fixture.predictorCount,
    tolerance: 1e-11
  };
  const first = computeSpatialPanelLagMaximumLikelihoodOnCPU(input, options);
  const shift = 2 ** 24;
  const second = computeSpatialPanelLagMaximumLikelihoodOnCPU(
    {
      ...input,
      predictors: Float32Array.from(input.predictors, value => value + shift),
      response: Float32Array.from(input.response, value => value + shift)
    },
    options
  );
  expect(first.status[0]).toBe(0);
  expect(second.status[0]).toBe(0);
  expectRegressionRowsClose(second.coefficients.subarray(1), first.coefficients.subarray(1), 1e-6);
  expectRegressionRowsClose(second.residuals, first.residuals, 3e-6);
  expectRegressionRowsClose(second.summary.subarray(0, 2), first.summary.subarray(0, 2), 1e-6);
  const width = fixture.predictorCount + 2;
  const transform = Array.from({length: width}, (_, row) =>
    Array.from({length: width}, (_, column) =>
      row === 0 ? (column === 0 ? 1 : -shift) : Number(row === column)
    )
  );
  for (let row = 0; row < width; row++)
    for (let column = 0; column < width; column++) {
      let expected = 0;
      for (let firstIndex = 0; firstIndex < width; firstIndex++)
        for (let secondIndex = 0; secondIndex < width; secondIndex++)
          expected +=
            transform[row][firstIndex] *
            first.covariance[firstIndex * width + secondIndex] *
            transform[column][secondIndex];
      expect(Math.abs(second.covariance[row * width + column] - expected)).toBeLessThanOrEqual(
        2e-6 *
          Math.sqrt(
            second.covariance[row * width + row] * second.covariance[column * width + column]
          )
      );
    }
});

it('has the closed-form zero-lag-residual maximum at rho=0', () => {
  // Reciprocal pairs; x=Wy and y orthogonal to [1,x]. Repeat with time signs, so
  // entity demeaning is the identity. RSS=8 and det(I-rho W)=(1-rho²)^2.
  const response = Float32Array.of(-1, 1, 0, 0, 1, -1, 0, 0);
  const input = {
    weights: {
      offsets: Uint32Array.of(0, 1, 2, 3, 4),
      neighbors: Uint32Array.of(2, 3, 0, 1),
      weights: Float32Array.of(1, 1, 1, 1)
    },
    response,
    predictors: Float32Array.of(0, 0, -1, 1, 0, 0, 1, -1)
  };
  const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(input, {
    periodCount: 2,
    predictorCount: 1,
    effects: 'unit'
  });
  expect(result.status[0]).toBe(0);
  expect(result.coefficients[0]).toBeNaN();
  expect(Math.abs(result.coefficients[1])).toBeLessThan(1e-7);
  expect(Math.abs(result.coefficients[2])).toBeLessThan(1e-7);
  expect(result.coefficients[3]).toBe(0);
  expect(result.summary[0]).toBe(0.5);
  for (let row = 0; row < response.length; row++)
    expect(Math.abs(result.residuals[row] - response[row])).toBeLessThan(1e-7);
  expect(result.covariance[0]).toBeNaN();
  expect(result.covariance[4]).toBe(0.125);
  // Information: T*(tr(W²)+tr(WW')) + ||Wx beta||²/sigma² = 16.
  expect(result.covariance[8]).toBe(0.0625);
});

it('publishes the defined iterate when the within design becomes singular', () => {
  const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(
    {
      weights: {
        offsets: Uint32Array.of(0, 1, 2, 3, 4),
        neighbors: Uint32Array.of(1, 2, 3, 0),
        weights: Float32Array.of(1, 1, 1, 1)
      },
      predictors: Float32Array.of(-1, 1, 2, -2, -1, 1, 2, -2),
      response: Float32Array.of(-1, 1, 2, -2, 1, -1, -2, 2)
    },
    {periodCount: 2, predictorCount: 1, effects: 'unit'}
  );
  expect(result.status[0]).toBe(2);
  expect(result.coefficients[0]).toBeNaN();
  expect(Math.abs(result.coefficients[1])).toBeLessThan(1e-15);
  expect(result.coefficients[2]).toBe(Math.fround(0.1));
  expect(result.coefficients[3]).toBe(0);
  expectRegressionRowsClose(result.summary, [0, Math.fround(-3.402823e38), 0, 0, 2, 0], 0);
  expect(result.covariance.every(Number.isNaN)).toBe(true);
  expect(result.fitted.every(Number.isNaN)).toBe(true);
  expect(result.residuals.every(Number.isNaN)).toBe(true);
});

it('does not call an exact zero-variance likelihood pole a finite maximum', () => {
  // y=x+0.5 Wy exactly, two periods with opposing signs. The augmented regression has
  // coefficients [1,0.5] and exactly zero RSS. No finite Gaussian ML maximum exists.
  const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(
    {
      weights: {
        offsets: Uint32Array.of(0, 1, 2, 3, 4),
        neighbors: Uint32Array.of(1, 2, 3, 0),
        weights: Float32Array.of(1, 1, 1, 1)
      },
      predictors: Float32Array.of(-1.5, 0, 3, -1.5, 1.5, 0, -3, 1.5),
      response: Float32Array.of(-1, 1, 2, -2, 1, -1, -2, 2)
    },
    {periodCount: 2, predictorCount: 1, effects: 'unit'}
  );
  expect(result.status[0]).toBe(2);
  for (const port of ['coefficients', 'covariance', 'summary', 'fitted', 'residuals'] as const)
    expect(result[port].every(Number.isNaN)).toBe(true);
});

it('follows a sub-1e-8 phi as a constrained fit with status 3, finite iterate and no covariance', () => {
  const scene = edges.belowLimit;
  const input = {
    weights: {
      offsets: Uint32Array.from(scene.weights.offsets),
      neighbors: Uint32Array.from(scene.weights.neighbors),
      weights: Float32Array.from(scene.weights.weights)
    },
    predictors: Float32Array.from(scene.predictors),
    response: Float32Array.from(scene.response)
  };
  const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(input, {
    periodCount: scene.periodCount,
    predictorCount: 1,
    tolerance: 1e-8,
    hausman: true
  });
  expect(result.status[0]).toBe(3);
  expect(result.coefficients.at(-1)).toBe(Math.fround(1e-8));
  expect(result.summary[5] & 6).toBe(6);
  expect(result.covariance.every(Number.isNaN)).toBe(true);
  expect(result.fitted.every(Number.isFinite)).toBe(true);
  expect(result.residuals.every(Number.isFinite)).toBe(true);
  expect(result.hausman!.status[0]).toBe(1);
  expect(result.hausman!.tests.every(Number.isNaN)).toBe(true);
});

it('matches committed spreg estimates and standard errors on a high-fit review scene', () => {
  const scene = edges.highFit;
  const input = {
    weights: {
      offsets: Uint32Array.from(scene.weights.offsets),
      neighbors: Uint32Array.from(scene.weights.neighbors),
      weights: Float32Array.from(scene.weights.weights)
    },
    predictors: Float32Array.from(scene.predictors),
    response: Float32Array.from(scene.response)
  };
  for (const effects of ['unit', 'random'] as const) {
    const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(input, {
      periodCount: scene.periodCount,
      predictorCount: 1,
      effects,
      tolerance: 1e-11
    });
    const expected = review.scenes[3][effects];
    expect(result.status[0]).toBe(0);
    expectRegressionRowsClose(
      result.coefficients.subarray(1, 3),
      [expected.slope, expected.parameter],
      2e-7
    );
    expectRegressionRowsClose(
      result.summary.subarray(0, 2),
      [expected.sigmaSquared, expected.logLikelihood],
      2e-5
    );
    const indices = effects === 'unit' ? [1, 2] : [0, 1, 2];
    expectRegressionRowsClose(
      indices.map(index => Math.sqrt(result.covariance[index * 3 + index])),
      expected.standardErrors,
      2e-5
    );
  }
});

it('ignores unused non-finite slot capacity without changing the fit', () => {
  const fixture = fixtures[0];
  const input = getInput();
  const options = {periodCount: fixture.periodCount, predictorCount: fixture.predictorCount};
  const first = computeSpatialPanelLagMaximumLikelihoodOnCPU(input, options);
  const second = computeSpatialPanelLagMaximumLikelihoodOnCPU(
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
  for (const port of [
    'coefficients',
    'covariance',
    'summary',
    'fitted',
    'residuals',
    'status'
  ] as const)
    expectRegressionRowsClose(second[port], first[port], 0);
  expect(second.status[0]).toBe(0);
});

it('honors an explicit outer iteration budget and retains its nonconverged iterate (R2)', () => {
  const fixture = fixtures[0];
  const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(getInput(), {
    periodCount: fixture.periodCount,
    predictorCount: fixture.predictorCount,
    maximumIterations: 1
  });
  expect(result.status[0]).toBe(3);
  expect(result.summary[2]).toBe(1);
  expect(result.covariance.every(Number.isNaN)).toBe(true);
  expect(result.fitted.every(Number.isFinite)).toBe(true);
});

it('returns defined failure layouts for incomplete weights and singular designs', () => {
  const fixture = fixtures[0];
  const input = getInput();
  const options = {
    periodCount: fixture.periodCount,
    predictorCount: fixture.predictorCount,
    hausman: true
  };
  const invalidCases = [
    {...input, weights: {...input.weights, status: {overflow: Uint32Array.of(1)}}},
    {
      ...input,
      response: Float32Array.from(input.response, (value, row) => (row === 0 ? NaN : value))
    },
    {
      ...input,
      weights: {
        ...input.weights,
        offsets: Uint32Array.from(input.weights.offsets, (value, row) => (row === 2 ? 999 : value))
      }
    },
    {
      ...input,
      weights: {
        ...input.weights,
        neighbors: Uint32Array.from(input.weights.neighbors, (value, slot) =>
          slot === 0 ? 999 : value
        )
      }
    },
    {
      ...input,
      weights: {
        ...input.weights,
        weights: Float32Array.from(input.weights.weights, (value, slot) =>
          slot === 0 ? -1 : value
        )
      }
    },
    {...input, weights: {...input.weights, weights: new Float32Array(input.weights.weights.length)}}
  ];
  for (const current of invalidCases) {
    const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(current, options);
    expect(result.status[0]).toBe(1);
    for (const port of ['coefficients', 'covariance', 'summary', 'fitted', 'residuals'] as const)
      expect(result[port].every(Number.isNaN)).toBe(true);
    expect(result.hausman!.status[0]).toBe(1);
  }
  const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(
    {...input, predictors: new Float32Array(input.predictors.length)},
    options
  );
  expect(result.status[0]).toBe(2);
  expect(result.coefficients.every(Number.isNaN)).toBe(true);
});

it('throws argument-naming errors for inconsistent arrays', () => {
  const fixture = fixtures[0];
  const input = getInput();
  for (const changes of [
    {predictors: new Float32Array(1)},
    {response: new Float32Array(1)},
    {weights: {...input.weights, weights: new Float32Array(1)}},
    {weights: {...input.weights, status: {overflow: new Uint32Array(0)}}}
  ])
    expect(() =>
      computeSpatialPanelLagMaximumLikelihoodOnCPU(
        {...input, ...changes},
        {periodCount: fixture.periodCount, predictorCount: fixture.predictorCount}
      )
    ).toThrow(new RegExp(Object.keys(changes)[0]));
});

it('validates CPU construction and GPU contributor construction and shares graph contracts', () => {
  const fixture = fixtures[0];
  const props: SpatialPanelLagMaximumLikelihoodOperationProps = {
    id: 'panel-test',
    entityCount: fixture.entityCount,
    periodCount: fixture.periodCount,
    predictorCount: fixture.predictorCount,
    weights: {
      offsets: 'offsets',
      neighbors: 'neighbors',
      weights: 'weights',
      status: {overflow: 'overflow'}
    },
    predictors: 'predictors',
    response: 'response',
    output: {
      coefficients: 'coefficients',
      covariance: 'covariance',
      summary: 'summary',
      status: 'status',
      fitted: 'fitted',
      residuals: 'residuals'
    },
    hausman: {tests: 'tests', status: 'testStatus'}
  };
  const shapes = {
    offsets: {format: 'uint32' as const, length: fixture.entityCount + 1},
    neighbors: {format: 'uint32' as const, length: fixture.weights.neighbors.length},
    weights: {format: 'float32' as const, length: fixture.weights.weights.length},
    weightsStatus_overflow: {format: 'uint32' as const, length: 1},
    predictors: {format: 'float32' as const, length: fixture.predictors.length},
    response: {format: 'float32' as const, length: fixture.response.length}
  };
  for (const changes of [
    {periodCount: 1},
    {predictorCount: 0},
    {maximumIterations: 0},
    {maximumSearchIterations: 0},
    {tolerance: 0},
    {tolerance: NaN},
    {effects: 'wrong' as never},
    {effects: 'unit' as const}
  ]) {
    expect(() => new CPUSpatialPanelLagMaximumLikelihoodOperation({...props, ...changes})).toThrow(
      /panel-test/
    );
    expect(() =>
      constructRegressionOperationContributor(
        new GPUSpatialPanelLagMaximumLikelihoodOperation({...props, ...changes}),
        shapes
      )
    ).toThrow(/panel-test/);
  }
  expect(
    () =>
      new CPUSpatialPanelLagMaximumLikelihoodOperation({
        ...props,
        entityCount: 40,
        periodCount: 40,
        predictorCount: 8,
        maximumIterations: 129,
        maximumSearchIterations: 65
      })
  ).not.toThrow();
  for (const Operation of [
    CPUSpatialPanelLagMaximumLikelihoodOperation,
    GPUSpatialPanelLagMaximumLikelihoodOperation
  ])
    expect(() => new Operation({...props, output: {...props.output, fitted: 'response'}})).toThrow(
      /panel-test/
    );
  expect(new CPUSpatialPanelLagMaximumLikelihoodOperation(props).contract.outputs).toEqual(
    new GPUSpatialPanelLagMaximumLikelihoodOperation(props).contract.outputs
  );
  expect(() =>
    constructRegressionOperationContributor(
      new GPUSpatialPanelLagMaximumLikelihoodOperation(props),
      shapes
    )
  ).not.toThrow();
  expect(() =>
    constructRegressionOperationContributor(
      new GPUSpatialPanelLagMaximumLikelihoodOperation({
        ...props,
        entityCount: props.entityCount + 1
      }),
      shapes
    )
  ).toThrow(/panel-test/);
});

it('forwards optional weights views and metadata and preserves their validation', async () => {
  const fixture = fixtures[0];
  const input = getInput();
  const information = {rowCount: fixture.entityCount, transform: 'row' as const, symmetric: false};
  const spatialContext = {coordinateSpace: 'planar' as const, metric: 'native' as const};
  const props: SpatialPanelLagMaximumLikelihoodOperationProps = {
    id: 'panel-optional-weights',
    entityCount: fixture.entityCount,
    periodCount: fixture.periodCount,
    predictorCount: fixture.predictorCount,
    predictors: 'predictors',
    response: 'response',
    weights: {
      offsets: 'offsets',
      neighbors: 'neighbors',
      weights: 'weights',
      distances: 'distances',
      information,
      spatialContext,
      status: {
        overflow: 'overflow',
        requiredCount: 'requiredCount',
        candidateOverflow: 'candidateOverflow',
        approximation: 'approximation'
      }
    },
    output: {
      coefficients: 'coefficients',
      covariance: 'covariance',
      summary: 'summary',
      status: 'status',
      fitted: 'fitted',
      residuals: 'residuals'
    }
  };
  const shapes = {
    offsets: {format: 'uint32' as const, length: fixture.entityCount + 1},
    neighbors: {format: 'uint32' as const, length: fixture.weights.neighbors.length},
    weights: {format: 'float32' as const, length: fixture.weights.weights.length},
    distances: {format: 'float32' as const, length: fixture.weights.neighbors.length},
    weightsStatus_overflow: {format: 'uint32' as const, length: 1},
    weightsStatus_requiredCount: {format: 'uint32' as const, length: 1},
    weightsStatus_candidateOverflow: {format: 'uint32' as const, length: 1},
    weightsStatus_approximation: {format: 'uint32' as const, length: 1},
    predictors: {format: 'float32' as const, length: fixture.predictors.length},
    response: {format: 'float32' as const, length: fixture.response.length}
  };
  const contributor = constructRegressionOperationContributor(
    new GPUSpatialPanelLagMaximumLikelihoodOperation(props),
    shapes
  );
  if (!(contributor instanceof GPUSpatialPanelLagMaximumLikelihood))
    throw new Error('expected panel lag contributor');
  expect(contributor.props.weights.distances?.length).toBe(fixture.weights.neighbors.length);
  expect(contributor.props.weights.information).toEqual(information);
  expect(contributor.props.weights.spatialContext).toEqual(spatialContext);
  expect(Object.keys(contributor.props.weights.status!).sort()).toEqual(
    ['overflow', 'requiredCount', 'candidateOverflow', 'approximation'].sort()
  );
  expect(contributor.props.weights.status!.overflow!.length).toBe(1);
  // Status is a scalar contract; forwarding must not hide a wrongly sized port.
  expect(() =>
    constructRegressionOperationContributor(
      new GPUSpatialPanelLagMaximumLikelihoodOperation(props),
      {...shapes, weightsStatus_overflow: {...shapes.weightsStatus_overflow, length: 2}}
    )
  ).toThrow(/overflow must contain one uint32 row/);
  const operation = new CPUSpatialPanelLagMaximumLikelihoodOperation(props);
  expect(operation.inputs).toEqual(new GPUSpatialPanelLagMaximumLikelihoodOperation(props).inputs);
  // A one-element distances view cannot represent every neighbor slot.
  expect(() =>
    constructRegressionOperationContributor(
      new GPUSpatialPanelLagMaximumLikelihoodOperation(props),
      {...shapes, distances: {...shapes.distances, length: 1}}
    )
  ).toThrow(/weights.distances length must equal neighbors length/);
  expect(() =>
    constructRegressionOperationContributor(
      new GPUSpatialPanelLagMaximumLikelihoodOperation(props),
      {...shapes, weightsStatus_requiredCount: {...shapes.weightsStatus_requiredCount, length: 0}}
    )
  ).toThrow(/requiredCount/);
  const rows = {
    offsets: input.weights.offsets,
    neighbors: input.weights.neighbors,
    weights: input.weights.weights,
    predictors: input.predictors,
    response: input.response,
    distances: new Float32Array(1),
    weightsStatus_overflow: Uint32Array.of(0),
    weightsStatus_requiredCount: Uint32Array.of(input.weights.neighbors.length),
    weightsStatus_candidateOverflow: Uint32Array.of(0),
    weightsStatus_approximation: Uint32Array.of(1)
  };
  const operationInputs = Object.fromEntries(
    Object.entries(rows).map(([port, values]) => [
      port,
      {
        format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
        length: values.length,
        chunks: [{values, length: values.length}],
        pack: () => values
      }
    ])
  );
  await expect(async () => operation.execute({inputs: operationInputs})).rejects.toThrow(
    /weights.distances length/
  );
  const small = edges.reviewCases[0];
  expect(() =>
    constructRegressionOperationContributor(
      new GPUSpatialPanelLagMaximumLikelihoodOperation({
        ...props,
        entityCount: 4,
        periodCount: 2,
        predictorCount: 1,
        weights: {...props.weights, information: {...information, rowCount: 4}}
      }),
      {
        ...shapes,
        offsets: {format: 'uint32', length: 5},
        neighbors: {format: 'uint32', length: 4},
        weights: {format: 'float32', length: 4},
        distances: {format: 'float32', length: 1},
        predictors: {format: 'float32', length: small.predictors.length},
        response: {format: 'float32', length: small.response.length}
      }
    )
  ).toThrow(/weights.distances length must equal neighbors length/);
  const expected = computeSpatialPanelLagMaximumLikelihoodOnCPU(input, {
    periodCount: fixture.periodCount,
    predictorCount: fixture.predictorCount
  });
  const optionalInput = {
    ...input,
    weights: {
      ...input.weights,
      information,
      distances: new Float32Array(input.weights.neighbors.length),
      status: {
        overflow: Uint32Array.of(0, 1),
        requiredCount: Uint32Array.of(input.weights.neighbors.length),
        candidateOverflow: Uint32Array.of(0),
        approximation: Uint32Array.of(1)
      }
    }
  };
  const actual = computeSpatialPanelLagMaximumLikelihoodOnCPU(optionalInput, {
    periodCount: fixture.periodCount,
    predictorCount: fixture.predictorCount
  });
  const distances = optionalInput.weights.distances;
  const executed = await operation.execute({
    inputs: {
      ...operationInputs,
      distances: {
        format: 'float32',
        length: distances.length,
        chunks: [{values: distances, length: distances.length}],
        pack: () => distances
      }
    }
  });
  for (const port of [
    'coefficients',
    'covariance',
    'summary',
    'status',
    'fitted',
    'residuals'
  ] as const) {
    expectRegressionRowsClose(actual[port], expected[port], 0);
    expectRegressionRowsClose(executed[port] as Float32Array | Uint32Array, expected[port], 0);
  }
  for (const changes of [
    {rowCount: fixture.entityCount + 1},
    {transform: 'bad' as never},
    {symmetric: 'bad' as never}
  ]) {
    const weights = {...props.weights, information: {...information, ...changes}};
    expect(() => new CPUSpatialPanelLagMaximumLikelihoodOperation({...props, weights})).toThrow(
      /panel-optional-weights weights.information/
    );
    expect(() =>
      constructRegressionOperationContributor(
        new GPUSpatialPanelLagMaximumLikelihoodOperation({...props, weights}),
        shapes
      )
    ).toThrow(/panel-optional-weights weights.information/);
    expect(() =>
      computeSpatialPanelLagMaximumLikelihoodOnCPU(
        {
          ...optionalInput,
          weights: {...optionalInput.weights, information: {...information, ...changes}}
        },
        {periodCount: fixture.periodCount, predictorCount: fixture.predictorCount}
      )
    ).toThrow(/weights.information/);
  }
  expect(() =>
    computeSpatialPanelLagMaximumLikelihoodOnCPU(
      {
        ...optionalInput,
        weights: {...optionalInput.weights, status: {requiredCount: new Uint32Array(0)}}
      },
      {periodCount: fixture.periodCount, predictorCount: fixture.predictorCount}
    )
  ).toThrow(/weights.status.requiredCount/);
});
