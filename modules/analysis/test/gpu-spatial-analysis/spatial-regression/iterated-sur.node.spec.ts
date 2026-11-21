// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeSeeminglyUnrelatedRegressionOnCPU,
  CPUSeeminglyUnrelatedRegressionOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-seemingly-unrelated-regression-operation';
import {GPUSeeminglyUnrelatedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression-operation';
import {GPUSeeminglyUnrelatedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression';
import {constructRegressionOperationContributor} from './regression-operation-test-utils';
import fixtures from './iterated-sur-fixtures.json';
import {expectIteratedSURResult, SUR_OVERFLOW_EXPECTED} from './iterated-sur-test-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

it('refreshes zero-response seeded residuals and reports singular covariance after one completed fit', () => {
  const errors = new Float32Array(16);
  errors[0] = 1;
  errors[9] = 1;
  const result = computeSeeminglyUnrelatedRegressionOnCPU(
    {
      predictors: Float32Array.from({length: 16}, (_, index) => (index % 2 ? 1 : -1)),
      responses: new Float32Array(16),
      initialResiduals: errors
    },
    {
      equationCount: 2,
      predictorCount: 1,
      rowCount: 8,
      iteration: {maximumIterations: 2, tolerance: 1e-100}
    }
  );
  expect(Array.from(result.status)).toEqual([3]);
  expect(Array.from(result.iterationCount)).toEqual([1]);
  for (const name of ['coefficients', 'covariance', 'residualCovariance', 'residuals'] as const)
    expect(result[name].every(Number.isNaN), name).toBe(true);
});

it('rejects GPU tolerances that become zero or subnormal instead of silently exhausting the budget', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const tolerance of [1e-100, Number.MIN_VALUE, 1e-40]) {
      const operation = new GPUSeeminglyUnrelatedRegressionOperation({
        equationCount: 2,
        predictorCount: 1,
        rowCount: 8,
        predictors: 'predictors',
        responses: 'responses',
        iteration: {maximumIterations: 2, tolerance},
        output: {
          coefficients: 'coefficients',
          covariance: 'covariance',
          residualCovariance: 'residualCovariance',
          residuals: 'residuals',
          status: 'status'
        }
      });
      expect(() =>
        constructRegressionOperationContributor(operation, {
          predictors: {format: 'float32', length: 16},
          responses: {format: 'float32', length: 16}
        })
      ).toThrow(/iteration.tolerance.*normal.*float32.*CPU/);
    }
  } finally {
    device.destroy();
  }
});

for (const iterated of [false, true]) {
  it(`matches the finite large-covariance closed form in both CPU forms: ${iterated ? 'iterated' : 'default'}`, async () => {
    const input = {
      predictors: Float32Array.from({length: 16}, (_, index) => (index % 2 ? 1 : -1)),
      responses: Float32Array.from({length: 16}, (_, index) => (index % 2 ? 3 : 1)),
      initialResiduals: new Float32Array(16)
    };
    input.initialResiduals[0] = 2 ** 64;
    input.initialResiduals[9] = 2 ** 64;
    const options = {
      equationCount: 2,
      predictorCount: 1,
      rowCount: 8,
      iteration: iterated ? {maximumIterations: 1} : undefined
    };
    expectIteratedSURResult(
      computeSeeminglyUnrelatedRegressionOnCPU(input, options),
      {...SUR_OVERFLOW_EXPECTED, status: [iterated ? 4 : 0]},
      2e-6
    );
    const device = createNullWebGPUDevice();
    try {
      const result = await runOperationGraph(device, {
        inputs: {
          predictors: {format: 'float32', values: input.predictors},
          responses: {format: 'float32', values: input.responses},
          initial: {format: 'float32', values: input.initialResiduals}
        },
        operations: [
          new CPUSeeminglyUnrelatedRegressionOperation({
            ...options,
            predictors: 'predictors',
            responses: 'responses',
            initialResiduals: 'initial',
            output: {
              coefficients: 'coefficients',
              covariance: 'covariance',
              residualCovariance: 'residualCovariance',
              residuals: 'residuals',
              status: 'status',
              iteration: {status: {iterationCount: 'iterationCount'}}
            }
          })
        ],
        outputs: [
          'coefficients',
          'covariance',
          'residualCovariance',
          'residuals',
          'status',
          'iterationCount'
        ]
      });
      expectIteratedSURResult(result, {...SUR_OVERFLOW_EXPECTED, status: [iterated ? 4 : 0]}, 2e-6);
    } finally {
      device.destroy();
    }
  });
}

for (const fixture of fixtures) {
  it(`executes the CPU operation against independent NumPy ${fixture.name} budget-2 outputs without a GPU`, async () => {
    const device = createNullWebGPUDevice();
    try {
      const result = await runOperationGraph(device, {
        inputs: {
          predictors: {format: 'float32', values: Float32Array.from(fixture.predictors)},
          responses: {format: 'float32', values: Float32Array.from(fixture.responses)},
          ...(fixture.weights
            ? {
                offsets: {
                  format: 'uint32' as const,
                  values: Uint32Array.from(fixture.weights.offsets)
                },
                neighbors: {
                  format: 'uint32' as const,
                  values: Uint32Array.from(fixture.weights.neighbors)
                },
                weights: {
                  format: 'float32' as const,
                  values: Float32Array.from(fixture.weights.weights)
                }
              }
            : {mask: {format: 'uint32' as const, values: Uint32Array.from(fixture.mask)}})
        },
        operations: [
          new CPUSeeminglyUnrelatedRegressionOperation({
            equationCount: fixture.equationCount,
            predictorCount: 2,
            rowCount: 40,
            predictors: 'predictors',
            responses: 'responses',
            mask: fixture.weights ? undefined : 'mask',
            iteration: {maximumIterations: 2},
            spatial: fixture.weights
              ? {
                  weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
                  output: {parameters: 'parameters', status: 'spatialStatus'}
                }
              : undefined,
            output: {
              coefficients: 'coefficients',
              covariance: 'covariance',
              residualCovariance: 'residualCovariance',
              residuals: 'residuals',
              status: 'status',
              iteration: {status: {iterationCount: 'iterationCount'}}
            }
          })
        ],
        outputs: [
          'coefficients',
          'covariance',
          'residualCovariance',
          'residuals',
          'status',
          'iterationCount',
          ...(fixture.weights ? ['parameters', 'spatialStatus'] : [])
        ]
      });
      expectIteratedSURResult(result, fixture.rounds[1].expected, 2e-6);
      if (fixture.parameters) {
        expect(Array.from(result['spatialStatus'])).toEqual([0, 0]);
        fixture.parameters.forEach((value, index) => {
          expect(Number.isFinite(result['parameters'][index])).toBe(true);
          expect(Math.abs(result['parameters'][index] - value)).toBeLessThan(2e-6);
        });
      }
    } finally {
      device.destroy();
    }
  });
}

for (const maximumIterations of [undefined, 2]) {
  it(`suppresses the joint spatial fit when every row is an island: ${maximumIterations ?? 'default'}`, () => {
    // W=0 has no row-standardized rows. Every GM parameter is invalid. Filtering
    // cannot pass through unlinked rows and certify a joint fit from invalid parameters.
    const result = computeSeeminglyUnrelatedRegressionOnCPU(
      {
        predictors: Float32Array.of(-1, 0, 1, 0, 1, 0),
        responses: Float32Array.of(0, 1, 0, 0, 0, 1),
        spatial: {
          weights: {
            offsets: new Uint32Array(4),
            neighbors: Uint32Array.of(0),
            weights: Float32Array.of(0)
          }
        }
      },
      {
        rowCount: 3,
        equationCount: 2,
        predictorCount: 1,
        iteration: maximumIterations ? {maximumIterations} : undefined
      }
    );
    expect(Array.from(result.spatial!.status)).toEqual([4, 4]);
    expect(result.spatial!.parameters.every(Number.isNaN)).toBe(true);
    expect(Array.from(result.status)).toEqual([1]);
    expect(Array.from(result.iterationCount)).toEqual([0]);
    for (const name of ['coefficients', 'covariance', 'residualCovariance', 'residuals'] as const)
      expect(result[name].every(Number.isNaN), name).toBe(true);
  });
}

for (const fixture of fixtures) {
  for (const round of fixture.rounds) {
    it(`matches independent NumPy ${fixture.name} FGLS with budget ${round.maximumIterations}`, () => {
      const result = computeSeeminglyUnrelatedRegressionOnCPU(
        {
          predictors: Float32Array.from(fixture.predictors),
          responses: Float32Array.from(fixture.responses),
          mask: fixture.weights ? undefined : Uint32Array.from(fixture.mask),
          spatial: fixture.weights
            ? {
                weights: {
                  offsets: Uint32Array.from(fixture.weights.offsets),
                  neighbors: Uint32Array.from(fixture.weights.neighbors),
                  weights: Float32Array.from(fixture.weights.weights)
                }
              }
            : undefined
        },
        {...fixture, iteration: {maximumIterations: round.maximumIterations}}
      );
      expectIteratedSURResult(result, round.expected, 2e-6);
      if (fixture.parameters) {
        expect(Array.from(result.spatial!.status)).toEqual([0, 0]);
        fixture.parameters.forEach((value, index) => {
          expect(Number.isFinite(result.spatial!.parameters[index])).toBe(true);
          expect(Math.abs(result.spatial!.parameters[index] - value)).toBeLessThan(2e-6);
        });
      }
    });
  }
}

it('keeps the one-step default identical and optionally counts that single fit', () => {
  const fixture = fixtures[0];
  const input = {
    predictors: Float32Array.from(fixture.predictors),
    responses: Float32Array.from(fixture.responses),
    mask: Uint32Array.from(fixture.mask)
  };
  const result = computeSeeminglyUnrelatedRegressionOnCPU(input, fixture);
  const oneRound = computeSeeminglyUnrelatedRegressionOnCPU(input, {
    ...fixture,
    iteration: {maximumIterations: 1}
  });
  expectIteratedSURResult(result, {...fixture.rounds[0].expected, status: [0]}, 2e-6);
  for (const name of ['coefficients', 'covariance', 'residualCovariance', 'residuals'] as const)
    expect(result[name]).toEqual(oneRound[name]);
});

it('returns status 3 and NaN on singular covariance without counting a failed round', () => {
  const fixture = fixtures[0];
  const result = computeSeeminglyUnrelatedRegressionOnCPU(
    {
      predictors: Float32Array.from(fixture.predictors),
      responses: Float32Array.from(fixture.responses),
      initialResiduals: new Float32Array(120).fill(2)
    },
    {...fixture, iteration: {maximumIterations: 2}}
  );
  expect(Array.from(result.status)).toEqual([3]);
  expect(Array.from(result.iterationCount)).toEqual([0]);
  for (const name of ['coefficients', 'covariance', 'residualCovariance', 'residuals'] as const)
    expect(result[name].every(Number.isNaN), name).toBe(true);
});

it('wires and snapshots iteration settings and optional count in both operation forms', () => {
  const fixture = fixtures[0];
  const iteration = {maximumIterations: 2, tolerance: 1e-5};
  const props = {
    ...fixture,
    parameters: undefined,
    iteration,
    predictors: 'predictors',
    responses: 'responses',
    mask: 'mask',
    output: {
      coefficients: 'coefficients',
      covariance: 'covariance',
      residualCovariance: 'residualCovariance',
      residuals: 'residuals',
      status: 'status',
      iteration: {status: {iterationCount: 'iterationCount'}}
    }
  };
  const operation = new GPUSeeminglyUnrelatedRegressionOperation(props);
  expect(new CPUSeeminglyUnrelatedRegressionOperation(props).contract.outputs).toEqual(
    operation.contract.outputs
  );
  iteration.maximumIterations = 7;
  const contributor = constructRegressionOperationContributor(operation, {
    predictors: {format: 'float32', length: 240},
    responses: {format: 'float32', length: 120},
    mask: {format: 'uint32', length: 40}
  }) as GPUSeeminglyUnrelatedRegression;
  expect(contributor.props.iteration).toEqual({maximumIterations: 2, tolerance: 1e-5});
  for (const iteration of [
    {maximumIterations: 0},
    {maximumIterations: 1.5},
    {maximumIterations: 2, tolerance: 0},
    {maximumIterations: 2, tolerance: NaN}
  ]) {
    expect(() =>
      computeSeeminglyUnrelatedRegressionOnCPU(
        {predictors: new Float32Array(240), responses: new Float32Array(120)},
        {...fixture, iteration}
      )
    ).toThrow(/iteration\./);
    expect(() => new GPUSeeminglyUnrelatedRegression({...contributor.props, iteration})).toThrow(
      /iteration\./
    );
  }
});

it('F13 SUR operation captures the optional nested iteration status', () => {
  const output = {
    coefficients: 'coefficients',
    covariance: 'covariance',
    residualCovariance: 'residualCovariance',
    residuals: 'residuals',
    status: 'status',
    iteration: {status: {iterationCount: 'rounds'}}
  };
  const props = {
    rowCount: 8,
    predictorCount: 1,
    equationCount: 2,
    predictors: 'predictors',
    responses: 'responses',
    output
  };
  for (const Operation of [
    GPUSeeminglyUnrelatedRegressionOperation,
    CPUSeeminglyUnrelatedRegressionOperation
  ]) {
    const operation = new Operation(props);
    expect(operation.contract.outputs).toHaveProperty('rounds');
    output.iteration.status.iterationCount = 'replacement';
    expect(operation.contract.outputs).not.toHaveProperty('replacement');
    output.iteration.status.iterationCount = 'rounds';
    const {iteration: _iteration, ...withoutCount} = output;
    expect(new Operation({...props, output: withoutCount}).contract.outputs).not.toHaveProperty(
      'rounds'
    );
  }
});
