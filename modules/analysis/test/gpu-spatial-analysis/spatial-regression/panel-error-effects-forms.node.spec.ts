// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createTransientView,
  GPUCommandGraph,
  type GPUCPUVector,
  type GPUCPUResult
} from '@luma.gl/graph/gpu-core';
import {GPUSpatialPanelErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {GPUSpatialPanelErrorMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood-operation';
import {
  CPUSpatialPanelErrorMaximumLikelihoodOperation,
  computeSpatialPanelErrorMaximumLikelihoodOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-error-maximum-likelihood-operation';
import {getSpatialLogDeterminantFromTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {constructRegressionOperationContributor} from './regression-operation-test-utils';
import {
  getLongPeriodPanelInput,
  getPanelErrorOperationProps,
  longPeriodPanelFixture
} from './panel-error-long-periods.test-utils';

function compare(actual: ArrayLike<number>, expected: readonly number[]) {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    const value = expected[index];
    if (Number.isNaN(value)) expect(actual[index], `slot ${index}`).toBeNaN();
    else {
      expect(Number.isFinite(actual[index]), `slot ${index}`).toBe(true);
      // Float64 computation rounded to float32; dense oracle optimizes the exact determinant.
      expect(Math.abs(actual[index] - value), `slot ${index}`).toBeLessThanOrEqual(
        2e-6 * Math.max(1, Math.abs(value))
      );
    }
  }
}

function createCPUVector(values: Float32Array | Uint32Array): GPUCPUVector {
  return {
    format: values instanceof Uint32Array ? 'uint32' : 'float32',
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  };
}

function createCPUInputs(input = getLongPeriodPanelInput()): Record<string, GPUCPUVector> {
  return {
    predictors: createCPUVector(input.predictors),
    response: createCPUVector(input.response),
    offsets: createCPUVector(input.weights.offsets),
    neighbors: createCPUVector(input.weights.neighbors),
    weights: createCPUVector(input.weights.weights)
  };
}

function compareOutput(actual: GPUCPUResult, expected: readonly number[]) {
  if (
    !(
      actual instanceof Float32Array ||
      actual instanceof Uint32Array ||
      actual instanceof Int32Array
    )
  )
    throw new Error('expected one packed numeric output');
  compare(actual, expected);
}

const inputShapes = {
  predictors: {format: 'float32' as const, length: 64},
  response: {format: 'float32' as const, length: 64},
  offsets: {format: 'uint32' as const, length: 5},
  neighbors: {format: 'uint32' as const, length: 4},
  weights: {format: 'float32' as const, length: 4}
};

for (const effects of ['time', 'twoWay'] as const) {
  it(`${effects}: pure CPU fit beyond the GPU period limit matches every dense-oracle output`, () => {
    const actual = computeSpatialPanelErrorMaximumLikelihoodOnCPU(getLongPeriodPanelInput(), {
      ...longPeriodPanelFixture,
      effects
    });
    const expected = longPeriodPanelFixture.models[effects];
    compare(actual.table, expected.table);
    compare(actual.summary, expected.summary);
    compare(actual.residuals, expected.residuals);
    compare(actual.diagnosticTests, longPeriodPanelFixture.diagnostics);
    expect(actual.status).toEqual(Uint32Array.of(0));
    expect(actual.converged).toEqual(Uint32Array.of(1));
    expect(actual.diagnosticStatus).toEqual(Uint32Array.of(0));
    expect(actual.powerTraceStandardErrors).toEqual(new Float32Array(128));
  });

  it(`${effects}: actual CPU operation callback beyond the GPU period limit matches every oracle port`, async () => {
    const props = getPanelErrorOperationProps(effects);
    const operation = new CPUSpatialPanelErrorMaximumLikelihoodOperation(props);
    const actual = await operation.execute({inputs: createCPUInputs()});
    const expected = longPeriodPanelFixture.models[effects];
    for (const [name, values] of Object.entries({
      table: expected.table,
      summary: expected.summary,
      residuals: expected.residuals,
      tests: longPeriodPanelFixture.diagnostics,
      code: [0],
      converged: [1],
      diagnosticStatus: [0],
      traceErrors: new Array<number>(128).fill(0)
    })) {
      compareOutput(actual[name], values);
    }
    expect(Object.keys(actual).sort()).toEqual(Object.keys(operation.outputs).sort());
  });

  it(`${effects}: GPU operation refuses the period-contrast limit with its reason and CPU alternative`, () => {
    const operation = new GPUSpatialPanelErrorMaximumLikelihoodOperation(
      getPanelErrorOperationProps(effects)
    );
    expect(() => constructRegressionOperationContributor(operation, inputShapes)).toThrow(
      /predictorCount \+ periodCount - 1 <= 15.*16-column solve.*computeSpatialPanelErrorMaximumLikelihoodOnCPU/
    );
  });
}

it('unit effects retain the same long-period GPU construction path', () => {
  expect(
    constructRegressionOperationContributor(
      new GPUSpatialPanelErrorMaximumLikelihoodOperation(getPanelErrorOperationProps('unit')),
      inputShapes
    )
  ).toBeInstanceOf(GPUSpatialPanelErrorMaximumLikelihood);
});

it('the shared likelihood constructor independently rejects sixteen predictors after removing a wrapper guard', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `shared-period-limit-${serial++}`, format, length);
    expect(
      () =>
        new GPUSpatialErrorMaximumLikelihood({
          predictorCount: 16,
          predictors: view('float32', 64 * 16),
          response: view('float32', 64),
          weights: {
            offsets: view('uint32', 65),
            neighbors: view('uint32', 64),
            weights: view('float32', 64)
          },
          output: {
            table: view('float32', 72),
            summary: view('float32', 22),
            status: {code: view('uint32', 1)}
          }
        })
    ).toThrow(/spatial-error-maximum-likelihood.*predictorCount.*15/);
  } finally {
    device.destroy();
  }
});

it('exact pooled affine degeneracy reaches the CPU operation diagnostic ports', async () => {
  const predictors = Float32Array.of(0.25, -1, 0.5, 2, -0.75, 0.125, 1.5, -0.5);
  const input = {
    predictors,
    response: Float32Array.from(predictors, value => 3 + 2 * value),
    weights: {
      offsets: Uint32Array.of(0, 1, 2),
      neighbors: Uint32Array.of(1, 0),
      weights: Float32Array.of(1, 1)
    }
  };
  for (const effects of ['unit', 'time', 'twoWay'] as const) {
    const operation = new CPUSpatialPanelErrorMaximumLikelihoodOperation({
      ...getPanelErrorOperationProps(effects),
      entityCount: 2,
      periodCount: 4
    });
    const actual = await operation.execute({inputs: createCPUInputs(input)});
    expect(actual['code']).toEqual(Uint32Array.of(3));
    expect(actual['converged']).toEqual(Uint32Array.of(0));
    expect(actual['diagnosticStatus']).toEqual(Uint32Array.of(2));
    expect(actual['tests']).toEqual(new Float32Array(6).fill(NaN));
  }
});

it('all four forms refuse the unsupported random covariance, with GPU operation validation at lowering', () => {
  const props = getPanelErrorOperationProps('random');
  expect(() =>
    computeSpatialPanelErrorMaximumLikelihoodOnCPU(getLongPeriodPanelInput(), props)
  ).toThrow(/effects 'random'.*covariance determinant.*dense/);
  expect(() => new CPUSpatialPanelErrorMaximumLikelihoodOperation(props)).toThrow(
    /effects 'random'.*covariance determinant.*dense/
  );
  expect(() =>
    constructRegressionOperationContributor(
      new GPUSpatialPanelErrorMaximumLikelihoodOperation(props),
      inputShapes
    )
  ).toThrow(/effects 'random'.*log\|I.*dense/);
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `random-refusal-${serial++}`, format, length);
    expect(
      () =>
        new GPUSpatialPanelErrorMaximumLikelihood({
          effects: 'random',
          periodCount: 4,
          predictorCount: 1,
          predictors: view('float32', 8),
          response: view('float32', 8),
          weights: {
            offsets: view('uint32', 3),
            neighbors: view('uint32', 2),
            weights: view('float32', 2)
          },
          output: {
            table: view('float32', 12),
            summary: view('float32', 22),
            status: {code: view('uint32', 1)}
          }
        })
    ).toThrow(/effects 'random'.*log\|I.*dense/);
  } finally {
    device.destroy();
  }
});

it('equal W traces and infinity norm cannot encode the independent-unit between determinant', () => {
  // panel-error-effects.oracle.py: W1=[[0,1,0],[0,0,0],[0,0,0]],
  // W2=[[0,1,0],[0,0,1],[0,0,0]], lambda=.4, phi=.5. Both are nilpotent,
  // row-standardized on their non-islands, and have infinity norm 1. Every W trace is zero.
  expect(getSpatialLogDeterminantFromTraces(0.4, new Array<number>(128).fill(0), 1) === 0).toBe(
    true
  );
  // Dense slogdet(I + (phi^-2-1) A A') gives different between determinants. In closed form,
  // with c=3,d=4: |M1|=d^3+c*d*lambda^2; |M2|=d^3+2*c*d*lambda^2+c^2*lambda^4.
  compare([Math.log(65.92), Math.log(68.0704)], [4.188441885601216, 4.2205424637463995]);
  expect(4.2205424637463995 - 4.188441885601216).toBeGreaterThan(0.03);
});
