// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  computeProbitRegressionOnCPU,
  CPUProbitRegressionOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-probit-regression-operation';
import {GPUProbitRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression-operation';
import {GPU_PROBIT_SPATIAL_LAG_STATUS} from '../../../src/gpu-spatial-analysis/spatial-regression/probit-spatial-lag-nodes';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  constructRegressionOperationContributor,
  expectRegressionRowsClose
} from './regression-operation-test-utils';
import {getProbitSpatialLagInput, PROBIT_SPATIAL_LAG_PROPS} from './probit-spatial-lag-test-utils';
import fixture from './probit-spatial-lag-fixture.json';
import auditFixture from './probit-audit-fixture.json';

function getProbitCallbackInputs(operation: CPUProbitRegressionOperation, oracle = fixture) {
  const input = getProbitSpatialLagInput(oracle);
  const values: Record<string, Float32Array | Uint32Array> = {
    predictors: input.predictors,
    response: input.response,
    mask: input.mask,
    offsets: input.spatialLag.weights.offsets,
    neighbors: input.spatialLag.weights.neighbors,
    weights: input.spatialLag.weights.weights,
    weightFlag: Uint32Array.of(1)
  };
  return Object.fromEntries(
    Object.entries(operation.inputs).map(([port, name]): [string, GPUCPUVector] => {
      const rows = values[name];
      return [
        port,
        {
          format: rows instanceof Uint32Array ? 'uint32' : 'float32',
          length: rows.length,
          chunks: [{values: rows, length: rows.length}],
          pack: () => rows
        }
      ];
    })
  );
}

it('executes the named CPU operation without a device against every independent oracle output', async () => {
  for (const oracle of [auditFixture, fixture]) {
    const props = {
      ...PROBIT_SPATIAL_LAG_PROPS,
      rowCount: oracle.response.length,
      tolerance: 1e-9,
      spatialLag: {
        ...PROBIT_SPATIAL_LAG_PROPS.spatialLag!,
        output: {...PROBIT_SPATIAL_LAG_PROPS.spatialLag!.output}
      }
    };
    const operation = new CPUProbitRegressionOperation(props);
    const inputs = getProbitCallbackInputs(operation, oracle);
    // The published callback must retain the validated scalar settings and output names.
    props.rowCount = 1;
    props.predictorCount = 1;
    props.spatialLag.output.coefficients = 'mutatedOutput';
    const result = await operation.execute({inputs});
    for (const [name, expected] of Object.entries({
      coefficients: oracle.expected.ordinaryCoefficients,
      standardErrors: oracle.expected.ordinaryStandardErrors,
      fitted: oracle.expected.fitted.map(value => (value === null ? NaN : value)),
      marginalEffects: oracle.expected.marginalEffects,
      tests: oracle.expected.tests,
      lagCoefficients: oracle.expected.coefficients,
      lagStandardErrors: oracle.expected.standardErrors
    }))
      expectRegressionRowsClose(result[name] as Float32Array, expected, 3e-6);
    expectRegressionRowsClose(
      (result['summary'] as Float32Array).subarray(0, 3),
      oracle.expected.summary,
      3e-6
    );
    expect(Array.from(result['status'] as Uint32Array)).toEqual([0]);
    expect(Array.from(result['diagnosticStatus'] as Uint32Array)).toEqual([0]);
    expect(Array.from(result['lagStatus'] as Uint32Array)).toEqual([oracle.expected.status]);
    expect(result['mutatedOutput']).toBeUndefined();
  }
});

it('executes named CPU budget and incomplete-weights paths with independently required statuses', async () => {
  const budget = new CPUProbitRegressionOperation({
    ...PROBIT_SPATIAL_LAG_PROPS,
    maximumIterations: 1
  });
  const limited = await budget.execute({inputs: getProbitCallbackInputs(budget)});
  expect(Array.from(limited['status'] as Uint32Array)).toEqual([3]);
  expect((limited['summary'] as Float32Array)[3]).toBe(1);
  expect(Array.from(limited['diagnosticStatus'] as Uint32Array)).toEqual([1]);
  expect(Array.from(limited['lagStatus'] as Uint32Array)).toEqual([1]);
  for (const name of ['standardErrors', 'tests', 'lagCoefficients', 'lagStandardErrors']) {
    expect((limited[name] as Float32Array).every(Number.isNaN), name).toBe(true);
  }
  const incomplete = new CPUProbitRegressionOperation({
    ...PROBIT_SPATIAL_LAG_PROPS,
    spatialDiagnostics: {
      ...PROBIT_SPATIAL_LAG_PROPS.spatialDiagnostics!,
      weights: {
        ...PROBIT_SPATIAL_LAG_PROPS.spatialDiagnostics!.weights,
        status: {candidateOverflow: 'weightFlag'}
      }
    },
    spatialLag: {
      ...PROBIT_SPATIAL_LAG_PROPS.spatialLag!,
      weights: {
        ...PROBIT_SPATIAL_LAG_PROPS.spatialLag!.weights,
        status: {candidateOverflow: 'weightFlag'}
      }
    }
  });
  const suppressed = await incomplete.execute({inputs: getProbitCallbackInputs(incomplete)});
  expect(Array.from(suppressed['status'] as Uint32Array)).toEqual([0]);
  expect(Array.from(suppressed['diagnosticStatus'] as Uint32Array)).toEqual([2]);
  expect(Array.from(suppressed['lagStatus'] as Uint32Array)).toEqual([5]);
  for (const name of ['tests', 'lagCoefficients', 'lagStandardErrors']) {
    expect((suppressed[name] as Float32Array).every(Number.isNaN), name).toBe(true);
  }
});

it('matches independent SciPy probit, dense IV projection and structural HC0 sandwich on masked tied rows and an island', () => {
  for (const oracle of [auditFixture, fixture]) {
    const fixture = oracle;
    const input = getProbitSpatialLagInput(fixture);
    const result = computeProbitRegressionOnCPU(
      {...input, spatialDiagnostics: input.spatialLag},
      {predictorCount: 2, tolerance: 1e-9}
    );
    expect(result.status[0]).toBe(0);
    expect(result.spatialLag!.status[0]).toBe(fixture.expected.status);
    expect(result.spatialDiagnostics!.status[0]).toBe(0);
    for (const [actual, expected] of [
      [result.spatialLag!.coefficients, fixture.expected.coefficients],
      [result.spatialLag!.standardErrors, fixture.expected.standardErrors],
      [result.coefficients, fixture.expected.ordinaryCoefficients],
      [result.standardErrors, fixture.expected.ordinaryStandardErrors],
      [result.marginalEffects, fixture.expected.marginalEffects],
      [result.spatialDiagnostics!.tests, fixture.expected.tests],
      [result.fitted, fixture.expected.fitted.map(value => (value === null ? NaN : value))],
      [result.summary.subarray(0, 3), fixture.expected.summary]
    ] as const)
      expectRegressionRowsClose(actual, expected, 3e-6);
    expect(Number.isFinite(result.fitted[0])).toBe(true); // island retained
    expect(result.fitted[4]).toBeNaN();
    expect(result.fitted[17]).toBeNaN();
  }
});

it('retains coefficient and sandwich equivariance when predictor units change', () => {
  const input = getProbitSpatialLagInput();
  input.predictors = Float32Array.from(
    input.predictors,
    (value, index) => value * (index % 2 ? -4 : 2)
  );
  const result = computeProbitRegressionOnCPU(input, {predictorCount: 2, tolerance: 1e-9});
  expect(result.spatialLag!.status[0]).toBe(0);
  expectRegressionRowsClose(
    result.spatialLag!.coefficients,
    fixture.expected.coefficients.map((value, index) => value / [1, 2, -4, 1][index]),
    3e-6
  );
  expectRegressionRowsClose(
    result.spatialLag!.standardErrors,
    fixture.expected.standardErrors.map((value, index) => value / [1, 2, 4, 1][index]),
    3e-6
  );
});

it('publishes an unconstrained rho and its errors with the out-of-range flag', () => {
  const input = getProbitSpatialLagInput();
  // Scaling W by 1/4 divides rho's design column by four, preserves the instrument span,
  // and multiplies rho and its error by four. This is an independent exact IV invariant.
  input.spatialLag.weights.weights = Float32Array.from(
    input.spatialLag.weights.weights,
    value => value / 4
  );
  const result = computeProbitRegressionOnCPU(input, {predictorCount: 2, tolerance: 1e-9});
  expect(result.spatialLag!.status[0]).toBe(GPU_PROBIT_SPATIAL_LAG_STATUS.RHO_OUT_OF_RANGE);
  expectRegressionRowsClose(
    result.spatialLag!.coefficients,
    fixture.expected.coefficients.map((value, index) => value * (index === 3 ? 4 : 1)),
    3e-6
  );
  expectRegressionRowsClose(
    result.spatialLag!.standardErrors,
    fixture.expected.standardErrors.map((value, index) => value * (index === 3 ? 4 : 1)),
    3e-6
  );
});

it('suppresses linearization after an exhausted ordinary fit and distinguishes singular stages', () => {
  const limited = computeProbitRegressionOnCPU(getProbitSpatialLagInput(), {
    predictorCount: 2,
    maximumIterations: 1
  });
  expect(limited.status[0]).toBe(3);
  expect(limited.summary[3]).toBe(1);
  expect(limited.spatialLag!.status[0]).toBe(GPU_PROBIT_SPATIAL_LAG_STATUS.MODEL_NOT_CONVERGED);
  expect(limited.spatialLag!.coefficients.every(Number.isNaN)).toBe(true);
  expect(limited.spatialLag!.standardErrors.every(Number.isNaN)).toBe(true);
  const input = getProbitSpatialLagInput();
  input.spatialLag.weights.weights.fill(0);
  const singular = computeProbitRegressionOnCPU(input, {predictorCount: 2});
  expect(singular.spatialLag!.status[0]).toBe(GPU_PROBIT_SPATIAL_LAG_STATUS.SINGULAR_FIRST_STAGE);
  expect(singular.spatialLag!.coefficients.every(Number.isNaN)).toBe(true);
  expect(singular.spatialLag!.standardErrors.every(Number.isNaN)).toBe(true);
  // Opposite outcomes at every x give beta0=0 exactly, hence G_rho=0, although
  // [1,x,Wx,W²x] has rank >= 3. Thus only the second stage is singular.
  const second = computeProbitRegressionOnCPU(
    {
      predictors: Float32Array.of(-2, -2, -1, -1, 1, 1, 2, 2),
      response: Float32Array.of(0, 1, 0, 1, 0, 1, 0, 1),
      spatialLag: {
        weights: {
          offsets: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8),
          neighbors: Uint32Array.of(1, 2, 3, 4, 5, 6, 7, 0),
          weights: new Float32Array(8).fill(1)
        }
      }
    },
    {predictorCount: 1}
  );
  expect(second.status[0]).toBe(0);
  expectRegressionRowsClose(second.summary, [8, -8 * Math.log(2), 16 * Math.log(2), 1], 1e-7);
  expect(second.spatialLag!.status[0]).toBe(GPU_PROBIT_SPATIAL_LAG_STATUS.SINGULAR_SECOND_STAGE);
  expect(second.spatialLag!.coefficients.every(Number.isNaN)).toBe(true);
  expect(second.spatialLag!.standardErrors.every(Number.isNaN)).toBe(true);
});

it('refuses malformed and incomplete weights even when their bad focus row is masked', () => {
  for (const scenario of [
    'negative',
    'origin',
    'order',
    'excludedOffset',
    'overflow',
    'candidateOverflow',
    'invalidCount',
    'iterationLimitReached',
    'converged',
    'approximation',
    'uncertainCount'
  ] as const) {
    const input = getProbitSpatialLagInput();
    const weights = input.spatialLag.weights;
    if (scenario === 'negative') weights.weights[0] = -1;
    else if (scenario === 'origin') weights.offsets[0] = 1;
    else if (scenario === 'order') weights.neighbors[1] = weights.neighbors[0];
    else if (scenario === 'excludedOffset') weights.offsets[4] = 0xffffffff;
    else
      Object.assign(weights, {
        status: {[scenario]: Uint32Array.of(scenario === 'converged' ? 0 : 1)}
      });
    const result = computeProbitRegressionOnCPU(input, {predictorCount: 2});
    expect(result.status[0], scenario).toBe(0);
    expect(result.spatialLag!.status[0], scenario).toBe(
      GPU_PROBIT_SPATIAL_LAG_STATUS.INVALID_WEIGHTS
    );
    expect(result.spatialLag!.coefficients.every(Number.isNaN), scenario).toBe(true);
    expect(result.spatialLag!.standardErrors.every(Number.isNaN), scenario).toBe(true);
  }
});

it('parses and audits every optional kernel at the maximum predictor count and counts its graph nodes', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const statuses = {
      overflow: 'weightFlag',
      candidateOverflow: 'weightFlag',
      invalidCount: 'weightFlag',
      iterationLimitReached: 'weightFlag',
      converged: 'weightFlag',
      approximation: 'weightFlag',
      uncertainCount: 'weightFlag'
    };
    const props = {
      ...PROBIT_SPATIAL_LAG_PROPS,
      predictorCount: 11,
      maximumIterations: 1,
      spatialLag: {
        ...PROBIT_SPATIAL_LAG_PROPS.spatialLag!,
        weights: {...PROBIT_SPATIAL_LAG_PROPS.spatialLag!.weights, status: statuses}
      },
      spatialDiagnostics: {
        ...PROBIT_SPATIAL_LAG_PROPS.spatialDiagnostics!,
        weights: {...PROBIT_SPATIAL_LAG_PROPS.spatialDiagnostics!.weights, status: statuses}
      }
    };
    const operation = new GPUProbitRegressionOperation(props);
    const formats = {
      predictors: ['float32', 660],
      response: ['float32', 60],
      mask: ['uint32', 60],
      offsets: ['uint32', 61],
      neighbors: ['uint32', 118],
      weights: ['float32', 118],
      weightFlag: ['uint32', 1]
    } as const;
    const ports = Object.fromEntries([
      ...Object.entries(formats).map(([name, [format, length]]) => [
        name,
        createTransientView(graph, name, format, length)
      ]),
      ...Object.entries(operation.outputs).map(([name, output]) => [
        name,
        createTransientView(graph, name, output.format, output.length)
      ]),
      ...Object.entries({
        lagOffsets: 'offsets',
        lagNeighbors: 'neighbors',
        lagWeights: 'weights'
      }).map(([name, source]) => [
        name,
        createTransientView(
          graph,
          name,
          formats[source as keyof typeof formats][0],
          formats[source as keyof typeof formats][1]
        )
      ]),
      ...Object.keys(operation.inputs)
        .filter(name => name.startsWith('lagWeight') || name.startsWith('diagnosticWeight'))
        .filter(name => name !== 'lagWeights')
        .map(name => [name, createTransientView(graph, name, 'uint32', 1)])
    ]);
    const contributor = operation.createContributor(name => ports[name] as never);
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBe(
      (contributor as {estimatedCommandNodeCount?: number}).estimatedCommandNodeCount
    );
    getParsedKernelSources(nodes);
    for (const kernel of getGPUCommandNodeKernelSources(nodes).filter(kernel =>
      kernel.id.includes('-spatial-lag-')
    ))
      expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    operation.destroy();
  } finally {
    device.destroy();
  }
});

it('validates optional port sizes and snapshots operation props', () => {
  const shapes = {
    predictors: {format: 'float32' as const, length: 120},
    response: {format: 'float32' as const, length: 60},
    mask: {format: 'uint32' as const, length: 60},
    offsets: {format: 'uint32' as const, length: 61},
    neighbors: {format: 'uint32' as const, length: 118},
    weights: {format: 'float32' as const, length: 118},
    lagOffsets: {format: 'uint32' as const, length: 61},
    lagNeighbors: {format: 'uint32' as const, length: 118},
    lagWeights: {format: 'float32' as const, length: 118}
  };
  const props = {
    ...PROBIT_SPATIAL_LAG_PROPS,
    spatialLag: {
      ...PROBIT_SPATIAL_LAG_PROPS.spatialLag!,
      output: {...PROBIT_SPATIAL_LAG_PROPS.spatialLag!.output}
    }
  };
  const operation = new GPUProbitRegressionOperation(props);
  props.predictorCount = 1;
  props.spatialLag.output.coefficients = 'wrong';
  expect(() => constructRegressionOperationContributor(operation, shapes)).not.toThrow();
  expect(
    () =>
      new CPUProbitRegressionOperation({
        ...PROBIT_SPATIAL_LAG_PROPS,
        spatialLag: {
          ...PROBIT_SPATIAL_LAG_PROPS.spatialLag!,
          output: {...PROBIT_SPATIAL_LAG_PROPS.spatialLag!.output, coefficients: 'predictors'}
        }
      })
  ).toThrow();
});
