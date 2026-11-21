// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeSpatialInteractionOnCPU,
  CPUSpatialInteractionOperation
} from '../../../src/gpu-spatial-analysis/spatial-interaction/cpu-spatial-interaction-operation';
import {
  GPUSpatialInteractionOperation,
  type SpatialInteractionOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-interaction/gpu-spatial-interaction-operation';
import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {GPUSpatialInteraction} from '../../../src/gpu-spatial-analysis/spatial-interaction/gpu-spatial-interaction';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {
  INPUT,
  OPTIONS,
  EXPECTED,
  EXHAUSTED,
  BOUNDED_INPUT,
  BOUNDED_OPTIONS,
  BOUNDED_EXPECTED,
  TIED_INPUT,
  TIED_OPTIONS,
  TIED_EXPECTED,
  OBSERVED_ZERO_INPUT,
  OBSERVED_ZERO_OPTIONS,
  OBSERVED_ZERO_EXPECTED,
  expectOracleResult,
  expectOracleRows
} from './spatial-interaction-audit-fixtures';

function createCPUInputs(): Record<string, GPUCPUVector> {
  return Object.fromEntries(
    Object.entries(INPUT).map(([name, values]) => [
      name,
      {
        format: name === 'flows' || name === 'covariate' ? 'float32' : 'uint32',
        length: values.length,
        chunks: [{values, length: values.length}],
        pack: () => values
      }
    ])
  );
}

it('parses stable dense and sparse Poisson deviance kernels', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const method of ['dense', 'sparse'] as const) {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const floating = (length: number) =>
        createTransientView(graph, `float-${serial++}`, 'float32', length);
      const unsigned = (length: number) =>
        createTransientView(graph, `uint-${serial++}`, 'uint32', length);
      const nodes = new GPUSpatialInteraction({
        origins: unsigned(4),
        destinations: unsigned(4),
        flows: floating(4),
        covariate: floating(4),
        originCount: 2,
        destinationCount: 2,
        constraint: 'doubly',
        coefficients: [0, 1],
        maximumIterations: 1,
        calibration: method === 'dense' ? 'bounded' : 'grid',
        maximumCalibrationIterations: 1,
        method,
        output: {
          predictions: floating(4),
          summary: floating(4),
          candidates: floating(8),
          status: unsigned(1),
          originResiduals: floating(2),
          destinationResiduals: floating(2),
          ...(method === 'dense' ? {calibration: floating(4)} : {})
        }
      }).getCommandNodes(graph);
      for (const {id, source} of getGPUCommandNodeKernelSources(nodes))
        expect(() => new WgslReflect(source), id).not.toThrow();
    }
  } finally {
    device.destroy();
  }
});

for (const constraint of ['unconstrained', 'origin', 'destination', 'doubly'] as const) {
  it(`matches every ${constraint} output to the independent Poisson oracle, including a shifted covariate`, () => {
    for (const shift of [0, 8]) {
      const actual = computeSpatialInteractionOnCPU(
        {...INPUT, covariate: INPUT.covariate.map(value => value + shift)},
        {...OPTIONS, constraint}
      );
      expectOracleResult(actual, EXPECTED[constraint], 0);
    }
  });

  it(`publishes the independent ${constraint} oracle through the CPU operation ports`, async () => {
    const operation = new CPUSpatialInteractionOperation({
      ...OPTIONS,
      constraint,
      pairCount: 8,
      origins: 'origins',
      destinations: 'destinations',
      flows: 'flows',
      covariate: 'covariate',
      pairState: 'pairState',
      output: {
        predictions: 'predictions',
        summary: 'summary',
        status: 'status',
        candidates: 'candidates',
        originResiduals: 'originResiduals',
        destinationResiduals: 'destinationResiduals'
      }
    });
    const outputs = await operation.execute({inputs: createCPUInputs()});
    expectOracleResult(
      outputs as Record<string, Float32Array | Uint32Array>,
      EXPECTED[constraint],
      0
    );
  });
}

it('finds a feasible bounded interior without increasing the one-sweep inner budget', async () => {
  for (const coefficients of [
    [-1, 1],
    [-1, 3]
  ]) {
    const input = {
      origins: Uint32Array.of(0, 0, 1, 1),
      destinations: Uint32Array.of(0, 1, 0, 1),
      flows: Float32Array.of(10, 10, 10, 10),
      covariate: Float32Array.of(0, 0, 0, 1)
    };
    const options = {
      originCount: 2,
      destinationCount: 2,
      constraint: 'doubly' as const,
      coefficients,
      calibration: 'bounded' as const,
      maximumIterations: 1,
      maximumCalibrationIterations: 2,
      coefficientTolerance: 1e-5
    };
    const grid = computeSpatialInteractionOnCPU(input, {
      ...options,
      calibration: 'grid',
      coefficients: [0]
    });
    expect(grid.summary[1]).toBeLessThan(1e-26);
    expectOracleRows(grid.predictions, [10, 10, 10, 10], 0);
    // beta=0 has kernel ones, hence one row/column sweep exactly fits four tens.
    // Budget two: the dyadic probe 0 (-1 + 2/2, or -1 + 4/4) is evaluated by the last
    // refinement and must be certified before exhaustion: calibration [0, 0, 0, 1].
    const actual = computeSpatialInteractionOnCPU(input, options);
    expect(actual.status[0]).toBe(0);
    expectOracleRows(actual.predictions, [10, 10, 10, 10], 1e-4);
    expect(Math.abs(actual.summary[0])).toBeLessThan(1e-5);
    expectOracleRows(actual.calibration!, [0, 0, 0, 1], 0);
    const operation = new CPUSpatialInteractionOperation({
      ...options,
      pairCount: 4,
      origins: 'origins',
      destinations: 'destinations',
      flows: 'flows',
      covariate: 'covariate',
      output: {
        predictions: 'predictions',
        summary: 'summary',
        candidates: 'candidates',
        status: 'status',
        calibration: 'calibration',
        originResiduals: 'originResiduals',
        destinationResiduals: 'destinationResiduals'
      }
    });
    const inputs: Record<string, GPUCPUVector> = Object.fromEntries(
      Object.entries(input).map(([name, values]) => [
        name,
        {
          format: values instanceof Float32Array ? 'float32' : 'uint32',
          length: values.length,
          chunks: [{values, length: values.length}],
          pack: () => values
        }
      ])
    );
    const outputs = (await operation.execute({inputs})) as Record<
      string,
      Float32Array | Uint32Array
    >;
    expectOracleRows(outputs['status'], [0], 0);
    expectOracleRows(outputs['predictions'], [10, 10, 10, 10], 1e-4);
    expectOracleRows(outputs['calibration'], [0, 0, 0, 1], 0);
  }
});

it('honors a one-sweep IPF budget and suppresses the unfinished result', () => {
  const actual = computeSpatialInteractionOnCPU(INPUT, {
    ...OPTIONS,
    coefficients: [-Math.log(2)],
    maximumIterations: 1
  });
  expectOracleResult(actual, EXHAUSTED, 3);
});

it('retains a feasible nonzero-deviance exploratory fit and never certifies an all-failed narrow bracket', () => {
  const input = {
    origins: Uint32Array.of(0, 0, 1, 1),
    destinations: Uint32Array.of(0, 1, 0, 1),
    flows: Float32Array.of(10, 20, 30, 40),
    covariate: Float32Array.of(0, 0, 0, 1)
  };
  const options = {
    originCount: 2,
    destinationCount: 2,
    constraint: 'doubly' as const,
    coefficients: [-1, 3],
    calibration: 'bounded' as const,
    maximumIterations: 1,
    maximumCalibrationIterations: 4,
    coefficientTolerance: 1e-5
  };
  const actual = computeSpatialInteractionOnCPU(input, options);
  // beta=0 gives independent means rowTotal*columnTotal/100=[12,18,28,42].
  expectOracleRows(actual.status, [4], 0);
  expectOracleRows(actual.predictions, [12, 18, 28, 42], 1e-5);
  expectOracleRows(actual.calibration!, [-1, 3, 4, 0], 0);
  const failed = computeSpatialInteractionOnCPU(
    {...input, flows: Float32Array.of(10, 10, 10, 10)},
    {...options, coefficients: [1, 1 + 2 ** -20]}
  );
  expectOracleRows(failed.status, [3], 0);
  expect(failed.calibration![3]).toBe(0);
});

it('selects the first coefficient on an exact deviance tie', () => {
  expectOracleResult(computeSpatialInteractionOnCPU(TIED_INPUT, TIED_OPTIONS), TIED_EXPECTED, 0);
});

it('keeps zero observed counts in support and in Poisson deviance', () => {
  expectOracleResult(
    computeSpatialInteractionOnCPU(OBSERVED_ZERO_INPUT, OBSERVED_ZERO_OPTIONS),
    OBSERVED_ZERO_EXPECTED,
    0
  );
});

for (const scene of ['no-mass', 'invalid'] as const) {
  it(`publishes every ${scene} output according to the support definition`, () => {
    const flows = new Float32Array(8);
    if (scene === 'invalid') flows[0] = -1;
    expectOracleResult(
      computeSpatialInteractionOnCPU({...INPUT, flows}, {...OPTIONS, coefficients: [0]}),
      {
        ...EXHAUSTED,
        candidates: [NaN, NaN, 0, scene === 'no-mass' ? 2 : 3]
      },
      scene === 'no-mass' ? 1 : 2
    );
  });
}

it('honors a one-refinement golden-section budget and publishes the final evaluated bracket', () => {
  const actual = computeSpatialInteractionOnCPU(BOUNDED_INPUT, BOUNDED_OPTIONS);
  expectOracleResult(actual, BOUNDED_EXPECTED, 4);
});

it('snapshots operation budgets and coefficients before the caller mutates them', async () => {
  const coefficients = [-Math.log(2)];
  const props: SpatialInteractionOperationProps = {
    ...OPTIONS,
    maximumIterations: 1,
    coefficients,
    pairCount: 8,
    origins: 'origins',
    destinations: 'destinations',
    flows: 'flows',
    covariate: 'covariate',
    pairState: 'pairState',
    output: {
      predictions: 'predictions',
      summary: 'summary',
      status: 'status',
      candidates: 'candidates',
      originResiduals: 'originResiduals',
      destinationResiduals: 'destinationResiduals'
    }
  };
  const cpu = new CPUSpatialInteractionOperation(props);
  const gpu = new GPUSpatialInteractionOperation(props);
  props.maximumIterations = 256;
  coefficients[0] = 0;
  const outputs = await cpu.execute({inputs: createCPUInputs()});
  expectOracleResult(outputs as Record<string, Float32Array | Uint32Array>, EXHAUSTED, 3);
  const contributor = constructNetworkOperationContributor(gpu, {
    origins: {format: 'uint32', length: 8},
    destinations: {format: 'uint32', length: 8},
    flows: {format: 'float32', length: 8},
    covariate: {format: 'float32', length: 8},
    pairState: {format: 'uint32', length: 8}
  }) as GPUSpatialInteraction;
  expect(contributor.props.maximumIterations).toBe(1);
  expect(contributor.props.coefficients).toEqual([-Math.log(2)]);
});
