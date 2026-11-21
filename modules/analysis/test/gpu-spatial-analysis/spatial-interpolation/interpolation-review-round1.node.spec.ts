// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {CPUVecchiaLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-log-likelihood-operation';
import {CPUVecchiaSpaceTimeLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-space-time-log-likelihood-operation';
import {CPUVecchiaFisherScoringOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-fisher-scoring-operation';
import {CPUConditionalGaussianSimulationOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-conditional-gaussian-simulation-operation';
import {GPUConditionalGaussianSimulationOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation-operation';
import {GPUConditionalGaussianSimulation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';
import {
  GPUKriging,
  type GPUKrigingOutput
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-kriging';
import {getGPUVecchiaLogLikelihoodParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-parameters';
import {getGPUVecchiaSpaceTimeLogLikelihoodParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {makeExtentGrid} from './extent-grid';

for (const family of ['spatial', 'space-time', 'fisher'] as const) {
  it(`B1 constructs and executes the CPU ${family} operation with caller-selected graph names`, async () => {
    const spatial = getGPUVecchiaLogLikelihoodParameterValues({
      model: 'exponential',
      sill: 1,
      range: 3
    });
    const spaceTime = getGPUVecchiaSpaceTimeLogLikelihoodParameterValues({
      type: 'metric',
      joint: {model: 'exponential', sill: 1, nugget: 0, range: 3},
      anisotropy: 1
    });
    const props = {
      id: 'round1-cpu',
      rowCount: 1,
      neighborCount: 2,
      positions: 'custom-positions',
      observations: 'custom-observations',
      parameters: 'custom-parameters',
      neighborhoodOffsets: 'custom-offsets',
      neighborhoods: 'custom-neighbors',
      output: {
        conditionals: 'custom-conditionals',
        conditionalStatus: 'custom-conditional-status',
        siteTerms: 'custom-terms',
        siteStatus: 'custom-site-status',
        logLikelihood: 'custom-likelihood',
        status: 'custom-status'
      }
    };
    const operation =
      family === 'spatial'
        ? new CPUVecchiaLogLikelihoodOperation(props)
        : family === 'fisher'
          ? new CPUVecchiaFisherScoringOperation({
              ...props,
              output: {
                siteTerms: 'custom-terms',
                siteStatus: 'custom-site-status',
                totals: 'custom-totals',
                status: 'custom-status'
              }
            })
          : new CPUVecchiaSpaceTimeLogLikelihoodOperation({
              ...props,
              times: 'custom-times',
              temporalContext: {
                unit: 'second',
                epoch: 0n,
                encoding: 'float32-relative',
                axis: {kind: 'irregular'}
              }
            });
    expect(operation.inputs['positions']).toBe('custom-positions');
    // Mutation after construction must not affect the CPU closure or output routing.
    props.rowCount = 99;
    props.neighborCount = 0;
    props.output.siteTerms = 'mutated-terms';
    const device = createNullWebGPUDevice();
    try {
      const result = await runOperationGraph(device, {
        inputs: {
          'custom-positions': {format: 'float32x2', values: Float32Array.of(0, 0)},
          'custom-observations': {format: 'float32x4', values: Float32Array.of(2, 0, 0, 1)},
          'custom-parameters': {
            format: 'float32',
            values: family === 'space-time' ? spaceTime : spatial
          },
          'custom-offsets': {format: 'uint32', values: Uint32Array.of(0, 0)},
          'custom-neighbors': {format: 'uint32', values: Uint32Array.of(0xffffffff)},
          ...(family === 'space-time'
            ? {'custom-times': {format: 'float32' as const, values: Float32Array.of(0)}}
            : {})
        },
        operations: [operation],
        outputs: ['custom-terms', 'custom-status']
      });
      expect(Array.from(result['custom-status'])).toEqual([0, 0, 1]);
      expect(result['custom-terms']).toHaveLength(family === 'fisher' ? 10 : 1);
      // Independent marginal Gaussian log density for y=2 and variance=1.
      expect(Number.isFinite(result['custom-terms'][0])).toBe(true);
      expect(result['custom-terms'][0]).toBeCloseTo(-2.9189385332046727, 6);
    } finally {
      device.destroy();
    }
  });
}

it('B1 constructs and executes CPU simulation with seed 7 and caller-selected names', async () => {
  const props = {
    id: 'round1-simulation',
    observationCount: 1,
    queryCount: 2,
    replicateCount: 3,
    seed: 7,
    mean: 3,
    covariance: {model: 'exponential' as const, sill: 1, range: 3},
    positions: 'sample-xy',
    observations: 'measurements',
    queries: 'query-xy',
    output: {values: 'draws', moments: 'conditional-moments', status: 'draw-status'}
  };
  const operation = new CPUConditionalGaussianSimulationOperation(props);
  expect(operation.inputs['positions']).toBe('sample-xy');
  props.seed = 99;
  props.mean = 100;
  props.covariance.range = 99;
  props.output.moments = 'mutated';
  const device = createNullWebGPUDevice();
  try {
    const result = await runOperationGraph(device, {
      inputs: {
        'sample-xy': {format: 'float32x2', values: Float32Array.of(0, 0)},
        measurements: {format: 'float32x4', values: Float32Array.of(2, 0, 0, 1)},
        'query-xy': {format: 'float32x2', values: Float32Array.of(0, 0, 1, 0)}
      },
      operations: [operation],
      outputs: ['draws', 'conditional-moments', 'draw-status']
    });
    expect(result['draws']).toHaveLength(6);
    expect(result['draws'].every(Number.isFinite)).toBe(true);
    expect(Array.from(result['draw-status'])).toEqual([0, 0, 0]);
    expect(Array.from(result['conditional-moments'])).toEqual([
      2,
      0,
      Math.fround(2.6321205588285577),
      Math.fround(0.8646647167633873)
    ]);
    expect([result['draws'][0], result['draws'][2], result['draws'][4]]).toEqual([2, 2, 2]);
  } finally {
    device.destroy();
  }
});

it('B5 captures only declared simulation output ports from a class-backed wrapper', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const values = createTransientView(graph, 'draws', 'float32', 1);
  const moments = createTransientView(graph, 'moments', 'float32x2', 1);
  const status = createTransientView(graph, 'status', 'uint32', 1);
  class Outputs {
    constructor(
      readonly values: GraphDataView<'float32'>,
      readonly moments: GraphDataView<'float32x2'>,
      readonly status: GraphDataView<'uint32'>
    ) {}
    reset() {
      throw new Error('not a graph port');
    }
    sum() {
      throw new Error('a focal output name is not a simulation port');
    }
  }
  const contributor = new GPUConditionalGaussianSimulation({
    positions: createTransientView(graph, 'positions', 'float32x2', 1),
    observations: createTransientView(graph, 'observations', 'float32x4', 1),
    queries: createTransientView(graph, 'queries', 'float32x2', 1),
    covariance: {model: 'exponential', sill: 1, range: 3},
    seed: 7,
    replicateCount: 1,
    output: new Outputs(values, moments, status)
  });
  expect(Object.keys(contributor.props.output).sort()).toEqual(['moments', 'status', 'values']);
  expect(contributor.props.output.values).toBe(values);
  expect(contributor.props.output.moments).toBe(moments);
  expect(contributor.props.output.status).toBe(status);
  expect(contributor.getCommandNodes(graph)).toHaveLength(4);
});

it('S1 snapshots an initially empty optional raster status wrapper', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const status: NonNullable<GPUKrigingOutput['status']> = {};
  const contributor = new GPUKriging({
    positions: createTransientView(graph, 'positions', 'float32x2', 3),
    values: createTransientView(graph, 'values', 'float32', 3),
    parameters: createTransientView(graph, 'parameters', 'float32', 12),
    grid: makeExtentGrid(1, 1, [-1, -1, 1, 1]),
    indexBounds: [-2, -2, 2, 2],
    indexGridSize: [2, 2],
    maximumNeighborCount: 3,
    output: {values: createTransientView(graph, 'prediction', 'float32', 1), status}
  });
  status.invalidCount = createTransientView(graph, 'late-invalid', 'uint32', 1);
  expect(contributor.props.output.status).not.toBe(status);
  expect(contributor.props.output.status?.invalidCount).toBeUndefined();
  expect(contributor.getCommandNodes(graph).map(node => node.id)).not.toContain(
    'kriging-clear-invalid'
  );
});

it.each([
  'positions',
  'observations',
  'queries'
] as const)('B6 rejects GPU simulation %s lengths inconsistent with declared counts', port => {
  const operation = new GPUConditionalGaussianSimulationOperation({
    id: 'shape-check',
    observationCount: 3,
    queryCount: 2,
    replicateCount: 3,
    seed: 7,
    covariance: {model: 'exponential', sill: 1, range: 3},
    positions: 'custom-positions',
    observations: 'custom-observations',
    queries: 'custom-queries',
    output: {values: 'draws', moments: 'moments', status: 'status'}
  });
  const shapes = {
    positions: {format: 'float32x2' as const, length: 3},
    observations: {format: 'float32x4' as const, length: 3},
    queries: {format: 'float32x2' as const, length: 2}
  };
  shapes[port].length = 1;
  expect(() => constructNetworkOperationContributor(operation, shapes)).toThrow(
    new RegExp(`^shape-check ${port}.*${port === 'queries' ? 'queryCount' : 'observationCount'}`)
  );
});
