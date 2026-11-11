// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUConditionalGaussianSimulationOperation,
  type ConditionalGaussianSimulationOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation-operation';
import {CPUConditionalGaussianSimulationOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-conditional-gaussian-simulation-operation';
import type {GPUPredictionCovarianceModel} from '../../../src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {expectLikelihoodFloat} from './likelihoods-operation-test-utils';
import fixture from './conditional-operation-fixture.json';

for (const model of [
  'spherical',
  'exponential',
  'gaussian',
  'matern-3/2',
  'matern-5/2'
] as GPUPredictionCovarianceModel[]) {
  it(`compares all correlated joint draws and moments for ${model}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: ConditionalGaussianSimulationOperationProps = {
      id: 'simulation-graph',
      observationCount: 3,
      queryCount: 4,
      positions: 'positions',
      observations: 'observations',
      queries: 'queries',
      covariance: {model, sill: fixture.sill, nugget: fixture.nugget, range: fixture.range},
      mean: fixture.mean,
      seed: fixture.seed,
      replicateCount: 3,
      replicateOffset: 7,
      output: {values: 'values', moments: 'moments', status: 'status'}
    };
    const inputs = {
      positions: {format: 'float32x2' as const, values: Float32Array.from(fixture.positions)},
      observations: {format: 'float32x4' as const, values: Float32Array.from(fixture.observations)},
      queries: {format: 'float32x2' as const, values: Float32Array.from(fixture.queries)}
    };
    const outputs = Object.values(props.output);
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUConditionalGaussianSimulationOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUConditionalGaussianSimulationOperation(props)],
      outputs
    });
    expect(Array.from(gpu['status'])).toEqual([0, 0, 0]);
    expect(Array.from(cpu['status'])).toEqual([0, 0, 0]);
    // Relative 1e-4 on every moment component. Realizations use an absolute bound of 2e-4
    // times the query's natural scale, so a near-zero draw still has a meaningful error bound.
    for (let index = 0; index < cpu['moments'].length; index++)
      expectLikelihoodFloat(gpu['moments'][index], cpu['moments'][index], 1e-4);
    for (let index = 0; index < cpu['values'].length; index++) {
      const query = index % 4;
      const scale = Math.max(
        Math.abs(cpu['moments'][query * 2]),
        Math.sqrt(cpu['moments'][query * 2 + 1])
      );
      expect(Math.abs(gpu['values'][index] - cpu['values'][index])).toBeLessThanOrEqual(
        2e-4 * scale
      );
    }
    expect(cpu['values'][0]).not.toBe(cpu['values'][4]);
  });
}

it('compares every output on singular duplicate noiseless observations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: ConditionalGaussianSimulationOperationProps = {
    id: 'simulation-singular',
    observationCount: 2,
    queryCount: 2,
    positions: 'positions',
    observations: 'observations',
    queries: 'queries',
    covariance: {model: 'exponential', sill: 1.25, nugget: 0.25, range: 2},
    seed: 17,
    replicateCount: 2,
    output: {values: 'values', moments: 'moments', status: 'status'}
  };
  const inputs = {
    positions: {format: 'float32x2' as const, values: new Float32Array([0, 0, 0, 0])},
    observations: {
      format: 'float32x4' as const,
      values: new Float32Array([1, 0, 0, 1, 2, 0, 0, 1])
    },
    queries: {format: 'float32x2' as const, values: new Float32Array([1, 0, 2, 0])}
  };
  for (const Operation of [
    GPUConditionalGaussianSimulationOperation,
    CPUConditionalGaussianSimulationOperation
  ]) {
    const rows = await runOperationGraph(device, {
      inputs,
      operations: [new Operation(props)],
      outputs: Object.values(props.output)
    });
    expect(Array.from(rows['status'])).toEqual([2, 2]);
    expect(rows['values'].length).toBe(4);
    expect(rows['moments'].length).toBe(4);
    expect(rows['values'].every(Number.isNaN)).toBe(true);
    expect(rows['moments'].every(Number.isNaN)).toBe(true);
  }
});
