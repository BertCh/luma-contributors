// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUVecchiaSpaceTimeLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood-operation';
import {CPUVecchiaSpaceTimeLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-space-time-log-likelihood-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixture from './likelihoods-fixture.json';
import {
  createLikelihoodInput,
  createStructureCodeLikelihoodInput,
  getSpaceTimeLikelihoodProps,
  getLikelihoodGraphInputs,
  expectLikelihoodGraphResults
} from './likelihoods-operation-test-utils';

for (const oracle of fixture.spaceTime) {
  it(`GPU and CPU vecchia-space-time-log-likelihood agree on correlated unequal-error rows, model ${oracle.typeCode}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props = getSpaceTimeLikelihoodProps();
    const input = createLikelihoodInput(oracle.parameters);
    const inputs = getLikelihoodGraphInputs(input, true);
    const outputs = Object.values(props.output);
    // Each run owns and destroys its operation. Use a fresh instance for every graph.
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUVecchiaSpaceTimeLogLikelihoodOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUVecchiaSpaceTimeLogLikelihoodOperation(props)],
      outputs
    });
    // Relative 1e-4 for all nonzero float outputs and decoded totals, absolute 1e-7 at zero.
    // Low words are compared at two float32 quanta of their high word; counts/statuses are exact.
    expectLikelihoodGraphResults(gpu, cpu);
    expect(Array.from(cpu['status'])).toEqual([0, 0, 5]);
    expect(cpu['conditionals'][4]).not.toBe(0);
  });
}

for (const edge of ['all-excluded', 'invalid-parameters', 'singular'] as const) {
  it(`GPU and CPU vecchia-space-time-log-likelihood compare every output on ${edge}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    let props = getSpaceTimeLikelihoodProps();
    const input = createLikelihoodInput(fixture.spaceTime[0].parameters);
    if (edge === 'all-excluded') {
      for (let row = 0; row < 5; row++) input.observations[row * 4 + 3] = 0;
    } else if (edge === 'invalid-parameters') {
      input.parameters.fill(0);
    } else {
      props = {...props, rowCount: 3};
      input.positions = new Float32Array([0, 0, 1, 0, 2, 0]);
      input.times = new Float32Array([0, 1, 2]);
      input.observations = new Float32Array([1, 0, 0, 1, 2, 0, 0, 1, 3, 0, 0, 1]);
      input.neighborhoodOffsets = new Uint32Array([0, 0, 1, 3]);
      input.neighborhoods = new Uint32Array([0, 0, 0]);
    }
    const inputs = getLikelihoodGraphInputs(input, true);
    const outputs = Object.values(props.output);
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUVecchiaSpaceTimeLogLikelihoodOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUVecchiaSpaceTimeLogLikelihoodOperation(props)],
      outputs
    });
    expectLikelihoodGraphResults(gpu, cpu);
    expect(Array.from(cpu['status'])).toEqual(
      edge === 'all-excluded' ? [0, 5, 0] : edge === 'invalid-parameters' ? [5, 0, 0] : [1, 0, 2]
    );
    expect(cpu['logLikelihood'].every(Number.isNaN)).toBe(true);
  });
}

it('compares every output when the loader accepts a negative structure code and WGSL clamps it to spherical', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props = {...getSpaceTimeLikelihoodProps(), rowCount: 2, neighborCount: 1};
  const inputs = getLikelihoodGraphInputs(createStructureCodeLikelihoodInput(), true);
  const outputs = Object.values(props.output);
  const gpu = await runOperationGraph(device, {
    inputs,
    operations: [new GPUVecchiaSpaceTimeLogLikelihoodOperation(props)],
    outputs
  });
  const cpu = await runOperationGraph(device, {
    inputs,
    operations: [new CPUVecchiaSpaceTimeLogLikelihoodOperation(props)],
    outputs
  });
  expectLikelihoodGraphResults(gpu, cpu);
  for (const rows of [gpu, cpu]) {
    // The hand-derived mean is zero and variance/pivot ratio are one. Keep the existing absolute
    // 1e-7 zero bound and relative 1e-4 float bounds; only the retained integer count is exact.
    expect(Math.abs(rows['conditionals'][4])).toBeLessThanOrEqual(1e-7);
    for (const component of [5, 6]) {
      expect(Math.abs(rows['conditionals'][component] - 1)).toBeLessThanOrEqual(1e-4);
    }
    expect(rows['conditionals'][7]).toBe(1);
    expect(Array.from(rows['status'])).toEqual([0, 0, 2]);
  }
});
