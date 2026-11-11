// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUVecchiaLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood-operation';
import {CPUVecchiaLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-log-likelihood-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixture from './likelihoods-fixture.json';
import {
  createLikelihoodInput,
  getLikelihoodProps,
  getLikelihoodGraphInputs,
  expectLikelihoodGraphResults
} from './likelihoods-operation-test-utils';

for (const oracle of fixture.spatial) {
  it(`GPU and CPU vecchia-log-likelihood agree on correlated unequal-error rows, model ${oracle.modelCode}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props = getLikelihoodProps();
    const input = createLikelihoodInput(oracle.parameters);
    const inputs = getLikelihoodGraphInputs(input, false);
    const outputs = Object.values(props.output);
    // Each run owns and destroys its operation. Use a fresh instance for every graph.
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUVecchiaLogLikelihoodOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUVecchiaLogLikelihoodOperation(props)],
      outputs
    });
    // Relative 1e-4 for nonzero float outputs and decoded totals, absolute 1e-7 at zero.
    // Conditional means also allow 2 * 2^-24 * sum(abs(weight * observation)), from the SciPy
    // oracle: their relative error is amplified when positive and negative terms nearly cancel.
    // Low words are compared at two float32 quanta of their high word; counts/statuses are exact.
    expectLikelihoodGraphResults(gpu, cpu, oracle.meanTermMagnitudes);
    expect(Array.from(cpu['status'])).toEqual([0, 0, 5]);
    expect(cpu['conditionals'][4]).not.toBe(0);
  });
}

for (const edge of ['all-excluded', 'invalid-parameters', 'singular'] as const) {
  it(`GPU and CPU vecchia-log-likelihood compare every output on ${edge}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    let props = getLikelihoodProps();
    const input = createLikelihoodInput(fixture.spatial[0].parameters);
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
    const inputs = getLikelihoodGraphInputs(input, false);
    const outputs = Object.values(props.output);
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUVecchiaLogLikelihoodOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUVecchiaLogLikelihoodOperation(props)],
      outputs
    });
    expectLikelihoodGraphResults(gpu, cpu);
    expect(Array.from(cpu['status'])).toEqual(
      edge === 'all-excluded' ? [0, 5, 0] : edge === 'invalid-parameters' ? [5, 0, 0] : [1, 0, 2]
    );
    expect(cpu['logLikelihood'].every(Number.isNaN)).toBe(true);
  });
}
