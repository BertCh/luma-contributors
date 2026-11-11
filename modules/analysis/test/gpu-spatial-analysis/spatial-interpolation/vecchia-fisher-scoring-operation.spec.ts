// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUVecchiaFisherScoringOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring-operation';
import {CPUVecchiaFisherScoringOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-fisher-scoring-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixture from './likelihoods-fixture.json';
import {
  createLikelihoodInput,
  getFisherProps,
  getLikelihoodGraphInputs,
  expectLikelihoodGraphResults
} from './likelihoods-operation-test-utils';

for (const oracle of fixture.spatial) {
  it(`GPU and CPU vecchia-fisher-scoring agree on correlated unequal-error rows, model ${oracle.modelCode}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props = getFisherProps();
    const input = createLikelihoodInput(oracle.parameters);
    const inputs = getLikelihoodGraphInputs(input, false);
    const outputs = Object.values(props.output);
    // Each run owns and destroys its operation. Use a fresh instance for every graph.
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUVecchiaFisherScoringOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUVecchiaFisherScoringOperation(props)],
      outputs
    });
    // Relative 1e-4 for all nonzero float outputs and decoded totals, absolute 1e-7 at zero.
    // Low words are compared at two float32 quanta of their high word; counts/statuses are exact.
    expectLikelihoodGraphResults(gpu, cpu);
    expect(Array.from(cpu['status'])).toEqual([0, 0, 5]);
    expect(cpu['siteTerms'][11]).not.toBe(cpu['siteTerms'][1]);
  });
}

for (const edge of ['all-excluded', 'invalid-parameters', 'singular'] as const) {
  it(`GPU and CPU vecchia-fisher-scoring compare every output on ${edge}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    let props = getFisherProps();
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
      operations: [new GPUVecchiaFisherScoringOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUVecchiaFisherScoringOperation(props)],
      outputs
    });
    expectLikelihoodGraphResults(gpu, cpu);
    expect(Array.from(cpu['status'])).toEqual(
      edge === 'all-excluded' ? [0, 5, 0] : edge === 'invalid-parameters' ? [5, 0, 0] : [1, 0, 2]
    );
    expect(cpu['totals'].every(Number.isNaN)).toBe(true);
  });
}
