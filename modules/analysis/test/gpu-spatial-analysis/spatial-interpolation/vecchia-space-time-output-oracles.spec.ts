// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUVecchiaSpaceTimeLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood-operation';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createPoisonedOracleGraph,
  runPoisonedOracleGraph
} from '../spatial-clustering/output-oracle-test-utils';
import fixture from './likelihoods-fixture.json';
import {
  createLikelihoodInput,
  getLikelihoodGraphInputs,
  getSpaceTimeLikelihoodProps
} from './likelihoods-operation-test-utils';
import {expectOracleFloats} from './vecchia-output-oracles-fixture';

for (const oracle of fixture.spaceTime) {
  it(`space-time contributor and operation write SciPy site literals, type ${oracle.typeCode}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const inputs = Object.fromEntries(
      Object.entries(getLikelihoodGraphInputs(createLikelihoodInput(oracle.parameters), true)).map(
        ([name, input]) => [
          name,
          {
            ...input,
            values: input.values as Float32Array | Uint32Array,
            length:
              input.values.length / (name === 'positions' ? 2 : name === 'observations' ? 4 : 1)
          }
        ]
      )
    );
    const harness = createPoisonedOracleGraph(device, inputs, {
      conditionals: {format: 'float32x4', length: 5},
      conditionalStatus: {format: 'uint32', length: 5},
      siteTerms: {format: 'float32', length: 5},
      siteStatus: {format: 'uint32', length: 5},
      logLikelihood: {format: 'float32', length: 2},
      status: {format: 'uint32', length: 3}
    });
    harness.graph.add(
      new GPUVecchiaSpaceTimeLogLikelihoodOperation(
        getSpaceTimeLikelihoodProps()
      ).createContributor(harness.getView)
    );
    const results = await runPoisonedOracleGraph(harness, device, submission => {
      if (submission === 1) harness.buffers['parameters'].write(new Float32Array(28));
    });
    for (const [submission, result] of results.entries()) {
      if (submission === 1) {
        expect(result['conditionalStatus']).toEqual([1, 1, 1, 1, 1]);
        expect(result['siteStatus']).toEqual([2, 2, 2, 2, 2]);
        expect(result['status']).toEqual([5, 0, 0]);
        expect(result['siteTerms'].every(Number.isNaN)).toBe(true);
        continue;
      }
      // Independent SciPy prefix Gaussian densities committed in likelihoods-fixture.json;
      // 2e-5 relative plus 1e-6 absolute covers these well-conditioned float32 solves/logs.
      expectOracleFloats(result['siteTerms'], oracle.siteTerms);
      expect(result['conditionalStatus']).toEqual([0, 0, 0, 0, 0]);
      expect(result['siteStatus']).toEqual([0, 0, 0, 0, 0]);
      expect(result['status']).toEqual([0, 0, 5]);
    }
  });
}
