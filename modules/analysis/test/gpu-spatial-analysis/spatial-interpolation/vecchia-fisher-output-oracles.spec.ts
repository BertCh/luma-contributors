// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUVecchiaFisherScoringOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring-operation';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createPoisonedOracleGraph,
  runPoisonedOracleGraph
} from '../spatial-clustering/output-oracle-test-utils';
import fixture from './likelihoods-fixture.json';
import {
  createLikelihoodInput,
  getLikelihoodGraphInputs,
  getFisherProps
} from './likelihoods-operation-test-utils';
import {expectOracleFloats} from './vecchia-output-oracles-fixture';

for (const oracle of fixture.spatial) {
  it(`Fisher contributor and operation overwrite every SciPy site derivative cell, model ${oracle.modelCode}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const inputs = Object.fromEntries(
      Object.entries(getLikelihoodGraphInputs(createLikelihoodInput(oracle.parameters))).map(
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
      siteTerms: {format: 'float32', length: 50},
      siteStatus: {format: 'uint32', length: 5},
      totals: {format: 'float32', length: 20},
      status: {format: 'uint32', length: 3}
    });
    harness.graph.add(
      new GPUVecchiaFisherScoringOperation(getFisherProps()).createContributor(harness.getView)
    );
    const results = await runPoisonedOracleGraph(harness, device, submission => {
      if (submission === 1) harness.buffers['parameters'].write(new Float32Array(8));
    });
    for (const [submission, result] of results.entries()) {
      if (submission === 1) {
        expect(result['siteStatus']).toEqual([2, 2, 2, 2, 2]);
        expect(result['status']).toEqual([5, 0, 0]);
        expect(result['siteTerms'].every(Number.isNaN)).toBe(true);
        continue;
      }
      // All ten cells of every row are independently pinned, including tail row and zeros.
      expectOracleFloats(result['siteTerms'], oracle.fisherSiteTerms);
      expect(result['siteStatus']).toEqual([0, 0, 0, 0, 0]);
      expect(result['status']).toEqual([0, 0, 5]);
    }
  });
}
