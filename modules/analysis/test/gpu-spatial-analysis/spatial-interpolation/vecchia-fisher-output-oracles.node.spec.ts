// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  CPUVecchiaFisherScoringOperation,
  computeVecchiaFisherScoringOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-fisher-scoring-operation';
import fixture from './likelihoods-fixture.json';
import {
  createLikelihoodInput,
  getLikelihoodGraphInputs,
  getFisherProps,
  makeLikelihoodCPUInputs
} from './likelihoods-operation-test-utils';
import {expectOracleFloats} from './vecchia-output-oracles-fixture';

for (const oracle of fixture.spatial) {
  it(`Fisher operation and host function match SciPy prefix derivative cells, model ${oracle.modelCode}`, async () => {
    const input = createLikelihoodInput(oracle.parameters);
    const results = [
      computeVecchiaFisherScoringOnCPU(input),
      await new CPUVecchiaFisherScoringOperation(getFisherProps()).execute({
        inputs: makeLikelihoodCPUInputs(getLikelihoodGraphInputs(input))
      })
    ];
    for (const result of results) {
      // likelihoods-fixture.py uses central differences of independent SciPy logpdf and
      // prefix differences of 1/2 trace(C^-1 C_j C^-1 C_k), not conditional derivatives.
      expectOracleFloats(result['siteTerms'] as Float32Array, oracle.fisherSiteTerms);
      expect(Array.from(result['siteStatus'] as Uint32Array)).toEqual([0, 0, 0, 0, 0]);
    }
  });
}

it('invalid Fisher scale publishes a nonzero failure code for every site', async () => {
  const input = createLikelihoodInput();
  input.parameters.fill(0);
  const result = await new CPUVecchiaFisherScoringOperation(getFisherProps()).execute({
    inputs: makeLikelihoodCPUInputs(getLikelihoodGraphInputs(input))
  });
  expect(Array.from(result['siteStatus'] as Uint32Array)).toEqual([2, 2, 2, 2, 2]);
  expect(Array.from(result['status'] as Uint32Array)).toEqual([5, 0, 0]);
  expect(Array.from(result['siteTerms'] as Float32Array).every(Number.isNaN)).toBe(true);
});
