// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {CPUVecchiaSpaceTimeLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-space-time-log-likelihood-operation';
import fixture from './likelihoods-fixture.json';
import {
  createLikelihoodInput,
  getLikelihoodGraphInputs,
  getSpaceTimeLikelihoodProps,
  makeLikelihoodCPUInputs
} from './likelihoods-operation-test-utils';
import {expectOracleFloats} from './vecchia-output-oracles-fixture';

for (const oracle of fixture.spaceTime) {
  it(`space-time operation site outputs match SciPy prefix densities, type ${oracle.typeCode}`, async () => {
    // likelihoods-fixture.py independently differences scipy.stats multivariate Gaussian
    // prefix log densities; no implementation/twin supplies expected values.
    const result = await new CPUVecchiaSpaceTimeLogLikelihoodOperation(
      getSpaceTimeLikelihoodProps()
    ).execute({
      inputs: makeLikelihoodCPUInputs(
        getLikelihoodGraphInputs(createLikelihoodInput(oracle.parameters), true)
      )
    });
    expectOracleFloats(result['siteTerms'] as Float32Array, oracle.siteTerms);
    expect(Array.from(result['conditionalStatus'] as Uint32Array)).toEqual([0, 0, 0, 0, 0]);
    expect(Array.from(result['siteStatus'] as Uint32Array)).toEqual([0, 0, 0, 0, 0]);
  });
}

it('invalid space-time scale publishes nonzero conditional and site failure codes', async () => {
  const input = createLikelihoodInput(fixture.spaceTime[0].parameters);
  input.parameters.fill(0);
  const result = await new CPUVecchiaSpaceTimeLogLikelihoodOperation(
    getSpaceTimeLikelihoodProps()
  ).execute({inputs: makeLikelihoodCPUInputs(getLikelihoodGraphInputs(input, true))});
  expect(Array.from(result['conditionalStatus'] as Uint32Array)).toEqual([1, 1, 1, 1, 1]);
  expect(Array.from(result['siteStatus'] as Uint32Array)).toEqual([2, 2, 2, 2, 2]);
  expect(Array.from(result['status'] as Uint32Array)).toEqual([5, 0, 0]);
  expect(Array.from(result['siteTerms'] as Float32Array).every(Number.isNaN)).toBe(true);
});
