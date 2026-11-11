// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeVecchiaFisherScoringOnCPU,
  CPUVecchiaFisherScoringOperation
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-fisher-scoring-operation';
import {GPUVecchiaFisherScoringOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring-operation';
import {getGPULocalKrigingParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance';
import {constructNetworkOperationContributor as constructLikelihoodContributor} from '../../utils/network-operation-test-utils';
import fixture from './likelihoods-fixture.json';
import {
  createLikelihoodInput,
  expectLikelihoodFloat,
  getFisherProps,
  getLikelihoodShapes,
  getLikelihoodGraphInputs,
  makeLikelihoodCPUInputs
} from './likelihoods-operation-test-utils';

for (const oracle of fixture.spatial) {
  it(`matches dense Gaussian finite-difference scores and trace information, model ${oracle.modelCode}`, () => {
    const result = computeVecchiaFisherScoringOnCPU(createLikelihoodInput(oracle.parameters));
    expect(Array.from(result.status)).toEqual([0, 0, 5]);
    expect(Array.from(result.siteStatus)).toEqual([0, 0, 0, 0, 0]);
    // Oracle scores differentiate scipy.stats multivariate_normal.logpdf for full prefix blocks.
    // Expected information uses the dense matrix trace identity; subtract prefixes for site cells.
    oracle.fisherSiteTerms.forEach((expected, index) =>
      expectLikelihoodFloat(result.siteTerms[index], expected, 2e-6)
    );
    oracle.fisherTotals.forEach((expected, cell) =>
      expectLikelihoodFloat(result.totals[cell * 2] + result.totals[cell * 2 + 1], expected, 2e-8)
    );
    // The high/low output must carry more information than its high float32 word alone.
    expect(Math.abs(result.totals[0] + result.totals[1] - oracle.logLikelihood)).toBeLessThan(
      1e-12
    );
    expect(result.totals[1]).not.toBe(0);
  });
}

it('has the closed-form marginal Fisher information and score with fixed observation error', () => {
  const input = {
    ...createLikelihoodInput(),
    positions: new Float32Array([0, 0]),
    observations: new Float32Array([2, 0.5, 0, 1]),
    neighborhoodOffsets: new Uint32Array([0, 0]),
    neighborhoods: new Uint32Array(0),
    parameters: getGPULocalKrigingParameterValues({
      covariance: {model: 'exponential', sill: 1, nugget: 0.5, range: 2}
    })
  };
  const result = computeVecchiaFisherScoringOnCPU(input);
  // y=2, total variance=2, derivatives [1,0,1/2]. Score = derivative/4;
  // expected information = derivative outer product / 8.
  expect(Array.from(result.siteTerms.subarray(1))).toEqual([
    0.25, 0, 0.125, 0.125, 0, 0.0625, 0, 0, 0.03125
  ]);
  expect(Array.from(result.status)).toEqual([0, 0, 1]);
});

it('preserves exclusion and singular statuses and fails every total if any site fails', () => {
  const input = createLikelihoodInput();
  input.observations.fill(0);
  const empty = computeVecchiaFisherScoringOnCPU(input);
  expect(Array.from(empty.status)).toEqual([0, 5, 0]);
  expect(empty.siteTerms.every(Number.isNaN)).toBe(true);
  expect(empty.totals.every(Number.isNaN)).toBe(true);
  const singular = computeVecchiaFisherScoringOnCPU({
    ...createLikelihoodInput(),
    positions: new Float32Array([0, 0, 1, 0, 2, 0]),
    observations: new Float32Array([1, 0, 0, 1, 2, 0, 0, 1, 3, 0, 0, 1]),
    neighborhoodOffsets: new Uint32Array([0, 0, 1, 3]),
    neighborhoods: new Uint32Array([0, 0, 0])
  });
  expect(Array.from(singular.siteStatus)).toEqual([0, 0, 2]);
  expect(Array.from(singular.status)).toEqual([1, 0, 2]);
  expect(singular.totals.every(Number.isNaN)).toBe(true);
});

it('validates arrays and both operation constructions and executes with matching port names', async () => {
  const input = createLikelihoodInput();
  for (const [argument, values] of [
    ['observations', new Float32Array(1)],
    ['positions', new Float32Array(1)],
    ['parameters', new Float32Array(1)],
    ['neighborhoodOffsets', new Uint32Array([0, 0, 1, 100, 6, 10])]
  ] as const) {
    expect(() => computeVecchiaFisherScoringOnCPU({...input, [argument]: values})).toThrow(
      new RegExp(argument)
    );
  }
  const props = getFisherProps();
  for (const Operation of [GPUVecchiaFisherScoringOperation, CPUVecchiaFisherScoringOperation]) {
    expect(() => new Operation({...props, rowCount: 0})).toThrow(/likelihood-test/);
    expect(() => new Operation({...props, output: {...props.output, totals: 'positions'}})).toThrow(
      /likelihood-test/
    );
  }
  expect(() => new CPUVecchiaFisherScoringOperation({...props, neighborCount: 0})).toThrow(
    /likelihood-test.*neighborCount/
  );
  expect(() => new CPUVecchiaFisherScoringOperation({...props, neighborCount: 33})).not.toThrow();
  expect(() =>
    constructLikelihoodContributor(
      new GPUVecchiaFisherScoringOperation({...props, neighborCount: 33}),
      getLikelihoodShapes()
    )
  ).toThrow(/likelihood-test.*neighborCount/);
  expect(() =>
    constructLikelihoodContributor(
      new GPUVecchiaFisherScoringOperation(props),
      getLikelihoodShapes()
    )
  ).not.toThrow();
  const operation = new CPUVecchiaFisherScoringOperation(props);
  const output = await operation.execute({
    inputs: makeLikelihoodCPUInputs(getLikelihoodGraphInputs(input))
  });
  expect(output['totals']).toEqual(computeVecchiaFisherScoringOnCPU(input).totals);
});
