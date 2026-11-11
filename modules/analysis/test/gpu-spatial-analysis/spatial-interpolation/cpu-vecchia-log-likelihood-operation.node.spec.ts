// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeVecchiaLogLikelihoodOnCPU,
  CPUVecchiaLogLikelihoodOperation
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-log-likelihood-operation';
import {GPUVecchiaLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood-operation';
import {getGPULocalKrigingParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance';
import {constructNetworkOperationContributor as constructLikelihoodContributor} from '../../utils/network-operation-test-utils';
import fixture from './likelihoods-fixture.json';
import {
  createLikelihoodInput,
  expectLikelihoodFloat,
  expectLikelihoodGraphResults,
  getLikelihoodGraphInputs,
  getLikelihoodProps,
  getLikelihoodShapes,
  makeLikelihoodCPUInputs
} from './likelihoods-operation-test-utils';

for (const oracle of fixture.spatial) {
  it(`matches SciPy dense Gaussian prefix distributions, model ${oracle.modelCode}`, () => {
    const result = computeVecchiaLogLikelihoodOnCPU(createLikelihoodInput(oracle.parameters));
    expect(Array.from(result.status)).toEqual([0, 0, 5]);
    expect(Array.from(result.siteStatus)).toEqual([0, 0, 0, 0, 0]);
    expect(Array.from(result.conditionalStatus)).toEqual([0, 0, 0, 0, 0]);
    for (let row = 0; row < 5; row++) {
      expectLikelihoodFloat(result.conditionals[row * 4], oracle.means[row], 1e-7);
      expectLikelihoodFloat(result.conditionals[row * 4 + 1], oracle.variances[row], 1e-7);
      expect(result.conditionals[row * 4 + 2]).toBeGreaterThan(0);
      expect(result.conditionals[row * 4 + 2]).toBeLessThanOrEqual(1);
      expect(result.conditionals[row * 4 + 3]).toBe(row);
      expectLikelihoodFloat(result.siteTerms[row], oracle.siteTerms[row], 1e-7);
    }
    expect(result.logLikelihood[0] + result.logLikelihood[1]).toBeCloseTo(oracle.logLikelihood, 12);
  });
}

it('compares cancelled means at their term scale while retaining other output tolerances', () => {
  const oracle = fixture.spatial[0];
  const result = computeVecchiaLogLikelihoodOnCPU(createLikelihoodInput(oracle.parameters));
  // Site 2 sums +0.090286958 and -0.090238443. The integrator observed this absolute
  // float32 GPU/CPU discrepancy; both signs must fit the same cancellation-aware budget.
  const reportedError = 1.3202225090935826e-8;
  expect(result.conditionals[8]).not.toBe(0);
  for (const direction of [-1, 1]) {
    const conditionals = result.conditionals.slice();
    conditionals[8] = result.conditionals[8] + direction * reportedError;
    const gpu = {...result, conditionals};
    expect(() => expectLikelihoodGraphResults(gpu, result)).toThrow();
    expect(() =>
      expectLikelihoodGraphResults(gpu, result, oracle.meanTermMagnitudes)
    ).not.toThrow();
  }
  const conditionals = result.conditionals.slice();
  conditionals[8] = 0;
  expect(() =>
    expectLikelihoodGraphResults({...result, conditionals}, result, oracle.meanTermMagnitudes)
  ).toThrow();
  conditionals.set(result.conditionals);
  conditionals[9] *= 1.001;
  expect(() =>
    expectLikelihoodGraphResults({...result, conditionals}, result, oracle.meanTermMagnitudes)
  ).toThrow();
});

it('keeps marginal conditionals for excluded sites and fails an entirely excluded likelihood', () => {
  const input = createLikelihoodInput();
  for (let row = 0; row < 5; row++) input.observations[row * 4 + 3] = 0;
  const result = computeVecchiaLogLikelihoodOnCPU(input);
  expect(Array.from(result.status)).toEqual([0, 5, 0]);
  expect(Array.from(result.siteStatus)).toEqual([1, 1, 1, 1, 1]);
  expect(result.siteTerms.every(Number.isNaN)).toBe(true);
  expect(result.logLikelihood.every(Number.isNaN)).toBe(true);
  for (let row = 0; row < 5; row++)
    expect(Array.from(result.conditionals.subarray(row * 4, row * 4 + 4))).toEqual([0, 1.5, 1, 0]);
});

it('preserves explicit bad-span and singular-solve statuses and rejects inconsistent array lengths', () => {
  const input = createLikelihoodInput();
  const malformed = computeVecchiaLogLikelihoodOnCPU({
    ...input,
    neighborhoodOffsets: new Uint32Array([0, 0, 1, 999, 6, 10])
  });
  expect(malformed.conditionalStatus[2]).toBe(1);
  expect(malformed.conditionalStatus[3]).toBe(1);
  expect(malformed.siteStatus[2]).toBe(2);
  expect(malformed.logLikelihood.every(Number.isNaN)).toBe(true);
  const observations = new Float32Array([1, 0, 0, 1, 2, 0, 0, 1, 3, 0, 0, 1]);
  const singular = computeVecchiaLogLikelihoodOnCPU({
    ...input,
    positions: new Float32Array([0, 0, 1, 0, 2, 0]),
    observations,
    neighborhoodOffsets: new Uint32Array([0, 0, 1, 3]),
    neighborhoods: new Uint32Array([0, 0, 0])
  });
  expect(Array.from(singular.conditionalStatus)).toEqual([0, 0, 2]);
  expect(Array.from(singular.siteStatus)).toEqual([0, 0, 2]);
  expect(Array.from(singular.status)).toEqual([1, 0, 2]);
  expect(Array.from(singular.conditionals.subarray(8))).toEqual([NaN, NaN, 0, 0]);
  for (const [argument, values] of [
    ['observations', new Float32Array(1)],
    ['positions', new Float32Array(1)],
    ['parameters', new Float32Array(1)],
    ['neighborhoodOffsets', new Uint32Array(1)]
  ] as const) {
    expect(() => computeVecchiaLogLikelihoodOnCPU({...input, [argument]: values})).toThrow(
      new RegExp(argument)
    );
  }
});

it('validates CPU props, GPU deferred construction and graph output aliases', () => {
  const props = getLikelihoodProps();
  for (const Operation of [GPUVecchiaLogLikelihoodOperation, CPUVecchiaLogLikelihoodOperation]) {
    expect(() => new Operation({...props, rowCount: 0})).toThrow(/likelihood-test/);
    expect(
      () => new Operation({...props, output: {...props.output, siteTerms: 'positions'}})
    ).toThrow(/likelihood-test/);
  }
  expect(() => new CPUVecchiaLogLikelihoodOperation({...props, neighborCount: 0})).toThrow(
    /likelihood-test.*neighborCount/
  );
  expect(() => new CPUVecchiaLogLikelihoodOperation({...props, neighborCount: 33})).not.toThrow();
  expect(() =>
    constructLikelihoodContributor(
      new GPUVecchiaLogLikelihoodOperation({...props, neighborCount: 33}),
      getLikelihoodShapes()
    )
  ).toThrow(/likelihood-test.*neighborCount/);
  expect(() =>
    constructLikelihoodContributor(
      new GPUVecchiaLogLikelihoodOperation(props),
      getLikelihoodShapes()
    )
  ).not.toThrow();
  const shapes = getLikelihoodShapes();
  shapes['observations'].length--;
  expect(() =>
    constructLikelihoodContributor(new GPUVecchiaLogLikelihoodOperation(props), shapes)
  ).toThrow(/likelihood-test.*observations/);
});

it('CPU operation uses every conditioning slot beyond the GPU private-array capacity', async () => {
  // Spherical supports farther apart than the range are independent: every mean is zero and
  // variance two, irrespective of nonuniform observed values. The final row retains 33 slots.
  const rowCount = 34;
  const positions = Float32Array.from({length: rowCount * 2}, (_, index) =>
    index % 2 === 0 ? index : 0
  );
  const observations = Float32Array.from({length: rowCount * 4}, (_, index) =>
    index % 4 === 3 ? 1 : index % 4 === 0 ? index : 0
  );
  const input = {
    ...createLikelihoodInput(),
    positions,
    observations,
    neighborhoodOffsets: Uint32Array.from({length: rowCount + 1}, (_, index) =>
      index === rowCount ? 33 : 0
    ),
    neighborhoods: Uint32Array.from({length: 33}, (_, index) => index),
    parameters: getGPULocalKrigingParameterValues({
      covariance: {model: 'spherical', sill: 2, range: 0.1}
    })
  };
  const operation = new CPUVecchiaLogLikelihoodOperation({
    ...getLikelihoodProps(),
    rowCount,
    neighborCount: 33
  });
  const output = await operation.execute({
    inputs: makeLikelihoodCPUInputs(getLikelihoodGraphInputs(input))
  });
  const conditionals = output['conditionals'] as Float32Array;
  expect(Array.from(conditionals.subarray(33 * 4))).toEqual([0, 2, 1, 33]);
  expect(() =>
    operation.execute({
      inputs: makeLikelihoodCPUInputs(
        getLikelihoodGraphInputs({...input, positions: new Float32Array(2)})
      )
    })
  ).toThrow(/positions/);
});
