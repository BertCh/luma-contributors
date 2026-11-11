// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeVecchiaSpaceTimeLogLikelihoodOnCPU,
  CPUVecchiaSpaceTimeLogLikelihoodOperation
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-space-time-log-likelihood-operation';
import {GPUVecchiaSpaceTimeLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood-operation';
import {constructNetworkOperationContributor as constructLikelihoodContributor} from '../../utils/network-operation-test-utils';
import fixture from './likelihoods-fixture.json';
import {
  createLikelihoodInput,
  createStructureCodeLikelihoodInput,
  expectLikelihoodFloat,
  getSpaceTimeLikelihoodProps,
  getLikelihoodShapes,
  getLikelihoodGraphInputs,
  makeLikelihoodCPUInputs
} from './likelihoods-operation-test-utils';

for (const oracle of fixture.spaceTime) {
  it(`matches SciPy dense Gaussian prefix distributions, space-time type ${oracle.typeCode}`, () => {
    const result = computeVecchiaSpaceTimeLogLikelihoodOnCPU(
      createLikelihoodInput(oracle.parameters)
    );
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

it('clamps finite negative structure codes to spherical covariance in accepted hand-written buffers', () => {
  const result = computeVecchiaSpaceTimeLogLikelihoodOnCPU(createStructureCodeLikelihoodInput());
  // Spherical correlation is zero at distance/range=1. The two unit-variance sites are
  // independent: means zero, variances one, and log L=-log(2*pi)-(2^2+1^2)/2.
  expect(result.conditionals).toEqual(new Float32Array([0, 1, 1, 0, 0, 1, 1, 1]));
  expect(Array.from(result.conditionalStatus)).toEqual([0, 0]);
  expect(Array.from(result.siteStatus)).toEqual([0, 0]);
  expect(Array.from(result.status)).toEqual([0, 0, 2]);
  expect(result.siteTerms).toEqual(
    Float32Array.of(-0.5 * (Math.log(2 * Math.PI) + 4), -0.5 * (Math.log(2 * Math.PI) + 1))
  );
  const total = -Math.log(2 * Math.PI) - 2.5;
  const high = Math.fround(total);
  expect(result.logLikelihood).toEqual(Float32Array.of(high, total - high));
});

it('documents the deterministic CPU default for a NaN code without comparing an indeterminate GPU conversion', () => {
  const result = computeVecchiaSpaceTimeLogLikelihoodOnCPU(createStructureCodeLikelihoodInput(NaN));
  // The Matern 5/2 default has correlation exp(-3) at its effective range of one.
  expectLikelihoodFloat(result.conditionals[4], 2 * Math.exp(-3), 1e-7);
  expectLikelihoodFloat(result.conditionals[5], 1 - Math.exp(-6), 1e-7);
  expect(Array.from(result.status)).toEqual([0, 0, 2]);
});

it('drops unusable conditioning rows, ignores missing IDs, and preserves excluded site terms', () => {
  const input = createLikelihoodInput(fixture.spaceTime[0].parameters);
  input.observations[3] = 0;
  input.neighborhoods[2] = 0xffffffff;
  const result = computeVecchiaSpaceTimeLogLikelihoodOnCPU(input);
  expect(Array.from(result.status)).toEqual([0, 1, 4]);
  expect(Array.from(result.siteStatus)).toEqual([1, 0, 0, 0, 0]);
  expect(Number.isNaN(result.siteTerms[0])).toBe(true);
  // Row 1 has only excluded row 0; row 2 has excluded row 0 and missing row ID.
  expect(Array.from(result.conditionals.subarray(4, 12))).toEqual([0, 1.5, 1, 0, 0, 1.5, 1, 0]);
  input.observations.fill(0);
  const empty = computeVecchiaSpaceTimeLogLikelihoodOnCPU(input);
  expect(Array.from(empty.status)).toEqual([0, 5, 0]);
  expect(empty.logLikelihood.every(Number.isNaN)).toBe(true);
});

it('rejects malformed arrays, preserves invalid-buffer failure and checks operation execution lengths', async () => {
  const input = createLikelihoodInput(fixture.spaceTime[0].parameters);
  for (const [argument, values] of [
    ['observations', new Float32Array(1)],
    ['positions', new Float32Array(1)],
    ['times', new Float32Array(1)],
    ['parameters', new Float32Array(1)],
    ['neighborhoodOffsets', new Uint32Array([0, 0, 1, 100, 6, 10])]
  ] as const) {
    expect(() => computeVecchiaSpaceTimeLogLikelihoodOnCPU({...input, [argument]: values})).toThrow(
      new RegExp(argument)
    );
  }
  const result = computeVecchiaSpaceTimeLogLikelihoodOnCPU({
    ...input,
    parameters: new Float32Array(28)
  });
  expect(Array.from(result.status)).toEqual([5, 0, 0]);
  expect(Array.from(result.siteStatus)).toEqual([2, 2, 2, 2, 2]);
  const operation = new CPUVecchiaSpaceTimeLogLikelihoodOperation(getSpaceTimeLikelihoodProps());
  const output = await operation.execute({
    inputs: makeLikelihoodCPUInputs(getLikelihoodGraphInputs(input, true))
  });
  expect(output['siteTerms']).toEqual(computeVecchiaSpaceTimeLogLikelihoodOnCPU(input).siteTerms);
});

it('validates both constructions and rejects unsupported temporal encoding in deferred GPU construction', () => {
  const props = getSpaceTimeLikelihoodProps();
  for (const Operation of [
    GPUVecchiaSpaceTimeLogLikelihoodOperation,
    CPUVecchiaSpaceTimeLogLikelihoodOperation
  ]) {
    expect(() => new Operation({...props, rowCount: 0})).toThrow(/likelihood-test/);
    expect(() => new Operation({...props, output: {...props.output, status: 'positions'}})).toThrow(
      /likelihood-test/
    );
  }
  const invalid = {
    ...props,
    temporalContext: {...props.temporalContext, encoding: 'double-single' as const}
  };
  expect(() => new CPUVecchiaSpaceTimeLogLikelihoodOperation(invalid)).toThrow(
    /likelihood-test.*double-single/
  );
  expect(() =>
    constructLikelihoodContributor(
      new GPUVecchiaSpaceTimeLogLikelihoodOperation(invalid),
      getLikelihoodShapes(true)
    )
  ).toThrow(/likelihood-test.*double-single/);
  expect(() => new CPUVecchiaSpaceTimeLogLikelihoodOperation({...props, neighborCount: 0})).toThrow(
    /likelihood-test.*neighborCount/
  );
  expect(() =>
    constructLikelihoodContributor(
      new GPUVecchiaSpaceTimeLogLikelihoodOperation({...props, neighborCount: 33}),
      getLikelihoodShapes(true)
    )
  ).toThrow(/likelihood-test.*neighborCount/);
  expect(
    () => new CPUVecchiaSpaceTimeLogLikelihoodOperation({...props, neighborCount: 33})
  ).not.toThrow();
  expect(() =>
    constructLikelihoodContributor(
      new GPUVecchiaSpaceTimeLogLikelihoodOperation(props),
      getLikelihoodShapes(true)
    )
  ).not.toThrow();
});
