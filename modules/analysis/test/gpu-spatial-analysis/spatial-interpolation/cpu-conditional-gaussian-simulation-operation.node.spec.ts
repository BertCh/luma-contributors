// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeConditionalGaussianSimulationOnCPU,
  CPUConditionalGaussianSimulationOperation,
  type CPUConditionalGaussianSimulationOptions
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-conditional-gaussian-simulation-operation';
import {
  GPUConditionalGaussianSimulationOperation,
  type ConditionalGaussianSimulationOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation-operation';
import type {GPUPredictionCovarianceModel} from '../../../src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance';
import {constructNetworkOperationContributor as constructSimulationContributor} from '../../utils/network-operation-test-utils';
import {makeLikelihoodCPUInputs, expectLikelihoodFloat} from './likelihoods-operation-test-utils';
import fixture from './conditional-operation-fixture.json';

function getOptions(
  model: GPUPredictionCovarianceModel = 'exponential'
): CPUConditionalGaussianSimulationOptions {
  return {
    covariance: {model, sill: fixture.sill, nugget: fixture.nugget, range: fixture.range},
    mean: fixture.mean,
    seed: fixture.seed,
    replicateCount: fixture.replicateCount,
    replicateOffset: fixture.replicateOffset
  };
}
function createInput() {
  return {
    positions: Float32Array.from(fixture.positions),
    observations: Float32Array.from(fixture.observations),
    queries: Float32Array.from(fixture.queries)
  };
}
function getProps(): ConditionalGaussianSimulationOperationProps {
  return {
    ...getOptions(),
    id: 'simulation-test',
    observationCount: 3,
    queryCount: 4,
    positions: 'positions',
    observations: 'observations',
    queries: 'queries',
    output: {values: 'values', moments: 'moments', status: 'status'}
  };
}
const SHAPES = {
  positions: {format: 'float32x2' as const, length: 3},
  observations: {format: 'float32x4' as const, length: 3},
  queries: {format: 'float32x2' as const, length: 4}
};

it('rejects retained preparation on both simulation operation forms (LIF-09)', () => {
  for (const Operation of [
    CPUConditionalGaussianSimulationOperation,
    GPUConditionalGaussianSimulationOperation
  ]) {
    const untyped = {...getProps(), reusePreparation: true};
    expect(() => new Operation(untyped)).toThrow(
      /simulation-test.*reusePreparation.*GPUConditionalGaussianSimulation/
    );
  }
});

for (const oracle of fixture.fixtures) {
  it(`matches SciPy Schur conditioning and independent integer Philox draws for ${oracle.model}`, () => {
    const input = createInput();
    const options = getOptions(oracle.model as GPUPredictionCovarianceModel);
    const result = computeConditionalGaussianSimulationOnCPU(input, options);
    expect(Array.from(result.status)).toEqual([0, 0, 0]);
    oracle.moments.forEach((expected, index) =>
      expectLikelihoodFloat(result.moments[index], expected, 1e-7)
    );
    oracle.values.forEach((expected, index) =>
      expectLikelihoodFloat(result.values[index], expected, 1e-7)
    );
    for (let replicate = 0; replicate < 3; replicate++) {
      const single = computeConditionalGaussianSimulationOnCPU(input, {
        ...options,
        replicateCount: 1,
        replicateOffset: fixture.replicateOffset + replicate
      });
      expect(single.values).toEqual(result.values.subarray(replicate * 4, replicate * 4 + 4));
      expect(single.moments).toEqual(result.moments);
    }
  });
}

it('has unconditional normal moments with no observations and preserves the registered counter stream', () => {
  const options = getOptions();
  const result = computeConditionalGaussianSimulationOnCPU(
    {
      positions: new Float32Array(0),
      observations: new Float32Array(0),
      queries: new Float32Array([1, 2])
    },
    options
  );
  expect(result.moments).toEqual(Float32Array.of(fixture.mean, fixture.sill + fixture.nugget));
  for (let replicate = 0; replicate < 3; replicate++)
    expect(result.values[replicate]).toBe(
      Math.fround(
        fixture.mean + Math.sqrt(fixture.sill + fixture.nugget) * fixture.normals[replicate * 4]
      )
    );
});

it('keeps finite independent results when a Matern argument would overflow a polynomial', () => {
  const options = {
    ...getOptions('matern-5/2'),
    covariance: {model: 'matern-5/2' as const, sill: 1, range: 1e-200}
  };
  const result = computeConditionalGaussianSimulationOnCPU(
    {
      positions: new Float32Array([0, 0]),
      observations: new Float32Array([1, 0, 0, 1]),
      queries: new Float32Array([1, 0])
    },
    options
  );
  expect(Array.from(result.status)).toEqual([0, 0, 0]);
  expect(result.moments).toEqual(Float32Array.of(fixture.mean, 1));
  expect(result.values.every(Number.isFinite)).toBe(true);
});

it('copies noiseless data hits and repeated queries exactly, ignoring excluded malformed observations', () => {
  const options = getOptions('gaussian');
  const result = computeConditionalGaussianSimulationOnCPU(
    {
      positions: new Float32Array([0, 0, NaN, NaN]),
      observations: new Float32Array([2, 0, 0, 1, NaN, -1, 0, 0]),
      queries: new Float32Array([0, 0, 0.5, 0.5, 0.5, 0.5])
    },
    options
  );
  expect(Array.from(result.status)).toEqual([0, 0, 0]);
  expect(result.moments[0]).toBe(2);
  expect(result.moments[1]).toBe(0);
  expect(result.moments.subarray(2, 4)).toEqual(result.moments.subarray(4));
  for (let replicate = 0; replicate < 3; replicate++) {
    expect(result.values[replicate * 3]).toBe(2);
    expect(result.values[replicate * 3 + 1]).toBe(result.values[replicate * 3 + 2]);
  }
});

it('distinguishes invalid input from singular observations, and checks inconsistent arrays/options', () => {
  const input = createInput();
  const options = getOptions();
  const invalid = computeConditionalGaussianSimulationOnCPU(
    {...input, queries: new Float32Array([NaN, 0])},
    options
  );
  expect(Array.from(invalid.status)).toEqual([1, 1, 1]);
  expect(invalid.moments.every(Number.isNaN)).toBe(true);
  const singular = computeConditionalGaussianSimulationOnCPU(
    {
      ...input,
      positions: new Float32Array([0, 0, 0, 0]),
      observations: new Float32Array([1, 0, 0, 1, 2, 0, 0, 1])
    },
    {...options, covariance: {model: 'exponential', sill: 1.25, nugget: 0.25, range: 2}}
  );
  expect(Array.from(singular.status)).toEqual([2, 2, 2]);
  expect(singular.values.every(Number.isNaN)).toBe(true);
  expect(singular.moments.every(Number.isNaN)).toBe(true);
  for (const [argument, values] of [
    ['positions', new Float32Array(1)],
    ['observations', new Float32Array(1)],
    ['queries', new Float32Array(1)]
  ] as const)
    expect(() =>
      computeConditionalGaussianSimulationOnCPU({...input, [argument]: values}, options)
    ).toThrow(new RegExp(argument));
  expect(() => computeConditionalGaussianSimulationOnCPU(input, {...options, seed: -1})).toThrow(
    /seed/
  );
  expect(() =>
    computeConditionalGaussianSimulationOnCPU(input, {...options, replicateOffset: 0xffffffff})
  ).toThrow(/replicateOffset/);
});

it('validates both operation constructions and executes the same graph port layouts', async () => {
  const props = getProps();
  for (const Operation of [
    GPUConditionalGaussianSimulationOperation,
    CPUConditionalGaussianSimulationOperation
  ]) {
    expect(() => new Operation({...props, queryCount: 0})).toThrow(/simulation-test/);
    expect(() => new Operation({...props, output: {...props.output, moments: 'queries'}})).toThrow(
      /simulation-test/
    );
  }
  expect(
    () => new CPUConditionalGaussianSimulationOperation({...props, observationCount: -1})
  ).toThrow(/simulation-test.*observationCount/);
  expect(() => new CPUConditionalGaussianSimulationOperation({...props, seed: -1})).toThrow(
    /simulation-test.*seed/
  );
  expect(
    () =>
      new CPUConditionalGaussianSimulationOperation({
        ...props,
        covariance: {...props.covariance, range: 0}
      })
  ).toThrow(/simulation-test.*range/);
  expect(
    () => new CPUConditionalGaussianSimulationOperation({...props, observationCount: 4096})
  ).not.toThrow();
  expect(() =>
    constructSimulationContributor(new GPUConditionalGaussianSimulationOperation(props), SHAPES)
  ).not.toThrow();
  expect(() =>
    constructSimulationContributor(
      new GPUConditionalGaussianSimulationOperation({...props, seed: -1}),
      SHAPES
    )
  ).toThrow(/simulation-test/);
  expect(() =>
    constructSimulationContributor(new GPUConditionalGaussianSimulationOperation(props), {
      ...SHAPES,
      observations: {...SHAPES.observations, length: 2}
    })
  ).toThrow(/simulation-test/);
  const operation = new CPUConditionalGaussianSimulationOperation(props);
  const input = createInput();
  const output = await operation.execute({
    inputs: makeLikelihoodCPUInputs({
      positions: {format: 'float32x2', values: input.positions},
      observations: {format: 'float32x4', values: input.observations},
      queries: {format: 'float32x2', values: input.queries}
    })
  });
  expect(output['values']).toEqual(computeConditionalGaussianSimulationOnCPU(input, props).values);
});
