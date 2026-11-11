// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeSpatialBalancedSampleOnCPU,
  CPUSpatialBalancedSampleOperation
} from '../../../src/gpu-spatial-analysis/point-process/cpu-spatial-balanced-sample-operation';
import {
  GPUSpatialBalancedSampleOperation,
  type SpatialBalancedSampleOperationProps
} from '../../../src/gpu-spatial-analysis/point-process/gpu-spatial-balanced-sample-operation';
import fixtures from './spatial-balanced-sample-operation-fixtures.json';
import {
  constructSamplingContributor,
  createSamplingCPUInputs
} from './validation-sampling-operation-test-utils';

it('the CPU callback reads every optional port and returns the declared graph values', async () => {
  const fixture = fixtures[1];
  const operation = new CPUSpatialBalancedSampleOperation({
    ...PROPS,
    parameters: 'parameters',
    inclusionTickets: 'tickets',
    allocations: fixture.allocations
  });
  const result = await operation.execute({
    inputs: createSamplingCPUInputs({
      positions: {format: 'float32x2', values: new Float32Array(fixture.positions)},
      frameIds: {format: 'uint32', values: new Uint32Array(fixture.frameIds)},
      strata: {format: 'uint32', values: new Uint32Array(fixture.strata)},
      inclusionTickets: {format: 'uint32', values: new Uint32Array(fixture.tickets!)},
      parameters: {format: 'uint32', values: new Uint32Array([fixture.seed, fixture.replicate])}
    })
  });
  expect(result).toEqual({
    selected: new Uint32Array(fixture.expectedIds),
    probabilities: new Float32Array(fixture.expectedProbabilities),
    weights: new Float32Array(fixture.expectedWeights),
    status: new Uint32Array(fixture.expectedStatus)
  });
});

const PROPS: SpatialBalancedSampleOperationProps = {
  id: 'balanced-test',
  rowCount: 12,
  positions: 'positions',
  frameIds: 'frameIds',
  strata: 'strata',
  bounds: [0, 0, 1, 1],
  seed: 7,
  allocations: [
    {stratumId: 9, sampleCount: 2},
    {stratumId: 2, sampleCount: 3}
  ],
  output: {
    frameIds: 'selected',
    inclusionProbabilities: 'probabilities',
    designWeights: 'weights',
    status: 'status'
  }
};

it('matches an independent recursive-tree and expanded-population Python oracle', () => {
  for (const fixture of fixtures) {
    const input = {
      positions: new Float32Array(fixture.positions),
      frameIds: new Uint32Array(fixture.frameIds),
      strata: new Uint32Array(fixture.strata),
      inclusionTickets: fixture.tickets ? new Uint32Array(fixture.tickets) : undefined,
      parameters: new Uint32Array([fixture.seed, fixture.replicate])
    };
    const result = computeSpatialBalancedSampleOnCPU(input, {
      ...PROPS,
      allocations: fixture.allocations
    });
    expect(Array.from(result.frameIds)).toEqual(fixture.expectedIds);
    expect(result.inclusionProbabilities).toEqual(new Float32Array(fixture.expectedProbabilities));
    expect(result.designWeights).toEqual(new Float32Array(fixture.expectedWeights));
    expect(Array.from(result.status)).toEqual(fixture.expectedStatus);
    const constructorParameters = computeSpatialBalancedSampleOnCPU(
      {...input, parameters: undefined},
      {
        ...PROPS,
        allocations: fixture.allocations,
        seed: fixture.seed,
        replicate: fixture.replicate
      }
    );
    expect(constructorParameters).toEqual(result);
  }
});

it('breaks coincident hierarchy ties by row and returns a census with unit weights', () => {
  const result = computeSpatialBalancedSampleOnCPU(
    {positions: new Float32Array(8), frameIds: new Uint32Array([9, 2, 7, 1])},
    {
      ...PROPS,
      allocations: [{stratumId: 0, sampleCount: 4}]
    }
  );
  // A circular census must contain exactly the row-order cycle, regardless of its phase.
  const selected = Array.from(result.frameIds);
  const first = [9, 2, 7, 1].indexOf(selected[0]);
  expect(selected).toEqual(
    Array.from({length: 4}, (_, index) => [9, 2, 7, 1][(first + index) % 4])
  );
  expect(Array.from(result.inclusionProbabilities)).toEqual([1, 1, 1, 1]);
  expect(Array.from(result.designWeights)).toEqual([1, 1, 1, 1]);
  expect(Array.from(result.status)).toEqual([4, 0]);
});

it('invalidates the whole frame for an unknown stratum but only the affected ticket allocation', () => {
  const fixture = fixtures[1];
  const input = {
    positions: new Float32Array(fixture.positions),
    frameIds: new Uint32Array(fixture.frameIds),
    strata: new Uint32Array(fixture.strata),
    inclusionTickets: new Uint32Array(fixture.tickets!)
  };
  input.inclusionTickets[0] = 0;
  const result = computeSpatialBalancedSampleOnCPU(input, {
    ...PROPS,
    allocations: fixture.allocations
  });
  expect(Array.from(result.status)).toEqual([2, 0, 0, 1]);
  expect(Array.from(result.frameIds.subarray(2))).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
  expect(result.inclusionProbabilities.subarray(2).every(Number.isNaN)).toBe(true);
  expect(result.designWeights.subarray(2).every(Number.isNaN)).toBe(true);
  input.strata[0] = 123;
  const invalid = computeSpatialBalancedSampleOnCPU(input, PROPS);
  expect(Array.from(invalid.status)).toEqual([0, 1, 0, 1]);
  expect(invalid.frameIds.every(value => value === 0xffffffff)).toBe(true);
});

it('rejects malformed arrays and validates both operation constructions', () => {
  const input = {positions: new Float32Array(24), frameIds: new Uint32Array(12)};
  for (const [argument, value] of Object.entries({
    positions: new Float32Array(22),
    strata: new Uint32Array(11),
    inclusionTickets: new Uint32Array(13),
    parameters: new Uint32Array(1)
  }))
    expect(() =>
      computeSpatialBalancedSampleOnCPU(
        {...input, [argument]: value},
        {...PROPS, allocations: [{stratumId: 0, sampleCount: 1}]}
      )
    ).toThrow(new RegExp(argument));
  const shapes = {
    positions: {format: 'float32x2' as const, length: 12},
    frameIds: {format: 'uint32' as const, length: 12},
    strata: {format: 'uint32' as const, length: 12}
  };
  expect(() =>
    constructSamplingContributor(new GPUSpatialBalancedSampleOperation(PROPS), shapes)
  ).not.toThrow();
  expect(() => new CPUSpatialBalancedSampleOperation(PROPS)).not.toThrow();
  for (const change of [
    {seed: -1},
    {bounds: [0, 0, 0, 1] as const},
    {allocations: [{stratumId: 0x1000000, sampleCount: 1}]}
  ]) {
    expect(() => new CPUSpatialBalancedSampleOperation({...PROPS, ...change})).toThrow(
      /balanced-test/
    );
    expect(() =>
      constructSamplingContributor(
        new GPUSpatialBalancedSampleOperation({...PROPS, ...change}),
        shapes
      )
    ).toThrow(/balanced-test/);
  }
  for (const Operation of [GPUSpatialBalancedSampleOperation, CPUSpatialBalancedSampleOperation])
    expect(
      () => new Operation({...PROPS, output: {...PROPS.output, frameIds: PROPS.positions}})
    ).toThrow(/balanced-test/);
  expect(() => new CPUSpatialBalancedSampleOperation({...PROPS, rowCount: 4097})).not.toThrow();
});
