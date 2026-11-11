// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeContinuousSpatialSampleOnCPU,
  CPUContinuousSpatialSampleOperation,
  type CPUContinuousSpatialSampleOptions
} from '../../../src/gpu-spatial-analysis/point-process/cpu-continuous-spatial-sample-operation';
import {
  GPUContinuousSpatialSampleOperation,
  type ContinuousSpatialSampleOperationProps
} from '../../../src/gpu-spatial-analysis/point-process/gpu-continuous-spatial-sample-operation';
import fixtures from './continuous-spatial-sample-operation-fixtures.json';
import {
  constructSamplingContributor,
  createSamplingCPUInputs
} from './validation-sampling-operation-test-utils';

it('the CPU callback preserves graph shapes for both legacy encodings and resident random parameters', async () => {
  for (const fixture of fixtures) {
    const area = fixture.frameType === 'area';
    const operation = new CPUContinuousSpatialSampleOperation({
      ...PROPS,
      frameRowCount: fixture.positions.length / 2,
      legacySiteCount: fixture.legacySites.length / (area ? 2 : 1),
      spacing: fixture.spacing,
      waves: fixture.waves,
      capacity: fixture.capacity,
      parameters: 'parameters',
      frame: area
        ? {
            type: 'area',
            positions: 'positions',
            legacySites: 'legacySites',
            bounds: fixture.bounds as [number, number, number, number]
          }
        : {type: 'line', positions: 'positions', legacySites: 'legacySites'}
    });
    const result = await operation.execute({
      inputs: createSamplingCPUInputs({
        positions: {format: 'float32x2', values: new Float32Array(fixture.positions)},
        legacySites: {
          format: area ? 'float32x2' : 'float32',
          values: new Float32Array(fixture.legacySites)
        },
        parameters: {format: 'uint32', values: new Uint32Array([fixture.seed, fixture.replicate])}
      })
    });
    expect(result).toEqual({
      selected: new Float32Array(fixture.expectedPositions),
      weights: new Float32Array(fixture.expectedWeights),
      provenance: new Uint32Array(fixture.expectedProvenance),
      status: new Uint32Array(fixture.expectedStatus)
    });
  }
});

const PROPS: ContinuousSpatialSampleOperationProps = {
  id: 'continuous-test',
  frameRowCount: 3,
  frame: {type: 'line', positions: 'positions'},
  spacing: 0.5,
  seed: 79,
  capacity: 16,
  output: {
    positions: 'selected',
    designWeights: 'weights',
    provenance: 'provenance',
    status: 'status'
  }
};

it('matches Shapely interpolation and triangle-union coverage with legacy exclusion and two waves', () => {
  for (const fixture of fixtures) {
    const options: CPUContinuousSpatialSampleOptions = {
      ...fixture,
      frame:
        fixture.frameType === 'line'
          ? {type: 'line'}
          : {type: 'area', bounds: fixture.bounds as [number, number, number, number]}
    };
    const result = computeContinuousSpatialSampleOnCPU(
      {
        positions: new Float32Array(fixture.positions),
        legacySites: new Float32Array(fixture.legacySites)
      },
      options
    );
    expect(result.positions).toEqual(new Float32Array(fixture.expectedPositions));
    expect(result.designWeights).toEqual(new Float32Array(fixture.expectedWeights));
    expect(Array.from(result.provenance)).toEqual(fixture.expectedProvenance);
    expect(Array.from(result.status)).toEqual(fixture.expectedStatus);
    expect(
      computeContinuousSpatialSampleOnCPU(
        {
          positions: new Float32Array(fixture.positions),
          legacySites: new Float32Array(fixture.legacySites),
          parameters: new Uint32Array([fixture.seed, fixture.replicate])
        },
        {...options, seed: 0, replicate: 0}
      )
    ).toEqual(result);
    const truncated = computeContinuousSpatialSampleOnCPU(
      {
        positions: new Float32Array(fixture.positions),
        legacySites: new Float32Array(fixture.legacySites)
      },
      {...options, capacity: 2, maximumCandidates: 1}
    );
    for (let wave = 0; wave < fixture.waves; wave++) {
      expect(Array.from(truncated.status.subarray(wave * 4, wave * 4 + 4))).toEqual([
        2,
        fixture.expectedStatus[wave * 4 + 1],
        fixture.expectedStatus[wave * 4 + 2],
        2
      ]);
      expect(truncated.positions.subarray(wave * 4, wave * 4 + 4)).toEqual(
        result.positions.subarray(wave * fixture.capacity * 2, wave * fixture.capacity * 2 + 4)
      );
      expect(truncated.designWeights.subarray(wave * 2, wave * 2 + 2)).toEqual(
        result.designWeights.subarray(wave * fixture.capacity, wave * fixture.capacity + 2)
      );
      expect(truncated.provenance.subarray(wave * 8, wave * 8 + 8)).toEqual(
        result.provenance.subarray(wave * fixture.capacity * 4, wave * fixture.capacity * 4 + 8)
      );
    }
  }
});

it('uses a closed-form lattice count and a constant ordinate on a horizontal interval', () => {
  const options = {...PROPS, frame: {type: 'line' as const}, spacing: 0.25, waves: 3, capacity: 10};
  const result = computeContinuousSpatialSampleOnCPU(
    {positions: new Float32Array([0, 2, 2, 2])},
    options
  );
  // Length / spacing is the integer 8: every phase in [0, spacing) yields exactly eight rows.
  for (let wave = 0; wave < 3; wave++) {
    expect(Array.from(result.status.subarray(wave * 4, wave * 4 + 4))).toEqual([8, 8, 8, 0]);
    for (let sample = 0; sample < 8; sample++) {
      const slot = wave * 10 + sample;
      expect(result.positions[slot * 2 + 1]).toBe(2);
      expect(result.designWeights[slot]).toBe(0.25);
      if (sample > 0)
        expect(result.positions[slot * 2] - result.positions[(slot - 1) * 2]).toBeCloseTo(0.25, 6);
    }
  }
});

it('writes four exactly representable half-unit weights on a length-two line', () => {
  const result = computeContinuousSpatialSampleOnCPU(
    {positions: new Float32Array([0, 0, 2, 0])},
    {
      ...PROPS,
      frame: {type: 'line'},
      spacing: 0.5,
      seed: 1,
      capacity: 4
    }
  );
  // Every phase in [0, 0.5) gives four lattice sites, each with inverse intensity 0.5.
  expect(Array.from(result.designWeights)).toEqual([0.5, 0.5, 0.5, 0.5]);
  expect(Array.from(result.status)).toEqual([4, 4, 4, 0]);
});

it('publishes invalid flags and initialized unused slots for collapsed frames and replicate overflow', () => {
  for (const frame of [
    {type: 'line' as const},
    {type: 'area' as const, bounds: [0, 0, 1, 1] as const}
  ]) {
    const result = computeContinuousSpatialSampleOnCPU(
      {positions: new Float32Array(6)},
      {...PROPS, frame, capacity: 2}
    );
    expect(Array.from(result.status)).toEqual([0, 0, 0, 1]);
    expect(Array.from(result.positions)).toEqual([0, 0, 0, 0]);
    expect(Array.from(result.designWeights)).toEqual([0, 0]);
    expect(Array.from(result.provenance)).toEqual([
      0xffffffff, 0, 0xffffffff, 0xffffffff, 0xffffffff, 0, 0xffffffff, 0xffffffff
    ]);
  }
  const result = computeContinuousSpatialSampleOnCPU(
    {positions: new Float32Array([0, 0, 1, 0]), parameters: new Uint32Array([1, 0xffffffff])},
    {...PROPS, frame: {type: 'line'}, waves: 2}
  );
  expect(Array.from(result.status.subarray(4))).toEqual([0, 0, 0, 1]);
  const badLegacy = computeContinuousSpatialSampleOnCPU(
    {positions: new Float32Array([0, 0, 1, 0]), legacySites: new Float32Array([0, 0.25])},
    {...PROPS, frame: {type: 'line'}}
  );
  expect(Array.from(badLegacy.status)).toEqual([0, 0, 0, 1]);
});

it('rejects malformed arrays and validates CPU and deferred GPU construction', () => {
  const input = {positions: new Float32Array([0, 0, 1, 0])};
  const options = {...PROPS, frame: {type: 'line' as const}};
  expect(() =>
    computeContinuousSpatialSampleOnCPU({positions: new Float32Array(3)}, options)
  ).toThrow(/positions/);
  expect(() =>
    computeContinuousSpatialSampleOnCPU({...input, parameters: new Uint32Array(1)}, options)
  ).toThrow(/parameters/);
  expect(() =>
    computeContinuousSpatialSampleOnCPU(
      {positions: new Float32Array(6), legacySites: new Float32Array(3)},
      {...options, frame: {type: 'area', bounds: [0, 0, 1, 1]}}
    )
  ).toThrow(/legacySites/);
  const shapes = {positions: {format: 'float32x2' as const, length: 3}};
  expect(() =>
    constructSamplingContributor(new GPUContinuousSpatialSampleOperation(PROPS), shapes)
  ).not.toThrow();
  expect(() => new CPUContinuousSpatialSampleOperation(PROPS)).not.toThrow();
  for (const change of [
    {spacing: 0},
    {waves: 0},
    {capacity: 0},
    {seed: -1},
    {waves: 2, replicate: 0xffffffff}
  ]) {
    expect(() => new CPUContinuousSpatialSampleOperation({...PROPS, ...change})).toThrow(
      /continuous-test/
    );
    expect(() =>
      constructSamplingContributor(
        new GPUContinuousSpatialSampleOperation({...PROPS, ...change}),
        shapes
      )
    ).toThrow(/continuous-test/);
  }
  for (const Operation of [
    GPUContinuousSpatialSampleOperation,
    CPUContinuousSpatialSampleOperation
  ])
    expect(
      () => new Operation({...PROPS, output: {...PROPS.output, positions: 'positions'}})
    ).toThrow(/continuous-test/);
  expect(
    () =>
      new CPUContinuousSpatialSampleOperation({
        ...PROPS,
        frameRowCount: 4097,
        waves: 4097,
        capacity: 65537
      })
  ).not.toThrow();
});
