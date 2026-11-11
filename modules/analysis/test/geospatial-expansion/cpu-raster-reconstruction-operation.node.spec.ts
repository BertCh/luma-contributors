// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeRasterReconstructionOnCPU,
  CPURasterReconstructionOperation
} from '../../src/raster-analysis/raster-morphology/cpu-raster-reconstruction-operation';
import {
  GPURasterReconstructionOperation,
  type RasterReconstructionOperationProps
} from '../../src/raster-analysis/raster-morphology/gpu-raster-reconstruction-operation';
import {constructRasterOperationContributor} from '../utils/raster-operation-test-utils';
import fixtures from './raster-morphology-operation-fixtures.json';

const PROPS: RasterReconstructionOperationProps = {
  id: 'reconstruct',
  grid: {width: 5, height: 1, affine: [1, 0, 0, 0, -1, 1], pixelInterpretation: 'area'},
  marker: 'marker',
  mask: 'mask',
  output: {
    values: 'values',
    validity: 'validity',
    status: {converged: 'converged', iterationCount: 'rounds', iterationLimitReached: 'limit'}
  }
};

it('matches skimage reconstruction in dilation and erosion with four and eight neighbors', () => {
  for (const fixture of fixtures.reconstruction) {
    const result = computeRasterReconstructionOnCPU(
      Float32Array.from(fixture.marker),
      Float32Array.from(fixture.mask),
      {
        grid: {...PROPS.grid, width: fixture.width, height: fixture.height},
        method: fixture.method as 'dilation' | 'erosion',
        connectivity: fixture.connectivity as 4 | 8
      }
    );
    expect(Array.from(result.values)).toEqual(fixture.expected);
    expect(Array.from(result.validity)).toEqual(fixture.expected.map(() => 1));
    expect(Array.from(result.converged)).toEqual([1]);
    expect(Array.from(result.iterationLimitReached)).toEqual([0]);
  }
});

it('has the closed-form bottleneck profile and verifies the last changed round', () => {
  // The only path from the left passes through a height-3 bottleneck.
  const result = computeRasterReconstructionOnCPU(
    Float32Array.of(9, 0, 0, 0, 0),
    Float32Array.of(9, 7, 3, 8, 6),
    PROPS
  );
  expect(Array.from(result.values)).toEqual([9, 7, 3, 3, 3]);
  expect(result.iterationCount[0]).toBe(5);
  const inverse = computeRasterReconstructionOnCPU(
    Float32Array.of(1, 10, 10, 10, 10),
    Float32Array.of(1, 3, 7, 2, 4),
    {...PROPS, method: 'erosion'}
  );
  expect(Array.from(inverse.values)).toEqual([1, 3, 7, 7, 7]);
  expect(inverse.iterationCount[0]).toBe(5);
});

it('keeps invalid marker/mask pairs as barriers and rejects undersized arrays', () => {
  const result = computeRasterReconstructionOnCPU(
    Float32Array.of(9, 0, 2, NaN, Infinity),
    Float32Array.of(9, 9, 1, 9, 9),
    PROPS
  );
  expect(Array.from(result.values)).toEqual([9, 9, NaN, NaN, NaN]);
  expect(Array.from(result.validity)).toEqual([1, 1, 0, 0, 0]);
  expect(result.iterationCount[0]).toBe(2);
  expect(() =>
    computeRasterReconstructionOnCPU(new Float32Array(4), new Float32Array(5), PROPS)
  ).toThrow(/marker/);
  expect(() =>
    computeRasterReconstructionOnCPU(new Float32Array(5), new Float32Array(4), PROPS)
  ).toThrow(/mask/);
  expect(
    computeRasterReconstructionOnCPU(new Float32Array(5), new Float32Array(6), PROPS).values
  ).toEqual(new Float32Array(5));
});

it('validates both classes at construction, deferring GPU props until contributor creation', () => {
  const inputs = {
    marker: {format: 'float32' as const, length: 5},
    mask: {format: 'float32' as const, length: 5}
  };
  expect(() => new CPURasterReconstructionOperation({...PROPS, maximumIterations: 0})).toThrow(
    /^reconstruct .*maximumIterations/
  );
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterReconstructionOperation({...PROPS, maximumIterations: 0}),
      inputs
    )
  ).toThrow(/^reconstruct .*maximumIterations/);
  for (const Operation of [CPURasterReconstructionOperation, GPURasterReconstructionOperation])
    expect(() => new Operation({...PROPS, grid: {...PROPS.grid, width: 0}})).toThrow(
      /^reconstruct /
    );
  expect(
    () => new CPURasterReconstructionOperation({...PROPS, maximumIterations: 30000})
  ).not.toThrow();
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterReconstructionOperation({...PROPS, maximumIterations: 30000}),
      inputs
    )
  ).toThrow(/^reconstruct /);
  expect(() =>
    constructRasterOperationContributor(new GPURasterReconstructionOperation(PROPS), inputs)
  ).not.toThrow();
  expect(() =>
    constructRasterOperationContributor(new GPURasterReconstructionOperation(PROPS), {
      ...inputs,
      mask: {format: 'float32', length: 4}
    })
  ).toThrow(/^reconstruct .*mask/);
});

it('executes renamed outputs without reusing an input port', async () => {
  const wrap = (values: Float32Array) => ({
    format: 'float32' as const,
    length: 5,
    chunks: [{values, length: 5}],
    pack: () => values
  });
  const result = await new CPURasterReconstructionOperation({
    ...PROPS,
    maximumIterations: 1
  }).execute({
    inputs: {
      marker: wrap(Float32Array.of(9, 0, 0, 0, 0)),
      mask: wrap(Float32Array.of(9, 7, 3, 8, 6))
    }
  });
  expect(result['values']).toEqual(Float32Array.of(9, 7, 3, 3, 3));
  expect(result['validity']).toEqual(new Uint32Array(5).fill(1));
  expect(result['rounds']).toEqual(Uint32Array.of(5));
  expect(Object.keys(result).sort()).toEqual([
    'converged',
    'limit',
    'rounds',
    'validity',
    'values'
  ]);
});

it('ignores surplus marker and mask rows independently on a unit grid', () => {
  const props = {...PROPS, grid: {...PROPS.grid, width: 1}};
  for (const marker of [Float32Array.of(2), Float32Array.of(2, NaN)]) {
    for (const mask of [Float32Array.of(3), Float32Array.of(3, NaN)]) {
      const result = computeRasterReconstructionOnCPU(marker, mask, props);
      // A singleton marker 2 already satisfies mask 3 and has no neighbor to raise it.
      expect(result.values).toEqual(Float32Array.of(2));
      expect(result.validity).toEqual(Uint32Array.of(1));
      expect(result.converged).toEqual(Uint32Array.of(1));
      expect(result.iterationCount).toEqual(Uint32Array.of(1));
      expect(result.iterationLimitReached).toEqual(Uint32Array.of(0));
      expect(() =>
        constructRasterOperationContributor(new GPURasterReconstructionOperation(props), {
          marker: {format: 'float32', length: marker.length},
          mask: {format: 'float32', length: mask.length}
        })
      ).not.toThrow();
    }
  }
});
