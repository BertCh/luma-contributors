// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeRasterGeometricMedianCompositeOnCPU,
  CPURasterGeometricMedianCompositeOperation
} from '../../../src/raster-analysis/raster-time-series/cpu-raster-geometric-median-composite-operation';
import {
  GPURasterGeometricMedianCompositeOperation,
  type RasterGeometricMedianCompositeOperationProps
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-geometric-median-composite-operation';
import {getGPURasterGeometricMedianCompositeParameterValues} from '../../../src/raster-analysis/raster-time-series/gpu-raster-geometric-median-composite';
import {constructRasterOperationContributor} from '../../utils/raster-operation-test-utils';

const PROPS: RasterGeometricMedianCompositeOperationProps = {
  id: 'geometric',
  cellCount: 1,
  sliceCount: 3,
  bandCount: 2,
  iterationCount: 64,
  values: 'values',
  parameters: 'parameters',
  output: {composite: 'composite', status: 'status', lastStep: 'lastStep'}
};
const PARAMETERS = getGPURasterGeometricMedianCompositeParameterValues();

it('recovers the analytic Fermat point of an isosceles triangle', () => {
  // All angles below 120 degrees: the geometric median has three incident 120-degree angles.
  // Symmetry gives x=2, and the bottom rays meet at 120 degrees when y=2/sqrt(3).
  const result = computeRasterGeometricMedianCompositeOnCPU(
    new Float32Array([0, 0, 4, 0, 2, 3]),
    PARAMETERS,
    PROPS
  );
  expect(result.composite[0]).toBe(2);
  expect(Math.abs(result.composite[1] - 2 / Math.sqrt(3))).toBeLessThan(1e-6);
  expect(result.lastStep[0]).toBeLessThan(1e-6);
  expect(Array.from(result.status)).toEqual([0]);
});

it('matches a hand-computed rational first pass and keeps singleton/missing layouts', () => {
  const options = {...PROPS, bandCount: 1, iterationCount: 1};
  // At mean 4 the weights are 1/4,1/2,1/6, yielding 32/11 and step 12/11.
  const result = computeRasterGeometricMedianCompositeOnCPU(
    new Float32Array([0, 2, 10]),
    PARAMETERS,
    options
  );
  expect(result.composite[0]).toBe(Math.fround(32 / 11));
  expect(result.lastStep[0]).toBe(Math.fround(12 / 11));
  expect(result.lastStep[0]).toBeGreaterThan(1);
  expect(Array.from(result.status)).toEqual([0]);
  const mask = new Uint32Array([0, 1, 0]);
  const singleton = computeRasterGeometricMedianCompositeOnCPU(
    new Float32Array([0, 2, 10]),
    PARAMETERS,
    options,
    mask
  );
  expect(singleton.composite[0]).toBe(2);
  expect(singleton.lastStep[0]).toBe(0);
  const empty = computeRasterGeometricMedianCompositeOnCPU(
    new Float32Array([NaN, 2, NaN]),
    PARAMETERS,
    options,
    new Uint32Array(3)
  );
  expect(empty.composite[0]).toBeNaN();
  expect(empty.lastStep[0]).toBeNaN();
  expect(empty.status[0]).toBe(1);
});

it('checks constructors and array arguments and accepts CPU counts above GPU caps', () => {
  const inputs = {
    values: {format: 'float32' as const, length: 6},
    parameters: {format: 'float32' as const, length: 1}
  };
  for (const Operation of [
    CPURasterGeometricMedianCompositeOperation,
    GPURasterGeometricMedianCompositeOperation
  ])
    expect(() => new Operation({...PROPS, cellCount: 0})).toThrow(/^geometric /);
  expect(
    () => new CPURasterGeometricMedianCompositeOperation({...PROPS, iterationCount: 0})
  ).toThrow(/^geometric iterationCount/);
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterGeometricMedianCompositeOperation({...PROPS, iterationCount: 0}),
      inputs
    )
  ).toThrow(/^geometric iterationCount/);
  expect(
    () =>
      new CPURasterGeometricMedianCompositeOperation({
        ...PROPS,
        sliceCount: 257,
        bandCount: 17,
        iterationCount: 65
      })
  ).not.toThrow();
  expect(() =>
    computeRasterGeometricMedianCompositeOnCPU(new Float32Array(5), PARAMETERS, PROPS)
  ).toThrow(/values/);
  expect(() =>
    computeRasterGeometricMedianCompositeOnCPU(new Float32Array(6), new Float32Array(0), PROPS)
  ).toThrow(/parameters/);
});

it.each([1, 3])('TWN-41 keeps %s identical nonzero observations exactly', sliceCount => {
  const result = computeRasterGeometricMedianCompositeOnCPU(
    new Float32Array(sliceCount).fill(456787.5),
    PARAMETERS,
    {...PROPS, sliceCount, bandCount: 1, iterationCount: 1}
  );
  expect(Array.from(result.composite)).toEqual([456787.5]);
  expect(Array.from(result.lastStep)).toEqual([0]);
  expect(Array.from(result.status)).toEqual([0]);
});

it('W3-05 rejects ambiguous CPU graph names before keying outputs', () => {
  for (const output of [
    {...PROPS.output, composite: ''},
    {...PROPS.output, status: 'composite'},
    {...PROPS.output, lastStep: 'parameters'}
  ])
    expect(() => new CPURasterGeometricMedianCompositeOperation({...PROPS, output})).toThrow(
      /^geometric /
    );
  expect(() => new CPURasterGeometricMedianCompositeOperation({...PROPS, values: ''})).toThrow(
    /^geometric /
  );
});

it.each([
  {values: [1, 1, 3, 1, 1, 2], expected: -0.0195126859, scalarMedian: 0},
  {values: [0.5, 0.25, 1, 0.25, 0.5, 0.75], expected: 0.2600577, scalarMedian: 1 / 3}
])('TWN-12 joint spectrum NDVI differs from acquisition median ($expected)', ({
  values,
  expected,
  scalarMedian
}) => {
  // Independent SciPy minimize(sum(norm(point-spectra,axis=1))) gives points
  // [1.25456932,1.30450369] and [.6056624,.3556624]; the scalar medians are hand-sorted.
  // Band order is [nearInfrared, red].
  const result = computeRasterGeometricMedianCompositeOnCPU(
    Float32Array.from(values),
    PARAMETERS,
    PROPS
  );
  expect(result.composite).toHaveLength(2);
  const expectedSpectrum = values[0] === 1 ? [1.25456932, 1.30450369] : [0.605662435, 0.355662437];
  for (const [index, value] of expectedSpectrum.entries()) {
    expect(Number.isFinite(result.composite[index])).toBe(true);
    expect(Math.abs(result.composite[index] - value)).toBeLessThan(1e-6);
  }
  expect(result.lastStep).toHaveLength(1);
  expect(Number.isFinite(result.lastStep[0])).toBe(true);
  expect(result.lastStep[0]).toBeLessThan(1e-6);
  const ndvi =
    (result.composite[0] - result.composite[1]) / (result.composite[0] + result.composite[1]);
  expect(Number.isFinite(ndvi)).toBe(true);
  expect(Math.abs(ndvi - expected)).toBeLessThan(1e-6);
  expect(Math.abs(ndvi - scalarMedian)).toBeGreaterThan(0.01);
  expect(Array.from(result.status)).toEqual([0]);
});

it('review B6 validates only declared required geometric destinations', () => {
  for (const key of ['composite', 'status'] as const)
    expect(
      () =>
        new CPURasterGeometricMedianCompositeOperation({
          ...PROPS,
          output: {
            ...PROPS.output,
            [key]: undefined
          } as unknown as RasterGeometricMedianCompositeOperationProps['output']
        })
    ).toThrow();
  expect(
    () =>
      new CPURasterGeometricMedianCompositeOperation({
        ...PROPS,
        output: {...PROPS.output, lastStep: undefined, other: PROPS.output.composite}
      } as RasterGeometricMedianCompositeOperationProps)
  ).not.toThrow();
});
