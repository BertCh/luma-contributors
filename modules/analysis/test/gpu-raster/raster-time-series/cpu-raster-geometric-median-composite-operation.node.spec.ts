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
  expect(result.status[0]).toBe(0);
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
  expect(result.status[0]).toBe(0);
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
