// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeRasterMedoidCompositeOnCPU,
  CPURasterMedoidCompositeOperation
} from '../../../src/raster-analysis/raster-time-series/cpu-raster-medoid-composite-operation';
import {
  GPURasterMedoidCompositeOperation,
  type RasterMedoidCompositeOperationProps
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-medoid-composite-operation';
import {constructRasterOperationContributor} from '../../utils/raster-operation-test-utils';

const PROPS: RasterMedoidCompositeOperationProps = {
  id: 'medoid',
  cellCount: 1,
  sliceCount: 5,
  bandCount: 1,
  values: 'values',
  output: {composite: 'composite', sliceIndex: 'sliceIndex', status: 'status'}
};

it('agrees with the order-statistic median oracle in one dimension', () => {
  const values = new Float32Array([40, -10, 7, 2, 9]);
  // For odd scalar observations the unique median minimizes total absolute distance.
  const median = Array.from(values).sort((first, second) => first - second)[2];
  const result = computeRasterMedoidCompositeOnCPU(values, PROPS);
  expect(result.composite[0]).toBe(median);
  expect(result.sliceIndex[0]).toBe(2);
  expect(result.status[0]).toBe(0);
});

it('selects a hand-computed multiband medoid and earliest ties, filtering whole observations', () => {
  const dimensions = {cellCount: 1, sliceCount: 4, bandCount: 2};
  const values = new Float32Array([0, 0, 3, 0, 0, 4, 1, 1]);
  const result = computeRasterMedoidCompositeOnCPU(values, dimensions);
  // Interior point sum sqrt(2)+sqrt(5)+sqrt(10) is below all three corner sums.
  expect(result.sliceIndex[0]).toBe(3);
  expect(Array.from(result.composite)).toEqual([1, 1]);
  const tie = computeRasterMedoidCompositeOnCPU(
    new Float32Array([0, 0, 3, 4, NaN, 0, 1, 1]),
    dimensions,
    new Uint32Array([1, 1, 1, 0])
  );
  expect(tie.sliceIndex[0]).toBe(0);
  const empty = computeRasterMedoidCompositeOnCPU(values, dimensions, new Uint32Array(4));
  expect(empty.sliceIndex[0]).toBe(0xffffffff);
  expect(empty.status[0]).toBe(1);
  expect(empty.composite.every(Number.isNaN)).toBe(true);
});

it('validates both constructors and CPU input lengths while dropping GPU caps', () => {
  for (const Operation of [CPURasterMedoidCompositeOperation, GPURasterMedoidCompositeOperation])
    expect(() => new Operation({...PROPS, cellCount: 0})).toThrow(/^medoid /);
  expect(() => new CPURasterMedoidCompositeOperation({...PROPS, sliceCount: 0})).toThrow(
    /^medoid sliceCount/
  );
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterMedoidCompositeOperation({...PROPS, sliceCount: 0}),
      {values: {format: 'float32', length: 5}}
    )
  ).toThrow(/^medoid sliceCount/);
  expect(
    () => new CPURasterMedoidCompositeOperation({...PROPS, sliceCount: 65, bandCount: 17})
  ).not.toThrow();
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterMedoidCompositeOperation({...PROPS, sliceCount: 65}),
      {values: {format: 'float32', length: 65}}
    )
  ).toThrow(/sliceCount/);
  expect(() => computeRasterMedoidCompositeOnCPU(new Float32Array(4), PROPS)).toThrow(/values/);
  expect(() =>
    computeRasterMedoidCompositeOnCPU(new Float32Array(5), PROPS, new Uint32Array(4))
  ).toThrow(/mask/);
});
