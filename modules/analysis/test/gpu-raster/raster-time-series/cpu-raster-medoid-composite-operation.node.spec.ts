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
  expect(Array.from(result.status)).toEqual([0]);
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

it.each([
  [0.1, 0.2, 0.3, 0.4],
  [0.1, 0.3, 0.2, 0.4]
])('TWN-41 selects earliest scalar median acquisition from %j', (...values) => {
  const stored = Float32Array.from(values);
  const result = computeRasterMedoidCompositeOnCPU(stored, {...PROPS, sliceCount: 4});
  expect(Array.from(result.sliceIndex)).toEqual([1]);
  expect(Array.from(result.composite)).toEqual([stored[1]]);
  expect(Array.from(result.status)).toEqual([0]);
});

it.each([1, 1e-25, 1e20])('URT-14 keeps the unique scalar medoid at scale %s', scale => {
  const result = computeRasterMedoidCompositeOnCPU(Float32Array.of(0, scale, 2 * scale), {
    ...PROPS,
    sliceCount: 3
  });
  expect(Array.from(result.sliceIndex)).toEqual([1]);
  expect(Array.from(result.composite)).toEqual([Math.fround(scale)]);
  expect(Array.from(result.status)).toEqual([0]);
});

it('W3-05 rejects ambiguous CPU graph names before keying outputs', () => {
  for (const output of [
    {...PROPS.output, composite: ''},
    {...PROPS.output, status: 'composite'},
    {...PROPS.output, sliceIndex: 'values'}
  ])
    expect(() => new CPURasterMedoidCompositeOperation({...PROPS, output})).toThrow(/^medoid /);
  expect(() => new CPURasterMedoidCompositeOperation({...PROPS, values: ''})).toThrow(/^medoid /);
});

it.each([
  {values: [7e-12, 1, 3, 5], bandCount: 1, expected: [1]},
  {
    values: [0.1, 0.875, 0.2, 0.875, 0.3, 0.875, 0.4, 0.875],
    bandCount: 2,
    expected: [Math.fround(0.2), 0.875]
  },
  {values: [1e24, 0, 1e24, 1, 1e24, 2], bandCount: 2, expected: [Math.fround(1e24), 1]}
])('review B4 B5 S1 CPU selects the earliest minimizer with $bandCount bands', ({
  values,
  bandCount,
  expected
}) => {
  const result = computeRasterMedoidCompositeOnCPU(Float32Array.from(values), {
    cellCount: 1,
    sliceCount: values.length / bandCount,
    bandCount
  });
  expect(Array.from(result.composite)).toEqual(expected);
  expect(Array.from(result.sliceIndex)).toEqual([1]);
  expect(Array.from(result.status)).toEqual([0]);
});

it('review B6 validates only declared required medoid destinations', () => {
  for (const key of ['composite', 'sliceIndex', 'status'] as const)
    expect(
      () =>
        new CPURasterMedoidCompositeOperation({
          ...PROPS,
          output: {
            ...PROPS.output,
            [key]: undefined
          } as unknown as RasterMedoidCompositeOperationProps['output']
        })
    ).toThrow();
  expect(
    () =>
      new CPURasterMedoidCompositeOperation({
        ...PROPS,
        output: {...PROPS.output, other: PROPS.output.composite}
      } as RasterMedoidCompositeOperationProps)
  ).not.toThrow();
});
