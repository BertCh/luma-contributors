// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {prepareRasterStackInput} from '@luma-contributors/analysis/raster-analysis';
import {computeRasterMedoidCompositeOnCPU} from '../../../src/raster-analysis/raster-time-series/cpu-raster-medoid-composite-operation';

const GRID = {
  width: 2,
  height: 1,
  affine: [1, 0, 0, 0, -1, 0] as const,
  pixelInterpretation: 'area' as const
};
it('ING-17 packs acquisition planes into the two-cell medoid input', () => {
  const packed = prepareRasterStackInput({
    grid: GRID,
    bandCount: 1,
    sourceLayout: 'acquisition-major',
    slices: [Float32Array.of(0, 10), Float32Array.of(100, 20), Float32Array.of(200, 30)],
    maskLayout: 'acquisition-major',
    mask: [Uint32Array.of(1, 2), Uint32Array.of(3, 4), Uint32Array.of(5, 6)]
  });
  expect(Array.from(packed.values)).toEqual([0, 100, 200, 10, 20, 30]);
  expect(Array.from(packed.mask!)).toEqual([1, 3, 5, 2, 4, 6]);
  expect([packed.cellCount, packed.sliceCount, packed.bandCount]).toEqual([2, 3, 1]);
  const result = computeRasterMedoidCompositeOnCPU(packed.values, packed, packed.mask);
  const unprepared = computeRasterMedoidCompositeOnCPU(
    Float32Array.of(0, 10, 100, 20, 200, 30),
    packed
  );
  expect(Array.from(unprepared.composite)).toEqual([10, 30]);
  // Each scalar triple's unique median: [0,100,200] -> 100; [10,20,30] -> 20.
  expect(Array.from(result.composite)).toEqual([100, 20]);
  expect(Array.from(result.status)).toEqual([0, 0]);
  expect(Array.from(result.sliceIndex)).toEqual([1, 1]);
});
it('ING-17 preserves multiband spectra and accepts explicit cell-major masks', () => {
  const packed = prepareRasterStackInput({
    grid: GRID,
    bandCount: 2,
    sourceLayout: 'acquisition-major',
    slices: [Float32Array.of(1, 2, 3, 4), Float32Array.of(5, 6, 7, 8)],
    maskLayout: 'cell-major',
    mask: [Uint32Array.of(1, 0), Uint32Array.of(1, 1)]
  });
  expect(Array.from(packed.values)).toEqual([1, 2, 5, 6, 3, 4, 7, 8]);
  expect(Array.from(packed.mask!)).toEqual([1, 0, 1, 1]);
  const retained = computeRasterMedoidCompositeOnCPU(packed.values, packed, packed.mask);
  expect(Array.from(retained.composite)).toEqual([1, 2, 3, 4]);
  expect(Array.from(retained.sliceIndex)).toEqual([0, 0]);
  expect(Array.from(retained.status)).toEqual([0, 0]);
  const again = prepareRasterStackInput({
    grid: GRID,
    bandCount: 2,
    sourceLayout: 'cell-major',
    slices: [packed.values.subarray(0, 4), packed.values.subarray(4)]
  });
  expect(again.values).toEqual(packed.values);
  expect(again.values).not.toBe(packed.values);
});
it('ING-17 rejects unknown layouts, ragged slices and masks without an explicit layout', () => {
  const props = {
    grid: GRID,
    bandCount: 1,
    sourceLayout: 'acquisition-major' as const,
    slices: [Float32Array.of(1, 2)]
  };
  expect(() =>
    prepareRasterStackInput({...props, sourceLayout: 'unknown' as 'cell-major'})
  ).toThrow(/sourceLayout.*acquisition-major.*cell-major/);
  expect(() => prepareRasterStackInput({...props, slices: [Float32Array.of(1)]})).toThrow(/slices/);
  expect(() => prepareRasterStackInput({...props, mask: [Uint32Array.of(1, 1)]})).toThrow(
    /maskLayout/
  );
});
