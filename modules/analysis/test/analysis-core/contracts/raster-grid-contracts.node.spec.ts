// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, expectTypeOf, it} from 'vitest';
import type {RasterGridAffine as CoreRasterGridAffine} from '@luma-contributors/analysis/analysis-core';
import type {RasterGridAffine as SpatialRasterGridAffine} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  resolveGPURasterContext,
  type RasterGridAffine
} from '../../../src/analysis-core/contracts/raster-grid';

expectTypeOf<CoreRasterGridAffine>().toEqualTypeOf<RasterGridAffine>();
expectTypeOf<SpatialRasterGridAffine>().toEqualTypeOf<RasterGridAffine>();

function grid(rowY: number): RasterGridAffine {
  return {width: 1, height: 1, affine: [1, 1, 0, 1, rowY, 0], pixelInterpretation: 'area'};
}

it('W5-05 uses the packed axes determinant for both basis-vector inverse round trips', () => {
  const resolved = resolveGPURasterContext('packed-inverse', {grid: grid(1 + 3 * 2 ** -25)});
  const words = resolved.parameterValues;
  // At 1 the binary32 ULP is 2^-23. Three quarters of that ULP rounds up once, so
  // det([1,1],[1,1+2^-23]) = 2^-23, independently of the original binary64 area.
  expect(words[6]).toBe(2 ** -23);
  expect(resolved.affineDeterminant).toBe(3 * 2 ** -25);
  const inverse = (x: number, y: number) => [
    (words[5] * x - words[3] * y) / words[6],
    (-words[4] * x + words[2] * y) / words[6]
  ];
  expect(inverse(words[2], words[4])).toEqual([1, 0]);
  expect(inverse(words[3], words[5])).toEqual([0, 1]);
});

it('W5-05 rejects axes that collapse exactly when packed to float32', () => {
  expect(() => resolveGPURasterContext('collapsed', {grid: grid(1 + 2 ** -25)})).toThrow(
    /collapsed.*invertible in float32/
  );
});

it('review earlier-fix captures one raster affine for validation, packing and physical metadata', () => {
  let reads = 0;
  const origin: [number, number] = [7, -11];
  const changingGrid: RasterGridAffine = {
    width: 1,
    height: 1,
    pixelInterpretation: 'area',
    get affine(): RasterGridAffine['affine'] {
      reads++;
      return reads <= 3 ? [2, 0, 7, 0, 1, -11] : [9, 0, 7, 0, 9, -11];
    }
  };
  const resolved = resolveGPURasterContext('capture-affine', {
    grid: changingGrid,
    localOrigin: origin
  });
  origin[0] = 900;
  expect(resolved.affineDeterminant).toBe(2);
  expect(resolved.georeference.cellSize).toEqual([2, 1]);
  expect(Array.from(resolved.parameterValues)).toEqual([0, 0, 2, 0, 0, 1, 2, 0]);
  expect(resolved.localOrigin).toEqual([7, -11]);
  expect(reads).toBe(1);
});
