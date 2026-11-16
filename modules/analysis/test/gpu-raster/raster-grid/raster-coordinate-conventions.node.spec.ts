// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';

it('declares outer contour extents and center-based flow coordinates with independent expected samples', async () => {
  const {getGPUIsolinesParameterValues} = await import(
    '../../../src/raster-analysis/isolines/isolines-parameters'
  );
  const {getGPUIsobandsParameterValues} = await import(
    '../../../src/raster-analysis/isolines/isobands-parameters'
  );
  const {sampleFieldOnCPU} = await import(
    '../../../src/raster-analysis/vector-field-common/vector-field-sampling'
  );
  const grid = {
    width: 4,
    height: 2,
    affine: [1, 0, 0, 0, -1, 0] as const,
    pixelInterpretation: 'point' as const
  };
  const extent = [10, 20, 18, 26] as const;
  const contours = getGPUIsolinesParameterValues({extent, levelCount: 1}, grid);
  expect(Array.from(contours)).toEqual([1, 10, 20, 2, 3, 0, 0, 0]);
  const bands = getGPUIsobandsParameterValues({extent, breakCount: 1}, grid);
  expect(Array.from(bands.slice(4, 10))).toEqual([10, 20, 18, 26, 2, 3]);
  // Closed-form centers are (11, 21.5), (17, 24.5), even for point metadata/north-up affine.
  expect([contours[1] + 0.5 * contours[3], contours[2] + 0.5 * contours[4]]).toEqual([11, 21.5]);
  expect([contours[1] + 3.5 * contours[3], contours[2] + 1.5 * contours[4]]).toEqual([17, 24.5]);
  const field = {width: 2, height: 2, velocities: Float32Array.of(1, 10, 2, 20, 3, 30, 4, 40)};
  expect(sampleFieldOnCPU(field, 11, 21.5, [10, 20, 2, 3])).toEqual([1, 10]);
  expect(sampleFieldOnCPU(field, 13, 24.5, [10, 20, 2, 3])).toEqual([4, 40]);
  expect(sampleFieldOnCPU(field, 12, 23, [10, 20, 2, 3])).toEqual([2.5, 25]);
  expect(sampleFieldOnCPU(field, 14, 26, [10, 20, 2, 3])).toBeUndefined();
  // Existing signed field placement is explicit; this does not change the default row direction.
  expect(sampleFieldOnCPU(field, 11, 24.5, [10, 26, 2, -3])).toEqual([1, 10]);
});
