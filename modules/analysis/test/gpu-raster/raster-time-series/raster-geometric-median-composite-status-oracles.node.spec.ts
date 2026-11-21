// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeRasterGeometricMedianCompositeOnCPU} from '../../../src/raster-analysis/raster-time-series/cpu-raster-geometric-median-composite-operation';
import {getGPURasterGeometricMedianCompositeParameterValues} from '../../../src/raster-analysis/raster-time-series/gpu-raster-geometric-median-composite';

it('derives geometric median CPU statuses from support, even before convergence', () => {
  // Retained observations, all masked, singleton retained, all nonfinite: support counts 3,0,1,0.
  const result = computeRasterGeometricMedianCompositeOnCPU(
    Float32Array.of(0, 2, 10, 0, 2, 10, NaN, 2, NaN, NaN, Infinity, -Infinity),
    getGPURasterGeometricMedianCompositeParameterValues(),
    {cellCount: 4, sliceCount: 3, bandCount: 1, iterationCount: 1},
    Uint32Array.of(1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1)
  );
  expect(Array.from(result.status)).toEqual([0, 1, 0, 1]);
  // A correct nonzero step does not change the support status. Float64 rounded to float32.
  expect(result.lastStep[0]).toBe(Math.fround(12 / 11));
});
