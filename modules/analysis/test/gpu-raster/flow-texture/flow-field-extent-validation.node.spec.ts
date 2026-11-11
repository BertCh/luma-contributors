// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {getGPULineIntegralConvolutionParameterValues} from '../../../src/raster-analysis/flow-texture/line-integral-convolution-parameters';
import {getGPUStreamlinesParameterValues} from '../../../src/raster-analysis/flow-texture/streamlines-parameters';
import {getGPUParticleAdvectionParameterValues} from '../../../src/raster-analysis/particle-advection/particle-advection-parameters';

type Extent = readonly [number, number, number, number];

const VALID_EXTENT: Extent = [10, 20, 2, 3];
// A size that is zero, rounds to zero in float32, or is not finite makes every field sample
// invalid in the kernel (a division by the cell size), so the packers refuse it.
const UNUSABLE_SIZES = [0, -0, 1e-50, Number.NaN, Number.POSITIVE_INFINITY, 1e39];

function withCellWidth(cellWidth: number): Extent {
  return [10, 20, cellWidth, 3];
}

function withCellHeight(cellHeight: number): Extent {
  return [10, 20, 2, cellHeight];
}

describe('flow field extent validation', () => {
  it('GPUParticleAdvection rejects an unusable field cell size', () => {
    for (const size of UNUSABLE_SIZES) {
      expect(() =>
        getGPUParticleAdvectionParameterValues(
          {fieldExtent: withCellWidth(size), timeStep: 1},
          [4, 4]
        )
      ).toThrow(/GPUParticleAdvection fieldExtent cellWidth/);
      expect(() =>
        getGPUParticleAdvectionParameterValues(
          {fieldExtent: withCellHeight(size), timeStep: 1},
          [4, 4]
        )
      ).toThrow(/GPUParticleAdvection fieldExtent cellHeight/);
    }
  });

  it('GPULineIntegralConvolution rejects an unusable field or output cell size', () => {
    for (const size of UNUSABLE_SIZES) {
      expect(() =>
        getGPULineIntegralConvolutionParameterValues({
          fieldExtent: withCellWidth(size),
          outputExtent: VALID_EXTENT
        })
      ).toThrow(/GPULineIntegralConvolution fieldExtent cellWidth/);
      expect(() =>
        getGPULineIntegralConvolutionParameterValues({
          fieldExtent: VALID_EXTENT,
          outputExtent: withCellHeight(size)
        })
      ).toThrow(/GPULineIntegralConvolution outputExtent cellHeight/);
    }
  });

  it('GPUStreamlines rejects an unusable field cell size', () => {
    for (const size of UNUSABLE_SIZES) {
      expect(() =>
        getGPUStreamlinesParameterValues({
          fieldExtent: withCellHeight(size),
          gridExtent: VALID_EXTENT,
          stepLength: 1
        })
      ).toThrow(/GPUStreamlines fieldExtent cellHeight/);
    }
  });

  it('GPUStreamlines rejects a grid cell size that is not positive and finite in float32', () => {
    for (const size of [...UNUSABLE_SIZES, -1]) {
      expect(() =>
        getGPUStreamlinesParameterValues({
          fieldExtent: VALID_EXTENT,
          gridExtent: withCellWidth(size),
          stepLength: 1
        })
      ).toThrow(/GPUStreamlines gridExtent cells/);
      expect(() =>
        getGPUStreamlinesParameterValues({
          fieldExtent: VALID_EXTENT,
          gridExtent: withCellHeight(size),
          stepLength: 1
        })
      ).toThrow(/GPUStreamlines gridExtent cells/);
    }
  });

  it('keeps accepting the extents that packed before, including a negative cell height', () => {
    const northUpExtent: Extent = [10, 127, 2, -3];
    expect(
      Array.from(
        getGPUParticleAdvectionParameterValues({fieldExtent: northUpExtent, timeStep: 1}, [4, 4])
      ).slice(0, 4)
    ).toEqual([10, 127, 2, -3]);
    expect(
      Array.from(
        getGPULineIntegralConvolutionParameterValues({
          fieldExtent: northUpExtent,
          outputExtent: VALID_EXTENT
        })
      ).slice(0, 8)
    ).toEqual([10, 127, 2, -3, 10, 20, 2, 3]);
    expect(
      Array.from(
        getGPUStreamlinesParameterValues({
          fieldExtent: northUpExtent,
          gridExtent: VALID_EXTENT,
          stepLength: 1
        })
      ).slice(0, 4)
    ).toEqual([10, 127, 2, -3]);
  });
});
