// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {getGPULineIntegralConvolutionParameterValues} from '../../../src/raster-analysis/flow-texture/line-integral-convolution-parameters';
import {getGPUStreamlinesParameterValues} from '../../../src/raster-analysis/flow-texture/streamlines-parameters';
import {getGPUParticleAdvectionParameterValues} from '../../../src/raster-analysis/particle-advection/particle-advection-parameters';

type Placement = {origin: readonly [number, number]; cellSize: readonly [number, number]};

const VALID_PLACEMENT: Placement = {origin: [10, 20], cellSize: [2, 3]};
// A size that is zero, rounds to zero in float32, or is not finite makes every field sample
// invalid in the kernel (a division by the cell size), so the packers refuse it.
const UNUSABLE_SIZES = [0, -0, 1e-50, Number.NaN, Number.POSITIVE_INFINITY, 1e39];

function withCellWidth(cellWidth: number): Placement {
  return {origin: [10, 20], cellSize: [cellWidth, 3]};
}

function withCellHeight(cellHeight: number): Placement {
  return {origin: [10, 20], cellSize: [2, cellHeight]};
}

describe('flow field extent validation', () => {
  it('GPUParticleAdvection rejects an unusable field cell size', () => {
    for (const size of UNUSABLE_SIZES) {
      expect(() =>
        getGPUParticleAdvectionParameterValues(
          {fieldPlacement: withCellWidth(size), timeStep: 1},
          [4, 4]
        )
      ).toThrow(/GPUParticleAdvection fieldPlacement cellWidth/);
      expect(() =>
        getGPUParticleAdvectionParameterValues(
          {fieldPlacement: withCellHeight(size), timeStep: 1},
          [4, 4]
        )
      ).toThrow(/GPUParticleAdvection fieldPlacement cellHeight/);
    }
  });

  it('GPULineIntegralConvolution rejects an unusable field or output cell size', () => {
    for (const size of UNUSABLE_SIZES) {
      expect(() =>
        getGPULineIntegralConvolutionParameterValues({
          fieldPlacement: withCellWidth(size),
          outputPlacement: VALID_PLACEMENT
        })
      ).toThrow(/GPULineIntegralConvolution fieldPlacement cellWidth/);
      expect(() =>
        getGPULineIntegralConvolutionParameterValues({
          fieldPlacement: VALID_PLACEMENT,
          outputPlacement: withCellHeight(size)
        })
      ).toThrow(/GPULineIntegralConvolution outputPlacement cellHeight/);
    }
  });

  it('GPUStreamlines rejects an unusable field cell size', () => {
    for (const size of UNUSABLE_SIZES) {
      expect(() =>
        getGPUStreamlinesParameterValues({
          fieldPlacement: withCellHeight(size),
          gridPlacement: VALID_PLACEMENT,
          stepLength: 1
        })
      ).toThrow(/GPUStreamlines fieldPlacement cellHeight/);
    }
  });

  it('GPUStreamlines rejects a grid cell size that is not positive and finite in float32', () => {
    for (const size of [...UNUSABLE_SIZES, -1]) {
      expect(() =>
        getGPUStreamlinesParameterValues({
          fieldPlacement: VALID_PLACEMENT,
          gridPlacement: withCellWidth(size),
          stepLength: 1
        })
      ).toThrow(/GPUStreamlines gridPlacement cells/);
      expect(() =>
        getGPUStreamlinesParameterValues({
          fieldPlacement: VALID_PLACEMENT,
          gridPlacement: withCellHeight(size),
          stepLength: 1
        })
      ).toThrow(/GPUStreamlines gridPlacement cells/);
    }
  });

  it('keeps accepting the extents that packed before, including a negative cell height', () => {
    const northUpPlacement: Placement = {origin: [10, 127], cellSize: [2, -3]};
    expect(
      Array.from(
        getGPUParticleAdvectionParameterValues(
          {fieldPlacement: northUpPlacement, timeStep: 1},
          [4, 4]
        )
      ).slice(0, 4)
    ).toEqual([10, 127, 2, -3]);
    expect(
      Array.from(
        getGPULineIntegralConvolutionParameterValues({
          fieldPlacement: northUpPlacement,
          outputPlacement: VALID_PLACEMENT
        })
      ).slice(0, 8)
    ).toEqual([10, 127, 2, -3, 10, 20, 2, 3]);
    expect(
      Array.from(
        getGPUStreamlinesParameterValues({
          fieldPlacement: northUpPlacement,
          gridPlacement: VALID_PLACEMENT,
          stepLength: 1
        })
      ).slice(0, 4)
    ).toEqual([10, 127, 2, -3]);
  });
});
