// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {packGPUSpatialParameterValues} from '../../../src/analysis-core/contracts/index';
import {
  getGPULineDensityParameterValues,
  getGPULineDensityRasterOutputMapping,
  GPU_LINE_DENSITY_PARAMETER_SCHEMA
} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-density';
const GRID: GPURasterMetadata = {
  width: 2,
  height: 3,
  affine: [1, 0, 0, 0, -1, 3],
  pixelInterpretation: 'area'
};
it('B4 metadata packing roundtrips through the existing minimum/positive-size public schema', () => {
  const words = getGPULineDensityParameterValues(GRID);
  expect([...words]).toEqual([0, 0, 1, 1]);
  const fields = Object.fromEntries(
    GPU_LINE_DENSITY_PARAMETER_SCHEMA.fields.map(field => [field.name, words[field.wordOffset]])
  );
  expect([...packGPUSpatialParameterValues(GPU_LINE_DENSITY_PARAMETER_SCHEMA, fields)]).toEqual([
    ...words
  ]);
  expect([
    ...getGPULineDensityParameterValues({minX: 0, minY: 0, cellWidth: 1, cellHeight: 1})
  ]).toEqual([0, 0, 1, 1]);
});
it('B3 supplies an explicit raster output mapping while traversal retains physical ownership', () => {
  expect(getGPULineDensityRasterOutputMapping(GRID)).toEqual({
    outputRowOrder: 'north-first',
    outputColumnOrder: 'west-first'
  });
  expect(getGPULineDensityRasterOutputMapping({...GRID, affine: [-1, 0, 2, 0, 1, 0]})).toEqual({
    outputRowOrder: 'south-first',
    outputColumnOrder: 'east-first'
  });
});
it('UPT-03 preserves origin/step raster packing when float32 corners coincide', () => {
  // 1,000,000 has float32 spacing .0625. The origin/step schema stores the
  // independently representable .03125 step even though the far corner rounds down.
  const grid: GPURasterMetadata = {
    width: 1,
    height: 1,
    affine: [0.03125, 0, 1000000, 0, 1, 0],
    pixelInterpretation: 'area'
  };
  expect([
    ...getGPULineDensityParameterValues({
      minX: 1000000,
      minY: 0,
      cellWidth: 0.03125,
      cellHeight: 1
    })
  ]).toEqual([1000000, 0, 0.03125, 1]);
  expect([...getGPULineDensityParameterValues(grid)]).toEqual([1000000, 0, 0.03125, 1]);
  expect(getGPULineDensityRasterOutputMapping(grid)).toEqual({
    outputRowOrder: 'south-first',
    outputColumnOrder: 'west-first'
  });
});
it('UPT-03 retains affine and encoded-step validation in origin/step raster packing', () => {
  for (const grid of [
    {...GRID, affine: [1, 0.5, 0, 0, -1, 3]},
    {...GRID, affine: [1e-50, 0, 0, 0, -1, 3]}
  ] satisfies GPURasterMetadata[]) {
    expect(() => getGPULineDensityParameterValues(grid)).toThrow();
    expect(() => getGPULineDensityRasterOutputMapping(grid)).toThrow();
  }
});
it('B1 rejects a minimum rounded by a whole cell without publishing parameters', () => {
  // The reversed cell is [999999.96875,1000000]. Float32 rounds its minimum
  // to 1000000, translating the entire .03125 cell onto an exterior segment.
  for (const axis of ['x', 'y'] as const) {
    const grid: GPURasterMetadata = {
      width: 1,
      height: 1,
      affine: axis === 'x' ? [-0.03125, 0, 1000000, 0, 1, 0] : [1, 0, 0, 0, -0.03125, 1000000],
      pixelInterpretation: 'area'
    };
    const settings = {
      minX: axis === 'x' ? 999999.96875 : 0,
      minY: axis === 'y' ? 999999.96875 : 0,
      cellWidth: axis === 'x' ? 0.03125 : 1,
      cellHeight: axis === 'y' ? 0.03125 : 1
    };
    for (const pack of [
      (target: Float32Array) => getGPULineDensityParameterValues(grid, target),
      (target: Float32Array) => getGPULineDensityParameterValues(settings, target)
    ]) {
      const target = Float32Array.of(77, 77, 77, 77, 88);
      expect(() => pack(target)).toThrow(/common.*origin/);
      expect([...target]).toEqual([77, 77, 77, 77, 88]);
    }
    expect(() => getGPULineDensityRasterOutputMapping(grid)).toThrow(/common.*origin/);
  }
});
it('S1 rejects independently encoded zero steps even when encoded corners differ', () => {
  const grid: GPURasterMetadata = {
    width: 1,
    height: 1,
    affine: [2 ** -152, 0, 2 ** -150, 0, 1, 0],
    pixelInterpretation: 'area'
  };
  // Endpoints straddle the float32 rounding midpoint; the independent step underflows.
  expect(Math.fround(2 ** -150)).toBe(0);
  expect(Math.fround(2 ** -150 + 2 ** -152)).toBe(2 ** -149);
  expect(() => getGPULineDensityRasterOutputMapping(grid)).toThrow(/cellWidth/);
});
it('S3 rejects pixelInterpretation outside area and point instead of silently shifting the grid', () => {
  expect(() =>
    getGPULineDensityParameterValues({...GRID, pixelInterpretation: 'points' as 'point'})
  ).toThrow(/pixelInterpretation.*area.*point.*points/);
});

it('B3 raster output permutations pass host WGSL audit with canonical positive grid words', async () => {
  const {GPUCommandGraph, createTransientView} = await import('@luma.gl/graph/gpu-core');
  const {GPULineDensity} = await import(
    '../../../src/gpu-spatial-analysis/line-density/gpu-line-density'
  );
  const {createNullWebGPUDevice, getParsedKernelSources} = await import(
    '../../utils/gpu-contributor-test-utils'
  );
  const {auditWGSLModule} = await import('../../analysis-core/kernel-audit/kernel-audit-wgsl');
  for (const outputRowOrder of ['north-first', 'south-first'] as const)
    for (const outputColumnOrder of ['east-first', 'west-first'] as const) {
      const graph = new GPUCommandGraph(createNullWebGPUDevice());
      const props = {
        positions: createTransientView(graph, 'positions', 'float32x2', 2),
        lineOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
        parameters: createTransientView(graph, 'parameters', 'float32', 4),
        columns: 2,
        rows: 3,
        outputRowOrder,
        outputColumnOrder,
        output: {
          lengths: createTransientView(graph, 'lengths', 'float32', 6),
          densities: createTransientView(graph, 'densities', 'float32', 6),
          status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
        }
      };
      for (const {id, source, constants} of getParsedKernelSources(
        new GPULineDensity(props).getCommandNodes(graph)
      ).configurations)
        expect(auditWGSLModule(source, constants), id).toEqual([]);
      expect(
        () => new GPULineDensity({...props, outputRowOrder: 'top-down' as 'north-first'})
      ).toThrow(/outputRowOrder.*south-first.*north-first.*top-down/);
    }
});
