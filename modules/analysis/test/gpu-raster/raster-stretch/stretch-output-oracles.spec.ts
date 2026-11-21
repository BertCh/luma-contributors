// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPURasterStretch,
  getGPURasterStretchParameterValues,
  type GPURasterStretchOutput
} from '../../../src/raster-analysis/raster-stretch/index';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {RecipeTestFixture} from '../../gpu-spatial-analysis/recipes/recipe-harness';
import {requireTestDevice} from '../../utils/require-test-device';
import {expectLiteralRows} from '../../analysis-core/mixed-execution/output-oracles-test-utils';
import {
  computeRasterStretchStatistics,
  computeRasterStretchLutOnCPU,
  stretchRasterOnCPU,
  getPaletteColorOnCPU
} from './raster-stretch-oracle';

it('linear stretch LUT colors select independently known palette entries with optional outputs isolated', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = {
    values: new Float32Array([3, 0, 3, 0]),
    width: 4,
    height: 1,
    binCount: 16,
    lutSize: 4
  };
  const palette = new Uint32Array([11, 22, 33, 44]);
  const parameters = getGPURasterStretchParameterValues({
    domain: [0, 3],
    mode: 'linear',
    gamma: 2,
    paletteInterpolation: 'nearest'
  });
  const statistics = computeRasterStretchStatistics(scene, parameters);
  expect(
    Array.from(computeRasterStretchLutOnCPU(scene, parameters, statistics), value =>
      getPaletteColorOnCPU(value, palette, false)
    )
  ).toEqual([22, 33, 44, 44]);
  expect(
    Array.from(stretchRasterOnCPU(scene, parameters, statistics), value =>
      getPaletteColorOnCPU(value, palette, false)
    )
  ).toEqual([44, 11, 44, 11]);
  for (const selected of [['lutColors'], ['lutColors', 'colors', 'lut'], ['colors', 'lut']]) {
    const fixture = new RecipeTestFixture(device, 'stretch-lut-oracle');
    const seedBuffer = (buffer: {write(data: Uint32Array): void}) =>
      buffer.write(new Uint32Array(4).fill(123));
    const outputs = {
      lutColors: selected.includes('lutColors')
        ? fixture.output('lutColors', 'uint32', 4)
        : undefined,
      colors: selected.includes('colors') ? fixture.output('colors', 'uint32', 4) : undefined,
      lut: selected.includes('lut') ? fixture.output('lut', 'float32', 4) : undefined
    };
    for (const entry of Object.values(outputs)) if (entry) seedBuffer(entry.buffer);
    // Each selection is seeded with one known requested terminal, including colors-only routes.
    const output: GPURasterStretchOutput = outputs.lutColors
      ? {lutColors: outputs.lutColors.view}
      : outputs.colors
        ? {colors: outputs.colors.view}
        : {lut: outputs.lut!.view};
    if (outputs.colors) output.colors = outputs.colors.view;
    if (outputs.lut) output.lut = outputs.lut.view;
    fixture.graph.add(
      new GPURasterStretch({
        grid: makeGPURasterMetadata({width: 4, height: 1}),
        // Histogram bins and LUT samples have independent capacities: minimum bins=16.
        // Linear mode uses the explicit domain, so the four LUT expectations stay unchanged.
        binCount: 16,
        lutSize: 4,
        values: fixture.input('values', scene.values, 'float32', 4),
        palette: fixture.input('palette', palette, 'uint32', 4),
        parameters: fixture.input('parameters', parameters, 'float32', 16),
        output
      })
    );
    try {
      fixture.run();
      // Gamma 2 gives sqrt([1/8,3/8,5/8,7/8]); floor(4*t)=[1,2,3,3].
      // The LUT, original palette and per-cell colors are now three distinct arrays.
      if (outputs.lutColors)
        expect(await fixture.readUint32(outputs.lutColors, 4)).toEqual([22, 33, 44, 44]);
      if (outputs.colors)
        expect(await fixture.readUint32(outputs.colors, 4)).toEqual([44, 11, 44, 11]);
      if (outputs.lut)
        expectLiteralRows(
          await fixture.readFloat32(outputs.lut, 4),
          [0.3535533905932738, 0.6123724356957945, 0.7905694150420949, 0.9354143466934853]
        );
    } finally {
      fixture.destroy();
    }
  }
});
