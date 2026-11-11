// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {addLandscapeMetricsRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from '../recipes/recipe-harness';

// Counts past 2^24 = 16,777,216, where a float32 counter that adds 1.0 stops growing. Expected
// values are closed forms for a full square of one class.

it('landscape metrics count more than 2^24 like sides exactly', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'landscape-large-counts');
  try {
    // One 2,100 x 2,100 patch. Every cell has four neighbours except along the four edges:
    // like sides = 4 * 2100 * 2100 - 4 * 2100 = 17,631,600 (each adjacency seen from both cells).
    // The largest count for 2100^2 cells is 2 * 2100 * 2099 adjacencies, so the aggregation
    // index is 100. A float32 counter stops at 16,777,216 and gives 95.15.
    const size = 2100;
    const patchMetrics = fixture.output('patch-metrics', 'float32x4', 1);
    const classMetrics = fixture.output('class-metrics', 'float32x4', 1);
    const aggregation = fixture.output('aggregation', 'float32', 1);
    const landscapeAggregation = fixture.output('landscape-aggregation', 'float32', 1);
    addLandscapeMetricsRecipe(fixture.graph, {
      patches: fixture.input('patches', new Uint32Array(size * size), 'uint32', size * size),
      patchClasses: fixture.input('classes', new Uint32Array([0]), 'uint32', 1),
      width: size,
      height: size,
      classCount: 1,
      cellSize: [1, 1],
      connectivity: 4,
      output: {
        patchMetrics: patchMetrics.view,
        classMetrics: classMetrics.view,
        aggregation: aggregation.view,
        landscapeAggregation: landscapeAggregation.view
      }
    });
    fixture.run();
    // Area, perimeter (4 * 2100 domain sides), cell count, like sides. All are exact in float32.
    expect(await fixture.readFloat32(patchMetrics, 4)).toEqual([4410000, 8400, 4410000, 17631600]);
    expect((await fixture.readFloat32(aggregation, 1))[0]).toBeCloseTo(100, 4);
    expect((await fixture.readFloat32(landscapeAggregation, 1))[0]).toBeCloseTo(100, 4);
    // Class area, perimeter and mesh (one patch: area squared over area).
    const classValues = await fixture.readFloat32(classMetrics, 4);
    expect(classValues.slice(0, 3)).toEqual([4410000, 8400, 4410000]);
  } finally {
    fixture.destroy();
  }
}, 120_000);

it('landscape aggregation uses the exact largest square for counts that are not squares', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'landscape-aggregation-remainders');
  try {
    // Three classes as full-width bands of a 4-column raster, by hand (He et al. 2000):
    // class 0: 4 x 1 = 4 cells, 3 adjacencies; n = 2, m = 0, maximum 2n(n-1) = 4: 75.
    // class 1: 4 x 2 minus one corner = 7 cells; n = 2, m = 3 > n, maximum 4 + 2m - 2 = 8.
    //          Adjacencies: 3 in the full row, 2 in the short row, 3 between the rows: 8: 100.
    // class 2: 4 + 2 = 6 cells; n = 2, m = 2 <= n, maximum 4 + 2m - 1 = 7.
    //          Adjacencies: 3 in the full row, 1 in the short row, 2 between: 6: 85.714287.
    // Rows of the raster, top to bottom (patch id = class id; 3 is background):
    const patches = new Uint32Array([
      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 3, 3
    ]);
    const aggregation = fixture.output('aggregation', 'float32', 3);
    addLandscapeMetricsRecipe(fixture.graph, {
      patches: fixture.input('patches', patches, 'uint32', patches.length),
      patchClasses: fixture.input('classes', new Uint32Array([0, 1, 2]), 'uint32', 3),
      width: 4,
      height: 6,
      classCount: 3,
      cellSize: [1, 1],
      connectivity: 4,
      output: {aggregation: aggregation.view}
    });
    fixture.run();
    const values = await fixture.readFloat32(aggregation, 3);
    expect(values[0]).toBeCloseTo(75, 4);
    expect(values[1]).toBeCloseTo(100, 4);
    expect(values[2]).toBeCloseTo(600 / 7, 4);
  } finally {
    fixture.destroy();
  }
});
