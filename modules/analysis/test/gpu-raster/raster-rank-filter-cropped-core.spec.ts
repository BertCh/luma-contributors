// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterRankFilter} from '../../src/raster-analysis/gpu-raster-rank-filter';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {createRasterTimeSeriesHarness} from './raster-time-series/raster-time-series-test-utils';

it('B5 inherited and non-enumerable ignore policies preserve a valid center beside nodata', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const inherited of [false, true]) {
    const harness = createRasterTimeSeriesHarness(device);
    const fields = {
      grid: {
        width: 2,
        height: 1,
        affine: [1, 0, 0, 0, 1, 0] as const,
        pixelInterpretation: 'point' as const
      },
      input: {
        id: 'input',
        format: 'float32' as const,
        storage: {kind: 'buffer' as const, values: harness.floats([1, NaN])}
      },
      radius: 1,
      output: {
        values: harness.output('values', 'float32', 2),
        validity: harness.output('validity', 'uint32', 2)
      }
    };
    const settings = inherited
      ? Object.assign(Object.create({noDataPolicy: 'ignore', borderMode: 'nodata'}), fields)
      : Object.defineProperties(fields, {
          noDataPolicy: {value: 'ignore'},
          borderMode: {value: 'nodata'}
        });
    const actual = await harness.run(new GPURasterRankFilter(settings));
    expect(actual['validity']).toEqual([1, 0]);
    expect(actual['values'][0]).toBe(1);
    expect(actual['values'][1]).toBeNaN();
  }
});

it('W3-11 rank keeps the missing center invalid even when ignoring missing neighbors', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  const result = await harness.run(
    new GPURasterRankFilter({
      grid: {width: 3, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'point'},
      input: {
        id: 'input',
        format: 'float32',
        storage: {kind: 'buffer', values: harness.floats([1, NaN, 3])}
      },
      radius: 1,
      noDataPolicy: 'ignore',
      borderMode: 'nodata',
      output: {
        values: harness.output('values', 'float32', 3),
        validity: harness.output('validity', 'uint32', 3)
      }
    })
  );
  expect(result['validity'][1]).toBe(0);
  expect(result['values'][1]).toBeNaN();
});

it('rank filter pins the upper median at an offset core boundary with a proper halo', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Full raster core: columns [2,4), row [2,3). Radius-one crop is columns [1,5),
  // rows [1,4). Around core (2,2), masked values leave [1,2,2,4,8,9]. At q=.5,
  // floor(q*n)=3 selects 4; the lower median is 2, so averaging/lower rank cannot pass.
  const values = Array(36).fill(50) as number[];
  const mask = Array(36).fill(1) as number[];
  for (const [index, value] of [
    [7, 1],
    [8, 2],
    [9, 2],
    [13, 4],
    [14, 8],
    [15, 9]
  ] as const)
    values[index] = value;
  for (const index of [19, 20, 21, 15]) mask[index] = 0;
  // Keep six valid neighbors, and use the adjacent core center (3,2) as invalid.
  mask[19] = 1;
  values[19] = 9;
  const cropValues = [1, 2, 3].flatMap(row => values.slice(row * 6 + 1, row * 6 + 5));
  const cropMask = [1, 2, 3].flatMap(row => mask.slice(row * 6 + 1, row * 6 + 5));
  const full = createRasterTimeSeriesHarness(device);
  const cropped = createRasterTimeSeriesHarness(device);
  async function run(
    harness: typeof full,
    data: number[],
    valid: number[],
    width: number,
    height: number
  ) {
    return harness.run(
      new GPURasterRankFilter({
        grid: {
          width,
          height,
          affine: [1, 0, width === 6 ? 0 : 1, 0, 1, width === 6 ? 0 : 1],
          pixelInterpretation: 'area'
        },
        input: {
          id: 'input',
          format: 'float32',
          storage: {kind: 'buffer', values: harness.floats(data)},
          validity: harness.integers(valid)
        },
        radius: 1,
        quantile: 0.5,
        noDataPolicy: 'ignore',
        borderMode: 'nodata',
        output: {
          values: harness.output('values', 'float32', data.length),
          validity: harness.output('validity', 'uint32', data.length)
        }
      })
    );
  }
  for (const [result, selected, invalid] of [
    [await run(full, values, mask, 6, 6), 14, 15],
    [await run(cropped, cropValues, cropMask, 4, 3), 5, 6]
  ] as const) {
    expect(result['values'][selected]).toBe(4);
    expect(result['validity'][selected]).toBe(1);
    expect(result['values'][invalid]).toBeNaN();
    expect(result['validity'][invalid]).toBe(0);
  }
});
