// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  RASTER_TIMESERIES_BINDINGS,
  validateRasterTimeStack,
  runRasterTimeSeries
} from '../src/analysis/raster-timeseries';
import {createRasterTimeSeriesHost} from '../../../examples/deck/kepler-compute/src/raster-timeseries-host';
import {
  bindRasterTimeStack,
  createTimeSeriesOutputSource
} from '../../../examples/deck/kepler-compute/src/raster-timeseries-stack';
import {
  createNumericRasterSource,
  createNumericRasterTile
} from '../../../examples/deck/kepler-compute/src/raster-tile-ingest';

function makeSource(offset = 0) {
  return createNumericRasterSource({
    id: `source-${offset}`,
    kind: 'xyz-tiles',
    crs: 'EPSG:4326',
    tiles: [0, 1].map(column =>
      createNumericRasterTile({
        id: `tile-${column}-${offset}`,
        width: 2,
        height: 1,
        extent: [column * 2, 0, column * 2 + 2, 1],
        zoom: 1,
        bands: [
          {values: new Float32Array([offset + column, offset + 10 + column])},
          {values: new Float32Array([offset + 20 + column, -9999]), noDataValue: -9999}
        ]
      })
    )
  });
}
const SLICES = [
  {time: '2020-01-01T00:00:00Z', bands: [1]},
  {time: '2020-02-01T00:00:00Z', layerId: 'second', bands: [2]}
];
const first = makeSource();
const second = makeSource(100);
const getSource = (id: string) => (id === 'first' ? first : id === 'second' ? second : undefined);

describe('raster time-stack binding', () => {
  test('matches grids independently of tile IDs and source tile order; preserves missing samples', () => {
    const bound = bindRasterTimeStack('first', SLICES, id =>
      id === 'second' ? {...second, tiles: [...second.tiles].reverse()} : getSource(id)
    );
    expect(Array.from(bound.stack.values)).toEqual([0, 120, 10, NaN, 1, 121, 11, NaN]);
    expect(bound.stack.times).toEqual([Date.UTC(2020, 0, 1), Date.UTC(2020, 1, 1)]);
    expect(bound.stack.cellCount).toBe(4);
  });
  test('supports time-major bands and multi-band observations in one source', () => {
    const bound = bindRasterTimeStack(
      'first',
      [
        {...SLICES[0], bands: [1, 2]},
        {...SLICES[1], layerId: 'first', bands: [2, 1]}
      ],
      getSource
    );
    expect(Array.from(bound.stack.values.slice(0, 8))).toEqual([0, 20, 20, 0, 10, NaN, NaN, 10]);
  });
  test('refuses missing bands, dates, differing grids, CRS, coverage and duplicate grids', () => {
    expect(() =>
      bindRasterTimeStack('first', [SLICES[0], {...SLICES[1], bands: [3]}], getSource)
    ).toThrow(/missing band/);
    for (const time of ['2020-01-01', '2020-01-01T00:00:00Z', 'bad'])
      expect(() =>
        bindRasterTimeStack('first', [SLICES[0], {...SLICES[1], time}], getSource)
      ).toThrow();
    for (const changed of [
      {...second, crs: 'EPSG:3857' as const},
      {...second, tiles: second.tiles.slice(0, 1)},
      {...second, tiles: second.tiles.map(tile => ({...tile, extent: [0, 0, 3, 1] as const}))},
      {...second, tiles: [second.tiles[0], second.tiles[0]]}
    ])
      expect(() =>
        bindRasterTimeStack('first', SLICES, id => (id === 'second' ? changed : first))
      ).toThrow();
  });
  test('refuses overlapping zooms and output components outside the selected layout', () => {
    const overlapping = {...first, tiles: [first.tiles[0], {...first.tiles[0], zoom: 2}]};
    expect(() => bindRasterTimeStack('first', SLICES, () => overlapping)).toThrow(
      /overlapping|zoom level/
    );
    expect(() => createTimeSeriesOutputSource(first, [1, 2, 3, 4], 1, 1, 'result')).toThrow(
      /component/
    );
    const source = createTimeSeriesOutputSource(
      first,
      [1, 2, NaN, 4, 5, 6, 7, 0xffffffff],
      2,
      1,
      'mean'
    );
    expect(Array.from(source.tiles[0].bands[0].values)).toEqual([2, 4]);
    expect(Array.from(source.tiles[1].bands[0].validity)).toEqual([1, 0]);
    expect(source.tiles[1].extent).toEqual(first.tiles[1].extent);
  });
  test('checks stack shapes and capacity before allocating a GPU session', () => {
    const stack = bindRasterTimeStack('first', SLICES, getSource).stack;
    expect(() => validateRasterTimeStack({...stack, cellCount: 65537})).toThrow(/cellCount/);
    expect(() => validateRasterTimeStack({...stack, values: new Float32Array(0)})).toThrow(
      /length/
    );
    expect(() => validateRasterTimeStack({...stack, times: [1, 1]})).toThrow(/increasing/);
    expect(() => validateRasterTimeStack({...stack, times: [1.5, 2]})).toThrow(/milliseconds/);
  });
  test('all 19 contributor classes have executable raster-native registrations', () => {
    const contributors = new Set(
      RASTER_TIMESERIES_BINDINGS.flatMap(binding => [...binding.contributors])
    );
    expect(contributors.size).toBe(19);
    expect(new Set(RASTER_TIMESERIES_BINDINGS.map(binding => binding.id)).size).toBe(19);
  });
  test('host refuses publication without a GPU device', async () => {
    let publications = 0;
    const host = createRasterTimeSeriesHost({
      getDevice: () => undefined,
      getSource,
      getPolygonSource: () => undefined,
      publishTable: async () => {
        publications++;
      }
    });
    await expect(
      host.run({operation: 'rolling-mean', rasterLayerId: 'first', slices: SLICES})
    ).rejects.toThrow(/GPU device/);
    expect(publications).toBe(0);
  });
  test('regular-time methods and univariate contributors reject incompatible inputs before GPU use', async () => {
    const stack = bindRasterTimeStack(
      'first',
      [
        {...SLICES[0], bands: [1, 2]},
        {...SLICES[1], bands: [1, 2]}
      ],
      getSource
    ).stack;
    await expect(runRasterTimeSeries(undefined as never, stack, 'hants')).rejects.toThrow(
      /one band/
    );
    const irregular = {
      ...stack,
      bandCount: 1,
      values: new Float32Array(12),
      sliceCount: 3,
      times: [0, 10, 30]
    };
    await expect(
      runRasterTimeSeries(undefined as never, irregular, 'rolling-mean')
    ).rejects.toThrow(/equal acquisition/);
  });
});
