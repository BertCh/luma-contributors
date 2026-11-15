// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {RASTER_ZONE_NONE} from '../../../examples/deck/kepler-compute/src/raster-gpu-zonal-zones';
import {
  computeRasterTerrainZonalStatistics,
  getTerrainOutputScale,
  padZoneIds
} from '../../../examples/deck/kepler-compute/src/raster-terrain-zonal-gpu';
import {
  createNumericRasterSource,
  createNumericRasterTile
} from '../../../examples/deck/kepler-compute/src/raster-tile-ingest';

const SIZE = 16;

function makeSource() {
  const values = new Float32Array(SIZE * SIZE);
  for (let row = 0; row < SIZE; row++)
    for (let column = 0; column < SIZE; column++) values[row * SIZE + column] = 4 * column + row;
  return createNumericRasterSource({
    id: 'ramp',
    kind: 'xyz-tiles',
    crs: 'EPSG:4326',
    tiles: [
      createNumericRasterTile({
        id: 't0',
        width: SIZE,
        height: SIZE,
        extent: [0, 0, 0.016, 0.016],
        zoom: 3,
        bands: [{values, noDataValue: -9999}]
      })
    ]
  });
}

const ZONE = [
  [
    [0.004, 0.004],
    [0.012, 0.004],
    [0.012, 0.012],
    [0.004, 0.012],
    [0.004, 0.004]
  ]
] as const;

test('padZoneIds leaves the halo without a zone', () => {
  const zones = new Uint32Array([1, 2, 3, 4]);
  const padded = padZoneIds(zones, 2, 2, 4, 4);
  expect(padded[0]).toBe(RASTER_ZONE_NONE);
  expect(Array.from(padded.slice(5, 7))).toEqual([1, 2]);
  expect(Array.from(padded.slice(9, 11))).toEqual([3, 4]);
  expect(padded[15]).toBe(RASTER_ZONE_NONE);
  expect(padZoneIds(zones, 2, 2, 2, 2)).toBe(zones);
});

test('output scales match the Kepler units', () => {
  expect(getTerrainOutputScale('slope')).toBe(1);
  expect(getTerrainOutputScale('hillshade')).toBe(255);
  expect(getTerrainOutputScale('curvature')).toBe(-100);
});

test('without a device the zonal terrain falls back to the CPU reference with a reason', async () => {
  const result = await computeRasterTerrainZonalStatistics({
    source: makeSource(),
    zonePolygons: [[ZONE as never], null],
    terrain: {operation: 'slope'}
  });
  expect(result.engine).toBe('cpu-reference');
  expect(result.fallbackReason).toBe('no GPU device');
  expect(result.engineNote).toContain('CPU reference');
  expect(result.readbackByteLength).toBe(0);
  expect(result.statistics[0]!.count).toBeGreaterThan(0);
  expect(result.statistics[0]!.mean).toBeGreaterThan(0);
  expect(result.statistics[1]).toBeNull();
});

test('terrain fallback preserves same-zoom tile seam populations with neighbor halos', async () => {
  const {createTerrainRasterSourceAsync} = await import(
    '../../../examples/deck/kepler-compute/src/analysis3-raster-terrain-gpu'
  );
  const source = makeSource();
  const left = source.tiles[0];
  const right = createNumericRasterTile({
    id: 't1',
    width: SIZE,
    height: SIZE,
    zoom: left.zoom,
    extent: [0.016, 0, 0.032, 0.016],
    bands: [
      {
        values: Float32Array.from(
          {length: SIZE * SIZE},
          (_, index) => 4 * (SIZE + (index % SIZE)) + Math.floor(index / SIZE)
        )
      }
    ]
  });
  const mosaic = {...source, tiles: [left, right]};
  const withHalo = await createTerrainRasterSourceAsync(
    mosaic,
    {operation: 'slope'},
    {engine: 'cpu'}
  );
  const withoutHalo = await createTerrainRasterSourceAsync(
    mosaic,
    {operation: 'slope'},
    {engine: 'cpu', useNeighborHalo: false}
  );
  const row = 8;
  expect(withHalo.source.tiles[0].bands[0].validity[row * SIZE + SIZE - 1]).toBe(1);
  expect(withHalo.source.tiles[1].bands[0].validity[row * SIZE]).toBe(1);
  expect(withoutHalo.source.tiles[0].bands[0].validity[row * SIZE + SIZE - 1]).toBe(0);
  expect(withHalo.source.tiles[0].bands[0].values[row * SIZE + SIZE - 1]).toBeCloseTo(
    withHalo.source.tiles[0].bands[0].values[row * SIZE + SIZE - 2],
    6
  );
  expect(withHalo.source.tiles[0].bands[0].validity[row * SIZE]).toBe(0);
});
