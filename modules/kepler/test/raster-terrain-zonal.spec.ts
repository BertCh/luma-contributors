// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {
  computeRasterTerrainZonalStatistics,
  computeRasterTerrainZonalStatisticsGpu
} from '../../../examples/deck/kepler-compute/src/raster-terrain-zonal-gpu';
import {
  createNumericRasterSource,
  createNumericRasterTile
} from '../../../examples/deck/kepler-compute/src/raster-tile-ingest';

const SIZE = 24;
const OPERATIONS = ['slope', 'aspect', 'hillshade', 'curvature'] as const;

function makeSource() {
  const tiles = [0, 1].map(column => {
    const width = SIZE / 2;
    const values = new Float32Array(width * SIZE);
    for (let row = 0; row < SIZE; row++)
      for (let local = 0; local < width; local++) {
        const x = column * width + local;
        values[row * width + local] = 30 * Math.sin(x / 3) * Math.cos(row / 4) + 2 * x + (row % 5);
      }
    return createNumericRasterTile({
      id: `tile-${column}`,
      width,
      height: SIZE,
      extent: [column * 0.012, 0, (column + 1) * 0.012, 0.024],
      zoom: 3,
      bands: [{values, noDataValue: -9999}]
    });
  });
  return createNumericRasterSource({id: 'bumps', kind: 'xyz-tiles', crs: 'EPSG:4326', tiles});
}

// Two disjoint zones; the first straddles the seam between the two tiles.
const ZONES = [
  [
    [
      [
        [0.008, 0.006],
        [0.016, 0.006],
        [0.016, 0.016],
        [0.008, 0.016],
        [0.008, 0.006]
      ]
    ]
  ],
  [
    [
      [
        [0.002, 0.018],
        [0.006, 0.018],
        [0.006, 0.022],
        [0.002, 0.022],
        [0.002, 0.018]
      ]
    ]
  ]
] as never;

for (const operation of OPERATIONS) {
  test(`GPU zonal ${operation} matches the CPU reference without reading the grid back`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const source = makeSource();
    const gpu = await computeRasterTerrainZonalStatisticsGpu({
      device,
      source,
      zonePolygons: ZONES,
      terrain: {operation}
    });
    expect(gpu.declinedReason).toBeUndefined();
    // Five zone columns per zone and tile; never the cell grid.
    expect(gpu.readbackByteLength).toBe(gpu.tileCount * 2 * 5 * 4);
    const cpu = await computeRasterTerrainZonalStatistics({
      source,
      zonePolygons: ZONES,
      terrain: {operation}
    });
    expect(cpu.engine).toBe('cpu-reference');
    for (let zone = 0; zone < 2; zone++) {
      const expected = cpu.statistics[zone]!;
      const actual = gpu.statistics[zone]!;
      expect(actual.count, `${operation} zone ${zone} count`).toBe(expected.count);
      const tolerance = (value: number) => Math.max(0.1, Math.abs(value) * 5e-3);
      expect(Math.abs(actual.mean! - expected.mean!)).toBeLessThan(tolerance(expected.mean!));
      expect(Math.abs(actual.minimum! - expected.minimum!)).toBeLessThan(
        tolerance(expected.minimum!) * (operation === 'aspect' ? 100 : 1)
      );
      expect(Math.abs(actual.maximum! - expected.maximum!)).toBeLessThan(
        tolerance(expected.maximum!) * (operation === 'aspect' ? 100 : 1)
      );
    }
  });
}

test('the fused path reports its engine and declines overlapping zones', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await computeRasterTerrainZonalStatistics({
    device,
    source: makeSource(),
    zonePolygons: ZONES,
    terrain: {operation: 'slope'}
  });
  expect(result.engine).toBe('gpu');
  expect(result.engineNote).toContain('no derived-grid readback');
  const overlapping = await computeRasterTerrainZonalStatistics({
    device,
    source: makeSource(),
    zonePolygons: [ZONES[0], ZONES[0]] as never,
    terrain: {operation: 'slope'}
  });
  expect(overlapping.engine).toBe('cpu-reference');
  expect(overlapping.fallbackReason).toBe('overlapping-zones');
});
