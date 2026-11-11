// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  computeRasterTerrainGrid,
  createTerrainRasterSource
} from '../../../examples/deck/kepler-compute/src/analysis2-raster-terrain';
import {runRasterAnalysis} from '../../../examples/deck/kepler-compute/src/analysis2-raster-source-binding';
import {
  createNumericRasterSource,
  createNumericRasterTile
} from '../../../examples/deck/kepler-compute/src/raster-tile-ingest';
import {encodeCloudOptimizedGeoTIFF} from '../../../examples/deck/kepler-compute/src/interpolation-cog';
import {validateCogBytes} from '../../../examples/deck/kepler-compute/scripts/validate-cog.mjs';

const SIZE = 6;
function createPlaneSource(elevationAt: (column: number, row: number) => number) {
  const values = new Float32Array(SIZE * SIZE);
  for (let row = 0; row < SIZE; row++)
    for (let column = 0; column < SIZE; column++)
      values[row * SIZE + column] = elevationAt(column, row);
  return createNumericRasterSource({
    id: 'plane',
    kind: 'single-raster',
    crs: 'EPSG:4326',
    tiles: [
      createNumericRasterTile({
        id: 'plane/0',
        width: SIZE,
        height: SIZE,
        extent: [0, 0, 0.006, 0.006],
        bands: [{values, noDataValue: -9999}]
      })
    ]
  });
}

test('slope of a tilted plane matches the analytic angle and borders are nodata', () => {
  const source = createPlaneSource(column => column * 10);
  const tile = source.tiles[0];
  const grid = computeRasterTerrainGrid(tile, source.crs, {operation: 'slope'});
  const cellSizeX = 0.001 * 111320 * Math.cos((0.003 * Math.PI) / 180);
  const expected = (Math.atan(10 / cellSizeX) * 180) / Math.PI;
  expect(grid.validity[0]).toBe(0);
  expect(grid.validity[2 * SIZE + 2]).toBe(1);
  expect(grid.values[2 * SIZE + 2]).toBeCloseTo(expected, 3);
});

test('aspect of a plane rising eastward faces west and flat cells are -1', () => {
  const tilted = createPlaneSource(column => column * 10);
  const aspect = computeRasterTerrainGrid(tilted.tiles[0], tilted.crs, {operation: 'aspect'});
  expect(aspect.values[2 * SIZE + 2]).toBeCloseTo(270, 3);
  const flat = createPlaneSource(() => 5);
  expect(computeRasterTerrainGrid(flat.tiles[0], flat.crs, {operation: 'aspect'}).values[14]).toBe(
    -1
  );
  const curvature = computeRasterTerrainGrid(flat.tiles[0], flat.crs, {operation: 'curvature'});
  expect(curvature.values[14]).toBeCloseTo(0, 6);
});

test('flat hillshade equals 255 * sin(altitude) and nodata neighbors invalidate', () => {
  const flat = createPlaneSource(() => 5);
  const shade = computeRasterTerrainGrid(flat.tiles[0], flat.crs, {
    operation: 'hillshade',
    altitudeDegrees: 45
  });
  expect(shade.values[14]).toBeCloseTo(255 * Math.SQRT1_2, 3);
  const holed = createPlaneSource((column, row) => (column === 2 && row === 2 ? -9999 : 5));
  const grid = computeRasterTerrainGrid(holed.tiles[0], holed.crs, {operation: 'slope'});
  expect(grid.validity[1 * SIZE + 1]).toBe(0);
  expect(grid.validity[3 * SIZE + 3]).toBe(0);
  expect(grid.validity[4 * SIZE + 4]).toBe(1);
});

test('terrain zonal statistics bind through the derived source', () => {
  const source = createPlaneSource(column => column * 10);
  const polygon = [
    [
      [-0.001, -0.001],
      [0.007, -0.001],
      [0.007, 0.007],
      [-0.001, 0.007],
      [-0.001, -0.001]
    ]
  ] as const;
  const result = runRasterAnalysis(source, [polygon], {
    kind: 'terrain-zonal',
    terrain: {operation: 'slope'}
  });
  expect(result.kind).toBe('statistics');
  if (result.kind === 'statistics') {
    expect(result.statistics.count).toBe(16);
    expect(result.statistics.nodataCount).toBe(20);
  }
  expect(createTerrainRasterSource(source, {operation: 'slope'}).tiles[0].bands[0].name).toBe(
    'slope'
  );
});

test('COG validator accepts the encoder output and rejects corruption', async () => {
  const width = 300;
  const height = 200;
  const values = new Float32Array(width * height).map((_, index) => index % 17);
  for (const compression of ['none', 'deflate'] as const) {
    const bytes = await encodeCloudOptimizedGeoTIFF({
      width,
      height,
      bounds: [0, 0, 3, 2],
      bands: [
        {name: 'a', values},
        {name: 'b', values}
      ],
      crs: {kind: 'geographic', epsg: 4326},
      tileSize: 128,
      compression
    });
    const report = validateCogBytes(bytes);
    expect(report.errors).toEqual([]);
    expect(report.levels.length).toBe(3);
    const corrupted = bytes.slice(0, bytes.byteLength - 40);
    // validate-cog.mjs adds `isValid` after building the report, so its inferred type lacks it.
    const corruptedReport: Record<string, unknown> = validateCogBytes(corrupted);
    expect(corruptedReport['isValid']).toBe(false);
  }
});
