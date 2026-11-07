// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';
import {computeRasterTerrainGrid} from '../../../examples/deck/kepler-compute/src/analysis2-raster-terrain';
import {
  createTerrainRasterSourceAsync,
  padRasterTileWithHalo
} from '../../../examples/deck/kepler-compute/src/analysis3-raster-terrain-gpu';
import {
  createNumericRasterSource,
  createNumericRasterTile,
  type NumericRasterCrs
} from '../../../examples/deck/kepler-compute/src/raster-tile-ingest';

// Oracle policy: the CPU Horn implementation (analysis2-raster-terrain.ts) is the oracle, run with
// the 'wgs84-per-row' ground model that the GPU contributors use (float32 on the GPU, float64 on
// the CPU), so parity is checked to 0.2% (aspect: 0.2 degrees) instead of the older 0.5% / 3%
// allowances that covered the tile-center spherical approximation.

const SIZE = 24;
const OPERATIONS = ['slope', 'aspect', 'hillshade', 'curvature'] as const;

function elevationAt(column: number, row: number): number {
  return 30 * Math.sin(column / 3) * Math.cos(row / 4) + 2 * column + (row % 5);
}

function createSource(crs: NumericRasterCrs, tileColumn = 0, singleTile = true) {
  const extent: [number, number, number, number] =
    crs === 'EPSG:4326' ? [0, 0, 0.024, 0.024] : [0, 0, 2400, 2400];
  const width = singleTile ? SIZE : SIZE / 2;
  const tiles = [];
  for (const column of singleTile ? [0] : [0, 1]) {
    const values = new Float32Array(width * SIZE);
    for (let row = 0; row < SIZE; row++)
      for (let local = 0; local < width; local++)
        values[row * width + local] = elevationAt(column * width + local, row);
    const span = (extent[2] - extent[0]) / (singleTile ? 1 : 2);
    tiles.push(
      createNumericRasterTile({
        id: `tile-${column + tileColumn}`,
        width,
        height: SIZE,
        extent: [extent[0] + column * span, extent[1], extent[0] + (column + 1) * span, extent[3]],
        zoom: 3,
        bands: [{values, noDataValue: -9999}]
      })
    );
  }
  return createNumericRasterSource({id: 'bumps', kind: 'xyz-tiles', crs, tiles});
}

async function getDevice() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

for (const crs of ['EPSG:4326', 'EPSG:3857'] as const) {
  test(`GPU terrain operators match the CPU oracle on ${crs}`, async () => {
    const device = await getDevice();
    const source = createSource(crs);
    const tolerance = 2e-3;
    for (const operation of OPERATIONS) {
      const gpu = await createTerrainRasterSourceAsync(
        source,
        {operation},
        {device, useNeighborHalo: false}
      );
      expect(gpu.engine, `${operation}: ${gpu.fallbackReason ?? ''}`).toBe('gpu');
      expect(gpu.engineNote).toContain('GPU');
      const cpu = computeRasterTerrainGrid(source.tiles[0], crs, {
        operation,
        groundModel: 'wgs84-per-row'
      });
      const band = gpu.source.tiles[0].bands[0];
      for (let sample = 0; sample < SIZE * SIZE; sample++) {
        expect(band.validity[sample], `${operation} validity ${sample}`).toBe(cpu.validity[sample]);
        if (!cpu.validity[sample]) continue;
        const expected = cpu.values[sample];
        const actual = band.values[sample];
        if (operation === 'aspect') {
          const difference = Math.abs(actual - expected);
          expect(Math.min(difference, 360 - difference)).toBeLessThan(0.2);
        } else {
          expect(Math.abs(actual - expected), `${operation} sample ${sample}`).toBeLessThan(
            Math.max(0.05, Math.abs(expected) * tolerance)
          );
        }
      }
    }
  });
}

test('neighbor halo fills the seam between two tiles and matches a single-tile run', async () => {
  const device = await getDevice();
  const whole = createSource('EPSG:4326');
  const split = createSource('EPSG:4326', 0, false);
  const wholeResult = await createTerrainRasterSourceAsync(
    whole,
    {operation: 'slope'},
    {device, useNeighborHalo: false}
  );
  const splitResult = await createTerrainRasterSourceAsync(
    split,
    {operation: 'slope'},
    {device, useNeighborHalo: true}
  );
  expect(splitResult.engine).toBe('gpu');
  const width = SIZE / 2;
  // Seam columns: last of tile 0 and first of tile 1.
  for (let row = 1; row < SIZE - 1; row++) {
    const left = splitResult.source.tiles[0].bands[0];
    const right = splitResult.source.tiles[1].bands[0];
    expect(left.validity[row * width + width - 1]).toBe(1);
    expect(right.validity[row * width]).toBe(1);
    expect(left.values[row * width + width - 1]).toBeCloseTo(
      wholeResult.source.tiles[0].bands[0].values[row * SIZE + width - 1],
      2
    );
    expect(right.values[row * width]).toBeCloseTo(
      wholeResult.source.tiles[0].bands[0].values[row * SIZE + width],
      2
    );
  }
  // The outer border of the mosaic stays nodata, like the CPU reference.
  expect(splitResult.source.tiles[0].bands[0].validity[5 * width]).toBe(0);
  const padded = padRasterTileWithHalo(split.tiles, split.tiles[0], 1);
  expect(padded.width).toBe(width + 2);
  expect(padded.validity[(1 + 3) * padded.width + width + 1]).toBe(1);
});
