// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  createTerrainRasterSourceAsync,
  padRasterTileWithHalo
} from '../../../examples/deck/kepler-compute/src/analysis3-raster-terrain-gpu';
import {
  createNumericRasterSource,
  createNumericRasterTile
} from '../../../examples/deck/kepler-compute/src/raster-tile-ingest';

function createTwoTileSource() {
  const tiles = [0, 1].map(column =>
    createNumericRasterTile({
      id: `tile-${column}`,
      width: 3,
      height: 3,
      extent: [column * 0.003, 0, (column + 1) * 0.003, 0.003],
      zoom: 2,
      bands: [{values: Float32Array.from({length: 9}, (_, index) => column * 100 + index)}]
    })
  );
  return createNumericRasterSource({id: 'two', kind: 'xyz-tiles', crs: 'EPSG:4326', tiles});
}

test('without a device the terrain source falls back to the CPU reference with a visible note', async () => {
  const source = createTwoTileSource();
  const result = await createTerrainRasterSourceAsync(source, {operation: 'slope'});
  expect(result.engine).toBe('cpu-reference');
  expect(result.fallbackReason).toBe('no GPU device');
  expect(result.engineNote).toBe('Terrain slope: CPU reference (no GPU device)');
  expect(result.source.tiles).toHaveLength(2);
  const forced = await createTerrainRasterSourceAsync(
    source,
    {operation: 'aspect'},
    {engine: 'cpu'}
  );
  expect(forced.engineNote).toBe('Terrain aspect: CPU reference');
});

test('halo padding copies the adjacent tile column and leaves unmatched sides invalid', () => {
  const source = createTwoTileSource();
  const padded = padRasterTileWithHalo(source.tiles, source.tiles[0], 1);
  expect(padded.width).toBe(5);
  expect(padded.height).toBe(5);
  // East halo column holds the first column of tile 1 (values 100, 103, 106).
  expect(padded.elevation[1 * 5 + 4]).toBe(100);
  expect(padded.elevation[2 * 5 + 4]).toBe(103);
  expect(padded.validity[1 * 5 + 4]).toBe(1);
  // No neighbor to the west or north.
  expect(padded.validity[2 * 5]).toBe(0);
  expect(padded.validity[0 * 5 + 2]).toBe(0);
  expect(padded.extent[0]).toBeCloseTo(-0.001, 9);
});
