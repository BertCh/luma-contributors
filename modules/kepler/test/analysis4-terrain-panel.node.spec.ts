// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {createRasterTerrainHost} from '../../../examples/deck/kepler-compute/src/analysis4-raster-terrain-host';
import {
  createNumericRasterSourceFromGrid,
  createTerrainCellTable,
  createTerrainMosaic
} from '../../../examples/deck/kepler-compute/src/analysis4-terrain-grid';
import {
  createNumericRasterSource,
  createNumericRasterTile
} from '../../../examples/deck/kepler-compute/src/raster-tile-ingest';

function createRampGrid(width: number, height: number) {
  return Float32Array.from({length: width * height}, (_, index) => (index % width) * 10);
}

test('mosaic joins same-zoom tiles and cell table has one polygon per valid cell', () => {
  const tiles = [0, 1].map(column =>
    createNumericRasterTile({
      id: `tile-${column}`,
      width: 4,
      height: 4,
      extent: [column * 0.004, 0, (column + 1) * 0.004, 0.004],
      zoom: 3,
      bands: [{values: createRampGrid(4, 4)}]
    })
  );
  const source = createNumericRasterSource({
    id: 'two',
    kind: 'xyz-tiles',
    crs: 'EPSG:4326',
    tiles
  });
  const mosaic = createTerrainMosaic(source);
  expect(mosaic.width).toBe(8);
  expect(mosaic.height).toBe(4);
  expect(mosaic.validCellCount).toBe(32);
  const table = createTerrainCellTable(mosaic, 'slope');
  expect(table?.numRows).toBe(32);
  expect(table?.schema.fields.map(field => field.name)).toContain('slope');
  const geometry = JSON.parse(String(table?.getChild('geometry')?.get(0)));
  expect(geometry.type).toBe('Polygon');
  expect(geometry.coordinates[0]).toHaveLength(5);
});

test('terrain host runs on a derived grid, publishes cells and exports a GeoTIFF', async () => {
  const width = 6;
  const height = 6;
  const published: {label: string; rowCount: number}[] = [];
  const host = createRasterTerrainHost({
    getDevice: () => undefined,
    getTilesSource: () => undefined,
    getDerivedGrid: () => ({
      isAvailable: () => true,
      read: async () => ({
        id: 'grid',
        sourceId: 'dataset',
        operation: 'idw-points',
        width,
        height,
        bounds: [0, 0, 0.006, 0.006],
        values: createRampGrid(width, height)
      })
    }),
    publishTable: async ({table, label}) => {
      published.push({label, rowCount: table.numRows});
    }
  });
  expect(host.isDerivedGridAvailable()).toBe(true);
  const result = await host.runTerrainAction({input: {kind: 'derived-grid'}, operation: 'slope'});
  expect(result.published).toBe(true);
  expect(result.engine).toBe('cpu-reference');
  // The border ring is nodata, so only the 4 x 4 interior has values.
  expect(result.rowCount).toBe(16);
  expect(published).toHaveLength(1);
  const exported = await host.exportTerrainGeoTIFF();
  expect(exported.filename.endsWith('.tif')).toBe(true);
  expect(exported.data.byteLength).toBeGreaterThan(100);
  const noPublish = await host.runTerrainAction({
    input: {kind: 'derived-grid'},
    operation: 'hillshade',
    publish: false
  });
  expect(noPublish.published).toBe(false);
  expect(noPublish.exportAvailable).toBe(true);
  expect(published).toHaveLength(1);
});

test('wrapping a derived grid keeps NaN cells invalid', () => {
  const values = createRampGrid(3, 3);
  values[4] = Number.NaN;
  const source = createNumericRasterSourceFromGrid({
    id: 'g',
    width: 3,
    height: 3,
    bounds: [0, 0, 1, 1],
    values
  });
  expect(source.tiles[0].bands[0].validity[4]).toBe(0);
});
