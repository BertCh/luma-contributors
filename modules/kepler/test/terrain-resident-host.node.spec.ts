// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {expect, test} from 'vitest';
import {createRasterTerrainHost} from '../../../examples/deck/kepler-compute/src/analysis4-raster-terrain-host';
import {
  createNumericRasterSource,
  createNumericRasterTile
} from '../../../examples/deck/kepler-compute/src/raster-tile-ingest';

function makeSource() {
  const values = Float32Array.from({length: 64}, (_, index) => (index % 8) * 10);
  return createNumericRasterSource({
    id: 'ramp',
    kind: 'xyz-tiles',
    crs: 'EPSG:4326',
    tiles: [
      createNumericRasterTile({
        id: 'tile',
        width: 8,
        height: 8,
        extent: [0, 0, 0.008, 0.008],
        zoom: 3,
        bands: [{values}]
      })
    ]
  });
}

test('resident publication skips the CPU table and the CPU mosaic', async () => {
  const published: string[] = [];
  const host = createRasterTerrainHost({
    getDevice: () => ({}) as Device,
    getTilesSource: () => makeSource(),
    publishTable: async () => {
      published.push('table');
    },
    publishResident: async options => {
      published.push(`resident:${options.operation}`);
      return {rowCount: 64, contributors: ['GPUTerrainDerivatives'], readbackBytes: 72};
    }
  });
  const result = await host.runTerrainAction({
    input: {kind: 'tiles', rasterLayerId: 'ramp'},
    operation: 'slope',
    publication: 'resident'
  });
  expect(published).toEqual(['resident:slope']);
  expect(result.publication).toBe('resident');
  expect(result.published).toBe(true);
  expect(result.publicationReadbackBytes).toBe(72);
  expect(result.exportAvailable).toBe(true);
  expect(host.getResultMosaic()).toBeUndefined();
});

test('without a device the resident publication is skipped and the table path answers', async () => {
  const published: string[] = [];
  const host = createRasterTerrainHost({
    getDevice: () => undefined,
    getTilesSource: () => makeSource(),
    publishTable: async () => {
      published.push('table');
    },
    publishResident: async () => {
      throw new Error('should not run without a device');
    }
  });
  const result = await host.runTerrainAction({
    input: {kind: 'tiles', rasterLayerId: 'ramp'},
    operation: 'slope',
    publication: 'resident'
  });
  expect(published).toEqual(['table']);
  expect(result.publication).toBe('table');
});

test('GeoTIFF export after a resident publication recomputes with a readback', async () => {
  const host = createRasterTerrainHost({
    getDevice: () => ({}) as Device,
    getTilesSource: () => makeSource(),
    publishTable: async () => {},
    publishResident: async () => ({rowCount: 64, contributors: [], readbackBytes: 0})
  });
  await host.runTerrainAction({
    input: {kind: 'tiles', rasterLayerId: 'ramp'},
    operation: 'slope',
    publication: 'resident'
  });
  expect(host.hasResult()).toBe(true);
  const exported = await host.exportTerrainGeoTIFF();
  expect(exported.filename.endsWith('.tif')).toBe(true);
  expect(exported.data.byteLength).toBeGreaterThan(100);
});
