// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Headless browser check for the Terrain panel (ANALYSIS4-1, PLATFORM4-5). A synthetic raster tile
// layer (dev-only injectRasterTestLayer hook) is analysed through the panel: a plain GPU run
// publishes polygon cells, then "Keep cells on the GPU" publishes a resident point dataset without
// cell readback; GeoTIFF export still works afterwards (it recomputes with a readback).
// Needs a running dev server; default headless Playwright only.
//
// Usage: node scripts/raster-terrain-gpu.mjs [url]
import assert from 'node:assert/strict';
import {
  getRasterTileExtent,
  injectElevationRaster,
  openRasterPage,
  publishTilePolygons
} from './raster-panels-utils.mjs';

const {page, failures, close} = await openRasterPage(process.argv[2]);
try {
  await injectElevationRaster(page, 'terrain-test-raster', 'Elevation ramp');
  // Publishing also re-renders the app so the panel sees the injected layer.
  await publishTilePolygons(page, await getRasterTileExtent(page));
  const panel = page.getByTestId('raster-terrain-panel');
  await panel.waitFor({timeout: 30000});
  const status = panel.getByTestId('terrain-status');
  const publicationCount = async () =>
    (await page.evaluate(() => window.__keplerComputeTest.getState())).publications.length;
  const run = async () => {
    const before = await publicationCount();
    await panel.getByTestId('terrain-run').click();
    await page.waitForFunction(
      count => window.__keplerComputeTest.getState().publications.length > count,
      before,
      {timeout: 90000}
    );
    await page.waitForFunction(
      () => document.querySelector('[data-testid="terrain-run"]')?.textContent === 'Compute terrain',
      undefined,
      {timeout: 90000}
    );
    const text = (await status.textContent()) ?? '';
    return text;
  };

  await panel.getByTestId('terrain-operation-select').selectOption('slope');
  await panel.getByTestId('terrain-engine-select').selectOption('gpu');
  const tableText = await run();
  console.log(`table run: ${tableText}`);
  assert.match(tableText, /Published \d+ cells as a native dataset/, 'table publication');
  // A successful GPU run shows no fallback note (the note renders only when fallbackReason is set).
  assert.equal(await panel.getByTestId('terrain-fallback-reason').count(), 0, 'no fallback note on a GPU run');

  // Keep cells on the GPU: one row per cell with longitude/latitude columns, nodata cells are null rows.
  await panel.getByTestId('terrain-keep-on-gpu').check();
  const residentText = await run();
  console.log(`resident run: ${residentText}`);
  assert.match(residentText, /cells GPU-resident \(\d+ bytes read back\)/, 'resident publication');
  const state = await page.evaluate(() => window.__keplerComputeTest.getState());
  const residentRowCount = state.publications.at(-1).rowCount;
  assert.equal(residentRowCount, 32 * 32, 'one row per cell including nodata cells');
  const bytesMatch = /\((\d+) bytes read back\)/.exec(residentText);
  assert.ok(Number(bytesMatch[1]) < residentRowCount * 8, 'cell values were not read back');
  // Coordinate roles need CPU cells, so the resident dataset is a table without a map layer.
  assert.equal(
    state.nativeLayers.filter(layer => layer.dataId === state.publications.at(-1).id).length,
    0,
    'resident terrain cells publish as a table, not a layer'
  );

  // GeoTIFF export still works afterwards by recomputing with a readback.
  const exported = await page.evaluate(async () => {
    const result = await window.__keplerComputeTest.exportRasterTerrainGeoTIFF();
    return {filename: result.filename, byteLength: result.bytes.length, magic: result.bytes.slice(0, 2)};
  });
  console.log(`export: ${exported.filename} ${exported.byteLength} bytes`);
  assert.ok(exported.byteLength > 1000, 'GeoTIFF has content');
  assert.deepEqual(exported.magic, [73, 73], 'little-endian TIFF header');

  // Unload: removing the resident dataset layer must not throw or leak validation errors.
  assert.deepEqual(failures, [], 'no WGSL or validation errors');
  console.log('PASS Terrain panel: table and GPU-resident publication, GeoTIFF export after resident run');
} finally {
  await close();
}
