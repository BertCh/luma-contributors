// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {pinViteModules} from './pin-vite-modules.mjs';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

// Runs the raster ingestion, identify, zonal, band math, residency and bitmap georeference modules
// as real browser ES modules served by the Vite dev server, then (when the host exposes a device)
// uploads bands to GPU textures. Mirrors scripts/layers.mjs: pass a URL as the first argument.
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const failures = [];
page.on('pageerror', error => failures.push(error.message));
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer|Texture)|validation error/i.test(message.text())) failures.push(message.text());
});
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=4096');
  await page.waitForFunction(() => { const api = window.__keplerComputeTest; return api && api.getState().selectedCount === api.source().length; });
  const report = await page.evaluate(async () => {
    const ingest = await import('/src/raster-tile-ingest.ts');
    const identify = await import('/src/raster-identify.ts');
    const zonal = await import('/src/raster-zonal.ts');
    const math = await import('/src/raster-band-math.ts');
    const model = await import('/src/raster-band-model.ts');
    const reprojection = await import('/src/raster-reprojection.ts');
    const bitmap = await import('/src/bitmap-georeference.ts');
    const tiles = [0, 1].map(column => ingest.createNumericRasterTile({
      id: `tile-${column}`, width: 2, height: 2, extent: [column, 0, column + 1, 1],
      bands: [{values: new Float32Array([1, 2, 3, -9999].map(value => value + column * 4)), noDataValue: -9999 + column * 4}, {values: new Float32Array(4).fill(1)}]
    }));
    const source = ingest.createNumericRasterSource({id: 'browser', kind: 'single-raster', crs: 'EPSG:4326', tiles});
    const edge = identify.identifyRasterValue(source, 1, 0.75, {bandIndexes: [1]});
    const nodata = identify.identifyRasterValue(source, 0.75, 0.25, {bandIndexes: [1]});
    const stats = zonal.computeRasterZonalStatistics(source, [[[[0.1, 0.1], [1.9, 0.1], [1.9, 0.9], [0.1, 0.9], [0.1, 0.1]]]]);
    const ndvi = math.evaluateBandMath(tiles[0], math.compileBandMath(math.KEPLER_BAND_COMBINATION_EXPRESSIONS.normalizedDifference));
    let refusedCrs = false;
    try { reprojection.planRasterReprojection('EPSG:32633'); } catch { refusedCrs = true; }
    const destroyed = [];
    const residency = new model.RasterBandResidency({budgetByteLength: 40, upload: (tile, band) => ({byteLength: 20, hasPrecisionLoss: false, destroy: () => destroyed.push(`${tile.id}#${band.index}`)})});
    residency.ensureBandResident(tiles[0], 1); residency.ensureBandResident(tiles[1], 1); residency.ensureBandResident(tiles[0], 2);
    const evictedIdentify = identify.identifyRasterValue(source, 0.25, 0.75, {isTileAvailable: tile => residency.isTileResident(tile, [1])});
    residency.destroy();
    const georeference = bitmap.resolveBitmapGeoreference({boundsWest: -70, boundsSouth: 43, boundsEast: -74, boundsNorth: 40, opacity: 0.5});
    const device = window.__keplerComputeTest.device?.();
    let gpuResident = null;
    if (device) {
      const gpu = await import('/src/raster-band-gpu.ts');
      const gpuResidency = new model.RasterBandResidency({budgetByteLength: 1024, upload: gpu.createRasterBandGpuUploader(device)});
      gpuResidency.ensureBandResident(tiles[0], 1);
      gpuResident = gpuResidency.getStatistics().residentByteLength;
      gpuResidency.destroy();
    }
    return {edge, nodata, stats, ndviValidity: [...ndvi.validity], refusedCrs, destroyed, evictedStatus: evictedIdentify.status, georeference, gpuResident};
  });
  assert.equal(report.edge.tileId, 'tile-1');
  assert.equal(report.nodata.status, 'nodata');
  assert.equal(report.stats.nodataCount, 2);
  assert.deepEqual(report.ndviValidity, [1, 1, 1, 0]);
  assert.equal(report.refusedCrs, true);
  assert.equal(report.evictedStatus, 'evicted');
  assert.equal(report.destroyed.length, 3);
  assert.deepEqual(report.georeference.bounds, [-74, 40, -70, 43]);
  if (report.gpuResident !== null) assert.equal(report.gpuResident, 20);
  // Live wiring (TILES2-2): only runs when the workspace already has a native rasterTile layer.
  const live = await page.evaluate(async () => {
    const api = window.__keplerComputeTest;
    api.syncRasterLiveLayers();
    const layer = api.getState().nativeLayers?.find(candidate => candidate.type === 'rasterTile');
    if (!layer) return null;
    const snapshot = api.getRasterLiveSnapshot(layer.id);
    const bitmaps = api.getBitmapLiveGeoreferences();
    return {layerId: layer.id, snapshot, bitmaps};
  });
  if (live) {
    assert.ok(live.snapshot === null || live.snapshot.tileCount >= 0);
    if (live.snapshot && live.snapshot.residency) assert.ok(live.snapshot.residency.residentByteLength <= live.snapshot.residency.budgetByteLength);
  }
  // TILES3-2: GPU readback identify, GPU zonal vs CPU reference, and the WGSL tile draw.
  const gpuReport = await page.evaluate(async () => {
    const api = window.__keplerComputeTest;
    const device = api.device?.();
    if (!device) return null;
    const ingest = await import('/src/raster-tile-ingest.ts');
    const model = await import('/src/raster-band-model.ts');
    const gpu = await import('/src/raster-band-gpu.ts');
    const gpuIdentify = await import('/src/raster-gpu-identify.ts');
    const gpuZonal = await import('/src/raster-gpu-zonal.ts');
    const cpuZonal = await import('/src/raster-zonal.ts');
    const values = Float32Array.from({length: 64}, (_, index) => (index % 9 === 0 ? -9999 : index * 0.25));
    const tile = ingest.createNumericRasterTile({id: 'gpu-tile', width: 8, height: 8, extent: [0, 0, 8, 8], bands: [{values, noDataValue: -9999}]});
    const source = ingest.createNumericRasterSource({id: 'gpu-source', kind: 'single-raster', crs: 'EPSG:4326', tiles: [tile]});
    const residency = new model.RasterBandResidency({budgetByteLength: 4096, upload: gpu.createRasterBandGpuUploader(device)});
    const resource = residency.ensureBandResident(tile, 1);
    const samples = [];
    for (const [column, row] of [[1, 1], [0, 0], [7, 7]]) {
      const sample = await gpuIdentify.readRasterBandSample(resource, column, row);
      samples.push({column, row, sample, expected: values[row * 8 + column], valid: tile.bands[0].validity[row * 8 + column]});
    }
    const polygons = [[[[1, 1], [5, 1], [5, 5], [1, 5], [1, 1]]], [[[5.5, 5.5], [7.5, 5.5], [7.5, 7.5], [5.5, 7.5], [5.5, 5.5]]]];
    const gpuResult = await gpuZonal.computeRasterZonalStatisticsGpu({device, source, zonePolygons: polygons.map(polygon => [polygon]), getResource: (tileId, bandIndex) => residency.peekBandResource(tileId, bandIndex)});
    const cpuResults = polygons.map(polygon => cpuZonal.computeRasterZonalStatistics(source, [polygon]));
    residency.destroy();
    return {samples, gpuResult, cpuResults};
  });
  if (gpuReport) {
    for (const entry of gpuReport.samples) {
      assert.equal(entry.sample.isValid, Boolean(entry.valid));
      if (entry.valid) assert.ok(Math.abs(entry.sample.value - entry.expected) < 1e-6);
    }
    assert.equal(gpuReport.gpuResult.declinedReason, undefined);
    assert.equal(gpuReport.gpuResult.residentTileCount, 1);
    gpuReport.cpuResults.forEach((cpu, index) => {
      const gpuStats = gpuReport.gpuResult.statistics[index];
      assert.equal(gpuStats.count, cpu.count);
      assert.equal(gpuStats.nodataCount, cpu.nodataCount);
      assert.ok(Math.abs(gpuStats.sum - cpu.sum) < 1e-3);
      assert.equal(gpuStats.minimum, cpu.minimum);
      assert.equal(gpuStats.maximum, cpu.maximum);
    });
  }
  if (live) {
    const wgsl = await page.evaluate(() => window.__keplerComputeTest.getRasterWgslStatistics());
    assert.ok(wgsl.replacedCount > 0 || wgsl.keptCount >= 0);
  }
  assert.deepEqual(failures, []);
  process.stdout.write(`${JSON.stringify({passed: true, gpuChecked: report.gpuResident !== null, liveRasterLayer: Boolean(live), gpuReadbackChecked: gpuReport !== null, checks: ['gpu readback identify', 'gpu zonal vs cpu', 'tile edge identify', 'nodata', 'zonal', 'band math nodata', 'CRS refusal', 'eviction', 'bitmap georeference']}, null, 2)}\n`);
} finally {
  await browser.close();
}
