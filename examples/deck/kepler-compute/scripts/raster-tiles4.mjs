// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {pinViteModules} from './pin-vite-modules.mjs';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

// TILES4: GPU zonal precision safeguards (centered uploads, compensated merge, error bound) against
// the CPU float64 reference on a band with a large offset, plus the WGSL plan for index/custom math.
// Mirrors scripts/raster-tiles.mjs: pass the dev server URL as the first argument. Headless only.
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
    const shader = await import('/src/raster-wgsl-shader.ts');
    const reader = await import('/src/raster-geotiff-reader.ts');
    const base = {imageBandCount: 8, hasColormap: false, hasRgba: false, hasMask: false, minPixelValue: 0, maxPixelValue: 1000, linearRescalingFactor: [0, 1]};
    const indexPlan = shader.planRasterRender({...base, bandCombination: 'normalizedDifference'});
    const customPlan = shader.planRasterRender({...base, bandCombination: 'custom', bandMathExpression: '(b5 - b8) / (b5 + b8)'});
    const levels = reader.selectGeoTiffLevel([{width: 1000, height: 1000, samplesPerPixel: 1, bitsPerSample: 32}, {width: 500, height: 500, samplesPerPixel: 1, bitsPerSample: 32}], {maximumDecodedByteLength: 1500000});
    const device = window.__keplerComputeTest.device?.();
    let zonal = null;
    if (device) {
      const ingest = await import('/src/raster-tile-ingest.ts');
      const gpuZonal = await import('/src/raster-gpu-zonal.ts');
      const cpuZonal = await import('/src/raster-zonal.ts');
      const size = 64;
      const values = Float32Array.from({length: size * size}, (_, index) => 16777216 + (index % 5));
      const tile = ingest.createNumericRasterTile({id: 'precision-tile', width: size, height: size, extent: [0, 0, size, size], bands: [{values, noDataValue: -9999}]});
      const source = ingest.createNumericRasterSource({id: 'precision-source', kind: 'single-raster', crs: 'EPSG:4326', tiles: [tile]});
      const polygon = [[[[1, 1], [63, 1], [63, 63], [1, 63], [1, 1]]]];
      const gpuResult = await gpuZonal.computeRasterZonalStatisticsGpu({device, source, zonePolygons: [polygon[0]], bandIndex: 1});
      const cpu = cpuZonal.computeRasterZonalStatistics(source, polygon[0]);
      zonal = {gpu: gpuResult.statistics[0], cpuSum: cpu.sum, cpuCount: cpu.count, centeredTileCount: gpuResult.centeredTileCount, note: gpuResult.precisionNote};
    }
    return {indexMode: indexPlan?.mode, customSlots: customPlan?.bandPositions.length, customMask: customPlan?.usedBandMask, levels, zonal};
  });
  assert.equal(report.indexMode, 'index');
  assert.equal(report.customSlots, 8);
  assert.equal(report.customMask, 0b10010000);
  assert.equal(report.levels.levelIndex, 1);
  if (report.zonal) {
    assert.equal(report.zonal.gpu.count, report.zonal.cpuCount);
    // Centered upload: the float32 reduction runs on values 0..4, so the sum is exact in float64.
    assert.ok(Math.abs(report.zonal.gpu.sum - report.zonal.cpuSum) <= 1, `sum ${report.zonal.gpu.sum} vs ${report.zonal.cpuSum}`);
    assert.ok(report.zonal.centeredTileCount >= 1);
    assert.ok(report.zonal.gpu.sumErrorBound >= 0);
    assert.match(report.zonal.note, /float32/);
  }
  assert.deepEqual(failures, []);
  process.stdout.write(`${JSON.stringify({passed: true, gpuChecked: report.zonal !== null, checks: ['index mode plan', 'eight band custom plan', 'overview level selection', 'centered gpu zonal vs cpu']}, null, 2)}\n`);
} finally {
  await browser.close();
}
