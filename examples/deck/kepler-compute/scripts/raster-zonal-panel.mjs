// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Headless browser check for the Zonal statistics panel (PLATFORM3-4). A synthetic Kepler-shaped
// raster tile layer (dev-only injectRasterTestLayer hook) with an elevation ramp is paired with a
// published polygon dataset; the panel is driven through the UI with "Statistic of" = terrain slope
// and the status must report the GPU engine. A band-value run and a CPU run are compared to the GPU
// results. Needs a running dev server; default headless Playwright only.
//
// Usage: node scripts/raster-zonal-panel.mjs [url]
import assert from 'node:assert/strict';
import {
  getRasterTileExtent,
  injectElevationRaster,
  openRasterPage,
  publishTilePolygons
} from './raster-panels-utils.mjs';

const {page, failures, close} = await openRasterPage(process.argv[2]);
try {
  await injectElevationRaster(page, 'zonal-test-raster', 'Elevation ramp');
  const polygonDatasetId = await publishTilePolygons(page, await getRasterTileExtent(page));
  const panel = page.getByTestId('raster-zonal-panel');
  await panel.waitFor({timeout: 30000});
  await page.waitForFunction(
    () => window.__keplerComputeTest.getRasterToolsPanelModel().polygonDatasets.length >= 1
  );
  const model = await page.evaluate(() => window.__keplerComputeTest.getRasterToolsPanelModel());
  assert.deepEqual(model.rasterLayers.map(layer => layer.id), ['zonal-test-raster']);
  assert.ok(model.polygonDatasets.some(dataset => dataset.id === polygonDatasetId));

  const status = panel.getByTestId('zonal-status');
  const runPanel = async (terrain, engine) => {
    await panel.getByTestId('zonal-engine-select').selectOption(engine);
    await panel.getByTestId('zonal-terrain-select').selectOption(terrain);
    const before = (await page.evaluate(() => window.__keplerComputeTest.getState())).publications
      .length;
    await panel.getByTestId('zonal-run').click();
    await page.waitForFunction(
      count => window.__keplerComputeTest.getState().publications.length > count,
      before,
      {timeout: 90000}
    );
    await page.waitForFunction(
      () =>
        document.querySelector('[data-testid="zonal-run"]')?.textContent ===
        'Compute zonal statistics',
      undefined,
      {timeout: 90000}
    );
    const text = (await status.textContent()) ?? '';
    const publications = (await page.evaluate(() => window.__keplerComputeTest.getState())).publications;
    const datasetId = publications.at(-1).id;
    const columns = await page.evaluate(id => {
      const api = window.__keplerComputeTest;
      return {
        names: api.describeRasterPolygonSource(id)?.columnNames ?? [],
        mean: api.readRasterPolygonColumn(id, 'zonal_mean'),
        count: api.readRasterPolygonColumn(id, 'zonal_count')
      };
    }, datasetId);
    return {text, ...columns};
  };

  const slopeGpuRun = await runPanel('slope', 'gpu');
  const slopeGpu = slopeGpuRun.text;
  console.log(`slope gpu: ${slopeGpu}`, JSON.stringify(slopeGpuRun.mean));
  assert.ok(slopeGpuRun.names.includes('zonal_mean'), 'zonal_* columns were published');
  assert.match(slopeGpu, /Published 2 rows \(gpu/, 'terrain slope zonal statistics ran on the GPU');
  assert.doesNotMatch(slopeGpu, /fell back/, 'no CPU fallback for slope');

  const slopeCpuRun = await runPanel('slope', 'cpu');
  const slopeCpu = slopeCpuRun.text;
  console.log(`slope cpu: ${slopeCpu}`, JSON.stringify(slopeCpuRun.mean));
  slopeGpuRun.mean.forEach((value, index) => {
    assert.ok(value > 0 && value < 90, `slope mean is a plausible angle (${value})`);
    assert.ok(Math.abs(value - slopeCpuRun.mean[index]) < 0.02 * value, 'GPU slope matches CPU');
    assert.equal(slopeGpuRun.count[index], slopeCpuRun.count[index], 'same pixel counts');
  });
  assert.match(slopeCpu, /Published 2 rows \(cpu/, 'CPU reference publishes the same zones');

  const bandGpuRun = await runPanel('', 'gpu');
  const bandGpu = bandGpuRun.text;
  console.log(`band gpu: ${bandGpu}`, JSON.stringify(bandGpuRun.mean));
  assert.ok(bandGpuRun.mean.every(value => value > 100), 'band means are elevations');
  assert.match(bandGpu, /Published 2 rows \(gpu/, 'band-value zonal statistics ran on the GPU');

  const publications = (await page.evaluate(() => window.__keplerComputeTest.getState())).publications;
  assert.ok(publications.length >= 4, 'each run published a native dataset');
  assert.ok(publications.slice(1).every(item => item.rowCount === 2), 'zonal datasets keep one row per zone');
  assert.deepEqual(failures, [], 'no WGSL or validation errors');
  console.log('PASS Zonal statistics panel: terrain slope zones on the GPU, CPU parity, band values');
} finally {
  await close();
}
