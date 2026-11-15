// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Raster time-stack panel + all raster-native contributors, using a generated 24-date stack.
// Needs the example dev server; Chromium/WebGPU runs on the driver, not the implementor sandbox.
// Usage from repository root:
//   node examples/deck/kepler-compute/scripts/raster-timeseries.mjs [http://127.0.0.1:5318/?rows=256]
// Run sequentially with raster-tiles.mjs, raster-zonal-panel.mjs and raster-terrain-gpu.mjs.
import assert from 'node:assert/strict';
import {openRasterPage, publishTilePolygons, getRasterTileExtent, RASTER_TILE_INDEX, RASTER_TILE_SIZE} from './raster-panels-utils.mjs';

const {page, failures, close} = await openRasterPage(process.argv[2], 300000);
try {
  await page.evaluate(({index, size}) => {
    const imageBands = Array.from({length: 24}, (_, slice) => ({width: size, height: size, data: Float32Array.from({length: size * size}, (_, cell) => 10 + cell * 0.01 + slice * 0.2 + Math.sin(slice * Math.PI / 6))}));
    const mask = new Float32Array(size * size).fill(1);
    mask[0] = 0;
    window.__keplerComputeTest.injectRasterTestLayer('timeseries-stack', 'Generated acquisition bands', [{index, data: {images: {imageBands, imageMask: {data: mask, width: size, height: size}}}}]);
  }, {index: RASTER_TILE_INDEX, size: RASTER_TILE_SIZE});
  const polygonDatasetId = await publishTilePolygons(page, await getRasterTileExtent(page));
  const slices = Array.from({length: 24}, (_, slice) => ({time: new Date(Date.UTC(2020, 0, 1) + slice * 86400000).toISOString(), bands: [slice + 1]}));
  const panel = page.getByTestId('raster-timeseries-panel');
  await panel.waitFor({timeout: 30000});
  await panel.evaluate(element => {
    element.open = true;
  });
  await panel.getByTestId('timeseries-source').selectOption('timeseries-stack');
  await panel.getByTestId('timeseries-zones').selectOption(polygonDatasetId);
  await panel.getByTestId('timeseries-mapping').fill(JSON.stringify(slices));
  await panel.getByTestId('timeseries-component').fill('23');
  await panel.getByTestId('timeseries-run').click();
  await page.waitForFunction(() => document.querySelector('[data-testid="timeseries-status"]')?.textContent?.startsWith('Published'), undefined, {timeout: 90000});
  assert.match(await panel.getByTestId('timeseries-status').textContent(), /gpu; GPURasterRollingMean/);
  const actual = await page.evaluate(() => window.__keplerComputeTest.getRasterTimeSeriesResult().columns.rollingMean.values);
  const expected = (21 + 22 + 23) * 0.2 / 3 + 10 + 0.01 + (Math.sin(21 * Math.PI / 6) + Math.sin(22 * Math.PI / 6) + Math.sin(23 * Math.PI / 6)) / 3;
  assert.ok(Math.abs(actual[24 + 23] - expected) < 1e-4, 'rolling mean independent oracle for valid cell 1');
  assert.ok(Number.isNaN(actual[23]), 'nodata cell remains missing');

  // Use the same live host entry point as the panel for the complete contributor matrix.
  const operations = ['whittaker', 'hants', 'harmonic-regression', 'bfast', 'cold', 'medoid', 'geometric-median', 'best-pixel', 'phenology', 'streaming', 'climate-anomaly', 'standardized-calibration', 'standardized-index', 'resample', 'lagged-correlation', 'snapshot-covariance', 'eof', 'climate-runs'];
  for (const operation of operations) {
    const result = await page.evaluate(async ({operation, slices, polygonDatasetId}) => {
      const monthly = slices.map((slice, index) => ({...slice, time: new Date(Date.UTC(2020, 0, 1) + index * 32 * 86400000).toISOString()}));
      const settings = operation === 'lagged-correlation' ? {referenceIndex: Array.from({length: 24}, (_, index) => index)} : {};
      return window.__keplerComputeTest.runRasterTimeSeriesAction({rasterLayerId: 'timeseries-stack', slices: operation === 'climate-runs' ? slices : monthly, operation, settings, polygonDatasetId});
    }, {operation, slices, polygonDatasetId});
    assert.equal(result.engine, 'gpu', `${operation} uses the GPU`);
    assert.equal(result.published, true);
    assert.ok(result.readbackBytes > 0);
    assert.ok(result.contributors.length > 0);
    assert.ok(Object.keys(result.columns).length > 0);
    console.log(`${operation}: ${result.contributors.join(', ')}; ${result.rowCount} published rows`);
  }
  const beforeInvalid = (await page.evaluate(() => window.__keplerComputeTest.getState())).publications.length;
  const invalid = await page.evaluate(async ({slices, polygonDatasetId}) => {
    try {await window.__keplerComputeTest.runRasterTimeSeriesAction({rasterLayerId: 'timeseries-stack', slices: [slices[1], slices[0]], operation: 'rolling-mean', polygonDatasetId}); return '';} catch (error) {return error.message;}
  }, {slices, polygonDatasetId});
  assert.match(invalid, /strictly increasing/);
  assert.equal((await page.evaluate(() => window.__keplerComputeTest.getState())).publications.length, beforeInvalid);
  assert.deepEqual(failures, [], 'no page, WGSL or validation errors');
  console.log('PASS Raster time series: UI publication, all 19 GPU contributors, independent rolling oracle, nodata and invalid-date refusal');
} finally {
  await close();
}
