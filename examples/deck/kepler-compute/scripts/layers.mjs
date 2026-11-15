// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {pinViteModules, openKeplerComputePage, discardStaleImportErrors} from './pin-vite-modules.mjs';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const failures = [];
const diagnostics = [];
page.on('pageerror', error => failures.push(error.message));
page.on('console', message => {
  if (message.type() === 'error') diagnostics.push(message.text());
  if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())) failures.push(message.text());
});
const results = [];
try {
  const failuresBeforeBoot = failures.length;
  await openKeplerComputePage(page, process.argv[2] || 'http://127.0.0.1:5318/?rows=4096', {
    timeout: 30000,
    // Discard only identified stale imports; genuine failures from every attempt survive.
    onAbandonedAttempt: ({staleImportErrors}) => {
      discardStaleImportErrors(failures, failuresBeforeBoot, staleImportErrors);
    }
  });
  const original = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const sourceCount = await page.evaluate(() => window.__keplerComputeTest.source().length);
  const filteredCount = await page.evaluate(() => window.__keplerComputeTest.source().filter(row => row[2] >= 40 && row[2] <= 75).length);
  for (const type of ['line', 'arc', 'grid', 'hexagon']) {
    const configuration = structuredClone(original);
    const layer = configuration.config.visState.layers[0];
    layer.id = `resident-test-${type}`;
    layer.type = type;
    layer.config.columnMode = 'points';
    layer.config.columns = type === 'line' || type === 'arc'
      ? {lat0: 'latitude', lng0: 'longitude', lat1: 'targetLatitude', lng1: 'targetLongitude'}
      : {lat: 'latitude', lng: 'longitude'};
    layer.config.color = [86, 203, 249];
    layer.config.colorField = null;
    layer.config.sizeField = null;
    layer.config.colorScale = 'quantize';
    layer.config.textLabel = [];
    layer.config.visConfig = {...layer.config.visConfig, thickness: 2, targetColor: [86, 203, 249], targetArrowEnabled: false,
      worldUnitSize: 0.3, coverage: 0.9, enable3d: false, percentile: [0, 100], colorRange: {colors: ['#21918c', '#fde725']}};
    const frameCount = await page.evaluate(() => window.__keplerComputeTest.getState().frameCount);
    await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), configuration);
    await page.waitForFunction(({id, count, frame}) => {
      const api = window.__keplerComputeTest;
      const state = api.getState();
      if (state.error) throw new Error(state.error);
      const invalidLayer = state.nativeLayers?.find(layer => layer.valid === false);
      if (invalidLayer) throw new Error(JSON.stringify(invalidLayer));
      return state.selectedCount === count && state.frameCount > frame && api.renderedLayers().some(layer => layer.id === id && layer.visible && layer.loaded);
    }, {id: layer.id, count: sourceCount, frame: frameCount}, {timeout: 30000});
    const screenshot = PNG.sync.read(await page.screenshot());
    let coloredPixels = 0;
    for (let y = 100; y < 850; y++) for (let x = 400; x < 1000; x++) {
      const offset = (y * screenshot.width + x) * 4;
      const channels = screenshot.data.subarray(offset, offset + 3);
      if (Math.max(...channels) > 100 && Math.max(...channels) - Math.min(...channels) > 30) coloredPixels++;
    }
    assert(coloredPixels > 500, `${type} must draw visible map pixels; found ${coloredPixels}`);
    await page.evaluate(() => window.__keplerComputeTest.setRange(40, 75));
    await page.waitForFunction(count => window.__keplerComputeTest.getState().selectedCount === count, filteredCount);
    results.push({type, coloredPixels, selectedCount: filteredCount});
  }
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), original);
  await page.waitForFunction(count => window.__keplerComputeTest.getState().selectedCount === count, sourceCount);
  await page.evaluate(() => window.__keplerComputeTest.addPolygon([[[-74.05,40.69],[-73.99,40.69],[-73.99,40.82],[-74.05,40.82],[-74.05,40.69]]]));
  await page.waitForFunction(count => window.__keplerComputeTest.getState().selectedCount < count, sourceCount);
  const spatial = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  spatial.config.visState.layers[0].type = 'line';
  spatial.config.visState.layers[0].config.columns = {lat0: 'latitude', lng0: 'longitude', lat1: 'targetLatitude', lng1: 'targetLongitude'};
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), spatial);
  const endpointCount = await page.evaluate(() => window.__keplerComputeTest.source().filter(row =>
    row[0] > -74.05 && row[0] < -73.99 && row[1] > 40.69 && row[1] < 40.82 &&
    row[5] > -74.05 && row[5] < -73.99 && row[6] > 40.69 && row[6] < 40.82).length);
  await page.waitForFunction(count => {
    const api = window.__keplerComputeTest;
    const state = api.getState();
    if (state.error) throw new Error(state.error);
      const invalidLayer = state.nativeLayers?.find(layer => layer.valid === false);
      if (invalidLayer) throw new Error(JSON.stringify(invalidLayer));
    return state.selectedCount === count && !state.unsupportedSelection && api.renderedLayers().some(layer => layer.id === 'resident-points' && layer.visible && layer.loaded);
  }, endpointCount);
  assert(await page.evaluate(() => window.__keplerComputeTest.renderedLayers().some(layer => layer.id === 'resident-points' && layer.visible && layer.loaded)));
  const endpointImage = PNG.sync.read(await page.screenshot());
  let endpointPixels = 0;
  for (let vertical = 100; vertical < 850; vertical++) for (let horizontal = 400; horizontal < 1000; horizontal++) {
    const offset = (vertical * endpointImage.width + horizontal) * 4;
    const channels = endpointImage.data.subarray(offset, offset + 3);
    if (Math.max(...channels) > 100 && Math.max(...channels) - Math.min(...channels) > 30) endpointPixels++;
  }
  assert(endpointPixels > 500, `Filtered line must draw visible map pixels; found ${endpointPixels}`);
  assert.deepEqual(failures, []);
  process.stdout.write(`${JSON.stringify({passed: true, results, endpointCount, endpointPixels, checks: ['native layer restore', 'visible GPU pixels', 'native filter updates', 'line endpoint polygon scope']}, null, 2)}\n`);
} catch (error) {
  process.stderr.write(`${JSON.stringify({diagnostics})}\n`);
  process.stderr.write(`${JSON.stringify(await page.evaluate(() => ({state: window.__keplerComputeTest?.getState(), renderedLayers: window.__keplerComputeTest?.renderedLayers(), configuration: window.__keplerComputeTest?.saveNativeConfig()})), null, 2)}\n`);
  throw error;
} finally {
  await browser.close();
}
