// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {openNativeData, openNativeExport} from './native-navigation.mjs';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const origin = Date.UTC(2026, 0, 1);
const path = latitude => JSON.stringify({type: 'LineString', coordinates: [
  [-74.03, latitude, 0, origin], [-74, latitude, 0, origin + 10000], [-73.97, latitude, 0, origin + 20000]
]});
const source = arrow.tableFromArrays({route: [null, path(40.74)], measure: new Float32Array([0, 1])})
  .concat(arrow.tableFromArrays({route: [path(40.77)], measure: new Float32Array([2])}));
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
async function coloredMapPixels() {
  // Published trips fit the map to their bounds, so the northern route sits on the top edge of
  // the map; sample a band around its projected position instead of a fixed central region.
  const [, routeVertical] = await page.evaluate(() => window.__keplerComputeTest.projectPosition(-74, 40.77));
  const image = PNG.sync.read(await page.screenshot());
  let pixels = 0;
  const top = Math.max(0, Math.floor(routeVertical - 30));
  const bottom = Math.min(image.height, Math.ceil(routeVertical + 30));
  for (let vertical = top; vertical < bottom; vertical++) for (let horizontal = 340; horizontal < 1380; horizontal++) {
    const offset = (vertical * image.width + horizontal) * 4;
    const values = image.data.subarray(offset, offset + 3);
    if (Math.max(...values) > 100 && Math.max(...values) - Math.min(...values) > 30) pixels++;
  }
  return pixels;
}
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: 'timed-paths.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))});
  await workspace.getByLabel('Source geometry column').selectOption('route');
  await workspace.getByLabel('Timed paths (Trips)', {exact: true}).check();
  await workspace.getByLabel('Trip timestamp encoding', {exact: true}).selectOption('unix-millisecond');
  await workspace.getByLabel('Trip time origin (epoch milliseconds)', {exact: true}).fill(String(origin));
  await workspace.getByLabel('Maximum trip time error (milliseconds)', {exact: true}).fill('0.1');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByText(/2 timed paths, 6 vertices/).waitFor();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.trips?.tripCount === 2, undefined, {timeout: 45000});
  const initial = await page.evaluate(() => window.__keplerComputeTest.getState());
  const datasetId = initial.publications[0].id;
  assert.deepEqual(initial.publications[0].sourceBatches, [2, 1]);
  assert.deepEqual(initial.publications[0].trips.domain, [origin, origin + 20000]);
  assert.equal(initial.publications[0].error, null);
  const nativeConfig = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  for (const layer of nativeConfig.config.visState.layers) {
    layer.config.isVisible = layer.config.dataId === datasetId;
    if (layer.type === 'trip') layer.config.visConfig.thickness = 8;
  }
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), nativeConfig);
  await page.waitForFunction(datasetId => {
    const api = window.__keplerComputeTest;
    const state = api.getState();
    if (state.error) throw new Error(state.error);
    const native = state.nativeLayers.find(layer => layer.id === `${datasetId}-trips`);
    if (native?.valid === false) throw new Error(JSON.stringify(native));
    return api.renderedLayers().some(layer => layer.id === `${datasetId}-trips` && layer.visible && layer.loaded);
  }, datasetId);
  assert.deepEqual(await page.evaluate(() => window.__keplerComputeTest.getState().animation.domain), [origin, origin + 20000]);
  await page.evaluate(time => window.__keplerComputeTest.setAnimationTime(time), origin);
  const earlyPixels = await coloredMapPixels();
  await page.evaluate(time => window.__keplerComputeTest.setAnimationTime(time), origin + 15000);
  await page.waitForFunction(time => window.__keplerComputeTest.getState().animation.currentTime === time, origin + 15000);
  const latePixels = await coloredMapPixels();
  assert(latePixels > earlyPixels + 100, `Animation must reveal timed segments (${earlyPixels} -> ${latePixels})`);
  const picked = await page.evaluate(datasetId => window.__keplerComputeTest.pickTripPosition(datasetId, -74.01, 40.74), datasetId);
  assert.equal(picked?.sourceRowIndex, 1, 'Trip picking preserves the null-prefixed source row');
  await page.evaluate(time => window.__keplerComputeTest.setAnimationTime(time), origin + 1000);
  const playback = page.locator('.animation-control-container .playback-control-button:visible').first();
  await playback.click();
  await page.waitForFunction(time => window.__keplerComputeTest.getState().animation.currentTime > time, origin + 1100);
  await playback.click();
  await page.evaluate(time => window.__keplerComputeTest.setAnimationTime(time), origin + 15000);
  await page.evaluate(datasetId => window.__keplerComputeTest.setPublishedRange(datasetId, 'measure', 2, 2), datasetId);
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0].selectedCount === 1);
  assert.equal(await page.evaluate(datasetId => window.__keplerComputeTest.pickTripPosition(datasetId, -74.01, 40.74), datasetId), null);
  assert.equal((await page.evaluate(datasetId => window.__keplerComputeTest.pickTripPosition(datasetId, -74.01, 40.77), datasetId))?.sourceRowIndex, 2);
  const final = await page.evaluate(() => window.__keplerComputeTest.getState());
  assert.equal(final.error, '');
  assert.equal(final.publications[0].trips.sourceUploadBytes, initial.publications[0].trips.sourceUploadBytes);
  assert.deepEqual(final.publications[0].trips.sourceBufferIds, initial.publications[0].trips.sourceBufferIds);
  assert.equal(final.uploadBytes, initial.uploadBytes);
  await openNativeExport(page, true);
  const sandbox = page.getByRole('region', {name: 'Save map', exact: true});
  await sandbox.getByRole('button', {name: 'Save state', exact: true}).click();
  const saved = JSON.parse(await sandbox.getByLabel('Saved sandbox state', {exact: true}).inputValue());
  assert.deepEqual(saved.sources.find(source => source.dataset.datasetId === datasetId).reference.parameters.tripOptions,
    {timestampEncoding: 'unix-millisecond', originMilliseconds: origin, maximumTimeErrorMilliseconds: 0.1});
  assert.deepEqual(errors, []);
  await page.screenshot({path: '/tmp/luma-kepler-trips.png'});
  console.log(JSON.stringify({passed: true, sourceBatches: [2, 1], originalRows: [1, 2], earlyPixels, latePixels,
    checks: ['explicit time precision', 'native animation domain', 'native playback', 'timed GPU trail', 'source-aligned picking', 'scalar selection', 'stable source allocations', 'saved trip policy']}));
} catch (error) {
  console.error(await page.evaluate(() => window.__keplerComputeTest?.getState()));
  console.error(await page.locator('body').innerText());
  throw error;
} finally {
  await browser.close();
}
