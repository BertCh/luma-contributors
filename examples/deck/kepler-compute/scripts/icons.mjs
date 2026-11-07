// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const source = arrow.tableFromArrays({
  longitude: new Float32Array([-74.03, -74.01]),
  latitude: new Float32Array([40.72, 40.74]),
  symbol: [null, 'diamond'],
  measure: new Float32Array([0, 1])
}).concat(arrow.tableFromArrays({
  longitude: new Float32Array([-73.98]),
  latitude: new Float32Array([40.77]),
  symbol: ['triangle'],
  measure: new Float32Array([2])
}));
const svgIcons = [
  {id: 'diamond', mesh: {positions: [[0,-1,0],[1,0,0],[0,1,0],[-1,0,0]], cells: [[0,1,2],[0,2,3]]}},
  {id: 'triangle', mesh: {positions: [[0,-1,0],[1,1,0],[-1,1,0]], cells: [[0,1,2]]}}
];
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
await page.route('**/icons/svg-icons.json', route => route.fulfill({json: {svgIcons}}));
const errors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())) errors.push(message.text());
});
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => {
    const overlay = document.querySelector('vite-error-overlay');
    if (overlay) throw new Error(overlay.shadowRoot?.textContent ?? 'Vite import failed');
    return window.__keplerComputeTest?.getState().selectedCount === 256;
  }, undefined, {timeout: 60000});
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: 'native-icons.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))});
  await workspace.getByLabel('Source longitude column').selectOption('longitude');
  await workspace.getByLabel('Source latitude column').selectOption('latitude');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 3);
  await closeNativeDialog(page);
  const initial = await page.evaluate(() => window.__keplerComputeTest.getState());
  const datasetId = initial.publications[0].id;
  assert.deepEqual(initial.publications[0].sourceBatches, [2, 1]);
  const configuration = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const nativeIcon = configuration.config.visState.layers.find(layer => layer.config.dataId === datasetId);
  for (const layer of configuration.config.visState.layers) layer.config.isVisible = layer === nativeIcon;
  nativeIcon.type = 'icon';
  // Native Icon has no alternate column modes; do not retain Point's 'points' mode.
  nativeIcon.config.columnMode = null;
  nativeIcon.config.columns = {lat: 'latitude', lng: 'longitude', icon: 'symbol'};
  nativeIcon.config.textLabel = [];
  nativeIcon.config.color = [245, 110, 40];
  nativeIcon.config.visConfig = {...nativeIcon.config.visConfig, radius: 200, fixedRadius: false, billboard: false};
  nativeIcon.visualChannels = {...nativeIcon.visualChannels, colorField: null, sizeField: null};
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), configuration);
  await page.waitForFunction(id => {
    const api = window.__keplerComputeTest;
    const rendered = api.renderedLayers().filter(layer => layer.id === id && layer.visible && layer.loaded);
    return rendered.length === 2;
  }, nativeIcon.id, {timeout: 45000});
  const iconBuffers = () => page.evaluate(id => window.__keplerComputeTest.renderedLayers()
    .filter(layer => layer.id === id && layer.iconGeometryBufferId)
    .map(layer => ({id: layer.renderId, mesh: layer.iconGeometryBufferId,
      positions: layer.positionBufferIds, masks: layer.maskBufferIds})), nativeIcon.id);
  const initialIconBuffers = await iconBuffers();
  assert.equal(initialIconBuffers.length, 2);
  const image = PNG.sync.read(await page.screenshot());
  let coloredPixels = 0;
  for (let vertical = 100; vertical < 850; vertical++) for (let horizontal = 360; horizontal < 1040; horizontal++) {
    const offset = (vertical * image.width + horizontal) * 4;
    if (image.data[offset] > 180 && image.data[offset + 1] < 160 && image.data[offset + 2] < 100) coloredPixels++;
  }
  assert(coloredPixels > 100, `Native SVG meshes should draw visible pixels; found ${coloredPixels}`);
  const pick = (longitude, latitude) => page.evaluate(({datasetId, longitude, latitude}) => window.__keplerComputeTest.pickPublishedPosition(datasetId, longitude, latitude), {datasetId, longitude, latitude});
  assert.equal((await pick(-74.01, 40.74))?.sourceRowIndex, 1);
  assert.equal((await pick(-73.98, 40.77))?.sourceRowIndex, 2);
  assert.equal(await pick(-74.03, 40.72), null);
  await page.evaluate(datasetId => window.__keplerComputeTest.setPublishedRange(datasetId, 'measure', 2, 2), datasetId);
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0].selectedCount === 1);
  assert.equal(await pick(-74.01, 40.74), null);
  assert.equal((await pick(-73.98, 40.77))?.sourceRowIndex, 2);
  const final = await page.evaluate(() => window.__keplerComputeTest.getState());
  assert.equal(final.uploadBytes, initial.uploadBytes);
  assert.deepEqual(await iconBuffers(), initialIconBuffers);
  assert.deepEqual(final.publications[0].sourceBufferIds, initial.publications[0].sourceBufferIds);
  assert.equal(final.publications[0].sourceUploadBytes, initial.publications[0].sourceUploadBytes);
  assert.equal(final.error, '');
  assert.deepEqual(errors, []);
  await page.screenshot({path: '/tmp/luma-kepler-icons.png'});
  console.log(JSON.stringify({passed: true, coloredPixels, sourceBatches: [2,1], originalRows: [1,2], checks: ['native SVG catalog', 'same-ID point to icon', 'two GPU meshes', 'source picking', 'scalar masks', 'stable source upload']}));
} catch (error) {
  console.error({pageErrors: errors});
  console.error(JSON.stringify(await page.evaluate(() => ({layers: window.__keplerComputeTest?.renderedLayers(), configuration: window.__keplerComputeTest?.saveNativeConfig()}))));
  console.error(await page.evaluate(() => window.__keplerComputeTest?.getState()));
  console.error(await page.locator('body').innerText());
  throw error;
} finally {
  await browser.close();
}
