// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// NATIVEUI6-3: point labels with collision on a globe, for a source of at most 100,000 rows and one
// above it (GPU collision). Labels of points on the far hemisphere are hidden; labels on the near
// hemisphere are placed. Viewing longitude 0 with points only around longitude 180 must show no
// label ink; rotating the globe to longitude 180 must show ink. Headless WebGPU only.
// Usage: node globe-label-collision.mjs [url]
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const baseUrl = process.argv[2] || 'http://127.0.0.1:5318/?rows=256';
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
page.setDefaultTimeout(60000);
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.stack ?? error.message));
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())) errors.push(message.text());
});
const watchdog = setTimeout(() => {
  console.error('globe-label-collision exceeded 420 seconds');
  process.exit(1);
}, 420000);

/** Points scattered over a 60 by 50 degree window centered at longitude 180 (the far side of lon 0). */
function makeTable(rowCount) {
  let seed = 4242;
  const random = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
  const longitude = new Float32Array(rowCount);
  const latitude = new Float32Array(rowCount);
  const names = new Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    longitude[row] = 150 + random() * 60;
    if (longitude[row] > 180) longitude[row] -= 360;
    latitude[row] = -25 + random() * 50;
    names[row] = `N${row % 500}`;
  }
  return new arrow.Table({longitude: arrow.makeVector(longitude), latitude: arrow.makeVector(latitude), name: arrow.vectorFromArray(names, new arrow.Utf8())});
}

async function countInk() {
  const image = PNG.sync.read(await page.screenshot());
  let ink = 0;
  for (let y = 60; y < 960; y++) for (let x = 360; x < 1420; x++) {
    const offset = (y * image.width + x) * 4;
    if (image.data[offset] > 200 && image.data[offset + 1] < 90 && image.data[offset + 2] > 200) ink++;
  }
  return ink;
}

async function readyLabels(layerId) {
  await page.waitForFunction(id => {
    const api = window.__keplerComputeTest;
    if (api.getState().error) throw new Error(api.getState().error);
    return api.renderedLayers().some(layer => layer.id === id && layer.layerClass === 'GPUPointLabelLayer' && layer.loaded);
  }, layerId, {timeout: 120000});
  await page.waitForTimeout(3000);
}

const results = [];
try {
  for (const rowCount of (process.env.GLOBE_ROWS ?? '20000,120000').split(',').map(Number)) {
    await page.goto(baseUrl, {waitUntil: 'domcontentloaded'});
    await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
    await openNativeData(page);
    const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
    await workspace.locator('input[type=file]').setInputFiles({name: `globe-labels-${rowCount}.arrow`, mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(makeTable(rowCount), 'stream'))});
    await workspace.getByLabel('Source longitude column').selectOption('longitude');
    await workspace.getByLabel('Source latitude column').selectOption('latitude');
    await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
    await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
    await page.waitForFunction(count => window.__keplerComputeTest.getState().publications[0]?.selectedCount === count, rowCount, {timeout: 120000});
    await closeNativeDialog(page);
    const datasetId = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);
    const layerId = `globe-labels-${rowCount}`;
    const configuration = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
    const layer = configuration.config.visState.layers.find(value => value.config.dataId === datasetId);
    layer.id = layerId;
    layer.config.visConfig = {...layer.config.visConfig, radius: 6, filled: false, outline: false};
    layer.config.textLabel = [{field: {name: 'name', type: 'string'}, color: [255, 0, 255], size: 22, weight: 700, anchor: 'middle', alignment: 'center', outlineWidth: 0, background: false, collisionEnabled: true}];
    configuration.config.visState.layers = [layer];
    const globeState = lon => ({...configuration.config.mapState, mapViewMode: 'MODE_GLOBE', globe: {...(configuration.config.mapState.globe ?? {}), enabled: true}, longitude: lon, latitude: 0, zoom: 1.6, pitch: 0, bearing: 0});
    configuration.config.mapState = globeState(0);
    await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), configuration);
    await readyLabels(layerId);
    const farInk = await countInk();
    configuration.config.mapState = globeState(180);
    await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), configuration);
    await readyLabels(layerId);
    const trace = [];
    for (let attempt = 0; attempt < 3; attempt++) {
      trace.push(await countInk());
      await page.waitForTimeout(3000);
    }
    if (process.env.GLOBE_SHOT) await page.screenshot({path: `${process.env.GLOBE_SHOT}-${rowCount}.png`});
    console.log('near ink trace', JSON.stringify(trace));
    const nearInk = trace.at(-1);
    results.push({rowCount, farInk, nearInk});
    console.log(JSON.stringify(results.at(-1)));
    assert(nearInk > 400, `${rowCount}: near-side labels must be placed (${nearInk} ink pixels)`);
    assert.equal(farInk, 0, `${rowCount}: far-side labels must be hidden (${farInk} ink pixels)`);
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, results}));
} catch (error) {
  console.error(error);
  console.error(JSON.stringify(results), errors);
  process.exitCode = 1;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
