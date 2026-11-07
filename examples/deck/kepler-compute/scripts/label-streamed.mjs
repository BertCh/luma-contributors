// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Headless check of streamed GPU label collision above the 100,000 row glyph cap (NATIVEUI6-1).
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {PNG} from 'pngjs';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const ROW_COUNT = 150001;
let seed = 4242;
const random = () => {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  return seed / 4294967296;
};
const longitudes = Float32Array.from({length: ROW_COUNT}, () => -74.05 + random() * 0.2);
const latitudes = Float32Array.from({length: ROW_COUNT}, () => 40.65 + random() * 0.2);
const names = Array.from({length: ROW_COUNT}, (_value, index) => `L${index}`);
const source = new arrow.Table({
  longitude: arrow.makeVector(longitudes),
  latitude: arrow.makeVector(latitudes),
  value: arrow.makeVector(Float32Array.from({length: ROW_COUNT}, (_value, index) => index)),
  name: arrow.vectorFromArray(names, new arrow.Utf8())
});
const bytes = arrow.tableToIPC(source, 'stream');
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
page.setDefaultTimeout(60000);
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.stack ?? error.message));
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|validation error|Invalid.*(?:Buffer|Pipeline)/i.test(message.text()))
    errors.push(message.text());
});
const watchdog = setTimeout(async () => {
  console.error('Streamed label browser exceeded five minutes');
  await Promise.race([browser.close(), new Promise(resolve => setTimeout(resolve, 3000))]);
  process.exit(1);
}, 300000);
const layerId = 'native-streamed-labels';
async function countInk(name) {
  const image = PNG.sync.read(await page.screenshot({path: `/tmp/luma-kepler-${name}.png`}));
  let count = 0;
  for (let y = 100; y < image.height - 100; y++)
    for (let x = 350; x < image.width - 80; x++) {
      const offset = (y * image.width + x) * 4;
      if (image.data[offset] > 170 && image.data[offset + 1] > 170 && image.data[offset + 2] > 170) count++;
    }
  return count;
}
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page, 'Prepare data');
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: 'streamed-labels.arrow',
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(bytes)
  });
  await workspace.getByLabel('Source longitude column').selectOption('longitude');
  await workspace.getByLabel('Source latitude column').selectOption('latitude');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(count => window.__keplerComputeTest.getState().publications[0]?.selectedCount === count, ROW_COUNT);
  await closeNativeDialog(page);
  const datasetId = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);
  const base = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const apply = async (collisionEnabled, zoom) => {
    const config = structuredClone(base);
    const layer = config.config.visState.layers.find(value => value.config.dataId === datasetId);
    layer.id = layerId;
    layer.config.visConfig = {...layer.config.visConfig, radius: 2, filled: true, outline: false};
    layer.config.textLabel = [
      {
        field: {name: 'name', type: 'string'},
        color: [255, 255, 255],
        size: 14,
        weight: 400,
        anchor: 'start',
        alignment: 'center',
        outlineWidth: 0,
        background: false,
        collisionEnabled
      }
    ];
    config.config.visState.layers = [layer];
    config.config.mapState = {...config.config.mapState, longitude: -73.95, latitude: 40.75, zoom, pitch: 0, bearing: 0};
    await page.evaluate(configuration => window.__keplerComputeTest.applyNativeConfig(configuration), config);
  };
  const ready = () =>
    page.waitForFunction(
      identifier => {
        const api = window.__keplerComputeTest;
        if (api.getState().error) throw new Error(api.getState().error);
        return api.renderedLayers().some(layer => layer.id === identifier && layer.layerClass === 'GPUPointLabelLayer' && layer.loaded);
      },
      layerId,
      {timeout: 120000}
    );
  await apply(true, 11);
  await ready();
  // Placement is asynchronous (collision readback); give it a few frames then measure.
  await page.waitForTimeout(1500);
  const wide = await countInk('streamed-labels-wide');
  assert(wide > 500, `Streamed GPU collision must draw visible labels above 100k rows (${wide} ink pixels)`);
  await apply(true, 14);
  await ready();
  await page.waitForTimeout(1500);
  const zoomed = await countInk('streamed-labels-zoomed');
  assert(zoomed > 500, `Streamed labels must re-place for a new viewport (${zoomed} ink pixels)`);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, rows: ROW_COUNT, wideInk: wide, zoomedInk: zoomed}, null, 2));
} catch (error) {
  console.error(JSON.stringify(await page.evaluate(() => ({state: window.__keplerComputeTest?.getState(), layers: window.__keplerComputeTest?.renderedLayers()})).catch(() => null), null, 2));
  console.error(errors);
  throw error;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
