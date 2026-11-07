// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// LAYERS5-5: an icon layer with a sizeField and a text label places each label beside its own icon.
// The label offset scales per row with the row radius, so the gap between each icon's right edge
// and its label stays constant while the icons themselves differ in size. Headless WebGPU only.
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const rows = [
  {longitude: -74.1, latitude: 40.75, size: 1},
  {longitude: -73.98, latitude: 40.75, size: 3},
  {longitude: -73.86, latitude: 40.75, size: 8}
];
const table = new arrow.Table({
  east: arrow.vectorFromArray(rows.map(row => Math.fround(row.longitude)), new arrow.Float32()),
  north: arrow.vectorFromArray(rows.map(row => Math.fround(row.latitude)), new arrow.Float32()),
  size: arrow.vectorFromArray(rows.map(row => row.size), new arrow.Float32()),
  symbol: arrow.vectorFromArray(rows.map(() => 'diamond')),
  name: arrow.vectorFromArray(rows.map((_, index) => `row ${index}`))
});
const svgIcons = [{id: 'diamond', mesh: {positions: [[0, -1, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]], cells: [[0, 1, 2], [0, 2, 3]]}}];
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
page.setDefaultTimeout(30000);
await pinViteModules(page);
await page.route('**/icons/svg-icons.json', route => route.fulfill({json: {svgIcons}}));
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: 'icon-label-size.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(table, 'stream'))});
  await workspace.getByLabel('Source longitude column').selectOption('east');
  await workspace.getByLabel('Source latitude column').selectOption('north');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 3);
  await closeNativeDialog(page);
  const datasetId = (await page.evaluate(() => window.__keplerComputeTest.getState())).publications[0].id;
  const configuration = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const layer = configuration.config.visState.layers.find(candidate => candidate.config.dataId === datasetId);
  for (const candidate of configuration.config.visState.layers) candidate.config.isVisible = candidate === layer;
  layer.type = 'icon';
  layer.config.columnMode = null;
  layer.config.columns = {lat: 'north', lng: 'east', icon: 'symbol'};
  layer.config.sizeField = {name: 'size', type: 'real'};
  layer.visualChannels = {...layer.visualChannels, sizeField: layer.config.sizeField, sizeScale: 'linear'};
  layer.config.textLabel = [{field: {name: 'name', type: 'string'}, size: 20, color: [255, 0, 255], offset: [0, 0], anchor: 'start', alignment: 'center', outlineWidth: 0, outlineColor: [255, 0, 0, 255], background: false, backgroundColor: [0, 0, 200, 255], weight: 700}];
  layer.config.color = [245, 110, 40];
  layer.config.visConfig = {...layer.config.visConfig, radius: 10, radiusRange: [50, 150], fixedRadius: false, billboard: false, opacity: 1};
  configuration.config.mapState = {...configuration.config.mapState, longitude: -73.98, latitude: 40.75, zoom: 10.5, pitch: 0, bearing: 0};
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), configuration);
  await page.waitForFunction(id => window.__keplerComputeTest.renderedLayers().filter(value => value.id === id && value.visible && value.loaded).length >= 2, layer.id, {timeout: 45000});
  await page.waitForTimeout(2500);
  const image = PNG.sync.read(await page.screenshot());
  const projected = [];
  for (const row of rows) projected.push(await page.evaluate(([longitude, latitude]) => window.__keplerComputeTest.projectPosition(longitude, latitude), [row.longitude, row.latitude]));
  const isIcon = (x, y) => {const o = (y * image.width + x) * 4; return image.data[o] > 200 && image.data[o + 1] > 80 && image.data[o + 1] < 150 && image.data[o + 2] < 90;};
  const isLabel = (x, y) => {const o = (y * image.width + x) * 4; return image.data[o] > 200 && image.data[o + 1] < 80 && image.data[o + 2] > 200;};
  const measures = projected.map((point, index) => {
    const centerY = Math.round(point[1]);
    const centerX = Math.round(point[0]);
    const windowEnd = Math.min(image.width - 1, centerX + 100);
    let iconRight = -Infinity;
    let iconLeft = Infinity;
    let labelLeft = Infinity;
    for (let y = centerY - 40; y <= centerY + 40; y++) for (let x = Math.max(0, centerX - 100); x <= windowEnd; x++) {
      if (isIcon(x, y)) {iconRight = Math.max(iconRight, x); iconLeft = Math.min(iconLeft, x);}
    }
    for (let y = centerY - 40; y <= centerY + 40; y++) for (let x = iconRight + 1; x <= Math.min(image.width - 1, centerX + 230); x++) if (isLabel(x, y)) labelLeft = Math.min(labelLeft, x);
    return {index, centerX, iconRadius: (iconRight - iconLeft) / 2, iconRight, labelLeft, gap: labelLeft - iconRight};
  });
  console.log(JSON.stringify(measures));
  assert(measures.every(measure => Number.isFinite(measure.gap)), 'every row has an icon and a label');
  assert(measures[2].iconRadius > measures[0].iconRadius * 1.5, 'size channel scales the icons');
  const gaps = measures.map(measure => measure.gap);
  // Without per-row scaling the big icon would overlap its label (gap shrinks by the radius difference).
  assert(Math.max(...gaps) - Math.min(...gaps) <= 6, `label gap must stay constant across icon sizes (${gaps})`);
  assert.equal(await page.evaluate(() => window.__keplerComputeTest.getState().error), '');
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, gaps, radii: measures.map(measure => measure.iconRadius)}));
} catch (error) {
  console.error(error);
  process.exitCode = 1;
} finally {
  await browser.close();
}
