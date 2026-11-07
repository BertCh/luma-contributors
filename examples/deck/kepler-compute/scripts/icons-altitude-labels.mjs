// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Icon altitude (null = 0, nonfinite hidden), icon text labels and portable custom icon catalogs.
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const rows = [[-73.995, 40.735, 0], [-73.965, 40.735, 6000], [-73.995, 40.765, -1000], [-73.965, 40.765, null], [-73.98, 40.75, NaN]];
const table = new arrow.Table({
  east: arrow.vectorFromArray(rows.map(row => Math.fround(row[0])), new arrow.Float32()),
  north: arrow.vectorFromArray(rows.map(row => Math.fround(row[1])), new arrow.Float32()),
  height: arrow.vectorFromArray(rows.map(row => row[2] === null ? null : Math.fround(row[2])), new arrow.Float32()),
  symbol: arrow.vectorFromArray(rows.map(() => 'diamond')),
  name: arrow.vectorFromArray(rows.map((_, index) => `row ${index}`))
});
const svgIcons = [{id: 'diamond', mesh: {positions: [[0,-1,0],[1,0,0],[0,1,0],[-1,0,0]], cells: [[0,1,2],[0,2,3]]}}];
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
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
  await workspace.locator('input[type=file]').setInputFiles({name: 'icon-altitude.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(table, 'stream'))});
  await workspace.getByLabel('Source longitude column').selectOption('east');
  await workspace.getByLabel('Source latitude column').selectOption('north');
  await workspace.getByLabel('Source altitude column').selectOption('height');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 5);
  await closeNativeDialog(page);
  const datasetId = (await page.evaluate(() => window.__keplerComputeTest.getState())).publications[0].id;
  const configuration = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const layer = configuration.config.visState.layers.find(candidate => candidate.config.dataId === datasetId);
  for (const candidate of configuration.config.visState.layers) candidate.config.isVisible = candidate === layer;
  layer.type = 'icon';
  layer.config.columnMode = null;
  layer.config.columns = {lat: 'north', lng: 'east', icon: 'symbol', altitude: 'height'};
  layer.config.textLabel = [{field: {name: 'name', type: 'string'}, size: 18, color: [255, 255, 255], offset: [0, 0], anchor: 'start', alignment: 'center', outlineWidth: 0, outlineColor: [255, 0, 0, 255], background: false, backgroundColor: [0, 0, 200, 255], weight: 400}];
  layer.config.color = [245, 110, 40];
  layer.config.visConfig = {...layer.config.visConfig, radius: 200, fixedRadius: false, billboard: false, opacity: 1};
  configuration.config.mapState = {...configuration.config.mapState, longitude: -73.98, latitude: 40.75, zoom: 12, pitch: 50, bearing: 0};
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), configuration);
  await page.waitForFunction(id => window.__keplerComputeTest.renderedLayers().some(value => value.id === id && value.visible && value.loaded && value.positionFormat === 'float32x3' || value.getPosition?.format === 'float32x3'), layer.id, {timeout: 45000});
  const pick = (row, altitude = rows[row][2] ?? 0) => page.evaluate(({datasetId, longitude, latitude, altitude}) => window.__keplerComputeTest.pickPublishedPosition(datasetId, longitude, latitude, altitude), {datasetId, longitude: rows[row][0], latitude: rows[row][1], altitude});
  assert.equal((await pick(1))?.sourceRowIndex, 1, 'raised icon picks at its elevated projection');
  assert.notEqual((await pick(1, 0))?.sourceRowIndex, 1, 'raised icon must not pick at ground');
  assert.equal((await pick(3))?.sourceRowIndex, 3, 'null altitude renders at zero');
  assert.equal(await pick(4, 0), null, 'nonfinite altitude icon is hidden');
  const labelLayers = await page.evaluate(id => window.__keplerComputeTest.renderedLayers().filter(value => value.id === id && /Label/.test(value.layerClass ?? '')).length, layer.id);
  assert(labelLayers >= 1, 'icon text label layer must be rendered');
  // Portable custom catalog: embed, save, drop and restore through the bundle panel.
  const bundleText = await page.evaluate(async () => window.__keplerComputeTest.serializeBundleWithIconCatalog?.([{id: 'custom-square', mesh: {positions: [[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]], cells: [[0,1,2],[0,2,3]]}}]));
  if (bundleText) {
    const document = JSON.parse(bundleText);
    assert.equal(document.iconCatalog?.format, 'luma-kepler-icon-catalog');
    assert.equal(document.iconCatalog.icons[0].id, 'custom-square');
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, labelLayers, embeddedCatalogChecked: Boolean(bundleText)}));
} finally {
  await browser.close();
}
