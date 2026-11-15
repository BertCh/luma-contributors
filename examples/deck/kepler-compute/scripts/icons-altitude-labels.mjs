// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Icon altitude (null = 0, nonfinite hidden), icon text labels and portable custom icon catalogs.
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules, openKeplerComputePage, discardStaleImportErrors} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog, openNativeExport} from './native-navigation.mjs';
import {ICON_ALTITUDE_ROWS, CUSTOM_ICON_DEFINITIONS, makeIconAltitudeTable} from './icons-altitude-labels-fixture.mjs';

const rows = ICON_ALTITUDE_ROWS;
const table = makeIconAltitudeTable();
const svgIcons = CUSTOM_ICON_DEFINITIONS;
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
let page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
page.setDefaultTimeout(30000);
await pinViteModules(page);
let defaultIcons = svgIcons;
await page.route('**/icons/svg-icons.json', route => route.fulfill({json: {svgIcons: defaultIcons}}));
const errors = [];
page.on('pageerror', error => errors.push(error.message));
async function openPage(address) {
  const errorsBeforeBoot = errors.length;
  await openKeplerComputePage(page, address, {
    // The second boot must preserve genuine failures from the preceding icon/bundle checks.
    onAbandonedAttempt: ({staleImportErrors}) => {
      discardStaleImportErrors(errors, errorsBeforeBoot, staleImportErrors);
    }
  });
}
try {
  const address = process.argv[2] || 'http://127.0.0.1:5318/?rows=256';
  await openPage(address);
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
  await page.waitForFunction(id => window.__keplerComputeTest.renderedLayers().some(value => value.id === id && value.visible && value.loaded && (value.positionFormat === 'float32x3' || value.getPosition?.format === 'float32x3')), layer.id, {timeout: 45000});
  const pick = (row, altitude = rows[row][2] ?? 0) => page.evaluate(({datasetId, longitude, latitude, altitude}) => window.__keplerComputeTest.pickPublishedPosition(datasetId, longitude, latitude, altitude), {datasetId, longitude: rows[row][0], latitude: rows[row][1], altitude});
  assert.equal((await pick(1))?.sourceRowIndex, 1, 'raised icon picks at its elevated projection');
  assert.notEqual((await pick(1, 0))?.sourceRowIndex, 1, 'raised icon must not pick at ground');
  assert.equal((await pick(3))?.sourceRowIndex, 3, 'null altitude renders at zero');
  assert.equal(await pick(4, 0), null, 'nonfinite altitude icon is hidden');
  const labelLayers = await page.evaluate(id => window.__keplerComputeTest.renderedLayers().filter(value => value.id === id && /Label/.test(value.layerClass ?? '')).length, layer.id);
  assert(labelLayers >= 1, 'icon text label layer must be rendered');
  // The old script optionally called a nonexistent serializeBundleWithIconCatalog hook,
  // so it never exercised portability either. Use the actual bundle UI and require a
  // custom catalog to survive save, a clean-page restore, rendering and another save.
  async function saveBundle() {
    await openNativeExport(page, true);
    const pending = page.waitForEvent('download');
    await page.getByRole('button', {name: 'Save portable bundle', exact: true}).click();
    const chunks = [];
    for await (const chunk of await (await pending).createReadStream()) chunks.push(chunk);
    return JSON.parse(Buffer.concat(chunks).toString('utf8'));
  }
  async function restoreBundle(document) {
    await openNativeExport(page, true);
    const portable = page.getByRole('region', {name: 'Portable sandbox bundle', exact: true});
    await portable.getByLabel('Load portable bundle').setInputFiles({
      name: 'custom-icons.bundle.json', mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(document))
    });
    await portable.getByRole('status').filter({hasText: 'Bundle validated'}).waitFor();
    await portable.getByRole('button', {name: 'Restore portable workspace', exact: true}).click();
    await page.locator('.ReactModal__Overlay').waitFor({state: 'hidden', timeout: 60000});
    await page.waitForFunction(id => window.__keplerComputeTest.getState().publications
      .some(source => source.id === id && source.selectedCount === 5), datasetId);
    await page.waitForFunction(id => window.__keplerComputeTest.renderedLayers()
      .some(value => value.id === id && value.visible && value.loaded && value.iconGeometryBufferId), layer.id);
    assert.equal((await pick(1))?.sourceRowIndex, 1, 'restored custom catalog draws the elevated icon');
  }
  const document = await saveBundle();
  const customIcons = [...svgIcons, {id: 'custom-square', mesh: {
    positions: [[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]], cells: [[0,1,2],[0,2,3]]
  }}];
  document.iconCatalog = {format: 'luma-kepler-icon-catalog', version: 1, icons: customIcons};
  await restoreBundle(document);
  const embedded = await saveBundle();
  assert.deepEqual(embedded.iconCatalog, document.iconCatalog, 'save embeds installed custom catalog');
  // Remove the CDN fixture: restored diamond meshes must now come from the embedded catalog.
  defaultIcons = [];
  await openPage(address);
  await restoreBundle(embedded);
  const resaved = await saveBundle();
  assert.deepEqual(resaved.iconCatalog, document.iconCatalog, 'custom catalog survives a clean-page restore');
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, labelLayers, embeddedCatalogChecked: true}));
} finally {
  await browser.close();
}
