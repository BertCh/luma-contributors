// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog, openNativeExport} from './native-navigation.mjs';

const locations = [[-73.995, 40.735, 0], [-73.965, 40.735, 6000], [-73.995, 40.765, -1000], [-73.965, 40.765, null], [-73.98, 40.75, NaN]].map(row => row.map(value => value === null ? null : Math.fround(value)));
const batches = [[0, 1], [], [2, 3, 4]].map(indices => {
  const table = new arrow.Table({east: arrow.vectorFromArray(indices.map(index => locations[index][0]), new arrow.Float32()), north: arrow.vectorFromArray(indices.map(index => locations[index][1]), new arrow.Float32()), height: arrow.vectorFromArray(indices.map(index => locations[index][2]), new arrow.Float32()), source_row: arrow.vectorFromArray(indices, new arrow.Float32())});
  return arrow.tableFromIPC(arrow.tableToIPC(table, 'stream')).batches[0];
});
const bytes = Buffer.from(arrow.tableToIPC(new arrow.Table(batches[0].schema, batches), 'stream'));
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
page.setDefaultTimeout(30000);
const watchdog = setTimeout(async () => {
  console.error('Native altitude workflow exceeded its four-minute deadline');
  await Promise.race([browser.close(), new Promise(resolve => setTimeout(resolve, 5000))]);
  process.exit(1);
}, 240000);
await pinViteModules(page);
const failures = [];
page.on('pageerror', error => failures.push(error.message));
let datasetId;
let layerId;
const checks = [];
const publication = () => page.evaluate(id => window.__keplerComputeTest.getState().publications.find(source => source.id === id), datasetId);
const rendered = () => page.evaluate(id => window.__keplerComputeTest.renderedLayers().find(layer => layer.id === id && layer.layerClass === 'GPUScatterplotLayer'), layerId);
async function pick(row, altitude = locations[row][2] ?? 0, identifier = datasetId) {
  return page.evaluate(({identifier, longitude, latitude, altitude}) => window.__keplerComputeTest.pickPublishedPosition(identifier, longitude, latitude, altitude), {identifier, longitude: locations[row][0], latitude: locations[row][1], altitude});
}
async function apply(configuration, format) {
  const frame = await page.evaluate(() => window.__keplerComputeTest.getState().frameCount);
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), configuration);
  await page.waitForFunction(({datasetId, layerId, frame, format}) => {
    const api = window.__keplerComputeTest;
    const state = api.getState();
    if (state.error) throw new Error(state.error);
    const source = state.publications.find(value => value.id === datasetId);
    const layer = api.renderedLayers().find(value => value.id === layerId && value.layerClass === 'GPUScatterplotLayer');
    return !source?.scopePending && state.frameCount > frame && layer?.loaded && layer.visible && layer.getPosition?.format === format;
  }, {datasetId, layerId, frame, format}, {timeout: 45000});
}
async function readDownload(button) {
  const pending = page.waitForEvent('download');
  await button.click();
  const stream = await (await pending).createReadStream();
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return Buffer.concat(chunks);
}
try {
  const address = process.argv[2] || 'http://127.0.0.1:5318/?rows=256';
  await page.goto(address, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: 'point-altitude.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: bytes});
  await workspace.getByLabel('Source longitude column').selectOption('east');
  await workspace.getByLabel('Source latitude column').selectOption('north');
  await workspace.getByLabel('Source altitude column').selectOption('height');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 5);
  await closeNativeDialog(page);
  const source = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0]);
  datasetId = source.id;
  assert.deepEqual(source.sourceBatches, [2, 0, 3]);
  const configuration = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const layer = configuration.config.visState.layers.find(candidate => candidate.config.dataId === datasetId);
  assert(layer, 'Altitude publication must create a native point layer');
  layerId = layer.id;
  for (const candidate of configuration.config.visState.layers) candidate.config.isVisible = candidate.id === layerId;
  layer.config.columnMode = 'points';
  layer.config.columns = {lng: 'east', lat: 'north', altitude: 'height'};
  layer.config.textLabel = [];
  layer.config.color = [240, 80, 30];
  layer.visualChannels = {...layer.visualChannels, colorField: null, sizeField: null, strokeColorField: null};
  layer.config.visConfig = {...layer.config.visConfig, radius: 10, fixedRadius: false, opacity: 1, filled: true, outline: false};
  configuration.config.mapState = {...configuration.config.mapState, longitude: -73.98, latitude: 40.75, zoom: 12, pitch: 50, bearing: 0};
  await apply(configuration, 'float32x3');
  const elevated = await rendered();
  assert.equal(elevated.depthCompare, 'less-equal');
  assert.equal(elevated.depthWriteEnabled, true);
  assert.equal((await pick(1))?.sourceRowIndex, 1, 'Raised source point must pick at its elevated projection');
  assert.notEqual((await pick(1, 0))?.sourceRowIndex, 1, 'Raised point must not remain at its ground projection');
  assert.equal((await pick(0))?.sourceRowIndex, 0);
  assert.equal((await pick(2))?.sourceRowIndex, 2);
  assert.equal((await pick(3))?.sourceRowIndex, 3, 'Arrow null altitude must render at zero');
  assert.equal(await pick(4, 0), null, 'Nonfinite altitude must be hidden');
  checks.push('pitched native XYZ picking, zero/negative/null/native Arrow semantics, nonfinite mask');
  await page.screenshot({path: '/tmp/luma-kepler-point-altitude.png'});
  const flat = structuredClone(configuration);
  flat.config.visState.layers.find(candidate => candidate.id === layerId).config.columns.altitude = null;
  await apply(flat, 'float32x2');
  assert.equal((await rendered()).depthCompare, 'always');
  assert.equal((await pick(1, 0))?.sourceRowIndex, 1);
  assert.equal((await pick(4, 0))?.sourceRowIndex, 4, 'Deselecting altitude restores original XY visibility');
  await apply(configuration, 'float32x3');
  assert.deepEqual((await rendered()).positionBufferIds, elevated.positionBufferIds);
  assert.deepEqual((await publication()).sourceBufferIds, source.sourceBufferIds);
  checks.push('native altitude deselection/reselection reuses XYZ and XY buffers');
  await page.evaluate(id => window.__keplerComputeTest.setPublishedRange(id, 'source_row', 1, 1), datasetId);
  await page.waitForFunction(id => window.__keplerComputeTest.getState().publications.find(source => source.id === id)?.selectedCount === 1, datasetId);
  assert.equal((await pick(1))?.sourceRowIndex, 1);
  assert.equal(await pick(0), null);
  await apply(configuration, 'float32x3');
  assert.deepEqual((await rendered()).positionBufferIds, elevated.positionBufferIds);
  assert.deepEqual((await publication()).sourceBufferIds, source.sourceBufferIds);
  checks.push('native filters retain source IDs and immutable position storage');
  await openNativeExport(page, true);
  const bundle = await readDownload(page.getByRole('button', {name: 'Save portable bundle', exact: true}));
  const saved = JSON.parse(bundle.toString('utf8'));
  const descriptor = saved.state.sources.find(source => source.dataset.datasetId === datasetId);
  assert.deepEqual(descriptor.reference.parameters.coordinates, {longitude: 'east', latitude: 'north', altitude: 'height'});
  await page.reload({waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeExport(page, true);
  const portable = page.getByRole('region', {name: 'Portable sandbox bundle', exact: true});
  await portable.getByLabel('Load portable bundle').setInputFiles({name: 'altitude.bundle.json', mimeType: 'application/json', buffer: bundle});
  await portable.getByRole('status').filter({hasText: 'Bundle validated'}).waitFor();
  await portable.getByRole('button', {name: 'Restore portable workspace', exact: true}).click();
  await page.locator('.ReactModal__Overlay').waitFor({state: 'hidden', timeout: 60000});
  await page.waitForFunction(id => window.__keplerComputeTest.getState().publications.find(source => source.id === id)?.selectedCount === 5, datasetId, {timeout: 45000});
  await apply(configuration, 'float32x3');
  assert.equal((await pick(1))?.sourceRowIndex, 1);
  assert.equal((await pick(3))?.sourceRowIndex, 3);
  assert.deepEqual((await publication()).sourceBatches, [2, 0, 3]);
  checks.push('portable bundle restores optional altitude role, original chunks and native XYZ config');
  await openNativeData(page);
  const stream = page.getByTestId('arrow-stream-panel');
  await stream.getByLabel('Arrow IPC stream file').setInputFiles({name: 'stream-altitude.arrows', mimeType: 'application/vnd.apache.arrow.stream', buffer: bytes});
  await stream.getByRole('button', {name: 'Open stream', exact: true}).click();
  await stream.getByRole('status').filter({hasText: 'paused: 1 batches'}).waitFor();
  await stream.getByRole('button', {name: 'Continue stream', exact: true}).click();
  await stream.getByRole('status').filter({hasText: 'exhausted:'}).waitFor();
  await stream.getByRole('button', {name: 'Finish stream', exact: true}).click();
  await stream.getByRole('status').filter({hasText: 'complete: 3 batches, 5 admitted rows'}).waitFor();
  await stream.getByLabel(/^Longitude/).selectOption('east');
  await stream.getByLabel(/^Latitude/).selectOption('north');
  await stream.getByLabel(/^Altitude \(optional\)/).selectOption('height');
  await stream.getByRole('button', {name: 'Add completed stream', exact: true}).click();
  await page.locator('.ReactModal__Overlay').waitFor({state: 'hidden'});
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications.length === 2);
  const streamedId = await page.evaluate(id => window.__keplerComputeTest.getState().publications.find(source => source.id !== id).id, datasetId);
  const streamedConfig = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  assert.equal(streamedConfig.config.visState.layers.find(layer => layer.config.dataId === streamedId).config.columns.altitude, 'height');
  checks.push('completed Arrow stream hands off its selected altitude role');
  assert.equal(Boolean((await page.evaluate(() => window.__keplerComputeTest.getState())).error), false);
  assert.deepEqual(failures, []);
  console.log(JSON.stringify({passed: true, checks}, null, 2));
} catch (error) {
  console.error(JSON.stringify({completedChecks: checks}));
  console.error(await Promise.race([page.evaluate(() => JSON.stringify({state: window.__keplerComputeTest?.getState(), layers: window.__keplerComputeTest?.renderedLayers(), config: window.__keplerComputeTest?.saveNativeConfig(), alerts: [...document.querySelectorAll('[role=alert]')].map(element => element.textContent), stream: document.querySelector('[data-testid=arrow-stream-panel]')?.textContent})), new Promise(resolve => setTimeout(() => resolve('diagnostics timed out'), 5000))]));
  throw error;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
