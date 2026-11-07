// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Headless check of the resident native 3D (glTF per point) layer: renders from a local glb,
// honors opacity and the altitude column, and picks source rows through filter masking.
// Usage: node scripts/native-scenegraph.mjs [url]
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const MODEL_URL = 'http://127.0.0.1:5999/models/quad.glb';
const locations = [
  [-73.995, 40.735, 0],
  [-73.965, 40.735, 6000],
  [-73.995, 40.765, 0],
  [-73.965, 40.765, 0]
].map(row => row.map(Math.fround));

/** A 400 m square (glTF Y-up) that faces the camera when tilted by the native orientation. */
function makeQuadGlb() {
  const positions = new Float32Array([-200, -200, 0, 200, -200, 0, 200, 200, 0, -200, 200, 0]);
  const normals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
  const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);
  const binary = Buffer.concat([positions, normals, indices].map(array => Buffer.from(array.buffer)));
  const json = {
    asset: {version: '2.0'},
    scene: 0,
    scenes: [{nodes: [0]}],
    nodes: [{mesh: 0}],
    meshes: [{primitives: [{attributes: {POSITION: 0, NORMAL: 1}, indices: 2, material: 0}]}],
    materials: [{pbrMetallicRoughness: {baseColorFactor: [0.9, 0.2, 0.2, 1], metallicFactor: 0}, doubleSided: true}],
    buffers: [{byteLength: binary.length}],
    bufferViews: [
      {buffer: 0, byteOffset: 0, byteLength: 48},
      {buffer: 0, byteOffset: 48, byteLength: 48},
      {buffer: 0, byteOffset: 96, byteLength: 12}
    ],
    accessors: [
      {bufferView: 0, componentType: 5126, count: 4, type: 'VEC3', min: [-200, -200, 0], max: [200, 200, 0]},
      {bufferView: 1, componentType: 5126, count: 4, type: 'VEC3'},
      {bufferView: 2, componentType: 5123, count: 6, type: 'SCALAR'}
    ]
  };
  const jsonBytes = Buffer.from(JSON.stringify(json));
  const jsonPadded = Buffer.concat([jsonBytes, Buffer.alloc((4 - (jsonBytes.length % 4)) % 4, 0x20)]);
  const binaryPadded = Buffer.concat([binary, Buffer.alloc((4 - (binary.length % 4)) % 4)]);
  const header = Buffer.alloc(12);
  header.write('glTF', 0, 'ascii');
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(12 + 8 + jsonPadded.length + 8 + binaryPadded.length, 8);
  const jsonHeader = Buffer.alloc(8);
  jsonHeader.writeUInt32LE(jsonPadded.length, 0);
  jsonHeader.write('JSON', 4, 'ascii');
  const binaryHeader = Buffer.alloc(8);
  binaryHeader.writeUInt32LE(binaryPadded.length, 0);
  binaryHeader.write('BIN\0', 4, 'binary');
  return Buffer.concat([header, jsonHeader, jsonPadded, binaryHeader, binaryPadded]);
}

const table = new arrow.Table({
  east: arrow.vectorFromArray(locations.map(row => row[0]), new arrow.Float32()),
  north: arrow.vectorFromArray(locations.map(row => row[1]), new arrow.Float32()),
  height: arrow.vectorFromArray(locations.map(row => row[2]), new arrow.Float32()),
  source_row: arrow.vectorFromArray([0, 1, 2, 3], new arrow.Float32())
});
const bytes = Buffer.from(arrow.tableToIPC(table, 'stream'));
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
page.setDefaultTimeout(30000);
const watchdog = setTimeout(async () => {
  console.error('Native scenegraph check exceeded its four-minute deadline');
  await Promise.race([browser.close(), new Promise(resolve => setTimeout(resolve, 5000))]);
  process.exit(1);
}, 240000);
await pinViteModules(page);
const failures = [];
const consoleErrors = [];
page.on('pageerror', error => failures.push(error.message));
page.on('console', message => {
  if (message.type() === 'error') consoleErrors.push(message.text().slice(0, 400));
});
await page.route(MODEL_URL, route =>
  route.fulfill({
    status: 200,
    contentType: 'model/gltf-binary',
    headers: {'access-control-allow-origin': '*'},
    body: makeQuadGlb()
  })
);
const checks = [];
let datasetId;
let layerId;
const rendered = () =>
  page.evaluate(id => window.__keplerComputeTest.renderedLayers().find(layer => layer.id === id), layerId);
const pick = (row, altitude = locations[row][2]) =>
  page.evaluate(
    ({identifier, longitude, latitude, altitude}) =>
      window.__keplerComputeTest.pickPublishedPosition(identifier, longitude, latitude, altitude),
    {identifier: datasetId, longitude: locations[row][0], latitude: locations[row][1], altitude}
  );
async function apply(configuration, predicate) {
  const frame = await page.evaluate(() => window.__keplerComputeTest.getState().frameCount);
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), configuration);
  await page.waitForFunction(
    ({layerId, frame, predicateSource}) => {
      const api = window.__keplerComputeTest;
      const state = api.getState();
      if (state.error) throw new Error(state.error);
      const layer = api.renderedLayers().find(value => value.id === layerId);
      return state.frameCount > frame && layer?.loaded && layer.visible && new Function('layer', predicateSource)(layer);
    },
    {layerId, frame, predicateSource: `return ${predicate}`},
    {timeout: 45000}
  );
}
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: 'scenegraph.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: bytes});
  await workspace.getByLabel('Source longitude column').selectOption('east');
  await workspace.getByLabel('Source latitude column').selectOption('north');
  await workspace.getByLabel('Source altitude column').selectOption('height');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 4);
  await closeNativeDialog(page);
  datasetId = (await page.evaluate(() => window.__keplerComputeTest.getState().publications[0])).id;
  const configuration = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const layers = configuration.config.visState.layers;
  const layer = layers.find(candidate => candidate.config.dataId === datasetId);
  assert(layer, 'publication must create a native layer');
  layerId = layer.id;
  for (const candidate of layers) candidate.config.isVisible = candidate.id === layerId;
  layer.type = '3D';
  layer.config.columnMode = 'points';
  layer.config.columns = {lng: 'east', lat: 'north', altitude: 'height'};
  layer.config.textLabel = [];
  layer.visualChannels = {};
  layer.config.visConfig = {...layer.config.visConfig, scenegraph: MODEL_URL, sizeScale: 4, opacity: 0.5};
  configuration.config.mapState = {...configuration.config.mapState, longitude: -73.98, latitude: 40.75, zoom: 12, pitch: 50, bearing: 0};
  await apply(configuration, "layer.layerClass === 'ResidentScenegraphPointLayer'");
  const first = await rendered();
  assert.equal(first.opacity, 0.5, 'native opacity reaches the 3D layer');
  assert.equal(failures.length, 0, failures.join('\n'));
  assert.equal((await pick(0))?.sourceRowIndex, 0, 'ground instance picks its source row');
  assert.equal((await pick(1))?.sourceRowIndex, 1, 'altitude instance picks at its elevated projection');
  assert.notEqual((await pick(1, 0))?.sourceRowIndex, 1, 'elevated instance is not at its ground projection');
  assert.equal((await pick(3))?.sourceRowIndex, 3);
  checks.push('3D layer renders, opacity 0.5 reaches the layer, picks source rows incl. elevated instance');
  await page.screenshot({path: '/tmp/luma-kepler-scenegraph.png'});
  await page.evaluate(id => window.__keplerComputeTest.setPublishedRange(id, 'source_row', 1, 1), datasetId);
  await page.waitForFunction(id => window.__keplerComputeTest.getState().publications.find(source => source.id === id)?.selectedCount === 1, datasetId);
  assert.equal((await pick(1))?.sourceRowIndex, 1, 'selected instance still picks');
  assert.equal(await pick(0), null, 'filtered-out instance is masked from picking');
  assert.equal(await pick(3), null, 'filtered-out instance is masked from picking');
  checks.push('filter masking hides instances from picking');
  assert.equal(consoleErrors.filter(text => /pipeline|wgsl|shader|attachment/i.test(text)).length, 0, consoleErrors.join('\n'));
  console.log(JSON.stringify({passed: true, checks}, null, 2));
} catch (error) {
  console.error(JSON.stringify({completedChecks: checks, consoleErrors: consoleErrors.slice(-5)}));
  console.error(await Promise.race([page.evaluate(() => JSON.stringify({layers: window.__keplerComputeTest?.renderedLayers().map(layer => ({id: layer.id, layerClass: layer.layerClass, loaded: layer.loaded, visible: layer.visible, opacity: layer.opacity})), publications: window.__keplerComputeTest?.getState().publications.map(source => source.id), hostError: window.__keplerComputeTest?.getState().error})), new Promise(resolve => setTimeout(() => resolve('diagnostics timed out'), 5000))]));
  throw error;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
