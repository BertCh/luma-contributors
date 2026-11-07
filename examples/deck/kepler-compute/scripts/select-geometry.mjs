// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {cellToBoundary, cellToLatLng, latLngToCell} from 'h3-js';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const polygon = {type: 'Feature', id: 'original-source-id', properties: {label: 'retained polygon', score: 7}, geometry: {type: 'Polygon', coordinates: [[[-74.02, 40.72], [-73.98, 40.72], [-73.98, 40.78], [-74.02, 40.78], [-74.02, 40.72]]]}};
const otherPolygon = {type: 'Polygon', coordinates: [[[-73.91, 40.72], [-73.89, 40.72], [-73.89, 40.78], [-73.91, 40.78], [-73.91, 40.72]]]};
const geometrySource = arrow.tableFromArrays({geometry: [null, JSON.stringify(polygon)], value: new Float32Array([0, 7])}).concat(arrow.tableFromArrays({geometry: [JSON.stringify(otherPolygon)], value: new Float32Array([9])}));
const firstCell = latLngToCell(37.78, -122.41, 7);
const secondCell = latLngToCell(37.84, -122.3, 7);
const h3Source = new arrow.Table({cell: arrow.vectorFromArray([null, BigInt(`0x${firstCell}`)], new arrow.Uint64())}).concat(new arrow.Table({cell: arrow.vectorFromArray([BigInt(`0x${secondCell}`)], new arrow.Uint64())}));
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
const watchdog = setTimeout(async () => {
  console.error('Native Select Geometry workflow exceeded three minutes');
  await Promise.race([browser.close(), new Promise(resolve => setTimeout(resolve, 3000))]);
  process.exit(1);
}, 180000);
page.on('pageerror', error => errors.push(error.stack || error.message));
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  const selectedIds = [];
  for (const [kind, table, column, center] of [
    ['geojson', geometrySource, 'geometry', [-74, 40.75]],
    ['hexagonId', h3Source, 'cell', cellToLatLng(firstCell).reverse()]
  ]) {
    console.log(JSON.stringify({stage: 'publish', kind}));
    await openNativeData(page);
    const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
    await workspace.locator('input[type=file]').setInputFiles({name: `${kind}-selection.arrow`, mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(table, 'stream'))});
    await workspace.getByLabel(kind === 'geojson' ? 'Source geometry column' : 'Source H3 column').selectOption(column);
    await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
    await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
    await page.waitForFunction(count => window.__keplerComputeTest.getState().publications.length === count, selectedIds.length + 1);
    await closeNativeDialog(page);
    const state = await page.evaluate(() => window.__keplerComputeTest.getState());
    const dataset = state.publications.at(-1);
    assert.deepEqual(dataset.sourceBatches, [2, 1]);
    const configuration = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
    const layer = configuration.config.visState.layers.find(candidate => candidate.config.dataId === dataset.id);
    layer.config.label = `${kind} selection target`;
    for (const candidate of configuration.config.visState.layers) candidate.config.isVisible = candidate.id === layer.id;
    layer.config.visConfig = {...layer.config.visConfig, filled: true, outline: false, enable3d: false, coverage: 0.5, opacity: 1};
    configuration.config.mapState = {...configuration.config.mapState, longitude: center[0], latitude: center[1], zoom: 11, pitch: 0, bearing: 0};
    await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), configuration);
    await page.waitForFunction(identifier => window.__keplerComputeTest.renderedLayers().some(candidate => candidate.id === identifier && candidate.visible && candidate.loaded), layer.id, {timeout: 45000});
    const picked = await page.evaluate(({datasetId, center}) => window.__keplerComputeTest.pickPublishedPosition(datasetId, ...center), {datasetId: dataset.id, center});
    assert.equal(picked.sourceRowIndex, 1);
    const previousFeatures = await page.evaluate(() => window.__keplerComputeTest.nativeEditor().features.length);
    const position = await page.evaluate(location => window.__keplerComputeTest.projectPosition(...location), center);
    console.log(JSON.stringify({stage: 'native geometry click', kind}));
    await page.mouse.click(position[0], position[1]);
    await page.getByRole('button', {name: 'Select Geometry', exact: true}).click();
    await page.waitForFunction(count => window.__keplerComputeTest.nativeEditor().features.length === count, previousFeatures + 1);
    const editor = await page.evaluate(() => window.__keplerComputeTest.nativeEditor());
    const selected = editor.selectedFeature;
    assert.equal(typeof selected.id, 'string');
    assert(!selectedIds.includes(selected.id));
    selectedIds.push(selected.id);
    assert.equal(editor.selectionContext.mapIndex, 0);
    assert.equal(editor.selectionContext.rightClick, true);
    assert(Math.abs(editor.selectionContext.position.x - position[0]) < 1);
    assert(Math.abs(editor.selectionContext.position.y - position[1]) < 1);
    if (kind === 'geojson') {
      assert.notEqual(selected.id, polygon.id);
      assert.deepEqual(selected.geometry, polygon.geometry);
      assert.equal(selected.properties.label, polygon.properties.label);
      assert.equal(selected.properties.score, 7);
    } else {
      assert.equal(selected.geometry.type, 'Polygon');
      const boundary = cellToBoundary(firstCell, true);
      assert.equal(selected.geometry.coordinates.length, 1);
      assert.equal(selected.geometry.coordinates[0].length, boundary.length);
      // Browser WASM and Node H3 trigonometry can differ by a few Float64 ulps.
      boundary.forEach((position, index) => position.forEach((coordinate, axis) =>
        assert(Math.abs(selected.geometry.coordinates[0][index][axis] - coordinate) < 1e-12)));
      assert.deepEqual(selected.geometry.coordinates[0][0], selected.geometry.coordinates[0].at(-1));
      assert.equal(selected.properties.isClosed, true);
    }
    // Native editor action uses the copied source geometry, independent of display coverage.
    console.log(JSON.stringify({stage: 'native editor filter', kind}));
    await page.locator('.feature-action-panel .editor-layers-list').hover();
    await page.locator('.feature-action-panel .layer-panel-item').filter({hasText: layer.config.label}).click();
    await page.waitForFunction(identifier => window.__keplerComputeTest.getState().publications.find(candidate => candidate.id === identifier)?.selectedCount === 1, dataset.id);
    const final = await page.evaluate(() => window.__keplerComputeTest.getState());
    assert.equal(final.error, '');
    assert.deepEqual(final.publications.find(candidate => candidate.id === dataset.id).sourceBufferIds, dataset.sourceBufferIds);
    await page.mouse.click(1200, 900);
  }
  assert.deepEqual(errors, []);
  await page.screenshot({path: '/tmp/luma-kepler-select-geometry.png'});
  console.log(JSON.stringify({passed: true, checks: ['actual native frozen popover', 'bounded original GeoJSON row with properties', 'exact original H3 cell independent of coverage', 'fresh editor feature IDs', 'native editor actions and targeted polygon filters', 'source batch identity and stable buffers']}));
} catch (error) {
  console.error(JSON.stringify(await page.evaluate(() => ({state: window.__keplerComputeTest?.getState(), editor: window.__keplerComputeTest?.nativeEditor()})), null, 2));
  console.error(await page.locator('body').innerText());
  console.error(errors);
  throw error;
} finally {clearTimeout(watchdog); await browser.close();}
