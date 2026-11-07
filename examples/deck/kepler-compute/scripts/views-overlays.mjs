// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Point labels and elevated points in the second Dual view, plus the Swipe right side.
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {PNG} from 'pngjs';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog, openNativeExport} from './native-navigation.mjs';

const baseUrl = process.env.KEPLER_COMPUTE_URL ?? 'http://127.0.0.1:5318/';
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
page.setDefaultTimeout(30000);
await pinViteModules(page);
const failures = [];
page.on('pageerror', error => failures.push(error.message));
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|validation error|device.*lost/i.test(message.text()))
    failures.push(message.text());
});

async function publish(bytes, name, columns) {
  await openNativeData(page, 'Prepare data');
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name,
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(bytes)
  });
  await workspace.getByLabel('Source longitude column').selectOption(columns.longitude);
  await workspace.getByLabel('Source latitude column').selectOption(columns.latitude);
  if (columns.altitude)
    await workspace.getByLabel('Source altitude column').selectOption(columns.altitude);
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0]?.selectedCount > 0
  );
  await closeNativeDialog(page);
  return page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);
}
async function chooseSplitMode(title) {
  await page.locator('.map-control-button.split-map').click();
  await page.locator('.toolbar-item', {hasText: new RegExp(`^${title}$`)}).click();
  await page.waitForFunction(
    expected =>
      window.__keplerComputeTest.saveNativeConfig().config.mapState.mapSplitMode === expected,
    {Dual: 'DUAL_MAP', Swipe: 'SWIPE_COMPARE', Single: 'SINGLE_MAP'}[title]
  );
  await page.waitForTimeout(2000);
}
async function setLayerMaps(layerId, left, right) {
  const {splitMaps} = await page.evaluate(
    () => window.__keplerComputeTest.saveNativeConfig().config.visState
  );
  const has = index => Boolean(splitMaps[index]?.layers?.[layerId]);
  if (has(0) !== left)
    await page.evaluate(id => window.__keplerComputeTest.toggleLayerForMap(0, id), layerId);
  await page.waitForTimeout(600);
  if (has(1) !== right)
    await page.evaluate(id => window.__keplerComputeTest.toggleLayerForMap(1, id), layerId);
  await page.waitForTimeout(1500);
}
async function whitePixels(viewId, longitude, latitude, maxX = Infinity) {
  const [centerX, centerY] = await page.evaluate(
    ([viewId, longitude, latitude]) =>
      window.__keplerComputeTest.projectPositionInView(viewId, longitude, latitude),
    [viewId, longitude, latitude]
  );
  const image = PNG.sync.read(await page.screenshot());
  let count = 0;
  for (let y = Math.max(100, Math.floor(centerY - 100)); y < Math.min(image.height - 100, centerY + 100); y++)
    for (let x = Math.max(350, Math.floor(centerX - 110)); x < Math.min(image.width - 80, centerX + 110, maxX); x++) {
      const offset = (y * image.width + x) * 4;
      if (image.data[offset] > 170 && image.data[offset + 1] > 170 && image.data[offset + 2] > 170)
        count++;
    }
  return count;
}

try {
  // ---- Point labels ----
  const labelPositions = [[-74, 40.74], [-73.96, 40.76]];
  const labelTable = new arrow.Table({
    longitude: arrow.makeVector(Float32Array.from(labelPositions, position => position[0])),
    latitude: arrow.makeVector(Float32Array.from(labelPositions, position => position[1])),
    name: arrow.vectorFromArray(['ALPHA', 'BETA'], new arrow.Utf8())
  });
  await page.goto(`${baseUrl}?rows=256`, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256);
  const labelDataset = await publish(arrow.tableToIPC(labelTable, 'stream'), 'labels.arrow', {
    longitude: 'longitude',
    latitude: 'latitude'
  });
  const labelConfig = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const labelLayer = labelConfig.config.visState.layers.find(
    layer => layer.config.dataId === labelDataset
  );
  labelLayer.id = 'view-labels';
  labelLayer.config.visConfig = {...labelLayer.config.visConfig, radius: 10, filled: true, outline: false};
  labelLayer.config.textLabel = [
    {field: {name: 'name', type: 'string'}, color: [255, 255, 255], size: 24, weight: 400, anchor: 'middle', alignment: 'center', outlineWidth: 0, background: false, collisionEnabled: false}
  ];
  labelConfig.config.visState.layers = [labelLayer];
  labelConfig.config.mapState = {...labelConfig.config.mapState, longitude: -73.98, latitude: 40.75, zoom: 12, pitch: 0, bearing: 0};
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), labelConfig);
  await page.waitForFunction(() =>
    window.__keplerComputeTest
      .renderedLayers()
      .some(layer => layer.id === 'view-labels' && layer.layerClass === 'GPUPointLabelLayer' && layer.loaded)
  , undefined, {timeout: 45000});
  await page.waitForTimeout(1500);
  assert.ok((await whitePixels(null, ...labelPositions[1])) > 60, 'single view draws labels');
  await chooseSplitMode('Dual');
  await setLayerMaps('view-labels', false, true);
  assert.ok((await whitePixels('view-1', ...labelPositions[1])) > 60, 'dual right view draws its labels');
  assert.ok((await whitePixels('view-0', ...labelPositions[1])) < 10, 'dual left view excludes the layer labels');
  await setLayerMaps('view-labels', true, true);
  assert.ok((await whitePixels('view-0', ...labelPositions[1])) > 60, 'dual left labels when enabled');
  assert.ok((await whitePixels('view-1', ...labelPositions[1])) > 60, 'dual right labels when enabled');
  await chooseSplitMode('Swipe');
  await setLayerMaps('view-labels', false, true);
  assert.ok((await whitePixels('view-1', ...labelPositions[1])) > 60, 'swipe right side draws labels');
  assert.ok((await whitePixels('view-0', ...labelPositions[0], 690)) < 10, 'swipe left side excludes the layer labels');

  // ---- Elevated points ----
  const locations = [[-73.995, 40.735, 0], [-73.965, 40.735, 6000], [-73.995, 40.765, -1000], [-73.965, 40.765, 0]].map(row => row.map(value => Math.fround(value)));
  const altitudeTable = new arrow.Table({
    east: arrow.vectorFromArray(locations.map(row => row[0]), new arrow.Float32()),
    north: arrow.vectorFromArray(locations.map(row => row[1]), new arrow.Float32()),
    height: arrow.vectorFromArray(locations.map(row => row[2]), new arrow.Float32())
  });
  await page.goto(`${baseUrl}?rows=256`, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256);
  const altitudeDataset = await publish(arrow.tableToIPC(altitudeTable, 'stream'), 'altitude.arrow', {
    longitude: 'east',
    latitude: 'north',
    altitude: 'height'
  });
  const altitudeConfig = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const altitudeLayer = altitudeConfig.config.visState.layers.find(layer => layer.config.dataId === altitudeDataset);
  const altitudeLayerId = altitudeLayer.id;
  for (const layer of altitudeConfig.config.visState.layers) layer.config.isVisible = layer.id === altitudeLayerId;
  altitudeLayer.config.columnMode = 'points';
  altitudeLayer.config.columns = {lng: 'east', lat: 'north', altitude: 'height'};
  altitudeLayer.config.textLabel = [];
  altitudeLayer.config.visConfig = {...altitudeLayer.config.visConfig, radius: 10, fixedRadius: false, opacity: 1, filled: true, outline: false};
  altitudeConfig.config.mapState = {...altitudeConfig.config.mapState, longitude: -73.98, latitude: 40.75, zoom: 12, pitch: 50, bearing: 0};
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), altitudeConfig);
  await page.waitForFunction(id => window.__keplerComputeTest.renderedLayers().some(layer => layer.id === id && layer.loaded && layer.visible && layer.getPosition?.format === 'float32x3'), altitudeLayerId, {timeout: 45000});
  const pick = (viewId, row, altitude) =>
    page.evaluate(
      ({id, viewId, longitude, latitude, altitude}) =>
        window.__keplerComputeTest.pickPublishedPosition(id, longitude, latitude, altitude, viewId),
      {id: altitudeDataset, viewId, longitude: locations[row][0], latitude: locations[row][1], altitude}
    );
  assert.equal((await pick(null, 1, locations[1][2]))?.sourceRowIndex, 1, 'single view picks the raised point');
  await chooseSplitMode('Dual');
  await setLayerMaps(altitudeLayerId, false, true);
  await page.waitForTimeout(1000);
  assert.equal((await pick('view-1', 1, locations[1][2]))?.sourceRowIndex, 1, 'dual right view picks the raised point at its elevated projection');
  assert.notEqual((await pick('view-1', 1, 0))?.sourceRowIndex, 1, 'raised point is not at its ground projection in view 1');
  assert.equal(await pick('view-0', 1, locations[1][2]), null, 'left view excludes the layer');
  await chooseSplitMode('Swipe');
  await setLayerMaps(altitudeLayerId, false, true);
  assert.equal((await pick('view-1', 1, locations[1][2]))?.sourceRowIndex, 1, 'swipe right side picks elevated rows');
  assert.equal(await pick('view-0', 0, locations[0][2]), null, 'swipe left side excludes the layer');

  // ---- Native Select Geometry from a resident polygon in view 1 ----
  const selectionPolygon = {type: 'Polygon', coordinates: [[[-74.02, 40.72], [-73.98, 40.72], [-73.98, 40.78], [-74.02, 40.78], [-74.02, 40.72]]]};
  const selectionTable = arrow.tableFromArrays({geometry: [JSON.stringify({type: 'Feature', id: 'original-source-id', properties: {label: 'retained polygon', score: 7}, geometry: selectionPolygon})], value: new Float32Array([7])});
  await page.goto(`${baseUrl}?rows=256`, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256);
  await openNativeData(page);
  const geometryWorkspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await geometryWorkspace.locator('input[type=file]').setInputFiles({name: 'selection.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(selectionTable, 'stream'))});
  await geometryWorkspace.getByLabel('Source geometry column').selectOption('geometry');
  await geometryWorkspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await geometryWorkspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications.length === 1);
  await closeNativeDialog(page);
  const geometryDataset = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);
  const geometryConfig = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const geometryLayer = geometryConfig.config.visState.layers.find(candidate => candidate.config.dataId === geometryDataset);
  for (const candidate of geometryConfig.config.visState.layers) candidate.config.isVisible = candidate.id === geometryLayer.id;
  geometryLayer.config.visConfig = {...geometryLayer.config.visConfig, filled: true, outline: false, enable3d: false, opacity: 1};
  geometryConfig.config.mapState = {...geometryConfig.config.mapState, longitude: -74.04, latitude: 40.75, zoom: 11, pitch: 0, bearing: 0};
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), geometryConfig);
  await page.waitForFunction(identifier => window.__keplerComputeTest.renderedLayers().some(candidate => candidate.id === identifier && candidate.visible && candidate.loaded), geometryLayer.id, {timeout: 45000});
  for (const title of ['Dual', 'Swipe']) {
    await chooseSplitMode(title);
    await setLayerMaps(geometryLayer.id, false, true);
    await page.waitForTimeout(1000);
    const position = await page.evaluate(() => window.__keplerComputeTest.projectPositionInView('view-1', -74, 40.75));
    assert.ok(position[0] > 760, `${title} polygon center is inside view 1 (${position})`);
    const before = await page.evaluate(() => window.__keplerComputeTest.nativeEditor().features.length);
    await page.mouse.move(position[0] - 4, position[1] - 4);
    await page.mouse.move(position[0], position[1], {steps: 3});
    await page.waitForTimeout(500);
    await page.mouse.click(position[0], position[1]);
    await page.getByRole('button', {name: 'Select Geometry', exact: true}).click();
    await page.waitForFunction(count => window.__keplerComputeTest.nativeEditor().features.length === count + 1, before, {timeout: 15000});
    const editor = await page.evaluate(() => window.__keplerComputeTest.nativeEditor());
    assert.equal(editor.selectionContext.mapIndex, 1, `${title} view 1 select geometry belongs to map 1`);
    assert.deepEqual(editor.selectedFeature.geometry, selectionPolygon, `${title} view 1 selects the original source geometry`);
    assert.equal(editor.selectedFeature.properties.label, 'retained polygon');
    // Remove the drawn selection so the next mode picks the resident polygon, not the editor feature.
    await page.evaluate(() => window.__keplerComputeTest.deleteNativeEditorFeatures());
    await page.mouse.click(1200, 900);
    await page.keyboard.press('Escape');
    await chooseSplitMode('Single');
  }

  // ---- Split state inside an exported portable bundle ----
  async function saveBundle() {
    await openNativeExport(page, true);
    const pending = page.waitForEvent('download', {timeout: 15000}).catch(async error => {
      throw new Error(`${error.message}\n${await page.getByRole('region', {name: 'Portable sandbox bundle', exact: true}).innerText()}`);
    });
    await page.getByRole('button', {name: 'Save portable bundle', exact: true}).click();
    const chunks = [];
    for await (const chunk of await (await pending).createReadStream()) chunks.push(chunk);
    return Buffer.concat(chunks);
  }
  async function restoreBundle(bundle) {
    await page.reload();
    await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
    assert.equal(await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig().config.mapState.mapSplitMode), 'SINGLE_MAP', 'a fresh session starts single');
    await openNativeExport(page, true);
    const portable = page.getByRole('region', {name: 'Portable sandbox bundle', exact: true});
    await portable.getByLabel('Load portable bundle').setInputFiles({name: 'workspace.bundle.json', mimeType: 'application/json', buffer: bundle});
    await portable.getByRole('status').filter({hasText: 'Bundle validated'}).waitFor({timeout: 30000});
    await portable.getByRole('button', {name: 'Restore portable workspace', exact: true}).click();
    await page.locator('.ReactModal__Overlay').waitFor({state: 'hidden', timeout: 60000});
  }
  await page.goto(`${baseUrl}?rows=256`, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256);
  for (const title of ['Swipe', 'Dual']) {
    await chooseSplitMode(title);
    await setLayerMaps('resident-points', true, false);
    if (title === 'Swipe') {
      await page.mouse.move(720, 500);
      await page.mouse.down();
      await page.mouse.move(504, 500, {steps: 6});
      await page.mouse.up();
      await page.waitForFunction(() => {
        const percentage = window.__keplerComputeTest.saveNativeConfig().config.mapState.swipeComparePercentage;
        return percentage > 30 && percentage < 40;
      });
    } else {
      await page.evaluate(() => window.__keplerComputeTest.toggleSplitMapViewport({isViewportSynced: false}));
      await page.waitForTimeout(1500);
    }
    await page.waitForTimeout(800);
    const expected = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig().config);
    const bundle = await saveBundle();
    const saved = JSON.parse(bundle.toString('utf8')).state.nativeConfig.config;
    assert.equal(saved.mapState.mapSplitMode, expected.mapState.mapSplitMode, `${title} mode is inside the bundle`);
    assert.equal(saved.mapState.swipeComparePercentage, expected.mapState.swipeComparePercentage, `${title} divider is inside the bundle`);
    assert.deepEqual(saved.visState.splitMaps, expected.visState.splitMaps, `${title} split layers are inside the bundle`);
    await restoreBundle(bundle);
    await page.waitForFunction(mode => window.__keplerComputeTest.saveNativeConfig().config.mapState.mapSplitMode === mode, expected.mapState.mapSplitMode, {timeout: 30000});
    await page.waitForTimeout(2500);
    const restored = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig().config);
    assert.equal(restored.mapState.swipeComparePercentage, expected.mapState.swipeComparePercentage, `${title} divider restored from the bundle`);
    assert.deepEqual(restored.visState.splitMaps, expected.visState.splitMaps, `${title} split layers restored from the bundle`);
    assert.equal(restored.mapState.isViewportSynced, expected.mapState.isViewportSynced, `${title} viewport sync restored`);
    assert.equal(await page.evaluate(() => window.__keplerComputeTest.getState().ready), true);
    if (title === 'Swipe') {
      const divider = await page.evaluate(() => document.querySelector('.layout-splitter .splitter')?.getBoundingClientRect().x ?? null);
      assert.ok(divider > 480 && divider < 530, `restored bundle divider position ${divider}`);
    } else {
      assert.equal(await page.locator('.maplibregl-map').count(), 2, 'restored Dual bundle mounts both maps');
    }
  }
  assert.deepEqual(failures, [], failures.join('\n'));
  console.log('views-overlays.mjs checks completed');
} finally {
  await browser.close();
}
