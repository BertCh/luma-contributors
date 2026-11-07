// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, openNativeExport, closeNativeDialog} from './native-navigation.mjs';

const ring = (west, south, east, north, altitude = 0) => [
  [west, south, altitude],
  [east, south, altitude],
  [east, north, altitude],
  [west, north, altitude],
  [west, south, altitude]
];
const feature = (geometry, properties = {}) =>
  JSON.stringify({type: 'Feature', geometry, properties});
// Original row 0 is null. Polygon, line and point identities remain independent across both batches.
const geometry = [
  null,
  feature(
    {
      type: 'Polygon',
      coordinates: [ring(-74.025, 40.735, -74.005, 40.755), ring(-74.02, 40.74, -74.01, 40.75)]
    },
    {
      elevation: 300,
      elevationOffset: 0,
      fillColor: [245, 40, 150],
      lineColor: [20, 220, 230],
      lineWidth: 3
    }
  ),
  feature(
    {
      type: 'LineString',
      coordinates: [
        [-74.03, 40.765, 30],
        [-74.015, 40.765, 30]
      ]
    },
    {lineColor: [20, 220, 230], lineWidth: 4}
  ),
  feature(
    {type: 'Point', coordinates: [-74.025, 40.78, 40]},
    {fillColor: [90, 235, 80], radius: 80, lineWidth: 2}
  ),
  feature(
    {type: 'Polygon', coordinates: [ring(-73.995, 40.735, -73.975, 40.755, 50)]},
    {elevation: 600, elevationOffset: -50, fillColor: [245, 40, 150]}
  ),
  feature(
    {type: 'Polygon', coordinates: [ring(-74.005, 40.768, -73.985, 40.788, 10)]},
    {elevation: 900, fillColor: [245, 40, 150]}
  )
];
const source = arrow
  .tableFromArrays({
    geometry: geometry.slice(0, 3),
    sourceId: new Float32Array([0, 1, 2]),
    heightMetric: new Float32Array([0, 100, 300]),
    widthMetric: new Float32Array([0, 2, 8])
  })
  .concat(
    arrow.tableFromArrays({
      geometry: geometry.slice(3),
      sourceId: new Float32Array([3, 4, 5]),
      heightMetric: new Float32Array([200, 400, 500]),
      widthMetric: new Float32Array([3, 4, 6])
    })
  );
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const failures = [];
page.on('pageerror', error => failures.push(error.message));
page.on('console', message => {
  if (
    message.type() === 'error' &&
    /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())
  )
    failures.push(message.text());
});
let datasetId;
let layerId;
const checks = [];
let currentStage = 'startup';
function stage(name) {
  currentStage = name;
  console.log(JSON.stringify({stage: name}));
}
const watchdog = setTimeout(() => {
  console.error(`GeoJSON vertical workflow exceeded 120 seconds at ${currentStage}`);
  void browser.close();
}, 120000);
const mapRegion = {left: 400, right: 1050, top: 140, bottom: 850};

async function state() {
  return page.evaluate(() => window.__keplerComputeTest.getState());
}
async function layerProperties() {
  const snapshot = await page.waitForFunction(
    id => {
      const api = window.__keplerComputeTest;
      if (api.getState().error) throw new Error(api.getState().error);
      const layers = api.renderedLayers().filter(layer => layer.id === id && layer.visible);
      return layers.length > 0 && layers.every(layer => layer.loaded) ? layers : false;
    },
    layerId,
    {timeout: 15000}
  );
  const layers = await snapshot.jsonValue();
  await snapshot.dispose();
  return layers;
}
async function nativeConfig() {
  return page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
}
function nativeLayer(config) {
  const layer = config.config.visState.layers.find(layer => layer.id === layerId);
  assert(layer, `Native GeoJSON layer ${layerId} must exist`);
  return layer;
}
async function apply(config) {
  const frame = (await state()).frameCount;
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), config);
  await page.waitForFunction(
    ({id, frame}) => {
      const api = window.__keplerComputeTest;
      const state = api.getState();
      if (state.error) throw new Error(state.error);
      const native = state.nativeLayers.find(layer => layer.id === id);
      if (native?.valid === false) throw new Error(JSON.stringify(native));
      return (
        state.frameCount > frame &&
        api.renderedLayers().some(layer => layer.id === id && layer.visible && layer.loaded)
      );
    },
    {id: layerId, frame},
    {timeout: 45000}
  );
}
async function changeVisConfig(changes, channels = {}) {
  const config = await nativeConfig();
  const layer = nativeLayer(config);
  layer.config.visConfig = {...layer.config.visConfig, ...changes};
  layer.visualChannels = {...layer.visualChannels, ...channels};
  await apply(config);
}
async function waitScalarBinding(channel, scale) {
  await page.waitForFunction(
    ({id, channel, scale}) => {
      const api = window.__keplerComputeTest;
      const current = api.getStyleLegend(id, channel);
      if (api.getState().error) throw new Error(api.getState().error);
      // Linear/identity scalar styles intentionally do not trigger legend readback.
      const native = api.saveNativeConfig().config.visState.layers.find(layer => layer.id === id);
      const fixed = channel === 'height' && native.config.visConfig.fixedHeight;
      const actualScale = fixed ? 'identity' : native.visualChannels[`${channel}Scale`];
      const rendered = api
        .renderedLayers()
        .find(layer => layer.renderId === `${id}-${channel === 'height' ? 'fill' : 'paths'}`);
      const value = channel === 'height' ? rendered?.getElevation : rendered?.getWidth;
      return (
        current?.expected.revision > 0 &&
        !current.reason &&
        actualScale === scale &&
        rendered?.loaded &&
        value &&
        typeof value === 'object' &&
        value.bufferIds.length > 0
      );
    },
    {id: layerId, channel, scale},
    {timeout: 45000}
  );
}
async function snapshot(name) {
  await page.evaluate(
    () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))
  );
  const legend = page.locator('.map-legend').first();
  const legendBounds = (await legend.count()) ? await legend.boundingBox() : null;
  const image = PNG.sync.read(
    await page.screenshot({path: `/tmp/luma-kepler-geojson-${name}.png`})
  );
  let coloredPixels = 0;
  const values = [];
  for (let vertical = mapRegion.top; vertical < mapRegion.bottom; vertical++) {
    for (let horizontal = mapRegion.left; horizontal < mapRegion.right; horizontal++) {
      if (
        legendBounds &&
        horizontal >= legendBounds.x &&
        horizontal <= legendBounds.x + legendBounds.width &&
        vertical >= legendBounds.y &&
        vertical <= legendBounds.y + legendBounds.height
      ) {
        values.push(null);
        continue;
      }
      const offset = (vertical * image.width + horizontal) * 4;
      const channels = Array.from(image.data.subarray(offset, offset + 3));
      values.push(channels);
      if (Math.max(...channels) > 120 && Math.max(...channels) - Math.min(...channels) > 60)
        coloredPixels++;
    }
  }
  assert(coloredPixels > 100, `${name} must draw real geometry pixels (${coloredPixels})`);
  return {coloredPixels, values};
}
function changedPixels(left, right) {
  return left.values.reduce(
    (count, channels, index) =>
      count +
      Number(
        Boolean(
          channels &&
            right.values[index] &&
            channels.some((value, channel) => Math.abs(value - right.values[index][channel]) > 8)
        )
      ),
    0
  );
}
function renderedLayer(layers, suffix) {
  const layer = layers.find(layer => layer.renderId === `${layerId}-${suffix}`);
  assert(layer?.loaded && layer.visible, `Expected loaded ${suffix} primitive`);
  return layer;
}
function vectorProperty(property, name) {
  assert(
    property && typeof property === 'object' && property.bufferIds.length > 0,
    `${name} must bind a resident vector`
  );
  return property;
}
async function pick(longitude, latitude) {
  return page.evaluate(
    ({id, longitude, latitude}) =>
      window.__keplerComputeTest.pickPublishedPosition(id, longitude, latitude),
    {id: datasetId, longitude, latitude}
  );
}
async function selectNativeField(label, field) {
  let group = page.locator('.layer-config-group').filter({hasText: label}).last();
  if (!(await group.count())) {
    await page.locator('.layer__enable-config svg').first().click();
    group = page.locator('.layer-config-group').filter({hasText: label}).last();
  }
  await group.waitFor();
  if ((await group.getAttribute('class'))?.includes('collapsed'))
    await group.locator('.layer-config-group__header').click();
  await group.locator('.field-selector').first().click();
  await page
    .locator('.field-selector_list-item')
    .filter({has: page.getByText(field, {exact: true})})
    .click();
}

try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {
    waitUntil: 'domcontentloaded'
  });
  await page.waitForFunction(
    () => {
      const overlay = document.querySelector('vite-error-overlay');
      if (overlay) throw new Error(overlay.shadowRoot?.textContent?.slice(0, 2000));
      const api = window.__keplerComputeTest;
      if (api?.getState().error) throw new Error(api.getState().error);
      return api?.getState().selectedCount === 256;
    },
    undefined,
    {timeout: 60000}
  );
  stage('prepare native Arrow source');
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: 'geojson-vertical.arrow',
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))
  });
  await workspace.getByLabel('Source geometry column').selectOption('geometry');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 6,
    undefined,
    {timeout: 45000}
  );
  await closeNativeDialog(page);
  const initial = await state();
  datasetId = initial.publications[0].id;
  layerId = `${datasetId}-geometry`;
  await page.evaluate(id => {
    const identity = window.__keplerComputeTest.resourceIdentity();
    window.__verticalSourceIdentity = {
      geometry: [...identity.geometryBuffers[id]],
      columns: [...identity.sourceBuffers[id]]
    };
  }, datasetId);
  assert.deepEqual(initial.publications[0].sourceBatches, [3, 3]);
  assert.equal(initial.publications[0].rowCount, 6);
  assert.equal(initial.publications[0].error, null);
  assert(
    initial.publications[0].geometry.sourceBufferIds.length > 0,
    'Geometry source storage must be observable'
  );
  const config = await nativeConfig();
  for (const layer of config.config.visState.layers) layer.config.isVisible = layer.id === layerId;
  const layer = nativeLayer(config);
  layer.config.color = [245, 40, 150];
  layer.config.visConfig = {
    ...layer.config.visConfig,
    filled: true,
    stroked: true,
    enable3d: false,
    wireframe: false,
    elevationOffset: 0,
    elevationScale: 1,
    thickness: 3,
    opacity: 1,
    strokeOpacity: 1,
    strokeColor: [20, 220, 230],
    heightRange: [0, 1500],
    sizeRange: [1, 12]
  };
  stage('configure native geometry');
  await apply(config);
  await page.evaluate(() =>
    window.__keplerComputeTest.setMapViewport({
      longitude: -74.005,
      latitude: 40.759,
      zoom: 12.2,
      pitch: 0,
      bearing: 0
    })
  );
  await page.waitForFunction(
    id =>
      window.__keplerComputeTest
        .renderedLayers()
        .filter(layer => layer.id === id && layer.visible && layer.loaded).length >= 4,
    layerId
  );
  stage('flat geometry and exact source picking');
  await snapshot('flat');
  assert.equal(
    (await pick(-74.023, 40.737))?.sourceRowIndex,
    1,
    'Polygon shell retains null-prefixed row1'
  );
  assert.equal(await pick(-74.015, 40.745), null, 'Polygon hole remains empty');
  assert.equal(
    (await pick(-74.0225, 40.765))?.sourceRowIndex,
    2,
    'Line keeps independent original row2'
  );
  assert.equal((await pick(-74.025, 40.78))?.sourceRowIndex, 3, 'XYZ point keeps original row3');
  checks.push('holed XYZ polygon and independent line/point picking preserve source rows');

  stage('native height dropdown');
  await changeVisConfig({enable3d: true});
  await selectNativeField(/Height based on/i, 'heightMetric');
  await page.waitForFunction(
    id =>
      window.__keplerComputeTest
        .saveNativeConfig()
        .config.visState.layers.find(layer => layer.id === id).visualChannels.heightField?.name ===
      'heightMetric',
    layerId
  );
  await changeVisConfig(
    {},
    {sizeField: {name: 'widthMetric', type: 'real'}, sizeScale: 'linear', heightScale: 'linear'}
  );
  stage('current scalar height and width bindings');
  await waitScalarBinding('height', 'linear');
  await waitScalarBinding('size', 'linear');
  let properties = await layerProperties();
  const heightVector = vectorProperty(
    renderedLayer(properties, 'fill').getElevation,
    'height field'
  );
  const widthVector = vectorProperty(
    renderedLayer(properties, 'paths').getWidth,
    'stroke width field'
  );
  assert.notDeepEqual(
    heightVector.bufferIds,
    widthVector.bufferIds,
    'Height and stroke width must use independent resident results'
  );
  assert(renderedLayer(properties, 'fill').extruded);
  checks.push('native height dropdown and independent stroke-width field');

  await page.evaluate(() => window.__keplerComputeTest.setMapViewport({pitch: 50, bearing: 0}));
  stage('mapped and fixed heights');
  const scaled = await snapshot('mapped-height');
  await changeVisConfig({fixedHeight: true, invertHeight: false});
  await waitScalarBinding('height', 'identity');
  const fixed = await snapshot('fixed-height');
  const fixedWidth = renderedLayer(await layerProperties(), 'paths');
  assert.equal(fixedWidth.widthScale, renderedLayer(properties, 'paths').widthScale);
  assert(
    changedPixels(scaled, fixed) > 100,
    'Fixed height changes mapped heights on the actual map'
  );
  stage('inverted fixed height');
  await changeVisConfig({invertHeight: true});
  await waitScalarBinding('height', 'identity');
  const inverted = await snapshot('inverted-height');
  assert(changedPixels(fixed, inverted) > 100, 'Inverted fixed height changes visible extrusion');
  assert.equal(renderedLayer(await layerProperties(), 'paths').widthScale, fixedWidth.widthScale);
  checks.push('fixed and inverted heights change pixels without coupling line width');

  stage('offset-only flat extrusion');
  await changeVisConfig(
    {enable3d: false, fixedHeight: false, invertHeight: false, elevationOffset: 120},
    {heightField: null}
  );
  properties = await layerProperties();
  const offsetOnly = renderedLayer(properties, 'fill');
  assert.equal(offsetOnly.extruded, true);
  assert.equal(
    offsetOnly.getElevation,
    0,
    'Positive uniform offset only must flatten column height'
  );
  vectorProperty(
    offsetOnly.getElevationOffset,
    'embedded zero/negative offsets with uniform fallback'
  );
  assert.deepEqual(JSON.parse(geometry[1]).properties.elevationOffset, 0);
  assert.deepEqual(JSON.parse(geometry[4]).properties.elevationOffset, -50);
  await snapshot('offset-only');
  checks.push(
    'positive uniform offset activates flat raised surfaces with resident zero/negative source overrides'
  );

  stage('filled versus wireframe pixels');
  await changeVisConfig({enable3d: true, filled: true, wireframe: false, elevationOffset: 0});
  const filled = await snapshot('filled');
  await changeVisConfig({filled: false, wireframe: true});
  properties = await layerProperties();
  const wireframe = renderedLayer(properties, 'fill');
  assert.equal(wireframe.filled, false);
  assert.equal(wireframe.wireframe, true);
  const wire = await snapshot('wireframe-only');
  assert(
    filled.coloredPixels > wire.coloredPixels + 100,
    `Wireframe-only removes filled surfaces (${filled.coloredPixels} -> ${wire.coloredPixels})`
  );
  checks.push('wireframe-only native config draws edges without polygon fill');

  await changeVisConfig(
    {filled: true, wireframe: true, fixedHeight: true, invertHeight: true},
    {heightField: {name: 'heightMetric', type: 'real'}, heightScale: 'linear'}
  );
  await waitScalarBinding('height', 'identity');
  stage('filtered source masks and picking');
  const beforeFilter = await layerProperties();
  await page.evaluate(
    id => window.__keplerComputeTest.setPublishedRange(id, 'sourceId', 4, 5),
    datasetId
  );
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0].selectedCount === 2
  );
  const afterFilter = await layerProperties();
  assert.deepEqual(
    renderedLayer(afterFilter, 'fill').maskBufferIds,
    renderedLayer(beforeFilter, 'fill').maskBufferIds
  );
  assert.deepEqual(
    renderedLayer(afterFilter, 'paths').maskBufferIds,
    renderedLayer(beforeFilter, 'paths').maskBufferIds
  );
  await page.evaluate(() => window.__keplerComputeTest.setMapViewport({pitch: 0}));
  await changeVisConfig({enable3d: false, wireframe: false, invertHeight: false});
  assert.equal(await pick(-74.023, 40.737), null);
  assert.equal(await pick(-74.0225, 40.765), null);
  assert.equal(await pick(-74.025, 40.78), null);
  assert.equal((await pick(-73.985, 40.745))?.sourceRowIndex, 4);
  assert.equal((await pick(-73.995, 40.778))?.sourceRowIndex, 5);
  checks.push('selection masks update in place and picking keeps surviving original rows4/5');

  await changeVisConfig({
    enable3d: true,
    filled: true,
    wireframe: true,
    fixedHeight: true,
    invertHeight: true
  });
  await waitScalarBinding('height', 'identity');
  stage('native save and replay');
  await openNativeExport(page, true);
  const saving = page.getByRole('region', {name: 'Save map', exact: true});
  await saving.getByRole('button', {name: 'Save state', exact: true}).click();
  const saved = JSON.parse(
    await saving.getByLabel('Saved sandbox state', {exact: true}).inputValue()
  );
  const savedLayer = nativeLayer(saved.nativeConfig);
  assert.equal(savedLayer.visualChannels.heightField.name, 'heightMetric');
  assert.equal(savedLayer.visualChannels.sizeField.name, 'widthMetric');
  assert.equal(savedLayer.config.visConfig.invertHeight, true);
  assert.equal(savedLayer.config.visConfig.fixedHeight, true);
  assert.equal(savedLayer.config.visConfig.wireframe, true);
  const changed = structuredClone(saved.nativeConfig);
  Object.assign(nativeLayer(changed).config.visConfig, {
    enable3d: false,
    wireframe: false,
    invertHeight: false
  });
  await apply(changed);
  await page.evaluate(
    id => window.__keplerComputeTest.setPublishedRange(id, 'sourceId', 0, 5),
    datasetId
  );
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0].selectedCount === 6
  );
  // Applying native config resets native dialogs; reopen the actual Save map workspace.
  await openNativeExport(page, true);
  await saving.getByText('Inspect or edit saved state', {exact: true}).click();
  await saving.getByLabel('Saved sandbox state', {exact: true}).fill(JSON.stringify(saved));
  await saving.getByRole('button', {name: 'Replay state', exact: true}).click();
  await closeNativeDialog(page);
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0].selectedCount === 2
  );
  await waitScalarBinding('height', 'identity');
  const replayed = nativeLayer(await nativeConfig());
  assert.deepEqual(replayed.visualChannels, savedLayer.visualChannels);
  assert.equal(replayed.config.visConfig.invertHeight, true);
  assert.equal(replayed.config.visConfig.wireframe, true);
  stage('immutable source allocation checks');
  const final = await state();
  assert.equal(final.error, '');
  assert.equal(final.publications[0].sourceUploadBytes, initial.publications[0].sourceUploadBytes);
  assert.deepEqual(final.publications[0].sourceBufferIds, initial.publications[0].sourceBufferIds);
  assert.equal(
    final.publications[0].geometry.sourceUploadBytes,
    initial.publications[0].geometry.sourceUploadBytes
  );
  assert.deepEqual(
    final.publications[0].geometry.sourceBufferIds,
    initial.publications[0].geometry.sourceBufferIds
  );
  assert.equal(final.uploadBytes, initial.uploadBytes);
  assert(
    await page.evaluate(id => {
      const identity = window.__keplerComputeTest.resourceIdentity();
      const initial = window.__verticalSourceIdentity;
      return (
        initial.geometry.length === identity.geometryBuffers[id].length &&
        initial.columns.length === identity.sourceBuffers[id].length &&
        initial.geometry.every((buffer, index) => buffer === identity.geometryBuffers[id][index]) &&
        initial.columns.every((buffer, index) => buffer === identity.sourceBuffers[id][index])
      );
    }, datasetId),
    'Source columns and geometry must retain the exact Buffer objects through replay'
  );
  checks.push(
    'native Save map/replay preserves vertical channels and immutable source allocations'
  );
  assert.deepEqual(failures, []);
  console.log(
    JSON.stringify(
      {
        passed: true,
        sourceBatches: [3, 3],
        sourceRows: [1, 2, 3, 4, 5],
        pixels: {
          mapped: scaled.coloredPixels,
          fixed: fixed.coloredPixels,
          inverted: inverted.coloredPixels,
          filled: filled.coloredPixels,
          wireframe: wire.coloredPixels
        },
        checks
      },
      null,
      2
    )
  );
} catch (error) {
  console.error(`Failed at ${currentStage}`, error);
  const diagnostic = page.evaluate(
    layerId => ({
      state: window.__keplerComputeTest?.getState(),
      rendered: window.__keplerComputeTest?.renderedLayers(),
      native: window.__keplerComputeTest?.saveNativeConfig(),
      heightLegend: window.__keplerComputeTest?.getStyleLegend(layerId, 'height'),
      widthLegend: window.__keplerComputeTest?.getStyleLegend(layerId, 'size'),
      text: document.body.innerText.slice(-12000)
    }),
    layerId
  );
  const timeout = new Promise(resolve =>
    setTimeout(() => resolve({diagnostic: 'page did not respond within 5 seconds'}), 5000)
  );
  console.error(
    JSON.stringify(
      await Promise.race([diagnostic.catch(error => ({diagnostic: String(error)})), timeout]),
      null,
      2
    )
  );
  throw error;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
