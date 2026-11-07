// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {openNativeData} from './native-navigation.mjs';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const origin = Date.UTC(2026, 0, 1);
const url = process.argv[2] || 'http://127.0.0.1:5318/?rows=256';
const lineString = coordinates =>
  JSON.stringify({
    type: 'LineString',
    coordinates: coordinates.map(([longitude, latitude, time]) => [longitude, latitude, 0, origin + time])
  });
const eastPath = latitude => lineString([[-74.03, latitude, 0], [-74, latitude, 10000], [-73.97, latitude, 20000]]);
// Trips 40.74 and 40.77 run east; the third runs north in the middle of the screen.
const northPath = lineString([[-74, 40.753, 0], [-74, 40.755, 10000], [-74, 40.757, 20000]]);
const routes = arrow
  .tableFromArrays({route: [null, eastPath(40.74)], measure: new Float32Array([0, 1])})
  .concat(arrow.tableFromArrays({route: [eastPath(40.77), northPath], measure: new Float32Array([2, 3])}));

const modelColumns = (ids, seconds, longitudes, latitudes, values) => ({
  trip: ids,
  time: new Float64Array(seconds.map(second => origin + second * 1000)),
  lon: new Float64Array(longitudes),
  lat: new Float64Array(latitudes),
  ...Object.fromEntries(Object.entries(values).map(([name, list]) => [name, new Float32Array(list)]))
});
// One grouped trip heading east through the middle of the screen with per-vertex datum columns.
const makeGrouped = values =>
  arrow.tableFromArrays(modelColumns(['A', 'A', 'A'], [0, 10, 20], [-74.03, -74, -73.97], [40.755, 40.755, 40.755], values));

/** A flat z-up triangle in meters whose tip points along +x: asymmetric about its origin. */
const ARROW_POSITIONS = [60, 0, 0, -30, 30, 0, -30, -30, 0];

function makePng(width, pixels) {
  const png = new PNG({width, height: 1});
  png.data = Buffer.from(pixels.flat());
  return PNG.sync.write(png);
}

/** Builds a GLB: an arrow, optionally textured (left red, right green) and/or spinning about z. */
function makeGlb({textured = false, animated = false} = {}) {
  const positions = new Float32Array(ARROW_POSITIONS);
  const normals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
  const uvs = new Float32Array([1, 0.5, 0, 0, 0, 1]);
  const times = new Float32Array([0, 1, 2, 3, 4]);
  const half = Math.SQRT1_2;
  const rotations = new Float32Array([0, 0, 0, 1, 0, 0, half, half, 0, 0, 1, 0, 0, 0, half, -half, 0, 0, 0, -1]);
  const image = makePng(4, [[255, 0, 0, 255], [255, 0, 0, 255], [0, 255, 0, 255], [0, 255, 0, 255]]);
  const pieces = [positions, normals];
  if (textured) pieces.push(uvs);
  if (animated) pieces.push(times, rotations);
  if (textured) pieces.push(image);
  let offset = 0;
  const bufferViews = pieces.map(piece => {
    const view = {buffer: 0, byteOffset: offset, byteLength: piece.byteLength};
    offset += Math.ceil(piece.byteLength / 4) * 4;
    return view;
  });
  const accessors = [
    {bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', min: [-30, -30, 0], max: [60, 30, 0]},
    {bufferView: 1, componentType: 5126, count: 3, type: 'VEC3'}
  ];
  let next = 2;
  const attributes = {POSITION: 0, NORMAL: 1};
  if (textured) {
    accessors.push({bufferView: next++, componentType: 5126, count: 3, type: 'VEC2'});
    attributes.TEXCOORD_0 = accessors.length - 1;
  }
  const animations = [];
  if (animated) {
    accessors.push({bufferView: next++, componentType: 5126, count: 5, type: 'SCALAR', min: [0], max: [4]});
    accessors.push({bufferView: next++, componentType: 5126, count: 5, type: 'VEC4'});
    animations.push({
      channels: [{sampler: 0, target: {node: 0, path: 'rotation'}}],
      samplers: [{input: accessors.length - 2, output: accessors.length - 1, interpolation: 'LINEAR'}]
    });
  }
  const material = {pbrMetallicRoughness: {baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 1}};
  if (textured) material.pbrMetallicRoughness.baseColorTexture = {index: 0};
  const json = JSON.stringify({
    asset: {version: '2.0'},
    scene: 0,
    scenes: [{nodes: [0]}],
    nodes: [{mesh: 0, ...(animated ? {rotation: [0, 0, 0, 1]} : {})}],
    meshes: [{primitives: [{attributes, material: 0}]}],
    materials: [material],
    ...(textured ? {textures: [{source: 0}], images: [{bufferView: next++, mimeType: 'image/png'}]} : {}),
    ...(animations.length ? {animations} : {}),
    buffers: [{byteLength: offset}],
    bufferViews,
    accessors
  });
  const jsonBytes = Buffer.from(json + ' '.repeat((4 - (json.length % 4)) % 4));
  const binary = Buffer.alloc(offset);
  pieces.forEach((piece, index) => Buffer.from(piece.buffer, piece.byteOffset, piece.byteLength).copy(binary, bufferViews[index].byteOffset));
  const total = 12 + 8 + jsonBytes.length + 8 + binary.length;
  const header = Buffer.alloc(12);
  header.writeUInt32LE(0x46546c67, 0);
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(total, 8);
  const jsonHeader = Buffer.alloc(8);
  jsonHeader.writeUInt32LE(jsonBytes.length, 0);
  jsonHeader.writeUInt32LE(0x4e4f534a, 4);
  const binaryHeader = Buffer.alloc(8);
  binaryHeader.writeUInt32LE(binary.length, 0);
  binaryHeader.writeUInt32LE(0x004e4942, 4);
  return Buffer.concat([header, jsonHeader, jsonBytes, binaryHeader, binary]);
}

const MODELS = {
  'plain.glb': makeGlb(),
  'textured.glb': makeGlb({textured: true}),
  'animated.glb': makeGlb({animated: true})
};
// PBR darkens the tint, so the hue is matched, not the full-intensity magenta.
const isMagenta = (red, green, blue) => red > 50 && blue > 50 && green < 45 && Math.abs(red - blue) < 40;
const isRed = (red, green, blue) => red > 50 && red > green * 2.5 && red > blue * 2.5;
const isGreen = (red, green, blue) => green > 35 && green > red * 2 && green > blue * 2;
const isBlue = (red, green, blue) => blue > 90 && blue > red * 2.5 && blue > green * 1.5;

async function openApp(label) {
  const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  await pinViteModules(page);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/trip-marker-*.glb', route => {
    const name = new URL(route.request().url()).pathname.split('trip-marker-')[1];
    return route.fulfill({status: 200, contentType: 'model/gltf-binary', body: MODELS[name]});
  });
  await page.goto(url, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page);
  return {browser, page, errors, label};
}

async function upload(page, name, table) {
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name, mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(table, 'stream'))});
  return workspace;
}
async function publish(page, workspace, tripCount) {
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(count => window.__keplerComputeTest.getState().publications[0]?.trips?.tripCount === count, tripCount, {timeout: 45000});
  return (await page.evaluate(() => window.__keplerComputeTest.getState())).publications[0].id;
}
async function publishRoutes(page) {
  const workspace = await upload(page, 'timed-paths.arrow', routes);
  await workspace.getByLabel('Source geometry column').selectOption('route');
  await workspace.getByLabel('Timed paths (Trips)', {exact: true}).check();
  await workspace.getByLabel('Trip timestamp encoding', {exact: true}).selectOption('unix-millisecond');
  await workspace.getByLabel('Trip time origin (epoch milliseconds)', {exact: true}).fill(String(origin));
  await workspace.getByLabel('Maximum trip time error (milliseconds)', {exact: true}).fill('0.1');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByText(/3 timed paths, 9 vertices/).waitFor();
  return publish(page, workspace, 3);
}
async function publishGrouped(page, values) {
  const workspace = await upload(page, 'grouped-trips.arrow', makeGrouped(values));
  await workspace.getByLabel('Grouped timed rows (Trips)', {exact: true}).check();
  await workspace.getByLabel('Trip ID column', {exact: true}).selectOption('trip');
  await workspace.getByLabel('Trip time column', {exact: true}).selectOption('time');
  await workspace.getByLabel('Trip longitude column', {exact: true}).selectOption('lon');
  await workspace.getByLabel('Trip latitude column', {exact: true}).selectOption('lat');
  await workspace.getByLabel('Trip timestamp encoding', {exact: true}).selectOption('unix-millisecond');
  await workspace.getByLabel('Trip time origin (epoch milliseconds)', {exact: true}).fill(String(origin));
  for (const field of ['time', 'lon', 'lat']) await workspace.getByLabel(`${field} (Float64)`, {exact: true}).uncheck();
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByText(/1 timed grouped trips, 3 vertices/).waitFor();
  return publish(page, workspace, 1);
}

/** Shows only this dataset's Trips with native marker settings, and waits for the marker layer. */
async function applyMarkerConfig(page, datasetId, visConfig, layerConfig = {}) {
  const nativeConfig = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  for (const layer of nativeConfig.config.visState.layers) {
    layer.config.isVisible = layer.config.dataId === datasetId;
    if (layer.type === 'trip' && layer.config.dataId === datasetId) {
      Object.assign(layer.config, layerConfig);
      Object.assign(layer.config.visConfig, {
        // Thin trail so the marker is the only marker-colored feature.
        thickness: 0.5,
        trailLength: 1,
        scenegraphEnabled: true,
        scenegraph: 'custom',
        scenegraphColorEnabled: false,
        scenegraphUseTrailColor: false,
        sizeScale: 2,
        ...visConfig
      });
    }
  }
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), nativeConfig);
  await page.waitForFunction(() => {
    const api = window.__keplerComputeTest;
    const state = api.getState();
    if (state.error) throw new Error(state.error);
    const diagnostic = state.nativeLayers.find(layer => layer.valid === false);
    if (diagnostic) throw new Error(JSON.stringify(diagnostic));
    return api.renderedLayers().some(layer => layer.layerClass === 'GPUTripMarkerLayer' && layer.visible && layer.loaded);
  }, undefined, {timeout: 45000});
}
const near = {halfWidth: 100, halfHeight: 100};
/** Native roll, pitch and yaw fields are not saved with Kepler configurations; set them live. */
async function setFields(page, datasetId, fields) {
  await page.evaluate(([id, value]) => window.__keplerComputeTest.setTripMarkerFields(`${id}-trips`, value), [datasetId, fields]);
  await page.waitForFunction(() => {
    const api = window.__keplerComputeTest;
    const diagnostic = api.getState().nativeLayers.find(layer => layer.valid === false);
    if (diagnostic) throw new Error(JSON.stringify(diagnostic));
    return api.renderedLayers().some(layer => layer.layerClass === 'GPUTripMarkerLayer' && layer.visible && layer.loaded);
  });
}
const modelUrl = (page, name) => `${new URL(page.url()).origin}/trip-marker-${name}`;
async function setTime(page, offset) {
  await page.evaluate(time => window.__keplerComputeTest.setAnimationTime(time), origin + offset);
  await page.waitForFunction(time => window.__keplerComputeTest.getState().animation.currentTime === time, origin + offset);
  await page.waitForTimeout(500);
}

/** Deck viewports project relative to the map canvas; screenshots and the mouse use page coordinates. */
async function projectToPage(page, longitude, latitude) {
  return page.evaluate(([lng, lat]) => {
    const [x, y] = window.__keplerComputeTest.projectPosition(lng, lat);
    const canvas = [...document.querySelectorAll('canvas')].sort((a, b) => b.clientWidth * b.clientHeight - a.clientWidth * a.clientHeight)[0];
    const rect = canvas.getBoundingClientRect();
    return [x + rect.left, y + rect.top];
  }, [longitude, latitude]);
}
/** Marker pixels near a projected position: count, extent and the thickness of each side. */
async function scan(page, longitude, latitude, matches = isMagenta, {halfWidth = 160, halfHeight = 60} = {}) {
  const [x, y] = await projectToPage(page, longitude, latitude);
  const image = PNG.sync.read(await page.screenshot());
  const columns = new Map();
  const rows = new Map();
  let count = 0;
  let sumY = 0;
  for (let vertical = Math.max(0, Math.floor(y - halfHeight)); vertical < Math.min(image.height, y + halfHeight); vertical++) {
    for (let horizontal = Math.max(0, Math.floor(x - halfWidth)); horizontal < Math.min(image.width, x + halfWidth); horizontal++) {
      const offset = (vertical * image.width + horizontal) * 4;
      const [red, green, blue] = image.data.subarray(offset, offset + 3);
      if (matches(red, green, blue)) {
        count++;
        sumY += vertical;
        columns.set(horizontal, (columns.get(horizontal) ?? 0) + 1);
        rows.set(vertical, (rows.get(vertical) ?? 0) + 1);
      }
    }
  }
  const sides = counts => {
    const keys = [...counts.keys()].sort((a, b) => a - b);
    const first = keys[0] ?? 0;
    const length = (keys.at(-1) ?? 0) - first + 1;
    const thickness = (from, to) => keys.filter(key => key >= first + length * from && key < first + length * to).reduce((sum, key) => sum + counts.get(key), 0);
    return {length, low: thickness(0, 0.3), high: thickness(0.7, 1)};
  };
  const horizontal = sides(columns);
  const vertical = sides(rows);
  return {count, centroidY: count ? sumY / count : 0, width: horizontal.length, height: vertical.length, leftThickness: horizontal.low, rightThickness: horizontal.high, topThickness: vertical.low, bottomThickness: vertical.high};
}

const results = {};
async function runScenario(label, run) {
  const app = await openApp(label);
  const {browser, page, errors} = app;
  try {
    results[label] = await run(app);
    const state = await page.evaluate(() => window.__keplerComputeTest.getState());
    assert.equal(state.error, '');
    assert.deepEqual(errors, [], `${label} page errors`);
  } catch (error) {
    await page.screenshot({path: `/tmp/luma-kepler-trip-markers-${label}-failed.png`}).catch(() => undefined);
    console.error(JSON.stringify(await page.evaluate(() => ({layers: window.__keplerComputeTest?.renderedLayers().map(l => [l.id, l.layerClass, l.visible, l.loaded].join(' ')), error: window.__keplerComputeTest?.getState().error})), null, 1).slice(0, 3000));
    console.error(errors);
    throw error;
  } finally {
    await page.screenshot({path: `/tmp/luma-kepler-trip-markers-${label}.png`}).catch(() => undefined);
    await browser.close();
  }
}

// Scenario 1: GeoJSON trips with a tinted flat model: position, heading, window and filtering.
await runScenario('geojson', async ({page}) => {
  const datasetId = await publishRoutes(page);
  await applyMarkerConfig(page, datasetId, {
    scenegraphCustomModelUrl: modelUrl(page, 'plain.glb'),
    scenegraphColorEnabled: true,
    scenegraphColor: [255, 0, 255]
  });
  // Positions at 5 s and 15 s: halfway along each leg of the 40.74 and 40.77 trips.
  await setTime(page, 5000);
  const early = await scan(page, -74.015, 40.74);
  const earlyNorth = await scan(page, -74.015, 40.77);
  // The northbound trip is at 40.754 after 5 s; the arrow is about 100 pixels long.
  const northbound = await scan(page, -74, 40.754, isMagenta, {halfWidth: 100, halfHeight: 100});
  await setTime(page, 15000);
  const late = await scan(page, -73.985, 40.74);
  const staleEarly = await scan(page, -74.015, 40.74);
  assert(early.count > 300 && earlyNorth.count > 300, `Markers must draw at both trips (${early.count}, ${earlyNorth.count})`);
  assert(late.count > 300, `Marker follows animation time (${late.count})`);
  assert(staleEarly.count < 20, `Marker leaves its earlier position (${staleEarly.count})`);
  // East-bound trips: the wide base trails on the left and the tip points right.
  assert(early.leftThickness > early.rightThickness * 1.5, `East heading follows the speed vector (${early.leftThickness} vs ${early.rightThickness})`);
  // North-bound trip: the base is at the bottom and the tip points up the screen.
  assert(northbound.count > 300, `Northbound marker draws (${northbound.count})`);
  assert(northbound.bottomThickness > northbound.topThickness * 1.5, `North heading points up (${northbound.bottomThickness} vs ${northbound.topThickness})`);
  assert(northbound.rightThickness < northbound.leftThickness * 1.5 && northbound.leftThickness < northbound.rightThickness * 1.5, 'North-bound marker is not skewed sideways');
  // Outside the trip's time window the marker disappears.
  await setTime(page, 25000);
  assert((await scan(page, -73.97, 40.74)).count < 20, 'No marker after trip end');
  // Scalar selection hides the filtered trip's marker only.
  await setTime(page, 15000);
  await page.evaluate(id => window.__keplerComputeTest.setPublishedRange(id, 'measure', 2, 2), datasetId);
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0].selectedCount === 1);
  await page.waitForTimeout(400);
  const filteredOut = await scan(page, -73.985, 40.74);
  const kept = await scan(page, -73.985, 40.77);
  assert(filteredOut.count < 20, `Filtered trip hides its marker (${filteredOut.count})`);
  assert(kept.count > 300, `Selected trip keeps its marker (${kept.count})`);
  return {early, earlyNorth, northbound, late, filteredOut, kept};
});

// Scenario 2: model textures and PBR materials (native `_lighting: 'pbr'`).
await runScenario('textured', async ({page}) => {
  const datasetId = await publishRoutes(page);
  await applyMarkerConfig(page, datasetId, {scenegraphCustomModelUrl: modelUrl(page, 'textured.glb')});
  await setTime(page, 5000);
  const red = await scan(page, -74, 40.754, isRed, near);
  const green = await scan(page, -74, 40.754, isGreen, near);
  const magenta = await scan(page, -74, 40.754, isMagenta, near);
  // The base is textured red, the tip green; no tint is applied.
  assert(red.count > 100 && green.count > 100, `Texture colors reach the marker (red ${red.count}, green ${green.count})`);
  assert(magenta.count < 20, 'No tint is applied to the textured model');
  // The northbound marker points up: red at the base (bottom), green at the tip (top).
  assert(red.centroidY > green.centroidY + 10, 'Texture coordinates map red to the base and green to the tip');
  return {red, green};
});

// Scenario 3: glTF model animations (native `_animations: {'*': {speed: 5}}`).
await runScenario('animated', async ({page}) => {
  const datasetId = await publishRoutes(page);
  await applyMarkerConfig(page, datasetId, {
    scenegraphCustomModelUrl: modelUrl(page, 'animated.glb'),
    scenegraphColorEnabled: true,
    scenegraphColor: [255, 0, 255]
  });
  await setTime(page, 5000);
  // Animation time comes from Deck's timeline, so a fixed trip time still animates the model.
  const signatures = new Set();
  for (let sample = 0; sample < 8; sample++) {
    const frame = await scan(page, -74, 40.754, isMagenta, {halfWidth: 130, halfHeight: 130});
    signatures.add(`${frame.width}x${frame.height}:${Math.round(frame.leftThickness / 40)}:${Math.round(frame.topThickness / 40)}`);
    await page.waitForTimeout(170);
  }
  assert(signatures.size >= 3, `The model animation changes the marker over time (${[...signatures].join(' ')})`);
  return {signatures: [...signatures]};
});

// Scenario 4: roll, pitch and yaw datum fields and per-timestamp colors for table-mode Trips.
await runScenario('table', async ({page}) => {
  const datasetId = await publishGrouped(page, {yawEast: [90, 90, 90], yawNorth: [0, 0, 0], rollQuarter: [90, 90, 90], level: [1, 0, 2]});
  const magenta = {scenegraphCustomModelUrl: modelUrl(page, 'plain.glb'), scenegraphColorEnabled: true, scenegraphColor: [255, 0, 255]};
  // Without datum fields the heading follows the speed vector (east).
  await applyMarkerConfig(page, datasetId, magenta);
  await setTime(page, 5000);
  const baseline = await scan(page, -74.015, 40.755, isMagenta, near);
  assert(baseline.count > 300 && baseline.leftThickness > baseline.rightThickness * 1.5, `Table-mode marker heads east (${baseline.leftThickness} vs ${baseline.rightThickness})`);
  // Native yaw = 90 - bearing: a bearing of 90 points east, a bearing of 0 points up the screen.
  await applyMarkerConfig(page, datasetId, magenta);
  await setFields(page, datasetId, {yaw: 'yawEast'});
  await setTime(page, 5000);
  const eastYaw = await scan(page, -74.015, 40.755, isMagenta, near);
  assert(eastYaw.count > 300 && eastYaw.leftThickness > eastYaw.rightThickness * 1.5, `Yaw field bearing 90 points east (${eastYaw.leftThickness} vs ${eastYaw.rightThickness})`);
  await setFields(page, datasetId, {yaw: 'yawNorth'});
  await setTime(page, 5000);
  const northYaw = await scan(page, -74.015, 40.755, isMagenta, near);
  assert(northYaw.count > 300 && northYaw.bottomThickness > northYaw.topThickness * 1.5, `Yaw field bearing 0 points up although the trip runs east (${northYaw.bottomThickness} vs ${northYaw.topThickness})`);
  // A roll field of 90 degrees stands the flat arrow on its edge, seen edge-on from above.
  await setFields(page, datasetId, {yaw: null, roll: 'rollQuarter'});
  await setTime(page, 5000);
  const rolled = await scan(page, -74.015, 40.755, isMagenta, near);
  assert(rolled.count < baseline.count * 0.4, `Roll field turns the model edge-on (${rolled.count} vs ${baseline.count})`);
  // Per-timestamp colors: bisectRight picks vertex 1 (below the median level, red) at 5 s and vertex 2 (above, blue) at 15 s.
  await applyMarkerConfig(page, datasetId, {
    scenegraphCustomModelUrl: modelUrl(page, 'plain.glb'),
    scenegraphColorEnabled: true,
    scenegraphUseTrailColor: true,
    colorRange: {name: 'Custom', type: 'sequential', category: 'Custom', colors: ['#FF0000', '#0000FF']}
  });
  // Live fields survive configuration changes: clear the roll field and choose the color field.
  await setFields(page, datasetId, {color: 'level'});
  await setTime(page, 5000);
  const firstColor = {red: await scan(page, -74.015, 40.755, isRed), blue: await scan(page, -74.015, 40.755, isBlue)};
  await setTime(page, 15000);
  const secondColor = {red: await scan(page, -73.985, 40.755, isRed), blue: await scan(page, -73.985, 40.755, isBlue)};
  assert(firstColor.red.count > 300 && firstColor.blue.count < 20, `The marker is red before the second vertex (${JSON.stringify(firstColor)})`);
  assert(secondColor.blue.count > 300 && secondColor.red.count < 20, `The marker turns blue for the last vertex (${JSON.stringify(secondColor)})`);
  return {baseline, eastYaw, northYaw, rolled, firstColor, secondColor};
});

// Scenario 5 (FLOW2-TRIP): scenegraph markers on a globe view draw at the trip position, face the
// speed vector, move with animation time and are hidden when the globe is turned away.
await runScenario('globe', async ({page}) => {
  const datasetId = await publishRoutes(page);
  await applyMarkerConfig(page, datasetId, {
    scenegraphCustomModelUrl: modelUrl(page, 'plain.glb'),
    scenegraphColorEnabled: true,
    scenegraphColor: [255, 0, 255]
  });
  const setView = async target => {
    const current = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
    current.config.mapState = {
      ...current.config.mapState, ...target, zoom: 12, pitch: 0, bearing: 0, dragRotate: false,
      mapViewMode: 'MODE_GLOBE', globe: {...(current.config.mapState.globe ?? {}), enabled: true}
    };
    await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), current);
    await page.waitForFunction(() => {
      const api = window.__keplerComputeTest;
      const diagnostic = api.getState().nativeLayers.find(layer => layer.valid === false);
      if (diagnostic) throw new Error(JSON.stringify(diagnostic));
      return api.renderedLayers().some(layer => layer.layerClass === 'GPUTripMarkerLayer' && layer.visible && layer.loaded);
    }, undefined, {timeout: 45000});
  };
  await setView({longitude: -74.015, latitude: 40.74});
  await setTime(page, 5000);
  const near = await scan(page, -74.015, 40.74, isMagenta, {halfWidth: 120, halfHeight: 50});
  assert(near.count > 100, `Globe marker draws at the trip position (${near.count})`);
  assert(near.leftThickness > near.rightThickness * 1.2, `Globe marker heads east (${near.leftThickness} vs ${near.rightThickness})`);
  await setTime(page, 15000);
  await setView({longitude: -73.985, latitude: 40.74});
  const later = await scan(page, -73.985, 40.74, isMagenta, {halfWidth: 120, halfHeight: 50});
  assert(later.count > 100, `Globe marker follows animation time (${later.count})`);
  // Antipode: the near side of the globe is turned away, so the marker is culled.
  await setView({longitude: 105.985, latitude: -40.74});
  const far = await scan(page, -73.985, 40.74, isMagenta, {halfWidth: 400, halfHeight: 400});
  assert(far.count < 20, `Far-side globe marker is culled (${far.count})`);
  return {near, later, far};
});

console.log(JSON.stringify({passed: true, results, checks: [
  'marker draws at current position', 'marker moves with animation time', 'east and north heading from speed vector',
  'hidden outside trip window', 'filtered trip hides marker', 'textured PBR model pixels', 'glTF animation changes over time',
  'yaw and roll datum fields', 'table-mode per-timestamp colors', 'globe marker position, heading, animation and culling'
]}));
