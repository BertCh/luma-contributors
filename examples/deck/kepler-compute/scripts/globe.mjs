// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Native GlobeView acceptance for resident layers. Each scenario is published twice, as a one-row
// "near" source at the view center and a one-row "far" source at the antipode. Every scenario
// asserts: the near source draws pixels and picks; the far source draws nothing and does not pick
// until the globe is rotated to face it, at which point it draws and picks.
//
// Usage: node scripts/globe.mjs [url] [scenario ...]
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {cellToLatLng, latLngToCell} from 'h3-js';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const args = process.argv.slice(2);
const url = args.find(value => value.startsWith('http')) || 'http://127.0.0.1:5318/?rows=256';
const requested = args.filter(value => !value.startsWith('http'));
const NEAR = {longitude: -30, latitude: 20};
const FAR = {longitude: 150, latitude: -20};
const ZOOM = 1.2;
const origin = Date.UTC(2026, 0, 1);

const box = (center, half, altitude = 0) => [
  [center.longitude - half, center.latitude - half, altitude],
  [center.longitude + half, center.latitude - half, altitude],
  [center.longitude + half, center.latitude + half, altitude],
  [center.longitude - half, center.latitude + half, altitude],
  [center.longitude - half, center.latitude - half, altitude]
];
const feature = geometry => JSON.stringify({type: 'Feature', geometry, properties: {}});
const lineAround = center => ({
  type: 'LineString',
  coordinates: [
    [center.longitude - 6, center.latitude - 2],
    [center.longitude, center.latitude + 2],
    [center.longitude + 6, center.latitude - 2]
  ]
});
const tripAround = center =>
  JSON.stringify({
    type: 'LineString',
    coordinates: [
      [center.longitude - 6, center.latitude - 2, 0, origin],
      [center.longitude, center.latitude + 2, 0, origin + 10000],
      [center.longitude + 6, center.latitude - 2, 0, origin + 20000]
    ]
  });
const one = columns => arrow.tableFromArrays({...columns, measure: new Float32Array([0])});
const cellIdentity = center => BigInt(`0x${latLngToCell(center.latitude, center.longitude, 1)}`);

const svgIcons = [
  {id: 'diamond', mesh: {positions: [[0, -1, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]], cells: [[0, 1, 2], [0, 2, 3]]}}
];

const geometryFill = workspace => workspace.getByLabel('Source geometry column').selectOption('geometry');
const polygonTable = half => center => one({geometry: [feature({type: 'Polygon', coordinates: [box(center, half)]})]});
const cellTable = center => one({cell: arrow.vectorFromArray([cellIdentity(center)], new arrow.Uint64())});
const lineTip = center => ({longitude: center.longitude, latitude: center.latitude + 2});

/** Three nearby points, for scenarios whose layer needs data bounds or aggregation. */
const spreadTable = center => arrow.tableFromArrays({
  longitude: new Float32Array([center.longitude - 3, center.longitude, center.longitude + 3]),
  latitude: new Float32Array([center.latitude - 2, center.latitude + 1, center.latitude - 1]),
  measure: new Float32Array([1, 1, 1])
});
const spreadFill = async workspace => {
  await workspace.getByLabel('Source longitude column').selectOption('longitude');
  await workspace.getByLabel('Source latitude column').selectOption('latitude');
};
const aggregate = layerType => ({
  rowCount: 3,
  table: spreadTable,
  fill: spreadFill,
  layerType,
  columns: {lat: 'latitude', lng: 'longitude'},
  columnMode: 'points',
  zoom: 3,
  visConfig: {worldUnitSize: 150, coverage: 1, enable3d: false, percentile: [0, 100], opacity: 1, colorAggregation: 'count',
    colorRange: {colors: ['#ff14aa', '#ff14aa']}, clusterRadius: 40, radiusRange: [20, 40], minClusterSize: 1},
  pick: () => null
});

/** Scenario: one-row Arrow source per center, workspace form fill, native layer edit, and the pick. */
const scenarios = {
  polygon: {
    table: polygonTable(4),
    fill: geometryFill,
    layerType: 'geojson',
    visConfig: {filled: true, stroked: false, enable3d: false, opacity: 1},
    pick: center => center
  },
  // A 40 degree half-width polygon has edges far longer than Deck's 5 degree globe cut.
  'polygon-wide': {
    table: polygonTable(40),
    fill: geometryFill,
    layerType: 'geojson',
    visConfig: {filled: true, stroked: false, enable3d: false, opacity: 1},
    zoom: 1.2,
    minPixels: 6000,
    pick: center => center
  },
  // A ring that jumps from +170 to -170 longitude crosses the antimeridian as one polygon.
  'polygon-dateline': {
    centers: {near: {longitude: 180, latitude: 15}, far: {longitude: 0, latitude: -15}},
    table: center => one({geometry: [feature({type: 'Polygon', coordinates: [box(center, 12).map(([longitude, latitude, altitude]) => [longitude > 180 ? longitude - 360 : longitude, latitude, altitude])]})]}),
    fill: geometryFill,
    layerType: 'geojson',
    visConfig: {filled: true, stroked: false, enable3d: false, opacity: 1},
    minPixels: 800,
    pick: center => center
  },
  'path-long': {
    centers: {near: {longitude: 0, latitude: 0}, far: {longitude: 180, latitude: 0}},
    table: center => one({geometry: [feature({type: 'LineString', coordinates: [[center.longitude - 60, 50], [center.longitude + 60, 50]]})]}),
    fill: geometryFill,
    layerType: 'geojson',
    visConfig: {filled: false, stroked: true, thickness: 8, strokeOpacity: 1},
    zoom: 1.2,
    pick: () => null
  },
  'polygon-outline': {
    table: polygonTable(4),
    fill: geometryFill,
    layerType: 'geojson',
    visConfig: {filled: false, stroked: true, thickness: 6, enable3d: false, strokeOpacity: 1},
    pick: () => null
  },
  extrusion: {
    table: polygonTable(2),
    fill: geometryFill,
    layerType: 'geojson',
    visConfig: {filled: true, stroked: false, enable3d: true, fixedHeight: true, opacity: 1},
    pick: center => center
  },
  path: {
    table: center => one({geometry: [feature(lineAround(center))]}),
    fill: geometryFill,
    layerType: 'geojson',
    visConfig: {filled: true, stroked: true, thickness: 8, strokeOpacity: 1},
    pick: lineTip
  },
  geopoint: {
    table: center => one({geometry: [feature({type: 'Point', coordinates: [center.longitude, center.latitude]})]}),
    fill: geometryFill,
    layerType: 'geojson',
    visConfig: {filled: true, stroked: false, radius: 40, opacity: 1},
    pick: center => center
  },
  trips: {
    table: center => one({route: [tripAround(center)]}),
    fill: async workspace => {
      await workspace.getByLabel('Source geometry column').selectOption('route');
      await workspace.getByLabel('Timed paths (Trips)', {exact: true}).check();
      await workspace.getByLabel('Trip timestamp encoding', {exact: true}).selectOption('unix-millisecond');
      await workspace.getByLabel('Trip time origin (epoch milliseconds)', {exact: true}).fill(String(origin));
      await workspace.getByLabel('Maximum trip time error (milliseconds)', {exact: true}).fill('0.1');
    },
    ready: dataset => dataset.trips?.tripCount === 1,
    layerType: 'trip',
    visConfig: {thickness: 8, trailLength: 600, fadeTrail: false, opacity: 1},
    animationTime: origin + 20000,
    pick: lineTip
  },
  icon: {
    table: center => one({longitude: new Float32Array([center.longitude]), latitude: new Float32Array([center.latitude]), symbol: ['diamond']}),
    fill: async workspace => {
      await workspace.getByLabel('Source longitude column').selectOption('longitude');
      await workspace.getByLabel('Source latitude column').selectOption('latitude');
    },
    layerType: 'icon',
    columns: {lat: 'latitude', lng: 'longitude', icon: 'symbol'},
    zoom: 11,
    visConfig: {radius: 3000, fixedRadius: false, billboard: false, opacity: 1},
    pick: center => center
  },
  labels: {
    table: center => one({longitude: new Float32Array([center.longitude]), latitude: new Float32Array([center.latitude]), name: ['ALPHA']}),
    fill: async workspace => {
      await workspace.getByLabel('Source longitude column').selectOption('longitude');
      await workspace.getByLabel('Source latitude column').selectOption('latitude');
    },
    layerType: 'point',
    textLabel: [{field: null, color: [255, 20, 170], size: 48, weight: 400, anchor: 'middle', alignment: 'center', outlineWidth: 0, background: false, collisionEnabled: false}],
    visConfig: {radius: 1, filled: true, outline: false, opacity: 1},
    // The native label field is chosen through the layer panel, as in point-labels.mjs.
    afterApply: async (page, layerId) => {
      const labels = page.locator('.layer-config-group').filter({has: page.getByText('Label 1', {exact: true})}).first();
      if (!(await labels.isVisible())) await page.locator('.layer__enable-config svg').first().click();
      if ((await labels.getAttribute('class')).includes('collapsed')) await labels.locator('.layer-config-group__header').click();
      await labels.locator('.field-selector').last().click();
      await page.locator('.field-selector_list-item').filter({has: page.getByText('name', {exact: true})}).click();
      await page.keyboard.press('Escape');
      await page.waitForFunction(id => window.__keplerComputeTest.renderedLayers().some(layer => layer.id === id && layer.layerClass === 'GPUPointLabelLayer' && layer.loaded), layerId, {timeout: 30000});
    },
    pick: () => null
  },
  'icon-billboard': {
    table: center => one({longitude: new Float32Array([center.longitude]), latitude: new Float32Array([center.latitude]), symbol: ['diamond']}),
    fill: async workspace => {
      await workspace.getByLabel('Source longitude column').selectOption('longitude');
      await workspace.getByLabel('Source latitude column').selectOption('latitude');
    },
    layerType: 'icon',
    columns: {lat: 'latitude', lng: 'longitude', icon: 'symbol'},
    zoom: 11,
    visConfig: {radius: 300000, fixedRadius: false, billboard: true, opacity: 1},
    pick: center => center
  },
  // Native frames the density texture from the data bounds and drapes it over a sphere mesh.
  grid: aggregate('grid'),
  hexagon: aggregate('hexagon'),
  // Clusters are viewport-bound, so the far-side view legitimately renders no cluster layer.
  cluster: {...aggregate('cluster'), optionalWhenHidden: true},
  heatmap: {
    rowCount: 3,
    // Native fills the density frame from the data bounds, so blobs need a wide spread to be visible.
    table: center => arrow.tableFromArrays({
      longitude: new Float32Array([center.longitude - 25, center.longitude, center.longitude + 25]),
      latitude: new Float32Array([center.latitude - 12, center.latitude + 8, center.latitude - 4]),
      measure: new Float32Array([1, 1, 1])
    }),
    fill: spreadFill,
    layerType: 'heatmap',
    columns: {lat: 'latitude', lng: 'longitude'},
    columnMode: 'points',
    zoom: 2,
    visConfig: {opacity: 1, radius: 400, intensity: 3, threshold: 0.01, aggregation: 'SUM', colorRange: {colors: ['#ff14aa', '#ff14aa']}},
    pick: () => null
  },
  'icon-wide': {
    table: center => one({longitude: new Float32Array([center.longitude]), latitude: new Float32Array([center.latitude]), symbol: ['diamond']}),
    fill: async workspace => {
      await workspace.getByLabel('Source longitude column').selectOption('longitude');
      await workspace.getByLabel('Source latitude column').selectOption('latitude');
    },
    layerType: 'icon',
    columns: {lat: 'latitude', lng: 'longitude', icon: 'symbol'},
    zoom: 3,
    visConfig: {radius: 1500, fixedRadius: false, billboard: false, opacity: 1},
    pick: center => center
  },
  h3: {
    table: cellTable,
    fill: workspace => workspace.getByLabel('Source H3 column').selectOption('cell'),
    layerType: 'hexagonId',
    visConfig: {filled: true, outline: false, enable3d: false, coverage: 1, opacity: 1},
    pick: center => center
  },
  'h3-extruded': {
    table: cellTable,
    fill: workspace => workspace.getByLabel('Source H3 column').selectOption('cell'),
    layerType: 'hexagonId',
    visConfig: {filled: true, outline: false, enable3d: true, coverage: 0.8, opacity: 1, elevationScale: 5},
    pick: center => center
  }
};

/** Demo-dataset scenarios: connection and point layers, with native target arrowheads. */
const demoScenarios = {
  'demo-point': {type: 'point', arrow: false},
  'demo-line': {type: 'line', arrow: false},
  'demo-arc': {type: 'arc', arrow: false},
  'demo-line-arrow': {type: 'line', arrow: true},
  'demo-arc-arrow': {type: 'arc', arrow: true}
};
const DEMO_CENTER = {longitude: -73.985, latitude: 40.755};
const DEMO_ANTIPODE = {longitude: 106.015, latitude: -40.755};

// Probes that run only when named; everything else is part of the default acceptance set.
const UNQUALIFIED = new Set(['icon-wide']);
const names = requested.length ? requested : [...Object.keys(scenarios).filter(name => !UNQUALIFIED.has(name)), ...Object.keys(demoScenarios)];
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const results = [];
const failures = [];

function isMagenta(red, green, blue) {
  return red > 150 && green < 110 && blue > 90 && red - green > 90;
}

async function countPixels(page, name) {
  // The globe geometry variant is prepared in time slices; wait for any in-flight preparation.
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications.every(publication => publication.globeGeometry?.state !== 'preparing'), undefined, {timeout: 45000});
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const image = PNG.sync.read(await page.screenshot({path: `/tmp/luma-kepler-globe-${name}.png`}));
  let pixels = 0;
  for (let vertical = 60; vertical < image.height - 40; vertical++) {
    for (let horizontal = 340; horizontal < image.width - 40; horizontal++) {
      const offset = (vertical * image.width + horizontal) * 4;
      if (isMagenta(image.data[offset], image.data[offset + 1], image.data[offset + 2])) pixels++;
    }
  }
  return pixels;
}

async function applyConfig(page, config, layerId, requireLoaded = true, layerOptional = false) {
  const frame = (await page.evaluate(() => window.__keplerComputeTest.getState())).frameCount;
  await page.evaluate(value => window.__keplerComputeTest.applyNativeConfig(value), config);
  await page.waitForFunction(({id, frame, requireLoaded, layerOptional}) => {
    const api = window.__keplerComputeTest;
    const state = api.getState();
    // Native icon meshes load asynchronously and report that state as an error until ready.
    if (state.error && !/loading/i.test(state.error)) throw new Error(state.error);
    const native = state.nativeLayers.find(layer => layer.id === id);
    if (native?.valid === false) throw new Error(JSON.stringify(native));
    return state.frameCount > frame && (layerOptional || api.renderedLayers().some(layer => layer.id.startsWith(id) && layer.visible && (layer.loaded || !requireLoaded)));
  }, {id: layerId, frame, requireLoaded, layerOptional}, {timeout: 45000});
}

let stage = '';
/** One page load: publish the one-row source at `center`, then probe it from `views`. */
async function runScenario(name, role) {
  const scenario = scenarios[name];
  const near = scenario.centers?.near ?? NEAR;
  const far = scenario.centers?.far ?? FAR;
  const center = role === 'near' ? near : far;
  const label = `${name}/${role}`;
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  await pinViteModules(page);
  const errors = [];
  // Vite's blocked socket makes its console wrapper throw this; it is unrelated noise.
  page.on('pageerror', error => !/send was called before connect/.test(error.message) && errors.push(error.stack?.slice(0, 1500) ?? error.message));
  page.on('console', message => {
    const text = message.text();
    // Follow-on "invalid due to a previous error" lines hide the root cause; keep the first few.
    if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(text) && !/previous error/.test(text) && errors.length < 3) errors.push(text.slice(0, 1200));
    else if (process.env.GLOBE_DEBUG && ['error', 'warning'].includes(message.type()) && !/send was called/.test(text)) console.error('console', message.type(), text.slice(0, 600));
  });
  try {
    stage = 'publish';
    await page.route('**/icons/svg-icons.json', route => route.fulfill({json: {svgIcons}}));
    await page.goto(url, {waitUntil: 'domcontentloaded'});
    await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
    await openNativeData(page);
    const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
    await workspace.locator('input[type=file]').setInputFiles({name: `globe-${name}.arrow`, mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(scenario.table(center), 'stream'))});
    await scenario.fill(workspace);
    await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
    await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
    await page.waitForFunction(({ready, rowCount}) => {
      const dataset = window.__keplerComputeTest.getState().publications[0];
      return dataset?.selectedCount === rowCount && (!ready || Function(`return (${ready})`)()(dataset));
    }, {ready: scenario.ready ? String(scenario.ready) : null, rowCount: scenario.rowCount ?? 1}, {timeout: 45000});
    await closeNativeDialog(page);
    const datasetId = (await page.evaluate(() => window.__keplerComputeTest.getState())).publications[0].id;
    const config = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
    const layer = config.config.visState.layers.find(candidate => candidate.config.dataId === datasetId);
    for (const candidate of config.config.visState.layers) candidate.config.isVisible = candidate === layer;
    // The label field is chosen in the panel, whose first layer must be this one.
    if (scenario.afterApply) config.config.visState.layers = [layer];
    layer.type = scenario.layerType;
    if (scenario.columns) {
      layer.config.columnMode = scenario.columnMode ?? null;
      layer.config.columns = scenario.columns;
    }
    layer.config.color = [255, 20, 170];
    layer.config.textLabel = scenario.textLabel ?? [];
    layer.config.visConfig = {...layer.config.visConfig, ...scenario.visConfig, strokeColor: [255, 20, 170]};
    layer.visualChannels = {...layer.visualChannels, colorField: null, sizeField: null, heightField: null, coverageField: null, strokeColorField: null};
    const view = target => ({...config.config.mapState, ...target, zoom: scenario.zoom ?? ZOOM, pitch: 0, bearing: 0, dragRotate: false,
      mapViewMode: process.env.GLOBE_FLAT ? 'MODE_PLANAR' : 'MODE_GLOBE', globe: {...(config.config.mapState.globe ?? {}), enabled: !process.env.GLOBE_FLAT}});
    const pick = location => location && page.evaluate(({id, location}) => window.__keplerComputeTest.pickPublishedPosition(id, location.longitude, location.latitude), {id: datasetId, location});
    let applied = false;
    const probe = async (target, visible) => {
      stage = `${visible ? 'visible' : 'hidden'}-view`;
      // Later probes keep whatever the panel changed since the first apply.
      const current = applied ? await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig()) : config;
      current.config.mapState = view(target);
      await applyConfig(page, current, layer.id, !(scenario.afterApply && !applied), Boolean(scenario.optionalWhenHidden) && !visible);
      if (!applied && scenario.afterApply) await scenario.afterApply(page, layer.id);
      applied = true;
      if (scenario.animationTime) await page.evaluate(time => window.__keplerComputeTest.setAnimationTime(time), scenario.animationTime);
      const pixels = await countPixels(page, `${name}-${role}-${visible ? 'visible' : 'hidden'}`);
      if (process.env.GLOBE_DEBUG) console.error('pixels', label, visible, pixels);
      const picked = await pick(scenario.pick(center));
      if (process.env.GLOBE_DEBUG) console.error('picked', label, JSON.stringify(picked));
      if (visible) {
        assert(pixels > (scenario.minPixels ?? 150), `${label}: must draw on the globe (${pixels}) projected=${JSON.stringify(await page.evaluate(({longitude, latitude}) => window.__keplerComputeTest.projectPosition(longitude, latitude), center))} layers=${JSON.stringify(await page.evaluate(() => window.__keplerComputeTest.renderedLayers().map(layer => [layer.id, layer.layerClass, layer.visible, layer.loaded, layer.heatmapReadiness])))}`);
        if (scenario.pick(center)) assert.equal(picked?.sourceRowIndex, 0, `${label}: must pick`);
      } else {
        assert(pixels < 20, `${label}: far-side geometry must be culled (${pixels})`);
        if (scenario.pick(center)) assert.equal(picked, null, `${label}: far-side geometry must not pick`);
      }
      return pixels;
    };
    const globeEnabled = (config.config.mapState.globe ?? {}).enabled;
    const visibleTarget = role === 'near' ? near : far;
    const hiddenTarget = role === 'near' ? far : near;
    const visiblePixels = await probe(visibleTarget, true);
    if (!process.env.GLOBE_FLAT) assert.equal(await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig().config.mapState.globe?.enabled), true, 'Native map state must be a globe');
    const hiddenPixels = await probe(hiddenTarget, false);
    const state = await page.evaluate(() => window.__keplerComputeTest.getState());
    assert.equal(state.error, '', `${label}: ${state.error}`);
    assert.deepEqual(errors, []);
    results.push({name, role, visiblePixels, hiddenPixels});
  } catch (error) {
    failures.push({name, role, stage, message: error.message});
    console.error(label, stage, error.message, errors, await page.evaluate(() => JSON.stringify(window.__keplerComputeTest.getState().error)).catch(() => ''));
    await page.screenshot({path: `/tmp/luma-kepler-globe-${name}-${role}-failure.png`}).catch(() => {});
  } finally {
    await page.close();
  }
}

async function runDemoScenario(name) {
  const scenario = demoScenarios[name];
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  await pinViteModules(page);
  const errors = [];
  page.on('pageerror', error => !/send was called before connect/.test(error.message) && errors.push(error.stack?.slice(0, 1500) ?? error.message));
  page.on('console', message => {
    const text = message.text();
    if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(text) && !/previous error/.test(text) && errors.length < 3) errors.push(text.slice(0, 1200));
  });
  try {
    stage = 'load';
    await page.goto(url, {waitUntil: 'domcontentloaded'});
    await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
    const config = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
    const layer = config.config.visState.layers[0];
    layer.id = `resident-globe-${name}`;
    layer.type = scenario.type;
    if (scenario.type !== 'point') {
      layer.config.columnMode = 'points';
      layer.config.columns = {lat0: 'latitude', lng0: 'longitude', lat1: 'targetLatitude', lng1: 'targetLongitude'};
    }
    layer.config.color = [255, 20, 170];
    layer.config.colorField = null;
    layer.config.sizeField = null;
    layer.config.textLabel = [];
    layer.config.visConfig = {...layer.config.visConfig, thickness: 6, opacity: 1, targetColor: [255, 20, 170], targetArrowEnabled: scenario.arrow,
      targetArrowShape: 'triangle', targetArrowSize: 28, radius: 12, fixedRadius: true, filled: true, stroked: false, colorRange: {colors: ['#ff14aa', '#ff14aa']}};
    layer.visualChannels = {...layer.visualChannels, colorField: null, sizeField: null};
    const view = center => ({...config.config.mapState, ...center, zoom: 9, pitch: 0, bearing: 0, dragRotate: false,
      mapViewMode: 'MODE_GLOBE', globe: {...(config.config.mapState.globe ?? {}), enabled: true}});
    stage = 'near';
    config.config.mapState = view(DEMO_CENTER);
    await applyConfig(page, config, layer.id);
    const visiblePixels = await countPixels(page, `${name}-near`);
    assert(visiblePixels > 100, `${name}: must draw on the globe (${visiblePixels})`);
    stage = 'far';
    config.config.mapState = view(DEMO_ANTIPODE);
    await applyConfig(page, config, layer.id);
    const hiddenPixels = await countPixels(page, `${name}-far`);
    assert(hiddenPixels < 20, `${name}: far-side geometry must be culled (${hiddenPixels})`);
    const state = await page.evaluate(() => window.__keplerComputeTest.getState());
    assert.equal(state.error, '', `${name}: ${state.error}`);
    assert.deepEqual(errors, []);
    results.push({name, visiblePixels, hiddenPixels});
  } catch (error) {
    failures.push({name, stage, message: error.message});
    console.error(name, stage, error.message, errors);
    await page.screenshot({path: `/tmp/luma-kepler-globe-${name}-failure.png`}).catch(() => {});
  } finally {
    await page.close();
  }
}

for (const name of names) {
  if (demoScenarios[name]) {
    await runDemoScenario(name);
    continue;
  }
  assert(scenarios[name], `Unknown scenario ${name}`);
  // Published at NEAR: drawn and picked facing it, culled and unpickable facing the antipode.
  await runScenario(name, 'near');
}
await browser.close();
console.log(JSON.stringify({passed: failures.length === 0, results, failures}, null, 1));
if (failures.length) process.exit(1);
