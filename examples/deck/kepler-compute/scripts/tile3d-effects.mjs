// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser checks for the WebGPU-gated parts of 3D tiles, scenegraph, effects, editor dashes and
// basemap fallback. Needs a running dev server (not started here) and a WebGPU-capable browser.
// Checks are assertions about observable host state driven through window.__keplerComputeTest;
// steps that cannot be observed are reported as SKIP rather than guessed.
//
// Usage: node scripts/tile3d-effects.mjs [url]
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

// A 200 m quad (glTF Y-up) served as a one-tile 3D Tiles 1.1 tileset with glb content.
function makeQuadGlb() {
  const positions = new Float32Array([-100, 0, -100, 100, 0, -100, 100, 0, 100, -100, 0, 100]);
  const normals = new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]);
  const indices = new Uint16Array([0, 2, 1, 0, 3, 2]);
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
      {bufferView: 0, componentType: 5126, count: 4, type: 'VEC3', min: [-100, 0, -100], max: [100, 0, 100]},
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
const TILESET_CENTER = {longitude: -122.4, latitude: 37.8};
const degreesToRadians = degrees => (degrees * Math.PI) / 180;
const halfDegrees = 0.0015;
// Tile content is placed by the tile transform: east-north-up axes at the tileset center and the
// WGS84 ECEF position of that center (column-major). Without it the glb sits at the Earth's center.
function makeEastNorthUpTransform({longitude, latitude}) {
  const lambda = degreesToRadians(longitude);
  const phi = degreesToRadians(latitude);
  const equatorialRadius = 6378137;
  const eccentricitySquared = 0.00669437999014;
  const primeVerticalRadius = equatorialRadius / Math.sqrt(1 - eccentricitySquared * Math.sin(phi) ** 2);
  const east = [-Math.sin(lambda), Math.cos(lambda), 0];
  const north = [-Math.sin(phi) * Math.cos(lambda), -Math.sin(phi) * Math.sin(lambda), Math.cos(phi)];
  const up = [Math.cos(phi) * Math.cos(lambda), Math.cos(phi) * Math.sin(lambda), Math.sin(phi)];
  const origin = [
    primeVerticalRadius * Math.cos(phi) * Math.cos(lambda),
    primeVerticalRadius * Math.cos(phi) * Math.sin(lambda),
    primeVerticalRadius * (1 - eccentricitySquared) * Math.sin(phi)
  ];
  return [...east, 0, ...north, 0, ...up, 0, ...origin, 1];
}
const tilesetJson = {
  asset: {version: '1.0'},
  geometricError: 100,
  root: {
    transform: makeEastNorthUpTransform(TILESET_CENTER),
    boundingVolume: {
      region: [
        degreesToRadians(TILESET_CENTER.longitude - halfDegrees),
        degreesToRadians(TILESET_CENTER.latitude - halfDegrees),
        degreesToRadians(TILESET_CENTER.longitude + halfDegrees),
        degreesToRadians(TILESET_CENTER.latitude + halfDegrees),
        0,
        50
      ]
    },
    geometricError: 0,
    refine: 'ADD',
    content: {uri: 'tile.glb'}
  }
};

const url = process.argv.slice(2).find(value => value.startsWith('http')) || 'http://127.0.0.1:5318/?rows=256';

const browser = await chromium.launch(await getPlaywrightLaunchOptions({headless: true}));
try {
  const page = await browser.newPage({viewport: {width: 1280, height: 800}});
  await pinViteModules(page);
  const consoleErrors = [];
  page.on('console', message => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  await page.goto(url, {waitUntil: 'load'});
  await page.waitForFunction(() => Boolean(window.__keplerComputeTest), null, {timeout: 120000});
  // Kepler's compute instance must finish loading data before effect actions are accepted.
  await page.waitForFunction(() => { const api = window.__keplerComputeTest; return api.getState().selectedCount === api.source().length; }, null, {timeout: 120000});

  // Basemap: an unreachable external style must report an error and keep the local background.
  const failed = await page.evaluate(() =>
    window.__keplerComputeTest.setBasemapStyleUrl('http://127.0.0.1:9/missing-style.json')
  );
  assert.equal(failed.status, 'failed');
  assert.equal(failed.activeStyleUrl, null);
  assert.match(failed.errorMessage, /data is unchanged/);
  const restored = await page.evaluate(() => window.__keplerComputeTest.setBasemapStyleUrl(null));
  assert.equal(restored.status, 'local');
  console.log('PASS basemap failure keeps local background');

  // Canvas must still render after the failed style (data not lost).
  const canvasCount = await page.evaluate(() => document.querySelectorAll('canvas').length);
  assert.ok(canvasCount > 0, 'map canvas remains after failed style');

  // Shader errors from the editor dash extension or effects would surface as console errors.
  const shaderErrors = consoleErrors.filter(text => /wgsl|shader|pipeline/i.test(text));
  assert.deepEqual(shaderErrors, [], 'no WGSL/pipeline errors during load');
  console.log('PASS no shader errors on load');

  // Effects: distance fog, surface fog and Light and Shadow run through WGSL replacements.
  // Kepler allows only one fog effect, so distance fog is replaced by surface fog in a second phase.
  const effectTypes = ['distanceFog', 'lightAndShadow'];
  for (const type of effectTypes) {
    await page.evaluate(effectType => window.__keplerComputeTest.addNativeEffect(effectType), type);
  }
  const nativeEffects = await page.evaluate(() => window.__keplerComputeTest.getNativeEffects());
  assert.equal(nativeEffects.length >= effectTypes.length, true, 'Kepler effects were added');
  await page.waitForFunction(
    () => window.__keplerComputeTest.getEffectReplacements().length >= 3,
    null,
    {timeout: 30000}
  );
  let replacements = await page.evaluate(() => window.__keplerComputeTest.getEffectReplacements());
  assert.ok(replacements.some(item => item.kind === 'distance-fog'), 'distance fog replaced by WGSL');
  const distanceFogId = nativeEffects.find(item => item.type === 'distanceFog')?.id;
  await page.evaluate(id => window.__keplerComputeTest.removeNativeEffect(id), distanceFogId);
  await page.evaluate(() => window.__keplerComputeTest.addNativeEffect('surfaceFog'));
  await page.waitForFunction(
    () => window.__keplerComputeTest.getEffectReplacements().some(item => item.kind === 'surface-fog'),
    null,
    {timeout: 30000}
  );
  replacements = await page.evaluate(() => window.__keplerComputeTest.getEffectReplacements());
  assert.ok(replacements.some(item => item.kind === 'surface-fog'), 'surface fog replaced by WGSL');
  assert.ok(replacements.some(item => item.kind === 'lighting'), 'Light and Shadow lighting replaced');
  // Fog must actually draw frames (renderedFrameCount grows) rather than silently skipping.
  await page.waitForFunction(
    () => window.__keplerComputeTest.getEffectReplacements().some(item => item.renderedFrameCount > 0),
    null,
    {timeout: 30000}
  );
  const effectErrors = consoleErrors.filter(text => /wgsl|shader|pipeline|validation/i.test(text));
  assert.deepEqual(effectErrors, [], 'WGSL fog and lighting compile without GPU validation errors');
  assert.ok(replacements.some(item => item.kind === 'shadow'), 'screen-space WGSL shadow effect added');
  assert.ok(
    !consoleErrors.some(text => /shadow maps are not rendered/i.test(text)),
    'no shadow gap message once the WGSL shadow pass exists'
  );
  console.log('PASS WGSL distance fog, surface fog and lighting and shadow effects render');

  // Basemap: a validated external style appears in Kepler's picker with visible attribution.
  const styleUrl = 'http://127.0.0.1:5999/basemap-test-style.json';
  await page.route(styleUrl, route =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      headers: {'access-control-allow-origin': '*'},
      body: JSON.stringify({
        version: 8,
        name: 'Test Basemap',
        sources: {
          test: {
            type: 'geojson',
            data: {type: 'FeatureCollection', features: []},
            attribution: '<a href="https://example.com">&copy; Test Basemap Contributors</a>'
          }
        },
        layers: [{id: 'background', type: 'background', paint: {'background-color': '#223344'}}]
      })
    })
  );
  const added = await page.evaluate(address => window.__keplerComputeTest.addBasemapStyle(address, 'Test Basemap'), styleUrl);
  assert.equal(added.state.status, 'external');
  assert.ok(added.styleId, 'validated style got a picker id');
  assert.equal(added.activeStyleType, added.styleId, 'Kepler selected the new style');
  assert.ok(added.pickerStyleIds.includes(added.styleId), 'style is listed in the Kepler map-style picker');
  await page.waitForFunction(
    () => /Test Basemap Contributors/.test(window.__keplerComputeTest.getBasemapAttributionText() ?? ''),
    null,
    {timeout: 10000}
  );
  const attributionText = await page.evaluate(() => document.getElementById('basemap-attribution')?.textContent);
  assert.doesNotMatch(attributionText, /</, 'attribution is plain text');
  assert.ok(await page.locator('#basemap-attribution').isVisible(), 'attribution is visible in the UI');
  console.log('PASS validated basemap style listed in picker with visible attribution');

  // 3D tiles: transforms, picking and resource cleanup.
  const glb = makeQuadGlb();
  // Serve the synthetic tileset with CORS (including preflight and HEAD) so loaders.gl fetches succeed.
  const corsHeaders = {
    'access-control-allow-origin': '*',
    'access-control-allow-methods': 'GET, HEAD, OPTIONS',
    'access-control-allow-headers': '*',
    'access-control-expose-headers': '*',
    'cache-control': 'no-store'
  };
  const tilesetBody = JSON.stringify(tilesetJson);
  await page.route('http://127.0.0.1:5999/tiles/**', route => {
    const request = route.request();
    const requestUrl = new URL(request.url());
    if (request.method() === 'OPTIONS') return route.fulfill({status: 204, headers: corsHeaders});
    if (requestUrl.pathname.endsWith('/tileset.json'))
      return route.fulfill({status: 200, contentType: 'application/json', headers: corsHeaders, body: tilesetBody});
    if (requestUrl.pathname.endsWith('/tile.glb'))
      return route.fulfill({status: 200, contentType: 'model/gltf-binary', headers: corsHeaders, body: glb});
    return route.fulfill({status: 404, headers: corsHeaders, body: 'not found'});
  });
  const resourcesBefore = await page.evaluate(() => window.__keplerComputeTest.getDeviceResourceCounts());
  await page.evaluate(
    ({view}) => window.__keplerComputeTest.setNativeMapView({...view, zoom: 17, pitch: 0}),
    {view: TILESET_CENTER}
  );
  await page.evaluate(() =>
    window.__keplerComputeTest.addNativeTile3DDataset('tile3d-check', 'http://127.0.0.1:5999/tiles/tileset.json')
  );
  let tileLayers = [];
  try {
    await page.waitForFunction(
      () => {
        const layers = window.__keplerComputeTest.getTile3DDiagnostics();
        return layers.some(layer => layer.tileset && layer.tileset.tileCount > 0) && layers.some(layer => layer.modelMatrix);
      },
      null,
      {timeout: 60000}
    );
    tileLayers = await page.evaluate(() => window.__keplerComputeTest.getTile3DDiagnostics());
  } catch (error) {
    const diagnostics = await page.evaluate(() => ({
      layers: window.__keplerComputeTest.getTile3DDiagnostics(),
      nativeLayers: window.__keplerComputeTest.getState().nativeLayers,
      hostError: window.__keplerComputeTest.getState().error
    }));
    diagnostics.consoleErrors = consoleErrors.slice(-6).map(text => text.slice(0, 300));
    throw new Error(`3D tileset did not load: ${JSON.stringify(diagnostics)} (${error.message})`);
  }
  const tilesetLayer = tileLayers.find(layer => layer.tileset);
  assert.ok(tilesetLayer.tileset.tileCount >= 1, 'tileset reported at least one tile');
  const transformed = tileLayers.filter(layer => layer.modelMatrix);
  assert.ok(transformed.length > 0, 'a tile sublayer carries a model matrix');
  for (const layer of transformed) {
    assert.equal(layer.modelMatrix.allFinite, true, `${layer.id} matrix is finite`);
    assert.ok(layer.modelMatrix.scaleDeterminant > 0, `${layer.id} matrix is not degenerate`);
  }
  assert.ok(
    transformed.some(layer => !layer.modelMatrix.isIdentity),
    'tile transform places the tile (non-identity model matrix)'
  );
  console.log('PASS 3D tile transforms are finite, non-degenerate and non-identity');

  const picked = await page.evaluate(() => window.__keplerComputeTest.pickTile3D());
  // tile3d-picking.ts clones Kepler's Tile3D layer with pickable: true, so picking works outside annotation mode.
  assert.ok(
    tileLayers.some(layer => layer.pickable),
    'Tile3D layer is pickable outside annotation mode'
  );
  assert.equal(
    picked.hit,
    true,
    `pickable tile sublayer returns a hit at the tileset center ${JSON.stringify(consoleErrors.slice(-40).map(text => text.slice(0, 330)))}`
  );
  assert.ok(picked.layerId, 'pick names the tile sublayer');
  console.log('PASS 3D tile picking returns a hit', JSON.stringify(picked));

  const resourcesLoaded = await page.evaluate(() => window.__keplerComputeTest.getDeviceResourceCounts());
  const tileNativeLayers = await page.evaluate(() =>
    window.__keplerComputeTest.getState().nativeLayers.filter(layer => layer.type === 'tile3d').map(layer => layer.id)
  );
  assert.ok(tileNativeLayers.length > 0, 'Kepler created a tile3d layer');
  for (const layerId of tileNativeLayers) {
    await page.evaluate(id => window.__keplerComputeTest.removeNativeLayer(id), layerId);
  }
  await page.waitForFunction(
    () => window.__keplerComputeTest.getTile3DDiagnostics().length === 0,
    null,
    {timeout: 30000}
  );
  await page.waitForTimeout(500);
  const resourcesAfter = await page.evaluate(() => window.__keplerComputeTest.getDeviceResourceCounts());
  assert.ok(
    resourcesAfter.Buffers < resourcesLoaded.Buffers || resourcesAfter.Textures < resourcesLoaded.Textures,
    `GPU resources drop after tile layer removal (loaded ${JSON.stringify(resourcesLoaded)}, after ${JSON.stringify(resourcesAfter)})`
  );
  assert.ok(
    resourcesAfter.Buffers <= resourcesBefore.Buffers + 8,
    `no resource leak beyond baseline (before ${JSON.stringify(resourcesBefore)}, after ${JSON.stringify(resourcesAfter)})`
  );
  console.log('PASS 3D tile layer removal releases GPU resources');

  console.log('SKIP editor dashed stroke and shadow-map rendering need Kepler UI driving / GLSL parity');
} finally {
  await browser.close();
}
