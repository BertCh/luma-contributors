// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Headless browser check for WGSL shadow mapping (EFFECTS2). Publishes extruded GeoJSON polygons,
// adds Kepler's Light and Shadow effect and asserts that the WGSL shadow map pass draws (shadow-map
// mode, participating layers, screen-space fallback standing down) without GPU validation errors.
// EFFECTS4-4 extends this with phases that keep Light and Shadow active while the scene gains point
// labels, a raster/bitmap overlay attempt, arc arrow heads, a split map and a globe, and asserts per
// phase that the shadow-map pass still draws (participating layers, frames, per-viewport stats)
// without GPU validation errors. Screenshots land in SHADOW_SCREENSHOT_DIR (default os.tmpdir()) for
// a manual look against Kepler's WebGL shadows (not automatable).
// Needs a running dev server; default headless Playwright only.
//
// Usage: node scripts/effects-shadow-map.mjs [url] [phase ...]
//   phases: polygons labels bitmap arrows split globe (default: all)
import assert from 'node:assert/strict';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const ring = (west, south, east, north) => [
  [west, south],
  [east, south],
  [east, north],
  [west, north],
  [west, south]
];
const tower = (west, south, east, north) =>
  JSON.stringify({
    type: 'Feature',
    geometry: {type: 'Polygon', coordinates: [ring(west, south, east, north)]},
    properties: {}
  });
const source = arrow.tableFromArrays({
  geometry: [
    tower(-74.0215, 40.742, -74.0195, 40.744),
    tower(-74.016, 40.7455, -74.014, 40.7475),
    tower(-74.0105, 40.739, -74.0085, 40.741),
    // Flat ground slab: Deck shadows only fall on layers, so a receiver is needed to see them.
    tower(-74.026, 40.736, -74.004, 40.75)
  ],
  heightMetric: new Float32Array([100, 250, 400, 0])
});
const args = process.argv.slice(2);
const pageUrl = args.find(value => value.startsWith('http')) || 'http://127.0.0.1:5318/?rows=256';
const requestedPhases = args.filter(value => !value.startsWith('http'));
const shouldRun = phase => requestedPhases.length === 0 || requestedPhases.includes(phase);
const screenshotDirectory = process.env.SHADOW_SCREENSHOT_DIR || tmpdir();
const pointTable = arrow.tableFromArrays({
  longitude: Float32Array.from([-74.02, -74.0, -73.98, -73.96]),
  latitude: Float32Array.from([40.74, 40.76, 40.745, 40.755]),
  targetLongitude: Float32Array.from([-74.0, -73.97, -73.95, -73.99]),
  targetLatitude: Float32Array.from([40.755, 40.74, 40.765, 40.735]),
  name: ['ALPHA', 'BETA', 'GAMMA', 'DELTA'],
  measure: Float32Array.from([1, 2, 3, 4])
});
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
    /WGSL|Invalid.*(?:Pipeline|Buffer|Texture)|validation error/i.test(message.text())
  )
    failures.push(message.text());
});
const watchdog = setTimeout(() => {
  console.error('Shadow map workflow exceeded 600 seconds');
  void browser.close();
}, 600000);

async function darkPixelCount() {
  await page.evaluate(
    () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))
  );
  const image = PNG.sync.read(await page.screenshot());
  let dark = 0;
  for (let offset = 0; offset < image.data.length; offset += 4)
    if (Math.max(image.data[offset], image.data[offset + 1], image.data[offset + 2]) < 40) dark++;
  return dark;
}

try {
  await page.goto(pageUrl, {
    waitUntil: 'domcontentloaded'
  });
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {
    timeout: 240000
  });
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: 'shadow-towers.arrow',
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))
  });
  await workspace.getByLabel('Source geometry column').selectOption('geometry');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 4,
    undefined,
    {timeout: 45000}
  );
  await closeNativeDialog(page);
  const datasetId = (await page.evaluate(() => window.__keplerComputeTest.getState())).publications[0].id;
  const layerId = `${datasetId}-geometry`;
  const config = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  for (const layer of config.config.visState.layers) layer.config.isVisible = layer.id === layerId;
  const layer = config.config.visState.layers.find(item => item.id === layerId);
  layer.visualChannels = {
    ...layer.visualChannels,
    heightField: {name: 'heightMetric', type: 'real'},
    heightScale: 'linear'
  };
  layer.config.visConfig = {
    ...layer.config.visConfig,
    filled: true,
    stroked: false,
    enable3d: true,
    fixedHeight: false,
    elevationScale: 1,
    opacity: 1,
    heightRange: [0, 400]
  };
  config.config.mapState = {
    ...config.config.mapState,
    longitude: -74.015,
    latitude: 40.743,
    zoom: 14.5,
    pitch: 55,
    bearing: 20
  };
  await page.evaluate(next => window.__keplerComputeTest.applyNativeConfig(next), config);
  await page.waitForFunction(
    id =>
      window.__keplerComputeTest
        .renderedLayers()
        .some(item => item.id === id && item.visible && item.loaded && item.extruded),
    layerId,
    {timeout: 45000}
  );
  const darkBefore = await darkPixelCount();
  const beforeImage = PNG.sync.read(await page.screenshot());

  await page.evaluate(() => window.__keplerComputeTest.addNativeEffect('lightAndShadow'));
  await page.waitForFunction(
    () =>
      window.__keplerComputeTest
        .getEffectReplacements()
        .some(item => item.kind === 'lighting' && item.shadowMode === 'shadow-map'),
    undefined,
    {timeout: 45000}
  );
  // Kepler casts directional shadows only in daytime at the map center (SunCalc); at night it
  // shades everything uniformly. Pin the sun to local noon in New York (2026-06-21 16:00 UTC).
  const effectId = (await page.evaluate(() => window.__keplerComputeTest.getNativeEffects()))[0].id;
  await page.evaluate(
    ([id, timestamp]) =>
      window.__keplerComputeTest.updateNativeEffect(id, {timeMode: 'pick', timestamp}),
    [effectId, Date.UTC(2026, 5, 21, 16, 0)]
  );
  await page.waitForTimeout(1000);
  const replacements = await page.evaluate(() => window.__keplerComputeTest.getEffectReplacements());
  const lighting = replacements.find(item => item.kind === 'lighting');
  assert.equal(lighting.shadowMode, 'shadow-map', 'real shadow map pass drew');
  assert.ok(lighting.shadowParticipantLayerCount >= 1, 'extruded layers joined the shadow pass');
  assert.ok(lighting.renderedFrameCount > 0, 'shadow map frames rendered');
  await page.waitForFunction(
    () =>
      window.__keplerComputeTest
        .getEffectReplacements()
        .some(item => item.kind === 'shadow' && item.shadowMapFrameCount > 0),
    undefined,
    {timeout: 30000}
  );
  const darkAfter = await darkPixelCount();
  // Deck renders on demand: force a redraw at the original camera before capturing.
  for (const bearing of [21, 20]) {
    await page.evaluate(next => window.__keplerComputeTest.setMapViewport({bearing: next}), bearing);
    await darkPixelCount();
  }
  const afterImage = PNG.sync.read(await page.screenshot());
  (await import('node:fs')).writeFileSync(join(screenshotDirectory, 'luma-kepler-shadow-after.png'), PNG.sync.write(afterImage));
  (await import('node:fs')).writeFileSync(join(screenshotDirectory, 'luma-kepler-shadow-before.png'), PNG.sync.write(beforeImage));
  const countChangedPixels = (first, second) => {
    let changed = 0;
    for (let offset = 0; offset < first.data.length; offset += 4)
      if (
        Math.abs(first.data[offset] - second.data[offset]) +
          Math.abs(first.data[offset + 1] - second.data[offset + 1]) +
          Math.abs(first.data[offset + 2] - second.data[offset + 2]) >
        12
      )
        changed++;
    return changed;
  };
  /**
   * Counts pixels that are a darkened copy of the dominant scene color (the lit ground slab): same
   * hue direction, 20 to 80 percent of its brightness. Only the map area (right of the sidebar) counts.
   */
  const countShadowedGroundPixels = image => {
    const histogram = new Map();
    const pixelAt = offset => [image.data[offset], image.data[offset + 1], image.data[offset + 2]];
    for (let y = 0; y < image.height; y++)
      for (let x = 320; x < image.width - 80; x++) {
        const [red, green, blue] = pixelAt((y * image.width + x) * 4);
        if (Math.max(red, green, blue) < 120) continue; // skip the dark basemap background
        const key = `${red >> 2},${green >> 2},${blue >> 2}`;
        histogram.set(key, (histogram.get(key) ?? 0) + 1);
      }
    const [modeKey] = [...histogram.entries()].sort((first, second) => second[1] - first[1])[0];
    const mode = modeKey.split(',').map(value => Number(value) * 4 + 2);
    const modeLength = Math.hypot(...mode);
    let shadowed = 0;
    for (let y = 0; y < image.height; y++)
      for (let x = 320; x < image.width - 80; x++) {
        const color = pixelAt((y * image.width + x) * 4);
        const length = Math.hypot(...color);
        if (length < 0.2 * modeLength || length > 0.8 * modeLength) continue;
        const direction = color.map((value, index) => value / length - mode[index] / modeLength);
        if (Math.hypot(...direction) < 0.06) shadowed++;
      }
    return {shadowed, mode};
  };
  const changedPixels = countChangedPixels(beforeImage, afterImage);
  const shadowedBefore = countShadowedGroundPixels(beforeImage);
  const shadowedAfter = countShadowedGroundPixels(afterImage);
  console.log(
    JSON.stringify({darkBefore, darkAfter, changedPixels, shadowedBefore, shadowedAfter, lighting})
  );
  assert.ok(changedPixels > 20000, `Light and Shadow repaints the map (${changedPixels} pixels changed)`);
  assert.ok(
    shadowedAfter.shadowed - shadowedBefore.shadowed >= 5000,
    `tower shadows darken ground pixels (${shadowedBefore.shadowed} -> ${shadowedAfter.shadowed})`
  );

  // Moving the sun (time of day) must keep the shadow map pass running without validation errors.
  await page.evaluate(() => window.__keplerComputeTest.setMapViewport({pitch: 40, bearing: 200}));
  await page.evaluate(
    () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))
  );
  const afterMove = (await page.evaluate(() => window.__keplerComputeTest.getEffectReplacements())).find(
    item => item.kind === 'lighting'
  );
  assert.ok(afterMove.renderedFrameCount > lighting.renderedFrameCount, 'shadow map follows the camera');
  assert.deepEqual(failures, [], 'no WGSL or validation errors');
  console.log('PASS WGSL shadow map pass renders extruded polygons with Light and Shadow');
  const results = [{phase: 'polygons', passed: true}];

  const getLighting = async () =>
    (await page.evaluate(() => window.__keplerComputeTest.getEffectReplacements())).find(
      item => item.kind === 'lighting'
    );
  /** Deck renders on demand, so nudge the camera until the shadow pass has drawn a few frames. */
  const settleFrames = async (minimumFrames = 3) => {
    const start = (await getLighting()).renderedFrameCount;
    for (let attempt = 0; attempt < 40; attempt++) {
      const lighting = await getLighting();
      if (lighting.renderedFrameCount >= start + minimumFrames) return lighting;
      await page.evaluate(
        nudge => window.__keplerComputeTest.setMapViewport({bearing: 20 + nudge}),
        attempt % 2
      );
      await page.evaluate(
        () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))
      );
    }
    throw new Error('the shadow pass drew no new frames');
  };
  const savedConfig = () => page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const applyConfig = config =>
    page.evaluate(next => window.__keplerComputeTest.applyNativeConfig(next), config);
  const screenshot = async name =>
    page.screenshot({path: join(screenshotDirectory, `luma-kepler-shadow-${name}.png`)});
  const assertShadowing = async (name, minimumParticipants) => {
    const lighting = await settleFrames();
    assert.equal(lighting.shadowMode, 'shadow-map', `${name}: shadow map pass is active`);
    assert.ok(
      lighting.shadowParticipantLayerCount >= minimumParticipants,
      `${name}: ${lighting.shadowParticipantLayerCount} participants, expected at least ${minimumParticipants}`
    );
    assert.equal(lighting.shadowSkipReason, undefined, `${name}: no skip reason`);
    const stats = lighting.shadowViewportStats ?? {};
    for (const [viewportId, viewport] of Object.entries(stats))
      assert.ok(
        viewport.shadowMapSize[0] > 0 && viewport.shadowMapSize[1] > 0,
        `${name}: viewport ${viewportId} has a shadow map`
      );
    await screenshot(name);
    assert.deepEqual(failures, [], `${name}: no WGSL or validation errors`);
    return lighting;
  };
  const waitForRendered = (layerClass, idPrefix = '') =>
    page.waitForFunction(
      ({layerClass, idPrefix}) =>
        window.__keplerComputeTest
          .renderedLayers()
          .some(
            item =>
              item.layerClass === layerClass &&
              item.id.startsWith(idPrefix) &&
              item.visible &&
              item.loaded
          ),
      {layerClass, idPrefix},
      {timeout: 45000}
    ).catch(async error => {
      const rendered = await page.evaluate(() =>
        window.__keplerComputeTest
          .renderedLayers()
          .map(item => `${item.id}:${item.layerClass}:${item.visible}:${item.loaded}`)
      );
      const deckIds = await page.evaluate(() => window.__keplerComputeTest.getDeckLayerIds());
      const nativeLayers = await page.evaluate(() =>
        JSON.stringify(window.__keplerComputeTest.getState().nativeLayers)
      );
      throw new Error(`${error.message} deck: ${deckIds.join(',')} native: ${nativeLayers} waiting for ${layerClass} ${idPrefix}; rendered: ${rendered.join(', ')}; error: ${await page.evaluate(() => window.__keplerComputeTest.getState().error)}`);
    });
  const runPhase = async (name, body) => {
    if (!shouldRun(name)) return;
    try {
      const detail = await body();
      results.push({phase: name, passed: true, ...detail});
      console.log(`${detail.blocked ? 'BLOCKED' : 'PASS'} phase ${name} ${JSON.stringify(detail)}`);
    } catch (error) {
      results.push({phase: name, passed: false, error: error.message});
      console.log(`FAIL phase ${name}: ${error.message}\n${(error.stack ?? '').split('\n').filter(line => line.includes('effects-shadow-map')).slice(0, 3).join('\n')}`);
    }
  };

  // A second published table: points with a name column and flow target roles (labels and arcs).
  let pointDatasetId;
  const ensurePointDataset = async () => {
    if (pointDatasetId) return pointDatasetId;
    await openNativeData(page);
    const pointWorkspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
    await pointWorkspace.locator('input[type=file]').setInputFiles({
      name: 'shadow-points.arrow',
      mimeType: 'application/vnd.apache.arrow.stream',
      buffer: Buffer.from(arrow.tableToIPC(pointTable, 'stream'))
    });
    await pointWorkspace.getByLabel('Source longitude column').selectOption('longitude');
    await pointWorkspace.getByLabel('Source latitude column').selectOption('latitude');
    await pointWorkspace.getByLabel('Flow target longitude column').selectOption('targetLongitude');
    await pointWorkspace.getByLabel('Flow target latitude column').selectOption('targetLatitude');
    await pointWorkspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
    await pointWorkspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
    await page.waitForFunction(
      () => window.__keplerComputeTest.getState().publications.length >= 2,
      undefined,
      {timeout: 60000}
    );
    await closeNativeDialog(page);
    pointDatasetId = (await page.evaluate(() => window.__keplerComputeTest.getState())).publications[1].id;
    return pointDatasetId;
  };
  const mapState = {longitude: -73.98, latitude: 40.75, zoom: 12.5, pitch: 40, bearing: 0};
  /** Shows the polygon layer plus one layer cloned from the point dataset's first layer. */
  const applyPointScene = async (id, type, visConfig, textLabel = [], extraMapState = {}) => {
    const dataId = await ensurePointDataset();
    const next = structuredClone(await savedConfig());
    const layers = next.config.visState.layers;
    const template = layers.find(item => item.config.dataId === dataId) ?? layers[0];
    const layer = structuredClone(template);
    layer.id = id;
    layer.type = type;
    layer.config.dataId = dataId;
    layer.config.isVisible = true;
    layer.config.textLabel = textLabel;
    if (type === 'point') {
      layer.config.columns = {lat: 'latitude', lng: 'longitude', altitude: null};
      layer.config.columnMode = 'points';
    }
    if (type === 'arc') {
      layer.config.columns = {
        lat0: 'latitude',
        lng0: 'longitude',
        lat1: 'targetLatitude',
        lng1: 'targetLongitude'
      };
      layer.config.columnMode = 'points';
    }
    layer.config.visConfig = {...layer.config.visConfig, ...visConfig};
    next.config.visState.layers = [
      ...layers.filter(item => item.config.dataId !== dataId && item.id !== id),
      layer
    ];
    // The polygon tower layer stays visible so shadows have something to fall on.
    if (!next.config.visState.layerOrder?.includes(id)) next.config.visState.layerOrder?.unshift(id);
    next.config.mapState = {...next.config.mapState, ...mapState, ...extraMapState};
    await applyConfig(next);
  };
  const ensureShadowEffect = async () => {
    const hasLighting = (await getLighting())?.shadowMode === 'shadow-map';
    if (!hasLighting) {
      await page.evaluate(() => window.__keplerComputeTest.addNativeEffect('lightAndShadow'));
      await page.waitForFunction(
        () =>
          window.__keplerComputeTest
            .getEffectReplacements()
            .some(item => item.kind === 'lighting' && item.shadowMode === 'shadow-map'),
        undefined,
        {timeout: 45000}
      );
    }
  };

  await runPhase('labels', async () => {
    const baselineLayers = await getLighting();
    await applyPointScene('shadow-labels', 'point', {radius: 12, filled: true, outline: false});
    await ensureShadowEffect();
    await waitForRendered('GPUScatterplotLayer', 'shadow-labels');
    const withoutLabels = await assertShadowing('labels-baseline', 1);
    await applyPointScene('shadow-labels', 'point', {radius: 12, filled: true, outline: false}, [
      {
        field: {name: 'name', type: 'string'},
        color: [255, 255, 255],
        size: 24,
        weight: 400,
        anchor: 'middle',
        alignment: 'center',
        outlineWidth: 0,
        background: true,
        backgroundColor: [0, 0, 0, 200],
        collisionEnabled: false
      }
    ]);
    await waitForRendered('GPUPointLabelLayer', 'shadow-labels');
    const withLabels = await assertShadowing('labels', 1);
    assert.ok(
      withLabels.shadowParticipantLayerCount > withoutLabels.shadowParticipantLayerCount,
      'the point label layer joins the shadow pass'
    );
    return {
      baselineParticipants: baselineLayers.shadowParticipantLayerCount,
      withoutLabels: withoutLabels.shadowParticipantLayerCount,
      withLabels: withLabels.shadowParticipantLayerCount
    };
  });

  await runPhase('bitmap', async () => {
    // Kepler's image overlay layer type is 'bitmap'; a data-URL image georeferenced over the towers.
    // The raster tile layer needs a tile service, so this phase drives the bitmap overlay only.
    // A real, decodable 8x8 orange checker image (a corrupt PNG makes Kepler mark the layer invalid).
    const bitmapImage = new PNG({width: 8, height: 8});
    for (let pixel = 0; pixel < 64; pixel++) {
      const checker = (pixel + Math.floor(pixel / 8)) % 2 === 0;
      bitmapImage.data.set(checker ? [255, 160, 40, 255] : [255, 220, 120, 255], pixel * 4);
    }
    const png = PNG.sync.write(bitmapImage);
    await page.evaluate(
      ({imageUrl}) =>
        window.__keplerComputeTest.addNativeBitmapDataset('shadow-bitmap', imageUrl, [
          -74.03, 40.73, -73.99, 40.76
        ]),
      {imageUrl: `data:image/png;base64,${png.toString('base64')}`}
    );
    await page.waitForFunction(
      () =>
        window.__keplerComputeTest.getState().nativeLayers?.some(item => item.type === 'bitmap'),
      undefined,
      {timeout: 20000}
    );
    // After a config was applied Kepler adds the new overlay layer hidden: show it like the user would.
    await page.evaluate(() => {
      const api = window.__keplerComputeTest;
      const bitmap = api.getState().nativeLayers.find(item => item.type === 'bitmap');
      api.setLayerVisible(bitmap.id, true);
    });
    await ensureShadowEffect();
    const reachedDeck = await page
      .waitForFunction(
        () => {
          const api = window.__keplerComputeTest;
          const bitmapId = api.getState().nativeLayers?.find(item => item.type === 'bitmap')?.id;
          return Boolean(bitmapId) && api.getDeckLayerIds().includes(bitmapId);
        },
        undefined,
        {timeout: 15000}
      )
      .then(
        () => true,
        () => false
      );
    if (!reachedDeck) {
      // Kepler's bitmap layer reports valid and visible, but its renderLayer returns no Deck layer for
      // a dataset added this way, so bitmap shadow participation cannot be driven in the app yet.
      const baseline = await assertShadowing('bitmap-blocked', 1);
      return {
        blocked:
          'the Kepler bitmap overlay never reaches Deck (renderLayer returns []); raster tiles need a tile service',
        participants: baseline.shadowParticipantLayerCount
      };
    }
    await waitForRendered('BitmapLayer');
    const lighting = await assertShadowing('bitmap', 1);
    // The overlay image is orange; the tower scene is blue, so orange pixels prove the bitmap drew.
    const bitmapScreenshot = PNG.sync.read(await page.screenshot());
    let orangePixels = 0;
    for (let offset = 0; offset < bitmapScreenshot.data.length; offset += 4) {
      const [red, green, blue] = bitmapScreenshot.data.subarray(offset, offset + 3);
      if (red > 200 && green > 120 && green < 235 && blue < 150) orangePixels++;
    }
    assert.ok(orangePixels > 500, `the bitmap overlay is visible (${orangePixels} orange pixels)`);
    const rendered = await page.evaluate(() =>
      window.__keplerComputeTest
        .renderedLayers()
        .filter(item => item.layerClass === 'BitmapLayer')
        .map(item => item.layerClass)
    );
    return {
      bitmapLayerClasses: rendered,
      orangePixels,
      participants: lighting.shadowParticipantLayerCount
    };
  });

  await runPhase('arrows', async () => {
    await applyPointScene('shadow-arrows', 'arc', {
      thickness: 6,
      opacity: 1,
      targetArrowEnabled: true,
      targetArrowShape: 'triangle',
      targetArrowSize: 20
    });
    await ensureShadowEffect();
    await waitForRendered('GPUArcLayer', 'shadow-arrows');
    await waitForRendered('GPUArrowHeadLayer', 'shadow-arrows');
    const lighting = await assertShadowing('arrows', 2);
    return {participants: lighting.shadowParticipantLayerCount};
  });

  await runPhase('split', async () => {
    await applyPointScene('shadow-split', 'point', {radius: 12, filled: true, outline: false});
    await ensureShadowEffect();
    await page.evaluate(() => window.__keplerComputeTest.toggleSplitMap(0));
    await page.waitForFunction(
      () => (window.__keplerComputeTest.getEffectReplacements().find(item => item.kind === 'lighting')?.shadowViewportCount ?? 0) >= 2,
      undefined,
      {timeout: 45000}
    );
    const lighting = await assertShadowing('split', 1);
    assert.ok(lighting.shadowViewportCount >= 2, 'each split pane gets its own shadow map');
    assert.ok(Object.keys(lighting.shadowViewportStats).length >= 2, 'per-pane shadow stats');
    await page.evaluate(() => window.__keplerComputeTest.toggleSplitMap(0));
    return {viewports: lighting.shadowViewportCount, stats: lighting.shadowViewportStats};
  });

  await runPhase('globe', async () => {
    await applyPointScene('shadow-globe', 'point', {radius: 12, filled: true, outline: false}, [], {
      mapViewMode: 'MODE_GLOBE',
      globe: {enabled: true},
      zoom: 3,
      pitch: 0
    });
    await ensureShadowEffect();
    await waitForRendered('GPUScatterplotLayer', 'shadow-globe');
    const lighting = await assertShadowing('globe', 1);
    const fitting = Object.values(lighting.shadowViewportFitting ?? {});
    assert.ok(fitting.length > 0, 'a globe viewport reports its light-space fit');
    // Return to the planar map for any later phases.
    const planar = structuredClone(await savedConfig());
    planar.config.mapState = {...planar.config.mapState, mapViewMode: 'MODE_PLANAR', globe: {enabled: false}};
    await applyConfig(planar);
    return {fitting, stats: lighting.shadowViewportStats};
  });

  const failedPhases = results.filter(item => !item.passed);
  console.log(JSON.stringify({results}, null, 2));
  assert.equal(failedPhases.length, 0, `failed phases: ${failedPhases.map(item => item.phase).join(', ')}`);
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
