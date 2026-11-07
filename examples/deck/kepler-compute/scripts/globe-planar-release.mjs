// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Planar GeometryRuntime release under a globe view. A wide GeoJSON polygon is published and drawn
// on the globe; after the grace period the planar copy must be released
// (publications[].globeGeometry.planar.state === 'released'), the polygon must keep drawing in
// Single, Swipe and Dual globes (and with the extruded variant), and leaving globe must restore the
// planar copy ('resident') and free the globe variant. Browser verification pending; run through
// the central tester.
//
// Usage: node scripts/globe-planar-release.mjs [url]
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';
import {GLOBE_SPLIT_REGIONS} from './globe-split-matrix.mjs';

const url = process.argv.slice(2).find(value => value.startsWith('http')) || 'http://127.0.0.1:5318/?rows=256';
const CENTER = {longitude: -30, latitude: 20};
const HALF = 30;
const ring = [[-HALF, -HALF], [HALF, -HALF], [HALF, HALF], [-HALF, HALF], [-HALF, -HALF]].map(([longitude, latitude]) => [CENTER.longitude + longitude, CENTER.latitude + latitude, 0]);
const table = arrow.tableFromArrays({
  geometry: [JSON.stringify({type: 'Feature', geometry: {type: 'Polygon', coordinates: [ring]}, properties: {}})],
  measure: new Float32Array([0])
});

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => !/send was called before connect/.test(error.message) && errors.push(error.message));
const watchdog = setTimeout(() => { console.error('Globe planar release exceeded 200 seconds'); void browser.close(); process.exit(1); }, 200000);

const isMagenta = (red, green, blue) => red > 150 && green < 110 && blue > 90 && red - green > 90;
const getGlobe = () => page.evaluate(() => window.__keplerComputeTest.getState().publications[0]?.globeGeometry ?? null);
const waitForGlobe = (predicate, label) => page.waitForFunction(`(${predicate})(window.__keplerComputeTest.getState().publications[0]?.globeGeometry)`, undefined, {timeout: 60000}).catch(async error => { throw new Error(`${label}: ${error.message} status=${JSON.stringify(await getGlobe())}`); });
async function countRegions() {
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const image = PNG.sync.read(await page.screenshot());
  const counts = {};
  for (const [name, [left, right]] of Object.entries(GLOBE_SPLIT_REGIONS)) {
    let pixels = 0;
    for (let vertical = 60; vertical < image.height - 40; vertical++)
      for (let horizontal = left; horizontal < right; horizontal++) {
        const offset = (vertical * image.width + horizontal) * 4;
        if (isMagenta(image.data[offset], image.data[offset + 1], image.data[offset + 2])) pixels++;
      }
    counts[name] = pixels;
  }
  return counts;
}
const chooseSplitMode = async title => {
  await page.locator('.map-control-button.split-map').click();
  await page.locator('.toolbar-item', {hasText: new RegExp(`^${title}$`)}).click();
};
const waitForMode = async mode => {
  await page.waitForFunction(expected => window.__keplerComputeTest.saveNativeConfig().config.mapState.mapSplitMode === expected, mode);
  await page.waitForTimeout(2500);
};

try {
  await page.goto(url, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: 'globe-planar-release.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(table, 'stream'))});
  await workspace.getByLabel('Source geometry column').selectOption('geometry');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 1, undefined, {timeout: 45000});
  await closeNativeDialog(page);
  const datasetId = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);

  const setView = async (globe, enable3d) => {
    const config = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
    const layer = config.config.visState.layers.find(candidate => candidate.config.dataId === datasetId);
    for (const candidate of config.config.visState.layers) candidate.config.isVisible = candidate === layer;
    layer.type = 'geojson';
    layer.config.color = [255, 20, 170];
    layer.config.visConfig = {...layer.config.visConfig, filled: true, stroked: false, enable3d, fixedHeight: true, opacity: 1, fillColor: [255, 20, 170]};
    config.config.mapState = {...config.config.mapState, ...CENTER, zoom: 2.2, pitch: 0, bearing: 0, dragRotate: false, mapSplitMode: 'SINGLE_MAP',
      mapViewMode: globe ? 'MODE_GLOBE' : 'MODE_PLANAR', globe: {...(config.config.mapState.globe ?? {}), enabled: globe}};
    await page.evaluate(value => window.__keplerComputeTest.applyNativeConfig(value), config);
    await page.waitForTimeout(1500);
  };

  const record = {};
  // 1. Globe: variant ready, then the planar copy is released after the grace period.
  await setView(true, false);
  await waitForGlobe(status => status?.state === 'ready' && status.planar?.state === 'released', 'planar released under globe');
  record.single = await countRegions();
  assert(record.single.full > 400, `released globe must draw (${JSON.stringify(record.single)})`);
  // 2. Extruded polygons use the same released-planar globe variant.
  await setView(true, true);
  await waitForGlobe(status => status?.state === 'ready' && status.planar?.state === 'released', 'planar stays released (extruded)');
  record.extruded = await countRegions();
  assert(record.extruded.full > 200, `extruded globe must draw (${JSON.stringify(record.extruded)})`);
  await setView(true, false);
  // 3. Swipe and Dual globes keep drawing on both sides while released.
  for (const [title, mode] of [['Swipe', 'SWIPE_COMPARE'], ['Dual', 'DUAL_MAP']]) {
    await chooseSplitMode(title);
    await waitForMode(mode);
    const split = await countRegions();
    assert(split.left > 100 && split.right > 100, `${title} globe must draw both halves (${JSON.stringify(split)})`);
    assert.equal((await getGlobe())?.planar?.state, 'released');
    record[title] = split;
    await chooseSplitMode('Single');
    await waitForMode('SINGLE_MAP');
  }
  // 4. Leaving globe rebuilds the planar copy, frees the globe variant and keeps drawing.
  await setView(false, false);
  await waitForGlobe(status => !status || (status.state === 'idle' && (status.planar?.state ?? 'resident') === 'resident'), 'planar restored after leaving globe');
  record.flat = await countRegions();
  assert(record.flat.full > 400, `restored planar must draw (${JSON.stringify(record.flat)})`);
  // 5. Re-entering globe works again from the restored planar copy.
  await setView(true, false);
  await waitForGlobe(status => status?.state === 'ready' && status.planar?.state === 'released', 'planar released again');
  const state = await page.evaluate(() => window.__keplerComputeTest.getState());
  assert.equal(state.error, '');
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({globePlanarRelease: 'ok', record}));
} catch (error) {
  console.error(error.message, errors);
  await page.screenshot({path: '/tmp/luma-kepler-globe-planar-release-failure.png'}).catch(() => {});
  process.exitCode = 1;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
