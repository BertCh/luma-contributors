// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Globe geometry variant lifecycle: the GeoJSON globe variant is prepared in time slices (never
// blocking the first globe frame), rebuilt at finer resolution per zoom band, and released when the
// map leaves globe mode. Reads publications[].globeGeometry from the test API.
//
// Usage: node scripts/globe-rebuild.mjs [url]
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const url = process.argv.slice(2).find(value => value.startsWith('http')) || 'http://127.0.0.1:5318/?rows=256';
const center = {longitude: -30, latitude: 20};
const half = 40;
const ring = [[-half, -half], [half, -half], [half, half], [-half, half], [-half, -half]].map(([longitude, latitude]) => [center.longitude + longitude, center.latitude + latitude, 0]);
const table = arrow.tableFromArrays({
  geometry: [JSON.stringify({type: 'Feature', geometry: {type: 'Polygon', coordinates: [ring]}, properties: {}})],
  measure: new Float32Array([0])
});

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => !/send was called before connect/.test(error.message) && errors.push(error.message));
const watchdog = setTimeout(() => { console.error('Globe rebuild check exceeded 140 seconds'); void browser.close(); process.exit(1); }, 140000);
const getGlobe = () => page.evaluate(() => window.__keplerComputeTest.getState().publications[0]?.globeGeometry ?? null);
const waitForGlobe = (predicate, label) => page.waitForFunction(`(${predicate})(window.__keplerComputeTest.getState().publications[0]?.globeGeometry)`, undefined, {timeout: 45000}).catch(async error => { throw new Error(`${label}: ${error.message} status=${JSON.stringify(await getGlobe())}`); });
try {
  await page.goto(url, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: 'globe-rebuild.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(table, 'stream'))});
  await workspace.getByLabel('Source geometry column').selectOption('geometry');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 1, undefined, {timeout: 45000});
  await closeNativeDialog(page);
  const datasetId = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);

  const setView = async (zoom, globe) => {
    const config = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
    const layer = config.config.visState.layers.find(candidate => candidate.config.dataId === datasetId);
    for (const candidate of config.config.visState.layers) candidate.config.isVisible = candidate === layer;
    layer.type = 'geojson';
    layer.config.visConfig = {...layer.config.visConfig, filled: true, stroked: false, enable3d: false, opacity: 1};
    config.config.mapState = {...config.config.mapState, ...center, zoom, pitch: 0, bearing: 0, dragRotate: false,
      mapViewMode: globe ? 'MODE_GLOBE' : 'MODE_PLANAR', globe: {...(config.config.mapState.globe ?? {}), enabled: globe}};
    await page.evaluate(value => window.__keplerComputeTest.applyNativeConfig(value), config);
  };

  // 1. Entering globe starts background preparation; the page keeps rendering while it runs.
  await setView(1.2, true);
  await waitForGlobe(status => status?.state === 'ready' && status.residentResolutionDegrees === 5, 'band 0 ready at 5 degrees');
  // 2. Zooming in rebuilds a finer variant; the coarse one is replaced, not kept alongside.
  await setView(3.6, true);
  await waitForGlobe(status => status?.state === 'ready' && status.residentResolutionDegrees === 2, 'band 1 ready at 2 degrees');
  await setView(5.6, true);
  await waitForGlobe(status => status?.state === 'ready' && status.residentResolutionDegrees === 1, 'band 2 ready at 1 degree');
  // 3. Leaving globe releases the variant so GeoJSON geometry is not resident twice.
  await setView(1.2, false);
  await waitForGlobe(status => !status || (status.state === 'idle' && status.residentResolutionDegrees === undefined), 'released after leaving globe');
  const state = await page.evaluate(() => window.__keplerComputeTest.getState());
  assert.equal(state.error, '');
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({globeRebuild: 'ok', final: await getGlobe()}));
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
