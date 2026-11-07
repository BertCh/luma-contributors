// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// LAYERS4-C3: native geojson path and outline primitives use rounded caps and joints. A single wide
// LineString must extend past its projected end points by about half its width at each end, and the
// rendered path primitive must report capRounded and jointRounded. Headless WebGPU only.
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const feature = (geometry) => JSON.stringify({type: 'Feature', geometry, properties: {}});
const lineStart = [-74.02, 40.75];
const lineEnd = [-73.98, 40.75];
const source = arrow.tableFromArrays({
  geometry: [feature({type: 'LineString', coordinates: [lineStart, lineEnd]}), feature({type: 'Polygon', coordinates: [[[-74.02, 40.77], [-73.98, 40.77], [-73.98, 40.79], [-74.02, 40.79], [-74.02, 40.77]]]})],
  sourceId: new Float32Array([0, 1])
});
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const failures = [];
page.on('pageerror', error => failures.push(error.message));
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())) failures.push(message.text());
});
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: 'rounded-caps.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))});
  await workspace.getByLabel('Source geometry column').selectOption('geometry');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 2, undefined, {timeout: 45000});
  await closeNativeDialog(page);
  const datasetId = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);
  const layerId = `${datasetId}-geometry`;
  const configuration = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  for (const layer of configuration.config.visState.layers) layer.config.isVisible = layer.id === layerId;
  const layer = configuration.config.visState.layers.find(candidate => candidate.id === layerId);
  assert(layer, 'native geojson layer exists');
  layer.config.visConfig = {...layer.config.visConfig, filled: false, stroked: true, enable3d: false, thickness: 12, opacity: 1, strokeOpacity: 1, strokeColor: [20, 220, 230]};
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), configuration);
  await page.evaluate(() => window.__keplerComputeTest.setMapViewport({longitude: -74, latitude: 40.77, zoom: 12.5}));
  await page.waitForFunction(id => {
    const api = window.__keplerComputeTest;
    if (api.getState().error) throw new Error(api.getState().error);
    return api.renderedLayers().some(candidate => candidate.renderId === `${id}-paths` && candidate.visible && candidate.loaded);
  }, layerId, {timeout: 45000});
  await page.waitForTimeout(1500);
  const paths = await page.evaluate(id => window.__keplerComputeTest.renderedLayers().filter(candidate => candidate.renderId.startsWith(`${id}-paths`)), layerId);
  assert(paths.length > 0, 'a path primitive is rendered');
  for (const path of paths) {
    assert.equal(path.capRounded, true, `${path.renderId} capRounded`);
    assert.equal(path.jointRounded, true, `${path.renderId} jointRounded`);
  }
  const start = await page.evaluate(([longitude, latitude]) => window.__keplerComputeTest.projectPosition(longitude, latitude), lineStart);
  const end = await page.evaluate(([longitude, latitude]) => window.__keplerComputeTest.projectPosition(longitude, latitude), lineEnd);
  const image = PNG.sync.read(await page.screenshot());
  const isInk = (x, y) => {
    const offset = (Math.round(y) * image.width + Math.round(x)) * 4;
    return image.data[offset] < 90 && image.data[offset + 1] > 160 && image.data[offset + 2] > 160;
  };
  const rowY = Math.round(start[1]);
  let left = Infinity;
  let right = -Infinity;
  for (let x = 0; x < image.width; x++) if (isInk(x, rowY)) {left = Math.min(left, x); right = Math.max(right, x);}
  let top = Infinity;
  let bottom = -Infinity;
  const midX = Math.round((start[0] + end[0]) / 2);
  for (let y = 0; y < image.height; y++) if (isInk(midX, y) && Math.abs(y - rowY) < 120) {top = Math.min(top, y); bottom = Math.max(bottom, y);}
  const width = bottom - top + 1;
  const projectedLength = end[0] - start[0];
  const overhang = (right - left + 1) - projectedLength;
  assert(width >= 6, `line must have visible width (${width})`);
  // A flat cap overhangs by 0; a round cap overhangs by one half width at each end.
  assert(overhang > width * 0.6, `rounded caps must extend past the end points (overhang ${overhang}, width ${width})`);
  // The cap is round, not square: the corner pixel of the end square is not inked.
  assert(!isInk(right, top) && !isInk(left, bottom), 'cap corners are rounded');
  assert.equal(await page.evaluate(() => window.__keplerComputeTest.getState().error), '');
  assert.deepEqual(failures, []);
  console.log(JSON.stringify({passed: true, width, overhang, projectedLength, primitives: paths.map(path => path.renderId)}));
} catch (error) {
  console.error({failures});
  console.error(error);
  process.exitCode = 1;
} finally {
  await browser.close();
}
