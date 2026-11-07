// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Qualifies a native point layer remapped to a different lng/lat pair than the dataset's primary
// positions: an on-demand position variant is uploaded, rendered, filtered, accounted in the memory
// registry and released when the mapping returns. Not yet run; see the NATIVEUI4-P backlog items.
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1600, height: 1100}});
page.setDefaultTimeout(30000);
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => {errors.push(error.message); console.error(error.stack);});
const api = fn => page.evaluate(fn);
const checks = [];
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  const initial = await page.evaluate(() => window.__keplerComputeTest.getState());
  const original = initial.nativeLayers.find(layer => layer.type === 'point');
  assert(original, 'a native point layer exists');
  const baseline = await api(() => window.__keplerComputeTest.getPointVariantState());
  assert.equal(baseline.variantCount, 0, 'no variant before remapping');

  // Target coordinates are numeric columns on the fixture; the duplicate maps them as lng/lat.
  await page.evaluate(id => window.__keplerComputeTest.duplicateNativeLayer(id), original.id);
  await page.waitForFunction(count => window.__keplerComputeTest.getState().nativeLayers.length === count + 1, initial.nativeLayers.length);
  const copy = (await page.evaluate(() => window.__keplerComputeTest.getState())).nativeLayers.find(
    layer => !initial.nativeLayers.some(entry => entry.id === layer.id)
  );
  await page.evaluate(id => window.__keplerComputeTest.setNativePointColumns(id, 'targetLongitude', 'targetLatitude'), copy.id);
  await page.waitForFunction(() => window.__keplerComputeTest.getPointVariantState().variantCount === 1);
  const variant = await api(() => window.__keplerComputeTest.getPointVariantState());
  assert(variant.gpuByteLength > 0, 'variant GPU bytes tracked');
  assert(variant.memoryUsage.totalBytes >= baseline.memoryUsage.totalBytes + variant.gpuByteLength, 'registry accounts the variant');
  checks.push('remapped layer uploads one position variant and the registry accounts it');

  const rendered = JSON.stringify(await api(() => window.__keplerComputeTest.renderedLayers()));
  assert(rendered.includes(copy.id), 'variant layer renders');
  const state = await page.evaluate(() => window.__keplerComputeTest.getState());
  assert.notEqual(state.nativeLayers.find(layer => layer.id === copy.id)?.valid, false, 'variant layer valid');
  checks.push('variant layer renders without a mismatch error');

  // Polygon filter gets an independent mask per layer id for the variant.
  await page.evaluate(coordinates => window.__keplerComputeTest.addPolygon(coordinates), [[[-180, -85], [-180, 0], [0, 0], [0, -85], [-180, -85]]]);
  await page.waitForFunction(() => window.__keplerComputeTest.getState().selectedCount < 256);
  checks.push('polygon filter applies with the variant layer registered');

  await page.evaluate(id => window.__keplerComputeTest.setNativePointColumns(id, 'longitude', 'latitude'), copy.id);
  await page.waitForFunction(() => window.__keplerComputeTest.getPointVariantState().variantCount === 0);
  const released = await api(() => window.__keplerComputeTest.getPointVariantState());
  assert.equal(released.gpuByteLength, 0, 'variant released when no layer uses it');
  checks.push('variant released after remapping back to the primary pair');

  assert.deepEqual(errors, [], `page errors: ${errors.join('; ')}`);
  console.log(JSON.stringify({ok: true, checks}, null, 2));
} finally {
  await browser.close();
}
