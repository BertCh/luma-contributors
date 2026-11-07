// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Qualifies several point layers on one resident dataset under a polygon filter: the duplicate
// resolves its own layer mask while sharing the dataset's resident table, and neither layer is
// silently emptied or left unmasked. Not yet run; see the NATIVEUI3-M backlog items.
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
const state = () => page.evaluate(() => window.__keplerComputeTest.getState());
const checks = [];
const check = name => checks.push(name);
const polygon = [[[-180, -85], [-180, 0], [0, 0], [0, -85], [-180, -85]]];
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  const initial = await state();
  const original = initial.nativeLayers.find(layer => layer.type === 'point');
  assert(original, 'a native point layer exists');

  // Polygon filter first, then duplicate: the copy must pick up its own mask.
  await page.evaluate(coordinates => window.__keplerComputeTest.addPolygon(coordinates), polygon);
  await page.waitForFunction(() => window.__keplerComputeTest.getState().selectedCount < 256);
  const masked = (await state()).selectedCount;
  await page.evaluate(id => window.__keplerComputeTest.duplicateNativeLayer(id), original.id);
  await page.waitForFunction(count => window.__keplerComputeTest.getState().nativeLayers.length === count + 1, initial.nativeLayers.length);
  const duplicated = await state();
  const copy = duplicated.nativeLayers.find(layer => !initial.nativeLayers.some(entry => entry.id === layer.id));
  assert(copy && copy.dataId === original.dataId, 'duplicate targets the same dataset');
  assert.equal(duplicated.selectedCount, masked, 'duplicating does not change the dataset selection');
  for (const layer of duplicated.nativeLayers) assert.notEqual(layer.valid, false, `${layer.id} valid: ${layer.error}`);
  check('duplicate under a polygon filter keeps both point layers valid with the same selection');

  // Per-layer masks: both ids are registered and rendered.
  const rendered = JSON.stringify(await page.evaluate(() => window.__keplerComputeTest.renderedLayers()));
  assert(rendered.includes(original.id) && rendered.includes(copy.id), 'both point layers render');
  check('both point layers render from the shared resident table');

  // Reorder and delete the original: the copy keeps its mask.
  const order = await page.evaluate(() => window.__keplerComputeTest.getNativeLayerOrder());
  await page.evaluate(next => window.__keplerComputeTest.reorderNativeLayers(next), [...order].reverse());
  await page.evaluate(id => window.__keplerComputeTest.removeNativeLayer(id), original.id);
  await page.waitForFunction(id => !window.__keplerComputeTest.getState().nativeLayers.some(layer => layer.id === id), original.id);
  const afterDelete = await state();
  assert(afterDelete.nativeLayers.some(layer => layer.id === copy.id && layer.valid !== false), 'copy survives');
  // The polygon filter targets the layer ids it was drawn for (native Kepler never retargets it to a
  // duplicate), so once its only target is deleted the dataset selection settles back to unfiltered.
  await page.waitForFunction(count => window.__keplerComputeTest.getState().selectedCount === count, initial.selectedCount, {timeout: 15000});
  check('deleting the filter target releases the polygon mask and keeps the duplicate valid');

  assert.deepEqual(errors, [], `page errors: ${errors.join('; ')}`);
  console.log(JSON.stringify({ok: true, checks}, null, 2));
} finally {
  await browser.close();
}
