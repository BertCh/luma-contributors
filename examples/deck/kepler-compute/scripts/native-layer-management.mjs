// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Qualifies native layer management on resident datasets: duplicate, reorder and delete must keep
// every surviving layer rendered and must release the deleted layer's resident resources.
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
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  const initial = await state();
  const original = initial.nativeLayers.find(layer => layer.type === 'point') ?? initial.nativeLayers[0];
  assert(original, 'a native layer exists');

  // Duplicate: a second layer appears, both stay valid, and the selection is untouched.
  await page.evaluate(id => window.__keplerComputeTest.duplicateNativeLayer(id), original.id);
  await page.waitForFunction(count => window.__keplerComputeTest.getState().nativeLayers.length === count + 1, initial.nativeLayers.length);
  const duplicated = await state();
  const copy = duplicated.nativeLayers.find(layer => !initial.nativeLayers.some(entry => entry.id === layer.id));
  assert(copy && copy.dataId === original.dataId, 'duplicate targets the same dataset');
  assert.equal(duplicated.selectedCount, 256);
  check('duplicate creates a second layer on the same resident dataset');

  // Reorder: swap the two layers; both remain valid and no page error occurs.
  const order = await page.evaluate(() => window.__keplerComputeTest.getNativeLayerOrder());
  await page.evaluate(next => window.__keplerComputeTest.reorderNativeLayers(next), [...order].reverse());
  await page.waitForFunction(first => window.__keplerComputeTest.getNativeLayerOrder()[0] === first, order[order.length - 1]);
  const reordered = await state();
  for (const layer of reordered.nativeLayers) assert.notEqual(layer.valid, false, `${layer.id} stays valid after reorder: ${layer.error}`);
  check('reorder keeps every layer valid');

  // Delete the copy: the original keeps rendering from the shared resident source.
  await page.evaluate(id => window.__keplerComputeTest.removeNativeLayer(id), copy.id);
  await page.waitForFunction(id => !window.__keplerComputeTest.getState().nativeLayers.some(layer => layer.id === id), copy.id);
  const afterDelete = await state();
  assert(afterDelete.nativeLayers.some(layer => layer.id === original.id && layer.valid !== false), 'original survives');
  assert.equal(afterDelete.selectedCount, 256);
  const rendered = await page.evaluate(() => window.__keplerComputeTest.renderedLayers());
  assert(!JSON.stringify(rendered).includes(copy.id), 'deleted layer is no longer rendered');
  check('deleting the duplicate leaves the original rendered and the source resident');

  // Delete the original too: no resident layer remains and nothing throws.
  await page.evaluate(id => window.__keplerComputeTest.removeNativeLayer(id), original.id);
  await page.waitForFunction(id => !window.__keplerComputeTest.getState().nativeLayers.some(layer => layer.id === id), original.id);
  check('deleting the last layer does not error');
  assert.deepEqual(errors, [], `page errors: ${errors.join('; ')}`);
  console.log(JSON.stringify({ok: true, checks}, null, 2));
} finally {
  await browser.close();
}
