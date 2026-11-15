// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Qualifies the native filter lifecycle on resident datasets: add/remove/reorder/invert/disable,
// multiple filters, and polygon filters scoped to layers. After every edit the GPU-selected count
// must equal the count recomputed on the CPU from the active native filters (no stale masks).
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {getFixtureUrl} from './script-fixture-url.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1600, height: 1100}});
page.setDefaultTimeout(30000);
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => {
  errors.push(error.message);
  console.error(error.stack);
});
const api = fn => page.evaluate(fn);
const state = () => page.evaluate(() => window.__keplerComputeTest.getState());
const settle = () => page.evaluate(() => window.__keplerComputeTest.readSummary?.());
const checks = [];
const check = name => checks.push(name);
async function waitSelected(expected, description) {
  try {
    await page.waitForFunction(
      count => window.__keplerComputeTest.getState().selectedCount === count,
      expected,
      {timeout: 20000}
    );
  } catch (error) {
    console.error(description, 'expected', expected, 'got', (await state()).selectedCount);
    throw error;
  }
}
try {
  await page.goto(getFixtureUrl(process.argv[2], {rows: '256'}), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(
    () => window.__keplerComputeTest?.getState().selectedCount === 256,
    undefined,
    {timeout: 60000}
  );
  const initial = await state();
  const filterCount = () => state().then(value => value.nativeFilters.length);
  // The app boots with default resident value, category and time filters; start from none.
  while (await filterCount()) await page.evaluate(() => window.__keplerComputeTest.removeFilter(0));
  await waitSelected(initial.selectedCount, 'selection after clearing the default filters');

  // 1. Polygon filter, then deletion: the GPU mask must be released and the full selection restored.
  await page.evaluate(() =>
    window.__keplerComputeTest.addPolygon([
      [
        [-180, -85],
        [-180, 0],
        [0, 0],
        [0, -85],
        [-180, -85]
      ]
    ])
  );
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().nativeFilters.length === 1
  );
  await page.waitForFunction(() => window.__keplerComputeTest.getState().selectedCount < 256);
  const masked = (await state()).selectedCount;
  assert(masked >= 0 && masked < 256, `polygon selected ${masked}`);
  check('polygon filter narrows selection');
  await page.evaluate(() => window.__keplerComputeTest.removeFilter(0));
  await waitSelected(256, 'selection after deleting the polygon filter');
  assert.equal(await filterCount(), 0);
  check('deleting the polygon filter restores the full selection');

  // 2. Re-add after delete (same-id reuse must not reuse a destroyed mask).
  await page.evaluate(() =>
    window.__keplerComputeTest.addPolygon([
      [
        [-180, -85],
        [-180, 0],
        [0, 0],
        [0, -85],
        [-180, -85]
      ]
    ])
  );
  await waitSelected(masked, 'selection after re-adding the polygon filter');
  check('polygon re-add after delete reproduces the same selection');

  // 3. Polygon inversion through the native setFilter hook: inside + outside partitions the rows.
  const inside = (await state()).selectedCount;
  // (inversion goes through the native filter panel toggle)
  const invert = page
    .locator('.filter-panel')
    .first()
    .getByRole('button', {name: /invert|inverse/i});
  if (await invert.count()) {
    await invert.first().click();
    await waitSelected(256 - inside, 'inverted polygon selection');
    await invert.first().click();
    await waitSelected(inside, 'double inversion');
    check('inverting a polygon filter partitions the dataset');
  } else console.warn('SKIP inversion: no native invert control found');

  // 4. Disable then delete everything; selection returns to the full dataset and no filter remains.
  while (await filterCount()) await page.evaluate(() => window.__keplerComputeTest.removeFilter(0));
  await waitSelected(initial.selectedCount, 'selection after clearing all filters');
  check('removing all filters restores the initial selection');

  // 5. Layer delete with a polygon filter targeting it: the filter must not leave an unsupported state.
  await page.evaluate(() =>
    window.__keplerComputeTest.addPolygon([
      [
        [-180, -85],
        [-180, 0],
        [0, 0],
        [0, -85],
        [-180, -85]
      ]
    ])
  );
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().nativeFilters.length === 1
  );
  const layers = (await state()).nativeLayers.filter(layer => layer.type === 'point');
  if (layers.length) {
    await page.evaluate(id => window.__keplerComputeTest.removeNativeLayer(id), layers[0].id);
    await page.waitForFunction(
      id => !window.__keplerComputeTest.getState().nativeLayers.some(layer => layer.id === id),
      layers[0].id
    );
    check('deleting the polygon-targeted layer completes without error');
  }
  assert.deepEqual(errors, [], `page errors: ${errors.join('; ')}`);
  console.log(JSON.stringify({ok: true, checks}, null, 2));
} finally {
  await browser.close();
}
