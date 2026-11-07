// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Select-by-click of an existing native editor feature in Single, Dual and Swipe maps.
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const baseUrl = process.env.KEPLER_COMPUTE_URL ?? 'http://127.0.0.1:5318/';
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await (await browser.newContext({viewport: {width: 1440, height: 1000}})).newPage();
await pinViteModules(page);
const failures = [];
page.on('pageerror', error => failures.push(error.message));

const readEditor = () => page.evaluate(() => window.__keplerComputeTest.nativeEditor());
async function chooseSplitMode(title) {
  await page.locator('.map-control-button.split-map').click();
  await page.locator('.toolbar-item', {hasText: new RegExp(`^${title}$`)}).click();
  await page.waitForTimeout(2500);
}
async function pointerClick(x, y) {
  await page.mouse.move(x, y);
  await page.waitForTimeout(700);
  await page.mouse.click(x, y);
  await page.waitForTimeout(1200);
}
async function drawRectangle([x0, y0, x1, y1]) {
  const previousCount = (await readEditor()).features.length;
  await page.locator('.map-control-button.map-draw').first().click();
  await page.getByText('Polygon', {exact: true}).first().click();
  for (const [x, y] of [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) await pointerClick(x, y);
  await page.mouse.dblclick(x0, y1);
  await page.waitForFunction(
    count => window.__keplerComputeTest.nativeEditor().features.length === count + 1,
    previousCount,
    {timeout: 10000}
  );
  return (await readEditor()).features.at(-1);
}

try {
  const regions = {
    Single: [400, 600, 700, 800],
    Swipe: [350, 790, 650, 900],
    Dual: [350, 790, 650, 900],
    // The right-hand view of a Dual map: the editor feature must also select there.
    DualRight: [900, 790, 1200, 900]
  };
  for (const title of ['Single', 'Swipe', 'Dual', 'DualRight']) {
    await page.goto(baseUrl);
    await page.locator('.maplibregl-canvas').waitFor();
    await page.waitForTimeout(2000);
    if (title !== 'Single') await chooseSplitMode(title === 'DualRight' ? 'Dual' : title);
    const region = regions[title];
    const feature = await drawRectangle(region);
    // Leave drawing, then click empty map: the selection is cleared.
    await page.locator('.map-control-button.map-draw').first().click();
    // Click the map right of the polygon, clear of the split divider near x=730: left of it lies the side panel, which is not map surface.
    await pointerClick(region[2] + 30, region[1] - 100);
    assert.equal((await readEditor()).selectedFeature, null, `${title}: empty click deselects`);
    // An unselected polygon is unfilled, so select it through its outline.
    await pointerClick((region[0] + region[2]) / 2, region[1]);
    assert.equal((await readEditor()).selectedFeature?.id, feature.id, `${title}: outline click selects the existing feature`);
  }
  assert.deepEqual(failures, [], failures.join('\n'));
  console.log('views-editor-select.mjs checks completed');
} finally {
  await browser.close();
}
