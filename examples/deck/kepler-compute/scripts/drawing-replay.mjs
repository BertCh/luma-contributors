// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// NATIVEUI6 browser check (headless, not yet run): drawing replay steps through drawn shapes with
// fly-to and selection.
// Usage: node scripts/drawing-replay.mjs [url]
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));

const square = (id, west, south, size) => ({
  type: 'Feature',
  id,
  properties: {isClosed: true},
  geometry: {
    type: 'Polygon',
    coordinates: [[[west, south], [west + size, south], [west + size, south + size], [west, south + size], [west, south]]]
  }
});

try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256');
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});

  // Drawing replay: two shapes, step with Next, each step moves the map and selects the shape.
  await page.evaluate(features => window.__keplerComputeTest.setDrawnFeatures(features), [
    square('replay-a', -74.02, 40.7, 0.02),
    square('replay-b', -73.9, 40.8, 0.05)
  ]);
  const replay = page.getByTestId('drawing-replay');
  await replay.waitFor({timeout: 20000});
  const status = page.getByTestId('drawing-replay-status');
  assert.match(await status.textContent(), /2 steps/);
  await replay.getByRole('button', {name: 'Next step'}).click();
  await page.waitForFunction(() => document.querySelector('[data-testid="drawing-replay-status"]')?.textContent?.includes('Step 2 of 2'));
  await replay.getByRole('button', {name: 'Previous step'}).click();
  await page.waitForFunction(() => document.querySelector('[data-testid="drawing-replay-status"]')?.textContent?.includes('Step 1 of 2'));
  await page.waitForFunction(() => document.querySelector('[data-testid="drawing-selected"]')?.textContent?.includes('Polygon'));
  await replay.getByTestId('drawing-replay-play').click();
  await page.waitForFunction(() => document.querySelector('[data-testid="drawing-replay-status"]')?.textContent?.includes('Step 2 of 2'), undefined, {timeout: 15000});
  const view = await page.evaluate(() => window.__keplerComputeTest.getState());
  console.log(JSON.stringify({replayStatus: await status.textContent(), selectedCount: view.selectedCount, errors}, null, 2));
  assert.deepEqual(errors, []);
} finally {
  await browser.close();
}
