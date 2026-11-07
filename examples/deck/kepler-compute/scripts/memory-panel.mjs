// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The Status section shows registry usage per category and updates live while a reservation exists.
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=1024');
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 1024, undefined, {timeout: 60000});
  const panel = page.getByTestId('memory-budget-panel');
  await panel.waitFor({timeout: 30000});
  assert.match(await panel.getByTestId('memory-total').innerText(), /Used .* of .*; available/);
  const before = Number(await panel.locator('[data-category=readback]').getAttribute('data-bytes'));
  // A registry reservation tagged as readback must appear without a re-render from the host.
  const released = await page.evaluate(async () => {
    const api = window.__keplerComputeTest;
    const reservation = api.reserveTaggedMemory?.(1 << 20, 'readback');
    await new Promise(resolve => setTimeout(resolve, 200));
    const bytes = Number(document.querySelector('[data-category=readback]').dataset.bytes);
    reservation?.release();
    await new Promise(resolve => setTimeout(resolve, 200));
    return {bytes, after: Number(document.querySelector('[data-category=readback]').dataset.bytes)};
  });
  assert.equal(released.bytes, before + (1 << 20));
  assert.equal(released.after, before);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, checks: ['memory panel visible', 'tagged reservation tracked live']}));
} catch (error) {
  console.error(await page.locator('body').innerText());
  throw error;
} finally {await browser.close();}
