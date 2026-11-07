// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Smoke test of the host Time contract panel: it renders, validates zones, and the spatial join
// CRS panel stays hidden without a spatial join. Headless only.
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
try {
  await page.goto(baseUrl);
  await page.locator('[data-testid="time-contract-panel"]').waitFor({timeout: 30000});
  await page.locator('[data-testid="time-contract-zone-input"]').fill('Not/AZone');
  await page.locator('[data-testid="time-contract-zone-apply"]').click();
  await page.getByText('Unknown time zone').waitFor();
  await page.locator('[data-testid="time-contract-zone-input"]').fill('America/New_York');
  await page.locator('[data-testid="time-contract-zone-apply"]').click();
  await page.waitForTimeout(500);
  assert.equal(await page.getByText('Unknown time zone').count(), 0);
  assert.equal(await page.locator('[data-testid="spatial-join-crs-panel"]').count(), 0);
  assert.deepEqual(failures, []);
  console.log('time-contract-panel: ok (smoke only; playback and zoned histograms need a time filter)');
} finally {
  await browser.close();
}
