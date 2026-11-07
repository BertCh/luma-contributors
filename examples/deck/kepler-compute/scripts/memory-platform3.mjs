// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Memory manager round 6: measured layer render buffers, the budget surviving a reload through
// localStorage, and the running-operations list. Headless Playwright only.
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
const url = process.argv[2] || 'http://127.0.0.1:5318/?rows=1024';
async function openModal() {
  await page.goto(url);
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 1024, undefined, {timeout: 60000});
  const section = page.getByTestId('native-memory');
  await section.waitFor({timeout: 30000});
  await section.locator('summary').click();
  await section.getByTestId('memory-manager-open').click();
  const modal = page.getByTestId('memory-manager-full');
  await modal.waitFor({timeout: 10000});
  return modal;
}
try {
  let modal = await openModal();
  // Layers report measured render buffers, not a divided dataset total.
  const layerRows = modal.getByTestId('memory-manager-layers').locator('tbody tr');
  assert.ok((await layerRows.count()) >= 1);
  const measured = await layerRows.first().locator('[data-render-buffer-bytes]').getAttribute('data-render-buffer-bytes');
  assert.ok(Number(measured) >= 0);
  assert.match(await modal.getByTestId('memory-manager-layers').innerText(), /Render buffers/);
  // The operations list exists and is empty at rest.
  assert.match(await modal.getByTestId('memory-manager-operations').innerText(), /No running operations/);
  await modal.getByTestId('memory-manager-budget-input').fill('900');
  await modal.getByTestId('memory-manager-budget-apply').click();
  assert.match(await modal.getByTestId('memory-manager-summary').innerText(), /900(\.0)? MiB/);
  assert.equal(await page.evaluate(() => localStorage.getItem('luma-kepler-memory-budget-bytes')), String(900 * 1024 * 1024));
  // After a reload the remembered budget is applied to the new session.
  modal = await openModal();
  assert.match(await modal.getByTestId('memory-manager-summary').innerText(), /900(\.0)? MiB/);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, checks: ['render buffers', 'operations list', 'budget persisted']}));
} catch (error) {
  console.error(await page.locator('body').innerText());
  throw error;
} finally {
  await browser.close();
}
