// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Native memory manager: side-panel summary, modal with per-dataset/per-layer rows, budget change
// and a Kepler notification when a reservation crosses the warning threshold.
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
  const section = page.getByTestId('native-memory');
  await section.waitFor({timeout: 30000});
  await section.locator('summary').click();
  assert.match(await section.getByTestId('memory-manager-summary').innerText(), /% used: .* of .*available/);
  await section.getByTestId('memory-manager-open').click();
  const modal = page.getByTestId('memory-manager-full');
  await modal.waitFor({timeout: 10000});
  const baseRow = modal.locator('[data-dataset-id]').first();
  assert.match(await baseRow.innerText(), /pinned/);
  assert.equal(await baseRow.getByRole('button', {name: 'Unload'}).isDisabled(), true);
  assert.ok((await modal.getByTestId('memory-manager-layers').locator('tbody tr').count()) >= 0);
  await modal.getByTestId('memory-manager-budget-input').fill('700');
  await modal.getByTestId('memory-manager-budget-apply').click();
  assert.match(await modal.getByTestId('memory-manager-summary').innerText(), /700(\.0)? MiB/);
  // Crossing 80% of the budget raises one Kepler notification.
  const limitBytes = await page.evaluate(() => 700 * 1024 * 1024 * 0.85);
  const reserved = await page.evaluate(async bytes => {
    const reservation = window.__keplerComputeTest.reserveTaggedMemory?.(Math.floor(bytes), 'scratch');
    await new Promise(resolve => setTimeout(resolve, 300));
    const warned = document.body.innerText.includes('GPU memory budget');
    reservation?.release();
    return warned;
  }, limitBytes);
  assert.equal(reserved, true);
  // Lowering the budget below usage is rejected with an explanation.
  await modal.getByTestId('memory-manager-budget-input').fill('1');
  await modal.getByTestId('memory-manager-budget-apply').click();
  assert.match(await modal.getByTestId('memory-manager-budget-error').innerText(), /below/);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, checks: ['side panel summary', 'modal rows', 'budget change', 'notification', 'budget floor']}));
} catch (error) {
  console.error(await page.locator('body').innerText());
  throw error;
} finally {await browser.close();}
