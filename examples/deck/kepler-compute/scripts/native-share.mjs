// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {openNativeExport} from './native-navigation.mjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const errors = [];
page.on('pageerror', error => errors.push(error.message));
async function waitForReady() {
  await page.waitForFunction(() => {
    const state = window.__keplerComputeTest?.getState();
    if (state?.error) throw new Error(state.error);
    return state?.ready && state.selectedCount !== undefined;
  }, undefined, {timeout: 60000});
}
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=4096');
  await waitForReady();
  const expectedCount = await page.evaluate(() => {
    const api = window.__keplerComputeTest;
    api.setRange(25, 65);
    return api.source().filter(row => row[2] >= 25 && row[2] <= 65).length;
  });
  await page.waitForFunction(count => window.__keplerComputeTest.getState().selectedCount === count, expectedCount);
  await openNativeExport(page, true);
  await page.getByRole('button', {name: 'Create share link', exact: true}).click();
  const link = await page.getByLabel('Sandbox share link').inputValue();
  assert.equal(new URL(link).search, '?rows=4096');
  // A reload exercises a fresh device/session and native dataset, not an in-memory replay.
  await page.goto(link);
  await page.reload();
  await waitForReady();
  await openNativeExport(page, true);
  await page.getByRole('status').filter({hasText: 'Shared state loaded'}).waitFor();
  const saved = JSON.parse(await page.getByLabel('Saved sandbox state').inputValue());
  assert.equal(saved.sources[0].reference.parameters.rowCount, 4096);
  assert.equal(await page.evaluate(() => window.__keplerComputeTest.getState().selectedCount), 4096);
  await page.getByRole('button', {name: 'Reimport and replay', exact: true}).click();
  // Replay re-adds data to the map, so Kepler closes the modal and its status text; the restored GPU selection proves completion.
  await page.waitForFunction(count => window.__keplerComputeTest.getState().selectedCount === count, expectedCount, {timeout: 60000});
  await openNativeExport(page, true);
  await page.getByRole('button', {name: 'Create share link', exact: true}).scrollIntoViewIfNeeded();
  await page.screenshot({path: '/tmp/kepler-share-ui.png'});
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, expectedCount, checks: ['share current native config', 'fresh session', 'explicit replay', 'restored GPU selection']}));
} finally {await browser.close();}
