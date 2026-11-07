// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {openNativeData, openNativeExport} from './native-navigation.mjs';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
const errors = [];
page.on('pageerror', error => errors.push(error.message));
async function downloadJson(button) {
  const pending = page.waitForEvent('download');
  await button.click();
  const stream = await (await pending).createReadStream();
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=4096', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => {
    const api = window.__keplerComputeTest;
    const state = api?.getState();
    if (state?.error) throw new Error(state.error);
    return state?.ready && state.selectedCount === api.source().length;
  }, undefined, {timeout: 60000});

  const guide = page.getByTestId('sandbox-capabilities');
  await guide.locator('summary').first().click();
  await guide.getByLabel('Workflow', {exact: true}).selectOption('Layers');
  assert.equal(await guide.getByRole('status').textContent(), '22 of 66 features');
  await guide.getByLabel('Find a feature or contributor').fill('raster');
  await guide.getByLabel('Integration stage').selectOption({label: 'Partial host integration'});
  await guide.getByText('Raster tiles', {exact: true}).click();
  const raster = guide.locator('.sandbox-capabilities-results > details').filter({
    has: page.getByText('Raster tiles', {exact: true})
  });
  assert(await raster.getByText('Remaining work:', {exact: true}).isVisible());
  assert(await raster.getByText('Acceptance:', {exact: true}).isVisible());
  await guide.getByLabel('Find a feature or contributor').fill('');
  await guide.getByLabel('Workflow', {exact: true}).selectOption('');
  await guide.getByLabel('Integration stage').selectOption('');
  assert.equal(await guide.getByRole('status').textContent(), '66 of 66 features');
  await guide.locator('summary').first().click();

  const expectedCount = await page.evaluate(() => {
    const api = window.__keplerComputeTest;
    api.setRange(25, 65);
    return api.source().filter(row => row[2] >= 25 && row[2] <= 65).length;
  });
  await page.waitForFunction(count => window.__keplerComputeTest.getState().selectedCount === count, expectedCount);
  await openNativeExport(page, true);
  const saving = page.getByRole('region', {name: 'Save map', exact: true});
  const saved = await downloadJson(saving.getByRole('button', {name: 'Save state', exact: true}));
  assert(saved && typeof saved === 'object');
  await page.evaluate(() => window.__keplerComputeTest.setRange(0, 100));
  await page.waitForFunction(() => window.__keplerComputeTest.getState().selectedCount === window.__keplerComputeTest.source().length);
  await saving.getByRole('button', {name: 'Replay state', exact: true}).click();
  await page.waitForFunction(count => window.__keplerComputeTest.getState().selectedCount === count, expectedCount);
  await page.locator('.ReactModal__Overlay').waitFor({state: 'hidden'});
  await openNativeExport(page, true);
  await saving.getByText('Inspect or edit saved state', {exact: true}).click();
  await saving.getByLabel('Saved sandbox state', {exact: true}).fill(JSON.stringify(saved));
  await saving.getByRole('button', {name: 'Reimport and replay', exact: true}).click();
  await page.waitForFunction(count => window.__keplerComputeTest.getState().selectedCount === count, expectedCount);
  await page.locator('.ReactModal__Overlay').waitFor({state: 'hidden'});

  await openNativeExport(page);
  const exported = await downloadJson(page.getByRole('button', {name: 'Export all selected', exact: true}));
  assert.equal(exported.mode, 'all-selected');
  assert.equal(exported.rowCount, expectedCount);
  assert.equal(exported.sourceRowIndices.length, expectedCount);
  await openNativeData(page, 'SQL');
  const sql = page.getByRole('region', {name: 'Native SQL Data Explorer', exact: true});
  // DuckDB initializes and registers the source lazily when a CPU-routed statement runs.
  await sql.getByRole('button', {name: 'Run SQL', exact: true}).click({timeout: 60000});
  await sql.getByRole('status').filter({hasText: /4 DuckDB result rows/}).waitFor({timeout: 60000});
  await sql.getByRole('button', {name: 'Publish result', exact: true}).click();
  // Native addDataToMap closes the Add Data dialog once the result is published; reopen it to inspect.
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications.some(source => source.id.endsWith('-sql-result')));
  await openNativeData(page, 'SQL');
  await sql.getByText(/Published: .*-sql-result/).waitFor();
  assert.equal(await sql.getByRole('alert').count(), 0);
  assert.deepEqual(errors, []);
  process.stdout.write(`${JSON.stringify({passed: true, selectedCount: expectedCount,
    checks: ['capability search', 'native state replay', 'explicit selected export',
      'source regeneration', 'optional DuckDB source registration', 'SQL grouping', 'explicit GPU upload']}, null, 2)}\n`);
} catch (error) {
  process.stderr.write(`${JSON.stringify(await page.evaluate(() => ({
    state: window.__keplerComputeTest?.getState(),
    statuses: Array.from(document.querySelectorAll('[role="status"]'), element => element.textContent)
  })), null, 2)}\n`);
  throw error;
} finally {
  await browser.close();
}
