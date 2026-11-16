// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// SHARING5: catalog runs and chunked point-weights policy travel in the portable bundle under
// state.settings.extensions and are restored after a reload (Saved runs list in the catalog panel).
import {setAnalysisSheetParameter} from './analysis-sheet-controls.mjs';
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeExport} from './native-navigation.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
await pinViteModules(page);
const failures = [];
page.on('pageerror', error => failures.push(error.message));
const watchdog = setTimeout(() => {
  console.error('Save extensions check exceeded 150 seconds');
  void browser.close();
  process.exit(1);
}, 150000);
const address = process.argv[2] || 'http://127.0.0.1:5318/?rows=256';
try {
  await page.goto(address, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  const panel = page.getByRole('region', {name: 'Analysis catalog', exact: true});
  await panel.getByLabel('Operation', {exact: true}).selectOption('network-service-area');
  await setAnalysisSheetParameter(panel, 'coordinateSystem', 'planar');
  await setAnalysisSheetParameter(panel, 'onewayProperty', 'oneway');
  await setAnalysisSheetParameter(panel, 'maximumSnapDistance', 1);
  await setAnalysisSheetParameter(panel, 'budgets', [15]);
  await panel.getByTestId('catalog-check-inputs').click();
  await panel.getByTestId('catalog-run').click();
  await panel.getByTestId('catalog-result').waitFor();
  await panel.getByTestId('catalog-saved-invocations').waitFor();
  // The chunked point-weights policy member is covered by the node spec; the page hook only sets it for published sources.

  await openNativeExport(page, true);
  const pending = page.waitForEvent('download');
  await page.getByRole('button', {name: 'Save portable bundle', exact: true}).click();
  const chunks = [];
  for await (const chunk of await (await pending).createReadStream()) chunks.push(chunk);
  const bundleBytes = Buffer.concat(chunks);
  const extensions = JSON.parse(bundleBytes.toString('utf8')).state.settings?.extensions;
  assert.equal(extensions.catalogInvocations.records.length, 1);
  assert.equal(extensions.catalogInvocations.records[0].operationId, 'network-service-area');
  console.log(JSON.stringify({savedExtensions: Object.keys(extensions)}));

  await page.reload();
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount > 0, undefined, {timeout: 60000});
  assert.equal(await page.getByTestId('catalog-saved-invocations').count(), 0, 'a fresh page has no saved runs');
  await openNativeExport(page, true);
  const portable = page.getByRole('region', {name: 'Portable sandbox bundle', exact: true});
  await portable.getByLabel('Load portable bundle').setInputFiles({name: 'workspace.bundle.json', mimeType: 'application/json', buffer: bundleBytes});
  await portable.getByRole('status').filter({hasText: 'Bundle validated'}).waitFor({timeout: 30000});
  await portable.getByRole('button', {name: 'Restore portable workspace', exact: true}).click();
  await page.locator('.ReactModal__Overlay').waitFor({state: 'hidden', timeout: 60000});
  const saved = page.getByTestId('catalog-saved-invocations');
  await saved.waitFor({timeout: 30000});
  await saved.getByRole('button', {name: 'Load into form'}).click();
  assert.equal(await page.getByRole('region', {name: 'Analysis catalog', exact: true}).getByLabel('Operation', {exact: true}).inputValue(), 'network-service-area');
  assert.deepEqual(failures, []);
  console.log(JSON.stringify({passed: true}));
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
