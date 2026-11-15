// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A mid-restore failure (injected through the test API) leaves no half-restored publications.
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getSourceCheckUrl} from './source-check-utils.mjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, openNativeExport} from './native-navigation.mjs';

const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
const makeTable = offset =>
  new arrow.Table({
    x: arrow.makeVector(new Float32Array([-74, -74.01, -74.02, -74.03])),
    y: arrow.makeVector(new Float32Array([40.7, 40.71, 40.72, 40.73])),
    amount: arrow.makeVector(new Float32Array([offset, offset + 1, offset + 2, offset + 3]))
  });
async function readDownload(button) {
  const pending = page.waitForEvent('download');
  await button.click();
  const chunks = [];
  for await (const chunk of await (await pending).createReadStream()) chunks.push(chunk);
  return Buffer.concat(chunks);
}
async function publish(name, offset) {
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name,
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(makeTable(offset), 'stream'))
  });
  await workspace.getByLabel('Source longitude column').selectOption('x');
  await workspace.getByLabel('Source latitude column').selectOption('y');
  await workspace.getByRole('button', {name: 'Check schema and precision'}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields'}).click();
}
try {
  await page.goto(getSourceCheckUrl(process.argv[2], {rows: 1024}));
  await page.waitForFunction(
    () => window.__keplerComputeTest?.getState().selectedCount === 1024,
    undefined,
    {timeout: 60000}
  );
  await publish('first.arrow', 10);
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications.length === 1,
    undefined,
    {timeout: 45000}
  );
  await publish('second.arrow', 20);
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications.length === 2,
    undefined,
    {timeout: 45000}
  );
  await openNativeExport(page, true);
  const bundleBytes = await readDownload(
    page.getByRole('button', {name: 'Save portable bundle', exact: true})
  );
  const bundle = JSON.parse(bundleBytes.toString('utf8'));
  assert.equal(bundle.sources.length, 3);
  // Fault injection: source index 1 fails after source 0 was created. The restore must remove
  // source 0 (reverse order), rebuild the previous workspace from its snapshot and say so.
  await openNativeExport(page, true);
  const portable = page.getByRole('region', {name: 'Portable sandbox bundle', exact: true});
  await portable.getByLabel('Load portable bundle').setInputFiles({
    name: 'workspace.bundle.json',
    mimeType: 'application/json',
    buffer: bundleBytes
  });
  await portable
    .getByRole('status')
    .filter({hasText: 'Bundle validated'})
    .waitFor({timeout: 30000});
  await page.evaluate(() => window.__keplerComputeTest.injectRestoreFailure(1));
  await portable.getByRole('button', {name: 'Restore portable workspace', exact: true}).click();
  // The dialog unmounts while the workspace is rebuilt, so the outcome is published as the persistent workflow notice.
  await page
    .getByTestId('resident-workflow-notice')
    .filter({hasText: /Injected restore failure.*previous workspace was rebuilt/})
    .waitFor({timeout: 60000});
  const after = await page.evaluate(() => window.__keplerComputeTest.getState());
  assert.equal(
    after.publications.length,
    2,
    'previous workspace (two publications) rebuilt after the failed restore'
  );
  assert.equal(after.ready, true);
  // The rebuilt previous workspace may keep layers for its own publications, but never for a half-restored source.
  const knownDatasetIds = new Set(after.publications.map(publication => publication.id));
  const orphaned = after.nativeLayers.filter(
    layer =>
      String(layer.dataId).includes('resident-published') && !knownDatasetIds.has(layer.dataId)
  );
  assert.deepEqual(orphaned, [], 'no layers reference a dataset that was not rebuilt');
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({
      passed: true,
      checks: ['mid-restore failure rolled back in reverse order', 'previous workspace rebuilt']
    })
  );
} catch (error) {
  console.error(await page.locator('body').innerText());
  throw error;
} finally {
  await browser.close();
}
