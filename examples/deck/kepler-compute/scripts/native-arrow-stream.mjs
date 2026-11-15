// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getSourceCheckUrl} from './source-check-utils.mjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const batches = [[1, 2], [], [3, 4, 5]].map(values => {
  const table = new arrow.Table({
    amount: arrow.vectorFromArray(
      values.map(value => (value === 2 ? null : value)),
      new arrow.Float32()
    ),
    category: arrow.vectorFromArray(
      values.map(value => (value < 3 ? 'initial' : 'later')),
      new arrow.Utf8()
    )
  });
  return arrow.tableFromIPC(arrow.tableToIPC(table, 'stream')).batches[0];
});
const bytes = Buffer.from(arrow.tableToIPC(new arrow.Table(batches[0].schema, batches), 'stream'));
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
page.setDefaultTimeout(30000);
const watchdog = setTimeout(async () => {
  console.error('Native stream fixture exceeded three minutes');
  await Promise.race([browser.close(), new Promise(resolve => setTimeout(resolve, 5000))]);
  process.exit(1);
}, 180000);
await pinViteModules(page);
const failures = [];
page.on('pageerror', error => failures.push(error.message));
try {
  await page.goto(getSourceCheckUrl(process.argv[2], {rows: 256}), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(
    () => window.__keplerComputeTest?.getState().selectedCount === 256,
    undefined,
    {timeout: 60000}
  );
  await openNativeData(page);
  const panel = page.getByTestId('arrow-stream-panel');
  await panel.getByLabel('Arrow IPC stream file').setInputFiles({
    name: 'streamed-observations.arrows',
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: bytes
  });
  await panel.getByRole('button', {name: 'Open stream', exact: true}).click();
  await panel.getByRole('status').filter({hasText: 'paused: 1 batches, 2 admitted rows'}).waitFor();
  assert.equal(
    await panel.getByRole('button', {name: 'Finish stream', exact: true}).isDisabled(),
    true
  );
  assert.equal(
    await panel.getByRole('button', {name: 'Add completed stream', exact: true}).count(),
    0
  );
  assert.equal(
    (await page.evaluate(() => window.__keplerComputeTest.getState())).publications.length,
    0
  );
  assert.match(await panel.innerText(), /Incomplete source preview/);
  await panel.getByRole('button', {name: 'Read next batch', exact: true}).click();
  await panel.getByRole('status').filter({hasText: 'paused: 2 batches, 2 admitted rows'}).waitFor();
  await panel.getByRole('button', {name: 'Continue stream', exact: true}).click();
  await panel
    .getByRole('status')
    .filter({hasText: 'exhausted: 3 batches, 5 admitted rows'})
    .waitFor();
  await panel.getByRole('button', {name: 'Finish stream', exact: true}).click();
  await panel
    .getByRole('status')
    .filter({hasText: 'complete: 3 batches, 5 admitted rows'})
    .waitFor();
  await panel.getByRole('button', {name: 'Add completed stream', exact: true}).click();
  // Native addDataToMap closes its modal; successful transfer must survive that unmount.
  await page.locator('.ReactModal__Overlay').waitFor({state: 'hidden'});
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 5
  );
  const first = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0]);
  assert.deepEqual(first.sourceBatches, [2, 0, 3]);
  await closeNativeDialog(page);
  await page.evaluate(
    id => window.__keplerComputeTest.setPublishedRange(id, 'amount', 3, 4),
    first.id
  );
  await page.waitForFunction(
    id =>
      window.__keplerComputeTest.getState().publications.find(source => source.id === id)
        ?.selectedCount === 2,
    first.id
  );
  const filtered = await page.evaluate(
    id => window.__keplerComputeTest.getState().publications.find(source => source.id === id),
    first.id
  );
  assert.deepEqual(filtered.sourceBufferIds, first.sourceBufferIds);
  await openNativeData(page);
  await panel.getByLabel('Arrow IPC stream file').setInputFiles({
    name: 'cancelled.arrows',
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: bytes
  });
  await panel.getByRole('button', {name: 'Open stream', exact: true}).click();
  await panel.getByRole('status').filter({hasText: 'paused: 1 batches, 2 admitted rows'}).waitFor();
  await panel.getByRole('button', {name: 'Cancel stream', exact: true}).click();
  await panel.getByRole('status').filter({hasText: 'cancelled:'}).waitFor();
  // Explicit partial persistence: confirmation is required and the prefix is labelled partial.
  await openNativeData(page);
  await panel.getByLabel('Arrow IPC stream file').setInputFiles({
    name: 'partial.arrows',
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: bytes
  });
  await panel.getByRole('button', {name: 'Open stream', exact: true}).click();
  await panel.getByRole('status').filter({hasText: 'paused: 1 batches, 2 admitted rows'}).waitFor();
  assert.equal(
    await panel.getByRole('button', {name: 'Save incomplete prefix', exact: true}).isDisabled(),
    true
  );
  await panel.getByLabel('I understand this is an incomplete prefix').check();
  await panel.getByRole('button', {name: 'Save incomplete prefix', exact: true}).click();
  await panel
    .getByRole('status')
    .filter({hasText: 'complete: 1 batches, 2 admitted rows'})
    .waitFor();
  assert.match(await panel.getByRole('note').innerText(), /Partial population: 2 rows, 1 batches/);
  await panel.getByRole('button', {name: 'Add completed stream', exact: true}).click();
  await page.locator('.ReactModal__Overlay').waitFor({state: 'hidden'});
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications.length === 2);
  // Remote URL ingestion shares the same controller; abort/cancel leaves nothing published.
  await page.route('**/remote-fixture.arrows', route =>
    route.fulfill({status: 200, contentType: 'application/vnd.apache.arrow.stream', body: bytes})
  );
  await openNativeData(page);
  await panel
    .getByLabel(/Remote Arrow IPC stream URL/)
    .fill(new URL('/remote-fixture.arrows', page.url()).href);
  await panel.getByRole('button', {name: 'Open stream', exact: true}).click();
  await panel.getByRole('status').filter({hasText: 'paused: 1 batches, 2 admitted rows'}).waitFor();
  await panel.getByRole('button', {name: 'Cancel stream', exact: true}).click();
  await panel.getByRole('status').filter({hasText: 'cancelled:'}).waitFor();
  await closeNativeDialog(page);
  const final = await page.evaluate(() => window.__keplerComputeTest.getState());
  assert.equal(final.publications.length, 2);
  assert.equal(final.publications[0].id, first.id);
  assert.equal(final.publications[0].selectedCount, 2);
  assert.deepEqual(final.publications[0].sourceBufferIds, first.sourceBufferIds);
  assert.equal(Boolean(final.error), false);
  assert.deepEqual(failures, []);
  await page.screenshot({path: '/tmp/luma-kepler-native-stream.png'});
  console.log(
    JSON.stringify({
      passed: true,
      checks: [
        'native IPC stream ingestion',
        'first-batch pause and bounded incomplete preview',
        'preserved empty batch [2,0,3]',
        'explicit EOF finish',
        'native completed publication without source reupload',
        'nullable filtering',
        'transferred ownership survives panel close',
        'cancelled prefix stays unpublished',
        'confirmed partial prefix recorded and published',
        'remote URL stream opens and cancels without publication'
      ]
    })
  );
} catch (error) {
  console.error(
    await Promise.race([
      page.evaluate(() =>
        JSON.stringify({
          state: window.__keplerComputeTest?.getState(),
          statuses: [...document.querySelectorAll('[role=status]')].map(
            element => element.textContent
          )
        })
      ),
      new Promise(resolve => setTimeout(() => resolve('diagnostics timed out'), 5000))
    ])
  );
  throw error;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
