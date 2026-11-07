// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Native-only export and Stream download: bounded pages, progress, cancel, Blob fallback.
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeExport} from './native-navigation.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
await pinViteModules(page);
// Current Chromium exposes showSaveFilePicker even headless (it rejects as aborted); hide it to exercise the Blob fallback.
await page.addInitScript(() => Object.defineProperty(window, 'showSaveFilePicker', {value: undefined, configurable: true}));
const errors = [];
page.on('pageerror', error => errors.push(error.message));
const rowCount = 60000;
const csv = Buffer.from('latitude,longitude,value,note\n' + Array.from({length: rowCount}, (_, index) => `${40.5 + (index % 50) / 100},${-74 + (index % 70) / 100},${index % 7 === 0 ? '' : index},"n,${index}"`).join('\n'));
async function readDownload(trigger) {
  const pending = page.waitForEvent('download');
  await trigger();
  const chunks = [];
  for await (const chunk of await (await pending).createReadStream()) chunks.push(chunk);
  return Buffer.concat(chunks);
}
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?autoPrepare=0');
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount > 0, undefined, {timeout: 60000});
  await page.getByRole('button', {name: 'Add Data', exact: true}).click();
  await page.locator('input[type=file]').setInputFiles({name: 'stream-native.csv', mimeType: 'text/csv', buffer: csv});
  await page.locator('.ReactModal__Content').getByRole('button', {name: 'Add Data', exact: true}).click();
  await page.waitForTimeout(3000);
  await openNativeExport(page, false);
  const panel = page.getByTestId('streaming-export');
  await panel.getByLabel('Stream dataset').selectOption({label: 'stream-native.csv (native table)'});
  // showSaveFilePicker is hidden by the init script above, so this exercises the Blob fallback.
  await panel.getByLabel('Native rows').selectOption('all');
  const csvBytes = await readDownload(() => panel.getByRole('button', {name: 'Stream download', exact: true}).click());
  const lines = csvBytes.toString('utf8').trimEnd().split('\r\n');
  assert.equal(lines[0], 'latitude,longitude,value,note');
  assert.equal(lines.length, rowCount + 1);
  assert.equal(lines[1].split(',')[2], '', 'null stays empty');
  await panel.getByLabel('Stream format').selectOption('arrow');
  const arrowBytes = await readDownload(() => panel.getByRole('button', {name: 'Stream download', exact: true}).click());
  const table = arrow.tableFromIPC(arrowBytes);
  assert.equal(table.numRows, rowCount);
  assert.deepEqual(table.schema.fields.map(field => field.name), ['latitude', 'longitude', 'value', 'note']);
  assert.equal(table.getChild('value').get(0), null);
  // Cancel publishes nothing.
  await panel.getByLabel('Stream format').selectOption('csv');
  // The 60k-row stream finishes within a few frames, so press Cancel in the same task as the first render of the busy state.
  await page.evaluate(async () => {
    const root = document.querySelector('[data-testid="streaming-export"]');
    const findButton = label => [...root.querySelectorAll('button')].find(button => button.textContent === label);
    findButton('Stream download').click();
    for (let attempt = 0; attempt < 200 && !findButton('Cancel stream'); attempt++) await new Promise(resolve => requestAnimationFrame(resolve));
    findButton('Cancel stream')?.click();
  });
  await panel.getByRole('status').filter({hasText: /cancelled/}).waitFor({timeout: 30000});
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, rowCount, checks: ['native CSV stream', 'native Arrow stream with nulls', 'cancel']}));
} catch (error) {
  console.error(await page.locator('body').innerText());
  throw error;
} finally {await browser.close();}
