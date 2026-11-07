// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {openNativeData, closeNativeDialog, openNativeExport} from './native-navigation.mjs';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
const original = new arrow.Table({x: arrow.makeVector(new Float32Array([-74, -74.01, -74.02, -74.03])), y: arrow.makeVector(new Float32Array([40.7, 40.71, 40.72, 40.73])), amount: arrow.makeVector(new Float32Array([10, 20, 30, 40]))});
const source = original.slice(0, 2).concat(original.slice(2));
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
await page.routeWebSocket('**/*', socket => {
  const server = socket.connectToServer();
  server.onMessage(message => {
    if (typeof message === 'string') {
      try {const packet = JSON.parse(message); if (packet.type === 'update' || packet.type === 'full-reload') return;} catch {}
    }
    socket.send(message);
  });
  socket.onMessage(message => server.send(message));
});
const errors = [];
page.on('pageerror', error => {errors.push(error.message); console.error(error.message);});
async function readDownload(button) {
  const pending = page.waitForEvent('download');
  await button.click();
  const stream = await (await pending).createReadStream();
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return Buffer.concat(chunks);
}
try {
  const address = process.argv[2] || 'http://127.0.0.1:5318/?rows=1024';
  await page.goto(address);
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 1024, undefined, {timeout: 60000});
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: 'Portable observations.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))});
  await workspace.getByLabel('Source longitude column').selectOption('x');
  await workspace.getByLabel('Source latitude column').selectOption('y');
  await workspace.getByRole('button', {name: 'Check schema and precision'}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields'}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 4, undefined, {timeout: 45000});
  const id = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);
  const expected = await page.evaluate(id => {
    const api = window.__keplerComputeTest;
    api.setPublishedRange(id, 'amount', 15, 35);
    api.setRange(20, 70);
    return api.source().filter(row => row[2] >= 20 && row[2] <= 70).length;
  }, id);
  await page.waitForFunction(expected => {
    const state = window.__keplerComputeTest.getState();
    return state.selectedCount === expected && state.publications[0]?.selectedCount === 2;
  }, expected);
  await openNativeExport(page, true);
  const bundle = await readDownload(page.getByRole('button', {name: 'Save portable bundle', exact: true}));
  const saved = JSON.parse(bundle.toString('utf8'));
  assert.equal(saved.sources.length, 2);
  assert.deepEqual(saved.state.sources[1].reference.parameters.coordinates, {longitude: 'x', latitude: 'y'});
  // Reload ensures the result is reconstructed from portable data rather than current registry handles.
  await page.reload();
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 1024, undefined, {timeout: 60000});
  await openNativeExport(page, true);
  const portable = page.getByRole('region', {name: 'Portable sandbox bundle', exact: true});
  await portable.getByLabel('Load portable bundle').setInputFiles({name: 'workspace.bundle.json', mimeType: 'application/json', buffer: bundle});
  await portable.getByRole('status').filter({hasText: 'Bundle validated'}).waitFor({timeout: 30000});
  assert.equal(await page.evaluate(() => window.__keplerComputeTest.getState().publications.length), 0);
  await portable.getByRole('button', {name: 'Restore portable workspace', exact: true}).click();
  await page.locator('.ReactModal__Overlay').waitFor({state: 'hidden', timeout: 60000});
  await page.waitForFunction(({id, expected}) => {
    const state = window.__keplerComputeTest.getState();
    return state.selectedCount === expected && state.publications[0]?.id === id && state.publications[0]?.selectedCount === 2;
  }, {id, expected}, {timeout: 45000});
  const state = await page.evaluate(() => window.__keplerComputeTest.getState());
  assert.deepEqual(state.publications[0].sourceBatches, [2, 2]);
  assert.equal(state.error, '');
  // A bad role passes byte integrity but must fail CPU preflight without changing the running sources.
  const invalid = structuredClone(saved);
  invalid.state.sources[1].reference.parameters.coordinates.longitude = 'missing';
  await openNativeExport(page, true);
  await portable.getByLabel('Load portable bundle').setInputFiles({name: 'invalid.bundle.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(invalid))});
  await portable.getByRole('status').filter({hasText: 'Invalid saved coordinate roles'}).waitFor();
  assert.equal(await portable.getByRole('button', {name: 'Restore portable workspace'}).count(), 0);
  const afterInvalid = await page.evaluate(() => window.__keplerComputeTest.getState());
  assert.equal(afterInvalid.selectedCount, expected);
  assert.deepEqual(afterInvalid.publications, state.publications);
  await closeNativeDialog(page);
  await page.locator('.show-data-table svg').nth(1).click();
  await page.getByLabel('Rows', {exact: true}).selectOption('selected');
  await page.waitForFunction(() => JSON.stringify(Array.from(document.querySelectorAll('.resident-table-scroll tbody tr th'), cell => Number(cell.textContent))) === '[1,2]');
  assert.deepEqual(await page.locator('.resident-table-scroll tbody tr td:last-child').allTextContents(), ['20', '30']);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, fixtureSelected: expected, uploadedSelected: 2, checks: ['portable source download', 'fresh-session restore', 'original IDs/coordinate roles/batches', 'native filters and table', 'invalid restore leaves current workspace untouched']}));
} catch (error) {
  console.error(await page.evaluate(() => ({state: window.__keplerComputeTest?.getState(), statuses: Array.from(document.querySelectorAll('[role=status]'), element => element.textContent)})));
  throw error;
} finally {await browser.close();}
