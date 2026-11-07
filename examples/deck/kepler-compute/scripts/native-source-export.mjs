// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {openNativeData, openNativeExport} from './native-navigation.mjs';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
const precise = [0.1, 0.123456789012345, 0.3, 0.4];
const timestamps = BigInt64Array.from([0n, 1700000000123456789n, 1700000000987654321n, -1n]);
const source = new arrow.Table({amount: arrow.vectorFromArray(precise, new arrow.Float64()), rank: arrow.makeVector(new Float32Array([1, 2, 3, 4])), instant: arrow.makeVector(arrow.makeData({type: new arrow.TimestampNanosecond('UTC'), data: timestamps})), geometry: arrow.vectorFromArray([{x: 0}, {x: 1}, {x: 2}, {x: 3}])});
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
await page.routeWebSocket('**/*', socket => {
  const server = socket.connectToServer();
  server.onMessage(message => {if (typeof message === 'string') {try {const packet = JSON.parse(message); if (packet.type === 'update' || packet.type === 'full-reload') return;} catch {}} socket.send(message);});
  socket.onMessage(message => server.send(message));
});
const errors = [];
page.on('pageerror', error => {errors.push(error.message); console.error(error.message);});
async function download(button) {
  const pending = page.waitForEvent('download');
  await button.click();
  const stream = await (await pending).createReadStream();
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return Buffer.concat(chunks);
}
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=1024');
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 1024, undefined, {timeout: 60000});
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: 'original-values.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(source.slice(0, 2).concat(source.slice(2)), 'stream'))});
  await workspace.getByLabel('Float64 conversion').selectOption('round');
  await workspace.getByRole('checkbox', {name: /^geometry/}).uncheck();
  await workspace.getByRole('button', {name: 'Check schema and precision'}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields'}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 4, undefined, {timeout: 45000});
  const id = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);
  await page.evaluate(id => window.__keplerComputeTest.setPublishedRange(id, 'rank', 2, 3), id);
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 2);
  await openNativeExport(page);
  await page.getByLabel('Export dataset', {exact: true}).selectOption(id);
  await page.getByLabel('Export format', {exact: true}).selectOption('arrow');
  const table = arrow.tableFromIPC(await download(page.getByRole('button', {name: 'Export all selected', exact: true})));
  assert.deepEqual([...table.getChild('amount')], precise.slice(1, 3));
  assert.deepEqual([...table.getChild('__luma_source_row_index')], [1, 2]);
  assert.deepEqual(table.getChild('instant').data.flatMap(data => Array.from(data.values, String)), ['1700000000123456789', '1700000000987654321']);
  // Nested Arrow types are retained as JSON text by the source type routes (ARROW-1), not as structs.
  assert.deepEqual([...table.getChild('geometry')].map(value => JSON.parse(value).x), [1, 2]);
  assert.equal(JSON.parse(table.schema.metadata.get('luma:source-export')).datasetId, id);
  await page.getByLabel('Export format', {exact: true}).selectOption('csv');
  const csv = (await download(page.getByRole('button', {name: 'Export all selected', exact: true}))).toString('utf8');
  assert.match(csv, /0\.123456789012345/);
  assert.match(csv, /2023-11-14T22:13:20\.123456789Z/);
  await page.getByLabel('Export format', {exact: true}).selectOption('prepared-json');
  const prepared = JSON.parse((await download(page.getByRole('button', {name: 'Export all selected', exact: true}))).toString('utf8'));
  assert.equal(prepared.datasetId, id);
  assert.equal(prepared.rowCount, 2);
  assert.equal(prepared.columns.amount[0], Math.fround(precise[1]));
  assert.equal(prepared.columns.geometry, undefined);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, selected: 2, checks: ['dataset export selector', 'original Arrow values/types/geometry/nanosecond time', 'original CSV text', 'prepared JSON retains explicit rounding contract']}));
} catch (error) {
  console.error(await page.evaluate(() => ({state: window.__keplerComputeTest?.getState(), statuses: Array.from(document.querySelectorAll('[role=status]'), element => element.textContent)})));
  throw error;
} finally {await browser.close();}
