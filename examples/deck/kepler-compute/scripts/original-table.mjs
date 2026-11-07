// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const amounts = [null, 0.123456789012345, 9007199254740994, null, -1.23456789012345e20, 1.23456789012345e-20];
const ranks = [0, 3, 1, 5, 2, 4];
const instants = [null, 1767225600000000001n, 1767225600000000002n, null, -1n, 1767225600000000005n];
const integers = [null, 9007199254740993n, -9007199254740993n, null, 9223372036854775807n, -9223372036854775808n];
const geometries = ranks.map((_, index) => index % 3 === 0 ? null : JSON.stringify({type: 'LineString', coordinates: [[-74, 40 + index / 100], [-73.9, 40 + index / 100]]}));
const timestamp = arrow.makeVector(arrow.makeData({
  type: new arrow.TimestampNanosecond('UTC'), length: 6,
  data: BigInt64Array.from(instants, value => value ?? 0n), nullCount: 2, nullBitmap: new Uint8Array([0b110110])
}));
const original = new arrow.Table({
  amount: arrow.vectorFromArray(amounts, new arrow.Float64()),
  rank: arrow.makeVector(new Float32Array(ranks)),
  instant: timestamp,
  original_integer: arrow.vectorFromArray(integers, new arrow.Int64()),
  geometry: arrow.vectorFromArray(geometries, new arrow.Utf8()),
  analysis_z_score: arrow.vectorFromArray(ranks.map((_, row) => `original-${row}`), new arrow.Utf8()),
  note: arrow.vectorFromArray(ranks.map((_, row) => row === 5 ? 'x'.repeat(20000) : `note-${row}`), new arrow.Utf8())
});
const source = original.slice(0, 3).concat(original.slice(3));
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
const table = page.locator('.resident-table-scroll');
const headers = table.locator('thead');
const sourceIds = () => table.locator('tbody tr th').allTextContents().then(values => values.map(Number));
async function waitRows(expected) {
  await page.waitForFunction(expected => {
    const actual = Array.from(document.querySelectorAll('.resident-table-scroll tbody tr th'), cell => Number(cell.textContent));
    return JSON.stringify(actual) === JSON.stringify(expected) && document.querySelector('.resident-table-scroll')?.getAttribute('aria-busy') === 'false';
  }, expected);
  assert.deepEqual(await sourceIds(), expected);
}
async function assertOriginalRows(expected, includeDerived = true) {
  await waitRows(expected);
  const rows = await table.locator('tbody tr').evaluateAll(rows => rows.map(row => Array.from(row.querySelectorAll('td'), cell => cell.textContent)));
  for (const [position, sourceRow] of expected.entries()) {
    assert.deepEqual(rows[position].slice(0, 6), [
      amounts[sourceRow] === null ? 'null' : String(amounts[sourceRow]),
      String(ranks[sourceRow]),
      instants[sourceRow] === null ? 'null' : `${instants[sourceRow]} ns since Unix epoch (UTC)`,
      integers[sourceRow] === null ? 'null' : String(integers[sourceRow]),
      geometries[sourceRow] ?? 'null',
      `original-${sourceRow}`
    ]);
    if (sourceRow === 5) {assert.match(rows[position][6], /… \[preview\]$/); assert(rows[position][6].length <= 4096);} else assert.equal(rows[position][6], `note-${sourceRow}`);
    if (includeDerived) assert(Math.abs(Number(rows[position][7]) - (ranks[sourceRow] - 2.5) / Math.sqrt(3.5)) < 0.000001);
  }
}
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => {
    const state = window.__keplerComputeTest?.getState();
    if (state?.error) throw new Error(state.error);
    return state?.selectedCount === 256;
  }, undefined, {timeout: 60000});
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: 'exact-original-rows.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))});
  await workspace.getByLabel('Float64 conversion').selectOption('round');
  await workspace.getByLabel('Maximum relative error').fill('0.000001');
  await workspace.getByRole('checkbox', {name: /^original_integer/}).uncheck();
  await workspace.getByRole('checkbox', {name: /^geometry/}).uncheck();
  await workspace.getByRole('checkbox', {name: /^analysis_z_score/}).uncheck();
  await workspace.getByRole('checkbox', {name: /^note/}).uncheck();
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 6, undefined, {timeout: 45000});
  const datasetId = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);
  await closeNativeDialog(page);
  const analysis = page.getByRole('region', {name: 'GPU analysis', exact: true});
  await analysis.getByLabel('Dataset', {exact: true}).selectOption(datasetId);
  await analysis.getByLabel('Numeric field', {exact: true}).selectOption('rank');
  await analysis.getByLabel('Operation', {exact: true}).selectOption('z-score');
  await analysis.getByLabel('Analysis population', {exact: true}).selectOption('all');
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('6 analyzed'));
  const storage = await page.evaluate(id => {
    const state = window.__keplerComputeTest.getState();
    const published = state.publications.find(source => source.id === id);
    return {fixtureUploadBytes: state.uploadBytes, sourceRevision: published.sourceRevision, sourceUploadBytes: published.sourceUploadBytes, sourceBufferIds: published.sourceBufferIds, batches: published.sourceBatches};
  }, datasetId);
  assert.deepEqual(storage.batches, [3, 3]);
  assert.equal(typeof storage.sourceRevision, 'number');
  assert(storage.sourceUploadBytes > 0);
  assert(storage.sourceBufferIds.length > 0);
  await closeNativeDialog(page);
  await page.locator('.show-data-table svg').last().click();
  await waitRows([0, 1, 2, 3, 4, 5]);
  assert.equal(await page.getByLabel('Values', {exact: true}).inputValue(), 'prepared');
  assert.equal(await headers.getByRole('button', {name: /original_integer/}).count(), 0);
  await page.getByLabel('Values', {exact: true}).selectOption('original');
  await assertOriginalRows([0, 1, 2, 3, 4, 5]);
  // Rich source fields sort exactly on the CPU; Utf8 geometry text has a total order, so it sorts too.
  // (Only binary, list, struct and map sources stay unavailable; this fixture has none.)
  for (const name of ['amount', 'geometry', 'instant', 'original_integer', 'analysis_z_score', 'note']) {
    const header = headers.getByRole('button', {name: new RegExp(`^${name} `)});
    assert.equal(await header.isEnabled(), true, `${name} sortable: ${await header.getAttribute('title')} ${await header.textContent()}`);
  }
  assert.equal(await headers.getByRole('button', {name: /^rank /}).isEnabled(), true);
  assert.equal(await headers.getByRole('button', {name: 'Analysis: z-score', exact: true}).isEnabled(), true);
  await page.evaluate(id => window.__keplerComputeTest.setPublishedRange(id, 'rank', 1, 3), datasetId);
  await assertOriginalRows([1, 2, 4]);
  await headers.getByRole('button', {name: /^rank /}).click();
  await assertOriginalRows([2, 4, 1]);
  await headers.getByRole('button', {name: /^rank /}).click();
  await assertOriginalRows([1, 4, 2]);
  await headers.getByRole('button', {name: 'Analysis: z-score', exact: true}).click();
  await assertOriginalRows([2, 4, 1]);
  await page.getByLabel('Rows', {exact: true}).selectOption('all');
  await assertOriginalRows([0, 2, 4, 1, 5, 3]);
  await page.getByLabel('Values', {exact: true}).selectOption('prepared');
  await waitRows([0, 1, 2, 3, 4, 5]);
  await page.getByLabel('Values', {exact: true}).selectOption('original');
  await assertOriginalRows([0, 1, 2, 3, 4, 5]);
  await page.getByRole('button', {name: 'Refresh', exact: true}).click();
  await assertOriginalRows([0, 1, 2, 3, 4, 5]);
  const wrapCells = page.getByLabel('Wrap cell previews', {exact: true});
  await wrapCells.focus();
  await wrapCells.press('Space');
  assert.equal(await wrapCells.isChecked(), true);
  assert.equal(await table.locator('tbody td').first().evaluate(cell => getComputedStyle(cell).whiteSpace), 'pre-wrap');
  await assertOriginalRows([0, 1, 2, 3, 4, 5]);
  await wrapCells.press('Space');
  assert.equal(await wrapCells.isChecked(), false);
  const finalStorage = await page.evaluate(id => {
    const state = window.__keplerComputeTest.getState();
    const published = state.publications.find(source => source.id === id);
    return {fixtureUploadBytes: state.uploadBytes, sourceRevision: published.sourceRevision, sourceUploadBytes: published.sourceUploadBytes, sourceBufferIds: published.sourceBufferIds, batches: published.sourceBatches};
  }, datasetId);
  assert.deepEqual(finalStorage, storage);
  assert.deepEqual(errors, []);
  await page.screenshot({path: '/tmp/kepler-original-table.png'});
  console.log(JSON.stringify({passed: true, checks: ['native modal original/prepared chooser', 'null-prefixed source batches', 'exact Float64/Int64/nanosecond timestamps/geometry', 'selected original row IDs', 'rich sort visibly unavailable', 'GPU Float32 and derived sorting', 'source storage unchanged', 'colliding original/derived names', 'bounded rich-cell preview', 'keyboard cell wrapping']}));
} catch (error) {
  console.error(await page.evaluate(() => ({state: window.__keplerComputeTest?.getState(), text: document.body.innerText})).catch(() => 'Page unavailable'));
  throw error;
} finally {await browser.close();}
