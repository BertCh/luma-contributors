// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser check for the bounded resident native Join. Written alongside the executor; it requires
// a WebGPU browser and a running dev server, so it is not part of the node test run.
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog, openNativeExport} from './native-navigation.mjs';

// Left keys repeat, include null and empty keys; right keys repeat and include an unmatched label.
const leftTable = new arrow.Table({
  key: arrow.vectorFromArray(['b', null, 'a', '', 'b', 'z'], new arrow.Utf8()),
  amount: arrow.vectorFromArray([1, 2, 3, 4, 5, 6], new arrow.Float32())
});
const rightTable = new arrow.Table({
  key: arrow.vectorFromArray(['a', 'b', null, 'a', 'x', 'b'], new arrow.Utf8()),
  weight: arrow.vectorFromArray([10, 20, 30, 40, 50, 60], new arrow.Float32())
});
const leftLabel = 'native-join-left.arrow';
const rightLabel = 'native-join-right.arrow';
const resultLabel = 'Native GPU join result';
// Native FULL join: left order, right matches in right order, then unmatched right rows.
const expectedRows = [
  ['b', 1, 'b', 20],
  ['b', 1, 'b', 60],
  [null, 2, null, null],
  ['a', 3, 'a', 10],
  ['a', 3, 'a', 40],
  ['', 4, null, null],
  ['b', 5, 'b', 20],
  ['b', 5, 'b', 60],
  ['z', 6, null, null],
  [null, null, null, 30],
  [null, null, 'x', 50]
];
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
const errors = [];
const sqlRequests = [];
let currentStage = 'startup';
const watchdog = setTimeout(() => {
  console.error(`Native Join exceeded 120 seconds at ${currentStage}`);
  void browser.close();
}, 120000);
function stage(name) {
  currentStage = name;
  console.log(JSON.stringify({stage: name}));
}
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error|exceeds the maximum/i.test(message.text()))
    errors.push(message.text());
});
page.on('request', request => {
  if (/duckdb.*(?:\.wasm|worker)/i.test(request.url())) sqlRequests.push(request.url());
});
async function publish(workspace, name, table) {
  await workspace.locator('input[type=file]').setInputFiles({
    name, mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(table, 'stream'))
  });
  await workspace.getByLabel('Source geometry column', {exact: true}).selectOption('');
  await workspace.getByLabel('Source longitude column', {exact: true}).selectOption('');
  await workspace.getByLabel('Source latitude column', {exact: true}).selectOption('');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
}
async function selectNativeOption(selector, value) {
  await selector.locator('.item-selector__dropdown').click();
  await page.locator('.list-selector:visible').getByText(value, {exact: true}).click();
}

try {
  await pinViteModules(page);
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  stage('publish two dictionary sources');
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await publish(workspace, leftLabel, leftTable);
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications.some(source => source.rowCount === 6), undefined, {timeout: 30000});
  // Native addDataToMap closes the Add Data dialog after each publication; reopen for the second.
  await openNativeData(page);
  await publish(workspace, rightLabel, rightTable);
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications.filter(source => source.rowCount === 6).length >= 2, undefined, {timeout: 30000});
  const sources = await page.evaluate(() => window.__keplerComputeTest.getState().publications.filter(source => source.rowCount === 6));
  const leftId = sources[0].id;
  await closeNativeDialog(page);
  stage('configure native Join: FULL on key');
  await page.locator(`.dataset-ops-menu__toggle[data-for="dataset-ops-${leftId}"]`).click();
  await page.locator('.dataset-ops-menu__join').click();
  const panel = page.locator('.dataset-ops-panel');
  await panel.waitFor();
  // The native panel shows key selectors only after the right dataset is chosen: right dataset,
  // then left key, then right key.
  await selectNativeOption(panel.locator('.data-source-selector').last(), rightLabel);
  await selectNativeOption(panel.locator('.field-selector').nth(0), 'key');
  await selectNativeOption(panel.locator('.field-selector').nth(1), 'key');
  await panel.getByLabel('FULL', {exact: true}).click();
  await panel.locator('input[type=text]').last().fill(resultLabel);
  stage('run GPU Join');
  await panel.getByRole('button', {name: 'Run', exact: true}).click();
  await page.waitForFunction(ids => window.__keplerComputeTest.getState().publications.some(source => !ids.includes(source.id) && source.rowCount === 11), sources.map(source => source.id), {timeout: 30000});
  const resultId = await page.evaluate(ids => window.__keplerComputeTest.getState().publications.find(source => !ids.includes(source.id)).id, sources.map(source => source.id));
  stage('export result Arrow and compare with native row order and null semantics');
  await openNativeExport(page);
  await page.getByLabel('Export dataset', {exact: true}).selectOption(resultId);
  await page.getByLabel('Export format', {exact: true}).selectOption('arrow');
  const download = page.waitForEvent('download');
  await page.getByRole('button', {name: 'Export all selected', exact: true}).click();
  const stream = await (await download).createReadStream();
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  const table = arrow.tableFromIPC(Buffer.concat(chunks));
  const names = table.schema.fields.map(field => field.name);
  assert.deepEqual(names.slice(0, 4), ['key', 'amount', `${rightLabel}_key`, `${rightLabel}_weight`]);
  const rows = [...table].map(row => names.slice(0, 4).map(name => row[name]));
  assert.deepEqual(rows, expectedRows);
  assert.equal(JSON.parse(table.schema.metadata.get('luma.gl.join')).joinType, 'FULL');
  await closeNativeDialog(page);
  assert.deepEqual(sqlRequests, []);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, checks: [
    'native FULL Join over prepared string dictionary keys',
    'left order, duplicate right matches in right order, unmatched right rows appended',
    'null and empty keys never match',
    'output names use the right dataset label prefix',
    'no SQL runtime started'
  ]}));
} catch (error) {
  console.error(JSON.stringify({stage: currentStage, errors, sqlRequests,
    body: (await page.locator('body').innerText().catch(() => '')).slice(-4000)}));
  throw error;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
