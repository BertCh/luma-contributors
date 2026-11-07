// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser check for the bounded resident native Suitability. Written alongside the executor and
// NOT YET RUN: it requires a WebGPU browser and a running dev server, so it is not part of the
// node test run. Panel selectors below follow the pinned Kepler suitability panel and may need
// adjustment on the first real run.
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog, openNativeExport} from './native-navigation.mjs';

// Both numeric fields are missing together in row 3 (a fully missing row scores null); neither is
// constant. The default native configuration weights every numeric field 1 with range
// standardization and normalized weights.
const cost = [10, 20, 15, null, 40, 25];
const access = [3, 1, 7, null, 5, 9];
const sourceTable = new arrow.Table({
  name: arrow.vectorFromArray(['a', 'b', 'c', 'd', 'e', 'f'], new arrow.Utf8()),
  cost: arrow.vectorFromArray(cost, new arrow.Float32()),
  access: arrow.vectorFromArray(access, new arrow.Float32())
});
const sourceLabel = 'native-suitability-source.arrow';
const resultLabel = 'Native GPU suitability result';

function standardize(values, method) {
  const defined = values.filter(value => value !== null);
  if (method === 'range') {
    const minimum = Math.min(...defined);
    const span = Math.max(...defined) - minimum;
    return values.map(value => (value === null ? null : (value - minimum) / span));
  }
  const mean = defined.reduce((sum, value) => sum + value, 0) / defined.length;
  const deviation = Math.sqrt(defined.reduce((sum, value) => sum + (value - mean) ** 2, 0) / defined.length);
  return values.map(value => (value === null ? null : (value - mean) / deviation));
}
function expectedScores(method) {
  const columns = [standardize(cost, method), standardize(access, method)];
  // Normalized equal weights, null when any weighted value is missing.
  return cost.map((_, row) => columns.some(column => column[row] === null) ? null : (columns[0][row] + columns[1][row]) / 2);
}

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
const errors = [];
const sqlRequests = [];
let currentStage = 'startup';
const watchdog = setTimeout(() => {
  console.error(`Native Suitability exceeded 150 seconds at ${currentStage}`);
  void browser.close();
}, 150000);
function stage(name) {
  currentStage = name;
  console.log(JSON.stringify({stage: name}));
}
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text()))
    errors.push(message.text());
});
page.on('request', request => {
  if (/duckdb.*(?:\.wasm|worker)/i.test(request.url())) sqlRequests.push(request.url());
});
async function selectNativeOption(selector, value) {
  await selector.locator('.item-selector__dropdown').click();
  await page.locator('.list-selector:visible').getByText(value, {exact: true}).click();
}
async function exportResult(resultId) {
  await openNativeExport(page);
  await page.getByLabel('Export dataset', {exact: true}).selectOption(resultId);
  await page.getByLabel('Export format', {exact: true}).selectOption('arrow');
  const download = page.waitForEvent('download');
  await page.getByRole('button', {name: 'Export all selected', exact: true}).click();
  const stream = await (await download).createReadStream();
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  await closeNativeDialog(page);
  return arrow.tableFromIPC(Buffer.concat(chunks));
}
function assertScores(table, expected) {
  // Native export appends the source-row index column to every resident table.
  assert.deepEqual(table.schema.fields.map(field => field.name), ['name', 'cost', 'access', 'score', '__luma_source_row_index']);
  assert.deepEqual([...table.getChild('__luma_source_row_index')], [0, 1, 2, 3, 4, 5]);
  const scores = [...table].map(row => row.score);
  assert.equal(scores.length, expected.length);
  for (const [row, value] of expected.entries()) {
    if (value === null) assert.equal(scores[row], null, `row ${row} must score null`);
    else assert.ok(Math.abs(scores[row] - value) < 1e-5, `row ${row}: ${scores[row]} vs ${value}`);
  }
  assert.deepEqual([...table].map(row => row.name), ['a', 'b', 'c', 'd', 'e', 'f']);
}

try {
  await pinViteModules(page);
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  stage('publish numeric source');
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: sourceLabel, mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(sourceTable, 'stream'))
  });
  await workspace.getByLabel('Source geometry column', {exact: true}).selectOption('');
  await workspace.getByLabel('Source longitude column', {exact: true}).selectOption('');
  await workspace.getByLabel('Source latitude column', {exact: true}).selectOption('');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications.some(source => source.rowCount === 6), undefined, {timeout: 30000});
  const sourceId = await page.evaluate(() => window.__keplerComputeTest.getState().publications.find(source => source.rowCount === 6).id);
  await closeNativeDialog(page);

  stage('configure native Suitability: range, normalized weights');
  await page.locator(`.dataset-ops-menu__toggle[data-for="dataset-ops-${sourceId}"]`).click();
  await page.locator('.dataset-ops-menu__suitability').click();
  const panel = page.locator('.dataset-ops-panel');
  await panel.waitFor();
  await panel.locator('input[type=text]').last().fill(resultLabel);
  stage('run GPU Suitability (range)');
  await panel.getByRole('button', {name: 'Run', exact: true}).click();
  await page.waitForFunction(id => window.__keplerComputeTest.getState().publications.some(source => source.id !== id && source.rowCount === 6), sourceId, {timeout: 30000});
  const rangeResultId = await page.evaluate(id => window.__keplerComputeTest.getState().publications.find(source => source.id !== id && source.rowCount === 6).id, sourceId);
  stage('export range result and compare with the native formula');
  const rangeTable = await exportResult(rangeResultId);
  assertScores(rangeTable, expectedScores('range'));
  const lineage = JSON.parse(rangeTable.schema.metadata.get('luma.gl.suitability'));
  assert.equal(lineage.dataStandardization, 'range');
  assert.equal(lineage.source.datasetId, sourceId);

  stage('reconfigure to z-score and rerun');
  await page.locator(`.dataset-ops-menu__toggle[data-for="dataset-ops-${sourceId}"]`).click();
  await page.locator('.dataset-ops-menu__suitability').click();
  const second = page.locator('.dataset-ops-panel');
  await second.waitFor();
  // Selector order follows the native panel: weight standardization, then data standardization.
  await selectNativeOption(second.locator('.item-selector').nth(1), 'Z-score');
  await second.locator('input[type=text]').last().fill(`${resultLabel} z`);
  await second.getByRole('button', {name: 'Run', exact: true}).click();
  await page.waitForFunction(ids => window.__keplerComputeTest.getState().publications.some(source => !ids.includes(source.id) && source.rowCount === 6), [sourceId, rangeResultId], {timeout: 30000});
  const zResultId = await page.evaluate(ids => window.__keplerComputeTest.getState().publications.find(source => !ids.includes(source.id) && source.rowCount === 6).id, [sourceId, rangeResultId]);
  assertScores(await exportResult(zResultId), expectedScores('zScore'));

  assert.deepEqual(sqlRequests, []);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, checks: [
    'native Suitability over prepared Float32 fields with range standardization',
    'z-score standardization rerun publishes a second separate result',
    'fully missing rows score null; carried columns keep source order',
    'result lineage carries the source and standardization',
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
