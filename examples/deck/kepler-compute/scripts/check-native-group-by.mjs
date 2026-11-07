// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog, openNativeExport} from './native-navigation.mjs';

// Dictionary order deliberately differs from first source occurrence; the unused label must not emit a row.
const dictionaryType = new arrow.Dictionary(new arrow.Utf8(), new arrow.Int32(), 41, false);
const category = arrow.makeVector(arrow.makeData({
  type: dictionaryType,
  data: new Int32Array([2, 0, 2, 0, 1, 0, 0, 1]),
  dictionary: arrow.vectorFromArray(['a', 'empty', 'b', 'unused'], new arrow.Utf8()),
  nullBitmap: new Uint8Array([0xb7]),
  nullCount: 2,
  length: 8
}));
const original = new arrow.Table({
  category,
  counted: arrow.vectorFromArray([null, 1, 2, null, null, null, 6, null], new arrow.Float32()),
  amount: arrow.vectorFromArray([2, null, 3, 7, null, 5, null, null], new arrow.Float32())
});
const source = original.slice(0, 3).concat(original.slice(3));
const sourceLabel = 'native-group-by.arrow';
const resultLabel = 'Native GPU group result';
const expectedKeys = ['b', 'a', null, 'empty'];
const expectedFields = ['category', 'counted_count', 'amount_sum'];
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
const errors = [];
const diagnostics = [];
const sqlRequests = [];
let currentStage = 'startup';
const watchdog = setTimeout(() => {
  console.error(`Native Group By exceeded 120 seconds at ${currentStage}`);
  void browser.close();
}, 120000);
function stage(name) {
  currentStage = name;
  console.log(JSON.stringify({stage: name}));
}
page.on('pageerror', error => {
  errors.push(error.message);
  console.error('pageerror:', error.stack ?? error.message);
});
page.on('console', message => {
  if (message.type() !== 'error') return;
  diagnostics.push(message.text());
  if (/WGSL|Invalid.*(?:Pipeline|Buffer)|validation error|Maximum call stack/i.test(message.text())) errors.push(message.text());
});
page.on('request', request => {
  if (/duckdb.*(?:\.wasm|worker)/i.test(request.url())) sqlRequests.push(request.url());
});

async function readDownload(button) {
  const pending = page.waitForEvent('download');
  await button.click();
  const download = await pending;
  assert.equal(await download.failure(), null);
  const stream = await download.createReadStream();
  assert(stream);
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return Buffer.concat(chunks);
}
async function selectNativeOption(selector, value) {
  await selector.locator('.item-selector__dropdown').click();
  await page.locator('.list-selector:visible').getByText(value, {exact: true}).click();
}
function aggregationRow(panel, name) {
  return panel.locator('.dataset-ops-column-list > div').filter({has: page.getByText(name, {exact: true})});
}
async function selectAggregation(panel, name, technique) {
  const row = aggregationRow(panel, name);
  stage(`enable native aggregation field ${name}`);
  await row.locator('input[type=checkbox]').check();
  stage(`select native ${name} aggregation ${technique}`);
  await selectNativeOption(row.locator('.dataset-ops-aggregation-selector'), technique);
}
async function inspectResult(resultId) {
  await closeNativeDialog(page);
  await page.locator(`.show-data-table[data-for="data-table-${resultId}"] svg`).click();
  await page.waitForFunction(() => document.querySelectorAll('.resident-table-scroll tbody tr').length === 4);
  const result = await page.locator('.resident-table-scroll').evaluate(table => {
    const headers = [...table.querySelectorAll('thead th')].map(header => header.textContent);
    const names = ['category', 'counted_count', 'amount_sum'];
    const indexes = names.map(name => headers.findIndex(header => header.includes(name)));
    return {indexes, rows: [...table.querySelectorAll('tbody tr')].map(row => ({
      sourceRow: row.querySelector('th')?.textContent,
      values: indexes.map(index => row.children[index]?.textContent)
    }))};
  });
  assert(result.indexes.every(index => index >= 0), 'Native result table must expose exact output field names');
  assert.deepEqual(result.rows.map(row => row.sourceRow), ['0', '1', '2', '3']);
  assert.deepEqual(result.rows.map(row => row.values), [
    ['b', '2', '5'], ['a', '2', '5'], ['null', '2', '7'], ['empty', '2', 'null']
  ]);
  await page.keyboard.press('Escape');
  await page.locator('.resident-table-modal').waitFor({state: 'hidden'});
}
async function exportResult(resultId) {
  await openNativeExport(page);
  await page.getByLabel('Export dataset', {exact: true}).selectOption(resultId);
  await page.getByLabel('Export format', {exact: true}).selectOption('arrow');
  const table = arrow.tableFromIPC(await readDownload(page.getByRole('button', {name: 'Export all selected', exact: true})));
  assert.deepEqual(table.schema.fields.map(field => field.name), [...expectedFields, '__luma_source_row_index']);
  assert.deepEqual([...table.getChild('category')], expectedKeys);
  // Native count counts every row, including a null named measure; sum ignores nulls.
  assert.deepEqual([...table.getChild('counted_count')], [2, 2, 2, 2]);
  assert.deepEqual([...table.getChild('amount_sum')], [5, 5, 7, null]);
  assert.deepEqual([...table.getChild('__luma_source_row_index')], [0, 1, 2, 3]);
  const provenance = JSON.parse(table.schema.metadata.get('luma.gl.groupBy'));
  assert.equal(provenance.kind, 'group-by');
  assert.equal(provenance.resultId, resultId);
  assert.equal(provenance.resultLabel, resultLabel);
  assert.equal(provenance.population, 'all');
  assert.equal(provenance.rowOrder, 'first-source-occurrence');
  assert.equal(provenance.numericalPolicy, 'float32');
  assert.equal(provenance.keyColumn, 'category');
  assert.deepEqual(provenance.aggregations, [
    {column: 'counted', technique: 'count', name: 'counted_count'},
    {column: 'amount', technique: 'sum', name: 'amount_sum'}
  ]);
  assert.deepEqual(provenance.fields.map(field => [field.name, field.type]), [
    ['category', 'string'], ['counted_count', 'integer'], ['amount_sum', 'real']
  ]);
  await closeNativeDialog(page);
  return provenance;
}

try {
  await pinViteModules(page);
  const address = process.argv[2] || 'http://127.0.0.1:5318/?rows=256';
  await page.goto(address, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  stage('prepare dictionary and nullable Float32 source through native Add Data');
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: sourceLabel, mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))
  });
  await workspace.getByLabel('Source geometry column', {exact: true}).selectOption('');
  await workspace.getByLabel('Source longitude column', {exact: true}).selectOption('');
  await workspace.getByLabel('Source latitude column', {exact: true}).selectOption('');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 8, undefined, {timeout: 30000});
  const initial = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0]);
  const sourceId = initial.id;
  assert.deepEqual(initial.sourceBatches, [3, 5]);
  await closeNativeDialog(page);
  stage('configure native Group By with an unsupported aggregation and recover in place');
  await page.locator(`.dataset-ops-menu__toggle[data-for="dataset-ops-${sourceId}"]`).click();
  await page.locator('.dataset-ops-menu__group-by').click();
  const panel = page.locator('.dataset-ops-panel');
  await panel.waitFor();
  stage('select native Group By field category');
  await selectNativeOption(panel.locator('.field-selector'), 'category');
  stage('expand native aggregation rules');
  // The header includes a nested help tooltip, so its complete text is not exactly the title.
  // Click the header's left edge rather than its help button, which stops click propagation.
  await panel.locator('.dataset-ops-collapsible__header').click({position: {x: 8, y: 8}});
  await panel.locator('.dataset-ops-column-list').waitFor();
  // Keep native row order when replacing default techniques rather than recreating controls.
  await selectAggregation(panel, 'counted', 'count');
  await selectAggregation(panel, 'amount', 'median');
  stage('name native Group By result and run unsupported median');
  const resultName = panel.locator('.dataset-ops-panel__body > :last-child input[type=text]');
  await resultName.fill(resultLabel);
  await panel.getByRole('button', {name: 'Run', exact: true}).click();
  await panel.getByText(/Group By aggregation median is unavailable on the GPU/).waitFor({timeout: 30000});
  assert.match(await panel.innerText(), /Use SQL for a complementary query/);
  assert.equal(await resultName.inputValue(), resultLabel);
  assert(await panel.getByRole('button', {name: 'Run', exact: true}).isEnabled());
  assert.equal(await page.evaluate(() => window.__keplerComputeTest.getState().publications.length), 1);
  await selectAggregation(panel, 'amount', 'sum');
  stage('run GPU Group By and inspect source-order keys and native count/sum null semantics');
  await panel.getByRole('button', {name: 'Run', exact: true}).click();
  await page.waitForFunction(id => window.__keplerComputeTest.getState().publications.some(source => source.id !== id && source.rowCount === 4), sourceId, {timeout: 30000});
  await panel.waitFor({state: 'hidden'});
  // Native hides Add Data while an operation panel is open, so the SQL route is checked once it closes.
  await openNativeData(page, 'SQL');
  const sql = page.getByRole('region', {name: 'Native SQL Data Explorer', exact: true});
  assert.equal((await sql.getByRole('status').innerText()).trim(), 'disabled');
  // DuckDB now initializes lazily when a CPU-routed statement runs, so only Run SQL is exposed.
  assert(await sql.getByRole('button', {name: 'Run SQL', exact: true}).isEnabled());
  assert.deepEqual(sqlRequests, []);
  await closeNativeDialog(page);
  const resultId = await page.evaluate(id => window.__keplerComputeTest.getState().publications.find(source => source.id !== id).id, sourceId);
  await inspectResult(resultId);
  const provenance = await exportResult(resultId);
  assert.deepEqual(provenance.source, {datasetId: sourceId, revision: initial.sourceRevision});
  const unchanged = await page.evaluate(id => window.__keplerComputeTest.getState().publications.find(source => source.id === id), sourceId);
  for (const field of ['sourceRevision', 'sourceUploadBytes', 'sourceBufferIds', 'sourceBatches'])
    assert.deepEqual(unchanged[field], initial[field], `Group By changed source ${field}`);
  stage('remove the original source while retaining the completed grouped snapshot');
  await openNativeData(page);
  await workspace.getByRole('button', {name: `Remove ${sourceLabel}`, exact: true}).click();
  await page.waitForFunction(({sourceId, resultId}) => {
    const sources = window.__keplerComputeTest.getState().publications;
    return !sources.some(source => source.id === sourceId) && sources.some(source => source.id === resultId && source.rowCount === 4);
  }, {sourceId, resultId});
  await closeNativeDialog(page);
  await inspectResult(resultId);
  stage('save the portable grouped snapshot with an absent historical parent');
  await openNativeExport(page, true);
  const bundle = await readDownload(page.getByRole('button', {name: 'Save portable bundle', exact: true}));
  const saved = JSON.parse(bundle.toString('utf8'));
  const savedResult = saved.state.sources.find(source => source.dataset.datasetId === resultId);
  assert(savedResult);
  assert.deepEqual(savedResult.reference.parameters.groupBy, provenance);
  assert.deepEqual(savedResult.schema.map(field => field.name), expectedFields);
  assert(!saved.state.sources.some(source => source.dataset.datasetId === sourceId));
  assert.equal(saved.sources.length, 2);
  stage('restore the grouped snapshot and historical lineage in a fresh page');
  await page.reload({waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeExport(page, true);
  const portable = page.getByRole('region', {name: 'Portable sandbox bundle', exact: true});
  await portable.getByLabel('Load portable bundle', {exact: true}).setInputFiles({name: 'group-by.bundle.json', mimeType: 'application/json', buffer: bundle});
  await portable.getByRole('status').filter({hasText: 'Bundle validated'}).waitFor({timeout: 30000});
  await portable.getByRole('button', {name: 'Restore portable workspace', exact: true}).click();
  await page.locator('.ReactModal__Overlay').waitFor({state: 'hidden', timeout: 45000});
  await page.waitForFunction(({sourceId, resultId}) => {
    const sources = window.__keplerComputeTest.getState().publications;
    return !sources.some(source => source.id === sourceId) && sources.find(source => source.id === resultId)?.rowCount === 4;
  }, {sourceId, resultId});
  await inspectResult(resultId);
  assert.deepEqual(await exportResult(resultId), provenance);
  await openNativeExport(page, true);
  const restoredBundle = JSON.parse((await readDownload(page.getByRole('button', {name: 'Save portable bundle', exact: true}))).toString('utf8'));
  assert.deepEqual(restoredBundle.state.sources.find(source => source.dataset.datasetId === resultId).reference.parameters.groupBy, provenance);
  await closeNativeDialog(page);
  assert.deepEqual(sqlRequests, []);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, checks: [
    'native Prepare data dictionary and nullable Float32 multibatch source',
    'native Group By median refusal remains editable without starting SQL',
    'native count and sum fields, first source occurrence, dedicated null group and all-null sum',
    'native table and Arrow export retain exact values and Group By provenance',
    'source buffers/uploads remain unchanged',
    'completed result survives parent removal and portable restore retains historical lineage'
  ]}));
} catch (error) {
  const controls = await page.locator('.dataset-ops-panel').evaluateAll(panels => panels.map(panel => ({
    aggregationHeaders: [...panel.querySelectorAll('.dataset-ops-collapsible__header')].map(element => element.textContent),
    aggregationRows: [...panel.querySelectorAll('.dataset-ops-column-list > div')].map(element => ({
      text: element.textContent,
      checked: element.querySelector('input[type=checkbox]')?.checked
    })),
    textInputs: [...panel.querySelectorAll('input[type=text]')].map(element => ({value: element.value, placeholder: element.placeholder}))
  }))).catch(() => []);
  console.error(JSON.stringify({stage: currentStage, errors, diagnostics, sqlRequests,
    controls,
    body: (await page.locator('body').innerText().catch(() => '')).slice(-6000)}));
  throw error;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
