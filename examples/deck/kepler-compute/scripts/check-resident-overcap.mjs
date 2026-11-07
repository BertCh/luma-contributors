// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser check for over-cap resident results (> 262,144 rows): Spatial Join, Join and Suitability
// results must be adopted GPU-resident (no result readback, lazy rows) and page the right rows in
// through ensureRows. Usage: check-resident-overcap.mjs [spatial|join|suitability|table|exports|analysis-export|all|
// [join|spatial|suitability|all-]recovery[-explicit]] [url]; no mode runs everything
// `recovery` loses the device on a 300k-row Join result with ?automatic=all, expects automatic
// recomputation from lineage, then evicts and rebuilds it; `recovery-explicit` uses the button.
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {closeNativeDialog, openNativeExport} from './native-navigation.mjs';
import {
  aggregateValues, attachErrorCollectors, createPolygonTable, matchPolygons, polygonRows, publishTable, selectNativeOption
} from './spatial-join-utils.mjs';

const CAP = 262144;
const ROW_COUNT = 300000;
const mode = process.argv[2] || 'all';
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1400}});
const errors = [];
let currentStage = 'startup';
const watchdog = setTimeout(() => {
  console.error(`Over-cap check exceeded 1800 seconds at ${currentStage}`);
  void browser.close();
}, 1800000);
function stage(name) {
  currentStage = name;
  console.log(JSON.stringify({stage: name, seconds: Math.round(performance.now() / 1000)}));
}
attachErrorCollectors(page, errors);

function isClose(actual, expected) {
  if (expected === null || actual === null) return actual === expected;
  return Math.abs(actual - expected) <= Math.max(1e-4, Math.abs(expected) * 1e-5);
}
function getSampleRows(rowCount) {
  const rows = new Set([0, 1, 2, 1023, 1024, 1025, CAP - 1, CAP, CAP + 1, rowCount - 2, rowCount - 1]);
  for (let row = 3; row < rowCount; row += 4999) rows.add(row);
  return [...rows].filter(row => row < rowCount).sort((a, b) => a - b);
}
async function waitForResult(knownIds, rowCount) {
  await page.waitForFunction(({ids, rowCount}) => window.__keplerComputeTest.getState().publications.some(source => !ids.includes(source.id) && source.rowCount === rowCount) || document.querySelector('.dataset-ops-panel [class*=error], .dataset-ops-panel .error'), {ids: knownIds, rowCount}, {timeout: 120000});
  assert.deepEqual(errors, []);
  return page.evaluate(({ids, rowCount}) => window.__keplerComputeTest.getState().publications.find(source => !ids.includes(source.id) && source.rowCount === rowCount).id, {ids: knownIds, rowCount});
}
/** The result must be lazy: edge rows page in without loading the dataset, then a spread sample is read. */
async function readLazyRows(resultId, rowCount) {
  const edge = await page.evaluate(([id, rows]) => window.__keplerComputeTest.readPublishedRows(id, rows), [resultId, [0, CAP, rowCount - 1]]);
  assert.equal(edge.resident, true, 'over-cap result must be adopted resident');
  assert.equal(edge.rowCount, rowCount);
  assert.ok(edge.loadedRowCount > 0 && edge.loadedRowCount <= 4096, `only the requested pages load (${edge.loadedRowCount})`);
  const sampleRows = getSampleRows(rowCount);
  const sample = await page.evaluate(([id, rows]) => window.__keplerComputeTest.readPublishedRows(id, rows), [resultId, sampleRows]);
  assert.ok(sample.loadedRowCount < rowCount, 'result must not be fully loaded by sampling');
  return {sampleRows, rows: sample.rows, edge, sample};
}
async function openOperation(sourceId, menuClass) {
  await page.locator(`.dataset-ops-menu__toggle[data-for="dataset-ops-${sourceId}"]`).click();
  await page.locator(menuClass).click();
  const panel = page.locator('.dataset-ops-panel');
  await panel.waitFor();
  return panel;
}
async function getPublicationIds() {
  return page.evaluate(() => window.__keplerComputeTest.getState().publications.map(source => source.id));
}
async function publishAndFind(label, table, options, rowCount) {
  const before = await getPublicationIds();
  await publishTable(page, label, table, options);
  const id = await waitForResult(before, rowCount);
  await closeNativeDialog(page);
  return id;
}

async function checkSpatialJoin() {
  stage('spatial: build 300k points');
  const longitude = new Float32Array(ROW_COUNT);
  const latitude = new Float32Array(ROW_COUNT);
  const amount = new Float32Array(ROW_COUNT);
  for (let row = 0; row < ROW_COUNT; row++) {
    longitude[row] = ((row * 7919) % 5000) / 100 - 2;
    latitude[row] = ((row * 104729) % 1200) / 100 - 0.5;
    amount[row] = row % 1000;
  }
  const points = new arrow.Table({
    longitude: arrow.vectorFromArray(longitude, new arrow.Float32()),
    latitude: arrow.vectorFromArray(latitude, new arrow.Float32()),
    amount: arrow.vectorFromArray(amount, new arrow.Float32())
  });
  const pointId = await publishAndFind('overcap-points.arrow', points, {longitude: 'longitude', latitude: 'latitude'}, ROW_COUNT);
  await publishAndFind('overcap-polygons.arrow', createPolygonTable(), {geometry: 'geometry'}, 4);
  const before = await getPublicationIds();
  stage('spatial: run over-cap GPU Spatial Join');
  const panel = await openOperation(pointId, '.dataset-ops-menu__spatial-join');
  await selectNativeOption(page, panel.locator('.data-source-selector').last(), 'overcap-polygons.arrow');
  await panel.getByText(/aggregation rules/i).click();
  const geometryRow = panel.locator('.dataset-ops-column-list > div').filter({hasText: /^geometry/});
  // Merge is bounded to the readback cap, so the default flow is refused explicitly over the cap.
  await panel.locator('input[type=text]').last().fill('Over-cap merge refusal');
  await panel.getByRole('button', {name: 'Run', exact: true}).click();
  await page.getByText(/uncheck merge/i).first().waitFor({timeout: 30000});
  assert.deepEqual(await getPublicationIds(), before, 'refused merge must not publish');
  await geometryRow.locator('input[type=checkbox]').click();
  await selectNativeOption(page, panel.locator('.dataset-ops-column-list > div').filter({hasText: /^weight/}), 'sum');
  await selectNativeOption(page, panel.locator('.dataset-ops-column-list > div').filter({hasText: /^rank/}), 'maximum');
  await panel.locator('input[type=text]').last().fill('Over-cap spatial result');
  await panel.getByRole('button', {name: 'Run', exact: true}).click();
  const resultId = await waitForResult(before, ROW_COUNT);
  stage('spatial: page rows lazily and compare with the CPU reference');
  const {sampleRows, rows, edge} = await readLazyRows(resultId, ROW_COUNT);
  const verify = (rows, sampleRows, label) => {
  let matched = 0;
  for (const [index, row] of rows.entries()) {
    const source = sampleRows[index];
    const matches = matchPolygons(longitude[source], latitude[source], 'intersects');
    const expected = {
      longitude: longitude[source], latitude: latitude[source], amount: amount[source], count: matches.length,
      weight_sum: aggregateValues(matches.map(polygon => polygonRows[polygon].weight === null ? null : Math.fround(polygonRows[polygon].weight)), 'sum'),
      rank_maximum: aggregateValues(matches.map(polygon => polygonRows[polygon].rank), 'maximum')
    };
    if (matches.length) matched++;
    for (const [name, value] of Object.entries(expected))
      assert.ok(isClose(row[name], value), `${label} spatial row ${source} ${name}: ${row[name]} vs ${value}`);
  }
  assert.ok(matched > 20, `sample must include matches (${matched})`);
  return matched;
  };
  const matched = verify(rows, sampleRows, 'resident');
  recoveryTargets.spatial = {resultId, rowCount: ROW_COUNT, verify: async () => {
    const recovered = await readLazyRows(resultId, ROW_COUNT);
    verify(recovered.rows, recovered.sampleRows, 'recovered');
  }};
  return {case: 'spatial', rowCount: ROW_COUNT, sampled: rows.length, matched, loadedAfterEdges: edge.loadedRowCount};
}

async function checkJoin() {
  stage('join: build 300k-row left table and keyed right table');
  const keys = Array.from({length: ROW_COUNT}, (_, row) => `k${row % 60}`);
  const amounts = Float32Array.from({length: ROW_COUNT}, (_, row) => row % 1000);
  const left = new arrow.Table({
    key: arrow.vectorFromArray(keys, new arrow.Utf8()),
    amount: arrow.vectorFromArray(amounts, new arrow.Float32())
  });
  // k0 repeats on the right to expand rows; k50..k59 are unmatched (LEFT join keeps them with nulls).
  const rightKeys = [...Array.from({length: 50}, (_, index) => `k${index}`), 'k0'];
  const rightWeights = rightKeys.map((_, index) => index + 0.5);
  const right = new arrow.Table({
    key: arrow.vectorFromArray(rightKeys, new arrow.Utf8()),
    weight: arrow.vectorFromArray(rightWeights, new arrow.Float32())
  });
  const expected = [];
  for (let row = 0; row < ROW_COUNT; row++) {
    const matches = rightKeys.flatMap((key, index) => key === keys[row] ? [index] : []);
    if (!matches.length) expected.push([keys[row], amounts[row], null, null]);
    for (const index of matches) expected.push([keys[row], amounts[row], rightKeys[index], rightWeights[index]]);
  }
  const leftId = await publishAndFind('overcap-join-left.arrow', left, {}, ROW_COUNT);
  await publishAndFind('overcap-join-right.arrow', right, {}, 51);
  const before = await getPublicationIds();
  stage('join: run over-cap GPU LEFT Join');
  const panel = await openOperation(leftId, '.dataset-ops-menu__join');
  await selectNativeOption(page, panel.locator('.data-source-selector').last(), 'overcap-join-right.arrow');
  await selectNativeOption(page, panel.locator('.field-selector').nth(0), 'key');
  await selectNativeOption(page, panel.locator('.field-selector').nth(1), 'key');
  await panel.getByLabel('LEFT', {exact: true}).click();
  await panel.locator('input[type=text]').last().fill('Over-cap join result');
  await panel.getByRole('button', {name: 'Run', exact: true}).click();
  const resultId = await waitForResult(before, expected.length);
  stage('join: page rows lazily and compare with the CPU reference');
  const {sampleRows, rows, edge} = await readLazyRows(resultId, expected.length);
  for (const [index, row] of rows.entries()) {
    const actual = Object.values(row);
    const want = expected[sampleRows[index]];
    for (let column = 0; column < 4; column++)
      assert.ok(typeof want[column] === 'number' || want[column] === null ? isClose(actual[column], want[column]) : actual[column] === want[column],
        `join row ${sampleRows[index]} column ${column}: ${actual[column]} vs ${want[column]}`);
  }
  joinResultForTable = {resultId, expected};
  recoveryTargets.join = {resultId, rowCount: expected.length, verify: () => verifyRecomputedJoin(resultId, expected)};
  return {case: 'join', rowCount: expected.length, sampled: rows.length, loadedAfterEdges: edge.loadedRowCount};
}

let joinResultForTable = null;
const recoveryTargets = {};
/** The native data table must show real cells for an over-cap resident result (not null placeholders). */
async function checkNativeTable() {
  if (!joinResultForTable) await checkJoin();
  const {expected} = joinResultForTable;
  stage('table: open the native data table on the over-cap join result');
  const table = page.locator('.resident-table-scroll');
  await page.locator('.show-data-table svg').last().click();
  await table.waitFor({timeout: 60000});
  const readCells = async () => {
    await page.waitForFunction(() => document.querySelector('.resident-table-scroll')?.getAttribute('aria-busy') === 'false' && document.querySelectorAll('.resident-table-scroll tbody tr').length > 0, undefined, {timeout: 60000});
    return table.locator('tbody tr').evaluateAll(rows => rows.map(row => ({
      sourceRow: Number(row.querySelector('th')?.textContent),
      cells: Array.from(row.querySelectorAll('td'), cell => cell.textContent)
    })));
  };
  const summary = {};
  for (const view of ['prepared', 'original']) {
    await page.getByLabel('Values', {exact: true}).selectOption(view);
    let rows = await readCells();
    // Jump to a late page too, so rows beyond the first lazily loaded pages are paged in.
    for (const label of ['first', 'late']) {
      if (label === 'late') {
        const pageSize = rows.length;
        await page.getByLabel('Rows', {exact: true}).selectOption('all').catch(() => undefined);
        const last = page.getByRole('button', {name: /last page/i});
        if (await last.count()) {await last.click(); rows = await readCells();} else summary[`${view}LatePageSkipped`] = pageSize;
      }
      assert.ok(rows.length > 0, `${view} ${label}: table has rows`);
      for (const row of rows) {
        const want = expected[row.sourceRow];
        if (!want) continue;
        const text = row.cells.join('|');
        assert.ok(!/(^|\|)null(\||$)/.test(row.cells.slice(0, 2).join('|')) && row.cells[0] !== 'null' && row.cells[1] !== 'null', `${view} ${label} row ${row.sourceRow} has null cells: ${text}`);
        if (view === 'original') {
          assert.equal(row.cells[0], want[0], `${view} key row ${row.sourceRow}`);
          assert.ok(isClose(Number(row.cells[1]), want[1]), `${view} amount row ${row.sourceRow}: ${row.cells[1]} vs ${want[1]}`);
          if (want[3] === null) assert.equal(row.cells[3], 'null');
          else assert.ok(isClose(Number(row.cells[3]), want[3]), `${view} weight row ${row.sourceRow}`);
        }
      }
      summary[`${view}_${label}`] = {rows: rows.length, firstSourceRow: rows[0].sourceRow, firstCells: rows[0].cells};
    }
  }
  await page.keyboard.press('Escape');
  await page.locator('.resident-table-modal').waitFor({state: 'hidden'});
  return {case: 'table', ...summary};
}

async function checkSuitability() {
  stage('suitability: build 300k-row numeric table');
  // The resident scorer requires the same rows to be missing in every weighted field.
  const cost = Array.from({length: ROW_COUNT}, (_, row) => row % 97 === 0 ? null : (row * 37) % 1000);
  const access = Array.from({length: ROW_COUNT}, (_, row) => row % 97 === 0 ? null : (row * 53) % 700);
  const table = new arrow.Table({
    cost: arrow.vectorFromArray(cost, new arrow.Float32()),
    access: arrow.vectorFromArray(access, new arrow.Float32())
  });
  const sourceId = await publishAndFind('overcap-suitability.arrow', table, {}, ROW_COUNT);
  const before = await getPublicationIds();
  stage('suitability: run over-cap GPU Suitability');
  const panel = await openOperation(sourceId, '.dataset-ops-menu__suitability');
  await panel.locator('input[type=text]').last().fill('Over-cap suitability result');
  await panel.getByRole('button', {name: 'Run', exact: true}).click();
  const resultId = await waitForResult(before, ROW_COUNT);
  stage('suitability: page rows lazily and compare with the CPU formula');
  const {sampleRows, rows, edge} = await readLazyRows(resultId, ROW_COUNT);
  const range = values => {
    const defined = values.filter(value => value !== null);
    const minimum = defined.reduce((least, value) => Math.min(least, value), Infinity);
    const span = defined.reduce((most, value) => Math.max(most, value), -Infinity) - minimum;
    return values.map(value => value === null ? null : (value - minimum) / span);
  };
  const costRange = range(cost);
  const accessRange = range(access);
  let nullScores = 0;
  const verify = (rows, sampleRows, label) => {
    nullScores = 0;
    for (const [index, row] of rows.entries()) {
      const source = sampleRows[index];
      const want = costRange[source] === null || accessRange[source] === null ? null : (costRange[source] + accessRange[source]) / 2;
      if (want === null) nullScores++;
      assert.ok(isClose(row.cost, cost[source]) && isClose(row.access, access[source]), `${label} suitability row ${source} carried columns`);
      assert.ok(isClose(row.score, want), `${label} suitability row ${source} score: ${row.score} vs ${want}`);
    }
  };
  verify(rows, sampleRows, 'resident');
  const scoreAt = row => costRange[row] === null || accessRange[row] === null ? null : (costRange[row] + accessRange[row]) / 2;
  recoveryTargets.suitability = {resultId, rowCount: ROW_COUNT, scoreAt, costAt: row => cost[row], verify: async () => {
    const recovered = await readLazyRows(resultId, ROW_COUNT);
    verify(recovered.rows, recovered.sampleRows, 'recovered');
  }};
  return {case: 'suitability', rowCount: ROW_COUNT, sampled: rows.length, nullScores, loadedAfterEdges: edge.loadedRowCount};
}


// Streamed materialization of over-cap resident results: source export, combined analysis export,
// portable sandbox bundle page GPU rows instead of refusing.
async function readDownload(trigger) {
  const pending = page.waitForEvent('download', download => !download.suggestedFilename().endsWith('.metadata.json'));
  await trigger();
  const stream = await (await pending).createReadStream();
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return Buffer.concat(chunks);
}
async function checkResidentExports() {
  stage('exports: build and run an over-cap Suitability result');
  const cost = Array.from({length: ROW_COUNT}, (_, row) => row % 97 === 0 ? null : (row * 37) % 1000);
  const access = Array.from({length: ROW_COUNT}, (_, row) => row % 97 === 0 ? null : (row * 53) % 700);
  const table = new arrow.Table({
    cost: arrow.vectorFromArray(cost, new arrow.Float32()),
    access: arrow.vectorFromArray(access, new arrow.Float32())
  });
  const sourceId = await publishAndFind('overcap-export-source.arrow', table, {}, ROW_COUNT);
  const before = await getPublicationIds();
  const panel = await openOperation(sourceId, '.dataset-ops-menu__suitability');
  await panel.locator('input[type=text]').last().fill('Over-cap export result');
  await panel.getByRole('button', {name: 'Run', exact: true}).click();
  const resultId = await waitForResult(before, ROW_COUNT);
  await closeNativeDialog(page);
  const summary = {case: 'exports', rowCount: ROW_COUNT};
  const expectScore = (row, rowIndex) => {
    assert.ok(isClose(row.cost, cost[rowIndex]) && isClose(row.access, access[rowIndex]), `exported carried columns row ${rowIndex}`);
  };

  stage('exports: source Arrow and CSV of every selected row');
  await openNativeExport(page);
  await page.getByLabel('Export dataset', {exact: true}).selectOption(resultId);
  await page.getByLabel('Export format', {exact: true}).selectOption('arrow');
  const exportButton = page.getByRole('button', {name: 'Export all selected', exact: true});
  const exported = arrow.tableFromIPC(await readDownload(() => exportButton.click()));
  assert.equal(exported.numRows, ROW_COUNT, 'every selected row is exported');
  const rowColumn = exported.getChild('__luma_source_row_index');
  for (const rowIndex of [0, 1, 97, 1024, CAP - 1, CAP, ROW_COUNT - 1]) {
    assert.equal(rowColumn.get(rowIndex), rowIndex);
    expectScore({cost: exported.getChild('cost').get(rowIndex), access: exported.getChild('access').get(rowIndex)}, rowIndex);
  }
  assert.equal(exported.getChild('score').get(0), null, 'null scores survive export');
  summary.arrowBatches = exported.batches.length;
  await page.getByLabel('Export format', {exact: true}).selectOption('csv');
  const csv = (await readDownload(() => exportButton.click())).toString('utf8').split('\r\n');
  assert.equal(csv.length - 1 - 1, ROW_COUNT, 'CSV has a header plus every row');
  assert.match(csv[0], /^cost,access,score,__luma_source_row_index$|score/);
  assert.equal(csv[CAP + 1].split(',').at(-1), String(CAP));

  stage('exports: portable sandbox bundle with the full resident source');
  await openNativeExport(page, true);
  const bundle = JSON.parse((await readDownload(() => page.getByRole('button', {name: 'Save portable bundle', exact: true}).click())).toString('utf8'));
  const encoded = bundle.sources.find(source => source.datasetId === resultId);
  assert.ok(encoded, 'bundle contains the resident result');
  const bundled = arrow.tableFromIPC(Buffer.from(encoded.data, 'base64'));
  assert.equal(bundled.numRows, ROW_COUNT);
  for (const rowIndex of [0, 1, 1024, CAP, ROW_COUNT - 1])
    expectScore({cost: bundled.getChild('cost').get(rowIndex), access: bundled.getChild('access').get(rowIndex)}, rowIndex);
  summary.bundleBytes = encoded.byteLength;

  return summary;
}

async function verifyRecomputedJoin(resultId, expected) {
  const {sampleRows, rows} = await readLazyRows(resultId, expected.length);
  for (const [index, row] of rows.entries()) {
    const actual = Object.values(row);
    const want = expected[sampleRows[index]];
    for (let column = 0; column < 4; column++)
      assert.ok(typeof want[column] === 'number' || want[column] === null ? isClose(actual[column], want[column]) : actual[column] === want[column],
        `recovered join row ${sampleRows[index]} column ${column}: ${actual[column]} vs ${want[column]}`);
  }
}
const CHECKS = {spatial: checkSpatialJoin, join: checkJoin, suitability: checkSuitability};
/** Device loss and eviction recompute over-cap results from lineage under the same dataset ID. */
async function checkRecovery(kinds, automatic) {
  for (const kind of kinds) if (!recoveryTargets[kind]) await CHECKS[kind]();
  const targets = kinds.map(kind => ({kind, ...recoveryTargets[kind]}));
  const getState = () => page.evaluate(() => window.__keplerComputeTest.getState());
  const initial = await getState();
  assert.equal(initial.recovery.hasCheckpoint, true, 'an over-cap result must not disable the recovery checkpoint');
  stage(`recovery(${kinds}): lose the device`);
  await page.evaluate(() => window.__keplerComputeTest.loseDevice());
  if (!automatic) {
    await page.waitForFunction(() => ['lost', 'failed'].includes(window.__keplerComputeTest.getState().recovery.phase));
    assert.equal((await getState()).recovery.hasCheckpoint, true);
    await page.getByRole('button', {name: 'Recover GPU workspace', exact: true}).click();
  }
  await page.waitForFunction(({targets, generation}) => {
    const state = window.__keplerComputeTest.getState();
    if (state.recovery.phase === 'failed') throw new Error(state.recovery.reason);
    return state.ready && state.recovery.phase === 'ready' && state.deviceGeneration > generation &&
      targets.every(target => state.publications.some(source => source.id === target.resultId && source.rowCount === target.rowCount));
  }, {targets: targets.map(({resultId, rowCount}) => ({resultId, rowCount})), generation: initial.deviceGeneration}, {timeout: 180000});
  assert.equal((await getState()).error, '');
  stage('recovery: compare recomputed rows with the CPU reference');
  for (const target of targets) await target.verify();
  const kind = kinds.join('+');
  if (!automatic) return {case: `recovery-explicit(${kind})`, rowCounts: targets.map(target => target.rowCount)};
  stage('recovery: evict the hidden results and rebuild them from lineage');
  // Each pressure round evicts the least recently used hidden table(s); inputs go first, so a
  // result's rebuild must also bring its evicted inputs back.
  const evictedAll = async () => (await getState()).automaticEviction.evictedIds;
  for (let round = 0; round < 12 && !targets.every(target => (evictedAllCache).includes(target.resultId)); round++) {
    await page.evaluate(() => { try { window.__keplerComputeTest.applyMemoryPressure(); } catch (error) { if (error.message !== 'Dataset memory budget exceeded') throw error; } });
    await page.waitForTimeout(1500);
    evictedAllCache = await evictedAll();
  }
  for (const target of targets) {
    if (!evictedAllCache.includes(target.resultId)) continue;
    console.log(JSON.stringify({evicted: target.kind}));
    await page.evaluate(id => window.__keplerComputeTest.ensureDatasetResident(id), target.resultId);
    await page.waitForFunction(({id, rowCount}) => {
      const state = window.__keplerComputeTest.getState();
      if (state.automaticEviction.failedIds.length) throw new Error(state.automaticEviction.lastError);
      return !state.automaticEviction.evictedIds.includes(id) && state.publications.some(source => source.id === id && source.rowCount === rowCount);
    }, {id: target.resultId, rowCount: target.rowCount}, {timeout: 180000});
    await target.verify();
  }
  for (const target of targets) assert.ok(evictedAllCache.includes(target.resultId), `the hidden ${target.kind} result must be evicted`);
  assert.equal((await getState()).error, '');
  return {case: `recovery(${kind})`, rowCounts: targets.map(target => target.rowCount)};
}
let evictedAllCache = [];

/** Analysis export of an over-cap resident Suitability result streams every selected row. */
async function checkAnalysisExport() {
  if (!recoveryTargets.suitability) await checkSuitability();
  const {resultId} = recoveryTargets.suitability;
  stage('analysis export: z-score the over-cap score field');
  const panel = page.getByRole('region', {name: 'GPU analysis', exact: true});
  await panel.getByLabel('Dataset', {exact: true}).selectOption(resultId);
  await panel.getByLabel('Numeric field', {exact: true}).selectOption('score');
  await panel.getByLabel('Operation', {exact: true}).selectOption('z-score');
  await page.waitForFunction(id => window.__keplerComputeTest.analysisForDataset(id)?.getTableColumns()?.isCurrent(), resultId, {timeout: 120000});
  await openNativeExport(page);
  await page.getByLabel('Export dataset', {exact: true}).selectOption(resultId);
  const exportButton = page.getByRole('button', {name: 'Export all selected', exact: true});
  const exportFormat = async format => {
    await page.getByLabel('Export format', {exact: true}).selectOption(format);
    return readDownload(() => exportButton.click());
  };
  const rowCount = recoveryTargets.suitability.rowCount;
  stage('analysis export: analysis-arrow');
  const table = arrow.tableFromIPC(await exportFormat('analysis-arrow'));
  assert.equal(table.numRows, rowCount, 'every selected row is exported');
  const rowColumn = table.getChild('__luma_source_row_index');
  const zColumn = table.getChild('analysis_z_score') ?? table.getChild(table.schema.fields.find(field => /z_?score/i.test(field.name))?.name);
  assert.ok(zColumn, `z-score column in ${table.schema.fields.map(field => field.name)}`);
  for (const row of [0, 1, 97, 1024, CAP - 1, CAP, rowCount - 1]) assert.equal(rowColumn.get(row), row);
  assert.equal(zColumn.get(0), null, 'null scores stay null');
  assert.equal(zColumn.get(97), null);
  // z-score is affine in score: the pairwise slope is identical across the cap boundary.
  const scores = recoveryTargets.suitability.scoreAt;
  const slope = (a, b) => (zColumn.get(a) - zColumn.get(b)) / (scores(a) - scores(b));
  const reference = slope(1, 2);
  for (const [a, b] of [[CAP, CAP + 1], [rowCount - 2, rowCount - 1], [1024, CAP + 5]]) {
    assert.ok(Math.abs(slope(a, b) - reference) <= Math.abs(reference) * 1e-3, `z-score slope ${a},${b}: ${slope(a, b)} vs ${reference}`);
  }
  stage('analysis export: combined-arrow');
  const combined = arrow.tableFromIPC(await exportFormat('combined-arrow'));
  assert.equal(combined.numRows, rowCount);
  for (const row of [0, 1, CAP, rowCount - 1]) assert.ok(isClose(combined.getChild('cost').get(row), recoveryTargets.suitability.costAt(row)), `combined cost row ${row}`);
  stage('analysis export: combined-csv');
  const csv = (await exportFormat('combined-csv')).toString('utf8').trimEnd().split(/\r?\n/);
  assert.equal(csv.length - 1, rowCount, 'CSV has a header plus every row');
  await closeNativeDialog(page);
  return {case: 'analysis-export', rowCount, slope: reference};
}

const summaries = [];
const defaultUrl = 'http://127.0.0.1:5318/?rows=256';
const automaticUrl = 'http://127.0.0.1:5318/?rows=256&automatic=all';
async function openApp(url) {
  await page.goto(url, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
}
try {
  await pinViteModules(page);
  // Modes: spatial|join|suitability|table|exports|analysis-export, <kind>-recovery[-explicit] with
  // kind join|spatial|suitability|all (bare recovery[-explicit] is join), and all (everything).
  const recoveryMatch = /^(?:(join|spatial|suitability|all)-)?(recovery(?:-explicit)?)$/.exec(mode);
  const recoveryKinds = !recoveryMatch ? [] : (recoveryMatch[1] ?? 'join') === 'all' ? ['join', 'spatial', 'suitability'] : [recoveryMatch[1] ?? 'join'];
  const automaticMode = recoveryMatch?.[2] === 'recovery';
  await openApp(process.argv[3] || (automaticMode ? automaticUrl : defaultUrl));
  if (mode === 'all' || mode === 'spatial') summaries.push(await checkSpatialJoin());
  if (mode === 'all' || mode === 'join') summaries.push(await checkJoin());
  // The table check opens the last dataset's table, so it runs right after the Join it inspects.
  if (mode === 'all' || mode === 'table') summaries.push(await checkNativeTable());
  if (mode === 'all' || mode === 'suitability') summaries.push(await checkSuitability());
  if (mode === 'all' || mode === 'exports') summaries.push(await checkResidentExports());
  if (mode === 'all') {
    // Resident datasets share a 512 MiB accounted budget; start analysis export from a clean page.
    for (const kind of Object.keys(recoveryTargets)) delete recoveryTargets[kind];
    await openApp(defaultUrl);
  }
  if (mode === 'all' || mode === 'analysis-export') summaries.push(await checkAnalysisExport());
  if (mode === 'all') {
    // Explicit recovery of all three results, then a fresh ?automatic=all page for automatic
    // recovery and eviction rebuilds (the device-loss checks run last: they replace the device).
    summaries.push(await checkRecovery(['join', 'spatial', 'suitability'], false));
    for (const kind of Object.keys(recoveryTargets)) delete recoveryTargets[kind];
    joinResultForTable = null;
    await openApp(automaticUrl);
    summaries.push(await checkRecovery(['join', 'spatial', 'suitability'], true));
  } else if (recoveryKinds.length) summaries.push(await checkRecovery(recoveryKinds, automaticMode));
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, summaries}));
} catch (error) {
  console.error(JSON.stringify({stage: currentStage, errors,
    body: (await page.locator('body').innerText().catch(() => '')).slice(-3000)}));
  throw error;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
