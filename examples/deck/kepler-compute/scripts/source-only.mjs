// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {openNativeData, closeNativeDialog, openNativeExport} from './native-navigation.mjs';
import * as arrow from 'apache-arrow';
import {cellToLatLng, latLngToCell} from 'h3-js';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const cellIds = [latLngToCell(40.74, -74.01, 7), latLngToCell(40.77, -73.94, 7)];
const cellValues = cellIds.map(value => BigInt(`0x${value}`));
const originalValues = [null, cellValues[0], 0n, cellValues[1], null, cellValues[0]];
const sourceBatch = values =>
  new arrow.Table({cell: arrow.vectorFromArray(values, new arrow.Uint64())});
const source = sourceBatch(originalValues.slice(0, 3)).concat(sourceBatch(originalValues.slice(3)));
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1600, height: 1100}, acceptDownloads: true});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => {
  errors.push(error.message);
  console.error(error.stack);
});
page.on('console', message => {
  if (
    message.type() === 'error' &&
    /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())
  )
    errors.push(message.text());
});
const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
const table = page.locator('.resident-table-scroll');
const card = page.locator('.resident-chart-card').first();

async function waitApplication() {
  await page.waitForFunction(
    () => {
      const overlay = document.querySelector('vite-error-overlay');
      if (overlay)
        throw new Error(
          overlay.shadowRoot?.textContent || overlay.textContent || 'Vite module error'
        );
      const state = window.__keplerComputeTest?.getState();
      if (state?.error) throw new Error(state.error);
      return state?.selectedCount === 256;
    },
    undefined,
    {timeout: 60000}
  );
}

async function publish(buffer, name, expectedPublications, excludeProvenance = false) {
  await openNativeData(page);
  await workspace
    .locator('input[type=file]')
    .setInputFiles({name, mimeType: 'application/vnd.apache.arrow.stream', buffer});
  await workspace.getByLabel('Source H3 column').selectOption('cell');
  if (excludeProvenance)
    await workspace.getByRole('checkbox', {name: /^__luma_source_row_index/}).uncheck();
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  assert.equal(
    await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).isEnabled(),
    true,
    'A source with no selected scalar fields must be publishable'
  );
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(
    expected => {
      const state = window.__keplerComputeTest.getState();
      if (
        state.error &&
        !state.error
          .split('\n')
          .every(
            reason => reason === 'H3 topology is pending the current submitted selection revision.'
          )
      )
        throw new Error(state.error);
      const publication = state.publications[expected - 1];
      if (
        publication?.error &&
        publication.error !==
          '1 source rows have invalid geometry; their table rows are retained and their geometry is omitted.'
      )
        throw new Error(publication.error);
      return state.publications.length === expected && publication?.selectedCount === 6;
    },
    expectedPublications,
    {timeout: 45000}
  );
  const publication = await page.evaluate(
    index => window.__keplerComputeTest.getState().publications[index],
    expectedPublications - 1
  );
  assert.equal(publication.rowCount, 6);
  assert.equal(publication.selectedCount, 6, 'Count includes source null/invalid geometry rows');
  assert.equal(publication.sourceUploadBytes, 0);
  assert.deepEqual(publication.sourceBufferIds, []);
  assert.deepEqual(publication.sourceBatches, [3, 3]);
  assert.equal(publication.h3.validRowCount, 3);
  assert.equal(publication.h3.invalidRows.length, 1);
  assert(publication.h3.sourceUploadBytes > 0);
  await closeNativeDialog(page);
  return publication;
}

async function showLayer(datasetId) {
  const configuration = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const nativeLayer = configuration.config.visState.layers.find(
    layer => layer.config.dataId === datasetId
  );
  assert.equal(nativeLayer?.type, 'hexagonId');
  for (const layer of configuration.config.visState.layers)
    layer.config.isVisible = layer === nativeLayer;
  nativeLayer.config.color = [245, 110, 40];
  nativeLayer.config.textLabel = [];
  nativeLayer.config.visConfig = {
    ...nativeLayer.config.visConfig,
    filled: true,
    outline: false,
    enable3d: false,
    coverage: 1,
    opacity: 1
  };
  nativeLayer.visualChannels = {
    ...nativeLayer.visualChannels,
    colorField: null,
    coverageField: null,
    sizeField: null
  };
  configuration.config.mapState = {
    ...configuration.config.mapState,
    longitude: -74.02,
    latitude: 40.755,
    zoom: 10,
    pitch: 0,
    bearing: 0
  };
  await page.evaluate(value => window.__keplerComputeTest.applyNativeConfig(value), configuration);
  await page.waitForFunction(
    identifier =>
      window.__keplerComputeTest
        .renderedLayers()
        .some(layer => layer.id === identifier && layer.visible && layer.loaded),
    nativeLayer.id,
    {timeout: 45000}
  );
  return nativeLayer.id;
}

async function assertPicking(datasetId) {
  const [latitude, longitude] = cellToLatLng(cellIds[1]);
  const picked = await page.evaluate(
    location =>
      window.__keplerComputeTest.pickPublishedPosition(
        location.datasetId,
        location.longitude,
        location.latitude
      ),
    {datasetId, longitude, latitude}
  );
  assert.equal(
    picked?.sourceRowIndex,
    3,
    'Picking must retain the row across the source batch seam'
  );
}

async function waitRows(columnCount) {
  await page.waitForFunction(columnCount => {
    const rows = document.querySelector('.resident-table-scroll');
    return (
      rows?.getAttribute('aria-busy') === 'false' &&
      rows.querySelectorAll('tbody td').length === columnCount * 6 &&
      JSON.stringify(
        Array.from(rows.querySelectorAll('tbody tr th'), cell => Number(cell.textContent))
      ) === '[0,1,2,3,4,5]'
    );
  }, columnCount);
}

async function waitCount() {
  await page.waitForFunction(
    () => document.querySelector('.resident-chart-card [aria-label="Chart value: 6"]') !== null
  );
}

async function download(button) {
  const pending = page.waitForEvent('download');
  await button.click();
  const stream = await (await pending).createReadStream();
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return Buffer.concat(chunks);
}

try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {
    waitUntil: 'domcontentloaded'
  });
  await waitApplication();
  const initial = await publish(
    Buffer.from(arrow.tableToIPC(source, 'stream')),
    'h3-only.arrow',
    1
  );
  await showLayer(initial.id);
  await assertPicking(initial.id);
  await closeNativeDialog(page);
  await page.locator('.show-data-table svg').last().click();
  await waitRows(0);
  assert.equal(await page.getByLabel('Values', {exact: true}).inputValue(), 'prepared');
  assert.equal(
    await table.locator('thead button').count(),
    0,
    'No internal row carrier may appear as a prepared field'
  );
  assert.equal(await table.locator('tbody td').count(), 0);
  await page.getByLabel('Values', {exact: true}).selectOption('original');
  await waitRows(1);
  assert.deepEqual(
    await table.locator('tbody td').allTextContents(),
    originalValues.map(value => (value === null ? 'null' : String(value)))
  );
  assert.equal(await table.locator('thead button').count(), 1);
  assert.equal(
    await table.locator('thead button').first().isDisabled(),
    true,
    'Exact H3 sorting has no resident scalar permutation'
  );
  await page.screenshot({path: '/tmp/kepler-source-only-table.png'});
  await page.keyboard.press('Escape');
  await page.locator('.resident-table-modal').waitFor({state: 'hidden'});

  await page.getByRole('button', {name: 'Toggle charts', exact: true}).click();
  await page.locator('.chart-config__type').click();
  await page.getByText('Big Number', {exact: true}).click();
  await card.getByLabel('Chart dataset', {exact: true}).selectOption(initial.id);
  await card.getByLabel('Chart field', {exact: true}).selectOption('');
  await card.getByLabel('Chart measure', {exact: true}).selectOption('count');
  await card.getByLabel('Chart title', {exact: true}).fill('H3 source rows');
  await waitCount();
  await card.getByLabel('Apply map filters', {exact: true}).uncheck();
  await waitCount();
  await card.getByLabel('Apply map filters', {exact: true}).check();
  await waitCount();
  const saved = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const savedChart = saved.config.visState.charts.find(chart => chart.title === 'H3 source rows');
  assert.equal(savedChart.dataId, initial.id);
  assert.equal(savedChart.axis.field, null);
  assert.equal(savedChart.axis.aggregation, 'count');
  await page.evaluate(
    configuration => window.__keplerComputeTest.applyNativeConfig(configuration),
    saved
  );
  await waitCount();
  await assertPicking(initial.id);
  const unchanged = await page.evaluate(
    identifier =>
      window.__keplerComputeTest
        .getState()
        .publications.find(publication => publication.id === identifier),
    initial.id
  );
  // scopeRevision counts derived selection-scope rebuilds (layer registration, config replay);
  // it is not source storage, so only the storage identity and upload counters must be stable.
  const {scopeRevision: initialScopeRevision, ...initialStorage} = initial;
  const {scopeRevision: unchangedScopeRevision, ...unchangedStorage} = unchanged;
  assert(unchangedScopeRevision >= initialScopeRevision, 'Scope revisions must not move backwards');
  assert.deepEqual(
    unchangedStorage,
    initialStorage,
    'Table, chart and config replay must not replace source storage'
  );
  await page.screenshot({path: '/tmp/kepler-source-only-count.png'});
  await card.getByRole('button', {name: 'Remove chart', exact: true}).click();
  await card.waitFor({state: 'hidden'});
  // The open chart panel overlays the map-control column, so close it from the keyboard.
  await page.getByRole('button', {name: 'Toggle charts', exact: true}).press('Enter');

  await openNativeExport(page);
  await page.getByLabel('Export dataset', {exact: true}).selectOption(initial.id);
  await page.getByLabel('Export format', {exact: true}).selectOption('arrow');
  const exportedBytes = await download(
    page.getByRole('button', {name: 'Export all selected', exact: true})
  );
  const exported = arrow.tableFromIPC(exportedBytes);
  assert.deepEqual(
    exported.schema.fields.map(field => field.name),
    ['cell', '__luma_source_row_index']
  );
  assert.equal(String(exported.getChild('cell').type), 'Uint64');
  assert.deepEqual([...exported.getChild('cell')], originalValues);
  assert.deepEqual([...exported.getChild('__luma_source_row_index')], [0, 1, 2, 3, 4, 5]);
  assert.deepEqual(
    exported.batches.map(batch => batch.numRows),
    [3, 3]
  );
  assert.equal(
    JSON.parse(exported.schema.metadata.get('luma:source-export')).datasetId,
    initial.id
  );
  const replayed = await publish(exportedBytes, 'h3-only-export-replay.arrow', 2, true);
  await showLayer(replayed.id);
  await assertPicking(replayed.id);
  await page.waitForFunction(() => window.__keplerComputeTest.getState().error === '', undefined, {
    timeout: 30000
  });
  const finalState = await page.evaluate(() => window.__keplerComputeTest.getState());
  assert.equal(finalState.error, '');
  assert.deepEqual(errors, []);
  await page.screenshot({path: '/tmp/kepler-source-only.png'});
  console.log(
    JSON.stringify({
      passed: true,
      sourceRows: 6,
      sourceBatches: [3, 3],
      scalarUploadBytes: 0,
      checks: [
        'H3-only native publication',
        'null and invalid source rows retained',
        'no synthetic prepared fields',
        'exact original Uint64 cells',
        'source picking across batches',
        'native Big Number Count without a field',
        'saved native chart replay',
        'unchanged source buffers',
        'exact original Arrow export',
        'export re-publication with provenance excluded'
      ]
    })
  );
} catch (error) {
  console.error({pageErrors: errors});
  console.error(
    await page
      .evaluate(() => ({
        state: window.__keplerComputeTest?.getState(),
        layers: window.__keplerComputeTest?.renderedLayers(),
        viteError: document.querySelector('vite-error-overlay')?.shadowRoot?.textContent,
        text: document.body.innerText
      }))
      .catch(() => 'Page unavailable')
  );
  throw error;
} finally {
  await browser.close();
}
