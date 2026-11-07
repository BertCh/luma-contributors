// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {openNativeData, closeNativeDialog, openNativeExport} from './native-navigation.mjs';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const original = new arrow.Table({
  longitude: arrow.makeVector(new Float32Array([-74, -74.01, -74.02, -74.03])),
  latitude: arrow.makeVector(new Float32Array([40.7, 40.72, 40.74, 40.76])),
  measure: arrow.vectorFromArray([10, 20, null, 40], new arrow.Float32()),
  population: arrow.makeVector(new Float32Array([100, 100, 100, 100]))
});
const source = original.slice(0, 2).concat(original.slice(2));
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
// Keep this check on one loaded implementation while other contributors keep editing.
await page.routeWebSocket('**/*', socket => {
  const server = socket.connectToServer();
  server.onMessage(message => {
    if (typeof message === 'string') {
      try {
        const packet = JSON.parse(message);
        if (packet.type === 'update' || packet.type === 'full-reload') return;
      } catch {}
    }
    socket.send(message);
  });
  socket.onMessage(message => server.send(message));
});
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: 'analysis-batches.arrow', mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))
  });
  await workspace.getByLabel('Source longitude column').selectOption('longitude');
  await workspace.getByLabel('Source latitude column').selectOption('latitude');
  await workspace.getByRole('button', {name: 'Check schema and precision'}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields'}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 4, undefined, {timeout: 45000});
  const datasetId = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);
  await closeNativeDialog(page);
  const panel = page.getByRole('region', {name: 'GPU analysis', exact: true});
  await panel.getByLabel('Dataset', {exact: true}).selectOption(datasetId);
  await panel.getByLabel('Numeric field', {exact: true}).selectOption('measure');
  await panel.getByLabel('Operation', {exact: true}).selectOption('z-score');
  await page.waitForFunction(() => window.__keplerComputeTest.analysisForDataset(''+window.__keplerComputeTest.getState().publications[0].id)?.getTableColumns()?.isCurrent());
  await panel.getByLabel('Operation', {exact: true}).selectOption('excess-risk');
  await panel.getByLabel('Population field', {exact: true}).selectOption('population');
  await page.waitForFunction(() => window.__keplerComputeTest.analysisForDataset(window.__keplerComputeTest.getState().publications[0].id)?.getTableColumns()?.isCurrent());
  await openNativeExport(page);
  await page.getByLabel('Export dataset', {exact: true}).selectOption(datasetId);
  const exports = [];
  page.on('download', download => exports.push(download));
  const exportButton = page.getByRole('button', {name: 'Export all selected', exact: true});
  async function exportFormat(format) {
    const pending = page.waitForEvent('download', download => !download.suggestedFilename().endsWith('.metadata.json'));
    await page.getByLabel('Export format', {exact: true}).selectOption(format);
    await exportButton.click();
    const stream = await (await pending).createReadStream();
    const chunks = []; for await (const chunk of stream) chunks.push(chunk);
    return Buffer.concat(chunks);
  }
  const table = arrow.tableFromIPC(await exportFormat('analysis-arrow'));
  assert.deepEqual([...table.getChild('__luma_source_row_index')], [0, 1, 2, 3]);
  const values = [...table.getChild('analysis_excess_risk')];
  assert.equal(values[2], null);
  for (const [row, expected] of [[0, 3 / 7], [1, 6 / 7], [3, 12 / 7]]) assert.ok(Math.abs(values[row] - expected) < 0.0001);
  const metadata = JSON.parse(table.schema.metadata.get('luma:analysis-export'));
  assert.equal(metadata.datasetId, datasetId);
  assert.deepEqual(metadata.recipe.operation, {kind: 'excess-risk', events: 'measure', populations: 'population'});
  assert.equal(metadata.recipe.population, 'all');
  await page.evaluate(id => window.__keplerComputeTest.setPublishedRange(id, 'measure', 20, 40), datasetId);
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 2 && window.__keplerComputeTest.analysisForDataset(window.__keplerComputeTest.getState().publications[0].id)?.getTableColumns()?.isCurrent());
  const json = JSON.parse((await exportFormat('analysis-json')).toString('utf8'));
  assert.deepEqual(json.rows.map(row => row.__luma_source_row_index), [1, 3]);
  assert.equal(json.selectedCount, 2);
  assert.equal(json.recipe.population, 'all');
  assert.ok(Math.abs(json.rows[0].analysis_excess_risk - 6 / 7) < 0.0001);
  const metadataDownload = page.waitForEvent('download', download => download.suggestedFilename().endsWith('.metadata.json'));
  const csv = (await exportFormat('analysis-csv')).toString('utf8');
  assert.match(csv, /^__luma_source_row_index,analysis_excess_risk/);
  const metadataStream = await (await metadataDownload).createReadStream();
  const metadataChunks = []; for await (const chunk of metadataStream) metadataChunks.push(chunk);
  const csvMetadata = JSON.parse(Buffer.concat(metadataChunks).toString('utf8'));
  assert.equal(csvMetadata.analysisRevision, json.analysisRevision);
  const joined = arrow.tableFromIPC(await exportFormat('combined-arrow'));
  assert.deepEqual([...joined.getChild('measure')], [20, 40]);
  assert.deepEqual([...joined.getChild('__luma_source_row_index')], [1, 3]);
  assert.ok(Math.abs(joined.getChild('analysis_excess_risk').get(0) - 6 / 7) < 0.0001);
  assert.equal(JSON.parse(joined.schema.metadata.get('luma:analysis-export')).recipe.operation.populations, 'population');
  const joinedMetadataDownload = page.waitForEvent('download', download => download.suggestedFilename().endsWith('.metadata.json'));
  const joinedCSV = (await exportFormat('combined-csv')).toString('utf8');
  assert.match(joinedCSV, /^longitude,latitude,measure,population,analysis_excess_risk,__luma_source_row_index/);
  await joinedMetadataDownload;
  await closeNativeDialog(page);
  await panel.getByLabel('Operation', {exact: true}).selectOption('off');
  await openNativeExport(page);
  await page.getByLabel('Export dataset', {exact: true}).selectOption(datasetId);
  await page.getByLabel('Export format', {exact: true}).selectOption('combined-csv');
  const before = exports.length;
  await exportButton.click();
  await page.getByRole('region', {name: 'Export data', exact: true}).getByRole('status').filter({hasText: /unavailable|stale|Run an analysis/}).waitFor();
  assert.equal(exports.length, before);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, checks: ['published multibatch excess-risk Arrow with null validity', 'original source IDs', 'selected JSON keeps all-population recipe', 'CSV plus metadata', 'disabled analysis cannot export stale results']}));

} catch (error) {
  console.error(await page.evaluate(() => ({state: window.__keplerComputeTest?.getState(), errors: Array.from(document.querySelectorAll('[role=alert], [role=status]'), element => element.textContent), workspace: document.querySelector('[aria-label="Resident datasets"]')?.textContent})));
  console.error(errors);
  throw error;
} finally {
  await browser.close();
}
