// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Default URL is ?rows=256 (the script's intended fixture size); pass an explicit URL to override.
import assert from 'node:assert/strict';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const source = arrow.tableFromArrays({longitude: new Float32Array([-74, -74.01]), latitude: new Float32Array([40.71, 40.73]), measure: new Float32Array([10, 20])})
  .concat(arrow.tableFromArrays({longitude: new Float32Array([-74.02, -74.03]), latitude: new Float32Array([40.75, 40.77]), measure: new Float32Array([30, 40])}));
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  console.log('Initial host imports and GPU fixture ready');
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: 'recovery-source.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))});
  await workspace.getByLabel('Source longitude column', {exact: true}).selectOption('longitude');
  await workspace.getByLabel('Source latitude column', {exact: true}).selectOption('latitude');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 4);
  const datasetId = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);
  await page.evaluate(datasetId => {
    window.__keplerComputeTest.setRange(40, 75);
    window.__keplerComputeTest.setPublishedRange(datasetId, 'measure', 20, 40);
  }, datasetId);
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 3);
  await closeNativeDialog(page);
  const analysis = page.getByRole('region', {name: 'GPU analysis', exact: true});
  await analysis.getByLabel('Dataset', {exact: true}).selectOption(datasetId);
  await analysis.getByLabel('Numeric field', {exact: true}).selectOption('measure');
  await analysis.getByLabel('Operation', {exact: true}).selectOption('z-score');
  await analysis.getByLabel('Analysis population', {exact: true}).selectOption('selected');
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('3 analyzed'));
  const before = await page.evaluate(() => {
    window.previousRecoveryResources = window.__keplerComputeTest.resourceIdentity();
    return window.__keplerComputeTest.getState();
  });
  assert.equal(before.recovery.hasCheckpoint, true);
  console.log('Published source, filters and selected analysis ready');
  await page.evaluate(() => window.__keplerComputeTest.loseDevice());
  await page.waitForFunction(() => ['lost', 'failed'].includes(window.__keplerComputeTest.getState().recovery.phase));
  const lost = await page.evaluate(() => window.__keplerComputeTest.getState());
  assert.equal(lost.ready, false);
  assert.equal(lost.selectedCount, undefined);
  assert.equal(lost.recovery.hasCheckpoint, true);
  assert.equal(lost.recovery.sourceCount, 2);
  assert.match(await page.getByTestId('selected-count').innerText(), /Device unavailable\s+selected observations/);
  assert.equal(await page.getByRole('region', {name: 'GPU analysis', exact: true}).count(), 0);
  await page.getByRole('button', {name: 'Recover GPU workspace', exact: true}).click();
  await page.waitForFunction(({datasetId, count}) => {
    const state = window.__keplerComputeTest.getState();
    if (state.recovery.phase === 'failed') throw new Error(state.recovery.reason);
    return state.ready && state.recovery.phase === 'ready' && state.selectedCount === count &&
      state.publications.find(source => source.id === datasetId)?.selectedCount === 3;
  }, {datasetId, count: before.selectedCount}, {timeout: 60000});
  const after = await page.evaluate(datasetId => {
    const api = window.__keplerComputeTest;
    const previous = window.previousRecoveryResources;
    const current = api.resourceIdentity();
    return {
      state: api.getState(),
      differentDevice: current.device !== previous.device,
      differentBuffers: current.sourceBuffers[datasetId].every((buffer, index) => buffer !== previous.sourceBuffers[datasetId][index]),
      analysis: api.analysisForDataset(datasetId).settings
    };
  }, datasetId);
  assert.equal(after.state.error, '');
  assert.equal(after.differentDevice, true);
  assert.equal(after.differentBuffers, true);
  assert(after.state.deviceGeneration > before.deviceGeneration);
  assert.deepEqual(after.state.publications[0].sourceBatches, [2, 2]);
  assert.equal(after.analysis.mode, 'z-score');
  assert.equal(after.analysis.column, 'measure');
  assert.equal(after.analysis.population, 'selected');
  await page.getByRole('region', {name: 'GPU analysis', exact: true}).getByLabel('Dataset', {exact: true}).selectOption(datasetId);
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('3 analyzed'));
  await page.waitForFunction(datasetId => window.__keplerComputeTest.renderedLayers().some(layer => layer.id === `${datasetId}-points` && layer.visible && layer.loaded), datasetId);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, checks: ['actual device destruction', 'stale result invalidation', 'explicit recovery button', 'replacement Deck device', 'new source buffers', 'original dataset IDs and batches', 'native filters restored', 'selected analysis recipe restored']}));
} catch (error) {
  console.error(await page.evaluate(() => window.__keplerComputeTest?.getState()));
  console.error(await page.locator('body').innerText());
  throw error;
} finally { await browser.close(); }
