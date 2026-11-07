// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser check for ?automatic=all: forced device loss recovers without the explicit button, and a
// hidden resident table is evicted under memory pressure and rebuilt on demand. Requires WebGPU
// and a running dev server, so it is not part of the node test run.
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const source = arrow.tableFromArrays({longitude: new Float32Array([-74, -74.01]), latitude: new Float32Array([40.71, 40.73]), measure: new Float32Array([10, 20])})
  .concat(arrow.tableFromArrays({longitude: new Float32Array([-74.02, -74.03]), latitude: new Float32Array([40.75, 40.77]), measure: new Float32Array([30, 40])}));
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer|BindGroup)|validation error|exceeds the maximum|Uncaught|destroyed/i.test(message.text()))
    errors.push(message.text());
});
const watchdog = setTimeout(() => { console.error('Automatic recovery check exceeded 140 seconds'); void browser.close(); process.exit(1); }, 140000);
const getState = () => page.evaluate(() => window.__keplerComputeTest.getState());
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256&automatic=all', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  const baseLayerIds = await page.evaluate(() => window.__keplerComputeTest.getState().nativeLayers.map(layer => layer.id));
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: 'automatic-source.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))});
  await workspace.getByLabel('Source longitude column', {exact: true}).selectOption('longitude');
  await workspace.getByLabel('Source latitude column', {exact: true}).selectOption('latitude');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 4);
  const datasetId = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);
  await closeNativeDialog(page);
  let initial = await getState();
  assert.equal(initial.automaticRecovery.enabled, true);
  assert.equal(initial.automaticEviction.enabled, true);
  assert.equal(initial.recovery.hasCheckpoint, true);
  const publishedLayerIds = initial.nativeLayers.map(layer => layer.id).filter(id => !baseLayerIds.includes(id));
  assert(publishedLayerIds.length > 0, 'publishing created native layers');
  console.log(JSON.stringify({stage: 'published', datasetId, publishedLayerIds}));

  // Part 2/3 run before and after recovery, since recovery replaces the registry and ledger.
  async function checkEviction(label) {
  // Eviction: a visible table is never evicted; a hidden one is evicted under pressure.
  const beforePressure = await getState();
  const beforeBytes = beforePressure.publications[0].sourceUploadBytes;
  const visibleResult = await page.evaluate(() => { try { window.__keplerComputeTest.applyMemoryPressure(); return 'reserved'; } catch (error) { return error.message; } });
  assert.equal(visibleResult, 'Dataset memory budget exceeded', 'nothing evictable while the table is visible');
  assert.deepEqual((await getState()).automaticEviction.evictedIds, [], 'visible table is not evicted');
  await page.evaluate(ids => ids.forEach(id => window.__keplerComputeTest.setLayerVisible(id, false)), publishedLayerIds);
  await page.waitForFunction(ids => window.__keplerComputeTest.getState().nativeLayers.filter(layer => ids.includes(layer.id)).every(layer => !layer.visible), publishedLayerIds);
  console.log(JSON.stringify({stage: 'pre-pressure', label, diagnostics: await page.evaluate(() => window.__keplerComputeTest.getEvictionDiagnostics())}));
  // Eviction releases bytes after GPU fences, so the triggering reservation itself may still fail.
  await page.evaluate(() => { try { window.__keplerComputeTest.applyMemoryPressure(); } catch (error) { if (error.message !== 'Dataset memory budget exceeded') throw error; } });
  await page.waitForFunction(datasetId => {
    const state = window.__keplerComputeTest.getState();
    return state.automaticEviction.evictionCount >= 1 && state.automaticEviction.evictedIds.includes(datasetId);
  }, datasetId, {timeout: 30000});
  await page.waitForFunction(datasetId => !window.__keplerComputeTest.getState().publications.some(source => source.id === datasetId), datasetId, {timeout: 30000});
  const evicted = await getState();
  assert.equal(evicted.error, '');
  assert.match(await page.getByTestId('automatic-eviction-status').innerText(), /evicted/);
  console.log(JSON.stringify({stage: 'evicted', label, automaticEviction: evicted.automaticEviction}));

  // Rebuild: showing the table again rebuilds it from the retained source.
  await page.evaluate(ids => ids.forEach(id => window.__keplerComputeTest.setLayerVisible(id, true)), publishedLayerIds);
  await page.waitForFunction(datasetId => {
    const state = window.__keplerComputeTest.getState();
    if (state.automaticEviction.failedIds.length) throw new Error(state.automaticEviction.lastError);
    return state.automaticEviction.rebuildCount >= 1 && state.automaticEviction.evictedIds.length === 0 &&
      state.publications.find(source => source.id === datasetId)?.selectedCount === 4;
  }, datasetId, {timeout: 60000});
  await page.waitForFunction(datasetId => window.__keplerComputeTest.renderedLayers().some(layer => layer.id === `${datasetId}-points` && layer.visible && layer.loaded), datasetId, {timeout: 30000});
  const rebuilt = await getState();
  assert.equal(rebuilt.error, '');
  assert.deepEqual(rebuilt.publications[0].sourceBatches, [2, 2]);
  assert.equal(rebuilt.publications[0].sourceUploadBytes, beforeBytes, label);
  assert.equal(rebuilt.automaticEviction.failedIds.length, 0);
  }
  await checkEviction('before-loss');
  // Part 1: automatic recovery, no button click.
  await page.evaluate(() => {
    window.previousRecoveryResources = window.__keplerComputeTest.resourceIdentity();
    window.__keplerComputeTest.loseDevice();
  });
  await page.waitForFunction(() => {
    const state = window.__keplerComputeTest.getState();
    return state.recovery.phase !== 'ready' || state.automaticRecovery.attempts > 0;
  });
  await page.waitForFunction(({datasetId, generation}) => {
    const state = window.__keplerComputeTest.getState();
    if (state.recovery.phase === 'failed' && state.automaticRecovery.state === 'exhausted') throw new Error(`Automatic recovery exhausted: ${state.recovery.reason}`);
    return state.ready && state.recovery.phase === 'ready' && state.deviceGeneration > generation &&
      state.publications.find(source => source.id === datasetId)?.selectedCount === 4;
  }, {datasetId, generation: initial.deviceGeneration}, {timeout: 60000});
  assert.equal(await page.getByRole('button', {name: 'Recover GPU workspace', exact: true}).count(), 0);
  const recovered = await page.evaluate(datasetId => {
    const api = window.__keplerComputeTest;
    const previous = window.previousRecoveryResources;
    const current = api.resourceIdentity();
    return {
      state: api.getState(),
      differentDevice: current.device !== previous.device,
      differentBuffers: current.sourceBuffers[datasetId].every((buffer, index) => buffer !== previous.sourceBuffers[datasetId][index])
    };
  }, datasetId);
  assert.equal(recovered.differentDevice, true);
  assert.equal(recovered.differentBuffers, true);
  assert.equal(recovered.state.error, '');
  assert.deepEqual(recovered.state.publications[0].sourceBatches, [2, 2]);
  assert(recovered.state.automaticRecovery.attempts <= 1 || recovered.state.automaticRecovery.state === 'idle');
  await page.waitForFunction(datasetId => window.__keplerComputeTest.renderedLayers().some(layer => layer.id === `${datasetId}-points` && layer.visible && layer.loaded), datasetId);
  console.log(JSON.stringify({stage: 'automatically-recovered', generation: recovered.state.deviceGeneration, automaticRecovery: recovered.state.automaticRecovery}));

  await checkEviction('after-recovery');
  await page.waitForTimeout(1000);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, checks: ['automatic recovery without button', 'new device generation and buffers', 'published source restored', 'visible table not evicted', 'hidden table evicted under pressure', 'rebuild on demand', 'no validation errors'], ...(await getState()).automaticEviction}));
} catch (error) {
  console.error(errors);
  console.error(await page.evaluate(() => window.__keplerComputeTest?.getState()).catch(() => undefined));
  console.error(await page.locator('body').innerText().catch(() => ''));
  throw error;
} finally { clearTimeout(watchdog); await browser.close(); }
