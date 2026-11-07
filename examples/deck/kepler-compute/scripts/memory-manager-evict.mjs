// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Tester-written (TESTER4) check of Memory manager Evict/Unload actions with ?automatic=evict.
// Original header: Browser check for ?automatic=all: forced device loss recovers without the explicit button, and a
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
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256&automatic=evict', {waitUntil: 'domcontentloaded'});
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
    assert.equal(initial.automaticEviction.enabled, true);
  const publishedLayerIds = initial.nativeLayers.map(layer => layer.id).filter(id => !baseLayerIds.includes(id));
  assert(publishedLayerIds.length > 0, 'publishing created native layers');
  console.log(JSON.stringify({stage: 'published', datasetId, publishedLayerIds}));


  async function openModal() {
    const section = page.getByTestId('native-memory');
    await section.waitFor({timeout: 30000});
    if (!(await page.getByTestId('memory-manager-open').isVisible())) await section.locator('summary').click();
    await page.getByTestId('memory-manager-open').click();
    const modal = page.getByTestId('memory-manager-full');
    await modal.waitFor({timeout: 10000});
    return modal;
  }
  // Evict: hide the layers so the table is evictable, press Evict.
  await page.evaluate(ids => ids.forEach(id => window.__keplerComputeTest.setLayerVisible(id, false)), publishedLayerIds);
  let modal = await openModal();
  const row = modal.locator(`[data-dataset-id="${datasetId}"]`);
  await row.waitFor({timeout: 10000});
  await row.getByRole('button', {name: 'Evict'}).click();
  await page.waitForFunction(id => window.__keplerComputeTest.getState().automaticEviction.evictedIds.includes(id), datasetId, {timeout: 30000});
  console.log(JSON.stringify({stage: 'evicted', status: (await getState()).automaticEviction}));
  // Layer shown again reloads the dataset.
  await page.keyboard.press('Escape');
  await page.evaluate(ids => ids.forEach(id => window.__keplerComputeTest.setLayerVisible(id, true)), publishedLayerIds);
  await page.waitForFunction(id => {
    const state = window.__keplerComputeTest.getState();
    return state.automaticEviction.evictedIds.length === 0 && state.publications.find(source => source.id === id)?.selectedCount === 4;
  }, datasetId, {timeout: 60000});
  console.log(JSON.stringify({stage: 'reloaded'}));
  // Unload removes dataset and its layers.
  modal = await openModal();
  await modal.locator(`[data-dataset-id="${datasetId}"]`).getByRole('button', {name: 'Unload'}).click();
  await page.waitForFunction(id => {
    const state = window.__keplerComputeTest.getState();
    return !state.publications.some(source => source.id === id);
  }, datasetId, {timeout: 30000});
  const after = await getState();
  assert(!after.nativeLayers.some(layer => publishedLayerIds.includes(layer.id)), 'unload removed the dataset layers');
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, checks: ['evict', 'reload', 'unload']}));
} finally { clearTimeout(watchdog); await browser.close(); }
