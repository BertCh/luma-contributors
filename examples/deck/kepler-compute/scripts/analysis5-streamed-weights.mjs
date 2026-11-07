// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const source = arrow.tableFromArrays({
  longitude: new Float32Array([-74, -74.01]),
  latitude: new Float32Array([40.7, 40.72]),
  measure: new Float32Array([10, 20]),
  population: new Float32Array([100, 100]),
  trend: new Float32Array([1, 2])
}).concat(arrow.tableFromArrays({
  longitude: new Float32Array([-74.02, -74.03]),
  latitude: new Float32Array([40.74, 40.76]),
  measure: new Float32Array([30, 40]),
  population: new Float32Array([100, 100]),
  trend: new Float32Array([3, 4])
}));
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
const diagnostics = [];
page.on('pageerror', error => { errors.push(error.message); console.error('pageerror:', error.stack ?? error.message); });
page.on('requestfailed', request => { diagnostics.push({url: request.url(), error: request.failure()?.errorText}); console.error('requestfailed:', request.url(), request.failure()?.errorText); });
page.on('console', message => { if (message.type() === 'error') { diagnostics.push({console: message.text()}); console.error('console:', message.text()); } });
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
  // Point hot spots on a distance band, first on the single CSR.
  await panel.getByText('Configure hot spots on geographic points', {exact: true}).click();
  await panel.getByLabel('Hot-spot longitude field', {exact: true}).selectOption('longitude');
  await panel.getByLabel('Hot-spot latitude field', {exact: true}).selectOption('latitude');
  await panel.getByLabel('Hot-spot topology', {exact: true}).selectOption('distance-band');
  await panel.getByLabel('Hot-spot radius in meters', {exact: true}).fill('10000');
  await panel.getByLabel('Hot-spot directed neighbor capacity', {exact: true}).fill('12');
  await panel.getByLabel('Hot-spot minimum neighbors', {exact: true}).fill('1');
  await panel.getByLabel('Point hot-spot correction', {exact: true}).selectOption('none');
  await panel.getByRole('button', {name: 'Apply neighbors and run point hot spots', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('Point weights path: single CSR'), undefined, {timeout: 45000});
  const singleText = await panel.innerText();
  const linkCount = /([\d,]+) point neighbor links/.exec(singleText)?.[1];
  assert(linkCount, 'single CSR reports its neighbor links');
  // Force the streamed path with tiny chunks: the same links, computed per chunk.
  await page.evaluate(id => window.__keplerComputeTest.setStreamedPointWeights(id, {forceStreamed: true, maximumSlotsPerChunk: 6}), datasetId);
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('Point weights path: streamed row chunks'), undefined, {timeout: 45000});
  const streamedText = await panel.innerText();
  assert.match(streamedText, /streamed row chunks \((\d+) chunks?, forced\)/);
  assert.match(streamedText, new RegExp(`${linkCount} point neighbor links`));
  assert(Number(/\((\d+) chunks?/.exec(streamedText)[1]) > 1, 'forced chunk size splits the rows');
  // Back to the device-limit decision: the single CSR again.
  await page.evaluate(id => window.__keplerComputeTest.setStreamedPointWeights(id, {}), datasetId);
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('Point weights path: single CSR'), undefined, {timeout: 45000});
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, checks: ['point hot spots on the single CSR report their path', 'forced streamed chunks report path, chunk count and the same neighbor links', 'returning to the device-limit decision restores the single CSR']}));
} catch (failure) {
  console.error(JSON.stringify({pageErrors: errors, diagnostics, body: (await page.locator('body').innerText()).slice(0, 2000)}));
  throw failure;
} finally {
  await browser.close();
}
