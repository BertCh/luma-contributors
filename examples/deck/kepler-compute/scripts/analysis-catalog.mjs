// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser check for the contributor catalog panel: discovery with reasons, a typed run of the
// built-in network service area on the GPU contributors (engine shown in the panel), the CPU
// reference engine as an explicit choice, GPU-only overlay and noding operations, serialization
// restore, and staging the result into the Sources workflow, then publishing it as a native dataset.
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const failures = [];
page.on('pageerror', error => failures.push(error.message));
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text()))
    failures.push(message.text());
});
const watchdog = setTimeout(() => {
  console.error('Catalog workflow exceeded 120 seconds');
  void browser.close();
}, 120000);

try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  const panel = page.getByRole('region', {name: 'Analysis catalog', exact: true});
  await panel.waitFor();

  // Discovery: some contributors are runnable, the rest list a reason.
  const summary = await panel.getByTestId('catalog-summary').innerText();
  assert.match(summary, /contributors are runnable here/);
  await panel.getByRole('button', {name: /not runnable/}).click();
  assert.match(await panel.getByTestId('catalog-blocked').innerText(), /No executable host binding/);

  // Service area on the sample network: statuses and source edge IDs.
  await panel.getByLabel('Operation', {exact: true}).selectOption('network-service-area');
  await panel.getByLabel(/^Coordinate system/).selectOption('planar');
  await panel.getByLabel(/^One-way property/).fill('oneway');
  await panel.getByLabel(/^Maximum snap distance/).fill('1');
  await panel.getByLabel(/^Cost budgets/).fill('15');
  await panel.getByTestId('catalog-run').click();
  const result = panel.getByTestId('catalog-result');
  await result.waitFor();
  const text = await result.innerText();
  assert.match(text, /reachedEdgeCount: 3/);
  assert.match(text, /unreachable-directed/);
  assert.match(text, /unreachable-disconnected/);

  // The panel names the engine that ran: GPU contributors with a device, CPU reference on request.
  assert.match(await panel.getByTestId('catalog-device').innerText(), /GPU device/);
  assert.match(await panel.getByTestId('catalog-engine').innerText(), /Engine: GPU \(.*GPUNetworkReachability/);
  await panel.getByLabel('Engine', {exact: true}).selectOption('cpu');
  await panel.getByTestId('catalog-run').click();
  await page.waitForFunction(
    () => /Engine: CPU reference/.test(document.querySelector('[data-testid="catalog-engine"]')?.textContent ?? '')
  );
  assert.match(await result.innerText(), /reachedEdgeCount: 3/);
  await panel.getByLabel('Engine', {exact: true}).selectOption('auto');

  // GPU-only operations: overlay intersection and crossing noding on their samples.
  await panel.getByLabel('Operation', {exact: true}).selectOption('geometry-overlay-intersection');
  await panel.getByTestId('catalog-run').click();
  await page.waitForFunction(
    () => /GPUPolygonOverlay/.test(document.querySelector('[data-testid="catalog-engine"]')?.textContent ?? '')
  );
  assert.match(await panel.getByTestId('catalog-result').innerText(), /Overlay intersection: [1-9]\d* rows/);
  await panel.getByLabel('Operation', {exact: true}).selectOption('network-crossing-noding');
  await panel.getByTestId('catalog-run').click();
  await page.waitForFunction(
    () => /GPUNetworkNoding/.test(document.querySelector('[data-testid="catalog-engine"]')?.textContent ?? '')
  );
  await panel.getByLabel('Operation', {exact: true}).selectOption('network-service-area');
  await panel.getByLabel(/^Coordinate system/).selectOption('planar');
  await panel.getByLabel(/^One-way property/).fill('oneway');
  await panel.getByLabel(/^Maximum snap distance/).fill('1');
  await panel.getByLabel(/^Cost budgets/).fill('15');

  // Serialization round trip restores parameters.
  await panel.getByRole('button', {name: 'Serialize invocation', exact: true}).click();
  const serialized = JSON.parse(await panel.getByLabel('Serialized invocation', {exact: true}).inputValue());
  assert.equal(serialized.operationId, 'network-service-area');
  assert.deepEqual(serialized.parameters.budgets, [15]);
  assert.equal(typeof serialized.inputs.lines.contentHash, 'string');
  await panel.getByLabel(/^Cost budgets/).fill('99');
  await panel.getByRole('button', {name: 'Restore parameters', exact: true}).click();
  assert.equal(await panel.getByLabel(/^Cost budgets/).inputValue(), '15');

  // Invalid parameter is refused with a visible error.
  await panel.getByLabel(/^Cost budgets/).fill('-3');
  await panel.getByTestId('catalog-run').click();
  await panel.getByTestId('catalog-error').waitFor();
  await panel.getByLabel(/^Cost budgets/).fill('15');

  // Geometry workflow: convex clip returns new part identities.
  await panel.getByLabel('Operation', {exact: true}).selectOption('geometry-convex-clip');
  await panel.getByTestId('catalog-run').click();
  assert.match(await panel.getByTestId('catalog-result').innerText(), /Clipped features: 1 rows/);

  // Stage the result in Sources and publish it as a native dataset.
  await panel.getByTestId('catalog-publish').click();
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.getByText('Clipped features').first().waitFor({timeout: 15000});
  assert.deepEqual(failures, []);
  await closeNativeDialog(page);
  console.log(JSON.stringify({ok: true}));
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
