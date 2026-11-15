// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Tester-written (TESTER4) discovery listing for PLATFORM2 catalog bindings.
// Original header: Browser check for the contributor catalog panel: discovery with reasons, a typed run of the
// built-in network service area on the GPU contributors (engine shown in the panel), the CPU
// reference engine as an explicit choice, GPU-only overlay and noding operations, serialization
// restore, and staging the result into the Sources workflow, then publishing it as a native dataset.
// Usage: node scripts/catalog-discovery.mjs [URL] (default: http://127.0.0.1:5318/?rows=256).
import {VALIDATION_OPERATIONS} from './analysis-catalog-validation.mjs';
import {checkPointCloudCatalogDiscovery} from './analysis-catalog-pointcloud.mjs';
import assert from 'node:assert/strict';
import {NETWORK_CATEGORICAL_OPERATIONS} from './network-categorical-catalog-utils.mjs';
import {assertTrajectoryCatalogDiscovery} from './catalog-trajectory.mjs';
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

  await assertTrajectoryCatalogDiscovery(panel);
  const operations = await panel.getByLabel('Operation', {exact: true}).locator('option').evaluateAll(options => options.map(option => option.value));
  console.log(JSON.stringify({summary, operationCount: operations.length, operations}));
  await checkPointCloudCatalogDiscovery(panel);
  assert(operations.includes('network-snap-points'), 'network-snap-points is listed');
  for (const [operationId] of NETWORK_CATEGORICAL_OPERATIONS) assert(operations.includes(operationId), `${operationId} is listed`);
  for (const [operation] of VALIDATION_OPERATIONS) assert(operations.includes(operation), `${operation} is listed`);
  assert(operations.length >= 40, 'at least 40 runnable operations');
  await panel.getByLabel('Operation', {exact: true}).selectOption('network-snap-points');
  await panel.getByTestId('catalog-run').click();
  await page.waitForTimeout(3000);
  console.log(JSON.stringify({run: (await panel.innerText()).slice(0, 1500)}));
  assert.deepEqual(failures, []);
  console.log(JSON.stringify({passed: true}));
} finally { clearTimeout(watchdog); await browser.close(); }
