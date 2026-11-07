// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => {
  if (/unresolved value|Compilation error|Error while parsing WGSL/.test(message.text()))
    errors.push(message.text());
});
const rowCount = 410;
const csv = Buffer.from(
  'latitude,longitude,value,note\n' +
    Array.from({length: rowCount}, (_, index) =>
      `${40.5 + (index % 5) / 32},${-74 + (index % 7) / 32},${index},${'x'.repeat(1500)}`
    ).join('\n')
);

try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => {
    const state = window.__keplerComputeTest?.getState();
    if (state?.error) throw new Error(state.error);
    return state?.selectedCount === 100000;
  }, undefined, {timeout: 60000});
  const baselineLayerIds = await page.evaluate(() =>
    window.__keplerComputeTest.renderedLayers().map(layer => layer.id)
  );
  await page.getByRole('button', {name: 'Add Data', exact: true}).click();
  await page.locator('input[type=file]').setInputFiles({
    name: 'small-observations.csv', mimeType: 'text/csv', buffer: csv
  });
  await page.locator('.ReactModal__Content').getByRole('button', {name: 'Add Data', exact: true}).click();
  // Native Load Files is prepared automatically: no preparation notice, no manual publish step.
  await page.waitForFunction(count => {
    const state = window.__keplerComputeTest.getState();
    if (state.error) throw new Error(state.error);
    return state.publications.some(publication => publication.selectedCount === count);
  }, rowCount, {timeout: 60000});
  const state = await page.evaluate(() => window.__keplerComputeTest.getState());
  const published = state.publications.find(publication => publication.selectedCount === rowCount);
  assert.equal(published.error, null);
  assert.equal(await page.getByTestId('resident-workflow-notice').count(), 0);
  const report = page.getByTestId('auto-prepare-report');
  await report.waitFor();
  const reportText = await report.innerText();
  assert.match(reportText, /small-observations/);
  assert.match(reportText, /Longitude\/latitude: longitude\/latitude/);
  assert.match(reportText, /Float64 fields: round to Float32/);
  await page.waitForFunction(
    baseline => window.__keplerComputeTest.renderedLayers().some(
      layer => !baseline.includes(layer.id) && layer.loaded && layer.visible
    ),
    baselineLayerIds,
    {timeout: 30000}
  );
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, rowCount, csvBytes: csv.byteLength,
    checks: ['default startup', 'native CSV ingestion', 'automatic preparation without notice', 'visible policy report', 'native point rendering', 'no WGSL errors']}));
} catch (error) {
  console.error(await page.locator('body').innerText());
  throw error;
} finally {
  await browser.close();
}
