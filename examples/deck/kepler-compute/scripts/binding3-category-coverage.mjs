// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Two resident datasets with different category dictionaries share one multiSelect filter that selects
// a label only one of them has. The coverage warning names the dataset lacking it and clears when the
// filter is removed. Run headless (default): node scripts/binding3-category-coverage.mjs [url]
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {getFixtureUrl} from './script-fixture-url.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
page.setDefaultTimeout(30000);
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
const state = () => page.evaluate(() => window.__keplerComputeTest.getState());
const makeTable = labels =>
  new arrow.Table({
    x: arrow.makeVector(new Float32Array(labels.map((_, index) => -74 - index / 100))),
    y: arrow.makeVector(new Float32Array(labels.map((_, index) => 40.7 + index / 100))),
    kind: arrow.vectorFromArray(labels)
  });
async function publish(name, labels) {
  const before = (await state()).publications.length;
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name,
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(makeTable(labels), 'stream'))
  });
  await workspace.getByLabel('Source longitude column').selectOption('x');
  await workspace.getByLabel('Source latitude column').selectOption('y');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(
    count => window.__keplerComputeTest.getState().publications.length === count + 1,
    before,
    {timeout: 45000}
  );
  await closeNativeDialog(page);
  return (await state()).publications[before].id;
}
const apply = config =>
  page.evaluate(value => window.__keplerComputeTest.applyNativeConfig(value), config);
const checks = [];
try {
  await page.goto(getFixtureUrl(process.argv[2], {rows: '256'}), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(
    () => window.__keplerComputeTest?.getState().selectedCount === 256,
    undefined,
    {timeout: 60000}
  );
  const first = await publish('coverage-a.arrow', ['red', 'blue', 'red', 'teal']);
  const second = await publish('coverage-b.arrow', ['blue', 'green', 'blue', 'red']);
  const base = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  assert.equal((await state()).categoryCoverage.length, 0, 'no warning before the filter');
  const withFilter = structuredClone(base);
  withFilter.config.visState.filters = [
    ...withFilter.config.visState.filters.filter(filter => filter.id !== 'coverage-kind'),
    {
      id: 'coverage-kind',
      dataId: [first, second],
      name: ['kind', 'kind'],
      type: 'multiSelect',
      value: ['red', 'teal'],
      fixedDomain: true
    }
  ];
  await apply(withFilter);
  // 'teal' exists only in the first dataset, so the second reports it as missing.
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().categoryCoverage.length > 0,
    undefined,
    {timeout: 30000}
  );
  const warnings = (await state()).categoryCoverage;
  assert(
    warnings.every(warning => warning.missingLabels.includes('teal')),
    `missing label teal: ${JSON.stringify(warnings)}`
  );
  assert(
    warnings.some(warning => warning.datasetId === second),
    'the second dataset reports the missing label'
  );
  assert(
    !warnings.some(warning => warning.datasetId === first),
    'the first dataset has every selected label'
  );
  const text = await page.getByTestId('category-coverage-warning').innerText();
  assert(text.includes('teal') && text.includes(second), `warning text: ${text}`);
  checks.push('warning shown for the dataset lacking the label, in the UI and debug state');
  await apply(base);
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().categoryCoverage.length === 0,
    undefined,
    {timeout: 30000}
  );
  assert.equal(await page.getByTestId('category-coverage-warning').count(), 0);
  checks.push('warning cleared when the filter is removed');
  assert.deepEqual(errors, [], `page errors: ${errors.join('; ')}`);
  console.log(JSON.stringify({passed: true, checks}));
} catch (error) {
  console.error(
    await page
      .locator('body')
      .innerText()
      .catch(() => '')
  );
  throw error;
} finally {
  await browser.close();
}
