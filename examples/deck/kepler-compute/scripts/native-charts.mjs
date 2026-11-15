// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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
const page = await browser.newPage({viewport: {width: 1600, height: 1100}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => {
  errors.push(error.message);
  console.error(error.message);
});
page.on('response', response => {
  if (response.status() >= 400) console.error(response.status(), response.url());
});
const card = page.locator('.resident-chart-card').first();
async function waitValue(expected, tolerance = 0.0001) {
  await page.waitForFunction(
    ({expected, tolerance}) => {
      const label = document
        .querySelector('.resident-chart-card [aria-label^="Chart value:"]')
        ?.getAttribute('aria-label');
      return label && Math.abs(Number(label.replace('Chart value: ', '')) - expected) <= tolerance;
    },
    {expected, tolerance}
  );
}
try {
  await page.goto(getFixtureUrl(process.argv[2], {rows: '256'}), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(
    () => {
      const state = window.__keplerComputeTest?.getState();
      if (state?.error) throw new Error(state.error);
      return state?.selectedCount === 256;
    },
    undefined,
    {timeout: 60000}
  );
  const source = new arrow.Table({
    amount: arrow.vectorFromArray([10, null, 20, 40, 80, 160], new arrow.Float32()),
    whole: arrow.vectorFromArray([10, null, 20, 40, 80, 160], new arrow.Int32()),
    rank: arrow.makeVector(new Float32Array([0, 1, 2, 3, 4, 5]))
  });
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: 'chart-measures.arrow',
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(source.slice(0, 3).concat(source.slice(3)), 'stream'))
  });
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 6
  );
  const datasetId = await page.evaluate(
    () => window.__keplerComputeTest.getState().publications[0].id
  );
  const originalStorage = await page.evaluate(() => {
    const state = window.__keplerComputeTest.getState();
    const source = state.publications[0];
    return {
      fixtureUpload: state.uploadBytes,
      revision: source.sourceRevision,
      upload: source.sourceUploadBytes,
      buffers: source.sourceBufferIds,
      batches: source.sourceBatches
    };
  });
  await closeNativeDialog(page);
  await page.getByRole('button', {name: 'Toggle charts', exact: true}).click();
  await page.locator('.chart-config__type').click();
  await page.getByText('Big Number', {exact: true}).click();
  await waitValue(256);
  const chartColors = await card.evaluate(element => ({
    label: getComputedStyle(element.querySelector('label')).color,
    value: getComputedStyle(element.querySelector('[aria-label="Chart value: 256"] > div > div'))
      .color,
    background: getComputedStyle(element).backgroundColor
  }));
  assert.deepEqual(chartColors, {
    label: 'rgb(237, 242, 247)',
    value: 'rgb(237, 242, 247)',
    background: 'rgb(36, 39, 48)'
  });
  await card.getByLabel('Chart field', {exact: true}).selectOption('value');
  await card.getByLabel('Chart measure', {exact: true}).selectOption('average');
  const fixtureMean = await page.evaluate(() => {
    const values = window.__keplerComputeTest.source().map(row => row[2]);
    return values.reduce((total, value) => total + value, 0) / values.length;
  });
  await waitValue(fixtureMean);
  await card.getByLabel('Chart dataset', {exact: true}).selectOption(datasetId);
  await card.getByLabel('Chart title', {exact: true}).fill('Source measurement summary');
  await card.getByLabel('Chart field', {exact: true}).selectOption('amount');
  await card.getByLabel('Chart measure', {exact: true}).selectOption('average');
  await waitValue(62);
  await page.evaluate(
    id => window.__keplerComputeTest.setPublishedRange(id, 'rank', 2, 4),
    datasetId
  );
  await waitValue(140 / 3);
  await card.getByLabel('Chart measure', {exact: true}).selectOption('count');
  await waitValue(3);
  await card.getByLabel('Chart measure', {exact: true}).selectOption('minimum');
  await waitValue(20);
  await card.getByLabel('Chart measure', {exact: true}).selectOption('maximum');
  await waitValue(80);
  await card.getByLabel('Chart measure', {exact: true}).selectOption('sum');
  await waitValue(140);
  await card.getByLabel('Apply map filters', {exact: true}).uncheck();
  await waitValue(310);
  await card.getByLabel('Show chart caption', {exact: true}).uncheck();
  assert.equal(await card.getByText('sum of amount', {exact: true}).count(), 0);
  await card.getByLabel('Chart measure', {exact: true}).selectOption('median');
  await waitValue(40);
  await card.getByLabel('Chart measure', {exact: true}).selectOption('stdev');
  await waitValue(Math.sqrt(3720), 0.001);
  // Int32 fields widen exactly on the GPU for the same measures.
  await card.getByLabel('Chart field', {exact: true}).selectOption('whole');
  await waitValue(Math.sqrt(3720), 0.001);
  await card.getByLabel('Chart measure', {exact: true}).selectOption('median');
  await waitValue(40);
  await card.getByLabel('Chart measure', {exact: true}).selectOption('sum');
  await waitValue(310);
  await card.getByLabel('Chart field', {exact: true}).selectOption('amount');
  await card.getByLabel('Chart measure', {exact: true}).selectOption('variance');
  await card.getByText(/variance measure needs a resident statistical-summary adapter/).waitFor();
  assert.equal(await card.locator('[aria-label^="Chart value:"]').count(), 0);
  await card.getByLabel('Chart measure', {exact: true}).selectOption('sum');
  await waitValue(310);
  await card.getByLabel('Keep chart visible', {exact: true}).check();
  const saved = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const savedChart = saved.config.visState.charts.find(
    chart => chart.title === 'Source measurement summary'
  );
  assert.equal(savedChart.dataId, datasetId);
  assert.equal(savedChart.axis.field.name, 'amount');
  assert.equal(savedChart.axis.aggregation, 'sum');
  assert.equal(savedChart.applyFilters, false);
  assert.equal(savedChart.pinned, true);
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), saved);
  await waitValue(310);
  const finalStorage = await page.evaluate(() => {
    const state = window.__keplerComputeTest.getState();
    const source = state.publications[0];
    return {
      fixtureUpload: state.uploadBytes,
      revision: source.sourceRevision,
      upload: source.sourceUploadBytes,
      buffers: source.sourceBufferIds,
      batches: source.sourceBatches
    };
  });
  assert.deepEqual(finalStorage, originalStorage);
  assert.deepEqual(errors, []);
  await page.screenshot({path: '/tmp/kepler-native-charts.png'});
  await card.getByRole('button', {name: 'Remove chart', exact: true}).click();
  await card.waitFor({state: 'hidden'});
  console.log(
    JSON.stringify({
      passed: true,
      checks: [
        'native chart creation',
        'two unrelated sources',
        'nullable multi-batch measures',
        'map-selected versus source population',
        'median and deviation on Float32 and Int32',
        'unsupported statistic clears previous value',
        'caption and pin controls',
        'native saved replay',
        'source storage unchanged',
        'chart removal'
      ]
    })
  );
} catch (error) {
  console.error(errors);
  console.error(
    await page
      .evaluate(() => ({
        state: window.__keplerComputeTest?.getState(),
        text: document.body.innerText
      }))
      .catch(() => 'Page unavailable')
  );
  throw error;
} finally {
  await browser.close();
}
