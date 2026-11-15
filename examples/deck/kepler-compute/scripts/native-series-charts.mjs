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
page.setDefaultTimeout(30000);
page.on('requestfailed', request =>
  console.error('Request failed', request.url(), request.failure())
);
page.on('response', response => {
  if (response.status() >= 400) console.error('HTTP', response.status(), response.url());
});
page.on('console', message => {
  if (message.type() === 'error') console.error(message.text());
});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => {
  errors.push(error.message);
  console.error(error.stack);
});
const card = page.locator('.resident-chart-card').first();
const plot = card.getByRole('status', {name: 'Chart category summary', exact: true});
async function waitBins(expected) {
  await page.waitForFunction(
    expected =>
      JSON.stringify(
        [...document.querySelectorAll('[aria-label="Chart category summary"] [role=button]')].map(
          button => button.getAttribute('aria-label')
        )
      ) === JSON.stringify(expected),
    expected
  );
}
async function readSeries() {
  return card
    .getByRole('table', {name: 'Chart series breakdown'})
    .locator('tbody tr')
    .evaluateAll(rows => rows.map(row => [...row.children].map(cell => cell.textContent.trim())));
}
try {
  await page.goto(getFixtureUrl(process.argv[2], {rows: '256'}), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(
    () => window.__keplerComputeTest?.getState().selectedCount === 256,
    undefined,
    {timeout: 60000}
  );
  const source = new arrow.Table({
    category: arrow.vectorFromArray([
      'Beta',
      'Alpha',
      'Beta',
      'Alpha',
      'Beta',
      'Alpha',
      'Beta',
      'Alpha',
      'Gamma',
      'Delta',
      '',
      null
    ]),
    series: arrow.vectorFromArray([
      null,
      'Two',
      'One',
      'One',
      'One',
      'Two',
      'Two',
      '',
      'Two',
      null,
      'One',
      'Two'
    ]),
    amount: arrow.vectorFromArray(
      [100, 4, 2, -8, 6, null, 10, 200, null, 400, 600, 800],
      new arrow.Float32()
    ),
    numeric: arrow.vectorFromArray(
      [0, 2, 0, 2, 0, 2, 0, 2, null, null, null, null],
      new arrow.Int32()
    )
  });
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: 'chart-series.arrow',
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(
      arrow.tableToIPC(source.slice(0, 3).concat(source.slice(3, 7), source.slice(7)), 'stream')
    )
  });
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 12
  );
  const initial = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0]);
  await closeNativeDialog(page);
  await page.getByRole('button', {name: 'Toggle charts', exact: true}).click();
  await page.locator('.chart-config__type').click();
  await page.getByRole('dialog').getByText('Bar', {exact: true}).click();
  await card.getByLabel('Chart dataset', {exact: true}).selectOption(initial.id);
  await card.getByLabel('Chart categories', {exact: true}).selectOption('category');
  await card.getByLabel('Chart field', {exact: true}).selectOption('amount');
  await card.getByLabel('Chart measure', {exact: true}).selectOption('average');
  await card.getByLabel('Chart series', {exact: true}).selectOption('series');
  const barMarks = [
    'Beta · One: 4 (2 rows)',
    'Alpha · One: -8 (1 row)',
    'Beta · Two: 10 (1 row)',
    'Gamma · Two: 0 (1 row)',
    'Alpha · Two: 4 (2 rows)'
  ];
  await waitBins(barMarks);
  assert.equal(
    await plot.locator('.resident-series-plot > title').textContent(),
    'Bar chart by series'
  );
  await plot.locator('.resident-series-plot').scrollIntoViewIfNeeded();
  await page.screenshot({path: '/tmp/kepler-native-series-bars.png'});
  const expectedSeries = [
    ['Beta', 'One', '4', '2'],
    ['Beta', 'Two', '10', '1'],
    ['Gamma', 'Two', '0', '1'],
    ['Alpha', 'Two', '4', '2'],
    ['Alpha', 'One', '-8', '1']
  ];
  assert.deepEqual(await readSeries(), expectedSeries);
  const betaMark = plot.getByRole('button', {name: 'Beta · One: 4 (2 rows)', exact: true});
  await betaMark.focus();
  await betaMark.press('Enter');
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 4
  );
  await waitBins(barMarks);
  assert.equal(await plot.locator('[role=button][aria-pressed=true]').count(), 2);
  assert.deepEqual(await readSeries(), expectedSeries);
  const selected = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const selectedChart = selected.config.visState.charts.find(chart => chart.dataId === initial.id);
  assert.deepEqual(
    selected.config.visState.filters.find(
      filter => filter.id === selectedChart.crossFilter.filterId
    ).value,
    ['Beta']
  );
  await card.getByLabel('Apply map filters', {exact: true}).uncheck();
  await card.getByLabel('Maximum displayed groups', {exact: true}).fill('1');
  await card.getByLabel('Combine remaining groups as Other', {exact: true}).check();
  await waitBins(['Beta · One: 4 (2 rows)', 'Beta · Two: 10 (1 row)']);
  await plot
    .getByRole('img', {name: 'Other · Combined groups: -4 (6 rows)', exact: true})
    .waitFor();
  assert.equal(await plot.getByRole('button', {name: /^Other/}).count(), 0);
  await card.getByLabel('Maximum displayed groups', {exact: true}).fill('10');
  await card.getByLabel('Combine remaining groups as Other', {exact: true}).uncheck();
  await waitBins(barMarks);
  const saved = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const chart = saved.config.visState.charts.find(chart => chart.dataId === initial.id);
  assert.equal(chart.groupBy.field.name, 'series');
  chart.type = 'lineChart';
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), saved);
  await waitBins([
    'Alpha · One: -8 (1 row)',
    'Beta · One: 4 (2 rows)',
    'Alpha · Two: 4 (2 rows)',
    'Beta · Two: 10 (1 row)',
    'Gamma · Two: 0 (1 row)'
  ]);
  assert.equal(
    await plot.locator('.resident-series-plot > title').textContent(),
    'Line chart by series'
  );
  // Delta has no series: Two's Gamma observation must not connect across that missing group.
  assert.deepEqual(
    await plot
      .locator('polyline')
      .evaluateAll(lines =>
        lines.map(line => line.getAttribute('points').trim().split(/\s+/).length)
      ),
    [2, 2]
  );
  assert.deepEqual(await readSeries(), [...expectedSeries.slice(3), ...expectedSeries.slice(0, 3)]);
  await page.screenshot({path: '/tmp/kepler-native-series-charts.png'});
  await card.getByLabel('Chart categories', {exact: true}).selectOption('numeric');
  await waitBins([
    '0 · One: 4 (2 rows)',
    '2 · One: -8 (1 row)',
    '0 · Two: 10 (1 row)',
    '2 · Two: 4 (2 rows)'
  ]);
  const numericSaved = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const numericChart = numericSaved.config.visState.charts.find(
    chart => chart.dataId === initial.id
  );
  numericChart.type = 'horizontalBar';
  [numericChart.xAxis, numericChart.yAxis] = [numericChart.yAxis, numericChart.xAxis];
  numericChart.yAxis.aggregation = 'numericBin';
  numericChart.numGroups = 4;
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), numericSaved);
  await waitBins([
    '0 – 0.5 · One: 4 (2 rows)',
    '1.5 – 2 · One: -8 (1 row)',
    '0 – 0.5 · Two: 10 (1 row)',
    '1.5 – 2 · Two: 4 (2 rows)'
  ]);
  assert.equal(
    await plot.locator('.resident-series-plot > title').textContent(),
    'Horizontal bar chart by series'
  );
  await plot.locator('.resident-series-plot').scrollIntoViewIfNeeded();
  await page.screenshot({path: '/tmp/kepler-native-series-horizontal.png'});
  await plot.getByRole('button', {name: '1.5 – 2 · One: -8 (1 row)', exact: true}).press(' ');
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 4
  );
  const intervalSelection = await page.evaluate(() =>
    window.__keplerComputeTest.saveNativeConfig()
  );
  const intervalChart = intervalSelection.config.visState.charts.find(
    chart => chart.dataId === initial.id
  );
  assert.deepEqual(
    intervalSelection.config.visState.filters.find(
      filter => filter.id === intervalChart.crossFilter.filterId
    ).value,
    [1.5, 2]
  );
  await card.getByLabel('Chart series', {exact: true}).selectOption('amount');
  // Numeric series are binned on category axes, but a numeric-bin axis still refuses them with this reason.
  await card
    .getByText(
      'A numeric series needs a category axis; numeric-bin axes with a numeric series are not supported.'
    )
    .waitFor();
  await card.getByLabel('Chart series', {exact: true}).selectOption('');
  await waitBins(['0 – 0.5: 29.5', '0.5 – 1: 0', '1 – 1.5: 0', '1.5 – 2: 65.33']);
  assert.equal(await card.getByRole('table', {name: 'Chart series breakdown'}).count(), 0);
  const final = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0]);
  for (const name of ['sourceRevision', 'sourceUploadBytes', 'sourceBufferIds', 'sourceBatches'])
    assert.deepEqual(final[name], initial[name]);
  assert.deepEqual(initial.sourceBatches, [3, 4, 5]);
  assert.deepEqual(errors, []);
  await card.getByRole('button', {name: 'Remove chart', exact: true}).click();
  await card.waitFor({state: 'hidden'});
  console.log(
    JSON.stringify({
      passed: true,
      checks: [
        'native series control',
        'separate grouped vertical series with negative and zero values',
        'missing and null series',
        'source-local series order',
        'keyboard parent-group selection and self-filter exclusion',
        'Other is an unselectable aggregate',
        'config replay',
        'separate line segments break at missing categories',
        'integer series groups',
        'horizontal interval series and exact Space selection',
        'numeric series refused on a numeric-bin axis',
        'series removal',
        'unchanged source buffers and chunks'
      ]
    })
  );
} catch (error) {
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
