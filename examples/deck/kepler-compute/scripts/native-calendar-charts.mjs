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

const timeUnit = process.argv.includes('--second') ? 'second' : 'millisecond';
// Run the additional exact-hour acceptance checks with --hour for either Arrow time unit.
const includeHour = process.argv.includes('--hour');
const sandboxUrl = getFixtureUrl(
  process.argv.slice(2).find(argument => !argument.startsWith('--')),
  {rows: '256'}
);
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1600, height: 1100}});
page.setDefaultTimeout(30000);
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => {
  errors.push(error.message);
  console.error(error.stack);
});
page.on('requestfailed', request =>
  console.error('Request failed', request.url(), request.failure())
);
page.on('console', message => {
  if (message.type() === 'error') console.error(message.text());
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
async function assertCalendarSelection(dataId, selectedCount, start, end) {
  const expectedRange = [Date.parse(start), Date.parse(end)];
  // Consecutive selections can have equal row counts; wait for the exact persisted range as well.
  await page.waitForFunction(
    ({dataId, selectedCount, expectedRange}) => {
      const saved = window.__keplerComputeTest.saveNativeConfig();
      const chart = saved.config.visState.charts.find(chart => chart.dataId === dataId);
      const filter = saved.config.visState.filters.find(
        filter => filter.id === chart?.crossFilter?.filterId
      );
      return (
        window.__keplerComputeTest.getState().publications[0]?.selectedCount === selectedCount &&
        JSON.stringify(filter?.value) === JSON.stringify(expectedRange)
      );
    },
    {dataId, selectedCount, expectedRange}
  );
  const saved = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const chart = saved.config.visState.charts.find(chart => chart.dataId === dataId);
  assert.deepEqual(
    saved.config.visState.filters.find(filter => filter.id === chart.crossFilter.filterId).value,
    expectedRange
  );
}
try {
  await page.goto(sandboxUrl, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(
    () => window.__keplerComputeTest?.getState().selectedCount === 256,
    undefined,
    {timeout: 60000}
  );
  const instants = [
    '1969-12-31T23:59:59.999Z',
    '1970-01-01T00:00:00Z',
    '2024-02-29T23:59:59.999Z',
    '2024-03-01T00:00:00Z',
    '2024-03-10T03:00:00-04:00',
    '2024-04-01T00:00:00Z'
  ].map(value => {
    const milliseconds = Date.parse(value);
    return timeUnit === 'second' ? Math.floor(milliseconds / 1000) * 1000 : milliseconds;
  });
  const source = new arrow.Table({
    time: arrow.vectorFromArray(
      [...instants, null],
      timeUnit === 'second'
        ? new arrow.TimestampSecond('America/New_York')
        : new arrow.TimestampMillisecond('America/New_York')
    ),
    amount: arrow.vectorFromArray([1, 2, null, 4, 6, 8, 10], new arrow.Float32()),
    series: arrow.vectorFromArray(['Old', 'Old', 'New', 'One', 'Two', 'One', 'Old'])
  });
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: `chart-calendar-${timeUnit}.arrow`,
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(source.slice(0, 2).concat(source.slice(2)), 'stream'))
  });
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 7
  );
  const initial = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0]);
  await closeNativeDialog(page);
  await page.getByRole('button', {name: 'Toggle charts', exact: true}).click();
  await page.locator('.chart-config__type').click();
  await page.getByRole('dialog').getByText('Line', {exact: true}).click();
  await card.getByLabel('Chart dataset', {exact: true}).selectOption(initial.id);
  await card.getByLabel('Chart categories', {exact: true}).selectOption('time');
  await card.getByLabel('Chart binning', {exact: true}).selectOption('timeBin');
  await card.getByLabel('Calendar interval (UTC)', {exact: true}).selectOption('month');
  await card.getByLabel('Chart measure', {exact: true}).selectOption('count');
  await waitBins(['1969-12: 1', '1970-01: 2', '2024-02: 1', '2024-03: 2', '2024-04: 1']);
  await plot.getByRole('button', {name: '1970-01: 2', exact: true}).click();
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 2
  );
  const epochSaved = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const epochChart = epochSaved.config.visState.charts.find(chart => chart.dataId === initial.id);
  const epochFilter = epochSaved.config.visState.filters.find(
    filter => filter.id === epochChart.crossFilter.filterId
  );
  epochFilter.id = 'persisted-calendar-range';
  epochChart.crossFilter.filterId = epochFilter.id;
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), epochSaved);
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 2
  );
  // Retain the exact same persisted filter, but remove its chart owner: ordinary time filters reject nulls.
  const ordinary = structuredClone(epochSaved);
  ordinary.config.visState.charts = [];
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), ordinary);
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 1
  );
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), epochSaved);
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 2
  );
  await waitBins(['1969-12: 1', '1970-01: 2', '2024-02: 1', '2024-03: 2', '2024-04: 1']);
  await plot.getByRole('button', {name: '2024-03: 2', exact: true}).click();
  // Native period crossfilters include both endpoints, including the next month's first instant.
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 3
  );
  await waitBins(['1969-12: 1', '1970-01: 2', '2024-02: 1', '2024-03: 2', '2024-04: 1']);
  const savedSelection = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  assert.deepEqual(
    savedSelection.config.visState.filters.find(
      filter => filter.dataId.includes(initial.id) && filter.name.includes('time')
    ).value,
    [Date.parse('2024-03-01T00:00:00Z'), Date.parse('2024-04-01T00:00:00Z')]
  );
  await card.getByLabel('Apply map filters', {exact: true}).uncheck();
  await card.getByLabel('Chart field', {exact: true}).selectOption('amount');
  await card.getByLabel('Chart measure', {exact: true}).selectOption('average');
  await waitBins(['1969-12: 1', '1970-01: 6', '2024-02: 0', '2024-03: 5', '2024-04: 8']);
  await card.getByLabel('Chart series', {exact: true}).selectOption('series');
  await waitBins([
    '2024-02 · New: 0 (1 row)',
    '1969-12 · Old: 1 (1 row)',
    '1970-01 · Old: 6 (2 rows)',
    '2024-03 · One: 4 (1 row)',
    '2024-04 · One: 8 (1 row)',
    '2024-03 · Two: 6 (1 row)'
  ]);
  assert.equal(await plot.locator('.resident-series-legend li').count(), 4);
  assert.equal(
    await plot.locator('.resident-series-plot > title').textContent(),
    'Line chart by series'
  );
  assert.deepEqual(
    await plot
      .locator('polyline')
      .evaluateAll(lines =>
        lines.map(line => line.getAttribute('points').trim().split(/\s+/).length)
      ),
    [2, 2]
  );
  await card.getByRole('table', {name: 'Chart series breakdown'}).scrollIntoViewIfNeeded();
  await page.screenshot({path: `/tmp/kepler-native-calendar-${timeUnit}-charts.png`});
  const saved = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const chart = saved.config.visState.charts.find(chart => chart.dataId === initial.id);
  chart.xAxis.interval = 'year';
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), saved);
  await waitBins([
    '2024 · New: 0 (1 row)',
    '1969 · Old: 1 (1 row)',
    '1970 · Old: 6 (2 rows)',
    '2024 · One: 6 (2 rows)',
    '2024 · Two: 6 (1 row)'
  ]);
  // The chart excludes its own filter while computing all calendar groups.
  await card.getByLabel('Apply map filters', {exact: true}).check();
  await card.getByLabel('Calendar interval (UTC)', {exact: true}).selectOption('day');
  const dayMarks = [
    '2024-02-29 · New: 0 (1 row)',
    '1969-12-31 · Old: 1 (1 row)',
    '1970-01-01 · Old: 6 (2 rows)',
    '2024-03-01 · One: 4 (1 row)',
    '2024-04-01 · One: 8 (1 row)',
    '2024-03-10 · Two: 6 (1 row)'
  ];
  await waitBins(dayMarks);
  // Leap-day's nullable measurement remains a real zero mark; its inclusive range also selects March 1 midnight.
  await plot.getByRole('button', {name: '2024-02-29 · New: 0 (1 row)', exact: true}).press('Enter');
  await assertCalendarSelection(initial.id, 2, '2024-02-29T00:00:00Z', '2024-03-01T00:00:00Z');
  await waitBins(dayMarks);
  await card.getByLabel('Calendar interval (UTC)', {exact: true}).selectOption('week');
  const weekMarks = [
    'Week of 2024-02-25 · New: 0 (1 row)',
    'Week of 1969-12-28 · Old: 4.333 (3 rows)',
    'Week of 2024-02-25 · One: 4 (1 row)',
    'Week of 2024-03-31 · One: 8 (1 row)',
    'Week of 2024-03-10 · Two: 6 (1 row)'
  ];
  await waitBins(weekMarks);
  await plot.locator('.resident-series-plot').scrollIntoViewIfNeeded();
  await page.screenshot({path: `/tmp/kepler-native-calendar-${timeUnit}-week.png`});
  // Epoch and null share the Sunday-start week containing zero, together with the pre-epoch observation.
  await plot
    .getByRole('button', {name: 'Week of 1969-12-28 · Old: 4.333 (3 rows)', exact: true})
    .click();
  await assertCalendarSelection(initial.id, 3, '1969-12-28T00:00:00Z', '1970-01-04T00:00:00Z');
  await waitBins(weekMarks);
  // The offset-bearing March 10 Sunday belongs to this UTC week, not the preceding week.
  await plot
    .getByRole('button', {name: 'Week of 2024-03-10 · Two: 6 (1 row)', exact: true})
    .press(' ');
  await assertCalendarSelection(initial.id, 1, '2024-03-10T00:00:00Z', '2024-03-17T00:00:00Z');
  await waitBins(weekMarks);
  if (includeHour) {
    await card.getByLabel('Calendar interval (UTC)', {exact: true}).selectOption('hour');
    const hourMarks = [
      '2024-02-29 23:00 · New: 0 (1 row)',
      '1969-12-31 23:00 · Old: 1 (1 row)',
      '1970-01-01 00:00 · Old: 6 (2 rows)',
      '2024-03-01 00:00 · One: 4 (1 row)',
      '2024-04-01 00:00 · One: 8 (1 row)',
      '2024-03-10 07:00 · Two: 6 (1 row)'
    ];
    await waitBins(hourMarks);
    // The leap-day observation is in 23:00 UTC; native inclusive hour filtering also selects the next midnight.
    await plot
      .getByRole('button', {name: '2024-02-29 23:00 · New: 0 (1 row)', exact: true})
      .press('Enter');
    await assertCalendarSelection(initial.id, 2, '2024-02-29T23:00:00Z', '2024-03-01T00:00:00Z');
    await waitBins(hourMarks);
    await plot
      .getByRole('button', {name: '1970-01-01 00:00 · Old: 6 (2 rows)', exact: true})
      .press(' ');
    await assertCalendarSelection(initial.id, 2, '1970-01-01T00:00:00Z', '1970-01-01T01:00:00Z');
    await waitBins(hourMarks);
    const hourSaved = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
    const hourChart = hourSaved.config.visState.charts.find(chart => chart.dataId === initial.id);
    assert.equal(hourChart.xAxis.interval, 'hour');
    assert.equal(hourChart.crossFilter.filterId, epochFilter.id);
    await card.getByLabel('Calendar interval (UTC)', {exact: true}).selectOption('day');
    await waitBins(dayMarks);
    await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), hourSaved);
    await waitBins(hourMarks);
    await assertCalendarSelection(initial.id, 2, '1970-01-01T00:00:00Z', '1970-01-01T01:00:00Z');
    // Reuse the same chart ID after unmount: an old reader must not retire the new reader's named scope.
    const withoutHourChart = structuredClone(hourSaved);
    withoutHourChart.config.visState.charts = withoutHourChart.config.visState.charts.filter(
      chart => chart.id !== hourChart.id
    );
    await page.evaluate(
      config => window.__keplerComputeTest.applyNativeConfig(config),
      withoutHourChart
    );
    await card.waitFor({state: 'hidden'});
    await page.waitForFunction(
      () => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 1
    );
    await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), hourSaved);
    await waitBins(hourMarks);
    await assertCalendarSelection(initial.id, 2, '1970-01-01T00:00:00Z', '1970-01-01T01:00:00Z');
    await plot.locator('.resident-series-plot').scrollIntoViewIfNeeded();
    await page.screenshot({path: `/tmp/kepler-native-calendar-${timeUnit}-hour.png`});
  }
  await card.getByLabel('Calendar interval (UTC)', {exact: true}).selectOption('');
  // Unset interval is native automatic periods; this seven-row span resolves to years.
  await waitBins([
    '2024 · New: 0 (1 row)',
    '1969 · Old: 1 (1 row)',
    '1970 · Old: 6 (2 rows)',
    '2024 · One: 6 (2 rows)',
    '2024 · Two: 6 (1 row)'
  ]);
  await card.getByLabel('Calendar interval (UTC)', {exact: true}).selectOption('month');
  await waitBins([
    '2024-02 · New: 0 (1 row)',
    '1969-12 · Old: 1 (1 row)',
    '1970-01 · Old: 6 (2 rows)',
    '2024-03 · One: 4 (1 row)',
    '2024-04 · One: 8 (1 row)',
    '2024-03 · Two: 6 (1 row)'
  ]);
  const final = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0]);
  for (const name of ['sourceRevision', 'sourceUploadBytes', 'sourceBufferIds', 'sourceBatches'])
    assert.deepEqual(final[name], initial[name]);
  assert.deepEqual(initial.sourceBatches, [2, 5]);
  assert.deepEqual(errors, []);
  await card.getByRole('button', {name: 'Remove chart', exact: true}).click();
  await card.waitFor({state: 'hidden'});
  console.log(
    JSON.stringify({
      passed: true,
      timeUnit,
      includeHour,
      checks: [
        'native UTC month control',
        'nullable timestamps group at epoch',
        'nullable epoch click selects exact null and zero rows',
        'arbitrary persisted chart filter ID replay',
        'removing chart ownership restores ordinary null exclusion',
        'exact pre-epoch and leap-day grouping',
        'Arrow timezone ignored for UTC parity',
        'inclusive period click',
        'owned time-filter exclusion',
        'nullable measurement mean',
        'separate calendar series, missing gaps and supplied zero',
        'year config replay',
        'native UTC day labels and leap-day inclusive boundary selection',
        'native Sunday-week labels and pre-epoch grouping',
        'epoch-containing week selects null and zero',
        'Sunday week keyboard selection keeps exact native range',
        'day and week exclude their owned filter',
        ...(includeHour
          ? [
              'native UTC hour labels include pre-epoch and offset conversion',
              'leap-hour Enter selects inclusive midnight boundary',
              'epoch-hour Space selects null and zero',
              'hour excludes its owned filter',
              'hour config replay and same-ID remove/readd preserve exact range and filter ownership'
            ]
          : []),
        'native automatic interval resolves to years',
        'unchanged time/source buffers and batches',
        'removal'
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
