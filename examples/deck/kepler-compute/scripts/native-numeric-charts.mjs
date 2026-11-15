// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {dirname} from 'node:path';
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
page.on('console', message => {
  if (message.type() === 'error') console.error(message.text());
});
await pinViteModules(page);
const errors = [];
const {displayBinKey} = createRequire(import.meta.url)('@kepler.gl/charts');
// Pinned native plot code is the oracle for the range filter plot of the published field.
const nativePlot = createRequire(import.meta.url)(
  `${dirname(createRequire(import.meta.url).resolve('@kepler.gl/utils'))}/plot.js`
);
page.on('pageerror', error => {
  errors.push(error.message);
  console.error(error.stack);
});
const card = page.locator('.resident-chart-card').first();
const bars = card.getByRole('status', {name: 'Chart category summary', exact: true});
async function waitBars(expected) {
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
try {
  await page.goto(getFixtureUrl(process.argv[2], {rows: '256'}), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(
    () => window.__keplerComputeTest?.getState().selectedCount === 256,
    undefined,
    {timeout: 60000}
  );
  const source = new arrow.Table({
    value: arrow.vectorFromArray([0, 1, 2, 3, 4, null, null], new arrow.Int32()),
    amount: arrow.vectorFromArray([10, 20, 30, 40, 50, 99, 99], new arrow.Float32()),
    rank: arrow.makeVector(new Float32Array([0, 1, 2, 3, 4, 5, 6]))
  });
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: 'chart-numeric.arrow',
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(source.slice(0, 3).concat(source.slice(3)), 'stream'))
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
  await page.getByText('Bar', {exact: true}).click();
  await card.getByLabel('Chart dataset', {exact: true}).selectOption(initial.id);
  await card.getByLabel('Chart categories', {exact: true}).selectOption('value');
  await card.getByLabel('Maximum displayed groups', {exact: true}).fill('10');
  await waitBars(['0: 1', '1: 1', '2: 1', '3: 1', '4: 1']);
  await card.getByLabel('Maximum displayed groups', {exact: true}).fill('3');
  const autoBoundaries = [0, 4 / 3, 8 / 3, 4];
  await waitBars(
    [0, 2, 1].map(
      index =>
        `${displayBinKey(`${autoBoundaries[index]}\u001f${autoBoundaries[index + 1]}`)}: ${index === 1 ? 1 : 2}`
    )
  );
  await card.getByLabel('Chart binning', {exact: true}).selectOption('numericBin');
  await card.getByLabel('Maximum displayed groups', {exact: true}).fill('4');
  await waitBars(['0 – 1: 1', '1 – 2: 1', '2 – 3: 1', '3 – 4: 2']);
  await bars.getByRole('button', {name: '1 – 2: 1', exact: true}).click();
  // Native numeric crossfilters use inclusive [minimum, maximum] even though interior bins are half-open.
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 2
  );
  await waitBars(['0 – 1: 1', '1 – 2: 1', '2 – 3: 1', '3 – 4: 2']);
  const savedSelection = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  assert.deepEqual(
    savedSelection.config.visState.filters.find(
      filter => filter.dataId.includes(initial.id) && filter.name.includes('value')
    ).value,
    [1, 2]
  );
  await page.evaluate(
    id => window.__keplerComputeTest.setPublishedRange(id, 'rank', 2, 6),
    initial.id
  );
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 1
  );
  await waitBars(['2 – 2.5: 1', '2.5 – 3: 0', '3 – 3.5: 1', '3.5 – 4: 1']);
  // The published rank range filter plots with the native edges (d3 ticks, numeric interval) and
  // native line-chart measures, with the chart's own value filter applied and its own range excluded.
  {
    const rankId = `published-${initial.id}-rank`;
    const columns = {
      value: [0, 1, 2, 3, 4, null, null],
      amount: [10, 20, 30, 40, 50, 99, 99],
      rank: [0, 1, 2, 3, 4, 5, 6]
    };
    const rows = columns.rank
      .map((_, row) => row)
      .filter(
        row => columns.value[row] !== null && columns.value[row] >= 1 && columns.value[row] <= 2
      );
    const edgesOf = bins => bins.map(bin => [bin.x0, bin.x1, bin.count]);
    const readFilter = () =>
      page.evaluate(
        id => window.__keplerComputeTest.getState().nativeFilters.find(filter => filter.id === id),
        rankId
      );
    const waitPlot = async (predicate, argument) => {
      await page.waitForFunction(
        ({id, predicate, argument}) => {
          const filter = window.__keplerComputeTest
            .getState()
            .nativeFilters.find(candidate => candidate.id === id);
          return (
            filter &&
            new Function('filter', 'argument', `return (${predicate})(filter, argument)`)(
              filter,
              argument
            )
          );
        },
        {id: rankId, predicate: predicate.toString(), argument}
      );
      return readFilter();
    };
    const setPlot = newProp =>
      page.evaluate(({id, newProp}) => window.__keplerComputeTest.setFilterPlot(id, newProp), {
        id: rankId,
        newProp
      });
    let rank = await waitPlot(
      filter =>
        filter.lumaComputePlot?.status === 'ready' && filter.bins?.[filter.dataId[0]]?.length
    );
    const accessor = row => columns.rank[row];
    assert.deepEqual(
      edgesOf(rank.bins[initial.id]),
      nativePlot
        .histogramFromDomain(rank.domain, rows, 30, accessor)
        .map(bin => [bin.x0, bin.x1, bin.count]),
      'rank plot uses native tick edges'
    );
    await setPlot({plotType: {numericInterval: 2}});
    rank = await waitPlot(
      filter =>
        filter.plotType?.numericInterval === 2 &&
        filter.lumaComputePlot?.status === 'ready' &&
        filter.bins?.[filter.dataId[0]]?.length
    );
    assert.deepEqual(
      edgesOf(rank.bins[initial.id]),
      edgesOf(
        nativePlot.histogramFromThreshold(
          nativePlot.getNumericBinThresholds(rank.domain, 2, 30),
          rows,
          accessor,
          false
        )
      ),
      'rank plot uses native numeric interval edges'
    );
    const dataset = {
      id: initial.id,
      fields: [{name: 'amount', type: 'real', valueAccessor: ({index}) => columns.amount[index]}]
    };
    await setPlot({plotType: {numericInterval: 6}});
    for (const aggregation of ['median', 'stdev', 'average']) {
      await setPlot({yAxis: {name: 'amount', type: 'real'}});
      await setPlot({plotType: {aggregation}});
      rank = await waitPlot(
        filter =>
          filter.plotType?.numericInterval === 6 &&
          filter.plotType?.aggregation === argument &&
          filter.lineChart?.aggregation === argument &&
          filter.lumaComputePlot?.status === 'ready',
        aggregation
      );
      const expected = nativePlot.getLineChart(
        {[initial.id]: dataset},
        {
          ...rank,
          lineChart: undefined,
          plotType: {type: 'lineChart', aggregation},
          bins: {
            [initial.id]: nativePlot.histogramFromThreshold(
              nativePlot.getNumericBinThresholds(rank.domain, 6, 30),
              rows,
              accessor,
              false
            )
          }
        }
      );
      const points = chart => chart.series.lines.map(line => line.map(point => [point.x, point.y]));
      assert.deepEqual(
        points(rank.lineChart).map(line => line.map(([x, y]) => [x, Math.round(y * 1e4) / 1e4])),
        points(expected).map(line => line.map(([x, y]) => [x, Math.round(y * 1e4) / 1e4])),
        `${aggregation} numeric series`
      );
      assert.equal(rank.lineChart.title, expected.title);
    }
    await setPlot({plotType: {numericInterval: null}});
    await setPlot({yAxis: null});
    await waitPlot(
      filter =>
        filter.plotType?.type === 'histogram' &&
        !filter.lineChart &&
        filter.lumaComputePlot?.status === 'ready' &&
        filter.bins?.[filter.dataId[0]]?.length
    );
  }
  await card.getByLabel('Crossfilter from chart', {exact: true}).uncheck();
  await card.getByLabel('Apply map filters', {exact: true}).uncheck();
  await card.getByLabel('Chart field', {exact: true}).selectOption('amount');
  await card.getByLabel('Chart measure', {exact: true}).selectOption('average');
  await waitBars(['0 – 1: 10', '1 – 2: 20', '2 – 3: 30', '3 – 4: 45']);
  await card.getByLabel('Combine remaining groups as Other', {exact: true}).check();
  await waitBars(['0 – 1: 10', '1 – 2: 20', '2 – 3: 30', '3 – 4: 45']);
  const saved = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const savedChart = saved.config.visState.charts.find(chart => chart.dataId === initial.id);
  savedChart.type = 'horizontalBar';
  [savedChart.xAxis, savedChart.yAxis] = [savedChart.yAxis, savedChart.xAxis];
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), saved);
  await waitBars(['0 – 1: 10', '1 – 2: 20', '2 – 3: 30', '3 – 4: 45']);
  await card.locator('.horizontal-bar-chart').waitFor();
  const final = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0]);
  for (const name of ['sourceRevision', 'sourceUploadBytes', 'sourceBufferIds', 'sourceBatches'])
    assert.deepEqual(final[name], initial[name]);
  assert.deepEqual(errors, []);
  await page.screenshot({path: '/tmp/kepler-native-numeric-charts.png'});
  // Native compact labels may collide at Uint32 extremes. Selection must retain exact interval edges.
  await openNativeData(page);
  const integers = new arrow.Table({
    value: arrow.makeVector(new Uint32Array([4294967293, 4294967294, 4294967295]))
  });
  await workspace.locator('input[type=file]').setInputFiles({
    name: 'chart-integers.arrow',
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(integers, 'stream'))
  });
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications.length === 2);
  const integerSource = await page.evaluate(
    () => window.__keplerComputeTest.getState().publications[1]
  );
  await closeNativeDialog(page);
  await card.getByLabel('Chart dataset', {exact: true}).selectOption(integerSource.id);
  await card.getByLabel('Chart categories', {exact: true}).selectOption('value');
  await card.getByLabel('Maximum displayed groups', {exact: true}).fill('4');
  await waitBars(['4294967293: 1', '4294967294: 1', '4294967295: 1']);
  await bars.getByRole('button', {name: '4294967293: 1', exact: true}).click();
  await page.waitForFunction(
    id =>
      window.__keplerComputeTest.getState().publications.find(source => source.id === id)
        ?.selectedCount === 1,
    integerSource.id
  );
  const singletonSelection = await page.evaluate(() =>
    window.__keplerComputeTest.saveNativeConfig()
  );
  assert.deepEqual(
    singletonSelection.config.visState.filters.find(
      filter => filter.dataId.includes(integerSource.id) && filter.name.includes('value')
    ).value,
    [4294967293, 4294967293]
  );
  await card.getByLabel('Chart binning', {exact: true}).selectOption('numericBin');
  await card.getByLabel('Maximum displayed groups', {exact: true}).fill('4');
  await waitBars(
    [1, 0, 1, 1].map(
      (count, index) =>
        `${displayBinKey(`${4294967293 + index / 2}\u001f${4294967293 + (index + 1) / 2}`)}: ${count}`
    )
  );
  await bars.getByRole('button').first().click();
  await page.waitForFunction(
    id =>
      window.__keplerComputeTest.getState().publications.find(source => source.id === id)
        ?.selectedCount === 1,
    integerSource.id
  );
  const integerSelection = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  assert.deepEqual(
    integerSelection.config.visState.filters.find(
      filter => filter.dataId.includes(integerSource.id) && filter.name.includes('value')
    ).value,
    [4294967293, 4294967293.5]
  );
  const lineChart = integerSelection.config.visState.charts.find(
    chart => chart.dataId === integerSource.id
  );
  lineChart.type = 'lineChart';
  lineChart.xAxis = {field: {name: 'value', type: 'integer'}, aggregation: 'uniqueBin'};
  lineChart.yAxis = {field: null, aggregation: 'count'};
  lineChart.applyFilters = false;
  lineChart.crossFilter.enabled = false;
  await page.evaluate(
    config => window.__keplerComputeTest.applyNativeConfig(config),
    integerSelection
  );
  await waitBars(['4294967293: 1', '4294967294: 1', '4294967295: 1']);
  await card.locator('svg:has(polyline)').waitFor();
  assert.equal(await card.getByLabel('Maximum displayed groups', {exact: true}).count(), 0);
  await bars.getByRole('button', {name: '4294967294: 1', exact: true}).click();
  const lineSelection = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  assert.deepEqual(
    lineSelection.config.visState.filters.find(
      filter => filter.dataId.includes(integerSource.id) && filter.name.includes('value')
    ).value,
    [4294967294, 4294967294]
  );
  await page.screenshot({path: '/tmp/kepler-native-line-chart.png'});
  assert.deepEqual(errors, []);
  await card.getByRole('button', {name: 'Remove chart', exact: true}).click();
  await card.waitFor({state: 'hidden'});
  console.log(
    JSON.stringify({
      passed: true,
      checks: [
        'native integer auto categories',
        'native integer interval fallback ranking',
        'exact singleton integer range selection',
        'native numeric bins',
        'range filter plot native tick and interval edges',
        'range filter plot median, deviation and average series',
        'nullable multi-batch counts',
        'inclusive native range selection',
        'own-field exclusion',
        'unrelated range domain',
        'empty bins',
        'all-source mean',
        'Other ignored for ordered bins',
        'horizontal native replay',
        'stable source buffers',
        'exact Uint32 interval selection despite identical compact labels',
        'native line replay and exact point selection',
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
