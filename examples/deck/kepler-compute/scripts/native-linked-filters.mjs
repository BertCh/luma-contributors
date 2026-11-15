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

// autoPrepare=0 keeps addNativeCpuDataset tables native CPU datasets (otherwise they become resident).
// Pinned native plot/time code is the oracle for CPU datasets, bin edges and line-chart series.
const require = createRequire(import.meta.url);
const nativeDirectory = dirname(require.resolve('@kepler.gl/utils'));
const nativePlot = createRequire(import.meta.url)(`${nativeDirectory}/plot.js`);
const nativeTime = createRequire(import.meta.url)(`${nativeDirectory}/time.js`);
const stopAfter = process.env.LINKED_PHASE ?? 'all';

const DAY = 86400000;
const START = Date.UTC(2020, 0, 1);
const makeTimes = days => days.map(day => (day === null ? null : START + day * DAY));
const SOURCE = {
  a: {
    amount: [5, 10, 15, 20, 25, 30, 35, 40, null, 45, 50, 12],
    days: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    rank: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    kind: [
      'red',
      'blue',
      'red',
      'green',
      'blue',
      'red',
      'amber',
      'green',
      'amber',
      'red',
      'blue',
      'green'
    ]
  },
  b: {
    amount: [20, 40, 60, 80, 100, null, 30, 50, 70, 90],
    days: [4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
    rank: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    score: [110, 120, 130, 140, 150, 160, 300, 170, 180, 190]
  },
  c: {
    amount: [-10, 0, 10, 20, 30, 40, 50, 60, null, 70, 15, 25, 35, 45],
    days: [-12, -8, -4, 0, 3, 6, 9, 12, 15, 18, 21, 22, 24, 27]
  }
};
const times = {
  a: makeTimes(SOURCE.a.days),
  b: makeTimes(SOURCE.b.days),
  c: makeTimes(SOURCE.c.days)
};
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1600, height: 1100}});
page.setDefaultTimeout(30000);
page.on('console', message => {
  if (message.type() === 'error') console.error(message.text());
});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => {
  errors.push(error.message);
  console.error(error.stack);
});
const state = () => page.evaluate(() => window.__keplerComputeTest.getState());
const checks = [];
function check(name) {
  checks.push(name);
}
async function publish(name, source) {
  const columns = {
    amount: arrow.vectorFromArray(source.amount, new arrow.Float32()),
    when: arrow.vectorFromArray(makeTimes(source.days), new arrow.TimestampMillisecond('UTC')),
    rank: arrow.vectorFromArray(source.rank, new arrow.Float32())
  };
  if (source.score) columns.score = arrow.vectorFromArray(source.score, new arrow.Float32());
  if (source.kind) columns.kind = arrow.vectorFromArray(source.kind);
  const table = new arrow.Table(columns);
  const before = (await state()).publications.length;
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: `${name}.arrow`,
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(table, 'stream'))
  });
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(
    count =>
      window.__keplerComputeTest.getState().publications.length === count + 1 &&
      window.__keplerComputeTest.getState().publications[count].selectedCount !== undefined,
    before
  );
  await closeNativeDialog(page);
  return (await state()).publications[before].id;
}
const panels = page.locator('.filter-panel');
// Native lists the newest filter first.
async function panelFor(id) {
  const filters = (await state()).nativeFilters;
  return panels.nth(filters.length - 1 - filters.findIndex(filter => filter.id === id));
}
// The native synced-dataset panel: "+" adds a row for the first compatible dataset (immediately
// when it has one compatible field, otherwise as a pending row); then pick its dataset and field.
async function linkDataset(id, datasetLabel, fieldName) {
  const panel = await panelFor(id);
  await panel.locator('button.add-sync-dataset').click();
  await panel.getByText('Resident observations', {exact: true}).last().click();
  await page.getByText(datasetLabel, {exact: true}).last().click();
  const placeholder = panel.getByText('Select a field', {exact: true});
  if (await placeholder.count()) {
    await placeholder.click();
    await page
      .locator('.list__item__anchor', {hasText: new RegExp(`^${fieldName}$`)})
      .last()
      .click();
  }
}
// The native numeric inputs under the range slider.
async function setRangeInputs(panel, minimum, maximum) {
  const inputs = panel.locator('input.kg-range-slider__input');
  await inputs.nth(0).fill(String(minimum));
  await inputs.nth(0).press('Enter');
  await inputs.nth(1).fill(String(maximum));
  await inputs.nth(1).press('Enter');
}
const filterById = async id => (await state()).nativeFilters.find(filter => filter.id === id);
async function waitFilter(id, predicate, description, argument) {
  try {
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
      {id, predicate: predicate.toString(), argument},
      {timeout: 20000}
    );
  } catch (error) {
    console.error(
      description,
      JSON.stringify(await filterById(id), (key, value) => (key === 'indexes' ? undefined : value))
    );
    throw error;
  }
  return filterById(id);
}
const readyLinked = filter => filter.lumaComputePlot?.status === 'ready';
// Native leaves the y domain undefined when no point has a value; identical values are equal.
const closeTo = (actual, expected, label) =>
  assert(
    actual === expected || Math.abs(actual - expected) < 1e-4,
    `${label}: ${actual} != ${expected}`
  );
const inRange = (value, range) => value !== null && value >= range[0] && value <= range[1];
// Resident numeric plots use the native range-plot edges (d3 ticks of the shared filter domain, or the
// numeric interval): the pinned native histogram functions are the oracle, nulls included.
function nativeHistogram(values, rows, domain, numericInterval) {
  const accessor = row => values[row];
  return Number.isFinite(numericInterval) && numericInterval > 0
    ? nativePlot.histogramFromThreshold(
        nativePlot.getNumericBinThresholds(domain, numericInterval, 30),
        rows,
        accessor,
        false
      )
    : nativePlot.histogramFromDomain(domain, rows, 30, accessor);
}
const nativeBins = (values, rows, domain, numericInterval) =>
  nativeHistogram(values, rows, domain, numericInterval).map(bin => [bin.x0, bin.x1, bin.count]);
const edgesOf = bins => bins.map(bin => [bin.x0, bin.x1, bin.count]);
const countsOf = bins => bins.map(bin => bin.count);
try {
  await page.goto(getFixtureUrl(process.argv[2], {rows: '256', autoPrepare: '0'}), {
    waitUntil: 'domcontentloaded'
  });
  await page.waitForFunction(
    () => window.__keplerComputeTest?.getState().selectedCount === 256,
    undefined,
    {timeout: 60000}
  );
  const ids = {a: await publish('link-a', SOURCE.a)};
  ids.b = await publish('link-b', SOURCE.b);
  ids.c = 'cpu-observations';
  await page.evaluate(
    ({id, amount, times}) =>
      window.__keplerComputeTest.addNativeCpuDataset(
        id,
        [
          {name: 'amount', type: 'real', format: '', analyzerType: 'FLOAT'},
          {name: 'when', type: 'timestamp', format: 'x', analyzerType: 'DATETIME'}
        ],
        amount.map((value, index) => [value, times[index]])
      ),
    {id: ids.c, amount: SOURCE.c.amount, times: times.c}
  );
  for (let count = 0; count < 3; count++)
    await page.evaluate(() => window.__keplerComputeTest.removeFilter(0));
  await page.locator('.side-panel__tab[data-for="filter-nav"]').click();

  const rankIds = SOURCE.a.rank.map((_, index) => index);

  // 1. Range filter linked through the native "Add synced filter" UI.
  const amountId = `published-${ids.a}-amount`;
  const rankId = `published-${ids.a}-rank`;
  const whenId = `published-${ids.a}-when`;
  await page.evaluate(
    ({a}) => {
      window.__keplerComputeTest.setPublishedRange(a, 'amount', 10, 40);
      window.__keplerComputeTest.setPublishedRange(a, 'rank', 1, 9);
    },
    {a: ids.a}
  );
  await linkDataset(amountId, 'link-b.arrow', 'amount');
  await linkDataset(amountId, 'cpu-observations', 'amount');
  let amount = await waitFilter(
    amountId,
    filter => filter.dataId.length === 3 && filter.lumaComputePlot?.status === 'ready',
    'linked amount'
  );
  assert.deepEqual(amount.dataId, [ids.a, ids.b, ids.c]);
  assert.deepEqual(amount.name, ['amount', 'amount', 'amount']);
  assert.deepEqual(amount.domain, [-10, 100], 'native merged domain');
  assert.deepEqual(amount.value, [10, 40]);
  // Once the linked time filter exists it applies to every dataset (resident time columns and CPU).
  let timeValue = null;
  const passesTime = (key, row) => timeValue === null || inRange(times[key][row], timeValue);
  const ownExcluded = () => ({
    a: rankIds.filter(row => inRange(SOURCE.a.rank[row], [1, 9]) && passesTime('a', row)),
    b: SOURCE.b.amount.map((_, row) => row).filter(row => passesTime('b', row)),
    c: SOURCE.c.amount.map((_, row) => row).filter(row => passesTime('c', row))
  });
  let numericInterval;
  async function assertAmountBins(
    filter,
    mapping = {a: 'amount', b: 'amount'},
    cachedCpuBins = undefined
  ) {
    const rows = ownExcluded();
    assert.deepEqual(
      edgesOf(filter.bins[ids.a]),
      nativeBins(SOURCE.a[mapping.a], rows.a, filter.domain, numericInterval),
      'resident A bins: native edges and counts (own filter excluded, rank filter applied)'
    );
    assert.deepEqual(
      edgesOf(filter.bins[ids.b]),
      nativeBins(SOURCE.b[mapping.b], rows.b, filter.domain, numericInterval),
      'resident B bins: native edges and counts'
    );
    for (const id of [ids.a, ids.b])
      for (const bin of filter.bins[id])
        assert.deepEqual(bin.indexes, [], 'resident bins expose no row arrays');
    // Native only drops the plot of the dataset whose field changed; other datasets keep their cached bins.
    if (filter.dataId.includes(ids.c)) {
      const oracle =
        cachedCpuBins ?? nativeBins(SOURCE.c.amount, rows.c, filter.domain, numericInterval);
      assert.deepEqual(edgesOf(filter.bins[ids.c]), oracle, 'native CPU dataset bins');
      // Resident and CPU series share their edges exactly.
      if (!cachedCpuBins)
        assert.deepEqual(
          filter.bins[ids.a].map(bin => [bin.x0, bin.x1]),
          filter.bins[ids.c].map(bin => [bin.x0, bin.x1]),
          'resident and CPU edges agree'
        );
    }
    assert.deepEqual(
      Object.keys(filter.bins).sort(),
      filter.dataId.slice().sort(),
      'one entry per dataset; never summed'
    );
  }
  await assertAmountBins(amount);
  async function assertMasks(filter, mapping = {a: 'amount', b: 'amount'}) {
    const rangeOf = filter.value;
    const expectedA = rankIds.filter(
      row =>
        inRange(SOURCE.a[mapping.a][row], rangeOf) &&
        inRange(SOURCE.a.rank[row], [1, 9]) &&
        passesTime('a', row)
    ).length;
    const expectedB = SOURCE.b[mapping.b].filter(
      (value, row) => inRange(value, rangeOf) && passesTime('b', row)
    ).length;
    const expectedC = SOURCE.c.amount.filter(value => inRange(value, rangeOf)).length;
    await page.waitForFunction(
      ({a, b, expectedA, expectedB}) => {
        const publications = window.__keplerComputeTest.getState().publications;
        return (
          publications.find(publication => publication.id === a)?.selectedCount === expectedA &&
          publications.find(publication => publication.id === b)?.selectedCount === expectedB
        );
      },
      {a: ids.a, b: ids.b, expectedA, expectedB}
    );
    // A native CPU dataset applies a GPU range filter through its channel range (value minus domain minimum).
    if (filter.dataId.includes(ids.c)) {
      const channel = filter.gpuChannel[filter.dataId.indexOf(ids.c)];
      assert.deepEqual(
        await page
          .evaluate(id => window.__keplerComputeTest.getNativeGpuFilterRange(id), ids.c)
          .then(ranges => ranges[channel]),
        [rangeOf[0] - filter.domain[0], rangeOf[1] - filter.domain[0]],
        'native CPU dataset channel range'
      );
    }
    return {expectedA, expectedB, expectedC};
  }
  console.log('amount masks', JSON.stringify(await assertMasks(amount)));
  check(
    'native synced-dataset UI links two resident datasets and one CPU dataset to a range filter'
  );
  check('merged native domain, per-dataset bins, own-filter exclusion, per-dataset masks');
  if (stopAfter === 'range') throw new Error('stopped after range phase');
  // 2. Moving the range slider keeps every dataset's mask and bins current.
  await setRangeInputs(await panelFor(amountId), 30, 60);
  amount = await waitFilter(
    amountId,
    filter =>
      filter.value[0] === 30 &&
      filter.lumaComputePlot?.status === 'ready' &&
      filter.bins &&
      filter.bins['cpu-observations'],
    'moved range'
  );
  await assertAmountBins(amount);
  console.log('moved masks', JSON.stringify(await assertMasks(amount)));
  check('range changes update resident and CPU masks and bins');
  await setRangeInputs(await panelFor(amountId), 10, 40);
  amount = await waitFilter(
    amountId,
    filter =>
      filter.value[0] === 10 &&
      filter.value[1] === 40 &&
      filter.lumaComputePlot?.status === 'ready',
    'reset range'
  );

  // 3. Linked time filter and its native interval selection.
  await page.evaluate(
    ({a, start, end}) => window.__keplerComputeTest.setPublishedRange(a, 'when', start, end),
    {a: ids.a, start: START + 1 * DAY, end: START + 30 * DAY}
  );
  // The native "+" first links the only compatible dataset with one time field (the resident
  // observations); native builds its default plot type for that merged domain and keeps it when the
  // row is then repointed at the requested dataset. Resident tables expose no mapped timestamps.
  const observationDomain = await page.evaluate(() => window.__keplerComputeTest.timestampDomain());
  const firstDomain = [
    Math.min(START, observationDomain[0]),
    Math.max(START + 11 * DAY, observationDomain[1])
  ];
  const unmapped = id => ({id, getColumnField: () => ({filterProps: {}})});
  const nativeDefault = nativeTime.getInitialInterval(
    {domain: firstDomain, dataId: [ids.a, 'resident-observations'], name: ['when', 'timestamp']},
    {[ids.a]: unmapped(ids.a), 'resident-observations': unmapped('resident-observations')}
  );
  await linkDataset(whenId, 'link-b.arrow', 'when');
  let when = await waitFilter(
    whenId,
    filter =>
      filter.dataId.length === 2 &&
      filter.dataId[1] === 'resident-published-2' &&
      filter.lumaComputePlot?.status === 'ready' &&
      filter.plotType?.aggregation &&
      filter.timeBins?.['resident-published-2'],
    'time linked to B'
  );
  assert.equal(
    when.plotType.interval,
    nativeDefault,
    'native default interval for the merged domain'
  );
  assert.equal(
    when.plotType.aggregation,
    'average',
    'native default aggregation is kept, not replaced by count'
  );
  const defaultInterval = when.plotType.interval;
  timeValue = when.value;
  await linkDataset(whenId, 'cpu-observations', 'when');
  when = await waitFilter(
    whenId,
    filter =>
      filter.dataId.length === 3 &&
      filter.lumaComputePlot?.status === 'ready' &&
      filter.timeBins?.['cpu-observations']?.[argument],
    'linked time',
    defaultInterval
  );
  assert.equal(
    when.plotType.interval,
    defaultInterval,
    'the linked interval persists when another dataset joins, as in native'
  );
  check(
    'linked time filter resolves the native default plot type once and every dataset shares it'
  );
  const otherRows = {
    a: rankIds.filter(
      row => inRange(SOURCE.a.amount[row], amount.value) && inRange(SOURCE.a.rank[row], [1, 9])
    ),
    b: SOURCE.b.amount
      .map((value, row) => (inRange(value, amount.value) ? row : -1))
      .filter(row => row >= 0),
    c: SOURCE.c.amount
      .map((value, row) => (inRange(value, amount.value) ? row : -1))
      .filter(row => row >= 0)
  };
  function assertTimeBins(filter, interval = filter.plotType.interval) {
    const thresholds = nativePlot.getBinThresholds(interval, filter.domain);
    for (const key of ['a', 'b', 'c'].filter(key => filter.dataId.includes(ids[key]))) {
      const oracle = nativePlot
        .histogramFromThreshold(thresholds, otherRows[key], row => times[key][row] ?? 0)
        .map(bin => [bin.x0, bin.x1, bin.count]);
      assert.deepEqual(
        filter.timeBins[ids[key]][interval].map(bin => [bin.x0, bin.x1, bin.count]),
        oracle,
        `time bins for ${key} at ${interval}`
      );
    }
    for (const id of [ids.a, ids.b])
      for (const bin of filter.timeBins[id][interval]) assert.deepEqual(bin.indexes, []);
  }
  assertTimeBins(when);
  check(
    'per-dataset time bins match native edges and counts with other filters applied and own filter excluded'
  );
  const widgetStart = async () => {
    if ((await filterById(whenId)).view === 'enlarged') return;
    await page.locator(`[data-for=${JSON.stringify(`tooltip.timePlayback_${whenId}`)}]`).click();
    await page.waitForFunction(
      id =>
        window.__keplerComputeTest.getState().nativeFilters.find(filter => filter.id === id)
          ?.view === 'enlarged',
      whenId
    );
  };
  await widgetStart();
  await page.locator('.time-widget--top .data-ex-icons-gear').last().click();
  const settings = page.locator('.time-widget--settings');
  await settings.waitFor();
  await page.screenshot({path: '/tmp/linked-time.png'});
  // Choose a different interval with the native unit selector: all datasets follow it.
  await settings.locator('.item-selector__dropdown').first().click();
  await page.getByText('Week', {exact: true}).last().click();
  when = await waitFilter(
    whenId,
    filter =>
      filter.plotType?.interval === '1-week' &&
      filter.lumaComputePlot?.status === 'ready' &&
      filter.timeBins?.['cpu-observations']?.['1-week'] &&
      filter.timeBins[filter.dataId[0]]?.['1-week'],
    'week interval'
  );
  assertTimeBins(when, '1-week');
  check('native interval selector changes the interval of resident and CPU datasets together');
  // Native y axis switches the plot to a line chart of the first dataset's reduced values.
  const columnsA = {rank: SOURCE.a.rank, amount: SOURCE.a.amount, kind: SOURCE.a.kind};
  const datasetA = () => ({
    id: ids.a,
    fields: Object.entries(columnsA).map(([name, column]) => ({
      name,
      type: name === 'kind' ? 'string' : 'real',
      valueAccessor: ({index}) => column[index]
    }))
  });
  // The native series builder is the oracle for resident series, grouped or not, on either bin kind.
  function expectedChart(filter, {interval, aggregation, yColumn, groupBy, bins}) {
    const plotType = {
      type: 'lineChart',
      interval,
      aggregation,
      ...(groupBy ? {groupBy: nativeMergeGroupBy(groupBy)} : {})
    };
    return nativePlot.getLineChart(
      {[ids.a]: datasetA()},
      {
        ...filter,
        yAxis: yColumn ? {name: yColumn, type: 'real'} : null,
        plotType,
        lineChart: undefined,
        ...(filter.type === 'range'
          ? {bins: {[ids.a]: bins}}
          : {timeBins: {[ids.a]: {[interval]: bins}}})
      }
    );
  }
  const nativeMergeGroupBy = groupBy => nativePlot.mergePlotGroupBy(undefined, groupBy);
  const expectedLineChart = (interval, aggregation, yColumn = 'rank', groupBy) =>
    expectedChart(when, {
      interval,
      aggregation,
      yColumn,
      groupBy,
      bins: nativePlot.histogramFromThreshold(
        nativePlot.getBinThresholds(interval, when.domain),
        otherRows.a,
        row => times.a[row] ?? 0
      )
    });
  const flatten = chart => ({
    lines: chart.series.lines.map(line => line.map(point => [point.x, point.y, point.pct])),
    markers: chart.series.markers.map(point => [point.x, point.y]),
    names: chart.series.names,
    colors: chart.series.colors,
    yDomain: chart.yDomain,
    xDomain: chart.xDomain,
    title: chart.title,
    aggregation: chart.aggregation,
    yAxis: chart.yAxis
  });
  function assertChart(actual, expected, label, yAxis = 'rank') {
    const left = flatten(actual);
    const right = flatten(expected);
    assert.equal(left.title, right.title, `${label} title`);
    assert.equal(left.yAxis, yAxis);
    assert.equal(left.aggregation, right.aggregation, `${label} aggregation`);
    assert.deepEqual(left.xDomain, right.xDomain, `${label} x domain`);
    assert.deepEqual(left.names, right.names, `${label} series names`);
    assert.deepEqual(left.colors, right.colors, `${label} series colors`);
    assert.equal(left.lines.length, right.lines.length, `${label} line segments`);
    for (const [index, line] of left.lines.entries()) {
      assert.equal(line.length, right.lines[index].length, `${label} points`);
      for (const [pointIndex, point] of line.entries()) {
        assert.equal(point[0], right.lines[index][pointIndex][0], `${label} x`);
        closeTo(point[1], right.lines[index][pointIndex][1], `${label} y`);
      }
    }
    assert.equal(left.markers.length, right.markers.length, `${label} markers`);
    for (const [index, domain] of left.yDomain.entries())
      closeTo(domain, right.yDomain[index], `${label} y domain`);
  }
  await settings.getByText('Y Axis', {exact: true}).last().click();
  await page
    .locator('.list__item__anchor', {hasText: /^rank$/})
    .last()
    .click();
  const chooseAggregation = async label => {
    await settings
      .locator('[data-testid=time-widget-aggregation] .item-selector__dropdown')
      .click();
    await page.getByText(label, {exact: true}).last().click();
  };
  const lineAggregations = {
    Average: 'average',
    Sum: 'sum',
    Maximum: 'maximum',
    Minimum: 'minimum',
    Median: 'median',
    'Std Deviation': 'stdev'
  };
  for (const [label, aggregation] of Object.entries(lineAggregations)) {
    if (aggregation !== 'average') await chooseAggregation(label);
    when = await waitFilter(
      whenId,
      filter =>
        filter.plotType?.type === 'lineChart' &&
        filter.plotType.aggregation === argument.aggregation &&
        filter.lineChart?.aggregation === argument.aggregation &&
        filter.lumaComputePlot?.status === 'ready',
      `line chart ${aggregation}`,
      {aggregation}
    );
    assert.equal(when.yAxis.name, 'rank');
    assertChart(
      when.lineChart,
      expectedLineChart('1-week', aggregation),
      `${aggregation} line chart`
    );
    assertTimeBins(when, '1-week');
  }
  await page.waitForTimeout(1000);
  await page.screenshot({path: '/tmp/linked-line-chart.png'});
  // A y axis with null values: native skips nulls per bin (median/deviation included).
  const setTimePlot = newProp =>
    page.evaluate(({id, newProp}) => window.__keplerComputeTest.setFilterPlot(id, newProp), {
      id: whenId,
      newProp
    });
  await setTimePlot({yAxis: {name: 'amount', type: 'real'}});
  for (const aggregation of Object.values(lineAggregations)) {
    await setTimePlot({plotType: {aggregation}});
    when = await waitFilter(
      whenId,
      filter =>
        filter.yAxis?.name === 'amount' &&
        filter.plotType?.aggregation === argument.aggregation &&
        filter.lineChart?.aggregation === argument.aggregation &&
        filter.lineChart?.yAxis === 'amount' &&
        filter.lumaComputePlot?.status === 'ready',
      `amount line chart ${aggregation}`,
      {aggregation}
    );
    assertChart(
      when.lineChart,
      expectedLineChart('1-week', aggregation, 'amount'),
      `${aggregation} amount line chart`,
      'amount'
    );
  }
  await setTimePlot({yAxis: {name: 'rank', type: 'real'}});
  await setTimePlot({plotType: {aggregation: 'average'}});
  when = await waitFilter(
    whenId,
    filter =>
      filter.plotType?.aggregation === 'average' &&
      filter.lineChart?.aggregation === 'average' &&
      filter.lineChart?.yAxis === 'rank' &&
      filter.lumaComputePlot?.status === 'ready',
    'average restored'
  );
  assertChart(when.lineChart, expectedLineChart('1-week', 'average'), 'restored line chart');
  check(
    'native y axis draws the first dataset line chart from GPU-reduced average, sum, maximum, minimum, median and sample standard deviation per bin, matching native series (null y values skipped)'
  );

  // Grouped series: native discovers labels bin by bin, keeps numGroups and optionally merges Others.
  for (const groupBy of [
    {fieldName: 'kind', numGroups: 3, groupOthers: true},
    {fieldName: 'kind', numGroups: 2, groupOthers: false},
    {fieldName: 'kind', numGroups: 'ALL', groupOthers: false}
  ]) {
    await page.evaluate(
      ({id, groupBy}) => window.__keplerComputeTest.setFilterPlot(id, {plotType: {groupBy}}),
      {id: whenId, groupBy}
    );
    when = await waitFilter(
      whenId,
      filter =>
        JSON.stringify([
          filter.plotType?.groupBy?.numGroups,
          filter.plotType?.groupBy?.groupOthers
        ]) === JSON.stringify(argument) &&
        filter.lineChart?.groupBy &&
        filter.lumaComputePlot?.status === 'ready',
      `grouped ${JSON.stringify(groupBy)}`,
      [groupBy.numGroups, groupBy.groupOthers]
    );
    for (const aggregation of ['average', 'median', 'stdev'])
      assertChart(
        await groupedTimeChart(groupBy, aggregation),
        expectedLineChart('1-week', aggregation, 'rank', when.plotType.groupBy),
        `grouped ${groupBy.numGroups}/${groupBy.groupOthers} ${aggregation}`
      );
  }
  async function groupedTimeChart(groupBy, aggregation) {
    await page.evaluate(
      ({id, aggregation}) =>
        window.__keplerComputeTest.setFilterPlot(id, {plotType: {aggregation}}),
      {id: whenId, aggregation}
    );
    const filter = await waitFilter(
      whenId,
      candidate =>
        candidate.plotType?.aggregation === argument &&
        candidate.lineChart?.aggregation === argument &&
        candidate.lineChart?.groupBy &&
        candidate.lumaComputePlot?.status === 'ready',
      'grouped aggregation',
      aggregation
    );
    return filter.lineChart;
  }
  await page.evaluate(
    id =>
      window.__keplerComputeTest.setFilterPlot(id, {
        plotType: {groupBy: null, aggregation: 'average'}
      }),
    whenId
  );
  when = await waitFilter(
    whenId,
    filter =>
      !filter.plotType?.groupBy &&
      filter.plotType?.aggregation === 'average' &&
      filter.lineChart &&
      !filter.lineChart.groupBy &&
      filter.lumaComputePlot?.status === 'ready',
    'grouping cleared'
  );
  assertChart(when.lineChart, expectedLineChart('1-week', 'average'), 'ungrouped again');
  check(
    'grouped line series match native: first-seen label order, numGroups, Others, per-series counts and reduced y'
  );

  // 3b. Numeric range plots: native numeric interval edges, then line charts of the first dataset.
  const setPlot = (id, newProp) =>
    page.evaluate(({id, newProp}) => window.__keplerComputeTest.setFilterPlot(id, newProp), {
      id,
      newProp
    });
  numericInterval = 25;
  await setPlot(amountId, {plotType: {numericInterval}});
  amount = await waitFilter(
    amountId,
    filter =>
      filter.plotType?.numericInterval === 25 &&
      filter.lumaComputePlot?.status === 'ready' &&
      filter.bins?.['cpu-observations'] &&
      filter.bins[filter.dataId[0]]?.length &&
      filter.bins[filter.dataId[1]]?.length,
    'numeric interval 25'
  );
  await assertAmountBins(amount);
  check(
    'native numeric interval gives every dataset the native threshold edges, resident and CPU alike'
  );
  const numericChart = (aggregation, yColumn, groupBy) =>
    expectedChart(amount, {
      aggregation,
      yColumn,
      groupBy,
      bins: nativeHistogram(SOURCE.a.amount, ownExcluded().a, amount.domain, numericInterval)
    });
  // Native applies one property per action; a combined object would replace plotType wholesale.
  await setPlot(amountId, {yAxis: {name: 'rank', type: 'real'}});
  for (const aggregation of ['average', 'sum', 'maximum', 'minimum', 'median', 'stdev']) {
    await setPlot(amountId, {plotType: {aggregation}});
    amount = await waitFilter(
      amountId,
      filter =>
        filter.plotType?.type === 'lineChart' &&
        filter.plotType.aggregation === argument &&
        filter.lineChart?.aggregation === argument &&
        filter.lumaComputePlot?.status === 'ready',
      `numeric ${aggregation}`,
      aggregation
    );
    assertChart(
      amount.lineChart,
      numericChart(aggregation, 'rank'),
      `numeric ${aggregation} line chart`
    );
    await assertAmountBins(amount);
  }
  await setPlot(amountId, {yAxis: {name: 'amount', type: 'real'}});
  await setPlot(amountId, {plotType: {aggregation: 'median'}});
  amount = await waitFilter(
    amountId,
    filter =>
      filter.yAxis?.name === 'amount' &&
      filter.lineChart?.yAxis === 'amount' &&
      filter.lineChart?.aggregation === 'median' &&
      filter.lumaComputePlot?.status === 'ready',
    'numeric amount median'
  );
  assertChart(
    amount.lineChart,
    numericChart('median', 'amount'),
    'numeric amount median',
    'amount'
  );
  await setPlot(amountId, {yAxis: {name: 'rank', type: 'real'}});
  await setPlot(amountId, {
    plotType: {
      aggregation: 'average',
      groupBy: {fieldName: 'kind', numGroups: 3, groupOthers: true}
    }
  });
  amount = await waitFilter(
    amountId,
    filter =>
      filter.plotType?.groupBy?.fieldName === 'kind' &&
      filter.lineChart?.groupBy &&
      filter.lumaComputePlot?.status === 'ready',
    'numeric grouped'
  );
  assertChart(
    amount.lineChart,
    numericChart('average', 'rank', amount.plotType.groupBy),
    'numeric grouped line chart'
  );
  check(
    'numeric range line charts: GPU-reduced y measures (including median and standard deviation) and grouped series match native'
  );
  numericInterval = undefined;
  await setPlot(amountId, {plotType: {groupBy: null, numericInterval: null}});
  await setPlot(amountId, {yAxis: null});
  amount = await waitFilter(
    amountId,
    filter =>
      filter.plotType?.type === 'histogram' &&
      !filter.plotType?.groupBy &&
      filter.lumaComputePlot?.status === 'ready' &&
      filter.bins?.['cpu-observations'] &&
      filter.bins[filter.dataId[0]]?.length &&
      !filter.lineChart,
    'numeric histogram restored'
  );
  await assertAmountBins(amount);

  // 4. Field mapping per dataset through the native field selectors (setFilter name/valueIndex).
  const amountPanel = await panelFor(amountId);
  const anchors = amountPanel.locator('.list__item__anchor');
  const chooseDatasetField = async (position, fieldName) => {
    const panel = await panelFor(amountId);
    await panel.locator('.list__item__anchor').nth(position).click();
    await page
      .locator('.list__item__anchor', {hasText: new RegExp(`^${fieldName}$`)})
      .last()
      .click();
  };
  amount = await filterById(amountId);
  const cpuBinsBeforeMapping = amount.bins[ids.c].map(bin => [bin.x0, bin.x1, bin.count]);
  await chooseDatasetField(1, 'score');
  amount = await waitFilter(
    amountId,
    filter =>
      filter.name?.[1] === 'score' &&
      filter.lumaComputePlot?.status === 'ready' &&
      filter.bins?.['cpu-observations'] &&
      filter.dataId.length === 3,
    'B mapped to score'
  );
  assert.deepEqual(amount.name, ['amount', 'score', 'amount']);
  assert.deepEqual(
    amount.domain,
    [-10, 300],
    'native domain merges the mapped fields of every dataset'
  );
  console.log(
    'mapped value',
    JSON.stringify(amount.value),
    JSON.stringify(await assertMasks(amount, {a: 'amount', b: 'score'}))
  );
  await assertAmountBins(amount, {a: 'amount', b: 'score'}, cpuBinsBeforeMapping);
  check(
    'native field selector maps a dataset to another field: merged domain, bins and mask follow the mapped column'
  );
  await chooseDatasetField(1, 'amount');
  amount = await waitFilter(
    amountId,
    filter =>
      filter.name?.[1] === 'amount' &&
      filter.lumaComputePlot?.status === 'ready' &&
      filter.bins?.['cpu-observations'] &&
      filter.domain[1] === 100,
    'B mapped back'
  );
  await assertAmountBins(amount, undefined, cpuBinsBeforeMapping);
  await assertMasks(amount);

  // 5. Removing and re-adding a dataset through the native row actions.
  await page.locator(`[data-for=${JSON.stringify(`tooltip.delete_${amountId}-${ids.c}`)}]`).click();
  amount = await waitFilter(
    amountId,
    filter =>
      filter.dataId.length === 2 &&
      filter.lumaComputePlot?.status === 'ready' &&
      !filter.bins?.['cpu-observations'],
    'CPU dataset unlinked'
  );
  assert.deepEqual(Object.keys(amount.bins).sort(), [ids.a, ids.b].sort());
  assert.deepEqual(amount.domain, [5, 100]);
  assert.deepEqual(
    edgesOf(amount.bins[ids.a]),
    nativeBins(SOURCE.a.amount, ownExcluded().a, amount.domain),
    'all-resident plots need no CPU pass'
  );
  assert.deepEqual(
    edgesOf(amount.bins[ids.b]),
    nativeBins(SOURCE.b.amount, ownExcluded().b, amount.domain)
  );
  await linkDataset(amountId, 'cpu-observations', 'amount');
  amount = await waitFilter(
    amountId,
    filter =>
      filter.dataId.length === 3 &&
      filter.lumaComputePlot?.status === 'ready' &&
      filter.bins?.['cpu-observations'],
    'CPU dataset relinked'
  );
  await assertAmountBins(amount);
  await assertMasks(amount);
  check('removing and re-adding a dataset recomputes domain, bins and masks');

  // 6. Save and restore. Restoring rebuilds resident datasets only (a native CPU dataset is not part
  // of the resident source manifest), so unlink the CPU dataset from both filters first.
  for (const id of [amountId, whenId])
    await page.locator(`[data-for=${JSON.stringify(`tooltip.delete_${id}-${ids.c}`)}]`).click();
  amount = await waitFilter(
    amountId,
    filter =>
      filter.dataId.length === 2 &&
      filter.lumaComputePlot?.status === 'ready' &&
      !filter.bins?.['cpu-observations'],
    'amount without CPU'
  );
  when = await waitFilter(
    whenId,
    filter =>
      filter.dataId.length === 2 &&
      filter.lumaComputePlot?.status === 'ready' &&
      filter.lineChart &&
      !filter.timeBins?.['cpu-observations'],
    'time without CPU'
  );
  await assertAmountBins(amount);
  assertTimeBins(when, '1-week');
  assertChart(
    when.lineChart,
    expectedLineChart('1-week', 'average'),
    'line chart after unlinking the CPU dataset'
  );
  const saved = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const savedFilters = saved.config.visState.filters;
  const savedAmount = savedFilters.find(filter => filter.id === amountId);
  assert.deepEqual(savedAmount.dataId, [ids.a, ids.b]);
  assert.deepEqual(savedAmount.name, ['amount', 'amount']);
  const savedWhen = savedFilters.find(filter => filter.id === whenId);
  assert.equal(savedWhen.plotType.interval, '1-week');
  assert.equal(savedWhen.plotType.type, 'lineChart');
  assert.equal(savedWhen.yAxis.name, 'rank');
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), saved);
  amount = await waitFilter(
    amountId,
    filter =>
      filter.dataId.length === 2 &&
      filter.lumaComputePlot?.status === 'ready' &&
      filter.bins?.[filter.dataId[1]]?.length &&
      filter.bins[filter.dataId[0]]?.length,
    'restored amount'
  );
  assert.deepEqual(amount.name, ['amount', 'amount']);
  await assertAmountBins(amount);
  await assertMasks(amount);
  when = await waitFilter(
    whenId,
    filter =>
      filter.plotType?.interval === '1-week' &&
      filter.plotType.type === 'lineChart' &&
      filter.lineChart?.aggregation === 'average' &&
      filter.lumaComputePlot?.status === 'ready' &&
      filter.timeBins?.[filter.dataId[1]]?.['1-week'] &&
      filter.timeBins[filter.dataId[0]]?.['1-week'],
    'restored time'
  );
  assertTimeBins(when, '1-week');
  assertChart(when.lineChart, expectedLineChart('1-week', 'average'), 'restored line chart');
  const resaved = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  for (const id of [amountId, whenId, rankId]) {
    const left = saved.config.visState.filters.find(filter => filter.id === id);
    const right = resaved.config.visState.filters.find(filter => filter.id === id);
    assert.deepEqual(
      {
        dataId: right.dataId,
        name: right.name,
        value: right.value,
        plotType: right.plotType,
        yAxis: right.yAxis
      },
      {
        dataId: left.dataId,
        name: left.name,
        value: left.value,
        plotType: left.plotType,
        yAxis: left.yAxis
      },
      `saved ${id} round trip`
    );
  }
  check(
    'save and restore preserve dataset links, per-dataset fields, interval, line chart y axis and every plot'
  );
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, checks}));
} catch (error) {
  console.error(error.message);
  console.error(
    await page
      .evaluate(() => ({text: document.body.innerText.slice(0, 1500)}))
      .catch(() => 'Page unavailable')
  );
  process.exitCode = 1;
} finally {
  await browser.close();
}
