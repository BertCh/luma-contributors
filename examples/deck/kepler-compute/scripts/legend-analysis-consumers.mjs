// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser check: GPU color/size legends, active analysis tooltip/legend and shared notices.
// The pre-port probe never enabled a legend or analysis and asserted nothing. Drive those
// states explicitly; use the repository browser flags and module pinning for the ported host.
// Usage: node scripts/legend-analysis-consumers.mjs [url]
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const url = new URL(process.argv[2] ?? 'http://127.0.0.1:5318/');
// Keep this consumer check bounded independently of the host's 100,000-row default.
if (!url.searchParams.has('rows')) url.searchParams.set('rows', '256');
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(url.href);
  await page.waitForFunction(
    () => {
      const api = window.__keplerComputeTest;
      if (api?.getState().error) throw new Error(api.getState().error);
      return api && api.getState().selectedCount === api.source().length;
    },
    undefined,
    {timeout: 60000}
  );
  const rows = await page.evaluate(() => window.__keplerComputeTest.source());
  assert(rows.length > 0, 'fixture must contain observations');
  const configuration = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  configuration.config.uiState = {
    ...configuration.config.uiState,
    mapControls: {...configuration.config.uiState?.mapControls, mapLegend: {active: true}}
  };
  const point = configuration.config.visState.layers.find(layer => layer.id === 'resident-points');
  assert(point, 'fixture point layer exists');
  point.visualChannels = {
    ...point.visualChannels,
    colorField: {name: 'value', type: 'real'},
    colorScale: 'quantile',
    sizeField: {name: 'value', type: 'real'},
    sizeScale: 'linear'
  };
  point.config.visConfig = {
    ...point.config.visConfig,
    radiusRange: [2, 8],
    fixedRadius: false
  };
  // Dynamic domains must follow the value filter for the empty-selection notice below.
  for (const filter of configuration.config.visState.filters)
    if (filter.name?.[0] === 'value') filter.fixedDomain = false;
  configuration.config.visState.interactionConfig.tooltip.fieldsToShow['resident-observations'] = [
    {name: 'value', format: null}
  ];
  await page.evaluate(
    configuration => window.__keplerComputeTest.applyNativeConfig(configuration),
    configuration
  );
  await page.waitForFunction(
    () => {
      const snapshot = window.__keplerComputeTest.getStyleLegend('resident-points', 'fillColor');
      return (
        snapshot?.legend &&
        snapshot.legend.revision === snapshot.expected.revision &&
        snapshot.legend.selectionRevision === snapshot.expected.selectionRevision
      );
    },
    undefined,
    {timeout: 45000}
  );
  const styleLegend = page.locator('[data-resident-style-legend="fillColor"]').first();
  await styleLegend.waitFor();
  await page.waitForFunction(() => {
    const legend = document.querySelector('[data-resident-style-legend="fillColor"]');
    return legend && !legend.querySelector('[data-resident-consumer-state]');
  });
  // Kepler names the point-radius visual channel `size`, which is the legend's channel key.
  const sizeLegend = page.locator('[data-resident-channel-legend="size"]').first();
  await sizeLegend.waitFor({timeout: 45000});
  const ticks = await sizeLegend.locator('div > span:last-child').allTextContents();
  const values = rows.map(row => row[2]).filter(Number.isFinite);
  const formatTick = value => Number(value.toPrecision(5)).toLocaleString('en-US');
  assert.equal(ticks.length, 5, 'size legend renders five GPU domain ticks');
  assert.equal(ticks[0], formatTick(Math.min(...values)));
  assert.equal(ticks.at(-1), formatTick(Math.max(...values)));
  assert.equal(await styleLegend.locator('strong').textContent(), 'value');

  const analysisPanel = page
    .getByTestId('native-analysis')
    .getByRole('region', {name: 'GPU analysis', exact: true});
  await analysisPanel.getByLabel('Operation', {exact: true}).selectOption('z-score');
  await page.waitForFunction(
    () => window.__keplerComputeTest.analysis()?.summary?.operation === 'z-score',
    undefined,
    {timeout: 45000}
  );
  const summary = await page.evaluate(() => window.__keplerComputeTest.analysis().summary);
  const analysisLegend = page.locator('[data-resident-analysis-legend]').first();
  await analysisLegend.getByText('Analysis (z-score)', {exact: true}).waitFor({timeout: 20000});
  const breaks = summary.breaks?.length ? summary.breaks : [summary.minimum, summary.maximum];
  const formatBreak = value =>
    Number(value.toPrecision(5)).toLocaleString('en-US', {maximumSignificantDigits: 5});
  const expectedClasses = breaks
    .slice(0, -1)
    .slice(0, 8)
    .map((value, index) => `${formatBreak(value)} to ${formatBreak(breaks[index + 1])}`);
  assert(expectedClasses.length > 0, 'analysis produces a nonempty ramp');
  assert.deepEqual(
    await analysisLegend.locator('div > span:last-child').allTextContents(),
    expectedClasses
  );

  // A projected point can be occluded by another point in both old and ported hosts. Find
  // an independently pickable row before asserting its tooltip, rather than assuming draw order.
  let sourceRowIndex = -1;
  for (let index = 0; index < Math.min(rows.length, 64); index++) {
    const picked = await page.evaluate(
      index => window.__keplerComputeTest.pickSourceRow(index),
      index
    );
    if (picked?.index === index) {
      sourceRowIndex = index;
      break;
    }
  }
  assert(sourceRowIndex >= 0, 'at least one observation can be picked');
  const position = await page.evaluate(
    index => window.__keplerComputeTest.projectSourceRow(index),
    sourceRowIndex
  );
  await page.mouse.move(position[0], position[1]);
  const tooltip = page.locator(`[data-resident-tooltip-row="${sourceRowIndex}"]`).first();
  await tooltip.getByText('Analysis (z-score) of value', {exact: true}).waitFor({timeout: 20000});
  const computedRow = await page.evaluate(
    index => window.__keplerComputeTest.analysis().readRow(index),
    sourceRowIndex
  );
  assert(computedRow && Number.isFinite(computedRow.value), 'picked analysis result is finite');
  const computedValue = tooltip
    .locator('dl > div')
    .filter({has: page.getByText('Analysis (z-score) of value', {exact: true})})
    .locator('dd');
  assert.equal(await computedValue.textContent(), String(computedRow.value));
  assert.equal(await tooltip.locator('[data-resident-consumer-state]').count(), 0);
  await page.mouse.move(1, 1);

  // Empty selection is a stable way to exercise a real shared notice. No pending-state
  // timing assumptions or synthetic DOM are needed; the GPU domain has no rows to label.
  await analysisPanel.getByLabel('Operation', {exact: true}).selectOption('off');
  await page.evaluate(() => window.__keplerComputeTest.setRange(200, 201));
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().selectedCount === 0,
    undefined,
    {timeout: 45000}
  );
  const notice = page
    .locator('[data-resident-consumer="legend"][data-resident-consumer-state="unsupported"]')
    .first();
  await notice.waitFor({timeout: 45000});
  assert.equal(await notice.getAttribute('role'), 'status');
  assert.match(await notice.textContent(), /No rows pass the active filters/);
  assert.equal(
    await page.locator('[data-resident-channel-legend="size"]').count(),
    0,
    'empty domain removes obsolete size ticks'
  );
  assert.deepEqual(errors, []);
  process.stdout.write(
    `${JSON.stringify(
      {
        sizeTicks: ticks,
        analysisClasses: expectedClasses,
        sourceRowIndex,
        consumerNotice: await notice.textContent(),
        errors
      },
      null,
      2
    )}\n`
  );
} finally {
  await browser.close();
}
