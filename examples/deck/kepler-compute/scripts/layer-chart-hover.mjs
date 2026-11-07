// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// NATIVEUI6-T2 browser check (headless WebGPU): native layer charts (Tooltip > Category) in the
// resident hover popover. A layerChart on the resident point layer renders a ChartRenderer for the
// hovered row; a chart with an id field notes that grouping needs a resident row lookup and shows
// the hovered row only. Usage: node layer-chart-hover.mjs [url]
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())) errors.push(message.text());
});
const watchdog = setTimeout(() => {
  console.error('layer-chart-hover exceeded 180 seconds');
  process.exit(1);
}, 180000);

const makeChart = (id, idField) => ({
  id,
  title: `Layer chart ${id}`,
  type: 'layerChart',
  dataId: 'resident-observations',
  layerId: 'resident-points',
  applyFilters: false,
  display: {},
  layerChartType: 'BREAKDOWN_BY_CATEGORY',
  axis: {field: {name: 'category', type: 'integer'}, aggregation: 'uniqueBin'},
  yAxis: {field: {name: 'value', type: 'real'}, aggregation: 'sum'},
  chartDisplay: {idField, numEntries: 5}
});

try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => {
    const state = window.__keplerComputeTest?.getState();
    if (state?.error) throw new Error(state.error);
    return state?.ready && state.selectedCount === 256;
  }, undefined, {timeout: 60000});
  const hoverRow = async row => {
    const position = await page.evaluate(index => window.__keplerComputeTest.projectSourceRow(index), row);
    await page.mouse.move(position[0] - 3, position[1] - 3);
    await page.mouse.move(position[0], position[1]);
  };

  // 1. A chart without an id field: the hovered-row ChartRenderer, no lookup note.
  await page.evaluate(chart => window.__keplerComputeTest.addNativeChart(chart), makeChart('plain', null));
  await hoverRow(0);
  const plain = page.locator('[data-resident-layer-chart]').filter({hasText: 'Layer chart plain'});
  await plain.waitFor({timeout: 30000});
  await plain.locator('svg, canvas, [role="img"], .bar-chart').first().waitFor({timeout: 30000});
  assert.equal(await plain.getByText(/Reading hovered rows|stale/).count(), 0, 'chart finished reading the hovered row');
  assert.equal(await plain.getByText(/resident row lookup/).count(), 0, 'no id field, no lookup note');

  // 2. A chart with an id field: the resident row lookup is unavailable, so it says so.
  await page.evaluate(chart => window.__keplerComputeTest.addNativeChart(chart), makeChart('grouped', 'category'));
  await page.mouse.move(1300, 900);
  await hoverRow(1);
  const grouped = page.locator('[data-resident-layer-chart]').filter({hasText: 'Layer chart grouped'});
  await grouped.waitFor({timeout: 30000});
  await grouped.getByText(/showing the hovered row only/).waitFor({timeout: 30000});

  // 3. The chart follows the hovered row: a different row re-reads.
  await page.mouse.move(1300, 900);
  await hoverRow(2);
  await page.locator('[data-resident-layer-chart]').filter({hasText: 'Layer chart plain'}).locator('svg, canvas, [role="img"], .bar-chart').first().waitFor({timeout: 30000});
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, charts: await page.locator('[data-resident-layer-chart]').count()}));
} catch (error) {
  console.error(await page.evaluate(() => [...document.querySelectorAll('[data-resident-tooltip], [data-resident-layer-chart]')].map(element => element.outerHTML.slice(0, 400))).catch(() => null));
  console.error(errors);
  throw error;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
