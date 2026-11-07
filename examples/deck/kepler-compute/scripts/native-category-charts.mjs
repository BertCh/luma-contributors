// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1600, height: 1100}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => {errors.push(error.message); console.error(error.stack);});
const card = page.locator('.resident-chart-card').first();
const bars = card.getByRole('status', {name: 'Chart category summary', exact: true});
async function waitBars(expected) {
  await page.waitForFunction(expected => {
    const buttons = [...document.querySelectorAll('[aria-label="Chart category summary"] [role=button]')];
    return JSON.stringify(buttons.map(button => button.getAttribute('aria-label'))) === JSON.stringify(expected);
  }, expected);
}
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  const source = new arrow.Table({category: arrow.vectorFromArray(['Beta', 'Alpha', 'Gamma', 'Beta', 'Alpha', 'Gamma', '', null, 'Delta']), amount: arrow.vectorFromArray([4, null, -2, 2, 3, 3, 100, 100, null], new arrow.Float32()), rank: arrow.makeVector(new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]))});
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: 'chart-categories.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(source.slice(0, 3).concat(source.slice(3)), 'stream'))});
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 9);
  const initial = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0]);
  await closeNativeDialog(page);
  await page.getByRole('button', {name: 'Toggle charts', exact: true}).click();
  await page.locator('.chart-config__type').click();
  await page.getByText('Bar', {exact: true}).click();
  await card.getByLabel('Chart dataset', {exact: true}).selectOption(initial.id);
  await card.getByLabel('Chart categories', {exact: true}).selectOption('category');
  await waitBars(['Beta: 2', 'Alpha: 2', 'Gamma: 2', 'Delta: 1']);
  await card.getByLabel('Chart field', {exact: true}).selectOption('amount');
  await card.getByLabel('Chart measure', {exact: true}).selectOption('average');
  await waitBars(['Beta: 3', 'Alpha: 3', 'Gamma: 0.5', 'Delta: 0']);
  await bars.getByRole('button', {name: 'Beta: 3', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 2);
  await waitBars(['Beta: 3', 'Alpha: 3', 'Gamma: 0.5', 'Delta: 0']);
  assert.equal(await bars.getByRole('button', {name: 'Beta: 3', exact: true}).getAttribute('aria-pressed'), 'true');
  await page.evaluate(id => window.__keplerComputeTest.setPublishedRange(id, 'rank', 1, 8), initial.id);
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 1);
  await waitBars(['Alpha: 3', 'Beta: 2', 'Gamma: 0.5', 'Delta: 0']);
  await bars.getByRole('button', {name: 'Beta: 2', exact: true}).press('Enter');
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 8);
  await card.getByLabel('Maximum displayed groups', {exact: true}).fill('2');
  await card.getByLabel('Combine remaining groups as Other', {exact: true}).check();
  await waitBars(['Alpha: 3', 'Beta: 2', 'Other: 0.5']);
  await card.getByLabel('Apply map filters', {exact: true}).uncheck();
  await waitBars(['Beta: 3', 'Alpha: 3', 'Other: 0.5']);
  const saved = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const chart = saved.config.visState.charts.find(chart => chart.dataId === initial.id);
  assert.equal(chart.xAxis.field.name, 'category'); assert.equal(chart.yAxis.aggregation, 'average');
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), saved);
  await waitBars(['Beta: 3', 'Alpha: 3', 'Other: 0.5']);
  const final = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0]);
  for (const name of ['sourceRevision', 'sourceUploadBytes', 'sourceBufferIds', 'sourceBatches']) assert.deepEqual(final[name], initial[name]);
  assert.deepEqual(errors, []);
  await page.screenshot({path: '/tmp/kepler-native-category-charts.png'});
  await card.getByRole('button', {name: 'Remove chart', exact: true}).click();
  await card.waitFor({state: 'hidden'});
  console.log(JSON.stringify({passed: true, checks: ['native categorical bars', 'nullable multi-batch measures', 'native ordering and ties', 'own-filter exclusion', 'other range retained', 'keyboard toggle', 'top-N and Other', 'all-source population', 'native replay', 'stable source storage', 'remove']}));
} catch (error) {
  console.error(await page.evaluate(() => ({state: window.__keplerComputeTest?.getState(), text: document.body.innerText})).catch(() => 'Page unavailable'));
  throw error;
} finally {await browser.close();}
