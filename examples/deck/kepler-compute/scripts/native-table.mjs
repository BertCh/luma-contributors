// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const errors = [];
page.on('pageerror', error => {errors.push(error.message); console.error(error.message);});
page.on('response', response => {if (response.status() >= 400) console.error(response.status(), response.url());});
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=10000', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => {const state = window.__keplerComputeTest?.getState(); if (state?.error) throw new Error(state.error); return state?.ready;}, undefined, {timeout: 30000});
  const rows = await page.evaluate(() => window.__keplerComputeTest.source());
  const rowIds = () => page.locator('.resident-table-scroll tbody tr th').allTextContents().then(values => values.map(Number));
  const waitRows = async expected => {
    await page.waitForFunction(expected => {
      const actual = Array.from(document.querySelectorAll('.resident-table-scroll tbody tr th'), cell => Number(cell.textContent));
      return JSON.stringify(actual) === JSON.stringify(expected);
    }, expected);
    assert.deepEqual(await rowIds(), expected);
  };
  const sorted = (selected, direction = 1) => rows.map((row, index) => ({row, index}))
    .filter(({row}) => !selected || row[2] >= 40 && row[2] <= 75)
    .sort((first, second) => direction * (first.row[2] - second.row[2]) || first.index - second.index)
    .map(({index}) => index);
  await page.locator('.show-data-table svg').first().click();
  await waitRows(Array.from({length: 50}, (_, index) => index));
  await page.getByRole('button', {name: 'Next page', exact: true}).click();
  await waitRows(Array.from({length: 50}, (_, index) => index + 50));
  await page.locator('.resident-table-scroll thead').getByRole('button', {name: 'value', exact: true}).click();
  await waitRows(sorted(false).slice(0, 50));
  await page.locator('.resident-table-scroll thead').getByRole('button', {name: 'value ↑', exact: true}).click();
  await waitRows(sorted(false, -1).slice(0, 50));
  await page.evaluate(() => window.__keplerComputeTest.setRange(40, 75));
  await waitRows(sorted(true, -1).slice(0, 50));
  await page.getByLabel('Rows', {exact: true}).selectOption('all');
  await waitRows(sorted(false, -1).slice(0, 50));
  await page.keyboard.press('Escape');
  await page.locator('.resident-table-modal').waitFor({state: 'hidden'});
  await page.locator('#analysis-operation').selectOption('z-score');
  await page.locator('#analysis-population').selectOption('selected');
  await page.waitForFunction(() => window.__keplerComputeTest.analysis()?.summary?.operation === 'z-score');
  await page.locator('.show-data-table svg').first().click();
  await page.getByRole('button', {name: 'Analysis: z-score', exact: true}).click();
  const selectedRows = sorted(true);
  await waitRows(selectedRows.slice(0, 50));
  const values = selectedRows.map(index => rows[index][2]);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const standardDeviation = Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1));
  const displayed = await page.locator('.resident-table-scroll tbody tr td:last-child').allTextContents();
  displayed.forEach((value, index) => assert(Math.abs(Number(value) - (values[index] - mean) / standardDeviation) < 0.0001));
  await page.getByLabel('Rows', {exact: true}).selectOption('all');
  await page.getByRole('button', {name: 'Analysis: z-score ↑', exact: true}).click();
  await waitRows(sorted(true, -1).slice(0, 50));
  await page.screenshot({path: '/tmp/kepler-native-table.png'});
  await page.evaluate(() => window.__keplerComputeTest.analysis().setMode('off'));
  await waitRows(Array.from({length: 50}, (_, index) => index));
  assert.equal(await page.getByRole('button', {name: /Analysis: z-score/}).count(), 0);
  await page.keyboard.press('Escape');
  await page.locator('#save-action').click();
  await page.getByText('Export Data', {exact: true}).click();
  await page.getByRole('heading', {name: 'Export data'}).waitFor();
  assert.equal(await page.getByRole('button', {name: 'Export', exact: true}).count(), 0);
  await page.getByLabel('Export dataset', {exact: true}).waitFor();
  await page.keyboard.press('Escape');
  await page.locator('#save-action').click();
  await page.getByText('Export Map', {exact: true}).click();
  await page.getByRole('heading', {name: 'Save and restore map'}).waitFor();
  await page.getByRole('button', {name: 'Save state', exact: true}).waitFor();
  console.log('Native table paging, GPU sorting, selection updates, derived values and export routes passed.');
  assert.deepEqual(errors, []);
} catch (error) { console.error(await page.locator('body').innerText().catch(() => 'No page text')); throw error; } finally { await browser.close(); }
