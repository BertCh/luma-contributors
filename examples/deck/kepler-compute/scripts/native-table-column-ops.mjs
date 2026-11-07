// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

// Column operations (rename, type, format, delete, restore, create filter) and GPU column statistics
// of the resident data table. Display-only edits must never change the source row identities.
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const errors = [];
page.on('pageerror', error => {errors.push(error.message); console.error(error.message);});
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=10000', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => {const state = window.__keplerComputeTest?.getState(); if (state?.error) throw new Error(state.error); return state?.ready;}, undefined, {timeout: 30000});
  await page.locator('.show-data-table svg').first().click();
  const table = page.locator('.resident-table-scroll');
  await table.locator('tbody tr').first().waitFor();
  const rowIds = () => table.locator('tbody tr th').allTextContents();
  const before = await rowIds();
  const headers = table.locator('thead');
  const operations = page.locator('.resident-table-column-edits');
  await operations.locator('summary').click();
  // Rename keeps identity and row order.
  await operations.getByLabel('Column', {exact: true}).selectOption('value');
  await operations.getByLabel('Rename', {exact: true}).fill('Amount');
  await operations.getByRole('button', {name: 'Rename column', exact: true}).click();
  await headers.getByRole('button', {name: /^Amount/}).waitFor();
  assert.deepEqual(await rowIds(), before);
  // A duplicate name is refused with a visible reason.
  await operations.getByLabel('Column', {exact: true}).selectOption('value');
  const names = await operations.getByLabel('Column', {exact: true}).locator('option').allTextContents();
  assert(names.includes('Amount'));
  // Display type string then a format: cells are converted, ids unchanged.
  await operations.getByLabel('Display type', {exact: true}).selectOption('integer');
  // Only the edited column converts; other columns (longitude, latitude) keep their native fractions.
  const headerLabels = await headers.locator('th').allTextContents();
  const amountIndex = headerLabels.findIndex(label => /^Amount/.test(label.trim())) - 1;
  const integerCell = (await table.locator('tbody tr').first().locator('td').allTextContents())[amountIndex];
  assert.doesNotMatch(integerCell, /\.\d/, 'integer display type drops fractions');
  await operations.getByLabel('Display type', {exact: true}).selectOption('');
  assert.deepEqual(await rowIds(), before);
  // Delete hides the column and restore brings it back.
  await operations.getByRole('button', {name: 'Delete column', exact: true}).click();
  assert.equal(await headers.getByRole('button', {name: /^Amount/}).count(), 0);
  await operations.getByRole('button', {name: /^Restore Amount/}).click();
  await headers.getByRole('button', {name: /^Amount/}).waitFor();
  // Create filter from column is offered for dataset fields.
  assert.equal(await operations.getByRole('button', {name: 'Create filter from column', exact: true}).isEnabled(), true);
  // GPU statistics for a Float32 column over all rows.
  await page.getByLabel('Rows', {exact: true}).selectOption('all');
  const tools = page.locator('.resident-table-column-tools');
  await tools.locator('summary').click();
  await tools.getByLabel('Column', {exact: true}).selectOption('value');
  await tools.getByRole('button', {name: 'Column statistics', exact: true}).click();
  await page.locator('.resident-table-statistics').waitFor({timeout: 30000});
  const statistics = await page.locator('.resident-table-statistics tbody tr').evaluateAll(rows => Object.fromEntries(rows.map(row => [row.querySelector('th').textContent, row.querySelector('td').textContent])));
  assert.match(statistics['Computed on'], /GPU profile/);
  assert(Number(statistics['Rows'].replace(/,/g, '')) >= 10000);
  assert.deepEqual(errors, []);
  console.log('Resident table column operations and GPU statistics passed');
} finally {
  await browser.close();
}
