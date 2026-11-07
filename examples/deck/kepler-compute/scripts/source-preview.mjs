// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {chromium} from 'playwright';

const browser = await chromium.launch({headless: true});
const page = await browser.newPage();
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/scripts/fixtures/source-preview.html');
  const publish = page.getByRole('button', {name: 'Publish selected fields'});
  await publish.waitFor();
  assert.equal(await publish.isDisabled(), true);
  await page.getByRole('button', {name: 'Check schema and precision'}).click();
  assert.equal(await publish.isDisabled(), true);
  assert.deepEqual(await page.evaluate(() => window.sourceUploadCalls), []);
  await page.getByLabel('Float64 conversion').selectOption('round');
  await page.getByRole('checkbox', {name: /^geometry/}).uncheck();
  await page.getByRole('button', {name: 'Check schema and precision'}).click();
  assert.equal(await publish.isEnabled(), true);
  await page.getByLabel('Maximum absolute error').fill('0.00001');
  assert.equal(await publish.isDisabled(), true);
  await page.getByRole('button', {name: 'Check schema and precision'}).click();
  await publish.click();
  await page.getByRole('status').filter({hasText: 'Selected fields published'}).waitFor();
  assert.deepEqual(await page.evaluate(() => window.sourceUploadCalls), [{rowCount: 2, normalization: {columns: ['amount', 'region'], float64: {mode: 'round', maximumAbsoluteError: 0.00001, maximumRelativeError: 0}}}]);
  assert.deepEqual(errors, []);
  console.log('Source preview UI checks passed: explicit precision, excluded fields, stale preview invalidation and explicit publication.');
} finally { await browser.close(); }
