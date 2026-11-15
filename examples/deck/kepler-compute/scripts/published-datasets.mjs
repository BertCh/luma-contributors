// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getSourceCheckUrl} from './source-check-utils.mjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage();
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(getSourceCheckUrl(process.argv[2], {fixture: 'published-dataset'}));
  await page.waitForFunction(() => window.publishedDatasetResult, undefined, {timeout: 60000});
  const result = await page.evaluate(() => window.publishedDatasetResult);
  assert.equal(result.passed, true, result.error);
  assert.deepEqual(errors, []);
  console.log(result);
} finally {
  await browser.close();
}
