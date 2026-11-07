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
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/scripts/fixtures/sandbox-share-panel.html');
  await page.getByRole('button', {name: 'Create share link'}).click();
  const link = await page.getByLabel('Sandbox share link').inputValue();
  assert.match(link, /#sandbox=v1\./);
  assert.equal(await page.evaluate(() => window.sharedStateLoads.length), 0);
  await page.evaluate(() => {Object.defineProperty(navigator, 'clipboard', {configurable: true, value: {writeText: async () => {throw new Error('denied');}}});});
  await page.getByRole('button', {name: 'Copy link', exact: true}).click();
  await page.getByRole('status').filter({hasText: 'copy it manually'}).waitFor();
  await page.goto(link);
  await page.waitForFunction(() => window.sharedStateLoads.length === 1);
  await page.reload();
  await page.waitForFunction(() => window.sharedStateLoads.length === 1);
  assert.equal(await page.evaluate(() => window.sharedStateLoads[0].sources[0].reference.parameters.rowCount), 120);
  await page.evaluate(() => window.setShareBusy(true));
  await page.waitForFunction(() => document.querySelector('button').disabled);
  await page.evaluate(() => {window.location.hash = '';});
  await page.waitForFunction(() => window.location.hash === '');
  await page.evaluate(hash => {window.location.hash = hash;}, new URL(link).hash);
  // The hashchange handler must run while the operation remains busy.
  await page.evaluate(() => new Promise(resolve => setTimeout(resolve, 100)));
  assert.equal(await page.evaluate(() => window.sharedStateLoads.length), 1);
  await page.evaluate(() => window.setShareBusy(false));
  await page.waitForFunction(() => window.sharedStateLoads.length === 2);
  await page.evaluate(() => {window.location.hash = '#sandbox=v9.invalid';});
  await page.getByRole('status').filter({hasText: 'Unsupported share link version'}).waitFor();
  assert.equal(await page.evaluate(() => window.sharedStateLoads.length), 2);
  assert.deepEqual(errors, []);
  console.log('Sandbox share UI checks passed: create, manual clipboard fallback, navigation/reload, deferred loading during operations and malformed-link errors.');
} finally {await browser.close();}
