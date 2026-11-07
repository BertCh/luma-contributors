// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {closeNativeDialog, openNativeExport} from './native-navigation.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1200, height: 900}, acceptDownloads: true});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256');
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await closeNativeDialog(page);
  const initial = await page.evaluate(() => ({state: window.__keplerComputeTest.getState(), config: window.__keplerComputeTest.saveNativeConfig()}));
  await openNativeExport(page, true);
  await page.getByRole('button', {name: 'Save standalone HTML', exact: true}).click();
  const download = await page.waitForEvent('download', {timeout: 60000});
  assert.equal(download.suggestedFilename(), 'kepler-workspace.html');
  const html = await readFile(await download.path(), 'utf8');
  assert(html.includes('id="luma-kepler-bundle"'), 'Bundle must be embedded');
  const hasViewer = !html.includes('runtime not included');
  if (hasViewer) {
    // Embedded viewer (yarn build:viewer): block all network, then render from the file alone.
    assert(!/<script[^>]+src=/.test(html), 'Viewer must not reference external scripts');
    const offlineContext = await browser.newContext({offline: true});
    await offlineContext.route('**/*', route => (route.request().url().startsWith('data:') || route.request().url().startsWith('about:') ? route.continue() : route.abort()));
    const offlinePage = await offlineContext.newPage();
    const offlineErrors = [];
    offlinePage.on('pageerror', error => offlineErrors.push(error.message));
    await offlinePage.setContent(html);
    await offlinePage.waitForFunction(() => document.querySelector('#map canvas'), undefined, {timeout: 30000});
    assert.match(await offlinePage.locator('#status').innerText(), /Rendering \d+ dataset/);
    assert.deepEqual(offlineErrors, []);
    await offlineContext.close();
  } else {
    // Fallback page: loads offline, verifies digests and offers the bundle again.
    const packaged = await browser.newPage();
    // Digest verification needs a secure context: serve the file from a loopback origin (about:blank has none).
    await packaged.route('http://127.0.0.1:5318/standalone-check.html', route => route.fulfill({contentType: 'text/html', body: html}));
    await packaged.goto('http://127.0.0.1:5318/standalone-check.html');
    await packaged.waitForFunction(() => document.querySelectorAll('#summary li').length > 1);
    const lines = await packaged.locator('#summary li').allInnerTexts();
    assert(lines.some(line => line.includes('sha256 verified')), lines.join('\n'));
    assert(!lines.some(line => line.includes('MISMATCH')));
  }
  const final = await page.evaluate(() => ({state: window.__keplerComputeTest.getState(), config: window.__keplerComputeTest.saveNativeConfig()}));
  assert.equal(final.state.deviceGeneration, initial.state.deviceGeneration);
  assert.deepEqual(final.config, initial.config, 'HTML export must not edit the workspace');
  assert.deepEqual(errors, []);
  console.log('Standalone HTML export passed: embedded bundle, digest verification, honest runtime message and unchanged workspace.');
} finally {
  await browser.close();
}
