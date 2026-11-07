// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 390, height: 844}});
await page.routeWebSocket('**/*', socket => {
  const server = socket.connectToServer();
  server.onMessage(message => {if (typeof message === 'string') {try {const packet = JSON.parse(message); if (packet.type === 'update' || packet.type === 'full-reload') return;} catch {}} socket.send(message);});
  socket.onMessage(message => server.send(message));
});
const errors = []; page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=128');
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 128, undefined, {timeout: 60000});
  assert.equal(await page.locator('.compute-panel').count(), 0);
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.getByText('Try a sample dataset', {exact: true}).click();
  await workspace.getByLabel('Sample dataset', {exact: true}).selectOption('districts');
  await workspace.getByRole('button', {name: 'Prepare sample', exact: true}).click();
  const roles = workspace.getByLabel('Source geometry column');
  await roles.scrollIntoViewIfNeeded();
  assert.equal(await roles.inputValue(), 'geometry');
  assert.equal(await roles.evaluate(element => {
    const bounds = element.getBoundingClientRect(); return document.elementFromPoint(bounds.left + bounds.width / 2, bounds.top + bounds.height / 2) === element;
  }), true);
  await page.screenshot({path: '/tmp/kepler-compute-mobile-open.png'});
  await closeNativeDialog(page);
  assert.equal(await page.locator('.ReactModal__Overlay').count(), 0);
  assert.deepEqual(errors, []);
  console.log('Native mobile Add Data passed: gallery preparation, geometry control hit-test, Escape close, no floating compute panel.');
} catch (error) {console.error(await page.evaluate(() => document.body.innerText.slice(-2000))); throw error;} finally {await browser.close();}
