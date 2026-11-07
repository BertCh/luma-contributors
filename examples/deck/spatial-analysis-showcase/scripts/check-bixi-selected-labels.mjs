// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const directory = path.resolve(process.argv[3] ?? '/tmp/bixi-selected-labels');
await mkdir(directory, {recursive: true});
const report = {passed: false, status: 'pending', failureReason: null, consoleErrors: [], tileErrors: [], selectedLabels: [], visibleLabels: [], mapContext: null};
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
let page;
try {
  page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => report.consoleErrors.push(error.message));
  page.on('console', message => {if (message.type() === 'error') report.consoleErrors.push(message.text());});
  page.on('requestfailed', request => {if (/cartocdn|basemap|\.mvt/.test(request.url())) report.tileErrors.push({url: request.url(), failure: request.failure()?.errorText});});
  const url = new URL(baseUrl);
  url.hash = '#/story/bixi-service-locations/capacity-order';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error, null, {timeout: 90000});
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null);
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  await page.getByRole('radiogroup', {name: 'Capacity assignment order', exact: true}).getByRole('radio', {name: 'Highest activity first', exact: true}).click();
  await page.getByRole('radiogroup', {name: 'Station encoding', exact: true}).getByRole('radio', {name: 'Assignments', exact: true}).click();
  await page.waitForFunction(() => {
    const value = id => document.querySelector(`[data-readout-id="${id}"] dd`)?.textContent ?? '';
    return value('objectiveValue') === '351624.19' && value('selectedStationNames').includes('H:');
  }, null, {timeout: 30000});
  report.selectedLabels = await page.locator('[data-readout-id="selectedStationNames"] dd').evaluate(element => [...element.textContent.matchAll(/(?:^| · )([A-H]):/g)].map(match => match[1]));
  await page.waitForFunction(labels => {
    const visible = [...document.querySelectorAll('.map-annotation-point:not(.is-hidden)')].map(element => element.textContent?.trim());
    return labels.every(label => visible.includes(label));
  }, report.selectedLabels, {timeout: 5000});
  await page.waitForLoadState('networkidle', {timeout: 10000}).catch(() => {});
  report.visibleLabels = await page.locator('.map-annotation-point:not(.is-hidden)').allTextContents();
  report.mapContext = await page.locator('.map-basemap').evaluate(element => ({hidden: element.hidden, opacity: getComputedStyle(element).opacity, canvasCount: element.querySelectorAll('canvas').length}));
  await page.screenshot({path: `${directory}/bixi-selected-labels-desktop.png`, fullPage: true});
  assert.ok(report.visibleLabels.includes('H'));
  assert.deepEqual(report.consoleErrors, []);
  report.passed = true; report.status = 'passed';
} catch (error) {
  report.failureReason = String(error); report.status = 'failed';
  if (page) await page.screenshot({path: `${directory}/bixi-selected-labels-failure.png`, fullPage: true}).catch(() => {});
  process.exitCode = 1;
} finally {
  await browser.close();
  await writeFile(`${directory}/bixi-selected-labels.json`, JSON.stringify(report, null, 2)+'\n');
}
