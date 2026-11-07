// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

// Independent expected topology is recorded by coverage-audit-reference.py / GEOS 3.13.1.
// Source remains valid at gaps 0,10,100m; state dissolve has five shells, combined has three.
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/coverage-audit-browser-reference.json';
const reports = [];
const errors = [];
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
let page;
try {
  page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
  const url = new URL(baseUrl);
  url.hash = '#/story/noding-and-coverage/coverage-validity';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || document.body.innerText.includes('This analysis could not start'), null, {timeout: 120000});
  assert.ok(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.ready), await page.locator('body').innerText());
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  await page.locator('#all-option-view').selectOption('coverage-audit');
  const read = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''])));
  const wait = gap => page.waitForFunction(expected => [...document.querySelectorAll('.readout')].some(row => row.querySelector('dt')?.textContent === 'GPU coverage validity' && (row.querySelector('dd')?.textContent?.includes(`gapWidth ${expected} m`) || row.querySelector('dd')?.textContent?.includes('graph build failed'))), gap, {timeout: 120000});
  for (const group of ['state', 'single']) for (const gap of ['0', '10', '100']) {
    await page.locator('#all-option-coverageAuditGroup').selectOption(group);
    await page.locator('#all-option-coverageAuditGapWidth').selectOption(gap);
    await wait(gap);
    const readouts = await read();
    const shells = group === 'state' ? 5 : 3;
    assert.match(readouts['GPU coverage validity'], /isValid=1; invalid segments 0;/);
    assert.match(readouts['Coverage dissolve status'], /written; labels .*; input isValid=1;/);
    assert.match(readouts['Coverage dissolve status'], /overflow=0; open=0;/);
    assert.match(readouts['Coverage dissolve output'], new RegExp(`^${shells} GPU output polygons; ${shells} rings \\(0 holes\\)`));
    reports.push({kind: 'source', group, gap, readouts});
    console.log(`Accepted coverage ${group}, gap ${gap} m`);
  }
  await page.locator('#all-option-coverageAuditInput').selectOption('perturbed');
  await wait('100');
  let readouts = await read();
  assert.match(readouts['GPU coverage validity'], /isValid=0;/);
  assert.match(readouts['Coverage dissolve status'], /withheld; labels .*; input isValid=0;/);
  reports.push({kind: 'synthetic-perturbation', readouts});
  await page.locator('#all-option-coverageAuditInput').selectOption('source');
  await page.locator('#all-option-coverageAuditGroup').selectOption('state');
  await page.locator('#all-option-coverageAuditGapWidth').selectOption('0');
  await wait('0');
  readouts = await read();
  assert.match(readouts['GPU coverage validity'], /isValid=1; invalid segments 0;/);
  assert.match(readouts['Coverage dissolve output'], /^5 GPU output polygons; 5 rings/);
  reports.push({kind: 'cached-source-return', readouts});
  assert.deepEqual(errors, []);
} finally {
  const lastBody = page ? await page.locator('body').innerText().catch(() => '') : '';
  const lastReadouts = page ? await page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? '']))).catch(() => ({})) : {};
  await browser.close();
  await writeFile(outputFile, JSON.stringify({baseUrl, reports, errors, lastBody, lastReadouts}, null, 2) + '\n');
}
