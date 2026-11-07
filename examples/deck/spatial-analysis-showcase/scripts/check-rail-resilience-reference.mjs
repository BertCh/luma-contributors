// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

// Run against a stable built showcase. References are independently computed by Dijkstra on
// the bundled directed network, not captured from the GPU implementation under test.
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/rail-resilience-reference.json';
const reference = JSON.parse(await readFile(new URL(
  '../../../../dev-docs/research/showcase-overhaul-2026-10-25/movement-network-reference.json',
  import.meta.url
), 'utf8'));
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const reports = [];
const errors = [];
let passed = false;
let failureReason = null;
try {
  const page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') errors.push(message.text());
  });
  const url = new URL(baseUrl);
  url.hash = '#/story/rail-resilience';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready, null, {timeout: 90_000});
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  const originControl = page.locator('#all-option-origin');
  const corridorControl = page.locator('#all-option-corridor');
  await page.locator('#all-option-trainTypes').selectOption('all');
  const eventControl = page.locator('#all-option-event');
  if (await eventControl.isEnabled()) await eventControl.selectOption('closure');
  else assert.equal(await eventControl.inputValue(), 'closure');
  for (const expected of reference.rail.scenarios) {
    const closureName = `${expected.closedStation.replace(' Centraal', '')} interchange offline`;
    await originControl.selectOption(expected.origin);
    await corridorControl.selectOption(closureName);
    await page.waitForFunction(expectedValues => {
      const read = label => [...document.querySelectorAll('.readout')]
        .find(row => row.querySelector('dt')?.textContent === label)?.querySelector('dd')?.textContent ?? '';
      return read('Routing status') === 'Both routing stages converged' &&
        Number.parseInt(read('Stations within budget before'), 10) === expectedValues.baselineWithin90Minutes &&
        Number.parseInt(read('Stations within budget after'), 10) === expectedValues.scenarioWithin90Minutes &&
        Number.parseInt(read('Newly unreachable'), 10) === expectedValues.newlyUnreachable;
    }, expected, {timeout: 30_000});
    const observed = await page.evaluate(() => Object.fromEntries(
      [...document.querySelectorAll('.readout')].map(row => [
        row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''
      ])
    ));
    assert.equal(Number.parseInt(observed['Newly unreachable'], 10), expected.newlyUnreachable);
    assert.equal(Number.parseInt(observed['Stations within budget before'], 10), expected.baselineWithin90Minutes);
    assert.equal(Number.parseInt(observed['Stations within budget after'], 10), expected.scenarioWithin90Minutes);
    reports.push({expected, observed});
  }
  assert.deepEqual(errors, [], 'Browser errors must not be hidden by numerical agreement');
  passed = true;
} catch (error) {
  failureReason = String(error);
  throw error;
} finally {
  await browser.close();
  await writeFile(outputFile, JSON.stringify({baseUrl, passed, status: passed ? 'passed' : 'failed', failureReason, reports, consoleErrors: errors}, null, 2) + '\n');
}
console.log(`Verified ${reports.length} rail scenarios against independent references: ${outputFile}`);
