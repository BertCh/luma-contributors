// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

// Independent directed Brandes results on the exact regional induced BTS inputs.
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/aviation-network-reference.json';
const reference = JSON.parse(await readFile(new URL(
  '../../../../dev-docs/research/showcase-overhaul-2026-10-25/aviation-network-reference.json',
  import.meta.url
), 'utf8'));
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const reports = [];
const errors = [];
let page;
let passed = false;
let failureReason = null;
try {
  page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  const url = new URL(baseUrl);
  url.hash = '#/story/aviation-network-lab';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready, null, {timeout: 90_000});
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  for (const [region, expected] of Object.entries(reference)) {
    await page.locator('#all-option-region').selectOption(region);
    await page.waitForFunction(arcs => {
      const read = label => [...document.querySelectorAll('.readout')]
        .find(row => row.querySelector('dt')?.textContent === label)?.querySelector('dd')?.textContent ?? '';
      return read('Regional graph').includes(`${arcs} directed`) &&
        read('GPU status') === 'Both centrality searches converged';
    }, expected.arcs, {timeout: 90_000});
    for (const [measure, referenceField] of [
      ['distanceBroker', 'distanceBetweenness'], ['hopBroker', 'hopBetweenness']
    ]) {
      await page.locator('#all-option-measure').selectOption(measure);
      // Control handlers publish the selected measure synchronously; allow a rendered frame.
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => resolve())));
      const observed = await page.evaluate(() => Object.fromEntries(
        [...document.querySelectorAll('.readout')].map(row => [
          row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''
        ])
      ));
      assert.equal(Number.parseInt(observed['Regional graph'], 10), expected.airports);
      assert.equal(Number.parseInt(observed['Scheduled flight legs'].replaceAll(',', ''), 10), expected.scheduledFlightLegs);
      const entries = observed['Highest broker airports'].split(' · ').map(entry => {
        const [iata, value] = entry.split(' ');
        return {iata, value: Number(value)};
      });
      assert.equal(entries.length, 5);
      const sortedReference = expected.centrality.map(row => row[referenceField]).sort((first, second) => second - first);
      for (let index = 0; index < entries.length; index++) {
        const entry = entries[index];
        const airport = expected.centrality.find(row => row.iata === entry.iata);
        assert.ok(airport, `Unexpected airport ${entry.iata}`);
        assert.ok(Math.abs(entry.value - airport[referenceField]) <= 0.011,
          `${region}/${measure}: ${entry.iata} GPU ${entry.value}, reference ${airport[referenceField]}`);
        assert.ok(Math.abs(entry.value - sortedReference[index]) <= 0.011,
          `${region}/${measure}: incorrect ranking at ${index}`);
      }
      reports.push({region, measure, observed});
    }
  }
  assert.deepEqual(errors, [], 'Browser errors must not be hidden by numerical agreement');
  passed = true;
} catch (error) {
  failureReason = String(error);
  if (page) {
    await page.screenshot({path: outputFile.replace(/\.json$/, '-failure.png'), fullPage: true});
    reports.push({failure: String(error), visibleText: await page.locator('body').innerText()});
  }
  throw error;
} finally {
  await browser.close();
  await writeFile(outputFile, JSON.stringify({baseUrl, passed, status: passed ? 'passed' : 'failed', failureReason, reports, consoleErrors: errors}, null, 2) + '\n');
}
console.log(`Verified ${reports.length} regional centrality rankings: ${outputFile}`);
