// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/chicago-route-reference.json';
const reference = JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/chicago-route-reference.json', import.meta.url), 'utf8'));
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const reports = [], errors = []; let page;
let passed = false;
let failureReason = null;
try {
  page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
  const url = new URL(baseUrl); url.hash = '#/story/chicago-route-workbench';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error, null, {timeout: 90_000});
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null, 'Scene must compile before numerical checks');
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  const readValues = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''])));
  async function slider(id, value) {await page.locator(`#all-option-${id}`).evaluate((input, nextValue) => {input.value = String(nextValue); input.dispatchEvent(new Event('input', {bubbles: true})); input.dispatchEvent(new Event('change', {bubbles: true}));}, value);}
  async function awaitResults() {await page.waitForFunction(() => [...document.querySelectorAll('.readout')].find(row => row.querySelector('dt')?.textContent === 'Alternative search status')?.querySelector('dd')?.textContent?.startsWith('3 routes certified'), null, {timeout: 30_000});}
  for (const scenario of reference.scenarios) {
    await page.locator('#all-option-routePair').selectOption(scenario.name);
    await awaitResults();
    let observed = await readValues();
    assert.equal(Number.parseInt(observed['Extract topology'], 10), reference.nodeCount);
    assert.ok(Math.abs(Number.parseFloat(observed['Point-on-link route cost']) - scenario.pointCost) < .051);
    const costs = observed['Alternative route costs'].split(' · ').map(value => Number.parseFloat(value));
    assert.equal(costs.length, 3);
    costs.forEach((value, index) => assert.ok(Math.abs(value - scenario.alternatives.found[index].cost) < .051));
    const first = new Set(scenario.alternatives.found[0].edges), shared = new Set(scenario.alternatives.found.slice(1).flatMap(path => path.edges.filter(edge => first.has(edge))));
    assert.equal(Number.parseInt(observed['Shared links'], 10), shared.size);
    assert.match(observed['Reusable cost table'], new RegExp(`${reference.costTableCounts[12]} entries cached`));
    reports.push({scenario: scenario.name, kind: 'static', observed});
    for (const expected of scenario.arrivals) {
      await slider('departureMinute', expected.departure);
      await awaitResults();
      observed = await readValues();
      assert.ok(Math.abs(Number.parseFloat(observed['Affine model arrival']) - expected.affine) < .051);
      assert.ok(Math.abs(Number.parseFloat(observed['Piecewise model arrival']) - expected.piecewise) < .051);
      for (const [model, value] of [['affine-duration', expected.affine], ['piecewise-arrival', expected.piecewise]]) {
        await page.getByRole('radiogroup', {name: 'Arrival model', exact: true}).getByRole('radio', {name: model === 'affine-duration' ? 'Affine duration' : 'Piecewise arrivals', exact: true}).click(); await awaitResults(); observed = await readValues();
        const clock = `${String(Math.floor((480 + value) / 60) % 24).padStart(2, '0')}:${String(Math.floor((480 + value) % 60)).padStart(2, '0')}`;
        assert.match(observed['Selected arrival clock time'], new RegExp(`^${clock} local`));
      }
      reports.push({scenario: scenario.name, kind: 'arrival', expected, observed});
    }
  }
  await slider('costLimitMinutes', 2); await awaitResults();
  const observed = await readValues(); assert.equal(observed['Reusable cost table'], 'No pair within the configured cost limit');
  reports.push({kind: 'bounded-cost-table-miss', observed});
  await page.locator('#all-option-routePair').selectOption('east-north');
  await slider('departureMinute', 240);
  await page.locator('#all-option-turnRestriction').selectOption('hypothetical-ban');
  await page.waitForFunction(() => [...document.querySelectorAll('.readout')].find(row => row.querySelector('dt')?.textContent === 'Expanded-state route status')?.querySelector('dd')?.textContent === 'Hypothetical rule active · expanded-state route converged', null, {timeout: 30_000});
  const restricted = await readValues();
  const turnReference = JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/chicago-restriction-reference.json', import.meta.url), 'utf8')).restriction;
  const comparison = restricted['Baseline vs turn-rule arrival'].match(/Baseline ([\d.]+) min · rule ([\d.]+) min · \+([\d.]+) min/);
  assert.ok(comparison, 'Restricted route comparison must be numeric');
  [turnReference.baselineArrivalMinuteAfter0800, turnReference.hypotheticalArrivalMinuteAfter0800, turnReference.addedMinutes].forEach((expected, index) => assert.ok(Math.abs(Number(comparison[index + 1]) - expected) <= .051));
  reports.push({kind: 'hypothetical-directed-sequence-restriction', observed: restricted, expected: turnReference});
  await page.locator('#all-option-turnRestriction').selectOption('none');
  await page.getByRole('radiogroup', {name: 'Arrival model', exact: true}).getByRole('radio', {name: 'Piecewise arrivals', exact: true}).click();
  const referenceAtZero = reference.scenarios.find(scenario => scenario.name === 'east-north').arrivals.find(arrival => arrival.departure === 0).piecewise;
  for (const [firstSample, departure] of [[15, 0], [60, 30], [60, 60]]) {
    await page.locator('#all-option-firstSampleMinute').selectOption(String(firstSample));
    await slider('departureMinute', departure);
    await awaitResults();
    const waiting = await readValues();
    // Time-translation invariance: a start no later than the shifted first sample has the
    // same route arrival as the independent zero-origin envelope, shifted by that sample.
    const expected = firstSample + referenceAtZero;
    assert.ok(Math.abs(Number.parseFloat(waiting['Piecewise model arrival']) - expected) <= .051);
    assert.equal(waiting['Piecewise first sample and waiting'], `${firstSample} min after 08:00 · initial wait ${Math.max(0, firstSample - departure)} min`);
    reports.push({kind: 'pre-first-sample-waiting', firstSample, departure, expected, observed: waiting});
  }
  assert.deepEqual(errors, []);
  passed = true;
} catch (error) {
  failureReason = String(error);if (page) {await page.screenshot({path: outputFile.replace(/\.json$/, '-failure.png'), fullPage: true}); reports.push({failure: String(error), readouts: await page.locator('.readout').allTextContents(), visibleText: await page.locator('body').innerText()});} throw error;
} finally {await browser.close(); await writeFile(outputFile, JSON.stringify({baseUrl, passed, status: passed ? 'passed' : 'failed', failureReason, reports, consoleErrors: errors}, null, 2) + '\n');}
console.log(`Verified ${reports.length} Chicago routing cases: ${outputFile}`);
