// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/rail-support-reference.json';
const reference = JSON.parse(await readFile(new URL(
  '../../../../dev-docs/research/showcase-overhaul-2026-10-25/rail-support-reference.json', import.meta.url
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
  url.hash = '#/story/rail-support-planning';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready, null, {timeout: 90_000});
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  async function setSlider(id, value) {
    await page.locator(`#all-option-${id}`).evaluate((input, nextValue) => {
      input.value = String(nextValue);
      input.dispatchEvent(new Event('input', {bubbles: true}));
      input.dispatchEvent(new Event('change', {bubbles: true}));
    }, value);
  }
  async function readValues() {
    return await page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')]
      .map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''])));
  }
  for (const expected of reference.serviceAreas) {
    await page.locator('#all-option-direction').selectOption(expected.direction);
    await setSlider('costLimitMinutes', expected.costLimitMinutes);
    await page.waitForFunction(expectedValues => {
      const read = label => [...document.querySelectorAll('.readout')]
        .find(row => row.querySelector('dt')?.textContent === label)?.querySelector('dd')?.textContent ?? '';
      return read('GPU status') === 'Service areas and cost weights converged' &&
        Number.parseInt(read('Stations assigned'), 10) === expectedValues.countsByHub.reduce((sum, count) => sum + count, 0) &&
        Number.parseInt(read('Weighted station-desk pairs'), 10) === expectedValues.eligiblePairs;
    }, expected, {timeout: 30_000});
    const observed = await readValues();
    assert.equal(Number.parseInt(observed['Outside catchment'], 10), expected.unassigned);
    for (let index = 0; index < reference.hubs.length; index++) {
      assert.ok(observed['Candidate hubs'].includes(`${reference.hubs[index].replace(' Centraal', '')}: ${expected.countsByHub[index]}`));
    }
    reports.push({expected, observed});
  }
  await page.locator('#all-option-direction').selectOption('inbound');
  await setSlider('costLimitMinutes', 60);
  await setSlider('beta', 0.02);
  const expected = reference.serviceAreas.find(row => row.direction === 'inbound' && row.costLimitMinutes === 60);
  for (const [decay, key] of [
    ['none', 'none'], ['exponential', 'exponentialBeta002PerMinute'],
    ['power', 'powerExponent002MinuteFloor1'], ['gaussian', 'gaussian']
  ]) {
    await page.locator('#all-option-decay').selectOption(decay);
    await page.waitForFunction(total => {
      const row = [...document.querySelectorAll('.readout')].find(element => element.querySelector('dt')?.textContent === 'Weighted station-desk pairs');
      const value = Number.parseFloat(row?.querySelector('dd')?.textContent?.split('total weight ')[1] ?? '');
      return Math.abs(value - total) < 0.06;
    }, expected.summedDecayWeights[key], {timeout: 30_000});
    reports.push({decay, expectedTotal: expected.summedDecayWeights[key], observed: await readValues()});
  }
  url.hash = '#/story/rail-support-allocation';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready, null, {timeout: 90_000});
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  for (const [objective, expectedText] of [
    ['pmedian', 'Utrecht + Rotterdam · 231.0 station-minutes'],
    ['backup', 'Amsterdam + Utrecht · 7 / 8 double-covered stations'],
    ['dispersion', 'Rotterdam + Zwolle · 84.0 min minimum separation']
  ]) {
    await page.locator('#all-option-objective').selectOption(objective);
    await page.waitForFunction(expectedValue => [...document.querySelectorAll('.readout')]
      .find(row => row.querySelector('dt')?.textContent === 'Placement objective')
      ?.querySelector('dd')?.textContent === expectedValue, expectedText, {timeout: 30_000});
    reports.push({objective, expectedText, observed: await readValues()});
  }
  await page.locator('#all-option-objective').selectOption('backup');
  await setSlider('thresholdMinutes', 30);
  await page.waitForFunction(() => [...document.querySelectorAll('.readout')]
    .find(row => row.querySelector('dt')?.textContent === 'Exhaustive search')
    ?.querySelector('dd')?.textContent?.includes('no feasible incumbent'), null, {timeout: 30_000});
  const infeasible = await readValues();
  assert.ok(!infeasible['Placement objective'].includes('NaN'));
  reports.push({objective: 'backup', thresholdMinutes: 30, expected: 'infeasible', observed: infeasible});
  for (const demandMode of ['integral', 'fractional']) {
    for (const stepMode of ['fraction', 'absolute']) {
      await page.locator('#all-option-rationalMode').selectOption(demandMode);
      await page.locator('#all-option-stepMode').selectOption(stepMode);
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => resolve())));
      const observed = await readValues();
      assert.ok(!observed['Rational allocation'].includes('NaN'));
      assert.match(observed['Rational allocation'], /0\.0 unserved units/);
      reports.push({demandMode, stepMode, observed});
    }
  }
  assert.deepEqual(errors, []);
  passed = true;
} catch (error) {
  failureReason = String(error);
  if (page) {
    await page.screenshot({path: outputFile.replace(/\.json$/, '-failure.png'), fullPage: true});
    reports.push({failure: String(error), readouts: await page.locator('.readout').allTextContents(), visibleText: await page.locator('body').innerText()});
  }
  throw error;
} finally {
  await browser.close();
  await writeFile(outputFile, JSON.stringify({baseUrl, passed, status: passed ? 'passed' : 'failed', failureReason, reports, consoleErrors: errors}, null, 2) + '\n');
}
console.log(`Verified ${reports.length} rail service-area and decay configurations: ${outputFile}`);
