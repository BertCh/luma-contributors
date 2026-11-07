// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

// Interaction/calibration-contract checks over real archived observations and their GPU null runs.
// This does not independently recompute the statistics from raw curves; contributor unit tests
// own that numerical formula comparison. The S2 companion independently checks spatial numbers.
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/point-pattern-global-test.json';
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const reports = [];
const errors = [];
let webgpuErrors = [];

function readPageReadouts() {
  return Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [
    row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''
  ]));
}

try {
  const page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
  await page.addInitScript(() => {
    globalThis.showcaseValidationGpuErrors = [];
    if (typeof GPUAdapter === 'undefined') return;
    const requestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (...argumentsList) {
      const device = await requestDevice.apply(this, argumentsList);
      device.addEventListener('uncapturederror', event => {
        globalThis.showcaseValidationGpuErrors.push(event.error.message);
      });
      return device;
    };
  });
  const url = new URL(baseUrl);
  url.hash = '#/story/point-patterns';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready, null, {timeout: 90_000});
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  const select = (name, value) => name === 'nullModel'
    ? page.locator('[role="radiogroup"][aria-labelledby="all-option-nullModel-label"]')
      .getByRole('radio', {name: value === 'city' ? 'On city land (GPU-sampled)' : 'In the rectangle (contrast)', exact: true}).click()
    : page.locator(`#all-option-${name}`).selectOption(value);
  const setRange = (name, value) => page.locator(`#all-option-${name}`).evaluate((input, nextValue) => {
    input.value = String(nextValue);
    input.dispatchEvent(new Event('input', {bubbles: true}));
    input.dispatchEvent(new Event('change', {bubbles: true}));
  }, value);
  await select('window', 'city');
  await select('nullModel', 'city');
  await setRange('jitter', 0);

  async function checkStatistic(statistic, significanceLevel) {
    await select('globalStatistic', statistic);
    await select('significanceLevel', significanceLevel);
    // The exact scene runs one null replicate per submitted/read-back job. Permit that work to
    // finish, then require a current result rather than accepting the initial story-ready flag.
    await page.waitForFunction(expected => {
      const read = label => [...document.querySelectorAll('.readout')]
        .find(row => row.querySelector('dt')?.textContent === label)?.querySelector('dd')?.textContent ?? '';
      return read('Random patterns run') === '39 of 39 random patterns' &&
        read('GPU global statistic').startsWith(`${expected.statistic}:`) &&
        read('Global test status').includes(`α = ${expected.significanceLevel}`) &&
        read('Global test status').includes('global test over the full radius range');
    }, {statistic, significanceLevel}, {timeout: 180_000});
    const observed = await page.evaluate(readPageReadouts);
    const statistics = observed['GPU global statistic'];
    const probabilityMatch = statistics.match(/conservative p = ([\d.]+)/);
    assert.ok(probabilityMatch, 'Conservative p must be numeric');
    const probability = Number(probabilityMatch[1]);
    assert.ok(probability >= 0.025 && probability <= 1, '39 simulations bound attainable conservative p');
    assert.ok(Math.abs(probability * 40 - Math.round(probability * 40)) < 0.001, 'p must remain on the 1/40 grid');
    assert.match(statistics, /39 simulations, resolution 0\.025/);
    assert.match(statistics, statistic === 'maximum-deviation' ? /[\d.] m;/ : statistic === 'integrated-squared-deviation' ? /[\d.] m³;/ : /[\d.] rank;/);
    const decision = observed['Global test status'];
    if (probability <= Number(significanceLevel)) assert.match(decision, /^Reject at/);
    else assert.ok(decision.startsWith('Do not reject') || decision.startsWith('Tie bounds straddle'), 'Conservative decision must agree with p');
    if (statistic === 'extreme-rank') {
      const intervalMatch = statistics.match(/tie bounds ([\d.]+)–([\d.]+)/);
      assert.ok(intervalMatch, 'Rank ties must have an explicit p interval');
      const lower = Number(intervalMatch[1]);
      const upper = Number(intervalMatch[2]);
      assert.ok(lower >= 0 && lower <= upper && upper === probability);
      assert.ok(Math.abs(lower * 40 - Math.round(lower * 40)) < 0.001);
    }
    reports.push({statistic, significanceLevel, observed});
    console.log(`Checked ${statistic} at alpha ${significanceLevel}: ${probability}`);
  }

  await checkStatistic('maximum-deviation', '0.05');
  await checkStatistic('integrated-squared-deviation', '0.05');
  await checkStatistic('extreme-rank', '0.05');
  await checkStatistic('extreme-rank', '0.01');

  async function checkUnavailable(reason) {
    await page.waitForFunction(expectedReason => {
      const read = label => [...document.querySelectorAll('.readout')]
        .find(row => row.querySelector('dt')?.textContent === label)?.querySelector('dd')?.textContent ?? '';
      return read('Global test status').includes(expectedReason) && !read('GPU global statistic').includes('conservative p');
    }, reason, {timeout: 10_000});
    reports.push({gate: reason, observed: await page.evaluate(readPageReadouts)});
    console.log(`Checked calibration gate: ${reason}`);
  }
  await select('nullModel', 'window');
  await checkUnavailable('match the random-point support');
  await select('nullModel', 'city');
  await setRange('jitter', 5);
  await checkUnavailable('turn jitter off');
  await setRange('jitter', 0);
  // Supersede observed work and a test option in quick succession. A completed cached curve
  // must not be published under the newer generation while its replacement is still pending.
  await setRange('maximumDistance', 1000);
  await select('globalStatistic', 'maximum-deviation');
  await setRange('maximumDistance', 1500);
  await checkStatistic('extreme-rank', '0.05');
  webgpuErrors = await page.evaluate(() => globalThis.showcaseValidationGpuErrors ?? []);
  assert.deepEqual(errors, [], 'Browser errors must not be hidden by successful readouts');
  assert.deepEqual(webgpuErrors, [], 'Uncaptured GPU errors must fail validation');
} finally {
  await browser.close();
  await writeFile(outputFile, JSON.stringify({baseUrl, reports, errors, webgpuErrors}, null, 2) + '\n');
}
console.log(`Verified ${reports.length} global-test states: ${outputFile}`);
