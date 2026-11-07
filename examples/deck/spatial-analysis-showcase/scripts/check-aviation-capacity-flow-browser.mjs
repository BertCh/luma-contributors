// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {dirname, resolve} from 'node:path';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const oracle = JSON.parse(
  await readFile(
    new URL(
      '../../../../dev-docs/research/showcase-overhaul-2026-10-25/aviation-capacity-flow-reference.json',
      import.meta.url
    ),
    'utf8'
  )
);
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/luma-showcase-overhaul-20261025/aviation-capacity-flow.json';
await mkdir(dirname(resolve(outputFile)), {recursive: true});
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
let page;
const reports = [];
const consoleErrors = [];
let passed = false;
let failureReason = null;

const readReadouts = () =>
  page.evaluate(() =>
    Object.fromEntries(
      [...document.querySelectorAll('.readout')].map(row => [
        row.querySelector('dt')?.textContent ?? '',
        row.querySelector('dd')?.textContent ?? ''
      ])
    )
  );

async function waitForReadout(label, text) {
  await page.waitForFunction(
    ({readoutLabel, expectedText}) => {
      const row = [...document.querySelectorAll('.readout')].find(
        item => item.querySelector('dt')?.textContent === readoutLabel
      );
      return row?.querySelector('dd')?.textContent?.includes(expectedText);
    },
    {readoutLabel: label, expectedText: text},
    {timeout: 60_000}
  );
}

async function setSlider(identifier, value) {
  await page.locator(`#all-option-${identifier}`).evaluate((input, nextValue) => {
    input.value = String(nextValue);
    input.dispatchEvent(new Event('input', {bubbles: true}));
    input.dispatchEvent(new Event('change', {bubbles: true}));
  }, value);
}

async function setSelect(identifier, value) {
  await page.locator(`#all-option-${identifier}`).selectOption(value);
}

function numberFrom(readouts, label) {
  const text = readouts[label] ?? '';
  const match = text.replaceAll(',', '').match(/-?\d+/);
  assert.ok(match, `${label} has no number: ${text}`);
  return Number(match[0]);
}

try {
  page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => consoleErrors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  const url = new URL(baseUrl);
  url.hash = '#/story/aviation-capacity-flow';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(
    () => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error,
    null,
    {timeout: 90_000}
  );
  assert.equal(
    await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error),
    null,
    'Flow scene must initialize before numerical checks'
  );
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  await waitForReadout('GPU flow and cut status', 'Complete');

  let readouts = await readReadouts();
  let value = numberFrom(readouts, 'Allocated monthly flow');
  assert.equal(value, oracle.cases.seaToSanMaximumAt100Percent.maximumFlow);
  assert.match(readouts['Residual cut'], /4,203 = flow 4,203/);
  reports.push({scenario: 'SEA to SAN maximum at 100%', readouts});

  await setSelect('objective', 'fixedBudget');
  await setSlider('flowBudget', 1000);
  await waitForReadout('Aggregate delay score', '6,790');
  readouts = await readReadouts();
  assert.equal(numberFrom(readouts, 'Allocated monthly flow'), 1000);
  assert.equal(numberFrom(readouts, 'Aggregate delay score'), oracle.cases.seaToSanDelayWeightedBudget1000At100Percent.totalDelayScore);
  reports.push({scenario: 'fixed 1,000-unit delay-score allocation', readouts});

  await setSelect('objective', 'maximum');
  await setSelect('capacityShare', '25');
  await waitForReadout('Residual cut', '1,045 = flow 1,045');
  readouts = await readReadouts();
  assert.equal(numberFrom(readouts, 'Allocated monthly flow'), oracle.cases.seaToSanMaximumAt25Percent.maximumFlow);
  reports.push({scenario: 'maximum at 25% illustrative budgets', readouts});

  await setSelect('objective', 'fixedBudget');
  await setSlider('flowBudget', 4000);
  await waitForReadout('GPU flow and cut status', 'Budget shortfall');
  await waitForReadout('Residual cut', '1,045 = flow 1,045');
  readouts = await readReadouts();
  assert.equal(numberFrom(readouts, 'Allocated monthly flow'), oracle.cases.seaToSanDelayWeightedBudget4000At25Percent.maximumFlow);
  assert.equal(numberFrom(readouts, 'Aggregate delay score'), oracle.cases.seaToSanDelayWeightedBudget4000At25Percent.totalDelayScore);
  reports.push({scenario: '4,000-unit budget shortfall at 25%', readouts});

  await setSelect('capacityShare', '100');
  await setSelect('terminalMode', 'balanced4');
  await setSlider('flowBudget', 1000);
  await waitForReadout('Aggregate delay score', '4,479');
  readouts = await readReadouts();
  assert.equal(numberFrom(readouts, 'Allocated monthly flow'), oracle.cases.balancedSeaPdxToLaxSanBudget1000At100Percent.maximumFlow);
  assert.equal(numberFrom(readouts, 'Aggregate delay score'), oracle.cases.balancedSeaPdxToLaxSanBudget1000At100Percent.totalDelayScore);
  // Balanced virtual supply gates total 1,000, so meeting every supply also certifies the augmented cut.
  assert.equal(readouts['Residual cut'], 'Augmented-graph cut 1,000 = flow 1,000');
  reports.push({scenario: 'balanced SEA/PDX to LAX/SAN demands', readouts});

  assert.deepEqual(consoleErrors, [], 'Browser or GPU errors');
  passed = true;
} catch (error) {
  failureReason = String(error);
  if (page) {
    await page.screenshot({path: outputFile.replace(/\.json$/, '-failure.png'), fullPage: true});
    reports.push({failureReason, readouts: await page.locator('.readout').allTextContents()});
  }
  throw error;
} finally {
  await browser.close();
  await writeFile(
    outputFile,
    `${JSON.stringify({baseUrl, passed, status: passed ? 'passed' : 'failed', failureReason, reports, consoleErrors}, null, 2)}\n`
  );
}
