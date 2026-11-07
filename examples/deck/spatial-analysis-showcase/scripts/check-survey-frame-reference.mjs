// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory = process.argv[3] ?? '/tmp/survey-frame-reference';
const reference = JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/pointcloud-dem-validation.json', import.meta.url), 'utf8'));
await mkdir(outputDirectory, {recursive: true});
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const reports = [];
const errors = [];
try {
  const page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
  const url = new URL(baseUrl);
  url.hash = '#/story/survey-frame-sensitivity';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error, null, {timeout: 90_000});
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null);
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  const regionLabels = {west: 'West', east: 'East', north: 'North', all: 'All paired cells'};
  const labels = {
    stableCount: 'Cells in hypothetical control area',
    medianBiasMeters: 'Raw median vertical offset',
    residualNmadMeters: 'Control residual NMAD',
    biasStandardErrorMeters: 'Estimated bias standard error',
    correctedVolumeCubicMeters: 'Corrected vertical surface volume',
    volumeStandardDeviationCubicMeters: 'Volume standard error',
    validAreaSquareMeters: 'Paired observed area'
  };
  for (const expected of reference.cases) {
    await page.getByRole('group', {name: 'Hypothetical control area'}).getByRole('button', {name: regionLabels[expected.region], exact: true}).click();
    const deviationControl = page.locator('#all-option-assumedDifferenceSd');
    if (Number(await deviationControl.inputValue()) !== expected.assumedDifferenceDeviationMeters) {
      await deviationControl.evaluate((element, value) => {
        element.value = String(value);
        element.dispatchEvent(new Event('input', {bubbles: true}));
        element.dispatchEvent(new Event('change', {bubbles: true}));
      }, expected.assumedDifferenceDeviationMeters);
    }
    await page.locator('#all-option-correlationLength').evaluate((element, value) => {
      element.value = String(value);
      element.dispatchEvent(new Event('input', {bubbles: true}));
      element.dispatchEvent(new Event('change', {bubbles: true}));
    }, expected.correlationLengthMeters);
    await page.waitForFunction(({expectedValues, names}) => {
      const read = name => [...document.querySelectorAll('.readout')].find(row => row.querySelector('dt')?.textContent === name)?.querySelector('dd')?.textContent ?? '';
      return Object.entries(names).every(([field, label]) => Math.abs(Number.parseFloat(read(label).replaceAll(',', '')) - expectedValues[field]) < 0.02);
    }, {expectedValues: expected, names: labels}, {timeout: 60_000});
    const observed = await page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].filter(row => row.querySelector('dd')).map(row => [row.querySelector('dt').textContent, row.querySelector('dd').textContent])));
    for (const [field, label] of Object.entries(labels)) {
      const tolerance = field.includes('Volume') || field.includes('volume') ? 0.02 : 0.001;
      assert.ok(Math.abs(Number.parseFloat(observed[label].replaceAll(',', '')) - expected[field]) < tolerance, `${label} must match independent reference for ${expected.region}/${expected.correlationLengthMeters}`);
    }
    reports.push({expected, observed});
  }
  await page.getByRole('radio', {name: 'After vertical shift', exact: true}).click();
  await page.getByRole('radio', {name: 'After vertical shift', exact: true}).and(page.locator('[aria-checked="true"]')).waitFor();
  await page.screenshot({path: `${outputDirectory}/corrected-volume.png`});
  assert.deepEqual(errors, [], 'Browser errors must not be hidden by numerical agreement');
} finally {
  await browser.close();
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({baseUrl, reports, errors}, null, 2) + '\n');
}
console.log(`Verified ${reports.length} survey-frame scenarios against independent references: ${outputDirectory}`);
