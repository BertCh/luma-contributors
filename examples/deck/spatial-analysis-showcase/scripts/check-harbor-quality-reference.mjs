// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/harbor-quality-reference.json';
const reference = JSON.parse(await readFile(new URL(
  '../../../../dev-docs/research/showcase-overhaul-2026-10-25/movement-network-reference.json',
  import.meta.url
), 'utf8')).ais.selectedTrack;
const polygonReference = JSON.parse(await readFile(new URL(
  '../../../../dev-docs/research/showcase-overhaul-2026-10-25/harbor-polygon-support-reference.json',
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
  url.hash = '#/story/harbor-track-quality';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error, null, {timeout: 90_000});
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null, 'Scene must compile before numerical checks');
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  for (const expected of reference.qualityScenarios) {
    for (const [id, value] of [
      ['maximumGapSeconds', expected.maximumGapSeconds],
      ['maximumSpeedKnots', expected.maximumSpeedKnots],
      ['maximumModelGapSeconds', expected.maximumGapSeconds]
    ]) {
      await page.locator(`#all-option-${id}`).evaluate((input, nextValue) => {
        input.value = String(nextValue);
        input.dispatchEvent(new Event('input', {bubbles: true}));
        input.dispatchEvent(new Event('change', {bubbles: true}));
      }, value);
    }
    await page.waitForFunction(expectedValues => {
      const read = label => [...document.querySelectorAll('.readout')]
        .find(row => row.querySelector('dt')?.textContent === label)?.querySelector('dd')?.textContent ?? '';
      return read('Raw threshold review') === `${expectedValues.flaggedObservations} flagged observations · ${expectedValues.gapEndpoints} long-gap endpoints` &&
        Number.parseInt(read('Filter starts'), 10) === expectedValues.filterResetsAtSameModelGap;
    }, expected, {timeout: 30_000});
    const observed = await page.evaluate(() => Object.fromEntries(
      [...document.querySelectorAll('.readout')].map(row => [
        row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''
      ])
    ));
    assert.equal(observed['Epoch precision audit'], '265/265 distinct relative times · naïve epoch float32 237/265, max error 64 s');
    assert.ok(Math.abs(Number.parseFloat(observed['Observed path length']) * 1000 - reference.planarLengthMeters) <= 51);
    assert.ok(Math.abs(Number.parseFloat(observed['Maximum step speed']) - reference.maximumDerivedKnots) <= 0.051);
    assert.ok(Math.abs(Number.parseFloat(observed['Observed span']) * 3600 - reference.durationSeconds) <= 181);
    reports.push({expected, observed});
  }
  async function slider(id, value) {
    await page.locator(`#all-option-${id}`).evaluate((input, nextValue) => {
      input.value = String(nextValue);
      input.dispatchEvent(new Event('input', {bubbles: true}));
      input.dispatchEvent(new Event('change', {bubbles: true}));
    }, value);
  }
  const readValues = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''])));
  await page.locator('#all-option-covarianceModel').selectOption('full');
  for (const [ratio, correlation] of [[1, 0], [4, .8], [4, -.8]]) {
    await slider('axisRatio', ratio);
    await slider('axisCorrelation', correlation);
    for (const display of ['filtered', 'smoothed']) {
      await page.locator('#all-option-display').selectOption(display);
      await page.waitForFunction(expected => {
        const value = [...document.querySelectorAll('.readout')].find(row => row.querySelector('dt')?.textContent === 'Model-implied position uncertainty')?.querySelector('dd')?.textContent ?? '';
        return value.includes(`ρ ${expected.correlation.toFixed(1)}, axis ratio ${expected.ratio.toFixed(1)}`);
      }, {ratio, correlation}, {timeout: 30_000});
      const observed = await readValues();
      const axes = observed['Model-implied position uncertainty'].match(/axes ([\d.]+) \/ ([\d.]+) m/);
      assert.ok(axes, 'Covariance must produce finite displayed axes');
      assert.ok(Number(axes[1]) >= Number(axes[2]) && Number(axes[2]) > 0);
      if (ratio === 1 && correlation === 0) {
        const difference = observed['Isotropic/full covariance parity'].match(/filter ([\d.]+) m · smoother ([\d.]+) m/);
        assert.ok(difference, 'Parity must compare valid states');
        assert.ok(Number(difference[1]) < .1 && Number(difference[2]) < .1, `Unexpected isotropic/full mismatch: ${difference[0]}`);
        assert.equal(axes[1], axes[2], 'Isotropic covariance must have equal axes');
      } else assert.ok(Number(axes[1]) > Number(axes[2]), 'Correlated anisotropic covariance must expose unequal axes');
      assert.equal(observed['Selected model status'], '265 valid · 0 invalid states · 0 singular RTS steps');
      assert.equal(observed['Raw threshold review'], '0 flagged observations · 0 long-gap endpoints');
      reports.push({covarianceModel: 'full', ratio, correlation, display, observed});
    }
  }
  await slider('maximumModelGapSeconds', polygonReference.assumptions.maximumGapSeconds);
  await slider('windowEndHour', polygonReference.assumptions.windowEndHourUtc);
  await page.locator('#all-option-display').selectOption('zone');
  await page.waitForFunction(expected => {
    const read = label => [...document.querySelectorAll('.readout')]
      .find(row => row.querySelector('dt')?.textContent === label)?.querySelector('dd')?.textContent ?? '';
    return read('Polygon clip and unknown gaps').startsWith(`${expected.mergedIntervalsWritten} merged intervals · ${expected.unknownGapRows} source rows unknown across gaps · 0 invalid`) &&
      read('GPU interval-union status') === `${expected.mergedIntervalsRequired} required · ${expected.mergedIntervalsWritten} written · ${expected.compactOverflow} overflow`;
  }, polygonReference.counts, {timeout: 30_000});
  const polygonObserved = await readValues();
  assert.equal(
    polygonObserved['Polygon clip and unknown gaps'],
    `${polygonReference.counts.mergedIntervalsWritten} merged intervals · ${polygonReference.counts.unknownGapRows} source rows unknown across gaps · 0 invalid`
  );
  assert.equal(
    polygonObserved['GPU interval-union status'],
    `${polygonReference.counts.mergedIntervalsRequired} required · ${polygonReference.counts.mergedIntervalsWritten} written · ${polygonReference.counts.compactOverflow} overflow`
  );
  reports.push({display: 'zone', polygonSupport: polygonReference.counts, observed: polygonObserved});
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
console.log(`Verified ${reports.length} AIS threshold scenarios: ${outputFile}`);
