// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/retail-network-reference.json';
const reference = JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/retail-network-intensity-reference.json', import.meta.url), 'utf8'));
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const reports = [], consoleErrors = [];
let page, passed = false, failureReason = null;
try {
  page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => consoleErrors.push(error.message));
  page.on('console', message => {if (message.type() === 'error') consoleErrors.push(message.text());});
  const url = new URL(baseUrl); url.hash = '#/story/retail-network-intensity';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error, null, {timeout: 120_000});
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null, 'Scene must compile before numerical checks');
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  const readValues = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''])));
  async function awaitResult() {
    await page.waitForFunction(() => {
      const text = [...document.querySelectorAll('.readout')].find(row => row.querySelector('dt')?.textContent === 'Complete GPU result status')?.querySelector('dd')?.textContent ?? '';
      return text.includes('all K radii valid') || text.includes('K withheld') || text.includes('failed') || text.includes('did not converge');
    }, null, {timeout: 60_000});
  }
  for (const category of ['retail', 'restaurant_cafe', 'grocery', 'combined']) {
    await page.locator('#all-option-category').selectOption(category);
    for (const bandwidth of [2000, 600]) {
      await page.locator('#all-option-bandwidth').selectOption(String(bandwidth));
      await awaitResult();
      const observed = await readValues(), expected = reference.categories[category].bandwidths[bandwidth];
      assert.equal(Number.parseInt(observed['Estimated event intensity'], 10), expected.positiveLeaveOneOutCount);
      assert.ok(Math.abs(Number.parseFloat(observed['Integrated network intensity']) - reference.sample.eventCount) <= .1);
      const median = Number(observed['Estimated event intensity'].match(/median ([\d.]+)/)?.[1]);
      assert.ok(Math.abs(median - expected.medianPositivePerMeter * 1000) <= .011, `${category}/${bandwidth} LOO median`);
      if (expected.zeroLeaveOneOutCount) assert.equal(observed['Inhomogeneous K at radius'], 'K curve withheld');
      else assert.match(observed['Complete GPU result status'], /all K radii valid/);
      reports.push({category, bandwidth, observed});
    }
  }
  await page.locator('#all-option-category').selectOption('retail');
  await page.locator('#all-option-bandwidth').selectOption('2000');
  for (const correction of ['none', 'ang']) for (const normalization of ['network-length', 'inverse-intensity', 'inverse-intensity-squared']) {
    await page.locator('#all-option-correction').selectOption(correction);
    await page.locator('#all-option-normalization').selectOption(normalization);
    await awaitResult();
    const observed = await readValues();
    assert.match(observed['Complete GPU result status'], /all K radii valid/);
    if (correction === 'none') {
      const expected = reference.categories.retail.bandwidths[2000].uncorrectedK[normalization] / 1000;
      const actual = Number.parseFloat(observed['Inhomogeneous K at radius']);
      assert.ok(Math.abs(actual - expected) <= Math.max(.002, expected * .002), `${normalization}: ${actual} vs ${expected}`);
    }
    assert.equal(observed['Inhomogeneous K at radius'].includes('identity'), correction === 'ang' && normalization === 'network-length');
    reports.push({correction, normalization, observed, independentEndpointComparison: correction === 'none'});
  }
  assert.deepEqual(consoleErrors, []);
  passed = true;
} catch (error) {
  failureReason = String(error);
  if (page) {
    await page.screenshot({path: outputFile.replace(/\.json$/, '-failure.png'), fullPage: true});
    reports.push({failure: failureReason, readouts: await page.locator('.readout').allTextContents(), visibleText: await page.locator('body').innerText()});
  }
  throw error;
} finally {
  await browser.close();
  await writeFile(outputFile, JSON.stringify({baseUrl, passed, status: passed ? 'passed' : 'failed', failureReason, reports, consoleErrors}, null, 2) + '\n');
}
console.log(`Verified ${reports.length} network-density configurations: ${outputFile}`);
