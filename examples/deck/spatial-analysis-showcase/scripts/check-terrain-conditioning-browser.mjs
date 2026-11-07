// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

// Independent source oracle: decode the shipped Terrarium RGB, average 32 × 32 blocks,
// then directly sum cells below each absolute stage with the declared centre-row area.
// These are not values copied from the contributor under test. See earth-systems.md.
const expectedStages = [
  {index: 0, elevation: 759.51904296875, area: 0.244118453125, volume: 0},
  {index: 3, elevation: 1359.51904296875, area: 246.315519203125, volume: 56224.89260100251},
  {index: 8, elevation: 2359.51904296875, area: 883.464681859375, volume: 600954.0748228274}
];
const expectedWindowArea = 999.909184;
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory = process.argv[3] ?? '/tmp/luma-showcase-overhaul-20261025/terrain-numerics';
await mkdir(outputDirectory, {recursive: true});
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const errors = [];
const results = [];
page.on('pageerror', error => errors.push(String(error)));
page.on('console', message => {
  if (message.type() === 'error') errors.push(message.text());
});
const closeTo = (actual, expected, tolerance, label) => {
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance,
    `${label}: ${actual}, expected ${expected} ± ${tolerance}`);
};
const getReadout = identifier => page.locator(`[data-readout="${identifier}"]`).first();
try {
  await page.goto(`${baseUrl.replace(/\/$/, '')}/#/story/terrain-conditioning/hierarchy`, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error, null, {timeout: 180000});
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null);
  await page.waitForFunction(() => document.querySelector('[data-readout="hierarchyStatus"]')?.textContent?.includes('root area'), null, {timeout: 180000});
  for (const expected of expectedStages) {
    await page.locator('input[type="range"][id$="option-stage"]').first().evaluate((input, value) => {
      input.value = String(value);
      input.dispatchEvent(new Event('input', {bubbles: true}));
    }, expected.index);
    await page.waitForFunction(stage => document.querySelector('[data-readout="hierarchyStatus"]')?.textContent?.includes(`stage ${stage.toFixed(3)} m`), expected.elevation);
    const text = await getReadout('hierarchyStatus').textContent();
    const match = text.match(/stage ([\d.]+) m · root area ([\d.]+) km² · volume ([\d.]+) million m³/);
    assert.ok(match, `Unexpected hierarchy readout: ${text}`);
    const observed = {index: expected.index, elevation: Number(match[1]), area: Number(match[2]), volume: Number(match[3])};
    closeTo(observed.elevation, expected.elevation, 0.001, 'stage elevation');
    closeTo(observed.area, expected.area, 0.003, 'sublevel area km²');
    closeTo(observed.volume, expected.volume, Math.max(0.003, expected.volume * 0.000002), 'sublevel volume million m³');
    results.push({expected, observed, text});
  }
  await page.evaluate(() => {window.location.hash = '#/story/terrain-conditioning/profiles';});
  await page.waitForFunction(() => document.querySelector('[data-readout="catchmentSummary"]')?.textContent?.includes('final hypsometry bin'));
  const summary = await getReadout('catchmentSummary').textContent();
  const match = summary.match(/window area ([\d.]+) km² · final hypsometry bin ([\d.]+) km²/);
  assert.ok(match, `Unexpected catchment readout: ${summary}`);
  closeTo(Number(match[1]), expectedWindowArea, 0.003, 'window area km²');
  closeTo(Number(match[2]), expectedWindowArea, 0.003, 'hypsometry endpoint km²');
  results.push({expectedWindowArea, summary});
  assert.deepEqual(errors, [], 'Browser or GPU console errors');
  await page.screenshot({path: `${outputDirectory}/hypsometry.png`, fullPage: true});
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({passed: true, results, errors}, null, 2));
  console.log(JSON.stringify({passed: true, results, errors}, null, 2));
} catch (error) {
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({passed: false, error: String(error), results, errors}, null, 2));
  throw error;
} finally {
  await browser.close();
}
