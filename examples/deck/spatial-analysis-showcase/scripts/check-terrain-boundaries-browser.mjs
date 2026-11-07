import {captureBuildEvidence} from './earth-build-evidence.mjs';
import {installReadoutIdentifiers} from './earth-browser-readouts.mjs';
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const oracle = JSON.parse(
  await readFile(
    new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/earth-terrain-boundaries-oracle.json', import.meta.url),
    'utf8'
  )
);
const expectedPeak = oracle.prominence.largestFinitePeak;
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory = process.argv[3] ?? '/tmp/luma-showcase-overhaul-20261025/terrain-boundaries-numerics';
await mkdir(outputDirectory, {recursive: true});
const verifyBuildEvidence = await captureBuildEvidence(baseUrl, outputDirectory);

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const errors = [];
const results = [];
page.on('pageerror', error => errors.push(String(error)));
page.on('console', message => {
  if (message.type() === 'error') errors.push(message.text());
});
const readout = identifier => page.locator(`[data-readout="${identifier}"]`).first();
const selectOption = (identifier, value) =>
  page.locator(`#panel-controls select[id$="option-${identifier}"]`).selectOption(String(value));
const closeTo = (actual, expected, tolerance, label) => {
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance,
    `${label}: ${actual}, expected ${expected} ± ${tolerance}`);
};
const parseInteger = text => Number(text.replaceAll(',', ''));

try {
  await page.goto(`${baseUrl.replace(/\/$/, '')}/#/story/terrain-boundaries/closed-window`, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(
    () => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error,
    null,
    {timeout: 180000}
  );
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null);
  await installReadoutIdentifiers(page, 'terrain-boundaries', 'hydrology');
  await page.waitForFunction(
    () => document.querySelector('[data-readout="selectedPeak"]')?.textContent?.includes('finite prominence'),
    null,
    {timeout: 180000}
  );

  const peakText = await readout('selectedPeak').textContent();
  const peakMatch = peakText.match(
    /row (\d+), column (\d+) · ([\d.]+) m peak · ([\d.]+) m key saddle · ([\d.]+) m finite prominence · ([\d,]+) finite peaks · ([\d,]+) infinite component elders · elder link row (\d+), column (\d+)/
  );
  assert.ok(peakMatch, `Unexpected finite-window peak readout: ${peakText}`);
  const observedPeak = {
    row: Number(peakMatch[1]),
    column: Number(peakMatch[2]),
    elevation: Number(peakMatch[3]),
    keySaddle: Number(peakMatch[4]),
    prominence: Number(peakMatch[5]),
    finitePeakCount: parseInteger(peakMatch[6]),
    elderCount: parseInteger(peakMatch[7]),
    parentRow: Number(peakMatch[8]),
    parentColumn: Number(peakMatch[9])
  };
  assert.equal(observedPeak.row, expectedPeak.row);
  assert.equal(observedPeak.column, expectedPeak.column);
  // The UI formats these readouts to 2 decimal places; allow half a centi-metre plus float margin.
  closeTo(observedPeak.elevation, expectedPeak.elevation, 0.0051, 'peak elevation');
  closeTo(observedPeak.keySaddle, expectedPeak.keySaddle, 0.0051, 'key saddle elevation');
  closeTo(observedPeak.prominence, expectedPeak.prominence, 0.0051, 'finite prominence');
  assert.equal(observedPeak.finitePeakCount, oracle.prominence.finitePeakCount);
  assert.equal(observedPeak.elderCount, oracle.prominence.infiniteComponentElderCount);
  assert.equal(observedPeak.parentRow, expectedPeak.parentRow);
  assert.equal(observedPeak.parentColumn, expectedPeak.parentColumn);
  results.push({kind: 'finite-window-prominence', expectedPeak, observedPeak, peakText});

  // Exercise all tile DAG choices against their matching whole-window reference.
  await page.evaluate(() => { window.location.hash = '#/story/terrain-boundaries/owned-tiles'; });
  await page.waitForFunction(
    () => document.querySelector('[data-readout="tiledFlow"]')?.textContent?.includes('D8 recipe'),
    null,
    {timeout: 180000}
  );
  await page.locator('#tab-controls').click();
  assert.equal(await page.locator('#panel-controls select[id$="option-iterations"]').inputValue(), '128');
  const runTiledCase = async ({routing, branchMethod, expectedLabel}) => {
    await selectOption('routing', routing);
    if (branchMethod) await selectOption('branchMethod', branchMethod);
    await page.waitForFunction(
      label => document.querySelector('[data-readout="tiledFlow"]')?.textContent?.includes(label) &&
        document.querySelector('[data-readout="tiledFlow"]')?.textContent?.includes('maximum accumulation difference'),
      expectedLabel,
      {timeout: 180000}
    );
    const text = await readout('tiledFlow').textContent();
    const match = text.match(
      /([\d,]+) owned cells resolved · ([\d,]+) unresolved · ([\d,]+) invalid · ([\d,]+) (?:D8 edges|branch links) cross a quadrant seam · .*? · ([\d,]+) cells compared · (?:maximum accumulation difference ([\d.e+-]+) cells|difference unavailable without shared resolved cells)/
    );
    assert.ok(match, `Unexpected tile-routing readout: ${text}`);
    const observed = {
      resolved: parseInteger(match[1]),
      unresolved: parseInteger(match[2]),
      invalid: parseInteger(match[3]),
      seamLinks: parseInteger(match[4]),
      compared: parseInteger(match[5]),
      maximumDifference: match[6] === undefined ? null : Number(match[6])
    };
    assert.equal(observed.resolved + observed.unresolved + observed.invalid, 64 * 64,
      'Every tile-owned cell must be accounted for exactly once');
    assert.ok(observed.seamLinks > 0, 'The selected routing should cross at least one owned-tile seam');
    assert.ok(observed.compared > 0 && observed.compared <= observed.resolved,
      'Comparison uses only resolved tile cells');
    if (observed.compared > 0) {
      assert.ok(Number.isFinite(observed.maximumDifference));
      closeTo(observed.maximumDifference, 0, 0.05, `${routing}/${branchMethod ?? 'default'} whole-window agreement`);
    } else {
      assert.equal(observed.maximumDifference, null, 'Unavailable comparison must not be displayed as zero');
    }
    results.push({kind: 'tiled-flow', routing, branchMethod, observed, text});
  };

  await runTiledCase({routing: 'd8', expectedLabel: 'D8 recipe'});
  await runTiledCase({routing: 'branching', branchMethod: 'd-infinity', expectedLabel: 'd-infinity whole-window reference'});
  await runTiledCase({routing: 'branching', branchMethod: 'mfd-freeman', expectedLabel: 'mfd-freeman whole-window reference'});

  assert.deepEqual(errors, [], 'Browser or GPU console errors');
  await page.screenshot({path: `${outputDirectory}/owned-tiles.png`, fullPage: true});
  await page.evaluate(() => { window.location.hash = '#/story/terrain-boundaries/drainage-recipe'; });
  await page.waitForFunction(() => document.body.textContent?.includes('Strahler stream order'));
  await page.locator('#tab-story').click();
  await page.screenshot({path: `${outputDirectory}/stream-order.png`, fullPage: true});
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({passed: true, results, errors}, null, 2));
  console.log(JSON.stringify({passed: true, checks: results.length, errors}, null, 2));
} catch (error) {
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({passed: false, error: String(error), results, errors}, null, 2));
  throw error;
} finally {
  await browser.close(); await verifyBuildEvidence();
}
