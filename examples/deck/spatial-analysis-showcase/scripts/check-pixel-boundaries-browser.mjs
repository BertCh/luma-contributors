import {captureBuildEvidence} from './earth-build-evidence.mjs';
import {installReadoutIdentifiers} from './earth-browser-readouts.mjs';
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

// Independent source sums and polygon-cell clipping are recorded in earth-source-oracles.json.
const expected = {
  before: {integral: 13132541.86951518, area: 58.4856},
  after: {integral: 3626581.951533258, area: 34.2576}
};
const sourceOracles = JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/earth-source-oracles.json', import.meta.url), 'utf8'));
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory = process.argv[3] ?? '/tmp/luma-showcase-overhaul-20261025/pixel-boundary-numerics';
await mkdir(outputDirectory, {recursive: true});
const verifyBuildEvidence = await captureBuildEvidence(baseUrl, outputDirectory);
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const errors = [];
const results = [];
page.on('pageerror', error => errors.push(String(error)));
page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
const readout = identifier => page.locator(`[data-readout="${identifier}"]`).first();
const select = (identifier, value) => page.locator(`#panel-controls select[id$="option-${identifier}"]`).selectOption(value);
const closeTo = (actual, reference, tolerance, label) => assert.ok(Number.isFinite(actual) && Math.abs(actual - reference) <= tolerance, `${label}: ${actual}, expected ${reference} ± ${tolerance}`);
try {
  await page.goto(`${baseUrl.replace(/\/$/, '')}/#/story/pixel-boundaries/weighted-values`, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error, null, {timeout: 180000});
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null);
 await installReadoutIdentifiers(page, 'pixel-boundaries');
  await page.locator('#tab-controls').click();
  await select('quantileMethod', 'inverse-cdf');
  for (const reference of sourceOracles.pixelBoundaries.records.filter(record => record.date === 'after')) {
    await select('selectedZone', String(reference.zone));
    const zoneName = ['Rectangle', 'Oblique transect', 'Interior hole', 'Overlapping query'][reference.zone];
    await page.waitForFunction(name => document.querySelector('[data-readout="zoneSummary"]')?.textContent?.includes(`${name} · mean`), zoneName, {timeout: 180000});
    const zoneSummary = await readout('zoneSummary').textContent();
    const zone = zoneSummary.match(/mean ([\d.]+) · population variance ([\d.]+) · sample variance ([\d.]+) · effective support ([\d.]+) cells · valid area ([\d.]+) km² · geometric area ([\d.]+) km²/);
    assert.ok(zone, zoneSummary);
    closeTo(Number(zone[1]), reference.mean, 0.001, 'September weighted mean');
    closeTo(Number(zone[2]), reference.populationVariance, 0.00001, 'Population variance');
    closeTo(Number(zone[3]), reference.sampleVariance, 0.00001, 'Sample variance');
    closeTo(Number(zone[4]), reference.effectiveSampleSize, 0.11, 'Effective weighted support');
    closeTo(Number(zone[5]), reference.validAreaM2 / 1e6, 0.00001, 'Valid area km²');
    closeTo(Number(zone[6]), reference.coveredAreaM2 / 1e6, 0.00001, 'Geometric area km²');
    results.push({zone: reference.zone, zoneSummary});
  }
  if (process.argv.includes('--quantity')) {
    for (const reference of sourceOracles.pixelBoundaries.records) {
      await select('sourceDate', reference.date);
      await select('selectedZone', String(reference.zone));
      const text = await readout('quantityContract').textContent();
      const match = text.match(/intensive integral ([\d.]+) · extensive sum ([\d.]+) reflectance·m² · difference (-?[\d.]+) · area-weighted mean ([\d.]+) · ([\d.]+) km² observed/);
      assert.ok(match, text);
      const integral = reference.mean * reference.validAreaM2;
      closeTo(Number(match[1]), integral, 5, 'Intensive integral');
      closeTo(Number(match[2]), integral, 5, 'Extensive cell-integral sum');
      closeTo(Number(match[3]), 0, 1, 'Quantity contract equivalence');
      closeTo(Number(match[4]), reference.mean, 0.00001, 'Quantity mean');
      const rounding = sourceOracles.pixelQuantityRounding.records.find(record => record.date === reference.date && record.zone === reference.zone);
      // This contributor sums fractions sequentially in f32, unlike the distribution reducer.
      // Bound partial additions plus exact-one binade crossings, input/output and display rounding.
      closeTo(Number(match[5]) * 1e6, reference.validAreaM2, rounding.conservativeAreaToleranceM2, 'Quantity observed area');
      results.push({quantity: true, date: reference.date, zone: reference.zone, rounding, text});
    }
  }
  if (process.argv.includes('--paired')) {
    await select('observationSupport', 'paired-dates');
    for (const reference of sourceOracles.pixelPairedSupport.records) {
      await select('sourceDate', reference.date);
      await select('selectedZone', String(reference.zone));
      await page.waitForFunction(expectedArea => {
        const text = document.querySelector('[data-readout="zoneSummary"]')?.textContent ?? '';
        const match = text.match(/paired-date statistical support ([\d.]+) km²/);
        return match && Math.abs(Number(match[1]) - expectedArea) < 0.00001;
      }, reference.supportAreaM2 / 1e6, {timeout: 180000});
      const text = await readout('zoneSummary').textContent();
      const values = text.match(/mean ([\d.]+) · population variance ([\d.]+) · sample variance ([\d.]+) · effective support ([\d.]+)/);
      assert.ok(values, text);
      closeTo(Number(values[1]), reference.mean, 0.001, 'Paired mean');
      closeTo(Number(values[2]), reference.populationVariance, 0.00001, 'Paired population variance');
      closeTo(Number(values[3]), reference.sampleVariance, 0.00001, 'Paired sample variance');
      closeTo(Number(values[4]), reference.effectiveSampleSize, 0.11, 'Paired effective support');
      const support = text.match(/paired-date statistical support ([\d.]+) km²/);
      assert.ok(support, text);
      closeTo(Number(support[1]), reference.supportAreaM2 / 1e6, 0.00001, 'Paired support area km²');
      results.push({paired: true, date: reference.date, zone: reference.zone, text});
    }
    await select('observationSupport', 'selected-date');
  }
  await page.evaluate(() => {window.location.hash = '#/story/pixel-boundaries/analyst-queries';});
  await page.waitForFunction(() => document.querySelector('[data-readout="coverage"]')?.textContent?.includes('raw GPU coverage tuples'));
  const coverage = await readout('coverage').textContent();
  // Raw tuple counts vary with float32 boundary slivers. One square metre is below
  // the independently computed smallest genuine overlap (5.042 m²) for these queries.
  assert.match(coverage, /7,228 overlaps above 1 m²/);
  assert.match(coverage, /4\/4 analyst queries valid/);
  results.push({coverage, expectedMeaningfulIntersections: 7228, diagnosticAreaToleranceM2: 1});
  await page.evaluate(() => {window.location.hash = '#/story/pixel-boundaries/new-grid';});
  await page.locator('#tab-controls').click();
  for (const date of ['after', 'before']) {
    await select('sourceDate', date);
    for (const phase of ['0', '120']) {
      await select('gridPhase', phase);
      for (const normalization of ['destination-area', 'covered-area', 'extensive']) {
        await select('normalization', normalization);
        await page.waitForFunction(({phase, normalization}) => {
          const text = document.querySelector('[data-readout="remapping"]')?.textContent ?? '';
          return text.includes(`phase ${phase} m`) && text.includes(`· ${normalization} ·`) && text.includes('reflectance·m²');
        }, {phase, normalization});
        const text = await readout('remapping').textContent();
        const match = text.match(/([\d.]+) km² valid area · ([\d.]+) reflectance·m²/);
        assert.ok(match, text);
        closeTo(Number(match[1]), expected[date].area, 0.002, `${date} valid area`);
        closeTo(Number(match[2]), expected[date].integral, 10, `${date} ${phase} ${normalization} integral`);
        results.push({date, phase, normalization, text});
      }
    }
  }
  assert.deepEqual(errors, [], 'Browser or GPU errors');
  await page.screenshot({path: `${outputDirectory}/remapping.png`, fullPage: true});
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({passed: true, results, errors}, null, 2));
  console.log(JSON.stringify({passed: true, results, errors}, null, 2));
} catch (error) {
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({passed: false, error: String(error), results, errors}, null, 2));
  throw error;
} finally {
  await browser.close(); await verifyBuildEvidence();
}
