import {captureBuildEvidence} from './earth-build-evidence.mjs';
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {installReadoutIdentifiers} from './earth-browser-readouts.mjs';

const oracle = JSON.parse(
  await readFile(
    new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/earth-spectral-regions-oracle.json', import.meta.url),
    'utf8'
  )
);
const zones = oracle.zones.values;
const thresholds = Object.keys(zones[0].paired.thresholds);
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory = process.argv[3] ?? '/tmp/luma-showcase-overhaul-20261025/spectral-regions-numerics';
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

const selectControl = (identifier, value) =>
  page.locator(`#panel-controls select[id$="option-${identifier}"]`).selectOption(String(value));
const clickTab = tab => page.locator(`#tab-${tab}`).click();
const evidence = () =>
  page.locator('#panel-story .step-detail:not([hidden]) .step-readouts dd[data-readout="regionalEvidence"]').first();
const activeChartTable = () =>
  page.locator('#panel-story .step-detail:not([hidden]) .step-readouts .readout-chart-row .chart-box table').first();
const closeTo = (actual, expected, tolerance, label) => {
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance,
    `${label}: ${actual}, expected ${expected} ± ${tolerance}`);
};
const parseCount = text => Number(text.replaceAll(',', ''));
const formatChartNumber = value => String(Number(value.toFixed(3)));

const getStatusReadout = async (zone, threshold) => {
  await page.waitForFunction(
    ({label, cutoff}) => {
      const activeStep = document.querySelector('#panel-story .step-detail:not([hidden])');
      const text = activeStep?.querySelector('[data-readout="regionalEvidence"]')?.textContent ?? '';
      return text.includes(`${label} · 1,024 assigned whole cells`) &&
        text.includes(`cutoff ${Number(cutoff).toFixed(2)} · stage flags `) &&
        text.includes(', invalid cutoff 0');
    },
    {label: zone.label, cutoff: threshold},
    {timeout: 180000}
  );
  const text = await evidence().textContent();
  assert.ok(text, 'Active story card has no regional evidence text');
  return text;
};

function verifyEvidence(text, zone, threshold) {
  const pattern =
    /^([^·]+) · 1,024 assigned whole cells \/ ([\d.]+) km² · Jul support ([\d,]+) QA-retained rows \/ ([\d,]+) finite NIR \/ ([\d,]+) masked-out cells · median ([\d.]+), numeric sample mode ([\d.]+), ([\d,]+) unique values · Sep support ([\d,]+) QA-retained rows \/ ([\d,]+) finite NIR \/ ([\d,]+) masked-out cells · median ([\d.]+), numeric sample mode ([\d.]+), ([\d,]+) unique values · paired common support ([\d,]+), mean Δ ([-\d.]+) · valid area ([\d.]+) km² · changed area ([\d.]+) km² · coverage ([\d.]+)% · cutoff ([\d.]+) · stage flags ([\d/]+), invalid cutoff ([\d,]+)/;
  const match = text.match(pattern);
  assert.ok(match, `Unexpected regional evidence text: ${text}`);
  const before = zone.before;
  const after = zone.after;
  const paired = zone.paired;
  assert.equal(match[1].trim(), zone.label);
  assert.equal(match[2], zone.totalAreaKm2.toFixed(4));
  assert.equal(parseCount(match[3]), before.validRows);
  assert.equal(parseCount(match[4]), before.validFiniteValues);
  assert.equal(parseCount(match[5]), before.maskedOutRows);
  assert.equal(match[6], before.median.toFixed(4));
  assert.equal(match[7], before.mode.toFixed(4));
  assert.equal(parseCount(match[8]), before.uniqueValues);
  assert.equal(parseCount(match[9]), after.validRows);
  assert.equal(parseCount(match[10]), after.validFiniteValues);
  assert.equal(parseCount(match[11]), after.maskedOutRows);
  assert.equal(match[12], after.median.toFixed(4));
  assert.equal(match[13], after.mode.toFixed(4));
  assert.equal(parseCount(match[14]), after.uniqueValues);
  assert.equal(parseCount(match[15]), paired.validCells);
  assert.equal(match[16], paired.meanDifference.toFixed(4));
  assert.equal(match[17], paired.validAreaKm2.toFixed(4));
  assert.equal(match[18], paired.thresholds[threshold].changedAreaKm2.toFixed(4));
  assert.equal(match[19], (paired.coverage * 100).toFixed(1));
  assert.equal(Number(match[20]), Number(threshold));
  assert.deepEqual(match[21].split('/').map(Number), oracle.zones.expectedStatus.stages.map(() => 0));
  assert.equal(parseCount(match[22]), 0);
  return {
    zone: zone.label,
    threshold,
    assignedCells: 1024,
    beforeRows: before.validRows,
    afterRows: after.validRows,
    commonCells: paired.validCells,
    meanDifference: Number(match[16]),
    validAreaKm2: Number(match[17]),
    changedAreaKm2: Number(match[18]),
    coveragePercent: Number(match[19]),
    statuses: match[21]
  };
}

async function verifyQuantileTable(zone) {
  const table = activeChartTable();
  const details = table.locator('xpath=ancestor::details[contains(concat(" ", normalize-space(@class), " "), " chart-table ")]');
  if (!(await details.getAttribute('open'))) await details.locator('summary').click();
  await table.waitFor({state: 'visible', timeout: 180000});
  const data = await table.evaluate(element => ({
    headers: Array.from(element.querySelectorAll('thead th'), cell => cell.textContent?.trim() ?? ''),
    rows: Array.from(element.querySelectorAll('tbody tr'), row =>
      Array.from(row.querySelectorAll('td'), cell => cell.textContent?.trim() ?? '')
    )
  }));
  assert.deepEqual(data.headers, ['linear quantile probability', '13 Jul 2021', '21 Sep 2021']);
  assert.equal(data.rows.length, 5, `Expected all five requested quantiles: ${JSON.stringify(data)}`);
  for (let index = 0; index < oracle.zones.probabilities.length; index++) {
    const probability = oracle.zones.probabilities[index];
    assert.equal(data.rows[index][0], formatChartNumber(probability));
    assert.equal(data.rows[index][1], formatChartNumber(zone.before.quantiles[index]));
    assert.equal(data.rows[index][2], formatChartNumber(zone.after.quantiles[index]));
  }
  return data;
}

let diagnostics = {};
try {
  await page.goto(`${baseUrl.replace(/\/$/, '')}/#/story/spectral-evidence/residual-limits`, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(
    () => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error,
    null,
    {timeout: 180000}
  );
  await installReadoutIdentifiers(page, 'spectral-evidence');
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null);
  assert.equal(await page.locator('#tab-controls').getAttribute('aria-selected'), 'false');

  // All interaction is made through the Controls tab. Read assertions from the active, visible
  // residual-limits story card after returning to Story; the collapsed “All readouts” copy is unused.
  await clickTab('controls');
  await selectControl('regionView', 'distribution');
  await selectControl('changeThreshold', '0.05');
  for (const zone of zones) {
    await selectControl('selectedRegion', zone.id);
    await clickTab('story');
    const readyText = await getStatusReadout(zone, '0.05');
    const evidenceCheck = verifyEvidence(readyText, zone, '0.05');
    const quantiles = await verifyQuantileTable(zone);
    results.push({kind: 'zone-distributions', ...evidenceCheck, quantiles});
    console.log(`PASS regional GPU distributions: ${zone.label} · Jul/Sep medians, modes, support and quantiles`);
    await clickTab('controls');
    await selectControl('regionView', 'distribution');
  }
  await clickTab('story');

  for (const zone of zones) {
    for (const threshold of thresholds) {
      await clickTab('controls');
      await selectControl('regionView', 'change');
      await selectControl('selectedRegion', zone.id);
      await selectControl('changeThreshold', Number(threshold).toFixed(2));
      await clickTab('story');
      const text = await getStatusReadout(zone, threshold);
      const check = verifyEvidence(text, zone, threshold);
      results.push({kind: 'regional-change', ...check, text});
      console.log(`PASS regional change: ${zone.label} · cutoff ${threshold}`);
    }
  }

  assert.equal(oracle.validity.beforeValid, zones.reduce((sum, zone) => sum + zone.before.validRows, 0));
  assert.equal(oracle.validity.afterValid, zones.reduce((sum, zone) => sum + zone.after.validRows, 0));
  assert.equal(oracle.validity.commonValid, zones.reduce((sum, zone) => sum + zone.paired.validCells, 0));
  assert.deepEqual(errors, [], 'Browser or GPU console errors');
  await page.screenshot({path: `${outputDirectory}/regional-change.png`, fullPage: true});
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({passed: true, results, errors}, null, 2));
  console.log(JSON.stringify({passed: true, checks: results.length, errors}, null, 2));
} catch (error) {
  diagnostics = await page.evaluate(() => ({
    url: window.location.href,
    activeScene: globalThis.spatialAnalysisShowcase?.sceneId,
    ready: globalThis.spatialAnalysisShowcase?.ready,
    appError: globalThis.spatialAnalysisShowcase?.error,
    bodyText: document.body.innerText.slice(0, 12000),
    evidence: Array.from(document.querySelectorAll('#panel-story .step-detail:not([hidden]) [data-readout="regionalEvidence"]'), element => element.textContent),
    controls: Array.from(document.querySelectorAll('#panel-controls select'), element => [element.id, element.value])
  }));
  await page.screenshot({path: `${outputDirectory}/failure.png`, fullPage: true}).catch(() => {});
  await writeFile(
    `${outputDirectory}/report.json`,
    JSON.stringify({passed: false, error: String(error), results, errors, diagnostics}, null, 2)
  );
  throw error;
} finally {
  await browser.close(); await verifyBuildEvidence();
}
