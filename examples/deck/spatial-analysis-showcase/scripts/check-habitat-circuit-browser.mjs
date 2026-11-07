// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {captureBuildEvidence} from './earth-build-evidence.mjs';
import {installReadoutIdentifiers} from './earth-browser-readouts.mjs';

const oracle = JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/earth-source-oracles.json', import.meta.url), 'utf8')).habitatCircuit;
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory = process.argv[3] ?? '/tmp/luma-showcase-overhaul-20261025/circuit-numerics';
await mkdir(outputDirectory, {recursive: true});
const verifyBuildEvidence = await captureBuildEvidence(baseUrl, outputDirectory);
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const results = [];
const errors = [];
let recovery = null;
page.on('pageerror', error => errors.push(String(error)));
page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
const select = (identifier, value) => page.locator(`#panel-controls select[id$="option-${identifier}"]`).selectOption(String(value));
const readout = identifier => page.locator(`[data-readout="${identifier}"]`).first().textContent();
const ready = () => page.waitForFunction(() => {
  const text = document.querySelector('[data-readout="circuitStatus"]')?.textContent;
  return text?.includes('effective resistance') || text?.includes('not converged · configured') || text?.includes('unreachable');
}, null, {timeout: 180000});
const closeTo = (actual, expected, tolerance, label) => assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance, `${label}: ${actual}, expected ${expected} ± ${tolerance}`);
const setChecked = async (locator, expected) => {
  if (await locator.isChecked() !== expected) {
    await locator.focus();
    await locator.press('Space');
  }
  assert.equal(await locator.isChecked(), expected);
};
const saveReport = async (passed, error) => writeFile(`${outputDirectory}/report.json`, JSON.stringify({
  passed, error, results, recovery, errors,
  acceptedNumericalCases: results.filter(result => result.accepted).length,
  withheldCases: results.filter(result => !result.accepted).length,
  note: 'Withholding checks validate publication policy, not numerical agreement for rejected solutions. The separate 16 × 16 oracle is not a certificate for the 64 × 64 solve.'
}, null, 2));

// Exercise public display controls. The terminal overlay has fixed opacity; only the
// computed potential/current layer responds. A withheld map must remain identical
// across both opacity extremes and both products. Accepted maps provide a positive control.
async function inspectMapPublication(accepted, caseName) {
  const opacity = page.locator('#panel-controls input[type="range"][id$="option-opacity"]');
  const canvas = page.locator('canvas:not(.maplibregl-canvas)');
  assert.equal(await canvas.count(), 1, 'Exactly one Deck analysis canvas is expected');
  const images = [];
  for (const product of ['potential', 'current']) {
    await select('circuitLayer', product);
    for (const endpoint of ['Home', 'End']) {
      await opacity.focus();
      await opacity.press(endpoint);
      assert.equal(await opacity.inputValue(), endpoint === 'Home' ? '0.2' : '1');
      await page.waitForTimeout(250);
      const path = `${outputDirectory}/${caseName}-${product}-${endpoint}.png`;
      const pixels = await canvas.screenshot({
        path, animations: 'disabled',
        // Isolate rendered analysis pixels from HTML legends and MapLibre furniture.
        style: 'body * { visibility: hidden !important; } canvas:not(.maplibregl-canvas) { visibility: visible !important; }'
      });
      images.push({product, endpoint, path, hash: createHash('sha256').update(pixels).digest('hex')});
    }
  }
  if (accepted) {
    assert.notEqual(images[0].hash, images[1].hash, 'Accepted potential map responds to opacity');
    assert.notEqual(images[2].hash, images[3].hash, 'Accepted current map responds to opacity');
  } else {
    assert.equal(new Set(images.map(image => image.hash)).size, 1, 'Withheld potential and current products contribute no pixels');
  }
  assert.equal(await page.locator('.readout-chart-row[data-readout-id^="circuit"]').count(), 0, 'No circuit solution chart is published');
  await select('circuitLayer', 'potential');
  return images;
}

async function verifyCase(reference, caseName, requireAccepted = false) {
  await select('resistance', reference.resistance); await ready();
  await setChecked(page.locator('#panel-controls input[id$="option-waterBarrier"]'), reference.waterBarrier); await ready();
  await select('circuitConnectivity', reference.connectivity); await ready();
  const terminals = await readout('circuitTerminals');
  assert.match(terminals, /source local16 131 \/ local64 2075 \/ source raster 404082/);
  assert.match(terminals, /sink local16 140 \/ local64 2084 \/ source raster 404091/);
  const status = await readout('circuitStatus');
  const residual = await readout('circuitResidual');
  const accepted = /^converged/.test(status);
  if (requireAccepted) assert.ok(accepted, 'Returning to mild contrast must recover an accepted map');
  if (accepted) {
    const resistance = status.match(/effective resistance ([\d.]+)/);
    assert.ok(resistance, status);
    closeTo(Number(resistance[1]), reference.effectiveResistance, reference.effectiveResistance * .001 + .003, 'Main 64² resistance');
    const certificate = residual.match(/current conservation ([\d.e+-]+) · energy ([\d.]+)/);
    assert.ok(certificate, residual);
    // The readout rounds to three significant digits; the underlying gate remains 1e-4.
    assert.ok(Number(certificate[1]) <= .000101);
    closeTo(Number(certificate[2]), reference.energy, reference.energy * .001 + .003, 'Main energy');
  } else {
    assert.match(status, /^not converged · configured budget 1024 · residual /);
    assert.match(residual, /^diagnostic only · current-conservation residual /);
    assert.doesNotMatch(status, /effective resistance/);
    assert.doesNotMatch(residual, /\benergy\b/);
    const certificate = residual.match(/current-conservation residual ([\d.e+-]+)/);
    assert.ok(certificate, residual);
    // The rendered rounding interval must overlap the rejected side of 1e-4.
    const displayedResidual = Number(certificate[1]);
    assert.ok(Number.isFinite(displayedResidual) && displayedResidual >= .0001);
  }
  const subset = oracle.records.find(record => record.size === 16 && record.resistance === reference.resistance && record.waterBarrier === reference.waterBarrier && record.connectivity === reference.connectivity);
  const comparison = await readout('circuitOracle');
  const subsetAccepted = comparison.includes('dense/sparse converged');
  if (subsetAccepted) {
    const energies = comparison.match(/energies ([\d.]+) \/ ([\d.]+)/);
    assert.ok(energies, comparison);
    for (const energy of energies.slice(1)) closeTo(Number(energy), subset.energy, subset.energy * .001 + .003, 'Separate independent 16² energy');
  } else {
    assert.match(comparison, /^comparison withheld · dense (?:not converged|converged), sparse (?:not converged|converged)$/);
    assert.match(comparison, /not converged/);
    assert.doesNotMatch(comparison, /energ(?:y|ies)/);
  }
  const publication = await inspectMapPublication(accepted, caseName);
  return {reference, subset, terminals, status, residual, comparison, accepted, subsetAccepted, publication};
}

try {
  await page.goto(`${baseUrl.replace(/\/$/, '')}/#/story/habitat-connectivity/circuits`, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error, null, {timeout: 180000});
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null);
  await installReadoutIdentifiers(page, 'habitat-connectivity');
  await page.locator('#tab-controls').click(); await ready();
  await select('circuitBudget', '1024'); await ready();
  await select('circuitTolerance', '0.0001'); await ready();
  const references = oracle.records.filter(record => record.size === 64);
  assert.equal(references.length, 8);
  for (const [index, reference] of references.entries()) {
    results.push(await verifyCase(reference, `case-${index + 1}`, reference.resistance === 'mild'));
    await saveReport(false, 'Verification in progress');
  }
  recovery = await verifyCase(references.find(reference => reference.resistance === 'mild' && !reference.waterBarrier && reference.connectivity === 8), 'mild-recovery', true);
  assert.deepEqual(errors, []);
  await page.locator('#tab-story').click();
  await page.screenshot({path: `${outputDirectory}/circuit-recovered.png`, fullPage: true});
  await saveReport(true);
  console.log(JSON.stringify({passed: true, cases: results.length, accepted: results.filter(result => result.accepted).length, withheld: results.filter(result => !result.accepted).length, recovery: recovery.accepted, errors}));
} catch (error) {
  await saveReport(false, String(error));
  throw error;
} finally {
  await browser.close();
  await verifyBuildEvidence();
}
