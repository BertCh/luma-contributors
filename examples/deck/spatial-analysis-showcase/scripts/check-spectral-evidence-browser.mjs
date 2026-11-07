import {captureBuildEvidence} from './earth-build-evidence.mjs';
import {installReadoutIdentifiers} from './earth-browser-readouts.mjs';
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const oracles = JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/earth-source-oracles.json', import.meta.url), 'utf8'));
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory = process.argv[3] ?? '/tmp/luma-showcase-overhaul-20261025/spectral-numerics';
await mkdir(outputDirectory, {recursive: true});
const verifyBuildEvidence = await captureBuildEvidence(baseUrl, outputDirectory);
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const errors = [];
const results = [];
page.on('pageerror', error => errors.push(String(error)));
page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
const readout = identifier => page.locator(`[data-readout="${identifier}"]`).first();
const select = (identifier, value) => page.locator(`select[id$="option-${identifier}"]`).first().selectOption(value);
const closeTo = (actual, reference, tolerance, label) => assert.ok(Number.isFinite(actual) && Math.abs(actual - reference) <= tolerance, `${label}: ${actual}, expected ${reference} ± ${tolerance}`);
try {
  await page.goto(`${baseUrl.replace(/\/$/, '')}/#/story/spectral-evidence/index-contrasts`, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error, null, {timeout: 180000});
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null);
 await installReadoutIdentifiers(page, 'spectral-evidence');
  for (const date of ['after', 'before']) {
    await select('date', date);
    await select('saviL', '0.5');
    const reference = oracles.spectralIndices[date];
    await page.waitForFunction(value => document.querySelector('[data-readout="indexEvidence"]')?.textContent?.includes(`NDVI · NIR vs red ${value}`), reference.ndvi.toFixed(3), {timeout: 180000});
    const text = await readout('indexEvidence').textContent();
    for (const [label, value] of [
      ['NDVI · NIR vs red', reference.ndvi], ['NBR · NIR vs SWIR2', reference.nbr],
      ['EVI · blue correction', reference.evi], ['SAVI · L = 0.5', reference.saviL0_5],
      ['Gao-style · NIR vs SWIR1', reference.ndwiGaoSwir1Adaptation], ['McFeeters · green vs NIR', reference.ndwiMcFeeters]
    ]) {
      const suffix = text.split(`${label} `)[1];
      assert.ok(suffix, `${label}: ${text}`);
      closeTo(Number(suffix.split(' · ')[0]), value, 0.001, `${date} ${label}`);
    }
    for (const soilAdjustment of ['0', '1']) {
      await select('saviL', soilAdjustment);
      const expected = (1 + Number(soilAdjustment)) * (reference.bands.nir - reference.bands.red) /
        (reference.bands.nir + reference.bands.red + Number(soilAdjustment));
      await page.waitForFunction(({level, expected}) => document.querySelector('[data-readout="indexEvidence"]')?.textContent?.includes(`SAVI · L = ${level} ${expected}`), {level: soilAdjustment, expected: expected.toFixed(3)});
    }
    results.push({date, text});
  }
  if (process.argv.includes('--extended')) {
    await select('saviL', '0.5');
    for (const reference of oracles.spectralIndexOptions.records) {
      await select('date', reference.date);
      await select('indexEpsilon', reference.epsilon === 0.000001 ? '0.000001' : String(reference.epsilon));
      await select('indexClamp', reference.clamp);
      await page.waitForFunction(() => document.querySelector('[data-readout="indexEvidence"]')?.textContent?.includes('GPU-valid index cells'), null, {timeout: 180000});
      const text = await readout('indexEvidence').textContent();
      const count = text.match(/GPU-valid index cells ([\d,]+) \/ 24576/);
      assert.ok(count, text);
      assert.equal(Number(count[1].replaceAll(',', '')), reference.totalValid);
      for (const [label, key] of [['NDVI · NIR vs red', 'ndvi'], ['NBR · NIR vs SWIR2', 'nbr'], ['EVI · blue correction', 'evi'], ['SAVI · L = 0.5', 'savi'], ['Gao-style · NIR vs SWIR1', 'ndwi-gao'], ['McFeeters · green vs NIR', 'ndwi-mcfeeters']]) {
        const suffix = text.split(`${label} `)[1];
        assert.ok(suffix, text);
        if (reference.probeIndices[key] === null) assert.ok(suffix.startsWith('invalid denominator/support'));
        else closeTo(Number(suffix.split(' · ')[0]), reference.probeIndices[key], 0.001, `${reference.date} ${key} epsilon/clamp`);
      }
      results.push({indexOptions: true, date: reference.date, epsilon: reference.epsilon, clamp: reference.clamp, text});
    }
    await select('indexEpsilon', '0.000001');
    await select('indexClamp', 'none');
  }
  await page.evaluate(() => {window.location.hash = '#/story/spectral-evidence/residual-limits';});
  await select('maximumIterations', '256');
  await select('tolerance', '0.0001');
  for (const date of ['after', 'before']) {
    await select('date', date);
    for (const constraint of ['unconstrained', 'nonnegative', 'simplex']) {
      await select('constraint', constraint);
      await page.waitForFunction(() => {
        const text = document.querySelector('[data-readout="modelEvidence"]')?.textContent ?? '';
        return text.includes('budget 256') && text.includes('tolerance 0.0001') && text.includes('squared residual');
      }, null, {timeout: 180000});
      const text = await readout('modelEvidence').textContent();
      const weights = [...text.matchAll(/(?:vegetation-like|dark low-NBR|bright low-NDVI) (-?[\d.]+)/g)].map(match => Number(match[1]));
      assert.equal(weights.length, 3, text);
      const residual = Number(text.match(/squared residual ([\d.]+)/)?.[1]);
      const reference = oracles.spectralUnmixing.records[date][constraint];
      if (constraint === 'unconstrained') {
        weights.forEach((value, index) => closeTo(value, reference.weights[index], 0.001, `${date} unconstrained weight`));
        closeTo(residual, reference.squaredResidual, 0.00001, 'Unconstrained residual');
      } else {
        assert.ok(weights.every(value => value >= -0.001), text);
        if (constraint === 'simplex') closeTo(weights.reduce((sum, value) => sum + value, 0), 1, 0.002, 'Simplex weight sum');
        assert.ok(residual >= reference.squaredResidual - 0.00001, 'Residual cannot beat the independent constrained optimum');
        closeTo(residual, reference.squaredResidual, 0.0002, `${date} bounded ${constraint} residual`);
      }
      assert.match(text, /solved|iteration limit reached/);
      results.push({date, constraint, text, optimum: reference.squaredResidual});
    }
  }
  assert.deepEqual(errors, [], 'Browser or GPU errors');
  await page.screenshot({path: `${outputDirectory}/spectral-fit.png`, fullPage: true});
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({passed: true, results, errors}, null, 2));
  console.log(JSON.stringify({passed: true, results, errors}, null, 2));
} catch (error) {
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({passed: false, error: String(error), results, errors}, null, 2));
  throw error;
} finally {
  await browser.close(); await verifyBuildEvidence();
}
