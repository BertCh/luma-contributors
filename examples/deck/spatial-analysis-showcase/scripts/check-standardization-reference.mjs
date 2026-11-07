// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/standardization-reference-browser.json';
const reference = JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/earthquake-standardization-reference.json', import.meta.url), 'utf8'));
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 960}});
const reports = [];
const errors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
const readouts = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''])));
async function choose(id, label) {
  await page.locator(`[role="radiogroup"][aria-labelledby="all-option-${id}-label"]`).getByRole('radio', {name: label, exact: true}).click();
}
function close(actual, expected, description) {
  const tolerance = Math.max(0.0001, Math.abs(expected) * 0.00001);
  assert.ok(Math.abs(actual - expected) <= tolerance, `${description}: ${actual} vs ${expected}, tolerance ${tolerance}`);
}
try {
  const url = new URL(baseUrl);
  url.hash = '#/story/earthquake-standardization';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready, null, {timeout: 90_000});
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  for (const [field, fieldLabel] of [['magnitude', 'Magnitude'], ['depth', 'Reported depth']]) {
    await choose('field', fieldLabel);
    for (const [population, populationLabel] of [['all', 'All earthquakes'], ['large', 'Magnitude 6+'], ['shallow', 'Depth ≤70 km']]) {
      await choose('referencePopulation', populationLabel);
      for (const [method, methodLabel, variance] of [
        ['deviation-from-mean', 'Deviation', null],
        ['z-score-sample', 'Z-score', 'Sample'],
        ['z-score-population', 'Z-score', 'Population'],
        ['mad', 'Mean absolute deviation', null],
        ['range-adjust', 'Range-adjust', null],
        ['range-standardize', 'Range-standardize', null]
      ]) {
        await choose('method', methodLabel);
        if (variance) await choose('variance', variance);
        await page.waitForFunction(() => [...document.querySelectorAll('.readout')].some(row => row.querySelector('dt')?.textContent === 'Transformed field' && row.querySelector('dd')?.textContent?.includes('null within population')), null, {timeout: 90_000});
        const observed = await readouts();
        const expectedField = reference.populations[population][field];
        const expected = expectedField.transforms[method];
        const summary = observed['Transformed field'].match(/^([\d,]+) finite · ([-\d.]+) to ([-\d.]+) · mean ([-\d.]+)/);
        assert.ok(summary, observed['Transformed field']);
        assert.equal(Number(summary[1].replaceAll(',', '')), expectedField.count);
        close(Number(summary[2]), expected.minimum, `${field}/${population}/${method} minimum`);
        close(Number(summary[3]), expected.maximum, `${field}/${population}/${method} maximum`);
        close(Number(summary[4]), expected.mean, `${field}/${population}/${method} mean`);
        const probe = observed['GPU probe event'].match(/^Source row ([\d,]+) · .+ → ([-\d.]+)/);
        assert.ok(probe, observed['GPU probe event']);
        assert.equal(Number(probe[1].replaceAll(',', '')), reference.fixedProbeRow);
        close(Number(probe[2]), expected.fixedProbe.output, `${field}/${population}/${method} fixed probe`);
        assert.match(observed['Original field'], /0 null within population/);
        reports.push({field, population, method, observed});
      }
    }
  }
  assert.deepEqual(errors, []);
  await page.screenshot({path: outputFile.replace(/\.json$/, '-verified.png')});
} catch (error) {
  reports.push({failure: String(error), visibleText: await page.locator('body').innerText()});
  await page.screenshot({path: outputFile.replace(/\.json$/, '-failure.png'), fullPage: true});
  throw error;
} finally {
  await browser.close();
  await writeFile(outputFile, JSON.stringify({baseUrl, reports, errors}, null, 2) + '\n');
}
console.log(`Verified ${reports.length} standardization scenarios: ${outputFile}`);
