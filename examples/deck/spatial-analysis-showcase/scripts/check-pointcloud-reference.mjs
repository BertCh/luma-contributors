// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

// Use a stable production preview. The Python reference is independently computed from shipped
// original-row samples and supplied normals; none of these expectations come from GPU readback.
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory = process.argv[3] ?? '/tmp/pointcloud-reference';
const reference = JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/pointcloud-data-validation.json', import.meta.url), 'utf8'));
await mkdir(outputDirectory, {recursive: true});
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const reports = [];
const errors = [];
try {
  const page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
  const url = new URL(baseUrl);
  url.hash = '#/story/point-cloud-change';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error, null, {timeout: 90_000});
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null);
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  const read = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].filter(row => row.querySelector('dd')).map(row => [row.querySelector('dt').textContent, row.querySelector('dd').textContent])));
  const numeric = value => Number.parseFloat(value.replaceAll(',', ''));
  const waitForResult = async () => {
    await page.waitForFunction(() => [...document.querySelectorAll('.readout')].some(row => row.querySelector('dt')?.textContent === 'Median signed difference' && /\d/.test(row.querySelector('dd')?.textContent ?? '')), null, {timeout: 60_000});
  };
  const changeControl = async action => {
    const before = await page.evaluate(() => globalThis.spatialAnalysisShowcase?.rebuildCount ?? 0);
    await action();
    await page.waitForFunction(count => (globalThis.spatialAnalysisShowcase?.rebuildCount ?? 0) > count || Boolean(globalThis.spatialAnalysisShowcase?.error), before, {timeout: 60_000});
    assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null);
    await waitForResult();
  };
  const setSlider = async (id, value) => {
    const input = page.locator(`#all-option-${id}`);
    if (Number(await input.inputValue()) === value) return;
    await changeControl(() => input.evaluate((element, next) => {
      element.value = String(next);
      element.dispatchEvent(new Event('input', {bubbles: true}));
      element.dispatchEvent(new Event('change', {bubbles: true}));
    }, value));
  };
  for (const expected of reference.cases) {
    await setSlider('radius', expected.radiusMeters);
    await waitForResult();
    const observed = await read();
    assert.equal(numeric(observed['Cores with usable measurements']), expected.validCores);
    assert.equal(numeric(observed['Cores with invalid supplied normals']), expected.invalidNormalCores);
    assert.equal(numeric(observed['Cores with insufficient support']), expected.insufficientSupportCores);
    assert.equal(numeric(observed['Intervals excluding zero']), expected.detectableCores);
    assert.ok(Math.abs(numeric(observed['Median signed difference']) - expected.medianDifferenceMeters) < 0.001);
    assert.ok(Math.abs(numeric(observed['Median detection limit']) - expected.medianDetectionLimitMeters) < 0.001);
    reports.push({kind: 'surveyed-cylinder', expected, observed});
  }
  await setSlider('radius', 1.5);
  await setSlider('registrationError', 0.2);
  let observed = await read();
  assert.ok(numeric(observed['Intervals excluding zero']) < 181, 'Larger registration uncertainty must reduce detected differences');
  assert.ok(numeric(observed['Median detection limit']) > 0.154629);
  reports.push({kind: 'registration-sensitivity', observed});
  await setSlider('registrationError', 0.03);
  await changeControl(() => page.getByRole('radiogroup', {name: 'Combine uncertainty'}).getByRole('radio', {name: 'Quadrature', exact: true}).click());
  observed = await read();
  assert.ok(numeric(observed['Intervals excluding zero']) >= 181);
  assert.ok(numeric(observed['Median detection limit']) < 0.154629);
  reports.push({kind: 'quadrature-sensitivity', observed});
  await changeControl(() => page.getByRole('radiogroup', {name: 'Combine uncertainty'}).getByRole('radio', {name: 'Linear', exact: true}).click());
  await changeControl(() => page.getByRole('radiogroup', {name: 'Comparison frame'}).getByRole('radio', {name: 'Exploratory local ICP', exact: true}).click());
  await setSlider('maximumDistance', 2);
  for (const expected of reference.localIcpReference.cases) {
    const select = page.locator('#all-option-robust');
    if (await select.inputValue() !== expected.robustMethod) await changeControl(() => select.selectOption(expected.robustMethod));
    observed = await read();
    assert.match(observed['Local ICP status'], /Converged|provisional/);
    assert.ok(numeric(observed['Final ICP inliers']) > 100);
    assert.ok(Math.abs(numeric(observed['Final ICP translation step']) - expected.history.at(-1).translationStepMeters) < 0.005);
    assert.ok(Math.abs(numeric(observed['Median signed difference']) - 0.281966) > 0.01, 'A local fit must visibly alter the moving-crop interpretation');
    reports.push({kind: 'local-icp', method: expected.robustMethod, observed});
  }
  await changeControl(() => page.locator('#all-option-metric').selectOption('point-to-plane'));
  for (const expected of reference.planeIcpReference.cases) {
    const select = page.locator('#all-option-robust');
    if (await select.inputValue() !== expected.robustMethod) await changeControl(() => select.selectOption(expected.robustMethod));
    observed = await read();
    assert.match(observed['Local ICP status'], /Converged|provisional/);
    assert.equal(numeric(observed['Usable target normals']), reference.planeIcpReference.validTargetNormals);
    assert.equal(numeric(observed['Weak target normals']), reference.planeIcpReference.unsupportedTargetNormals);
    assert.equal(numeric(observed['Truncated target neighborhoods']), reference.planeIcpReference.truncatedTargetNormals);
    assert.equal(numeric(observed['Maximum 3 m neighborhood support']), 34);
    assert.ok(numeric(observed['Final ICP inliers']) > 100);
    assert.ok(Math.abs(numeric(observed['Final ICP translation step']) - expected.history.at(-1).translationStepMeters) < 0.005, `Point-to-plane ${expected.robustMethod} step must match independent least-squares reference`);
    reports.push({kind: 'local-plane-icp', method: expected.robustMethod, observed});
  }
  await page.getByRole('radio', {name: 'Signed change', exact: true}).click();
  await page.getByRole('radio', {name: 'Signed change', exact: true}).and(page.locator('[aria-checked="true"]')).waitFor();
  await page.screenshot({path: `${outputDirectory}/local-icp.png`});
  await changeControl(() => page.getByRole('radiogroup', {name: 'Comparison frame'}).getByRole('radio', {name: 'Supplied survey frame', exact: true}).click());
  await page.screenshot({path: `${outputDirectory}/surveyed-frame.png`});
  assert.deepEqual(errors, [], 'Browser errors must not be hidden by numerical agreement');
} finally {
  await browser.close();
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({baseUrl, reports, errors}, null, 2) + '\n');
}
console.log(`Verified ${reports.length} point-cloud scenarios against independent references: ${outputDirectory}`);
