// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/data-topology-reference.json';
const selectedScene = process.argv[4] ?? 'both';
const referenceRoot = new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/', import.meta.url);
const reference = JSON.parse(await readFile(new URL('fire-topology-reference.json', referenceRoot), 'utf8'));
const extended = JSON.parse(await readFile(new URL('fire-topology-extended-reference.json', referenceRoot), 'utf8'));
const reports = [];
const errors = [];
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 960}});
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
const readouts = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''])));
const number = value => Number.parseFloat(value.replaceAll(',', ''));
async function openScene(scene) {
  const url = new URL(baseUrl);
  url.hash = `#/story/${scene}`;
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(sceneId => globalThis.spatialAnalysisShowcase?.sceneId === sceneId && globalThis.spatialAnalysisShowcase?.ready, scene, {timeout: 90_000});
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
}
async function slider(id, value) {
  await page.locator(`#all-option-${id}`).evaluate((input, nextValue) => {
    input.value = String(nextValue);
    input.dispatchEvent(new Event('input', {bubbles: true}));
    input.dispatchEvent(new Event('change', {bubbles: true}));
  }, value);
}
async function waitReadout(label, expected) {
  await page.waitForFunction(({label, expected}) => {
    const value = [...document.querySelectorAll('.readout')].find(row => row.querySelector('dt')?.textContent === label)?.querySelector('dd')?.textContent ?? '';
    return Number.parseFloat(value.replaceAll(',', '')) === expected;
  }, {label, expected}, {timeout: 90_000});
}
try {
  if (selectedScene === 'both' || selectedScene === 'earthquake-query-lab') {
    await openScene('earthquake-query-lab');
    await waitReadout('Rows after query', 77215);
    let observed = await readouts();
    assert.equal(number(observed['Source rows']), 77231);
    assert.equal(number(observed['Non-earthquake types']), 16);
    assert.match(observed['Type-dimension lookup'], /^77,231 found, 0 missing/);
    assert.equal(observed['Join capacity'], 'no join overflow');
    reports.push({scene: 'earthquake-query-lab', scenario: 'earthquakes-default', observed});
    await page.locator('#all-option-includeOtherTypes').press('Space');
    await waitReadout('Rows after query', 77231);
    reports.push({scene: 'earthquake-query-lab', scenario: 'all-types', observed: await readouts()});
    await page.locator('#all-option-includeOtherTypes').press('Space');
    await slider('minimumMagnitude', 6);
    await waitReadout('Rows after query', 626);
    reports.push({scene: 'earthquake-query-lab', scenario: 'earthquakes-magnitude6', observed: await readouts()});
  }
  if (selectedScene === 'both' || selectedScene === 'fire-scar-topology') {
    await openScene('fire-scar-topology');
    await waitReadout('Common valid pixels', reference.validCells);
    await page.locator('#all-option-cleanup').selectOption('none');
    for (const [threshold, expected] of Object.entries(reference.thresholds)) {
      await slider('thresholdValue', threshold);
      for (const [connectivity, key] of [['4', 'fourNeighbors'], ['8', 'eightNeighbors']]) {
        await page.locator('#all-option-connectivity').selectOption(connectivity);
        await waitReadout('Connected components', expected[key].componentCount);
        const observed = await readouts();
        assert.equal(number(observed['Cells in mask']), expected.foregroundCells);
        assert.ok(Math.abs(number(observed['Mask area on accepted support']) - expected.foregroundAreaHa) <= 0.051);
        assert.ok(Math.abs(number(observed['Largest patch']) - expected[key].largestAreaHa) <= 0.051);
        assert.match(observed['Topology status'], /^converged/);
        reports.push({scene: 'fire-scar-topology', threshold, connectivity, cleanup: 'none', observed});
      }
    }
    await slider('thresholdValue', 0.27);
    for (const [scenario, expected] of Object.entries(extended.transformedMasks)) {
      const [cleanup, radius] = scenario.split('-radius');
      await page.locator('#all-option-cleanup').selectOption(cleanup);
      await slider('radius', radius);
      await waitReadout('Connected components', expected.eightNeighbors.componentCount);
      const observed = await readouts();
      assert.equal(number(observed['Cells in mask']), expected.foregroundCells);
      assert.ok(Math.abs(number(observed['Largest patch']) - expected.eightNeighbors.largestAreaHa) <= 0.051);
      reports.push({scene: 'fire-scar-topology', scenario, connectivity: '8', observed});
    }
    await page.locator('#all-option-cleanup').selectOption('none');
    await page.locator('#all-option-thresholdMethod').selectOption('otsu');
    await waitReadout('Cells in mask', 18656);
    const observed = await readouts();
    assert.equal(number(observed['Otsu dNBR split']), 0.453);
    reports.push({scene: 'fire-scar-topology', scenario: 'otsu-raw', observed});
  }
  if (selectedScene === 'fire-focus') {
    await openScene('fire-scar-topology');
    await page.locator('#all-option-thresholdMethod').selectOption('fixed');
    await slider('thresholdValue', 0.27);
    await page.locator('#all-option-cleanup').selectOption('none');
    await page.locator('#all-option-connectivity').selectOption('8');
    await page.locator('#all-option-mapLayer').selectOption('classContacts');
    await waitReadout('Observed class boundary', 231.48);
    reports.push({scene: 'fire-scar-topology', scenario: 'raw-class-contact', observed: await readouts()});
    await page.locator('#all-option-mapLayer').selectOption('skeleton');
    await waitReadout('Skeleton cells', extended.focus.skeletonCells);
    for (const mapLayer of ['skeleton', 'medialAxis', 'watershed', 'watershedMarkers']) {
      await page.locator('#all-option-mapLayer').selectOption(mapLayer);
      await page.waitForFunction(() => [...document.querySelectorAll('.readout')].some(row => row.querySelector('dt')?.textContent === 'Watershed status' && row.querySelector('dd')?.textContent?.includes('complete queue')), null, {timeout:90_000});
      const observed = await readouts();
      assert.equal(number(observed['Focus accepted support']), extended.focus.validCells);
      assert.equal(number(observed['Watershed markers']), 16);
      assert.ok(Math.abs(number(observed['Maximum inscribed radius']) - extended.focus.maximumInscribedRadiusMeters) <= 0.051);
      assert.equal(number(observed['Skeleton cells']), extended.focus.skeletonCells);
      reports.push({scene:'fire-scar-topology',scenario:mapLayer,observed});
      await page.screenshot({path:outputFile.replace(/\.json$/, `-${mapLayer}.png`)});
    }
  }
  assert.deepEqual(errors, [], 'Browser errors must not be hidden by numerical agreement');
} catch (error) {
  await page.screenshot({path: outputFile.replace(/\.json$/, '-failure.png'), fullPage: true});
  reports.push({failure: String(error), visibleText: await page.locator('body').innerText()});
  throw error;
} finally {
  await browser.close();
  await writeFile(outputFile, JSON.stringify({baseUrl, reports, errors}, null, 2) + '\n');
}
console.log(`Verified ${reports.length} data/topology scenarios: ${outputFile}`);
