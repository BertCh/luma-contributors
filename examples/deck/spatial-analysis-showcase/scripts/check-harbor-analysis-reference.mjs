// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/harbor-analysis-reference.json';
const reference = JSON.parse(await readFile(new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/ais-encounter-reference.json', import.meta.url), 'utf8'));
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const reports = [], errors = [];
let page;
let passed = false;
let failureReason = null;
try {
  page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
  const readValues = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''])));
  async function openScene(scene) {
    const url = new URL(baseUrl); url.hash = `#/story/${scene}`;
    await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
    await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error, null, {timeout: 90_000});
    assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null, 'Scene must compile before numerical checks');
    await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  }
  async function slider(id, value) {
    await page.locator(`#all-option-${id}`).evaluate((input, nextValue) => {input.value = String(nextValue); input.dispatchEvent(new Event('input', {bubbles: true})); input.dispatchEvent(new Event('change', {bubbles: true}));}, value);
  }
  if (!process.argv.includes('--similarity-only')) {
  await openScene('harbor-continuous-encounters');
  for (const radius of [50, 100, 250, 500]) {
    await slider('distanceMeters', radius);
    for (const step of [60, 300, 600]) {
      await page.locator('#all-option-clockStep').selectOption(String(step));
      const expected = reference.pairCounts[radius];
      const expectedText = `${expected.continuous} continuous · ${expected[`clock${step}`]} at ${step}s samples`;
      await page.waitForFunction(value => [...document.querySelectorAll('.readout')].find(row => row.querySelector('dt')?.textContent === 'Continuous vs sampled pairs')?.querySelector('dd')?.textContent === value, expectedText, {timeout: 30_000});
      const observed = await readValues();
      assert.match(observed['Sampling support'], /max observed interval 267 s/);
      assert.match(observed['GPU capacities'], /pair searches complete/);
      assert.match(observed['GPU capacities'], /restriction invalid 0, overflow 0/);
      const expectedPairs = reference.pairs.filter(pair => pair.minimumMeters <= radius).sort((first, second) => first.minimumMeters - second.minimumMeters);
      if (expectedPairs.length) {
        const values = observed['Continuous minimum distances'].split(' · ');
        assert.equal(values.length, Math.min(8, expectedPairs.length));
        for (let index = 0; index < values.length; index++) {
          const [pair, distance] = values[index].split(': ');
          assert.equal(pair, `${expectedPairs[index].firstPath}/${expectedPairs[index].secondPath}`);
          assert.ok(Math.abs(Number.parseFloat(distance) - expectedPairs[index].minimumMeters) < .1, `${pair}: ${distance}`);
        }
      } else assert.equal(observed['Continuous minimum distances'], 'No pair within threshold');
      reports.push({scene: 'continuous', radius, step, observed});
    }
  }
  await slider('distanceMeters', 100);
  for (const timeBoundary of ['closed', 'open', 'left-closed', 'right-closed']) for (const geometryBoundary of ['closed', 'open']) {
    await page.locator('#all-option-timeBoundary').selectOption(timeBoundary);
    await page.locator('#all-option-geometryBoundary').selectOption(geometryBoundary);
    await page.waitForFunction(boundary => [...document.querySelectorAll('.readout')].find(row => row.querySelector('dt')?.textContent === 'Pairwise restriction')?.querySelector('dd')?.textContent?.includes(boundary), `${timeBoundary} / ${geometryBoundary}`, {timeout: 30_000});
    const observed = await readValues();
    assert.match(observed['GPU capacities'], /restriction invalid 0, overflow 0/);
    reports.push({scene: 'restriction', timeBoundary, geometryBoundary, observed});
  }
  }
  await openScene('harbor-track-similarity');
  for (const [metric, field, threshold, ascending] of [['frechet', 'frechetMeters', 10000, true], ['dynamic-time-warping', 'dtwAccumulatedMeters', 300000, true], ['longest-common-subsequence', 'lcss150Meters', .05, false]]) {
    await page.locator('#all-option-metric').selectOption(metric);
    for (let query = 0; query < 8; query++) {
      await page.locator('#all-option-queryTrack').selectOption(String(query));
      await page.waitForFunction(path => [...document.querySelectorAll('.readout')].find(row => row.querySelector('dt')?.textContent === 'Query path')?.querySelector('dd')?.textContent?.startsWith(`Path ${path} ·`), reference.tracks[query].path, {timeout: 30_000});
      const observed = await readValues();
      assert.match(observed['Search capacity'], /0 over capacity · top-K complete/);
      const ranked = reference.similarities.filter(pair => (pair.firstTrack === query || pair.secondTrack === query) && (ascending ? pair[field] <= threshold : pair[field] >= threshold)).map(pair => ({path: pair.firstTrack === query ? pair.secondPath : pair.firstPath, value: pair[field]})).sort((first, second) => ascending ? first.value - second.value : second.value - first.value).slice(0, 3);
      if (ranked.length) {
        const values = observed['Ranked matches'].split(' · ');
        assert.equal(values.length, ranked.length);
        values.forEach((entry, index) => {
          const [path, text] = entry.split(': '); assert.equal(path, `Path ${ranked[index].path}`);
          const value = Number.parseFloat(text) * (ascending ? 1000 : 1);
          assert.ok(Math.abs(value - ranked[index].value) < (ascending ? 1.1 : .00006), `${metric}/${query}: ${entry} vs ${ranked[index].value}`);
        });
      } else assert.equal(observed['Ranked matches'], 'No candidates meet the threshold');
      reports.push({scene: 'similarity', metric, query, observed});
    }
  }
  await page.locator('#all-option-metric').selectOption('frechet');
  await slider('frechetThresholdMeters', 1000);
  await page.waitForFunction(() => [...document.querySelectorAll('.readout')].find(row => row.querySelector('dt')?.textContent === 'Ranked matches')?.querySelector('dd')?.textContent === 'No candidates meet the threshold', null, {timeout: 30_000});
  reports.push({scene: 'similarity-pruning', observed: await readValues()});
  assert.deepEqual(errors, []);
  passed = true;
} catch (error) {
  failureReason = String(error);
  if (page) {await page.screenshot({path: outputFile.replace(/\.json$/, '-failure.png'), fullPage: true}); reports.push({failure: String(error), readouts: await page.locator('.readout').allTextContents(), visibleText: await page.locator('body').innerText()});}
  throw error;
} finally {await browser.close(); await writeFile(outputFile, JSON.stringify({baseUrl, passed, status: passed ? 'passed' : 'failed', failureReason, reports, consoleErrors: errors}, null, 2) + '\n');}
console.log(`Verified ${reports.length} harbor analysis cases: ${outputFile}`);
