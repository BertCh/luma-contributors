// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/osm-semantics-reference-browser.json';
const selected = process.argv[4] ?? 'both';
const referenceRoot = new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/', import.meta.url);
const groupReference = JSON.parse(await readFile(new URL('osm-group-reference.json', referenceRoot), 'utf8'));
const temporalReference = JSON.parse(await readFile(new URL('osm-temporal-reference.json', referenceRoot), 'utf8'));
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 960}});
const reports = [];
const errors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
const readouts = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''])));
async function openScene(scene) {
  const url = new URL(baseUrl);
  const step = scene === 'osm-history-contributors' ? 'group-moments' : 'which-event-was-first';
  url.hash = `#/story/${scene}/${step}`;
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(sceneId => (globalThis.spatialAnalysisShowcase?.ready && globalThis.spatialAnalysisShowcase?.sceneId === sceneId) || document.body.textContent?.includes('This analysis could not start:'), scene, {timeout: 60_000});
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.ready), true, await page.locator('body').innerText());
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  if (scene === 'osm-history-contributors') {
    assert.equal(await page.locator('#all-option-colorBy option[value="within-group-z-score"]').count(), 1, 'Frozen production build is missing the group-moment controls');
  } else {
    assert.ok(await page.getByText('Sorted range (GPU)', {exact: true}).count() > 0, 'Frozen production build is missing the exact temporal audit');
  }
}
async function slider(id, value) {
  await page.locator(`#all-option-${id}`).evaluate((input, next) => {
    input.value = String(next);
    input.dispatchEvent(new Event('input', {bubbles: true}));
    input.dispatchEvent(new Event('change', {bubbles: true}));
  }, value);
}
async function waitReadout(label, value) {
  await page.waitForFunction(({label, value}) => [...document.querySelectorAll('.readout')].some(row => row.querySelector('dt')?.textContent === label && row.querySelector('dd')?.textContent?.includes(value)), {label, value}, {timeout: 15_000});
}
function close(actual, expected, tolerance, description) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${description}: ${actual} vs ${expected}`);
}
async function checkGroup(window, rank) {
  const expected = groupReference.windows[window].groups[rank];
  await slider('inspect', rank);
  await waitReadout('Inspected group moments', `Contributor #${rank}: n=${expected.count.toLocaleString('en-US')} `);
  if (expected.displaySupported) await waitReadout('GPU z-score probe', `Sample row ${expected.firstSourceRow}: GPU`);
  else await waitReadout('GPU z-score probe', 'unavailable');
  const observed = await readouts();
  assert.match(observed['GPU group table status'], /overflow 0/);
  if (expected.displaySupported) {
    const moments = observed['Inspected group moments'].match(/sample SD ([-\d.]+) days · skewness g₁ ([-\d.]+) · excess kurtosis g₂ ([-\d.]+)/);
    assert.ok(moments, observed['Inspected group moments']);
    close(Number(moments[1]), expected.sampleStandardDeviation, 0.06, 'sample SD');
    close(Number(moments[2]), expected.skewness, 0.003, 'skewness');
    close(Number(moments[3]), expected.excessKurtosis, 0.003, 'excess kurtosis');
    const probe = observed['GPU z-score probe'].match(/z-score ([-\d.]+)/);
    assert.ok(probe, observed['GPU z-score probe']);
    close(Number(probe[1]), expected.firstSourceRowZ, 0.001, 'row z-score');
  } else assert.match(observed['Inspected group moments'], /withheld/);
  reports.push({scene: 'osm-history-contributors', window, rank, observed});
}
try {
  if (selected !== 'temporal') {
    await openScene('osm-history-contributors');
    await slider('years-low', 2007.5);
    await slider('years-high', 2026.4);
    await page.locator('#all-option-kind').selectOption('all');
    await page.locator('#all-option-colorBy').selectOption('within-group-z-score');
    for (const rank of [1, 2, 10, 50]) await checkGroup('all', rank);
    await slider('years-low', 2013.5);
    await slider('years-high', 2014.5);
    await page.locator('#all-option-kind').selectOption('1');
    for (const rank of [1, 2, 10]) await checkGroup('concentrated-other', rank);
    await page.screenshot({path: outputFile.replace(/\.json$/, '-groups.png')});
  }
  if (selected !== 'groups') {
    await openScene('osm-history-draws-itself');
    for (const scenario of temporalReference.cases) {
      await slider('time', scenario.day);
      for (const group of scenario.groups) {
        const fragment = group.count ? `${group.label}: n ${group.count.toLocaleString('en-US')};` : `${group.label}: empty`;
        await waitReadout('Sorted range (GPU)', fragment);
        await waitReadout('Full scan (GPU)', fragment);
      }
      const observed = await readouts();
      assert.match(observed['GPU comparison'], /^0 mismatches/);
      for (const label of ['Sorted range (GPU)', 'Full scan (GPU)']) {
        assert.ok(observed[label].includes(`Occupied: ${scenario.occupiedCount}`));
        assert.match(observed[label], /overflow 0/);
        for (const group of scenario.groups) {
          if (!group.count) continue;
          assert.ok(observed[label].includes(`row ${group.minimumRow}–${group.maximumRow};`));
          for (const endpoint of ['first', 'last']) {
            const value = group[endpoint];
            assert.ok(observed[label].includes(`${endpoint} row ${value.row}, contributor #${value.contributor}, ${value.timestamp}`), `${label} ${endpoint}: ${JSON.stringify(group)}`);
          }
        }
      }
      reports.push({scene: 'osm-history-draws-itself', day: scenario.day, observed});
    }
    await page.screenshot({path: outputFile.replace(/\.json$/, '-temporal.png')});
  }
  assert.deepEqual(errors, []);
} catch (error) {
  reports.push({failure: String(error), readouts: await readouts(), visibleText: await page.locator('body').innerText()});
  await page.screenshot({path: outputFile.replace(/\.json$/, '-failure.png'), fullPage: true});
  throw error;
} finally {
  await browser.close();
  await writeFile(outputFile, JSON.stringify({baseUrl, reports, errors}, null, 2) + '\n');
}
console.log(`Verified ${reports.length} OSM semantics scenarios: ${outputFile}`);
