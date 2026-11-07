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
    new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/earth-source-oracles.json', import.meta.url),
    'utf8'
  )
).tiledObservationTopology;
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory = process.argv[3] ?? '/tmp/luma-showcase-overhaul-20261025/tiled-topology';
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
const select = (id, value) =>
  page.locator(`#panel-controls select[id$="option-${id}"]`).selectOption(String(value));
const readout = id => page.locator(`[data-readout="${id}"]`).first().textContent();
const waitForTopology = () =>
  page.waitForFunction(
    () =>
      document
        .querySelector('[data-readout="topologyComponents"]')
        ?.textContent?.includes('all local/global passes converged, no overflow'),
    null,
    {timeout: 180000}
  );

function assertRecord(record, actual) {
  const low = record.classes.low;
  const high = record.classes.high;
  const componentMatch = actual.components.match(
    /low: (\d+) core-local → (\d+) global \((\d+) seam joins\) · high: (\d+) → (\d+) \((\d+)\)/
  );
  assert.ok(componentMatch, actual.components);
  assert.deepEqual(
    componentMatch.slice(1).map(Number),
    [
      low.localComponentCountSum,
      low.globalComponentCount,
      low.seamMerges,
      high.localComponentCountSum,
      high.globalComponentCount,
      high.seamMerges
    ],
    `${record.date}: local/global component reconciliation`
  );
  const populationMatch = actual.population.match(
    /(\d+) valid cells · low < 0\.2: (\d+) · high ≥ 0\.2: (\d+)/
  );
  assert.ok(populationMatch, actual.population);
  assert.deepEqual(
    populationMatch.slice(1).map(Number),
    [record.validPopulation, low.validPopulation, high.validPopulation],
    `${record.date}: finite valid class populations`
  );
  const adjacencyMatch = actual.adjacency.match(/(\d+) low\/high region pairs · (\d+) shared 4-edge contacts/);
  assert.ok(adjacencyMatch, actual.adjacency);
  assert.deepEqual(
    adjacencyMatch.slice(1).map(Number),
    [record.adjacency.lowHighRegionPairCount, record.adjacency.lowHighSharedEdgeContacts],
    `${record.date}: whole-focus four-edge adjacency`
  );
  assert.match(actual.components, /128 slots\/class/);
  assert.match(actual.adjacency, /complete after 32 rounds/);
}

try {
  await page.goto(`${baseUrl.replace(/\/$/, '')}/#/story/tiled-observations/ownership`, {
    waitUntil: 'domcontentloaded'
  });
  await page.waitForFunction(
    () => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error,
    null,
    {timeout: 180000}
  );
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null);
  await installReadoutIdentifiers(page, 'tiled-observations');
  await page.locator('#tab-controls').click();
  for (const record of oracle.records) {
    if (record.date === '2021-09-21') await select('date', 'september');
    await waitForTopology();
    const actual = {
      components: await readout('topologyComponents'),
      population: await readout('topologyPopulation'),
      adjacency: await readout('regionAdjacency')
    };
    assertRecord(record, actual);
    results.push({date: record.date, actual});
    console.log(`PASS ${record.date}: global labels, valid populations, region contacts`);
    await page.locator('#tab-story').click();
    await page.screenshot({path: `${outputDirectory}/topology-${record.date}.png`, fullPage: true});
    if (record.date !== oracle.records.at(-1).date) await page.locator('#tab-controls').click();
  }
  assert.deepEqual(errors, []);
  await writeFile(
    `${outputDirectory}/report.json`,
    JSON.stringify({passed: true, results, errors, source: 'earth-source-oracles.json'}, null, 2)
  );
  console.log(JSON.stringify({passed: true, checks: results.length, errors}));
} catch (error) {
  await writeFile(
    `${outputDirectory}/report.json`,
    JSON.stringify({passed: false, error: String(error), results, errors}, null, 2)
  );
  throw error;
} finally {
  await browser.close();
  await verifyBuildEvidence();
}
