// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {EARTH_RADIUS_METERS, measureHaversineAngleRadians, toRadians} from './lib/geodesy.mjs';

// Independent sequential CPU OPTICS reference over unchanged archived coordinates. The GPU
// implements a resident parallel frontier; this reference uses sorted queue entries and doubles.
// We compare extraction cardinalities, not arbitrary cluster label numbers or tied visit ranks.
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/observation-hierarchy-reference.json';
const directory = new URL('../public/data/chicago-nature/', import.meta.url);
const [positions, categories] = await Promise.all([
  readFile(new URL('position.bin', directory)), readFile(new URL('category.bin', directory))
]);
const sourcePoints = [];
const frameLatitude = toRadians(41.8595);
const frameLongitude = toRadians(-87.607);
for (let row = 0; row < categories.length; row++) {
  const longitude = positions.readFloatLE(row * 8);
  const latitude = positions.readFloatLE(row * 8 + 4);
  if (categories[row] !== 1 || longitude < -87.617 || longitude > -87.597 || latitude < 41.849 || latitude > 41.87) continue;
  const latitudeRadians = toRadians(latitude);
  const longitudeDelta = toRadians(longitude) - frameLongitude;
  const angle = measureHaversineAngleRadians(latitudeRadians, frameLatitude, longitudeDelta);
  const scale = EARTH_RADIUS_METERS * (angle === 0 ? 1 : angle / Math.sin(angle));
  sourcePoints.push([
    Math.fround(scale * Math.cos(latitudeRadians) * Math.sin(longitudeDelta)),
    Math.fround(scale * (Math.cos(frameLatitude) * Math.sin(latitudeRadians) - Math.sin(frameLatitude) * Math.cos(latitudeRadians) * Math.cos(longitudeDelta)))
  ]);
}
assert.equal(sourcePoints.length, 246);
const distances = sourcePoints.map(first => sourcePoints.map(second => Math.hypot(first[0] - second[0], first[1] - second[1])));
function referenceCuts(minimumSamples) {
  const core = distances.map(row => row.filter(value => value <= 1000).sort((first, second) => first - second)[minimumSamples - 1] ?? Infinity);
  const visited = new Set();
  const reachability = sourcePoints.map(() => Infinity);
  const ordering = [];
  const queue = [];
  for (let start = 0; start < sourcePoints.length; start++) {
    if (visited.has(start)) continue;
    queue.push([Infinity, start]);
    while (queue.length) {
      queue.sort((first, second) => (first[0] === second[0] ? first[1] - second[1] : first[0] - second[0]));
      const [, focus] = queue.shift();
      if (visited.has(focus)) continue;
      visited.add(focus);
      ordering.push(focus);
      if (!Number.isFinite(core[focus])) continue;
      for (let neighbor = 0; neighbor < sourcePoints.length; neighbor++) {
        if (visited.has(neighbor) || distances[focus][neighbor] > 1000) continue;
        const candidate = Math.max(core[focus], distances[focus][neighbor]);
        if (candidate < reachability[neighbor]) {
          reachability[neighbor] = candidate;
          queue.push([candidate, neighbor]);
        }
      }
    }
  }
  return [75, 150, 300].map(extractionDistance => {
    let cluster = -1;
    const labels = sourcePoints.map(() => -1);
    for (const row of ordering) {
      if (reachability[row] > extractionDistance) {
        if (core[row] <= extractionDistance) labels[row] = ++cluster;
      } else labels[row] = cluster;
    }
    const counts = new Map();
    for (const label of labels) counts.set(label, (counts.get(label) ?? 0) + 1);
    const noise = counts.get(-1) ?? 0;
    counts.delete(-1);
    return {minimumSamples, extractionDistance, clusters: counts.size, noise, sizes: [...counts.values()].sort((first, second) => first - second)};
  });
}
const scenarios = [4, 8, 16].flatMap(referenceCuts);
if (process.argv.includes('--reference-only')) {
  await writeFile(outputFile, JSON.stringify({sourceRows: sourcePoints.length, scenarios}, null, 2) + '\n');
  console.log(JSON.stringify(scenarios));
} else {
  const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
  const reports = [];
  const errors = [];
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 960}});
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
    const url = new URL(baseUrl);
    url.hash = '#/story/observation-hierarchy';
    await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
    await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready, null, {timeout: 90_000});
    await page.getByRole('tab', {name: 'All controls', exact: false}).click();
    await page.locator('#all-option-algorithm').selectOption('optics-dbscan');
    for (const expected of scenarios) {
      await page.locator('#all-option-minimumSamples').selectOption(String(expected.minimumSamples));
      await page.locator('#all-option-extractionDistance').selectOption(String(expected.extractionDistance));
      await page.waitForFunction(() => [...document.querySelectorAll('.readout')].some(row =>
        row.querySelector('dt')?.textContent === 'GPU algorithm status' &&
        row.querySelector('dd')?.textContent?.includes('OPTICS visited 246 of 246 rows; complete; 0 invalid rows')
      ), null, {timeout: 90_000});
      const readouts = await page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [
        row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''
      ])));
      const match = readouts['Cluster sizes'].match(/(\d+) clusters, (\d+) noise records; sizes (.*)/);
      assert.ok(match, 'Current complete cluster counts must be published');
      const observed = {clusters: Number(match[1]), noise: Number(match[2]), sizes: match[3].split(', ').filter(value => value !== 'none').map(Number).sort((first, second) => first - second)};
      assert.deepEqual(observed, {clusters: expected.clusters, noise: expected.noise, sizes: expected.sizes}, `Independent OPTICS cut k=${expected.minimumSamples}, epsilon=${expected.extractionDistance}`);
      assert.equal(observed.sizes.reduce((sum, count) => sum + count, observed.noise), 246);
      reports.push({expected, observed, readouts});
      console.log(`Checked OPTICS k=${expected.minimumSamples}, epsilon=${expected.extractionDistance}: ${observed.clusters} clusters, ${observed.noise} noise`);
    }
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
    await writeFile(outputFile, JSON.stringify({baseUrl, sourceRows: sourcePoints.length, reports, errors}, null, 2) + '\n');
  }
}
