// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {EARTH_RADIUS_METERS, measureHaversineDistance} from './lib/geodesy.mjs';

// Independent double-precision references from unchanged archived coordinates. Run only against
// a stable built showcase; the parent validation sweep owns GPU/browser scheduling.
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/spherical-observation-reference.json';
const datasetUrl = new URL('../public/data/poopdeck-earthquakes/', import.meta.url);
const [positionBytes, eventTypes] = await Promise.all([
  readFile(new URL('position.bin', datasetUrl)),
  readFile(new URL('type.bin', datasetUrl))
]);
assert.equal(positionBytes.byteLength, eventTypes.length * 8);
const earthRadiusKilometers = EARTH_RADIUS_METERS / 1000;
const centers = {Japan: [140, 35], Türkiye: [38, 36], Andes: [-72, -24], Sunda: [107.5, -1]};
const earthquakePositions = [];
for (let index = 0; index < eventTypes.length; index++) {
  if (eventTypes[index] === 0) earthquakePositions.push([
    positionBytes.readFloatLE(index * 8), positionBytes.readFloatLE(index * 8 + 4)
  ]);
}
function measureDistance(first, second) {
  return measureHaversineDistance(first, second, earthRadiusKilometers);
}
const scenarios = [
  {center: 'Japan', radius: 500, level: '4', mode: 'exterior'},
  {center: 'Japan', radius: 1000, level: '5', mode: 'interior'},
  {center: 'Türkiye', radius: 500, level: '4', mode: 'exterior'},
  {center: 'Andes', radius: 1000, level: '5', mode: 'interior'},
  {center: 'Sunda', radius: 250, level: '5', mode: 'interior'},
  {center: 'Japan', radius: 700, level: '4', mode: 'exterior'}
].map(scenario => {
  const distances = earthquakePositions.map(position => measureDistance(centers[scenario.center], position));
  return {
    ...scenario,
    count: distances.filter(distance => distance <= scenario.radius).length,
    area: 2 * Math.PI * earthRadiusKilometers ** 2 * (1 - Math.cos(scenario.radius / earthRadiusKilometers)),
    closestBoundaryKilometers: distances.reduce((closest, distance) => Math.min(closest, Math.abs(distance - scenario.radius)), Infinity)
  };
});
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const reports = [];
const errors = [];
try {
  const page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
  const url = new URL(baseUrl);
  url.hash = '#/story/spherical-observation-support';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready, null, {timeout: 90_000});
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  for (const expected of scenarios) {
    await page.locator('#all-option-level').selectOption(expected.level);
    await page.locator('#all-option-coverMode').selectOption(expected.mode);
    await page.locator('#all-option-center').selectOption(expected.center);
    await page.locator('#all-option-radiusKm').evaluate((input, radius) => {
      input.value = String(radius);
      input.dispatchEvent(new Event('input', {bubbles: true}));
      input.dispatchEvent(new Event('change', {bubbles: true}));
    }, expected.radius);
    await page.waitForFunction(expectedValues => {
      const read = label => [...document.querySelectorAll('.readout')]
        .find(row => row.querySelector('dt')?.textContent === label)?.querySelector('dd')?.textContent ?? '';
      const numeric = value => Number(value.replaceAll(',', ''));
      const count = numeric(read('Earthquakes within cap').split(' of ')[0]);
      const areas = read('Cover area / analytic cap area').match(/([\d,]+) km² \/ ([\d,]+) km² cap/);
      if (!areas || count !== expectedValues.count || Math.abs(numeric(areas[2]) - expectedValues.area) > 1) return false;
      const coverArea = numeric(areas[1]);
      return read('GPU cover status') === 'GPU cover and cap-event search complete' &&
        (expectedValues.mode === 'interior' ? coverArea <= expectedValues.area + 1 : coverArea >= expectedValues.area - 1);
    }, expected, {timeout: 60_000});
    const observed = await page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [
      row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''
    ])));
    assert.equal(Number(observed['Earthquakes within cap'].split(' of ')[0].replaceAll(',', '')), expected.count);
    const coverCounts = observed['Cap candidate cells'].match(/([\d,]+) cells \(([\d,]+) required\)/);
    assert.ok(coverCounts, 'Cover counts must be explicit');
    assert.equal(coverCounts[1], coverCounts[2], 'Full-grid capacity must not truncate the cover');
    reports.push({expected, observed});
  }
  assert.deepEqual(errors, [], 'Browser errors must not be hidden by numerical agreement');
} finally {
  await browser.close();
  await writeFile(outputFile, JSON.stringify({baseUrl, earthquakeRows: earthquakePositions.length, reports, errors}, null, 2) + '\n');
}
console.log(`Verified ${reports.length} spherical scenarios against independent references: ${outputFile}`);
