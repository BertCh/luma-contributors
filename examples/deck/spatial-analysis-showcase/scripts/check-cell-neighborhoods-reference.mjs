// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import * as h3 from 'h3-js';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory = process.argv[3] ?? '/tmp/cell-neighborhoods-reference';
const sourceBytes = await readFile(new URL('../public/data/openflights/position.bin', import.meta.url));
const coordinates = new Float32Array(sourceBytes.buffer, sourceBytes.byteOffset, sourceBytes.byteLength / 4);
const airportRows = {ORD: 2033, MDW: 1703, JFK: 1247, LAX: 1501, SYD: 2589};
const keyAt = (words, index) => ((BigInt(words[index + 1]) << 32n) | BigInt(words[index])).toString(16);
const cellAt = (code, resolution) => h3.latLngToCell(coordinates[airportRows[code] * 2 + 1], coordinates[airportRows[code] * 2], resolution);
const scenarios = [3, 4, 5].flatMap(resolution => ['MDW', 'JFK', 'LAX', 'SYD'].map(destination => ({resolution, destination, capacity: 256})));
scenarios.push({resolution: 4, destination: 'LAX', capacity: 16}, {resolution: 5, destination: 'LAX', capacity: 64});
const reports = [];
const errors = [];
await mkdir(outputDirectory, {recursive: true});
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
try {
  const page = await browser.newPage({viewport: {width: 1440, height: 960}});
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
  await page.addInitScript(() => {
    // Observe actual production readback without adding a runtime testing interface to the scene.
    const original = GPUBuffer.prototype.getMappedRange;
    GPUBuffer.prototype.getMappedRange = function (...arguments_) {
      const bytes = original.apply(this, arguments_);
      if ([16, 64, 256].some(capacity => bytes.byteLength === 32 + 120 * capacity)) {
        const words = new Uint32Array(bytes);
        if (words[1] >>> 27 === 1 && words[4] >>> 27 === 1) globalThis.__cellReferenceReadback = Array.from(words);
      }
      return bytes;
    };
  });
  const url = new URL(baseUrl);
  url.hash = '#/story/cell-neighborhoods';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error, null, {timeout: 90_000});
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null);
  await page.getByRole('tab', {name: 'All controls', exact: false}).click();
  for (const scenario of scenarios) {
    for (const [option, value] of Object.entries({destination: scenario.destination, resolution: String(scenario.resolution), pathCapacity: String(scenario.capacity)})) {
      const control = page.locator(`#all-option-${option}`);
      if (await control.inputValue() !== value) await control.selectOption(value);
    }
    const origin = cellAt('ORD', scenario.resolution);
    const destination = cellAt(scenario.destination, scenario.resolution);
    await page.waitForFunction(({originCell, destinationCell, capacity}) => {
      const words = globalThis.__cellReferenceReadback;
      if (!words || words.length !== 8 + 30 * capacity) return false;
      const key = index => ((BigInt(words[index + 1]) << 32n) | BigInt(words[index])).toString(16);
      return key(0) === originCell && key(3) === destinationCell;
    }, {originCell: origin, destinationCell: destination, capacity: scenario.capacity}, {timeout: 60_000});
    const words = Uint32Array.from(await page.evaluate(() => globalThis.__cellReferenceReadback));
    assert.equal(keyAt(words, 0), origin);
    assert.equal(keyAt(words, 3), destination);
    assert.equal(words[2], 1);
    assert.equal(words[5], 1);
    let distance;
    try { distance = h3.gridDistance(origin, destination); } catch { distance = 0xffffffff; }
    assert.equal(words[6], distance);
    const requiredCount = words[7];
    if (distance === 0xffffffff) assert.equal(requiredCount, 0);
    else assert.equal(requiredCount, distance + 1, 'These chosen airport paths must have complete local-IJ solutions');
    const storedCount = Math.min(requiredCount, scenario.capacity);
    const cells = Array.from({length: storedCount}, (_, index) => keyAt(words, 8 + 2 * index));
    if (storedCount) assert.equal(cells[0], origin);
    if (requiredCount && requiredCount <= scenario.capacity) assert.equal(cells.at(-1), destination);
    for (let index = 1; index < storedCount; index++) assert.ok(h3.areNeighborCells(cells[index - 1], cells[index]), 'Every stored path step must be adjacent');
    const floats = new Float32Array(words.buffer);
    const areaOffset = 8 + scenario.capacity * 2;
    const perimeterOffset = areaOffset + scenario.capacity;
    const edgeCountOffset = perimeterOffset + scenario.capacity;
    const pentagonOffset = edgeCountOffset + scenario.capacity;
    const edgeIdentityOffset = pentagonOffset + scenario.capacity;
    const vertexIdentityOffset = edgeIdentityOffset + scenario.capacity * 12;
    let largestAreaRelativeError = 0;
    for (let index = 0; index < storedCount; index++) {
      const cell = cells[index];
      const expectedArea = h3.cellArea(cell, 'km2');
      const areaRelativeError = Math.abs(floats[areaOffset + index] - expectedArea) / expectedArea;
      largestAreaRelativeError = Math.max(largestAreaRelativeError, areaRelativeError);
      assert.ok(areaRelativeError < 0.003, `Spherical area must meet the documented resolution≤5 fallback accuracy: ${cell}`);
      const boundary = h3.cellToBoundary(cell);
      const perimeter = boundary.reduce((sum, point, vertex) => sum + h3.greatCircleDistance(point, boundary[(vertex + 1) % boundary.length], 'km'), 0);
      assert.ok(Math.abs(floats[perimeterOffset + index] - perimeter) / perimeter < 0.003);
      assert.equal(words[edgeCountOffset + index], boundary.length);
      assert.equal(words[pentagonOffset + index], Number(h3.isPentagon(cell)));
      const identities = offset => Array.from({length: 6}, (_, slot) => keyAt(words, offset + index * 12 + slot * 2)).filter(key => key !== '0').sort();
      assert.deepEqual(identities(edgeIdentityOffset), h3.originToDirectedEdges(cell).sort());
      assert.deepEqual(identities(vertexIdentityOffset), h3.cellToVertexes(cell).sort());
    }
    reports.push({...scenario, origin, destination, distance: distance === 0xffffffff ? 'undefined' : distance, requiredCount, storedCount, largestAreaRelativeError});
  }
  await page.getByRole('tab', {name: 'Story', exact: true}).click();
  await page.screenshot({path: `${outputDirectory}/path-prefix.png`});
  assert.deepEqual(errors, []);
} finally {
  await browser.close();
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({baseUrl, reports, errors}, null, 2) + '\n');
}
console.log(`Verified ${reports.length} H3 scenarios against independent h3-js identities, path invariants and spherical measures: ${outputDirectory}`);
