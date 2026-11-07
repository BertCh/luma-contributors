// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {EARTH_RADIUS_METERS, toRadians} from './lib/geodesy.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/pacific-boundary-reference.json';
const dot = (first, second) => first.reduce((sum, value, index) => sum + value * second[index], 0);
const cross = (first, second) => [first[1] * second[2] - first[2] * second[1], first[2] * second[0] - first[0] * second[2], first[0] * second[1] - first[1] * second[0]];
const normalize = value => value.map(coordinate => coordinate / Math.hypot(...value));
const unit = (longitude, latitude) => {
  return [Math.cos(toRadians(latitude)) * Math.cos(toRadians(longitude)), Math.cos(toRadians(latitude)) * Math.sin(toRadians(longitude)), Math.sin(toRadians(latitude))];
};
const windows = {A: [[145, -15], [-150, -15], [-150, 55], [145, 55]], B: [[170, -35], [-115, -35], [-115, 30], [170, 30]]};
const vectors = Object.fromEntries(Object.entries(windows).map(([name, points]) => [name, points.map(point => unit(...point))]));
const normals = Object.fromEntries(Object.entries(vectors).map(([name, points]) => [name, points.map((point, index) => normalize(cross(point, points[(index + 1) % points.length])))]));
const classify = (point, name) => {
  const sides = normals[name].map(normal => dot(point, normal));
  const inside = sides.every(side => side >= -1e-6);
  const boundary = inside && sides.some(side => Math.abs(side) <= 1e-6);
  return {inside, boundary};
};
const directory = new URL('../public/data/poopdeck-earthquakes/', import.meta.url);
const [positions, types] = await Promise.all(['position.bin', 'type.bin'].map(name => readFile(new URL(name, directory))));
const classifications = [];
for (let row = 0; row < types.length; row++) if (types[row] === 0) {
  const point = unit(positions.readFloatLE(row * 8), positions.readFloatLE(row * 8 + 4));
  classifications.push({A: classify(point, 'A'), B: classify(point, 'B')});
}
assert.equal(classifications.length, 77215);
const contains = [];
for (const window of ['A', 'B']) for (const boundary of ['closed', 'open']) for (const complement of [false, true]) {
  let accepted = 0;
  let boundaryCount = 0;
  for (const point of classifications) {
    const result = point[window];
    if (result.boundary) boundaryCount++;
    if (result.boundary ? boundary === 'closed' : complement ? !result.inside : result.inside) accepted++;
  }
  contains.push({window, boundary, complement, accepted, boundaryCount});
}
// Spherical Sutherland-Hodgman in double precision: normalized chord-plane crossings
// lie on the same minor great-circle arc. No luma clipping or area helpers are used.
let clipped = vectors.A;
for (const normal of normals.B) {
  const output = [];
  for (let index = 0; index < clipped.length; index++) {
    const first = clipped[index];
    const second = clipped[(index + 1) % clipped.length];
    const firstSide = dot(first, normal);
    const secondSide = dot(second, normal);
    if (firstSide >= 0) output.push(first);
    if ((firstSide >= 0) !== (secondSide >= 0)) {
      const fraction = firstSide / (firstSide - secondSide);
      output.push(normalize(first.map((value, axis) => value + fraction * (second[axis] - value))));
    }
  }
  clipped = output;
}
const area = polygon => {
  let total = 0;
  for (let index = 1; index + 1 < polygon.length; index++) {
    const [first, second, third] = [polygon[0], polygon[index], polygon[index + 1]];
    total += 2 * Math.atan2(Math.abs(dot(first, cross(second, third))), 1 + dot(first, second) + dot(second, third) + dot(third, first));
  }
  return total;
};
const intersectionArea = area(clipped);
assert.equal(clipped.length, 4);
const boolean = (left, right, operation) => operation === 'intersection' ? left && right : operation === 'union' ? left || right : operation === 'difference' ? left && !right : left !== right;
const arcs = [];
for (const complement of [false, true]) for (const operation of ['intersection', 'union', 'difference', 'symmetric-difference']) {
  const accepted = classifications.filter(point => boolean(complement ? !point.A.inside : point.A.inside, point.B.inside, operation)).length;
  arcs.push({complement, operation, accepted, referenceInside: boolean(complement, false, operation)});
}
const reference = {events: classifications.length, contains, arcs, intersectionVertices: clipped.length, intersectionArea, intersectionKm2: intersectionArea * EARTH_RADIUS_METERS ** 2 / 1e6, sourceFraction: intersectionArea / area(vectors.A), targetFraction: intersectionArea / area(vectors.B), unionArea: area(vectors.A) + area(vectors.B) - intersectionArea};
const reports = [];
const errors = [];
if (!process.argv.includes('--reference-only')) {
  const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
  let page;
  try {
    page = await browser.newPage({viewport: {width: 1440, height: 960}});
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
    const url = new URL(baseUrl);
    url.hash = '#/story/pacific-boundary-logic';
    await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
    await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || document.body.innerText.includes('This analysis could not start'), null, {timeout: 120000});
    assert.ok(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.ready), await page.locator('body').innerText());
    await page.getByRole('tab', {name: 'All controls', exact: false}).click();
    const read = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''])));
    const wait = text => page.waitForFunction(expected => [...document.querySelectorAll('.readout')].some(row => row.querySelector('dt')?.textContent === 'Selected GPU result status' && row.querySelector('dd')?.textContent?.includes(expected)), text, {timeout: 120000});
    const setComplement = async value => {const checkbox = page.locator('#all-option-complement'); if (await checkbox.isChecked() !== value) await checkbox.locator('..').click();};
    for (const scenario of contains) {
      await page.locator('#all-option-window').selectOption(scenario.window);
      await page.locator('#all-option-boundary').selectOption(scenario.boundary);
      await setComplement(scenario.complement);
      await wait(`${scenario.window}-${scenario.boundary}-${scenario.complement}`);
      const readouts = await read();
      assert.equal(Number(readouts['Accepted archived earthquakes'].replaceAll(',', '')), scenario.accepted);
      assert.equal(Number(readouts['Archive points in boundary band'].replaceAll(',', '')), scenario.boundaryCount);
      assert.equal(Number(readouts['Invalid archive point statuses']), 0);
      assert.match(readouts['Separate synthetic corner probes'], /^4\/4/);
      reports.push({kind: 'contains', scenario, readouts});
    }
    await page.locator('#all-option-analysis').selectOption('convex');
    for (const capacity of ['3', '16']) {
      await page.locator('#all-option-capacity').selectOption(capacity);
      await wait(`capacity ${capacity}`);
      const readouts = await read();
      if (capacity === '3') {
        assert.match(readouts['Selected GPU result status'], /status 2/);
        assert.match(readouts['Intersection spherical area'], /withheld/);
        assert.match(readouts['GPU intersection vertices'], /4 required/);
      } else {
        assert.match(readouts['Selected GPU result status'], /status 0/);
        assert.match(readouts['GPU intersection vertices'], /^4 written \/ 4 required$/);
        const resultArea = Number(readouts['Intersection spherical area'].split(' km²')[0].replaceAll(',', ''));
        assert.ok(Math.abs(resultArea - reference.intersectionKm2) < 30, `${resultArea} versus ${reference.intersectionKm2}`);
      }
      reports.push({kind: 'convex', capacity, readouts});
    }
    await page.locator('#all-option-analysis').selectOption('arc');
    for (const scenario of arcs) {
      await setComplement(scenario.complement);
      await page.locator('#all-option-operation').selectOption(scenario.operation);
      await wait(`GPUSphericalArcOverlay · ${scenario.operation}`);
      const readouts = await read();
      assert.match(readouts['Selected GPU result status'], /status 0/);
      assert.equal(Number(readouts['Archive events by A/B Boolean membership'].split(' ')[0].replaceAll(',', '')), scenario.accepted, JSON.stringify(readouts));
      assert.ok(readouts['Reference membership at 0°E, 0°N'].includes(`→ ${scenario.referenceInside ? 'inside' : 'outside'} ${scenario.operation}`));
      reports.push({kind: 'arc', scenario, readouts});
    }
    assert.deepEqual(errors, []);
  } finally {
    const lastBody = page ? await page.locator('body').innerText().catch(() => '') : '';
    await browser.close();
    await writeFile(outputFile, JSON.stringify({baseUrl, reference, reports, errors, lastBody}, null, 2) + '\n');
  }
} else {
  await writeFile(outputFile, JSON.stringify({reference}, null, 2) + '\n');
  console.log(JSON.stringify(reference));
}
