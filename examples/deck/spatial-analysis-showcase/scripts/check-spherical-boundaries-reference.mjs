// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {EARTH_RADIUS_METERS, toRadians} from './lib/geodesy.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/spherical-boundaries-reference.json';
const collection = JSON.parse(await readFile(new URL('../public/data/natural-earth/ne_110m_admin_0_countries.geojson', import.meta.url), 'utf8'));
const ring = collection.features.find(feature => feature.properties.ISO_A3 === 'FJI').geometry.coordinates[1][0].slice(0, -1);
const dot = (first, second) => first.reduce((sum, value, index) => sum + value * second[index], 0);
const cross = (first, second) => [first[1] * second[2] - first[2] * second[1], first[2] * second[0] - first[0] * second[2], first[0] * second[1] - first[1] * second[0]];
const unit = ([longitude, latitude]) => {
  const longitudeRadians = toRadians(longitude);
  const latitudeRadians = toRadians(latitude);
  return [Math.cos(latitudeRadians) * Math.cos(longitudeRadians), Math.cos(latitudeRadians) * Math.sin(longitudeRadians), Math.sin(latitudeRadians)];
};
const vertices = ring.map(unit);
// Independent solid-angle fan area. Source and probes remain double precision here; the GPU
// receives float32 unit vectors. This measures the generalized great-circle shell, not a coast.
let signedArea = 0;
for (let index = 1; index < vertices.length - 1; index++) {
  const first = vertices[0];
  const second = vertices[index];
  const third = vertices[index + 1];
  signedArea += 2 * Math.atan2(dot(first, cross(second, third)), 1 + dot(first, second) + dot(second, third) + dot(third, first));
}
const areaKm2 = Math.abs(signedArea) * (EARTH_RADIUS_METERS / 1000) ** 2;
const longitudes = ring.map(point => point[0]);
const latitudes = ring.map(point => point[1]);
const west = Math.min(...longitudes);
const south = Math.min(...latitudes);
const width = Math.max(...longitudes) - west;
const height = Math.max(...latitudes) - south;
const probes = [];
for (let row = 0; row <= 18; row++) for (let column = 0; column <= 22; column++) probes.push(unit([west + width * column / 22, south + height * row / 18]));
probes.push(...vertices);
let interior = 0;
let boundary = 0;
for (const probe of probes) {
  const onVertex = vertices.some(vertex => Math.hypot(...vertex.map((value, index) => value - probe[index])) <= 1e-6);
  if (onVertex) {boundary++; continue;}
  let winding = 0;
  for (let index = 0; index < vertices.length; index++) {
    const first = vertices[index];
    const second = vertices[(index + 1) % vertices.length];
    winding += Math.atan2(dot(probe, cross(first, second)), dot(first, second) - dot(probe, first) * dot(probe, second));
  }
  if (Math.abs(winding) > Math.PI) interior++;
}
const reference = {probes: probes.length, interior, boundary, exterior: probes.length - interior - boundary, areaKm2};
assert.deepEqual([reference.probes, interior, boundary, reference.exterior], [445, 283, 8, 154]);
assert.ok(Math.abs(areaKm2 - 11580.368460243442) < 1e-5);
const reports = [];
const errors = [];
if (!process.argv.includes('--reference-only')) {
  const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 960}});
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
    const url = new URL(baseUrl);
    url.hash = '#/story/spherical-boundaries';
    await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
    await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready, null, {timeout: 120_000});
    await page.getByRole('tab', {name: 'All controls', exact: false}).click();
    const select = (name, value) => page.locator(`#all-option-${name}`).selectOption(String(value));
    const read = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''])));
    const wait = text => page.waitForFunction(expected => [...document.querySelectorAll('.readout')].some(row => row.querySelector('dt')?.textContent === 'Selected GPU result' && row.querySelector('dd')?.textContent?.includes(expected)), text, {timeout: 120_000});
    const number = value => Number(value.replaceAll(',', '').match(/^[\d.]+/)?.[0]);
    await select('view', 'contains');
    for (const complement of [false, true]) for (const boundaryRule of ['closed', 'open']) {
      await select('boundary', boundaryRule);
      const complementSwitch = page.locator('#all-option-complement');
      if (await complementSwitch.isChecked() !== complement) {
        await complementSwitch.locator('..').click();
      }
      assert.equal(await complementSwitch.isChecked(), complement);
      await wait('Containment:');
      const readouts = await read();
      const expected = (complement ? reference.exterior : interior) + (boundaryRule === 'closed' ? boundary : 0);
      assert.equal(number(readouts['Accepted spherical probes']), expected);
      assert.equal(number(readouts['Boundary probes']), boundary);
      assert.equal(number(readouts['Invalid probes']), 0);
      reports.push({kind: 'contains', complement, boundaryRule, expected, readouts});
      console.log(`Checked ${boundaryRule}, complement=${complement}: ${expected} accepted`);
    }
    await select('view', 'overlay');
    for (const operation of ['intersection', 'union', 'difference', 'symmetric-difference']) {
      await select('operation', operation);
      await wait('complete output polygon(s)');
      const readouts = await read();
      assert.match(readouts['Overlay status and provenance counts'], /Output is complete/);
      assert.ok(number(readouts['GPU Boolean output polygons']) > 0);
      reports.push({kind: 'overlay', operation, readouts});
      console.log(`Checked complete ${operation} output and provenance status`);
    }
    await select('view', 'cover');
    for (const [minimumLevel, maximumLevel] of [[3, 6], [3, 8], [5, 10]]) {
      await select('minimumLevel', minimumLevel);
      await select('maximumLevel', maximumLevel);
      for (const mode of ['exterior', 'interior']) {
        await select('coverMode', mode);
        await wait('Cover and measures are complete.');
        const readouts = await read();
        const status = readouts['Cover traversal and capacity status'];
        assert.match(status, /status 0/);
        assert.match(status, /0 cell measure errors/);
        const counts = status.replaceAll(',', '').match(/(\d+) written (\d+) required so far/);
        assert.ok(counts);
        assert.equal(Number(counts[1]), Number(counts[2]));
        const supportArea = number(readouts['Summed S2 support area']);
        assert.ok(Number.isFinite(supportArea));
        assert.ok(mode === 'exterior' ? supportArea >= areaKm2 - 1 : supportArea <= areaKm2 + 1, `${mode} area must bracket independent spherical source area`);
        reports.push({kind: 'cover', minimumLevel, maximumLevel, mode, supportArea, readouts});
        console.log(`Checked S2 ${minimumLevel}–${maximumLevel} ${mode}: ${counts[1]} cells, ${supportArea} km²`);
      }
    }
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
    await writeFile(outputFile, JSON.stringify({baseUrl, reference, reports, errors}, null, 2) + '\n');
  }
} else {
  await writeFile(outputFile, JSON.stringify({reference}, null, 2) + '\n');
  console.log(JSON.stringify(reference));
}
