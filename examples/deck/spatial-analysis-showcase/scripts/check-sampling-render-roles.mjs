// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, mkdir, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {WebMercatorViewport} from '@deck.gl/core';
import {addMetersToLngLat} from '@math.gl/web-mercator';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory = path.resolve(process.argv[3] ?? '/tmp/sampling-render-roles');
await mkdir(outputDirectory, {recursive: true});
const origin = [-87.6, 41.86];
const reports = [];
const errors = [];
const readBuildEvidence = async () => {
  const response = await fetch(new URL('build-evidence.json', baseUrl), {headers: {connection: 'close'}});
  assert.ok(response.ok);
  return response.json();
};
const before = await readBuildEvidence();
await writeFile(path.join(outputDirectory, 'build-evidence.json'), JSON.stringify(before, null, 2));
const parks = JSON.parse(await readFile(new URL('../public/data/chicago-parks/green-space.geojson', import.meta.url), 'utf8'));
const park = parks.features.find(feature => Number(feature.properties?.osmId) === 2914325);
assert.equal(park.geometry.type, 'Polygon');
assert.equal(park.geometry.coordinates.length, 1);
const ring = park.geometry.coordinates[0];
const insideSource = point => {
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
    const start = ring[previous];
    const end = ring[index];
    if ((start[1] > point[1]) !== (end[1] > point[1]) &&
        point[0] < (end[0] - start[0]) * (point[1] - start[1]) / (end[1] - start[1]) + start[0]) inside = !inside;
  }
  return inside;
};
let browser;
try {
  browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
  const page = await browser.newPage({viewport: {width: 1440, height: 960}, deviceScaleFactor: 1});
  page.on('pageerror', error => errors.push(String(error)));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  // Test-only interception retains the exact native buffers used by production rendering.
  // It neither replaces their contents nor changes production graph authoring.
  await page.addInitScript(() => {
    globalThis.__samplingBuffers = new Map();
    const createBuffer = GPUDevice.prototype.createBuffer;
    GPUDevice.prototype.createBuffer = function(descriptor) {
      const buffer = createBuffer.call(this, descriptor);
      const match = descriptor.label?.match(/sampling-frames-\d+-(area-sample-(?:positions|roles|provenance|weights|status))$/);
      if (match) globalThis.__samplingBuffers.set(match[1], {buffer, device: this, size: descriptor.size, label: descriptor.label});
      return buffer;
    };
  });
  const url = new URL(baseUrl);
  url.hash = '#/story/sampling-frames/area-frame';
  await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => {
    const status = document.querySelector('[data-readout-value="status"]')?.textContent ?? '';
    return /^wave 0:.*flags 0/.test(status) || /GPU design error/.test(document.body.innerText);
  }, null, {timeout: 120000});
  assert.doesNotMatch(await page.locator('body').innerText(), /GPU design error/);
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  await page.waitForTimeout(1000);
  const canvasBounds = await page.locator('[aria-label="Map"] canvas').first().boundingBox();
  assert.ok(canvasBounds);
  const screenshot = await page.screenshot({path: path.join(outputDirectory, 'area-render.png')});
  const actual = await page.evaluate(async () => {
    const output = {};
    for (const [name, record] of globalThis.__samplingBuffers) {
      const readback = record.device.createBuffer({size: record.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
      try {
        const encoder = record.device.createCommandEncoder();
        encoder.copyBufferToBuffer(record.buffer, 0, readback, 0, record.size);
        record.device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const bytes = readback.getMappedRange();
        const values = /positions|weights/.test(name) ? new Float32Array(bytes) : new Uint32Array(bytes);
        output[name] = {label: record.label, values: Array.from(values)};
        readback.unmap();
      } finally {readback.destroy();}
    }
    return output;
  });
  await writeFile(path.join(outputDirectory, 'native-buffer-evidence.json'), JSON.stringify(actual, null, 2));
  for (const suffix of ['positions', 'roles', 'provenance', 'weights', 'status']) assert.ok(actual[`area-sample-${suffix}`], `Native area ${suffix} buffer captured`);
  const positions = actual['area-sample-positions'].values;
  const roles = actual['area-sample-roles'].values;
  const provenance = actual['area-sample-provenance'].values;
  const weights = actual['area-sample-weights'].values;
  const status = actual['area-sample-status'].values;
  assert.equal(status.length, 4, 'One default area sampling wave');
  assert.equal(status[3], 0, 'Completed design');
  assert.equal(status[0], status[1], 'All required rows written');
  assert.equal(roles.length, positions.length / 2);
  const occupied = [];
  let unused = 0;
  for (let index = 0; index < roles.length; index++) {
    assert.equal(roles[index], provenance[index * 4 + 3], `Rendered role matches current producer provenance at row ${index}`);
    if (index >= status[0]) {
      assert.equal(roles[index], 0xffffffff, `Unused row ${index} hidden by noDataValue`);
      unused++;
      continue;
    }
    const position = positions.slice(index * 2, index * 2 + 2);
    const longitudeLatitude = addMetersToLngLat(origin, position);
    assert.ok(insideSource(longitudeLatitude), `Occupied row ${index} inside independent source polygon`);
    assert.ok(Math.hypot(...position) > 1, `Occupied row ${index} is not the unused zero/origin sentinel`);
    if (roles[index] === 2) assert.equal(weights[index], 0);
    else assert.equal(weights[index], 40000);
    occupied.push({index, role: roles[index], position, longitudeLatitude, weight: weights[index]});
  }
  const legacy = occupied.filter(row => row.role === 2);
  assert.equal(legacy.length, 1);
  assert.equal(occupied.filter(row => row.role < 2).length, status[2]);
  reports.push({name: 'actual-render-buffer-roles-and-source-membership', passed: true, written: status[0], accepted: status[2], unused, occupied});

  const pixels = PNG.sync.read(screenshot);
  const viewport = new WebMercatorViewport({width: canvasBounds.width, height: canvasBounds.height, longitude: origin[0], latitude: origin[1], zoom: 14.3});
  const screenPoint = longitudeLatitude => {
    const projected = viewport.project(longitudeLatitude);
    return [projected[0] + canvasBounds.x, projected[1] + canvasBounds.y];
  };
  const countColor = (center, expected, radius = 5) => {
    let count = 0;
    for (let positionY = Math.floor(center[1] - radius); positionY <= Math.ceil(center[1] + radius); positionY++) {
      for (let positionX = Math.floor(center[0] - radius); positionX <= Math.ceil(center[0] + radius); positionX++) {
        if (positionX < 0 || positionY < 0 || positionX >= pixels.width || positionY >= pixels.height) continue;
        const offset = (positionY * pixels.width + positionX) * 4;
        if (expected.every((value, channel) => Math.abs(pixels.data[offset + channel] - value) <= 5)) count++;
      }
    }
    return count;
  };
  const originScreen = screenPoint(origin);
  const legacyScreen = screenPoint(legacy[0].longitudeLatitude);
  const greenAtOrigin = countColor(originScreen, [80, 137, 112]);
  const orangeAtLegacy = countColor(legacyScreen, [183, 105, 73]);
  assert.equal(greenAtOrigin, 0, 'No green phantom at unused zero-coordinate origin');
  assert.ok(orangeAtLegacy >= 5, `Actual zero-weight legacy point renders orange (${orangeAtLegacy} interior pixels)`);
  reports.push({name: 'actual-render-pixels', passed: true, originScreen, legacyScreen, greenAtOrigin, orangeAtLegacy});
} catch (error) {
  errors.push(String(error));
} finally {
  if (browser) await browser.close();
  const after = await readBuildEvidence();
  await writeFile(path.join(outputDirectory, 'build-evidence-after.json'), JSON.stringify(after, null, 2));
  if (after.bundleSha256 !== before.bundleSha256) errors.push('Served bundle changed');
  await writeFile(path.join(outputDirectory, 'report.json'), JSON.stringify({baseUrl, bundleSha256: before.bundleSha256, reports, errors, browsersClosed: true}, null, 2));
}
assert.deepEqual(errors, []);
assert.equal(reports.length, 2);
console.log('Sampling render roles, source membership, orange legacy and origin suppression passed.');
