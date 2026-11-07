// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {addMetersToLngLat, getDistanceScales} from '@math.gl/web-mercator';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {measureHaversineDistance} from './lib/geodesy.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/route-footpoints-reference.json';
const directory = new URL('../public/data/cta-transit/', import.meta.url);
const manifest = JSON.parse(await readFile(new URL('manifest.json', directory), 'utf8'));
const names = ['stopPosition', 'stopMode', 'stopRouteOffsets', 'stopRoutes', 'shapeVertices', 'shapePathOffsets', 'shapeRoute', 'shapeDirection'];
const inputs = Object.fromEntries(await Promise.all(names.map(async name => {
  const column = manifest.columns[name];
  const bytes = await readFile(new URL(column.file, directory));
  const size = column.dtype === 'uint8' ? 1 : column.dtype === 'uint16' ? 2 : 4;
  const values = Array.from({length: bytes.length / size}, (_, index) => column.dtype === 'uint8' ? bytes.readUInt8(index) : column.dtype === 'uint16' ? bytes.readUInt16LE(index * 2) : column.dtype === 'uint32' ? bytes.readUInt32LE(index * 4) : bytes.readFloatLE(index * 4));
  return [name, values];
})));
const origin = [(manifest.bbox[0] + manifest.bbox[2]) / 2, (manifest.bbox[1] + manifest.bbox[3]) / 2];
const scales = getDistanceScales({longitude: origin[0], latitude: origin[1], highPrecision: true});
const metersPerDegree = scales.unitsPerDegree.map((value, index) => value * scales.metersPerUnit[index]);
// Match the documented frame conversion, then independently compute every point/segment minimum.
function project(longitude, latitude) {
  let horizontal = (longitude - origin[0]) * metersPerDegree[0];
  let vertical = (latitude - origin[1]) * metersPerDegree[1];
  const estimated = addMetersToLngLat(origin, [horizontal, vertical]);
  horizontal += (longitude - estimated[0]) * metersPerDegree[0];
  vertical += (latitude - estimated[1]) * metersPerDegree[1];
  return [Math.fround(horizontal), Math.fround(vertical)];
}
const stations = Array.from({length: inputs.stopPosition.length / 2}, (_, row) => project(inputs.stopPosition[row * 2], inputs.stopPosition[row * 2 + 1]));
const vertices = Array.from({length: inputs.shapeVertices.length / 2}, (_, row) => project(inputs.shapeVertices[row * 2], inputs.shapeVertices[row * 2 + 1]));
const routeReferences = {};
for (const route of ['Blue', 'Red', 'Brn', 'G']) {
  const routeIndex = manifest.properties.routes.findIndex(row => row.id === route && row.type === 'rail');
  const sourceRows = stations.flatMap((_, row) => inputs.stopMode[row] === 1 && inputs.stopRoutes.slice(inputs.stopRouteOffsets[row], inputs.stopRouteOffsets[row + 1]).includes(routeIndex) ? [row] : []);
  const parts = inputs.shapeRoute.flatMap((value, row) => value === routeIndex && inputs.shapeDirection[row] === 0 ? [vertices.slice(inputs.shapePathOffsets[row], inputs.shapePathOffsets[row + 1])] : []).filter(part => part.length >= 2);
  const distances = sourceRows.map(sourceRow => {
    const point = stations[sourceRow];
    let nearest = {distance: Infinity, part: -1, segment: -1, position: [NaN, NaN]};
    parts.forEach((part, partIndex) => {
      for (let segment = 0; segment < part.length - 1; segment++) {
        const start = part[segment];
        const end = part[segment + 1];
        const delta = [end[0] - start[0], end[1] - start[1]];
        const squaredLength = delta[0] ** 2 + delta[1] ** 2;
        const fraction = squaredLength === 0 ? 0 : Math.max(0, Math.min(1, ((point[0] - start[0]) * delta[0] + (point[1] - start[1]) * delta[1]) / squaredLength));
        const position = [start[0] + fraction * delta[0], start[1] + fraction * delta[1]];
        const distance = Math.hypot(point[0] - position[0], point[1] - position[1]);
        if (distance < nearest.distance) nearest = {distance, part: partIndex, segment, position};
      }
    });
    const referenceLonLat = addMetersToLngLat(origin, parts[0][0]).map(Math.fround);
    const referenceDistance = measureHaversineDistance(
      [inputs.stopPosition[sourceRow * 2], inputs.stopPosition[sourceRow * 2 + 1]],
      referenceLonLat
    );
    return {sourceRow, ...nearest, referenceDistance};
  });
  const first = stations[sourceRows[0]];
  const queries = Object.fromEntries(['radius', 'bounds', 'polygon'].map(kind => [kind, sourceRows.filter(sourceRow => {
    const point = stations[sourceRow];
    const horizontal = Math.abs(point[0] - first[0]);
    const vertical = Math.abs(point[1] - first[1]);
    return kind === 'radius' ? Math.hypot(horizontal, vertical) <= 1200 : Math.max(horizontal, vertical) <= (kind === 'polygon' ? 400 : 1200);
  })]));
  routeReferences[route] = {sourceRows, partCount: parts.length, distances, queries};
}
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
    url.hash = '#/story/route-footpoints';
    await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
    await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || document.body.innerText.includes('This analysis could not start'), null, {timeout: 120_000});
    assert.ok(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.ready), await page.locator('body').innerText());
    await page.getByRole('tab', {name: 'All controls', exact: false}).click();
    const read = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''])));
    const wait = (label, text) => page.waitForFunction(([name, expected]) => [...document.querySelectorAll('.readout')].some(row => row.querySelector('dt')?.textContent === name && row.querySelector('dd')?.textContent?.includes(expected)), [label, text], {timeout: 120_000});
    for (const [route, reference] of Object.entries(routeReferences)) {
      await page.locator('#all-option-route').selectOption(route);
      for (const selected of [0, Math.floor(reference.sourceRows.length / 2), reference.sourceRows.length - 1]) {
        await page.locator('#all-option-sample').evaluate((element, value) => {element.value = String(value); element.dispatchEvent(new Event('input', {bubbles: true}));}, selected);
        await wait('GPU distance to returned segment', 'nearest linestring');
        const readouts = await read();
        const result = reference.distances[selected];
        assert.match(readouts['Selected station / nearest result'], new RegExp(`source row ${result.sourceRow} `));
        for (const label of ['GPU distance to returned segment', 'Point distance to returned footpoint']) {
          const match = readouts[label].match(/^([\d.]+) m · nearest linestring ([\d.]+) m/);
          assert.ok(match, label);
          assert.ok(Math.abs(Number(match[1]) - result.distance) < 0.03, `${route} row ${result.sourceRow} independent minimum`);
          assert.ok(Math.abs(Number(match[1]) - Number(match[2])) < 0.03, `${label} parity`);
        }
        reports.push({kind: 'nearest', route, selected, expected: result, readouts});
        console.log(`Checked ${route} station ${selected}: ${result.distance.toFixed(3)} m`);
      }
      for (const kind of ['radius', 'bounds', 'polygon']) {
        await page.locator('#all-option-query').selectOption(kind);
        await wait('Station query output', 'overflow no');
        const readouts = await read();
        const match = readouts['Station query output'].match(/^(\d+) returned · (\d+) required · overflow no/);
        assert.ok(match);
        assert.equal(Number(match[1]), reference.queries[kind].length);
        assert.equal(Number(match[2]), reference.queries[kind].length);
        reports.push({kind: 'query', route, query: kind, expectedRows: reference.queries[kind], readouts});
      }
    }
    for (const units of ['meters', 'kilometers']) {
      await page.locator('#all-option-haversineUnits').selectOption(units);
      await wait('Great-circle distance to same reference', units === 'meters' ? ' m great-circle' : ' km great-circle');
      const readouts = await read();
      const match = readouts['Great-circle distance to same reference'].match(/^([\d.]+) (m|km) great-circle/);
      assert.ok(match);
      const meters = Number(match[1]) * (match[2] === 'km' ? 1000 : 1);
      const expected = routeReferences.G.distances[0].referenceDistance;
      assert.ok(Math.abs(meters - expected) < 2, 'Independent spherical distance, allowing float32 trigonometry and displayed rounding');
      reports.push({kind: 'units', units, expected, readouts});
    }
    assert.deepEqual(errors, []);
  } finally {
    const lastBody = page ? await page.locator('body').innerText().catch(() => '') : '';
    await browser.close();
    await writeFile(outputFile, JSON.stringify({baseUrl, routeReferences, reports, errors, lastBody}, null, 2) + '\n');
  }
} else {
  await writeFile(outputFile, JSON.stringify({routeReferences}, null, 2) + '\n');
  console.log(JSON.stringify(Object.fromEntries(Object.entries(routeReferences).map(([route, reference]) => [route, {stations: reference.sourceRows.length, parts: reference.partCount, queryCounts: Object.fromEntries(Object.entries(reference.queries).map(([kind, rows]) => [kind, rows.length]))}]))));
}
