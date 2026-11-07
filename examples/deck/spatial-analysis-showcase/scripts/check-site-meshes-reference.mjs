// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {addMetersToLngLat, getDistanceScales} from '@math.gl/web-mercator';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/site-meshes-reference.json';
const directory = new URL('../public/data/', import.meta.url);
const [positions, categories, buildingText] = await Promise.all([
  readFile(new URL('chicago-facilities/position.bin', directory)),
  readFile(new URL('chicago-facilities/category.bin', directory)),
  readFile(new URL('chicago-buildings/buildings.geojson', directory), 'utf8')
]);
const origin = [-87.7, 41.86];
const scales = getDistanceScales({longitude: origin[0], latitude: origin[1], highPrecision: true});
const metersPerDegree = scales.unitsPerDegree.map((value, index) => value * scales.metersPerUnit[index]);
// The scene's declared display-frame conversion is reproduced to compare the same float32
// sites. Hull topology below is an independent CPU monotone-chain construction.
function project(longitude, latitude) {
  let horizontal = (longitude - origin[0]) * metersPerDegree[0];
  let vertical = (latitude - origin[1]) * metersPerDegree[1];
  const estimated = addMetersToLngLat(origin, [horizontal, vertical]);
  horizontal += (longitude - estimated[0]) * metersPerDegree[0];
  vertical += (latitude - estimated[1]) * metersPerDegree[1];
  return [Math.fround(horizontal), Math.fround(vertical)];
}
const sites = [];
for (let row = 0; row < categories.length; row++) if (categories[row] === 1) sites.push(project(positions.readFloatLE(row * 8), positions.readFloatLE(row * 8 + 4)));
sites.sort((first, second) => first[0] - second[0] || first[1] - second[1]);
const cross = (first, second, third) => (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0]);
const chain = sequence => {
  const result = [];
  for (const point of sequence) {
    while (result.length > 1 && cross(result.at(-2), result.at(-1), point) <= 0) result.pop();
    result.push(point);
  }
  return result;
};
const hullVertices = chain(sites).length + chain(sites.toReversed()).length - 2;
const building = JSON.parse(buildingText).features.find(feature => Number(feature.properties.bldg_id) === 354712);
const rings = building.geometry.coordinates[0];
const courtyardVertices = rings.reduce((sum, ring) => sum + ring.length - 1, 0);
const reference = {
  libraries: sites.length,
  hullVertices,
  delaunayTriangles: 2 * sites.length - 2 - hullVertices,
  courtyardVertices,
  courtyardHoles: rings.length - 1,
  courtyardTriangles: courtyardVertices + 2 * (rings.length - 1) - 2
};
assert.deepEqual(reference, {libraries: 82, hullVertices: 12, delaunayTriangles: 150, courtyardVertices: 22, courtyardHoles: 1, courtyardTriangles: 22});
const reports = [];
const errors = [];
if (!process.argv.includes('--reference-only')) {
  const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 960}});
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
    const url = new URL(baseUrl);
    url.hash = '#/story/site-meshes';
    await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
    await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready, null, {timeout: 120_000});
    await page.getByRole('tab', {name: 'All controls', exact: false}).click();
    const read = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.readout')].map(row => [row.querySelector('dt')?.textContent ?? '', row.querySelector('dd')?.textContent ?? ''])));
    const wait = (label, text) => page.waitForFunction(([name, expected]) => [...document.querySelectorAll('.readout')].some(row => row.querySelector('dt')?.textContent === name && row.querySelector('dd')?.textContent?.includes(expected)), [label, text], {timeout: 120_000});
    await wait('Delaunay / Voronoi status', 'required triangles');
    let readouts = await read();
    assert.match(readouts['Delaunay / Voronoi output'], /^150 triangles,/);
    assert.match(readouts['Delaunay / Voronoi status'], /exact duplicate sites 0/);
    reports.push({kind: 'mesh', readouts});
    await page.locator('#all-option-view').selectOption('ratio');
    await page.locator('#all-option-ratio').press('End');
    await wait('Hull status', 'converged');
    readouts = await read();
    assert.match(readouts['Hull rule'], /ratio 1;/);
    assert.match(readouts['Hull output'], /^12 boundary edges; 150 retained triangles$/);
    reports.push({kind: 'convex-limit', readouts});
    await page.locator('#all-option-view').selectOption('courtyard');
    await wait('Courtyard triangulation status', 'Earcut valid');
    readouts = await read();
    assert.match(readouts['Courtyard triangulation status'], /Earcut valid 1; CDT polygon valid/);
    assert.match(readouts['Courtyard triangle counts'], /^Earcut 22 triangles; CDT 22 triangles; source has 2 rings$/);
    reports.push({kind: 'courtyard', readouts});
    await page.locator('#all-option-maximumFlips').press('Home');
    await wait('Courtyard triangulation status', 'Earcut valid');
    readouts = await read();
    assert.match(readouts['Courtyard triangulation status'], /Earcut valid 1; CDT polygon valid/);
    assert.match(readouts['Courtyard triangle counts'], /^Earcut 22 triangles; CDT 22 triangles;/);
    reports.push({kind: 'bounded-flips', readouts});
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
    await writeFile(outputFile, JSON.stringify({baseUrl, reference, reports, errors}, null, 2) + '\n');
  }
} else {
  await writeFile(outputFile, JSON.stringify({reference}, null, 2) + '\n');
  console.log(JSON.stringify(reference));
}
