// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// DATA4-5 browser check (headless WebGPU): resident Spatial Join across coordinate systems.
// Case A: points in UTM zone 18N (EPSG:32618, meters) joined to polygons in EPSG:4326.
// Case B: points in EPSG:4326 joined to polygons whose vertices are UTM zone 18N meters.
// The per-point match count of the GPU join is compared with a proj4 CPU join. Points sit at least
// 40 m from every polygon edge so Float32 storage and the GPU transverse Mercator series cannot move
// a point across an edge. Usage: node check-spatial-join-crs.mjs [url]
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import proj4 from 'proj4';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {closeNativeDialog} from './native-navigation.mjs';
import {
  attachErrorCollectors, createMultiPolygonFeature, createPolygonFeature, exportArrow, makeSquareRing, publishTable, selectNativeOption
} from './spatial-join-utils.mjs';

const UTM = '+proj=utm +zone=18 +datum=WGS84 +units=m +no_defs';
const toUtm = proj4('EPSG:4326', UTM);
const POINT_COUNT = 600;
const MARGIN_METERS = 40;

// Polygon rings in longitude/latitude: overlap, hole and a multipolygon.
const polygonShapes = [
  {rings: [makeSquareRing(-74.1, 40.65, -73.95, 40.75)]},
  {rings: [makeSquareRing(-74.02, 40.7, -73.9, 40.82)]},
  {rings: [makeSquareRing(-74.3, 40.55, -74.15, 40.7), makeSquareRing(-74.25, 40.6, -74.2, 40.65)]},
  {multi: true, rings: [makeSquareRing(-73.88, 40.55, -73.84, 40.59), makeSquareRing(-73.78, 40.55, -73.74, 40.59)]}
];

const mapRing = (ring, transform) => ring.map(([x, y]) => transform(x, y));
const makeShapeFeature = (shape, transform) =>
  shape.multi
    ? createMultiPolygonFeature(shape.rings.map(ring => [mapRing(ring, transform)]))
    : createPolygonFeature(shape.rings.map(ring => mapRing(ring, transform)));
const identity = (x, y) => [x, y];
const forwardUtm = (x, y) => toUtm.forward([x, y]);

function isInsideRing(x, y, ring) {
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
    const [ax, ay] = ring[previous];
    const [bx, by] = ring[index];
    if (ay > y !== by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax) inside = !inside;
  }
  return inside;
}
/** Rings are lon/lat; a point inside a part's outer ring and outside its holes is inside the shape. */
function isInsideShape(longitude, latitude, shape) {
  const parts = shape.multi ? shape.rings.map(ring => [ring]) : [shape.rings];
  return parts.some(([outer, ...holes]) => isInsideRing(longitude, latitude, outer) && !holes.some(hole => isInsideRing(longitude, latitude, hole)));
}
/** Distance in meters from a lon/lat point to the nearest polygon edge, measured in UTM. */
function getEdgeDistance(longitude, latitude) {
  const [x, y] = forwardUtm(longitude, latitude);
  let best = Infinity;
  for (const shape of polygonShapes)
    for (const ring of shape.rings) {
      const projected = ring.map(([ringLongitude, ringLatitude]) => forwardUtm(ringLongitude, ringLatitude));
      for (let index = 1; index < projected.length; index++) {
        const [ax, ay] = projected[index - 1];
        const [bx, by] = projected[index];
        const lengthSquared = (bx - ax) ** 2 + (by - ay) ** 2;
        const t = Math.max(0, Math.min(1, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / lengthSquared));
        best = Math.min(best, Math.hypot(x - (ax + t * (bx - ax)), y - (ay + t * (by - ay))));
      }
    }
  return best;
}

let seed = 20261031;
const random = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
const points = [];
while (points.length < POINT_COUNT) {
  const longitude = -74.35 + random() * 0.65;
  const latitude = 40.5 + random() * 0.35;
  if (getEdgeDistance(longitude, latitude) >= MARGIN_METERS) points.push([longitude, latitude]);
}
const expectedCounts = points.map(([longitude, latitude]) => polygonShapes.filter(shape => isInsideShape(longitude, latitude, shape)).length);
assert(expectedCounts.some(count => count === 0) && expectedCounts.some(count => count >= 2), 'fixture needs unmatched and doubly matched points');

function makePointTable(coordinates) {
  return new arrow.Table({
    // Native Spatial Join finds the point geometry by the longitude/latitude column names; the values
    // are in the declared CRS (UTM meters in case A).
    longitude: arrow.vectorFromArray(coordinates.map(row => row[0]), new arrow.Float32()),
    latitude: arrow.vectorFromArray(coordinates.map(row => row[1]), new arrow.Float32()),
    id: arrow.vectorFromArray(coordinates.map((_, index) => index), new arrow.Int32())
  });
}
function makePolygonTable(transform) {
  return new arrow.Table({
    geometry: arrow.vectorFromArray(polygonShapes.map(shape => makeShapeFeature(shape, transform)), new arrow.Utf8()),
    rank: arrow.vectorFromArray(polygonShapes.map((_, index) => index), new arrow.Int32())
  });
}

const cases = [
  {name: 'utm-points-4326-polygons', pointRows: points.map(([longitude, latitude]) => toUtm.forward([longitude, latitude])), polygonTransform: identity, leftCrs: 'EPSG:32618', rightCrs: 'EPSG:4326'},
  {name: '4326-points-utm-polygons', pointRows: points, polygonTransform: forwardUtm, leftCrs: 'EPSG:4326', rightCrs: 'EPSG:32618'}
];

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1400}, acceptDownloads: true});
const errors = [];
let currentStage = 'startup';
const watchdog = setTimeout(() => {
  console.error(`Spatial Join CRS check exceeded 240 seconds at ${currentStage}`);
  process.exit(1);
}, 240000);
attachErrorCollectors(page, errors);

async function setCrs(testId, value) {
  const input = page.locator(`[data-testid="${testId}-input"]`).first();
  await input.fill(value);
  await input.blur();
  await page.waitForFunction(({testId, value}) => {
    const select = document.querySelector(`[data-testid="${testId}-select"]`);
    return !document.querySelector(`[data-testid="${testId}-error"]`) && (select?.value === value || select?.value === 'custom');
  }, {testId, value});
  // The field commits on blur; the recorded CRS shows in the select (custom for a UTM code).
  const expectedSelect = value === 'EPSG:4326' || value === 'EPSG:3857' ? value : 'custom';
  await page.waitForFunction(({testId, expectedSelect}) => document.querySelector(`[data-testid="${testId}-select"]`)?.value === expectedSelect, {testId, expectedSelect});
}

try {
  await pinViteModules(page);
  const results = [];
  for (const testCase of cases.filter(candidate => !process.env.CRS_CASE || candidate.name.includes(process.env.CRS_CASE))) {
    currentStage = testCase.name;
    await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
    await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
    const pointLabel = `crs-points-${testCase.name}.arrow`;
    const polygonLabel = `crs-polygons-${testCase.name}.arrow`;
    await publishTable(page, pointLabel, makePointTable(testCase.pointRows), {longitude: 'longitude', latitude: 'latitude'});
    await page.waitForFunction(count => window.__keplerComputeTest.getState().publications.some(source => source.rowCount === count), POINT_COUNT, {timeout: 60000});
    await publishTable(page, polygonLabel, makePolygonTable(testCase.polygonTransform), {geometry: 'geometry'});
    await page.waitForFunction(() => window.__keplerComputeTest.getState().publications.some(source => source.rowCount === 4), undefined, {timeout: 60000});
    await closeNativeDialog(page);
    const pointId = await page.evaluate(count => window.__keplerComputeTest.getState().publications.find(source => source.rowCount === count).id, POINT_COUNT);
    const knownIds = await page.evaluate(() => window.__keplerComputeTest.getState().publications.map(source => source.id));
    await page.locator(`.dataset-ops-menu__toggle[data-for="dataset-ops-${pointId}"]`).click();
    await page.locator('.dataset-ops-menu__spatial-join').click();
    const panel = page.locator('.dataset-ops-panel');
    await panel.waitFor();
    await panel.locator('.data-source-selector').last().locator('.item-selector__dropdown').click();
    await page.locator('.list-selector:visible').getByText(polygonLabel, {exact: true}).click();
    await panel.getByText(/aggregation rules/i).click();
    const rankRow = panel.locator('.dataset-ops-column-list > div').filter({hasText: /^rank/});
    await selectNativeOption(page, rankRow, 'maximum');
    await setCrs('spatial-join-left-crs', testCase.leftCrs);
    await setCrs('spatial-join-right-crs', testCase.rightCrs);
    await panel.locator('input[type=text]').last().fill(`CRS join ${testCase.name}`);
    await panel.getByRole('button', {name: 'Run', exact: true}).click();
    await page.waitForFunction(ids => window.__keplerComputeTest.getState().publications.some(source => !ids.includes(source.id)), knownIds, {timeout: 90000});
    const resultId = await page.evaluate(ids => window.__keplerComputeTest.getState().publications.find(source => !ids.includes(source.id)).id, knownIds);
    const table = await exportArrow(page, resultId);
    assert.equal(table.numRows, POINT_COUNT);
    const counts = table.getChild('count').toArray().map(Number);
    const ids = table.getChild('id').toArray().map(Number);
    let mismatches = 0;
    const firstMismatches = [];
    for (let row = 0; row < POINT_COUNT; row++) {
      const expected = expectedCounts[ids[row]];
      if (counts[row] !== expected) {
        mismatches++;
        if (firstMismatches.length < 5) firstMismatches.push({id: ids[row], actual: counts[row], expected, point: points[ids[row]]});
      }
    }
    const matchedCount = counts.filter(Boolean).length;
    const expectedMatched = expectedCounts.filter(Boolean).length;
    results.push({case: testCase.name, rows: POINT_COUNT, mismatches, matched: matchedCount, expectedMatched});
    console.log(JSON.stringify(results.at(-1)));
    // Known root cause of a case A failure (checked 2026-11-01): point-position-source.ts treats
    // coordinates outside longitude +-180 / latitude +-90 as invalid and stores NaN positions, so
    // projected left coordinates never reach the GPU reprojection (EPSG:3857 points fail the same way).
    assert.equal(mismatches, 0, `${testCase.name}: GPU CRS join differs from the proj4 CPU join (${matchedCount} matched of ${expectedMatched} expected): ${JSON.stringify(firstMismatches)}`);
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, results}));
} catch (error) {
  console.error(currentStage, errors);
  throw error;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
