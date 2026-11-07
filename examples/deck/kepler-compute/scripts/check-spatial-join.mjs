// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser check for the bounded resident native Spatial Join (<= 262,144 rows, readback path):
// GPU point-in-polygon enrichment compared with a CPU reference over overlapping polygons, a hole,
// a multipolygon, boundary points, null left values and null right aggregates.
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {closeNativeDialog} from './native-navigation.mjs';
import {
  aggregateValues, attachErrorCollectors, createMultiPolygonFeature, createPolygonFeature, createPolygonTable, exportArrow, matchPolygons, polygonRows,
  publishTable, selectNativeOption
} from './spatial-join-utils.mjs';

// [longitude, latitude, amount, tag]; boundary and vertex points sit exactly on Float32 values.
const pointRows = [
  [1, 1, 1, 'a'], [7, 7, 2, 'b'], [12, 12, null, 'a'], [5, 5, 4, null], [10, 10, 5, 'c'],
  [10, 7, 6, 'b'], [22, 2, 7, 'a'], [25, 5, 8, 'c'], [20, 5, 9, 'a'], [41, 1, 10, 'b'],
  [45, 1, 11, 'c'], [43, 1, 12, 'a'], [-5, -5, 13, 'b'], [7.5, 7.5, null, 'c'], [30, 5, 15, 'a'],
  [0, 0, 16, 'b'], [15, 15, 17, 'a'], [24, 4, 18, 'c'], [100, 50, 19, 'a'], [2, 8, 20, 'b']
];
const pointTable = new arrow.Table({
  longitude: arrow.vectorFromArray(pointRows.map(row => row[0]), new arrow.Float32()),
  latitude: arrow.vectorFromArray(pointRows.map(row => row[1]), new arrow.Float32()),
  amount: arrow.vectorFromArray(pointRows.map(row => row[2]), new arrow.Float32()),
  tag: arrow.vectorFromArray(pointRows.map(row => row[3]), new arrow.Utf8())
});
const pointLabel = 'spatial-join-points.arrow';
const polygonLabel = 'spatial-join-polygons.arrow';
const runs = [
  {predicate: 'Intersects', id: 'intersects', weight: 'sum', rank: 'maximum'},
  {predicate: 'Within', id: 'within', weight: 'average', rank: 'minimum'},
  {predicate: 'Intersects', id: 'intersects', weight: 'count', rank: 'average'}
];

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1400}, acceptDownloads: true});
const errors = [];
const sqlRequests = [];
let currentStage = 'startup';
const watchdog = setTimeout(() => {
  console.error(`Spatial Join check exceeded 150 seconds at ${currentStage}`);
  void browser.close();
}, 150000);
function stage(name) {
  currentStage = name;
  console.log(JSON.stringify({stage: name}));
}
attachErrorCollectors(page, errors);
page.on('request', request => {
  if (/duckdb.*(?:\.wasm|worker)/i.test(request.url())) sqlRequests.push(request.url());
});
async function setAggregation(panel, field, technique) {
  const row = panel.locator('.dataset-ops-column-list > div').filter({hasText: new RegExp(`^${field}`)});
  await selectNativeOption(page, row, technique);
}
async function assertMergeChecked(panel) {
  // The default flow keeps the native merge aggregation on the polygon geometry column.
  const row = panel.locator('.dataset-ops-column-list > div').filter({hasText: /^geometry/});
  assert.ok(await row.locator('input[type=checkbox]').isChecked(), 'merge must be checked by default');
  assert.match(await row.innerText(), /merge/i);
}
function getMergedGeometry(matches) {
  // Native mergeGeometries over polygon rows: none null, one itself, several polygons one MultiPolygon.
  const geometries = matches.map(index => JSON.parse(createGeometryText(polygonRows[index])).geometry);
  if (!geometries.length) return null;
  if (geometries.length === 1) return {type: 'Feature', geometry: geometries[0], properties: {}};
  if (geometries.every(geometry => geometry.type === 'Polygon'))
    return {type: 'Feature', geometry: {type: 'MultiPolygon', coordinates: geometries.map(geometry => geometry.coordinates)}, properties: {}};
  if (geometries.every(geometry => geometry.type === 'MultiPolygon'))
    return {type: 'Feature', geometry: {type: 'MultiPolygon', coordinates: geometries.flatMap(geometry => geometry.coordinates)}, properties: {}};
  return {type: 'Feature', geometry: {type: 'GeometryCollection', geometries}, properties: {}};
}
function createGeometryText(row) {
  return row.multi
    ? createMultiPolygonFeature([[row.rings[0]], [row.rings[1]]])
    : createPolygonFeature(row.rings);
}
function expectedRow(rowIndex, run) {
  const [longitude, latitude, amount, tag] = pointRows[rowIndex];
  const matches = matchPolygons(longitude, latitude, run.id);
  const weights = matches.map(index => polygonRows[index].weight === null ? null : Math.fround(polygonRows[index].weight));
  const ranks = matches.map(index => polygonRows[index].rank);
  return [longitude, latitude, amount, tag, matches.length, aggregateValues(weights, run.weight), aggregateValues(ranks, run.rank)];
}
function isClose(actual, expected) {
  if (expected === null || actual === null) return actual === expected;
  return Math.abs(actual - expected) <= Math.max(1e-5, Math.abs(expected) * 1e-6);
}

try {
  await pinViteModules(page);
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  stage('publish points and polygons');
  await publishTable(page, pointLabel, pointTable, {longitude: 'longitude', latitude: 'latitude'});
  await page.waitForFunction(count => window.__keplerComputeTest.getState().publications.some(source => source.rowCount === count), pointRows.length, {timeout: 30000});
  await publishTable(page, polygonLabel, createPolygonTable(), {geometry: 'geometry'});
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications.some(source => source.rowCount === 4), undefined, {timeout: 30000});
  await closeNativeDialog(page);
  const pointId = await page.evaluate(count => window.__keplerComputeTest.getState().publications.find(source => source.rowCount === count).id, pointRows.length);
  const knownIds = await page.evaluate(() => window.__keplerComputeTest.getState().publications.map(source => source.id));
  const summaries = [];
  for (const [runIndex, run] of runs.entries()) {
    stage(`run ${runIndex}: ${run.predicate} weight=${run.weight} rank=${run.rank}`);
    await page.locator(`.dataset-ops-menu__toggle[data-for="dataset-ops-${pointId}"]`).click();
    await page.locator('.dataset-ops-menu__spatial-join').click();
    const panel = page.locator('.dataset-ops-panel');
    await panel.waitFor();
    await selectNativeOption(page, panel.locator('.data-source-selector').last(), polygonLabel);
    if (run.predicate !== 'Intersects') {
      await panel.locator('.item-selector').filter({hasText: 'Intersects'}).locator('.item-selector__dropdown').click();
      await page.getByText(run.predicate, {exact: true}).last().click();
    }
    await panel.getByText(/aggregation rules/i).click();
    await assertMergeChecked(panel);
    await setAggregation(panel, 'weight', run.weight);
    await setAggregation(panel, 'rank', run.rank);
    const resultLabel = `Spatial join result ${runIndex}`;
    await panel.locator('input[type=text]').last().fill(resultLabel);
    await panel.getByRole('button', {name: 'Run', exact: true}).click();
    await page.waitForFunction(ids => window.__keplerComputeTest.getState().publications.some(source => !ids.includes(source.id)) || /Spatial Join/i.test(document.querySelector('.dataset-ops-panel')?.textContent ?? '') && document.querySelector('.dataset-ops-panel [class*=error]'), knownIds, {timeout: 40000});
    const resultId = await page.evaluate(ids => window.__keplerComputeTest.getState().publications.find(source => !ids.includes(source.id))?.id, knownIds);
    assert.ok(resultId, `run ${runIndex} did not publish: ${(await panel.innerText().catch(() => '')).slice(-600)}`);
    knownIds.push(resultId);
    const state = await page.evaluate(id => window.__keplerComputeTest.getState().publications.find(source => source.id === id), resultId);
    assert.equal(state.rowCount, pointRows.length);
    const table = await exportArrow(page, resultId);
    const names = table.schema.fields.map(field => field.name);
    const expectedNames = ['longitude', 'latitude', 'amount', 'tag', 'count', `weight_${run.weight}`, `rank_${run.rank}`];
    assert.deepEqual(names.filter(name => name !== 'geometry_merge' && name !== '__luma_source_row_index'), expectedNames);
    assert.ok(names.includes('geometry_merge'), `merged geometry column missing: ${names}`);
    const lineage = JSON.parse(table.schema.metadata.get('luma.gl.spatialJoin') ?? 'null');
    assert.equal(lineage?.predicate, run.id);
    const rows = [...table].map(row => expectedNames.map(name => row[name]));
    const mergedRows = [...table].map(row => row['geometry_merge']);
    assert.equal(rows.length, pointRows.length);
    let matchedRows = 0;
    for (let rowIndex = 0; rowIndex < pointRows.length; rowIndex++) {
      const expected = expectedRow(rowIndex, run);
      const actual = rows[rowIndex];
      if (expected[4]) matchedRows++;
      const expectedMerge = getMergedGeometry(matchPolygons(...pointRows[rowIndex].slice(0, 2), run.id));
      const actualMerge = mergedRows[rowIndex] === null ? null : JSON.parse(mergedRows[rowIndex]);
      assert.deepEqual(actualMerge, expectedMerge, `run ${runIndex} row ${rowIndex} merged geometry`);
      for (let column = 0; column < 7; column++)
        assert.ok(column === 3 ? actual[column] === expected[column] : isClose(actual[column], expected[column]),
          `run ${runIndex} row ${rowIndex} column ${expectedNames[column]}: ${actual[column]} vs ${expected[column]} (point ${pointRows[rowIndex].slice(0, 2)})`);
    }
    summaries.push({run: runIndex, predicate: run.id, matchedRows});
  }
  assert.deepEqual(sqlRequests, []);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, summaries, checks: [
    'GPU Spatial Join over overlapping polygons, a hole, a multipolygon and boundary points',
    'intersects (closed) and within (open) predicates, five aggregation techniques',
    'null left values, string dictionary gather and null-only right aggregates',
    'default merge aggregation of the polygon geometry column equals native mergeGeometries',
    'no SQL runtime started'
  ]}));
} catch (error) {
  console.error(JSON.stringify({stage: currentStage, errors, sqlRequests,
    body: (await page.locator('body').innerText().catch(() => '')).slice(-3000)}));
  throw error;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
