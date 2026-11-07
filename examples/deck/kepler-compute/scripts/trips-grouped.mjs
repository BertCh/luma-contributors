// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {openNativeData, openNativeExport} from './native-navigation.mjs';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const origin = Date.UTC(2026, 0, 1);
const columns = (ids, seconds, longitudes, latitudes, measures) => ({
  trip: ids,
  time: new Float64Array(seconds.map(second => origin + second * 1000)),
  lon: new Float64Array(longitudes),
  lat: new Float64Array(latitudes),
  measure: new Float32Array(measures)
});
// Trip A = rows 0, 1, 3 and trip B = rows 2, 4, 5, deliberately interleaved across two batches.
const source = arrow.tableFromArrays(columns(['A', 'A', 'B'], [0, 10, 0], [-74.03, -74, -74.03], [40.74, 40.74, 40.77], [0, 1, 0]))
  .concat(arrow.tableFromArrays(columns(['A', 'B', 'B'], [20, 10, 20], [-73.97, -74, -73.97], [40.74, 40.77, 40.77], [1, 0, 0])));
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: 'grouped-trips.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))});
  await workspace.getByLabel('Grouped timed rows (Trips)', {exact: true}).check();
  await workspace.getByLabel('Trip ID column', {exact: true}).selectOption('trip');
  await workspace.getByLabel('Trip time column', {exact: true}).selectOption('time');
  await workspace.getByLabel('Trip longitude column', {exact: true}).selectOption('lon');
  await workspace.getByLabel('Trip latitude column', {exact: true}).selectOption('lat');
  await workspace.getByLabel('Trip timestamp encoding', {exact: true}).selectOption('unix-millisecond');
  await workspace.getByLabel('Trip time origin (epoch milliseconds)', {exact: true}).fill(String(origin));
  // Float64 role columns stay in the retained source; exclude them from scalar upload like the UI directs.
  for (const field of ['time', 'lon', 'lat']) await workspace.getByLabel(`${field} (Float64)`, {exact: true}).uncheck();
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByText(/2 timed grouped trips, 6 vertices/).waitFor();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.trips?.tripCount === 2, undefined, {timeout: 45000});
  const initial = await page.evaluate(() => window.__keplerComputeTest.getState());
  const datasetId = initial.publications[0].id;
  assert.deepEqual(initial.publications[0].sourceBatches, [3, 3]);
  assert.deepEqual(initial.publications[0].trips.domain, [origin, origin + 20000]);
  assert.equal(initial.publications[0].error, null);
  await page.waitForFunction(datasetId => {
    const api = window.__keplerComputeTest;
    const state = api.getState();
    if (state.error) throw new Error(state.error);
    const native = state.nativeLayers.find(layer => layer.id === `${datasetId}-trips`);
    if (native?.valid === false) throw new Error(JSON.stringify(native));
    return api.renderedLayers().some(layer => layer.id === `${datasetId}-trips` && layer.visible && layer.loaded);
  }, datasetId);
  const pick = (longitude, latitude) => page.evaluate(([id, x, y]) => window.__keplerComputeTest.pickTripPosition(id, x, y), [datasetId, longitude, latitude]);
  await page.evaluate(time => window.__keplerComputeTest.setAnimationTime(time), origin + 20000);
  // Primary source row is the first valid row of each trip: A -> row 0, B -> row 2.
  // The first pick after an animation-time change can precede the redrawn frame; poll briefly.
  let firstPick;
  for (let attempt = 0; attempt < 50 && firstPick?.sourceRowIndex !== 0; attempt++) {
    firstPick = await pick(-74.015, 40.74);
    if (firstPick?.sourceRowIndex !== 0) await page.waitForTimeout(100);
  }
  assert.equal(firstPick?.sourceRowIndex, 0);
  // Select rows 1 and 3 only. Native drops row 0 from trip A, so its first segment disappears.
  await page.evaluate(id => window.__keplerComputeTest.setPublishedRange(id, 'measure', 1, 1), datasetId);
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0].selectedCount === 2);
  assert.equal(await pick(-74.015, 40.74), null, 'Unselected first vertex is dropped from the path');
  assert.equal((await pick(-73.985, 40.74))?.sourceRowIndex, 0, 'Trip stays visible while any member row is selected');
  await page.evaluate(id => window.__keplerComputeTest.setPublishedRange(id, 'measure', 0, 0), datasetId);
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0].selectedCount === 4);
  assert.equal(await pick(-74.015, 40.74), null, 'Trip A keeps one selected vertex, so it has no drawn segment');
  assert.equal((await pick(-74.015, 40.77))?.sourceRowIndex, 2, 'Trip B spans both source batches');
  const final = await page.evaluate(() => window.__keplerComputeTest.getState());
  assert.equal(final.error, '');
  assert.equal(final.publications[0].trips.sourceUploadBytes, initial.publications[0].trips.sourceUploadBytes);
  assert.deepEqual(final.publications[0].trips.sourceBufferIds, initial.publications[0].trips.sourceBufferIds);
  assert.equal(final.uploadBytes, initial.uploadBytes);
  await openNativeExport(page, true);
  const sandbox = page.getByRole('region', {name: 'Save map', exact: true});
  await sandbox.getByRole('button', {name: 'Save state', exact: true}).click();
  const saved = JSON.parse(await sandbox.getByLabel('Saved sandbox state', {exact: true}).inputValue());
  const parameters = saved.sources.find(source => source.dataset.datasetId === datasetId).reference.parameters;
  assert.equal(parameters.geometryColumn, 'trip');
  assert.deepEqual(parameters.tripOptions, {
    timestampEncoding: 'unix-millisecond', originMilliseconds: origin, maximumTimeErrorMilliseconds: 0,
    grouping: {timestampColumn: 'time', longitudeColumn: 'lon', latitudeColumn: 'lat'}
  });
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, sourceBatches: [3, 3], checks: ['grouped UI roles', 'cross-batch grouping', 'any-row visibility', 'native per-vertex filtering', 'stable source allocations', 'saved grouping']}));
} catch (error) {
  console.error(await page.evaluate(() => window.__keplerComputeTest?.getState()));
  console.error(await page.locator('body').innerText());
  throw error;
} finally {
  await browser.close();
}
