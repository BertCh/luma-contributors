// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser verification (headless by default): Flow Field positions from a GeoJSON column (centroid
// of all vertices) bind to the resident GPU path, and visConfig.debugGrid renders the PathLayer and
// PolygonLayer overlay without counting against gridReadbackCount.
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {openPublishedTable} from './flow-field-utils.mjs';

const SIDE = 24;
const geometry = [];
const u = [];
const v = [];
for (let row = 0; row < SIDE; row++)
  for (let column = 0; column < SIDE; column++) {
    const longitude = -74.1 + column * 0.01;
    const latitude = 40.65 + row * 0.01;
    // Small square: the centroid (average of 5 vertices incl. the closing one) offsets slightly.
    geometry.push(JSON.stringify({type: 'Polygon', coordinates: [[[longitude - 0.001, latitude - 0.001], [longitude + 0.001, latitude - 0.001], [longitude + 0.001, latitude + 0.001], [longitude - 0.001, latitude + 0.001], [longitude - 0.001, latitude - 0.001]]]}));
    u.push(Math.sin(column / 4) * 5);
    v.push(Math.cos(row / 4) * 5);
  }
const table = arrow.tableFromArrays({geometry, u: Float64Array.from(u), v: Float64Array.from(v)});
const {browser, page, errors, datasetId, applyLayer} = await openPublishedTable({table, fileName: 'flow-field-centroids.arrow', url: process.argv[2]});
try {
  const mapState = {longitude: -73.98, latitude: 40.77, zoom: 11, pitch: 0, bearing: 0};
  const id = 'resident-flow-field-centroid';
  await applyLayer(id, 'flowField', {columns: {geojson: 'geometry', u: 'u', v: 'v'}, color: [186, 233, 223], mapState}, {gridResolution: SIDE, smoothing: 1, debugGrid: true}, 'UV');
  await page.waitForFunction(identifier => window.__keplerComputeTest.getFlowFieldStats()[identifier], id, {timeout: 60000});
  const state = await page.evaluate(identifier => window.__keplerComputeTest.getFlowFieldStats()[identifier], id);
  assert.equal(state.gridReadbackCount, 0, 'debug readbacks are counted separately');
  await page.waitForFunction(identifier => window.__keplerComputeTest.getDeckLayerIds().filter(layer => layer.startsWith(identifier)).length >= 2, id, {timeout: 30000});
  const layerIds = await page.evaluate(() => window.__keplerComputeTest.getDeckLayerIds());
  assert(layerIds.some(layer => layer.endsWith('-debug-grid')) && layerIds.some(layer => layer.endsWith('-debug-filled')), 'debug grid layers present');
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, state, layerIds}, null, 2));
} finally {
  await browser.close();
}
