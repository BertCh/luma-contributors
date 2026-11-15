// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Usage: node scripts/raster-embeddings-lagrangian.mjs [app-url] [embedding-geotiff-url]
// Run from examples/deck/kepler-compute with its dev server already running (default :5318).
// Exercises native embedding publication, every generated-fixture operation, and forward/backward
// flow-map + FTLE publication and numerical validation from an active native Flow Field grid. Optional GeoTIFF
// URL is loaded through the existing COG reader and its first tile runs usable-mask (float bands).
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {openPublishedTable} from './flow-field-utils.mjs';

const longitudes = [];
const latitudes = [];
for (let row = 0; row < 8; row++) for (let column = 0; column < 8; column++) {
  longitudes.push(-74.1 + column * 0.01);
  latitudes.push(40.6 + row * 0.01);
}
const table = arrow.tableFromArrays({longitude: longitudes, latitude: latitudes,
  u: new Float32Array(64).fill(2), v: new Float32Array(64).fill(1)});
const {browser, page, errors, applyLayer} = await openPublishedTable({table, fileName: 'diagnostic-flow.arrow', url: process.argv[2]});
const watchdog = setTimeout(() => {console.error('Embedding/FTLE script exceeded 180 seconds'); void browser.close();}, 180000);
try {
  const panel = page.getByTestId('raster-embeddings-panel');
  await panel.locator('summary').click();
  await panel.getByRole('button', {name: 'Publish embedding result', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[data-testid="raster-embeddings-status"]')?.textContent?.includes('GPURasterEmbeddingSimilarity'), undefined, {timeout: 60000});
  const embeddings = await page.evaluate(async () => {
    const RASTER_EMBEDDING_OPERATIONS = ['kmeans', 'nearest-prototype', 'nearest-pixels', 'knn', 'linear-probe', 'change', 'covariance', 'pca', 'projection', 'dequantize', 'quantize', 'pyramid', 'similarity', 'zone-mean', 'usable-mask'];
    const results = [];
    for (const operation of RASTER_EMBEDDING_OPERATIONS) results.push({operation, ...await window.__keplerComputeTest.runRasterEmbeddingAction({sourceId: 'fixture', options: {operation}, demo: true, publish: false})});
    return results;
  });
  assert.equal(embeddings.length, 15);
  for (const result of embeddings) {
    assert(result.rowCount > 0, result.operation);
    assert(result.contributors.length > 0, result.operation);
  }
  if (process.argv[3]) {
    const loaded = await page.evaluate(url => window.__keplerComputeTest.loadRasterEmbeddingGeoTiff(url), process.argv[3]);
    assert(loaded.bandCount > 0);
    const result = await page.evaluate(async () => {
      const tile = window.__keplerComputeTest.getRasterEmbeddingTiles('loaded-geotiff')[0];
      assertTile(tile);
      function assertTile(value) {if (!value) throw new Error('GeoTIFF has no loaded tiles');}
      return window.__keplerComputeTest.runRasterEmbeddingAction({sourceId: 'loaded-geotiff', tileId: tile.id, options: {operation: 'usable-mask'}, publish: true});
    });
    assert(result.rowCount > 0);
  }
  const layerId = 'lagrangian-velocity-source';
  await applyLayer(layerId, 'flowField', {columns: {lat: 'latitude', lng: 'longitude', u: 'u', v: 'v'}, mapState: {longitude: -74.06, latitude: 40.64, zoom: 11}}, {gridResolution: 8, smoothing: 1, linesPerScreen: 100}, 'UV');
  // A previous revision's build count can be nonzero while current filters are still settling.
  // Wait for the loaded render layer (as flow-field.mjs does) and the actual current-grid signal.
  await page.waitForFunction(id => {
    const api = window.__keplerComputeTest;
    const state = api.getState();
    if (state.error) throw new Error(state.error);
    return api.getFlowFieldStats()[id]?.gridReady && api.renderedLayers().some(layer => layer.id === `${id}-resident-flow-field` && layer.visible && layer.loaded);
  }, layerId, {timeout: 60000});
  const flows = [];
  for (const timeStep of [1, -1]) {
    const result = await page.evaluate(request => window.__keplerComputeTest.runFlowLagrangianAction(request), {layerId, timeStep, stepCount: 4});
    assert.equal(result.validCount, result.rowCount, 'All inset seeds and FTLE stencils must remain valid in the filled constant-velocity grid');
    assert.deepEqual(result.contributors, ['GPULagrangianFlowMap', 'GPUFiniteTimeLyapunovExponent']);
    const columns = await page.evaluate(() => {
      const api = window.__keplerComputeTest;
      const publication = api.getState().publications.at(-1);
      if (!publication) throw new Error('Flow diagnostic table was not published');
      return Object.fromEntries(['geometry', 'final_longitude', 'final_latitude', 'ftle', 'flow_valid', 'ftle_valid'].map(name => [name, api.readRasterPolygonColumn(publication.id, name)]));
    });
    for (const [name, values] of Object.entries(columns)) assert.equal(values?.length, result.rowCount, `Published ${name} column must contain every seed`);
    // Ground velocity is u=2 m/s east, v=1 m/s north, for T=±4 s: displacement (±8, ±4) m.
    // Convert published degree displacement to seed-local ground metres independently of the
    // adapter. Allow 0.75 m for native Float32 coordinate narrowing near 74°W and RK2/resampling.
    // A uniform translation has FTLE=0 s^-1. Mercator scale varies with latitude: its small
    // analytic strain is about 2.2e-7 s^-1 here; 2e-6 s^-1 also covers Float32 stencil error.
    const integrationSeconds = timeStep * 4;
    const earthRadiusMeters = 6378137;
    const radiansPerDegree = Math.PI / 180;
    const displacementToleranceMeters = 0.75;
    const ftleTolerancePerSecond = 2e-6;
    let maximumDisplacementErrorMeters = 0;
    let maximumAbsoluteFtle = 0;
    for (let index = 0; index < result.rowCount; index++) {
      assert.equal(columns.flow_valid[index], 1, `flow validity of seed ${index}`);
      assert.equal(columns.ftle_valid[index], 1, `FTLE validity of seed ${index}`);
      const seed = JSON.parse(columns.geometry[index]).coordinates;
      const eastMeters = (columns.final_longitude[index] - seed[0]) * radiansPerDegree * earthRadiusMeters * Math.cos(seed[1] * radiansPerDegree);
      const northMeters = (columns.final_latitude[index] - seed[1]) * radiansPerDegree * earthRadiusMeters;
      const expectedEastMeters = 2 * integrationSeconds;
      const expectedNorthMeters = integrationSeconds;
      assert(Number.isFinite(eastMeters) && Math.abs(eastMeters - expectedEastMeters) <= displacementToleranceMeters,
        `seed ${index}: east displacement ${eastMeters} m; expected ${expectedEastMeters} m (=2 m/s × ${integrationSeconds} s), tolerance ±${displacementToleranceMeters} m`);
      assert(Number.isFinite(northMeters) && Math.abs(northMeters - expectedNorthMeters) <= displacementToleranceMeters,
        `seed ${index}: north displacement ${northMeters} m; expected ${expectedNorthMeters} m (=1 m/s × ${integrationSeconds} s), tolerance ±${displacementToleranceMeters} m`);
      const ftle = columns.ftle[index];
      assert(Number.isFinite(ftle) && Math.abs(ftle) <= ftleTolerancePerSecond,
        `seed ${index}: FTLE ${ftle} s^-1; expected near 0 s^-1 for constant ground velocity, tolerance ±${ftleTolerancePerSecond} s^-1 including Mercator strain`);
      maximumDisplacementErrorMeters = Math.max(maximumDisplacementErrorMeters, Math.abs(eastMeters - expectedEastMeters), Math.abs(northMeters - expectedNorthMeters));
      maximumAbsoluteFtle = Math.max(maximumAbsoluteFtle, Math.abs(ftle));
    }
    flows.push({...result, integrationSeconds, maximumDisplacementErrorMeters, maximumAbsoluteFtle});
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({embeddings, flows}, null, 2));
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
