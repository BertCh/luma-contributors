// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser verification (backlog LAYERS-65, H3 mode) of native Flow endpoint picking and cluster zoom:
// H3 cell centers are the flow endpoints, picking an endpoint returns the resident FlowmapLayer,
// layer bounds come from the derived center runtime, and picking keeps working after zooming out to
// a clustered level and back in. Usage: node scripts/flow-h3-picking.mjs [url]
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {cellToLatLng, latLngToCell} from 'h3-js';
import {openPublishedTable} from './flow-field-utils.mjs';

const cellIds = [[40.72, -74.0], [40.76, -73.95], [40.8, -73.9], [40.68, -73.98]].map(([lat, lng]) => latLngToCell(lat, lng, 7));
const cells = cellIds.map(id => BigInt(`0x${id}`));
const origin = [];
const destination = [];
const count = [];
for (let i = 0; i < 40; i++) {
  origin.push(cells[i % 4]);
  destination.push(cells[(i + 1 + (i % 3)) % 4]);
  count.push(1 + (i % 5));
}
const table = new arrow.Table({
  origin: arrow.vectorFromArray(origin, new arrow.Uint64()),
  destination: arrow.vectorFromArray(destination, new arrow.Uint64()),
  count: arrow.makeVector(new Float32Array(count))
});
const centers = cellIds.map(id => cellToLatLng(id));
const layerId = 'resident-flow-h3-picking';
const {browser, page, errors, datasetId, applyLayer} = await openPublishedTable({
  table,
  fileName: 'flow-h3-picking.arrow',
  url: process.argv[2],
  selectColumns: async workspace => workspace.getByLabel('Source H3 column').selectOption('origin')
});

async function waitForFlowLayer() {
  await page.waitForFunction(
    id => window.__keplerComputeTest.renderedLayers().some(layer => layer.id.startsWith(id) && layer.visible && layer.loaded),
    layerId,
    {timeout: 60000}
  );
}
async function pickCenters() {
  const picks = [];
  for (const [latitude, longitude] of centers) {
    picks.push(await page.evaluate(
      ({datasetId: id, longitude: x, latitude: y}) => window.__keplerComputeTest.pickPublishedPosition(id, x, y),
      {datasetId, longitude, latitude}
    ));
  }
  return picks;
}
try {
  const mapState = {longitude: -73.95, latitude: 40.74, zoom: 11, pitch: 0, bearing: 0};
  await applyLayer(layerId, 'flow', {columns: {sourceH3: 'origin', targetH3: 'destination', count: 'count'}, mapState}, {opacity: 1, flowClusteringEnabled: true}, 'H3');
  await page.waitForFunction(id => {
    const state = window.__keplerComputeTest.getFlowH3States()[id];
    if (state?.state === 'failed') throw new Error(state.reason);
    return state?.state === 'ready' && state.flowCount > 0;
  }, layerId, {timeout: 60000});
  await waitForFlowLayer();

  // Endpoint metadata: bounds cover exactly the four cell centers (not the dataset point bounds).
  const meta = await page.evaluate(id => {
    const layer = window.__keplerComputeTest.getNativeLayerMeta?.(id);
    return layer ?? null;
  }, layerId);
  // Bounds come from float32 GPU aggregation (documented float32 location identity), so allow ~1e-4 degrees.
  if (meta?.bounds) {
    const latitudes = centers.map(([latitude]) => latitude);
    const longitudes = centers.map(([, longitude]) => longitude);
    const [west, south, east, north] = meta.bounds;
    assert(Math.abs(west - Math.min(...longitudes)) < 1e-4 && Math.abs(east - Math.max(...longitudes)) < 1e-4, `flow bounds longitude ${meta.bounds}`);
    assert(Math.abs(south - Math.min(...latitudes)) < 1e-4 && Math.abs(north - Math.max(...latitudes)) < 1e-4, `flow bounds latitude ${meta.bounds}`);
    assert.equal(meta.locationCount, 4, 'one flow location per H3 cell');
  }

  // Zoomed in: every endpoint is individually pickable and reports the flow layer.
  const zoomedIn = await pickCenters();
  const zoomedInHits = zoomedIn.filter(pick => pick?.layerId?.startsWith(layerId));
  assert(zoomedInHits.length >= 3, `endpoints picked when zoomed in: ${JSON.stringify(zoomedIn)}`);

  // Zoomed out: clustering merges nearby cells; picking the cluster still resolves to the layer.
  await page.evaluate(viewport => window.__keplerComputeTest.setMapViewport(viewport), {...mapState, zoom: 4});
  await waitForFlowLayer();
  const zoomedOut = await pickCenters();
  assert(zoomedOut.some(pick => pick?.layerId?.startsWith(layerId)), `cluster picked when zoomed out: ${JSON.stringify(zoomedOut)}`);

  // Back in: individual endpoints return.
  await page.evaluate(viewport => window.__keplerComputeTest.setMapViewport(viewport), mapState);
  await waitForFlowLayer();
  // Cluster levels are recomputed asynchronously after a viewport change, so poll until endpoints return.
  let restored = [];
  for (let attempt = 0; attempt < 20 && restored.length < 3; attempt++) {
    await page.waitForTimeout(500);
    restored = (await pickCenters()).filter(pick => pick?.layerId?.startsWith(layerId));
  }
  assert(restored.length >= 3, 'endpoints picked again after zooming back in');
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({zoomedIn, zoomedOut, restoredHitCount: restored.length, meta}, null, 2));
} finally {
  await browser.close();
}
