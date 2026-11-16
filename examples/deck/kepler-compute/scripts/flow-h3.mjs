// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser verification of Flow in H3 endpoint mode: cell centers become endpoints, flows aggregate
// per origin-destination pair, and the native FlowmapLayer renders with picking and zoom clustering.
import {setAnalysisSheetParameter} from './analysis-sheet-controls.mjs';
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {latLngToCell, cellToLatLng} from 'h3-js';
import {runNetworkCategoricalCatalog} from './network-categorical-catalog-utils.mjs';
import {openPublishedTable} from './flow-field-utils.mjs';

const cells = [
  [40.72, -74.0],
  [40.76, -73.95],
  [40.8, -73.9],
  [40.68, -73.98]
].map(([lat, lng]) => BigInt(`0x${latLngToCell(lat, lng, 7)}`));
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
const {browser, page, errors, applyLayer} = await openPublishedTable({
  table,
  fileName: 'flow-h3.arrow',
  url: process.argv[2],
  selectColumns: async workspace => workspace.getByLabel('Source H3 column').selectOption('origin')
});
try {
  await applyLayer(
    'resident-flow-h3',
    'flow',
    {
      columns: {sourceH3: 'origin', targetH3: 'destination', count: 'count'},
      mapState: {longitude: -73.95, latitude: 40.74, zoom: 10, pitch: 0, bearing: 0}
    },
    {opacity: 1},
    'H3'
  );
  await page.waitForFunction(
    () => {
      const state = window.__keplerComputeTest.getFlowH3States()['resident-flow-h3'];
      if (state?.state === 'failed') throw new Error(state.reason);
      return state?.state === 'ready' && state.flowCount > 0;
    },
    undefined,
    {timeout: 60000}
  );
  const state = await page.evaluate(
    () => window.__keplerComputeTest.getFlowH3States()['resident-flow-h3']
  );
  // 4 cells produce at most 12 directed pairs (no self pairs).
  assert(state.flowCount > 0 && state.flowCount <= 12, `aggregated H3 flows: ${state.flowCount}`);
  await page.waitForFunction(
    () =>
      window.__keplerComputeTest
        .renderedLayers()
        .some(layer => layer.id.startsWith('resident-flow-h3') && layer.visible && layer.loaded),
    undefined,
    {timeout: 60000}
  );
  // Reuse these H3 centers as catalog zones: smooth the same aggregated OD pairs on the library.
  const coordinates = cells.map(cell => cellToLatLng(cell.toString(16)).reverse());
  // Include a singleton MultiPoint centroid in the real UI flow run.
  const zones = {
    type: 'FeatureCollection',
    features: coordinates.map((position, index) => ({
      id: String(index),
      geometry:
        index === 0
          ? {type: 'MultiPoint', coordinates: [position]}
          : {type: 'Point', coordinates: position},
      properties: {population: 1}
    }))
  };
  const pairs = new Map();
  origin.forEach((source, index) => {
    const from = cells.indexOf(source);
    const to = cells.indexOf(destination[index]);
    if (from !== to) pairs.set(`${from}:${to}`, (pairs.get(`${from}:${to}`) ?? 0) + count[index]);
  });
  const flows = {
    type: 'FeatureCollection',
    features: Array.from(pairs, ([key, weight]) => {
      const [from, to] = key.split(':').map(Number);
      return {
        geometry: {type: 'LineString', coordinates: [coordinates[from], coordinates[to]]},
        properties: {origin: String(from), destination: String(to), weight}
      };
    })
  };
  const panel = page.getByRole('region', {name: 'Analysis catalog', exact: true});
  await panel.getByLabel('Operation', {exact: true}).selectOption('kernel-flow-smoothing');
  await panel.getByLabel('Zone centroids source', {exact: true}).selectOption('text');
  await panel.getByLabel('Zone centroids GeoJSON', {exact: true}).fill(JSON.stringify(zones));
  await panel.getByLabel('Unique OD flows source', {exact: true}).selectOption('text');
  await panel.getByLabel('Unique OD flows GeoJSON', {exact: true}).fill(JSON.stringify(flows));
  await setAnalysisSheetParameter(panel, 'neighborhoodSize', 0);
  await runNetworkCategoricalCatalog(page, panel, 'GPUKernelFlowSmoothing');
  assert.match(
    await panel.getByTestId('catalog-result').innerText(),
    /Kernel flow smoothing: 16 rows/
  );
  await panel.getByRole('button', {name: 'Serialize invocation', exact: true}).click();
  const invocation = JSON.parse(
    await panel.getByLabel('Serialized invocation', {exact: true}).inputValue()
  );
  assert.equal(invocation.inputs.zones.featureCount, 4);
  assert.equal(invocation.inputs.flows.featureCount, pairs.size);
  assert.equal(invocation.parameters.neighborhoodSize, 0);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({state}, null, 2));
} finally {
  await browser.close();
}
