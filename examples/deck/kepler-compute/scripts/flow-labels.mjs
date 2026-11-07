// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// FLOW2: flowmap location labels, scale lock and filter-driven re-aggregation of the WGSL flowmap.
// Labels reuse the point-label GPU text path; the scalar filter stands in for a Kepler time filter
// because both drive the same resident layer mask that feeds the GPU origin-destination aggregation.
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {openPublishedTable} from './flow-field-utils.mjs';

const names = ['Alpha', 'Bravo', 'Charlie', 'Delta'];
const places = [[40.72, -74.0], [40.76, -73.95], [40.8, -73.9], [40.68, -73.98]];
const rows = 48;
const column = (map) => Array.from({length: rows}, (_, row) => map(row));
const origin = column(row => row % 4);
const destination = column(row => (row + 1 + (row % 3)) % 4);
const table = new arrow.Table({
  lat0: arrow.makeVector(new Float64Array(origin.map(index => places[index][0]))),
  lon0: arrow.makeVector(new Float64Array(origin.map(index => places[index][1]))),
  lat1: arrow.makeVector(new Float64Array(destination.map(index => places[index][0]))),
  lon1: arrow.makeVector(new Float64Array(destination.map(index => places[index][1]))),
  count: arrow.makeVector(new Float32Array(column(row => 1 + (row % 5)))),
  hour: arrow.makeVector(new Float32Array(column(row => row))),
  sourceName: arrow.vectorFromArray(origin.map(index => names[index])),
  targetName: arrow.vectorFromArray(destination.map(index => names[index]))
});
const {browser, page, errors, datasetId, applyLayer} = await openPublishedTable({
  table,
  fileName: 'flow-labels.arrow',
  url: process.argv[2],
  // The resident flow runtime binds only to a source prepared with the Flow map roles.
  selectColumns: async workspace => {
    await workspace.getByLabel('Source longitude column').selectOption('lon0');
    await workspace.getByLabel('Source latitude column').selectOption('lat0');
    await workspace.getByLabel('Flow target longitude column').selectOption('lon1');
    await workspace.getByLabel('Flow target latitude column').selectOption('lat1');
    await workspace.getByLabel('Flow count column').selectOption('count');
    await workspace.getByLabel('Flow source name column').selectOption('sourceName');
    await workspace.getByLabel('Flow target name column').selectOption('targetName');
  }
});
const mapState = {longitude: -73.95, latitude: 40.74, zoom: 10, pitch: 0, bearing: 0};
const columns = {lat0: 'lat0', lng0: 'lon0', lat1: 'lat1', lng1: 'lon1', count: 'count', sourceName: 'sourceName', targetName: 'targetName'};
const stats = layerId => page.evaluate(id => window.__keplerComputeTest.getFlowmapStats()[id], layerId);

/** Host-only flow switches are not part of the saved Kepler config, so they are toggled in the layer panel. */
async function setSwitch(property, enabled) {
  const input = page.locator(`[id="resident-flow-labels-${property}-switch"]`);
  if (!(await input.count())) await page.locator('.layer-panel__header').first().locator('.layer__enable-config, [class*=enable-config]').first().click();
  await input.first().waitFor({state: 'attached', timeout: 15000});
  if ((await input.first().isChecked()) !== enabled) await input.first().evaluate(element => element.click());
}
async function waitStats(layerId, predicate, description) {
  try {
    await page.waitForFunction(({id, source}) => {
      const value = window.__keplerComputeTest.getFlowmapStats()[id];
      return value && new Function('stats', `return (${source})(stats)`)(value);
    }, {id: layerId, source: predicate.toString()}, {timeout: 60000});
  } catch (error) {
    console.error(description, JSON.stringify(await stats(layerId)));
    throw error;
  }
}
try {
  // Native default: labels off.
  await applyLayer('resident-flow-labels', 'flow', {columns, mapState}, {opacity: 1}, 'LAT_LNG');
  await waitStats('resident-flow-labels', s => s.flowCount > 0 && s.locationCount > 0, 'flow ready');
  const base = await stats('resident-flow-labels');
  assert.equal(base.labelCount, 0, 'labels are off by default, as in Kepler');

  // Labels on: one GPU text label per in-view location, drawn by the point-label layer.
  await applyLayer('resident-flow-labels', 'flow', {columns, mapState}, {opacity: 1}, 'LAT_LNG');
  await setSwitch('flowLocationLabelsEnabled', true);
  await waitStats('resident-flow-labels', s => s.labelCount > 0, 'labels admitted');
  await page.waitForFunction(() => window.__keplerComputeTest.renderedLayers().some(layer => layer.layerClass === 'FlowmapGPULayer' && layer.visible), undefined, {timeout: 60000});
  const labeled = await stats('resident-flow-labels');
  assert(labeled.labelCount > 0 && labeled.labelCount <= labeled.locationCount, `labels ${labeled.labelCount} of ${labeled.locationCount}`);
  assert(!labeled.labelDiagnostic, `label diagnostic: ${labeled.labelDiagnostic}`);

  // Scale lock freezes the magnitude domain, so filtering rows keeps the domain.
  await applyLayer('resident-flow-labels', 'flow', {columns, mapState}, {opacity: 1}, 'LAT_LNG');
  await setSwitch('flowLocationLabelsEnabled', true);
  await setSwitch('flowScaleLockEnabled', true);
  await waitStats('resident-flow-labels', s => s.scaleLocked, 'scale locked');

  // Filter-driven re-aggregation: keep the first half of the rows by "hour".
  await applyLayer('resident-flow-labels', 'flow', {columns, mapState}, {opacity: 1}, 'LAT_LNG');
  await setSwitch('flowScaleLockEnabled', false);
  await waitStats('resident-flow-labels', s => s.flowCount > 0, 'flow ready again');
  const before = await stats('resident-flow-labels');
  await page.evaluate(([id]) => window.__keplerComputeTest.setPublishedRange(id, 'hour', 0, 5), [datasetId]);
  await waitStats('resident-flow-labels', s => s.flowCount > 0, 'flow after filter');
  await page.waitForFunction(({before}) => {
    const value = Object.values(window.__keplerComputeTest.getFlowmapStats())[0];
    return value && value.flowCount < before;
  }, {before: before.flowCount}, {timeout: 60000});
  const filtered = await stats('resident-flow-labels');
  assert(filtered.flowCount < before.flowCount, `filter re-aggregates flows (${before.flowCount} -> ${filtered.flowCount})`);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({base, labeled, before, filtered}, null, 2));
} finally {
  await browser.close();
}
