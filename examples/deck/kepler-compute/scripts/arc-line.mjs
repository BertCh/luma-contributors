// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser verification of Arc and Line on a PUBLISHED tabular dataset (not the built-in sample):
// endpoint binding, filtering, target arrowheads, and specific reasons for altitude and
// neighbor/GeoArrow column modes.
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {PNG} from 'pngjs';
import {openPublishedTable} from './flow-field-utils.mjs';

const count = 64;
const table = arrow.tableFromArrays({
  latitude: Float64Array.from({length: count}, (_, i) => 40.7 + (i % 8) * 0.02),
  longitude: Float64Array.from({length: count}, (_, i) => -74.05 + Math.floor(i / 8) * 0.02),
  targetLatitude: Float64Array.from({length: count}, (_, i) => 40.8 - (i % 8) * 0.01),
  targetLongitude: Float64Array.from({length: count}, (_, i) => -73.9 + Math.floor(i / 8) * 0.01),
  altitude: Float64Array.from({length: count}, (_, i) => i * 10),
  measure: Float64Array.from({length: count}, (_, i) => i)
});
const {browser, page, errors, applyLayer} = await openPublishedTable({table, fileName: 'arc-line.arrow', url: process.argv[2],
  // Arc/Line endpoints are published through the flow target roles (origins are the source coordinates).
  selectColumns: async workspace => {
    await workspace.getByLabel('Flow target longitude column').selectOption('targetLongitude');
    await workspace.getByLabel('Flow target latitude column').selectOption('targetLatitude');
  }
});
const columns = {lat0: 'latitude', lng0: 'longitude', lat1: 'targetLatitude', lng1: 'targetLongitude'};
const mapState = {longitude: -73.97, latitude: 40.75, zoom: 10.5, pitch: 0, bearing: 0};
const results = [];
const waitRendered = (identifier, classPattern) => page.waitForFunction(({identifier, classPattern}) => {
  const api = window.__keplerComputeTest;
  if (api.getState().error) throw new Error(api.getState().error);
  return api.renderedLayers().some(layer => layer.id.startsWith(identifier) && layer.visible && layer.loaded && new RegExp(classPattern).test(layer.layerClass));
}, {identifier, classPattern}, {timeout: 60000});
const reasons = () => page.evaluate(() => (window.__keplerComputeTest.getState().error ?? '').split('\n').filter(Boolean));
try {
  for (const type of ['line', 'arc']) {
    await applyLayer(`resident-${type}`, type, {columns, color: [86, 203, 249], mapState}, {thickness: 4, opacity: 1, targetArrowEnabled: true, targetArrowShape: 'triangle', targetArrowSize: 20, targetColor: [250, 120, 40]}, 'points');
    await waitRendered(`resident-${type}`, type === 'line' ? 'GPULineLayer|GPUArcLayer' : 'GPUArcLayer');
    await waitRendered(`resident-${type}`, 'GPUArrowHeadLayer');
    const screenshot = PNG.sync.read(await page.screenshot());
    let colored = 0;
    for (let i = 0; i < screenshot.data.length; i += 4) if (Math.max(screenshot.data[i], screenshot.data[i + 1], screenshot.data[i + 2]) > 150 && screenshot.data[i + 2] - screenshot.data[i] > 60) colored++;
    assert(colored > 300, `${type} must draw visible strokes; found ${colored}`);
    results.push({type, colored});
  }
  // Specific unsupported reasons.
  await applyLayer('resident-line-altitude', 'line', {columns: {...columns, alt0: 'altitude', alt1: 'altitude'}, mapState}, {}, 'points');
  await page.waitForFunction(() => /Line altitude/.test(window.__keplerComputeTest.getState().error ?? ''), undefined, {timeout: 30000});
  await applyLayer('resident-arc-neighbors', 'arc', {columns: {lat: 'latitude', lng: 'longitude', neighbors: 'measure'}, mapState}, {}, 'neighbors');
  await page.waitForFunction(() => /neighbor/.test(window.__keplerComputeTest.getState().error ?? ''), undefined, {timeout: 30000});
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({results, reasons: await reasons()}, null, 2));
} finally {
  await browser.close();
}
