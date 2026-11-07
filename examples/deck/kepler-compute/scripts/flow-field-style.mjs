// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// FLOW3-5: headless visual check of the Flow Field particle styling on a GPU: color by speed across a
// multi-color palette (screenshot color classes), elevation lift under pitch (streak centroid moves
// up the screen), and the debug grid layers, with no WebGPU validation errors (collected by the utils).
// Run headless (default): node scripts/flow-field-style.mjs [url]
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {PNG} from 'pngjs';
import {openPublishedTable} from './flow-field-utils.mjs';

const GRID_SIDE = 24;
const rows = [];
for (let row = 0; row < GRID_SIDE; row++)
  for (let column = 0; column < GRID_SIDE; column++) {
    const dx = column - 11.5;
    const dy = row - 11.5;
    const norm = Math.hypot(dx, dy) || 1;
    rows.push({
      longitude: -74.1 + column * 0.01,
      latitude: 40.65 + row * 0.01,
      u: (-dy / norm) * norm,
      v: (dx / norm) * norm,
      altitude: 200 - norm * 10
    });
  }
const table = arrow.tableFromArrays({
  longitude: Float64Array.from(rows, r => r.longitude),
  latitude: Float64Array.from(rows, r => r.latitude),
  u: Float64Array.from(rows, r => r.u),
  v: Float64Array.from(rows, r => r.v),
  altitude: Float64Array.from(rows, r => r.altitude)
});
const {browser, page, errors, applyLayer} = await openPublishedTable({table, fileName: 'flow-field-style.arrow', url: process.argv[2]});
const mapState = {longitude: -73.98, latitude: 40.77, zoom: 11, pitch: 0, bearing: 0};
const position = {lat: 'latitude', lng: 'longitude'};
const classes = {
  red: (r, g, b) => r > 180 && g < 90 && b < 90,
  yellow: (r, g, b) => r > 200 && g > 200 && b < 100,
  magenta: (r, g, b) => r > 180 && b > 180 && g < 90,
  white: (r, g, b) => r > 235 && g > 235 && b > 235
};
async function paint(matches) {
  const image = PNG.sync.read(await page.screenshot());
  let count = 0;
  let sumY = 0;
  // Skip the side panel and bottom controls: only the map area.
  for (let y = 0; y < image.height - 80; y++)
    for (let x = 330; x < image.width - 60; x++) {
      const offset = (y * image.width + x) * 4;
      if (matches(image.data[offset], image.data[offset + 1], image.data[offset + 2])) {
        count++;
        sumY += y;
      }
    }
  return {count, centroidY: count ? sumY / count : 0};
}
async function show(id, config, visConfig, mode) {
  await applyLayer(id, 'flowField', {...config, color: [186, 233, 223]}, {opacity: 1, strokeWidth: 4, linesPerScreen: 400, gridResolution: GRID_SIDE, smoothing: 1, ...visConfig}, mode);
  await page.waitForFunction(identifier => {
    const api = window.__keplerComputeTest;
    if (api.getState().error) throw new Error(api.getState().error);
    return api.getFlowFieldStats()[identifier]?.gridMode === 'gpu' && api.renderedLayers().some(layer => layer.id === `${identifier}-resident-flow-field` && layer.visible && layer.loaded);
  }, id, {timeout: 60000});
  await page.waitForTimeout(2500);
}
const results = {};
try {
  const palette = {colors: ['#ff0000', '#ffff00', '#ff00ff', '#ffffff', '#ff0000']};
  // Color by speed: the palette bands appear across the streaks; without it only the layer color paints.
  await show('resident-flow-style-speed', {columns: {...position, u: 'u', v: 'v'}, mapState}, {colorBySpeed: true, colorRange: palette}, 'UV');
  const colored = {};
  for (const [name, matches] of Object.entries(classes)) colored[name] = (await paint(matches)).count;
  const present = Object.values(colored).filter(count => count > 15).length;
  assert(present >= 3, `speed colors must span the palette: ${JSON.stringify(colored)}`);
  results.colored = colored;
  await show('resident-flow-style-plain', {columns: {...position, u: 'u', v: 'v'}, mapState}, {colorBySpeed: false, colorRange: palette}, 'UV');
  const plain = {};
  for (const [name, matches] of Object.entries(classes)) plain[name] = (await paint(matches)).count;
  assert(Object.values(plain).filter(count => count > 15).length < present, `plain color must not span the palette: ${JSON.stringify(plain)}`);
  results.plain = plain;

  // Elevation: a single magenta palette color; the streaks lift up the screen under pitch.
  const pitched = {...mapState, pitch: 60};
  const magenta = {colors: ['#ff00ff']};
  await show('resident-flow-style-flat', {columns: {...position, altitude: 'altitude'}, mapState: pitched}, {colorBySpeed: true, colorRange: magenta, elevationMultiplier: 0}, 'ELEVATION');
  const flat = await paint(classes.magenta);
  await show('resident-flow-style-lifted', {columns: {...position, altitude: 'altitude'}, mapState: pitched}, {colorBySpeed: true, colorRange: magenta, elevationMultiplier: 3}, 'ELEVATION');
  const lifted = await paint(classes.magenta);
  assert(flat.count > 15 && lifted.count > 15, `elevation streaks must paint (${flat.count}, ${lifted.count})`);
  assert(lifted.centroidY < flat.centroidY - 1, `elevation must lift streaks up the screen (${flat.centroidY} -> ${lifted.centroidY})`);
  results.elevation = {flat, lifted};

  // Debug grid layers.
  await show('resident-flow-style-debug', {columns: {...position, u: 'u', v: 'v'}, mapState}, {debugGrid: true}, 'UV');
  const debugLayers = await page.evaluate(() => window.__keplerComputeTest.renderedLayers().map(layer => layer.id).filter(id => id.includes('debug-grid')));
  assert(debugLayers.length > 0, 'debugGrid must render the debug grid layers');
  results.debugLayers = debugLayers;
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, results}, null, 2));
} finally {
  await browser.close();
}
