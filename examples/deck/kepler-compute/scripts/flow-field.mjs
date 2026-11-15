// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser verification of the resident Flow Field layer: u/v, speed/direction and altitude modes,
// GPU particle animation, field units and lifecycle, and specific unsupported reasons.
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {PNG} from 'pngjs';
import {openPublishedTable} from './flow-field-utils.mjs';

const rows = [];
for (let row = 0; row < 24; row++)
  for (let column = 0; column < 24; column++) {
    const longitude = -74.1 + column * 0.01;
    const latitude = 40.65 + row * 0.01;
    // Counter-clockwise swirl with a 5 m/s peak and a deterministic altitude hill.
    const dx = column - 11.5;
    const dy = row - 11.5;
    const norm = Math.hypot(dx, dy) || 1;
    rows.push({longitude, latitude, u: (-dy / norm) * 5, v: (dx / norm) * 5, speed: 5, direction: (Math.atan2(-dx, -dy) * 180) / Math.PI + 180, altitude: 200 - norm * 10});
  }
const table = arrow.tableFromArrays({
  longitude: Float64Array.from(rows, r => r.longitude),
  latitude: Float64Array.from(rows, r => r.latitude),
  u: Float64Array.from(rows, r => r.u),
  v: Float64Array.from(rows, r => r.v),
  speed: Float64Array.from(rows, r => r.speed),
  direction: Float64Array.from(rows, r => r.direction),
  altitude: Float64Array.from(rows, r => r.altitude)
});
const {browser, page, errors, applyLayer} = await openPublishedTable({table, fileName: 'flow-field.arrow', url: process.argv[2]});
const results = [];
try {
  const mapState = {longitude: -73.98, latitude: 40.77, zoom: 11, pitch: 0, bearing: 0};
  const columnsFor = mode => mode === 'UV'
    ? {lat: 'latitude', lng: 'longitude', u: 'u', v: 'v'}
    : mode === 'SPEED_DIR' ? {lat: 'latitude', lng: 'longitude', speed: 'speed', direction: 'direction'}
    : {lat: 'latitude', lng: 'longitude', altitude: 'altitude'};
  for (const mode of ['UV', 'SPEED_DIR', 'ELEVATION']) {
    const id = `resident-flow-field-${mode}`;
    await applyLayer(id, 'flowField', {columns: columnsFor(mode), color: [186, 233, 223], mapState}, {opacity: 1, strokeWidth: 3, linesPerScreen: 400, gridResolution: 24, smoothing: 1}, mode);
    await page.waitForFunction(identifier => {
      const api = window.__keplerComputeTest;
      const state = api.getState();
      if (state.error) throw new Error(state.error);
      return api.renderedLayers().some(layer => layer.id === `${identifier}-resident-flow-field` && layer.visible && layer.loaded);
    }, id, {timeout: 60000});
    // Animation: the simulation must advance without CPU readback, and the particles must paint pixels.
    const first = await page.evaluate(identifier => window.__keplerComputeTest.getFlowFieldStats()[identifier], id);
    await page.waitForTimeout(1500);
    const second = await page.evaluate(identifier => window.__keplerComputeTest.getFlowFieldStats()[identifier], id);
    assert(second.stepCount > first.stepCount, `${mode} particles must animate (steps ${first.stepCount} -> ${second.stepCount})`);
    assert.equal(second.particleCount, first.particleCount);
    const screenshot = PNG.sync.read(await page.screenshot());
    let painted = 0;
    for (let i = 0; i < screenshot.data.length; i += 4) {
      const [r, g, b] = [screenshot.data[i], screenshot.data[i + 1], screenshot.data[i + 2]];
      if (g > 190 && b > 180 && r > 150 && r < 215) painted++;
    }
    assert(painted > 200, `${mode} must paint particle streaks; found ${painted}`);
    results.push({mode, stats: second, painted});
  }
  // Field units: physical policy uses velocityScale 1 and moves slowly; visual policy rescales.
  await page.evaluate(() => window.__keplerComputeTest.setFlowFieldUnitPolicy('physical'));
  await applyLayer('resident-flow-field-physical', 'flowField', {columns: columnsFor('UV'), color: [186, 233, 223], mapState}, {linesPerScreen: 400, gridResolution: 24, strokeWidth: 3, cycleSeconds: 21}, 'UV');
  await page.waitForFunction(() => window.__keplerComputeTest.getFlowFieldStats()['resident-flow-field-physical']);
  const physical = await page.evaluate(() => window.__keplerComputeTest.getFlowFieldStats()['resident-flow-field-physical']);
  assert.equal(physical.velocityScale, 1, 'physical policy must not rescale field units');
  await page.evaluate(() => window.__keplerComputeTest.setFlowFieldUnitPolicy('visual'));
  // The old unsupported-options assertion was stale: the pre-port runtime already implemented
  // speed color/debug grid. Require successful rendering here; flow-field-style checks pixels.
  await applyLayer('resident-flow-field-limits', 'flowField', {columns: columnsFor('UV'), color: [186, 233, 223], mapState}, {colorBySpeed: true, debugGrid: true, gridResolution: 24}, 'UV');
  await page.waitForFunction(() => {
    const api = window.__keplerComputeTest;
    if (api.getState().error) throw new Error(api.getState().error);
    return api.renderedLayers().some(layer => layer.id === 'resident-flow-field-limits-resident-flow-field' && layer.visible && layer.loaded);
  }, undefined, {timeout: 30000});
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({results}, null, 2));
} finally {
  await browser.close();
}
