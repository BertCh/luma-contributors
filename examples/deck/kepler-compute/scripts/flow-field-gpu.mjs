// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser verification (headless by default) that the Flow Field velocity grid is GPU-resident and
// filter-aware: the grid is binned, smoothed and scaled by compute passes; a range filter re-bins
// it from the GPU mask with no mask, grid or scalar readback; layer removal releases the runtime.
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {openPublishedTable} from './flow-field-utils.mjs';

const GRID_SIDE = 24;
const rows = [];
for (let row = 0; row < GRID_SIDE; row++)
  for (let column = 0; column < GRID_SIDE; column++) {
    const dx = column - 11.5;
    const dy = row - 11.5;
    const norm = Math.hypot(dx, dy) || 1;
    rows.push({longitude: -74.1 + column * 0.01, latitude: 40.65 + row * 0.01, u: (-dy / norm) * 5, v: (dx / norm) * 5, speed: 5, direction: (Math.atan2(-dx, -dy) * 180) / Math.PI + 180, altitude: 200 - norm * 10});
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
const {browser, page, errors, datasetId, applyLayer} = await openPublishedTable({table, fileName: 'flow-field-gpu.arrow', url: process.argv[2]});
const results = [];
try {
  const mapState = {longitude: -73.98, latitude: 40.77, zoom: 11, pitch: 0, bearing: 0};
  const columnsFor = mode => mode === 'UV'
    ? {lat: 'latitude', lng: 'longitude', u: 'u', v: 'v'}
    : mode === 'SPEED_DIR' ? {lat: 'latitude', lng: 'longitude', speed: 'speed', direction: 'direction'}
    : {lat: 'latitude', lng: 'longitude', altitude: 'altitude'};
  const stats = id => page.evaluate(identifier => window.__keplerComputeTest.getFlowFieldStats()[identifier], id);
  const validCells = async id => {
    const texels = await page.evaluate(identifier => window.__keplerComputeTest.readFlowFieldGridForDebug(identifier), id);
    assert(texels, `${id} must expose GPU texels for debugging`);
    let valid = 0;
    let fastest = 0;
    let normalizedMinimum = Infinity;
    let normalizedMaximum = -Infinity;
    for (let index = 0; index < texels.length; index += 4) if (texels[index + 2] > 0.5) {
      valid++;
      fastest = Math.max(fastest, Math.hypot(texels[index], texels[index + 1]));
      normalizedMinimum = Math.min(normalizedMinimum, texels[index + 3]);
      normalizedMaximum = Math.max(normalizedMaximum, texels[index + 3]);
    }
    return {valid, fastest, normalizedMinimum, normalizedMaximum};
  };
  for (const mode of ['UV', 'SPEED_DIR', 'ELEVATION']) {
    const id = `resident-flow-field-gpu-${mode}`;
    await applyLayer(id, 'flowField', {columns: columnsFor(mode), color: [186, 233, 223], mapState}, {opacity: 1, strokeWidth: 3, linesPerScreen: 400, gridResolution: GRID_SIDE, smoothing: 1}, mode);
    await page.waitForFunction(identifier => window.__keplerComputeTest.getFlowFieldStats()[identifier], id, {timeout: 60000});
    await page.waitForTimeout(800);
    const state = await stats(id);
    assert.equal(state.gridMode, 'gpu', `${mode} grid must be GPU-resident`);
    assert.equal(state.gridReadbackCount, 0, `${mode} grid must not be read back`);
    assert(state.gridBuildCount >= 1, `${mode} grid must be built by compute passes`);
    assert.equal(state.velocityScale, null, `${mode} visual scale is resolved on the GPU`);
    const field = await validCells(id);
    assert(field.valid > 100, `${mode} must fill the grid; valid cells ${field.valid}`);
    assert(field.fastest > 0, `${mode} must have nonzero speeds`);
    // FLOW3: the w channel holds speed normalized over [minimum, maximum] (color by speed).
    assert(field.normalizedMinimum >= 0 && field.normalizedMaximum <= 1, `${mode} normalized speed must stay in [0, 1]`);
    assert(field.normalizedMaximum === 1 || (field.normalizedMaximum === 0 && field.normalizedMinimum === 0), `${mode} fastest cell must normalize to 1 (or the whole field has one speed)`);
    assert.equal(state.hasAltitude, mode === 'ELEVATION', `${mode} altitude texture presence`);
    results.push({mode, state, field});
  }
  // Filter-aware: a range filter on longitude re-bins the same texture from the GPU mask.
  const id = 'resident-flow-field-gpu-UV';
  await applyLayer(id, 'flowField', {columns: columnsFor('UV'), color: [186, 233, 223], mapState}, {opacity: 1, strokeWidth: 3, linesPerScreen: 400, gridResolution: GRID_SIDE, smoothing: 1}, 'UV');
  await page.waitForFunction(identifier => window.__keplerComputeTest.getFlowFieldStats()[identifier]?.gridMode === 'gpu', id, {timeout: 60000});
  const before = await stats(id);
  const beforeField = await validCells(id);
  await page.evaluate(identifier => window.__keplerComputeTest.setPublishedRange(identifier, 'longitude', -74.1, -73.98), datasetId);
  await page.waitForFunction(identifier => window.__keplerComputeTest.getState().publications.find(value => value.id === identifier)?.selectedCount < 576, datasetId, {timeout: 30000});
  await page.waitForFunction(({identifier, builds}) => window.__keplerComputeTest.getFlowFieldStats()[identifier]?.gridBuildCount > builds, {identifier: id, builds: before.gridBuildCount}, {timeout: 30000});
  const filtered = await stats(id);
  const filteredField = await validCells(id);
  assert.equal(filtered.gridReadbackCount, 0, 'a filter change must not read back the mask or grid');
  assert.equal(filtered.columns, before.columns, 'filtering must not resize the texture');
  // FLOW3: the bounds follow the filtered rows (GPU min/max reduction); only that result is read back.
  assert(filtered.bounds[2] < before.bounds[2], `filtered bounds must shrink east (${before.bounds[2]} -> ${filtered.bounds[2]})`);
  assert(filtered.extentReadbackCount >= 1, 'a filter revision must measure the selected rows on the GPU');
  assert.equal(before.selectionRevision === null || filtered.selectionRevision !== before.selectionRevision, true);
  assert(filtered.stepCount >= before.stepCount, 'the simulation keeps running through a filter change');
  assert(filteredField.valid > 0, 'filtered grid keeps the selected cells');
  await page.evaluate(identifier => window.__keplerComputeTest.setPublishedRange(identifier, 'longitude', -74.1, -73.86), datasetId);
  await page.waitForFunction(({identifier, east}) => window.__keplerComputeTest.getFlowFieldStats()[identifier]?.bounds[2] > east, {identifier: id, east: filtered.bounds[2]}, {timeout: 30000});
  results.push({filter: {before: beforeField, filtered: filteredField, filteredStats: filtered}});
  // Layer removal releases the GPU runtime: switching the layer id drops the previous entry.
  const replacement = 'resident-flow-field-gpu-replacement';
  await applyLayer(replacement, 'flowField', {columns: columnsFor('SPEED_DIR'), color: [186, 233, 223], mapState}, {gridResolution: GRID_SIDE}, 'SPEED_DIR');
  await page.waitForFunction(({previous, next}) => { const all = window.__keplerComputeTest.getFlowFieldStats(); return all[next] && !all[previous]; }, {previous: id, next: replacement}, {timeout: 30000});
  // CPU reference mode stays available as a fallback and reports its own grid mode.
  await page.evaluate(() => window.__keplerComputeTest.setFlowFieldGridMode('cpu'));
  await applyLayer('resident-flow-field-cpu-reference', 'flowField', {columns: columnsFor('UV'), color: [186, 233, 223], mapState}, {gridResolution: GRID_SIDE}, 'UV');
  await page.waitForFunction(() => window.__keplerComputeTest.getFlowFieldStats()['resident-flow-field-cpu-reference']?.gridMode === 'cpu', undefined, {timeout: 30000});
  await page.evaluate(() => window.__keplerComputeTest.setFlowFieldGridMode('auto'));
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, results}, null, 2));
} finally {
  await browser.close();
}
