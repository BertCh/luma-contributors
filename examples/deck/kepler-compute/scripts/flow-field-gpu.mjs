// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Run: node scripts/flow-field-gpu.mjs http://127.0.0.1:5328/?rows=256
// Browser verification (headless by default) that the Flow Field velocity grid is GPU-resident and
// filter-aware: the grid is binned, smoothed and scaled by compute passes; a range filter re-bins
// it from the GPU mask with no mask, grid or scalar readback; layer removal releases the runtime.
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
const {browser, page, errors, datasetId, applyLayer} = await openPublishedTable({table, fileName: 'flow-field-gpu.arrow', url: process.argv[2] ?? 'http://127.0.0.1:5328/?rows=256'});
const results = [];
try {
  const mapState = {longitude: -73.98, latitude: 40.77, zoom: 11, pitch: 0, bearing: 0};
  const columnsFor = mode => mode === 'UV'
    ? {lat: 'latitude', lng: 'longitude', u: 'u', v: 'v'}
    : mode === 'SPEED_DIR' ? {lat: 'latitude', lng: 'longitude', speed: 'speed', direction: 'direction'}
    : {lat: 'latitude', lng: 'longitude', altitude: 'altitude'};
  const stats = id => page.evaluate(identifier => window.__keplerComputeTest.getFlowFieldStats()[identifier], id);
  async function waitForStableField(identifier) {
    const deadline = Date.now() + 30000;
    let previousKey;
    let stableSince = Date.now();
    let latest;
    while (Date.now() < deadline) {
      latest = await stats(identifier);
      const key = latest && JSON.stringify({
        viewportBuildCount: latest.viewportBuildCount, simulationBounds: latest.simulationBounds,
        gridBuildCount: latest.gridBuildCount, bounds: latest.bounds,
        selectionRevision: latest.selectionRevision, extentReadbackCount: latest.extentReadbackCount,
        gridPending: latest.gridPending
      });
      if (key !== previousKey || latest?.gridPending || !latest?.spawnRegion || !(latest.stepCount > 3)) {
        previousKey = key;
        stableSince = Date.now();
      } else if (Date.now() - stableSince >= 500) {
        return latest;
      }
      await page.waitForTimeout(50);
    }
    assert.fail(`flow field did not stabilise within 30 s: ${JSON.stringify(latest)}`);
  }
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
  // Port regression coverage: published luma has no setSpawnRegion. The host crops the field
  // through public textures + setField, keeping all particles in a padded, settled viewport.
  // Exercise both GPU and CPU grid paths; no old assertion or tolerance is relaxed.
  for (const gridMode of ['auto', 'cpu']) {
    await page.evaluate(value => window.__keplerComputeTest.setFlowFieldGridMode(value), gridMode);
    const identifier = `resident-flow-field-viewport-${gridMode}`;
    await applyLayer(identifier, 'flowField', {columns: {...columnsFor('UV'), altitude: 'altitude'}, color: [255, 0, 255], mapState: {...mapState, zoom: 14}}, {colorBySpeed: false, opacity: 1, strokeWidth: 3, linesPerScreen: 400, gridResolution: GRID_SIDE, smoothing: 1}, 'UV');
    await page.waitForFunction(identifier => {
      const state = window.__keplerComputeTest.getFlowFieldStats()[identifier];
      return state?.spawnRegion && state.viewportBuildCount > 0 && state.stepCount > 3;
    }, identifier, {timeout: 30000});
    // Initial selection extent readback and a late map-panel resize can legitimately change the
    // first field/crop after layer creation, especially after a cold Vite reload. Await completed
    // field work and 500 ms of unchanged counts/bounds, then keep the independent reuse assertion.
    const zoomed = await waitForStableField(identifier);
    assert.equal(zoomed.gridMode, gridMode === 'auto' ? 'gpu' : 'cpu');
    assert.equal(zoomed.particleCount, 9600, 'viewport uses the full lines-per-screen particle budget');
    assert(zoomed.simulationBounds[2] - zoomed.simulationBounds[0] < (zoomed.bounds[2] - zoomed.bounds[0]) / 2, 'zoom must concentrate the particle domain');
    assert(zoomed.simulationBounds.every((value, index) => index < 2 ? value >= zoomed.bounds[index] : value <= zoomed.bounds[index]), 'viewport domain stays inside the data bounds');
    await page.waitForTimeout(500);
    const stable = await stats(identifier);
    assert.equal(stable.viewportBuildCount, zoomed.viewportBuildCount,
      `a settled camera must reuse its cropped textures: builds ${zoomed.viewportBuildCount} -> ${stable.viewportBuildCount}; simulationBounds ${JSON.stringify(zoomed.simulationBounds)} -> ${JSON.stringify(stable.simulationBounds)}; grid builds ${zoomed.gridBuildCount} -> ${stable.gridBuildCount}; extent reads ${zoomed.extentReadbackCount} -> ${stable.extentReadbackCount}; pending ${zoomed.gridPending} -> ${stable.gridPending}`);
    assert.equal(stable.gridReadbackCount, 0, 'viewport crop must not read the GPU grid back');
    assert.deepEqual(errors, [], 'viewport crop must not produce GPU validation or Deck draw errors');
    const image = PNG.sync.read(await page.screenshot());
    let painted = 0;
    for (let vertical = 0; vertical < image.height - 80; vertical++)
      for (let horizontal = 330; horizontal < image.width - 60; horizontal++) {
        const offset = (vertical * image.width + horizontal) * 4;
        if (image.data[offset] > 180 && image.data[offset + 1] < 90 && image.data[offset + 2] > 180) painted++;
      }
    assert(painted > 15, `${gridMode} viewport-cropped particles must paint magenta streaks (${painted})`);

    // Pan the same runtime via native configuration (same id, columns and style).
    const configuration = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
    configuration.config.mapState.longitude += 0.04;
    await page.evaluate(value => window.__keplerComputeTest.applyNativeConfig(value), configuration);
    await page.waitForFunction(({identifier, west}) => window.__keplerComputeTest.getFlowFieldStats()[identifier]?.simulationBounds[0] > west, {identifier, west: zoomed.simulationBounds[0]}, {timeout: 30000});
    const panned = await stats(identifier);
    assert.equal(panned.particleCount, zoomed.particleCount);
    configuration.config.mapState.zoom = 8;
    await page.evaluate(value => window.__keplerComputeTest.applyNativeConfig(value), configuration);
    await page.waitForFunction(identifier => window.__keplerComputeTest.getFlowFieldStats()[identifier]?.spawnRegion === null, identifier, {timeout: 30000});
    const wide = await stats(identifier);
    assert.deepEqual(wide.simulationBounds, wide.bounds, 'zooming out restores the original whole field');
    results.push({viewport: {gridMode, painted, zoomed, stable, panned, wide}});
  }
  await page.evaluate(() => window.__keplerComputeTest.setFlowFieldGridMode('auto'));
  // Filter-aware: a range filter on longitude re-bins the same texture from the GPU mask.
  const id = 'resident-flow-field-gpu-UV';
  await applyLayer(id, 'flowField', {columns: columnsFor('UV'), color: [186, 233, 223], mapState}, {opacity: 1, strokeWidth: 3, linesPerScreen: 400, gridResolution: GRID_SIDE, smoothing: 1}, 'UV');
  await page.waitForFunction(identifier => window.__keplerComputeTest.getFlowFieldStats()[identifier]?.gridMode === 'gpu', id, {timeout: 60000});
  // The GPU extent (20-byte readback) lands asynchronously after the first grid build; sample `before`
  // only once it did, else the CPU-derived initial bounds are compared with the first GPU extent.
  await page.waitForFunction(identifier => window.__keplerComputeTest.getFlowFieldStats()[identifier]?.extentReadbackCount >= 1, id, {timeout: 30000});
  const before = await stats(id);
  const beforeField = await validCells(id);
  await page.evaluate(identifier => window.__keplerComputeTest.setPublishedRange(identifier, 'longitude', -74.1, -73.98), datasetId);
  await page.waitForFunction(identifier => window.__keplerComputeTest.getState().publications.find(value => value.id === identifier)?.selectedCount < 576, datasetId, {timeout: 30000});
  await page.waitForFunction(({identifier, builds}) => window.__keplerComputeTest.getFlowFieldStats()[identifier]?.gridBuildCount > builds, {identifier: id, builds: before.gridBuildCount}, {timeout: 30000});
  // The filtered extent also arrives asynchronously after the grid rebuild (see above).
  await page.waitForFunction(({identifier, east}) => window.__keplerComputeTest.getFlowFieldStats()[identifier]?.bounds[2] < east, {identifier: id, east: before.bounds[2]}, {timeout: 30000});
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
