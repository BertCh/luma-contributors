// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {dirname} from 'node:path';
import {pinViteModules} from './pin-vite-modules.mjs';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

// Pinned native plot code is the oracle for range-filter histogram edges (d3 ticks of the filter domain).
const requireNative = createRequire(import.meta.url);
const nativePlot = requireNative(`${dirname(requireNative.resolve('@kepler.gl/utils'))}/plot.js`);
const url = process.argv[2] || 'http://127.0.0.1:5318/?rows=10000';
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
let rejectStartup;
const startupFailure = new Promise((resolve, reject) => { rejectStartup = reject; });
page.on('pageerror', error => {
  errors.push(error.message);
  rejectStartup(error);
});
page.on('console', message => {
  if (message.type() === 'error') process.stderr.write(`${message.location().url}: ${message.text()}\n`);
});
page.on('response', response => {
  if (response.status() >= 400) process.stderr.write(`HTTP ${response.status()} ${response.url()}\n`);
  if (response.status() >= 500) rejectStartup(new Error(`HTTP ${response.status()} ${response.url()}`));
});

try {
  await Promise.race([startupFailure, (async () => {
    await page.goto(url, {waitUntil: 'domcontentloaded'});
    await page.waitForFunction(() => {
      const state = window.__keplerComputeTest?.getState();
      if (state?.error) throw new Error(state.error);
      return state?.ready && Number.isInteger(state.selectedCount);
    }, undefined, {timeout: 60000});
  })()]);
  const initial = await page.evaluate(() => window.__keplerComputeTest.getState());
  const sourceCount = await page.evaluate(() => window.__keplerComputeTest.source().length);
  assert.equal(initial.selectedCount, sourceCount);
  assert.equal(initial.unsupportedSelection, false);
  assert.equal(initial.nativeFilters.length, 3);

  const rendered = await page.evaluate(async () => {
    const api = window.__keplerComputeTest;
    return {layers: api.renderedLayers(), picked: await api.pickSourceRow(0)};
  });
  assert(rendered.layers.some(layer => layer.id === 'resident-points' && layer.visible && layer.loaded));
  assert.equal(rendered.picked?.layerId, 'resident-points');
  assert(rendered.picked.index >= 0 && rendered.picked.index < sourceCount);
  const pixels = PNG.sync.read(await page.screenshot());
  let visibleMapPixels = 0;
  for (let y = 30; y < pixels.height - 40; y++) {
    for (let x = 320; x < pixels.width - 380; x++) {
      const offset = (y * pixels.width + x) * 4;
      if (pixels.data[offset + 1] > 100 && pixels.data[offset + 2] > 100) visibleMapPixels++;
    }
  }
  assert(visibleMapPixels > 300, 'Resident observations must be visible on the map');

  const expected = await page.evaluate(() => {
    const rows = window.__keplerComputeTest.source();
    const selected = [];
    const histogramValues = [];
    for (let index = 0; index < rows.length; index++) {
      const row = rows[index];
      if ((row[3] === 1 || row[3] === 3) && row[4] >= 100 && row[4] <= 900) {
        histogramValues.push(row[2]);
        if (row[2] >= 40 && row[2] <= 75) selected.push(index);
      }
    }
    window.__keplerComputeTest.setRange(40, 75);
    window.__keplerComputeTest.setCategories([1, 3]);
    window.__keplerComputeTest.setTime(100, 900);
    return {count: selected.length, firstRows: selected.slice(0, 10), histogramValues};
  });
  await page.waitForFunction(({count, revision}) => {
    const state = window.__keplerComputeTest.getState();
    if (state.error) throw new Error(state.error);
    return state.selectedCount === count && state.revision > revision;
  }, {count: expected.count, revision: initial.revision}, {timeout: 30000});

  const filtered = await page.evaluate(async () => {
    const api = window.__keplerComputeTest;
    const summary = await api.readSummary();
    const result = await api.readPage(0, 10);
    return {
      state: api.getState(),
      histogram: Array.from(summary.histograms.value),
      sourceRows: Array.from(result.sourceRowIndices),
      values: Array.from(result.columns.value),
      categories: Array.from(result.columns.category),
      times: Array.from(result.columns.time),
      expectedValues: Array.from(result.sourceRowIndices, index => api.source()[index][2])
    };
  });
  // The value histogram is the native range plot of the filter domain: d3 tick edges, own filter excluded.
  assert.deepEqual(filtered.histogram, nativePlot.histogramFromDomain([0, 100], expected.histogramValues, 30).map(bin => bin.count));
  assert.deepEqual(filtered.sourceRows, expected.firstRows);
  assert.deepEqual(filtered.values, filtered.expectedValues);
  assert(filtered.categories.every(category => category === 1 || category === 3));
  assert(filtered.times.every(time => time >= 100 && time <= 900));
  assert.equal(filtered.state.uploadBytes, initial.uploadBytes);
  assert(filtered.state.frameCount > initial.frameCount);
  assert.equal(filtered.state.unsupportedSelection, false);
  assert.deepEqual(filtered.state.nativeFilters[0].value, [40, 75]);
  assert.deepEqual(filtered.state.nativeFilters[1].value, [1, 3]);

  const disabledConfiguration = await page.evaluate(() => {
    const api = window.__keplerComputeTest;
    const configuration = api.saveNativeConfig();
    for (const filter of configuration.config.visState.filters) filter.enabled = false;
    api.applyNativeConfig(configuration);
    return configuration;
  });
  await page.waitForFunction(count => window.__keplerComputeTest.getState().selectedCount === count,
    sourceCount, {timeout: 30000});
  assert(await page.evaluate(() => window.__keplerComputeTest.getState().nativeFilters.every(filter => filter.enabled === false)));
  await page.evaluate(configuration => {
    for (const filter of configuration.config.visState.filters) filter.enabled = true;
    window.__keplerComputeTest.applyNativeConfig(configuration);
  }, disabledConfiguration);
  await page.waitForFunction(count => window.__keplerComputeTest.getState().selectedCount === count,
    expected.count, {timeout: 30000});

  const duplicate = await page.evaluate(() => {
    const api = window.__keplerComputeTest;
    const original = api.saveNativeConfig();
    const configuration = structuredClone(original);
    const valueFilter = configuration.config.visState.filters.find(filter => filter.name?.includes('value'));
    configuration.config.visState.filters.push({...valueFilter, id: 'duplicate-value', value: [50, 60]});
    const count = api.source().filter(row => row[2] >= 50 && row[2] <= 60 &&
      (row[3] === 1 || row[3] === 3) && row[4] >= 100 && row[4] <= 900).length;
    const originalValues = [];
    const duplicateValues = [];
    for (const row of api.source()) {
      if (!(row[3] === 1 || row[3] === 3) || row[4] < 100 || row[4] > 900) continue;
      if (row[2] >= 50 && row[2] <= 60) originalValues.push(row[2]);
      if (row[2] >= 40 && row[2] <= 75) duplicateValues.push(row[2]);
    }
    api.applyNativeConfig(configuration);
    return {original, count, originalValues, duplicateValues};
  });
  await page.waitForFunction(count => {
    const state = window.__keplerComputeTest.getState();
    return state.selectedCount === count && Object.keys(state.plotDiagnostics).length === 0 &&
      state.nativeFilters.filter(filter => filter.id === 'resident-value' || filter.id === 'duplicate-value').every(filter => filter.lumaComputePlot?.status === 'ready');
  }, duplicate.count, {timeout: 30000});
  assert.equal(await page.evaluate(() => window.__keplerComputeTest.getState().unsupportedSelection), false);
  assert.equal(await page.evaluate(() => window.__keplerComputeTest.getState().error), '');
  const duplicateBins = await page.evaluate(() => Object.fromEntries(
    window.__keplerComputeTest.getState().nativeFilters.filter(filter => filter.id === 'resident-value' || filter.id === 'duplicate-value')
      .map(filter => [filter.id, {domain: filter.domain, bins: filter.bins['resident-observations'].map(bin => [bin.x0, bin.x1, bin.count])}])
  ));
  const nativeBins = (domain, values) => nativePlot.histogramFromDomain(domain, values, 30).map(bin => [bin.x0, bin.x1, bin.count]);
  assert.deepEqual(duplicateBins['resident-value'].bins, nativeBins(duplicateBins['resident-value'].domain, duplicate.originalValues));
  assert.deepEqual(duplicateBins['duplicate-value'].bins, nativeBins(duplicateBins['duplicate-value'].domain, duplicate.duplicateValues));
  await page.evaluate(configuration => window.__keplerComputeTest.applyNativeConfig(configuration), duplicate.original);
  await page.waitForFunction(count => {
    const state = window.__keplerComputeTest.getState();
    return state.selectedCount === count && Object.keys(state.plotDiagnostics).length === 0;
  }, expected.count, {timeout: 30000});

  // Kepler's empty category list clears that filter; it is not an empty selection.
  await page.evaluate(() => {
    window.__keplerComputeTest.setRange(0, 100);
    window.__keplerComputeTest.setCategories([]);
    window.__keplerComputeTest.setTime(0, 1439);
  });
  await page.waitForFunction(count => window.__keplerComputeTest.getState().selectedCount === count,
    sourceCount, {timeout: 30000});
  assert.deepEqual(errors, []);
  const screenshot = process.env.KEPLER_SMOKE_SCREENSHOT;
  if (screenshot) await page.screenshot({path: screenshot});
  process.stdout.write(`${JSON.stringify({
    passed: true,
    url,
    sourceCount,
    selectedCount: expected.count,
    inspectedRows: filtered.sourceRows.length,
    checks: ['native filter actions', 'GPU selection count', 'self-excluding histogram',
      'bounded inspection', 'stable source identity', 'category clear semantics', 'stable upload accounting',
      'visible resident map', 'actual Deck picking', 'native filter disable and reenable',
      'duplicate filter plot guard and recovery']
  }, null, 2)}\n`);
} catch (error) {
  process.stderr.write(`${JSON.stringify(await page.evaluate(() => {
    const state = window.__keplerComputeTest?.getState();
    return state ? {...state, nativeFilters: state.nativeFilters.map(filter => ({
      id: filter.id, name: filter.name, type: filter.type, value: filter.value, enabled: filter.enabled,
      plot: filter.lumaComputePlot
    }))} : null;
  }), null, 2)}\n`);
  throw error;
} finally {
  await browser.close();
}
