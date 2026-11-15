// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// LAYERS4-C2 and LAYERS5-4: a cluster layer on a large dataset is admitted through the memory
// registry (scratch bytes rise during the build, settle to a measured footprint), zoom, radius and
// mask changes rebuild and redraw, hover members work, and removing the layer releases scratch.
// Usage: node cluster-admission-zoom.mjs [url] [rows]. Headless WebGPU only.
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const baseUrl = process.argv[2] || 'http://127.0.0.1:5318/';
const rowCount = Number(process.argv[3] || 1000000);
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1200, height: 900}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer|BindGroup)|validation error|Uncaught/i.test(message.text())) errors.push(message.text());
});
const watchdog = setTimeout(() => {
  console.error('cluster-admission-zoom exceeded 420 seconds');
  process.exit(1);
}, 420000);

const readScratch = () => page.evaluate(() => Number(document.querySelector('[data-category=scratch]')?.dataset.bytes ?? NaN));

async function applyCluster(layerId, clusterRadius, valueRange) {
  await page.evaluate(({layerId, clusterRadius, valueRange}) => {
    const api = window.__keplerComputeTest;
    const configuration = structuredClone(api.saveNativeConfig());
    const layer = configuration.config.visState.layers[0];
    layer.id = layerId;
    layer.type = 'cluster';
    layer.config.columnMode = 'points';
    layer.config.columns = {lat: 'latitude', lng: 'longitude'};
    layer.config.colorField = null;
    layer.config.sizeField = null;
    layer.config.textLabel = [];
    layer.config.visConfig = {...layer.config.visConfig, clusterRadius, radiusRange: [1, 40], minClusterSize: 1, colorAggregation: 'count'};
    layer.visualChannels = {...layer.visualChannels, colorField: null, sizeField: null};
    if (valueRange) configuration.config.visState.filters.find(filter => filter.name?.includes('value')).value = valueRange;
    return api.applyNativeConfig(configuration);
  }, {layerId, clusterRadius, valueRange});
}

async function waitForClusters(layerId, zoom) {
  await page.waitForFunction(async ({id, zoom}) => {
    const api = window.__keplerComputeTest;
    const state = api.getState();
    if (state.error) throw new Error(state.error);
    const invalid = state.nativeLayers?.find(layer => layer.id === id && layer.valid === false);
    if (invalid) throw new Error(JSON.stringify(invalid));
    const read = await api.readClusters(id, 1);
    return Boolean(read && read.clusterCount > 0 && (zoom === undefined || Math.abs(read.view.zoom - zoom) < 1e-9));
  }, {id: layerId, zoom}, {timeout: 120000, polling: 500});
}

// The binding is briefly absent while the index re-queries after a viewport or settle change (the
// pre-port harness read it once and was racy about the same window); poll until it is present.
const readClustersWhenPresent = async (memberLimit, timeout = 60000) => {
  const deadline = Date.now() + timeout;
  for (;;) {
    const read = await page.evaluate(limit => window.__keplerComputeTest.readClusters('cluster-admission', limit), memberLimit);
    if (read) return read;
    if (Date.now() > deadline) throw new Error('cluster binding did not become present');
    await page.waitForTimeout(250);
  }
};

const summary = {};
const startedAt = Date.now();
const stage = name => console.log(JSON.stringify({stage: name, seconds: Math.round((Date.now() - startedAt) / 1000)}));
try {
  // The driver may supply ?rows=256 (or a memory budget). Appending another '?' made
  // rows='256?rows=1000000', which becomes NaN even in the pre-port example.
  const targetUrl = new URL(baseUrl);
  targetUrl.searchParams.set('rows', String(rowCount));
  // Match acceptance.mjs (also present before the port): million-row fixtures exceed
  // the default 512 MiB registry budget before cluster scratch is admitted. Honor overrides.
  if (rowCount >= 1000000 && !targetUrl.searchParams.has('memoryBudgetMebibytes')) {
    targetUrl.searchParams.set('memoryBudgetMebibytes', '32768');
  }
  await page.goto(targetUrl.href, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(count => {
    const state = window.__keplerComputeTest?.getState();
    if (state?.error) throw new Error(state.error);
    return state?.selectedCount === count;
  }, rowCount, {timeout: 180000});
  stage('loaded');
  await page.getByTestId('memory-budget-panel').waitFor({timeout: 30000});
  const baseline = await readScratch();
  // Sample scratch while the build runs: admission must be visible, then settle lower.
  await page.evaluate(() => {
    window.__clusterScratchPeak = 0;
    window.__clusterScratchTimer = setInterval(() => {
      const bytes = Number(document.querySelector('[data-category=scratch]')?.dataset.bytes ?? 0);
      if (bytes > window.__clusterScratchPeak) window.__clusterScratchPeak = bytes;
    }, 25);
  });
  await applyCluster('cluster-admission', 40);
  await waitForClusters('cluster-admission');
  stage('built');
  await page.waitForTimeout(1500);
  const peak = await page.evaluate(() => window.__clusterScratchPeak);
  const settled = await readScratch();
  summary.scratch = {baseline, peak, settled};
  assert(peak > baseline, `build must reserve scratch memory (peak ${peak} vs ${baseline})`);
  assert(settled <= peak, `reservation must settle at or below its peak (${settled} vs ${peak})`);
  assert(settled > baseline, 'a built cluster layer keeps a measured footprint');

  const initial = await readClustersWhenPresent(3);
  assert(initial.clusterCount > 0);
  const inspected = initial.clusters.find(cluster => cluster.total > 1) ?? initial.clusters[0];
  assert(inspected.rows.length > 0 && inspected.total >= inspected.rows.length, 'hover members are read from the index');
  summary.initial = {clusterCount: initial.clusterCount, zoom: initial.view.zoom};

  // Zoom in first, then out: the index extends instead of stalling the main thread.
  const startZoom = initial.view.zoom;
  await page.evaluate(zoom => window.__keplerComputeTest.setMapViewport({zoom}), startZoom + 3);
  await waitForClusters('cluster-admission', startZoom + 3);
  await page.waitForTimeout(4000);
  const zoomedIn = await readClustersWhenPresent(1);
  await page.evaluate(() => {
    window.__longTaskMax = 0;
    new PerformanceObserver(list => {
      for (const entry of list.getEntries()) window.__longTaskMax = Math.max(window.__longTaskMax, entry.duration);
    }).observe({entryTypes: ['longtask']});
  });
  const outZoom = Math.max(1, startZoom - 2);
  await page.evaluate(zoom => window.__keplerComputeTest.setMapViewport({zoom}), outZoom);
  await waitForClusters('cluster-admission', outZoom);
  await page.waitForTimeout(4000);
  const zoomedOut = await readClustersWhenPresent(1);
  assert(zoomedOut.clusterCount !== zoomedIn.clusterCount || zoomedOut.view.zoom !== zoomedIn.view.zoom, 'zoom out requeries the index');
  const longTask = await page.evaluate(() => window.__longTaskMax);
  assert(longTask < 3000, `zooming out after a zoomed-in build must not stall the main thread (${longTask} ms)`);
  const afterZoomScratch = await readScratch();
  summary.zoom = {in: zoomedIn.clusterCount, out: zoomedOut.clusterCount, longTask, scratchAfterZoom: afterZoomScratch};
  assert(afterZoomScratch >= settled * 0.5, 'reservation remains covering the coarser hierarchy');

  stage('zoomed');
  // Radius and mask changes rebuild and redraw.
  await applyCluster('cluster-admission', 80);
  await page.waitForFunction(async () => {
    const read = await window.__keplerComputeTest.readClusters('cluster-admission', 1);
    return read && read.clusterCount > 0;
  }, undefined, {timeout: 120000, polling: 500});
  await page.waitForTimeout(3000);
  const radiusRead = await page.waitForFunction(() => window.__keplerComputeTest.readClusters('cluster-admission', 1), undefined, {timeout: 60000}).then(handle => handle.jsonValue());
  summary.radius = radiusRead.clusterCount;
  assert.notEqual(radiusRead.clusterCount, zoomedOut.clusterCount, 'radius change rebuilds the clusters');
  await applyCluster('cluster-admission', 80, [20, 60]);
  await page.waitForFunction(() => window.__keplerComputeTest.getState().selectedCount < window.__keplerComputeTest.source().length, undefined, {timeout: 60000}).catch(() => {});
  await page.waitForTimeout(3000);
  const rendered = await page.evaluate(() => window.__keplerComputeTest.renderedLayers().filter(layer => layer.id.startsWith('cluster-admission') && layer.visible));
  if (!rendered.length) console.error(JSON.stringify(await page.evaluate(() => window.__keplerComputeTest.renderedLayers().map(layer => ({id: layer.id, renderId: layer.renderId, visible: layer.visible, loaded: layer.loaded})))));
  assert(rendered.length > 0, 'cluster layer is drawn after mask change');

  // Teardown releases the reservation.
  await page.evaluate(() => window.__keplerComputeTest.removeNativeLayer('cluster-admission'));
  await page.waitForFunction(({baseline}) => Number(document.querySelector('[data-category=scratch]')?.dataset.bytes) <= baseline, {baseline}, {timeout: 30000});
  summary.released = await readScratch();
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, rowCount, summary}));
} catch (error) {
  console.error(error, JSON.stringify(summary));
  console.error({pageErrors: errors});
  process.exitCode = 1;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
