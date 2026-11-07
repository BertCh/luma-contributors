// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser check for the resident cluster adapter. Every source size uses exact native supercluster,
// built in a Web Worker from chunked GPU readbacks. This compares cluster count, order, centers,
// member sets and native aggregates against plain supercluster run in node, at 300,000+ rows,
// several zooms, with and without a filter mask. Requires a WebGPU browser and a dev server.
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import Supercluster from 'supercluster';
import {max, mean, median, min, sum} from 'd3-array';
import * as geoViewport from '@mapbox/geo-viewport';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const largeRowCount = Number(process.argv[3] || 300000);
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const errors = [];
const allMessages = [];
const watchdog = setTimeout(() => {
  console.error('Cluster check exceeded 400 seconds');
  void browser.close();
  process.exit(1);
}, 400000);

async function openPage(rowCount) {
  const page = await browser.newPage({viewport: {width: 1200, height: 900}});
  await pinViteModules(page);
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error' || message.type() === 'warning') allMessages.push(message.text().slice(0, 600));
    if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer|BindGroup)|validation error|exceeds the maximum|Uncaught|destroyed/i.test(message.text()))
      errors.push(message.text());
  });
  await page.goto(`${process.argv[2] || 'http://127.0.0.1:5318/'}?rows=${rowCount}`, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(count => window.__keplerComputeTest?.getState().selectedCount === count, rowCount, {timeout: 90000});
  return page;
}

async function applyClusterLayer(page, layerId, minClusterSize, colorField = null, colorAggregation = 'count', valueRange = null) {
  const original = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const configuration = structuredClone(original);
  const layer = configuration.config.visState.layers[0];
  layer.id = layerId;
  layer.type = 'cluster';
  layer.config.columnMode = 'points';
  layer.config.columns = {lat: 'latitude', lng: 'longitude'};
  layer.config.colorField = colorField ? {name: colorField, type: 'real'} : null;
  layer.visualChannels = {...layer.visualChannels, colorField: colorField ? {name: colorField, type: 'real'} : null};
  layer.config.sizeField = null;
  layer.config.colorScale = 'quantize';
  layer.config.textLabel = [];
  layer.config.visConfig = {...layer.config.visConfig, clusterRadius: 40, radiusRange: [1, 40], minClusterSize, colorAggregation,
    colorRange: {colors: ['#21918c', '#fde725']}};
  if (valueRange) {
    const valueFilter = configuration.config.visState.filters.find(filter => filter.name?.includes('value'));
    assert(valueFilter, 'the sample has a value filter');
    valueFilter.value = valueRange;
  }
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), configuration);
}

async function waitForClusters(page, layerId, predicate = () => true) {
  await page.waitForFunction(async ({id, check}) => {
    const api = window.__keplerComputeTest;
    const state = api.getState();
    if (state.error) throw new Error(state.error);
    const invalid = state.nativeLayers?.find(layer => layer.id === id && layer.valid === false);
    if (invalid) throw new Error(JSON.stringify(invalid));
    const read = await api.readClusters(id, 1);
    return Boolean(read && read.clusterCount > 0 && (check === 0 || read.view.zoom === check));
  }, {id: layerId, check: 0}, {timeout: 60000, polling: 500});
  await page.waitForTimeout(500);
}

const aggregators = {sum, average: mean, minimum: min, maximum: max, median};

/** Plain supercluster (default options, no custom reduce) over the float32 positions. */
function buildOracle(rows, include, clusterRadius) {
  const features = [];
  for (let row = 0; row < rows.length; row++) {
    if (!include(rows[row])) continue;
    features.push({type: 'Feature', properties: {}, id: row, geometry: {type: 'Point', coordinates: [rows[row][0], rows[row][1]]}});
  }
  return new Supercluster({maxZoom: 20, radius: clusterRadius}).load(features);
}

/** Expected clusters of one view, as native clustersAtZoom reports them. */
function queryOracle(oracle, view, minClusterSize) {
  const zoom = Math.round(view.zoom);
  const bounds = geoViewport.bounds([view.longitude, view.latitude], zoom, [view.width, view.height]);
  const cutoff = Number.isFinite(minClusterSize) ? minClusterSize : 1;
  const clusters = [];
  for (const feature of oracle.getClusters(bounds, zoom)) {
    const members = feature.properties.cluster ? oracle.getLeaves(feature.properties.cluster_id, Infinity).map(leaf => leaf.id) : [feature.id];
    if (members.length < cutoff) continue;
    clusters.push({members: members.sort((a, b) => a - b), center: feature.geometry.coordinates});
  }
  return clusters;
}

/** Returns a description of the first difference between the binding and plain supercluster. */
function findDifference(read, expected, rows, aggregation) {
  if (read.clusterCount !== expected.length) return `cluster count ${read.clusterCount} vs ${expected.length}`;
  for (let index = 0; index < expected.length; index++) {
    const cluster = read.clusters[index];
    const reference = expected[index];
    if (cluster.total !== reference.members.length) return `cluster ${index} size ${cluster.total} vs ${reference.members.length}`;
    if (read.centers[index * 2] !== Math.fround(reference.center[0]) || read.centers[index * 2 + 1] !== Math.fround(reference.center[1]))
      return `cluster ${index} center ${read.centers[index * 2]},${read.centers[index * 2 + 1]} vs ${reference.center}`;
    const members = [...cluster.rows].sort((a, b) => a - b);
    if (members.length !== reference.members.length || members.some((row, position) => row !== reference.members[position])) return `cluster ${index} members differ`;
    const expectedValue = aggregation === 'count' ? reference.members.length : aggregators[aggregation](reference.members, row => rows[row][2]);
    const tolerance = Math.max(1e-6, Math.abs(expectedValue) * 1e-6);
    if (!(Math.abs(cluster.value - expectedValue) <= tolerance)) return `cluster ${index} ${aggregation} ${cluster.value} vs ${expectedValue}`;
  }
  return null;
}

async function compareToOracle(page, layerId, label, oracle, rows, minClusterSize, aggregation, expectedView) {
  let last = 'no clusters';
  const deadline = Date.now() + 40000;
  while (Date.now() < deadline) {
    const read = await page.evaluate(id => window.__keplerComputeTest.readClusters(id, 10000000), layerId);
    if (read && read.clusterCount > 0 && (!expectedView || Math.abs(read.view.zoom - expectedView.zoom) < 1e-9)) {
      const expected = queryOracle(oracle, read.view, minClusterSize);
      const difference = findDifference(read, expected, rows, aggregation);
      if (!difference) return {clusterCount: read.clusterCount, rowCount: expected.reduce((total, cluster) => total + cluster.members.length, 0), view: read.view};
      last = difference;
    }
    await page.waitForTimeout(500);
  }
  assert.fail(`${label}: ${last}`);
}

const summary = {};
try {
  const page = await openPage(largeRowCount);
  const rows = await page.evaluate(() => window.__keplerComputeTest.source());
  assert.equal(rows.length, largeRowCount);
  const everyRow = () => true;
  // 1. Unmasked, default member-count measure, at several zooms.
  await applyClusterLayer(page, 'cluster-native', 1);
  await waitForClusters(page, 'cluster-native');
  let oracle = buildOracle(rows, everyRow, 40);
  summary.initial = await compareToOracle(page, 'cluster-native', 'initial', oracle, rows, 1, 'count');
  console.log(JSON.stringify({initial: summary.initial}));
  const view = summary.initial.view;
  const changes = [['pan', {longitude: view.longitude + 0.02, latitude: view.latitude + 0.01}], ['zoom-in', {zoom: view.zoom + 1.2}], ['zoom-out', {zoom: Math.max(1, view.zoom - 2)}], ['zoom-fraction', {zoom: view.zoom + 0.3}], ['zoom-far-in', {zoom: 17}]];
  for (const [name, change] of changes) {
    await page.evaluate(change_ => window.__keplerComputeTest.setMapViewport(change_), change);
    summary[name] = await compareToOracle(page, 'cluster-native', name, oracle, rows, 1, 'count', change.zoom ? {zoom: change.zoom} : undefined);
    console.log(JSON.stringify({[name]: summary[name]}));
  }
  // 2. minClusterSize filter.
  await applyClusterLayer(page, 'cluster-native-min', 3);
  await waitForClusters(page, 'cluster-native-min');
  summary.min = await compareToOracle(page, 'cluster-native-min', 'min=3', oracle, rows, 3, 'count');
  console.log(JSON.stringify({min3: summary.min}));
  // 3. Native aggregations of the value field (min, max and median included).
  for (const aggregation of ['sum', 'average', 'minimum', 'maximum', 'median']) {
    await applyClusterLayer(page, `cluster-native-${aggregation}`, 1, 'value', aggregation);
    await waitForClusters(page, `cluster-native-${aggregation}`);
    summary[aggregation] = await compareToOracle(page, `cluster-native-${aggregation}`, aggregation, oracle, rows, 1, aggregation);
    console.log(JSON.stringify({[aggregation]: summary[aggregation]}));
  }
  // 4. Filter mask: the value filter removes rows before clustering.
  const valueRange = [20, 60];
  await applyClusterLayer(page, 'cluster-native-masked', 1, 'value', 'average', valueRange);
  const expectedSelected = rows.filter(row => row[2] >= valueRange[0] && row[2] <= valueRange[1]).length;
  await page.waitForFunction(count => window.__keplerComputeTest.getState().selectedCount === count, expectedSelected, {timeout: 60000});
  await waitForClusters(page, 'cluster-native-masked');
  oracle = buildOracle(rows, row => row[2] >= valueRange[0] && row[2] <= valueRange[1], 40);
  summary.masked = await compareToOracle(page, 'cluster-native-masked', 'masked', oracle, rows, 1, 'average');
  assert.equal(summary.masked.rowCount <= expectedSelected, true);
  console.log(JSON.stringify({masked: summary.masked, expectedSelected}));
  await page.evaluate(change_ => window.__keplerComputeTest.setMapViewport(change_), {zoom: Math.max(1, view.zoom - 1)});
  summary.maskedZoom = await compareToOracle(page, 'cluster-native-masked', 'masked zoom', oracle, rows, 1, 'average', {zoom: Math.max(1, view.zoom - 1)});
  console.log(JSON.stringify({maskedZoom: summary.maskedZoom}));
  await page.close();
  assert.deepEqual(errors, [], `validation or page errors:\n${errors.join('\n')}`);
  console.log(JSON.stringify({passed: true, largeRowCount, summary}));
} catch (error) {
  console.error(error);
  if (errors.length) console.error('errors:', errors);
  console.error('console errors/warnings:', allMessages.slice(0, 12));
  process.exitCode = 1;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
