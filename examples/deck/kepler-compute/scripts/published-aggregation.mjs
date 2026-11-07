// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const coordinates = [[-74, 40.74], [-73.96, 40.76]];
const makeBatch = (rows, locations, measurements) => new arrow.Table({
  east: arrow.makeVector(new Float32Array(locations.map(location => location[0]))),
  north: arrow.makeVector(new Float32Array(locations.map(location => location[1]))),
  sourceRow: arrow.makeVector(new Float32Array(rows)),
  measurement: arrow.vectorFromArray(measurements, new arrow.Float32())
});
// Coordinate-null publication is currently rejected by the UI; the runtime GPU suite covers it.
// Nullable weights and the empty source batch must survive publication and native config replay.
const source = makeBatch([0, 1], [coordinates[0], coordinates[0]], [2, null]).concat(
  makeBatch([], [], []),
  makeBatch([2, 3, 4], [coordinates[0], coordinates[1], coordinates[1]], [6, -2, 10])
);
const sourceBytes = arrow.tableToIPC(source, 'stream');
assert.deepEqual(arrow.tableFromIPC(sourceBytes).batches.map(batch => batch.numRows), [2, 0, 3]);
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
page.setDefaultTimeout(30000);
const watchdog = setTimeout(async () => {
  console.error('Published aggregation fixture exceeded its three-minute deadline');
  await Promise.race([browser.close(), new Promise(resolve => setTimeout(resolve, 5000))]);
  process.exit(1);
}, 180000);
await pinViteModules(page);
const failures = [];
page.on('pageerror', error => failures.push(error.message));
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())) failures.push(message.text());
});
let datasetId;
const checks = [];
const cells = {};
const publication = async () => page.evaluate(identifier => window.__keplerComputeTest.getState().publications.find(value => value.id === identifier), datasetId);
async function apply(config, visibleId) {
  const frame = await page.evaluate(() => window.__keplerComputeTest.getState().frameCount);
  await page.evaluate(configuration => window.__keplerComputeTest.applyNativeConfig(configuration), config);
  await page.waitForFunction(({datasetId, visibleId, frame}) => {
    const api = window.__keplerComputeTest;
    const state = api.getState();
    if (state.error) throw new Error(state.error);
    const source = state.publications.find(value => value.id === datasetId);
    return !source?.scopePending && state.frameCount > frame && api.renderedLayers().some(layer => layer.id === visibleId && layer.loaded && layer.visible);
  }, {datasetId, visibleId, frame}, {timeout: 45000});
}
async function readCell(layerId, cellId) {
  const before = (await publication()).aggregation.readbackBytes;
  const value = await page.evaluate(({datasetId, layerId, cellId}) => window.__keplerComputeTest.readPublishedAggregate(datasetId, layerId, cellId), {datasetId, layerId, cellId});
  const after = (await publication()).aggregation.readbackBytes;
  assert.equal(after - before, 12, 'One cell inspection must transfer exactly count/value/overflow words');
  return value;
}
async function pickCell(layerId, locationIndex) {
  const [longitude, latitude] = coordinates[locationIndex];
  const picked = await page.evaluate(({datasetId, longitude, latitude}) => window.__keplerComputeTest.pickPublishedPosition(datasetId, longitude, latitude), {datasetId, longitude, latitude});
  assert.equal(picked?.layerId, layerId, 'Native GPU picking must select the published aggregate layer');
  assert(Number.isSafeInteger(picked.sourceRowIndex) && picked.sourceRowIndex >= 0);
  return picked.sourceRowIndex;
}
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page, 'Prepare data');
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: 'published-aggregates.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(sourceBytes)});
  await workspace.getByLabel('Source longitude column').selectOption('east');
  await workspace.getByLabel('Source latitude column').selectOption('north');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 5);
  await closeNativeDialog(page);
  const initialState = await page.evaluate(() => window.__keplerComputeTest.getState());
  datasetId = initialState.publications[0].id;
  const initialSource = await publication();
  assert.deepEqual(initialSource.sourceBatches, [2, 0, 3]);
  const configuration = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const template = configuration.config.visState.layers.find(layer => layer.config.dataId === datasetId);
  assert(template, 'Publication must create a native layer');
  for (const layer of configuration.config.visState.layers) layer.config.isVisible = false;
  const aggregateLayers = ['grid', 'hexagon'].map(type => {
    const layer = structuredClone(template);
    layer.id = `published-${type}`;
    layer.type = type;
    layer.config.columnMode = 'points';
    layer.config.columns = {lat: 'north', lng: 'east'};
    layer.config.textLabel = [];
    layer.config.isVisible = false;
    layer.config.visConfig = {...layer.config.visConfig, worldUnitSize: 0.5, coverage: 1, enable3d: false,
      percentile: [0, 100], opacity: 1, colorAggregation: 'average', colorRange: {colors: ['#f56e28', '#ffd52a']}};
    layer.visualChannels = {...layer.visualChannels, colorField: null, sizeField: null, colorScale: 'quantize'};
    return layer;
  });
  configuration.config.visState.layers.push(...aggregateLayers);
  configuration.config.mapState = {...configuration.config.mapState, longitude: -73.98, latitude: 40.75, zoom: 12, pitch: 0, bearing: 0};
  for (const layer of aggregateLayers) {
    for (const other of aggregateLayers) other.config.isVisible = other === layer;
    await apply(configuration, layer.id);
    cells[layer.id] = [await pickCell(layer.id, 0), await pickCell(layer.id, 1)];
    const observations = [];
    for (const cellId of cells[layer.id]) observations.push((await readCell(layer.id, cellId)).Observations);
    assert.deepEqual(observations, [3, 2]);
    const projection = (await publication()).aggregation;
    assert.equal(projection.contract, 'mercator-centroid-meters-float32');
    assert.deepEqual(projection.projectedChunks, [2, 0, 3]);
    checks.push(`${layer.type}: native published count, picking and bounded cells`);
  }
  const projected = (await publication()).aggregation;
  await page.evaluate(identifier => {
    window.__publishedAggregateBuffers = window.__keplerComputeTest.resourceIdentity().aggregationBuffers[identifier];
  }, datasetId);
  for (const layer of aggregateLayers) {
    for (const other of aggregateLayers) other.config.isVisible = other === layer;
    layer.visualChannels.colorField = {name: 'measurement', type: 'real'};
    layer.config.visConfig.colorAggregation = 'average';
    await apply(configuration, layer.id);
    const mean = await readCell(layer.id, cells[layer.id][0]);
    assert.equal(mean.Observations, 2, 'Missing weights are excluded from the weighted population');
    assert.equal(mean.mean, 4);
    layer.config.visConfig.colorAggregation = 'sum';
    await apply(configuration, layer.id);
    assert.equal((await readCell(layer.id, cells[layer.id][0])).sum, 8);
    checks.push(`${layer.type}: native nullable average and sum controls`);
  }
  // A source filter applies to both independent aggregation programs without source uploads.
  await page.evaluate(identifier => window.__keplerComputeTest.setPublishedRange(identifier, 'sourceRow', 0, 3), datasetId);
  await page.waitForFunction(identifier => window.__keplerComputeTest.getState().publications.find(value => value.id === identifier)?.selectedCount === 4, datasetId);
  assert.equal((await readCell('published-hexagon', cells['published-hexagon'][1])).sum, -2);
  assert.equal((await page.evaluate(() => window.__keplerComputeTest.getState())).selectedCount, 256, 'Published filters must not alter the fixture dataset');
  checks.push('source-specific range filtering');
  await page.evaluate(() => window.__keplerComputeTest.addPolygon([[[-74.01, 40.73], [-73.99, 40.73], [-73.99, 40.75], [-74.01, 40.75], [-74.01, 40.73]]]));
  await page.waitForFunction(() => window.__keplerComputeTest.getState().nativeFilters.some(filter => filter.type === 'polygon'));
  const filtered = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  for (const layer of filtered.config.visState.layers.filter(layer => layer.id.startsWith('published-'))) layer.config.isVisible = true;
  const polygon = filtered.config.visState.filters.find(filter => filter.type === 'polygon');
  assert(polygon, 'Native polygon action must create a serializable filter');
  polygon.dataId = [datasetId];
  polygon.layerId = ['published-grid'];
  polygon.name = ['Published grid'];
  await apply(filtered, 'published-grid');
  assert.equal(await readCell('published-grid', cells['published-grid'][1]), null, 'Targeted grid polygon must exclude its eastern cell');
  assert.equal((await readCell('published-hexagon', cells['published-hexagon'][1])).sum, -2, 'Untargeted Hex must retain its independent layer population');
  checks.push('independent layer polygon scope');
  const saved = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  await apply(configuration, 'published-hexagon');
  assert.equal((await readCell('published-hexagon', cells['published-hexagon'][1])).sum, 8);
  await apply(saved, 'published-grid');
  assert.equal(await readCell('published-grid', cells['published-grid'][1]), null);
  assert.equal((await readCell('published-hexagon', cells['published-hexagon'][1])).sum, -2);
  const finalSource = await publication();
  assert.deepEqual(finalSource.sourceBufferIds, initialSource.sourceBufferIds);
  assert.equal(finalSource.sourceUploadBytes, initialSource.sourceUploadBytes);
  assert.deepEqual(finalSource.sourceBatches, [2, 0, 3]);
  assert.deepEqual(finalSource.aggregation.projectedBufferIds, projected.projectedBufferIds);
  assert.equal(finalSource.aggregation.uploadBytes, projected.uploadBytes);
  assert(await page.evaluate(identifier => window.__keplerComputeTest.resourceIdentity().aggregationBuffers[identifier].every((buffer, index) => buffer === window.__publishedAggregateBuffers[index]), datasetId));
  checks.push('saved native config replay with stable source and projection buffers');
  await page.screenshot({path: '/tmp/luma-kepler-published-aggregation.png'});
  assert.deepEqual(failures, []);
  console.log(JSON.stringify({passed: true, contract: projected.contract, sourceBatches: [2, 0, 3], nullableWeight: true, checks}, null, 2));
} catch (error) {
  const diagnostic = await Promise.race([
    page.evaluate(() => ({state: window.__keplerComputeTest?.getState(), layers: window.__keplerComputeTest?.renderedLayers(), config: window.__keplerComputeTest?.saveNativeConfig()})).catch(error => ({error: String(error)})),
    new Promise(resolve => setTimeout(() => resolve({diagnostic: 'Page did not respond within five seconds'}), 5000))
  ]);
  console.error(JSON.stringify(diagnostic, null, 2));
  throw error;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
