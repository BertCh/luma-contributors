// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {PNG} from 'pngjs';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const points = [[-74, 40.74], [-73.96, 40.76]];
const makeBatch = (rows, indices, values) => new arrow.Table({
  east: arrow.makeVector(new Float32Array(indices.map(index => points[index][0]))),
  north: arrow.makeVector(new Float32Array(indices.map(index => points[index][1]))),
  sourceRow: arrow.makeVector(new Float32Array(rows)),
  measurement: arrow.vectorFromArray(values, new arrow.Float32())
});
const source = makeBatch([0, 1], [0, 0], [2, 4]).concat(makeBatch([], [], []), makeBatch([2, 3, 4], [0, 1, 1], [null, 6, 8]));
const sourceBytes = arrow.tableToIPC(source, 'stream');
assert.deepEqual(arrow.tableFromIPC(sourceBytes).batches.map(batch => batch.numRows), [2, 0, 3]);
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
page.setDefaultTimeout(30000);
await pinViteModules(page);
const failures = [];
const checks = [];
let datasetId;
const layerId = 'published-heatmap';
page.on('pageerror', error => failures.push(error.stack ?? error.message));
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())) failures.push(message.text());
});
const watchdog = setTimeout(async () => {
  console.error('Native heatmap workflow exceeded three minutes');
  await Promise.race([browser.close(), new Promise(resolve => setTimeout(resolve, 3000))]);
  process.exit(1);
}, 180000);
async function waitReady(scale, domain) {
  await page.waitForFunction(({layerId, scale, domain}) => {
    const api = window.__keplerComputeTest;
    const state = api.getState();
    if (state.error) throw new Error(state.error);
    const weight = api.getHeatmapState(layerId);
    const rendered = api.renderedLayers().find(layer => layer.id === layerId && layer.visible);
    const readiness = rendered?.heatmapReadiness;
    return rendered?.loaded && readiness?.revision === weight?.revision &&
      readiness.completedRevision === readiness.revision && !readiness.densityPending &&
      readiness.pipelinesReady && readiness.presentationReady &&
      (!scale || (weight?.legend?.scale === scale && weight.legend.revision === weight.styleRevision &&
        (!domain || weight.legend.domain.every((value, index) => Math.abs(value - domain[index]) < 1e-5))));
  }, {layerId, scale, domain}, {timeout: 45000});
}
async function apply(configuration, scale, domain) {
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), configuration);
  await waitReady(scale, domain);
}
async function pixels(name) {
  const image = PNG.sync.read(await page.screenshot({path: `/tmp/luma-kepler-heatmap-${name}.png`}));
  const projected = await page.evaluate(points => points.map(([longitude, latitude]) => window.__keplerComputeTest.projectPosition(longitude, latitude)), points);
  let colored = 0;
  for (const [centerX, centerY] of projected)
    for (let positionY = Math.max(100, Math.floor(centerY - 90)); positionY < Math.min(image.height - 100, centerY + 90); positionY++)
      for (let positionX = Math.max(360, Math.floor(centerX - 90)); positionX < Math.min(image.width - 100, centerX + 90); positionX++) {
        const offset = (positionY * image.width + positionX) * 4;
        const red = image.data[offset], green = image.data[offset + 1], blue = image.data[offset + 2];
        if (red > 120 && red - blue > 50 && red >= green) colored++;
      }
  assert(colored > 100, `${name} must draw actual density pixels around source positions, found ${colored}`);
  return colored;
}
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page, 'Prepare data');
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: 'native-heatmap.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(sourceBytes)});
  await workspace.getByLabel('Source longitude column').selectOption('east');
  await workspace.getByLabel('Source latitude column').selectOption('north');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 5);
  await closeNativeDialog(page);
  const initial = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0]);
  datasetId = initial.id;
  await page.evaluate(identifier => {window.__heatmapSourceBuffers = window.__keplerComputeTest.resourceIdentity().sourceBuffers[identifier];}, datasetId);
  const config = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const layer = config.config.visState.layers.find(layer => layer.config.dataId === datasetId);
  assert(layer);
  layer.id = layerId;
  layer.type = 'heatmap';
  layer.config.label = 'Native resident density';
  layer.config.columns = {lat: 'north', lng: 'east'};
  layer.config.columnMode = 'points';
  layer.config.isVisible = true;
  layer.config.textLabel = [];
  layer.config.visConfig = {...layer.config.visConfig, opacity: 1, radius: 80, intensity: 1, threshold: 0.05, aggregation: 'SUM', colorRange: {colors: ['#ff5500', '#ffff00']}};
  layer.visualChannels = {weightField: null, weightScale: 'linear'};
  config.config.visState.layers = [layer];
  config.config.mapState = {...config.config.mapState, longitude: -73.98, latitude: 40.75, zoom: 12, pitch: 0, bearing: 0};
  await apply(config);
  await pixels('constant');
  checks.push('native constant-one density over custom coordinate roles');
  const weight = page.locator('.layer-config-group').filter({hasText: /Weight/i}).filter({has: page.locator('.field-selector')}).first();
  if (!(await weight.isVisible())) await page.locator('.layer__enable-config svg').first().click();
  if (!(await weight.locator('.field-selector').count())) await weight.locator('.layer-config-group__header').click();
  await weight.locator('.field-selector').first().click();
  await page.locator('.field-selector_list-item').filter({has: page.getByText('measurement', {exact: true})}).click();
  await waitReady('linear', [2, 8]);
  const normalized = await page.evaluate(id => window.__keplerComputeTest.getHeatmapState(id), layerId);
  assert.deepEqual(normalized.weightChunks, [2, 0, 3]);
  assert.equal(normalized.legend.finiteCount, 4);
  await pixels('native-weight');
  checks.push('actual native weight dropdown and nullable GPU [0,1] normalization');
  const weightedConfig = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const nativeLayer = weightedConfig.config.visState.layers[0];
  for (const aggregation of ['MEAN', 'SUM']) {
    nativeLayer.config.visConfig.aggregation = aggregation;
    nativeLayer.config.visConfig.radius = aggregation === 'MEAN' ? 45 : 90;
    await apply(weightedConfig, 'linear', [2, 8]);
    await pixels(aggregation.toLowerCase());
  }
  checks.push('native SUM/MEAN, radius and intensity configuration');
  for (const scale of ['sqrt', 'log', 'linear']) {
    nativeLayer.visualChannels.weightScale = scale;
    await apply(weightedConfig, scale, [2, 8]);
    assert.deepEqual((await page.evaluate(id => window.__keplerComputeTest.getHeatmapState(id), layerId)).weightBufferIds, normalized.weightBufferIds);
  }
  checks.push('native linear/sqrt/log scales reuse resident normalized buffers');
  await page.evaluate(identifier => window.__keplerComputeTest.setPublishedRange(identifier, 'sourceRow', 0, 2), datasetId);
  await page.waitForFunction(identifier => window.__keplerComputeTest.getState().publications.find(value => value.id === identifier)?.selectedCount === 3, datasetId);
  const filtered = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const range = filtered.config.visState.filters.find(filter => filter.dataId.includes(datasetId));
  assert(range);
  range.fixedDomain = false;
  await apply(filtered, 'linear', [2, 4]);
  await pixels('selected-domain');
  await apply(weightedConfig, 'linear', [2, 8]);
  await page.waitForFunction(identifier => window.__keplerComputeTest.getState().publications.find(value => value.id === identifier)?.selectedCount === 5, datasetId);
  checks.push('independent display and dynamic DOMAIN scopes plus empty-filter replay');
  const finalSource = await page.evaluate(identifier => window.__keplerComputeTest.getState().publications.find(value => value.id === identifier), datasetId);
  assert.equal(finalSource.sourceUploadBytes, initial.sourceUploadBytes);
  assert.deepEqual(finalSource.sourceBufferIds, initial.sourceBufferIds);
  assert.deepEqual(finalSource.sourceBatches, [2, 0, 3]);
  assert(await page.evaluate(identifier => window.__keplerComputeTest.resourceIdentity().sourceBuffers[identifier].every((buffer, index) => buffer === window.__heatmapSourceBuffers[index]), datasetId));
  await pixels('final');
  assert.deepEqual(failures, []);
  console.log(JSON.stringify({passed: true, sourceBatches: [2, 0, 3], checks}, null, 2));
} catch (error) {
  console.error(JSON.stringify(await page.evaluate(id => ({state: window.__keplerComputeTest?.getState(), heatmap: window.__keplerComputeTest?.getHeatmapState(id), rendered: window.__keplerComputeTest?.renderedLayers()}), layerId).catch(() => null), null, 2));
  console.error(failures);
  throw error;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
