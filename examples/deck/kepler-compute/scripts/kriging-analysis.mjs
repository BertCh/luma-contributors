// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Usage: node scripts/kriging-analysis.mjs [URL] (default: http://127.0.0.1:5318/?rows=256).
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
// The old fork obtained UTIF accidentally through three's hoisted dependencies.
// Declare the standalone decoder so this check also runs in the published-package workspace.
import tiffDecoder from 'utif';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const bounds = [-74, 40, -73.5, 40.5];
const longitudes = [-73.875, -73.625, -73.875, -73.625];
const latitudes = [40.375, 40.375, 40.125, 40.125];
// Both batches explicitly declare a nullable Float32 response, including the all-valid first batch.
const source = arrow.tableFromArrays({
  longitude: new Float32Array(longitudes.slice(0, 2)),
  latitude: new Float32Array(latitudes.slice(0, 2)),
  response: arrow.vectorFromArray([4, 4], new arrow.Float32())
}).concat(arrow.tableFromArrays({
  longitude: new Float32Array(longitudes.slice(2)),
  latitude: new Float32Array(latitudes.slice(2)),
  response: arrow.vectorFromArray([4, null], new arrow.Float32())
}));
assert.deepEqual(source.batches.map(batch => batch.numRows), [2, 2]);
assert.equal(source.schema.fields.find(field => field.name === 'response').nullable, true);
const operation = {
  kind: 'kriging-points', column: 'response', longitudeColumn: 'longitude', latitudeColumn: 'latitude',
  coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'meters',
  grid: {width: 2, height: 2, bounds}, searchRadiusMeters: 100000,
  neighborCount: 3, minimumNeighbors: 2,
  variogram: {model: 'spherical', nugget: 0, sill: 1, rangeMeters: 100000}
};
// Independent binary64 ordinary-kriging reference at the fourth cell, using the three
// non-null samples above: solve [Gamma, 1; 1^T, 0] [weights, multiplier] = [gamma, 1],
// then variance = weights dot gamma + multiplier. Distances are haversine meters on
// R=6371008.8; nugget=0, sill=1, range=100000, with Matérn effective-range factors
// 4.749031386012701 (3/2) and 5.924462618951797 (5/2). These are rounded to six decimals.
const REFERENCE_INTERPOLATED_VARIANCES = {
  spherical: 0.458716,
  'matern-3/2': 0.370828,
  'matern-5/2': 0.252149
};
// Allow float32 GPU distance/solve rounding and the inspector's six significant digits;
// this is much smaller than the minimum inter-family variance difference (over 0.08).
const MODEL_VARIANCE_TOLERANCE = 0.00001;
function assertModelVariance(model, variance) {
  assert(Math.abs(variance - REFERENCE_INTERPOLATED_VARIANCES[model]) < MODEL_VARIANCE_TOLERANCE,
    `${model} variance ${variance} differs from independent reference ${REFERENCE_INTERPOLATED_VARIANCES[model]}`);
}
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const failures = [];
const diagnostics = [];
let currentStage = 'startup';
const watchdog = setTimeout(() => {
  console.error(`Kriging workflow exceeded 120 seconds at ${currentStage}`);
  void browser.close();
}, 120000);
function stage(name) {
  currentStage = name;
  console.log(JSON.stringify({stage: name}));
}
page.on('pageerror', error => {
  failures.push(error.message);
  console.error('pageerror:', error.stack ?? error.message);
});
page.on('console', message => {
  if (message.type() !== 'error') return;
  diagnostics.push(message.text());
  if (/WGSL|Invalid.*(?:Pipeline|Buffer)|validation error|Maximum call stack/i.test(message.text()))
    failures.push(message.text());
  console.error('console:', message.text());
});
page.on('requestfailed', request => diagnostics.push(`${request.url()}: ${request.failure()?.errorText}`));

function assertPredictions(values) {
  assert.equal(values.length, 4);
  for (const value of values) assert(Math.abs(value - 4) < 0.0001, `Unexpected prediction ${value}`);
}
function assertVariances(values) {
  assert.equal(values.length, 4);
  for (const value of values.slice(0, 3)) assert.equal(value, 0);
  assert(Number.isFinite(values[3]) && values[3] > 0, `Expected positive interpolated variance, got ${values[3]}`);
}
async function waitForGrid(datasetId) {
  await page.waitForFunction(id => {
    const panel = document.querySelector('[aria-label="GPU analysis"]');
    return panel?.textContent.includes('4 analyzed') && panel.textContent.includes('0 undefined') &&
      window.__keplerComputeTest.renderedLayers().some(layer =>
        layer.id === `resident-kriging-points-${id}` && layer.layerClass === 'BitmapLayer' && layer.visible && layer.loaded);
  }, datasetId, {timeout: 30000});
}
async function readGridRows(panel) {
  const details = panel.locator('.interpolation-grid-results');
  if (!(await details.evaluate(element => element.open)))
    await panel.getByText('Inspect interpolation grid cells', {exact: true}).click();
  const table = panel.getByRole('table', {name: 'Interpolation grid cells', exact: true});
  await page.waitForFunction(() => document.querySelectorAll('[aria-label="Interpolation grid cells"] tbody tr').length === 4);
  assert.deepEqual(await table.locator('thead th').allTextContents(),
    ['Cell', 'Longitude', 'Latitude', 'Prediction', 'Variance (squared response units)']);
  assert.match(await details.innerText(), /3 contributing source candidates/);
  const rows = await table.locator('tbody tr').evaluateAll(elements => elements.map(row =>
    [...row.querySelectorAll('td')].map(cell => Number(cell.textContent.replaceAll(',', '')))));
  assert.deepEqual(rows.map(row => row[0]), [0, 1, 2, 3]);
  assert.deepEqual(rows.map(row => row[1]), longitudes);
  assert.deepEqual(rows.map(row => row[2]), latitudes);
  assertPredictions(rows.map(row => row[3]));
  assertVariances(rows.map(row => row[4]));
  const diagnosticText = await panel.getByRole('region', {name: 'Kriging grid diagnostics', exact: true}).innerText();
  assert.match(diagnosticText, /0 cells unavailable from invalid settings/);
  assert.match(diagnosticText, /0 cells unavailable from numerically unreliable solutions/);
  assert.match(diagnosticText, /4 defined and 0 nodata cells/);
  return rows;
}
async function downloadBand(panel, metric, savedRecipe) {
  const pending = page.waitForEvent('download');
  await panel.getByRole('button', {name: `Download kriging ${metric} GeoTIFF`, exact: true}).click();
  const download = await pending;
  assert.equal(await download.failure(), null);
  assert(download.suggestedFilename().endsWith(`-kriging-${metric}.tif`));
  const stream = await download.createReadStream();
  assert(stream);
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  const bytes = Uint8Array.from(Buffer.concat(chunks));
  const images = tiffDecoder.decode(bytes.buffer);
  assert.equal(images.length, 1);
  const image = images[0];
  tiffDecoder.decodeImage(bytes.buffer, image);
  assert.deepEqual([image.width, image.height], [2, 2]);
  assert.deepEqual(image.t277, [1]);
  assert.deepEqual(image.t339, [3]);
  assert.deepEqual(image.t33550, [0.25, 0.25, 0]);
  assert.deepEqual(image.t33922, [0, 0, 0, -74, 40.5, 0]);
  assert.deepEqual(image.t34735, [1, 1, 1, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326]);
  assert.equal(image.t42113[0], 'nan');
  const provenance = JSON.parse(image.t270[0]);
  assert.equal(provenance.metric, metric);
  assert.equal(provenance.coordinateReferenceSystem, 'EPSG:4326');
  assert.deepEqual(provenance.recipe, savedRecipe);
  assert.deepEqual(provenance.band, {
    index: 1, name: `Ordinary kriging ${metric}`, sourceColumn: 'response', valueUnitsPower: metric === 'variance' ? 2 : 1
  });
  assert(image.t42112[0].includes(`role="description">Ordinary kriging ${metric}</Item>`));
  const raster = new DataView(image.data.buffer, image.data.byteOffset, image.data.byteLength);
  const values = Array.from({length: 4}, (_, index) => raster.getFloat32(index * 4, true));
  if (metric === 'prediction') assertPredictions(values);
  else assertVariances(values);
  return values;
}

try {
  await pinViteModules(page);
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  stage('publish nullable multibatch geographic samples through Prepare data');
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: 'kriging-analysis.arrow', mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))
  });
  await workspace.getByLabel('Source longitude column', {exact: true}).selectOption('longitude');
  await workspace.getByLabel('Source latitude column', {exact: true}).selectOption('latitude');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 4, undefined, {timeout: 30000});
  const initial = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0]);
  const datasetId = initial.id;
  assert.deepEqual(initial.sourceBatches, [2, 2]);
  await closeNativeDialog(page);
  const panel = page.getByRole('region', {name: 'GPU analysis', exact: true});
  await panel.getByLabel('Dataset', {exact: true}).selectOption(datasetId);
  await panel.getByLabel('Numeric field', {exact: true}).selectOption('response');
  await panel.getByText('Configure ordinary kriging grid', {exact: true}).click();
  await panel.getByLabel('Kriging longitude field', {exact: true}).selectOption('longitude');
  await panel.getByLabel('Kriging latitude field', {exact: true}).selectOption('latitude');
  await panel.getByLabel('Kriging variogram model', {exact: true}).selectOption('spherical');
  for (const [label, value] of [
    ['Kriging west longitude', -74], ['Kriging south latitude', 40],
    ['Kriging east longitude', -73.5], ['Kriging north latitude', 40.5],
    ['Kriging columns', 2], ['Kriging rows', 2], ['Kriging search radius in meters', 100000],
    ['Kriging nearest samples', 3], ['Kriging minimum distinct positions', 2],
    ['Kriging nugget in squared response units', 0], ['Kriging partial sill in squared response units', 1],
    ['Kriging effective range in meters', 100000]
  ]) await panel.getByLabel(label, {exact: true}).fill(String(value));
  stage('run manually configured ordinary kriging and inspect both output metrics');
  await panel.getByRole('button', {name: 'Apply model and run kriging', exact: true}).click();
  await waitForGrid(datasetId);
  assert.equal(await panel.getByLabel('Operation', {exact: true}).inputValue(), 'kriging-points');
  await readGridRows(panel);
  await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
  const savedRecipe = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
  assert.equal(savedRecipe.source.datasetId, datasetId);
  assert.equal(savedRecipe.source.revision, initial.sourceRevision);
  assert.equal(savedRecipe.population, 'all');
  assert.deepEqual(savedRecipe.operation, operation);
  stage('download and independently decode prediction and variance GeoTIFF bands');
  const prediction = await downloadBand(panel, 'prediction', savedRecipe);
  const variance = await downloadBand(panel, 'variance', savedRecipe);
  assertModelVariance('spherical', variance[3]);
  assert.notDeepEqual(prediction, variance);
  stage('compare real bitmap pixels with the overlay disabled at a fixed camera');
  const cameraDispatch = await page.evaluate(() => {
    window.__keplerComputeTest.setMapViewport({longitude: -73.75, latitude: 40.25, zoom: 9, bearing: 0, pitch: 0});
    return performance.now();
  });
  await page.waitForFunction(({datasetId, timestamp}) => {
    const frame = window.__keplerComputeTest.getPerformanceSnapshot(datasetId).frame;
    return frame?.stage === 'submitted' && frame.timestamp >= timestamp;
  }, {datasetId, timestamp: cameraDispatch});
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const center = await page.evaluate(() => window.__keplerComputeTest.projectPosition(-73.75, 40.25));
  const clip = {x: Math.floor(center[0] - 32), y: Math.floor(center[1] - 32), width: 64, height: 64};
  assert(clip.x >= 0 && clip.x + clip.width <= 1440 && clip.y >= 0 && clip.y + clip.height <= 1000);
  await page.mouse.move(20, 20);
  const enabledPixels = PNG.sync.read(await page.screenshot({clip, scale: 'css'}));
  const previousFrame = await page.evaluate(() => window.__keplerComputeTest.getState().frameCount);
  await panel.getByLabel('Operation', {exact: true}).selectOption('z-score');
  await page.waitForFunction(({datasetId, previousFrame}) => window.__keplerComputeTest.getState().frameCount > previousFrame &&
    !window.__keplerComputeTest.renderedLayers().some(layer => layer.id === `resident-kriging-points-${datasetId}`), {datasetId, previousFrame});
  const disabledPixels = PNG.sync.read(await page.screenshot({clip, scale: 'css'}));
  let changedPixels = 0;
  for (let offset = 0; offset < enabledPixels.data.length; offset += 4) {
    const difference = [0, 1, 2].reduce((total, component) => total + Math.abs(enabledPixels.data[offset + component] - disabledPixels.data[offset + component]), 0);
    if (difference > 40) changedPixels++;
  }
  assert(changedPixels > 2000, `Expected the enabled prediction bitmap to change its interior, got ${changedPixels} pixels`);
  const unchangedCenter = await page.evaluate(() => window.__keplerComputeTest.projectPosition(-73.75, 40.25));
  assert(Math.hypot(unchangedCenter[0] - center[0], unchangedCenter[1] - center[1]) < 0.5);
  stage('restore the saved kriging recipe through native controls');
  await panel.getByRole('button', {name: 'Restore recipe', exact: true}).click();
  await waitForGrid(datasetId);
  await readGridRows(panel);
  assert.equal(await panel.getByLabel('Kriging effective range in meters', {exact: true}).inputValue(), '100000');
  assert.equal(await panel.getByLabel('Kriging nearest samples', {exact: true}).inputValue(), '3');
  assert.equal(await panel.getByLabel('Kriging variogram model', {exact: true}).inputValue(), 'spherical');
  await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
  assert.deepEqual(JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue()), savedRecipe);
  const final = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0]);
  for (const field of ['sourceRevision', 'sourceUploadBytes', 'sourceBufferIds', 'sourceBatches'])
    assert.deepEqual(final[field], initial[field], `Analysis changed ${field}`);
  stage('select and run both Matérn variogram families');
  const modelVariances = [variance[3]];
  for (const model of ['matern-3/2', 'matern-5/2']) {
    // Retire the previous surface first so readiness and cell reads cannot pass on stale outputs.
    await panel.getByLabel('Operation', {exact: true}).selectOption('z-score');
    await page.waitForFunction(id => !window.__keplerComputeTest.renderedLayers().some(layer =>
      layer.id === `resident-kriging-points-${id}`), datasetId);
    await panel.getByLabel('Kriging variogram model', {exact: true}).selectOption(model);
    assert.equal(await panel.getByLabel('Kriging variogram model', {exact: true}).inputValue(), model);
    await panel.getByRole('button', {name: 'Apply model and run kriging', exact: true}).click();
    await waitForGrid(datasetId);
    // Keep the constant-field invariants, and independently verify that the selected model ran.
    const modelRows = await readGridRows(panel);
    const modelVariance = modelRows[3][4];
    assertModelVariance(model, modelVariance);
    for (const previousVariance of modelVariances) {
      assert(Math.abs(modelVariance - previousVariance) > 2 * MODEL_VARIANCE_TOLERANCE,
        `${model} must change the interpolated variance (${modelVariance} vs ${previousVariance})`);
    }
    modelVariances.push(modelVariance);
    await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
    const modelRecipe = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
    assert.equal(modelRecipe.operation.variogram.model, model);
  }
  stage('remove the source and its prediction overlay');
  await openNativeData(page);
  await workspace.getByRole('button', {name: 'Remove kriging-analysis.arrow', exact: true}).click();
  await page.waitForFunction(id => window.__keplerComputeTest.getState().publications.length === 0 &&
    !window.__keplerComputeTest.renderedLayers().some(layer => layer.id === `resident-kriging-points-${id}`), datasetId);
  await closeNativeDialog(page);
  stage('publish varying-response samples for Vecchia covariance fitting');
  const fitSource = arrow.tableFromArrays({
    longitude: Float32Array.from({length: 16}, (_, index) => -74 + (index % 4) * 0.002),
    latitude: Float32Array.from({length: 16}, (_, index) => 40 + Math.floor(index / 4) * 0.002),
    response: Float32Array.from({length: 16}, (_, index) => Math.sin(index * 1.4))
  });
  await openNativeData(page);
  await workspace.locator('input[type=file]').setInputFiles({
    name: 'kriging-vecchia-fit.arrow', mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(fitSource, 'stream'))
  });
  await workspace.getByLabel('Source longitude column', {exact: true}).selectOption('longitude');
  await workspace.getByLabel('Source latitude column', {exact: true}).selectOption('latitude');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 16, undefined, {timeout: 30000});
  const fitDatasetId = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);
  await closeNativeDialog(page);
  await panel.getByLabel('Dataset', {exact: true}).selectOption(fitDatasetId);
  await panel.getByLabel('Numeric field', {exact: true}).selectOption('response');
  if (!(await panel.locator('.analysis-kriging-settings').evaluate(element => element.open)))
    await panel.getByText('Configure ordinary kriging grid', {exact: true}).click();
  await panel.getByLabel('Kriging longitude field', {exact: true}).selectOption('longitude');
  await panel.getByLabel('Kriging latitude field', {exact: true}).selectOption('latitude');
  for (const [label, value] of [
    ['Kriging west longitude', -74], ['Kriging south latitude', 40],
    ['Kriging east longitude', -73.992], ['Kriging north latitude', 40.008],
    ['Kriging columns', 2], ['Kriging rows', 2], ['Kriging search radius in meters', 100000],
    ['Kriging nearest samples', 8], ['Kriging minimum distinct positions', 3]
  ]) await panel.getByLabel(label, {exact: true}).fill(String(value));
  stage('fit a covariance through each GPU Vecchia method and preserve fit provenance');
  const models = await panel.getByLabel('Kriging variogram model', {exact: true}).locator('option').evaluateAll(options => options.map(option => option.value));
  assert(models.includes('matern-3/2') && models.includes('matern-5/2'));
  for (const method of ['vecchia-likelihood', 'vecchia-fisher']) {
    await panel.getByLabel('Variogram fitting method', {exact: true}).selectOption(method);
    await panel.getByLabel('Kriging variogram model', {exact: true}).selectOption('exponential');
    await panel.getByLabel('Kriging nugget in squared response units', {exact: true}).fill('0.05');
    await panel.getByLabel('Kriging partial sill in squared response units', {exact: true}).fill('1');
    await panel.getByLabel('Kriging effective range in meters', {exact: true}).fill('300');
    await panel.getByRole('button', {name: 'Fit variogram automatically', exact: true}).click();
    const diagnosticsTable = panel.getByRole('table', {name: 'Variogram fit diagnostics', exact: true});
    await diagnosticsTable.waitFor({timeout: 60000});
    assert.match(await diagnosticsTable.innerText(), /log_likelihood/);
    assert.match(await diagnosticsTable.innerText(), /converged/);
    await panel.getByRole('button', {name: 'Apply model and run kriging', exact: true}).click();
    await waitForGrid(fitDatasetId);
    await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
    const fittedRecipe = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
    assert.equal(fittedRecipe.source.datasetId, fitDatasetId);
    assert.equal(fittedRecipe.operation.variogramFit.method, method);
    assert.equal(fittedRecipe.operation.variogramFit.weighting, 'conditional-likelihood');
    assert(Number.isFinite(fittedRecipe.operation.variogramFit.criterion));
  }
  stage('remove the covariance-fitting source and its prediction overlay');
  await openNativeData(page);
  await workspace.getByRole('button', {name: 'Remove kriging-vecchia-fit.arrow', exact: true}).click();
  await page.waitForFunction(id => window.__keplerComputeTest.getState().publications.length === 0 &&
    !window.__keplerComputeTest.renderedLayers().some(layer => layer.id === `resident-kriging-points-${id}`), fitDatasetId);
  await closeNativeDialog(page);
  assert.deepEqual(failures, []);
  console.log(JSON.stringify({passed: true, changedPixels, checks: [
    'native nullable Float32 multibatch publication and manual ordinary kriging controls',
    'three exact hits with prediction4/variance0 plus finite positive fourth-cell variance',
    'grid inspector distinguishes prediction/variance without fabricated IDW sample counts',
    'native separate prediction/variance downloads decoded with full recipe and band provenance',
    'actual enabled/disabled geographic bitmap pixels at a fixed camera',
    'saved recipe restore, unchanged source uploads/chunks and dataset/overlay removal',
    'both Matérn families selectable with independent model-specific variances, distinct outputs and saved model identity',
    'separate varying-response fixture, both GPU Vecchia fit methods, provenance and cleanup'
  ]}));
} catch (failure) {
  console.error(JSON.stringify({stage: currentStage, failures, diagnostics,
    body: (await page.locator('body').innerText().catch(() => '')).slice(-6000)}));
  throw failure;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
