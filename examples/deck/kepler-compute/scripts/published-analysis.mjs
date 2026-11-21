// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
// The old fork obtained UTIF accidentally through three's hoisted dependencies.
// Declare the standalone decoder so this check also runs in the published-package workspace.
import tiffDecoder from 'utif';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';
import {setAnalysisSheetControl} from './analysis-sheet-controls.mjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const source = arrow.tableFromArrays({
  longitude: new Float32Array([-74, -74.01]),
  latitude: new Float32Array([40.7, 40.72]),
  measure: new Float32Array([10, 20]),
  population: new Float32Array([100, 100]),
  trend: new Float32Array([1, 2])
}).concat(arrow.tableFromArrays({
  longitude: new Float32Array([-74.02, -74.03]),
  latitude: new Float32Array([40.74, 40.76]),
  measure: new Float32Array([30, 40]),
  population: new Float32Array([100, 100]),
  trend: new Float32Array([3, 4])
}));
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
// Keep this check on one loaded implementation while other contributors keep editing.
await page.routeWebSocket('**/*', socket => {
  const server = socket.connectToServer();
  server.onMessage(message => {
    if (typeof message === 'string') {
      try {
        const packet = JSON.parse(message);
        if (packet.type === 'update' || packet.type === 'full-reload') return;
      } catch {}
    }
    socket.send(message);
  });
  socket.onMessage(message => server.send(message));
});
const errors = [];
const diagnostics = [];
page.on('pageerror', error => { errors.push(error.message); console.error('pageerror:', error.stack ?? error.message); });
page.on('requestfailed', request => { diagnostics.push({url: request.url(), error: request.failure()?.errorText}); console.error('requestfailed:', request.url(), request.failure()?.errorText); });
page.on('console', message => { if (message.type() === 'error') { diagnostics.push({console: message.text()}); console.error('console:', message.text()); } });
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: 'analysis-batches.arrow', mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))
  });
  await workspace.getByLabel('Source longitude column').selectOption('longitude');
  await workspace.getByLabel('Source latitude column').selectOption('latitude');
  await workspace.getByRole('button', {name: 'Check schema and precision'}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields'}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 4, undefined, {timeout: 45000});
  const datasetId = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);
  await closeNativeDialog(page);
  const panel = page.getByRole('region', {name: 'GPU analysis', exact: true});
  await panel.getByLabel('Dataset', {exact: true}).selectOption(datasetId);
  await panel.getByLabel('Numeric field', {exact: true}).selectOption('measure');
  await panel.getByLabel('Operation', {exact: true}).selectOption('z-score');
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('4 analyzed'));
  await page.locator('.show-data-table svg').last().click();
  await page.waitForFunction(() => document.querySelector('.resident-table-scroll')?.textContent.includes('Analysis: z-score'));
  await page.waitForFunction(() => document.querySelectorAll('.resident-table-scroll tbody tr').length === 4 && document.querySelector('.resident-table-scroll')?.textContent.includes('-1.16'));
  const tableText = await page.locator('.resident-table-scroll').innerText();
  assert.match(tableText, /-1\.16/);
  assert.match(tableText, /1\.16/);
  await page.keyboard.press('Escape');
  await page.locator('.resident-table-modal').waitFor({state: 'hidden'});
  await panel.getByLabel('Analysis population', {exact: true}).selectOption('selected');
  await page.evaluate(id => window.__keplerComputeTest.setPublishedRange(id, 'measure', 20, 40), datasetId);
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('3 analyzed'));
  await panel.getByLabel('Operation', {exact: true}).selectOption('quantile');
  await page.locator('#analysis-class-count').focus();
  await page.locator('#analysis-class-count').press('Home');
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('Edges: 20 · 30 · 40'));
  await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
  const recipe = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
  assert.equal(recipe.source.datasetId, datasetId);
  assert.equal(recipe.operation.column, 'measure');
  assert.equal(recipe.population, 'selected');
  await panel.getByLabel('Operation', {exact: true}).selectOption('excess-risk');
  await panel.getByLabel('Population field', {exact: true}).selectOption('population');
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('0.667 to 1.333'));
  await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
  const rateRecipe = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
  assert.deepEqual(rateRecipe.operation, {kind: 'excess-risk', events: 'measure', populations: 'population'});
  await panel.getByLabel('Analysis population', {exact: true}).selectOption('all');
  await panel.getByText('Configure hot spots on a row-major grid', {exact: true}).click();
  await panel.locator('.analysis-hotspot-settings').getByTestId('ax-analysis-sheet').waitFor();
  await setAnalysisSheetControl(panel.getByLabel('Grid width', {exact: true}), '2');
  await setAnalysisSheetControl(panel.getByLabel('Grid height', {exact: true}), '2');
  await setAnalysisSheetControl(panel.getByLabel('Neighborhood', {exact: true}), 'rook');
  await setAnalysisSheetControl(panel.getByLabel('Minimum neighbors', {exact: true}), '1');
  await setAnalysisSheetControl(panel.getByLabel('Multiple-testing correction', {exact: true}), 'none');
  await panel.getByRole('button', {name: 'Apply grid and run hot spots', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('8 neighbor links'));
  assert.equal(await panel.getByLabel('Operation', {exact: true}).inputValue(), 'hot-spots-lattice');
  await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
  const hotspotRecipe = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
  assert.equal(hotspotRecipe.operation.kind, 'hot-spots-lattice');
  assert.equal(hotspotRecipe.operation.width, 2);
  assert.equal(hotspotRecipe.operation.height, 2);
  assert.equal(hotspotRecipe.operation.criterion, 'rook');
  await panel.getByText('Configure geographic neighbors and spatial rates', {exact: true}).click();
  await setAnalysisSheetControl(panel.getByLabel('Spatial rate population field', {exact: true}), 'population');
  await setAnalysisSheetControl(panel.getByLabel('Neighbor longitude field', {exact: true}), 'longitude');
  await setAnalysisSheetControl(panel.getByLabel('Neighbor latitude field', {exact: true}), 'latitude');
  await setAnalysisSheetControl(panel.getByLabel('Nearest neighbors', {exact: true}), '3');
  await setAnalysisSheetControl(panel.getByLabel('Rate scale', {exact: true}), '100');
  await panel.getByRole('button', {name: 'Apply neighbors and run spatial rates', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('Spatial rate range: 25 to 25'));
  assert.match(await panel.innerText(), /12 point neighbor links/);
  await panel.getByLabel('Analysis population', {exact: true}).selectOption('selected');
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('Spatial rate range: 30 to 30'));
  assert.match(await panel.innerText(), /12 point neighbor links/);
  await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
  const spatialRecipe = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
  assert.deepEqual(spatialRecipe.operation, {
    kind: 'spatial-rate-points', events: 'measure', populations: 'population',
    weights: {kind: 'knn', longitudeColumn: 'longitude', latitudeColumn: 'latitude', k: 3,
      coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'meters'},
    scale: 100, islandPolicy: 'undefined'
  });
  await panel.getByLabel('Operation', {exact: true}).selectOption('off');
  await panel.getByLabel('Operation', {exact: true}).selectOption('z-score');
  await panel.getByRole('button', {name: 'Restore recipe', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('Spatial rate range: 30 to 30'));
  await panel.getByLabel('Analysis population', {exact: true}).selectOption('all');
  if ((await panel.locator('details.analysis-point-weights-settings').getAttribute('open')) === null)
    await panel.getByText('Configure geographic neighbors and spatial rates', {exact: true}).click();
  await setAnalysisSheetControl(panel.getByLabel('Neighbor topology', {exact: true}), 'distance-band');
  await setAnalysisSheetControl(panel.getByLabel('Neighbor radius in meters', {exact: true}), '10000');
  await setAnalysisSheetControl(panel.getByLabel('Neighbor directed neighbor capacity', {exact: true}), '1');
  await panel.getByRole('button', {name: 'Apply neighbors and run spatial rates', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('12 slots are required'));
  await setAnalysisSheetControl(panel.getByLabel('Neighbor directed neighbor capacity', {exact: true}), '12');
  await panel.getByRole('button', {name: 'Apply neighbors and run spatial rates', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('Spatial rate range: 25 to 25'));
  assert.match(await panel.innerText(), /12 point neighbor links/);
  await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
  const distanceBandRecipe = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
  assert.equal(distanceBandRecipe.operation.weights.kind, 'distance-band');
  assert.equal(distanceBandRecipe.operation.weights.radiusMeters, 10000);
  assert.equal(distanceBandRecipe.operation.weights.neighborCapacity, 12);
  assert(!Object.hasOwn(distanceBandRecipe.operation.weights, 'k'));
  await panel.getByLabel('Operation', {exact: true}).selectOption('z-score');
  await panel.getByRole('button', {name: 'Restore recipe', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('Spatial rate range: 25 to 25'));
  await panel.getByText('Configure hot spots on geographic points', {exact: true}).click();
  await setAnalysisSheetControl(panel.getByLabel('Hot-spot longitude field', {exact: true}), 'longitude');
  await setAnalysisSheetControl(panel.getByLabel('Hot-spot latitude field', {exact: true}), 'latitude');
  await setAnalysisSheetControl(panel.getByLabel('Hot-spot topology', {exact: true}), 'knn');
  await setAnalysisSheetControl(panel.getByLabel('Hot-spot nearest neighbors', {exact: true}), '1');
  await setAnalysisSheetControl(panel.getByLabel('Hot-spot minimum neighbors', {exact: true}), '1');
  await setAnalysisSheetControl(panel.getByLabel('Point hot-spot correction', {exact: true}), 'none');
  await panel.getByRole('button', {name: 'Apply neighbors and run point hot spots', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('4 point neighbor links') && document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('4 analyzed'));
  assert.equal(await panel.getByLabel('Operation', {exact: true}).inputValue(), 'hot-spots-points');
  await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
  const pointHotspotRecipe = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
  assert.deepEqual(pointHotspotRecipe.operation, {
    kind: 'hot-spots-points', column: 'measure',
    weights: {kind: 'knn', longitudeColumn: 'longitude', latitudeColumn: 'latitude', k: 1,
      coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'meters'},
    minimumNeighbors: 1, selfWeight: 1, multipleTesting: 'none', islands: 'exclude', inference: 'analytic-two-sided'
  });
  await panel.getByLabel('Operation', {exact: true}).selectOption('z-score');
  await panel.getByRole('button', {name: 'Restore recipe', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('4 point neighbor links'));
  await page.locator('.show-data-table svg').last().click();
  await page.waitForFunction(() => document.querySelector('.resident-table-scroll')?.textContent.includes('Analysis: hot spot z-score'));
  assert.match(await page.locator('.resident-table-scroll').innerText(), /p.value|p-value/i);
  await page.keyboard.press('Escape');
  await page.locator('.resident-table-modal').waitFor({state: 'hidden'});
  await panel.getByText('Configure Local Moran on geographic points', {exact: true}).click();
  await setAnalysisSheetControl(panel.getByLabel('Local Moran longitude field', {exact: true}), 'longitude');
  await setAnalysisSheetControl(panel.getByLabel('Local Moran latitude field', {exact: true}), 'latitude');
  await setAnalysisSheetControl(panel.getByLabel('Local Moran nearest neighbors', {exact: true}), '1');
  await setAnalysisSheetControl(panel.getByLabel('Local Moran minimum neighbors', {exact: true}), '1');
  await setAnalysisSheetControl(panel.getByLabel('Local Moran correction', {exact: true}), 'none');
  await setAnalysisSheetControl(panel.getByLabel('Local Moran significance level', {exact: true}), '0.9999');
  await panel.getByRole('button', {name: 'Apply neighbors and run Local Moran', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('Local Moran I range:'));
  assert.match(await panel.innerText(), /4 analyzed/);
  assert.match(await panel.innerText(), /Map: gray = unreported/);
  await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
  const localMoranRecipe = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
  assert.equal(localMoranRecipe.operation.kind, 'local-moran-points');
  assert.equal(localMoranRecipe.operation.significanceLevel, 0.9999);
  assert.equal(localMoranRecipe.operation.inference, 'analytic-conditional-normal-two-sided');
  await panel.getByLabel('Operation', {exact: true}).selectOption('off');
  // Restore controls are hidden when off, so select an existing operation first.
  await panel.getByLabel('Operation', {exact: true}).selectOption('z-score');
  await panel.getByRole('button', {name: 'Restore recipe', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('Local Moran I range:'));
  await page.locator('.show-data-table svg').last().click();
  await page.waitForFunction(() => document.querySelector('.resident-table-scroll')?.textContent.includes('Analysis: local moran i'));
  assert.match(await page.locator('.resident-table-scroll').innerText(), /adjusted p value/);
  assert.match(await page.locator('.resident-table-scroll').innerText(), /quadrant/);
  await page.keyboard.press('Escape');
  await page.locator('.resident-table-modal').waitFor({state: 'hidden'});
  if ((await panel.locator('details.analysis-local-moran-settings').getAttribute('open')) === null)
    await panel.getByText('Configure Local Moran on geographic points', {exact: true}).click();
  await setAnalysisSheetControl(panel.getByLabel('Local Moran inference', {exact: true}), 'permutation-conditional-two-sided');
  await setAnalysisSheetControl(panel.getByLabel('Local Moran permutations', {exact: true}), '99');
  await setAnalysisSheetControl(panel.getByLabel('Local Moran permutation seed', {exact: true}), '314159');
  await panel.getByRole('button', {name: 'Apply neighbors and run Local Moran', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('Local Moran I range:') && document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('without permutation inference'));
  await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
  const permutationMoranRecipe = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
  assert.equal(permutationMoranRecipe.operation.inference, 'permutation-conditional-two-sided');
  assert.equal(permutationMoranRecipe.operation.permutationCount, 99);
  assert.equal(permutationMoranRecipe.operation.seed, 314159);
  await panel.getByLabel('Operation', {exact: true}).selectOption('z-score');
  await panel.getByRole('button', {name: 'Restore recipe', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('Local Moran I range:') && document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('without permutation inference'));
  await panel.getByText('Configure Quantile LISA on geographic points', {exact: true}).click();
  await setAnalysisSheetControl(panel.getByLabel('Quantile LISA longitude field', {exact: true}), 'longitude');
  await setAnalysisSheetControl(panel.getByLabel('Quantile LISA latitude field', {exact: true}), 'latitude');
  await setAnalysisSheetControl(panel.getByLabel('Quantile LISA nearest neighbors', {exact: true}), '1');
  await setAnalysisSheetControl(panel.getByLabel('Quantile LISA quantile count', {exact: true}), '2');
  await setAnalysisSheetControl(panel.getByLabel('Quantile LISA selected quantile', {exact: true}), '2');
  await setAnalysisSheetControl(panel.getByLabel('Quantile LISA permutations', {exact: true}), '19');
  await setAnalysisSheetControl(panel.getByLabel('Quantile LISA seed', {exact: true}), '314159');
  await setAnalysisSheetControl(panel.getByLabel('Quantile LISA significance level', {exact: true}), '0.9');
  await panel.getByRole('button', {name: 'Apply neighbors and run Quantile LISA', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('Quantile interval: [25, ∞)'));
  assert.match(await panel.innerText(), /4 finite coordinate-valid observations/);
  assert.match(await panel.innerText(), /anti-conservative/);
  await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
  const quantileLisaRecipe = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
  assert.equal(quantileLisaRecipe.operation.kind, 'quantile-lisa-points');
  assert.equal(quantileLisaRecipe.operation.permutationCount, 19);
  assert.equal(quantileLisaRecipe.operation.seed, 314159);
  assert.equal(quantileLisaRecipe.operation.quantileMethod, 'type-5');
  assert.equal(quantileLisaRecipe.operation.alternative, 'geoda-legacy');
  assert.equal(quantileLisaRecipe.operation.multipleTesting, 'none');
  await panel.getByLabel('Operation', {exact: true}).selectOption('z-score');
  await panel.getByRole('button', {name: 'Restore recipe', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('Quantile interval: [25, ∞)'));
  await page.locator('.show-data-table svg').last().click();
  await page.waitForFunction(() => document.querySelector('.resident-table-scroll')?.textContent.includes('Analysis: quantile lisa pseudo p value'));
  assert.match(await page.locator('.resident-table-scroll').innerText(), /membership/);
  assert.match(await page.locator('.resident-table-scroll').innerText(), /tail count/);
  await page.keyboard.press('Escape');
  await page.locator('.resident-table-modal').waitFor({state: 'hidden'});
  await panel.getByLabel('Operation', {exact: true}).selectOption('off');
  const mapBeforeGrid = PNG.sync.read(await page.screenshot());
  await panel.getByText('Configure inverse-distance interpolation grid', {exact: true}).click();
  await setAnalysisSheetControl(panel.getByLabel('Interpolation longitude field', {exact: true}), 'longitude');
  await setAnalysisSheetControl(panel.getByLabel('Interpolation latitude field', {exact: true}), 'latitude');
  await setAnalysisSheetControl(panel.getByLabel('Interpolation west longitude', {exact: true}), '-74.05');
  await setAnalysisSheetControl(panel.getByLabel('Interpolation south latitude', {exact: true}), '40.68');
  await setAnalysisSheetControl(panel.getByLabel('Interpolation east longitude', {exact: true}), '-73.98');
  await setAnalysisSheetControl(panel.getByLabel('Interpolation north latitude', {exact: true}), '40.78');
  await setAnalysisSheetControl(panel.getByLabel('Interpolation columns', {exact: true}), '8');
  await setAnalysisSheetControl(panel.getByLabel('Interpolation rows', {exact: true}), '8');
  await setAnalysisSheetControl(panel.getByLabel('Interpolation search radius in meters', {exact: true}), '20000');
  await setAnalysisSheetControl(panel.getByLabel('Interpolation nearest samples', {exact: true}), '4');
  await setAnalysisSheetControl(panel.getByLabel('Interpolation minimum samples', {exact: true}), '2');
  await panel.getByRole('button', {name: 'Apply grid and run interpolation', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('64 analyzed'));
  await page.waitForFunction(id => window.__keplerComputeTest.renderedLayers().some(layer => layer.id === `resident-idw-points-${id}` && layer.layerClass === 'BitmapLayer' && layer.visible && layer.loaded), datasetId);
  const mapWithGrid = PNG.sync.read(await page.screenshot());
  let changedGridPixels = 0;
  for (let vertical = 150; vertical < 800; vertical++) for (let horizontal = 450; horizontal < 1000; horizontal++) {
    const offset = (vertical * mapWithGrid.width + horizontal) * 4;
    if (Math.abs(mapBeforeGrid.data[offset] - mapWithGrid.data[offset]) + Math.abs(mapBeforeGrid.data[offset + 1] - mapWithGrid.data[offset + 1]) + Math.abs(mapBeforeGrid.data[offset + 2] - mapWithGrid.data[offset + 2]) > 40) changedGridPixels++;
  }
  assert(changedGridPixels > 1000, `Interpolation must render a geographic surface; changed ${changedGridPixels} map pixels`);
  await panel.getByText('Inspect interpolation grid cells', {exact: true}).click();
  await page.waitForFunction(() => document.querySelectorAll('.interpolation-grid-results tbody tr').length === 16);
  assert.match(await panel.locator('.interpolation-grid-results').innerText(), /4 contributing source candidates/);
  const gridDownloadPromise = page.waitForEvent('download');
  await panel.getByRole('button', {name: 'Download grid GeoTIFF', exact: true}).click();
  const gridDownload = await gridDownloadPromise;
  assert.equal(await gridDownload.failure(), null);
  assert.match(gridDownload.suggestedFilename(), /-idw-prediction\.tif$/);
  const gridDownloadPath = await gridDownload.path();
  assert(gridDownloadPath);
  const gridFile = new Uint8Array(await readFile(gridDownloadPath));
  const gridImage = tiffDecoder.decode(gridFile.buffer)[0];
  tiffDecoder.decodeImage(gridFile.buffer, gridImage);
  assert.deepEqual([gridImage.width, gridImage.height], [8, 8]);
  assert.equal(gridImage.t339[0], 3);
  assert.equal(gridImage.t34735.at(-1), 4326);
  const gridDescription = JSON.parse(gridImage.t270[0]);
  assert.equal(gridDescription.recipe.source.datasetId, datasetId);
  assert.equal(gridDescription.recipe.operation.kind, 'idw-points');
  const exportedGridValues = new Float32Array(gridImage.data.buffer, gridImage.data.byteOffset, 64);
  assert(Array.from(exportedGridValues).every(value => value >= 10 && value <= 40));
  await panel.getByRole('button', {name: 'Next grid cells', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('.interpolation-grid-results tbody td')?.textContent === '16');
  await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
  const interpolationRecipe = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
  assert.equal(interpolationRecipe.operation.kind, 'idw-points');
  assert.equal(interpolationRecipe.operation.grid.width, 8);
  assert.equal(interpolationRecipe.operation.metric, 'great-circle');
  await panel.getByLabel('Operation', {exact: true}).selectOption('off');
  await page.waitForFunction(id => !window.__keplerComputeTest.renderedLayers().some(layer => layer.id === `resident-idw-points-${id}`), datasetId);
  await panel.getByLabel('Operation', {exact: true}).selectOption('z-score');
  await panel.getByRole('button', {name: 'Restore recipe', exact: true}).click();
  await page.waitForFunction(id => window.__keplerComputeTest.renderedLayers().some(layer => layer.id === `resident-idw-points-${id}` && layer.loaded), datasetId);
  await panel.getByText('Configure DBSCAN on geographic points', {exact: true}).click();
  await setAnalysisSheetControl(panel.getByLabel('DBSCAN longitude field', {exact: true}), 'longitude');
  await setAnalysisSheetControl(panel.getByLabel('DBSCAN latitude field', {exact: true}), 'latitude');
  await setAnalysisSheetControl(panel.getByLabel('DBSCAN neighborhood radius in meters', {exact: true}), '1000');
  await setAnalysisSheetControl(panel.getByLabel('DBSCAN minimum samples including self', {exact: true}), '2');
  await panel.getByRole('button', {name: 'Apply radius and run DBSCAN', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('0 clusters · 4 noise'));
  await panel.getByText('Configure DBSCAN on geographic points', {exact: true}).click();
  await setAnalysisSheetControl(panel.getByLabel('DBSCAN neighborhood radius in meters', {exact: true}), '3000');
  await panel.getByRole('button', {name: 'Apply radius and run DBSCAN', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('1 clusters · 0 noise'));
  await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
  const clusteringRecipe = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
  assert.equal(clusteringRecipe.operation.kind, 'dbscan-points');
  assert.equal(clusteringRecipe.operation.epsilon, 3000);
  assert.equal(clusteringRecipe.operation.metric, 'great-circle');
  assert(!Object.hasOwn(clusteringRecipe.operation, 'column'));
  await panel.getByLabel('Operation', {exact: true}).selectOption('z-score');
  await panel.getByRole('button', {name: 'Restore recipe', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('1 clusters · 0 noise'));
  await page.locator('.show-data-table svg').last().click();
  await page.waitForFunction(() => document.querySelector('.resident-table-scroll')?.textContent.includes('Analysis: dbscan cluster'));
  assert.match(await page.locator('.resident-table-scroll').innerText(), /core distance meters/);
  await page.keyboard.press('Escape');
  await page.locator('.resident-table-modal').waitFor({state: 'hidden'});
  await panel.getByText('Configure ordinary least squares regression', {exact: true}).click();
  await panel.locator('.analysis-ols-settings').getByTestId('ax-analysis-sheet').waitFor();
  await setAnalysisSheetControl(panel.getByLabel('OLS response field', {exact: true}), 'measure');
  await setAnalysisSheetControl(panel.getByLabel('OLS predictor 1', {exact: true}), 'population');
  await panel.getByRole('button', {name: 'Apply fields and fit regression', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('OLS design is singular'));
  assert.equal(await panel.getByRole('region', {name: 'Regression report', exact: true}).count(), 0);
  await panel.getByText('Configure ordinary least squares regression', {exact: true}).click();
  await setAnalysisSheetControl(panel.getByLabel('OLS predictor 1', {exact: true}), 'trend');
  await panel.getByRole('button', {name: 'Apply fields and fit regression', exact: true}).click();
  const regressionReport = panel.getByRole('region', {name: 'Regression report', exact: true});
  await page.waitForFunction(() => document.querySelector('[aria-label="Regression report"]')?.textContent.includes('4 complete observations'));
  const slope = Number(await regressionReport.locator('table').first().locator('tbody tr').nth(1).locator('td').first().innerText());
  assert(Math.abs(slope - 10) < 0.0001);
  await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
  const regressionRecipe = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
  assert.deepEqual(regressionRecipe.operation.predictorColumns, ['trend']);
  assert.equal(regressionRecipe.operation.kind, 'ols');
  assert.equal(regressionRecipe.operation.missing, 'complete-cases');
  assert.equal(regressionRecipe.operation.ridgeLambda, 0);
  await panel.getByLabel('Analysis population', {exact: true}).selectOption('selected');
  await page.waitForFunction(() => document.querySelector('[aria-label="Regression report"]')?.textContent.includes('3 complete observations'));
  await panel.getByLabel('Analysis population', {exact: true}).selectOption('all');
  await page.waitForFunction(() => document.querySelector('[aria-label="Regression report"]')?.textContent.includes('4 complete observations'));
  await panel.getByText('Configure ordinary least squares regression', {exact: true}).click();
  await setAnalysisSheetControl(panel.getByLabel('OLS ridge penalty', {exact: true}), '6');
  await panel.getByRole('button', {name: 'Apply fields and fit regression', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="Regression report"]')?.textContent.includes('Ridge descriptive fit'));
  assert.match(await regressionReport.innerText(), /Standard errors, t tests, p-values/);
  const ridgeSlope = Number(await regressionReport.locator('table').first().locator('tbody tr').nth(1).locator('td').first().innerText());
  assert(Math.abs(ridgeSlope - 50 / 11) < 0.0001);
  await panel.getByRole('button', {name: 'Restore recipe', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="Regression report"]')?.textContent.includes('Ordinary least squares fit'));
  await page.locator('.show-data-table svg').last().click();
  await page.waitForFunction(() => document.querySelector('.resident-table-scroll')?.textContent.includes('Analysis: ols residual'));
  assert.match(await page.locator('.resident-table-scroll').innerText(), /ols fitted/);
  await page.keyboard.press('Escape');
  await page.locator('.resident-table-modal').waitFor({state: 'hidden'});
  assert.deepEqual(await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].sourceBatches), [2, 2]);
  await openNativeData(page);
  await workspace.getByRole('button', {name: 'Remove analysis-batches.arrow', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications.length === 0);
  await closeNativeDialog(page);
  assert.equal(await panel.getByLabel('Operation', {exact: true}).inputValue(), 'off');
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, checks: ['published dataset and field controls', 'global multibatch z-score', 'derived native table', 'selected quantile', 'source recipe identity', 'selected pooled excess risk', 'explicit lattice hotspot controls', 'geographic kNN rates and fixed topology under selection', 'point weights recipe restore and distance-band capacity refusal/recovery/replay', 'geographic point hot spots, native table and recipe restore', 'Local Moran analytic and conditional permutation controls, categorical colors, table statistics and recipe restore', 'Quantile LISA controls, legacy inference contract, table outputs and deterministic recipe restore', 'IDW geographic texture pixels, bounded grid cells, GeoTIFF download and saved recipe', 'geographic DBSCAN noise, clusters, source table and replay', 'OLS singular gating, ordinary/ridge fit, selected population, source table and recipe replay', 'dataset removal fallback']}));
} catch (failure) {
  console.error(JSON.stringify({pageErrors: errors, diagnostics, body: (await page.locator('body').innerText()).slice(0, 2000)}));
  throw failure;
} finally {
  await browser.close();
}
