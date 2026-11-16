// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Native workflow: automatic variogram fit, prediction intervals and leave-one-out diagnostics.
// Usage: node scripts/kriging-fit-confidence.mjs [url]

import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {setAnalysisSheetControl} from './analysis-sheet-controls.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const side = 20;
const bounds = [-74, 40, -73.6, 40.4];
const longitudes = new Float32Array(side * side);
const latitudes = new Float32Array(side * side);
const response = new Float32Array(side * side);
for (let row = 0; row < side * side; row++) {
  const x = (row % side) / side;
  const y = Math.floor(row / side) / side;
  longitudes[row] = bounds[0] + 0.4 * x;
  latitudes[row] = bounds[1] + 0.4 * y;
  response[row] = 10 * Math.sin(3 * x) * Math.cos(2 * y) + Math.sin(8 * x + 3 * y);
}
const source = arrow.tableFromArrays({longitude: longitudes, latitude: latitudes, response,
  other: response.map(value => value * 2 + 1)});
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const failures = [];
let currentStage = 'startup';
const watchdog = setTimeout(() => {
  console.error(`Kriging fit workflow exceeded 120 seconds at ${currentStage}`);
  void browser.close();
}, 120000);
page.on('pageerror', error => failures.push(error.message));
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text()))
    failures.push(message.text());
});

try {
  await pinViteModules(page);
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  currentStage = 'publish samples';
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: 'kriging-fit.arrow', mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))
  });
  await workspace.getByLabel('Source longitude column', {exact: true}).selectOption('longitude');
  await workspace.getByLabel('Source latitude column', {exact: true}).selectOption('latitude');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 400, undefined, {timeout: 30000});
  const datasetId = (await page.evaluate(() => window.__keplerComputeTest.getState().publications[0])).id;
  await closeNativeDialog(page);
  const panel = page.getByRole('region', {name: 'GPU analysis', exact: true});
  await panel.getByLabel('Dataset', {exact: true}).selectOption(datasetId);
  await panel.getByLabel('Numeric field', {exact: true}).selectOption('response');
  await panel.getByText('Configure ordinary kriging grid', {exact: true}).click();
  await setAnalysisSheetControl(panel.getByLabel('Kriging longitude field', {exact: true}), 'longitude');
  await setAnalysisSheetControl(panel.getByLabel('Kriging latitude field', {exact: true}), 'latitude');
  for (const [label, value] of [
    ['Kriging west longitude', bounds[0]], ['Kriging south latitude', bounds[1]],
    ['Kriging east longitude', bounds[2]], ['Kriging north latitude', bounds[3]],
    ['Kriging columns', 8], ['Kriging rows', 8], ['Kriging search radius in meters', 20000],
    ['Kriging nearest samples', 8], ['Kriging minimum distinct positions', 3]
  ]) await setAnalysisSheetControl(panel.getByLabel(label, {exact: true}), value);
  currentStage = 'fit variogram automatically';
  await panel.getByRole('button', {name: 'Fit variogram automatically', exact: true}).click();
  const status = panel.getByRole('status').filter({hasText: 'Fitted'});
  await status.waitFor({timeout: 30000});
  // Automatic fitting now considers both analysis Matérn families as well as the original three.
  assert.match(await status.innerText(), /Fitted (spherical|exponential|gaussian|matern-3\/2|matern-5\/2): nugget .*range \d+ m from [\d,]+ pairs/);
  const fittedRange = Number(await panel.getByLabel('Kriging effective range in meters', {exact: true}).inputValue());
  assert(fittedRange > 0, `Expected a positive fitted range, got ${fittedRange}`);
  await setAnalysisSheetControl(panel.getByLabel('Prediction interval level (blank: none, for example 0.95)', {exact: true}), '0.95');
  await setAnalysisSheetControl(panel.getByLabel('Leave-one-out cross-validation', {exact: true}), true);
  currentStage = 'run kriging with intervals and cross-validation';
  await panel.getByRole('button', {name: 'Apply model and run kriging', exact: true}).click();
  await page.waitForFunction(id => window.__keplerComputeTest.renderedLayers().some(layer =>
    layer.id === `resident-kriging-points-${id}` && layer.visible && layer.loaded), datasetId, {timeout: 60000});
  await panel.getByText('Inspect interpolation grid cells', {exact: true}).click();
  const diagnostics = panel.getByRole('region', {name: 'Kriging grid diagnostics', exact: true});
  await page.waitForFunction(() => /Leave-one-out cross-validation over/.test(
    document.querySelector('[aria-label="Kriging grid diagnostics"]')?.textContent ?? ''), undefined, {timeout: 30000});
  const text = await diagnostics.innerText();
  assert.match(text, /95\.0% prediction interval half-width/);
  assert.match(text, /Leave-one-out cross-validation over 400 retained samples/);
  const headers = await panel.getByRole('table', {name: 'Interpolation grid cells', exact: true}).locator('thead th').allTextContents();
  assert(headers.includes('Interval lower') && headers.includes('Interval upper'), headers.join('|'));
  await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
  const saved = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
  assert.equal(saved.operation.confidenceLevel, 0.95);
  assert.equal(saved.operation.crossValidation, true);
  assert.equal(saved.operation.variogramFit.method, 'weighted-least-squares');
  assert.equal(saved.operation.variogram.rangeMeters, fittedRange);
  // Reapplying a restored fit keeps provenance; edits alone keep the accepted recipe unchanged.
  await panel.getByRole('button', {name: 'Restore recipe', exact: true}).click();
  await panel.getByRole('button', {name: 'Apply model and run kriging', exact: true}).click();
  const reapplied = await page.evaluate(identifier => JSON.parse(window.__keplerComputeTest.analysisForDataset(identifier).serializeRecipe()), datasetId);
  assert.deepEqual(reapplied.operation, saved.operation);
  currentStage = 'invalidate fit provenance after changing the response';
  await panel.getByLabel('Numeric field', {exact: true}).selectOption('other');
  await panel.getByRole('button', {name: 'Apply model and run kriging', exact: true}).click();
  const changedResponse = await page.evaluate(identifier => JSON.parse(window.__keplerComputeTest.analysisForDataset(identifier).serializeRecipe()), datasetId);
  assert.equal(changedResponse.operation.column, 'other');
  assert.equal(changedResponse.operation.variogramFit, undefined);
  await panel.getByRole('button', {name: 'Restore recipe', exact: true}).click();
  await panel.getByRole('button', {name: 'Apply model and run kriging', exact: true}).click();
  const restoredResponse = await page.evaluate(identifier => JSON.parse(window.__keplerComputeTest.analysisForDataset(identifier).serializeRecipe()), datasetId);
  assert.deepEqual(restoredResponse.operation, saved.operation);
  assert.deepEqual(failures, []);
  console.log(JSON.stringify({passed: true, fittedRange, checks: [
    'automatic weighted-least-squares variogram fit fills explicit replayable parameters',
    'prediction interval columns and half-width diagnostics',
    'leave-one-out diagnostics over all retained samples',
    'saved recipe carries confidence level, cross-validation flag and fit provenance',
    'response changes invalidate fit provenance while recipe replay restores it'
  ]}));
} catch (failure) {
  console.error(JSON.stringify({stage: currentStage, failures,
    body: (await page.locator('body').innerText().catch(() => '')).slice(-4000)}));
  throw failure;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
