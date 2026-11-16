// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';
import {setAnalysisSheetControl} from './analysis-sheet-controls.mjs';

// X > 180 is valid in the declared projected CRS; the last row has a missing X coordinate.
const source = arrow.tableFromArrays({
  x: arrow.vectorFromArray([180, 181], new arrow.Float32()),
  y: new Float32Array([0, 0]),
  measure: new Float32Array([10, 20]),
  population: new Float32Array([100, 100])
}).concat(arrow.tableFromArrays({
  x: arrow.vectorFromArray([182, 200, null], new arrow.Float32()),
  y: new Float32Array([0, 0, 0]),
  measure: new Float32Array([60, 90, 1000]),
  population: new Float32Array([100, 100, 100])
}));

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const failures = [];
const diagnostics = [];
page.on('pageerror', error => {
  failures.push(error.message);
  console.error('pageerror:', error.stack ?? error.message);
});
page.on('console', message => {
  if (message.type() !== 'error') return;
  diagnostics.push(message.text());
  if (/WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())) failures.push(message.text());
  console.error('console:', message.text());
});
page.on('requestfailed', request => diagnostics.push(`${request.url()}: ${request.failure()?.errorText}`));
let currentStage = 'startup';
function stage(name) {
  currentStage = name;
  console.log(JSON.stringify({stage: name}));
}
const watchdog = setTimeout(() => {
  console.error(`Projected analysis workflow exceeded 120 seconds at ${currentStage}`);
  void browser.close();
}, 120000);

async function waitForProjectedRates() {
  await page.waitForFunction(() => {
    const text = document.querySelector('[aria-label="GPU analysis"]')?.textContent ?? '';
    const diagnostics = document.querySelector('[aria-label="Rate diagnostics"]');
    const definedOutputs = [...(diagnostics?.querySelectorAll('dt') ?? [])]
      .find(term => term.textContent === 'Defined outputs')?.nextElementSibling?.textContent;
    return text.includes('Spatial rate range: 15 to 40') && definedOutputs === '3' &&
      text.includes('4 point neighbor links') && text.includes('1 invalid coordinate rows');
  }, undefined, {timeout: 30000});
}

try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  stage('publish multibatch projected fields without geographic map roles');
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: 'projected-analysis.arrow', mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))
  });
  await workspace.getByLabel('Source geometry column', {exact: true}).selectOption('');
  await workspace.getByLabel('Source longitude column', {exact: true}).selectOption('');
  await workspace.getByLabel('Source latitude column', {exact: true}).selectOption('');
  assert.equal(await workspace.getByLabel('Source longitude column', {exact: true}).inputValue(), '');
  assert.equal(await workspace.getByLabel('Source latitude column', {exact: true}).inputValue(), '');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 5, undefined, {timeout: 45000});
  const datasetId = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);
  await closeNativeDialog(page);
  const panel = page.getByRole('region', {name: 'GPU analysis', exact: true});
  await panel.getByLabel('Dataset', {exact: true}).selectOption(datasetId);
  await panel.getByLabel('Numeric field', {exact: true}).selectOption('measure');
  await panel.getByText('Configure geographic neighbors and spatial rates', {exact: true}).click();
  await setAnalysisSheetControl(panel.getByLabel('Spatial rate population field', {exact: true}), 'population');
  await setAnalysisSheetControl(panel.getByLabel('Neighbor coordinate system', {exact: true}), 'planar');
  await setAnalysisSheetControl(panel.getByLabel('Neighbor projected X field', {exact: true}), 'x');
  await setAnalysisSheetControl(panel.getByLabel('Neighbor projected Y field', {exact: true}), 'y');
  await setAnalysisSheetControl(panel.getByLabel('Neighbor declared projected CRS', {exact: true}), 'EPSG:3857');
  for (const [label, value] of [['minimum X', '179'], ['minimum Y', '-1'], ['maximum X', '201'], ['maximum Y', '1']])
    await setAnalysisSheetControl(panel.getByLabel(`Neighbor projected ${label} in meters`, {exact: true}), value);
  await setAnalysisSheetControl(panel.getByLabel('Neighbor topology', {exact: true}), 'distance-band');
  await setAnalysisSheetControl(panel.getByLabel('Neighbor radius in meters', {exact: true}), '1.1');
  await setAnalysisSheetControl(panel.getByLabel('Neighbor directed neighbor capacity', {exact: true}), '4');
  await setAnalysisSheetControl(panel.getByLabel('Rate scale', {exact: true}), '100');
  await setAnalysisSheetControl(panel.getByLabel('Rows without eligible neighbors', {exact: true}), 'undefined');
  stage('projected distance band preserves valid X beyond geographic longitude bounds');
  await panel.getByRole('button', {name: 'Apply neighbors and run spatial rates', exact: true}).click();
  await waitForProjectedRates();
  await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
  const recipe = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
  assert.equal(recipe.source.datasetId, datasetId);
  assert.equal(recipe.population, 'all');
  assert.deepEqual(recipe.operation, {
    kind: 'spatial-rate-points', events: 'measure', populations: 'population',
    weights: {
      kind: 'distance-band', coordinateSpace: 'planar', xColumn: 'x', yColumn: 'y',
      crs: 'EPSG:3857', bounds: [179, -1, 201, 1], metric: 'native', units: 'meters',
      radiusMeters: Math.fround(1.1), neighborCapacity: 4
    },
    scale: 100, islandPolicy: 'undefined'
  });
  stage('restore projected recipe and all coordinate controls');
  await panel.getByLabel('Operation', {exact: true}).selectOption('z-score');
  await page.waitForFunction(() => document.querySelector('[aria-label="GPU analysis"]')?.textContent.includes('5 analyzed'));
  await panel.getByRole('button', {name: 'Restore recipe', exact: true}).click();
  await waitForProjectedRates();
  assert.equal(await panel.getByLabel('Neighbor coordinate system', {exact: true}).inputValue(), 'planar');
  assert.equal(await panel.getByLabel('Neighbor projected X field', {exact: true}).inputValue(), 'x');
  assert.equal(await panel.getByLabel('Neighbor projected Y field', {exact: true}).inputValue(), 'y');
  assert.equal(await panel.getByLabel('Neighbor declared projected CRS', {exact: true}).inputValue(), 'EPSG:3857');
  for (const [label, value] of [['minimum X', '179'], ['minimum Y', '-1'], ['maximum X', '201'], ['maximum Y', '1']])
    assert.equal(await panel.getByLabel(`Neighbor projected ${label} in meters`, {exact: true}).inputValue(), value);
  stage('inspect original projected fields and source-aligned rates in native table');
  await page.locator('.show-data-table svg').last().click();
  await page.waitForFunction(() => {
    const table = document.querySelector('.resident-table-scroll');
    return table?.textContent.includes('Analysis: spatial rate') && table.querySelectorAll('tbody tr').length === 5;
  });
  const rows = await page.locator('.resident-table-scroll').evaluate(table => {
    const headers = [...table.querySelectorAll('thead th')];
    const rateColumn = headers.findIndex(header => header.textContent.includes('Analysis: spatial rate'));
    const xColumn = headers.findIndex(header => header.textContent.trim() === 'x');
    return [...table.querySelectorAll('tbody tr')].map(row => ({
      sourceRow: row.querySelector('th').textContent,
      x: row.children[xColumn]?.textContent,
      rate: row.children[rateColumn]?.textContent
    }));
  });
  assert.deepEqual(rows.map(row => row.sourceRow), ['0', '1', '2', '3', '4']);
  assert.deepEqual(rows.slice(0, 4).map(row => Number(row.x)), [180, 181, 182, 200]);
  assert.equal(rows[4].x, 'null');
  assert.deepEqual(rows.slice(0, 3).map(row => Number(row.rate)), [15, 30, 40]);
  assert.match(rows[3].rate, /^(NaN|null)$/);
  assert.match(rows[4].rate, /^(NaN|null)$/);
  await page.keyboard.press('Escape');
  await page.locator('.resident-table-modal').waitFor({state: 'hidden'});
  assert.deepEqual(await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].sourceBatches), [2, 3]);
  stage('remove projected table source');
  await openNativeData(page);
  await workspace.getByRole('button', {name: 'Remove projected-analysis.arrow', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications.length === 0);
  await closeNativeDialog(page);
  assert.deepEqual(failures, []);
  console.log(JSON.stringify({passed: true, checks: [
    'native table-only publication with multibatch projected coordinates and null validity',
    'planar meter distance-band rates, explicit CRS/bounds and recipe restoration',
    'native source-aligned table preserves X=181 and null coordinate row',
    'source batches retained and published dataset removed'
  ]}));
} catch (failure) {
  console.error(JSON.stringify({stage: currentStage, failures, diagnostics,
    body: (await page.locator('body').innerText().catch(() => '')).slice(-6000)}));
  throw failure;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
