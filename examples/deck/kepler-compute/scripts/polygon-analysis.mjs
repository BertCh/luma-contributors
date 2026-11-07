// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

function square(west, south, east, north) {
  return JSON.stringify({
    type: 'Feature',
    properties: {},
    geometry: {
      type: 'Polygon',
      coordinates: [[[west, south], [east, south], [east, north], [west, north], [west, south]]]
    }
  });
}

// A–B share an edge; B–C share only a corner; D is isolated; E retains a null geometry row.
const source = arrow.tableFromArrays({
  geometry: [square(0, 0, 1, 1), square(1, 0, 2, 1)],
  measure: new Float32Array([10, 20]),
  population: new Float32Array([100, 100])
}).concat(arrow.tableFromArrays({
  geometry: [square(2, 1, 3, 2), square(10, 10, 11, 11), null],
  measure: new Float32Array([60, 40, 50]),
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
  console.error(`Polygon analysis workflow exceeded 150 seconds at ${currentStage}`);
  void browser.close();
}, 150000);

async function waitForRate(linkCount, minimum, maximum, analyzedCount) {
  await page.waitForFunction(({linkCount, minimum, maximum, analyzedCount}) => {
    const text = document.querySelector('[aria-label="GPU analysis"]')?.textContent ?? '';
    const diagnostics = document.querySelector('[aria-label="Rate diagnostics"]');
    const definedOutputs = [...(diagnostics?.querySelectorAll('dt') ?? [])]
      .find(term => term.textContent === 'Defined outputs')?.nextElementSibling?.textContent;
    return text.includes(`${linkCount} polygon neighbor links`) &&
      text.includes(`Spatial rate range: ${minimum} to ${maximum}`) &&
      definedOutputs === String(analyzedCount);
  }, {linkCount, minimum, maximum, analyzedCount}, {timeout: 30000});
}

try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  stage('publish multibatch polygon source through Prepare data');
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: 'polygon-analysis.arrow', mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))
  });
  await workspace.getByLabel('Source geometry column', {exact: true}).selectOption('geometry');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 5, undefined, {timeout: 45000});
  const datasetId = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);
  await closeNativeDialog(page);
  const panel = page.getByRole('region', {name: 'GPU analysis', exact: true});
  await panel.getByLabel('Dataset', {exact: true}).selectOption(datasetId);
  await panel.getByLabel('Numeric field', {exact: true}).selectOption('measure');
  await panel.getByText('Configure polygon contiguity analysis', {exact: true}).click();
  await panel.getByLabel('Polygon statistic', {exact: true}).selectOption('spatial-rate-polygons');
  await panel.getByLabel('Polygon geometry field', {exact: true}).selectOption('geometry');
  await panel.getByLabel('Polygon contiguity rule', {exact: true}).selectOption('queen');
  await panel.getByLabel('Polygon directed neighbor capacity', {exact: true}).fill('4');
  await panel.getByLabel('Polygon candidate pair capacity', {exact: true}).fill('64');
  await panel.getByLabel('Polygon rate population field', {exact: true}).selectOption('population');
  await panel.getByLabel('Polygon rate scale', {exact: true}).fill('100');
  await panel.getByLabel('Polygon rate rows without neighbors', {exact: true}).selectOption('undefined');
  stage('queen rates preserve source rows and isolate D/E');
  await panel.getByRole('button', {name: 'Apply contiguity and run polygon analysis', exact: true}).click();
  // Queen rates are A=15, B=30, C=40; D has no neighbors and E has no polygon.
  await waitForRate(4, 15, 40, 3);
  assert.match(await panel.innerText(), /1 rows without eligible polygon parts/);
  await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
  const queenRecipe = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
  assert.equal(queenRecipe.source.datasetId, datasetId);
  assert.equal(queenRecipe.population, 'all');
  assert.equal(queenRecipe.operation.kind, 'spatial-rate-polygons');
  assert.deepEqual(queenRecipe.operation.weights, {
    kind: 'contiguity', geometryColumn: 'geometry', criterion: 'queen',
    equality: 'exact-source-xy', neighborCapacity: 4, pairCapacity: 64
  });
  assert.equal(queenRecipe.operation.events, 'measure');
  assert.equal(queenRecipe.operation.populations, 'population');
  assert.equal(queenRecipe.operation.scale, 100);
  assert.equal(queenRecipe.operation.islandPolicy, 'undefined');

  stage('rook removes corner-only B–C adjacency');
  await panel.getByLabel('Polygon contiguity rule', {exact: true}).selectOption('rook');
  await panel.getByRole('button', {name: 'Apply contiguity and run polygon analysis', exact: true}).click();
  await waitForRate(2, 15, 15, 2);
  stage('restore queen recipe and inspect native derived table');
  await panel.getByRole('button', {name: 'Restore recipe', exact: true}).click();
  await waitForRate(4, 15, 40, 3);
  assert.equal(await panel.getByLabel('Polygon contiguity rule', {exact: true}).inputValue(), 'queen');
  await page.locator('.show-data-table svg').last().click();
  await page.waitForFunction(() => {
    const table = document.querySelector('.resident-table-scroll');
    return table?.textContent.includes('Analysis: spatial rate') && table.querySelectorAll('tbody tr').length === 5;
  });
  const derivedRows = await page.locator('.resident-table-scroll').evaluate(table => {
    const columnIndex = [...table.querySelectorAll('thead th')].findIndex(header => header.textContent.includes('Analysis: spatial rate'));
    return [...table.querySelectorAll('tbody tr')].map(row => ({
      sourceRow: row.querySelector('th').textContent,
      value: row.children[columnIndex]?.textContent
    }));
  });
  assert.deepEqual(derivedRows.map(row => row.sourceRow), ['0', '1', '2', '3', '4']);
  assert.deepEqual(derivedRows.slice(0, 3).map(row => Number(row.value)), [15, 30, 40]);
  assert.match(derivedRows[3].value, /^(NaN|null)$/);
  assert.equal(derivedRows[4].value, 'null');
  await page.keyboard.press('Escape');
  await page.locator('.resident-table-modal').waitFor({state: 'hidden'});

  stage('configure polygon Quantile LISA legacy inference');
  await panel.getByLabel('Polygon statistic', {exact: true}).selectOption('quantile-lisa-polygons');
  await panel.getByLabel('Polygon Quantile LISA quantile count', {exact: true}).fill('2');
  await panel.getByLabel('Polygon Quantile LISA selected quantile', {exact: true}).fill('1');
  await panel.getByLabel('Polygon Quantile LISA seed', {exact: true}).fill('7');
  await panel.getByLabel('Polygon Quantile LISA permutations', {exact: true}).fill('99');
  await panel.getByLabel('Polygon Quantile LISA significance level', {exact: true}).fill('0.05');
  await panel.getByRole('button', {name: 'Apply contiguity and run polygon analysis', exact: true}).click();
  async function waitForQuantile() {
    await page.waitForFunction(() => {
      const text = document.querySelector('[aria-label="GPU analysis"]')?.textContent ?? '';
      return text.includes('Quantile interval: [−∞, 30)') &&
        text.includes('4 finite polygon-valid observations') && text.includes('1 isolated') &&
        text.includes('4 polygon neighbor links');
    }, undefined, {timeout: 30000});
  }
  await waitForQuantile();
  assert.match(await panel.innerText(), /legacy directed rule without correction/);
  await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
  const quantileRecipe = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
  assert.deepEqual(quantileRecipe.operation, {
    kind: 'quantile-lisa-polygons', column: 'measure', weights: queenRecipe.operation.weights,
    quantileCount: 2, selectedQuantile: 1, seed: 7, permutationCount: 99,
    significanceLevel: 0.05, inference: 'legacy-directed-pseudo', multipleTesting: 'none',
    islands: 'exclude', quantileMethod: 'type-5'
  });
  stage('restore polygon Quantile LISA recipe through native controls');
  await panel.getByLabel('Polygon statistic', {exact: true}).selectOption('spatial-rate-polygons');
  await panel.getByLabel('Polygon rate scale', {exact: true}).fill('100');
  await panel.getByRole('button', {name: 'Apply contiguity and run polygon analysis', exact: true}).click();
  await waitForRate(4, 15, 40, 3);
  await panel.getByRole('button', {name: 'Restore recipe', exact: true}).click();
  await waitForQuantile();
  assert.equal(await panel.getByLabel('Polygon statistic', {exact: true}).inputValue(), 'quantile-lisa-polygons');
  assert.equal(await panel.getByLabel('Polygon Quantile LISA quantile count', {exact: true}).inputValue(), '2');
  assert.equal(await panel.getByLabel('Polygon Quantile LISA selected quantile', {exact: true}).inputValue(), '1');
  assert.equal(await panel.getByLabel('Polygon Quantile LISA seed', {exact: true}).inputValue(), '7');
  assert.equal(await panel.getByLabel('Polygon Quantile LISA permutations', {exact: true}).inputValue(), '99');
  assert.equal(await panel.getByLabel('Polygon Quantile LISA significance level', {exact: true}).inputValue(), '0.05');

  stage('configure polygon Local Moran conditional inference');
  await panel.getByLabel('Polygon statistic', {exact: true}).selectOption('local-moran-polygons');
  await panel.getByLabel('Polygon Local Moran inference', {exact: true}).selectOption('permutation-conditional-two-sided');
  await panel.getByLabel('Polygon Local Moran permutations', {exact: true}).fill('99');
  await panel.getByLabel('Polygon Local Moran permutation seed', {exact: true}).fill('7');
  await panel.getByLabel('Polygon multiple-testing correction', {exact: true}).selectOption('none');
  await panel.getByRole('button', {name: 'Apply contiguity and run polygon analysis', exact: true}).click();
  await page.waitForFunction(() => {
    const text = document.querySelector('[aria-label="GPU analysis"]')?.textContent ?? '';
    return text.includes('Local Moran I range:') && text.includes('included rows without permutation inference') && text.includes('4 polygon neighbor links');
  });
  assert.match(await panel.innerText(), /z-score remains a separate analytic diagnostic/);
  await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
  const moranRecipe = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
  assert.equal(moranRecipe.operation.kind, 'local-moran-polygons');
  assert.equal(moranRecipe.operation.inference, 'permutation-conditional-two-sided');
  assert.equal(moranRecipe.operation.permutationCount, 99);
  assert.equal(moranRecipe.operation.seed, 7);
  assert.equal(moranRecipe.operation.minimumNeighbors, 1);
  assert.deepEqual(moranRecipe.operation.weights, queenRecipe.operation.weights);
  assert.deepEqual(await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].sourceBatches), [2, 3]);

  stage('remove published source');
  await openNativeData(page);
  await workspace.getByRole('button', {name: 'Remove polygon-analysis.arrow', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications.length === 0);
  await closeNativeDialog(page);
  assert.deepEqual(failures, []);
  console.log(JSON.stringify({passed: true, checks: [
    'native Prepare data publication with original multibatch polygon rows',
    'queen shared vertices and rook complete edges produce distinct spatial rates',
    'per-analysis queen recipe restore and native source-aligned derived table',
    'polygon Quantile LISA legacy policy, source validity and recipe restore',
    'polygon conditional Local Moran policy and recipe settings',
    'dataset removal'
  ]}));
} catch (failure) {
  console.error(JSON.stringify({stage: currentStage, failures, diagnostics,
    body: (await page.locator('body').innerText().catch(() => '')).slice(-6000)}));
  throw failure;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
