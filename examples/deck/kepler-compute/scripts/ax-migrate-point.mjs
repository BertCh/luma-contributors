// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Lead-run gate: all five resident point sheets retain their methods, results and saved recipes.
import {deepStrictEqual, strictEqual, match} from 'node:assert';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {closeNativeDialog, openNativeData} from './native-navigation.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {setAnalysisSheetControl} from './analysis-sheet-controls.mjs';

const address = new URL(process.argv[2] || 'http://localhost:5318/');
address.searchParams.set('rows', '256');
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => {
  if (/Cannot update a component|Maximum update depth|getSnapshot.*cached/.test(message.text()))
    errors.push(message.text());
});
const source = arrow.tableFromArrays({
  east: new Float32Array([0, 0.0001, 0.0002, 0.03, 0.1]),
  north: new Float32Array([0, 0, 0, 0, 0]),
  reading: new Float32Array([1, 2, 3, 4, 5]),
  other: new Float32Array([3, 1, 4, 2, 5]),
  population: new Float32Array([100, 100, 100, 100, 100])
});

async function readSummary(datasetId) {
  return page.evaluate(identifier => window.__keplerComputeTest.analysisForDataset(identifier).summary?.dbscan, datasetId);
}
async function waitForSummary(datasetId, noiseCount) {
  await page.waitForFunction(({identifier, noise}) => {
    const summary = window.__keplerComputeTest.analysisForDataset(identifier).summary?.dbscan;
    return summary?.complete && summary.clusterCount === 1 && summary.noiseCount === noise;
  }, {identifier: datasetId, noise: noiseCount}, {timeout: 60000});
}

try {
  await page.goto(address.href, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: 'point-sheet.arrow', mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))
  });
  await workspace.getByLabel('Source longitude column', {exact: true}).selectOption('east');
  await workspace.getByLabel('Source latitude column', {exact: true}).selectOption('north');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 5, undefined, {timeout: 60000});
  const datasetId = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);
  await closeNativeDialog(page);
  const analysis = page.getByRole('region', {name: 'GPU analysis', exact: true});
  await analysis.getByLabel('Dataset', {exact: true}).selectOption(datasetId);
  const panel = analysis.locator('details.analysis-dbscan-settings');
  await panel.locator(':scope > summary').click();
  strictEqual(await panel.getByTestId('ax-analysis-sheet').count(), 1);
  strictEqual(await panel.locator('#analysis-dbscan-longitude').inputValue(), 'east');
  strictEqual(await panel.locator('#analysis-dbscan-latitude').inputValue(), 'north');
  match(await panel.getByTestId('ax-resident-method').innerText(), /Catalog density clustering uses a local planar frame/);
  const apply = panel.getByRole('button', {name: 'Apply radius and run DBSCAN', exact: true});
  await setAnalysisSheetControl(panel.locator('#analysis-dbscan-epsilon'), 30);
  await setAnalysisSheetControl(panel.locator('#analysis-dbscan-minimum'), 3);
  await apply.click();
  await waitForSummary(datasetId, 2);
  deepStrictEqual(await readSummary(datasetId), {clusterCount: 1, noiseCount: 2, validCount: 5, excludedCount: 0, complete: true});
  const rows = await page.evaluate(async identifier => {
    const runtime = window.__keplerComputeTest.analysisForDataset(identifier);
    // readRow is a bounded picked-row API (at most four in flight), not a bulk reader.
    const labels = [];
    for (const row of [0, 1, 2, 3, 4]) labels.push((await runtime.readRow(row))?.value);
    return labels;
  }, datasetId);
  deepStrictEqual(rows, [0, 0, 0, -1, -1]);
  await analysis.getByRole('button', {name: 'Save recipe', exact: true}).click();
  const recipeText = await analysis.getByLabel('Saved analysis recipe', {exact: true}).inputValue();
  const recipe = JSON.parse(recipeText);
  deepStrictEqual(recipe.operation, {kind: 'dbscan-points', longitudeColumn: 'east', latitudeColumn: 'north',
    coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'meters', epsilon: 30, minimumSamples: 3});

  // Invalid transient input must disable Apply without removing the accepted result.
  await setAnalysisSheetControl(panel.locator('#analysis-dbscan-epsilon'), -1);
  strictEqual(await apply.isDisabled(), true);
  deepStrictEqual(await readSummary(datasetId), {clusterCount: 1, noiseCount: 2, validCount: 5, excludedCount: 0, complete: true});
  // Releasing valid text changes only the draft until Apply.
  await setAnalysisSheetControl(panel.locator('#analysis-dbscan-epsilon'), 4000);
  strictEqual(await page.evaluate(identifier => window.__keplerComputeTest.analysisForDataset(identifier).settings.dbscan.epsilon, datasetId), 30);
  await apply.click();
  await waitForSummary(datasetId, 1);
  await analysis.getByRole('button', {name: 'Restore recipe', exact: true}).click();
  await waitForSummary(datasetId, 2);
  strictEqual(await panel.locator('#analysis-dbscan-epsilon').inputValue(), '30');
  strictEqual(await panel.locator('#analysis-dbscan-minimum').inputValue(), '3');
  strictEqual(await panel.locator('#analysis-dbscan-longitude').inputValue(), 'east');
  strictEqual(await panel.locator('#analysis-dbscan-latitude').inputValue(), 'north');
  await analysis.getByLabel('Numeric field', {exact: true}).selectOption('reading');
  strictEqual(await analysis.locator('.analysis-dbscan-settings, .analysis-point-weights-settings, .analysis-point-hotspot-settings, .analysis-local-moran-settings, .analysis-quantile-lisa-settings').getByTestId('ax-analysis-sheet').count(), 5, 'all five point forms are generated sheets');
  const cases = [
    {prefix: 'analysis-point', className: 'analysis-point-weights-settings', button: 'Apply neighbors and run spatial rates',
      kind: 'spatial-rate-points', neighbors: 'analysis-point-neighbors', controls: {population: 'population', neighbors: 2, scale: 100}},
    {prefix: 'analysis-point-hotspot', className: 'analysis-point-hotspot-settings', button: 'Apply neighbors and run point hot spots',
      kind: 'hot-spots-points', neighbors: 'analysis-point-hotspot-neighbors', controls: {topology: 'knn', neighbors: 2, minimum: 1, correction: 'none'}},
    {prefix: 'analysis-moran', className: 'analysis-local-moran-settings', button: 'Apply neighbors and run Local Moran',
      kind: 'local-moran-points', neighbors: 'analysis-moran-neighbors', controls: {neighbors: 2, minimum: 1, correction: 'none'}},
    {prefix: 'analysis-quantile-lisa', className: 'analysis-quantile-lisa-settings', button: 'Apply neighbors and run Quantile LISA',
      kind: 'quantile-lisa-points', neighbors: 'analysis-quantile-lisa-k', controls: {k: 2, permutationCount: 19, seed: 2 ** 40 + 5}},
    {prefix: 'analysis-moran', className: 'analysis-local-moran-settings', button: 'Apply neighbors and run Local Moran',
      kind: 'local-moran-points', neighbors: 'analysis-moran-neighbors', controls: {mode: 'bivariate', 'second-column': 'other', permutations: 19}},
    {prefix: 'analysis-moran', className: 'analysis-local-moran-settings', button: 'Apply neighbors and run Local Moran',
      kind: 'multivariate-local-geary-points', neighbors: 'analysis-moran-neighbors', controls: {mode: 'multivariate', 'variable-0': 'reading', 'variable-1': 'other'}}
  ];
  async function waitForPointResult() {
    await page.waitForFunction(identifier => {
      const runtime = window.__keplerComputeTest.analysisForDataset(identifier);
      return runtime.summary?.operation === runtime.settings.mode && !runtime.error;
    }, datasetId, {timeout: 60000});
  }
  for (const item of cases) {
    const pointPanel = analysis.locator(`details.${item.className}`);
    for (const [name, value] of Object.entries(item.controls))
      await setAnalysisSheetControl(pointPanel.locator(`#${item.prefix}-${name}`), value);
    await pointPanel.getByRole('button', {name: item.button, exact: true}).click();
    await waitForPointResult();
    await analysis.getByRole('button', {name: 'Save recipe', exact: true}).click();
    const savedText = await analysis.getByLabel('Saved analysis recipe', {exact: true}).inputValue();
    const saved = JSON.parse(savedText);
    strictEqual(saved.operation.kind, item.kind);
    strictEqual(saved.operation.weights.k, 2);
    strictEqual(saved.operation.weights.longitudeColumn, 'east');
    strictEqual(saved.operation.weights.latitudeColumn, 'north');
    if (item.controls.mode === 'bivariate') strictEqual(saved.operation.secondColumn, 'other');
    if (item.controls.mode === 'multivariate') deepStrictEqual(saved.operation.columns, ['reading', 'other']);
    if (item.kind === 'quantile-lisa-points') strictEqual(saved.operation.seed, 2 ** 40 + 5);
    // Edits preserve the accepted result until Apply, then restoring the saved recipe resets controls.
    await setAnalysisSheetControl(pointPanel.locator(`#${item.neighbors}`), 1);
    const retained = JSON.parse(await page.evaluate(identifier => window.__keplerComputeTest.analysisForDataset(identifier).serializeRecipe(), datasetId));
    deepStrictEqual(retained.operation, saved.operation);
    await pointPanel.getByRole('button', {name: item.button, exact: true}).click();
    await waitForPointResult();
    await analysis.getByRole('button', {name: 'Restore recipe', exact: true}).click();
    await waitForPointResult();
    strictEqual(await pointPanel.locator(`#${item.neighbors}`).inputValue(), '2');
    const restored = JSON.parse(await page.evaluate(identifier => window.__keplerComputeTest.analysisForDataset(identifier).serializeRecipe(), datasetId));
    deepStrictEqual(restored.operation, saved.operation);
  }
  deepStrictEqual(errors, []);
  process.stdout.write(`${JSON.stringify({passed: true, operations: ['dbscan', 'point-weights', 'point-hot-spots', 'local-moran', 'quantile-lisa'], checks: [
    'published coordinate roles', 'generated sheet', 'method difference', 'hand-computed clusters and source-row labels',
    'invalid draft blocks Apply', 'accepted result retained during edits', 'manual rebuild', 'all point recipes restored',
    'bivariate Moran and ordered multivariate Geary', 'exact large Quantile LISA seed'
  ]}, null, 2)}\n`);
} catch (error) {
  process.stderr.write(`${await page.locator('body').innerText()}\n`);
  throw error;
} finally {
  await browser.close();
}
