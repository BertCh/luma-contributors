// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Lead-run gate: real catalog staging/publication, field-named native legends, reports and screenshots.
import {setAnalysisSheetParameter} from './analysis-sheet-controls.mjs';
import {deepStrictEqual, strictEqual, match} from 'node:assert';
import {mkdir} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const address = new URL(process.argv[2] || 'http://localhost:5318/?rows=256');
address.searchParams.set('rows', '256');
const outputDirectory = process.argv[3] || '.ax/streams/result-publication/artifacts';
await mkdir(outputDirectory, {recursive: true});
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const failures = [];
page.on('pageerror', error => failures.push(error.message));
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/.test(message.text()))
    failures.push(message.text());
});
const watchdog = setTimeout(() => {void browser.close();}, 360000);

try {
  await page.goto(address.href, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256,
    undefined, {timeout: 240000});
  const panel = page.getByRole('region', {name: 'Analysis catalog', exact: true});
  const operations = ['hotspot-gi-star', 'local-moran-statistics', 'recipe-points-in-polygons-choropleth'];
  for (const operation of operations) {
    await panel.getByLabel('Operation', {exact: true}).selectOption(operation);
    await panel.getByLabel('Engine', {exact: true}).selectOption('cpu');
    await setAnalysisSheetParameter(panel, 'coordinateSystem', 'planar', {optional: true});
    await panel.getByTestId('catalog-run').click();
    await panel.getByTestId('catalog-result').waitFor({timeout: 60000});
    const previousIds = await page.evaluate(() => window.__keplerComputeTest.getState().publications.map(source => source.id));
    await panel.getByTestId('catalog-publish').first().click();
    await openNativeData(page);
    const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
    const report = workspace.getByTestId('analysis-result-report');
    await report.waitFor();
    const expected = await report.evaluate(element => ({
      column: element.querySelector('h3').textContent.replace('Colour: ', ''),
      entries: [...element.querySelectorAll('li')].map(entry => ({
        label: entry.textContent.replace(/ \(\d+\)$/, ''),
        color: [...entry.querySelector('span').style.backgroundColor.matchAll(/\d+/g)]
          .map(value => Number(value[0]))
      }))
    }));
    const preparation = workspace.getByRole('region', {name: /^Prepare /});
    await preparation.getByLabel('Float64 conversion', {exact: true}).selectOption('round');
    await preparation.getByLabel('Maximum absolute error', {exact: true}).fill('1e-37');
    await preparation.getByLabel('Maximum relative error', {exact: true}).fill(String(2 ** -23));
    await preparation.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
    await preparation.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
    await page.waitForFunction(known => window.__keplerComputeTest.getState().publications.some(source =>
      !known.includes(source.id) && source.selectedCount === source.rowCount), previousIds, {timeout: 60000});
    const configuration = await page.evaluate(known => {
      const source = window.__keplerComputeTest.getState().publications.find(candidate => !known.includes(candidate.id));
      const config = window.__keplerComputeTest.saveNativeConfig();
      return config.config.visState.layers.find(layer => layer.config.dataId === source.id);
    }, previousIds);
    strictEqual(configuration.visualChannels.colorScale, 'customOrdinal');
    match(configuration.visualChannels.colorField.name, new RegExp(expected.column));
    strictEqual(configuration.config.visConfig.colorRange.name, expected.column);
    const toHex = color => `#${color.map(value => value.toString(16).padStart(2, '0')).join('')}`;
    deepStrictEqual(configuration.config.visConfig.colorRange.colorMap,
      expected.entries.map(entry => [entry.label, toHex(entry.color)]));
    await closeNativeDialog(page);
    // Require the renderer's legend as well as the serialized configuration.
    if (!await page.locator('.map-control-panel').isVisible())
      await page.locator('.map-control-button.show-legend').click();
    // Several outputs name their source column classification. Match this published layer.
    const legend = page.locator(`[data-resident-style-legend="fillColor"][data-resident-legend-layer="${configuration.id}"]`);
    await legend.waitFor({timeout: 60000});
    strictEqual(await legend.locator('strong').innerText(), expected.column);
    for (const entry of expected.entries)
      strictEqual((await legend.innerText()).includes(entry.label), true, `Missing exact class label: ${entry.label}`);
    await page.screenshot({path: `${outputDirectory}/${operation}.png`});
  }
  await panel.getByLabel('Operation', {exact: true}).selectOption('global-spatial-statistics');
  await setAnalysisSheetParameter(panel, 'coordinateSystem', 'planar', {optional: true});
  await panel.getByTestId('catalog-run').click();
  await panel.getByTestId('catalog-result').waitFor({timeout: 60000});
  const beforeReport = await page.evaluate(() => window.__keplerComputeTest.getState().publications.map(source => source.id));
  await panel.getByTestId('catalog-publish').first().click();
  await page.getByRole('dialog').filter({has: page.getByTestId('analysis-result-report')}).waitFor();
  deepStrictEqual(await page.evaluate(() => window.__keplerComputeTest.getState().publications.map(source => source.id)), beforeReport);
  await page.screenshot({path: `${outputDirectory}/global-report.png`});
  await page.getByRole('button', {name: 'Close report', exact: true}).click();

  // Exercise the actual workbench host's report action independently of Sources selection.
  const output = await page.evaluate(async () => {
    const {createWorkbenchHost} = await import('/src/workflow/workbench-host.ts');
    const host = createWorkbenchHost();
    const request = {workflowId: 'ax-report', workflowTitle: 'Report', stepId: 'global', output: 'global',
      label: 'Workbench global report', data: {kind: 'table', table: {id: 'global', label: 'Global', rowCount: 1,
        columns: {morans_i: [0.12345678912345]}, columnTypes: {morans_i: 'float64'}},
        analysisPresentation: {tableId: 'global', display: {}, legend: []}}};
    const published = await host.publishOutput(request, new AbortController().signal);
    window.axReportCleanup = () => host.retractOutput(published.datasetId);
    return published.datasetId;
  });
  match(output, /^workflow-/);
  const workbenchReport = page.getByRole('dialog', {name: 'Workbench global report', exact: true});
  await workbenchReport.waitFor();
  match(await workbenchReport.innerText(), /0.12345678912345/);
  deepStrictEqual(await page.evaluate(() => window.__keplerComputeTest.getState().publications.map(source => source.id)), beforeReport);
  await page.evaluate(() => window.axReportCleanup());
  await workbenchReport.waitFor({state: 'hidden'});
  deepStrictEqual(failures, []);
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
