// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {closeNativeDialog, openNativeData, openNativeExport} from './native-navigation.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256');
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  assert.equal(await page.locator('.compute-panel').count(), 0);
  assert.equal(await page.getByText('Compute tools', {exact: true}).count(), 0);
  assert.equal(await page.getByLabel('Minimum value', {exact: true}).count(), 0);
  const canvasBounds = await page.locator('canvas').first().boundingBox();
  assert(canvasBounds.x + canvasBounds.width > 1400, 'Kepler must retain the right side of the map');
  const position = await page.evaluate(() => window.__keplerComputeTest.projectSourceRow(0));
  await page.mouse.move(position[0], position[1]);
  await page.getByRole('region', {name: 'Data inspection', exact: true}).getByText('Row', {exact: true}).waitFor();
  await page.screenshot({path: '/tmp/kepler-native-ui-tooltip.png'});
  await page.mouse.move(1200, 900);
  const originalConfiguration = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  await page.evaluate(configuration => {
    configuration.config.visState.interactionConfig.tooltip.fieldsToShow['resident-observations'] = [{name: 'value', format: null}];
    return window.__keplerComputeTest.applyNativeConfig(configuration);
  }, structuredClone(originalConfiguration));
  await page.mouse.move(position[0], position[1]);
  const inspection = page.getByRole('region', {name: 'Data inspection', exact: true});
  await inspection.getByText('value', {exact: true}).waitFor();
  assert.deepEqual(await inspection.locator('dt').allTextContents(), ['Row', 'value']);
  await page.mouse.move(1200, 900);
  await page.evaluate(configuration => window.__keplerComputeTest.applyNativeConfig(configuration), originalConfiguration);
  const analysis = page.getByTestId('native-analysis').getByRole('region', {name: 'GPU analysis', exact: true});
  await analysis.getByLabel('Operation', {exact: true}).selectOption('z-score');
  await analysis.getByText(/256 analyzed/).waitFor();
  await page.screenshot({path: '/tmp/kepler-native-ui-layers.png'});

  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.getByText('Try a sample dataset', {exact: true}).click();
  await workspace.getByLabel('Sample dataset', {exact: true}).selectOption('districts');
  await workspace.getByRole('button', {name: 'Prepare sample', exact: true}).click();
  assert.equal(await workspace.getByLabel('Source geometry column').inputValue(), 'geometry');
  await page.screenshot({path: '/tmp/kepler-native-ui-data.png'});
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 16);

  await closeNativeDialog(page);
  assert.equal(await analysis.getByLabel('Operation', {exact: true}).inputValue(), 'z-score');
  const datasetId = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);
  await analysis.getByLabel('Dataset', {exact: true}).selectOption(datasetId);
  await page.locator('.side-panel__tab[data-for="filter-nav"]').click();
  await page.locator('.side-panel__tab[data-for="layer-nav"]').click();
  await page.waitForFunction(id => document.querySelector('#analysis-dataset')?.value === id, datasetId);


  await openNativeExport(page);
  assert.equal(await page.getByRole('button', {name: 'Open sandbox controls'}).count(), 0);
  assert.equal(await page.getByRole('button', {name: 'Save state', exact: true}).count(), 0);
  const pendingDownload = page.waitForEvent('download');
  await page.getByRole('button', {name: 'Export 100-row page', exact: true}).click();
  const stream = await (await pendingDownload).createReadStream();
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  assert.equal(JSON.parse(Buffer.concat(chunks).toString()).rowCount, 100);
  await page.screenshot({path: '/tmp/kepler-native-ui-export.png'});
  await openNativeExport(page, true);
  assert.equal(await page.getByLabel('Export dataset', {exact: true}).count(), 0);
  await page.getByRole('button', {name: 'Save state', exact: true}).click();
  await page.getByRole('button', {name: 'Replay state', exact: true}).click();
  await page.locator('.ReactModal__Overlay').waitFor({state: 'hidden'});
  await page.waitForFunction(() => window.__keplerComputeTest.analysis()?.settings.mode === 'z-score');
  await closeNativeDialog(page);

  await openNativeData(page, 'SQL');
  await page.getByRole('region', {name: 'Native SQL Data Explorer'}).waitFor();
  await closeNativeDialog(page);
  await page.setViewportSize({width: 390, height: 844});
  await openNativeData(page);
  await workspace.getByText('Try a sample dataset', {exact: true}).click();
  await workspace.getByLabel('Sample dataset', {exact: true}).selectOption('districts');
  await workspace.getByRole('button', {name: 'Prepare sample', exact: true}).click();
  const geometry = workspace.getByLabel('Source geometry column');
  await geometry.scrollIntoViewIfNeeded();
  assert(await geometry.evaluate(element => {
    const bounds = element.getBoundingClientRect();
    return document.elementFromPoint(bounds.left + bounds.width / 2, bounds.top + bounds.height / 2) === element;
  }), 'Native mobile modal controls must be reachable');
  await page.screenshot({path: '/tmp/kepler-native-ui-mobile.png'});
  assert.deepEqual(errors, []);
  console.log('Native Layers analysis, Add Data preparation/SQL, direct export/save, replay, full map width and mobile controls passed.');
} catch (error) {
  console.error((await page.locator('body').innerText()).slice(-12000));
  await page.screenshot({path: '/tmp/kepler-native-ui-failure.png'});
  throw error;
} finally {
  await browser.close();
}
