// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Lead-run browser gate: point form defaults, two-action point statistic, overrides and lattice metadata.
import {strictEqual, deepStrictEqual, match} from 'node:assert';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {closeNativeDialog, openNativeData} from './native-navigation.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

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

async function publishSample(identifier) {
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.getByText('Try a sample dataset', {exact: true}).click();
  const samples = workspace.getByLabel('Sample dataset', {exact: true});
  await samples.selectOption(identifier);
  const title = await samples.locator(`option[value="${identifier}"]`).innerText();
  await workspace.getByRole('button', {name: 'Prepare sample', exact: true}).click();
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(label => [...document.querySelectorAll('#analysis-dataset option')]
    .some(option => option.textContent === label), title, {timeout: 60000});
  await closeNativeDialog(page);
  const analysis = page.getByRole('region', {name: 'GPU analysis', exact: true});
  await analysis.getByLabel('Dataset', {exact: true}).selectOption({label: title});
  return analysis;
}

try {
  await page.goto(address.href, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256,
    undefined, {timeout: 60000});
  const analysis = await publishSample('points-time');
  strictEqual(await analysis.locator('#analysis-column').inputValue(), 'reading');
  const coordinateControls = [
    ['interpolation-longitudeColumn', 'interpolation-latitudeColumn'],
    ['kriging-longitudeColumn', 'kriging-latitudeColumn'],
    ['analysis-dbscan-longitude', 'analysis-dbscan-latitude'],
    ['analysis-point-longitude', 'analysis-point-latitude'],
    ['analysis-point-hotspot-longitude', 'analysis-point-hotspot-latitude'],
    ['analysis-moran-longitude', 'analysis-moran-latitude'],
    ['analysis-quantile-lisa-longitudeColumn', 'analysis-quantile-lisa-latitudeColumn']
  ];
  // Read mounted controls without opening the other forms, preserving the measured action count.
  for (const [longitude, latitude] of coordinateControls) {
    strictEqual(await analysis.locator(`#${longitude}`).inputValue(), 'longitude', longitude);
    strictEqual(await analysis.locator(`#${latitude}`).inputValue(), 'latitude', latitude);
  }
  const extent = await page.evaluate(() => {
    const runtime = window.__keplerComputeTest.analysisForDataset(document.querySelector('#analysis-dataset').value);
    return runtime.formInitialization.interpolation;
  });
  for (const prefix of ['interpolation', 'kriging']) {
    for (const edge of ['west', 'south', 'east', 'north'])
      strictEqual(await analysis.locator(`#${prefix}-${edge}`).inputValue(), extent[edge]);
  }
  // The AX convention excludes dataset selection: opening the form and Apply are two actions.
  let actions = 0;
  await analysis.locator('.analysis-point-hotspot-settings > summary').click(); actions++;
  await analysis.getByRole('button', {name: 'Apply neighbors and run point hot spots', exact: true}).click(); actions++;
  await analysis.locator('[aria-live="polite"]').filter({hasText: 'Hot-spot z-score range'}).waitFor({timeout: 60000});
  strictEqual(actions, 2);
  match(await analysis.locator('[aria-live="polite"]').innerText(), /[\d,]+ analyzed/);

  await analysis.locator('.analysis-interpolation-settings > summary').click();
  await analysis.locator('#interpolation-west').fill('-75');
  const profiles = page.getByTestId('analysis-profiles');
  await profiles.locator('summary').click();
  const datasetId = await analysis.getByLabel('Dataset', {exact: true}).inputValue();
  const profile = profiles.locator(`fieldset[data-dataset-id="${datasetId}"]`);
  await profile.getByRole('button', {name: 'Compute bounded statistics', exact: true}).click();
  await profile.getByRole('status').filter({hasText: 'Bounded statistics ready'}).waitFor({timeout: 60000});
  strictEqual(await analysis.locator('#interpolation-west').inputValue(), '-75', 'profile refinement preserves unsaved bounds');

  const lattice = await publishSample('numeric-grid');
  strictEqual(await lattice.locator('#analysis-grid-width').inputValue(), '16');
  strictEqual(await lattice.locator('#analysis-grid-height').inputValue(), '16');
  deepStrictEqual(errors, []);
  process.stdout.write(`${JSON.stringify({passed: true, pointForms: coordinateControls.length,
    pointStatisticActions: actions, checks: ['selected value', 'shared coordinates', 'dataset extent',
      'unsaved override after statistics', 'lattice metadata']}, null, 2)}\n`);
} catch (error) {
  process.stderr.write(`${await page.locator('body').innerText()}\n`);
  throw error;
} finally {
  await browser.close();
}
