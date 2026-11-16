// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Native CSV auto-prepare -> shared metadata -> bounded statistics -> catalog -> saved declarations.
// Run with a Vite server listening on localhost. GPU/browser execution belongs to the lead verifier.
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {closeNativeDialog, openNativeExport} from './native-navigation.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const address = new URL(process.argv[2] || 'http://localhost:5318/');
address.searchParams.set('rows', '256');
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => {
  if (/Cannot update a component|Maximum update depth|getSnapshot.*cached/.test(message.text()))
    errors.push(message.text());
});
async function downloadJson(button) {
  const download = page.waitForEvent('download');
  await button.click();
  const stream = await (await download).createReadStream();
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}
try {
  await page.goto(address.href, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => {
    const state = window.__keplerComputeTest?.getState();
    if (state?.error) throw new Error(state.error);
    return state?.selectedCount === 256;
  }, undefined, {timeout: 60000});
  const csv = 'longitude,latitude,value,recorded\n' + Array.from({length: 24}, (_, row) =>
    `${-74 + row / 10000},${40.7 + row % 4 / 10000},${row + 1},2026-11-14T12:${String(row).padStart(2, '0')}:00Z`
  ).join('\n');
  await page.getByRole('button', {name: 'Add Data', exact: true}).click();
  await page.locator('.ReactModal__Content input[type=file]').setInputFiles({
    name: 'ax-profile-observations.csv', mimeType: 'text/csv', buffer: Buffer.from(csv)
  });
  await page.locator('.ReactModal__Content').getByRole('button', {name: 'Add Data', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications.some(source => source.rowCount === 24), undefined, {timeout: 60000});
  await closeNativeDialog(page);
  const profiles = page.getByTestId('analysis-profiles');
  await profiles.locator('summary').click();
  const dataset = profiles.locator('fieldset').filter({has: page.locator('legend').filter({hasText: 'ax-profile-observations'})});
  await dataset.waitFor();
  const datasetId = await dataset.getAttribute('data-dataset-id');
  assert.ok(datasetId, 'uploaded source has stable dataset identity');
  assert.match(await dataset.innerText(), /longitude\/latitude \(publication roles\)/);
  assert.match(await dataset.innerText(), /Metadata ready; statistics not read/);
  assert.match(await profiles.innerText(), /20,000 rows/);
  assert.match(await profiles.innerText(), /200,000 vertices/);
  const rowMeaning = dataset.getByLabel(/^Row meaning for/);
  const coordinateSystem = dataset.getByLabel(/^Coordinate system for/);
  const measurement = dataset.getByLabel(/^Measurement of value for/);
  await rowMeaning.selectOption('surface-samples');
  await coordinateSystem.selectOption('geographic');
  await measurement.selectOption('amount');
  await dataset.getByRole('button', {name: 'Compute bounded statistics', exact: true}).click();
  await dataset.getByRole('status').filter({hasText: 'Bounded statistics ready'}).waitFor({timeout: 30000});
  assert.equal(await rowMeaning.inputValue(), 'surface-samples');
  assert.match(await dataset.innerText(), /value: amount \(declared\)/);

  const catalog = page.getByRole('region', {name: 'Analysis catalog', exact: true});
  await catalog.getByLabel('Operation', {exact: true}).selectOption('geometry-point-buffer');
  await catalog.getByLabel('Engine', {exact: true}).selectOption('auto');
  await catalog.getByLabel('Points source', {exact: true}).selectOption('native');
  await catalog.getByLabel('Points dataset', {exact: true}).selectOption(datasetId);
  // This chooses the publication role mapping. No GeoJSON source column or manual coordinate guesses.
  await catalog.getByLabel('Points geometry field', {exact: true}).selectOption('@point:["",""]');
  await catalog.getByTestId('catalog-run').click();
  await catalog.getByTestId('catalog-result').waitFor({timeout: 60000});
  assert.equal(await catalog.getByTestId('catalog-error').count(), 0);
  assert.match(await catalog.getByTestId('catalog-engine').innerText(), /Engine: GPU/);
  assert.match(await catalog.getByTestId('catalog-result').innerText(), /24 rows/);

  await openNativeExport(page, true);
  const saving = page.getByRole('region', {name: 'Save map', exact: true});
  const saved = await downloadJson(saving.getByRole('button', {name: 'Save state', exact: true}));
  const declarations = saved.settings.extensions.analysisProfileDeclarations;
  assert.deepEqual(JSON.parse(declarations.datasets[datasetId]), {
    version: 1, measurement: {value: 'amount'}, rowSemantics: 'surface-samples', coordinateModel: 'geographic'
  });
  await closeNativeDialog(page);
  await rowMeaning.selectOption('objects');
  await coordinateSystem.selectOption('planar');
  await measurement.selectOption('rate');
  await openNativeExport(page, true);
  // The modal is remounted on open; replay the actual downloaded state explicitly.
  await saving.getByText('Inspect or edit saved state', {exact: true}).click();
  await saving.getByLabel('Saved sandbox state', {exact: true}).fill(JSON.stringify(saved));
  await saving.getByRole('button', {name: 'Replay state', exact: true}).click();
  await closeNativeDialog(page);
  await page.waitForFunction(sourceDatasetId => {
    const source = document.querySelector(`[data-dataset-id="${CSS.escape(sourceDatasetId)}"]`);
    return source?.querySelector('select[aria-label^="Row meaning for"]')?.value === 'surface-samples';
  }, datasetId);
  assert.equal(await coordinateSystem.inputValue(), 'geographic');
  assert.equal(await measurement.inputValue(), 'amount');
  assert.deepEqual(errors, []);
  process.stdout.write(`${JSON.stringify({passed: true, datasetId, rows: 24,
    checks: ['auto-prepared CSV without GeoJSON', 'metadata before readback', 'publication role precedence',
      'visible caps', 'bounded statistics preserve declarations', 'resident catalog operation', 'sandbox declaration save/replay']}, null, 2)}\n`);
} catch (error) {
  process.stderr.write(`${await page.locator('body').innerText()}\n`);
  throw error;
} finally {
  await browser.close();
}
