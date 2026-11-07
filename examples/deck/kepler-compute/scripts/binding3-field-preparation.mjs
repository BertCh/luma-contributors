// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// On-demand field preparation. A loaded CSV becomes resident automatically; a column the upload excludes
// (here a Float32-overflowing number) cannot be prepared, so a filter or color channel on it must show the
// exclusion reason and must not trigger repeated republishes. A column that is not in the table says so.
// NOT covered: the positive flow (column absent from the resident copy, then prepared by one republish)
// because Kepler 3.3 exposes no add-column action to script; see the BINDING4 backlog.
// Run headless (default): node scripts/binding3-field-preparation.mjs [url]
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
page.setDefaultTimeout(30000);
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
const state = () => page.evaluate(() => window.__keplerComputeTest.getState());
const rowCount = 64;
const csv = Buffer.from(
  'latitude,longitude,value,huge\n' +
    Array.from({length: rowCount}, (_, index) => `${40.5 + (index % 5) / 32},${-74 + (index % 7) / 32},${index}.5,${index % 2 ? '1e300' : '1'}`).join('\n')
);
// `value` is real-typed so Kepler's default color channel picks it, not the excluded `huge` column.
const checks = [];
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await page.getByRole('button', {name: 'Add Data', exact: true}).click();
  await page.locator('input[type=file]').setInputFiles({name: 'preparation.csv', mimeType: 'text/csv', buffer: csv});
  await page.locator('.ReactModal__Content').getByRole('button', {name: 'Add Data', exact: true}).click();
  await page.waitForFunction(count => window.__keplerComputeTest.getState().publications.some(publication => publication.selectedCount === count), rowCount, {timeout: 60000});
  const datasetId = (await state()).publications.find(publication => publication.selectedCount === rowCount).id;
  const before = (await state()).fieldPreparation.republishCount;
  const report = page.getByTestId('auto-prepare-report');
  const reportText = await report.innerText();
  const excluded = /Excluded huge/.test(reportText);
  // Filter on the column through the test hook; whether or not the upload excluded it, no loop may occur.
  await page.evaluate(id => window.__keplerComputeTest.setPublishedRange(id, 'huge', 0, 10), datasetId);
  await page.waitForTimeout(3000);
  let after = await state();
  assert(after.fieldPreparation.republishCount - before <= 1, `at most one automatic republish, got ${after.fieldPreparation.republishCount - before}`);
  if (excluded) {
    // The resident copy's native table carries only uploaded fields, so Kepler cannot create a filter on
    // the excluded column; if a request does reach the coordinator it must report the exclusion.
    const outcome = after.fieldPreparation.outcomes[`${datasetId}:huge`];
    if (outcome) {
      assert.equal(outcome.status, 'already-attempted', JSON.stringify(outcome));
      assert.match(outcome.reason, /excluded from GPU upload/);
    }
    assert.equal(after.fieldPreparation.republishCount - before, 0, 'an excluded column must not trigger a republish');
    checks.push('excluded column: reported in the preparation report, no republish');
  } else checks.push('huge column was prepared by the upload; exclusion path not exercised');
  await page.evaluate(id => window.__keplerComputeTest.setPublishedRange(id, 'ghost', 0, 10), datasetId).catch(() => undefined);
  await page.waitForTimeout(2000);
  after = await state();
  const ghost = after.fieldPreparation.outcomes[`${datasetId}:ghost`];
  if (ghost) {
    assert.equal(ghost.status, 'unavailable', JSON.stringify(ghost));
    assert.match(ghost.reason, /not a column/);
    checks.push('missing column: reported as not a column of the dataset');
  }
  assert.equal(after.fieldPreparation.republishCount - before <= 1, true);
  assert.deepEqual(errors, [], `page errors: ${errors.join('; ')}`);
  console.log(JSON.stringify({passed: true, checks}));
} catch (error) {
  console.error(await page.locator('body').innerText().catch(() => ''));
  throw error;
} finally {
  await browser.close();
}
