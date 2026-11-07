// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Headless browser check for the Suitability missing-value panel (ANALYSIS4-8). A prepared table has
// rows missing in only some weighted fields. The panel's "Exclude" policy then "Renormalize" policy
// is set on the pending native Suitability operation and the native Run executes it: the workflow
// notice reports the partial row count, partial rows score null (exclude) or over their defined
// fields (renormalize), and the saved result lineage carries the policy.
// Needs a running dev server; default headless Playwright only.
//
// Usage: node scripts/check-suitability-missing-panel.mjs [url]
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog, openNativeExport} from './native-navigation.mjs';

// Rows b and c each miss one weighted field (partial), row a and d are complete.
const cost = [10, 20, null, 40];
const access = [3, null, 7, 5];
const sourceTable = new arrow.Table({
  name: arrow.vectorFromArray(['a', 'b', 'c', 'd'], new arrow.Utf8()),
  cost: arrow.vectorFromArray(cost, new arrow.Float32()),
  access: arrow.vectorFromArray(access, new arrow.Float32())
});
const sourceLabel = 'suitability-missing-source.arrow';
// Range standardization over defined values, normalized equal weights.
const standardizeRange = values => {
  const defined = values.filter(value => value !== null);
  const minimum = Math.min(...defined);
  const span = Math.max(...defined) - minimum;
  return values.map(value => (value === null ? null : (value - minimum) / span));
};
const standardizedCost = standardizeRange(cost);
const standardizedAccess = standardizeRange(access);
const expectedExclude = cost.map((_, row) =>
  standardizedCost[row] === null || standardizedAccess[row] === null
    ? null
    : (standardizedCost[row] + standardizedAccess[row]) / 2
);
const expectedRenormalize = cost.map((_, row) => {
  const defined = [standardizedCost[row], standardizedAccess[row]].filter(value => value !== null);
  return defined.reduce((sum, value) => sum + value, 0) / defined.length;
});

const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
const errors = [];
let currentStage = 'startup';
const watchdog = setTimeout(() => {
  console.error(`Suitability missing-value check exceeded 300 seconds at ${currentStage}`);
  void browser.close();
}, 300000);
const stage = name => {
  currentStage = name;
  console.log(JSON.stringify({stage: name}));
};
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => {
  if (
    message.type() === 'error' &&
    /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())
  )
    errors.push(message.text());
});
async function exportTable(datasetId) {
  await openNativeExport(page);
  await page.getByLabel('Export dataset', {exact: true}).selectOption(datasetId);
  await page.getByLabel('Export format', {exact: true}).selectOption('arrow');
  const download = page.waitForEvent('download');
  await page.getByRole('button', {name: 'Export all selected', exact: true}).click();
  const chunks = [];
  for await (const chunk of await (await download).createReadStream()) chunks.push(chunk);
  await closeNativeDialog(page);
  return arrow.tableFromIPC(Buffer.concat(chunks));
}
const getPublicationIds = () =>
  page.evaluate(() => window.__keplerComputeTest.getState().publications.map(item => item.id));
async function runSuitability(sourceId, resultLabel, policy) {
  const knownIds = await getPublicationIds();
  const dismiss = page.getByTestId('resident-workflow-notice').getByRole('button', {name: 'Dismiss'});
  if (await dismiss.count()) await dismiss.click();
  await page.locator(`.dataset-ops-menu__toggle[data-for="dataset-ops-${sourceId}"]`).click();
  await page.locator('.dataset-ops-menu__suitability').click();
  const panel = page.locator('.dataset-ops-panel');
  await panel.waitFor();
  await panel.locator('input[type=text]').last().fill(resultLabel);
  const policyPanel = page.getByTestId('suitability-policy-panel');
  await policyPanel.waitFor({timeout: 30000});
  // The panel lists every pending operation; the newest is last.
  await policyPanel.locator('select[data-testid^="suitability-missing-"]').last().selectOption(policy);
  await panel.getByRole('button', {name: 'Run', exact: true}).click();
  await page.waitForFunction(
    ids => window.__keplerComputeTest.getState().publications.some(item => !ids.includes(item.id)),
    knownIds,
    {timeout: 30000}
  );
  const resultId = (await getPublicationIds()).find(id => !knownIds.includes(id));
  const notice = await page
    .getByTestId('resident-workflow-notice')
    .innerText({timeout: 30000});
  return {resultId, notice, table: await exportTable(resultId)};
}
function assertScores(table, expected, description) {
  const scores = [...table].map(row => row.score);
  assert.equal(scores.length, expected.length, `${description}: row count`);
  for (const [row, value] of expected.entries()) {
    if (value === null) assert.equal(scores[row], null, `${description}: row ${row} scores null`);
    else
      assert.ok(
        scores[row] !== null && Math.abs(scores[row] - value) < 1e-5,
        `${description}: row ${row} ${scores[row]} vs ${value}`
      );
  }
}

try {
  await pinViteModules(page);
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {
    waitUntil: 'domcontentloaded'
  });
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {
    timeout: 240000
  });
  stage('publish numeric source with partial nulls');
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: sourceLabel,
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(sourceTable, 'stream'))
  });
  for (const label of ['Source geometry column', 'Source longitude column', 'Source latitude column'])
    await workspace.getByLabel(label, {exact: true}).selectOption('');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications.some(item => item.rowCount === 4),
    undefined,
    {timeout: 60000}
  );
  const sourceId = await page.evaluate(
    () => window.__keplerComputeTest.getState().publications.find(item => item.rowCount === 4).id
  );
  await closeNativeDialog(page);

  stage('Exclude policy');
  const excluded = await runSuitability(sourceId, 'Suitability exclude', 'exclude');
  assert.match(excluded.notice, /2 rows were missing in some weighted fields and were scored null/);
  assertScores(excluded.table, expectedExclude, 'exclude');
  const excludeLineage = JSON.parse(excluded.table.schema.metadata.get('luma.gl.suitability'));
  assert.equal(excludeLineage.missingValues, 'exclude', 'result lineage carries the policy');

  stage('Renormalize policy');
  const renormalized = await runSuitability(sourceId, 'Suitability renormalize', 'renormalize');
  assert.match(
    renormalized.notice,
    /2 rows were missing in some weighted fields and were scored over their defined fields with renormalized weights/
  );
  assertScores(renormalized.table, expectedRenormalize, 'renormalize');
  const renormalizeLineage = JSON.parse(renormalized.table.schema.metadata.get('luma.gl.suitability'));
  assert.equal(renormalizeLineage.missingValues, 'renormalize');

  assert.deepEqual(errors, [], 'no WGSL or validation errors');
  console.log(
    JSON.stringify({
      passed: true,
      exclude: [...excluded.table].map(row => row.score),
      renormalize: [...renormalized.table].map(row => row.score)
    })
  );
} catch (error) {
  if (process.env.SUITABILITY_SCREENSHOT)
    await page.screenshot({path: process.env.SUITABILITY_SCREENSHOT}).catch(() => {});
  console.error(
    JSON.stringify({
      stage: currentStage,
      errors,
      body: (await page.locator('body').innerText().catch(() => '')).slice(-3000)
    })
  );
  throw error;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
