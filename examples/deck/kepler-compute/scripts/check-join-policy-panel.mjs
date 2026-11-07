// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Headless browser check for the Join policies panel (ANALYSIS4-3). Two prepared datasets share the
// key column "id" and the column name "val"; the right dataset has duplicate keys. The pending
// native Join is configured, the companion panel sets "First match only", a suffix and drop-key,
// and the native Run executes them: output rows and column names must follow the policy. A second
// run with "Last match only" and no suffix checks the other policy values.
// Needs a running dev server; default headless Playwright only.
//
// Usage: node scripts/check-join-policy-panel.mjs [url]
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog, openNativeExport} from './native-navigation.mjs';

const leftTable = new arrow.Table({
  id: arrow.vectorFromArray(['a', 'b', 'c'], new arrow.Utf8()),
  val: arrow.vectorFromArray([1, 2, 3], new arrow.Float32())
});
const rightTable = new arrow.Table({
  id: arrow.vectorFromArray(['a', 'a', 'b', 'd'], new arrow.Utf8()),
  val: arrow.vectorFromArray([10, 11, 20, 40], new arrow.Float32())
});
const leftLabel = 'join-policy-left.arrow';
const rightLabel = 'join-policy-right.arrow';

const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
const errors = [];
let currentStage = 'startup';
const watchdog = setTimeout(() => {
  console.error(`Join policies check exceeded 300 seconds at ${currentStage}`);
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
    /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error|exceeds the maximum/i.test(message.text())
  )
    errors.push(message.text());
});
async function publish(name, table) {
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name,
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(table, 'stream'))
  });
  for (const label of ['Source geometry column', 'Source longitude column', 'Source latitude column'])
    await workspace.getByLabel(label, {exact: true}).selectOption('');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
}
async function selectNativeOption(selector, value) {
  await selector.locator('.item-selector__dropdown').click();
  await page.locator('.list-selector:visible').getByText(value, {exact: true}).click();
}
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
async function getPublicationIds() {
  return page.evaluate(() => window.__keplerComputeTest.getState().publications.map(item => item.id));
}
/** Opens the native Join for the left dataset, applies the policy panel, runs, returns the export. */
async function runJoin(leftId, resultLabel, policy) {
  const knownIds = await getPublicationIds();
  // A previous run leaves its policy notice in the sidebar, which crowds the native panel out.
  const dismiss = page.getByTestId('resident-workflow-notice').getByRole('button', {name: 'Dismiss'});
  if (await dismiss.count()) await dismiss.click();
  await page.locator(`.dataset-ops-menu__toggle[data-for="dataset-ops-${leftId}"]`).click();
  await page.locator('.dataset-ops-menu__join').click();
  const panel = page.locator('.dataset-ops-panel');
  await panel.waitFor();
  if (process.env.JOIN_POLICY_SCREENSHOT) await page.screenshot({path: process.env.JOIN_POLICY_SCREENSHOT.replace('.png', `-${resultLabel.replace(/\W/g, '')}.png`)});
  // Reopening the Join of a dataset that already ran keeps its selections.
  const isConfigured = (await panel.innerText()).includes(rightLabel);
  if (!isConfigured) {
    await selectNativeOption(panel.locator('.data-source-selector').last(), rightLabel);
    await selectNativeOption(panel.locator('.field-selector').nth(0), 'id');
    await selectNativeOption(panel.locator('.field-selector').nth(1), 'id');
    await panel.getByLabel('INNER', {exact: true}).click();
  }
  await panel.locator('input[type=text]').last().fill(resultLabel);
  // The companion panel lists every pending attribute join; the latest one is this join.
  const policyPanel = page.getByTestId('join-policy-panel');
  await policyPanel.waitFor({timeout: 30000});
  const fieldset = policyPanel.locator('fieldset').last();
  await fieldset.getByTestId('join-duplicate-select').selectOption(policy.duplicate);
  const suffixToggle = fieldset.getByTestId('join-suffix-enabled');
  if (policy.suffix === undefined) {
    if (await suffixToggle.isChecked()) await suffixToggle.uncheck();
  } else {
    if (!(await suffixToggle.isChecked())) await suffixToggle.check();
    await fieldset.getByTestId('join-suffix-input').fill(policy.suffix);
    const dropKey = fieldset.getByTestId('join-drop-key');
    if ((await dropKey.isChecked()) !== policy.dropKey) await dropKey.setChecked(policy.dropKey);
  }
  await panel.getByRole('button', {name: 'Run', exact: true}).click();
  await page.waitForFunction(
    ids =>
      window.__keplerComputeTest.getState().publications.some(item => !ids.includes(item.id)),
    knownIds,
    {timeout: 30000}
  );
  const resultId = (await getPublicationIds()).find(id => !knownIds.includes(id));
  return exportTable(resultId);
}
const readRows = (table, names) => [...table].map(row => names.map(name => row[name]));

try {
  await pinViteModules(page);
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {
    waitUntil: 'domcontentloaded'
  });
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {
    timeout: 240000
  });
  stage('publish left and right sources');
  await publish(leftLabel, leftTable);
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications.some(item => item.rowCount === 3),
    undefined,
    {timeout: 60000}
  );
  await publish(rightLabel, rightTable);
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications.some(item => item.rowCount === 4),
    undefined,
    {timeout: 60000}
  );
  await closeNativeDialog(page);
  const publications = await page.evaluate(() => window.__keplerComputeTest.getState().publications);
  const leftId = publications.find(item => item.rowCount === 3).id;

  stage('First match, suffix "_r", drop key');
  const firstTable = await runJoin(leftId, 'Join policy first', {
    duplicate: 'first',
    suffix: '_r',
    dropKey: true
  });
  const firstNames = firstTable.schema.fields.map(field => field.name);
  assert.deepEqual(firstNames.slice(0, 3), ['id', 'val', 'val_r'], 'suffix and drop-key naming');
  assert.deepEqual(
    readRows(firstTable, ['id', 'val', 'val_r']),
    [
      ['a', 1, 10],
      ['b', 2, 20]
    ],
    'first match only, unmatched left row dropped by the INNER join'
  );
  const firstLineage = JSON.parse(firstTable.schema.metadata.get('luma.gl.join'));
  assert.equal(firstLineage.duplicateRightMatches, 'first');
  assert.equal(firstLineage.rightColumnSuffix, '_r');

  stage('Last match, no suffix');
  const lastTable = await runJoin(leftId, 'Join policy last', {
    duplicate: 'last',
    suffix: undefined,
    dropKey: false
  });
  const lastNames = lastTable.schema.fields.map(field => field.name);
  assert.deepEqual(
    lastNames.slice(0, 4),
    ['id', 'val', `${rightLabel}_id`, `${rightLabel}_val`],
    'without a suffix the native right-label prefix naming applies'
  );
  assert.equal(lastTable.numRows, 2, 'one row per matched left key');
  const lastValues = [...lastTable].map(row => row[lastNames[3]]);
  assert.ok(lastValues.includes(11), `last match picks the later duplicate (${lastValues})`);
  assert.ok(!lastValues.includes(10), 'the earlier duplicate is not output');

  assert.deepEqual(errors, [], 'no WGSL or validation errors');
  console.log(
    JSON.stringify({
      passed: true,
      first: {names: firstNames, rows: readRows(firstTable, firstNames.slice(0, 3))},
      last: {names: lastNames, values: lastValues}
    })
  );
} catch (error) {
  if (process.env.JOIN_POLICY_SCREENSHOT) await page.screenshot({path: process.env.JOIN_POLICY_SCREENSHOT}).catch(() => {});
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
