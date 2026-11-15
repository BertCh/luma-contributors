// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors


import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getSourceCheckUrl} from './source-check-utils.mjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => {
  if (/unresolved value|Compilation error|Error while parsing WGSL/.test(message.text()))
    errors.push(message.text());
});
const rowCount = 120;
const indices = Array.from({length: rowCount}, (_, index) => index);
const table = arrow.tableFromArrays({
  latitude: arrow.vectorFromArray(
    indices.map(index => 40.5 + (index % 5) / 32),
    new arrow.Float64()
  ),
  longitude: arrow.vectorFromArray(
    indices.map(index => -74 + (index % 7) / 32),
    new arrow.Float64()
  ),
  identity: arrow.vectorFromArray(
    indices.map(index => 2n ** 60n + BigInt(index)),
    new arrow.Int64()
  ),
  amount: arrow.makeVector({
    type: new arrow.Decimal(2, 12, 128),
    data: Uint32Array.from(indices.flatMap(index => [index * 100 + 5, 0, 0, 0])),
    length: rowCount
  }),
  half: arrow.vectorFromArray(
    indices.map(index => index / 4),
    new arrow.Float16()
  ),
  day: arrow.vectorFromArray(
    indices.map(index => new Date(86400000 * (19000 + index))),
    new arrow.DateDay()
  ),
  flag: arrow.vectorFromArray(
    indices.map(index => (index % 3 ? index % 2 === 0 : null)),
    new arrow.Bool()
  ),
  tags: arrow.vectorFromArray(indices.map(index => [index, index + 1])),
  shape: arrow.vectorFromArray(
    indices.map(index => `POINT (${index} ${index})`),
    new arrow.Utf8()
  )
});
const bytes = Buffer.from(arrow.tableToIPC(table, 'file'));

try {
  await page.goto(getSourceCheckUrl(process.argv[2], {rows: 100000}), {
    waitUntil: 'domcontentloaded'
  });
  await page.waitForFunction(
    () => {
      const state = window.__keplerComputeTest?.getState();
      if (state?.error) throw new Error(state.error);
      return state?.selectedCount === 100000;
    },
    undefined,
    {timeout: 60000}
  );
  await page.getByRole('button', {name: 'Add Data', exact: true}).click();
  await page.locator('.ReactModal__Content input[type=file]').setInputFiles({
    name: 'typed-observations.arrow',
    mimeType: 'application/octet-stream',
    buffer: bytes
  });
  await page
    .locator('.ReactModal__Content')
    .getByRole('button', {name: 'Add Data', exact: true})
    .click();
  await page.waitForFunction(
    count => {
      const state = window.__keplerComputeTest.getState();
      if (state.error) throw new Error(state.error);
      return state.publications.some(publication => publication.selectedCount === count);
    },
    rowCount,
    {timeout: 60000}
  );
  const report = page.getByTestId('auto-prepare-report');
  await report.waitFor();
  // Prepared datasets collapse to their summary line; expand them to read the full report.
  await report.evaluate(element => {
    for (const details of element.querySelectorAll('details')) details.open = true;
  });
  const reportText = await report.innerText();
  assert.match(reportText, /typed-observations/);
  assert.match(reportText, /Type route identity \(Int64\): converted/);
  assert.match(reportText, /Type route tags .*retained-original/);
  assert.match(reportText, /Excluded from GPU upload[\s\S]*tags/);
  const state = await page.evaluate(() => window.__keplerComputeTest.getState());
  const published = state.publications.find(publication => publication.selectedCount === rowCount);
  assert.equal(published.error, null);
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({
      passed: true,
      rowCount,
      reportText,
      checks: [
        'typed Arrow IPC via native Add Data',
        'visible type routes',
        'excluded nested column',
        'no errors'
      ]
    })
  );
} catch (error) {
  console.error(await page.locator('body').innerText());
  throw error;
} finally {
  await browser.close();
}
