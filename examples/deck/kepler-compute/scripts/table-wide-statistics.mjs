// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// NATIVEUI6-T3 browser check (headless WebGPU): resident data table column statistics and sorting
// for Int64, timestamp, Float64 and dictionary-string columns. A 100,000-row dataset uses the bounded
// CPU statistics (at most 262,144 rows); a 300,000-row dataset takes the GPU profile (Float32 copy
// relative to the column minimum; dictionary codes). Both are compared with closed-form figures of
// the generated columns, and the exact 64-bit GPU sort of the larger table is compared row by row.
// Not covered: the GPU-only prepared-Float32 sort fallback (needs a dataset without retained Arrow
// words). Usage: node table-wide-statistics.mjs [url]
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {getFixtureUrl} from './script-fixture-url.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const INT64_BASE = 10_000_000_000;
const TIMESTAMP_BASE = 1_600_000_000_000;
const MULTIPLIER = 7919;
const TIME_MULTIPLIER = 104729;
const LABEL_COUNT = 1000;
const NULL_PERIOD = 1000;

/** Columns whose statistics have closed forms: permutations of 0..N-1 for Int64 and timestamp. */
function makeTable(rowCount) {
  const permutation = new Array(rowCount);
  const times = new Array(rowCount);
  const doubles = new Array(rowCount);
  const labels = new Array(rowCount);
  const longitude = new Float32Array(rowCount);
  const latitude = new Float32Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    permutation[row] = BigInt(INT64_BASE + ((row * MULTIPLIER) % rowCount));
    times[row] = TIMESTAMP_BASE + ((row * TIME_MULTIPLIER) % rowCount) * 1000;
    doubles[row] = row % NULL_PERIOD === 0 ? null : Math.sin(row * 0.001) * 500 + row * 1e-4;
    labels[row] = `label-${row % LABEL_COUNT}`;
    longitude[row] = -74.1 + (row % 500) / 2500;
    latitude[row] = 40.6 + Math.floor(row / 500) / 5000;
  }
  const expectedDoubles = doubles.filter(value => value !== null);
  const mean = expectedDoubles.reduce((sum, value) => sum + value, 0) / expectedDoubles.length;
  const variance =
    expectedDoubles.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    (expectedDoubles.length - 1);
  // The pre-port oracle spread the 300,000-row fixture into Math.min/max, exceeding Node's
  // argument limit before the GPU route could run. Reduce the same full population instead.
  const minimum = expectedDoubles.reduce((result, value) => Math.min(result, value), Infinity);
  const maximum = expectedDoubles.reduce((result, value) => Math.max(result, value), -Infinity);
  const table = new arrow.Table({
    longitude: arrow.makeVector(longitude),
    latitude: arrow.makeVector(latitude),
    i64: arrow.vectorFromArray(Array.from(permutation), new arrow.Int64()),
    stamp: arrow.vectorFromArray(times, new arrow.TimestampMillisecond()),
    f64: arrow.vectorFromArray(doubles, new arrow.Float64()),
    kind: arrow.vectorFromArray(labels)
  });
  return {
    table,
    expected: {
      i64: {
        mean: INT64_BASE + (rowCount - 1) / 2,
        std: Math.sqrt((rowCount * (rowCount + 1)) / 12),
        range: rowCount - 1
      },
      stamp: {
        min: TIMESTAMP_BASE,
        max: TIMESTAMP_BASE + (rowCount - 1) * 1000,
        range: (rowCount - 1) * 1000
      },
      f64: {
        mean,
        std: Math.sqrt(variance),
        nulls: expectedDoubles.length === rowCount ? 0 : rowCount - expectedDoubles.length,
        range: maximum - minimum
      },
      kind: {unique: LABEL_COUNT}
    }
  };
}

/** Modular inverse of MULTIPLIER (mod rowCount), so the row holding the k-th smallest Int64 is k * inverse. */
function getInverse(rowCount) {
  let [previous, current] = [BigInt(rowCount), BigInt(MULTIPLIER)];
  let [previousCoefficient, coefficient] = [0n, 1n];
  while (current) {
    const quotient = previous / current;
    [previous, current] = [current, previous - quotient * current];
    [previousCoefficient, coefficient] = [
      coefficient,
      previousCoefficient - quotient * coefficient
    ];
  }
  return Number(((previousCoefficient % BigInt(rowCount)) + BigInt(rowCount)) % BigInt(rowCount));
}

const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
page.setDefaultTimeout(90000);
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => {
  if (
    message.type() === 'error' &&
    /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())
  )
    errors.push(message.text());
});
const watchdog = setTimeout(() => {
  console.error('table-wide-statistics exceeded 480 seconds');
  process.exit(1);
}, 480000);

async function publish(rowCount) {
  const {table, expected} = makeTable(rowCount);
  const fileName = `wide-${rowCount}.arrow`;
  const before = await page.evaluate(
    () => window.__keplerComputeTest.getState().publications.length
  );
  log(`publish ${rowCount}`);
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  // The pre-port file loader also converted Int64 to decimal labels before the preview, so this
  // check could not reach its numeric statistics/sort paths. Explicit IPC preservation retains
  // the Int64 source words; the measured precision choice below controls only the GPU copy.
  await workspace.getByLabel('Preserve original Arrow types', {exact: true}).check();
  await workspace.locator('input[type=file]').setInputFiles({
    name: fileName,
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(table, 'stream'))
  });
  await workspace.getByLabel('Source longitude column').selectOption('longitude');
  await workspace.getByLabel('Source latitude column').selectOption('latitude');
  // SourcePreviewPanel selects every scalar field by default, including both wide numeric fields,
  // time and dictionary strings. Fail here if a fixture field was lost or excluded before upload.
  const fields = workspace.getByRole('group', {name: 'Fields to upload', exact: true});
  for (const name of ['longitude', 'latitude', 'i64', 'stamp', 'f64', 'kind']) {
    assert.equal(
      await fields.getByRole('checkbox', {name: new RegExp(`^${name} \\(`)}).isChecked(),
      true,
      `${rowCount}: ${name} must be selected for upload`
    );
  }
  // Float64 needs an explicit precision policy before the schema check accepts the column.
  await workspace
    .getByLabel('Float64 conversion')
    .selectOption({label: 'Allow a measured rounding error'});
  await workspace.getByLabel('Maximum absolute error').fill('0.001');
  await workspace.getByLabel('Maximum relative error').fill('0.000001');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(
    ({count, rowCount}) =>
      window.__keplerComputeTest.getState().publications[count]?.selectedCount === rowCount,
    {count: before, rowCount},
    {timeout: 180000}
  );
  await closeNativeDialog(page);
  // loadLocalSource uses file.name as the candidate/native dataset label. The dev-test publication
  // record exposes id/counts, not label; do not derive the rendered button name from that record.
  return {expected, datasetLabel: fileName};
}

async function readStatistics(column) {
  console.log(`statistics ${column}`);
  const tools = page.locator('.resident-table-column-tools');
  if (!(await tools.evaluate(element => element.open))) await tools.locator('summary').click();
  const selector = tools.getByLabel('Column', {exact: true});
  const available = await selector
    .locator('option')
    .evaluateAll(options => options.map(option => option.value));
  assert(
    available.includes(column),
    `Statistics column ${column} is missing; available: ${available.join(', ')}`
  );
  await selector.selectOption(column);
  await tools.getByRole('button', {name: 'Column statistics', exact: true}).click();
  await page.locator('.resident-table-statistics').waitFor({timeout: 180000});
  await page.waitForFunction(
    name =>
      document.querySelector('.resident-table-statistics caption')?.textContent?.startsWith(name),
    column
  );
  return page
    .locator('.resident-table-statistics tbody tr')
    .evaluateAll(rows =>
      Object.fromEntries(
        rows.map(row => [row.querySelector('th').textContent, row.querySelector('td').textContent])
      )
    );
}
const toNumber = text => Number(text.replace(/,/g, ''));
/** Kepler's quantile rows keep the label and a formatted value; the median is the 50% entry. */
const readMedian = statistics => toNumber(statistics['Median (50%)']);

const results = [];
const startedAt = Date.now();
const log = stage => console.log(`[${Math.round((Date.now() - startedAt) / 1000)}s] ${stage}`);
try {
  await page.goto(getFixtureUrl(process.argv[2], {rows: '256'}), {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(
    () => window.__keplerComputeTest?.getState().selectedCount === 256,
    undefined,
    {timeout: 60000}
  );
  for (const rowCount of [100000, 300000]) {
    const gpu = rowCount > 262144;
    const {expected, datasetLabel} = await publish(rowCount);
    log(`published ${rowCount}; open table`);
    await page.locator('.show-data-table svg').last().click();
    const datasetButton = page
      .getByRole('navigation', {name: 'Table datasets', exact: true})
      .getByRole('button', {name: datasetLabel, exact: true});
    // The modal renders aria-pressed from its active dataset ID; publishing commonly selects it.
    if ((await datasetButton.getAttribute('aria-pressed')) !== 'true') await datasetButton.click();
    // The pre-port table also defaults to prepared scalar columns, which omit exact time words.
    // This checks retained source statistics and Int64 ordering, so select the original view explicitly.
    await page.getByLabel('Values', {exact: true}).selectOption('original');
    await page.locator('.resident-table-scroll tbody tr').first().waitFor();
    await page.getByLabel('Rows', {exact: true}).selectOption('all');
    // Int64 (a permutation of 0..N-1 plus 1e10): exact mean, closed-form deviation, median near the middle.
    const integer = await readStatistics('i64');
    assert.match(
      integer['Computed on'] ?? '',
      gpu ? /GPU profile/ : /CPU/,
      `${rowCount}: i64 route`
    );
    assert.equal(toNumber(integer['Rows']), rowCount);
    assert(
      Math.abs(toNumber(integer['Mean']) - expected.i64.mean) <= expected.i64.range * 1e-4 + 1,
      `${rowCount}: i64 mean ${integer['Mean']} vs ${expected.i64.mean}`
    );
    assert(
      Math.abs(toNumber(integer['Std. dev.']) - expected.i64.std) <= expected.i64.std * 1e-3,
      `${rowCount}: i64 deviation ${integer['Std. dev.']} vs ${expected.i64.std}`
    );
    assert(
      Math.abs(readMedian(integer) - expected.i64.mean) <= expected.i64.range * 0.02,
      `${rowCount}: i64 median ${integer['Median (50%)']}`
    );
    // Timestamp: minimum and maximum come back as ISO instants within the Float32 resolution of the range.
    const stamp = await readStatistics('stamp');
    assert(
      Math.abs(Date.parse(stamp['Minimum']) - expected.stamp.min) <=
        Math.max(1, expected.stamp.range * 1e-6),
      `${rowCount}: stamp minimum ${stamp['Minimum']}`
    );
    assert(
      Math.abs(Date.parse(stamp['Maximum']) - expected.stamp.max) <=
        Math.max(1, expected.stamp.range * 1e-6) + 1000,
      `${rowCount}: stamp maximum ${stamp['Maximum']}`
    );
    // Float64 with a null every 1000th row: null count exact, mean and deviation to Float32 range resolution.
    const double = await readStatistics('f64');
    assert.match(
      double['Nulls'],
      new RegExp(`^${expected.f64.nulls.toLocaleString('en-US')} `),
      `${rowCount}: f64 nulls ${double['Nulls']}`
    );
    assert(
      Math.abs(toNumber(double['Mean']) - expected.f64.mean) <= expected.f64.range * 1e-4,
      `${rowCount}: f64 mean ${double['Mean']} vs ${expected.f64.mean}`
    );
    assert(
      Math.abs(toNumber(double['Std. dev.']) - expected.f64.std) <= expected.f64.std * 1e-3,
      `${rowCount}: f64 deviation ${double['Std. dev.']} vs ${expected.f64.std}`
    );
    // Dictionary strings: distinct estimate within the profile's stated error (about 1.6%), CPU is exact.
    const dictionary = await readStatistics('kind');
    const unique = toNumber(dictionary['Unique values']);
    assert(
      Math.abs(unique - expected.kind.unique) <= expected.kind.unique * (gpu ? 0.05 : 0),
      `${rowCount}: unique labels ${unique}`
    );
    results.push({
      rowCount,
      route: integer['Computed on'],
      mean: integer['Mean'],
      stampMinimum: stamp['Minimum'],
      stampMaximum: stamp['Maximum'],
      nulls: double['Nulls'],
      unique
    });
    console.log(JSON.stringify(results.at(-1)));
    if (gpu) {
      // Exact 64-bit GPU sort: the first page of the Int64 column is the 50 smallest values, then the 50 largest.
      const inverse = getInverse(rowCount);
      const rowIds = () =>
        page
          .locator('.resident-table-scroll tbody tr th')
          .allTextContents()
          .then(values => values.map(Number));
      const waitRows = async expectedRows => {
        await page.waitForFunction(
          expectedRows =>
            JSON.stringify(
              Array.from(document.querySelectorAll('.resident-table-scroll tbody tr th'), cell =>
                Number(cell.textContent)
              )
            ) === JSON.stringify(expectedRows),
          expectedRows,
          {timeout: 120000}
        );
        assert.deepEqual(await rowIds(), expectedRows);
      };
      // Rows selects the population in the pre-port UI too; page size has its own control.
      await page.getByLabel('Rows per page', {exact: true}).selectOption('50');
      await page
        .locator('.resident-table-scroll thead')
        .getByRole('button', {name: 'i64 (Int64, original)', exact: true})
        .click();
      await waitRows(
        Array.from({length: 50}, (_, rank) =>
          Number((BigInt(rank) * BigInt(inverse)) % BigInt(rowCount))
        )
      );
      await page
        .locator('.resident-table-scroll thead')
        .getByRole('button', {name: 'i64 (Int64, original) ↑', exact: true})
        .click();
      await waitRows(
        Array.from({length: 50}, (_, rank) =>
          Number((BigInt(rowCount - 1 - rank) * BigInt(inverse)) % BigInt(rowCount))
        )
      );
      results.at(-1).sorted = 'int64 ascending and descending first pages equal the closed form';
    }
    await page.keyboard.press('Escape');
    await page.locator('.resident-table-modal').waitFor({state: 'hidden'});
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, results}));
} catch (error) {
  console.error('Completed stages:', JSON.stringify(results));
  console.error('Page errors:', errors);
  throw error;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
