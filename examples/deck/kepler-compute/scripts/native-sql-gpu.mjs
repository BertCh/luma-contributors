// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {closeNativeDialog, openNativeData} from './native-navigation.mjs';
import {EXPECTED_BACKEND_BY_GROUP, SQL_FIXTURE_QUERIES} from './native-sql-fixture-queries.mjs';

/**
 * Routing check for the native SQL Data Explorer (headless). The default dataset has numeric
 * columns only (value float32, category uint32), so this covers CAST, GROUP BY empty-slot removal,
 * and honest refusals; string predicates, aliases, ORDER BY and right-column joins are covered by
 * modules/kepler/test/resident-sql-strings-gpu.spec.ts on a real WebGPU device.
 *
 * The fixture URL `?sqlFixture=strings` also publishes `sql_trips` (2000 rows, string/numeric/epoch
 * columns) and `sql_zones` (50 rows) so every query in native-sql-fixture-queries.mjs runs end to
 * end: one PASS/FAIL line each (expected backend and result row count), non-zero exit on any FAIL.
 * Env: KEPLER_SQL_QUERIES=eq,group:join,ilike selects a subset (names, or group:<name>);
 * KEPLER_SQL_BACKEND_OVERRIDES=name=duckdb,group:join=duckdb flips expected backends;
 * KEPLER_SQL_SKIP_BASELINE=1 skips the numeric-only baseline checks.
 */
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
// Monaco rejects its pending async work with a bare "Canceled" when each run publishes (the data
// modal and its editor unmount); that is editor teardown, not an application error.
page.on('pageerror', error => {
  if (error.message !== 'Canceled') errors.push(error.message);
});
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text()))
    errors.push(message.text());
});

try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256&sqlFixture=strings');
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page, 'SQL');
  const explorer = page.getByRole('region', {name: 'Native SQL Data Explorer'});
  await explorer.waitFor();

  // A successful GPU run publishes its result with addDataToMap, which makes Kepler close the data
  // modal; the explorer (and its status text) is then unmounted. So a GPU result is detected by a new
  // publication, and the explorer is reopened for the next statement. Fallback runs and errors keep
  // the explorer open and are read from its status text.
  async function runSql(sql) {
    if (!(await explorer.count())) {
      await openNativeData(page, 'SQL');
      await explorer.waitFor();
    }
    const knownIds = await page.evaluate(() =>
      window.__keplerComputeTest.getState().publications.map(source => source.id)
    );
    await explorer.locator('.monaco-editor').first().click();
    await page.keyboard.press(process.platform === 'darwin' ? 'Meta+A' : 'Control+A');
    await page.keyboard.insertText(sql);
    await explorer.getByRole('button', {name: 'Run SQL', exact: true}).click();
    // Let React disable the button so the previous result is not mistaken for this one.
    await page.waitForTimeout(150);
    const outcome = await page.waitForFunction(
      known => {
        const published = window.__keplerComputeTest
          .getState()
          .publications.find(source => !known.includes(source.id));
        const root = document.querySelector('[data-native-sql-explorer]');
        if (published && !root) return {kind: 'published', id: published.id, rowCount: published.rowCount};
        const button = [...(root?.querySelectorAll('button') ?? [])].find(item => item.textContent === 'Run SQL');
        if (root && button && !button.disabled) return {kind: 'explorer'};
        return null;
      },
      knownIds,
      {timeout: 120000}
    );
    const result = await outcome.jsonValue();
    if (result.kind === 'published')
      return {backend: 'gpu', text: `published ${result.id}`, rowCount: result.rowCount, failed: false};
    const text = await explorer.innerText();
    const backend = await explorer.locator('[data-sql-backend]').getAttribute('data-sql-backend');
    const rowCountMatch = text.match(backend === 'gpu' ? /(\d+) GPU result rows/ : /(\d+) DuckDB result rows/);
    return {backend, text, rowCount: rowCountMatch ? Number(rowCountMatch[1]) : null, failed: await explorer.getByRole('alert').count() > 0};
  }

  const table = '"resident-observations"';
  if (process.env.KEPLER_SQL_SKIP_BASELINE !== '1') {
  const cast = await runSql(`SELECT category, CAST(value AS INTEGER) AS value_int FROM ${table} WHERE CAST(category AS DOUBLE) < 2`);
  assert.equal(cast.backend, 'gpu', cast.text);
  assert.equal(cast.rowCount, 128, cast.text);

  const grouped = await runSql(`SELECT category, COUNT(*) AS rows FROM ${table} WHERE value > 50 GROUP BY category`);
  assert.equal(grouped.backend, 'gpu', grouped.text);

  const refused = await runSql(`SELECT category FROM ${table} WHERE value = 'abc'`);
  assert.equal(refused.backend, 'duckdb');
  assert.match(refused.text, /String literal outside a supported predicate/);

  // CAST(numeric AS VARCHAR) resolves its labels with the GPU distinct pass (SQL5/SQL6).
  const castVarchar = await runSql(`SELECT CAST(category AS VARCHAR) AS label FROM ${table}`);
  assert.equal(castVarchar.backend, 'gpu', castVarchar.text);
  assert.equal(castVarchar.rowCount, 256, castVarchar.text);

  console.log('PASS baseline: CAST, GROUP BY and the string-literal refusal and CAST to VARCHAR on the numeric default dataset');
  }

  // Fixture queries: string, CAST, BETWEEN, function, join and fallback groups.
  const overrides = Object.fromEntries(
    (process.env.KEPLER_SQL_BACKEND_OVERRIDES ?? '').split(',').filter(Boolean).map(entry => entry.split('='))
  );
  const selectors = (process.env.KEPLER_SQL_QUERIES ?? '').split(',').filter(Boolean);
  const selected = SQL_FIXTURE_QUERIES.filter(
    query => !selectors.length || selectors.some(selector => selector === query.name || selector === `group:${query.group}`)
  );
  assert.ok(selected.length, `No query matches KEPLER_SQL_QUERIES=${selectors.join(',')}`);
  await page.waitForFunction(
    () => ['sql_trips', 'sql_zones'].every(id => window.__keplerComputeTest.getState().publications.some(source => source.id === id)),
    undefined,
    {timeout: 120000}
  );
  // Reopen the explorer so its table list includes the fixture tables.
  await closeNativeDialog(page);
  await openNativeData(page, 'SQL');
  await explorer.waitFor();
  const failures = [];
  for (const query of selected) {
    const expectedBackend = overrides[query.name] ?? overrides[`group:${query.group}`] ?? query.backend;
    let verdict;
    try {
      const result = await runSql(query.sql);
      const problems = [];
      if (result.failed) problems.push(`error shown: ${(result.text.split('DuckDB fallback')[1] ?? result.text.slice(-300)).slice(0, 700).replace(/\n+/g, ' ')}`);
      if (result.backend !== expectedBackend) problems.push(`backend ${result.backend}, expected ${expectedBackend}`);
      if (result.rowCount !== query.rowCount) problems.push(`rows ${result.rowCount}, expected ${query.rowCount}`);
      if (query.fieldTypes && result.backend === 'gpu' && result.text.startsWith('published ')) {
        // The published Kepler fields must carry the SQL type (CAST ... AS TIMESTAMP is a timestamp).
        const published = result.text.slice('published '.length);
        const fieldTypes = await page.evaluate(
          datasetId => window.__keplerComputeTest.getDatasetFieldTypes(datasetId),
          published
        );
        for (const [name, type] of Object.entries(query.fieldTypes))
          if (fieldTypes[name] !== type)
            problems.push(`field ${name} type ${fieldTypes[name]}, expected ${type}`);
      }
      verdict = problems.length ? problems.join('; ') : null;
    } catch (error) {
      verdict = `threw ${error.message.split('\n')[0]}`;
    }
    console.log(`${verdict ? 'FAIL' : 'PASS'} [${query.group}] ${query.name} (${expectedBackend}, ${query.rowCount} rows)${verdict ? ` - ${verdict}` : ''}`);
    if (verdict) failures.push(query.name);
  }
  void EXPECTED_BACKEND_BY_GROUP;
  await closeNativeDialog(page);
  assert.deepEqual(errors, []);
  if (failures.length) {
    process.exitCode = 1;
    console.error(`Native SQL fixture: ${failures.length}/${selected.length} FAILED: ${failures.join(', ')}`);
  } else console.log(`Native SQL explorer: ${selected.length} fixture queries passed.`);
} catch (error) {
  console.error((await page.locator('body').innerText()).slice(-4000));
  throw error;
} finally {
  await browser.close();
}
