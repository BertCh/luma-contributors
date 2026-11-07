// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
const browser = await chromium.launch({headless: true});
const page = await browser.newPage();
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/scripts/fixtures/sandbox-bundle.html');
  await page.waitForFunction(() => window.bundleChecks);
  const result = await page.evaluate(() => window.bundleChecks);
  assert.deepEqual(result.batches, [2, 1]);
  assert.deepEqual(result.fieldNames, ['cost €', 'region', 'time']);
  assert.deepEqual(result.costs, [0.1, null, 30.25]);
  assert.deepEqual(result.categories, ['北', 'East', '北']);
  assert.equal(result.timeType, 'Timestamp<MICROSECOND, UTC>');
  assert.deepEqual(result.timeWords, result.originalTimeWords);
  assert.equal(result.sourceCount, 2);
  for (const key of ['stateMatches', 'corrupt', 'duplicate', 'wrongRevision', 'wrongSchema', 'missingReference', 'stale', 'documentLimit', 'sourceLimit', 'rowLimit', 'countLimit']) assert.equal(result[key], true, key);
  console.log('Portable bundle checks passed: multiple sources, Arrow batch boundaries, Float64/nulls/dictionaries/exact time words, source revision and checksum validation, stale-export rejection and explicit capacities.');
} finally {await browser.close();}
