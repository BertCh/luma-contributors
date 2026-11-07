// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
const browser = await chromium.launch({headless: true});
const page = await browser.newPage();
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/scripts/fixtures/source-export.html');
  await page.waitForFunction(() => window.sourceExportChecks);
  const result = await page.evaluate(() => window.sourceExportChecks);
  assert.deepEqual(result.separateDictionaryValues, ['south', 'west']);
  assert.deepEqual(result.amount, [0.123456789012345, null, 4.25]);
  assert.deepEqual(result.labels, ['comma, quote"', 'line\nbreak', '北']);
  assert.deepEqual(result.words, ['-1', '1700000000123456789', '1700000000987654321']);
  assert.deepEqual(result.geometry.map(value => ({type: value.type, coordinates: Array.from(value.coordinates)})), [1, 2, 4].map(index => ({type: 'Point', coordinates: [-74 + index, 40]})));
  assert.deepEqual(result.indices, [1, 2, 4]);
  assert.deepEqual(result.originalIndexColumn, [11, 12, 14]);
  assert.equal(result.sourceRowColumn, '__luma_source_row_index_');
  assert.deepEqual(result.batches, [2, 1]);
  assert.equal(result.metadata, 'preserve source metadata');
  assert.equal(result.exportMetadata.selectionRevision, 4);
  assert.match(result.csv, /1969-12-31T23:59:59\.999999999Z/);
  assert.match(result.csv, /2023-11-14T22:13:20\.123456789Z/);
  assert.match(result.csv, /"comma, quote"""/);
  assert.equal(result.empty.rows, 0);
  assert.equal(result.empty.fields.length, 6);
  for (const key of ['oversized', 'invalid', 'stale', 'canceled']) assert.equal(result[key], true, key);
  console.log('Selected-source codec passed: original Float64/nulls/dictionaries/geometry/nanosecond values, identity collisions, metadata, empty selection, CSV escaping, byte capacity and cancellation.');
} finally {await browser.close();}
