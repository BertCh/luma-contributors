// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
const browser = await chromium.launch({headless: true});
try {
  const page = await browser.newPage();
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/scripts/fixtures/analysis-export.html');
  await page.waitForFunction(() => window.analysisExportChecks);
  const result = await page.evaluate(() => window.analysisExportChecks);
  assert.deepEqual(result.identities, [10, 13, 99]);
  assert.equal(result.sourceRowColumn, '__luma_source_row_index_');
  assert.deepEqual(result.score, [0.25, null, 'NaN']);
  assert.deepEqual(result.fields.map(field => field.type), ['Uint32', 'Float32', 'Uint32', 'Int32']);
  assert.equal(result.metadata.recipe.operation.populations, 'population');
  assert.equal(result.metadata.recipe.population, 'all');
  assert.equal(result.metadata.selectionRevision, 7);
  assert.equal(result.metadata.analysisRevision, '4:7');
  assert.equal(result.metadata.offset, 5);
  assert.equal(result.json.rows[1].score, null);
  assert.equal(result.json.rows[2].score, 'NaN');
  assert.equal(result.json.rows[2].__luma_source_row_index_, 99);
  assert.match(result.csv, /13,,2,2/);
  assert.deepEqual(result.csvMetadata, result.metadata);
  assert.equal(result.emptyRows, 0);
  assert.deepEqual(result.combined.sourceValues, [10, 13, 99].map(row => 0.123456789012345 + row));
  assert.deepEqual(result.combined.sourceLabels, ['even', 'odd', 'odd']);
  assert.deepEqual(result.combined.sourceWords, [10n, 13n, 99n].map(row => String(1700000000123456789n + row)));
  assert.deepEqual(result.combined.derivedValues, [0.25, null, 'NaN']);
  assert.deepEqual(result.combined.indices, [10, 13, 99]);
  assert.equal(result.combined.sourceRowColumn, '__luma_source_row_index__');
  assert.equal(result.combined.mapping.score, 'score_analysis_analysis');
  assert.equal(result.combined.metadata.sourceRowColumn, result.combined.sourceRowColumn);
  assert.equal(result.combined.metadata.recipe.operation.events, 'cases');
  assert.deepEqual(result.combined.sourceBatches, [12, 38, 50]);
  assert.deepEqual(result.combined.outputBatches, [1, 1, 1]);
  assert.match(result.combined.csv, /2023-11-14T22:13:20\.123456799Z/);
  assert.equal(result.combined.csvMetadata.derivedColumns.score, 'score_analysis_analysis');
  for (const key of ['invalid', 'oversized', 'stale', 'canceled']) assert.equal(result[key], true, key);
  console.log('Analysis codecs passed: Arrow typed/null/nonfinite values, original IDs, recipe/revisions, JSON, CSV metadata, empty selection, limits and cancellation.');
} finally {await browser.close();}
