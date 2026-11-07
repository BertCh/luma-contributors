// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=4096');
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().ready, undefined, {timeout: 60000});
  // Loads the router module from the dev server; it has no network or provider access.
  const results = await page.evaluate(async () => {
    const {routeAssistantOperation} = await import('/src/assistant-operation-router.ts');
    const context = {
      datasets: {obs: {resident: true, rowCount: 4096, population: 'complete', fields: {value: {type: 'real', precision: 'float32'}, group: {type: 'string'}}}},
      layers: {},
      limits: {maximumExportRows: 1000, maximumLoadBytes: 1e6, maximumGroupByAggregations: 4},
      memory: {residentBytes: 0, budgetBytes: 1e6},
      supportedLayerTypes: ['point'],
      supportedAnalyses: [],
      supportedExportFormats: ['csv'],
      supportedViewCount: 1
    };
    const route = operation => routeAssistantOperation({approved: true, operation}, context);
    return {
      unknown: route({type: 'teleport'}),
      extract: route({type: 'extractData'}),
      precision: route({type: 'runAggregation', datasetId: 'obs', groupByField: 'group', aggregations: {value: 'sum'}, requiredPrecision: 'float64'}),
      fullExport: route({type: 'exportData', datasetId: 'obs', format: 'csv', scope: 'all'}),
      split: route({type: 'setMapSplit', mode: 'DUAL_MAP'})
    };
  });
  for (const [name, result] of Object.entries(results)) assert.equal(result.accepted, false, name);
  assert.equal(results.precision.code, 'precision');
  assert.equal(results.fullExport.code, 'export');
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, checks: ['unknown rejected', 'extract rejected', 'precision guard', 'export guard', 'single view']}));
} finally {await browser.close();}
