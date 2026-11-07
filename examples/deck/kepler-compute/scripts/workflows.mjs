// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const url = process.argv[2] || 'http://127.0.0.1:5318/?rows=4096';
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const errors = [];
page.on('pageerror', error => errors.push(error.message));

try {
  await page.goto(url, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => {
    const state = window.__keplerComputeTest?.getState();
    if (state?.error) throw new Error(state.error);
    return state?.ready && state.selectedCount === window.__keplerComputeTest.source().length;
  }, undefined, {timeout: 60000});

  const expected = await page.evaluate(() => {
    const api = window.__keplerComputeTest;
    // Native polygon selection with a hole, expressed in source geographic coordinates.
    const outer = [[-74.06, 40.68], [-73.95, 40.68], [-73.95, 40.82], [-74.06, 40.82], [-74.06, 40.68]];
    const hole = [[-74.03, 40.71], [-74.01, 40.71], [-74.01, 40.73], [-74.03, 40.73], [-74.03, 40.71]];
    const inside = (row, minimumX, minimumY, maximumX, maximumY) =>
      row[0] > minimumX && row[0] < maximumX && row[1] > minimumY && row[1] < maximumY;
    const indices = [];
    const values = [];
    api.source().forEach((row, index) => {
      if (inside(row, -74.06, 40.68, -73.95, 40.82) && !inside(row, -74.03, 40.71, -74.01, 40.73)) {
        indices.push(index);
        values.push(row[2]);
      }
    });
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const deviation = Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1));
    api.addPolygon([outer, hole]);
    return {
      count: indices.length,
      firstRows: indices.slice(0, 10),
      firstStandardized: (values[0] - mean) / deviation,
      sourceCount: api.source().length
    };
  });
  await page.waitForFunction(count => {
    const state = window.__keplerComputeTest.getState();
    if (state.error) throw new Error(state.error);
    return state.selectedCount === count && !state.unsupportedSelection;
  }, expected.count, {timeout: 30000});
  const sourceRows = await page.evaluate(async () => {
    const result = await window.__keplerComputeTest.readPage(0, 10);
    return Array.from(result.sourceRowIndices);
  });
  assert.deepEqual(sourceRows, expected.firstRows);

  const polygonConfiguration = await page.evaluate(() => {
    const api = window.__keplerComputeTest;
    const configuration = api.saveNativeConfig();
    for (const filter of configuration.config.visState.filters) {
      if (filter.type === 'polygon') filter.enabled = false;
    }
    api.applyNativeConfig(configuration);
    return configuration;
  });
  await page.waitForFunction(count => window.__keplerComputeTest.getState().selectedCount === count,
    expected.sourceCount, {timeout: 30000});
  await page.evaluate(configuration => {
    for (const filter of configuration.config.visState.filters) {
      if (filter.type === 'polygon') filter.enabled = true;
    }
    window.__keplerComputeTest.applyNativeConfig(configuration);
  }, polygonConfiguration);
  await page.waitForFunction(count => window.__keplerComputeTest.getState().selectedCount === count,
    expected.count, {timeout: 30000});

  await page.evaluate(() => {
    const analysis = window.__keplerComputeTest.analysis();
    analysis.setPopulation('selected');
    analysis.setMode('z-score');
  });
  await page.waitForFunction(count => {
    const analysis = window.__keplerComputeTest.analysis();
    if (analysis.error) throw new Error(analysis.error);
    return analysis.summary?.operation === 'z-score' && analysis.summary.count === count;
  }, expected.count, {timeout: 30000});
  const standardized = await page.evaluate(async sourceRow => {
    const analysis = window.__keplerComputeTest.analysis();
    return {row: await analysis.readRow(sourceRow), recipe: analysis.serializeRecipe()};
  }, expected.firstRows[0]);
  assert.equal(standardized.row.sourceRowIndex, expected.firstRows[0]);
  assert(Math.abs(standardized.row.value - expected.firstStandardized) < 0.0001);

  await page.evaluate(() => {
    const analysis = window.__keplerComputeTest.analysis();
    analysis.setClassCount(4);
    analysis.setMode('quantile');
  });
  await page.waitForFunction(() => {
    const analysis = window.__keplerComputeTest.analysis();
    if (analysis.error) throw new Error(analysis.error);
    return analysis.summary?.operation === 'quantile';
  }, undefined, {timeout: 30000});
  const classification = await page.evaluate(() => window.__keplerComputeTest.analysis().summary);
  assert.equal(classification.classCounts.reduce((sum, count) => sum + count, 0), expected.count);
  assert.equal(classification.breaks.length, 5);
  assert(classification.breaks.every((value, index, values) => index === 0 || value >= values[index - 1]));

  await page.evaluate(recipe => window.__keplerComputeTest.analysis().restoreRecipe(recipe), standardized.recipe);
  await page.waitForFunction(() => window.__keplerComputeTest.analysis().summary?.operation === 'z-score',
    undefined, {timeout: 30000});
  await page.evaluate(() => {
    const api = window.__keplerComputeTest;
    api.analysis().setMode('off');
    const filters = api.getState().nativeFilters;
    const polygonIndex = filters.findIndex(filter => filter.type === 'polygon');
    if (polygonIndex < 0) throw new Error('Native polygon filter was not registered');
    api.removeFilter(polygonIndex);
  });
  await page.waitForFunction(count => window.__keplerComputeTest.getState().selectedCount === count,
    expected.sourceCount, {timeout: 30000});
  const staleRecipeRejected = await page.evaluate(recipe => {
    try {
      window.__keplerComputeTest.analysis().restoreRecipe(recipe);
      return false;
    } catch {
      return true;
    }
  }, standardized.recipe);
  assert.equal(staleRecipeRejected, true);
  const timeSelection = await page.evaluate(() => {
    const api = window.__keplerComputeTest;
    const [minimum] = api.timestampDomain();
    const maximum = minimum + 6 * 60 * 60 * 1000;
    const indices = [];
    api.source().forEach((row, index) => {
      if (row[7] >= minimum && row[7] <= maximum) indices.push(index);
    });
    api.setTimestamp(minimum, maximum);
    return {count: indices.length, firstRows: indices.slice(0, 10)};
  });
  await page.waitForFunction(count => {
    const state = window.__keplerComputeTest.getState();
    if (state.error) throw new Error(state.error);
    return state.selectedCount === count;
  }, timeSelection.count, {timeout: 30000});
  assert.deepEqual(await page.evaluate(async () => Array.from(
    (await window.__keplerComputeTest.readPage(0, 10)).sourceRowIndices)), timeSelection.firstRows);
  assert.deepEqual(errors, []);
  if (process.env.KEPLER_SMOKE_SCREENSHOT) await page.screenshot({path: process.env.KEPLER_SMOKE_SCREENSHOT});
  process.stdout.write(`${JSON.stringify({passed: true, sourceCount: expected.sourceCount,
    polygonCount: expected.count, checks: ['native polygon holes', 'bounded source identity',
      'selected z-score CPU oracle', 'resident classification', 'recipe restore',
        'polygon removal', 'stale recipe rejection', 'native absolute timestamp filter',
        'native polygon disable and reenable']}, null, 2)}\n`);
} finally {
  await browser.close();
}
