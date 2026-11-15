// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getSourceCheckUrl} from './source-check-utils.mjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(getSourceCheckUrl(process.argv[2], {rows: 4096}));
  await page.waitForFunction(
    () => {
      const state = window.__keplerComputeTest?.getState();
      if (state?.error) throw new Error(state.error);
      return state?.ready && state.selectedCount !== undefined && window.__lumaLocalCloudProvider;
    },
    undefined,
    {timeout: 60000}
  );
  const expectedCount = await page.evaluate(() => {
    const api = window.__keplerComputeTest;
    api.setRange(25, 65);
    return api.source().filter(row => row[2] >= 25 && row[2] <= 65).length;
  });
  await page.waitForFunction(
    count => window.__keplerComputeTest.getState().selectedCount === count,
    expectedCount
  );
  const saved = await page.evaluate(async () => {
    const provider = window.__lumaLocalCloudProvider;
    const mapData = {
      map: {info: {title: 'cloud-check', description: 'native cloud provider script'}}
    };
    const item = await provider.uploadMap({mapData, options: {}});
    const conflict = await provider.uploadMap({mapData, options: {}}).then(
      () => null,
      error => error.message
    );
    const publicError = await provider.uploadMap({mapData, options: {isPublic: true}}).then(
      () => null,
      error => error.message
    );
    const maps = await provider.listMaps();
    return {id: item.id, conflict, publicError, titles: maps.map(map => map.title)};
  });
  assert.equal(saved.conflict, 'file_conflict');
  assert.match(saved.publicError, /Public publishing is unavailable/);
  assert.ok(saved.titles.includes('cloud-check'));
  // Change the selection, then restore the saved bundle through the provider.
  await page.evaluate(() => window.__keplerComputeTest.setRange(0, 5));
  await page.evaluate(async id => {
    const result = await window.__lumaLocalCloudProvider.downloadMap({id});
    if (result.format !== 'keplergl') throw new Error('Unexpected download format');
  }, saved.id);
  await page.waitForFunction(
    count => window.__keplerComputeTest.getState().selectedCount === count,
    expectedCount,
    {timeout: 60000}
  );
  await page.evaluate(id => window.__lumaLocalCloudProvider.deleteMap(id), saved.id);
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({
      passed: true,
      expectedCount,
      checks: ['upload via bundle', 'conflict', 'public refused', 'download replays resident data']
    })
  );
} finally {
  await browser.close();
}
