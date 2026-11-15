// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {withSyntheticSample} from './native-navigation.mjs';

// Host Time contract: zone validation, micro/nanosecond source admission, GPU selection and
// exact native filter playback with unchanged source allocations. Headless only.
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

// Match the other browser checks: the driver's positional URL takes precedence over the env default.
const baseUrl = process.argv[2] || process.env.KEPLER_COMPUTE_URL || 'http://127.0.0.1:5318/';
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await (await browser.newContext({viewport: {width: 1440, height: 1000}})).newPage();
await pinViteModules(page);
const failures = [];
page.on('pageerror', error => failures.push(error.message));
try {
  await page.goto(withSyntheticSample(baseUrl));
  await page.locator('[data-testid="time-contract-panel"]').waitFor({timeout: 30000});
  await page.locator('[data-testid="time-contract-zone-input"]').fill('Not/AZone');
  await page.locator('[data-testid="time-contract-zone-apply"]').click();
  await page.getByText('Unknown time zone').waitFor();
  await page.locator('[data-testid="time-contract-zone-input"]').fill('America/New_York');
  await page.locator('[data-testid="time-contract-zone-apply"]').click();
  await page.waitForTimeout(500);
  assert.equal(await page.getByText('Unknown time zone').count(), 0);
  assert.equal(await page.locator('[data-testid="spatial-join-crs-panel"]').count(), 0);
  const origin = Date.UTC(2026, 0, 1);
  const timeUnit = process.env.KEPLER_TIME_UNIT ?? 'nanosecond';
  assert(['microsecond', 'nanosecond'].includes(timeUnit));
  const scale = timeUnit === 'microsecond' ? 1000n : 1000000n;
  const type = timeUnit === 'microsecond' ? new arrow.TimestampMicrosecond('UTC') : new arrow.TimestampNanosecond('UTC');
  const ticks = [BigInt(origin) * scale + 1n, BigInt(origin + 500) * scale + 1n, BigInt(origin + 2000) * scale + 1n];
  const source = new arrow.Table({time: arrow.makeVector(arrow.makeData({type, data: BigInt64Array.from(ticks), length: ticks.length}))});
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: 'exact-playback.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))});
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 3);
  const initial = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0]);
  await closeNativeDialog(page);
  await page.evaluate(({id, origin}) => window.__keplerComputeTest.setPublishedRange(id, 'time', origin, origin + 1000), {id: initial.id, origin});
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 2);
  await page.locator('[data-testid="time-contract-mode-select"]').selectOption('free');
  await page.locator('[data-testid="time-contract-play"]').click();
  await page.locator('[data-testid="time-contract-pause"]').waitFor();
  await page.waitForFunction(origin => window.__keplerComputeTest.getState().nativeFilters.some(filter => filter.type === 'timeRange' && filter.value[0] > origin), origin);
  await page.locator('[data-testid="time-contract-pause"]').click();
  const final = await page.evaluate(() => window.__keplerComputeTest.getState());
  const filter = final.nativeFilters.find(filter => filter.type === 'timeRange' && filter.dataId.includes(initial.id));
  assert(filter.value.every(Number.isSafeInteger), 'Native playback boundaries follow the integral millisecond control policy');
  for (const field of ['sourceRevision', 'sourceUploadBytes', 'sourceBufferIds', 'sourceBatches']) assert.deepEqual(final.publications[0][field], initial[field]);
  assert.equal(final.error, '');
  assert.deepEqual(failures, []);
  console.log(`time-contract-panel: ok (zone validation, ${timeUnit} admission, exact GPU selection, integral-ms playback, stable source allocations)`);
} finally {
  await browser.close();
}
