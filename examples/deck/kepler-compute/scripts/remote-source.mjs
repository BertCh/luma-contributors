// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData} from './native-navigation.mjs';

const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));

const csv = 'lat,lng,value\n' + Array.from({length: 50}, (_, index) => `${40 + index / 100},${-74 + index / 100},${index}`).join('\n');
const geojson = JSON.stringify({
  type: 'FeatureCollection',
  features: Array.from({length: 30}, (_, index) => ({
    type: 'Feature',
    geometry: {type: 'Point', coordinates: [-74 + index / 100, 40 + index / 100]},
    properties: {index, name: `p${index}`}
  }))
});
const fixtures = {
  '/remote/points.csv': {body: csv, type: 'text/csv', rows: 50},
  '/remote/points.geojson': {body: geojson, type: 'application/geo+json', rows: 30}
};

try {
  await page.route('https://fixtures.example/**', async route => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/remote/slow.csv') {
      // Never fulfilled until the page cancels; exercises the Cancel button.
      await new Promise(resolve => setTimeout(resolve, 15000));
      return route.abort();
    }
    if (path === '/remote/missing.csv') return route.fulfill({status: 404, body: 'no'});
    const fixture = fixtures[path];
    return route.fulfill({
      status: 200,
      contentType: fixture.type,
      headers: {'access-control-allow-origin': '*'},
      body: fixture.body
    });
  });
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => {
    const state = window.__keplerComputeTest?.getState();
    if (state?.error) throw new Error(state.error);
    return state?.selectedCount === 100000;
  }, undefined, {timeout: 60000});
  // The URL control lives in the Add Data > Prepare data workspace dialog.
  await openNativeData(page);
  const control = page.getByTestId('remote-source-control');
  await control.waitFor({timeout: 30000});
  for (const [path, fixture] of Object.entries(fixtures)) {
    await control.getByLabel('Source URL').fill(`https://fixtures.example${path}`);
    await control.getByRole('button', {name: 'Open URL'}).click();
    await page.getByTestId('remote-source-status').filter({hasText: `${fixture.rows.toLocaleString()} rows`}).waitFor({timeout: 30000});
  }
  await control.getByLabel('Source URL').fill('https://fixtures.example/remote/missing.csv');
  await control.getByRole('button', {name: 'Open URL'}).click();
  await page.getByTestId('remote-source-error').filter({hasText: /HTTP 404/}).waitFor();
  await control.getByLabel('Source URL').fill('https://fixtures.example/remote/slow.csv');
  await control.getByRole('button', {name: 'Open URL'}).click();
  await control.getByRole('button', {name: 'Cancel'}).click();
  await page.getByTestId('remote-source-error').filter({hasText: /cancelled/i}).waitFor();
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true,
    checks: ['open CSV from URL', 'open GeoJSON from URL', 'HTTP error surfaced', 'cancel in flight']}));
} catch (error) {
  console.error(await page.locator('body').innerText());
  throw error;
} finally {
  await browser.close();
}
