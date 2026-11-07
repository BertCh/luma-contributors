// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {PNG} from 'pngjs';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {closeNativeDialog} from './native-navigation.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1200, height: 900}, acceptDownloads: true});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
const preview = page.getByRole('img', {name: 'Map preview', exact: true});
async function openImage() {
  await closeNativeDialog(page);
  await page.locator('#save-action').click();
  await page.getByText('Export Image', {exact: true}).click();
  await page.locator('.export-image-modal').waitFor();
}
async function waitImage() {
  const image = await page.waitForFunction(() => {
    const error = document.querySelector('.preview-image--error');
    if (error) throw new Error(error.textContent);
    const preview = document.querySelector('img[alt="Map preview"]');
    const source = preview?.getAttribute('src');
    return source?.startsWith('data:image/png;') && !document.querySelector('.native-image-export-map') ? source : false;
  }, undefined, {timeout: 45000});
  return image.jsonValue();
}
function readPreview(data) { return PNG.sync.read(Buffer.from(data.split(',')[1], 'base64')); }
function countPixels(image, predicate) {
  let count = 0;
  for (let offset = 0; offset < image.data.length; offset += 4) if (predicate(...image.data.subarray(offset, offset + 4))) count++;
  return count;
}
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256');
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await page.getByRole('button', {name: 'Toggle charts', exact: true}).click();
  await page.locator('.chart-config__type').click();
  await page.getByText('Big Number', {exact: true}).click();
  await page.locator('[aria-label="Chart value: 256"]').waitFor();
  const initial = await page.evaluate(() => {
    const state = window.__keplerComputeTest.getState();
    window.__imageExportSourceBuffers = Object.values(window.__keplerComputeTest.resourceIdentity().sourceBuffers).flat();
    return {state, config: window.__keplerComputeTest.saveNativeConfig()};
  });
  await openImage();
  const opaque = readPreview(await waitImage());
  assert.equal(opaque.width, 1200);
  assert.equal(opaque.height, 900);
  assert(countPixels(opaque, (red, green, blue, alpha) => red === 16 && green === 29 && blue === 43 && alpha === 255) > 1000, 'PNG must retain the actual local basemap color');
  assert(countPixels(opaque, (red, green, blue, alpha) => alpha === 255) > opaque.width * opaque.height * 0.9, 'Basemap must be composed into the PNG');
  assert(countPixels(opaque, (red, green, blue, alpha) => alpha > 0 && Math.max(red, green, blue) > 120 && Math.max(red, green, blue) - Math.min(red, green, blue) > 40) > 50, 'PNG must include the resident GPU layer');
  await page.locator('#export-image-file-name').fill('resident-map.png');
  const downloadEvent = page.waitForEvent('download');
  await page.getByRole('button', {name: 'Download', exact: true}).click();
  const download = await downloadEvent;
  assert.equal(download.suggestedFilename(), 'resident-map.png');
  const downloaded = PNG.sync.read(await readFile(await download.path()));
  assert.deepEqual(downloaded.data, opaque.data, 'Native download must contain the preview pixels');
  await openImage();
  await waitImage();
  const transparency = page.locator('#export-transparent-background');
  await transparency.check({force: true});
  const transparent = readPreview(await waitImage());
  assert(countPixels(transparent, (red, green, blue, alpha) => alpha === 0) > transparent.width * transparent.height * 0.5, 'Transparent export must omit the basemap');
  assert(countPixels(transparent, (red, green, blue, alpha) => alpha > 0) > 50, 'Transparent export must retain resident geometry');
  const withoutLegend = await preview.getAttribute('src');
  await page.locator('#add-map-legend').check({force: true});
  const withLegend = readPreview(await waitImage());
  assert.notDeepEqual(withLegend.data, readPreview(withoutLegend).data, 'Legend option must affect the composed pixels');
  await page.locator('#add-map-charts').check({force: true});
  const exportedChart = page.locator('.native-image-export-map .resident-chart-card');
  await exportedChart.waitFor();
  assert.equal(await exportedChart.locator('input, select').count(), 0, 'Exported chart must hide editing controls');
  assert.equal(await exportedChart.getByText('Refresh chart', {exact: true}).count(), 0);
  const withCharts = readPreview(await waitImage());
  assert.notDeepEqual(withCharts.data, withLegend.data, 'Charts option must include the native chart overlay');
  await page.locator('#export-image-modal__option_resolution').getByText('2x', {exact: true}).click();
  const larger = readPreview(await waitImage());
  assert.equal(larger.width, 2400);
  assert.equal(larger.height, 1800);
  await page.locator('#export-image-modal__option_ratio').getByText('16:9', {exact: true}).click();
  const widescreen = readPreview(await waitImage());
  assert.equal(widescreen.width / widescreen.height, 16 / 9);
  // Cancelling while a new option is rendering must restore the original live map/device.
  await page.locator('#export-transparent-background').uncheck({force: true});
  await closeNativeDialog(page);
  await page.waitForFunction(() => !document.querySelector('.native-image-export-map'));
  const final = await page.evaluate(() => ({state: window.__keplerComputeTest.getState(), config: window.__keplerComputeTest.saveNativeConfig()}));
  assert.equal(final.state.deviceId, initial.state.deviceId);
  assert.equal(final.state.deviceGeneration, initial.state.deviceGeneration);
  assert.equal(final.state.uploadBytes, initial.state.uploadBytes);
  assert(await page.evaluate(() => {
    const sourceBuffers = Object.values(window.__keplerComputeTest.resourceIdentity().sourceBuffers).flat();
    return sourceBuffers.length === window.__imageExportSourceBuffers.length && sourceBuffers.every((buffer, index) => buffer === window.__imageExportSourceBuffers[index]);
  }), 'Export must retain the original source buffer objects');
  assert.deepEqual(final.config, initial.config, 'Image export must not edit the workspace');
  assert.deepEqual(errors, []);
  console.log(`Native map image export passed: ${opaque.width}×${opaque.height}, real download, basemap, resident pixels, alpha, legend/charts, resolution/aspect ratio, cancellation and stable device/source buffers/uploads.`);
} catch (error) {
  console.error((await page.locator('body').innerText()).slice(-10000));
  throw error;
} finally {
  await browser.close();
}
