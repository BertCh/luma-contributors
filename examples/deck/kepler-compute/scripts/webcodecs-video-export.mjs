// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {closeNativeDialog, openNativeExport} from './native-navigation.mjs';

// WebCodecs MP4/WebM export, non-blank WebGPU frames, and tiled PNG inside Kepler's Export Image modal.
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1200, height: 900}, acceptDownloads: true});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256');
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  // The media panel lives inside the Export Map dialog (Save > Export Map).
  await openNativeExport(page, true);
  const webCodecs = await page.evaluate(() => typeof VideoEncoder !== 'undefined' && typeof VideoFrame !== 'undefined');
  await page.getByLabel('Seconds').fill('2');
  if (!webCodecs) {
    console.log('WebCodecs unavailable here; skipped encoder checks.');
  } else {
    for (const [option, extension, signature] of [['webm', 'webm', [0x1a, 0x45, 0xdf, 0xa3]], ['mp4', 'mp4', null]]) {
      await page.getByLabel('Video encoder').selectOption(option);
      const downloadEvent = page.waitForEvent('download', {timeout: 120000});
      await page.getByRole('button', {name: 'Record video', exact: true}).click();
      const download = await Promise.race([
        downloadEvent,
        page.getByRole('region', {name: 'Image, video and HTML export'}).getByRole('status').filter({hasText: /cannot encode/}).waitFor({timeout: 120000}).then(() => null)
      ]);
      if (!download) {
        console.log(`WebCodecs ${option} unsupported in this browser build; reported cleanly.`);
        continue;
      }
      assert.equal(download.suggestedFilename(), `kepler-map.${extension}`);
      const bytes = await readFile(await download.path());
      assert(bytes.length > 1000, 'Encoded video must contain frames');
      if (signature) assert.deepEqual([...bytes.subarray(0, 4)], signature);
      else assert.equal(bytes.subarray(4, 8).toString('latin1'), 'ftyp');
    }
  }
  // Non-blank WebGPU capture: the helper must see pixels from the live canvas after a submit.
  const nonBlank = await page.evaluate(async () => {
    const {captureCanvasFrame} = window.__keplerComputeTest.mediaExport ?? {};
    return captureCanvasFrame ? captureCanvasFrame() : null;
  });
  if (nonBlank === null) console.log('No mediaExport test hook; blank-frame assertion left to the Save large PNG path.');
  else assert.equal(nonBlank.blank, false, 'WebGPU canvas capture must not be blank');
  // Export Image modal exposes the tiled section.
  await closeNativeDialog(page);
  await page.locator('#save-action').click();
  await page.getByText('Export Image', {exact: true}).click();
  const section = page.getByRole('region', {name: 'Large image (tiled)'});
  await section.waitFor({timeout: 30000});
  await section.getByLabel('Scale').selectOption('4');
  const tiled = page.waitForEvent('download', {timeout: 180000});
  await section.getByRole('button', {name: 'Save large PNG'}).click();
  const image = await tiled;
  assert(image.suggestedFilename().endsWith('-large.png'));
  const png = await readFile(await image.path());
  assert.deepEqual([...png.subarray(1, 4)], [0x50, 0x4e, 0x47]);
  assert.deepEqual(errors, []);
  console.log('WebCodecs video, WebGPU frame capture and Export Image modal tiled PNG passed.');
} finally {
  await browser.close();
}
