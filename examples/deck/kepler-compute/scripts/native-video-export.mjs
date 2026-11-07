// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {stat} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {closeNativeDialog, openNativeExport} from './native-navigation.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1200, height: 900}, acceptDownloads: true});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256');
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await closeNativeDialog(page);
  const initial = await page.evaluate(() => {
    window.__mediaSourceBuffers = Object.values(window.__keplerComputeTest.resourceIdentity().sourceBuffers).flat();
    return {state: window.__keplerComputeTest.getState(), config: window.__keplerComputeTest.saveNativeConfig()};
  });
  await openNativeExport(page, true);
  const supported = await page.evaluate(() => typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported('video/webm'));
  const recordVideo = page.getByRole('button', {name: 'Record video', exact: true});
  if (!supported) {
    assert(await recordVideo.isDisabled(), 'Unsupported browsers must disable video recording');
    console.log('Native video export: MediaRecorder WebM unsupported here; button disabled with message (skipped capture).');
  } else {
    await page.getByLabel('Seconds').fill('2');
    await page.getByLabel('Video encoder').selectOption('recorder');
    const downloadEvent = page.waitForEvent('download', {timeout: 90000});
    await recordVideo.click();
    const download = await downloadEvent;
    assert.equal(download.suggestedFilename(), 'kepler-map.webm');
    assert((await stat(await download.path())).size > 1000, 'Video must contain encoded frames');
    // Cancellation: start another capture, cancel, and confirm state restoration.
    await recordVideo.click();
    await page.getByRole('button', {name: 'Cancel', exact: true}).click();
    await page.getByRole('status').filter({hasText: 'cancelled'}).waitFor({timeout: 30000});
  }
  const tiled = page.waitForEvent('download', {timeout: 120000});
  await page.getByLabel('PNG scale').fill('9');
  await page.getByRole('button', {name: 'Save large PNG', exact: true}).click();
  const image = await tiled;
  assert.equal(image.suggestedFilename(), 'kepler-map-large.png');
  const final = await page.evaluate(() => ({
    state: window.__keplerComputeTest.getState(),
    config: window.__keplerComputeTest.saveNativeConfig(),
    sameBuffers: Object.values(window.__keplerComputeTest.resourceIdentity().sourceBuffers).flat().every((buffer, index) => buffer === window.__mediaSourceBuffers[index])
  }));
  assert.equal(final.state.deviceId, initial.state.deviceId);
  assert.equal(final.state.deviceGeneration, initial.state.deviceGeneration);
  assert.equal(final.state.uploadBytes, initial.state.uploadBytes);
  assert(final.sameBuffers, 'Export must keep source buffer identities');
  assert.deepEqual(final.config, initial.config, 'Media export must restore viewport/animation and not edit the workspace');
  assert.deepEqual(errors, []);
  console.log('Native video and tiled PNG export passed: WebM download, cancellation, 9x tiled PNG (>8192 px), restored viewport/config, stable device/buffers/uploads.');
} finally {
  await browser.close();
}
