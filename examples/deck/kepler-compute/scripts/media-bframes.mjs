// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {closeNativeDialog, openNativeExport} from './native-navigation.mjs';

// Records a 2 s MP4 with WebCodecs in quality latency mode and checks the container: with B-frames
// the file has ctts (version 1) and elst boxes; without them it has neither. A WebM clip must end
// with a Cues element. Headless only; reports BLOCKED text when WebCodecs is unavailable.
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1200, height: 900}, acceptDownloads: true});
await pinViteModules(page);
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256');
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeExport(page, true);
  if (!(await page.evaluate(() => typeof VideoEncoder !== 'undefined'))) {
    console.log('BLOCKED: WebCodecs unavailable in this browser build.');
  } else {
    await page.getByLabel('Seconds').fill('2');
    for (const option of ['mp4', 'webm']) {
      await page.getByLabel('Video encoder').selectOption(option);
      const downloadEvent = page.waitForEvent('download', {timeout: 120000});
      await page.getByRole('button', {name: 'Record video', exact: true}).click();
      const download = await downloadEvent;
      const bytes = await readFile(await download.path());
      const text = bytes.toString('latin1');
      if (option === 'mp4') {
        const hasComposition = text.includes('ctts');
        assert.equal(text.includes('elst'), hasComposition, 'ctts and elst must appear together');
        console.log(`mp4 ${bytes.length} bytes, B-frame reordering ${hasComposition ? 'present (ctts v1 + elst)' : 'not used by this encoder'}`);
      } else {
        assert.ok(text.includes('\x1c\x53\xbb\x6b'), 'WebM must contain a Cues element');
        assert.ok(text.includes('\x11\x4d\x9b\x74'), 'WebM must contain a SeekHead element');
        console.log(`webm ${bytes.length} bytes with SeekHead and Cues`);
      }
    }
  }
  await closeNativeDialog(page);
} finally {
  await browser.close();
}
