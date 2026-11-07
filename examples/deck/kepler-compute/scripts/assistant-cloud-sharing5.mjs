// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Headless browser check: local cloud provider save/list/delete round trip through
// the __lumaLocalCloudProvider test hook. Usage: node scripts/assistant-cloud-sharing5.mjs [url]
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const url = process.argv[2] ?? 'http://127.0.0.1:5318/?rows=256';
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await page.goto(url);
await page.waitForFunction(() => {
  const state = window.__keplerComputeTest?.getState();
  if (state?.error) throw new Error(state.error);
  return state?.ready && state.selectedCount !== undefined && window.__lumaLocalCloudProvider;
}, undefined, {timeout: 60000});
const result = await page.evaluate(async () => {
  const provider = globalThis.__lumaLocalCloudProvider;
  const user = await provider.getUser();
  const before = (await provider.listMaps()).length;
  const saved = await provider.uploadMap({
    mapData: {map: {info: {title: 'sharing5-check', description: ''}}},
    options: {overwrite: true}
  });
  const url = provider.getMapUrl({id: saved.id});
  let publicRefused = false;
  try {
    await provider.uploadMap({mapData: {map: {info: {title: 'public'}}}, options: {isPublic: true}});
  } catch {
    publicRefused = true;
  }
  await provider.deleteMap(saved.id);
  return {user, before, url, publicRefused, after: (await provider.listMaps()).length};
});
await browser.close();
const ok =
  result.user?.name === 'Local user' &&
  result.url.startsWith('local://map/') &&
  result.publicRefused &&
  result.after === result.before;
console.log(JSON.stringify(result), ok ? 'PASS' : 'FAIL');
process.exit(ok ? 0 : 1);
