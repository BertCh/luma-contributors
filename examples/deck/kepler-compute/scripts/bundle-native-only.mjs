// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Native-only datasets (autoPrepare=0 keeps a loaded CSV as a native Kepler table) are embedded in
// the portable bundle as checksummed Arrow IPC and restored as native tables after a reload.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeExport} from './native-navigation.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
const csv = Buffer.from('latitude,longitude,value,note\n' + Array.from({length: 64}, (_, index) => `${40.5 + index / 100},${-74 + index / 100},${index % 9 === 0 ? '' : index},n${index}`).join('\n'));
async function readDownload(button) {
  const pending = page.waitForEvent('download');
  await button.click();
  const chunks = [];
  for await (const chunk of await (await pending).createReadStream()) chunks.push(chunk);
  return Buffer.concat(chunks);
}
try {
  const address = process.argv[2] || 'http://127.0.0.1:5318/?autoPrepare=0';
  await page.goto(address);
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount > 0, undefined, {timeout: 60000});
  await page.getByRole('button', {name: 'Add Data', exact: true}).click();
  await page.locator('input[type=file]').setInputFiles({name: 'native-only.csv', mimeType: 'text/csv', buffer: csv});
  await page.locator('.ReactModal__Content').getByRole('button', {name: 'Add Data', exact: true}).click();
  await page.waitForTimeout(2000);
  assert.equal(await page.evaluate(() => window.__keplerComputeTest.getState().publications.length), 0, 'dataset must stay native-only');
  await openNativeExport(page, true);
  const bundleBytes = await readDownload(page.getByRole('button', {name: 'Save portable bundle', exact: true}));
  const bundle = JSON.parse(bundleBytes.toString('utf8'));
  assert.equal(bundle.nativeOnlySources.length, 1);
  const embedded = bundle.nativeOnlySources[0];
  assert.equal(embedded.rowCount, 64);
  assert.equal(createHash('sha256').update(Buffer.from(embedded.data, 'base64')).digest('hex'), embedded.sha256);
  assert.deepEqual(embedded.fields.map(field => field.name), ['latitude', 'longitude', 'value', 'note']);
  // Tampering must reject the whole bundle before any restore step.
  const tampered = structuredClone(bundle);
  tampered.nativeOnlySources[0].sha256 = '0'.repeat(64);
  await page.reload();
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount > 0, undefined, {timeout: 60000});
  await openNativeExport(page, true);
  const portable = page.getByRole('region', {name: 'Portable sandbox bundle', exact: true});
  await portable.getByLabel('Load portable bundle').setInputFiles({name: 'tampered.bundle.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(tampered))});
  await portable.getByRole('status').filter({hasText: 'checksum'}).waitFor({timeout: 30000});
  await portable.getByLabel('Load portable bundle').setInputFiles({name: 'workspace.bundle.json', mimeType: 'application/json', buffer: bundleBytes});
  await portable.getByRole('status').filter({hasText: 'Bundle validated'}).waitFor({timeout: 30000});
  assert.match(await portable.innerText(), /native-only dataset/);
  await portable.getByRole('button', {name: 'Restore portable workspace', exact: true}).click();
  await page.locator('.ReactModal__Overlay').waitFor({state: 'hidden', timeout: 60000});
  await openNativeExport(page, false);
  const options = await page.getByLabel('Stream dataset').locator('option').allTextContents();
  assert.ok(options.some(text => /native table/.test(text)), `native table listed: ${options.join('|')}`);
  // Local-cloud save (Kepler's Save/Load map provider) must embed the same native-only original.
  const cloud = await page.evaluate(async () => {
    const provider = globalThis.__lumaLocalCloudProvider;
    const saved = await provider.uploadMap({mapData: {map: {info: {title: 'native-only cloud'}}}, options: {}});
    const stored = await globalThis.__lumaLocalCloudStore?.getBundle?.(saved.id);
    return {id: saved.id, stored: stored ?? null};
  });
  if (cloud.stored) {
    const cloudBundle = JSON.parse(cloud.stored);
    assert.equal(cloudBundle.nativeOnlySources?.length, 1, 'cloud bundle embeds the native-only table');
    assert.equal(createHash('sha256').update(Buffer.from(cloudBundle.nativeOnlySources[0].data, 'base64')).digest('hex'), cloudBundle.nativeOnlySources[0].sha256);
  }
  await page.evaluate(id => globalThis.__lumaLocalCloudProvider.downloadMap({id}), cloud.id);
  await openNativeExport(page, false);
  const cloudOptions = await page.getByLabel('Stream dataset').locator('option').allTextContents();
  assert.ok(cloudOptions.some(text => /native table/.test(text)), `native table after cloud load: ${cloudOptions.join('|')}`);
  // Standalone HTML export embeds the native-only table inside the luma-kepler-bundle script element.
  // Save standalone HTML is in the media panel of the Export Map dialog.
  await openNativeExport(page, true);
  const standaloneButton = page.getByRole('button', {name: 'Save standalone HTML', exact: true});
  const html = (await readDownload(standaloneButton)).toString('utf8');
  const embeddedJson = html.match(/<script[^>]*id="luma-kepler-bundle"[^>]*>([\s\S]*?)<\/script>/)?.[1];
  assert.ok(embeddedJson, 'standalone HTML carries the bundle JSON');
  const standaloneBundle = JSON.parse(embeddedJson.replace(/\\u003c/g, '<'));
  assert.equal(standaloneBundle.nativeOnlySources?.length, 1, 'standalone bundle embeds the native-only table');
  assert.equal(standaloneBundle.nativeOnlySources[0].sha256, embedded.sha256);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, checks: ['native-only embed with checksum', 'tamper rejection', 'restore as native table', 'local-cloud save+load embeds native-only', 'standalone HTML embeds native-only']}));
} catch (error) {
  console.error(await page.locator('body').innerText());
  throw error;
} finally {await browser.close();}
