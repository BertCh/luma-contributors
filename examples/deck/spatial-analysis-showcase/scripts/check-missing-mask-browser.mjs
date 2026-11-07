// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import {PNG} from 'pngjs';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {captureBuildEvidence} from './earth-build-evidence.mjs';
import {installReadoutIdentifiers} from './earth-browser-readouts.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory = process.argv[3] ?? '/tmp/luma-showcase-overhaul-20261025/missing-mask-visual';
await mkdir(outputDirectory, {recursive: true});
const verifyBuildEvidence = await captureBuildEvidence(baseUrl, outputDirectory);
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const results = [];
const errors = [];
page.on('pageerror', error => errors.push(String(error)));
page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
const select = (identifier, value) => page.locator(`#panel-controls select[id$="option-${identifier}"]`).selectOption(value);
const capture = async name => {
  await page.waitForTimeout(250);
  return PNG.sync.read(await page.locator('canvas:not(.maplibregl-canvas)').screenshot({
    path: `${outputDirectory}/${name}.png`,
    style: 'body * { visibility: hidden !important; } canvas:not(.maplibregl-canvas) { visibility: visible !important; }'
  }));
};
const isOrange = (pixels, offset) => pixels[offset] > 170 && pixels[offset] - pixels[offset + 1] > 25 && pixels[offset + 1] - pixels[offset + 2] > 30;
try {
  await page.goto(`${baseUrl.replace(/\/$/, '')}/#/story/missing-observations/known-truth`, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error, null, {timeout: 180000});
  assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null);
  await installReadoutIdentifiers(page, 'missing-observations');
  await page.waitForFunction(() => document.querySelector('[data-readout="holdoutError"]')?.textContent?.includes('81 / 81'), null, {timeout: 180000});
  await page.locator('#tab-controls').click();
  await select('mapLayer', 'source');
  const source = await capture('measured-source');
  await select('mapLayer', 'mask');
  const mask = await capture('block-mask');
  assert.equal(source.width, mask.width); assert.equal(source.height, mask.height);
  const background = Array.from(source.data.subarray(0, 3));
  let sourcePixels = 0;
  let orangePixels = 0;
  let sourceOrangePixels = 0;
  for (let offset = 0; offset < source.data.length; offset += 4) {
    if (background.some((channel, index) => Math.abs(source.data[offset + index] - channel) > 10)) sourcePixels++;
    if (isOrange(source.data, offset)) sourceOrangePixels++;
    if (isOrange(mask.data, offset)) orangePixels++;
  }
  assert.ok(sourcePixels > 50000, 'The observed64² crop occupies a measurable raster area');
  assert.equal(sourceOrangePixels, 0, 'Measured reflectance is not encoded as missing orange');
  const orangeFraction = orangePixels / sourcePixels;
  const expectedFraction = 81 / 4096;
  // A roughly45-pixel-wide9×9 block has at most one antialiased boundary pixel
  // per edge position; marker/outline pixels add a small denominator perturbation.
  // This screen-space bound is separate from exact81-cell numerical support checks.
  assert.ok(Math.abs(orangeFraction - expectedFraction) < .004, `Orange area ${orangeFraction}, expected81/4096=${expectedFraction}`);
  results.push({sourcePixels, orangePixels, orangeFraction, expectedFraction, tolerance: .004});
  await page.locator('#tab-story').click();
  await page.screenshot({path: `${outputDirectory}/known-truth-desktop.png`, fullPage: true});
  await page.setViewportSize({width: 390, height: 844});
  await page.waitForTimeout(300);
  await page.screenshot({path: `${outputDirectory}/known-truth-mobile.png`, fullPage: true});
  assert.deepEqual(errors, []);
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({passed: true, results, errors}, null, 2));
  console.log(JSON.stringify({passed: true, results, errors}));
} catch (error) {
  await writeFile(`${outputDirectory}/report.json`, JSON.stringify({passed: false, error: String(error), results, errors}, null, 2));
  throw error;
} finally {
  await browser.close();
  await verifyBuildEvidence();
}
