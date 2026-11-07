// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdir, writeFile} from 'node:fs/promises';
import {PNG} from 'pngjs';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {captureBuildEvidence} from './earth-build-evidence.mjs';

const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory = process.argv[3] ?? '/tmp/street-density-polygons';
await mkdir(outputDirectory, {recursive: true});
const verifyBuildEvidence = await captureBuildEvidence(baseUrl, outputDirectory);
const report = {passed: false, status: 'pending', failureReason: null, consoleErrors: [], cases: []};
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
let page;

async function captureAnalysis(name) {
  const canvas = page.locator('canvas:not(.maplibregl-canvas)');
  assert.equal(await canvas.count(), 1);
  const encoded = await canvas.screenshot({
    path: `${outputDirectory}/${name}.png`,
    animations: 'disabled',
    // Only the GPU polygon/outline layers remain: basemap, labels, legend and HTML annotations
    // cannot provide false positive colored pixels. This does not change any scene option.
    style: 'body * { visibility: hidden !important; } canvas:not(.maplibregl-canvas) { visibility: visible !important; }'
  });
  const pixels = PNG.sync.read(encoded);
  let warmFillPixels = 0;
  for (let offset = 0; offset < pixels.data.length; offset += 4) {
    const [red, green, blue, alpha] = pixels.data.subarray(offset, offset + 4);
    // YlOrBr polygon classes are warm; white outlines and neutral canvas ground are excluded.
    if (alpha > 128 && red > blue + 20 && green > blue + 10) warmFillPixels++;
  }
  return {warmFillPixels, pixelSha256: createHash('sha256').update(pixels.data).digest('hex')};
}

try {
  for (const scenario of [
    {step: 'by-neighbourhood', set: 'areas', readyReadout: 'topLength', expected: /km/},
    {step: 'per-resident', set: 'tracts', readyReadout: 'residentMedian', expected: /m per resident/}
  ]) {
    // A new page prevents the previous scene's computed state from masking a first-entry bug.
    page = await browser.newPage({viewport: {width: 1440, height: 960}});
    page.on('pageerror', error => report.consoleErrors.push(error.message));
    page.on('console', message => {if (message.type() === 'error') report.consoleErrors.push(message.text());});
    const url = new URL(baseUrl);
    url.hash = `#/story/street-density/${scenario.step}`;
    await page.goto(url.toString(), {waitUntil: 'domcontentloaded'});
    await page.waitForFunction(() => globalThis.spatialAnalysisShowcase?.ready || globalThis.spatialAnalysisShowcase?.error, null, {timeout: 90000});
    assert.equal(await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error), null);
    await page.waitForFunction(identifier => /\d/.test(document.querySelector(`[data-readout-id="${identifier}"] dd`)?.textContent ?? ''), scenario.readyReadout, {timeout: 90000});
    assert.match(await page.locator(`[data-readout-id="${scenario.readyReadout}"] dd`).first().textContent(), scenario.expected);
    await page.waitForTimeout(2000);
    await page.screenshot({path: `${outputDirectory}/${scenario.set}-first-entry.png`, fullPage: true});
    const initial = await captureAnalysis(`${scenario.set}-initial-analysis`);
    const record = {set: scenario.set, initial, outlinesOff: null, restored: null, passed: false};
    report.cases.push(record);

    // Public outline controls force a layer refresh, providing a positive control for the
    // original failure: computed polygons appeared only after a subsequent option change.
    await page.getByRole('tab', {name: 'All controls', exact: false}).click();
    const outlines = page.locator('#all-option-showOutlines');
    assert.equal(await outlines.isChecked(), true);
    await outlines.focus();
    await outlines.press('Space');
    await page.waitForTimeout(250);
    record.outlinesOff = await captureAnalysis(`${scenario.set}-without-outlines`);
    await outlines.press('Space');
    await page.waitForTimeout(250);
    record.restored = await captureAnalysis(`${scenario.set}-restored-analysis`);
    assert.ok(record.outlinesOff.warmFillPixels > 0, `${scenario.set}: positive control contains warm polygon fill with outlines disabled`);
    assert.ok(initial.warmFillPixels > 0, `${scenario.set}: first entry must draw polygon fill before any option interaction`);
    assert.equal(initial.pixelSha256, record.restored.pixelSha256, `${scenario.set}: first entry and explicit layer refresh draw identical analysis pixels`);
    assert.notEqual(record.outlinesOff.pixelSha256, record.restored.pixelSha256, `${scenario.set}: outline control changes actual rendered boundaries`);
    record.passed = true;
    await page.close();
    page = null;
  }
  assert.deepEqual(report.consoleErrors, []);
  report.passed = true;
  report.status = 'passed';
} catch (error) {
  report.status = 'failed';
  report.failureReason = String(error);
  if (page) await page.screenshot({path: `${outputDirectory}/failure.png`, fullPage: true}).catch(() => {});
  process.exitCode = 1;
} finally {
  await browser.close();
  await writeFile(`${outputDirectory}/report.json`, `${JSON.stringify(report, null, 2)}\n`);
  await verifyBuildEvidence();
}
