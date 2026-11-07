// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Native 3D H3 hover highlight: extruded columns/prisms request the translucent white highlight.
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  const layers = await page.evaluate(() => window.__keplerComputeTest.renderedLayers().filter(layer => /Column|SolidPolygon/.test(layer.layerClass ?? '') && layer.extruded));
  // Requires an extruded H3 layer in the loaded configuration; the tester applies enable3d first.
  for (const layer of layers) {
    assert.equal(layer.autoHighlight, true);
    assert.deepEqual(layer.highlightColor, [255, 255, 255, 60]);
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, extrudedLayers: layers.length}));
} finally {
  await browser.close();
}
