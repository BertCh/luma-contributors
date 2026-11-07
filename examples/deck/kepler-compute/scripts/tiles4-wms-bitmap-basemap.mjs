// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {pinViteModules} from './pin-vite-modules.mjs';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

// TILES4: WMS current/default/nearest TIME semantics, bitmap bounds patch + panel, attribution links.
// Headless only. Pass a URL as the first argument to override the default dev server.
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const failures = [];
page.on('pageerror', error => failures.push(error.message));
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=4096');
  await page.waitForFunction(() => { const api = window.__keplerComputeTest; return api && api.getState().selectedCount === api.source().length; });
  const report = await page.evaluate(async () => {
    const time = await import('/src/wms-time-extent.ts');
    const attribution = await import('/src/basemap-attribution.ts');
    const bitmap = await import('/src/bitmap-georeference.ts');
    const extent = '2024-01-01T00:00:00Z/2024-01-05T00:00:00Z/P1D';
    const current = time.resolveWmsTimeValue(extent, 'current', {now: Date.parse('2024-01-03T11:00:00Z')});
    const nearest = time.resolveWmsTimeValue(extent, '2023-06-01T00:00:00Z', {allowNearest: true});
    attribution.renderBasemapAttribution(attribution.getStyleAttributionSegments({sources: {a: {attribution: '<a href="https://osm.org/c">OSM</a> <a href="javascript:1">bad</a>'}}}));
    const element = document.getElementById(attribution.BASEMAP_ATTRIBUTION_ELEMENT_ID);
    const anchors = [...element.querySelectorAll('a')].map(anchor => [anchor.href, anchor.rel]);
    attribution.renderBasemapAttribution([]);
    const patch = bitmap.makeBitmapBoundsPatch([-10, -5, 10, 5], 0.5);
    const api = window.__keplerComputeTest;
    return {current, nearest, anchors, patch, hasBitmapModel: Boolean(api.getRasterToolsPanelModel?.().bitmapLayers)};
  });
  assert.equal(report.current.time, '2024-01-03T00:00:00Z');
  assert.equal(report.nearest.time, '2024-01-01T00:00:00Z');
  assert.equal(report.anchors.length, 1);
  assert.match(report.anchors[0][1], /noopener/);
  assert.equal(report.patch.boundsEast, 10);
  assert.equal(report.hasBitmapModel, true);
  assert.deepEqual(failures, []);
  process.stdout.write(`${JSON.stringify({passed: true, checks: ['wms current/nearest', 'attribution links', 'bitmap patch', 'panel model']}, null, 2)}\n`);
} finally {
  await browser.close();
}
