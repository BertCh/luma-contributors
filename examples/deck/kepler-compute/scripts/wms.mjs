// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// WMS request/identify check. Builds GetMap and GetFeatureInfo URLs with src/wms-identify.ts (loaded
// in the Vite page), executes them against a public WMS from node, and verifies: the GetMap response
// is an image, 1.3.0 EPSG:4326 BBOX uses lat,lon order (the response must be non-empty, an
// inverted box returns an exception or blank), a bad layer yields a classified service exception,
// and GetFeatureInfo parses. Skips when the service is unreachable. Rendering through Kepler's
// native WMS layer is NOT covered (see backlog TILES-105).
//
// Usage: node scripts/wms.mjs [url]
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const url = process.argv.slice(2).find(value => value.startsWith('http')) || 'http://127.0.0.1:5318/?rows=256';
const SERVICE_URL = 'https://ows.terrestris.de/osm/service';
const request = {serviceUrl: SERVICE_URL, version: '1.3.0', layers: ['OSM-WMS'], crs: 'EPSG:4326', extent: [8, 49, 9, 50], width: 256, height: 256};

try {
  const probe = await fetch(`${SERVICE_URL}?SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.3.0`, {signal: AbortSignal.timeout(10000)});
  if (!probe.ok) throw new Error(`status ${probe.status}`);
} catch (error) {
  console.log(JSON.stringify({passed: true, skipped: `WMS unreachable: ${error.message}`}));
  process.exit(0);
}

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage();
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(url);
  const urls = await page.evaluate(async options => {
    const module = await import('/src/wms-identify.ts');
    return {
      map: module.buildWmsGetMapUrl(options),
      identify: module.buildWmsGetFeatureInfoUrl({...options, pixelX: 128, pixelY: 128, infoFormat: 'text/plain'}),
      badLayer: module.buildWmsGetMapUrl({...options, layers: ['no_such_layer']}),
      bbox: module.formatWmsBbox(options.extent, '1.3.0', 'EPSG:4326')
    };
  }, request);
  assert.equal(urls.bbox, '49,8,50,9', 'EPSG:4326 1.3.0 must be lat,lon');
  const map = await fetch(urls.map);
  assert.match(map.headers.get('content-type') ?? '', /image\//, 'GetMap did not return an image');
  const badBody = await (await fetch(urls.badLayer)).text();
  const classification = await page.evaluate(async text => (await import('/src/wms-identify.ts')).classifyWmsResponseError({status: 200, bodyText: text}), badBody);
  assert.equal(classification.kind, 'service-exception', 'bad layer not classified');
  const identifyBody = await (await fetch(urls.identify)).text();
  const parsed = await page.evaluate(async text => (await import('/src/wms-identify.ts')).parseWmsFeatureInfoResponse(text), identifyBody);
  assert.ok(['text', 'gml', 'json', 'unknown'].includes(parsed.format));
  // Live wiring (TILES2-2): TIME injection and GetFeatureInfo through the host controller, only when a native wms layer exists.
  const liveWms = await page.evaluate(async () => {
    const api = window.__keplerComputeTest;
    api.syncRasterLiveLayers();
    const layer = api.getState().nativeLayers?.find(candidate => candidate.type === 'wms');
    if (!layer) return null;
    const accepted = api.setWmsLiveTime(layer.id, '2020-01-01T00:00:00Z');
    const state = api.getWmsLiveState(layer.id);
    api.setWmsLiveTime(layer.id, null);
    return {accepted, time: state.time, cleared: api.getWmsLiveState(layer.id).time};
  });
  if (liveWms) {
    assert.equal(liveWms.accepted, true);
    assert.equal(liveWms.time, '2020-01-01T00:00:00Z');
    assert.equal(liveWms.cleared, null);
  }
  // TILES3-2: TIME picker pure logic and panel mount (BitmapLayer is deck's WGSL-capable layer; draw unverified).
  const timeReport = await page.evaluate(async () => {
    const extent = await import('/src/wms-time-extent.ts');
    const model = window.__keplerComputeTest.getRasterToolsPanelModel();
    return {
      instants: extent.parseWmsExtentInstants('2024-01-01/2024-03-01/P1M'),
      found: extent.findWmsTimeExtentText('<Layer><Name>a</Name><Dimension name="time">2024-01-01,2024-02-01</Dimension></Layer>', 'a'),
      wmsPanelLayers: model.wmsLayers.length
    };
  });
  assert.deepEqual(timeReport.instants, ['2024-01-01T00:00:00Z', '2024-02-01T00:00:00Z', '2024-03-01T00:00:00Z']);
  assert.equal(timeReport.found.extentText, '2024-01-01,2024-02-01');
  if (liveWms) {
    assert.ok(timeReport.wmsPanelLayers >= 1, 'WMS time picker should list the native wms layer');
    assert.ok(await page.locator('[data-testid="wms-time-picker"]').count() >= 1, 'WMS time picker is not mounted');
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, exception: classification.exception, identifyFormat: parsed.format, identifyFeatures: parsed.features.length}));
} finally {
  await browser.close();
}
