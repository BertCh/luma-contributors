// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser qualification of native Grid/Hexagon statistics, scales, percentiles and extrusion.
// Usage: node scripts/aggregation-stats.mjs [url]   (dev server on 127.0.0.1:5318 by default)
import assert from 'node:assert/strict';
import {pinViteModules} from './pin-vite-modules.mjs';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const failures = [];
const diagnostics = [];
page.on('pageerror', error => failures.push(error.message));
page.on('console', message => {
  if (message.type() === 'error') diagnostics.push(message.text());
  if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())) failures.push(message.text());
});

const AGGREGATIONS = ['count', 'average', 'sum', 'minimum', 'maximum', 'median', 'stdev', 'variance', 'mode', 'countUnique'];
const SCALES = ['quantize', 'quantile', 'linear'];
const results = [];

async function countColoredPixels() {
  const screenshot = PNG.sync.read(await page.screenshot());
  let colored = 0;
  const signature = new Map();
  for (let y = 100; y < 850; y += 1) for (let x = 400; x < 1000; x += 1) {
    const offset = (y * screenshot.width + x) * 4;
    const channels = screenshot.data.subarray(offset, offset + 3);
    if (Math.max(...channels) > 100 && Math.max(...channels) - Math.min(...channels) > 30) {
      colored++;
      const key = `${channels[0] >> 4},${channels[1] >> 4},${channels[2] >> 4}`;
      signature.set(key, (signature.get(key) ?? 0) + 1);
    }
  }
  return {colored, distinctColors: signature.size};
}

async function applyAggregation(original, type, id, mutate) {
  const configuration = structuredClone(original);
  const layer = configuration.config.visState.layers[0];
  layer.id = id;
  layer.type = type;
  layer.config.columnMode = 'points';
  layer.config.columns = {lat: 'latitude', lng: 'longitude'};
  layer.config.color = [86, 203, 249];
  layer.config.colorField = null;
  layer.config.sizeField = null;
  layer.config.colorScale = 'quantize';
  layer.config.sizeScale = 'linear';
  layer.config.textLabel = [];
  layer.config.visConfig = {...layer.config.visConfig, worldUnitSize: 0.3, coverage: 0.9, enable3d: false, percentile: [0, 100],
    elevationPercentile: [0, 100], elevationScale: 5, sizeRange: [0, 500], sizeAggregation: 'sum', colorAggregation: 'count',
    colorRange: {colors: ['#21918c', '#5ec962', '#fde725']}};
  mutate(layer);
  const frameCount = await page.evaluate(() => window.__keplerComputeTest.getState().frameCount);
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), configuration);
  await page.waitForFunction(({id: layerId, frame}) => {
    const api = window.__keplerComputeTest;
    const state = api.getState();
    const invalid = state.nativeLayers?.find(candidate => candidate.id === layerId && candidate.valid === false);
    if (invalid) throw new Error(`${layerId}: ${JSON.stringify(invalid)} ${state.error ?? ''}`);
    return state.frameCount > frame && api.renderedLayers().some(candidate => candidate.id === layerId && candidate.visible && candidate.loaded);
  }, {id, frame: frameCount}, {timeout: 45000});
  await page.waitForTimeout(400);
  return layer;
}

try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=4096');
  await page.waitForFunction(() => { const api = window.__keplerComputeTest; return api && api.getState().selectedCount === api.source().length; });
  const original = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());

  for (const type of ['grid', 'hexagon']) {
    // 1. Every native aggregation renders, with every supported scale, on the weighted value field.
    for (const aggregation of AGGREGATIONS) for (const scale of SCALES) {
      const id = `agg-${type}-${aggregation}-${scale}`;
      await applyAggregation(original, type, id, layer => {
        if (aggregation !== 'count') {
          layer.config.colorField = {name: 'value', type: 'real'};
          layer.visualChannels = {...layer.visualChannels, colorField: layer.config.colorField};
        }
        layer.config.colorScale = scale;
        layer.config.visConfig.colorAggregation = aggregation;
      });
      const pixels = await countColoredPixels();
      assert(pixels.colored > 300, `${id} must draw cells; found ${pixels.colored}`);
      results.push({id, ...pixels});
    }

    // 2. Percentile filters hide cells: the colored area must shrink.
    const full = await (async () => {
      await applyAggregation(original, type, `pct-${type}-full`, () => {});
      return countColoredPixels();
    })();
    await applyAggregation(original, type, `pct-${type}-trim`, layer => { layer.config.visConfig.percentile = [40, 80]; });
    const trimmed = await countColoredPixels();
    assert(trimmed.colored < full.colored, `${type} percentile [40,80] must hide cells (${trimmed.colored} < ${full.colored})`);
    results.push({id: `percentile-${type}`, full: full.colored, trimmed: trimmed.colored});

    // 3. Custom scale draws; jenks is covered in step 6.
    await applyAggregation(original, type, `custom-${type}`, layer => {
      layer.config.colorScale = 'custom';
      layer.config.visConfig.colorRange = {colors: ['#21918c', '#5ec962', '#fde725'], colorMap: [[2, '#21918c'], [6, '#5ec962'], [null, '#fde725']]};
    });
    const custom = await countColoredPixels();
    assert(custom.colored > 300, `${type} custom scale must draw`);
    results.push({id: `custom-${type}`, ...custom});

    // 4. Extrusion: enable3d with an elevation field and range renders (taller geometry changes the picture).
    const flat = await (async () => {
      await applyAggregation(original, type, `flat-${type}`, () => {});
      return countColoredPixels();
    })();
    await applyAggregation(original, type, `extruded-${type}`, layer => {
      layer.config.sizeField = {name: 'value', type: 'real'};
      layer.visualChannels = {...layer.visualChannels, sizeField: layer.config.sizeField};
      layer.config.visConfig.enable3d = true;
      layer.config.visConfig.sizeAggregation = 'average';
      layer.config.visConfig.sizeRange = [0, 3000];
    });
    const extruded = await countColoredPixels();
    assert(extruded.colored > 100, `${type} extruded cells must draw`);
    assert(extruded.colored !== flat.colored, `${type} enable3d must change the rendered image`);
    results.push({id: `extruded-${type}`, flat: flat.colored, extruded: extruded.colored});

    // 5. Elevation percentile hides tall/short cells even when flat.
    await applyAggregation(original, type, `elevation-percentile-${type}`, layer => {
      layer.config.sizeField = {name: 'value', type: 'real'};
      layer.config.visConfig.sizeAggregation = 'sum';
      layer.config.visConfig.elevationPercentile = [50, 100];
    });
    const elevationFiltered = await countColoredPixels();
    assert(elevationFiltered.colored < flat.colored, `${type} elevationPercentile [50,100] must hide cells`);
    results.push({id: `elevation-percentile-${type}`, flat: flat.colored, filtered: elevationFiltered.colored});
  }

  // 6. Jenks is a supported native scale for resident grid and hexagon layers: it must draw cells.
  for (const type of ['grid', 'hexagon']) {
    await applyAggregation(original, type, `jenks-${type}`, layer => {
      layer.config.colorField = {name: 'value', type: 'real'};
      layer.visualChannels = {...layer.visualChannels, colorField: layer.config.colorField};
      layer.config.colorScale = 'jenks';
      layer.config.visConfig.colorAggregation = 'average';
    });
    const jenks = await countColoredPixels();
    assert(jenks.colored > 300, `${type} jenks scale must draw cells; found ${jenks.colored}`);
    results.push({id: `jenks-${type}`, ...jenks});
  }

  assert.deepEqual(failures, []);
  process.stdout.write(`${JSON.stringify({passed: true, results}, null, 2)}\n`);
} catch (error) {
  process.stderr.write(`${JSON.stringify({diagnostics: diagnostics.slice(-20), results})}\n`);
  process.stderr.write(`${JSON.stringify(await page.evaluate(() => ({state: window.__keplerComputeTest?.getState(), renderedLayers: window.__keplerComputeTest?.renderedLayers()})), null, 2)}\n`);
  throw error;
} finally {
  await browser.close();
}
