// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Vector-tile working set, cross-tile ids and analysis-scope check. Fetches real CARTO MVT tiles in
// node (@loaders.gl/mvt), then runs src/vector-tile-*.ts inside the Vite page (dynamic import of
// the TypeScript sources) to verify: bounded LRU admission, stable ids that survive evict and
// reload, and that a viewport population is refused for global analysis. Skips when the tile server
// is unreachable. The live controller is exercised with a stand-in MVTLayer; the native Kepler vectorTile layer path needs a manual/tileset run (TILES2-100).
//
// Usage: node scripts/vector-tiles.mjs [url]
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {parse} from '@loaders.gl/core';
import {MVTLoader} from '@loaders.gl/mvt';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const url = process.argv.slice(2).find(value => value.startsWith('http')) || 'http://127.0.0.1:5318/?rows=256';
const TILES = [[4, 7, 5], [4, 8, 5], [4, 7, 6], [4, 8, 6]];
const tileUrl = ([zoom, column, row]) =>
  `https://tiles-a.basemaps.cartocdn.com/vectortiles/carto.streets/v1/${zoom}/${column}/${row}.mvt`;

async function loadTile(coordinates) {
  const response = await fetch(tileUrl(coordinates), {signal: AbortSignal.timeout(10000)});
  if (!response.ok) throw new Error(`tile ${coordinates} status ${response.status}`);
  const [zoom, column, row] = coordinates;
  const features = await parse(await response.arrayBuffer(), MVTLoader, {
    mvt: {coordinates: 'wgs84', tileIndex: {x: column, y: row, z: zoom}, shape: 'geojson'}
  });
  return {key: coordinates.join('/'), features: features.map(feature => ({id: feature.id ?? null, properties: feature.properties ?? {}, geometry: feature.geometry}))};
}

let tiles;
try {
  tiles = await Promise.all(TILES.map(loadTile));
} catch (error) {
  console.log(JSON.stringify({passed: true, skipped: `tile server unreachable: ${error.message}`}));
  process.exit(0);
}

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage();
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(url);
  const result = await page.evaluate(async decodedTiles => {
    const {VectorTileWorkingSet, estimateDecodedTileByteLength} = await import('/src/vector-tile-working-set.ts');
    const {describeVectorTileAnalysisScope, decideVectorTileAnalysis} = await import('/src/vector-tile-analysis-scope.ts');
    const sizes = decodedTiles.map(tile => estimateDecodedTileByteLength(tile.features.map(feature => ({...feature, coordinateCount: 4}))));
    const budgetBytes = Math.floor(sizes.slice(0, 3).reduce((total, size) => total + size, 0) * 1.05);
    const evicted = [];
    const set = new VectorTileWorkingSet({budgetBytes, featureIdOptions: {sourceLayer: 'tile'}, onEvictTile: key => evicted.push(key)});
    const idsBefore = new Map();
    decodedTiles.forEach((tile, index) => {
      const admission = set.admitTile(tile.key, sizes[index], tile.features);
      idsBefore.set(tile.key, set.featureRegistry.getTileFeatureIds(tile.key));
      if (!admission.admitted) throw new Error(`rejected ${tile.key}`);
    });
    const afterAdmit = set.getStatus();
    // Reload the first tile (evicted by LRU) and compare ids.
    const first = decodedTiles[0];
    set.admitTile(first.key, sizes[0], first.features);
    const idsAfter = set.featureRegistry.getTileFeatureIds(first.key);
    const identical = idsAfter.length === idsBefore.get(first.key).length && idsAfter.every((id, index) => id === idsBefore.get(first.key)[index]);
    const scope = describeVectorTileAnalysisScope({datasetId: 'carto', zoom: 4, visibleTileKeys: decodedTiles.map(tile => tile.key), loadedTileKeys: set.getResidentTileKeys(), rejectedTileKeys: set.getStatus().rejectedTileKeys, featureSummary: set.featureRegistry.summarize()});
    return {afterAdmit, evicted, identical, scope, global: decideVectorTileAnalysis(scope, {requiresCompleteDataset: true}), budgetBytes};
  }, tiles);
  // Live wiring: controller on the real host (registry reservations + GPU buffers when a runtime exists).
  const live = await page.evaluate(async decodedTiles => {
    const hook = window.__keplerComputeTest;
    if (!hook?.getVectorTileState) throw new Error('vector tile test hook missing');
    const {VectorTileLiveController} = await import('/src/vector-tile-live-controller.ts');
    const controller = new VectorTileLiveController({getReserveMemory: () => null, getCreateBuffer: () => null});
    class FakeMVTLayer { static layerName = 'MVTLayer'; constructor(id, props = {}) { this.id = id; this.props = props; } clone(props) { return new FakeMVTLayer(this.id, {...this.props, ...props}); } }
    const wrapped = controller.wrapLayer(new FakeMVTLayer('live-tiles'));
    const tiles = decodedTiles.map(tile => { const [z, x, y] = tile.key.split('/').map(Number); return {index: {x, y, z}, content: tile.features}; });
    for (const tile of tiles) wrapped.props.onTileLoad(tile);
    wrapped.props.onViewportLoad(tiles);
    const state = controller.getState()[0];
    wrapped.props.onTileUnload(tiles[0]);
    return {state, afterUnload: controller.getState()[0].workingSet.tileCount, hookState: hook.getVectorTileState(), badge: document.querySelector('[data-testid="vector-tile-scope"]')?.textContent ?? null};
  }, tiles);
  assert.equal(live.state.workingSet.tileCount, tiles.length);
  assert.equal(live.afterUnload, tiles.length - 1);
  assert.equal(live.state.completeDatasetDecision.allowed, false);
  assert.ok(Array.isArray(live.hookState));
  // Resident GPU drawing (TILES3-1): geometry upload, GPU filter mask and GPU style color, with Deck sublayers suppressed.
  const resident = await page.evaluate(async decodedTiles => {
    const hook = window.__keplerComputeTest;
    const device = hook?.getVectorTileDevice?.();
    if (!device) return {skipped: 'no WebGPU device in the host runtime'};
    const {VectorTileLiveController} = await import('/src/vector-tile-live-controller.ts');
    const {VectorTileResidentRenderer} = await import('/src/vector-tile-resident-layers.ts');
    const {evaluateVectorTileStyleReference} = await import('/src/vector-tile-resident-style.ts');
    const renderer = new VectorTileResidentRenderer({getDevice: () => device, getReserveMemory: () => null});
    const controller = new VectorTileLiveController({getReserveMemory: () => null, getCreateBuffer: () => null, residentRenderer: renderer});
    class FakeMVTLayer { static layerName = 'MVTLayer'; constructor(id, props = {}) { this.id = id; this.props = props; } clone(props) { return new FakeMVTLayer(this.id, {...this.props, ...props}); } }
    const wrapped = controller.wrapLayer(new FakeMVTLayer('resident-tiles'));
    const sublayers = wrapped.props.renderSubLayers();
    const tiles = decodedTiles.map(tile => { const [z, x, y] = tile.key.split('/').map(Number); return {index: {x, y, z}, content: tile.features}; });
    for (const tile of tiles) wrapped.props.onTileLoad(tile);
    wrapped.props.onViewportLoad(tiles);
    const layers = renderer.getLayers();
    const numeric = controller.getState()[0].numericFieldNames;
    const field = numeric[0];
    const loadedState = renderer.getState()[0];
    const before = await renderer.readFilterMasks('resident-tiles');
    let styleResult = null;
    let colors = null;
    let after = null;
    if (field) {
      controller.setFilters('resident-tiles', [{kind: 'range', field, min: Number.MAX_VALUE, max: Number.MAX_VALUE}]);
      after = await renderer.readFilterMasks('resident-tiles');
      controller.setFilters('resident-tiles', []);
      styleResult = renderer.setStyleFromNativeLayer('resident-tiles', {id: 'resident-tiles', type: 'vectorTile', config: {dataId: 'x', color: [1, 2, 3], colorField: {name: field}, colorScale: 'linear', colorDomain: [0, 1], visConfig: {opacity: 0.8, thickness: 1, colorRange: {colors: ['#000000', '#ffffff']}}}});
      colors = await renderer.readColors('resident-tiles');
    }
    const sum = masks => Object.values(masks ?? {}).reduce((total, mask) => total + mask.reduce((a, b) => a + b, 0), 0);
    let styleMatches = null;
    if (styleResult?.supported && styleResult.style.kind === 'field') {
      styleMatches = true;
      for (const tile of tiles) {
        const gpu = colors[tile.key] ?? colors[Object.keys(colors).find(key => key.endsWith(tile.key))];
        if (!gpu) continue;
        tile.content.forEach((feature, row) => {
          const value = feature.properties?.[field];
          const expected = evaluateVectorTileStyleReference(styleResult.style, typeof value === 'number' ? value : Number.NaN);
          // Packed channels may differ by one from the f32 mix; compare the red channel with tolerance.
          if (Math.abs((gpu[row] & 255) - (expected & 255)) > 1) styleMatches = false;
        });
      }
    }
    return {sublayers: sublayers.length, layerCount: layers.length, loadedState, field, passingBefore: sum(before), passingAfter: after ? sum(after) : null, styleSupported: styleResult?.supported ?? null, styleMatches};
  }, tiles);
  if (resident.skipped) console.log(JSON.stringify({residentSkipped: resident.skipped}));
  else {
    assert.equal(resident.sublayers, 0, 'Deck sublayers not suppressed');
    assert.ok(resident.loadedState.rejectedTiles.length === 0 || resident.layerCount > 0, JSON.stringify(resident.loadedState.rejectedTiles));
    assert.ok(resident.layerCount > 0, 'no resident draw layers');
    if (resident.field) {
      assert.ok(resident.passingBefore > 0, 'GPU mask passed nothing before filtering ' + JSON.stringify(resident));
      assert.equal(resident.passingAfter, 0, 'GPU filter mask did not reject all features');
      assert.equal(resident.styleSupported, true);
      assert.notEqual(resident.styleMatches, false, 'GPU style colors differ from the CPU reference');
    }
  }
  // TILES4-2: style, filters, widths, categories and picking derived automatically from Kepler's Deck props.
  const derived = await page.evaluate(async decodedTiles => {
    const hook = window.__keplerComputeTest;
    const device = hook?.getVectorTileDevice?.();
    if (!device) return {skipped: 'no WebGPU device in the host runtime'};
    const {VectorTileLiveController} = await import('/src/vector-tile-live-controller.ts');
    const {VectorTileResidentRenderer} = await import('/src/vector-tile-resident-layers.ts');
    const {evaluateVectorTileStyleReference} = await import('/src/vector-tile-resident-style.ts');
    const picks = [];
    const renderer = new VectorTileResidentRenderer({getDevice: () => device, getReserveMemory: () => null, onPick: event => picks.push(event.pick)});
    const controller = new VectorTileLiveController({getReserveMemory: () => null, getCreateBuffer: () => null, residentRenderer: renderer});
    class FakeMVTLayer { static layerName = 'MVTLayer'; constructor(id, props = {}) { this.id = id; this.props = props; } clone(props) { return new FakeMVTLayer(this.id, {...this.props, ...props}); } }
    const tiles = decodedTiles.map(tile => { const [z, x, y] = tile.key.split('/').map(Number); return {index: {x, y, z}, content: tile.features}; });
    const numericName = (() => { for (const feature of tiles[0].content) for (const [name, value] of Object.entries(feature.properties ?? {})) if (typeof value === 'number') return name; return null; })();
    const stringName = (() => { for (const feature of tiles[0].content) for (const [name, value] of Object.entries(feature.properties ?? {})) if (typeof value === 'string') return name; return null; })();
    if (!numericName || !stringName) return {skipped: 'tiles carry no numeric and string property'};
    const props = (colorField, colorScale, filterRange) => ({
      opacity: 0.7, stroked: true, filterRange,
      updateTriggers: {
        getFilterValue: {gpuFilter_0: {name: numericName, domain0: 0}, gpuFilter_1: null, gpuFilter_2: null, gpuFilter_3: null},
        getFillColor: {color: [10, 20, 30], colorField: {name: colorField}, colorScale, colorDomain: [0, 1], colorRange: {colors: ['#ff0000', '#00ff00', '#0000ff', '#ffff00']}},
        getLineColor: {strokeColorField: null},
        getLineWidth: {sizeField: {name: numericName}, sizeScale: 'linear', sizeRange: [1, 6], strokeWidth: 1},
        getPointRadius: {radius: 5, radiusField: null}
      }
    });
    const open = [[-1e30, 1e30], [0, 0], [0, 0], [0, 0]];
    let wrapped = controller.wrapLayer(new FakeMVTLayer('auto-tiles', props(stringName, 'ordinal', open)));
    for (const tile of tiles) wrapped.props.onTileLoad(tile);
    wrapped.props.onViewportLoad(tiles);
    renderer.getLayers();
    const state = renderer.getState()[0];
    const colors = await renderer.readColors('auto-tiles');
    const distinct = new Set(Object.values(colors).flat());
    wrapped = controller.wrapLayer(new FakeMVTLayer('auto-tiles', props(stringName, 'ordinal', [[Number.MAX_VALUE, Number.MAX_VALUE], [0, 0], [0, 0], [0, 0]])));
    const masks = await renderer.readFilterMasks('auto-tiles');
    const passing = Object.values(masks).reduce((total, mask) => total + mask.reduce((a, b) => a + b, 0), 0);
    return {state, distinctColors: distinct.size, passingAfterFilter: passing, categoryCounts: state.categoryCounts, styleReasons: state.styleReasons, deckCache: wrapped.props.maxCacheSize, hasPickHandlers: renderer.getLayers().every(layer => layer.props.pickable !== false)};
  }, tiles);
  if (derived.skipped) console.log(JSON.stringify({derivedSkipped: derived.skipped}));
  else {
    assert.ok(derived.state.style.kind === 'field' && derived.state.style.scale === 'ordinal', 'ordinal style not derived automatically ' + JSON.stringify(derived.styleReasons));
    assert.ok(derived.distinctColors > 1, 'categorical colors collapsed to one color');
    assert.equal(derived.passingAfterFilter, 0, 'Kepler GPU filter range was not applied automatically');
    assert.ok(derived.deckCache > 0, 'Deck cache not bounded');
    assert.ok(derived.hasPickHandlers, 'resident layers are not pickable');
  }
  assert.ok(result.afterAdmit.usedBytes <= result.budgetBytes, 'working set exceeds budget');
  assert.ok(result.afterAdmit.evictionCount >= 1, 'budget forced no eviction');
  assert.ok(result.evicted.length >= 1);
  assert.ok(result.identical, 'ids changed after evict/reload');
  assert.equal(result.scope.kind, 'viewport-working-set');
  assert.equal(result.scope.canInferGlobally, false);
  assert.equal(result.global.allowed, false);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, evicted: result.evicted, label: result.scope.label}));
} finally {
  await browser.close();
}
