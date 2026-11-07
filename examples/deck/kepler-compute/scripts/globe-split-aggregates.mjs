// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Globe aggregates (grid, hexagon, heatmap) inside Single, Swipe and Dual maps. A wide three-point
// source is published once; each aggregate layer type must draw magenta pixels on both sides of the
// divider when its layer is in both maps, and only on its own side when it is removed from one map.
// Browser verification pending; run through the central tester.
//
// Usage: node scripts/globe-split-aggregates.mjs [url] [layerType ...]
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';
import {GLOBE_SPLIT_AGGREGATES, GLOBE_SPLIT_REGIONS} from './globe-split-matrix.mjs';

const args = process.argv.slice(2);
const url = args.find(value => value.startsWith('http')) || 'http://127.0.0.1:5318/?rows=256';
const requested = args.filter(value => !value.startsWith('http'));
const names = requested.length ? requested : Object.keys(GLOBE_SPLIT_AGGREGATES);
const CENTER = {longitude: -30, latitude: 20};

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => !/send was called before connect/.test(error.message) && errors.push(error.message));
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text()) && !/previous error/.test(message.text()) && errors.length < 3) errors.push(message.text().slice(0, 1000));
});
const watchdog = setTimeout(() => { console.error('Globe split aggregates exceeded 280 seconds'); void browser.close(); process.exit(1); }, 280000);

const isMagenta = (red, green, blue) => red > 150 && green < 110 && blue > 90 && red - green > 90;
async function countRegions() {
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications.every(publication => publication.globeGeometry?.state !== 'preparing'), undefined, {timeout: 45000});
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const image = PNG.sync.read(await page.screenshot());
  const counts = {};
  for (const [name, [left, right]] of Object.entries(GLOBE_SPLIT_REGIONS)) {
    let pixels = 0;
    // The legend panel (bottom right, below y=700) has magenta swatches; count only the globe area.
    for (let vertical = 60; vertical < Math.min(image.height - 40, 690); vertical++)
      for (let horizontal = left; horizontal < right; horizontal++) {
        const offset = (vertical * image.width + horizontal) * 4;
        if (isMagenta(image.data[offset], image.data[offset + 1], image.data[offset + 2])) pixels++;
      }
    counts[name] = pixels;
  }
  return counts;
}
const chooseSplitMode = async title => {
  await page.locator('.map-control-button.split-map').click();
  await page.locator('.toolbar-item', {hasText: new RegExp(`^${title}$`)}).click();
};
const waitForMode = async mode => {
  await page.waitForFunction(expected => window.__keplerComputeTest.saveNativeConfig().config.mapState.mapSplitMode === expected, mode);
  await page.waitForTimeout(2500);
};
const getLayerMaps = layerId => page.evaluate(id => window.__keplerComputeTest.saveNativeConfig().config.visState.splitMaps.map(map => Boolean(map?.layers?.[id])), layerId);
async function setLayerMaps(layerId, left, right) {
  const present = await getLayerMaps(layerId);
  if (present[0] !== left) await page.evaluate(id => window.__keplerComputeTest.toggleLayerForMap(0, id), layerId);
  if (present[1] !== right) await page.evaluate(id => window.__keplerComputeTest.toggleLayerForMap(1, id), layerId);
  await page.waitForTimeout(1800);
}

try {
  await page.goto(url, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  const table = arrow.tableFromArrays({
    longitude: new Float32Array([CENTER.longitude - 25, CENTER.longitude, CENTER.longitude + 25]),
    latitude: new Float32Array([CENTER.latitude - 12, CENTER.latitude + 8, CENTER.latitude - 4]),
    measure: new Float32Array([1, 1, 1])
  });
  await workspace.locator('input[type=file]').setInputFiles({name: 'globe-split.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(table, 'stream'))});
  await workspace.getByLabel('Source longitude column').selectOption('longitude');
  await workspace.getByLabel('Source latitude column').selectOption('latitude');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 3, undefined, {timeout: 45000});
  await closeNativeDialog(page);
  const datasetId = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);

  const results = [];
  for (const name of names) {
    const scenario = GLOBE_SPLIT_AGGREGATES[name];
    assert(scenario, `Unknown layer type ${name}`);
    const config = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
    const layer = config.config.visState.layers.find(candidate => candidate.config.dataId === datasetId);
    for (const candidate of config.config.visState.layers) candidate.config.isVisible = candidate === layer;
    layer.type = scenario.layerType;
    layer.config.columnMode = 'points';
    layer.config.columns = {lat: 'latitude', lng: 'longitude'};
    layer.config.color = [255, 20, 170];
    layer.config.visConfig = {...layer.config.visConfig, ...scenario.visConfig};
    layer.visualChannels = {...layer.visualChannels, colorField: null, sizeField: null, heightField: null, coverageField: null, strokeColorField: null};
    config.config.mapState = {...config.config.mapState, ...CENTER, zoom: scenario.zoom, pitch: 0, bearing: 0, dragRotate: false,
      mapSplitMode: 'SINGLE_MAP', mapViewMode: 'MODE_GLOBE', globe: {...(config.config.mapState.globe ?? {}), enabled: true}};
    await page.evaluate(value => window.__keplerComputeTest.applyNativeConfig(value), config);
    await page.waitForFunction(id => window.__keplerComputeTest.renderedLayers().some(rendered => rendered.id.startsWith(id) && rendered.visible), layer.id, {timeout: 45000});
    await page.waitForTimeout(2000);

    const single = await countRegions();
    assert(single.full > scenario.minPixels, `${name}: single globe must draw (${JSON.stringify(single)})`);
    const record = {name, single: single.full};
    for (const [title, mode] of [['Swipe', 'SWIPE_COMPARE'], ['Dual', 'DUAL_MAP']]) {
      await chooseSplitMode(title);
      await waitForMode(mode);
      await setLayerMaps(layer.id, true, true);
      const both = await countRegions();
      assert(both.left > scenario.minPixels / 4 && both.right > scenario.minPixels / 4, `${name}/${title}: both halves must draw (${JSON.stringify(both)})`);
      await setLayerMaps(layer.id, true, false);
      const leftOnly = await countRegions();
      assert(leftOnly.left > scenario.minPixels / 4 && leftOnly.right < 20, `${name}/${title}: map 0 only (${JSON.stringify(leftOnly)})`);
      await setLayerMaps(layer.id, false, true);
      const rightOnly = await countRegions();
      assert(rightOnly.right > scenario.minPixels / 4 && rightOnly.left < 20, `${name}/${title}: map 1 only (${JSON.stringify(rightOnly)})`);
      record[title] = {both, leftOnly, rightOnly};
      await chooseSplitMode('Single');
      await waitForMode('SINGLE_MAP');
    }
    const state = await page.evaluate(() => window.__keplerComputeTest.getState());
    assert.equal(state.error, '', `${name}: ${state.error}`);
    assert.deepEqual(errors, []);
    results.push(record);
  }
  console.log(JSON.stringify({passed: true, results}, null, 1));
} catch (error) {
  console.error(error.message, errors);
  await page.screenshot({path: '/tmp/luma-kepler-globe-split-failure.png'}).catch(() => {});
  process.exitCode = 1;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
