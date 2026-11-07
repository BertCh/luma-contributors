// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';
import * as arrow from 'apache-arrow';
import {cellToLatLng, latLngToCell} from 'h3-js';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const cellIds = [latLngToCell(40.74, -74.01, 7), latLngToCell(40.77, -73.94, 7), latLngToCell(40.75, -74.4, 5)];
const cells = cellIds.map(id => BigInt(`0x${id}`));
const batch = (identities, measure, category) => new arrow.Table({
  cell: arrow.vectorFromArray(identities, new arrow.Uint64()),
  measure: arrow.makeVector(new Float32Array(measure)),
  category: arrow.vectorFromArray(category, new arrow.Utf8())
});
const source = batch([null, cells[0], 0n], [0, 1, 2], ['skip', 'fine', 'skip']).concat(batch([cells[1], cells[2], null], [3, 4, 5], ['fine', 'coarse', 'skip']));
/** Deck viewports project relative to the map canvas; screenshots and the mouse use page coordinates. */
async function projectToPage(page, longitude, latitude) {
  return page.evaluate(([lng, lat]) => {
    const [x, y] = window.__keplerComputeTest.projectPosition(lng, lat);
    const canvas = [...document.querySelectorAll('canvas')].sort((a, b) => b.clientWidth * b.clientHeight - a.clientWidth * a.clientHeight)[0];
    const rect = canvas.getBoundingClientRect();
    return [x + rect.left, y + rect.top];
  }, [longitude, latitude]);
}
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())) errors.push(message.text());
});
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page, 'Prepare data');
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: 'native-h3.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))});
  await workspace.getByLabel('Source H3 column').selectOption('cell');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 6);
  await closeNativeDialog(page);
  assert.equal(await page.locator('.ReactModal__Overlay:visible').count(), 0, 'Native data dialog must close before map assertions');
  const initial = await page.evaluate(() => window.__keplerComputeTest.getState());
  const datasetId = initial.publications[0].id;
  assert.deepEqual(initial.publications[0].sourceBatches, [3, 3]);
  assert.equal(initial.publications[0].h3.validRowCount, 3);
  assert.equal(initial.publications[0].h3.invalidRows.length, 1);
  const configuration = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const nativeLayer = configuration.config.visState.layers.find(layer => layer.config.dataId === datasetId);
  assert.equal(nativeLayer.type, 'hexagonId');
  for (const layer of configuration.config.visState.layers) layer.config.isVisible = layer === nativeLayer;
  nativeLayer.config.color = [245, 110, 40];
  nativeLayer.config.textLabel = [];
  nativeLayer.config.visConfig = {...nativeLayer.config.visConfig, filled: true, outline: false, enable3d: false, coverage: 1, opacity: 1};
  nativeLayer.visualChannels = {...nativeLayer.visualChannels, colorField: null, coverageField: null, sizeField: null};
  configuration.config.mapState = {...configuration.config.mapState, longitude: -74.02, latitude: 40.755, zoom: 10, pitch: 0, bearing: 0};
  const apply = config => page.evaluate(value => window.__keplerComputeTest.applyNativeConfig(value), config);
  const waitBranch = column => page.waitForFunction(({identifier, column}) => {
    const layers = window.__keplerComputeTest.renderedLayers().filter(layer => layer.id === identifier && layer.visible && layer.loaded);
    return layers.length > 0 && layers.some(layer => layer.layerClass === 'GPUMeshColumnLayer') === column;
  }, {identifier: nativeLayer.id, column}, {timeout: 45000});
  await apply(configuration);
  await waitBranch(false);
  const pick = index => {
    const [latitude, longitude] = cellToLatLng(cellIds[index]);
    return page.evaluate(location => window.__keplerComputeTest.pickPublishedPosition(location.datasetId, location.longitude, location.latitude), {datasetId, longitude, latitude});
  };
  assert.equal((await pick(0))?.sourceRowIndex, 1);
  assert.equal((await pick(1))?.sourceRowIndex, 3);
  // Native polygon sublayers brush per vertex: only cells with a vertex inside the radius draw.
  const countOrangeNear = async (longitude, latitude, halfSize = 45) => {
    const [x, y] = await projectToPage(page, longitude, latitude);
    const image = PNG.sync.read(await page.screenshot());
    let colored = 0;
    for (let vertical = Math.max(0, Math.floor(y - halfSize)); vertical < Math.min(image.height, y + halfSize); vertical++) {
      for (let horizontal = Math.max(0, Math.floor(x - halfSize)); horizontal < Math.min(image.width, x + halfSize); horizontal++) {
        const offset = (vertical * image.width + horizontal) * 4;
        if (image.data[offset] > 180 && image.data[offset + 1] < 160 && image.data[offset + 2] < 100) colored++;
      }
    }
    return {colored, x, y};
  };
  const [nearLatitude, nearLongitude] = cellToLatLng(cellIds[0]);
  const [farLatitude, farLongitude] = cellToLatLng(cellIds[1]);
  const unbrushedNear = (await countOrangeNear(nearLongitude, nearLatitude)).colored;
  const unbrushedFar = (await countOrangeNear(farLongitude, farLatitude)).colored;
  assert(unbrushedNear > 30 && unbrushedFar > 30, `Both res-7 cells draw without brushing (${unbrushedNear}, ${unbrushedFar})`);
  const brushed = structuredClone(configuration);
  brushed.config.visState.interactionConfig.brush = {...brushed.config.visState.interactionConfig.brush, enabled: true, config: {size: 2}};
  await apply(brushed);
  await waitBranch(false);
  const target = await countOrangeNear(nearLongitude, nearLatitude);
  await page.mouse.move(target.x, target.y);
  await page.mouse.move(target.x + 1, target.y + 1);
  await page.waitForTimeout(600);
  const brushedNear = (await countOrangeNear(nearLongitude, nearLatitude)).colored;
  const brushedFar = (await countOrangeNear(farLongitude, farLatitude)).colored;
  assert(brushedNear > unbrushedNear * 0.5, `Cell within the brush radius stays visible (${unbrushedNear} -> ${brushedNear})`);
  assert(brushedFar < unbrushedFar * 0.1, `Cell beyond the brush radius is hidden (${unbrushedFar} -> ${brushedFar})`);
  // Moving the brush onto the other cell swaps visibility.
  const farTarget = await countOrangeNear(farLongitude, farLatitude, 1);
  await page.mouse.move(farTarget.x, farTarget.y);
  await page.mouse.move(farTarget.x + 1, farTarget.y + 1);
  await page.waitForTimeout(600);
  assert((await countOrangeNear(nearLongitude, nearLatitude)).colored < unbrushedNear * 0.1, 'Former cell hides after the brush moves');
  assert((await countOrangeNear(farLongitude, farLatitude)).colored > unbrushedFar * 0.5, 'New cell shows under the brush');
  await page.screenshot({path: '/tmp/luma-kepler-h3-brushed.png'});
  // Brushing is render-only: it must not change selection or picking identity.
  assert.equal((await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].selectedCount)), 6);
  await page.mouse.move(2, 2);
  await apply(configuration);
  await waitBranch(false);
  await page.mouse.move(1430, 990);
  // Ordinary native GPU range filtering must not change H3's CPU topology decision.
  await page.evaluate(id => window.__keplerComputeTest.setPublishedRange(id, 'measure', 1, 3), datasetId);
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0].selectedCount === 3);
  await waitBranch(false);
  // Native categorical CPU filtering removes the only lower-resolution cell.
  const lowConfiguration = structuredClone(configuration);
  lowConfiguration.config.visState.filters.push({id: 'h3-category', dataId: [datasetId], name: ['category'], type: 'multiSelect', value: ['fine'], fixedDomain: true});
  await apply(lowConfiguration);
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0].selectedCount === 2);
  await waitBranch(true);
  assert.equal((await pick(0))?.sourceRowIndex, 1);
  assert.equal((await pick(1))?.sourceRowIndex, 3);
  const countPixels = async () => {
    const image = PNG.sync.read(await page.screenshot());
    let colored = 0;
    for (let vertical = 100; vertical < 850; vertical++) for (let horizontal = 350; horizontal < 1050; horizontal++) {
      const offset = (vertical * image.width + horizontal) * 4;
      if (image.data[offset] > 180 && image.data[offset + 1] < 160 && image.data[offset + 2] < 100) colored++;
    }
    return colored;
  };
  const fullPixels = await countPixels();
  assert(fullPixels > 30, `H3 columns must render visible pixels; found ${fullPixels}`);
  await page.screenshot({path: '/tmp/luma-kepler-h3-visible.png'});
  lowConfiguration.config.visState.layers.find(layer => layer.id === nativeLayer.id).config.visConfig.coverage = 0.4;
  await apply(lowConfiguration);
  await waitBranch(true);
  const reducedPixels = await countPixels();
  assert(reducedPixels > 0 && reducedPixels < fullPixels * 0.7, `Native coverage should shrink cell footprints: ${fullPixels} -> ${reducedPixels}`);
  const initialMesh = await page.evaluate(id => window.__keplerComputeTest.getH3ViewportMesh(id), nativeLayer.id);
  const farViewport = {...lowConfiguration.config.mapState, longitude: -73.6, latitude: 40.9};
  await page.evaluate(viewport => window.__keplerComputeTest.setMapViewport(viewport), farViewport);
  const expectedCenter = latLngToCell(farViewport.latitude, farViewport.longitude, 7);
  await page.waitForFunction(({identifier, expectedCenter}) => window.__keplerComputeTest.getH3ViewportMesh(identifier)?.centerCellId === expectedCenter,
    {identifier: nativeLayer.id, expectedCenter});
  assert.notEqual(expectedCenter, initialMesh.centerCellId);
  // A single final viewport update must retain the previous mesh for native sub-10km movement.
  await page.evaluate(viewport => window.__keplerComputeTest.setMapViewport(viewport), {...farViewport, longitude: farViewport.longitude + 0.001});
  await page.waitForFunction(({identifier, expectedCenter}) => window.__keplerComputeTest.getH3ViewportMesh(identifier)?.centerCellId === expectedCenter,
    {identifier: nativeLayer.id, expectedCenter});
  await page.evaluate(viewport => window.__keplerComputeTest.setMapViewport(viewport), lowConfiguration.config.mapState);
  await waitBranch(true);
  await page.waitForFunction(({identifier, expectedCenter}) => window.__keplerComputeTest.getH3ViewportMesh(identifier)?.centerCellId === expectedCenter,
    {identifier: nativeLayer.id, expectedCenter: latLngToCell(lowConfiguration.config.mapState.latitude, lowConfiguration.config.mapState.longitude, 7)});
  assert(await countPixels() > 0, 'Restored camera must show the selected cells');
  const final = await page.evaluate(() => window.__keplerComputeTest.getState());
  assert.deepEqual(final.publications[0].h3.sourceBufferIds, initial.publications[0].h3.sourceBufferIds);
  assert.equal(final.publications[0].h3.sourceUploadBytes, initial.publications[0].h3.sourceUploadBytes);
  assert.deepEqual(final.publications[0].sourceBufferIds, initial.publications[0].sourceBufferIds);
  assert.equal(final.error, '');
  for (const [configurationToExtrude, columnBranch] of [[configuration, false], [lowConfiguration, true]]) {
    const extrudedConfiguration = structuredClone(configurationToExtrude);
    const extrudedLayer = extrudedConfiguration.config.visState.layers.find(layer => layer.id === nativeLayer.id);
    extrudedLayer.config.visConfig.enable3d = true;
    extrudedLayer.config.visConfig.elevationScale = 1;
    extrudedConfiguration.config.mapState.pitch = 35;
    await apply(extrudedConfiguration);
    await page.waitForFunction(expected => window.__keplerComputeTest.getState().publications[0].selectedCount === expected, columnBranch ? 2 : 6);
    await waitBranch(columnBranch);
    await page.waitForFunction(identifier => window.__keplerComputeTest.renderedLayers().some(layer => layer.id === identifier && layer.visible && layer.loaded && layer.extruded), nativeLayer.id);
    const extrudedImage = PNG.sync.read(await page.screenshot({path: `/tmp/luma-kepler-h3-${columnBranch ? 'columns' : 'polygons'}-extruded.png`}));
    let extrudedPixels = 0;
    for (let vertical = 100; vertical < 850; vertical++) for (let horizontal = 350; horizontal < 1050; horizontal++) {
      const offset = (vertical * extrudedImage.width + horizontal) * 4;
      const [red, green, blue] = extrudedImage.data.subarray(offset, offset + 3);
      if (red > 70 && red > green * 1.3 && green > blue * 1.2) extrudedPixels++;
    }
    assert(extrudedPixels > 20, `Native ${columnBranch ? 'column' : 'polygon'} extrusion must render; found ${extrudedPixels}`);
  }
  await apply(lowConfiguration);
  await waitBranch(true);
  assert(await countPixels() > 0);
  assert.deepEqual(errors, []);
  await page.screenshot({path: '/tmp/luma-kepler-h3.png'});
  console.log(JSON.stringify({passed: true, fullPixels, reducedPixels, checks: ['exact Uint64 IDs', 'source batches/null/invalid slots', 'native high and column branches', 'polygon-branch per-vertex brushing', 'native GPU filter topology exclusion', 'category topology change', 'coverage', 'final pan mesh/cache', 'native polygon and column extrusion', 'source picking', 'stable upload']}));
} catch (error) {
  console.error({pageErrors: errors});
  console.error(JSON.stringify(await page.evaluate(() => ({state: window.__keplerComputeTest?.getState(), layers: window.__keplerComputeTest?.renderedLayers(), h3: window.__keplerComputeTest?.getState().nativeLayers?.filter(layer => layer.type === 'hexagonId').map(layer => ({layerId: layer.id, metadata: window.__keplerComputeTest.getH3Metadata(layer.id)}))})), null, 2));
  console.error(await page.locator('body').innerText());
  throw error;
} finally {
  await browser.close();
}
