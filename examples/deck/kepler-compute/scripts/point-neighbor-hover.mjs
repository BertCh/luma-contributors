// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// LAYERS4-A1: a point layer with showNeighborOnHover and a list<int> neighbors column dims the layer
// to 0.2 while a point is hovered and draws a `-hovered` overlay (neighbors plus the hovered row);
// moving the pointer away restores the layer. Headless WebGPU only.
import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const longitudes = [-74.03, -74.0, -73.97, -73.94];
const latitudes = [40.72, 40.75, 40.72, 40.75];
const neighbors = [[1, 2], [0, 3], [], [1]];
const source = arrow.tableFromArrays({
  longitude: new Float32Array(longitudes),
  latitude: new Float32Array(latitudes),
  neighbors: arrow.vectorFromArray(neighbors, new arrow.List(new arrow.Field('item', new arrow.Int32(), true)))
});
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const failures = [];
page.on('pageerror', error => failures.push(error.message));
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())) failures.push(message.text());
});
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: 'neighbors.arrow', mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))});
  await workspace.getByLabel('Source longitude column').selectOption('longitude');
  await workspace.getByLabel('Source latitude column').selectOption('latitude');
  // The list column stays in the retained source table for hover; only scalar columns upload.
  const neighborsBox = workspace.getByRole('checkbox', {name: /^neighbors/});
  if ((await neighborsBox.count()) && (await neighborsBox.isChecked())) await neighborsBox.uncheck();
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await page.waitForTimeout(1500);
  if (await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).isDisabled()) console.error((await workspace.innerText()).split('\n').slice(40).join('\n'));
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 4, undefined, {timeout: 45000});
  await closeNativeDialog(page);
  const datasetId = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);
  const configuration = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const layer = configuration.config.visState.layers.find(candidate => candidate.config.dataId === datasetId);
  for (const candidate of configuration.config.visState.layers) candidate.config.isVisible = candidate === layer;
  layer.config.columnMode = 'points';
  layer.config.columns = {...layer.config.columns, lat: 'latitude', lng: 'longitude', neighbors: 'neighbors'};
  layer.config.textLabel = [];
  layer.config.color = [245, 110, 40];
  layer.config.visConfig = {...layer.config.visConfig, radius: 400, fixedRadius: true, showNeighborOnHover: true, allowHover: true, opacity: 1};
  layer.visualChannels = {...layer.visualChannels, colorField: null, sizeField: null};
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), configuration);
  await page.evaluate(() => window.__keplerComputeTest.setMapViewport({longitude: -73.985, latitude: 40.735, zoom: 11.5}));
  const layerId = layer.id;
  const rendered = () => page.evaluate(id => window.__keplerComputeTest.renderedLayers().filter(candidate => candidate.id === id).map(candidate => ({renderId: candidate.renderId, layerClass: candidate.layerClass, opacity: candidate.opacity, visible: candidate.visible, loaded: candidate.loaded})), layerId);
  await page.waitForFunction(id => {
    const api = window.__keplerComputeTest;
    if (api.getState().error) throw new Error(api.getState().error);
    return api.renderedLayers().some(candidate => candidate.id === id && candidate.visible && candidate.loaded);
  }, layerId, {timeout: 45000});
  const idle = await rendered();
  assert(!idle.some(candidate => candidate.renderId.endsWith('-hovered')), 'no overlay before hover');
  const baseOpacity = idle.find(candidate => candidate.layerClass.includes('Scatterplot'))?.opacity;
  const point = await page.evaluate(([longitude, latitude]) => window.__keplerComputeTest.projectPosition(longitude, latitude), [longitudes[0], latitudes[0]]);
  await page.mouse.move(point[0] - 40, point[1] - 40);
  await page.mouse.move(point[0], point[1], {steps: 6});
  await page.waitForFunction(id => window.__keplerComputeTest.renderedLayers().some(candidate => candidate.renderId.startsWith(id) && candidate.renderId.endsWith('-hovered')), layerId, {timeout: 15000}).catch(async () => {
    throw new Error(`hover overlay never appeared; layers ${JSON.stringify(await rendered())}; hover ${JSON.stringify(await page.evaluate(() => window.__keplerComputeTest.getHoverInfo()))}`);
  });
  const hovered = await rendered();
  const dimmed = hovered.find(candidate => candidate.layerClass.includes('Scatterplot') && !candidate.renderId.endsWith('-hovered'));
  assert.equal(dimmed?.opacity, 0.2, `base layer dims to 0.2 (was ${baseOpacity})`);
  const overlay = await page.evaluate(id => window.__keplerComputeTest.renderedLayers().find(candidate => candidate.renderId.startsWith(id) && candidate.renderId.endsWith('-hovered')), layerId);
  assert(overlay?.visible, `overlay is visible ${JSON.stringify(overlay)}`);
  const overlayData = await page.evaluate(id => {
    const layers = window.__keplerComputeTest.renderedLayers().filter(candidate => candidate.renderId.endsWith('-hovered'));
    return layers.length;
  }, layerId);
  assert(overlayData >= 1);
  await page.mouse.move(5, 5, {steps: 4});
  await page.waitForFunction(id => !window.__keplerComputeTest.renderedLayers().some(candidate => candidate.renderId.startsWith(id) && candidate.renderId.endsWith('-hovered')), layerId, {timeout: 15000});
  const restored = (await rendered()).find(candidate => candidate.layerClass.includes('Scatterplot'));
  assert.equal(restored?.opacity, baseOpacity, 'opacity restored after hover ends');
  assert.equal(await page.evaluate(() => window.__keplerComputeTest.getState().error), '');
  assert.deepEqual(failures, []);
  console.log(JSON.stringify({passed: true, baseOpacity, dimmedOpacity: dimmed.opacity}));
} catch (error) {
  console.error({failures});
  console.error(error);
  process.exitCode = 1;
} finally {
  await browser.close();
}
