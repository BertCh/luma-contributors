// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';
import {setAnalysisSheetControl} from './analysis-sheet-controls.mjs';

// Projected centers near New York. The first two circles overlap; the third has zero area.
const centerX = -8237642;
const centerY = 4970352;
const xValues = [centerX, centerX + 8, centerX + 200, centerX + 300, centerX + 400];
const bounds = [-8237700, 4970300, -8237200, 4970400];
const expectedRadius = Math.sqrt(10000 / Math.PI);
function unprojectMercator(x, y) {
  return [x / 6378137 * 180 / Math.PI, Math.atan(Math.sinh(y / 6378137)) * 180 / Math.PI];
}
const source = arrow.tableFromArrays({
  x: new Float32Array(xValues.slice(0, 2)),
  y: new Float32Array([centerY, centerY]),
  area: arrow.vectorFromArray([100, 100], new arrow.Float32())
}).concat(arrow.tableFromArrays({
  x: new Float32Array(xValues.slice(2)),
  y: new Float32Array([centerY, centerY, centerY]),
  area: arrow.vectorFromArray([0, null, -1], new arrow.Float32())
}));

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const failures = [];
const diagnostics = [];
page.on('pageerror', error => {
  failures.push(error.message);
  console.error('pageerror:', error.stack ?? error.message);
});
page.on('console', message => {
  if (message.type() !== 'error') return;
  diagnostics.push(message.text());
  if (/WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())) failures.push(message.text());
  console.error('console:', message.text());
});
page.on('requestfailed', request => diagnostics.push(`${request.url()}: ${request.failure()?.errorText}`));
let currentStage = 'startup';
function stage(name) {
  currentStage = name;
  console.log(JSON.stringify({stage: name}));
}
const watchdog = setTimeout(() => {
  console.error(`Cartogram workflow exceeded 120 seconds at ${currentStage}`);
  void browser.close();
}, 120000);

async function waitForInitialLayout() {
  await page.waitForFunction(() => {
    const text = document.querySelector('[aria-label="Cartogram results"]')?.textContent ?? '';
    return text.includes('Iteration budget exhausted; finite layout available') &&
      text.includes('3 eligible circles') && text.includes('2 excluded or unselected source rows') &&
      text.includes('0 iterations');
  }, undefined, {timeout: 30000});
}

try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  stage('publish multibatch projected table through Prepare data');
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: 'cartogram-analysis.arrow', mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))
  });
  await workspace.getByLabel('Source geometry column', {exact: true}).selectOption('');
  await workspace.getByLabel('Source longitude column', {exact: true}).selectOption('');
  await workspace.getByLabel('Source latitude column', {exact: true}).selectOption('');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 5, undefined, {timeout: 45000});
  const datasetId = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].id);
  await closeNativeDialog(page);
  const panel = page.getByRole('region', {name: 'GPU analysis', exact: true});
  await panel.getByLabel('Dataset', {exact: true}).selectOption(datasetId);
  await panel.getByText('Configure Dorling circle cartogram', {exact: true}).click();
  await panel.locator('.analysis-cartogram-panel').getByTestId('ax-analysis-sheet').waitFor();
  await setAnalysisSheetControl(panel.getByLabel('Cartogram projected X field', {exact: true}), 'x');
  await setAnalysisSheetControl(panel.getByLabel('Cartogram projected Y field', {exact: true}), 'y');
  await setAnalysisSheetControl(panel.getByLabel('Cartogram nonnegative area-value field', {exact: true}), 'area');
  await setAnalysisSheetControl(panel.getByLabel('Cartogram declared projected CRS', {exact: true}), 'EPSG:3857');
  for (const [index, label] of ['minimum X', 'minimum Y', 'maximum X', 'maximum Y'].entries())
    await setAnalysisSheetControl(panel.getByLabel(`Cartogram ${label} in projected meters`, {exact: true}), String(bounds[index]));
  await setAnalysisSheetControl(panel.getByLabel('Cartogram area scale in projected square meters per value', {exact: true}), '100');
  await setAnalysisSheetControl(panel.getByLabel('Cartogram maximum iterations', {exact: true}), '0');
  await setAnalysisSheetControl(panel.getByLabel('Cartogram overlap tolerance in projected meters', {exact: true}), '1');
  await setAnalysisSheetControl(panel.getByLabel('Cartogram attraction', {exact: true}), '0');
  stage('retain finite overlapping initial layout after zero iterations');
  await panel.getByRole('button', {name: 'Apply fields and run cartogram', exact: true}).click();
  await waitForInitialLayout();
  assert.equal(await panel.getByLabel('Operation', {exact: true}).inputValue(), 'dorling-cartogram');
  const report = panel.getByRole('region', {name: 'Cartogram results', exact: true});
  const metricValues = await report.locator('dl').evaluate(list => Object.fromEntries(
    [...list.querySelectorAll('dt')].map(term => [term.textContent, term.nextElementSibling.textContent])
  ));
  assert(Number.parseFloat(metricValues['Maximum remaining overlap penetration']) > 100);
  assert.equal(Number.parseFloat(metricValues['Maximum displacement from original centers']), 0);
  assert.equal(metricValues['Declared projected CRS'], 'EPSG:3857');
  await panel.getByRole('button', {name: 'Save recipe', exact: true}).click();
  const recipe = JSON.parse(await panel.getByLabel('Saved analysis recipe', {exact: true}).inputValue());
  assert.equal(recipe.source.datasetId, datasetId);
  assert.equal(recipe.population, 'all');
  assert.deepEqual(recipe.operation, {
    kind: 'dorling-cartogram', coordinateSpace: 'planar', metric: 'native', units: 'meters',
    xColumn: 'x', yColumn: 'y', valueColumn: 'area', crs: 'EPSG:3857', bounds,
    areaScale: 100, maximumIterations: 0, overlapTolerance: 1, attraction: 0
  });
  stage('restore saved initial-layout recipe through native analysis controls');
  await panel.getByLabel('Operation', {exact: true}).selectOption('z-score');
  await page.waitForFunction(() => !document.querySelector('[aria-label="Cartogram results"]'));
  await panel.getByRole('button', {name: 'Restore recipe', exact: true}).click();
  await waitForInitialLayout();
  assert.equal(await panel.getByLabel('Cartogram maximum iterations', {exact: true}).inputValue(), '0');
  assert.equal(await panel.getByLabel('Cartogram nonnegative area-value field', {exact: true}).inputValue(), 'area');
  assert.equal(await panel.getByLabel('Cartogram declared projected CRS', {exact: true}).inputValue(), 'EPSG:3857');
  stage('set Web Mercator camera and compare cartogram pixels with overlay disabled');
  const location = unprojectMercator(centerX, centerY);
  const cameraDispatch = await page.evaluate(({location, datasetId}) => {
    const api = window.__keplerComputeTest;
    const sourceRevision = api.getPerformanceSnapshot(datasetId).sourceRevision;
    api.setMapViewport({longitude: location[0], latitude: location[1], zoom: 18, bearing: 0, pitch: 0});
    return {sourceRevision, dispatchedAt: performance.now()};
  }, {location, datasetId});
  await page.waitForFunction(({datasetId, sourceRevision, dispatchedAt}) => {
    const snapshot = window.__keplerComputeTest.getPerformanceSnapshot(datasetId);
    return snapshot.sourceRevision === sourceRevision && snapshot.frame?.stage === 'submitted' &&
      snapshot.frame.sourceRevision === sourceRevision && snapshot.frame.timestamp >= dispatchedAt;
  }, {datasetId, ...cameraDispatch}, {timeout: 30000});
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => resolve())));
  await page.waitForFunction(id => window.__keplerComputeTest.renderedLayers().some(
    layer => layer.id === `resident-cartogram-${id}` && layer.layerClass === 'CartogramMercatorLayer' && layer.visible && layer.loaded
  ), datasetId, {timeout: 30000});
  const screenPositions = await page.evaluate(locations => locations.map(
    location => window.__keplerComputeTest.projectPosition(...location)
  ), [location, unprojectMercator(centerX + 8, centerY), unprojectMercator(centerX + expectedRadius, centerY), unprojectMercator(centerX - 53, centerY)]);
  const [firstCenter, secondCenter, radiusEdge, exposedPoint] = screenPositions;
  const screenRadius = Math.abs(radiusEdge[0] - firstCenter[0]);
  assert(screenRadius > 100 && screenRadius < 300, `Unexpected projected radius: ${screenRadius}`);
  const viewport = page.viewportSize();
  const left = Math.max(0, Math.floor(firstCenter[0] - screenRadius * 1.3));
  const top = Math.max(0, Math.floor(firstCenter[1] - screenRadius * 1.3));
  const clip = {
    x: left, y: top,
    width: Math.min(viewport.width - left, Math.ceil(secondCenter[0] + screenRadius * 1.3) - left),
    height: Math.min(viewport.height - top, Math.ceil(firstCenter[1] + screenRadius * 1.3) - top)
  };
  const overlayPixels = PNG.sync.read(await page.screenshot({clip, scale: 'css'}));
  const offFrame = await page.evaluate(() => window.__keplerComputeTest.getState().frameCount);
  await panel.getByLabel('Operation', {exact: true}).selectOption('z-score');
  await page.waitForFunction(({datasetId, frame}) => window.__keplerComputeTest.getState().frameCount > frame &&
    !window.__keplerComputeTest.renderedLayers().some(layer => layer.id === `resident-cartogram-${datasetId}`), {datasetId, frame: offFrame});
  const backgroundPixels = PNG.sync.read(await page.screenshot({clip, scale: 'css'}));
  assert.equal(overlayPixels.width, backgroundPixels.width);
  assert.equal(overlayPixels.height, backgroundPixels.height);
  let changedInside = 0;
  let blueInside = 0;
  let innerPixelCount = 0;
  let changedOutside = 0;
  let outerPixelCount = 0;
  for (let vertical = 0; vertical < overlayPixels.height; vertical++) {
    for (let horizontal = 0; horizontal < overlayPixels.width; horizontal++) {
      const offset = (vertical * overlayPixels.width + horizontal) * 4;
      const screenX = clip.x + horizontal + 0.5;
      const screenY = clip.y + vertical + 0.5;
      const distance = Math.min(...[firstCenter, secondCenter].map(center => Math.hypot(screenX - center[0], screenY - center[1])));
      const changed = Math.abs(overlayPixels.data[offset] - backgroundPixels.data[offset]) +
        Math.abs(overlayPixels.data[offset + 1] - backgroundPixels.data[offset + 1]) +
        Math.abs(overlayPixels.data[offset + 2] - backgroundPixels.data[offset + 2]) > 35;
      if (distance < screenRadius * 0.85) {
        innerPixelCount++;
        if (changed) changedInside++;
        if (changed && overlayPixels.data[offset + 2] > overlayPixels.data[offset] + 20 &&
          overlayPixels.data[offset + 2] > overlayPixels.data[offset + 1] + 5) blueInside++;
      } else if (distance > screenRadius * 1.15) {
        outerPixelCount++;
        if (changed) changedOutside++;
      }
    }
  }
  assert(changedInside > 1000 && blueInside > 1000, `Missing blue circle pixels: changed=${changedInside}, blue=${blueInside}`);
  assert(changedInside / innerPixelCount > 0.7, 'Most predicted circle interiors must change when the overlay is removed');
  assert(changedOutside / outerPixelCount < 0.1, 'Pixels outside the predicted circles must remain mostly unchanged');
  console.log(JSON.stringify({stage: 'cartogram pixel comparison', changedInside, blueInside, innerPixelCount, changedOutside, outerPixelCount}));
  const restoreFrame = await page.evaluate(() => window.__keplerComputeTest.getState().frameCount);
  await panel.getByRole('button', {name: 'Restore recipe', exact: true}).click();
  await waitForInitialLayout();
  await page.waitForFunction(({datasetId, frame}) => window.__keplerComputeTest.getState().frameCount > frame &&
    window.__keplerComputeTest.renderedLayers().some(layer => layer.id === `resident-cartogram-${datasetId}` && layer.visible && layer.loaded), {datasetId, frame: restoreFrame});
  const restoredCenter = await page.evaluate(location => window.__keplerComputeTest.projectPosition(...location), location);
  assert(Math.hypot(restoredCenter[0] - firstCenter[0], restoredCenter[1] - firstCenter[1]) < 0.5, 'Overlay comparison must use the same camera');
  stage('hover and pin the exposed left circle through actual map events');
  await page.mouse.move(exposedPoint[0], exposedPoint[1], {steps: 4});
  const inspection = panel.getByRole('region', {name: 'Cartogram circle inspection', exact: true});
  await inspection.getByRole('heading', {name: 'Hovered circle', exact: true}).waitFor();
  const inspectedValues = await inspection.locator('dl').evaluate(list => Object.fromEntries(
    [...list.querySelectorAll('dt')].map(term => [term.textContent, term.nextElementSibling.textContent])
  ));
  assert.equal(inspectedValues['Source dataset'], datasetId);
  assert.equal(inspectedValues['Source row'], '0');
  assert.equal(inspectedValues['Source ID'], '0');
  assert(Math.abs(Number.parseFloat(inspectedValues['Computed circle radius']) - expectedRadius) < 0.001);
  await page.mouse.click(exposedPoint[0], exposedPoint[1]);
  await inspection.getByRole('heading', {name: 'Pinned circle', exact: true}).waitFor();
  await page.mouse.move(20, 20);
  await inspection.getByRole('heading', {name: 'Pinned circle', exact: true}).waitFor();
  await inspection.getByRole('button', {name: 'Clear inspection', exact: true}).click();
  await inspection.waitFor({state: 'hidden'});
  stage('inspect source-aligned centers, radii and identities in the native table');
  await page.locator('.show-data-table svg').last().click();
  await page.waitForFunction(() => {
    const table = document.querySelector('.resident-table-scroll');
    return table?.textContent.includes('Analysis: cartogram radius meters') && table.querySelectorAll('tbody tr').length === 5;
  });
  const rows = await page.locator('.resident-table-scroll').evaluate(table => {
    const headers = [...table.querySelectorAll('thead th')];
    const column = label => headers.findIndex(header => header.textContent.includes(label));
    const centerXColumn = column('Analysis: cartogram center x');
    const centerYColumn = column('Analysis: cartogram center y');
    const radiusColumn = column('Analysis: cartogram radius meters');
    const identityColumn = column('Analysis: cartogram source id');
    return [...table.querySelectorAll('tbody tr')].map(row => ({
      sourceRow: row.querySelector('th').textContent,
      centerX: row.children[centerXColumn]?.textContent,
      centerY: row.children[centerYColumn]?.textContent,
      radius: row.children[radiusColumn]?.textContent,
      sourceId: row.children[identityColumn]?.textContent
    }));
  });
  assert.deepEqual(rows.map(row => row.sourceRow), ['0', '1', '2', '3', '4']);
  assert.deepEqual(rows.map(row => Number(row.sourceId)), [0, 1, 2, 3, 4]);
  assert.deepEqual(rows.slice(0, 3).map(row => Number(row.centerX)), xValues.slice(0, 3));
  assert.deepEqual(rows.slice(0, 3).map(row => Number(row.centerY)), [centerY, centerY, centerY]);
  for (const row of rows.slice(0, 2)) assert(Math.abs(Number(row.radius) - expectedRadius) < 0.001);
  assert.equal(Number(rows[2].radius), 0);
  for (const row of rows.slice(3)) {
    assert.equal(row.centerX, 'null');
    assert.equal(row.centerY, 'null');
    assert.equal(row.radius, 'null');
  }
  await page.keyboard.press('Escape');
  await page.locator('.resident-table-modal').waitFor({state: 'hidden'});
  assert.deepEqual(await page.evaluate(() => window.__keplerComputeTest.getState().publications[0].sourceBatches), [2, 3]);
  stage('remove cartogram source and registered result layer');
  await openNativeData(page);
  await workspace.getByRole('button', {name: 'Remove cartogram-analysis.arrow', exact: true}).click();
  await page.waitForFunction(id => window.__keplerComputeTest.getState().publications.length === 0 &&
    !window.__keplerComputeTest.renderedLayers().some(layer => layer.id === `resident-cartogram-${id}`), datasetId);
  await closeNativeDialog(page);
  assert.deepEqual(failures, []);
  console.log(JSON.stringify({passed: true, checks: [
    'native multibatch projected-table publication and explicit cartogram settings',
    'finite exhausted layout with residual overlap after zero iterations',
    'recipe replay and source-aligned centers/radii/identities with zero/null/negative values',
    'Web Mercator blue circle pixels compared with overlay disabled at a fixed camera',
    'actual exposed-circle hover, click pinning, mouseout persistence and clear inspection',
    'source batch preservation and dataset/layer removal'
  ]}));
} catch (failure) {
  const renderSnapshot = await page.evaluate(location => {
    const api = window.__keplerComputeTest;
    const state = api?.getState();
    const datasetId = state?.publications[0]?.id;
    return api ? {
      frameCount: state.frameCount,
      submittedFrame: datasetId ? api.getPerformanceSnapshot(datasetId).frame : null,
      layers: api.renderedLayers().slice(0, 20).map(layer => ({
        id: layer.id, layerClass: layer.layerClass, visible: layer.visible, loaded: layer.loaded
      })),
      projectedTestCenter: api.projectPosition(...location)
    } : {apiAvailable: false};
  }, unprojectMercator(centerX, centerY)).catch(error => ({snapshotError: String(error)}));
  console.error(JSON.stringify({stage: currentStage, failures, diagnostics,
    renderSnapshot,
    body: (await page.locator('body').innerText().catch(() => '')).slice(-6000)}));
  throw failure;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
