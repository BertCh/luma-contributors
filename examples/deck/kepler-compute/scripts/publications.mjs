// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getSourceCheckUrl} from './source-check-utils.mjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const first = arrow.tableFromArrays({
  x: new Float32Array([-74, -74.01, -74.02]),
  y: new Float32Array([40.7, 40.72, 40.74]),
  measure: new Float64Array([1.5, 2.5, 3.5])
});
const second = arrow.tableFromArrays({
  x: new Float32Array([-74.03, -74.04]),
  y: new Float32Array([40.76, 40.78]),
  measure: new Float64Array([4.5, 5.5])
});
const source = first.concat(second);
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
// Pin this browser's loaded source while other contributors edit the development server.
await page.routeWebSocket('**/*', socket => {
  const server = socket.connectToServer();
  server.onMessage(message => {
    if (typeof message === 'string') {
      try {
        const packet = JSON.parse(message);
        if (packet.type === 'update' || packet.type === 'full-reload') return;
      } catch {}
    }
    socket.send(message);
  });
  socket.onMessage(message => server.send(message));
});
const errors = [];
page.on('pageerror', error => {
  errors.push(error.message);
  console.error(error);
});
page.on('console', message => {
  if (message.type() === 'error') console.error(message.text());
});
try {
  await page.goto(getSourceCheckUrl(process.argv[2], {rows: 4096}), {
    waitUntil: 'domcontentloaded'
  });
  await page.waitForFunction(
    () => window.__keplerComputeTest?.getState().selectedCount === 4096,
    undefined,
    {timeout: 60000}
  );
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: 'two-batches.arrow',
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))
  });
  await workspace.getByLabel('Source longitude column').selectOption('x');
  await workspace.getByLabel('Source latitude column').selectOption('y');
  await workspace.getByLabel('Float64 conversion').selectOption('exact');
  await workspace.getByRole('button', {name: 'Check schema and precision'}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields'}).click();
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 5,
    undefined,
    {timeout: 45000}
  );
  const state = await page.evaluate(() => window.__keplerComputeTest.getState());
  assert.deepEqual(state.publications[0].sourceBatches, [3, 2]);
  assert.equal(state.publications[0].error, null);
  assert.equal(state.selectedCount, 4096);
  const id = state.publications[0].id;
  await page.waitForFunction(
    id =>
      window.__keplerComputeTest
        .renderedLayers()
        .some(layer => layer.id === `${id}-points` && layer.loaded && layer.visible),
    id
  );
  await closeNativeDialog(page);
  await page.locator('.show-data-table svg').last().click();
  await page.waitForFunction(
    () => document.querySelectorAll('.resident-table-scroll tbody tr').length === 5
  );
  const text = await page.locator('.resident-table-scroll').innerText();
  assert.match(text, /measure/);
  assert.match(text, /5\.5/);
  await page.evaluate(id => window.__keplerComputeTest.setPublishedRange(id, 'measure', 2, 4), id);
  await page.waitForFunction(
    () => document.querySelectorAll('.resident-table-scroll tbody tr').length === 2
  );
  assert.deepEqual(await page.locator('.resident-table-scroll tbody tr th').allTextContents(), [
    '1',
    '2'
  ]);
  assert.equal(
    (await page.evaluate(() => window.__keplerComputeTest.getState())).selectedCount,
    4096
  );
  await page.screenshot({path: '/tmp/luma-kepler-publication-table.png'});
  await page.keyboard.press('Escape');
  await page.locator('.resident-table-modal').waitFor({state: 'hidden'});
  const geometrySource = arrow.tableFromArrays({
    geometry: [
      JSON.stringify({type: 'Point', coordinates: [-74, 40.75]}),
      JSON.stringify({
        type: 'MultiPoint',
        coordinates: [
          [-74.01, 40.76],
          [-74.015, 40.77]
        ]
      }),
      JSON.stringify({
        type: 'LineString',
        coordinates: [
          [-74.02, 40.72],
          [-73.96, 40.78]
        ]
      }),
      JSON.stringify({
        type: 'Polygon',
        coordinates: [
          [
            [-74, 40.7],
            [-73.98, 40.7],
            [-73.98, 40.72],
            [-74, 40.72],
            [-74, 40.7]
          ]
        ]
      })
    ],
    metric: new Int32Array([1, 2, 3, 4])
  });
  await openNativeData(page);
  await workspace.locator('input[type=file]').setInputFiles({
    name: 'mixed-geometry.arrow',
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(geometrySource, 'stream'))
  });
  await workspace.getByLabel('Source geometry column').selectOption('geometry');
  await workspace.getByRole('button', {name: 'Check schema and precision'}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields'}).click();
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[1]?.selectedCount === 4,
    undefined,
    {timeout: 45000}
  );
  const geometryId = await page.evaluate(
    () => window.__keplerComputeTest.getState().publications[1].id
  );
  await page.waitForFunction(
    id =>
      window.__keplerComputeTest
        .renderedLayers()
        .filter(layer => layer.id.startsWith(`${id}-geometry`) && layer.loaded && layer.visible)
        .length >= 3,
    geometryId
  );
  await closeNativeDialog(page);
  await page.locator('.show-data-table svg').last().click();
  await page.waitForFunction(
    () => document.querySelectorAll('.resident-table-scroll tbody tr').length === 4
  );
  assert.deepEqual(await page.locator('.resident-table-scroll tbody tr th').allTextContents(), [
    '0',
    '1',
    '2',
    '3'
  ]);
  await page.evaluate(
    id => window.__keplerComputeTest.setPublishedRange(id, 'metric', 2, 3),
    geometryId
  );
  await page.waitForFunction(
    () => document.querySelectorAll('.resident-table-scroll tbody tr').length === 2
  );
  assert.deepEqual(await page.locator('.resident-table-scroll tbody tr th').allTextContents(), [
    '1',
    '2'
  ]);
  await page.keyboard.press('Escape');
  await page.locator('.resident-table-modal').waitFor({state: 'hidden'});
  await openNativeData(page);
  await workspace.getByRole('button', {name: 'Remove two-batches.arrow', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications.length === 1);
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({
      nativeDataset: id,
      sourceBatches: [3, 2],
      pointLayerLoaded: true,
      nativeFilteredRows: [1, 2],
      geometryDataset: geometryId,
      primitiveTypes: 3,
      geometryFilteredSourceRows: [1, 2],
      removedSource: true
    })
  );
} catch (error) {
  console.error(await page.evaluate(() => window.__keplerComputeTest?.getState()));
  console.error(await page.locator('body').innerText());
  throw error;
} finally {
  await browser.close();
}
