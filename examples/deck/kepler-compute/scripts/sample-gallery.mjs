// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {openNativeData, openNativeExport} from './native-navigation.mjs';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
const samples = [
  ['points-time', 'Observations through time', 256, false],
  ['districts', 'District counts and populations', 16, true],
  ['connections', 'Origin and destination flows', 24, true],
  ['trips', 'Timed routes', 8, true],
  ['numeric-grid', 'Numeric grid', 256, false],
  ['directed-network', 'Directed network edges', 48, true]
];
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
await page.routeWebSocket('**/*', socket => {
  const server = socket.connectToServer();
  server.onMessage(message => {if (typeof message === 'string') {try {const packet = JSON.parse(message); if (packet.type === 'update' || packet.type === 'full-reload') return;} catch {}} socket.send(message);});
  socket.onMessage(message => server.send(message));
});
const errors = [];
page.on('pageerror', error => errors.push(error.message));
async function readDownload(button) {
  const pending = page.waitForEvent('download'); await button.click();
  const download = await pending; const stream = await download.createReadStream();
  const chunks = []; for await (const chunk of stream) chunks.push(chunk);
  return {name: download.suggestedFilename(), bytes: Buffer.concat(chunks)};
}
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=128');
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 128, undefined, {timeout: 60000});
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.getByText('Try a sample dataset', {exact: true}).click();
  for (const [index, [id, title, count, geometry]] of samples.entries()) {
    if (index > 0) {
      // Publishing adds data to the map, which Kepler answers by closing the modal.
      await openNativeData(page);
      await workspace.getByText('Try a sample dataset', {exact: true}).click();
    }
    await workspace.getByLabel('Sample dataset', {exact: true}).selectOption(id);
    const download = await readDownload(workspace.getByRole('button', {name: 'Download sample Arrow', exact: true}));
    assert.equal(download.name, `kepler-sample-${id}.arrow`);
    const table = arrow.tableFromIPC(download.bytes);
    assert.equal(table.numRows, count);
    assert.deepEqual(table.batches.map(batch => batch.numRows), [count / 2, count / 2]);
    assert.equal(JSON.parse(table.schema.metadata.get('luma:sandbox-sample')).id, id);
    await workspace.getByRole('button', {name: 'Prepare sample', exact: true}).click();
    const preview = page.getByRole('region', {name: `Prepare ${title}`, exact: true});
    await preview.waitFor();
    assert.equal(await workspace.getByLabel('Source geometry column').inputValue(), geometry ? 'geometry' : '');
    if (id === 'trips') {
      assert.equal(await workspace.getByLabel('Timed paths (Trips)', {exact: true}).isChecked(), true);
      assert.equal(await workspace.getByLabel('Trip timestamp encoding', {exact: true}).inputValue(), 'unix-second');
      assert.equal(JSON.parse(table.schema.metadata.get('luma:sandbox-sample')).tripOptions.timestampEncoding, 'unix-second');
    }
    if (!geometry) {
      assert.equal(await workspace.getByLabel('Source longitude column').inputValue(), 'longitude');
      assert.equal(await workspace.getByLabel('Source latitude column').inputValue(), 'latitude');
    }
    assert.equal(await preview.getByRole('button', {name: 'Publish selected fields', exact: true}).isDisabled(), true);
    await preview.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
    await preview.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
    await page.waitForFunction(({index, count}) => {
      const published = window.__keplerComputeTest.getState().publications[index];
      return published?.rowCount === count && published?.selectedCount === count;
    }, {index, count}, {timeout: 45000});
    const publication = await page.evaluate(index => window.__keplerComputeTest.getState().publications[index], index);
    assert.equal(publication.error, null);
    assert.deepEqual(publication.sourceBatches, [count / 2, count / 2]);
    await page.waitForFunction(({id, suffix}) => window.__keplerComputeTest.renderedLayers().some(layer => layer.id === `${id}-${suffix}` && layer.loaded && layer.visible), {id: publication.id, suffix: id === 'trips' ? 'trips' : geometry ? 'geometry' : 'points'});
  }
  await openNativeExport(page, true);
  const bundle = JSON.parse((await readDownload(page.getByRole('button', {name: 'Save portable bundle', exact: true}))).bytes.toString('utf8'));
  assert.equal(bundle.sources.length, 7);
  assert.equal(bundle.state.sources.length, 7);
  assert.deepEqual(bundle.state.sources[4].reference.parameters.tripOptions, {timestampEncoding: 'unix-second', maximumTimeErrorMilliseconds: 0.1});
  await page.setViewportSize({width: 390, height: 844});
  await openNativeData(page);
  await workspace.getByText('Try a sample dataset', {exact: true}).click();
  await workspace.getByLabel('Sample dataset', {exact: true}).selectOption('numeric-grid');
  await workspace.getByRole('button', {name: 'Prepare sample', exact: true}).focus();
  assert.equal(await workspace.getByRole('button', {name: 'Prepare sample', exact: true}).evaluate(element => document.activeElement === element), true);
  await page.screenshot({path: '/tmp/kepler-gallery-mobile.png'});
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, checks: ['six source downloads and metadata', 'explicit schema review before publication', 'suggested geometry/coordinate roles', 'six rendered resident datasets', 'two preserved batches per sample', 'portable workspace includes all samples', 'keyboard focus at 390px']}));
} catch (error) {
  console.error(await page.evaluate(() => ({state: window.__keplerComputeTest?.getState(), notices: Array.from(document.querySelectorAll('[role=status], [role=alert]'), element => element.textContent)})));
  console.error(errors); throw error;
} finally {await browser.close();}
