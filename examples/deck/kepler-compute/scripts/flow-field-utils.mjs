// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

/** Opens the sandbox, uploads an Arrow IPC table and publishes it. Returns page helpers. */
export async function openPublishedTable({table, fileName, url, selectColumns}) {
  const arrow = await import('apache-arrow');
  const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  await pinViteModules(page);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error' && /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())) errors.push(message.text());
  });
  await page.goto(url || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page, 'Prepare data');
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({name: fileName, mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(table, 'stream'))});
  if (selectColumns) await selectColumns(workspace);
  await workspace.getByLabel('Float64 conversion').selectOption('round');
  await workspace.getByLabel('Maximum relative error').fill('0.000001');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount > 0, undefined, {timeout: 60000});
  await closeNativeDialog(page);
  const datasetId = (await page.evaluate(() => window.__keplerComputeTest.getState())).publications[0].id;
  const configuration = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const apply = config => page.evaluate(value => window.__keplerComputeTest.applyNativeConfig(value), config);
  /** Rewrites the published dataset's first layer to the given type/config and applies it. */
  async function applyLayer(id, type, config, visConfig, columnMode) {
    const next = structuredClone(configuration);
    const layers = next.config.visState.layers;
    let layer = layers.find(candidate => candidate.config.dataId === datasetId);
    if (!layer) {
      // Tables without lat/lng (or other auto-detectable columns) get no native layer: clone any
      // existing layer as a template and point it at the published dataset.
      const template = layers[0];
      assert(template, 'the native configuration has no layer to use as a template');
      layer = structuredClone(template);
      layer.config.dataId = datasetId;
      layers.push(layer);
    }
    for (const other of layers) other.config.isVisible = other === layer;
    layer.id = id;
    const order = next.config.visState.layerOrder;
    if (Array.isArray(order) && !order.includes(id)) order.unshift(id);
    layer.type = type;
    layer.config = {...layer.config, ...config, textLabel: []};
    if (columnMode) layer.config.columnMode = columnMode;
    layer.config.visConfig = {...layer.config.visConfig, ...visConfig};
    next.config.mapState = {...next.config.mapState, ...(config.mapState ?? {})};
    await apply(next);
    return layer;
  }
  return {browser, page, errors, datasetId, configuration, apply, applyLayer, assert};
}
