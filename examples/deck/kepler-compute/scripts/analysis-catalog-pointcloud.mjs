// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

export const POINT_CLOUD_OPERATIONS = ['ground', 'smrf', 'height', 'surfaces', 'neighborhoods', 'features', 'statistical-filter', 'normal-scale', 'icp', 'change', 'voxel-merge'].map(operation => `pointcloud-${operation}`);

export async function checkPointCloudCatalogDiscovery(panel) {
  const operations = await panel.getByLabel('Operation', {exact: true}).locator('option').evaluateAll(options => options.map(option => option.value));
  for (const operation of POINT_CLOUD_OPERATIONS) assert(operations.includes(operation), `${operation} is discoverable`);
  const blocked = await panel.getByTestId('catalog-blocked').innerText();
  for (const contributor of ['Ground', 'SMRF', 'Height', 'Surfaces', 'Neighborhoods', 'Features', 'StatisticalFilter', 'NormalScale', 'ICP', 'Change', 'VoxelMerge']) assert(!blocked.includes(`GPUPointCloud${contributor} ·`), `${contributor} must be runnable`);
}

/** Real CSV source → mapped XYZ catalog run → Sources → styled native point layer with altitude. */
export async function checkPointCloudCatalogWorkflow(page, panel) {
  await panel.getByLabel('Engine', {exact: true}).selectOption('auto');
  for (const operation of POINT_CLOUD_OPERATIONS) {
    await panel.getByLabel('Operation', {exact: true}).selectOption(operation);
    if (operation === 'pointcloud-change') await panel.getByLabel('Epochs share an aligned frame', {exact: true}).check();
    if (operation === 'pointcloud-voxel-merge') {
      await panel.getByLabel(/^Coordinate system/).selectOption('planar');
      await panel.getByLabel('Parts own disjoint rows on the same voxel grid', {exact: true}).check();
    }
    await panel.getByTestId('catalog-run').click();
    await panel.getByTestId('catalog-result').waitFor();
    await panel.getByRole('button', {name: 'Run', exact: true}).waitFor();
    assert.equal(await panel.getByTestId('catalog-error').count(), 0, `${operation} run should succeed`);
    assert.match(await panel.getByTestId('catalog-engine').innerText(), /Engine: GPU/);
  }
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  const csv = 'longitude,latitude,altitude,classification,intensity\n' + Array.from({length: 10}, (_, row) => {
    const cell = row === 9 ? 4 : row;
    return `${-87.65 + cell % 3 * 0.00001},${41.88 + Math.floor(cell / 3) * 0.00001},${row === 9 ? 103 : 100},${row === 9 ? 1 : 2},${row * 10}`;
  }).join('\n');
  await workspace.locator('input[type=file]').setInputFiles({name: 'pointcloud-catalog.csv', mimeType: 'text/csv', buffer: Buffer.from(csv)});
  await workspace.getByLabel('Source longitude column').selectOption('longitude');
  await workspace.getByLabel('Source latitude column').selectOption('latitude');
  await workspace.getByLabel('Source altitude column').selectOption('altitude');
  await workspace.getByLabel('Float64 conversion', {exact: true}).selectOption('round');
  await workspace.getByLabel('Maximum absolute error', {exact: true}).fill('0.00001');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications.some(source => source.rowCount === 10 && source.selectedCount === 10));
  await closeNativeDialog(page);
  await panel.getByLabel('Operation', {exact: true}).selectOption('pointcloud-height');
  await panel.getByLabel('Points cloud source', {exact: true}).selectOption('native');
  const sourceOption = await panel.getByLabel('Points cloud dataset', {exact: true}).locator('option').evaluateAll(options => options.find(option => option.textContent.includes('pointcloud-catalog'))?.value);
  assert(sourceOption, 'retained CSV point table appears in catalog dataset choices');
  await panel.getByLabel('Points cloud dataset', {exact: true}).selectOption(sourceOption);
  const fields = await panel.getByLabel('Points cloud geometry field', {exact: true}).locator('option').evaluateAll(options => options.map(option => option.value));
  const mappedField = fields.find(field => field.startsWith('XYZ columns:'));
  assert(mappedField, 'XYZ role mapping is discoverable without a GeoJSON column');
  await panel.getByLabel('Points cloud geometry field', {exact: true}).selectOption(mappedField);
  await panel.getByTestId('catalog-run').click();
  await panel.getByTestId('catalog-result').waitFor();
  await panel.getByRole('button', {name: 'Run', exact: true}).waitFor();
  assert.equal(await panel.getByTestId('catalog-error').count(), 0);
  assert.match(await panel.getByTestId('catalog-result').innerText(), /height_above_ground/);
  await panel.getByRole('button', {name: 'Serialize invocation', exact: true}).click();
  const invocation = JSON.parse(await panel.getByLabel('Serialized invocation', {exact: true}).inputValue());
  assert.equal(invocation.operationId, 'pointcloud-height');
  assert.equal(invocation.inputs.points.featureCount, 10);
  const before = await page.evaluate(() => window.__keplerComputeTest.getState().publications.map(source => source.id));
  await panel.getByTestId('catalog-publish').click();
  await openNativeData(page);
  assert.equal(await workspace.getByLabel('Source altitude column').inputValue(), 'altitude');
  await workspace.getByLabel('Float64 conversion', {exact: true}).selectOption('round');
  await workspace.getByLabel('Maximum absolute error', {exact: true}).fill('0.00001');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(previous => window.__keplerComputeTest.getState().publications.some(source => !previous.includes(source.id) && source.selectedCount === 10), before);
  await closeNativeDialog(page);
  const datasetId = await page.evaluate(previous => window.__keplerComputeTest.getState().publications.find(source => !previous.includes(source.id)).id, before);
  const configuration = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const layer = configuration.config.visState.layers.find(candidate => candidate.config.dataId === datasetId);
  assert(layer, 'catalog output creates a native point layer');
  assert.equal(layer.config.columns.altitude, 'altitude');
  layer.config.columns = {lng: 'longitude', lat: 'latitude', altitude: 'altitude'};
  layer.visualChannels = {...layer.visualChannels, colorField: {name: 'height_above_ground', type: 'real'}, colorScale: 'quantile'};
  configuration.config.mapState = {...configuration.config.mapState, longitude: -87.64999, latitude: 41.88001, zoom: 18, pitch: 45};
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), configuration);
  await page.waitForFunction(identifier => window.__keplerComputeTest.renderedLayers().some(candidate => candidate.id === identifier && candidate.layerClass === 'GPUScatterplotLayer' && candidate.loaded && candidate.getPosition?.format === 'float32x3'), layer.id);
  const native = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  assert.equal(native.config.visState.layers.find(candidate => candidate.id === layer.id)?.visualChannels.colorField.name, 'height_above_ground');
}
