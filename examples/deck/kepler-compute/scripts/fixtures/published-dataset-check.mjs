// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {luma} from '@luma.gl/core';
import {webgpuAdapter} from '@luma.gl/core/webgpu';
import {KeplerComputeSession} from '@luma-contributors/kepler';
import * as arrow from 'apache-arrow';
import {PublishedDatasetRuntime} from '../../src/published-dataset-runtime';
import {previewArrowSource} from '../../src/source-preview';

function equal(actual, expected) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error(`Expected ${JSON.stringify(expected)}, received ${JSON.stringify(actual)}`);
}

async function runChecks() {
  const device = await luma.createDevice({type: 'webgpu', featureLevel: 'core', adapters: [webgpuAdapter], createCanvasContext: {width: 1, height: 1}});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  let published;
  try {
    const first = arrow.tableFromArrays({'sales USD': new Float32Array([30, 10]), region: ['north', 'south']});
    const second = arrow.tableFromArrays({'sales USD': new Float32Array([40, 20]), region: ['west', 'north']});
    const source = first.concat(second);
    const preview = previewArrowSource(source);
    equal(preview.canUpload, true);
    equal(preview.batchRowCounts, [2, 2]);
    const dataset = session.importArrowDataset({id: 'sql-result-with-original-names', table: source});
    const uploadedBytes = dataset.value.uploadedBufferBytes;
    let updates = 0;
    let notifications = 0;
    published = new PublishedDatasetRuntime({session, datasetId: dataset.id,
      explorer: {dimensions: [{id: 'sales', kind: 'range', column: 'sales USD'}], workingMemoryBudgetBytes: 8 * 1024 * 1024},
      sortMemoryBudgetBytes: 8 * 1024 * 1024, requestUpdate: () => updates++});
    published.subscribe(() => notifications++);
    const render = async () => {
      const encoder = device.createCommandEncoder();
      published.encode(encoder);
      device.submit(encoder.finish());
      await published.readSummary();
    };
    await render();
    equal(published.summary.selectedCount, 4);
    equal(published.table.fields.map(field => field.name), ['sales USD', 'region']);
    const request = {offset: 0, limit: 4, columns: ['sales USD', 'region'], population: 'selected', sort: {column: 'sales USD', direction: 'ascending'}};
    const page = await published.table.readPage(request);
    equal(Array.from(page.sourceRowIndices), [1, 3, 0, 2]);
    equal(Array.from(page.columns['sales USD']), [10, 20, 30, 40]);
    const formatRegion = published.table.fields.find(field => field.name === 'region').format;
    equal(Array.from(page.columns.region, formatRegion), ['south', 'north', 'north', 'west']);
    const previousNotifications = notifications;
    published.setRange('sales', 20, 30);
    equal(published.summary, null);
    equal(notifications, previousNotifications + 1);
    equal(updates, 1);
    await render();
    equal(published.summary.selectedCount, 2);
    equal(Array.from((await published.table.readPage(request)).sourceRowIndices), [3, 0]);
    equal(dataset.value.uploadedBufferBytes, uploadedBytes);
    equal(dataset.value.columns['sales USD'].data.map(chunk => chunk.length), [2, 2]);
    session.importArrowDataset({id: dataset.id, table: first});
    equal(published.summary, null);
    equal(await published.table.readPage(request), null);
    await published.destroy();
    equal(session.registry.getDataset(dataset.id).rowCount, 2);
    return {passed: true, checks: ['original field names', 'batch preservation', 'category labels', 'GPU sorting', 'selection notifications', 'stale source rejection', 'borrowed dataset lifetime']};
  } finally {
    await published?.destroy();
    const fence = device.createFence();
    try { await session.destroy(fence.signaled); }
    finally { fence.destroy(); device.destroy(); }
  }
}

try {
  window.publishedDatasetResult = await runChecks();
  document.querySelector('#result').textContent = 'Passed';
} catch (error) {
  window.publishedDatasetResult = {passed: false, error: error.stack ?? String(error)};
  document.querySelector('#result').textContent = String(error);
}
