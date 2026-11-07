// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import test from 'test/utils/vitest-tape';
import {
  appendGPUTableFromArrowBatches,
  iterateGPURecordBatchesFromArrowBatches,
  makeArrowFixedSizeListVector
} from '@luma-contributors/graph/arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import * as arrow from 'apache-arrow';

function makeRecordBatch(positions: number[], ids: number[]): arrow.RecordBatch {
  return new arrow.Table({
    positions: makeArrowFixedSizeListVector(new arrow.Float32(), 2, new Float32Array(positions)),
    id: arrow.vectorFromArray(ids, new arrow.Uint32())
  }).batches[0];
}

async function* streamBatches(batches: unknown[]) {
  for (const batch of batches) {
    yield batch as any;
  }
}

test('appendGPUTableFromArrowBatches preserves batch boundaries and skips empty batches', async t => {
  const device = new NullDevice({});
  const first = makeRecordBatch([0, 1, 2, 3], [1, 2]);
  const second = makeRecordBatch([4, 5], [3]);
  const table = await appendGPUTableFromArrowBatches(
    device,
    streamBatches([
      {shape: 'arrow-table', batchType: 'metadata', length: 0, data: null},
      {shape: 'arrow-table', batchType: 'data', length: 2, data: new arrow.Table([first])},
      {shape: 'arrow-table', batchType: 'data', length: 0, data: new arrow.Table([])},
      {shape: 'arrow-table', batchType: 'data', length: 1, data: new arrow.Table([second])},
      {shape: 'arrow-table', batchType: 'final', length: 0, data: new arrow.Table([])}
    ])
  );

  t.ok(table, 'creates a table');
  t.equal(table!.batches.length, 2, 'one GPURecordBatch per non-empty source batch');
  t.equal(table!.numRows, 3, 'counts rows across batches');
  t.deepEqual(
    table!.batches.map(batch => batch.numRows),
    [2, 1],
    'preserves batch row counts'
  );
  t.deepEqual(
    table!.batches.map(batch => batch.sourceInfo?.sourceRowIndexOffset),
    [0, 2],
    'tracks source row offsets'
  );
  t.deepEqual(
    table!.batches.map(batch => batch.sourceInfo?.sourceBatchIndex),
    [0, 1],
    'tracks source batch indices'
  );
  t.deepEqual(
    table!.schema.fields.map(field => [field.name, field.format]),
    [
      ['positions', 'float32x2'],
      ['id', 'uint32']
    ],
    'maps Arrow types to GPU vector formats'
  );
  t.equal(table!.batches[0].gpuData.id.length, 2, 'one GPUData chunk per column');
  table!.destroy();
  device.destroy();
  t.end();
});

test('appendGPUTableFromArrowBatches splits multi-batch tables and appends to existing tables', async t => {
  const device = new NullDevice({});
  const wrapper = {
    shape: 'arrow-table',
    batchType: 'data',
    length: 3,
    data: new arrow.Table([makeRecordBatch([0, 1], [1]), makeRecordBatch([2, 3, 4, 5], [2, 3])])
  };
  const table = await appendGPUTableFromArrowBatches(device, [wrapper], {
    columns: [{name: 'id'}]
  });
  t.equal(table!.batches.length, 2, 'keeps record-batch boundaries within one loader batch');
  t.deepEqual(
    table!.schema.fields.map(field => field.name),
    ['id'],
    'honors column selection'
  );

  const appended = await appendGPUTableFromArrowBatches(device, [makeRecordBatch([6, 7], [4])], {
    table,
    columns: [{name: 'id'}]
  });
  t.equal(appended, table, 'returns the supplied table');
  t.equal(table!.batches.length, 3, 'appends without repacking');
  t.equal(table!.batches[2].sourceInfo?.sourceRowIndexOffset, 3, 'continues row offsets');

  table!.destroy();
  device.destroy();
  t.end();
});

test('iterateGPURecordBatchesFromArrowBatches yields nothing for empty streams', async t => {
  const device = new NullDevice({});
  let count = 0;
  for await (const prepared of iterateGPURecordBatchesFromArrowBatches(
    device,
    streamBatches([{shape: 'arrow-table', batchType: 'metadata', data: null}])
  )) {
    prepared.destroy();
    count++;
  }
  t.equal(count, 0, 'metadata-only streams produce no batches');
  t.equal(await appendGPUTableFromArrowBatches(device, streamBatches([])), undefined);
  device.destroy();
  t.end();
});
