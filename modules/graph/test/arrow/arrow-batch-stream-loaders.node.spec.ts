// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import test from 'test/utils/vitest-tape';
import {appendGPUTableFromArrowBatches} from '@luma-contributors/graph/arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import {parseInBatches} from '@loaders.gl/core';
import {ArrowLoader, CSVLoader} from '@loaders.gl/tables';
import * as arrow from 'apache-arrow';

test('appendGPUTableFromArrowBatches consumes loaders.gl CSV parseInBatches output', async t => {
  const device = new NullDevice({});
  const rows = ['x,y,id'];
  for (let index = 0; index < 20; index++) {
    rows.push(`${index + 0.5},${index * 2},${index}`);
  }
  const loaderBatches = (await parseInBatches(
    new Response(rows.join('\n'), {headers: {'content-type': 'text/csv'}}),
    [CSVLoader],
    {
      csv: {shape: 'arrow-table', header: true, dynamicTyping: true}
    } as any
  )) as AsyncIterable<any>;

  const table = await appendGPUTableFromArrowBatches(device, loaderBatches, {
    columns: [{name: 'x'}, {name: 'id'}]
  });

  t.ok(table, 'builds a GPU table');
  t.equal(table!.numRows, 20, 'uploads every row');
  t.ok(table!.batches.length >= 1, 'uploads at least one loader batch');
  t.deepEqual(
    table!.batches.map(batch => batch.sourceInfo?.sourceRowIndexOffset),
    table!.batches.map((_, index) =>
      table!.batches.slice(0, index).reduce((count, batch) => count + batch.numRows, 0)
    ),
    'offsets are cumulative'
  );
  t.deepEqual(
    table!.schema.fields.map(field => field.name),
    ['x', 'id'],
    'selects columns'
  );
  table!.destroy();
  device.destroy();
  t.end();
});

test('appendGPUTableFromArrowBatches consumes loaders.gl Arrow IPC parseInBatches output', async t => {
  const device = new NullDevice({});
  const writer = new arrow.RecordBatchStreamWriter();
  const makeTable = (values: number[]) =>
    new arrow.Table({id: arrow.vectorFromArray(values, new arrow.Float32())});
  writer.write(makeTable([1, 2, 3]).batches[0]);
  writer.write(makeTable([4, 5]).batches[0]);
  writer.close();
  const bytes = await writer.toUint8Array();

  const loaderBatches = (await parseInBatches(
    new Response(bytes, {headers: {'content-type': 'application/vnd.apache.arrow.stream'}}),
    [ArrowLoader],
    {arrow: {shape: 'arrow-table'}} as any
  )) as AsyncIterable<any>;
  const table = await appendGPUTableFromArrowBatches(device, loaderBatches);

  t.ok(table, 'builds a GPU table');
  t.deepEqual(
    table!.batches.map(batch => batch.numRows),
    [3, 2],
    'preserves IPC record-batch boundaries'
  );
  t.equal(table!.schema.fields[0].format, 'float32', 'maps Arrow Float32 to float32');
  table!.destroy();
  device.destroy();
  t.end();
});
