// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test} from 'vitest';
import {NullDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession} from '../src/session/compute-session';
import {makeSourceCandidateFromBytes} from '../../../examples/deck/kepler-compute/src/remote-source-loader';
import {restoreArrowSourceRoleColumns} from '../../../examples/deck/kepler-compute/src/source-type-routes';
import {serializeSelectedSource} from '../../../examples/deck/kepler-compute/src/source-export';

test('file routing, role assignment, source-only publication and Arrow export retain original Uint64 cells', async () => {
  const cells = [null, 0x8928308280fffffn, 0n, 0x8928308280bffffn, null, 0x8928308280fffffn];
  const original = new arrow.Table({
    cell: arrow.vectorFromArray(cells, new arrow.Uint64()),
    scalar: arrow.vectorFromArray(
      cells.map(() => 2n ** 60n),
      new arrow.Int64()
    )
  });
  original.schema.fields[0].metadata.set('example:identity', 'exact H3 source');
  const source = original.slice(0, 3).concat(original.slice(3));
  const candidate = await makeSourceCandidateFromBytes(
    arrow.tableToIPC(source, 'stream'),
    'cells.arrow'
  );
  expect(String(candidate.table.getChild('cell')!.type)).toBe('Utf8');
  const table = restoreArrowSourceRoleColumns(candidate.table, candidate.originalTable, ['cell']);
  expect(String(table.getChild('cell')!.type)).toBe('Uint64');
  expect([...table.getChild('cell')!]).toEqual(cells);
  expect(table.schema.fields[0].metadata.get('example:identity')).toBe('exact H3 source');
  // Other wide scalar columns still follow the explicit decimal-category policy.
  expect(String(table.getChild('scalar')!.type)).toBe('Utf8');
  expect(table.batches.map(batch => batch.numRows)).toEqual([3, 3]);
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 1024 * 1024});
  try {
    const dataset = session.importArrowDataset({id: 'cells', table, normalization: {columns: []}});
    expect(dataset.value.uploadedBufferBytes).toBe(0);
    const output = await serializeSelectedSource(
      dataset.value.source,
      new Uint32Array([0, 1, 2, 3, 4, 5]),
      {
        datasetId: dataset.id,
        datasetRevision: dataset.revision,
        selectionRevision: 1,
        selectedCount: 6,
        offset: 0
      },
      {format: 'arrow', maximumOutputBytes: 65536, assertCurrent() {}}
    );
    const exported = arrow.tableFromIPC(output.bytes);
    expect(String(exported.getChild('cell')!.type)).toBe('Uint64');
    expect([...exported.getChild('cell')!]).toEqual(cells);
    expect(exported.batches.map(batch => batch.numRows)).toEqual([3, 3]);
    expect(exported.schema.fields[0].metadata.get('example:identity')).toBe('exact H3 source');
    const replay = await makeSourceCandidateFromBytes(output.bytes, 'replay.arrow');
    expect([
      ...restoreArrowSourceRoleColumns(replay.table, replay.originalTable, ['cell']).getChild(
        'cell'
      )!
    ]).toEqual(cells);
  } finally {
    await session.destroy(Promise.resolve());
    device.destroy();
  }
});
