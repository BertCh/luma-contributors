// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {makeContext, makeCapture} from '../fixtures/v4-record';
import {TABLE_VIEW} from '../../../src/computation-views/views/table-view';
import {buildDatumReceiptModel} from '../../../src/computation-views/views/datum-receipt';
import {
  buildResultViewModel,
  renderResultView
} from '../../../src/computation-views/views/result-view';
import {getViewNodeText} from '../../../src/computation-views/view-node';

test('PTS-130 catchment unknown counts stay raw and render unknown across readers', () => {
  const context = makeContext({bindings: {entity: 1}});
  context.record.id = 'gpu-spatial-analysis.catchment-accessibility';
  context.record.graphs[0].nodes.find(node => node.id === 'finish')!.op =
    'GPUCatchmentAccessibility';
  const capture = makeCapture({
    name: 'result.reachableFacilities',
    elementType: 'u32',
    values: new Uint32Array([2, 0xffffffff, 6]),
    role: 'output'
  });
  const unrelated = makeCapture({
    id: 'unrelated',
    name: 'unrelated',
    elementType: 'u32',
    values: new Uint32Array([2, 0xffffffff, 6])
  });
  context.record.captures!.snapshots = [capture, unrelated];
  const table = TABLE_VIEW.build(context).sections.find(
    section => section.id === 'captures'
  )!.table;
  const rows = table.rows.map(row =>
    Object.fromEntries(table.columns.map((column, index) => [column.name, row[index]]))
  );
  expect(rows[0]['max']).toBe('6'); // Hand maximum of the two known counts, 2 and 6.
  expect(rows[0]['value at 1']).toBe('unknown');
  expect(rows[1]['max']).toBe('4294967295');
  expect(rows[1]['value at 1']).toBe('4294967295');
  const receipt = buildDatumReceiptModel(context);
  expect(receipt.rows.find(row => row.name === capture.name)!.value).toBe('unknown');
  expect(receipt.rows.find(row => row.name === unrelated.name)!.value).toBe('4294967295');
  expect(getViewNodeText(renderResultView(buildResultViewModel(context), context))).toContain(
    'unknown'
  );
  expect(Array.from(capture.values)).toEqual([2, 0xffffffff, 6]);
  context.record.id = 'other';
  context.record.graphs[0].nodes.find(node => node.id === 'finish')!.op = 'CensusCount';
  expect(buildDatumReceiptModel(context).rows.find(row => row.name === capture.name)!.value).toBe(
    '4294967295'
  );
});
