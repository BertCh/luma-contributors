// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test} from 'vitest';
import type {KeplerArrowDataset} from '@luma-contributors/kepler';
import {
  estimateResidentMaterializedBytes,
  materializeResidentRows
} from '../../../examples/deck/kepler-compute/src/resident-materialization';

function makeFakeResident(batchLengths: number[], pageRows: number) {
  const requests: number[][] = [];
  const pages = new Set<string>();
  const batches = batchLengths.map(
    length => new arrow.Table({value: arrow.vectorFromArray(new Float32Array(length))}).batches[0]
  );
  const source = new arrow.Table(batches);
  const offsets = batchLengths.map((_, index) =>
    batchLengths.slice(0, index).reduce((total, length) => total + length, 0)
  );
  const value = {
    source,
    resident: {
      pageRows,
      async ensureRows(rows: readonly number[]) {
        requests.push([...rows]);
        for (const row of rows) {
          const batch = offsets.filter(offset => offset <= row).length - 1;
          pages.add(`${batch}:${Math.floor((row - offsets[batch]) / pageRows)}`);
        }
      }
    }
  } as unknown as Pick<KeplerArrowDataset, 'resident' | 'source'>;
  return {value, requests, pages};
}

test('materializeResidentRows pages every page of every batch in bounded steps', async () => {
  const {value, requests, pages} = makeFakeResident([1000, 777, 5], 100);
  await materializeResidentRows(value);
  // 10 + 8 + 1 pages; none are skipped at batch boundaries.
  expect(pages.size).toBe(19);
  expect(Math.max(...requests.map(rows => rows.length))).toBeLessThanOrEqual(64);
});

test('materializeResidentRows reads only the requested rows in bounded steps', async () => {
  const {value, requests} = makeFakeResident([200000], 1024);
  const rows = Uint32Array.from({length: 200000 / 3}, (_, index) => index * 3);
  await materializeResidentRows(value, rows);
  expect(requests.length).toBeGreaterThan(1);
  expect(Math.max(...requests.map(request => request.length))).toBeLessThanOrEqual(1024 * 64);
  expect(requests.flat().length).toBe(rows.length);
});

test('materializeResidentRows stops when the source becomes stale', async () => {
  const {value, requests} = makeFakeResident([100000], 100);
  let calls = 0;
  await expect(
    materializeResidentRows(value, undefined, {
      assertCurrent() {
        if (++calls > 2) throw new Error('stale');
      }
    })
  ).rejects.toThrow('stale');
  expect(requests.length).toBeLessThan(5);
});

test('materializeResidentRows leaves CPU datasets untouched', async () => {
  const {value} = makeFakeResident([10], 4);
  await materializeResidentRows({source: value.source});
  expect(estimateResidentMaterializedBytes(value)).toBe(42);
});
