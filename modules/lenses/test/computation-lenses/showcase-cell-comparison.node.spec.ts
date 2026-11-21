// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  readCellComparisonSnapshot,
  type CellComparisonIdentity
} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/cells/cell-pyramid-comparison';

function publication(): ArrayBuffer {
  const bytes = new ArrayBuffer(48);
  // One occupied cell, no cell loss, saturated sum; numeric values are just sentinels.
  new Uint32Array(bytes).set([1, 0, 1, 1, 0, 1, 123, 456]);
  new Float32Array(bytes, 32).set([17, 19, 2, 0]);
  return bytes;
}

test('DGGS-102 B6: a pending sum publication is discarded after switching to count', async () => {
  const produced: CellComparisonIdentity = {
    key: '6|sum|standardized',
    inputGeneration: 1,
    measure: 'sum',
    resolution: 6
  };
  let current: CellComparisonIdentity | null = produced;
  let finish!: (bytes: ArrayBuffer) => void;
  const pending = new Promise<ArrayBuffer>(resolve => {
    finish = resolve;
  }).then(bytes => readCellComparisonSnapshot(bytes, 1, produced, current));
  current = {key: '8|count|poisson', inputGeneration: 2, measure: 'count', resolution: 8};
  finish(publication());
  expect(await pending).toBeNull();
  // An A -> B -> A switch still invalidates an earlier read of A.
  expect(
    readCellComparisonSnapshot(publication(), 1, produced, {...produced, inputGeneration: 3})
  ).toBeNull();
  expect(readCellComparisonSnapshot(publication(), 1, produced, null)).toBeNull();
});

test('DGGS-102 B6: accepted publications retain the producing measure, resolution and loss word', () => {
  const identity: CellComparisonIdentity = {
    key: '6|sum|standardized',
    inputGeneration: 4,
    measure: 'sum',
    resolution: 6
  };
  const snapshot = readCellComparisonSnapshot(publication(), 1, identity, identity)!;
  expect(snapshot.measure).toBe('sum');
  expect(snapshot.resolution).toBe(6);
  expect(snapshot.inputGeneration).toBe(4);
  expect(snapshot.sumOverflow).toBe(true);
  expect(snapshot.rows).toBe(1);
  expect(Array.from(snapshot.delta)).toEqual([2]);
});
