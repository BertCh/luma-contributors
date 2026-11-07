// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {KeplerDatasetRegistry} from '../src/runtime/dataset-registry';

test('setMemoryBudget raises always and rejects below usage', () => {
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1000});
  let notifications = 0;
  registry.subscribeMemory(() => notifications++);
  const reservation = registry.reserveMemory(600, 'gpu', 'output');
  expect(() => registry.setMemoryBudget(500)).toThrow();
  expect(registry.memoryBudgetBytes).toBe(1000);
  registry.setMemoryBudget(2000);
  expect(registry.memoryBudgetBytes).toBe(2000);
  expect(registry.memoryUsage.availableBytes).toBe(1400);
  expect(notifications).toBeGreaterThan(0);
  registry.setMemoryBudget(600);
  expect(() => registry.reserveMemory(1, 'gpu')).toThrow(/budget/);
  reservation.release();
});

test('getDatasetMemory splits memory kinds and ownership', () => {
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 10_000});
  const reservations = [registry.reserveMemory(110, 'gpu'), registry.reserveMemory(40, 'cpu')];
  registry.registerDataset(
    {
      id: 'a',
      value: {},
      batches: [{id: 'b', rowCount: 4}],
      resources: [
        {
          id: 'gpu-1',
          value: {},
          byteLength: 100,
          memoryKind: 'gpu',
          ownership: 'owned',
          destroy() {}
        },
        {
          id: 'cpu-1',
          value: {},
          byteLength: 40,
          memoryKind: 'cpu',
          ownership: 'owned',
          destroy() {}
        },
        {id: 'gpu-2', value: {}, byteLength: 10, memoryKind: 'gpu', ownership: 'borrowed'}
      ]
    } as never,
    reservations
  );
  const [row] = registry.getDatasetMemory();
  expect(row).toMatchObject({
    id: 'a',
    gpuBytes: 110,
    cpuBytes: 40,
    ownedBytes: 140,
    borrowedBytes: 10,
    sharedBytes: 0,
    rowCount: 4
  });
});
