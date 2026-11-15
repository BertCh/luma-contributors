// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {KeplerDatasetRegistry} from '../src/runtime/dataset-registry';
import {subscribeMemoryBudgetUpdates} from '../../../examples/deck/kepler-compute/src/memory-budget-updates';
import {trackOperation} from '../../../examples/deck/kepler-compute/src/operation-tracker';

test('budget panel defers render-time reservations and coalesces operation updates', async () => {
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1000});
  const observedUsage: number[] = [];
  const unsubscribe = subscribeMemoryBudgetUpdates(registry, () => {
    observedUsage.push(registry.memoryUsage.totalBytes);
  });
  const first = registry.reserveMemory(10, 'gpu', 'output');
  const second = registry.reserveMemory(20, 'gpu', 'scratch');
  const operation = trackOperation('Prepare resident layer', () => {});
  try {
    expect(registry.memoryUsage.totalBytes).toBe(30);
    expect(observedUsage).toEqual([]);
    await Promise.resolve();
    expect(observedUsage).toEqual([30]);
    first.release();
    second.release();
    operation.finish();
    await Promise.resolve();
    expect(observedUsage).toEqual([30, 0]);
  } finally {
    unsubscribe();
    first.release();
    second.release();
    operation.finish();
  }
});

test('budget panel cancels pending delivery and releases subscriptions on unmount', async () => {
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1000});
  let updates = 0;
  const unsubscribe = subscribeMemoryBudgetUpdates(registry, () => updates++);
  const reservation = registry.reserveMemory(10, 'gpu', 'output');
  unsubscribe();
  const operation = trackOperation('After unmount', () => {});
  reservation.release();
  operation.finish();
  await Promise.resolve();
  expect(updates).toBe(0);
});

test('rebinding a budget panel cancels the old registry update and observes the new registry', async () => {
  const previous = new KeplerDatasetRegistry({memoryBudgetBytes: 1000});
  const current = new KeplerDatasetRegistry({memoryBudgetBytes: 1000});
  const observedUsage: number[] = [];
  const stopPrevious = subscribeMemoryBudgetUpdates(previous, () => observedUsage.push(-1));
  const previousReservation = previous.reserveMemory(10, 'gpu', 'output');
  stopPrevious();
  const stopCurrent = subscribeMemoryBudgetUpdates(current, () => {
    observedUsage.push(current.memoryUsage.totalBytes);
  });
  const currentReservation = current.reserveMemory(20, 'gpu', 'output');
  try {
    await Promise.resolve();
    expect(observedUsage).toEqual([20]);
  } finally {
    stopCurrent();
    previousReservation.release();
    currentReservation.release();
  }
});
