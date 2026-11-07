// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';
import {TripRuntime} from '../../../examples/deck/kepler-compute/src/trip-runtime';
import {
  estimateTripOutputBytes,
  resolveTripWorkingBudget
} from '../../../examples/deck/kepler-compute/src/trip-working-budget';
import {createTabularTripRuntimeFixture} from './utils/tabular-trip-runtime-fixture';

test('budget stays at the configured value when the outputs fit or growth is unavailable', () => {
  expect(resolveTripWorkingBudget(1000, 400, 5000)).toBe(1000);
  expect(resolveTripWorkingBudget(1000, 4000, undefined)).toBe(1000);
});

test('budget grows to outputs plus a quarter headroom, limited by registry availability', () => {
  expect(resolveTripWorkingBudget(1000, 4000, 100000)).toBe(5000);
  expect(resolveTripWorkingBudget(1000, 4000, 3000)).toBe(3000);
  expect(resolveTripWorkingBudget(1000, 4000, 10)).toBe(1000);
});

test('estimate matches the runtime accounting and a large grouped trip grows instead of failing', async () => {
  const device = createNullWebGPUDevice();
  const fixture = createTabularTripRuntimeFixture(device, [0, 1, 0, 0, 0, 0]);
  try {
    const estimate = estimateTripOutputBytes(fixture.trips, {
      selectionMask: fixture.selectionMask
    });
    expect(estimate).toBe(124);
    const registry = {memoryUsage: {availableBytes: 1024 * 1024}} as never;
    expect(
      () =>
        new TripRuntime({
          device,
          column: 'id',
          trips: fixture.trips,
          selectionMask: fixture.selectionMask,
          workingMemoryBudgetBytes: 64
        })
    ).toThrow('outputs exceed');
    const runtime = new TripRuntime({
      device,
      column: 'id',
      trips: fixture.trips,
      selectionMask: fixture.selectionMask,
      workingMemoryBudgetBytes: 64,
      memoryRegistry: {
        ...(registry as object),
        reserveMemory: () => ({release() {}})
      } as never,
      growWorkingMemoryToRegistry: true
    });
    expect(runtime.accountedResidentByteLength).toBe(124);
    await runtime.destroy(Promise.resolve());
  } finally {
    fixture.destroy();
    device.destroy();
  }
});
