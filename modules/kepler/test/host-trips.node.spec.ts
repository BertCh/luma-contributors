// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';
import {expect, test, vi} from 'vitest';
import {KeplerDatasetRegistry} from '../src/runtime/dataset-registry';
import {TripRuntime} from '../../../examples/deck/kepler-compute/src/trip-runtime';
import {createTripRuntimeFixture} from './utils/trip-runtime-fixture';

test('trip gathers preserve empty output chunks and own only admitted outputs until their fence settles', async () => {
  const device = createNullWebGPUDevice();
  const fixture = createTripRuntimeFixture(device);
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1024});
  const runtime = new TripRuntime({
    device,
    column: 'route',
    ...fixture,
    workingMemoryBudgetBytes: 128,
    memoryRegistry: registry
  });
  try {
    expect(runtime.binding.getPath).toBe(fixture.trips.getPath);
    expect(runtime.binding.getTimestamps).toBe(fixture.trips.getTimestamps);
    expect(runtime.binding.selectionMask.data.map(chunk => chunk.length)).toEqual([1, 0, 1]);
    expect(runtime.binding.getColor!.data.map(chunk => chunk.length)).toEqual([1, 0, 1]);
    expect(runtime.binding.resolveSourceRow(0)).toBe(1);
    expect(runtime.binding.resolveSourceRow(1)).toBe(2);
    expect(runtime.binding.animationDomainMilliseconds).toEqual([1000, 1100]);
    expect(runtime.accountedResidentByteLength).toBe(36);
    expect(registry.memoryUsage.totalBytes).toBe(128);
    expect(runtime.resolve('other')).toBeUndefined();
    const completion = Promise.withResolvers<void>();
    const destroyed = runtime.destroy(completion.promise);
    expect(runtime.destroy(Promise.resolve())).toBe(destroyed);
    expect(runtime.resolve('route')).toBeUndefined();
    expect(runtime.resources.every(buffer => !buffer.destroyed)).toBe(true);
    expect(runtime.accountedResidentByteLength).toBe(36);
    expect(registry.memoryUsage.totalBytes).toBe(128);
    completion.resolve();
    await destroyed;
    expect(runtime.resources.every(buffer => buffer.destroyed)).toBe(true);
    expect(registry.memoryUsage.totalBytes).toBe(0);
    expect(runtime.accountedResidentByteLength).toBe(0);
    expect(fixture.trips.getPath.data.every(chunk => !chunk.buffer.destroyed)).toBe(true);
    expect(fixture.selectionMask.data.every(chunk => !chunk.buffer.destroyed)).toBe(true);
  } finally {
    await runtime.destroy(Promise.resolve());
    fixture.destroy();
    device.destroy();
  }
});

test('trip admission rejects insufficient budgets before allocation and allocation failure returns its reservation', () => {
  const device = createNullWebGPUDevice();
  const fixture = createTripRuntimeFixture(device);
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 128});
  const createBuffer = vi.spyOn(device, 'createBuffer');
  try {
    expect(
      () =>
        new TripRuntime({
          device,
          column: 'route',
          ...fixture,
          workingMemoryBudgetBytes: 35,
          memoryRegistry: registry
        })
    ).toThrow('outputs exceed');
    expect(createBuffer).not.toHaveBeenCalled();
    expect(registry.memoryUsage.totalBytes).toBe(0);
    const allocate = createBuffer.getMockImplementation() ?? device.createBuffer.bind(device);
    createBuffer.mockRestore();
    const buffer = allocate({byteLength: 4});
    const failing = vi
      .spyOn(device, 'createBuffer')
      .mockReturnValueOnce(buffer)
      .mockImplementationOnce(() => {
        throw new Error('allocation failed');
      });
    expect(
      () =>
        new TripRuntime({
          device,
          column: 'route',
          ...fixture,
          workingMemoryBudgetBytes: 128,
          memoryRegistry: registry
        })
    ).toThrow('allocation failed');
    expect(buffer.destroyed).toBe(true);
    expect(registry.memoryUsage.totalBytes).toBe(0);
    failing.mockRestore();
  } finally {
    createBuffer.mockRestore();
    fixture.destroy();
    device.destroy();
  }
});
