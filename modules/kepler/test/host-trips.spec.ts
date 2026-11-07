// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {type GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {TripRuntime} from '../../../examples/deck/kepler-compute/src/trip-runtime';
import {KeplerDatasetRegistry} from '../src/runtime/dataset-registry';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {createTripRuntimeFixture} from './utils/trip-runtime-fixture';

async function readWords(vector: GPUVector): Promise<number[]> {
  const result: number[] = [];
  for (const chunk of vector.data) {
    if (!chunk.length) continue;
    const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 4);
    result.push(...new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length));
  }
  return result;
}

it('trip rendering inputs gather source masks/styles on GPU across unequal chunks without rescanning or uploading paths', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createTripRuntimeFixture(device);
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1024 * 1024});
  const runtime = new TripRuntime({
    device,
    column: 'route',
    ...fixture,
    workingMemoryBudgetBytes: 1024 * 1024,
    memoryRegistry: registry
  });
  const sourceBuffers = [
    fixture.trips.getPath,
    fixture.trips.getTimestamps,
    fixture.trips.sourceRowIds,
    fixture.selectionMask,
    fixture.colors,
    fixture.widths
  ].flatMap(vector => vector.data.map(chunk => chunk.buffer));
  const reads = sourceBuffers.map(buffer => vi.spyOn(buffer, 'readAsync'));
  const submit = vi.spyOn(device, 'submit');
  try {
    runtime.prepare();
    const binding = runtime.resolve('route')!;
    const allocations = [...runtime.resources];
    const createBuffer = vi.spyOn(device, 'createBuffer');
    const encoder = device.createCommandEncoder();
    runtime.encode(encoder);
    expect(submit).not.toHaveBeenCalled();
    for (const read of reads) expect(read).not.toHaveBeenCalled();
    device.submit(encoder.finish());
    expect(await readWords(binding.selectionMask)).toEqual([0, 1]);
    expect(await readWords(binding.getColor!)).toEqual([0xff060504, 0xff090807]);
    expect(await readWords(binding.getWidth!)).toEqual(
      Array.from(new Uint32Array(new Float32Array([20, 30]).buffer))
    );
    // Simulate upstream compute changing borrowed data; the runtime only records gathers.
    fixture.selectionMask.data[1].buffer.write(new Uint32Array([1, 0, 1]));
    fixture.widths.data[1].buffer.write(new Float32Array([25, 35, 45]));
    createBuffer.mockClear();
    const writes = sourceBuffers.map(buffer => vi.spyOn(buffer, 'write'));
    const next = device.createCommandEncoder();
    runtime.encode(next);
    expect(createBuffer).not.toHaveBeenCalled();
    for (const write of writes) {
      expect(write).not.toHaveBeenCalled();
      write.mockRestore();
    }
    device.submit(next.finish());
    expect(await readWords(binding.selectionMask)).toEqual([1, 0]);
    expect(await readWords(binding.getWidth!)).toEqual(
      Array.from(new Uint32Array(new Float32Array([25, 35]).buffer))
    );
    expect(runtime.resolve('route')).toBe(binding);
    expect(runtime.resources).toEqual(allocations);
    expect(binding.selectionMask.data.map(chunk => chunk.length)).toEqual([1, 0, 1]);
    expect(binding.resolveSourceRow(0)).toBe(1);
    expect(binding.resolveSourceRow(1)).toBe(2);
    expect(runtime.readbackByteLength).toBe(0);
    for (const read of reads) expect(read).not.toHaveBeenCalled();
    createBuffer.mockRestore();
  } finally {
    submit.mockRestore();
    for (const read of reads) read.mockRestore();
    const fence = device.createFence();
    try {
      await runtime.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
    expect(registry.memoryUsage.totalBytes).toBe(0);
    fixture.destroy();
  }
});
