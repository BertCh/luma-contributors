// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {createNullWebGPUDevice} from '../../graph/test/utils/gpu-contributor-test-utils';
import {KeplerDatasetRegistry} from '../src/runtime/dataset-registry';
import {
  IconRuntime,
  makeGPUVectorsFromNativeIcons
} from '../../../examples/deck/kepler-compute/src/icon-runtime';
import {
  prepareNativeIconGeometry,
  prepareNativeIconSource
} from '../../../examples/deck/kepler-compute/src/icon-source';

function source() {
  return prepareNativeIconSource(
    [
      {id: 'first', icons: ['alpha', null]},
      {id: 'empty', icons: []},
      {id: 'last', icons: ['alpha']}
    ],
    prepareNativeIconGeometry({alpha: [0, 1, 0, -1, -1, 0, 1, -1, 0]})
  );
}
function vectors(device: Device) {
  const positions = new GPUVector({
    type: 'data',
    name: 'positions',
    format: 'float32x2',
    data: [
      new GPUData({
        format: 'float32x2',
        length: 3,
        ownsBuffer: true,
        buffer: device.createBuffer({
          data: new Float32Array([0, 0, 1, 1, 2, 2]),
          usage: Buffer.STORAGE | Buffer.VERTEX
        })
      })
    ]
  });
  const selectionMask = new GPUVector({
    type: 'data',
    name: 'mask',
    format: 'uint32',
    data: [
      new GPUData({
        format: 'uint32',
        length: 3,
        ownsBuffer: true,
        buffer: device.createBuffer({
          data: new Uint32Array([1, 1, 1]),
          usage: Buffer.STORAGE | Buffer.VERTEX
        })
      })
    ]
  });
  return {
    positions,
    selectionMask,
    destroy() {
      positions.destroy();
      selectionMask.destroy();
    }
  };
}

test('native icon reservations account retained CPU arrays and GPU outputs until their respective fences settle', async () => {
  const device = createNullWebGPUDevice();
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1024});
  const prepared = source();
  const icons = makeGPUVectorsFromNativeIcons(device, prepared, {
    workingMemoryBudgetBytes: 128,
    memoryRegistry: registry
  });
  const inputs = vectors(device);
  const runtime = new IconRuntime({
    device,
    columns: {lat: 'lat', lng: 'lng', icon: 'marker'},
    icons,
    ...inputs,
    workingMemoryBudgetBytes: 128,
    memoryRegistry: registry
  });
  expect(prepared.typedStorageByteLength).toBe(44);
  expect(icons.gpuByteLength).toBe(48);
  expect(registry.memoryUsage.reservedBytes).toEqual({cpu: 44, gpu: 176});
  expect(runtime.binding.groups[0].positions.data.map(chunk => chunk.length)).toEqual([1, 0, 1]);
  expect(runtime.accountedResidentByteLength).toBe(32);
  const completion = Promise.withResolvers<void>();
  const pending = runtime.destroy(completion.promise);
  expect(runtime.resources.every(buffer => !buffer.destroyed)).toBe(true);
  completion.resolve();
  await pending;
  expect(registry.memoryUsage.totalBytes).toBe(92);
  expect(icons.groups[0].iconGeometry.buffer.destroyed).toBe(false);
  expect(inputs.positions.data[0].buffer.destroyed).toBe(false);
  await icons.destroy(Promise.resolve());
  expect(registry.memoryUsage.totalBytes).toBe(0);
  inputs.destroy();
  device.destroy();
});

test('captured device loss retires icon source and gather reservations without waiting for a stranded fence', async () => {
  const device = createNullWebGPUDevice();
  const loss = Promise.withResolvers<{reason: 'destroyed'; message: string}>();
  Object.defineProperty(device, 'lost', {value: loss.promise, configurable: true});
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1024});
  const icons = makeGPUVectorsFromNativeIcons(device, source(), {
    workingMemoryBudgetBytes: 128,
    memoryRegistry: registry
  });
  const inputs = vectors(device);
  const runtime = new IconRuntime({
    device,
    columns: {lat: 'lat', lng: 'lng', icon: 'marker'},
    icons,
    ...inputs,
    workingMemoryBudgetBytes: 128,
    memoryRegistry: registry
  });
  const stranded = new Promise<void>(() => {});
  Object.defineProperty(device, 'lost', {value: stranded});
  const pending = [runtime.destroy(stranded), icons.destroy(stranded)];
  expect(icons.isCurrent()).toBe(false);
  expect(registry.memoryUsage.totalBytes).toBe(220);
  loss.resolve({reason: 'destroyed', message: 'test loss'});
  await Promise.all(pending);
  expect(registry.memoryUsage.totalBytes).toBe(0);
  expect(runtime.resources.every(buffer => buffer.destroyed)).toBe(true);
  inputs.destroy();
  device.destroy();
});

test('partial source allocation failure releases both reservations and every prior allocation', () => {
  const device = createNullWebGPUDevice();
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1024});
  const buffer = device.createBuffer({byteLength: 36});
  const allocation = vi
    .spyOn(device, 'createBuffer')
    .mockReturnValueOnce(buffer)
    .mockImplementationOnce(() => {
      throw new Error('allocation failure');
    });
  expect(() =>
    makeGPUVectorsFromNativeIcons(device, source(), {
      workingMemoryBudgetBytes: 128,
      memoryRegistry: registry
    })
  ).toThrow('allocation failure');
  expect(buffer.destroyed).toBe(true);
  expect(registry.memoryUsage.totalBytes).toBe(0);
  allocation.mockRestore();
  device.destroy();
});
