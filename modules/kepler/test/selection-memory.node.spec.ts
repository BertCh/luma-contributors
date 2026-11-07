// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {NullDevice} from '@luma.gl/core/test-utils';
import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {expect, test, vi} from 'vitest';
import {KeplerDatasetRegistry} from '../src/runtime/dataset-registry';
import {GPUKeplerSelectionScopes} from '../src/selection/gpu-selection-scopes';
import {GPUKeplerSpatialSelection} from '../src/selection/gpu-spatial-selection';

function fixture(budget = 1024, scopeBudget = 512) {
  const device = new NullDevice({});
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: budget});
  const scopes = new GPUKeplerSelectionScopes({
    device,
    registry,
    sourceChunkLengths: [2, 0, 1],
    workingMemoryBudgetBytes: scopeBudget
  });
  return {device, registry, scopes};
}
function points(device: NullDevice) {
  return new GPUVector({
    type: 'data',
    name: 'points',
    format: 'float32x2',
    data: [2, 0, 1].map(
      length =>
        new GPUData({
          buffer: device.createBuffer({byteLength: Math.max(8, length * 8), usage: Buffer.STORAGE}),
          format: 'float32x2',
          length,
          ownsBuffer: true
        })
    ),
    ownsData: true
  });
}

test('scope masks reserve live topology, reject growth before allocation and roll back partial layer creation', async () => {
  const {device, registry, scopes} = fixture(256, 256);
  const positions = points(device);
  // Five masks, each retaining the explicit empty source chunk: (8 + 4 + 4) * 5.
  expect(scopes.reservedMemoryByteLength).toBe(80);
  expect(registry.memoryUsage.totalBytes).toBe(80);
  const competitor = registry.reserveMemory(151);
  const allocate = vi.spyOn(device, 'createBuffer');
  expect(() =>
    scopes.setLayers([{id: 'points', geometryRevision: 1, geometry: {kind: 'point', positions}}])
  ).toThrow('budget');
  expect(allocate).toHaveBeenCalledTimes(3);
  for (const result of allocate.mock.results) expect(result.value.destroyed).toBe(true);
  expect(scopes.reservedMemoryByteLength).toBe(80);
  expect(scopes.workingMemoryByteLength).toBe(80);
  expect(() => scopes.getLayerMask('points')).toThrow('not registered');
  competitor.resize(161);
  allocate.mockClear();
  expect(() => scopes.getPlotMask('new')).toThrow('budget');
  expect(allocate).not.toHaveBeenCalled();
  expect(registry.memoryUsage.totalBytes).toBe(241);
  competitor.release();
  scopes.setLayers([{id: 'points', geometryRevision: 1, geometry: {kind: 'point', positions}}]);
  expect(scopes.reservedMemoryByteLength).toBe(112);
  allocate.mockRestore();
  await scopes.destroy(Promise.resolve());
  expect(registry.memoryUsage.totalBytes).toBe(0);
  positions.destroy();
  await registry.destroy();
  device.destroy();
});

test('retired masks retain their charge until the consumer fence settles', async () => {
  const {device, registry, scopes} = fixture();
  const mask = scopes.getPlotMask('filter');
  const buffer = mask.data[0].buffer;
  expect(scopes.reservedMemoryByteLength).toBe(96);
  scopes.removePlot('filter');
  let complete!: () => void;
  const fence = new Promise<void>(resolve => {
    complete = resolve;
  });
  const retiring = scopes.collectRetired(fence);
  await Promise.resolve();
  expect(scopes.reservedMemoryByteLength).toBe(96);
  expect(buffer.destroyed).toBe(false);
  complete();
  await retiring;
  expect(buffer.destroyed).toBe(true);
  expect(scopes.reservedMemoryByteLength).toBe(80);
  const completion = scopes.destroy(Promise.resolve());
  expect(registry.memoryUsage.totalBytes).toBe(80);
  await completion;
  expect(scopes.reservedMemoryByteLength).toBe(0);
  expect(registry.memoryUsage.totalBytes).toBe(0);
  device.destroy();
});

test('scope hard capacity stays separate from the smaller live registry charge', async () => {
  const {device, registry, scopes} = fixture(4096, 96);
  scopes.getPlotMask('first');
  const allocate = vi.spyOn(device, 'createBuffer');
  expect(() => scopes.getPlotMask('second')).toThrow('working memory');
  expect(allocate).not.toHaveBeenCalled();
  expect(registry.memoryUsage.totalBytes).toBe(96);
  allocate.mockRestore();
  registry.handleDeviceLoss();
  expect(scopes.reservedMemoryByteLength).toBe(0);
  expect(() => scopes.getPlotMask('after-loss')).toThrow('inactive');
  await scopes.destroy(Promise.resolve());
  expect(registry.memoryUsage.totalBytes).toBe(0);
  device.destroy();
});

test('spatial owned-output admission rejects before creating any buffer', () => {
  const device = new NullDevice({});
  // Exercise the WebGPU-only adapter's preallocation boundary without starting a GPU backend.
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  const positions = points(device);
  const allocate = vi.spyOn(device, 'createBuffer');
  const admitMemory = vi.fn(() => {
    throw new Error('not admitted');
  });
  expect(
    () =>
      new GPUKeplerSpatialSelection({
        device,
        positions,
        selection: {kind: 'rectangle', bounds: [0, 0, 1, 1]},
        coordinateSpace: 'planar',
        admitMemory
      })
  ).toThrow('not admitted');
  expect(admitMemory).toHaveBeenCalledWith(120);
  expect(allocate).not.toHaveBeenCalled();
  allocate.mockRestore();
  positions.destroy();
  device.destroy();
});

test('composition planning rejects unavailable scratch before allocation and rolls back failed compilation', async () => {
  const {device, registry, scopes} = fixture(1024, 512);
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.assign(device.limits, {
    maxComputeWorkgroupsPerDimension: 65535,
    maxComputeWorkgroupSizeX: 256,
    maxComputeInvocationsPerWorkgroup: 256,
    maxStorageBufferBindingSize: 2 ** 27,
    maxStorageBuffersPerShaderStage: 16,
    maxBindGroups: 4,
    maxBindingsPerBindGroup: 1000
  });
  const allocate = vi.spyOn(device, 'createBuffer');
  const compile = vi.spyOn(GPUCommandGraph.prototype, 'compile');
  const competitor = registry.reserveMemory(944);
  expect(() => scopes.encode(device.commandEncoder)).toThrow('budget');
  expect(compile).not.toHaveBeenCalled();
  expect(allocate).not.toHaveBeenCalled();
  expect(scopes.reservedMemoryByteLength).toBe(80);
  competitor.release();
  // NullDevice deliberately rejects executable pipelines after graph transients were allocated.
  // Compilation must destroy those transients before the scope returns its temporary charge.
  expect(() => scopes.encode(device.commandEncoder)).toThrow('ComputePipeline is not supported');
  expect(compile).toHaveBeenCalledTimes(1);
  expect(allocate).toHaveBeenCalled();
  for (const result of allocate.mock.results) expect(result.value.destroyed).toBe(true);
  expect(scopes.workingMemoryByteLength).toBe(80);
  expect(scopes.reservedMemoryByteLength).toBe(80);
  compile.mockRestore();
  allocate.mockRestore();
  await scopes.destroy(Promise.resolve());
  expect(registry.memoryUsage.totalBytes).toBe(0);
  device.destroy();
});
