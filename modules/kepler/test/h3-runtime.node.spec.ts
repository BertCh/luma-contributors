// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {WebMercatorViewport} from '@deck.gl/core';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';
import {KeplerDatasetRegistry} from '../src/runtime/dataset-registry';
import {
  prepareH3Source,
  prepareH3ViewportMesh
} from '../../../examples/deck/kepler-compute/src/h3-source';
import {
  H3MetadataRuntime,
  H3Runtime,
  makeH3GPUVectors
} from '../../../examples/deck/kepler-compute/src/h3-runtime';

const CELL = '872830828ffffff';
const PENTAGON = '8009fffffffffff';
const viewport = new WebMercatorViewport({
  longitude: -122.42,
  latitude: 37.77,
  zoom: 10,
  width: 800,
  height: 600
});

function createLifecycleDevice(): Device {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cachePipelines: false, _cacheShaders: false}
  });
  // CPU lifecycle tests compile graph metadata only; shader execution is covered by WebGPU tests.
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {id?: string}) => ({id: props.id ?? '', hash: '', destroy: () => {}})
  });
  return device;
}
function source() {
  return prepareH3Source([
    {id: 'first', values: [CELL, null]},
    {id: 'empty', values: []},
    {id: 'last', values: [PENTAGON, CELL]}
  ]);
}
function mask(device: Device): GPUVector<'uint32'> {
  return new GPUVector({
    type: 'data',
    name: 'scope',
    format: 'uint32',
    data: [
      new GPUData({
        format: 'uint32',
        length: 4,
        ownsBuffer: true,
        buffer: device.createBuffer({
          data: new Uint32Array([1, 1, 0, 1]),
          usage: Buffer.STORAGE | Buffer.VERTEX
        })
      })
    ]
  });
}

test('H3 upload preserves source slots, primitive provenance and admitted CPU/GPU ownership through fences', async () => {
  const device = createLifecycleDevice();
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1 << 20});
  const prepared = source();
  const uploaded = makeH3GPUVectors(device, prepared, {
    workingMemoryBudgetBytes: 1 << 16,
    memoryRegistry: registry
  });
  const buffers = [
    ...new Set([
      ...uploaded.geometry.buffers,
      ...[
        uploaded.centers,
        uploaded.validity,
        uploaded.resolutions,
        uploaded.pentagons,
        uploaded.primitiveSourceRows,
        uploaded.outlineVertexRows
      ].flatMap(vector => vector.data.map(chunk => chunk.buffer))
    ])
  ];
  expect(uploaded.centers.data.map(chunk => chunk.length)).toEqual([2, 0, 2]);
  expect(uploaded.primitiveSourceRows.data.map(chunk => chunk.length)).toEqual([1, 0, 2]);
  expect([0, 1, 2, 3, -1].map(index => uploaded.resolveSourceRow(index))).toEqual([
    0, 2, 3, -1, -1
  ]);
  expect(uploaded.gpuByteLength).toBe(
    buffers.reduce((total, buffer) => total + buffer.byteLength, 0)
  );
  expect(registry.memoryUsage.reservedBytes).toEqual({
    cpu: uploaded.typedStorageByteLength,
    gpu: 1 << 16
  });
  expect(prepared.batches[0].sourceIds).toEqual([CELL, null]);
  const completion = createDeferred<void>();
  const pending = uploaded.destroy(completion.promise);
  expect(uploaded.isCurrent()).toBe(false);
  expect(buffers.every(buffer => !buffer.destroyed)).toBe(true);
  completion.resolve();
  await pending;
  expect(buffers.every(buffer => buffer.destroyed)).toBe(true);
  expect(registry.memoryUsage.totalBytes).toBe(0);
  device.destroy();
});

test('H3 metadata and render programs release only owned resources after their fence settles', async () => {
  const device = createLifecycleDevice();
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1 << 20});
  const uploaded = makeH3GPUVectors(device, source(), {
    workingMemoryBudgetBytes: 1 << 16,
    memoryRegistry: registry
  });
  const selectionMask = mask(device);
  const sourceBudget = registry.memoryUsage.totalBytes;
  const metadata = new H3MetadataRuntime({
    device,
    source: uploaded,
    mask: selectionMask,
    revision: 1,
    workingMemoryBudgetBytes: 1 << 14,
    memoryRegistry: registry
  });
  const runtime = new H3Runtime({
    device,
    column: 'cell',
    source: uploaded,
    branch: 'columns',
    selectionMask,
    coverage: 1,
    viewportMesh: prepareH3ViewportMesh({resolution: 7, viewport}),
    workingMemoryBudgetBytes: 1 << 16,
    memoryRegistry: registry
  });
  expect(runtime.binding.columns!.positions.data.map(chunk => chunk.length)).toEqual([1, 0, 2]);
  expect(runtime.binding.columns!.resolveSourceRow!(2)).toBe(3);
  expect(runtime.accountedResidentByteLength).toBeGreaterThan(0);
  expect(metadata.accountedResidentByteLength).toBeGreaterThan(0);
  expect(metadata.snapshot).toBeUndefined();
  expect(await metadata.readMetadata()).toBeNull();
  expect(metadata.readbackByteLength).toBe(0);
  const completion = createDeferred<void>();
  const pending = [runtime.destroy(completion.promise), metadata.destroy(completion.promise)];
  expect(runtime.accountedResidentByteLength).toBeGreaterThan(0);
  expect(metadata.accountedResidentByteLength).toBeGreaterThan(0);
  expect(registry.memoryUsage.totalBytes).toBe(sourceBudget + (1 << 14) + (1 << 16));
  completion.resolve();
  await Promise.all(pending);
  expect(runtime.accountedResidentByteLength).toBe(0);
  expect(metadata.accountedResidentByteLength).toBe(0);
  expect(registry.memoryUsage.totalBytes).toBe(sourceBudget);
  expect(uploaded.geometry.buffers.every(buffer => !buffer.destroyed)).toBe(true);
  expect(selectionMask.data[0].buffer.destroyed).toBe(false);
  await uploaded.destroy(Promise.resolve());
  selectionMask.destroy();
  device.destroy();
});

test('failed H3 source admission rolls back allocations and CPU reservations', () => {
  const device = createLifecycleDevice();
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1 << 20});
  const allocations: Buffer[] = [];
  const createBuffer = device.createBuffer.bind(device);
  const spy = vi.spyOn(device, 'createBuffer').mockImplementation(props => {
    const buffer = createBuffer(props);
    allocations.push(buffer);
    return buffer;
  });
  expect(() =>
    makeH3GPUVectors(device, source(), {workingMemoryBudgetBytes: 0, memoryRegistry: registry})
  ).toThrow();
  expect(allocations.every(buffer => buffer.destroyed)).toBe(true);
  expect(registry.memoryUsage.totalBytes).toBe(0);
  spy.mockRestore();
  device.destroy();
});

function createDeferred<Value>(): {promise: Promise<Value>; resolve(value: Value): void} {
  let resolve!: (value: Value) => void;
  const promise = new Promise<Value>(complete => {
    resolve = complete;
  });
  return {promise, resolve};
}

test('H3 Float32-collapsed boundaries upload fp64 low centers and are flagged for the fp64 column and polygon paths', async () => {
  const device = createLifecycleDevice();
  const prepared = prepareH3Source([{id: 'precision', values: ['8f6e434330114cc']}]);
  expect(prepared.validRowCount).toBe(1);
  expect(prepared.precision.collapsedRows).toHaveLength(1);
  const vectors = makeH3GPUVectors(device, prepared, {workingMemoryBudgetBytes: 1 << 20});
  expect(vectors.requiresFp64).toBe(true);
  expect(vectors.centersLow?.format).toBe('float32x2');
  expect(vectors.centersLow?.length).toBe(vectors.centers.length);
  // Polygon and outline vertex streams carry the low part too, so mixed/pentagon sets can render.
  expect(vectors.geometry.polygons.positionsLow?.format).toBe('vertex-list<float32x2>');
  expect(vectors.geometry.polygons.positionsLow?.data[0].valueLength).toBe(
    vectors.geometry.polygons.positions.data[0].valueLength
  );
  expect(vectors.geometry.polygonOutlines.getPathLow?.format).toBe('vertex-list<float32x2>');
  await vectors.destroy(Promise.resolve());
  device.destroy();
});

test('device loss releases H3 reservations even when the submitted completion fence cannot settle', async () => {
  const device = createLifecycleDevice();
  const loss = createDeferred<{reason: 'destroyed'; message: string}>();
  Object.defineProperty(device, 'lost', {value: loss.promise});
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1 << 20});
  const uploaded = makeH3GPUVectors(device, source(), {
    workingMemoryBudgetBytes: 1 << 16,
    memoryRegistry: registry
  });
  const selectionMask = mask(device);
  const metadata = new H3MetadataRuntime({
    device,
    source: uploaded,
    mask: selectionMask,
    revision: 1,
    workingMemoryBudgetBytes: 1 << 14,
    memoryRegistry: registry
  });
  const runtime = new H3Runtime({
    device,
    column: 'cell',
    source: uploaded,
    branch: 'columns',
    selectionMask,
    coverage: 1,
    viewportMesh: prepareH3ViewportMesh({resolution: 7, viewport}),
    workingMemoryBudgetBytes: 1 << 16,
    memoryRegistry: registry
  });
  const stranded = new Promise<void>(() => {});
  const pending = [
    metadata.destroy(stranded),
    runtime.destroy(stranded),
    uploaded.destroy(stranded)
  ];
  loss.resolve({reason: 'destroyed', message: 'simulated loss'});
  await Promise.all(pending);
  expect(registry.memoryUsage.totalBytes).toBe(0);
  expect(metadata.accountedResidentByteLength).toBe(0);
  expect(runtime.accountedResidentByteLength).toBe(0);
  expect(selectionMask.data[0].buffer.destroyed).toBe(false);
  selectionMask.destroy();
  device.destroy();
});
