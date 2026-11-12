// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {NullDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {expect, test, vi} from 'vitest';
import {GPUResidentExplorer} from '../src/exploration/gpu-resident-explorer';
import {KeplerAllocationError} from '../src/runtime/dataset-registry';

// Use real buffer copies and readback tickets, replacing only GPU graph compilation/execution.
function createExplorerFixture() {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxComputeWorkgroupsPerDimension: 65535,
      maxStorageBufferBindingSize: 128 * 1024 * 1024
    }
  });
  const source = device.createBuffer({
    data: new Uint32Array([10, 20]),
    usage: Buffer.STORAGE | Buffer.COPY_SRC
  });
  const column = new GPUVector({
    name: 'value',
    type: 'data',
    format: 'uint32',
    data: [new GPUData({buffer: source, format: 'uint32', length: 2})]
  });
  const selectedCount = device.createBuffer({
    data: new Uint32Array([2]),
    usage: Buffer.COPY_SRC
  });
  const selectedRows = device.createBuffer({
    data: new Uint32Array([0, 1]),
    usage: Buffer.COPY_SRC
  });
  const destroyCompiled = vi.fn();
  const explorer: GPUResidentExplorer = Object.assign(
    Object.create(GPUResidentExplorer.prototype),
    {
      device,
      columns: {value: column},
      validity: {},
      rowCount: 2,
      datasetRevision: 1,
      maximumPageSize: 2,
      maximumReadbackBytes: 64,
      maximumConcurrentReads: 2,
      selectedCount,
      selectedRows: new GPUVector({
        name: 'selected-rows',
        type: 'data',
        format: 'uint32',
        data: [new GPUData({buffer: selectedRows, format: 'uint32', length: 2})]
      }),
      histogramBuffers: new Map(),
      domainBuffers: new Map(),
      predicates: new Map(),
      domainPredicates: new Map(),
      domainFilters: [],
      vectors: [],
      compiled: {destroy: destroyCompiled},
      filter: {destroy: vi.fn()},
      destroyed: false,
      activeReads: 0,
      readbackRings: [],
      revision: 1,
      encodedRevision: 1,
      transferredBytes: 0,
      resourcesReleased: false,
      releaseWaiters: []
    }
  );
  return {
    device,
    explorer,
    destroyCompiled,
    async destroy() {
      await explorer.destroyAsync();
      source.destroy();
      selectedCount.destroy();
      selectedRows.destroy();
      device.destroy();
    }
  };
}

test('readback retries after the first staging allocation fails', async () => {
  const fixture = createExplorerFixture();
  const failure = new Error('First staging allocation failed');
  const createBuffer = fixture.device.createBuffer.bind(fixture.device);
  const stagingBuffers: Buffer[] = [];
  const allocations = vi.spyOn(fixture.device, 'createBuffer').mockImplementation(() => {
    throw failure;
  });
  try {
    const error = await fixture.explorer.readSummary().catch(reason => reason);
    expect(error).toBeInstanceOf(KeplerAllocationError);
    expect(error.reason).toBe('device-out-of-memory');
    expect(error.cause).toBe(failure);
    expect(allocations).toHaveBeenCalledOnce();
    allocations.mockImplementation(props => {
      const buffer = createBuffer(props);
      stagingBuffers.push(buffer);
      return buffer;
    });
    expect(await fixture.explorer.readSelectedCount()).toEqual({
      revision: 1,
      datasetRevision: 1,
      selectedCount: 2
    });
    expect(await fixture.explorer.readSummary()).toMatchObject({selectedCount: 2});
    expect(allocations).toHaveBeenCalledTimes(2);
    expect(stagingBuffers).toHaveLength(1);
    await fixture.explorer.destroyAsync();
    expect(fixture.destroyCompiled).toHaveBeenCalledOnce();
    expect(stagingBuffers[0].destroyed).toBe(true);
  } finally {
    allocations.mockRestore();
    await fixture.destroy();
  }
});

test('failed second ring allocation preserves the busy first read and releases its staging', async () => {
  const fixture = createExplorerFixture();
  const failure = new Error('Second staging allocation failed');
  let finishMapping!: () => void;
  const mapping = new Promise<void>(resolve => {
    finishMapping = resolve;
  });
  const createBuffer = fixture.device.createBuffer.bind(fixture.device);
  const stagingBuffers: Buffer[] = [];
  const allocations = vi.spyOn(fixture.device, 'createBuffer').mockImplementation(props => {
    if (stagingBuffers.length) throw failure;
    const buffer = createBuffer(props);
    stagingBuffers.push(buffer);
    const read = buffer.readAsync.bind(buffer);
    vi.spyOn(buffer, 'readAsync').mockImplementationOnce(async (byteOffset, byteLength) => {
      await mapping;
      return read(byteOffset, byteLength);
    });
    return buffer;
  });
  try {
    const firstRead = fixture.explorer.readSummary().catch(reason => reason);
    await vi.waitFor(() => expect(stagingBuffers[0]?.readAsync).toHaveBeenCalledOnce());
    const error = await fixture.explorer.readSelectedCount().catch(reason => reason);
    expect(error).toBeInstanceOf(KeplerAllocationError);
    expect(error.reason).toBe('device-out-of-memory');
    expect(error.cause).toBe(failure);
    expect(allocations).toHaveBeenCalledTimes(2);
    expect(stagingBuffers).toHaveLength(1);
    expect(stagingBuffers[0].destroyed).toBe(false);
    finishMapping();
    expect(await firstRead).toMatchObject({selectedCount: 2});
    expect(await fixture.explorer.readSummary()).toMatchObject({selectedCount: 2});
    expect(allocations).toHaveBeenCalledTimes(2);
    await fixture.explorer.destroyAsync();
    expect(fixture.destroyCompiled).toHaveBeenCalledOnce();
    expect(stagingBuffers.every(buffer => buffer.destroyed)).toBe(true);
  } finally {
    finishMapping();
    allocations.mockRestore();
    await fixture.destroy();
  }
});

test('single-slot rings grow only for busy reads, stay within the reservation and are reused', async () => {
  const fixture = createExplorerFixture();
  let finishMapping!: () => void;
  const mapping = new Promise<void>(resolve => {
    finishMapping = resolve;
  });
  const createBuffer = fixture.device.createBuffer.bind(fixture.device);
  const stagingBuffers: Buffer[] = [];
  const allocations = vi.spyOn(fixture.device, 'createBuffer').mockImplementation(props => {
    const buffer = createBuffer(props);
    stagingBuffers.push(buffer);
    const read = buffer.readAsync.bind(buffer);
    vi.spyOn(buffer, 'readAsync').mockImplementationOnce(async (byteOffset, byteLength) => {
      await mapping;
      return read(byteOffset, byteLength);
    });
    return buffer;
  });
  try {
    expect(allocations).not.toHaveBeenCalled();
    const firstRead = fixture.explorer.readSummary().catch(reason => reason);
    expect(stagingBuffers).toHaveLength(1);
    const secondRead = fixture.explorer.readSelectedCount();
    expect(stagingBuffers).toHaveLength(2);
    expect(stagingBuffers.map(buffer => buffer.byteLength)).toEqual([16, 16]);
    expect(
      stagingBuffers.reduce((total, buffer) => total + buffer.byteLength, 0)
    ).toBeLessThanOrEqual(fixture.explorer.inspectionReservationByteLength / 2);
    expect(await fixture.explorer.readSummary()).toBeNull();
    expect(allocations).toHaveBeenCalledTimes(2);
    finishMapping();
    expect(await firstRead).toMatchObject({selectedCount: 2});
    expect(await secondRead).toMatchObject({selectedCount: 2});
    expect(await fixture.explorer.readSummary()).toMatchObject({selectedCount: 2});
    expect(allocations).toHaveBeenCalledTimes(2);
    expect(fixture.explorer.readbackByteLength).toBe(12);
    await fixture.explorer.destroyAsync();
    expect(stagingBuffers.every(buffer => buffer.destroyed)).toBe(true);
  } finally {
    finishMapping();
    allocations.mockRestore();
    await fixture.destroy();
  }
});

test.each([
  'replace',
  'grow'
])('readRows captures source rows before pending compilation when callers %s the array', async mutation => {
  const fixture = createExplorerFixture();
  let finishCompilation!: (compiled: CompiledGPUCommandGraph) => void;
  const compilation = new Promise<CompiledGPUCommandGraph>(resolve => {
    finishCompilation = resolve;
  });
  const compile = vi.spyOn(GPUCommandGraph.prototype, 'compileAsync').mockReturnValue(compilation);
  const indices: Buffer[] = [];
  const uploads: number[][] = [];
  const createBuffer = fixture.device.createBuffer.bind(fixture.device);
  const allocations = vi.spyOn(fixture.device, 'createBuffer').mockImplementation(props => {
    const buffer = createBuffer(props);
    if (props && 'id' in props && props.id === 'inspection-indices') {
      indices.push(buffer);
      const write = buffer.write.bind(buffer);
      vi.spyOn(buffer, 'write').mockImplementation((data, byteOffset) => {
        uploads.push(Array.from(data as Uint32Array));
        write(data, byteOffset);
      });
    }
    return buffer;
  });
  const destroyGather = vi.fn();
  try {
    const sourceRows = [0];
    const reading = fixture.explorer.readRows(sourceRows, ['value']);
    expect(compile).toHaveBeenCalledOnce();
    expect(uploads).toEqual([]);
    sourceRows[0] = 1;
    if (mutation === 'grow') sourceRows.push(0, 1);
    finishCompilation({
      encode: vi.fn(),
      destroy: destroyGather
    } as unknown as CompiledGPUCommandGraph);
    const page = await reading;
    expect(uploads).toEqual([[0]]);
    expect(page?.sourceRowIndices).toEqual(new Uint32Array([0]));
    expect(indices[0].byteLength).toBe(4);
    expect(indices[0].destroyed).toBe(true);
    expect(destroyGather).toHaveBeenCalledOnce();
  } finally {
    compile.mockRestore();
    allocations.mockRestore();
    await fixture.destroy();
  }
});

test('readPage captures requested columns before its count mapping suspends', async () => {
  const fixture = createExplorerFixture();
  let finishMapping!: () => void;
  const mapping = new Promise<void>(resolve => {
    finishMapping = resolve;
  });
  const createBuffer = fixture.device.createBuffer.bind(fixture.device);
  const allocations = vi.spyOn(fixture.device, 'createBuffer').mockImplementation(props => {
    const buffer = createBuffer(props);
    const read = buffer.readAsync.bind(buffer);
    vi.spyOn(buffer, 'readAsync').mockImplementationOnce(async (byteOffset, byteLength) => {
      await mapping;
      return read(byteOffset, byteLength);
    });
    return buffer;
  });
  const destroyGather = vi.fn();
  const compile = vi.spyOn(GPUCommandGraph.prototype, 'compileAsync').mockResolvedValue({
    encode: vi.fn(),
    destroy: destroyGather
  } as unknown as CompiledGPUCommandGraph);
  try {
    const columns = ['value'];
    const reading = fixture.explorer.readPage(0, 1, columns);
    columns[0] = 'missing';
    finishMapping();
    const page = await reading;
    expect(Object.keys(page!.columns)).toEqual(['value']);
    expect(page?.sourceRowIndices).toEqual(new Uint32Array([0]));
    expect(destroyGather).toHaveBeenCalledOnce();
  } finally {
    allocations.mockRestore();
    compile.mockRestore();
    await fixture.destroy();
  }
});
