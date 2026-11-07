// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUResidentExplorer} from '../src/exploration/gpu-resident-explorer';

it('resident explorer composes nullable, category and spatial filters with bounded inspection', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU is required');
  const value = makeVector(device, 'float32', [
    new Float32Array([5, 15, 25]),
    new Float32Array(),
    new Float32Array([35, 45, 55])
  ]);
  const category = makeVector(device, 'uint32', [
    new Uint32Array([0, 1, 2]),
    new Uint32Array(),
    new Uint32Array([0, 1, 2])
  ]);
  const validity = makeVector(device, 'uint32', [
    new Uint32Array([1, 0, 1]),
    new Uint32Array(),
    new Uint32Array([1, 1, 1])
  ]);
  const spatialMask = makeVector(device, 'uint32', [
    new Uint32Array([1, 1, 0]),
    new Uint32Array(),
    new Uint32Array([1, 1, 1])
  ]);
  const explorer = new GPUResidentExplorer({
    device,
    columns: {value, category},
    validity: {value: validity},
    dimensions: [
      {id: 'value', kind: 'range', column: 'value'},
      {id: 'category', kind: 'category', column: 'category', capacity: 4},
      {id: 'spatial', kind: 'external', mask: spatialMask}
    ],
    histograms: [
      {id: 'distribution', column: 'value', dimension: 'value', domain: [0, 60], binCount: 3}
    ],
    domains: [
      {id: 'selected', column: 'value'},
      {id: 'global', column: 'value', dimensions: []},
      {id: 'category-domain', column: 'value', dimensions: ['category']}
    ],
    datasetRevision: 3,
    maximumPageSize: 3
  });
  try {
    expect(explorer.selectionMask.data.map(data => data.length)).toEqual([3, 0, 3]);
    expect(await explorer.readSummary()).toBeNull();
    submit(device, explorer);
    const bytesBeforeCount = explorer.readbackByteLength;
    expect(await explorer.readSelectedCount()).toEqual({
      revision: 0,
      datasetRevision: 3,
      selectedCount: 6
    });
    expect(explorer.readbackByteLength - bytesBeforeCount).toBe(4);
    expect(explorer.maximumInspectionPageSize).toBe(3);
    expect(explorer.maximumInspectionReadbackBytes).toBe(1024 * 1024);
    expect(await explorer.readSummary()).toMatchObject({
      selectedCount: 6,
      domains: {selected: [5, 55], global: [5, 55], 'category-domain': [5, 55]},
      datasetRevision: 3,
      histograms: {distribution: new Uint32Array([1, 2, 2])}
    });
    explorer.setRange('value', 20, 50);
    explorer.setCategories('category', [0, 2]);
    expect(await explorer.readSelectedCount()).toBeNull();
    expect(await explorer.readSummary()).toBeNull();
    submit(device, explorer);
    expect(await explorer.readSummary()).toMatchObject({
      selectedCount: 2,
      histograms: {distribution: new Uint32Array([1, 2, 1])}
    });
    expect((await explorer.readSummary())?.domains).toEqual({
      selected: [25, 35],
      global: [5, 55],
      'category-domain': [5, 55]
    });
    const page = await explorer.readPage(1, 1);
    expect(page?.sourceRowIndices).toEqual(new Uint32Array([3]));
    expect(page?.columns['value']).toEqual(new Float32Array([35]));
    expect(page?.columns['category']).toEqual(new Uint32Array([0]));
    expect(page?.validity['value']).toEqual(new Uint32Array([1]));
    const tooltip = await explorer.readRows([1, 5], ['value']);
    expect(tooltip?.columns['value']).toEqual(new Float32Array([15, 55]));
    expect(tooltip?.validity['value']).toEqual(new Uint32Array([0, 1]));
    explorer.setExternalActive('spatial', true);
    submit(device, explorer);
    expect((await explorer.readSummary())?.selectedCount).toBe(1);
    explorer.setCategories('category', []);
    submit(device, explorer);
    expect((await explorer.readSummary())?.selectedCount).toBe(0);
    expect((await explorer.readPage(0, 3))?.sourceRowIndices.length).toBe(0);
    expect((await explorer.readSummary())?.domains['selected']).toBeNull();
    explorer.setEmpty('value');
    submit(device, explorer);
    expect((await explorer.readSummary())?.selectedCount).toBe(0);
    explorer.clearAll();
    submit(device, explorer);
    const stale = explorer.readSummary();
    explorer.setRange('value', 0, 30);
    expect(await stale).toBeNull();
    submit(device, explorer);
    expect((await explorer.readSummary())?.selectedCount).toBe(2);
    expect((await explorer.readSummary())?.domains).toEqual({
      selected: [5, 25],
      global: [5, 55],
      'category-domain': [5, 55]
    });
    await expect(explorer.readPage(0, 4)).rejects.toThrow('bounded');
    explorer.discardEncoding();
    expect(await explorer.readSummary()).toBeNull();
  } finally {
    explorer.destroy();
    expect((value.data[0].buffer as Buffer).destroyed).toBe(false);
    value.destroy();
    category.destroy();
    validity.destroy();
    spatialMask.destroy();
  }
});

it('resident explorer preserves half-open intervals, bounded maximum-category plans and empty chunks', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU is required');
  const value = makeVector(device, 'uint32', [new Uint32Array([0, 1, 2, 0xffffffff])]);
  const explorer = new GPUResidentExplorer({
    device,
    columns: {value},
    dimensions: [
      {id: 'range', kind: 'range', column: 'value', exclusiveMaximum: true},
      {id: 'category', kind: 'category', column: 'value', capacity: 64}
    ]
  });
  try {
    explorer.setCategories('category', [0xffffffff, 1, 2]);
    submit(device, explorer);
    expect((await explorer.readSummary())?.selectedCount).toBe(3);
    explorer.setRange('range', 1, 2);
    submit(device, explorer);
    expect((await explorer.readPage(0, 4))?.sourceRowIndices).toEqual(new Uint32Array([1]));
    const pending = explorer.readSummary();
    explorer.destroy();
    expect(await pending).toBeNull();
    expect((value.data[0].buffer as Buffer).destroyed).toBe(false);
  } finally {
    explorer.destroy();
    value.destroy();
  }
  const empty = makeVector(device, 'float32', [new Float32Array()]);
  const emptyExplorer = new GPUResidentExplorer({device, columns: {empty}, dimensions: []});
  try {
    submit(device, emptyExplorer);
    expect((await emptyExplorer.readSummary())?.selectedCount).toBe(0);
  } finally {
    emptyExplorer.destroy();
    empty.destroy();
  }
});

it('resident explorer bounds concurrent readback and releases failed construction allocations', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU is required');
  const value = makeVector(device, 'float32', [new Float32Array([1, 2, 3, 4])]);
  const wrongValidity = makeVector(device, 'uint32', [new Uint32Array([1])]);
  const allocations = vi.spyOn(device, 'createBuffer');
  expect(
    () =>
      new GPUResidentExplorer({
        device,
        columns: {value},
        validity: {value: wrongValidity},
        dimensions: []
      })
  ).toThrow();
  for (const result of allocations.mock.results) {
    if (result.type === 'return') expect(result.value.destroyed).toBe(true);
  }
  allocations.mockRestore();
  const sourceRead = vi.spyOn(value.data[0].buffer as Buffer, 'readAsync');
  const explorer = new GPUResidentExplorer({
    device,
    columns: {value},
    dimensions: [],
    maximumConcurrentReads: 2,
    maximumReadbackBytes: 64,
    maximumPageSize: 16
  });
  try {
    submit(device, explorer);
    const first = explorer.readSummary();
    const second = explorer.readSummary();
    expect(await explorer.readSummary()).toBeNull();
    expect((await first)?.selectedCount).toBe(4);
    expect((await second)?.selectedCount).toBe(4);
    expect((await explorer.readPage(0, 2))?.columns['value']).toEqual(new Float32Array([1, 2]));
    expect(sourceRead).not.toHaveBeenCalled();
    await expect(explorer.readPage(0, 8)).rejects.toThrow('byte budget');
    expect(explorer.accountedResidentByteLength).toBeGreaterThanOrEqual(
      explorer.ownedOutputByteLength
    );
    expect(explorer.inspectionReservationByteLength).toBe(256);
    expect(explorer.readbackByteLength).toBeGreaterThan(0);
    const pending = explorer.readSummary();
    const destroyed = explorer.destroyAsync();
    expect(await pending).toBeNull();
    await destroyed;
    expect((explorer.selectionMask.data[0].buffer as Buffer).destroyed).toBe(true);
    expect((value.data[0].buffer as Buffer).destroyed).toBe(false);
  } finally {
    sourceRead.mockRestore();
    explorer.destroy();
    value.destroy();
    wrongValidity.destroy();
  }
});

function makeVector<Format extends 'float32' | 'uint32'>(
  device: Device,
  format: Format,
  chunks: (Float32Array | Uint32Array)[]
): GPUVector<Format> {
  return new GPUVector({
    name: 'fixture',
    type: 'data',
    format,
    ownsData: true,
    data: chunks.map(
      chunk =>
        new GPUData({
          format,
          length: chunk.length,
          ownsBuffer: true,
          buffer: device.createBuffer({
            byteLength: Math.max(4, chunk.byteLength),
            data: chunk.byteLength ? chunk : undefined,
            usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
          })
        })
    )
  });
}
function submit(device: Device, explorer: GPUResidentExplorer): void {
  const encoder = device.createCommandEncoder();
  explorer.encode(encoder);
  device.submit(encoder.finish());
}
