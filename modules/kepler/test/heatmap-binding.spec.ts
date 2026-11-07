// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession} from '../src/session/compute-session';
import {HeatmapBindingRuntime} from '../../../examples/deck/kepler-compute/src/heatmap-binding-runtime';

function makeMask(device: Device, chunks: readonly (readonly number[])[]): GPUVector<'uint32'> {
  return new GPUVector({
    name: 'heatmap-domain',
    type: 'data',
    format: 'uint32',
    ownsData: true,
    data: chunks.map(
      values =>
        new GPUData({
          format: 'uint32',
          length: values.length,
          ownsBuffer: true,
          buffer: device.createBuffer({
            data: new Uint32Array(values.length ? values : [0]),
            usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
          })
        })
    )
  });
}
async function readValues(vector: GPUVector<'float32'>): Promise<number[]> {
  const values: number[] = [];
  for (const chunk of vector.data) {
    if (!chunk.length) continue;
    const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 4);
    values.push(...new Float32Array(bytes.buffer, bytes.byteOffset, chunk.length));
  }
  return values;
}

test('heatmap weights match native normalized unclamped DOMAIN scales without source reads or uploads', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU required');
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const tables = [[2, 4], [], [null, 6, 8]].map(
    values => new arrow.Table({value: arrow.vectorFromArray(values, new arrow.Float32())})
  );
  const dataset = session.importArrowDataset({
    id: 'points',
    table: new arrow.Table(tables.flatMap(table => table.batches))
  });
  const source = dataset.value.columns['value'];
  const sourceReads = source.data.map(chunk => vi.spyOn(chunk.buffer, 'readAsync'));
  const sourceWrites = source.data.map(chunk => vi.spyOn(chunk.buffer, 'write'));
  const domainMask = makeMask(device, [[1, 1], [], [1, 1, 1]]);
  const displayMask = makeMask(device, [[1, 0], [], [1, 0, 1]]);
  const positions = new GPUVector({
    name: 'unused-render-positions',
    type: 'data',
    format: 'float32x2',
    data: []
  });
  const runtime = new HeatmapBindingRuntime(session, vi.fn());
  const options = {
    layerId: 'density',
    dataset,
    positions,
    selectionMask: displayMask,
    columns: {lat: 'north', lng: 'east'},
    revision: 1,
    weightColumn: 'value',
    weightScale: 'linear',
    domainMask,
    domainRevision: 1
  };
  const submit = async () => {
    const encoder = device.createCommandEncoder();
    runtime.encode(encoder);
    device.submit(encoder.finish());
    await runtime.readSummary();
  };
  try {
    expect(runtime.resolve(options)).toEqual({});
    await submit();
    const first = runtime.resolve(options).binding!;
    expect(first.weights).toBeDefined();
    expect(first.weights!.data.map(chunk => chunk.length)).toEqual([2, 0, 3]);
    const normalized = await readValues(first.weights!);
    [0, 1 / 3, 0, 2 / 3, 1].forEach((value, index) =>
      expect(normalized[index]).toBeCloseTo(value, 6)
    );
    expect(runtime.resolve(options).binding?.revision).toBe(first.revision);
    domainMask.data[0].buffer.write(new Uint32Array([0, 1]));
    const selected = {...options, domainRevision: 2};
    expect(runtime.resolve(selected)).toEqual({});
    await submit();
    const second = runtime.resolve(selected).binding!;
    expect(second.weights).toBe(first.weights);
    expect(second.revision).toBeGreaterThan(first.revision);
    const selectedWeights = await readValues(second.weights!);
    [-0.5, 0, 0, 0.5, 1].forEach((value, index) =>
      expect(selectedWeights[index]).toBeCloseTo(value, 6)
    );
    for (const scale of ['sqrt', 'log']) {
      const scaled = {...selected, weightScale: scale};
      expect(runtime.resolve(scaled)).toEqual({});
      await submit();
      const binding = runtime.resolve(scaled).binding!;
      const values = await readValues(binding.weights!);
      const transform = scale === 'sqrt' ? Math.sqrt : Math.log;
      [2, 4, null, 6, 8].forEach((value, index) =>
        expect(values[index]).toBeCloseTo(
          value === null ? 0 : (transform(value) - transform(4)) / (transform(8) - transform(4)),
          5
        )
      );
    }
    for (const read of sourceReads) expect(read).not.toHaveBeenCalled();
    for (const write of sourceWrites) expect(write).not.toHaveBeenCalled();
  } finally {
    const fence = device.createFence();
    await runtime.destroy(fence.signaled);
    fence.destroy();
    domainMask.destroy();
    displayMask.destroy();
    for (const spy of [...sourceReads, ...sourceWrites]) spy.mockRestore();
    await session.destroy(Promise.resolve());
  }
});
