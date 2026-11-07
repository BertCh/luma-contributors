// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUVector} from '@luma.gl/graph/gpu-data';
import {KeplerDatasetRegistry} from '@luma-contributors/kepler';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {
  AggregationRuntime,
  type NativeAggregationLayer
} from '../../../examples/deck/kepler-compute/src/aggregation-runtime';

function vector<Format extends 'uint32' | 'float32'>(
  device: Device,
  format: Format,
  values: Uint32Array | Float32Array
): GPUVector<Format> {
  return new GPUVector({
    type: 'buffer',
    name: 'aggregate-test-column',
    format,
    length: values.length,
    buffer: device.createBuffer({
      data: values,
      usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
    }),
    ownsBuffer: true
  });
}

it('native metric grids and hexagons reuse resident masks, colors and bounded aggregate picking', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const mask = vector(device, 'uint32', new Uint32Array([1, 1, 1, 1]));
  const weights = vector(device, 'float32', new Float32Array([2, 4, 10, -2]));
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 64 * 1024 * 1024});
  const runtime = new AggregationRuntime({
    memoryRegistry: registry,
    device,
    longitude: [-74, -74, -74, -74],
    latitude: [40.7, 40.7, 40.7, 40.7],
    selectionMask: mask,
    columns: {value: {values: weights}}
  });
  const submitSpy = vi.spyOn(device, 'submit');
  const sourceReadSpy = vi.spyOn(runtime.positions.data[0].buffer, 'readAsync');
  try {
    for (const kind of ['grid', 'hexagon'] as const) {
      const layer: NativeAggregationLayer = {
        id: kind,
        type: kind,
        config: {
          columns: {lat: {value: 'latitude'}, lng: {value: 'longitude'}},
          colorField: {name: 'value'},
          colorScale: 'quantile',
          visConfig: {
            worldUnitSize: 0.5,
            coverage: 0.7,
            colorAggregation: 'average',
            colorRange: {colors: ['#112233', '#abcdef']}
          }
        }
      };
      const resolved = runtime.resolve(layer);
      expect(resolved.reason).toBeUndefined();
      const binding = resolved.binding!;
      expect(binding.coordinateSystem).toBe('cartesian');
      expect(runtime.resolve(layer).binding).toBe(binding);
      expect(registry.memoryUsage.totalBytes).toBeGreaterThan(runtime.uploadBytes);
      const encoder = device.createCommandEncoder();
      runtime.encode(encoder);
      device.submit(encoder.finish());
      const bytes = await binding.counts.data[0].buffer.readAsync();
      const counts = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
      const cellId = counts.findIndex(count => count === 4);
      expect(cellId).toBeGreaterThanOrEqual(0);
      expect(counts.reduce((sum, count) => sum + count, 0)).toBe(4);
      const colors = binding.getFillColor;
      if (!(colors instanceof GPUVector)) throw new Error('Expected resident aggregate colors');
      const colorBytes = await colors.data[0].buffer.readAsync(
        cellId * (kind === 'grid' ? 4 : 6) * 4,
        4
      );
      expect(colorBytes[3]).toBe(255);
      expect([0x11, 0xab]).toContain(colorBytes[0]);
      expect(await runtime.readCell(kind, cellId)).toMatchObject({
        count: 4,
        value: 3.5,
        statistic: 'mean'
      });
      mask.data[0].buffer.write(new Uint32Array([0, 1, 0, 1]));
      const secondEncoder = device.createCommandEncoder();
      runtime.encode(secondEncoder);
      device.submit(secondEncoder.finish());
      expect(await runtime.readCell(kind, cellId)).toMatchObject({count: 2, value: 1});
      expect(runtime.resolve(layer).binding).toBe(binding);
      mask.data[0].buffer.write(new Uint32Array(4));
      const emptyEncoder = device.createCommandEncoder();
      runtime.encode(emptyEncoder);
      device.submit(emptyEncoder.finish());
      expect(await runtime.readCell(kind, cellId)).toBeNull();
      mask.data[0].buffer.write(new Uint32Array([1, 1, 1, 1]));
      runtime.remove(kind);
    }
    expect(sourceReadSpy).not.toHaveBeenCalled();
    expect(runtime.readbackByteLength).toBe(72);
    expect(submitSpy).toHaveBeenCalledTimes(6);
  } finally {
    submitSpy.mockRestore();
    sourceReadSpy.mockRestore();
    const fence = device.createFence();
    await runtime.destroy(fence.signaled);
    fence.destroy();
    expect(registry.memoryUsage.totalBytes).toBe(0);
    expect(mask.data[0].buffer.destroyed).toBe(false);
    mask.destroy();
    weights.destroy();
  }
});

it('native aggregate layers retain independent source masks and rebuild only replaced mask bindings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const defaultMask = vector(device, 'uint32', new Uint32Array([1, 1, 1, 1]));
  const firstMask = vector(device, 'uint32', new Uint32Array([1, 0, 0, 0]));
  const secondMask = vector(device, 'uint32', new Uint32Array([0, 1, 1, 1]));
  const replacementMask = vector(device, 'uint32', new Uint32Array([1, 0, 1, 0]));
  const weights = vector(device, 'float32', new Float32Array([2, 4, 10, -2]));
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 64 * 1024 * 1024});
  const runtime = new AggregationRuntime({
    device,
    memoryRegistry: registry,
    longitude: [-74, -74, -74, -74],
    latitude: [40.7, 40.7, 40.7, 40.7],
    selectionMask: defaultMask,
    columns: {value: {values: weights}}
  });
  const positions = runtime.positions;
  const sourceReadSpy = vi.spyOn(positions.data[0].buffer, 'readAsync');
  const createLayer = (kind: 'grid' | 'hexagon'): NativeAggregationLayer => ({
    id: kind,
    type: kind,
    config: {
      columns: {lat: {value: 'latitude'}, lng: {value: 'longitude'}},
      colorField: {name: 'value'},
      colorScale: 'quantize',
      visConfig: {worldUnitSize: 0.5, colorAggregation: 'average'}
    }
  });
  const firstLayer = createLayer('grid');
  const secondLayer = createLayer('hexagon');
  const submit = () => {
    const encoder = device.createCommandEncoder();
    runtime.encode(encoder);
    device.submit(encoder.finish());
  };
  const readCell = async (layer: NativeAggregationLayer, mask: GPUVector<'uint32'>) => {
    const binding = runtime.resolve(layer, mask).binding!;
    const bytes = await binding.counts.data[0].buffer.readAsync();
    const counts = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    return runtime.readCell(
      layer.id,
      counts.findIndex(count => count > 0)
    );
  };
  try {
    const firstBinding = runtime.resolve(firstLayer, firstMask).binding!;
    const secondBinding = runtime.resolve(secondLayer, secondMask).binding!;
    expect(firstBinding).toBeDefined();
    expect(secondBinding).toBeDefined();
    submit();
    expect(await readCell(firstLayer, firstMask)).toMatchObject({count: 1, value: 2});
    expect(await readCell(secondLayer, secondMask)).toMatchObject({count: 3, value: 4});

    firstMask.data[0].buffer.write(new Uint32Array([0, 1, 0, 0]));
    expect(runtime.resolve(firstLayer, firstMask).binding).toBe(firstBinding);
    submit();
    expect(await readCell(firstLayer, firstMask)).toMatchObject({count: 1, value: 4});
    expect(await readCell(secondLayer, secondMask)).toMatchObject({count: 3, value: 4});

    const replacement = runtime.resolve(firstLayer, replacementMask).binding;
    expect(replacement).toBeDefined();
    expect(replacement).not.toBe(firstBinding);
    expect(runtime.resolve(secondLayer, secondMask).binding).toBe(secondBinding);
    expect(runtime.positions).toBe(positions);
    submit();
    expect(await readCell(firstLayer, replacementMask)).toMatchObject({count: 2, value: 6});
    expect(await readCell(secondLayer, secondMask)).toMatchObject({count: 3, value: 4});
    expect(sourceReadSpy).not.toHaveBeenCalled();
  } finally {
    sourceReadSpy.mockRestore();
    await runtime.destroy();
    expect(registry.memoryUsage.totalBytes).toBe(0);
    for (const borrowed of [defaultMask, firstMask, secondMask, replacementMask, weights]) {
      expect(borrowed.data[0].buffer.destroyed).toBe(false);
      borrowed.destroy();
    }
  }
});

it('published nullable multi-batch points aggregate with independent native scopes, bounded picks and fenced source ownership', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const selection = vector(device, 'uint32', new Uint32Array([1, 1, 0, 1, 1]));
  const weights = vector(device, 'float32', new Float32Array([2, 999, 10, -2, 6]));
  const validity = vector(device, 'uint32', new Uint32Array([1, 1, 1, 1, 0]));
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 64 * 1024 * 1024});
  const runtime = new AggregationRuntime({
    device,
    longitude: [-74, null, -74, -74, -74],
    latitude: [40.7, 40.7, 40.7, 40.7, 40.7],
    sourceChunkLengths: [2, 0, 3],
    selectionMask: selection,
    coordinateColumns: {lat: 'north', lng: 'east'},
    columns: {measurement: {values: weights, validity}},
    memoryRegistry: registry,
    maximumConcurrentReads: 2
  });
  const sourceReads = runtime.positions.data.map(chunk => vi.spyOn(chunk.buffer, 'readAsync'));
  const sourceWrites = runtime.positions.data.map(chunk => vi.spyOn(chunk.buffer, 'write'));
  const makeLayer = (
    id: string,
    type: 'grid' | 'hexagon',
    weighted: boolean
  ): NativeAggregationLayer => ({
    id,
    type,
    config: {
      columns: {lat: {value: 'north'}, lng: {value: 'east'}},
      colorField: weighted ? {name: 'measurement'} : null,
      colorScale: 'quantize',
      visConfig: {worldUnitSize: 0.5, colorAggregation: 'average'}
    }
  });
  const countLayer = makeLayer('published-count', 'grid', false);
  const meanLayer = makeLayer('published-mean', 'hexagon', true);
  let releaseRead: (() => void) | undefined;
  try {
    expect(runtime.positions.data.map(chunk => chunk.length)).toEqual([2, 0, 3]);
    expect(
      runtime.resolve({
        ...countLayer,
        config: {
          ...countLayer.config,
          columns: {lat: {value: 'latitude'}, lng: {value: 'longitude'}}
        }
      }).reason
    ).toMatch(/coordinate columns/);
    const count = runtime.resolve(countLayer, selection, 1).binding!;
    const mean = runtime.resolve(meanLayer, selection, 1).binding!;
    const encoder = device.createCommandEncoder();
    runtime.encode(encoder);
    device.submit(encoder.finish());
    const findCell = async (binding: typeof count) => {
      const bytes = await binding.counts.data[0].buffer.readAsync();
      return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).findIndex(
        value => value > 0
      );
    };
    const countCell = await findCell(count);
    const meanCell = await findCell(mean);
    expect(await runtime.readCell(countLayer.id, countCell)).toMatchObject({count: 3, value: 3});
    expect(await runtime.readCell(meanLayer.id, meanCell)).toMatchObject({count: 2, value: 0});
    const buffer = count.counts.data[0].buffer;
    const readOriginal = buffer.readAsync.bind(buffer);
    const gate = new Promise<void>(resolve => {
      releaseRead = resolve;
    });
    const readSpy = vi.spyOn(buffer, 'readAsync').mockImplementation(async (...args) => {
      const bytes = await readOriginal(...args);
      await gate;
      return bytes;
    });
    const first = runtime.readCell(countLayer.id, countCell);
    const second = runtime.readCell(countLayer.id, countCell);
    const bytesBeforeRejected = runtime.readbackByteLength;
    expect(await runtime.readCell(countLayer.id, countCell)).toBeNull();
    expect(runtime.readbackByteLength).toBe(bytesBeforeRejected);
    expect(runtime.resolve(countLayer, selection, 2).binding).toBe(count);
    expect(await runtime.readCell(countLayer.id, countCell)).toBeNull();
    releaseRead?.();
    expect(await first).toBeNull();
    expect(await second).toBeNull();
    readSpy.mockRestore();
    const nextEncoder = device.createCommandEncoder();
    runtime.encode(nextEncoder);
    device.submit(nextEncoder.finish());
    expect(await runtime.readCell(countLayer.id, countCell)).toMatchObject({count: 3});
    const delayed = new Promise<void>(resolve => {
      releaseRead = resolve;
    });
    const finalReadSpy = vi.spyOn(buffer, 'readAsync').mockImplementation(async (...args) => {
      const bytes = await readOriginal(...args);
      await delayed;
      return bytes;
    });
    const finalRead = runtime.readCell(countLayer.id, countCell);
    runtime.remove(countLayer.id);
    const destruction = runtime.destroy(Promise.resolve());
    expect(runtime.positions.data.every(chunk => !chunk.buffer.destroyed)).toBe(true);
    releaseRead?.();
    expect(await finalRead).toBeNull();
    await destruction;
    finalReadSpy.mockRestore();
    expect(registry.memoryUsage.totalBytes).toBe(0);
    for (const spy of [...sourceReads, ...sourceWrites]) expect(spy).not.toHaveBeenCalled();
    expect([selection, weights, validity].every(column => !column.data[0].buffer.destroyed)).toBe(
      true
    );
  } finally {
    releaseRead?.();
    sourceReads.forEach(spy => spy.mockRestore());
    sourceWrites.forEach(spy => spy.mockRestore());
    await runtime.destroy();
    selection.destroy();
    weights.destroy();
    validity.destroy();
  }
});
