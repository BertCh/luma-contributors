// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';
import {KeplerDatasetRegistry} from '../src/runtime/dataset-registry';
import {
  prepareNativeIconGeometry,
  prepareNativeIconSource
} from '../../../examples/deck/kepler-compute/src/icon-source';
import {
  IconRuntime,
  makeGPUVectorsFromNativeIcons
} from '../../../examples/deck/kepler-compute/src/icon-runtime';

test('native icon groups gather exact source-row styles across original chunks and retire borrowed sources only after fences', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const catalog = prepareNativeIconGeometry({
    alpha: [0, 1, 0.5, -1, -1, 0, 1, -1, 0],
    beta: [-1, 1, 0, 1, 1, 0, 0, -1, 0]
  });
  const prepared = prepareNativeIconSource(
    [
      {id: 'first', icons: ['alpha', null]},
      {id: 'empty', icons: []},
      {id: 'last', icons: ['beta', 'alpha', 'beta']}
    ],
    catalog
  );
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1 << 20});
  const icons = makeGPUVectorsFromNativeIcons(device, prepared, {
    workingMemoryBudgetBytes: 1 << 16,
    memoryRegistry: registry
  });
  const positions = makeVector(
    device,
    'float32x3',
    [
      [0, 1, 10, 2, 3, 20, 4, 5, 30],
      [6, 7, 40, 8, 9, 50]
    ],
    3
  );
  const mask = makeVector(device, 'uint32', [[1], [1, 0, 1, 1]], 1);
  const colors = makeVector(
    device,
    'unorm8x4',
    [
      [0xff000001, 0xff000002, 0xff000003],
      [0xff000004, 0xff000005]
    ],
    1
  );
  const radii = makeVector(device, 'float32', [[1, 2], [], [3, 4, 5]], 1);
  const sourceBuffers = [
    ...positions.data,
    ...colors.data,
    ...radii.data,
    ...icons.groups.flatMap(group => [group.iconGeometry, ...group.sourceRowIds.data])
  ].map(chunk => chunk.buffer);
  const reads = sourceBuffers.map(buffer => vi.spyOn(buffer, 'readAsync'));
  const writes = sourceBuffers.map(buffer => vi.spyOn(buffer, 'write'));
  const runtime = new IconRuntime({
    device,
    icons,
    columns: {lat: 'lat', lng: 'lng', icon: 'marker'},
    positions,
    selectionMask: mask,
    colors,
    radii,
    workingMemoryBudgetBytes: 1 << 16,
    memoryRegistry: registry
  });
  const groups = runtime.binding.groups;
  const outputBuffers = [...runtime.resources];
  try {
    const encoder = device.createCommandEncoder();
    runtime.encode(encoder);
    device.submit(encoder.finish());
    expect(groups.map(group => group.positions.data.map(chunk => chunk.length))).toEqual([
      [1, 0, 1],
      [0, 0, 2]
    ]);
    expect(await readValues(groups[0].positions, 3, true)).toEqual([0, 1, 10, 6, 7, 40]);
    expect(await readValues(groups[1].positions, 3, true)).toEqual([4, 5, 30, 8, 9, 50]);
    expect(await readValues(groups[1].selectionMask, 1, false)).toEqual([0, 1]);
    expect(await readValues(groups[0].colors!, 1, false)).toEqual([0xff000001, 0xff000004]);
    expect(await readValues(groups[1].radii!, 1, true)).toEqual([3, 5]);
    expect(groups[0].resolveSourceRow(1)).toBe(3);
    expect(groups[1].resolveSourceRow(0)).toBe(2);
    const allocations = vi.spyOn(device, 'createBuffer');
    const replay = device.createCommandEncoder();
    runtime.encode(replay);
    device.submit(replay.finish());
    expect(allocations).not.toHaveBeenCalled();
    allocations.mockRestore();
    await readValues(groups[0].selectionMask, 1, false);
    expect(runtime.resources).toEqual(outputBuffers);
    for (const spy of [...reads, ...writes]) expect(spy).not.toHaveBeenCalled();
    expect(runtime.readbackByteLength).toBe(0);
    let complete!: () => void;
    const completion = new Promise<void>(resolve => {
      complete = resolve;
    });
    const pending = runtime.destroy(completion);
    expect(outputBuffers.every(buffer => !buffer.destroyed)).toBe(true);
    expect(runtime.accountedResidentByteLength).toBeGreaterThan(0);
    complete();
    await pending;
    expect(outputBuffers.every(buffer => buffer.destroyed)).toBe(true);
    expect(runtime.accountedResidentByteLength).toBe(0);
    expect(sourceBuffers.every(buffer => !buffer.destroyed)).toBe(true);
    await icons.destroy(Promise.resolve());
    expect(positions.data.every(chunk => !chunk.buffer.destroyed)).toBe(true);
  } finally {
    [...reads, ...writes].forEach(spy => spy.mockRestore());
    await runtime.destroy(Promise.resolve());
    await icons.destroy(Promise.resolve());
    positions.destroy();
    mask.destroy();
    colors.destroy();
    radii.destroy();
  }
});

function makeVector<Format extends 'float32x3' | 'float32' | 'uint32' | 'unorm8x4'>(
  device: Device,
  format: Format,
  chunks: number[][],
  components: number
): GPUVector<Format> {
  return new GPUVector({
    type: 'data',
    name: 'source',
    format,
    data: chunks.map(
      values =>
        new GPUData({
          format,
          length: values.length / components,
          ownsBuffer: true,
          buffer: device.createBuffer({
            ...(values.length
              ? {
                  data: format.startsWith('float')
                    ? new Float32Array(values)
                    : new Uint32Array(values)
                }
              : {byteLength: Math.max(4, components * 4)}),
            usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
          })
        })
    )
  });
}
async function readValues(
  vector: GPUVector,
  components: number,
  floatingPoint: boolean
): Promise<number[]> {
  return (
    await Promise.all(
      vector.data.map(async chunk => {
        if (!chunk.length) return [];
        const bytes = await chunk.buffer.readAsync(0, chunk.length * components * 4);
        return Array.from(
          floatingPoint
            ? new Float32Array(bytes.buffer, bytes.byteOffset, chunk.length * components)
            : new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length * components)
        );
      })
    )
  ).flat();
}
