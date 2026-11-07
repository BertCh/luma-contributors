// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession} from '../src/session/compute-session';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';
import {PointAltitudeBindingRuntime} from '../../../examples/deck/kepler-compute/src/point-altitude-binding-runtime';
import {
  prepareTabularPointPositions,
  makeGPUVectorsFromTabularPointPositions
} from '../../../examples/deck/kepler-compute/src/point-position-source';

test('altitude masks preserve source chunks and native null-to-zero without changing XY layers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const batches = [[0, -50], [], [null, 150, NaN]].map(
    heights =>
      new arrow.Table({
        east: arrow.vectorFromArray(
          heights.map(() => -73),
          new arrow.Float32()
        ),
        north: arrow.vectorFromArray(
          heights.map(() => 40),
          new arrow.Float32()
        ),
        height: arrow.vectorFromArray(heights, new arrow.Float32())
      })
  );
  const source = batches[0].concat(...batches.slice(1));
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 16 * 1024 * 1024});
  const dataset = session.importArrowDataset({id: 'altitude-source', table: source});
  const positions = makeGPUVectorsFromTabularPointPositions(
    device,
    prepareTabularPointPositions(dataset.value.prepared, {
      longitude: 'east',
      latitude: 'north',
      altitude: 'height'
    }),
    {memoryRegistry: session.registry}
  );
  const buffers = [2, 0, 3].map(length =>
    device.createBuffer({
      ...(length ? {data: new Uint32Array(length).fill(1)} : {byteLength: 4}),
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    })
  );
  const selection = new GPUVector({
    type: 'data',
    name: 'selection',
    format: 'uint32',
    ownsData: false,
    data: buffers.map(
      (buffer, index) =>
        new GPUData({format: 'uint32', buffer, length: [2, 0, 3][index], ownsBuffer: false})
    )
  });
  const update = vi.fn();
  const runtime = new PointAltitudeBindingRuntime(session, update);
  try {
    const output = runtime.resolve(dataset.id, 'xyz-layer', selection, positions.altitudeValidity);
    expect(output.data.map(chunk => chunk.length)).toEqual([2, 0, 3]);
    expect(runtime.resolve(dataset.id, 'xy-layer', selection, undefined)).toBe(selection);
    const encoder = device.createCommandEncoder();
    runtime.encode(encoder);
    device.submit(encoder.finish());
    expect(await readMask(output)).toEqual([1, 1, 1, 1, 0]);
    buffers[0].write(new Uint32Array([0, 1]));
    buffers[2].write(new Uint32Array([1, 0, 1]));
    const createBuffer = vi.spyOn(device, 'createBuffer');
    expect(runtime.resolve(dataset.id, 'xyz-layer', selection, positions.altitudeValidity)).toBe(
      output
    );
    const next = device.createCommandEncoder();
    runtime.encode(next);
    expect(createBuffer).not.toHaveBeenCalled();
    createBuffer.mockRestore();
    device.submit(next.finish());
    expect(await readMask(output)).toEqual([0, 1, 1, 0, 0]);
    expect(await readMask(selection)).toEqual([0, 1, 1, 0, 1]);
    expect(update).toHaveBeenCalledTimes(1);
    const owned = output.data.map(chunk => chunk.buffer);
    let release!: () => void;
    const pending = runtime.destroy(
      new Promise<void>(resolve => {
        release = resolve;
      })
    );
    await Promise.resolve();
    expect(owned.every(buffer => !buffer.destroyed)).toBe(true);
    release();
    await pending;
    expect(owned.every(buffer => buffer.destroyed)).toBe(true);
  } finally {
    await runtime.destroy(Promise.resolve());
    await positions.destroy(Promise.resolve());
    selection.destroy();
    for (const buffer of buffers) buffer.destroy();
    await session.destroy(Promise.resolve());
  }
});

async function readMask(vector: GPUVector<'uint32'>): Promise<number[]> {
  const values: number[] = [];
  for (const chunk of vector.data)
    if (chunk.length) {
      const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 4);
      values.push(...new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length));
    }
  return values;
}
