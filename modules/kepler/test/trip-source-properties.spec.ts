// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {GPUVector} from '@luma-contributors/graph/gpu-data';
import {makeGeoJSONTripGPUVectors, prepareGeoJSONTrips} from '@luma.gl/engine/arrow';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';
import {TripRuntime} from '../../../examples/deck/kepler-compute/src/trip-runtime';
import {GeoJSONPropertyRuntime} from '../../../examples/deck/kepler-compute/src/geojson-property-runtime';

async function readWords(vector: GPUVector): Promise<number[]> {
  const result: number[] = [];
  for (const chunk of vector.data) {
    if (!chunk.length) continue;
    const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 4);
    result.push(...new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length));
  }
  return result;
}

it('real trip properties resolve mutable defaults on GPU before primitive gathering without reuploading source geometry or time', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const feature = (times: number[], properties: Record<string, unknown>) => ({
    type: 'Feature',
    properties,
    geometry: {type: 'LineString', coordinates: times.map((time, index) => [index, 2, 3, time])}
  });
  const source = prepareGeoJSONTrips(
    [
      {
        id: 'first',
        rows: [
          null,
          feature([0, 1], {lineColor: [10, 20, 30, 0], lineWidth: -2}),
          feature([2, 3], {lineColor: null, lineWidth: 0})
        ]
      },
      {id: 'empty', rows: []},
      {id: 'last', rows: [feature([5, 4], {lineWidth: 99}), feature([6, 7], {})]}
    ],
    {timestampEncoding: 'unix-millisecond'}
  );
  const trips = makeGeoJSONTripGPUVectors(device, source);
  const colors = new GeoJSONPropertyRuntime(device, trips.properties!.lineColor!, [90, 80, 70]);
  const widths = new GeoJSONPropertyRuntime(device, trips.properties!.lineWidth!, 1);
  const selectionMask = new GPUVector({
    type: 'buffer',
    name: 'trip-selection-mask',
    format: 'uint32',
    length: 5,
    buffer: device.createBuffer({
      data: new Uint32Array([1, 1, 0, 1, 1]),
      usage: Buffer.STORAGE | Buffer.COPY_DST
    }),
    ownsBuffer: true
  });
  const runtime = new TripRuntime({
    device,
    column: 'route',
    trips,
    selectionMask,
    colors: colors.output,
    widths: widths.output,
    workingMemoryBudgetBytes: 1024 * 1024
  });
  const sourceBuffers = [
    ...trips.getPath.data,
    ...trips.getTimestamps.data,
    ...trips.sourceRowIds.data,
    ...Object.values(trips.properties!).flatMap(channel => [
      ...channel.values.data,
      ...channel.validity.data
    ])
  ].map(chunk => chunk.buffer);
  const reads = sourceBuffers.map(buffer => vi.spyOn(buffer, 'readAsync'));
  const writes = sourceBuffers.map(buffer => vi.spyOn(buffer, 'write'));
  try {
    runtime.prepare();
    const binding = runtime.resolve('route')!;
    const resources = [...runtime.resources];
    const encode = () => {
      const encoder = device.createCommandEncoder();
      colors.encode(encoder);
      widths.encode(encoder);
      runtime.encode(encoder);
      device.submit(encoder.finish());
    };
    encode();
    expect(await readWords(binding.selectionMask)).toEqual([1, 0, 1]);
    expect(await readWords(binding.getColor!)).toEqual([0x001e140a, 0xff46505a, 0xff46505a]);
    expect(await readWords(binding.getWidth!)).toEqual([
      ...new Uint32Array(new Float32Array([-2, 1, 1]).buffer)
    ]);
    expect(binding.resolveSourceRow(0)).toBe(1);
    expect(binding.resolveSourceRow(2)).toBe(4);
    expect(trips.properties!.lineWidth!.values.data.map(chunk => chunk.length)).toEqual([3, 0, 2]);
    expect(binding.getPath.data.map(chunk => chunk.length)).toEqual([2, 0, 1]);
    colors.setFallback([1, 2, 3]);
    widths.setFallback(5);
    selectionMask.data[0].buffer.write(new Uint32Array([1, 0, 1, 1, 1]));
    encode();
    expect(await readWords(binding.selectionMask)).toEqual([0, 1, 1]);
    expect(await readWords(binding.getColor!)).toEqual([0x001e140a, 0xff030201, 0xff030201]);
    expect(await readWords(binding.getWidth!)).toEqual([
      ...new Uint32Array(new Float32Array([-2, 5, 5]).buffer)
    ]);
    expect(runtime.resolve('route')).toBe(binding);
    expect(runtime.resources).toEqual(resources);
    for (const spy of [...reads, ...writes]) expect(spy).not.toHaveBeenCalled();
    expect(source.batches.map(batch => [...batch.epochMilliseconds])).toEqual([
      [0, 1, 2, 3],
      [],
      [6, 7]
    ]);
  } finally {
    [...reads, ...writes].forEach(spy => spy.mockRestore());
    const fence = device.createFence();
    try {
      await runtime.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
    colors.destroy();
    widths.destroy();
    expect(sourceBuffers.every(buffer => !buffer.destroyed)).toBe(true);
    trips.destroy();
    selectionMask.destroy();
  }
});
