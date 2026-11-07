// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {makeGeometryGPUVectors, prepareGeometrySource} from '@luma.gl/engine/arrow';
import {GeometryRuntime} from '../../../examples/deck/kepler-compute/src/geometry-runtime';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {GeoJSONPropertyRuntime} from '../../../examples/deck/kepler-compute/src/geojson-property-runtime';

test('embedded source properties preserve chunks and native truthiness while mutable defaults replay without source uploads', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const values = makeVector(device, 'float32', [[0, -2], [], [7, 0]]);
  const validity = makeVector(device, 'uint32', [[0, 1], [], [1, 0]]);
  const colorValues = makeVector(device, 'unorm8x4', [[0x00030201, 0], [], [0xff000000, 0]]);
  const properties = new GeoJSONPropertyRuntime(device, {values, validity}, 1);
  const colors = new GeoJSONPropertyRuntime(device, {values: colorValues, validity}, [10, 20, 30]);
  const reads = [...values.data, ...colorValues.data, ...validity.data].map(chunk =>
    vi.spyOn(chunk.buffer, 'readAsync')
  );
  const writes = [...values.data, ...colorValues.data, ...validity.data].map(chunk =>
    vi.spyOn(chunk.buffer, 'write')
  );
  const snapshots = [0, 1].map(() =>
    device.createBuffer({byteLength: 4, usage: Buffer.COPY_DST | Buffer.COPY_SRC})
  );
  const outputChunks = [...properties.output.data];
  try {
    const outputBuffer = properties.output.data[0].buffer;
    if (!(outputBuffer instanceof Buffer)) throw new Error('Expected static output storage');
    const encoder = device.createCommandEncoder();
    properties.encode(encoder);
    colors.encode(encoder);
    encoder.copyBufferToBuffer({
      sourceBuffer: outputBuffer,
      destinationBuffer: snapshots[0],
      size: 4
    });
    properties.setFallback(5);
    properties.encode(encoder);
    encoder.copyBufferToBuffer({
      sourceBuffer: outputBuffer,
      destinationBuffer: snapshots[1],
      size: 4
    });
    device.submit(encoder.finish());
    const snapshotBytes = await Promise.all(snapshots.map(buffer => buffer.readAsync(0, 4)));
    expect(
      snapshotBytes.map(bytes => new Float32Array(bytes.buffer, bytes.byteOffset, 1)[0])
    ).toEqual([1, 5]);
    expect(await readVector(properties.output, true)).toEqual([5, -2, 7, 5]);
    expect(await readVector(colors.output, false)).toEqual([0xff1e140a, 0, 0xff000000, 0xff1e140a]);
    expect(properties.output.data.map(chunk => chunk.length)).toEqual([2, 0, 2]);
    expect(properties.output.data).toEqual(outputChunks);
    const upload = vi.spyOn(device, 'writeBufferViaCommandEncoder');
    const unchanged = device.createCommandEncoder();
    properties.setFallback(5);
    properties.encode(unchanged);
    expect(upload).not.toHaveBeenCalled();
    properties.discardEncoding();
    properties.encode(unchanged);
    expect(upload).toHaveBeenCalledTimes(1);
    device.submit(unchanged.finish());
    await readVector(properties.output, true);
    upload.mockRestore();
    for (const spy of [...reads, ...writes]) expect(spy).not.toHaveBeenCalled();
  } finally {
    properties.destroy();
    colors.destroy();
    for (const chunk of [...values.data, ...validity.data, ...colorValues.data])
      expect(chunk.buffer.destroyed).toBe(false);
    [...reads, ...writes].forEach(spy => spy.mockRestore());
    values.destroy();
    validity.destroy();
    colorValues.destroy();
    snapshots.forEach(buffer => buffer.destroy());
  }
});

function makeVector<Format extends 'float32' | 'uint32' | 'unorm8x4'>(
  device: Device,
  format: Format,
  chunks: number[][]
): GPUVector<Format> {
  return new GPUVector({
    type: 'data',
    name: 'property',
    format,
    data: chunks.map(
      values =>
        new GPUData({
          format,
          length: values.length,
          byteOffset: 4,
          ownsBuffer: true,
          buffer: device.createBuffer({
            data:
              format === 'float32'
                ? new Float32Array([999, ...values])
                : new Uint32Array([999, ...values]),
            usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
          })
        })
    )
  });
}
async function readVector(
  vector: GPUVector<'float32'> | GPUVector<'unorm8x4'>,
  floatingPoint: boolean
): Promise<number[]> {
  return (
    await Promise.all(
      vector.data.map(async chunk => {
        if (!chunk.length) return [];
        const bytes = await chunk.buffer.readAsync(0, chunk.length * 4);
        return Array.from(
          floatingPoint
            ? new Float32Array(bytes.buffer, bytes.byteOffset, chunk.length)
            : new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length)
        );
      })
    )
  ).flat();
}

test('prepared polygon vertical properties preserve zero offsets, native height fallback, XYZ and warm chunked gathers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygon = (elevation: unknown, elevationOffset: unknown) => ({
    type: 'Feature',
    properties: {elevation, elevationOffset},
    geometry: {
      type: 'Polygon',
      coordinates: [
        [
          [0, 0, 7],
          [1, 0, 8],
          [0, 1, 9],
          [0, 0, 7]
        ]
      ]
    }
  });
  const source = prepareGeometrySource([
    {id: 'first', rows: [polygon(0, 0), polygon(-3, -2)]},
    {id: 'empty', rows: []},
    {id: 'last', rows: [polygon(null, null), polygon(9, 4)]}
  ]);
  const geometry = makeGeometryGPUVectors(device, source);
  const height = new GeoJSONPropertyRuntime(device, geometry.properties!.elevation!, 500);
  const offset = new GeoJSONPropertyRuntime(device, geometry.properties!.elevationOffset!, 50);
  // Different mask chunking must not change polygon source-row alignment.
  const mask = makeVector(device, 'uint32', [[1], [0, 1, 1]]);
  const runtime = new GeometryRuntime({
    device,
    column: 'geometry',
    geometry,
    selectionMask: mask,
    elevations: height.values,
    elevationOffsets: offset.values,
    stylesResolved: true
  });
  const sourceReads = geometry.buffers.map(buffer => vi.spyOn(buffer, 'readAsync'));
  const sourceWrites = geometry.buffers.map(buffer => vi.spyOn(buffer, 'write'));
  try {
    const polygonBinding = runtime.binding.polygons!;
    expect(polygonBinding.positions).toBe(geometry.polygons.positions);
    expect(polygonBinding.extrusionRings!.positions).toBe(geometry.polygonOutlines.getPath);
    const gatheredHeight = polygonBinding.getElevation;
    const gatheredOffset = polygonBinding.getElevationOffset;
    if (!(gatheredHeight instanceof GPUVector) || !(gatheredOffset instanceof GPUVector))
      throw new Error('Missing gathered vertical outputs');
    expect(gatheredHeight.data.map(chunk => chunk.length)).toEqual([2, 0, 2]);
    const encoder = device.createCommandEncoder();
    height.encode(encoder);
    offset.encode(encoder);
    runtime.encode(encoder);
    device.submit(encoder.finish());
    expect(await readVector(gatheredHeight, true)).toEqual([500, -3, 500, 9]);
    expect(await readVector(gatheredOffset, true)).toEqual([0, -2, 50, 4]);
    const resources = [...runtime.resources];
    offset.setFallback(-20);
    const replay = device.createCommandEncoder();
    height.encode(replay);
    offset.encode(replay);
    runtime.encode(replay);
    device.submit(replay.finish());
    expect(await readVector(gatheredOffset, true)).toEqual([0, -2, -20, 4]);
    expect(polygonBinding.getElevationOffset).toBe(gatheredOffset);
    expect(runtime.resources).toEqual(resources);
    expect(runtime.readbackByteLength).toBe(0);
    for (const spy of [...sourceReads, ...sourceWrites]) expect(spy).not.toHaveBeenCalled();
    runtime.destroy();
    height.destroy();
    offset.destroy();
    expect(geometry.buffers.every(buffer => !buffer.destroyed)).toBe(true);
    expect(height.accountedResidentByteLength).toBe(0);
    expect(offset.accountedResidentByteLength).toBe(0);
  } finally {
    sourceReads.forEach(spy => spy.mockRestore());
    sourceWrites.forEach(spy => spy.mockRestore());
    runtime.destroy();
    height.destroy();
    offset.destroy();
    mask.destroy();
    geometry.destroy();
  }
});
