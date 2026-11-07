// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {WebMercatorViewport} from '@deck.gl/core';
import {modifyPolygonWindingDirection, WINDING} from '@math.gl/polygon';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {cellToBoundary, cellToLatLng} from 'h3-js';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';
import {
  H3MetadataRuntime,
  H3Runtime,
  makeH3GPUVectors
} from '../../../examples/deck/kepler-compute/src/h3-runtime';
import {
  prepareH3Source,
  prepareH3ViewportMesh
} from '../../../examples/deck/kepler-compute/src/h3-source';

const CELL = '872830828ffffff';
const PENTAGON = '8009fffffffffff';
const FACE = '81017ffffffffff';
const DATELINE = '877eb5722ffffff';
const viewport = new WebMercatorViewport({
  longitude: -122.42,
  latitude: 37.77,
  zoom: 10,
  width: 800,
  height: 600
});

function vector<Format extends 'uint32' | 'float32' | 'unorm8x4'>(
  device: Device,
  format: Format,
  chunks: number[][]
): GPUVector<Format> {
  return new GPUVector({
    type: 'data',
    name: 'source',
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
            usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
          })
        })
    )
  });
}
async function readVector(
  values: GPUVector,
  components: number,
  floatingPoint = false
): Promise<number[]> {
  return (
    await Promise.all(
      values.data.map(async chunk => {
        const length = chunk.format?.startsWith('vertex-list<') ? chunk.valueLength : chunk.length;
        if (!length) return [];
        const bytes = await chunk.buffer.readAsync(chunk.byteOffset, length * components * 4);
        return Array.from(
          floatingPoint
            ? new Float32Array(bytes.buffer, bytes.byteOffset, length * components)
            : new Uint32Array(bytes.buffer, bytes.byteOffset, length * components)
        );
      })
    )
  ).flat();
}
function submit(
  device: Device,
  runtime: {encode(encoder: ReturnType<Device['createCommandEncoder']>): void}
): void {
  const encoder = device.createCommandEncoder();
  runtime.encode(encoder);
  device.submit(encoder.finish());
}

// h3-js is the independent geometry oracle; normalization follows the pinned native coverage contract.
function expectedBoundary(cell: string, coverage: number): number[][] {
  const boundary = cellToBoundary(cell, true);
  const [latitude, longitude] = cellToLatLng(cell);
  const reference = coverage === 1 ? boundary[0][0] : longitude;
  return boundary.map(([vertexLongitude, vertexLatitude]) => {
    if (vertexLongitude - reference > 180) vertexLongitude -= 360;
    else if (vertexLongitude - reference < -180) vertexLongitude += 360;
    return coverage === 1
      ? [vertexLongitude, vertexLatitude]
      : [
          longitude + (vertexLongitude - longitude) * coverage,
          latitude + (vertexLatitude - latitude) * coverage
        ];
  });
}
function expectBoundary(actual: number[], expected: number[][]): void {
  expect(actual.length / 3).toBe(expected.length);
  for (let index = 0; index < actual.length; index += 3) {
    const distance = Math.min(
      ...expected.map(position =>
        Math.hypot(position[0] - actual[index], position[1] - actual[index + 1])
      )
    );
    expect(distance).toBeLessThan(0.00008);
    expect(actual[index + 2]).toBe(0);
  }
}

test('H3 topology metadata uses its own mask, excludes invalid source rows and reads only one bounded summary per revision', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const source = makeH3GPUVectors(
    device,
    prepareH3Source([
      {id: 'first', values: [CELL, null]},
      {id: 'empty', values: []},
      {id: 'last', values: [PENTAGON, FACE, CELL]}
    ]),
    {workingMemoryBudgetBytes: 1 << 20}
  );
  const topology = vector(device, 'uint32', [
    [1, 1, 0],
    [0, 1]
  ]);
  const allocations = vi.spyOn(device, 'createBuffer');
  const metadata = new H3MetadataRuntime({
    device,
    source,
    mask: topology,
    revision: 1,
    workingMemoryBudgetBytes: 1 << 16
  });
  const summary = allocations.mock.results[0].value as Buffer;
  allocations.mockRestore();
  const spies = [source.validity, source.resolutions, source.pentagons, topology].flatMap(values =>
    values.data.map(chunk => vi.spyOn(chunk.buffer, 'readAsync'))
  );
  try {
    expect(await metadata.readMetadata()).toBeNull();
    submit(device, metadata);
    const first = metadata.readMetadata();
    expect(metadata.readMetadata()).toBe(first);
    expect(await first).toEqual({
      revision: 1,
      count: 2,
      minimumResolution: 7,
      maximumResolution: 7,
      hasPentagon: false
    });
    await metadata.readMetadata();
    expect(metadata.readbackByteLength).toBe(16);
    const controlWrites = vi.spyOn(device, 'writeBufferViaCommandEncoder');
    submit(device, metadata);
    expect(controlWrites).not.toHaveBeenCalled();
    controlWrites.mockRestore();
    topology.data[0].buffer.write(new Uint32Array([0, 1, 1]), 4);
    topology.data[1].buffer.write(new Uint32Array([1, 1]), 4);
    metadata.setRevision(2);
    expect(metadata.snapshot).toBeUndefined();
    submit(device, metadata);
    expect(await metadata.readMetadata()).toEqual({
      revision: 2,
      count: 3,
      minimumResolution: 0,
      maximumResolution: 7,
      hasPentagon: true
    });
    topology.data[0].buffer.write(new Uint32Array([0, 0, 0]), 4);
    topology.data[1].buffer.write(new Uint32Array([0, 0]), 4);
    metadata.setRevision(3);
    submit(device, metadata);
    expect(await metadata.readMetadata()).toEqual({
      revision: 3,
      count: 0,
      minimumResolution: 0xffffffff,
      maximumResolution: 0,
      hasPentagon: false
    });
    expect(metadata.readbackByteLength).toBe(48);
    metadata.setRevision(4);
    submit(device, metadata);
    const deferred = createDeferred<Uint8Array>();
    const readSummary = vi
      .spyOn(summary, 'readAsync')
      .mockImplementationOnce(() => deferred.promise);
    const stale = metadata.readMetadata();
    metadata.setRevision(5);
    metadata.setRevision(4);
    submit(device, metadata);
    deferred.resolve(new Uint8Array(new Uint32Array([0, 7, 1, 99]).buffer));
    expect(await stale).toBeNull();
    expect(metadata.snapshot).toBeUndefined();
    expect(await metadata.readMetadata()).toMatchObject({
      revision: 4,
      count: 0,
      hasPentagon: false
    });
    expect(metadata.readbackByteLength).toBe(80);
    metadata.setRevision(6);
    submit(device, metadata);
    const discardedRead = createDeferred<Uint8Array>();
    readSummary.mockImplementationOnce(() => discardedRead.promise);
    const discarded = metadata.readMetadata();
    metadata.discardEncoding();
    submit(device, metadata);
    discardedRead.resolve(new Uint8Array(new Uint32Array([0, 7, 1, 99]).buffer));
    expect(await discarded).toBeNull();
    expect(await metadata.readMetadata()).toMatchObject({
      revision: 6,
      count: 0,
      hasPentagon: false
    });
    expect(metadata.readbackByteLength).toBe(112);
    readSummary.mockRestore();
    for (const spy of spies) expect(spy).not.toHaveBeenCalled();
  } finally {
    spies.forEach(spy => spy.mockRestore());
    await metadata.destroy(Promise.resolve());
    await source.destroy(Promise.resolve());
    topology.destroy();
  }
});

test('H3 canonical GPU coverage preserves variable native boundaries and snapshots zero/half/full coverage without source mutation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const source = makeH3GPUVectors(
    device,
    prepareH3Source([
      {id: 'cells', values: [DATELINE, null, PENTAGON, FACE]},
      {id: 'empty', values: []}
    ]),
    {workingMemoryBudgetBytes: 1 << 20}
  );
  const mask = vector(device, 'uint32', [
    [1, 1],
    [1, 0]
  ]);
  const runtime = new H3Runtime({
    device,
    column: 'h3',
    source,
    branch: 'polygons',
    selectionMask: mask,
    coverage: 1,
    workingMemoryBudgetBytes: 1 << 20
  });
  const outlines = runtime.binding.polygonOutlines!.getPath;
  const firstChunk = outlines.data[0];
  const snapshots = [0, 1].map(() =>
    device.createBuffer({
      byteLength: firstChunk.valueLength * 12,
      usage: Buffer.COPY_SRC | Buffer.COPY_DST
    })
  );
  const sourceReads = source.geometry.polygons.positions.data.map(chunk =>
    vi.spyOn(chunk.buffer, 'readAsync')
  );
  try {
    const encoder = device.createCommandEncoder();
    runtime.setCoverage(0);
    runtime.encode(encoder);
    encoder.copyBufferToBuffer({
      sourceBuffer:
        firstChunk.buffer instanceof Buffer ? firstChunk.buffer : firstChunk.buffer.buffer,
      destinationBuffer: snapshots[0],
      size: firstChunk.valueLength * 12
    });
    runtime.setCoverage(1);
    runtime.encode(encoder);
    encoder.copyBufferToBuffer({
      sourceBuffer:
        firstChunk.buffer instanceof Buffer ? firstChunk.buffer : firstChunk.buffer.buffer,
      destinationBuffer: snapshots[1],
      size: firstChunk.valueLength * 12
    });
    device.submit(encoder.finish());
    const cells = [DATELINE, PENTAGON, FACE];
    for (const [snapshotIndex, coverage] of [0, 1].entries()) {
      const bytes = await snapshots[snapshotIndex].readAsync();
      const actual = Array.from(
        new Float32Array(bytes.buffer, bytes.byteOffset, firstChunk.valueLength * 3)
      );
      cells.forEach((cell, index) =>
        expectBoundary(
          actual.slice(
            firstChunk.valueOffsets![index] * 3,
            firstChunk.valueOffsets![index + 1] * 3
          ),
          expectedBoundary(cell, coverage)
        )
      );
    }
    runtime.setCoverage(0.5);
    submit(device, runtime);
    const half = await readVector(outlines, 3, true);
    cells.forEach((cell, index) =>
      expectBoundary(
        half.slice(firstChunk.valueOffsets![index] * 3, firstChunk.valueOffsets![index + 1] * 3),
        expectedBoundary(cell, 0.5)
      )
    );
    expect(await readVector(runtime.binding.polygonOutlines!.selectionMask!, 1)).toEqual([1, 1, 0]);
    expect(
      [0, 1, 2].map(index => runtime.binding.polygonOutlines!.resolveSourceRow!(index))
    ).toEqual([0, 2, 3]);
    for (const spy of sourceReads) expect(spy).not.toHaveBeenCalled();
    expect(runtime.readbackByteLength).toBe(0);
  } finally {
    sourceReads.forEach(spy => spy.mockRestore());
    snapshots.forEach(buffer => buffer.destroy());
    await runtime.destroy(Promise.resolve());
    await source.destroy(Promise.resolve());
    mask.destroy();
  }
});

test('H3 column branch gathers source styles and display masks while viewport mesh updates preserve output topology', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const source = makeH3GPUVectors(
    device,
    prepareH3Source([
      {id: 'first', values: [CELL, null]},
      {id: 'empty', values: []},
      {id: 'last', values: [CELL, CELL]}
    ]),
    {workingMemoryBudgetBytes: 1 << 20}
  );
  const selectionMask = vector(device, 'uint32', [[1], [1, 0, 1]]);
  const colors = vector(device, 'unorm8x4', [[0xff000001, 0xff000002, 0xff000003], [0xff000004]]);
  const strokeColors = vector(device, 'unorm8x4', [
    [0xff000011, 0xff000012],
    [0xff000013, 0xff000014]
  ]);
  const coverages = vector(device, 'float32', [[0.2, 0.4], [], [0.6, 0.8]]);
  const elevations = vector(device, 'float32', [[10, 20, 30, 40]]);
  const mesh = prepareH3ViewportMesh({resolution: 7, viewport})!;
  const runtime = new H3Runtime({
    device,
    column: 'h3',
    source,
    branch: 'columns',
    selectionMask,
    colors,
    strokeColors,
    coverages,
    elevations,
    coverage: 0.5,
    viewportMesh: mesh,
    workingMemoryBudgetBytes: 1 << 20
  });
  const binding = runtime.binding.columns!;
  const sourceBuffers = [
    source.centers,
    source.primitiveSourceRows,
    selectionMask,
    colors,
    strokeColors,
    coverages,
    elevations
  ].flatMap(values => values.data.map(chunk => chunk.buffer));
  const sourceReads = sourceBuffers.map(buffer => vi.spyOn(buffer, 'readAsync'));
  const sourceWrites = sourceBuffers.map(buffer => vi.spyOn(buffer, 'write'));
  const initialEdgeDistance = binding.edgeDistance;
  const stableBuffers = [
    binding.positions,
    binding.selectionMask!,
    binding.colors!,
    binding.strokeColors!,
    binding.coverages!,
    binding.elevations!
  ].flatMap(values => values.data.map(chunk => chunk.buffer));
  try {
    submit(device, runtime);
    expect(binding.positions.data.map(chunk => chunk.length)).toEqual([1, 0, 2]);
    expect(await readVector(binding.selectionMask!, 1)).toEqual([1, 0, 1]);
    expect(await readVector(binding.colors!, 1)).toEqual([0xff000001, 0xff000003, 0xff000004]);
    expect(await readVector(binding.strokeColors!, 1)).toEqual([
      0xff000011, 0xff000013, 0xff000014
    ]);
    expect(await readVector(binding.coverages!, 1, true)).toEqual(
      Array.from(new Float32Array([0.2, 0.6, 0.8]))
    );
    expect(await readVector(binding.elevations!, 1, true)).toEqual([10, 30, 40]);
    const expectedCenter = cellToLatLng(CELL).reverse().map(Math.fround);
    expect(await readVector(binding.positions, 2, true)).toEqual([
      ...expectedCenter,
      ...expectedCenter,
      ...expectedCenter
    ]);
    const updated = {...mesh, positions: Float64Array.from(mesh.positions, value => value * 0.75)};
    runtime.setViewportMesh(updated);
    expect(binding.edgeDistance).toBeCloseTo(initialEdgeDistance * 0.75, 5);
    submit(device, runtime);
    const bytes = await binding.vertices.buffer.readAsync();
    expect(
      Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, binding.vertices.length * 2))
    ).toEqual(nativeColumnVertices(updated.positions));
    expect(stableBuffers).toEqual(
      [
        binding.positions,
        binding.selectionMask!,
        binding.colors!,
        binding.strokeColors!,
        binding.coverages!,
        binding.elevations!
      ].flatMap(values => values.data.map(chunk => chunk.buffer))
    );
    expect([0, 1, 2, 3].map(index => binding.resolveSourceRow!(index))).toEqual([0, 2, 3, -1]);
    const controls = vi.spyOn(device, 'writeBufferViaCommandEncoder');
    runtime.setViewportMesh(updated);
    submit(device, runtime);
    expect(controls).not.toHaveBeenCalled();
    runtime.discardEncoding();
    submit(device, runtime);
    expect(controls).toHaveBeenCalledTimes(2);
    controls.mockRestore();
    await readVector(binding.selectionMask!, 1);
    const meshBuffer = binding.vertices.buffer;
    expect(binding.vertices.length).toBe(6);
    for (const [resolution, centerHexagon] of [
      [0, PENTAGON],
      [1, FACE]
    ] as const) {
      const differentTopology = prepareH3ViewportMesh({resolution, viewport, centerHexagon})!;
      runtime.setViewportMesh(differentTopology);
      submit(device, runtime);
      expect(binding.vertices.buffer).toBe(meshBuffer);
      const changedBytes = await meshBuffer.readAsync();
      expect(
        Array.from(new Float32Array(changedBytes.buffer, changedBytes.byteOffset, 12))
      ).toEqual(nativeColumnVertices(differentTopology.positions));
      const radiusMean =
        Array.from({length: 6}, (_, index) =>
          Math.hypot(
            differentTopology.positions[index * 2],
            differentTopology.positions[index * 2 + 1]
          )
        ).reduce((sum, radius) => sum + radius, 0) / 6;
      expect(binding.edgeDistance).toBeCloseTo(Math.cos(Math.PI / 6) * radiusMean, 5);
    }
    for (const spy of [...sourceReads, ...sourceWrites]) expect(spy).not.toHaveBeenCalled();
    expect(runtime.readbackByteLength).toBe(0);
  } finally {
    [...sourceReads, ...sourceWrites].forEach(spy => spy.mockRestore());
    await runtime.destroy(Promise.resolve());
    await source.destroy(Promise.resolve());
    [selectionMask, colors, strokeColors, coverages, elevations].forEach(values =>
      values.destroy()
    );
  }
});

function createDeferred<Value>(): {promise: Promise<Value>; resolve(value: Value): void} {
  let resolve!: (value: Value) => void;
  const promise = new Promise<Value>(complete => {
    resolve = complete;
  });
  return {promise, resolve};
}

function nativeColumnVertices(positions: Float64Array): number[] {
  const normalized = positions.slice();
  modifyPolygonWindingDirection(normalized, WINDING.COUNTER_CLOCKWISE);
  return Array.from(Float32Array.from(normalized.subarray(0, 12)));
}
