// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {expect, test, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import {
  makeCPUVectorFromArrow,
  makeArrowVectorFromCPUVector,
  projectCPUVector,
  tessellateCPUVectorPolygons
} from '@luma-contributors/graph/arrow';
import {
  makeGPUVectorFromCPUVector,
  makeCPUVectorFromGPUVector,
  CPUData,
  CPUVector
} from '@luma-contributors/graph/gpu-data';
import {projectionEngine} from '@math.gl/geospatial/projection';

test('numeric Arrow CPU GPU round trip retains sliced validity and chunk boundaries', async () => {
  const first = arrow.vectorFromArray([1, null, 3, 4], new arrow.Float32()).slice(1, 4);
  const second = arrow.vectorFromArray([5, null], new arrow.Float32());
  const source = first.concat(second);
  const cpu = makeCPUVectorFromArrow(source);
  expect(cpu.data.map(chunk => chunk.length)).toEqual([3, 2]);
  expect(makeArrowVectorFromCPUVector(cpu).toArray()).toEqual(source.toArray());
  const device = new NullDevice({});
  const gpu = makeGPUVectorFromCPUVector(device, cpu);
  try {
    const returned = makeArrowVectorFromCPUVector(await makeCPUVectorFromGPUVector(gpu));
    expect(Array.from(returned)).toEqual(Array.from(source));
    expect(returned.data.map(chunk => chunk.length)).toEqual([3, 2]);
  } finally {
    gpu.destroy();
    device.destroy();
  }
});

test('single-level coordinate lists project and tessellate through CPU vectors', async () => {
  const type = new arrow.List(
    new arrow.Field(
      'coordinates',
      new arrow.FixedSizeList(2, new arrow.Field('value', new arrow.Float32(), false)),
      false
    )
  );
  const source = arrow.vectorFromArray(
    [
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0]
      ],
      null
    ],
    type
  );
  const cpu = makeCPUVectorFromArrow(source);
  const roundtrip = makeArrowVectorFromCPUVector(cpu);
  expect(roundtrip.get(0)?.toArray().length).toBe(5);
  expect(roundtrip.get(1)).toBeNull();
  const projection = await projectionEngine.createProjectionAsync({
    from: 'EPSG:4326',
    to: 'EPSG:3857'
  });
  const projected = await projectCPUVector(cpu, projection);
  expect(makeArrowVectorFromCPUVector(projected).get(1)).toBeNull();
  const mesh = tessellateCPUVectorPolygons(projected);
  expect(mesh.positions).toBeInstanceOf(CPUVector);
  expect(mesh.indices.valueLength).toBe(6);
  expect(mesh.positions.data[0].valueOffsets?.length).toBe(3);
  const device = new NullDevice({});
  const gpu = makeGPUVectorFromCPUVector(device, mesh.indices, {
    usage: Buffer.INDEX
  });
  try {
    expect(gpu.data[0].buffer.usage & Buffer.INDEX).toBeTruthy();
    expect(gpu.data[0].buffer.indexType).toBe('uint32');
    const returned = await makeCPUVectorFromGPUVector(gpu);
    expect(returned.data[0].valueOffsets).toEqual(mesh.indices.data[0].valueOffsets);
    expect(returned.data[0].nullBitmap).toEqual(mesh.indices.data[0].nullBitmap);
  } finally {
    gpu.destroy();
    device.destroy();
  }
});

test('rejects nullable list elements instead of discarding child validity', () => {
  const type = new arrow.List(new arrow.Field('item', new arrow.Float32(), true));
  expect(() => makeCPUVectorFromArrow(arrow.vectorFromArray([[1, null, 3]], type))).toThrow();
});

test('Float16 values round-trip without decoding their stored bits', () => {
  const source = arrow.vectorFromArray([1, 2, null, -3], new arrow.Float16());
  const result = makeArrowVectorFromCPUVector(makeCPUVectorFromArrow(source));
  expect(result.type.toString()).toBe(source.type.toString());
  expect(Array.from(result)).toEqual(Array.from(source));
});

test('projects a point chunk in one batched call', async () => {
  const type = new arrow.FixedSizeList(2, new arrow.Field('value', new arrow.Float32(), false));
  const source = makeCPUVectorFromArrow(
    arrow.vectorFromArray(
      [
        [1, 2],
        [3, 4],
        [5, 6]
      ],
      type
    )
  );
  const projectFlat = vi.fn(
    <T extends number[] | Float32Array | Float64Array>(values: T) => values
  );
  await projectCPUVector(source, {projectFlat});
  expect(projectFlat).toHaveBeenCalledTimes(1);
  expect(projectFlat.mock.calls[0][0].length).toBe(6);
});

test('direct GeoArrow adapters borrow coordinate buffers and preserve topology', async () => {
  const {makeGeoArrowColumnFromCPUVector, makeCPUVectorFromGeoArrowColumn} = await import(
    '@luma-contributors/graph/gpu-project/geoarrow'
  );
  const source = makeCPUVectorFromArrow(
    arrow.vectorFromArray(
      [
        [1, 2],
        [3, 4]
      ],
      new arrow.FixedSizeList(2, new arrow.Field('value', new arrow.Float32(), false))
    )
  );
  const column = makeGeoArrowColumnFromCPUVector(source);
  const returned = makeCPUVectorFromGeoArrowColumn(column);
  expect(returned.data[0].buffer).toBe(source.data[0].buffer);
  expect(returned.data[0].byteOffset).toBe(source.data[0].byteOffset);
  expect(returned.length).toBe(source.length);
});

test('direct GeoArrow tessellation handles polygon holes without Apache Arrow conversion', async () => {
  const {makeGeoArrowColumnFromGeometryRows} = await import('@math.gl/geospatial/geoarrow');
  const {tessellateGeoArrowColumnToCPUVectors} = await import('@luma-contributors/graph/arrow');
  const column = makeGeoArrowColumnFromGeometryRows(
    [
      {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [4, 0],
            [4, 4],
            [0, 4],
            [0, 0]
          ],
          [
            [1, 1],
            [1, 2],
            [2, 2],
            [2, 1],
            [1, 1]
          ]
        ]
      }
    ],
    {encoding: 'geoarrow.polygon'}
  );
  const result = tessellateGeoArrowColumnToCPUVectors(column);
  expect(result.positions.valueLength).toBe(8);
  expect(result.indices.valueLength).toBe(24);
});

test('projection preserves value-list format and vector name', async () => {
  const {CPUData} = await import('@luma-contributors/graph/gpu-data');
  const source = new CPUVector({
    name: 'named-paths',
    format: 'value-list<float32x2>',
    data: [
      new CPUData({
        format: 'value-list<float32x2>',
        buffer: new Float32Array([1, 2, 3, 4]),
        length: 1,
        valueLength: 2,
        valueOffsets: new Int32Array([0, 2])
      })
    ]
  });
  const projection = await projectionEngine.createProjectionAsync({
    from: 'EPSG:4326',
    to: 'EPSG:3857'
  });
  const result = await projectCPUVector(source, projection);
  expect(result.name).toBe(source.name);
  expect(result.format).toBe(source.format);
  expect(result.data[0].format).toBe(source.format);
});

/** Reference implementation: the previous per-tuple byte gather. */
function readCoordinatesByBytes(chunk: CPUData, count: number, components: number): Float32Array {
  const values = new Float32Array(count * components);
  const bytes = new Uint8Array(values.buffer);
  for (let index = 0; index < count; index++)
    bytes.set(
      new Uint8Array(chunk.buffer, chunk.byteOffset + index * chunk.byteStride, components * 4),
      index * components * 4
    );
  return values;
}

test('direct coordinate reads equal the byte-gather reference for packed, offset and padded chunks', async () => {
  const {readFloat32Coordinates} = await import('@luma-contributors/graph/gpu-project/geoarrow');
  const backing = Float32Array.from({length: 64}, (_, index) => index * 1.5 - 7);
  const layouts = [
    {byteOffset: 0, byteStride: 8, components: 2},
    {byteOffset: 8, byteStride: 8, components: 2},
    {byteOffset: 4, byteStride: 16, components: 2},
    {byteOffset: 0, byteStride: 16, components: 3},
    {byteOffset: 12, byteStride: 12, components: 3}
  ] as const;
  for (const {byteOffset, byteStride, components} of layouts) {
    const chunk = new CPUData({
      buffer: backing,
      format: `float32x${components}` as 'float32x2' | 'float32x3',
      length: 4,
      byteOffset,
      byteStride
    });
    const direct = readFloat32Coordinates(chunk, 4, components, 'copy');
    expect(Array.from(direct), JSON.stringify({byteOffset, byteStride})).toEqual(
      Array.from(readCoordinatesByBytes(chunk, 4, components))
    );
    const borrowed = readFloat32Coordinates(chunk, 4, components, 'borrow');
    expect(Array.from(borrowed)).toEqual(Array.from(direct));
    if (byteStride === components * 4) expect(borrowed.buffer).toBe(backing.buffer);
    else expect(borrowed.buffer).not.toBe(backing.buffer);
  }
});

test('GeoArrow columns from padded CPU vectors keep list topology and null rows', async () => {
  const {makeGeoArrowColumnFromCPUVector} = await import(
    '@luma-contributors/graph/gpu-project/geoarrow'
  );
  const backing = Float32Array.from([0, 0, 99, 99, 1, 0, 99, 99, 1, 1, 99, 99, 0, 1, 99, 99]);
  const vector = new CPUVector({
    format: 'vertex-list<float32x2>',
    data: [
      new CPUData({
        buffer: backing,
        format: 'vertex-list<float32x2>',
        length: 3,
        valueLength: 4,
        valueOffsets: Int32Array.from([0, 4, 4, 4]),
        nullBitmap: Uint8Array.from([0b001]),
        byteStride: 16
      })
    ]
  });
  const column = makeGeoArrowColumnFromCPUVector(vector, {encoding: 'geoarrow.polygon'});
  const rings = column.chunks[0] as any;
  expect(Array.from(rings.offsets)).toEqual([0, 1, 2, 3]);
  expect(Array.from(rings.child.offsets)).toEqual([0, 4, 4, 4]);
  expect(Array.from(rings.child.child.child.values)).toEqual([0, 0, 1, 0, 1, 1, 0, 1]);
  const mesh = tessellateCPUVectorPolygons(vector);
  expect(mesh.indices.valueLength).toBe(6);
  expect(Array.from(mesh.indices.data[0].valueOffsets!)).toEqual([0, 6, 6, 6]);
});

test('projectCPUVector with an identity projection equals padded source tuples', async () => {
  const backing = Float32Array.from([1, 2, 99, 99, 3, 4, 99, 99, 5, 6, 99, 99]);
  const source = new CPUVector({
    format: 'float32x2',
    data: [new CPUData({buffer: backing, format: 'float32x2', length: 3, byteStride: 16})]
  });
  const projected = await projectCPUVector(source, {projectFlat: values => values});
  expect(
    Array.from(new Float32Array(projected.data[0].buffer, projected.data[0].byteOffset, 6))
  ).toEqual([1, 2, 3, 4, 5, 6]);
});
