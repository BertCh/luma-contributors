// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuProj.

import {assert} from '@luma.gl/core';
import {CPUData, CPUVector, getGPUVectorFormatInfo} from '@luma-contributors/graph/gpu-data';
import type {
  GeoArrowColumn,
  GeoArrowDimension,
  GeoArrowArray,
  GeoArrowFixedSizeList
} from '@math.gl/geospatial/geoarrow';

/** Borrow CPU coordinate chunks as native GeoArrow points or single-level coordinate lists. */
export function makeGeoArrowColumnFromCPUVector(
  vector: CPUVector,
  options: {
    encoding?:
      | 'geoarrow.point'
      | 'geoarrow.linestring'
      | 'geoarrow.multipoint'
      | 'geoarrow.polygon';
    dimension?: GeoArrowDimension;
  } = {}
): GeoArrowColumn {
  const info = getGPUVectorFormatInfo(vector.format);
  assert(info.signedDataType === 'float32' && info.components >= 2 && !info.fixedSizeList);
  const variable = info.vertexList || info.valueList;
  const encoding = options.encoding ?? (variable ? 'geoarrow.linestring' : 'geoarrow.point');
  assert(variable === (encoding !== 'geoarrow.point'));
  const dimension =
    options.dimension ?? (info.components === 2 ? 'xy' : info.components === 3 ? 'xyz' : 'xyzm');
  assert((dimension === 'xy' ? 2 : dimension === 'xyzm' ? 4 : 3) === info.components);
  const chunks = vector.data.map(chunk => {
    // GeoArrow primitive scalar stride cannot express padding between coordinate tuples.
    const count = variable ? chunk.valueLength : chunk.length;
    const values = readFloat32Coordinates(chunk, count, info.components, 'borrow');
    const coordinates: GeoArrowFixedSizeList = {
      kind: 'fixed-size-list',
      length: count,
      size: info.components,
      child: {
        kind: 'primitive',
        length: count * info.components,
        values
      }
    };
    const validity = chunk.nullBitmap ? {values: chunk.nullBitmap} : undefined;
    if (!variable) return {...coordinates, validity};
    assert(chunk.valueOffsets);
    const rings: GeoArrowArray = {
      kind: 'list',
      length: chunk.length,
      offsets: chunk.valueOffsets,
      child: coordinates,
      validity
    };
    return encoding === 'geoarrow.polygon'
      ? {
          kind: 'list' as const,
          length: chunk.length,
          offsets: makeIdentityOffsets(chunk.length),
          child: rings,
          validity
        }
      : rings;
  });
  return {encoding, dimension, coordinateLayout: 'interleaved', chunks};
}

/** Borrow packed Float32 GeoArrow points/lines/multipoints as CPU chunks. Nested polygons stay GeoArrow. */
export function makeCPUVectorFromGeoArrowColumn(column: GeoArrowColumn): CPUVector {
  assert(
    column.coordinateLayout === 'interleaved' &&
      ['geoarrow.point', 'geoarrow.linestring', 'geoarrow.multipoint'].includes(column.encoding)
  );
  const components = column.dimension === 'xy' ? 2 : column.dimension === 'xyzm' ? 4 : 3;
  const variable = column.encoding !== 'geoarrow.point';
  const element = `float32x${components}` as 'float32x2' | 'float32x3' | 'float32x4';
  const format = variable ? (`vertex-list<${element}>` as const) : element;
  const data = column.chunks.map(chunk => {
    assert(!chunk.validity?.bitOffset);
    const coordinates = variable && chunk.kind === 'list' ? chunk.child : chunk;
    assert(
      coordinates.kind === 'fixed-size-list' &&
        coordinates.size === components &&
        !coordinates.offset &&
        (!variable || !coordinates.validity)
    );
    const primitive = coordinates.child;
    assert(
      primitive.kind === 'primitive' &&
        primitive.values instanceof Float32Array &&
        !primitive.validity &&
        (!primitive.stride || primitive.stride === 1)
    );
    const offset = primitive.offset ?? 0;
    const values = primitive.values.subarray(offset, offset + coordinates.length * components);
    let valueOffsets: Int32Array | undefined;
    if (variable) {
      assert(
        chunk.kind === 'list' &&
          chunk.offsets instanceof Int32Array &&
          !chunk.offset &&
          !chunk.offsetBase
      );
      valueOffsets = chunk.offsets;
    }
    return new CPUData({
      buffer: values,
      format,
      length: chunk.length,
      valueLength: variable ? coordinates.length : undefined,
      valueOffsets,
      nullBitmap: chunk.validity?.values
    });
  });
  return new CPUVector({format, data});
}

/** One list offset per row, for polygons that hold exactly one ring per row. */
function makeIdentityOffsets(length: number): Int32Array {
  const offsets = new Int32Array(length + 1);
  for (let index = 1; index <= length; index++) offsets[index] = index;
  return offsets;
}

/**
 * Reads `count` Float32 tuples of a CPU chunk straight from its buffer. Packed aligned chunks are
 * borrowed when `mode` is `borrow` and copied when it is `copy`. Padded rows are gathered with
 * typed-array reads (GeoArrow primitive scalar stride cannot express padding between tuples).
 */
export function readFloat32Coordinates(
  chunk: CPUData,
  count: number,
  components: number,
  mode: 'borrow' | 'copy'
): Float32Array {
  const tupleByteLength = components * Float32Array.BYTES_PER_ELEMENT;
  const isAligned = chunk.byteOffset % 4 === 0;
  if (isAligned && chunk.byteStride === tupleByteLength) {
    const values = new Float32Array(chunk.buffer, chunk.byteOffset, count * components);
    return mode === 'borrow' ? values : values.slice();
  }
  const values = new Float32Array(count * components);
  if (isAligned && chunk.byteStride % 4 === 0) {
    const source = new Float32Array(chunk.buffer, chunk.byteOffset);
    const elementStride = chunk.byteStride / 4;
    for (let index = 0; index < count; index++)
      for (let component = 0; component < components; component++)
        values[index * components + component] = source[index * elementStride + component];
    return values;
  }
  const view = new DataView(chunk.buffer, chunk.byteOffset);
  for (let index = 0; index < count; index++)
    for (let component = 0; component < components; component++)
      values[index * components + component] = view.getFloat32(
        index * chunk.byteStride + component * 4,
        true
      );
  return values;
}
