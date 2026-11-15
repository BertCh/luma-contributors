// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUData,
  GPUVector,
  type GPUDataFromBufferProps,
  type GPUVectorCreateProps,
  type GPUVectorFormat
} from '@luma.gl/graph/gpu-data';

/** Typed arrays that can be uploaded as storage and vertex data. */
export type StorageBufferValues = Float32Array | Uint32Array | Int32Array | Uint8Array;

/** One borrowed buffer chunk of a batched vector. */
export type BatchedGPUVectorChunk = {
  buffer: Buffer;
  length: number;
  valueLength?: number;
  valueOffsets?: Int32Array | Uint32Array;
};

/**
 * Creates a storage and vertex buffer from typed values. Empty values allocate a 4 byte buffer
 * because zero-sized GPU buffers are invalid on some backends.
 */
export function allocateStorageBuffer(
  device: Device,
  name: string,
  values: StorageBufferValues
): Buffer {
  return device.createBuffer({
    id: name,
    ...(values.byteLength ? {data: values} : {byteLength: 4}),
    usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
  });
}

/** Wraps borrowed buffer chunks, one per source batch, without packing them. */
export function makeBatchedGPUVector<Format extends GPUVectorFormat>(
  name: string,
  format: Format,
  chunks: readonly BatchedGPUVectorChunk[]
): GPUVector<Format> {
  return new GPUVector({
    type: 'data',
    name,
    format,
    ownsData: false,
    data: chunks.map(
      chunk =>
        new GPUData({
          format,
          buffer: chunk.buffer,
          length: chunk.length,
          ...(chunk.valueLength !== undefined ? {valueLength: chunk.valueLength} : {}),
          ...(chunk.valueOffsets ? {valueOffsets: chunk.valueOffsets} : {}),
          ownsBuffer: false
        } as GPUDataFromBufferProps<Format>)
    )
  } as GPUVectorCreateProps<Format>);
}

/**
 * Maps a global primitive index to its source row. Uses cumulative offsets and a binary search so
 * picking cost is logarithmic in the number of batches. Returns -1 for out of range indices.
 */
export function makeSourceRowResolver(
  sourceRows: readonly ArrayLike<number>[]
): (primitiveIndex: number) => number {
  const cumulativeEnds = new Float64Array(sourceRows.length);
  let total = 0;
  for (let batchIndex = 0; batchIndex < sourceRows.length; batchIndex++) {
    total += sourceRows[batchIndex].length;
    cumulativeEnds[batchIndex] = total;
  }
  return primitiveIndex => {
    if (!Number.isSafeInteger(primitiveIndex) || primitiveIndex < 0 || primitiveIndex >= total)
      return -1;
    // Find the first batch whose cumulative end is greater than the index.
    let low = 0;
    let high = sourceRows.length - 1;
    while (low < high) {
      const middle = (low + high) >> 1;
      if (cumulativeEnds[middle] > primitiveIndex) high = middle;
      else low = middle + 1;
    }
    const batchStart = low === 0 ? 0 : cumulativeEnds[low - 1];
    return sourceRows[low][primitiveIndex - batchStart];
  };
}

/** One style property channel of a prepared batch. */
export type PreparedPropertyChannel = {values: Float32Array | Uint8Array; validity: Uint32Array};

/** Uploads one per-row style property as a values vector plus a validity vector. */
export function makePropertyGPUVectors<Format extends 'float32' | 'unorm8x4'>(options: {
  namePrefix: string;
  name: string;
  format: Format;
  batches: readonly {rowCount: number; channel: PreparedPropertyChannel}[];
  allocateBuffer: (name: string, values: StorageBufferValues) => Buffer;
}): {values: GPUVector<Format>; validity: GPUVector<'uint32'>} {
  const {namePrefix, name, format, batches, allocateBuffer} = options;
  return {
    values: makeBatchedGPUVector(
      `${namePrefix}-property-${name}`,
      format,
      batches.map(({rowCount, channel}, index) => ({
        buffer: allocateBuffer(`${namePrefix}-property-${name}-${index}`, channel.values),
        length: rowCount
      }))
    ),
    validity: makeBatchedGPUVector(
      `${namePrefix}-property-${name}-validity`,
      'uint32',
      batches.map(({rowCount, channel}, index) => ({
        buffer: allocateBuffer(
          `${namePrefix}-property-${name}-validity-${index}`,
          channel.validity
        ),
        length: rowCount
      }))
    )
  };
}
