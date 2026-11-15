// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {PickingInfo} from '@deck.gl/core';
import {assert, Buffer, type BufferLayout, type Device, type VertexFormat} from '@luma.gl/core';
import type {GPUData, GPUVector, GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {getGPUVectorFormatInfo} from '@luma.gl/graph/gpu-data';
import {getGPUVectorModelBatches} from '@luma.gl/engine';
import {DynamicBuffer} from '@luma.gl/core/dynamic';

/** Maximum row count addressable by Deck's RGB picking color (zero is no hit). */
export const GPU_VECTOR_MAXIMUM_PICKABLE_ROWS = 0xffffff;

/** Shared resident selection contract. The caller retains ownership of the mask. */
export type GPUVectorLayerSelectionProps = {
  /**
   * One uint32 flag per source feature: zero rejects, any nonzero value selects.
   * Must match source chunk boundaries and use buffers with VERTEX usage. Compute writers should
   * also request STORAGE usage. Rewriting flags does not require replacing the vector or model.
   */
  selectionMask?: GPUVector<'uint32'>;
};

/** Rejects ambiguous RGB picking IDs without reading source data. */
export function validateGPUVectorPickingCapacity(rowCount: number, pickable: boolean): void {
  // RGB picking reserves zero for no hit. Split larger pickable inputs into separate layers.
  assert(!pickable || rowCount <= GPU_VECTOR_MAXIMUM_PICKABLE_ROWS);
}

/** Describes the optional instance mask, defaulting to a single accepted flag. */
export function makeGPUVectorSelectionLayout(selectionMask?: GPUVector<'uint32'>): BufferLayout {
  return selectionMask
    ? makeGPUVectorBufferLayout(selectionMask, 'selectionMask')
    : {
        name: 'selectionMask',
        byteStride: 0,
        stepMode: 'instance',
        attributes: [{attribute: 'selectionMask', format: 'uint32', byteOffset: 0}]
      };
}

/** One physical, row-aligned set of GPUVector chunks rendered as a model draw batch. */
export type GPUVectorLayerBatch = {
  batchIndex: number;
  rowIndexOffset: number;
  rowCount: number;
  data: Record<string, GPUData>;
};

/** Global row and physical chunk provenance attached to GPUVector layer picking results. */
export type GPUVectorLayerPickingInfo = PickingInfo & {
  gpuVector?: {rowIndex: number; batchIndex: number; batchRowIndex: number};
};

/** Resolves global picking identity back to a physical GPUVector chunk. */
export function getGPUVectorPickingProvenance(
  vector: GPUVector,
  rowIndex: number
): {rowIndex: number; batchIndex: number; batchRowIndex: number} {
  if (!Number.isInteger(rowIndex) || rowIndex < 0 || rowIndex >= vector.length) {
    return {rowIndex, batchIndex: -1, batchRowIndex: -1};
  }
  let rowIndexOffset = 0;
  for (const [batchIndex, data] of vector.data.entries()) {
    if (rowIndex < rowIndexOffset + data.length) {
      return {rowIndex, batchIndex, batchRowIndex: rowIndex - rowIndexOffset};
    }
    rowIndexOffset += data.length;
  }
  return {rowIndex, batchIndex: -1, batchRowIndex: -1};
}

/** Validates vector formats and returns aligned physical batches without packing or copying. */
export function getGPUVectorLayerBatches(
  ownerName: string,
  vectors: Record<string, GPUVector | undefined>,
  formats: Record<string, readonly GPUVectorFormat[]>
): GPUVectorLayerBatch[] {
  const entries = Object.entries(vectors).filter((entry): entry is [string, GPUVector] =>
    Boolean(entry[1])
  );
  if (entries.length === 0) return [];
  for (const [name, vector] of entries) {
    const acceptedFormats = formats[name];
    if (!vector.format || !acceptedFormats?.includes(vector.format)) {
      throw new Error(
        `${ownerName} ${name} GPUVector.format "${vector.format ?? 'undefined'}" must be one of ${acceptedFormats?.join(', ') ?? 'the declared formats'}`
      );
    }
  }
  return getGPUVectorModelBatches(ownerName, Object.fromEntries(entries));
}

/** Returns the luma buffer behind a fixed-width GPUData chunk. */
export function getGPUDataBuffer(data: GPUData): Buffer {
  return data.buffer instanceof DynamicBuffer ? data.buffer.buffer : data.buffer;
}

/** Returns the first physical buffer used to initialize a GPUVectorModel attribute. */
export function getGPUVectorBuffer(vector: GPUVector): Buffer {
  const data = vector.data[0];
  if (!data) throw new Error('GPU deck layers require a non-empty GPUVector');
  return getGPUDataBuffer(data);
}

/** Describes one borrowed GPUData chunk as a model vertex buffer. */
export function makeGPUDataBufferLayout(data: GPUData, name: string): BufferLayout {
  if (!data.format || typeof data.format !== 'string') {
    throw new Error('GPU deck layers require a scalar GPUVectorFormat');
  }
  const formatInfo = getGPUVectorFormatInfo(data.format);
  if (formatInfo.vertexList || formatInfo.valueList || formatInfo.fixedSizeList) {
    throw new Error(`GPU deck binary attributes do not accept ${data.format}`);
  }
  return {
    name,
    stepMode: 'instance',
    byteStride: data.byteStride,
    attributes: [
      {
        attribute: name,
        format: data.format as VertexFormat,
        byteOffset: data.byteOffset
      }
    ]
  };
}

/** Describes a fixed-width GPUVector as a GPUVectorModel vertex buffer. */
export function makeGPUVectorBufferLayout(vector: GPUVector, name: string): BufferLayout {
  const data = vector.data[0];
  if (!data) throw new Error('GPU deck layers require a non-empty GPUVector');
  return makeGPUDataBufferLayout(data, name);
}

/** Allocates an aligned uniform region for each physical draw batch. */
export function createGPUVectorStyleBuffer(
  device: Device,
  id: string,
  byteLength: number,
  batchCount: number
): Buffer {
  const byteStride = getGPUVectorStyleByteStride(device, byteLength);
  return device.createBuffer({
    id,
    byteLength: byteStride * Math.max(batchCount, 1),
    usage: Buffer.UNIFORM | Buffer.COPY_DST
  });
}

/** Grows style storage when caller-owned vectors append more physical chunks. */
export function resizeGPUVectorStyleBuffer(
  buffer: Buffer,
  byteLength: number,
  batchCount: number
): Buffer {
  if (getGPUVectorStyleByteStride(buffer.device, byteLength) * batchCount <= buffer.byteLength) {
    return buffer;
  }
  const resizedBuffer = createGPUVectorStyleBuffer(
    buffer.device,
    buffer.id,
    byteLength,
    batchCount
  );
  buffer.destroy();
  return resizedBuffer;
}

/** Binds only the batch's region, retaining its contents until the encoded pass executes. */
export function getGPUVectorStyleBufferBinding(
  buffer: Buffer,
  byteLength: number,
  batchIndex = 0
): {buffer: Buffer; offset: number; size: number} {
  return {
    buffer,
    offset: batchIndex * getGPUVectorStyleByteStride(buffer.device, byteLength),
    size: byteLength
  };
}

function getGPUVectorStyleByteStride(device: Device, byteLength: number): number {
  const alignment = device.limits.minUniformBufferOffsetAlignment;
  return Math.ceil(byteLength / alignment) * alignment;
}
