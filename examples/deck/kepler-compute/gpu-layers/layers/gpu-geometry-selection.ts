// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import {GPUVector, type GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {getGPUDataBuffer} from './gpu-vector-layer-utils';

/** Feature-aligned storage selection used after paths or polygons expand into vertices. */
export const GPU_GEOMETRY_SELECTION_SHADER = /* wgsl */ `
struct GeometrySelectionConfig {
  enabled: u32,
  rowIndexBase: u32,
  wordOffset: u32,
  wordStride: u32,
  rowCount: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
};
@group(0) @binding(auto) var<uniform> geometrySelectionConfig: GeometrySelectionConfig;
@group(0) @binding(auto) var<storage, read> geometrySelection: array<u32>;
fn isGeometryRowSelected(rowIndex: u32) -> bool {
  if (geometrySelectionConfig.enabled == 0u) { return true; }
  if (rowIndex < geometrySelectionConfig.rowIndexBase) { return false; }
  let localRowIndex = rowIndex - geometrySelectionConfig.rowIndexBase;
  if (localRowIndex >= geometrySelectionConfig.rowCount) { return false; }
  return geometrySelection[geometrySelectionConfig.wordOffset + localRowIndex * geometrySelectionConfig.wordStride] != 0u;
}
`;

/** Borrows one physical geometry chunk, without copying values or changing its ownership. */
export function borrowGPUGeometryChunk<Format extends GPUVectorFormat>(
  vector: GPUVector<Format>,
  chunkIndex: number
): GPUVector<Format> {
  const chunk = vector.data[chunkIndex];
  // Geometry and selection adapters must preserve matching physical chunk boundaries.
  assert(chunk);
  return new GPUVector({
    type: 'data',
    name: vector.name,
    format: vector.format,
    data: [chunk],
    ownsData: false,
    dataType: vector.dataType
  });
}

/** Validates feature alignment without inspecting or downloading mask values. */
export function validateGPUGeometrySelection(
  device: Device,
  reference: GPUVector,
  selectionMask?: GPUVector<'uint32'>
): void {
  if (!selectionMask) return;
  // One uint32 mask row per source feature, preserving physical chunk boundaries.
  assert(selectionMask.format === 'uint32');
  assert(selectionMask.length === reference.length);
  assert(selectionMask.data.length === reference.data.length);
  for (const [chunkIndex, chunk] of selectionMask.data.entries()) {
    const buffer = getGPUDataBuffer(chunk);
    assert(chunk.length === reference.data[chunkIndex].length);
    // Storage addressing supports sliced/interleaved masks without repacking their data.
    assert(chunk.byteOffset % 4 === 0 && chunk.byteStride % 4 === 0);
    assert(chunk.byteStride >= 4);
    assert(buffer.device === device && (buffer.usage & Buffer.STORAGE) !== 0);
    assert(buffer.byteLength <= device.limits.maxStorageBufferBindingSize);
  }
}

/** Owns only its tiny uniform/default buffers; every input mask remains caller-owned. */
export class GPUGeometrySelectionBinding {
  readonly configuration: Buffer;
  readonly defaultMask: Buffer;

  constructor(device: Device, id: string) {
    this.configuration = device.createBuffer({
      id: `${id}-selection-configuration`,
      byteLength: 32,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    this.defaultMask = device.createBuffer({
      id: `${id}-default-selection`,
      data: new Uint32Array([1]),
      usage: Buffer.STORAGE
    });
  }

  update(selectionMask: GPUVector<'uint32'> | undefined, chunkIndex: number, rowIndexBase: number) {
    const chunk = selectionMask?.data[chunkIndex];
    this.configuration.write(
      new Uint32Array([
        chunk ? 1 : 0,
        rowIndexBase,
        chunk ? chunk.byteOffset / 4 : 0,
        chunk ? chunk.byteStride / 4 : 1,
        chunk?.length ?? 0,
        0,
        0,
        0
      ])
    );
    return {
      geometrySelectionConfig: this.configuration,
      geometrySelection: chunk ? getGPUDataBuffer(chunk) : this.defaultMask
    };
  }

  destroy(): void {
    this.configuration.destroy();
    this.defaultMask.destroy();
  }
}

/** Validates row/chunk topology before per-chunk models borrow any geometry inputs. */
export function validateGPUGeometryChunks(
  reference: GPUVector,
  inputs: Array<GPUVector | undefined>
): void {
  for (const input of inputs) {
    if (!input) continue;
    // Row-oriented style/geometry vectors must share the source physical batches.
    assert(input.length === reference.length && input.data.length === reference.data.length);
    for (const [chunkIndex, chunk] of input.data.entries()) {
      assert(chunk.length === reference.data[chunkIndex].length);
    }
  }
}
