// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';

/** Reads `byteLength` bytes of a buffer into a freshly allocated, aligned `ArrayBuffer`. */
async function readBufferArrayBuffer(
  buffer: Buffer,
  byteOffset: number,
  byteLength: number
): Promise<ArrayBuffer> {
  const bytes = await buffer.readAsync(byteOffset, byteLength);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + byteLength) as ArrayBuffer;
}

/** Reads `elementCount` float32 values starting at `byteOffset`; the result owns its memory. */
export async function readBufferFloat32(
  buffer: Buffer,
  elementCount: number,
  byteOffset = 0
): Promise<Float32Array> {
  return new Float32Array(await readBufferArrayBuffer(buffer, byteOffset, elementCount * 4));
}

/** Reads `elementCount` uint32 values starting at `byteOffset`; the result owns its memory. */
export async function readBufferUint32(
  buffer: Buffer,
  elementCount: number,
  byteOffset = 0
): Promise<Uint32Array> {
  return new Uint32Array(await readBufferArrayBuffer(buffer, byteOffset, elementCount * 4));
}

/** Reads `elementCount` int32 values starting at `byteOffset`; the result owns its memory. */
export async function readBufferInt32(
  buffer: Buffer,
  elementCount: number,
  byteOffset = 0
): Promise<Int32Array> {
  return new Int32Array(await readBufferArrayBuffer(buffer, byteOffset, elementCount * 4));
}

/** Reads one uint32 scalar at `byteOffset`. */
export async function readBufferUint32Scalar(buffer: Buffer, byteOffset = 0): Promise<number> {
  return (await readBufferUint32(buffer, 1, byteOffset))[0];
}
