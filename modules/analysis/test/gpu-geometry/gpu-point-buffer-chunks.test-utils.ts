// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';

/** Reads one float32x2 ring without copying the whole chunk back from the GPU. */
export async function readPointBufferRing(
  buffer: Pick<Buffer, 'readAsync'>,
  ringIndex: number,
  verticesPerRing: number
): Promise<number[]> {
  const componentCount = verticesPerRing * 2;
  const ringByteLength = componentCount * Float32Array.BYTES_PER_ELEMENT;
  const bytes = await buffer.readAsync(ringIndex * ringByteLength, ringByteLength);
  // readAsync returns Uint8Array: reinterpret its backing range, rather than converting bytes.
  return Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, componentCount));
}
