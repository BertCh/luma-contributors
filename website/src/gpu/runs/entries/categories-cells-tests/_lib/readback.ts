// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Typed reads of GPU buffers an operation wrote, for runs that post-process results in `afterRun`.

import type {RunBuffer, RunFormat} from '../../../define-run';

function copyBytes(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

/** The buffer's contents as float32 values. */
export async function readFloats(buffer: RunBuffer<RunFormat>): Promise<Float32Array> {
  return new Float32Array(copyBytes(await buffer.buffer.readAsync()));
}

/** The buffer's contents as uint32 values. */
export async function readUints(buffer: RunBuffer<RunFormat>): Promise<Uint32Array> {
  return new Uint32Array(copyBytes(await buffer.buffer.readAsync()));
}
