// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export function readRegressionFloat32Word(bytes: Uint8Array): number {
  if (bytes.byteLength !== 4) throw new RangeError('Expected one float32 word');
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getFloat32(0, true);
}

export function readRegressionUint32Word(bytes: Uint8Array): number {
  if (bytes.byteLength !== 4) throw new RangeError('Expected one uint32 word');
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(0, true);
}
