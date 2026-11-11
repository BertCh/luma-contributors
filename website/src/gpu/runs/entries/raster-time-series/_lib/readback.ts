// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Readbacks of an operation's output into per-cell maps. The operations write per-cell records
// (several words per cell, one value per slice); a map shows one number per cell, so `afterRun`
// reads the record back and uploads the number as a raster.

import type {RunBuffer, RunBuilder} from '../../../define-run';

export async function readFloat32(source: RunBuffer): Promise<Float32Array> {
  const bytes = await source.buffer.readAsync();
  return new Float32Array(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
  );
}

export async function readUint32(source: RunBuffer): Promise<Uint32Array> {
  const bytes = await source.buffer.readAsync();
  return new Uint32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

/** A map buffer of NaN, drawn until the first readback arrives. */
export function createBlankMap(builder: RunBuilder, id: string, cellCount: number) {
  const blank = builder.allocate(id, 'float32', cellCount);
  blank.write(new Float32Array(cellCount).fill(Number.NaN));
  return blank;
}
