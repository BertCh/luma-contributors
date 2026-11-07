// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import type {PreparedArrowAnalyticsTable} from '@luma.gl/graph/arrow';

type PreparedTimeColumn = PreparedArrowAnalyticsTable['timeColumns'][string];

/** True for result columns tagged as Kepler time columns (`luma.gl.time.format`). */
export function isResidentSqlTimeColumn(
  metadata: Readonly<Record<string, string>> | undefined
): boolean {
  return Boolean(metadata?.['luma.gl.time.format']);
}

/** Exact int64 second words (low then high) and validity for one chunk of epoch seconds. */
export function makeTimeChunkFromEpochSeconds(
  seconds: Int32Array,
  validity: Uint32Array | undefined
): {words: Uint32Array; validity: Uint32Array} {
  const words = new Uint32Array(seconds.length * 2);
  const mask = new Uint32Array(seconds.length);
  for (let rowIndex = 0; rowIndex < seconds.length; rowIndex++) {
    if (validity && validity[rowIndex] === 0) continue;
    // Sign extension of the int32 into the high word keeps pre-1970 instants exact.
    words[rowIndex * 2] = seconds[rowIndex] >>> 0;
    words[rowIndex * 2 + 1] = seconds[rowIndex] < 0 ? 0xffffffff : 0;
    mask[rowIndex] = 1;
  }
  return {words, validity: mask};
}

/** Bytes of CPU time words plus GPU copies for rows: 12 per row on each side. */
export function estimateTimeColumnByteLength(rowCount: number): number {
  return rowCount * 12;
}

/**
 * Reads one epoch-second column (and its validity) back chunk by chunk and prepares the exact time
 * words the native time filter needs. The read is bounded to 8 bytes per row; the resident column
 * itself is unchanged and keeps serving SQL, charts and display.
 */
export async function readResidentSqlTimeColumn(
  name: string,
  values: GPUVector<'sint32'>,
  validity: GPUVector<'uint32'> | undefined
): Promise<PreparedTimeColumn> {
  const chunks: Uint32Array[] = [];
  const masks: Uint32Array[] = [];
  for (const [chunkIndex, data] of values.data.entries()) {
    const byteLength = data.length * 4;
    const valueBytes = byteLength
      ? await (data.buffer as Buffer).readAsync(data.byteOffset, byteLength)
      : new Uint8Array(0);
    const validityData = validity?.data[chunkIndex];
    const validityBytes =
      validityData && byteLength
        ? await (validityData.buffer as Buffer).readAsync(validityData.byteOffset, byteLength)
        : undefined;
    const chunk = makeTimeChunkFromEpochSeconds(
      new Int32Array(valueBytes.slice().buffer),
      validityBytes ? new Uint32Array(validityBytes.slice().buffer) : undefined
    );
    chunks.push(chunk.words);
    masks.push(chunk.validity);
  }
  return Object.freeze({
    name,
    sourceType: 'Timestamp<SECOND>',
    unit: 'second' as const,
    timeZone: 'UTC',
    nullable: true,
    metadata: new Map([['luma.gl.time.format', 'X']]),
    chunks: Object.freeze(chunks),
    validity: Object.freeze(masks)
  });
}
