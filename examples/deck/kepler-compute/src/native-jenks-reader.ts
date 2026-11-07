// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import type {KeplerMemoryReservation} from '@luma-contributors/kepler';
import type {NativeJenksThresholds} from './native-style-plan';

/**
 * Jenks is exact at every size: the pinned Kepler `naturalBreaks` is a deterministic seeded
 * permutation search (no sampling, no Math.random) over every sorted finite value, so it is
 * reproduced by reading the finite selected Float32 values in accounted chunks and running it.
 */
/** Bytes per row of the transferred Float32 list, the worker's sorted copy and its unique positions. */
const BYTES_PER_ROW = 4 + 4 + 8;
/** Bytes per row of one chunk's value, validity and mask reads plus its finite copy. */
const BYTES_PER_CHUNK_ROW = 4 + 4 + 4 + 4;

export type NativeJenksReadProps = {
  source: {datasetId: string; revision: number};
  column: string;
  classCount: number;
  values: GPUVector<'float32'>;
  validity?: GPUVector<'uint32'>;
  /** Present for selected domains; nonzero rows are included. */
  mask?: GPUVector<'uint32'>;
  selectionRevision?: number;
  reserveMemory(
    byteLength: number,
    memoryKind?: 'cpu' | 'gpu',
    category?: 'source' | 'output' | 'scratch' | 'readback'
  ): KeplerMemoryReservation;
  /** Computes native `naturalBreaks` over the transferred finite values (worker or in place). */
  computeBreaks(values: Float32Array, classCount: number): Promise<number[]>;
};

async function readChunk<Format extends 'float32' | 'uint32'>(
  vector: GPUVector<Format>,
  chunkIndex: number,
  asFloat: boolean
): Promise<Float32Array | Uint32Array> {
  const chunk = vector.data[chunkIndex];
  const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * chunk.byteStride);
  // Copy to an aligned buffer; mapped byte views carry no alignment guarantee.
  const aligned = bytes.slice().buffer as ArrayBuffer;
  return asFloat ? new Float32Array(aligned) : new Uint32Array(aligned);
}

async function readFiniteChunk(
  props: NativeJenksReadProps,
  chunkIndex: number
): Promise<Float32Array> {
  const [values, validity, mask] = await Promise.all([
    readChunk(props.values, chunkIndex, true) as Promise<Float32Array>,
    props.validity
      ? (readChunk(props.validity, chunkIndex, false) as Promise<Uint32Array>)
      : undefined,
    props.mask ? (readChunk(props.mask, chunkIndex, false) as Promise<Uint32Array>) : undefined
  ]);
  const finiteValues = new Float32Array(values.length);
  let count = 0;
  for (let row = 0; row < values.length; row++) {
    if (validity && !validity[row]) continue;
    if (mask && !mask[row]) continue;
    if (Number.isFinite(values[row])) finiteValues[count++] = values[row];
  }
  return finiteValues.subarray(0, count);
}

/**
 * Accounted readback returning revision-bound thresholds identical to native `naturalBreaks`.
 * The memory reservation covers the full value list, the worker copy and the largest chunk; it is
 * refused only when the registry refuses it.
 */
export async function readNativeJenksThresholds(
  props: NativeJenksReadProps
): Promise<NativeJenksThresholds> {
  const rowCount = props.values.length;
  let chunkRows = 0;
  for (const chunk of props.values.data) chunkRows = Math.max(chunkRows, chunk.length);
  const reservation = props.reserveMemory(
    rowCount * BYTES_PER_ROW + chunkRows * BYTES_PER_CHUNK_ROW,
    'cpu',
    'readback'
  );
  try {
    const finiteValues = new Float32Array(rowCount);
    let finiteCount = 0;
    for (let chunkIndex = 0; chunkIndex < props.values.data.length; chunkIndex++) {
      const chunkValues = await readFiniteChunk(props, chunkIndex);
      finiteValues.set(chunkValues, finiteCount);
      finiteCount += chunkValues.length;
    }
    // The buffer is transferred to the worker, so hand over an exactly sized copy-free view.
    const values = finiteCount === rowCount ? finiteValues : finiteValues.slice(0, finiteCount);
    return {
      source: props.source,
      column: props.column,
      classCount: props.classCount,
      ...(props.selectionRevision !== undefined
        ? {selectionRevision: props.selectionRevision}
        : {}),
      values: await props.computeBreaks(values, props.classCount)
    };
  } finally {
    reservation.release();
  }
}
