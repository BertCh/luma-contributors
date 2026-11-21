// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getPackedFloat32Values,
  getPackedUint32Values
} from '../../analysis-core/kernels/packed-readback';
import {GPU_DATUM_GRID_TRANSFORM_STATUS} from './gpu-datum-grid-transform';

type FloatReadback = Float32Array | Uint8Array | ArrayBuffer;
type StatusReadback = Uint32Array | Uint8Array | ArrayBuffer;

export type GPUDatumGridTransformDecodedPositions = {
  /** Packed XYZ in degrees, degrees and the producer's declared height unit, summed in binary64. */
  positions: Float64Array;
  /** One unchanged status word per XYZ row, in chunk order. Zero denotes success. */
  status: Uint32Array;
};

/**
 * Reconstructs packed or identically aligned chunk readbacks, preserving byte windows and statuses.
 * Every position chunk must contain three components per row and exactly one status per row;
 * low words, when supplied, must match each high-word chunk. Empty output is valid. Failed rows
 * retain their producer values (normally NaNs); consult status before using a row.
 * Equivalent for one chunk: Float64Array.from(high, (value, index) => value + low[index]).
 */
export function decodeGPUDatumGridTransformPositions(input: {
  positions: FloatReadback | readonly FloatReadback[];
  positionsLow?: FloatReadback | readonly FloatReadback[];
  status: StatusReadback | readonly StatusReadback[];
}): GPUDatumGridTransformDecodedPositions {
  const identifier = 'decodeGPUDatumGridTransformPositions';
  const highChunks: readonly FloatReadback[] = Array.isArray(input.positions)
    ? input.positions
    : [input.positions as FloatReadback];
  const lowChunks: readonly FloatReadback[] | undefined =
    input.positionsLow === undefined
      ? undefined
      : Array.isArray(input.positionsLow)
        ? input.positionsLow
        : [input.positionsLow as FloatReadback];
  const statusChunks: readonly StatusReadback[] = Array.isArray(input.status)
    ? input.status
    : [input.status as StatusReadback];
  validateRequirement(
    identifier,
    statusChunks.length === highChunks.length &&
      (lowChunks === undefined || lowChunks.length === highChunks.length),
    `chunk counts must match: positions ${highChunks.length}, positionsLow ${lowChunks?.length}, status ${statusChunks.length}`
  );
  const chunks = highChunks.map((high, chunkIndex) => {
    const positions = getPackedFloat32Values(high);
    const positionsLow = lowChunks ? getPackedFloat32Values(lowChunks[chunkIndex]) : undefined;
    const status = getPackedUint32Values(statusChunks[chunkIndex]);
    validateRequirement(
      identifier,
      positions.length % 3 === 0,
      `positions chunk ${chunkIndex} length received ${positions.length}; must contain three components per row`
    );
    validateRequirement(
      identifier,
      status.length === positions.length / 3 &&
        (positionsLow === undefined || positionsLow.length === positions.length),
      `chunk ${chunkIndex} row alignment requires positionsLow length ${positions.length} and status length ${positions.length / 3}; received ${positionsLow?.length}, ${status.length}`
    );
    return {positions, positionsLow, status};
  });
  const positions = new Float64Array(
    chunks.reduce((count, chunk) => count + chunk.positions.length, 0)
  );
  const status = new Uint32Array(positions.length / 3);
  let componentOffset = 0;
  for (const chunk of chunks) {
    for (let index = 0; index < chunk.positions.length; index++) {
      positions[componentOffset + index] =
        chunk.positions[index] + (chunk.positionsLow?.[index] ?? 0);
    }
    status.set(chunk.status, componentOffset / 3);
    componentOffset += chunk.positions.length;
  }
  return {positions, status};
}

/** Decodes one row's coded status from a typed or raw readback; missing words are rejected. */
export function decodeGPUDatumGridTransformStatus(input: StatusReadback) {
  const words = getPackedUint32Values(input, 1);
  validateRequirement(
    'decodeGPUDatumGridTransformStatus',
    words.length >= 1,
    `status length received ${words.length}; must contain at least one word`
  );
  const code = words[0];
  return {
    complete: code === GPU_DATUM_GRID_TRANSFORM_STATUS.success,
    invalidInput: (code & GPU_DATUM_GRID_TRANSFORM_STATUS.invalid) !== 0,
    noCoverage: (code & GPU_DATUM_GRID_TRANSFORM_STATUS.noCoverage) !== 0,
    nodata: (code & GPU_DATUM_GRID_TRANSFORM_STATUS.nodata) !== 0,
    iterationLimit: (code & GPU_DATUM_GRID_TRANSFORM_STATUS.inverseExhausted) !== 0
  };
}
