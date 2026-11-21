// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCPUHostOffsets, type CPUHostRowsOptions} from './compact-host-offsets';

/** Packed trajectory arrays shared by CPU trajectory computations. */
export type CPUTrajectoryRows = {
  positions: Float32Array;
  timestamps: Float32Array | Uint32Array;
  trackOffsets: Uint32Array;
  /** Explicit resident-capacity storage retains GPU offset clamping. Defaults to compact. */
  storage?: CPUHostRowsOptions['storage'];
};

/**
 * Checks compact trajectory lengths and exact terminal offsets, returning a time-difference
 * accessor. A single [0] encodes zero tracks; [0,0] encodes one empty track.
 * Use storage: resident-capacity explicitly for GPU-compatible clamped offsets.
 * Int64 words are subtracted as integers before conversion to binary64, preserving short steps
 * even near the signed Int64 endpoints. In resident-capacity mode, rows outside the offset range
 * belong to no track.
 * @throws Error naming the inconsistent argument.
 * @internal
 */
export function getCPUTrajectoryRows(
  input: CPUTrajectoryRows,
  options: CPUHostRowsOptions = {storage: input.storage}
): {
  rowCount: number;
  trackOffsets: Uint32Array;
  getTimeDifference(later: number, earlier: number): number;
} {
  const {positions, timestamps} = input;
  const rowCount = positions.length / 2;
  if (!Number.isInteger(rowCount)) throw new Error('positions must contain float32x2 rows');
  const isWordTime = timestamps instanceof Uint32Array;
  if (timestamps.length !== rowCount * (isWordTime ? 2 : 1)) {
    throw new Error('timestamps length must match positions');
  }
  validateCPUHostOffsets(
    'getCPUTrajectoryRows',
    'trackOffsets',
    input.trackOffsets,
    rowCount,
    options
  );
  const trackOffsets = input.trackOffsets.map(offset => Math.min(offset, rowCount));
  const getWords = (row: number) =>
    BigInt.asIntN(64, (BigInt(timestamps[row * 2 + 1]) << 32n) | BigInt(timestamps[row * 2]));
  return {
    rowCount,
    trackOffsets,
    getTimeDifference: (later, earlier) =>
      isWordTime
        ? Number(getWords(later) - getWords(earlier))
        : timestamps[later] - timestamps[earlier]
  };
}
