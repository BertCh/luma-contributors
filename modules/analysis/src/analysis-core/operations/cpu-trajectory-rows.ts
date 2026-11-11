// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Packed trajectory arrays shared by CPU trajectory computations. */
export type CPUTrajectoryRows = {
  positions: Float32Array;
  timestamps: Float32Array | Uint32Array;
  trackOffsets: Uint32Array;
};

/**
 * Checks packed trajectory array lengths, clamps offsets to the row count as the GPU row-bounds
 * helper does, and returns the normalized offsets and a time-difference accessor. Offsets must
 * be nondecreasing after clamping; overlapping tracks are not supported.
 * Int64 words are subtracted as integers before conversion to binary64, preserving short steps
 * even near the signed Int64 endpoints. Rows before/after the offset range belong to no track.
 * @throws Error naming the inconsistent argument.
 * @internal
 */
export function getCPUTrajectoryRows(input: CPUTrajectoryRows): {
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
  const trackOffsets = input.trackOffsets.map(offset => Math.min(offset, rowCount));
  if (
    trackOffsets.length < 2 ||
    Array.from(trackOffsets).some((offset, index) => index > 0 && offset < trackOffsets[index - 1])
  ) {
    throw new Error('trackOffsets must contain at least two nondecreasing offsets after clamping');
  }
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
